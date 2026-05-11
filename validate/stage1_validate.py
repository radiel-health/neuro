import argparse
from pathlib import Path

import pandas as pd
import torch

from pipeline.core.context import PatientContext
from pipeline.core.results import Stage1Explanation
from pipeline.explainability.cam import CAMExplainer
from pipeline.stages.stage1 import Stage1PatchConfig, stage1_variant_for_classes
from pipeline.visualization.overlays import VolumeOverlayViewer
from cams import infer_num_classes_from_checkpoint, latest_checkpoint
from utils import DEFAULT_DATA_ROOT, VERTEBRA_LABELS, load_img


DEFAULT_MODEL_PATH = "output/stage1/4_class/2026-03-20/best.pth"


def parse_patch_size(text):
    parts = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("patch size must be D,H,W, e.g. 96,96,64")
    return tuple(parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a stage1 checkpoint on one vertebra and display a 3D CAM overlay."
    )
    parser.add_argument("--patient-id", type=str, default="", help="Patient id to resolve CT/SEG from root dir.")
    parser.add_argument("--ct", type=str, default="", help="Path to CT NIfTI. Overrides --patient-id CT path.")
    parser.add_argument("--seg", type=str, default="", help="Path to segmentation NIfTI. Overrides --patient-id SEG path.")
    parser.add_argument("--root-dir", type=str, default=DEFAULT_DATA_ROOT, help="Dataset root directory.")
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default="../../datasets/radiel/vertebra_dataset.csv",
        help="CSV with patient_id, vertebra, label columns for ground-truth lookup.",
    )
    parser.add_argument("--vertebra", type=str, required=True, help="Vertebra to classify, e.g. T8 or L2.")
    parser.add_argument("--model-path", type=str, default="", help="Stage1 checkpoint. Defaults to the known 4-class best checkpoint.")
    parser.add_argument(
        "--use-latest",
        action="store_true",
        help="Use the most recently modified checkpoint under --checkpoint-root instead of DEFAULT_MODEL_PATH.",
    )
    parser.add_argument("--checkpoint-root", type=str, default="output/stage1/4_class", help="Root used with --use-latest.")
    parser.add_argument("--checkpoint-name", type=str, default="best.pth", help="Checkpoint filename used with --use-latest.")
    parser.add_argument("--num-classes", type=int, default=0, help="Override class count if checkpoint inference fails.")
    parser.add_argument("--target-class", type=str, default="", help="Class index/name for CAM. Defaults to predicted class.")
    parser.add_argument("--cam-method", type=str, default="gradcam", choices=["gradcam", "gradcam++", "layercam"], help="CAM method.")
    parser.add_argument("--cam-threshold", type=float, default=0.35, help="Minimum normalized CAM value to render.")
    parser.add_argument("--patch-size", type=parse_patch_size, default=(96, 96, 64), help="Stage1 patch size as D,H,W.")
    parser.add_argument("--norm-mode", type=str, default="zscore_sigmoid", choices=["zscore_sigmoid", "robust_minmax"])
    parser.add_argument("--zscore-scale", type=float, default=1.5)
    parser.add_argument("--foreground-floor", type=float, default=0.15)
    parser.add_argument("--no-show", action="store_true", help="Print prediction and compute CAM without opening vedo.")
    return parser.parse_args()


def resolve_model_path(args):
    if args.model_path:
        return Path(args.model_path)
    if args.use_latest:
        return latest_checkpoint(args.checkpoint_root, filename=args.checkpoint_name)
    return Path(DEFAULT_MODEL_PATH)


def parse_target_class(value, class_names):
    if value == "":
        return None
    value_key = str(value).strip().lower()
    if value_key.isdigit():
        return int(value_key)
    names = {name.lower(): idx for idx, name in enumerate(class_names)}
    if value_key not in names:
        raise ValueError(f"Unknown target class '{value}'. Use one of: {', '.join(class_names)}")
    return names[value_key]


def load_ground_truth_label(csv_path, patient_id, vertebra):
    if not csv_path or not patient_id:
        return None
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    required = {"patient_id", "vertebra", "label"}
    if not required.issubset(df.columns):
        return None
    rows = df[
        (df["patient_id"].astype(str) == str(patient_id))
        & (df["vertebra"].astype(str).str.upper() == str(vertebra).upper())
    ]
    if rows.empty:
        return None
    return int(rows.iloc[0]["label"])


def build_context(args):
    if args.ct and args.seg:
        ct_path = Path(args.ct)
        seg_path = Path(args.seg)
        if not ct_path.exists():
            raise FileNotFoundError(f"CT not found: {ct_path}")
        if not seg_path.exists():
            raise FileNotFoundError(f"Segmentation not found: {seg_path}")
        ct, affine = load_img(str(ct_path), canonical=False)
        seg, _ = load_img(str(seg_path), canonical=False)
        return PatientContext(
            patient_id=args.patient_id or ct_path.parent.name,
            ct=ct,
            seg=seg,
            affine=affine,
            ct_path=ct_path,
            seg_path=seg_path,
        )
    if not args.patient_id:
        raise ValueError("Provide --patient-id, or provide both --ct and --seg.")
    return PatientContext.load(args.patient_id, root_dir=args.root_dir, canonical=False)


class Stage1Validator:
    def __init__(self, stage1, viewer=None):
        self.stage1 = stage1
        self.viewer = viewer or VolumeOverlayViewer()

    def run(self, context, vertebra, target_class=None, gt_label=None, show=True, cam_threshold=0.35):
        patch, mask = self.stage1.build_patch(context, vertebra, with_coords=False)
        prediction = self.stage1.predict_patch(vertebra, patch)
        cam_result = self.stage1.cam.compute(self.stage1.model, patch, target_class=target_class)
        explanation = Stage1Explanation(
            prediction=prediction,
            cam=cam_result.cam,
            mask=mask,
            coords=None,
        )

        self.print_prediction(context, prediction, self.stage1.class_names, gt_label, cam_result.target_class)
        if show:
            self.show(context, patch, explanation, gt_label, cam_result.target_class, cam_threshold)
        return explanation

    def print_prediction(self, context, prediction, class_names, gt_label, target_class):
        print(f"patient: {context.patient_id}")
        print(f"vertebra: {prediction.vertebra}")
        print(f"model: {self.stage1.model_path}")
        if gt_label is not None:
            gt_name = class_names[gt_label] if 0 <= gt_label < len(class_names) else str(gt_label)
            print(f"ground truth: {gt_label} ({gt_name})")
        else:
            print("ground truth: unavailable")
        print(f"prediction: {prediction.class_id} ({prediction.class_name})")
        for idx, prob in enumerate(prediction.probabilities):
            name = class_names[idx] if idx < len(class_names) else str(idx)
            print(f"  p[{idx} {name}]: {prob:.4f}")
        target_name = class_names[target_class] if target_class < len(class_names) else str(target_class)
        print(f"CAM: method={self.stage1.cam.method} target={target_class} ({target_name})")

    def show(self, context, patch, explanation, gt_label, target_class, cam_threshold):
        class_names = self.stage1.class_names
        gt_text = "GT unavailable"
        if gt_label is not None:
            gt_name = class_names[gt_label] if 0 <= gt_label < len(class_names) else str(gt_label)
            gt_text = f"GT {gt_label} {gt_name}"
        target_name = class_names[target_class] if target_class < len(class_names) else str(target_class)
        p = explanation.prediction
        self.viewer.show_stage1_patch_cam(
            patch,
            explanation,
            title=(
                f"{context.patient_id} {p.vertebra} | {self.stage1.cam.method}\n"
                f"{gt_text} | Pred {p.class_id} {p.class_name} "
                f"({p.confidence:.3f}) | CAM target {target_class} {target_name}"
            ),
            cam_threshold=cam_threshold,
        )


def main():
    args = parse_args()
    vertebra = args.vertebra.upper()
    if vertebra not in VERTEBRA_LABELS:
        raise ValueError(f"Unknown vertebra: {args.vertebra}")

    model_path = resolve_model_path(args)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    num_classes = args.num_classes or infer_num_classes_from_checkpoint(model_path)
    stage_cls = stage1_variant_for_classes(num_classes)
    stage1 = stage_cls(
        model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cam=CAMExplainer(method=args.cam_method),
        patch_config=Stage1PatchConfig(
            patch_size=args.patch_size,
            norm_mode=args.norm_mode,
            zscore_scale=args.zscore_scale,
            foreground_floor=args.foreground_floor,
        ),
    )
    context = build_context(args)
    target_class = parse_target_class(args.target_class, stage1.class_names)
    gt_label = load_ground_truth_label(args.dataset_csv, context.patient_id, vertebra)
    Stage1Validator(stage1).run(
        context,
        vertebra,
        target_class=target_class,
        gt_label=gt_label,
        show=not args.no_show,
        cam_threshold=args.cam_threshold,
    )


if __name__ == "__main__":
    main()
