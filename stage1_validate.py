import argparse
from pathlib import Path

import pandas as pd
import torch

from cams import (
    build_stage1_patch_and_mask,
    class_names_for_count,
    compute_cam,
    infer_num_classes_from_checkpoint,
    latest_checkpoint,
    load_stage1_model,
    show_cam_overlay,
)
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


def resolve_paths(args):
    if args.ct and args.seg:
        return Path(args.ct), Path(args.seg)
    if not args.patient_id:
        raise ValueError("Provide --patient-id, or provide both --ct and --seg.")
    pdir = Path(args.root_dir) / args.patient_id
    return pdir / f"{args.patient_id}_ct.nii.gz", pdir / f"{args.patient_id}_seg-1.nii.gz"


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


def main():
    args = parse_args()
    vertebra = args.vertebra.upper()
    if vertebra not in VERTEBRA_LABELS:
        raise ValueError(f"Unknown vertebra: {args.vertebra}")

    ct_path, seg_path = resolve_paths(args)
    model_path = resolve_model_path(args)
    if not ct_path.exists():
        raise FileNotFoundError(f"CT not found: {ct_path}")
    if not seg_path.exists():
        raise FileNotFoundError(f"Segmentation not found: {seg_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = args.num_classes or infer_num_classes_from_checkpoint(model_path)
    class_names = class_names_for_count(num_classes)
    model = load_stage1_model(model_path, device=device, num_classes=num_classes)
    gt_label = load_ground_truth_label(args.dataset_csv, args.patient_id, vertebra)

    ct, _ = load_img(str(ct_path), canonical=False)
    seg, _ = load_img(str(seg_path), canonical=False)
    patch, mask = build_stage1_patch_and_mask(
        ct,
        seg,
        vertebra,
        patch_size=args.patch_size,
        norm_mode=args.norm_mode,
        zscore_scale=args.zscore_scale,
        foreground_floor=args.foreground_floor,
    )

    target_class = parse_target_class(args.target_class, class_names)
    cam_result = compute_cam(
        model,
        patch,
        target_class=target_class,
        method=args.cam_method,
    )

    probs = cam_result.probabilities[0].tolist()
    pred_name = class_names[cam_result.predicted_class] if cam_result.predicted_class < len(class_names) else str(cam_result.predicted_class)
    target_name = class_names[cam_result.target_class] if cam_result.target_class < len(class_names) else str(cam_result.target_class)

    print(f"patient: {args.patient_id or ct_path.parent.name}")
    print(f"vertebra: {vertebra}")
    print(f"model: {model_path}")
    if gt_label is not None:
        gt_name = class_names[gt_label] if 0 <= gt_label < len(class_names) else str(gt_label)
        print(f"ground truth: {gt_label} ({gt_name})")
    else:
        print("ground truth: unavailable")
    print(f"prediction: {cam_result.predicted_class} ({pred_name})")
    for idx, prob in enumerate(probs):
        name = class_names[idx] if idx < len(class_names) else str(idx)
        print(f"  p[{idx} {name}]: {prob:.4f}")
    print(f"CAM: method={args.cam_method} target={cam_result.target_class} ({target_name})")

    if not args.no_show:
        gt_text = "GT unavailable"
        if gt_label is not None:
            gt_name = class_names[gt_label] if 0 <= gt_label < len(class_names) else str(gt_label)
            gt_text = f"GT {gt_label} {gt_name}"
        show_cam_overlay(
            patch,
            cam_result.cam,
            mask=mask,
            title=(
                f"{args.patient_id or ct_path.parent.name} {vertebra} | {args.cam_method}\n"
                f"{gt_text} | Pred {cam_result.predicted_class} {pred_name} "
                f"({probs[cam_result.predicted_class]:.3f}) | CAM target {cam_result.target_class} {target_name}"
            ),
            cam_threshold=args.cam_threshold,
        )


if __name__ == "__main__":
    main()
