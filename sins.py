import os
import argparse
import torch
import numpy as np

from cams import (
    build_stage1_patch_mask_and_coords,
    class_names_for_count,
    compute_cam,
    load_stage1_model,
    show_full_ct_cam_overlay,
)
from utils import (
    INV_LABELS,
    DEFAULT_DATA_ROOT,
    load_img,
)

from stage2_collapse import measure_height, collapse_score
from stage3_postlateral import (
    posterolateral_score_from_ijk,
    prepare_ct_seg_arrays,
    resolve_thresholds,
)
from stage4_alignment import compute_patient_alignment

ROOT = DEFAULT_DATA_ROOT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_stage1(model_path=None):
    if model_path is None:
        model_path = "output/stage1/4_class/2026-03-20/last.pth"
    return load_stage1_model(model_path, device=DEVICE, num_classes=4)


def location_score(v):
    if v in ["T1", "T2", "T11", "T12", "L1", "L5"]:
        return 3
    if v in ["L2", "L3", "L4"]:
        return 2
    if v in ["T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]:
        return 1
    return 0


def bone_score(label):
    if label == 2:
        return 1
    if label == 3:
        return 2
    return 0


def run(patient_id, model_path=None, show_cam=False, cam_method="gradcam"):
    model = load_stage1(model_path)
    class_names = class_names_for_count(4)
    pdir = os.path.join(ROOT, patient_id)
    ct_path = os.path.join(pdir, f"{patient_id}_ct.nii.gz")
    seg_path = os.path.join(pdir, f"{patient_id}_seg-1.nii.gz")

    ct, aff = load_img(ct_path, canonical=True)
    seg, _ = load_img(seg_path, canonical=True)
    ct, seg = prepare_ct_seg_arrays(ct, seg)
    low_hu, high_hu, threshold_meta = resolve_thresholds(ct, seg)

    align_result = compute_patient_alignment(
        seg_path,
        6,
        10,
        10,
    )
    align_score = 0
    if align_result is not None:
        align_summary, _ = align_result
        align_score = align_summary["alignment_score"]

    results = []
    cam_points = []
    cam_summary = []
    stage1_predictions = []
    for lab in range(1, 18):
        vname = INV_LABELS[lab]
        if not np.any(seg == lab):
            continue

        patch, patch_mask, patch_coords = build_stage1_patch_mask_and_coords(ct, seg, vname)
        patch = patch.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(patch)
            probs = torch.softmax(out, dim=1)
            cls = out.argmax().item()
            pred_name = class_names[cls] if cls < len(class_names) else str(cls)
            stage1_predictions.append(f"{vname}: {pred_name} {float(probs[0, cls]):.2f}")

        if show_cam:
            cam_result = compute_cam(model, patch.detach(), method=cam_method)
            coords = patch_coords[patch_mask].astype(np.float32)
            values = cam_result.cam[patch_mask].astype(np.float32)
            cam_points.append({"vertebra": vname, "coords": coords, "values": values})
            pred_name = class_names[cam_result.predicted_class] if cam_result.predicted_class < len(class_names) else str(cam_result.predicted_class)
            cam_summary.append(
                f"{vname}: {pred_name} {float(cam_result.probabilities[0, cam_result.predicted_class]):.2f}"
            )

        if cls == 0:
            continue

        h = measure_height(seg, aff, lab)
        if h is None:
            collapse = 0
        else:
            ant, post = h
            ratio = min(ant, post) / max(ant, post)
            collapse = collapse_score(ratio)

        ijk = np.argwhere(seg == lab)
        post_score = posterolateral_score_from_ijk(
            ct,
            aff,
            ijk,
            lesion_low_hu=low_hu,
            lesion_high_hu=high_hu,
            threshold_meta=threshold_meta,
        )

        loc = location_score(vname)
        bone = bone_score(cls)

        sins = loc + bone + collapse + post_score + align_score
        sins_category = "Stable"
        if sins >= 10:
            sins_category = "Unstable"
        elif sins >= 4:
            sins_category = "Potentially Unstable"

        results.append({
            "vertebra": vname,
            "SINS": sins,
            "SINS_category": sins_category,
            "loc": loc,
            "bone": bone,
            "collapse": collapse,
            "posterolateral": post_score,
            "align": align_score,
        })

    results = sorted(results, key=lambda x: x["SINS"], reverse=True)

    if show_cam and stage1_predictions:
        print("Stage1 predictions")
        print(" | ".join(stage1_predictions[:8]))
        if len(stage1_predictions) > 8:
            print(" | ".join(stage1_predictions[8:]))

    if not results:
        print("No suspicious vertebrae found.")
    else:
        print("Top 3")
        for r in results[:3]:
            print(
                f"{r['vertebra']}: SINS={r['SINS']} "
                f"({r['SINS_category']}) | "
                f"loc={r['loc']} bone={r['bone']} collapse={r['collapse']} "
                f"posterolateral={r['posterolateral']} align={r['align']}"
            )

    if show_cam:
        if not cam_points:
            print("[WARN] No vertebrae found for full-volume CAM overlay.")
        else:
            print(f"Showing stage1 CAMs over the original CT volume for {len(cam_points)} vertebrae.")
            show_full_ct_cam_overlay(
                ct,
                seg,
                cam_points,
                title=(
                    f"{patient_id} | original CT + stage1 {cam_method} CAMs | no CAM threshold\n"
                    + " | ".join(cam_summary[:8])
                    + ("\n" + " | ".join(cam_summary[8:16]) if len(cam_summary) > 8 else "")
                ),
                cam_threshold=None,
                show_seg_context=False,
            )

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run the SINS pipeline for one patient.")
    parser.add_argument("patient_id", nargs="?", default="", help="Patient id. If omitted, prompts interactively.")
    parser.add_argument("--model-path", type=str, default="", help="Stage1 checkpoint path.")
    parser.add_argument("--show-cam", action="store_true", help="Show every vertebra stage1 CAM over the original CT volume.")
    parser.add_argument("--cam-method", type=str, default="gradcam", choices=["gradcam", "gradcam++", "layercam"], help="CAM method.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pid = args.patient_id or input("Patient ID: ")
    run(
        pid,
        model_path=args.model_path or None,
        show_cam=args.show_cam,
        cam_method=args.cam_method,
    )
