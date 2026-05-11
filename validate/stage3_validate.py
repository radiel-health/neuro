import argparse
from pathlib import Path

import numpy as np
from vedo import Line, Points, Sphere, Text2D, show

from pipeline.scripts.stage3_postlateral import analyze_posterolateral_from_ijk, prepare_ct_seg_arrays, resolve_thresholds
from utils import DEFAULT_DATA_ROOT, VERTEBRA_LABELS, load_img


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize stage 3 posterolateral scoring geometry with vedo.")
    parser.add_argument("--seg", type=str, default="", help="Path to segmentation NIfTI.")
    parser.add_argument("--ct", type=str, default="", help="Path to CT NIfTI if using --seg directly.")
    parser.add_argument("--root-dir", type=str, default=DEFAULT_DATA_ROOT, help="Dataset root directory.")
    parser.add_argument("--patient-id", type=str, default="", help="Patient id to resolve CT/SEG paths.")
    parser.add_argument("--vertebra", type=str, required=True, help="Vertebra name, e.g. T8 or L2.")
    parser.add_argument("--point-size", type=float, default=5.0, help="Point size for rendered clouds.")
    parser.add_argument("--max-points", type=int, default=12000, help="Maximum points per rendered subset.")
    parser.add_argument("--seg-alpha", type=float, default=0.15, help="Opacity for full vertebra cloud.")
    parser.add_argument("--threshold-mode", choices=["auto_patient", "fixed"], default="auto_patient")
    parser.add_argument("--lesion-low-hu", type=float, default=None, help="Optional low HU cutoff override.")
    parser.add_argument("--lesion-high-hu", type=float, default=None, help="Optional high HU cutoff override.")
    return parser.parse_args()


def resolve_paths(args):
    if args.seg:
        if not args.ct:
            raise ValueError("Use --ct together with --seg.")
        return Path(args.ct), Path(args.seg)
    if args.patient_id:
        pdir = Path(args.root_dir) / args.patient_id
        return pdir / f"{args.patient_id}_ct.nii.gz", pdir / f"{args.patient_id}_seg-1.nii.gz"
    raise ValueError("Provide either --patient-id or both --ct and --seg.")


def sample_points(points, max_points):
    if len(points) <= max_points:
        return points
    step = max(int(np.ceil(len(points) / max_points)), 1)
    return points[::step]


def main():
    args = parse_args()
    ct_path, seg_path = resolve_paths(args)
    if not ct_path.exists():
        raise FileNotFoundError(f"CT not found: {ct_path}")
    if not seg_path.exists():
        raise FileNotFoundError(f"Segmentation not found: {seg_path}")

    vname = args.vertebra.upper()
    if vname not in VERTEBRA_LABELS:
        raise ValueError(f"Unknown vertebra: {args.vertebra}")
    label = VERTEBRA_LABELS[vname]

    seg, aff = load_img(str(seg_path), canonical=False)
    ct, _ = load_img(str(ct_path), canonical=False)
    ct, seg = prepare_ct_seg_arrays(ct, seg)
    low_hu, high_hu, threshold_meta = resolve_thresholds(
        ct,
        seg,
        threshold_mode=args.threshold_mode,
        lesion_low_hu=args.lesion_low_hu,
        lesion_high_hu=args.lesion_high_hu,
    )

    ijk = np.argwhere(seg == label)
    if ijk.size == 0:
        raise ValueError(f"Label {vname} is not present in {seg_path}")

    result = analyze_posterolateral_from_ijk(
        ct,
        aff,
        ijk,
        lesion_low_hu=low_hu,
        lesion_high_hu=high_hu,
        threshold_mode=args.threshold_mode,
        threshold_meta=threshold_meta,
    )
    if result is None:
        raise ValueError("Could not compute stage 3 geometry for this vertebra.")

    all_points = Points(sample_points(result["all_xyz"], args.max_points), r=args.point_size).c("lightgray").alpha(args.seg_alpha)
    scored_points = Points(sample_points(result["scored_xyz"], args.max_points), r=max(args.point_size - 1.0, 2.0)).c("silver")
    abnormal_points = Points(sample_points(result["abnormal_xyz"], args.max_points), r=args.point_size).c("gold")
    posterior_points = Points(sample_points(result["posterior_xyz"], args.max_points), r=args.point_size + 0.5).c("deepskyblue")
    left_points = Points(sample_points(result["left_xyz"], args.max_points), r=args.point_size + 1).c("lime")
    right_points = Points(sample_points(result["right_xyz"], args.max_points), r=args.point_size + 1).c("tomato")

    centroid = result["centroid"]
    ref_center = result["reference_center"]
    centroid_actor = Sphere(centroid, r=1.6).c("white")
    ref_actor = Sphere(ref_center, r=2.0).c("violet")
    x_line = Line([result["center_x"], result["center_y"], ref_center[2] - 20], [result["center_x"], result["center_y"], ref_center[2] + 20]).c("white").lw(3)
    split_line = Line([result["center_x"], result["center_y"] - 40, ref_center[2]], [result["center_x"], result["center_y"] + 40, ref_center[2]]).c("white").lw(2)
    posterior_line = Line([result["center_x"] - 30, result["center_y"], ref_center[2]], [result["center_x"] + 30, result["center_y"], ref_center[2]]).c("violet").lw(2)

    title = Text2D(
        (
            f"{seg_path.name} | {vname}\n"
            f"score={result['score']} | ref_x={result['center_x']:.1f} ref_y={result['center_y']:.1f}\n"
            f"used_canal_center={int(result['used_canal_center'])} | centroid_x={result['centroid_x']:.1f} centroid_y={result['centroid_y']:.1f}\n"
            f"threshold_mode={result['threshold_mode']} | HU low/high = {result['lesion_low_hu']:.0f} / {result['lesion_high_hu']:.0f}\n"
            f"patient q01/q50/q95 = {result['threshold_patient_q01']:.0f} / {result['threshold_patient_q50']:.0f} / {result['threshold_patient_q95']:.0f}\n"
            f"raw adaptive low/high = {result['threshold_raw_low']:.0f} / {result['threshold_raw_high']:.0f} | fallback={int(result['threshold_fallback_used'])}\n"
            f"vertebra q05/q50/q95 = {result['ct_q05']:.0f} / {result['ct_q50']:.0f} / {result['ct_q95']:.0f}\n"
            f"bone-like voxels used = {result['bone_voxel_count']} ({100.0 * result['bone_voxel_fraction']:.1f}%)\n"
            f"posterior abnormal total={result['total_posterior_abnormal']} | left={result['left_count']} right={result['right_count']}\n"
            f"logic: raw voxel-space CT/SEG, patient-adaptive HU by default"
        ),
        pos="top-left",
        s=0.9,
    )

    print(f"ct_path: {ct_path}")
    print(f"seg_path: {seg_path}")
    print(f"vertebra: {vname}")
    print(f"score: {result['score']}")
    print(f"left_count: {result['left_count']}")
    print(f"right_count: {result['right_count']}")
    print(f"threshold_mode: {result['threshold_mode']}")
    print(f"HU low/high: {result['lesion_low_hu']:.1f} / {result['lesion_high_hu']:.1f}")
    print(f"patient q01/q50/q95: {result['threshold_patient_q01']:.1f} / {result['threshold_patient_q50']:.1f} / {result['threshold_patient_q95']:.1f}")
    print(f"raw adaptive low/high: {result['threshold_raw_low']:.1f} / {result['threshold_raw_high']:.1f}")
    print(f"threshold fallback used: {int(result['threshold_fallback_used'])}")
    print(f"vertebra q05/q50/q95: {result['ct_q05']:.1f} / {result['ct_q50']:.1f} / {result['ct_q95']:.1f}")
    print(f"bone-like voxels used: {result['bone_voxel_count']} ({100.0 * result['bone_voxel_fraction']:.1f}%)")

    show(
        all_points,
        scored_points,
        abnormal_points,
        posterior_points,
        left_points,
        right_points,
        centroid_actor,
        ref_actor,
        x_line,
        split_line,
        posterior_line,
        title,
        axes=1,
        bg="black",
        bg2="gray3",
        title="Stage 3 Validation",
        interactive=True,
    )


if __name__ == "__main__":
    main()
