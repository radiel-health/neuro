import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from vedo import Line, Points, Text2D, Volume, show

from pipeline.scripts.stage4_alignment import analyze_patient_alignment_geometry
from utils import DEFAULT_DATA_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize stage 4 spine alignment geometry.")
    parser.add_argument("--patient-id", type=str, default="", help="Patient id to resolve CT/SEG paths.")
    parser.add_argument("--ct", type=str, default="", help="Path to CT NIfTI.")
    parser.add_argument("--seg", type=str, default="", help="Path to segmentation NIfTI.")
    parser.add_argument("--root-dir", type=str, default=DEFAULT_DATA_ROOT, help="Dataset root directory.")
    parser.add_argument("--translation-thresh-mm", type=float, default=6.0)
    parser.add_argument("--scoliosis-max-dev-mm", type=float, default=10.0)
    parser.add_argument("--kyphosis-max-dev-mm", type=float, default=10.0)
    parser.add_argument("--ct-alpha", type=float, default=0.08, help="CT volume opacity.")
    parser.add_argument("--ct-wl", type=float, default=300.0, help="CT window level.")
    parser.add_argument("--ct-ww", type=float, default=1500.0, help="CT window width.")
    parser.add_argument("--seg-alpha", type=float, default=0.12, help="Segmentation surface opacity.")
    parser.add_argument("--point-size", type=float, default=10.0, help="Centroid point size.")
    return parser.parse_args()


def resolve_paths(args):
    if args.ct and args.seg:
        return Path(args.ct), Path(args.seg)
    if args.patient_id:
        pdir = Path(args.root_dir) / args.patient_id
        return pdir / f"{args.patient_id}_ct.nii.gz", pdir / f"{args.patient_id}_seg-1.nii.gz"
    raise ValueError("Provide --patient-id or both --ct and --seg.")


def load_volume(path):
    img = nib.as_closest_canonical(nib.load(str(path)))
    return np.asanyarray(img.dataobj).astype(np.float32), img.affine, img


def world_to_voxel_points(xyz, aff):
    inv = np.linalg.inv(aff)
    xyz = np.asarray(xyz, dtype=np.float64)
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
    ijk_h = (inv @ xyz_h.T).T
    return ijk_h[:, :3].astype(np.float32)


def add_lines_between(points_a, points_b, color, width=3):
    actors = []
    for a, b in zip(points_a, points_b):
        actors.append(Line(a, b).c(color).lw(width))
    return actors


def main():
    args = parse_args()
    ct_path, seg_path = resolve_paths(args)
    if not ct_path.exists():
        raise FileNotFoundError(f"CT not found: {ct_path}")
    if not seg_path.exists():
        raise FileNotFoundError(f"Segmentation not found: {seg_path}")

    result = analyze_patient_alignment_geometry(
        str(seg_path),
        translation_thresh_mm=args.translation_thresh_mm,
        scoliosis_max_dev_mm=args.scoliosis_max_dev_mm,
        kyphosis_max_dev_mm=args.kyphosis_max_dev_mm,
    )
    if result is None:
        raise ValueError("Could not compute stage 4 alignment geometry; not enough vertebrae were found.")

    ct, ct_aff, ct_img = load_volume(ct_path)
    seg_img = nib.Nifti1Image(result["seg"].astype(np.int16), result["affine"])
    seg_img = nib.as_closest_canonical(seg_img)
    if seg_img.shape != ct_img.shape or not np.allclose(seg_img.affine, ct_img.affine, atol=1e-5):
        seg_img = resample_from_to(seg_img, ct_img, order=0)
    seg = np.asanyarray(seg_img.dataobj).astype(np.int16)

    centers = world_to_voxel_points(result["centers"], ct_aff)
    fitted = world_to_voxel_points(result["fitted"], ct_aff)
    fitted_dense = world_to_voxel_points(result["fitted_dense"], ct_aff)
    summary = result["summary"]

    wl = float(args.ct_wl)
    ww = max(float(args.ct_ww), 1.0)
    ct_vis = np.clip(ct, wl - ww / 2.0, wl + ww / 2.0)
    actors = [
        Volume(ct_vis)
        .cmap("bone")
        .alpha([0.0, 0.0, float(args.ct_alpha) * 0.4, float(args.ct_alpha)])
    ]

    if np.any(seg > 0):
        try:
            actors.append(Volume((seg > 0).astype(np.uint8)).isosurface(0.5).c("lightgray").alpha(args.seg_alpha))
        except Exception:
            pass

    actors.append(Points(centers, r=args.point_size).c("deepskyblue"))
    actors.append(Points(fitted, r=max(args.point_size - 2.0, 4.0)).c("gold"))

    for i in range(len(fitted_dense) - 1):
        actors.append(Line(fitted_dense[i], fitted_dense[i + 1]).c("gold").lw(5))
    actors.extend(add_lines_between(centers, fitted, color="tomato", width=3))

    flag_labels = {flag["vertebra"] for flag in result["flags"]}
    if flag_labels:
        flag_points = [
            centers[i]
            for i, name in enumerate(result["vertebrae"])
            if name in flag_labels
        ]
        if flag_points:
            actors.append(Points(np.asarray(flag_points), r=args.point_size + 4.0).c("red"))

    title = Text2D(
        (
            f"{ct_path.parent.name} | Stage 4 Alignment\n"
            f"score={summary['alignment_score']} | n={summary['n_vertebrae_found']} | "
            f"coronal max dev={summary['coronal_max_dev_mm']:.1f} mm | "
            f"sagittal max dev={summary['sagittal_max_dev_mm']:.1f} mm\n"
            f"centers=blue | fitted cubic sagittal curve=gold | center-to-curve deviation=red lines | flagged=red"
        ),
        pos="top-left",
        s=0.8,
    )
    actors.append(title)

    print(f"ct_path: {ct_path}")
    print(f"seg_path: {seg_path}")
    print(f"alignment_score: {summary['alignment_score']}")
    print(f"coronal_max_dev_mm: {summary['coronal_max_dev_mm']:.2f}")
    print(f"sagittal_max_dev_mm: {summary['sagittal_max_dev_mm']:.2f}")
    print(f"has_subluxation: {summary['has_subluxation']}")
    print(f"has_curve_deviation: {summary['has_curve_deviation']}")
    for name, dev in zip(result["vertebrae"], result["deviation_mm"]):
        print(f"{name}: center_to_curve={dev:.2f} mm")
    if result["flags"]:
        print("flags:")
        for flag in result["flags"]:
            print(f"  {flag['vertebra']}: disp={flag['disp_mm']:.2f} mm dx={flag['dx_mm']:.2f} dy={flag['dy_mm']:.2f}")

    show(
        *actors,
        axes=1,
        bg="black",
        bg2="gray3",
        title="Stage 4 Validation",
        interactive=True,
    )


if __name__ == "__main__":
    main()
