import argparse
from pathlib import Path

import numpy as np
from vedo import Line, Points, Sphere, Text2D, show

from stage2_collapse import analyze_height_geometry, build_label_ijk_map
from utils import DEFAULT_DATA_ROOT, VERTEBRA_LABELS, load_seg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize stage 2 vertebral height measurement geometry with vedo."
    )
    parser.add_argument("--seg", type=str, default="", help="Path to a segmentation NIfTI.")
    parser.add_argument("--root-dir", type=str, default=DEFAULT_DATA_ROOT, help="Dataset root directory.")
    parser.add_argument("--patient-id", type=str, default="", help="Patient id to resolve seg path from root dir.")
    parser.add_argument("--vertebra", type=str, required=True, help="Vertebra name, e.g. T8 or L2.")
    parser.add_argument("--seg-alpha", type=float, default=0.35, help="Segmentation surface opacity.")
    parser.add_argument("--point-size", type=float, default=5.0, help="Point size for sampled voxel clouds.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=12000,
        help="Maximum voxel points to render per cloud to keep the viewer responsive.",
    )
    return parser.parse_args()


def resolve_seg_path(args):
    if args.seg:
        return Path(args.seg)
    if args.patient_id:
        return Path(args.root_dir) / args.patient_id / f"{args.patient_id}_seg-1.nii.gz"
    raise ValueError("Provide either --seg or --patient-id.")


def sample_points(points, max_points):
    if len(points) <= max_points:
        return points
    step = max(int(np.ceil(len(points) / max_points)), 1)
    return points[::step]


def format_basis(basis):
    rows = []
    for i, name in enumerate(["LR", "AP", "SI"]):
        vec = basis[:, i]
        rows.append(f"{name}=({vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f})")
    return " | ".join(rows)


def main():
    args = parse_args()
    seg_path = resolve_seg_path(args)
    if not seg_path.exists():
        raise FileNotFoundError(f"Segmentation not found: {seg_path}")

    vname = args.vertebra.upper()
    if vname not in VERTEBRA_LABELS:
        raise ValueError(f"Unknown vertebra: {args.vertebra}")
    label = VERTEBRA_LABELS[vname]

    seg, aff = load_seg(str(seg_path))
    label_ijk = build_label_ijk_map(seg)
    ijk = label_ijk.get(label)
    if ijk is None or ijk.size == 0:
        raise ValueError(f"Label {vname} is not present in {seg_path}")

    result = analyze_height_geometry(ijk, aff)
    if result is None:
        raise ValueError("Could not compute stage 2 height geometry for this vertebra.")

    all_points = Points(sample_points(result["xyz"], args.max_points), r=args.point_size).c("lightgray").alpha(args.seg_alpha)
    slab_points = Points(sample_points(result["slab_xyz"], args.max_points), r=args.point_size).c("gold")
    body_points = Points(sample_points(result["body_xyz"], args.max_points), r=args.point_size + 0.5).c("deepskyblue")
    ant_points = Points(sample_points(result["ant"]["points"], args.max_points), r=args.point_size + 1).c("lime")
    post_points = Points(sample_points(result["post"]["points"], args.max_points), r=args.point_size + 1).c("tomato")
    canal_points = Points(sample_points(result["canal"]["points_world"], args.max_points), r=args.point_size + 1).c("magenta")
    canal_center = Sphere(result["canal"]["center_world"], r=1.8).c("magenta")

    ant_line = Line(result["ant"]["p0"], result["ant"]["p1"]).c("lime").lw(6)
    post_line = Line(result["post"]["p0"], result["post"]["p1"]).c("tomato").lw(6)
    ant_top = Sphere(result["ant"]["p1"], r=1.5).c("lime")
    ant_bottom = Sphere(result["ant"]["p0"], r=1.5).c("lime")
    post_top = Sphere(result["post"]["p1"], r=1.5).c("tomato")
    post_bottom = Sphere(result["post"]["p0"], r=1.5).c("tomato")

    title = Text2D(
        (
            f"{seg_path.name} | {vname}\n"
            f"local frame: {format_basis(result['frame_basis'])}\n"
            f"local center-x slab: |x - {result['center_x']:.1f}| < 15 mm\n"
            f"local mid-z band used to estimate body walls: [{result['z_mid_lo']:.1f}, {result['z_mid_hi']:.1f}] mm\n"
            f"local canal center: x={result['canal']['center_x']:.1f}, y={result['canal']['center_y']:.1f} mm\n"
            f"local body band y=[{result['body_y_min']:.1f}, {result['body_y_max']:.1f}] mm\n"
            f"local anterior y={result['y_ant']:.1f} height={result['ant']['height_mm']:.1f} mm "
            f"(slab +/- {result['ant']['width_mm']:.0f} mm)\n"
            f"local posterior y={result['y_post']:.1f} height={result['post']['height_mm']:.1f} mm "
            f"(slab +/- {result['post']['width_mm']:.0f} mm)"
        ),
        pos="top-left",
        s=0.9,
    )

    print(f"seg_path: {seg_path}")
    print(f"vertebra: {vname}")
    print(f"anterior height: {result['ant']['height_mm']:.2f} mm")
    print(f"posterior height: {result['post']['height_mm']:.2f} mm")

    show(
        all_points,
        slab_points,
        body_points,
        ant_points,
        post_points,
        canal_points,
        canal_center,
        ant_line,
        post_line,
        ant_top,
        ant_bottom,
        post_top,
        post_bottom,
        title,
        axes=1,
        bg="black",
        bg2="gray3",
        title="Stage 2 Validation",
        interactive=True,
    )


if __name__ == "__main__":
    main()
