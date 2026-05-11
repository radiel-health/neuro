import os
import argparse
import numpy as np
import pandas as pd
from scipy import ndimage
from utils import VERTEBRA_LABELS, INV_LABELS, load_seg, vox2world

LIT_FILE = "data/vertebra_height.csv"
HEIGHT_LOW_PERCENTILE = 5.0
HEIGHT_HIGH_PERCENTILE = 95.0


def _compute_local_frame(xyz, target_basis=None):
    center = np.median(xyz, axis=0)
    centered = xyz - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    pca_axes = vh.copy()

    if target_basis is None:
        target_basis = np.eye(3, dtype=np.float64)

    remaining = list(range(3))
    ordered = []
    for target_idx in [0, 1, 2]:
        target = target_basis[:, target_idx]
        best_idx = max(remaining, key=lambda i: abs(np.dot(pca_axes[i], target)))
        vec = pca_axes[best_idx].copy()
        if np.dot(vec, target) < 0:
            vec *= -1.0
        ordered.append(vec)
        remaining.remove(best_idx)

    basis = np.stack(ordered, axis=1)
    local_xyz = centered @ basis
    return center, basis, local_xyz


def _project_to_frame(xyz, center, basis):
    xyz = np.asarray(xyz, dtype=np.float64)
    return (xyz - center) @ basis


def _to_world(local_pts, center, basis):
    local_pts = np.asarray(local_pts, dtype=np.float64)
    if local_pts.ndim == 1:
        local_pts = local_pts[None, :]
    return local_pts @ basis.T + center


MAX_TILT_CORRECTION_DEG = 5


def _normalize(vec):
    vec = np.asarray(vec, dtype=np.float64)
    n = np.linalg.norm(vec)
    if n < 1e-8:
        return vec
    return vec / n


def _rotate_toward(base_vec, target_vec, max_angle_deg):
    base = _normalize(base_vec)
    target = _normalize(target_vec)
    dot = float(np.clip(np.dot(base, target), -1.0, 1.0))
    angle = float(np.arccos(dot))
    max_angle = float(np.deg2rad(max_angle_deg))
    if angle < 1e-8:
        return base
    if angle <= max_angle:
        return target
    ortho = target - dot * base
    ortho_n = np.linalg.norm(ortho)
    if ortho_n < 1e-8:
        return base
    ortho /= ortho_n
    return _normalize(np.cos(max_angle) * base + np.sin(max_angle) * ortho)


def _basis_from_si(initial_basis, si_vec):
    si_vec = _normalize(si_vec)
    lr_seed = initial_basis[:, 0] - np.dot(initial_basis[:, 0], si_vec) * si_vec
    if np.linalg.norm(lr_seed) < 1e-8:
        lr_seed = initial_basis[:, 1] - np.dot(initial_basis[:, 1], si_vec) * si_vec
    lr_vec = _normalize(lr_seed)
    ap_vec = _normalize(np.cross(si_vec, lr_vec))
    lr_vec = _normalize(np.cross(ap_vec, si_vec))
    return np.stack([lr_vec, ap_vec, si_vec], axis=1)


def _refine_basis_mildly(initial_basis, body_xyz, max_angle_deg=MAX_TILT_CORRECTION_DEG):
    _, _, vh = np.linalg.svd(body_xyz - np.median(body_xyz, axis=0), full_matrices=False)
    body_pca = vh.copy()
    si_target = max(body_pca, key=lambda v: abs(np.dot(v, initial_basis[:, 2])))
    if np.dot(si_target, initial_basis[:, 2]) < 0:
        si_target *= -1.0
    si_vec = _rotate_toward(initial_basis[:, 2], si_target, max_angle_deg)
    return _basis_from_si(initial_basis, si_vec)


def _detect_canal_hole(mid_xyz, center_x):
    x = mid_xyz[:, 0]
    y = mid_xyz[:, 1]
    if len(mid_xyz) < 100:
        return None

    bin_width = 1.5
    x_edges = np.arange(float(x.min()), float(x.max()) + bin_width, bin_width, dtype=np.float64)
    y_edges = np.arange(float(y.min()), float(y.max()) + bin_width, bin_width, dtype=np.float64)
    if len(x_edges) < 6 or len(y_edges) < 6:
        return None

    hist, x_edges, y_edges = np.histogram2d(x, y, bins=(x_edges, y_edges))
    occ = hist >= 1
    occ = ndimage.binary_closing(occ, structure=np.ones((3, 3), dtype=bool))
    filled = ndimage.binary_fill_holes(occ)
    holes = filled & (~occ)
    labels, num = ndimage.label(holes)
    if num <= 0:
        return None

    best = None
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    global_y_mid = float(np.median(y))
    for lab in range(1, num + 1):
        mask = labels == lab
        area = int(mask.sum())
        if area < 12:
            continue
        xi, yi = np.where(mask)
        hx = x_centers[xi]
        hy = y_centers[yi]
        cx = float(np.mean(hx))
        cy = float(np.mean(hy))
        if cy >= global_y_mid:
            continue
        score = (-abs(cx - center_x), area, -abs(cy - global_y_mid))
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "center_x": cx,
                "center_y": cy,
                "x_min": float(hx.min()),
                "x_max": float(hx.max()),
                "y_min": float(hy.min()),
                "y_max": float(hy.max()),
                "points": np.column_stack([hx, hy]),
            }
    return best


def _select_body_subset(slab_xyz, center_x):
    if len(slab_xyz) < 200:
        return None

    y = slab_xyz[:, 1]
    z = slab_xyz[:, 2]
    if float(y.max()) <= float(y.min()) or float(z.max()) <= float(z.min()):
        return None

    z_lo = float(np.percentile(z, 30))
    z_hi = float(np.percentile(z, 70))
    mid_keep = (z >= z_lo) & (z <= z_hi)
    mid_xyz = slab_xyz[mid_keep]
    if len(mid_xyz) < 100:
        mid_xyz = slab_xyz

    canal = _detect_canal_hole(mid_xyz, center_x)
    if canal is None:
        return None

    z_centers = np.arange(float(mid_xyz[:, 2].min()), float(mid_xyz[:, 2].max()) + 2.0, 2.0, dtype=np.float64)
    posterior_edges = []
    anterior_edges = []
    x_half_width = max((canal["x_max"] - canal["x_min"]) * 0.75, 6.0)

    for zc in z_centers:
        slice_mask = np.abs(mid_xyz[:, 2] - zc) < 1.5
        slice_xyz = mid_xyz[slice_mask]
        if len(slice_xyz) < 40:
            continue

        central = slice_xyz[np.abs(slice_xyz[:, 0] - canal["center_x"]) <= x_half_width]
        if len(central) < 20:
            continue

        anterior_of_canal = central[central[:, 1] >= canal["y_max"]]
        if len(anterior_of_canal) < 15:
            continue

        posterior_edges.append(float(np.percentile(anterior_of_canal[:, 1], 5)))
        anterior_edges.append(float(np.percentile(anterior_of_canal[:, 1], 95)))

    if len(posterior_edges) < 5 or len(anterior_edges) < 5:
        return None

    posterior_edges = np.asarray(posterior_edges, dtype=np.float64)
    anterior_edges = np.asarray(anterior_edges, dtype=np.float64)

    body_y_min = float(np.percentile(posterior_edges, 50))
    body_y_max = float(np.percentile(anterior_edges, 50))
    if body_y_max <= body_y_min:
        return None

    keep = (slab_xyz[:, 1] >= body_y_min) & (slab_xyz[:, 1] <= body_y_max)
    body_xyz = slab_xyz[keep]
    if len(body_xyz) < 100:
        return None

    body_xz = body_xyz[:, [0, 2]]
    center = np.median(body_xz, axis=0)
    mad = np.median(np.abs(body_xz - center), axis=0)
    scale = np.maximum(mad * 1.4826, 1.0)
    xz_norm = ((body_xz - center) / scale) ** 2
    compact_keep = xz_norm.sum(axis=1) <= 9.0
    compact_xyz = body_xyz[compact_keep]
    if len(compact_xyz) >= 100:
        body_xyz = compact_xyz

    body_z_min = float(body_xyz[:, 2].min())
    body_z_max = float(body_xyz[:, 2].max())
    return body_xyz, body_y_min, body_y_max, body_z_min, body_z_max, float(z_lo), float(z_hi), canal


def analyze_height_geometry(ijk, aff):
    if ijk.size == 0:
        return None

    xyz = vox2world(aff, ijk)

    frame_center = np.median(xyz, axis=0)
    initial_basis = np.eye(3, dtype=np.float64)
    local_xyz = xyz - frame_center
    x, y, z = local_xyz[:, 0], local_xyz[:, 1], local_xyz[:, 2]
    initial_cx = np.median(x)
    initial_keep = np.abs(x - initial_cx) < 15
    initial_slab_local_xyz = local_xyz[initial_keep]
    if len(initial_slab_local_xyz) < 200:
        return None

    initial_body_result = _select_body_subset(initial_slab_local_xyz, float(initial_cx))
    if initial_body_result is None:
        return None
    initial_body_local_xyz = initial_body_result[0]
    initial_body_xyz = _to_world(initial_body_local_xyz, frame_center, initial_basis)

    frame_basis = _refine_basis_mildly(initial_basis, initial_body_xyz)
    local_xyz = _project_to_frame(xyz, frame_center, frame_basis)
    x, y, z = local_xyz[:, 0], local_xyz[:, 1], local_xyz[:, 2]
    cx = np.median(x)
    keep = np.abs(x - cx) < 15

    slab_local_xyz = local_xyz[keep]
    if len(slab_local_xyz) < 200:
        return None

    body_result = _select_body_subset(slab_local_xyz, float(cx))
    if body_result is None:
        return None
    body_local_xyz, body_y_min, body_y_max, body_z_min, body_z_max, z_mid_lo, z_mid_hi, canal = body_result
    body_y = body_local_xyz[:, 1]

    y_ant = np.percentile(body_y, 90)
    y_post = np.percentile(body_y, 10)

    def height_at_y(y0):
        slab = np.abs(body_y - y0) < 3
        slab_width = 3.0
        if slab.sum() < 20:
            slab = np.abs(body_y - y0) < 6
            slab_width = 6.0
        if slab.sum() < 10:
            return None
        local_pts = body_local_xyz[slab]
        zz = local_pts[:, 2]
        z_low = float(np.percentile(zz, HEIGHT_LOW_PERCENTILE))
        z_high = float(np.percentile(zz, HEIGHT_HIGH_PERCENTILE))
        p0_local = np.array([np.median(local_pts[:, 0]), np.median(local_pts[:, 1]), z_low], dtype=np.float64)
        p1_local = np.array([np.median(local_pts[:, 0]), np.median(local_pts[:, 1]), z_high], dtype=np.float64)
        return {
            "height_mm": z_high - z_low,
            "points_local": local_pts,
            "points": _to_world(local_pts, frame_center, frame_basis),
            "width_mm": slab_width,
            "z_low": z_low,
            "z_high": z_high,
            "p0_local": p0_local,
            "p1_local": p1_local,
            "p0": _to_world(p0_local, frame_center, frame_basis)[0],
            "p1": _to_world(p1_local, frame_center, frame_basis)[0],
        }

    h_ant = height_at_y(y_ant)
    h_post = height_at_y(y_post)

    if h_ant is None or h_post is None:
        return None

    slab_xyz = _to_world(slab_local_xyz, frame_center, frame_basis)
    body_xyz = _to_world(body_local_xyz, frame_center, frame_basis)
    canal_local_pts = np.column_stack(
        [
            canal["points"][:, 0],
            canal["points"][:, 1],
            np.full(len(canal["points"]), 0.5 * (z_mid_lo + z_mid_hi), dtype=np.float64),
        ]
    )
    canal_world_pts = _to_world(canal_local_pts, frame_center, frame_basis)
    canal_world_center = _to_world(
        np.array([canal["center_x"], canal["center_y"], 0.5 * (z_mid_lo + z_mid_hi)], dtype=np.float64),
        frame_center,
        frame_basis,
    )[0]

    return {
        "xyz": xyz,
        "local_xyz": local_xyz,
        "slab_xyz": slab_xyz,
        "slab_local_xyz": slab_local_xyz,
        "body_xyz": body_xyz,
        "body_local_xyz": body_local_xyz,
        "frame_center": frame_center,
        "frame_basis": frame_basis,
        "initial_frame_center": frame_center,
        "initial_frame_basis": initial_basis,
        "center_x": float(cx),
        "body_y_min": body_y_min,
        "body_y_max": body_y_max,
        "body_z_min": body_z_min,
        "body_z_max": body_z_max,
        "z_mid_lo": z_mid_lo,
        "z_mid_hi": z_mid_hi,
        "canal": {
            **canal,
            "points_local": canal_local_pts,
            "points_world": canal_world_pts,
            "center_world": canal_world_center,
        },
        "y_ant": float(y_ant),
        "y_post": float(y_post),
        "ant": h_ant,
        "post": h_post,
    }


def measure_height_from_ijk(ijk, aff):
    result = analyze_height_geometry(ijk, aff)
    if result is None:
        return None
    return result["ant"]["height_mm"], result["post"]["height_mm"]


def measure_height(seg, aff, label):
    ijk = np.argwhere(seg == label)
    return measure_height_from_ijk(ijk, aff)


def build_label_ijk_map(seg):
    """Build per-label voxel coordinates with a single pass over segmentation."""
    fg = np.argwhere(seg > 0)
    if fg.size == 0:
        return {}
    labels = seg[fg[:, 0], fg[:, 1], fg[:, 2]].astype(np.int16)
    out = {}
    for lab in range(1, 18):
        m = labels == lab
        if np.any(m):
            out[lab] = fg[m]
    return out


def collapse_score(r):
    if r < 0.5:
        return 3
    if r < 0.8:
        return 2
    return 0


def load_lit():
    if not os.path.exists(LIT_FILE):
        return {}
    df = pd.read_csv(LIT_FILE)
    return {
        r["vertebra"]: (
            r["mean_anterior_mm"],
            r["mean_posterior_mm"],
        )
        for _, r in df.iterrows()
    }


def run_batch(root):
    lit = load_lit()
    rows = []

    for pid in os.listdir(root):
        pdir = os.path.join(root, pid)
        if not os.path.isdir(pdir):
            continue
        seg_path = os.path.join(
            pdir,
            f"{pid}_seg-1.nii.gz"
        )
        if not os.path.exists(seg_path):
            continue

        seg, aff = load_seg(seg_path)
        label_ijk = build_label_ijk_map(seg)
        for lab in range(1, 18):
            ijk = label_ijk.get(lab, np.empty((0, 3), dtype=np.int64))
            res = measure_height_from_ijk(ijk, aff)
            if res is None:
                continue
            h_ant, h_post = res
            rows.append({
                "patient_id": pid,
                "vertebra": INV_LABELS[lab],
                "ant": h_ant,
                "post": h_post,
            })

    df = pd.DataFrame(rows)
    mean = df.groupby("vertebra")[["ant", "post"]].mean()
    mean_map = {
        k: (v["ant"], v["post"])
        for k, v in mean.to_dict("index").items()
    }

    out = []

    for _, r in df.iterrows():
        v = r["vertebra"]
        ant = r["ant"]
        post = r["post"]

        c_ant, c_post = mean_map[v]

        if v in lit:
            l_ant, l_post = lit[v]
        else:
            l_ant, l_post = c_ant, c_post

        normal_ant = max(c_ant, l_ant)
        normal_post = max(c_post, l_post)

        ratio_ant = ant / normal_ant
        ratio_post = post / normal_post

        worst = min(ratio_ant, ratio_post)

        score = collapse_score(worst)

        out.append({
            "patient_id": r["patient_id"],
            "vertebra": v,
            "ant": ant,
            "post": post,
            "ratio": worst,
            "collapse_score": score,
        })

    out = pd.DataFrame(out)
    out.to_csv("output/stage2_collapse.csv", index=False)
    print("saved output/stage2_collapse.csv")


def run_single(seg_path, vertebra):
    seg, aff = load_seg(seg_path)
    label = VERTEBRA_LABELS[vertebra]
    res = measure_height(seg, aff, label)
    print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str)
    parser.add_argument("--seg", type=str)
    parser.add_argument("--vertebra", type=str)

    args = parser.parse_args()

    if args.root:
        run_batch(args.root)
    elif args.seg:
        run_single(args.seg, args.vertebra)
