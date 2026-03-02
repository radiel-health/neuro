import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

VERTEBRA_LABELS = {
    "T1": 1, "T2": 2, "T3": 3, "T4": 4, "T5": 5, "T6": 6, "T7": 7, "T8": 8, "T9": 9, "T10": 10,
    "T11": 11, "T12": 12,
    "L1": 13, "L2": 14, "L3": 15, "L4": 16, "L5": 17,
}
INV_LABELS = {v: k for k, v in VERTEBRA_LABELS.items()}
VERTEBRAE = list(VERTEBRA_LABELS.keys())

DEFAULT_DATA_ROOT = "data/Spine-Mets-CT-SEG-Nifti"

def load_img(path, canonical=True):
    img = nib.load(path)
    if canonical:
        img = nib.as_closest_canonical(img)
    data = img.get_fdata()
    aff = img.affine
    return data, aff

def load_seg(seg_path):
    seg, aff = load_img(seg_path, canonical=True)
    return seg.astype(np.int16), aff

def vox2world(aff, ijk):
    ijk_h = np.concatenate(
        [ijk, np.ones((ijk.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    xyz_h = (aff @ ijk_h.T).T
    return xyz_h[:, :3]

def crop_from_mask(vol, mask, label, margin=5):
    coords = np.where(mask == label)
    if len(coords[0]) == 0:
        return None

    z0, z1 = coords[0].min(), coords[0].max()
    y0, y1 = coords[1].min(), coords[1].max()
    x0, x1 = coords[2].min(), coords[2].max()

    z0 = max(0, z0 - margin)
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)

    z1 = min(vol.shape[0], z1 + margin)
    y1 = min(vol.shape[1], y1 + margin)
    x1 = min(vol.shape[2], x1 + margin)

    return vol[z0:z1, y0:y1, x0:x1]

def resize_patch(patch, size=(64, 64, 64)):
    t = torch.tensor(patch).float()[None, None]
    t = F.interpolate(
        t,
        size=size,
        mode="trilinear",
        align_corners=False,
    )
    return t[0]
