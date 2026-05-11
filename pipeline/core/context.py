from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils import DEFAULT_DATA_ROOT, load_img


@dataclass
class PatientContext:
    patient_id: str
    ct: np.ndarray
    seg: np.ndarray
    affine: np.ndarray
    ct_path: Path
    seg_path: Path

    @classmethod
    def load(cls, patient_id, root_dir=DEFAULT_DATA_ROOT, canonical=True):
        root = Path(root_dir)
        patient_id = str(patient_id)
        patient_dir = root / patient_id
        ct_path = patient_dir / f"{patient_id}_ct.nii.gz"
        seg_path = patient_dir / f"{patient_id}_seg-1.nii.gz"
        if not ct_path.exists():
            raise FileNotFoundError(f"CT not found: {ct_path}")
        if not seg_path.exists():
            raise FileNotFoundError(f"Segmentation not found: {seg_path}")

        ct, affine = load_img(str(ct_path), canonical=canonical)
        seg, _ = load_img(str(seg_path), canonical=canonical)
        return cls(
            patient_id=patient_id,
            ct=ct,
            seg=seg,
            affine=affine,
            ct_path=ct_path,
            seg_path=seg_path,
        )
