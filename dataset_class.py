import os
import numpy as np
import pandas as pd
import torch
import re
from collections import Counter

from torch.utils.data import Dataset
from utils import (
    VERTEBRA_LABELS,
    VERTEBRAE,
    load_img,
    crop_from_mask,
    resize_patch,
)


def extract_levels(text):
    if pd.isna(text):
        return []
    text = str(text)
    return re.findall(r"[TL]\d+", text)


def build_labels(row):
    labels = {v: 0 for v in VERTEBRAE}
    blastic = extract_levels(row["Blastic"])
    lytic = extract_levels(row["Lytic"])
    mixed = extract_levels(row["Mixed"])

    for v in blastic:
        if v in labels:
            labels[v] = 1
    for v in lytic:
        if v in labels:
            labels[v] = 2
    for v in mixed:
        if v in labels:
            labels[v] = 3
    return labels


def build_dataset_csv(metadata_path="data/patient_metadata.csv", output_path="data/vertebra_dataset.csv"):
    df = pd.read_csv(metadata_path)
    rows = []

    for _, row in df.iterrows():
        pid = row["Case"]
        labels = build_labels(row)

        for v in VERTEBRAE:
            rows.append(
                {
                    "patient_id": pid,
                    "vertebra": v,
                    "label": labels[v],
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    counts = dict(Counter(out["label"]))
    print(f"Saved: {output_path}")
    print(f"Label counts: {counts}")
    return out

class VertebraDataset(Dataset):

    def __init__(self, csv_path, root_dir):
        self.df = pd.read_csv(csv_path)
        self.root = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pid = str(row["patient_id"])
        vname = row["vertebra"]
        label = int(row["label"])

        vid = VERTEBRA_LABELS[vname]

        patient_dir = os.path.join(self.root, pid)

        ct_path = os.path.join(
            patient_dir,
            f"{pid}_ct.nii.gz"
        )

        seg_path = os.path.join(
            patient_dir,
            f"{pid}_seg-1.nii.gz"
        )

        ct, _ = load_img(ct_path, canonical=False)
        seg, _ = load_img(seg_path, canonical=False)

        patch = crop_from_mask(ct, seg, vid)

        if patch is None:
            patch = np.zeros((32,32,32))

        # HU window
        patch = np.clip(patch, -200, 1000)
        patch = resize_patch(patch)
        patch = patch / 1000.0
        return patch, torch.tensor(label).long()
    

if __name__ == "__main__":
    build_dataset_csv(
        metadata_path="data/patient_metadata.csv",
        output_path="data/vertebra_dataset.csv",
    )
    dataset = VertebraDataset(
        csv_path="data/vertebra_dataset.csv",
        root_dir="data/Spine-Mets-CT-SEG-Nifti",
    )
    print(f"Dataset size: {len(dataset)}")
    patch, label = dataset[0]
    print(f"Patch shape: {patch.shape}, Label: {label}")
    
    
    
    
    
