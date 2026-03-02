import os
import argparse
import numpy as np
import pandas as pd
from utils import INV_LABELS, load_img, vox2world


def lesion_mask(ct):
    return np.logical_or(
        ct < 100,
        ct > 600
    )


def posterolateral_score(ct, seg, aff, label):
    mask = seg == label
    if mask.sum()==0:
        return 0
    ijk = np.argwhere(mask)
    xyz = vox2world(aff, ijk)
    x = xyz[:,0]
    y = xyz[:,1]
    cx = np.median(x)
    cy = np.median(y)
    ct_vals = ct[mask]
    abnormal = np.logical_or(
        ct_vals < 100,
        ct_vals > 600
    )

    if abnormal.sum()==0:
        return 0
    x_abn = x[abnormal]
    y_abn = y[abnormal]

    # posterior only
    posterior = y_abn < cy

    if posterior.sum()==0:
        return 0

    x_abn = x_abn[posterior]
    left = x_abn < cx
    right = x_abn > cx

    left_flag = left.sum() > 50
    right_flag = right.sum() > 50

    if left_flag and right_flag:
        return 3
    if left_flag or right_flag:
        return 1
    return 0

def run(root):
    rows=[]
    for pid in os.listdir(root):
        pdir = os.path.join(root,pid)
        if not os.path.isdir(pdir):
            continue
        seg_path = os.path.join(pdir,f"{pid}_seg-1.nii.gz")
        ct_path  = os.path.join(pdir,f"{pid}_ct.nii.gz")
        
        if not os.path.exists(seg_path):
            continue

        seg,aff = load_img(seg_path)
        ct,_    = load_img(ct_path)

        for lab in range(1,18):
            score = posterolateral_score(
                ct,seg,aff,lab
            )
            rows.append({
                "patient_id":pid,
                "vertebra":INV_LABELS[lab],
                "posterolateral_score":score
            })

    df = pd.DataFrame(rows)
    df.to_csv(
        "stage3_posterolateral.csv",
        index=False
    )
    print("saved stage3_posterolateral.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    args = parser.parse_args()
    run(args.root)
