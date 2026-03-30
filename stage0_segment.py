import os
import subprocess

CT_ROOT = "data/Spine-Mets-CT-SEG-Nifti"
OUTPUT_ROOT = "output/stage0"
DEVICE = "gpu"
VERTEBRA_ROIS = [
    "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5",
    "vertebrae_T6", "vertebrae_T7", "vertebrae_T8", "vertebrae_T9", "vertebrae_T10",
    "vertebrae_T11", "vertebrae_T12", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3",
    "vertebrae_L4", "vertebrae_L5",
]

# make sure output root exists
os.makedirs(OUTPUT_ROOT, exist_ok=True)



def run_totalsegmentator(input_path, output_path):
    cmd = [
        "TotalSegmentator",
        "-i", input_path,
        "-o", output_path,
        "--device", DEVICE,
        "--roi_subset", *VERTEBRA_ROIS,
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    for id in os.listdir(CT_ROOT):
        if id.endswith(".json") or id.endswith(".txt"):
            continue
    
        file = os.path.join(CT_ROOT, id, f"{id}_ct.nii.gz")
    
        output_path = os.path.join(OUTPUT_ROOT, id)

        # Skip if already processed
        # if os.path.exists(output_path) and len(os.listdir(output_path)) > 0:
        #     print(f"Skipping {id}, already processed.")
        #     continue

        os.makedirs(output_path, exist_ok=True)

        try:
            run_totalsegmentator(file, output_path)
            print(f"Finished {id}")
        except subprocess.CalledProcessError:
            print(f"Error processing {id}")

if __name__ == "__main__":
    main()
