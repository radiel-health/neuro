from dataclasses import dataclass, field
from pathlib import Path
import subprocess

from pipeline.stages.base import PipelineStage, register_stage
from utils import DEFAULT_DATA_ROOT, VERTEBRAE


DEFAULT_VERTEBRA_ROIS = [f"vertebrae_{name}" for name in VERTEBRAE]


@dataclass
class Stage0Config:
    output_root: str = "output/stage0"
    device: str = "gpu"
    rois: list[str] = field(default_factory=lambda: list(DEFAULT_VERTEBRA_ROIS))
    executable: str = "TotalSegmentator"


@register_stage("stage0")
class Stage0Segmentation(PipelineStage):
    def __init__(self, config=None):
        self.config = config or Stage0Config()

    def output_dir(self, patient_id):
        return Path(self.config.output_root) / str(patient_id)

    def build_command(self, input_path, output_path):
        return [
            self.config.executable,
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--device",
            self.config.device,
            "--roi_subset",
            *self.config.rois,
        ]

    def run_file(self, input_path, output_path):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        cmd = self.build_command(input_path, output_path)
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return output_path

    def run_patient(self, patient_id, root_dir=DEFAULT_DATA_ROOT, skip_existing=False):
        patient_id = str(patient_id)
        ct_path = Path(root_dir) / patient_id / f"{patient_id}_ct.nii.gz"
        if not ct_path.exists():
            raise FileNotFoundError(f"CT not found: {ct_path}")
        output_path = self.output_dir(patient_id)
        if skip_existing and output_path.exists() and any(output_path.iterdir()):
            print(f"Skipping {patient_id}, already processed.")
            return output_path
        return self.run_file(ct_path, output_path)

    def run_directory(self, root_dir=DEFAULT_DATA_ROOT, skip_existing=False):
        root_dir = Path(root_dir)
        outputs = []
        for patient_dir in sorted(root_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            try:
                outputs.append(self.run_patient(patient_dir.name, root_dir=root_dir, skip_existing=skip_existing))
                print(f"Finished {patient_dir.name}")
            except subprocess.CalledProcessError:
                print(f"Error processing {patient_dir.name}")
        return outputs
