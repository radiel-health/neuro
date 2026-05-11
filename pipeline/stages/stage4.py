from pipeline.scripts.stage4_alignment import compute_patient_alignment

from pipeline.stages.base import PipelineStage, register_stage


@register_stage("stage4")
class Stage4Alignment(PipelineStage):
    def __init__(self, min_labels=6, neighbor_window=10, angle_threshold=10):
        self.min_labels = min_labels
        self.neighbor_window = neighbor_window
        self.angle_threshold = angle_threshold

    def run_patient(self, context):
        result = compute_patient_alignment(
            str(context.seg_path),
            self.min_labels,
            self.neighbor_window,
            self.angle_threshold,
        )
        if result is None:
            return 0
        summary, _ = result
        return summary["alignment_score"]
