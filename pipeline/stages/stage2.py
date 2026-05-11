import numpy as np

from pipeline.scripts.stage2_collapse import collapse_score, measure_height
from utils import VERTEBRA_LABELS

from pipeline.stages.base import PipelineStage, register_stage


@register_stage("stage2")
class Stage2Collapse(PipelineStage):
    def run_vertebra(self, context, vertebra):
        label = VERTEBRA_LABELS[str(vertebra).upper()]
        heights = measure_height(context.seg, context.affine, label)
        if heights is None:
            return 0
        ant, post = heights
        ratio = min(ant, post) / max(ant, post)
        if not np.isfinite(ratio):
            return 0
        return collapse_score(ratio)
