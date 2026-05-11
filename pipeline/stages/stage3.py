import numpy as np

from pipeline.scripts.stage3_postlateral import posterolateral_score_from_ijk, resolve_thresholds
from utils import VERTEBRA_LABELS

from pipeline.stages.base import PipelineStage, register_stage


@register_stage("stage3")
class Stage3Posterolateral(PipelineStage):
    def prepare(self, context):
        low_hu, high_hu, threshold_meta = resolve_thresholds(context.ct, context.seg)
        return low_hu, high_hu, threshold_meta

    def run_vertebra(self, context, vertebra, thresholds=None):
        label = VERTEBRA_LABELS[str(vertebra).upper()]
        low_hu, high_hu, threshold_meta = thresholds or self.prepare(context)
        ijk = np.argwhere(context.seg == label)
        return posterolateral_score_from_ijk(
            context.ct,
            context.affine,
            ijk,
            lesion_low_hu=low_hu,
            lesion_high_hu=high_hu,
            threshold_meta=threshold_meta,
        )
