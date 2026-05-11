from pipeline.stages.base import STAGE_REGISTRY, PipelineStage, register_stage
from pipeline.stages.stage0 import Stage0Config, Stage0Segmentation
from pipeline.stages.stage1 import Stage1Binary, Stage1CancerType, Stage1Classifier, Stage1FourClass
from pipeline.stages.stage2 import Stage2Collapse
from pipeline.stages.stage3 import Stage3Posterolateral
from pipeline.stages.stage4 import Stage4Alignment

__all__ = [
    "STAGE_REGISTRY",
    "PipelineStage",
    "Stage0Config",
    "Stage0Segmentation",
    "Stage1Binary",
    "Stage1CancerType",
    "Stage1Classifier",
    "Stage1FourClass",
    "Stage2Collapse",
    "Stage3Posterolateral",
    "Stage4Alignment",
    "register_stage",
]
