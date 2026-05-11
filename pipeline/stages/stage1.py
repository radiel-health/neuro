from dataclasses import dataclass

import torch

from cams import (
    build_stage1_patch_and_mask,
    build_stage1_patch_mask_and_coords,
    class_names_for_count,
    infer_num_classes_from_checkpoint,
    load_stage1_model,
)
from pipeline.core.results import Stage1Explanation, Stage1Prediction
from pipeline.explainability.cam import CAMExplainer
from pipeline.stages.base import PipelineStage, register_stage
from utils import VERTEBRAE


@dataclass
class Stage1PatchConfig:
    patch_size: tuple[int, int, int] = (96, 96, 64)
    norm_mode: str = "zscore_sigmoid"
    zscore_scale: float = 1.5
    foreground_floor: float = 0.15


class Stage1Classifier(PipelineStage):
    name = "stage1"
    num_classes = 4
    class_names = ["none", "blastic", "lytic", "mixed"]

    def __init__(self, model_path, device=None, cam: CAMExplainer | None = None, patch_config=None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cam = cam
        self.patch_config = patch_config or Stage1PatchConfig()
        self.model = load_stage1_model(
            model_path,
            device=self.device,
            num_classes=self.num_classes,
        )

    @classmethod
    def from_checkpoint(cls, model_path, device=None, cam=None, patch_config=None):
        num_classes = infer_num_classes_from_checkpoint(model_path)
        if cls is Stage1Classifier:
            variant = stage1_variant_for_classes(num_classes)
            return variant(model_path, device=device, cam=cam, patch_config=patch_config)
        return cls(model_path, device=device, cam=cam, patch_config=patch_config)

    def build_patch(self, context, vertebra, with_coords=False):
        kwargs = {
            "patch_size": self.patch_config.patch_size,
            "norm_mode": self.patch_config.norm_mode,
            "zscore_scale": self.patch_config.zscore_scale,
            "foreground_floor": self.patch_config.foreground_floor,
        }
        if with_coords:
            return build_stage1_patch_mask_and_coords(context.ct, context.seg, vertebra, **kwargs)
        return build_stage1_patch_and_mask(context.ct, context.seg, vertebra, **kwargs)

    def predict_patch(self, vertebra, patch):
        x = patch.unsqueeze(0).to(self.device) if patch.ndim == 4 else patch.to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu()
        class_id = int(probs.argmax().item())
        class_name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
        return Stage1Prediction(
            vertebra=str(vertebra).upper(),
            class_id=class_id,
            class_name=class_name,
            probabilities=[float(v) for v in probs.tolist()],
        )

    def predict_vertebra(self, context, vertebra):
        patch, _ = self.build_patch(context, vertebra, with_coords=False)
        return self.predict_patch(vertebra, patch)

    def explain_vertebra(self, context, vertebra, target_class=None):
        if self.cam is None:
            raise RuntimeError("CAM is not enabled for stage1.")
        patch, mask, coords = self.build_patch(context, vertebra, with_coords=True)
        prediction = self.predict_patch(vertebra, patch)
        cam_result = self.cam.compute(self.model, patch, target_class=target_class)
        return Stage1Explanation(
            prediction=prediction,
            cam=cam_result.cam,
            mask=mask,
            coords=coords,
        )

    def run(self, context):
        predictions = []
        for vertebra in VERTEBRAE:
            if not (context.seg == self._label_for_vertebra(vertebra)).any():
                continue
            predictions.append(self.predict_vertebra(context, vertebra))
        return predictions

    @staticmethod
    def _label_for_vertebra(vertebra):
        from utils import VERTEBRA_LABELS

        return VERTEBRA_LABELS[str(vertebra).upper()]


@register_stage("stage1_4class")
class Stage1FourClass(Stage1Classifier):
    num_classes = 4
    class_names = ["none", "blastic", "lytic", "mixed"]


@register_stage("stage1_binary")
class Stage1Binary(Stage1Classifier):
    num_classes = 2
    class_names = ["none", "cancer"]


@register_stage("stage1_cancer_type")
class Stage1CancerType(Stage1Classifier):
    num_classes = 3
    class_names = ["blastic", "lytic", "mixed"]


def stage1_variant_for_classes(num_classes):
    if int(num_classes) == 2:
        return Stage1Binary
    if int(num_classes) == 3:
        return Stage1CancerType
    return Stage1FourClass
