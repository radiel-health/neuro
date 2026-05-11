from dataclasses import dataclass

import numpy as np

from cams import compute_cam


@dataclass
class CAMExplainer:
    method: str = "gradcam"
    target_layer: str | None = None

    def compute(self, model, patch, target_class=None):
        return compute_cam(
            model,
            patch,
            target_class=target_class,
            method=self.method,
            target_layer=self.target_layer,
        )

    def patient_points(self, explanation):
        if explanation.coords is None:
            raise ValueError("Patient-space CAM points need explanation.coords.")
        mask = np.asarray(explanation.mask, dtype=bool)
        return {
            "vertebra": explanation.prediction.vertebra,
            "coords": explanation.coords[mask].astype(np.float32),
            "values": explanation.cam[mask].astype(np.float32),
        }
