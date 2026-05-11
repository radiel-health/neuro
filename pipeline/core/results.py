from dataclasses import dataclass

import numpy as np


@dataclass
class Stage1Prediction:
    vertebra: str
    class_id: int
    class_name: str
    probabilities: list[float]

    @property
    def confidence(self):
        if 0 <= self.class_id < len(self.probabilities):
            return float(self.probabilities[self.class_id])
        return 0.0

    @property
    def is_suspicious(self):
        return int(self.class_id) != 0


@dataclass
class Stage1Explanation:
    prediction: Stage1Prediction
    cam: np.ndarray
    mask: np.ndarray
    coords: np.ndarray | None = None


@dataclass
class SINSResult:
    vertebra: str
    total: int
    category: str
    location: int
    bone: int
    collapse: int
    posterolateral: int
    alignment: int
