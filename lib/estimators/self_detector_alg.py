from typing import Self
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from lib.utils import IMPOSTOR_LABEL, GENUINE_LABEL


def rank_transform_vector(vec: np.ndarray) -> np.ndarray:
    ranks = vec.argsort(kind="stable").argsort(kind="stable").astype(float) + 1.0
    norm = np.linalg.norm(ranks)
    return ranks / norm if norm > 0 else ranks


@dataclass
class _Detector:
    center: np.ndarray
    recent_usage: int
    usage_count: int = 0


class SelfDetectorAlg(BaseEstimator):
    """Positive Selection / Immune Artificial System (Pisani et al., 2015).

    Estimator that represents a genuine user by a set of detectors.
    Uses rank transformation and cosine similarity.
    Supports strategies:
      - 'none': static baseline
      - 'sliding': sliding window
      - 'usage_control_r': removes least used detector with recent_usage == 0
      - 'usage_control_2': removes all detectors with recent_usage == 0
    """

    radius: float
    inclusion_radius: float | None
    strategy: str
    max_recent_usage: int
    _detectors: list[_Detector]

    def __init__(
        self,
        radius: float = 0.02,
        strategy: str = "none",
        max_recent_usage: int = 10,
        inclusion_radius: float | None = None,
    ):
        self.radius = radius
        self.inclusion_radius = inclusion_radius
        self.strategy = strategy
        self.max_recent_usage = max_recent_usage
        self._detectors = []

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {
            "radius": self.radius,
            "inclusion_radius": self.inclusion_radius,
            "strategy": self.strategy,
            "max_recent_usage": self.max_recent_usage,
        }

    def set_params(self, **params) -> Self:
        self.radius = params.get("radius", self.radius)
        self.inclusion_radius = params.get("inclusion_radius", self.inclusion_radius)
        self.strategy = params.get("strategy", self.strategy)
        self.max_recent_usage = params.get("max_recent_usage", self.max_recent_usage)
        return self

    def fit(self, x: pd.DataFrame, y: list[int] | None = None) -> Self:
        self._detectors = []
        for _, row in x.iterrows():
            transformed = rank_transform_vector(row.to_numpy(dtype=float))
            self._detectors.append(_Detector(center=transformed, recent_usage=self.max_recent_usage))
        return self

    def _match(self, transformed: np.ndarray) -> tuple[bool, int | None, float]:
        if not self._detectors:
            return False, None, 0.0

        # In usage control R and 2, check from newest to oldest
        reverse_check = self.strategy in ("usage_control_r", "usage_control_2")
        indices = range(len(self._detectors) - 1, -1, -1) if reverse_check else range(len(self._detectors))

        best_sim = -1.0
        matched_idx = None

        for idx in indices:
            det = self._detectors[idx]
            sim = float(det.center @ transformed)
            if sim > best_sim:
                best_sim = sim
            if (1.0 - sim) < self.radius and matched_idx is None:
                matched_idx = idx

        return (matched_idx is not None), matched_idx, best_sim

    def calculate_distance(self, x: pd.Series) -> float:
        transformed = rank_transform_vector(x.to_numpy(dtype=float))
        _, _, best_sim = self._match(transformed)
        return float(1.0 - best_sim)

    def should_include_sample(self, x: pd.Series | pd.DataFrame) -> bool:
        effective_radius = self.inclusion_radius if self.inclusion_radius is not None else self.radius
        if isinstance(x, pd.Series):
            return self.calculate_distance(x) < effective_radius
        return all(self.calculate_distance(row) < effective_radius for _, row in x.iterrows())

    def predict_vec(self, x: pd.Series) -> int:
        transformed = rank_transform_vector(x.to_numpy(dtype=float))
        matched, _, _ = self._match(transformed)
        return GENUINE_LABEL if matched else IMPOSTOR_LABEL

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_vec(row) for _, row in x.iterrows()])

    def adapt(self, x: pd.Series | pd.DataFrame) -> bool:
        """Adapts detector gallery with a query. Returns True if adaptation took place."""
        if isinstance(x, pd.Series):
            rows = [x]
        else:
            rows = [row for _, row in x.iterrows()]

        effective_radius = self.inclusion_radius if self.inclusion_radius is not None else self.radius
        adapted_any = False
        for row in rows:
            transformed = rank_transform_vector(row.to_numpy(dtype=float))
            matched, matched_idx, best_sim = self._match(transformed)
            if not matched or matched_idx is None or (1.0 - best_sim) >= effective_radius:
                continue

            for idx, det in enumerate(self._detectors):
                if idx == matched_idx:
                    det.usage_count += 1
                    det.recent_usage = self.max_recent_usage
                else:
                    det.recent_usage = max(0, det.recent_usage - 1)

            new_detector = _Detector(center=transformed, recent_usage=self.max_recent_usage)

            if self.strategy == "none":
                continue
            elif self.strategy == "sliding":
                if self._detectors:
                    self._detectors.pop(0)
                self._detectors.append(new_detector)
                adapted_any = True
            elif self.strategy == "usage_control_r":
                idle_indices = [i for i, d in enumerate(self._detectors) if d.recent_usage == 0]
                if idle_indices:
                    least_idx = min(idle_indices, key=lambda i: self._detectors[i].usage_count)
                    self._detectors.pop(least_idx)
                    self._detectors.append(new_detector)
                    adapted_any = True
            elif self.strategy == "usage_control_2":
                self._detectors = [d for d in self._detectors if d.recent_usage > 0]
                self._detectors.append(new_detector)
                adapted_any = True

        return adapted_any
