from typing import Self
from collections import deque
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from lib.utils import IMPOSTOR_LABEL, GENUINE_LABEL

MIN_MAD = 1e-4


class ScaledManhattanAlg(BaseEstimator):
    """Scaled Manhattan Distance Classifier with Mean Absolute Deviation (MAD).

    Based on the benchmark champion from Killourhy & Maxion (2009).
    Trained strictly on genuine samples (one-class).
    Distance:
        d(x, ref) = (1 / n) * sum_{i=1}^n (|x_i - center_i| / MAD_i)
    Classification:
        GENUINE if d(x, ref) <= threshold, else IMPOSTOR.
    """

    threshold: float
    inclusion_threshold: float | None
    sliding_window_size: int | None
    window_strategy: str
    _center: pd.Series | None
    _mad: pd.Series | None
    _samples: deque[pd.Series]

    def __init__(
        self,
        threshold: float = 2.0,
        sliding_window_size: int | None = None,
        inclusion_threshold: float | None = None,
        window_strategy: str | None = None,
    ):
        self.threshold = threshold
        self.inclusion_threshold = inclusion_threshold
        self.sliding_window_size = sliding_window_size
        if window_strategy is not None:
            self.window_strategy = window_strategy
        else:
            self.window_strategy = "sliding" if sliding_window_size is not None else "none"
        self._samples = deque(maxlen=sliding_window_size if self.window_strategy == "sliding" else None)
        self._center = None
        self._mad = None

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "inclusion_threshold": self.inclusion_threshold,
            "sliding_window_size": self.sliding_window_size,
            "window_strategy": self.window_strategy,
        }

    def set_params(self, **params) -> Self:
        self.threshold = params.get("threshold", self.threshold)
        self.inclusion_threshold = params.get("inclusion_threshold", self.inclusion_threshold)
        self.sliding_window_size = params.get("sliding_window_size", self.sliding_window_size)
        if "window_strategy" in params and params["window_strategy"] is not None:
            self.window_strategy = params["window_strategy"]
        elif "sliding_window_size" in params:
            self.window_strategy = "sliding" if self.sliding_window_size is not None else "none"
        return self

    def fit(self, x: pd.DataFrame, y: list[int] | None = None) -> Self:
        if self.window_strategy == "sliding":
            maxlen = self.sliding_window_size if self.sliding_window_size is not None else max(len(x), 1)
        else:
            maxlen = None
        self._samples = deque([row.copy() for _, row in x.iterrows()], maxlen=maxlen)
        self._recompute()
        return self

    def _recompute(self) -> None:
        if not self._samples:
            return
        df = pd.DataFrame(list(self._samples)).astype(float)
        self._center = df.mean()
        deviations = (df - self._center).abs()
        self._mad = deviations.mean().clip(lower=MIN_MAD)

    def calculate_distance(self, x: pd.Series) -> float:
        if self._center is None or self._mad is None:
            raise ValueError("Model has not been fitted yet.")
        scaled_diff = (x.astype(float) - self._center).abs() / self._mad
        return float(scaled_diff.mean())

    def should_include_sample(self, x: pd.Series | pd.DataFrame) -> bool:
        effective_limit = self.inclusion_threshold if self.inclusion_threshold is not None else self.threshold
        if isinstance(x, pd.Series):
            return self.calculate_distance(x) <= effective_limit
        return all(self.calculate_distance(row) <= effective_limit for _, row in x.iterrows())

    def calibrate_user_threshold(
        self,
        std_multiplier: float = 2.5,
        inclusion_ratio: float | None = None,
    ) -> float:
        if not self._samples:
            raise ValueError("Model has not been fitted yet.")
        distances = np.array([self.calculate_distance(sample) for sample in self._samples], dtype=float)
        if len(distances) > 1 and std_multiplier > 0:
            upper_bound = float(np.mean(distances) + std_multiplier * np.std(distances, ddof=1))
            non_outliers = distances[distances <= upper_bound]
            if len(non_outliers) > 0:
                distances = non_outliers
        self.threshold = float(np.max(distances) * 1.15)
        if inclusion_ratio is not None:
            self.inclusion_threshold = self.threshold * inclusion_ratio
        return self.threshold

    def predict_vec(self, x: pd.Series) -> int:
        dist = self.calculate_distance(x)
        return GENUINE_LABEL if dist <= self.threshold else IMPOSTOR_LABEL

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_vec(row) for _, row in x.iterrows()])

    def adapt(self, x: pd.Series | pd.DataFrame, window: str | None = None) -> None:
        strategy = window if window is not None else self.window_strategy
        if strategy == "none":
            return

        if isinstance(x, pd.Series):
            new_rows = [x.copy()]
        else:
            new_rows = [row.copy() for _, row in x.iterrows()]

        if strategy == "growing" and self._samples.maxlen is not None:
            self._samples = deque(self._samples, maxlen=None)
        elif strategy == "sliding" and self._samples.maxlen is None:
            maxlen = self.sliding_window_size if self.sliding_window_size is not None else max(len(self._samples), 1)
            self._samples = deque(self._samples, maxlen=maxlen)

        for row in new_rows:
            self._samples.append(row)
        self._recompute()
