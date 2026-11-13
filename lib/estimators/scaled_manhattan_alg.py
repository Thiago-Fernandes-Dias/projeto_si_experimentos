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
    sliding_window_size: int | None
    _center: pd.Series
    _mad: pd.Series
    _samples: deque[pd.Series]

    def __init__(self, threshold: float = 2.0, sliding_window_size: int | None = None):
        self.threshold = threshold
        self.sliding_window_size = sliding_window_size
        self._samples = deque(maxlen=sliding_window_size)
        self._center = None
        self._mad = None

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "sliding_window_size": self.sliding_window_size,
        }

    def set_params(self, **params) -> Self:
        self.threshold = params.get("threshold", self.threshold)
        self.sliding_window_size = params.get("sliding_window_size", self.sliding_window_size)
        return self

    def fit(self, x: pd.DataFrame, y: list[int] | None = None) -> Self:
        self._samples = deque([row.copy() for _, row in x.iterrows()], maxlen=self.sliding_window_size)
        self._recompute()
        return self

    def _recompute(self) -> None:
        if not self._samples:
            return
        df = pd.DataFrame(list(self._samples))
        self._center = df.mean()
        deviations = (df - self._center).abs()
        self._mad = deviations.mean().clip(lower=MIN_MAD)

    def calculate_distance(self, x: pd.Series) -> float:
        if self._center is None or self._mad is None:
            raise ValueError("Model has not been fitted yet.")
        scaled_diff = (x - self._center).abs() / self._mad
        return float(scaled_diff.mean())

    def predict_vec(self, x: pd.Series) -> int:
        dist = self.calculate_distance(x)
        return GENUINE_LABEL if dist <= self.threshold else IMPOSTOR_LABEL

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_vec(row) for _, row in x.iterrows()])

    def adapt(self, x: pd.Series | pd.DataFrame) -> None:
        if isinstance(x, pd.Series):
            self._samples.append(x.copy())
        else:
            for _, row in x.iterrows():
                self._samples.append(row.copy())
        self._recompute()
