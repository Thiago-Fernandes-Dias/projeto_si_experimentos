from typing import Self
from collections import deque
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from lib.utils import IMPOSTOR_LABEL, GENUINE_LABEL

MIN_STD = 1e-4


class KNNInstanceAlg(BaseEstimator):
    """Instance-based One-Class k-Nearest Neighbors classifier.

    Standardizes features by gallery mean and standard deviation.
    Computes average Euclidean distance to the k nearest neighbors.
    Classification:
        GENUINE if average distance <= threshold, else IMPOSTOR.
    """

    k: int
    threshold: float
    sliding_window_size: int | None
    _samples: deque[pd.Series]
    _mean: pd.Series
    _std: pd.Series

    def __init__(self, k: int = 1, threshold: float = 1.5, sliding_window_size: int | None = None):
        self.k = k
        self.threshold = threshold
        self.sliding_window_size = sliding_window_size
        self._samples = deque(maxlen=sliding_window_size)
        self._mean = None
        self._std = None

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {
            "k": self.k,
            "threshold": self.threshold,
            "sliding_window_size": self.sliding_window_size,
        }

    def set_params(self, **params) -> Self:
        self.k = params.get("k", self.k)
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
        self._mean = df.mean()
        self._std = df.std().clip(lower=MIN_STD)

    def calculate_distance(self, x: pd.Series) -> float:
        if self._mean is None or self._std is None or not self._samples:
            raise ValueError("Model has not been fitted yet.")
        df = pd.DataFrame(list(self._samples))
        std_gallery = ((df - self._mean) / self._std).to_numpy(dtype=float)
        std_x = ((x - self._mean) / self._std).to_numpy(dtype=float)
        dists = np.linalg.norm(std_gallery - std_x, axis=1) / np.sqrt(len(x))
        effective_k = min(self.k, len(dists))
        k_dists = np.partition(dists, effective_k - 1)[:effective_k]
        return float(np.mean(k_dists))

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
