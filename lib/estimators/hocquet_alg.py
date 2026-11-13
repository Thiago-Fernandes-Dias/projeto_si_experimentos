from typing import Self
import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from lib.utils import IMPOSTOR_LABEL, GENUINE_LABEL

MIN_STD = 1e-4


class HocquetStatisticalAlg(BaseEstimator):
    """Statistical classifier based on Hocquet et al. (2007) and Mhenni et al. (2016).

    Distance to reference:
        d(V, ref) = 1 - (1/n) * sum_{i=1}^n exp(-|V_i - mu_i| / sigma_i)
    Classification:
        GENUINE if d(V, ref) <= threshold, else IMPOSTOR.
    """

    threshold: float
    _mean: pd.Series
    _std: pd.Series
    _samples: list[pd.Series]

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._samples = []
        self._mean = None
        self._std = None

    def get_params(self, deep: bool = True) -> dict[str, float]:
        return {"threshold": self.threshold}

    def set_params(self, **params) -> Self:
        self.threshold = params.get("threshold", self.threshold)
        return self

    def fit(self, x: pd.DataFrame, y: list[int] | None = None) -> Self:
        self._samples = [row.copy() for _, row in x.iterrows()]
        self._recompute_stats()
        return self

    def _recompute_stats(self) -> None:
        if not self._samples:
            return
        df = pd.DataFrame(self._samples)
        self._mean = df.mean()
        self._std = df.std().clip(lower=MIN_STD)

    def calculate_distance(self, x: pd.Series) -> float:
        if self._mean is None or self._std is None:
            raise ValueError("Model has not been fitted yet.")
        diff = (x - self._mean).abs() / self._std
        similarities = np.exp(-diff)
        return float(1.0 - similarities.mean())

    def predict_vec(self, x: pd.Series) -> int:
        dist = self.calculate_distance(x)
        return GENUINE_LABEL if dist <= self.threshold else IMPOSTOR_LABEL

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_vec(row) for _, row in x.iterrows()])

    def adapt(self, x: pd.Series | pd.DataFrame, window: str = "sliding", max_size: int = 10) -> None:
        """Adapts reference template with new accepted genuine sample(s)."""
        if isinstance(x, pd.Series):
            new_rows = [x.copy()]
        else:
            new_rows = [row.copy() for _, row in x.iterrows()]

        if window == "sliding":
            for row in new_rows:
                if len(self._samples) >= max_size:
                    self._samples.pop(0)
                self._samples.append(row)
        elif window == "growing":
            self._samples.extend(new_rows)

        self._recompute_stats()

    def calculate_variable_threshold_decrement(self) -> float:
        """Mhenni et al. (2016), eq. (2): decrement = exp(-mu / sigma)
        where mu is mean of mean vector, and sigma is std of std vector.
        """
        if self._mean is None or self._std is None:
            return 0.0
        mu = float(self._mean.mean())
        sigma = float(self._std.std())
        if sigma <= 0:
            return 0.0
        return float(math.exp(-mu / sigma))
