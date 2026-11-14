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
    Template Inclusion (Double Thresholding):
        Included in gallery if d(V, ref) <= inclusion_threshold (defaults to threshold).
    """

    threshold: float
    inclusion_threshold: float | None
    window_size: int | None
    _mean: pd.Series | None
    _std: pd.Series | None
    _samples: list[pd.Series]
    _effective_window_size: int

    def __init__(
        self,
        threshold: float = 0.5,
        inclusion_threshold: float | None = None,
        window_size: int | None = None,
    ):
        self.threshold = threshold
        self.inclusion_threshold = inclusion_threshold
        self.window_size = window_size
        self._samples = []
        self._mean = None
        self._std = None
        self._effective_window_size = window_size if window_size is not None else 10

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "inclusion_threshold": self.inclusion_threshold,
            "window_size": self.window_size,
        }

    def set_params(self, **params) -> Self:
        self.threshold = params.get("threshold", self.threshold)
        self.inclusion_threshold = params.get("inclusion_threshold", self.inclusion_threshold)
        self.window_size = params.get("window_size", self.window_size)
        return self

    def fit(self, x: pd.DataFrame, y: list[int] | None = None) -> Self:
        self._samples = [row.copy() for _, row in x.iterrows()]
        self._effective_window_size = self.window_size if self.window_size is not None else max(len(self._samples), 1)
        self._recompute_stats()
        return self

    def _recompute_stats(self) -> None:
        if not self._samples:
            return
        df = pd.DataFrame(self._samples).astype(float)
        self._mean = df.mean()
        self._std = df.std().clip(lower=MIN_STD)

    def calculate_distance(self, x: pd.Series) -> float:
        if self._mean is None or self._std is None:
            raise ValueError("Model has not been fitted yet.")
        diff = (x.astype(float) - self._mean).abs() / self._std
        similarities = np.exp(-diff.to_numpy(dtype=float))
        return float(1.0 - np.mean(similarities))

    def predict_vec(self, x: pd.Series) -> int:
        dist = self.calculate_distance(x)
        return GENUINE_LABEL if dist <= self.threshold else IMPOSTOR_LABEL

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_vec(row) for _, row in x.iterrows()])

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
        """Calibrates user-specific threshold from the genuine gallery alone."""
        if not self._samples:
            raise ValueError("Model has not been fitted yet.")
        distances = np.array([self.calculate_distance(sample) for sample in self._samples], dtype=float)
        if len(distances) > 1 and std_multiplier > 0:
            upper_bound = float(np.mean(distances) + std_multiplier * np.std(distances, ddof=1))
            non_outliers = distances[distances <= upper_bound]
            if len(non_outliers) > 0:
                distances = non_outliers
        # Small margin above max non-outlier gallery distance avoids zero-tolerance lockouts on 5-10 samples
        calibrated = float(np.max(distances) * 1.15)
        self.threshold = min(calibrated, 0.95)
        if inclusion_ratio is not None:
            self.inclusion_threshold = self.threshold * inclusion_ratio
        return self.threshold

    def adapt(
        self,
        x: pd.Series | pd.DataFrame,
        window: str = "sliding",
        max_size: int | None = None,
    ) -> None:
        """Adapts reference template with new accepted genuine sample(s)."""
        if isinstance(x, pd.Series):
            new_rows = [x.copy()]
        else:
            new_rows = [row.copy() for _, row in x.iterrows()]

        window_limit = max_size if max_size is not None else self._effective_window_size

        if window == "sliding":
            for row in new_rows:
                if len(self._samples) >= window_limit:
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
