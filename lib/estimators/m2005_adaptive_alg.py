from typing import Self
from collections import deque
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from lib.estimators.m2005_alg import M2005Alg, DigraphsMetrics
from lib.utils import IMPOSTOR_LABEL, GENUINE_LABEL


class _WelfordOnlineStats:
    """Welford's algorithm for online mean and sample standard deviation."""

    def __init__(self, df: pd.DataFrame):
        self.count = len(df)
        self.mean = df.mean().to_dict()
        self.squared_dev = {col: 0.0 for col in df.columns}
        for _, row in df.iterrows():
            for col in df.columns:
                delta = row[col] - self.mean[col]
                self.squared_dev[col] += delta * (row[col] - self.mean[col])

    def update(self, row: pd.Series):
        self.count += 1
        for col in row.index:
            delta = row[col] - self.mean[col]
            self.mean[col] += delta / self.count
            self.squared_dev[col] += delta * (row[col] - self.mean[col])

    def get_metrics(self) -> dict[str, DigraphsMetrics]:
        metrics = {}
        for col, mean_val in self.mean.items():
            variance = self.squared_dev[col] / (self.count - 1) if self.count > 1 else 0.0
            std_val = float(np.sqrt(variance))
            metrics[str(col)] = DigraphsMetrics(mean_val, std_val, mean_val)
        return metrics


class AdaptiveM2005Alg(BaseEstimator):
    """M2005 with Double Parallel (DB) and Improved Double Parallel (IDB) (Pisani et al., 2015).

    Maintains two models:
      1. Growing model (Welford incremental in IDB; growing gallery in DB)
      2. Sliding model (bounded FIFO gallery of size window_size)
    Score is the average of both models.
    """

    threshold: float
    inclusion_threshold: float | None
    variant: str
    window_size: int | None

    def __init__(
        self,
        threshold: float = 0.7,
        variant: str = "idb",
        window_size: int | None = None,
        inclusion_threshold: float | None = None,
    ):
        self.threshold = threshold
        self.inclusion_threshold = inclusion_threshold
        self.variant = variant
        self.window_size = window_size
        self._sliding_gallery = deque(maxlen=window_size)
        self._growing_gallery = []
        self._welford = None
        self._sliding_estimator = M2005Alg(threshold=threshold)
        self._growing_estimator = M2005Alg(threshold=threshold)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "inclusion_threshold": self.inclusion_threshold,
            "variant": self.variant,
            "window_size": self.window_size,
        }

    def set_params(self, **params) -> Self:
        self.threshold = params.get("threshold", self.threshold)
        self.inclusion_threshold = params.get("inclusion_threshold", self.inclusion_threshold)
        self.variant = params.get("variant", self.variant)
        self.window_size = params.get("window_size", self.window_size)
        return self

    def fit(self, x: pd.DataFrame, y: list[int] | None = None) -> Self:
        effective_maxlen = self.window_size if self.window_size is not None else max(len(x), 1)
        self._sliding_gallery = deque([row.copy() for _, row in x.iterrows()], maxlen=effective_maxlen)
        self._sliding_estimator.fit(pd.DataFrame(list(self._sliding_gallery)))

        if self.variant == "idb":
            self._welford = _WelfordOnlineStats(x)
            self._growing_estimator._digraphs_metrics = self._welford.get_metrics()
        else:
            self._growing_gallery = [row.copy() for _, row in x.iterrows()]
            self._growing_estimator.fit(pd.DataFrame(self._growing_gallery))

        return self

    def predict_score(self, x: pd.Series) -> float:
        score_sliding = self._sliding_estimator.calculate_score(x)
        score_growing = self._growing_estimator.calculate_score(x)
        return (score_sliding + score_growing) / 2.0

    def should_include_sample(self, x: pd.Series | pd.DataFrame) -> bool:
        effective_threshold = self.inclusion_threshold if self.inclusion_threshold is not None else self.threshold
        if isinstance(x, pd.Series):
            return self.predict_score(x) >= effective_threshold
        return all(self.predict_score(row) >= effective_threshold for _, row in x.iterrows())

    def predict_vec(self, x: pd.Series) -> int:
        score = self.predict_score(x)
        return GENUINE_LABEL if score >= self.threshold else IMPOSTOR_LABEL

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_vec(row) for _, row in x.iterrows()])

    def adapt(self, x: pd.Series | pd.DataFrame) -> None:
        if isinstance(x, pd.Series):
            rows = [x]
        else:
            rows = [row for _, row in x.iterrows()]

        for row in rows:
            self._sliding_gallery.append(row.copy())
            if self.variant == "idb" and self._welford is not None:
                self._welford.update(row)
                self._growing_estimator._digraphs_metrics = self._welford.get_metrics()
            elif self.variant == "db":
                self._growing_gallery.append(row.copy())
                self._growing_estimator.fit(pd.DataFrame(self._growing_gallery))

        self._sliding_estimator.fit(pd.DataFrame(list(self._sliding_gallery)))
