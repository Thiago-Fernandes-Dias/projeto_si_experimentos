from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from lib.utils import IMPOSTOR_LABEL, GENUINE_LABEL


class DigraphsMetrics:
    def __init__(self, average: float, std_dev: float, median: float):
        self.average = average
        self.std_dev = std_dev
        self.median = median

    average: float
    std_dev: float
    median: float


class M2005Alg(BaseEstimator):
    """M2005 Statistical Classifier (Magalhães et al., 2005; Pisani, 2024).

    Classifies keystroke patterns based on per-digraph mean, median,
    and standard deviation boundaries. Supports sliding/growing gallery
    updates and user-specific threshold calibration ('min', 'min_non_outlier').
    """

    threshold: float
    inclusion_threshold: float | None
    window_strategy: str
    window_size: int | None
    threshold_method: str | None
    dynamic_threshold: bool
    _digraphs_metrics: dict[str, DigraphsMetrics]
    _training_data: pd.DataFrame | None
    _gallery: list[pd.Series]
    _effective_window_size: int

    def __init__(
        self,
        threshold: float = 0.7,
        inclusion_threshold: float | None = None,
        window_strategy: str = "none",
        window_size: int | None = None,
        threshold_method: str | None = None,
        dynamic_threshold: bool = False,
    ):
        self.threshold = threshold
        self.inclusion_threshold = inclusion_threshold
        self.window_strategy = window_strategy
        self.window_size = window_size
        self.threshold_method = threshold_method
        self.dynamic_threshold = dynamic_threshold
        self._digraphs_metrics = {}
        self._training_data = None
        self._gallery = []
        self._effective_window_size = window_size if window_size is not None else 50

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "inclusion_threshold": self.inclusion_threshold,
            "window_strategy": self.window_strategy,
            "window_size": self.window_size,
            "threshold_method": self.threshold_method,
            "dynamic_threshold": self.dynamic_threshold,
        }

    def set_params(self, **params) -> Self:
        self.threshold = params.get("threshold", self.threshold)
        self.inclusion_threshold = params.get("inclusion_threshold", self.inclusion_threshold)
        self.window_strategy = params.get("window_strategy", self.window_strategy)
        self.window_size = params.get("window_size", self.window_size)
        self.threshold_method = params.get("threshold_method", self.threshold_method)
        self.dynamic_threshold = params.get("dynamic_threshold", self.dynamic_threshold)
        return self

    def _compute_metrics_from_dataframe(self, df: pd.DataFrame) -> None:
        self._digraphs_metrics = {}
        for digraph_name, values in df.items():
            average = float(values.mean())
            std_dev = float(values.std())
            median = float(values.median())
            self._digraphs_metrics[str(digraph_name)] = DigraphsMetrics(average, std_dev, median)
        self._training_data = df

    def fit(self, x: pd.DataFrame, y: list[int] | None = None) -> Self:
        self._gallery = [row.copy() for _, row in x.iterrows()]
        self._effective_window_size = self.window_size if self.window_size is not None else max(len(self._gallery), 1)
        self._compute_metrics_from_dataframe(x)
        if self.threshold_method is not None:
            self.calibrate_threshold(method=self.threshold_method)
        return self

    def calculate_score(self, x: pd.Series) -> float:
        hits = []
        for digraph_name, value in x.items():
            digraph_metrics = self._digraphs_metrics.get(str(digraph_name))
            if digraph_metrics is None:
                continue
            av = digraph_metrics.average if digraph_metrics.average != 0 else 1.0
            std = digraph_metrics.std_dev
            md = digraph_metrics.median
            res = (min(av, md) * (0.95 - std / av)) <= value <= (max(av, md) * (1.05 + std / av))
            hits.append(res)
        length: int = len(hits)
        if length == 0:
            return 0.0
        score_sum: float = 1.0 if hits[0] else 0.0
        for i in range(1, length):
            if hits[i]:
                score_sum += 1.5 if hits[i - 1] else 1.0
        return float(score_sum / ((length - 1) * 1.5 + 1.0))

    def calibrate_threshold(
        self,
        method: str = "min_non_outlier",
        outlier_std_factor: float = 2.5,
    ) -> float:
        """Computes user-specific threshold from gallery scores (Pisani, SBSeg 2024)."""
        if not self._gallery:
            return self.threshold
        scores = np.array([self.calculate_score(sample) for sample in self._gallery], dtype=float)
        if method == "min" or len(scores) <= 1:
            self.threshold = float(np.min(scores))
            return self.threshold

        # Pisani (2024) min-non-outlier: S_j^n = {s in S_j | s > mean(S_j) - 2.5 * std(S_j)}
        lower_cutoff = float(np.mean(scores) - outlier_std_factor * np.std(scores, ddof=1))
        filtered_scores = scores[scores > lower_cutoff]
        self.threshold = float(np.min(filtered_scores)) if len(filtered_scores) > 0 else float(np.min(scores))
        return self.threshold

    def should_include_sample(self, x: pd.Series | pd.DataFrame) -> bool:
        effective_threshold = self.inclusion_threshold if self.inclusion_threshold is not None else self.threshold
        if isinstance(x, pd.Series):
            return self.calculate_score(x) >= effective_threshold
        return all(self.calculate_score(row) >= effective_threshold for _, row in x.iterrows())

    def adapt(
        self,
        x: pd.Series | pd.DataFrame,
        window: str | None = None,
        max_size: int | None = None,
    ) -> None:
        strategy = window if window is not None else self.window_strategy
        if strategy == "none":
            return

        if isinstance(x, pd.Series):
            new_rows = [x.copy()]
        else:
            new_rows = [row.copy() for _, row in x.iterrows()]

        window_limit = max_size if max_size is not None else self._effective_window_size

        if strategy == "sliding":
            for row in new_rows:
                if len(self._gallery) >= window_limit:
                    self._gallery.pop(0)
                self._gallery.append(row)
        elif strategy == "growing":
            self._gallery.extend(new_rows)

        if self._gallery:
            self._compute_metrics_from_dataframe(pd.DataFrame(self._gallery))
            if self.dynamic_threshold and self.threshold_method is not None:
                self.calibrate_threshold(method=self.threshold_method)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        results: list[int] = []
        for _, vec in x.iterrows():
            results.append(self.predict_vec(vec))
        return np.array(results)

    def predict_vec(self, x: pd.Series) -> int:
        score = self.calculate_score(x)
        return GENUINE_LABEL if score >= self.threshold else IMPOSTOR_LABEL


# Aliases for backwards compatibility and nomenclature variations
M2005 = M2005Alg
ImprovedStatisticalAlg = M2005Alg
