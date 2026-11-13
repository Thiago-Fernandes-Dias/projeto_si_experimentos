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
    """M2005 Statistical Classifier (Magalhães et al., 2005).

    Classifies keystroke patterns based on per-digraph mean, median,
    and standard deviation boundaries.
    """

    threshold: float
    _digraphs_metrics: dict[str, DigraphsMetrics]
    _training_data: pd.DataFrame | None

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self._digraphs_metrics = {}
        self._training_data = None

    def get_params(self, deep: bool = True) -> dict[str, float]:
        return {
            "threshold": self.threshold
        }

    def set_params(self, **params) -> Self:
        self.threshold = params.get("threshold", self.threshold)
        return self

    def fit(self, x: pd.DataFrame, y: list[int] | None = None) -> Self:
        self._digraphs_metrics = {}
        for digraph_name, values in x.items():
            average = values.mean()
            std_dev = values.std()
            median = values.median()
            self._digraphs_metrics[str(digraph_name)] = DigraphsMetrics(average, std_dev, median)
        self._training_data = x
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        results: list[int] = []
        for _, vec in x.iterrows():
            results.append(self.predict_vec(vec))
        return np.array(results)

    def predict_vec(self, x: pd.Series) -> int:
        hits = []
        for digraph_name, value in x.items():
            digraph_metrics = self._digraphs_metrics[str(digraph_name)]
            av = digraph_metrics.average if digraph_metrics.average != 0 else 1
            std = digraph_metrics.std_dev
            md = digraph_metrics.median
            res = (min(av, md) * (0.95 - std / av)) <= value <= (max(av, md) * (1.05 + std / av))
            hits.append(res)
        length: int = len(hits)
        s: float = 1 if hits[0] else 0
        for i in range(1, length):
            if hits[i]:
                s += 1.5 if hits[i - 1] else 1
        result = GENUINE_LABEL if (s / ((length - 1) * 1.5 + 1)) >= self.threshold else IMPOSTOR_LABEL
        return result


# Aliases for backwards compatibility and nomenclature variations
M2005 = M2005Alg
ImprovedStatisticalAlg = M2005Alg
