import logging
from datetime import datetime
from typing import Callable, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, ParameterGrid

from lib.datasets.dataset import Dataset
from lib.logging_config import log_structured
from lib.utils import create_labels, dict_values_average, sorted_nicely, IMPOSTOR_LABEL, GENUINE_LABEL


class GlobalHPTuning:
    _dataset: Dataset
    _parameter_grid: list[dict[str, Any]]
    _estimator_factory: Callable[[], BaseEstimator]
    _use_impostor_samples: bool
    _seed: int

    def __init__(self, dataset: Dataset, estimator_factory: Callable[[], BaseEstimator],
                 parameter_grid: list[dict[str, Any]], use_impostor_samples: bool, seed: int):
        self._dataset = dataset
        self._estimator_factory = estimator_factory
        self._parameter_grid = parameter_grid
        self._use_impostor_samples = use_impostor_samples
        self._seed = seed
        self.logger = logging.getLogger(__name__)

    def search(self) -> dict[str, Any]:
        start_time = datetime.now()
        grid = list(ParameterGrid(self._parameter_grid))
        total_configs = len(grid)
        log_structured(
            self.logger,
            logging.INFO,
            f"Starting global HPO search (seed={self._seed}, configs={total_configs})",
            event="hpo_global_started",
            seed=self._seed,
            total_configs=total_configs,
        )

        results: list[tuple[float, dict[str, Any]]] = []
        for config_idx, param_config in enumerate(grid, start=1):
            config_start = datetime.now()
            bacc, cfg = self._evaluate_config(param_config)
            results.append((bacc, cfg))
            config_elapsed = round((datetime.now() - config_start).total_seconds(), 4)
            log_structured(
                self.logger,
                logging.INFO,
                f"Global HPO config {config_idx}/{total_configs} evaluated in {config_elapsed:.4f}s (bacc={bacc:.4f})",
                event="hpo_global_config_evaluated",
                seed=self._seed,
                config_progress=f"{config_idx}/{total_configs}",
                bacc=round(bacc, 6),
                params=cfg,
                elapsed_seconds=config_elapsed,
            )

        best_bacc, best_param_config = max(results, key=lambda item: item[0])
        elapsed = round((datetime.now() - start_time).total_seconds(), 4)
        log_structured(
            self.logger,
            logging.INFO,
            f"Global HPO search (seed={self._seed}) finished in {elapsed:.4f}s",
            event="hpo_global_completed",
            seed=self._seed,
            best_bacc=round(best_bacc, 6),
            best_params=best_param_config,
            elapsed_seconds=elapsed,
        )

        return best_param_config

    def _evaluate_config(self, param_config: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        cv = KFold(n_splits=5, shuffle=True, random_state=self._seed)
        users_bacc_map: dict[str, float] = {}
        ordered_users = sorted_nicely([str(uk) for uk in self._dataset.user_keys()])
        for uk in ordered_users:
            user_baccs: list[float] = []
            x_genuine, y_genuine, x_impostor, y_impostor = \
                self._dataset.two_class_training_set(uk)
            x_genuine = x_genuine.drop(columns=self._dataset.get_drop_columns())
            x_impostor = x_impostor.drop(columns=self._dataset.get_drop_columns())
            for gss, iss in zip(cv.split(x_genuine, y_genuine),
                                cv.split(x_impostor, y_impostor)):
                x_train = x_genuine.iloc[gss[0]]
                x_g_test = x_genuine.iloc[gss[1]]
                x_i_test = x_impostor.iloc[iss[1]]
                estimator = self._estimator_factory().set_params(**param_config)
                y_train = create_labels(x_train, GENUINE_LABEL)
                if self._use_impostor_samples:
                    x_i_train = x_impostor.iloc[iss[1]]
                    x_train = pd.concat([x_train, x_i_train])
                    y_train = y_train + create_labels(x_i_train, IMPOSTOR_LABEL)
                estimator.fit(x_train, y_train)
                g_pred = estimator.predict(x_g_test)
                i_pred = estimator.predict(x_i_test)
                g_recall = accuracy_score(create_labels(x_g_test, GENUINE_LABEL), g_pred)
                i_recall = accuracy_score(create_labels(x_i_test, IMPOSTOR_LABEL), i_pred)
                user_baccs.append((g_recall + i_recall) / 2)
            users_bacc_map[uk] = np.average(user_baccs).item()
        average_bacc = dict_values_average(users_bacc_map)
        return average_bacc, param_config
