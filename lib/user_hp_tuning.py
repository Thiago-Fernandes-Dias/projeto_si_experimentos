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
from lib.utils import create_labels, sorted_nicely, IMPOSTOR_LABEL, GENUINE_LABEL


class UserHPTuning:
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
        ordered_users = sorted_nicely([str(uk) for uk in self._dataset.user_keys()])
        total_users = len(ordered_users)

        log_structured(
            self.logger,
            logging.INFO,
            f"Starting user HPO search (seed={self._seed}, users={total_users}, configs={len(grid)})",
            event="hpo_user_started",
            seed=self._seed,
            total_users=total_users,
            total_configs=len(grid),
        )

        user_best_param_config_map: dict[str, Any] = {}
        for user_idx, uk in enumerate(ordered_users, start=1):
            user_start_time = datetime.now()
            evaluations = [self._evaluate_config_by_user(pc, uk) for pc in grid]
            best_bacc, best_params = max(evaluations, key=lambda item: item[0])
            user_best_param_config_map[uk] = best_params
            user_elapsed = round((datetime.now() - user_start_time).total_seconds(), 4)
            log_structured(
                self.logger,
                logging.INFO,
                f"User HPO for {uk} ({user_idx}/{total_users}) completed in {user_elapsed:.4f}s (bacc={best_bacc:.4f})",
                event="hpo_user_completed",
                seed=self._seed,
                user_id=str(uk),
                user_progress=f"{user_idx}/{total_users}",
                best_bacc=round(best_bacc, 6),
                best_params=best_params,
                elapsed_seconds=user_elapsed,
            )

        elapsed = round((datetime.now() - start_time).total_seconds(), 4)
        log_structured(
            self.logger,
            logging.INFO,
            f"User HPO search (seed={self._seed}) finished in {elapsed:.4f}s",
            event="hpo_user_search_completed",
            seed=self._seed,
            total_users=total_users,
            elapsed_seconds=elapsed,
        )

        return user_best_param_config_map

    def _evaluate_config_by_user(self, param_config: dict[str, Any], uk: str) -> tuple[float, dict[str, Any]]:
        cv = KFold(n_splits=5, shuffle=True, random_state=self._seed)
        split_baccs = []
        x_genuine, y_genuine, x_impostor, y_impostor = \
            self._dataset.two_class_training_set(uk)
        for gss, iss in zip(cv.split(x_genuine, y_genuine),
                            cv.split(x_impostor, y_impostor)):
            x_train = x_genuine.drop(columns=self._dataset.get_drop_columns()).iloc[gss[0]]
            x_g_test = x_genuine.drop(columns=self._dataset.get_drop_columns()).iloc[gss[1]]
            x_i_test = x_impostor.drop(columns=self._dataset.get_drop_columns()).iloc[iss[1]]
            estimator = self._estimator_factory().set_params(**param_config)
            y_train = create_labels(x_train, GENUINE_LABEL)
            if self._use_impostor_samples:
                x_i_train = x_impostor.drop(columns=self._dataset.get_drop_columns()).iloc[iss[1]]
                x_train = pd.concat([x_train, x_i_train])
                y_train = y_train + create_labels(x_i_train, IMPOSTOR_LABEL)
            estimator.fit(x_train, y_train)
            g_pred = estimator.predict(x_g_test)
            i_pred = estimator.predict(x_i_test)
            g_recall = accuracy_score(create_labels(x_g_test, GENUINE_LABEL), g_pred)
            i_recall = accuracy_score(create_labels(x_i_test, IMPOSTOR_LABEL), i_pred)
            split_baccs.append((g_recall + i_recall) / 2)
        average_bacc = np.average(split_baccs).item()
        return average_bacc, param_config
