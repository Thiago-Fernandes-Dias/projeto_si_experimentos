from abc import abstractmethod, ABC
from datetime import datetime
import logging

from typing import Any
import pandas as pd

from lib.datasets.dataset import Dataset
from lib.logging_config import experiment_log_context, log_structured
from lib.repositories.results_repository import ResultsRepository
from lib.user_model_prediction import UserModelPrediction
from lib.utils import default_seeds_range, sorted_nicely


class ExperimentRunner(ABC):
    _dataset: Dataset
    _results_repository: ResultsRepository
    _exp_name: str
    _use_impostor_samples: bool
    _X_genuine_training: dict[str, pd.DataFrame]
    _y_genuine_training: dict[str, list[int]]
    _X_impostor_training: dict[str, pd.DataFrame]
    _y_impostor_training: dict[str, list[int]]
    _X_genuine_test: dict[str, pd.DataFrame]
    _y_genuine_test: dict[str, list[int]]
    _X_impostors_test: dict[str, pd.DataFrame]
    _y_impostors_test: dict[str, list[int]]
    _seeds_range: list[int]

    def __init__(self, dataset: Dataset, results_repo: ResultsRepository, exp_name: str,
                 use_impostor_samples: bool = False, seeds_range: list[int] | None = None) -> None:
        self._dataset = dataset
        self._results_repository = results_repo
        self._exp_name = exp_name
        self._use_impostor_samples = use_impostor_samples
        self._X_genuine_training = {}
        self._y_genuine_training = {}
        self._X_impostor_training = {}
        self._y_impostor_training = {}
        self._X_genuine_test = {}
        self._y_genuine_test = {}
        self._X_impostors_test = {}
        self._y_impostors_test = {}
        self._seeds_range = seeds_range if seeds_range is not None else list(default_seeds_range)

        self._set_vectors_and_true_labels()
        self._dataset.add_seed_change_cb(self._set_vectors_and_true_labels)

        self.logger = logging.getLogger(__name__)

    def add_name_suffix(self, s: str):
        self._exp_name += f" ({s})"

    @abstractmethod
    def exec(self) -> None:
        pass

    def _set_vectors_and_true_labels(self, seed: int = 0) -> None:
        for uk in self._dataset.user_keys():
            self._X_genuine_training[uk], self._y_genuine_training[uk], \
                self._X_impostor_training[uk], self._y_impostor_training[uk] = self._dataset.two_class_training_set(uk)
            self._X_genuine_test[uk], self._y_genuine_test[uk] = self._dataset.user_test_set(uk)
            self._X_impostors_test[uk], self._y_impostors_test[uk] = self._dataset.impostors_test_set(uk)

    def _get_user_training_vectors(self, uk):
        x_training, y_training = self._X_genuine_training[uk], self._y_genuine_training[uk]
        if self._use_impostor_samples:
            x_training = pd.concat([x_training, self._X_impostor_training[uk]])
            y_training = y_training + self._y_impostor_training[uk]
        return x_training, y_training

    def _test_user_model(self, estimator, uk, seed: int = 0) -> list[pd.Series]:
        pred_frames = []
        x_test = pd.concat([self._X_genuine_test[uk], self._X_impostors_test[uk]])
        y_test = self._y_genuine_test[uk] + self._y_impostors_test[uk]
        for (_, x), y in zip(x_test.iterrows(), y_test):
            x_filtered = pd.DataFrame([x.drop(labels=self._dataset.get_drop_columns()).astype(float)])
            pred = UserModelPrediction(
                user_id=str(uk),
                expected=int(y),
                predicted=int(estimator.predict(x_filtered)[0]),
                session=int(x[self._dataset.get_session_key_name()]),
                repetition=int(x[self._dataset.get_repetition_key_name()]),
            )
            pred_dict = pred.to_dict()
            pred_dict['seed'] = int(seed)
            pred_frame = pd.Series(pred_dict)
            pred_frames.append(pred_frame)
        return pred_frames

    def _ordered_user_keys(self) -> list[str]:
        return sorted_nicely([str(uk) for uk in self._dataset.user_keys()])

    def _log_experiment_start(self, start_time: datetime, total_seeds: int | None = None) -> None:
        seeds_count = total_seeds if total_seeds is not None else len(self._seeds_range)
        total_users = len(self._dataset.user_keys())
        log_structured(
            self.logger,
            logging.INFO,
            f"Starting experiment '{self._exp_name}' ({self.__class__.__name__})",
            event="experiment_started",
            exp_name=self._exp_name,
            runner=self.__class__.__name__,
            total_seeds=seeds_count,
            total_users=total_users,
            started_at=start_time.isoformat(timespec="seconds"),
        )

    def _log_seed_start(self, seed: int, seed_idx: int, total_seeds: int) -> None:
        log_structured(
            self.logger,
            logging.INFO,
            f"[{self._exp_name}] Starting seed {seed} ({seed_idx}/{total_seeds})",
            event="seed_started",
            exp_name=self._exp_name,
            seed=int(seed),
            seed_progress=f"{seed_idx}/{total_seeds}",
            total_users=len(self._dataset.user_keys()),
        )

    def _log_seed_completion(self, seed: int, seed_idx: int, total_seeds: int, seed_start_time: datetime) -> None:
        elapsed = round((datetime.now() - seed_start_time).total_seconds(), 4)
        log_structured(
            self.logger,
            logging.INFO,
            f"[{self._exp_name}] Seed {seed} ({seed_idx}/{total_seeds}) finished in {elapsed:.4f}s",
            event="seed_completed",
            exp_name=self._exp_name,
            seed=int(seed),
            seed_progress=f"{seed_idx}/{total_seeds}",
            elapsed_seconds=elapsed,
        )

    def _log_user_completion(
        self,
        uk: str,
        user_idx: int,
        total_users: int,
        seed: int,
        user_start_time: datetime,
        **extra_fields: Any,
    ) -> None:
        elapsed = round((datetime.now() - user_start_time).total_seconds(), 4)
        log_structured(
            self.logger,
            logging.INFO,
            f"[{self._exp_name}] User {uk} ({user_idx}/{total_users}) completed in {elapsed:.4f}s",
            event="user_completed",
            exp_name=self._exp_name,
            seed=int(seed),
            user_id=str(uk),
            user_progress=f"{user_idx}/{total_users}",
            elapsed_seconds=elapsed,
            **extra_fields,
        )

    def _log_experiment_completion(self, start_time: datetime) -> None:
        elapsed = round((datetime.now() - start_time).total_seconds(), 4)
        log_structured(
            self.logger,
            logging.INFO,
            f"Experiment '{self._exp_name}' finished in {elapsed:.4f}s",
            event="experiment_completed",
            exp_name=self._exp_name,
            runner=self.__class__.__name__,
            elapsed_seconds=elapsed,
        )
