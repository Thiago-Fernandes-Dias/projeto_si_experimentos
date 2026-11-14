from datetime import datetime
import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from lib.datasets.dataset import Dataset
from lib.logging_config import experiment_log_context
from lib.repositories.results_repository import ResultsRepository
from lib.runners.experiment_runner import ExperimentRunner
from lib.user_model_prediction import UserModelPrediction
from lib.utils import GENUINE_LABEL, IMPOSTOR_LABEL, interleave_preserving_chronology


class AdaptiveStreamRunner(ExperimentRunner):
    """Runner that evaluates keystroke models in a sequential data stream context
    (Pisani et al., 2015; Pisani, 2024).

    Test queries arrive sequentially in chronological order. When a query is predicted
    as GENUINE_LABEL and satisfies the inclusion threshold (double thresholding),
    the estimator is adapted online using only accepted genuine candidates.
    """

    _estimator_factory: Callable[[int], BaseEstimator]
    _impostor_ratio: float
    _threshold_method: str | None
    _dynamic_threshold: bool

    def __init__(
        self,
        dataset: Dataset,
        estimator_factory: Callable[[int], BaseEstimator],
        exp_name: str,
        results_repo: ResultsRepository,
        impostor_ratio: float = 0.3,
        threshold_method: str | None = None,
        dynamic_threshold: bool = False,
        seeds_range: list[int] | None = None,
    ) -> None:
        super().__init__(
            dataset=dataset,
            use_impostor_samples=False,
            exp_name=exp_name,
            results_repo=results_repo,
            seeds_range=seeds_range,
        )
        self._estimator_factory = estimator_factory
        self._impostor_ratio = impostor_ratio
        self._threshold_method = threshold_method
        self._dynamic_threshold = dynamic_threshold
        self.logger = logging.getLogger(__name__)

    def exec(self) -> None:
        start_time = datetime.now()
        seeds_to_run = list(self._seeds_range)
        total_seeds = len(seeds_to_run)

        with experiment_log_context(self._exp_name, start_time):
            self._log_experiment_start(start_time, total_seeds=total_seeds)

            for seed_idx, seed in enumerate(seeds_to_run, start=1):
                seed_start_time = datetime.now()
                self._log_seed_start(seed=seed, seed_idx=seed_idx, total_seeds=total_seeds)
                pred_df = pd.DataFrame(self._train_and_predict_stream(seed=seed, start_time=start_time))
                self._results_repository.add_predictions_frame(
                    predictions_frame=pred_df,
                    date=start_time,
                    exp_name=self._exp_name,
                    seed=seed,
                )
                self._log_seed_completion(
                    seed=seed,
                    seed_idx=seed_idx,
                    total_seeds=total_seeds,
                    seed_start_time=seed_start_time,
                )

            self._log_experiment_completion(start_time)

    def _train_and_predict_stream(self, start_time: datetime, seed: int = 0) -> list[pd.Series]:
        pred_frames: list[pd.Series] = []
        self._dataset.set_seed(seed)
        rng = np.random.default_rng(seed)

        base_estimator = self._estimator_factory(seed)
        self._results_repository.add_hp(
            hp=base_estimator.get_params(),
            exp_name=self._exp_name,
            date=start_time,
            seed=seed,
        )

        session_col = self._dataset.get_session_key_name()
        rep_col = self._dataset.get_repetition_key_name()
        user_col = self._dataset.user_key_name()

        ordered_users = self._ordered_user_keys()
        total_users = len(ordered_users)

        for user_idx, uk in enumerate(ordered_users, start=1):
            user_start_time = datetime.now()
            estimator = self._estimator_factory(seed)
            x_training, y_training = self._get_user_training_vectors(uk)
            clean_training = x_training.drop(columns=self._dataset.get_drop_columns())
            estimator.fit(clean_training, y_training)

            if self._threshold_method is not None and hasattr(estimator, "calibrate_threshold"):
                estimator.calibrate_threshold(method=self._threshold_method)

            genuine_test = self._X_genuine_test[uk].sort_values(
                [session_col, rep_col],
                kind="stable",
            )
            impostor_test = self._X_impostors_test[uk]

            total_genuine = len(genuine_test)
            target_impostors = int(round(self._impostor_ratio / (1.0 - self._impostor_ratio) * total_genuine))
            target_impostors = min(target_impostors, len(impostor_test))

            sampled_impostors = impostor_test.sample(n=target_impostors, random_state=seed).sort_values(
                [user_col, session_col, rep_col],
                kind="stable",
            )

            genuine_test_labeled = genuine_test.copy()
            genuine_test_labeled["_label"] = GENUINE_LABEL
            sampled_impostors_labeled = sampled_impostors.copy()
            sampled_impostors_labeled["_label"] = IMPOSTOR_LABEL

            chronological_stream = interleave_preserving_chronology(
                genuine_df=genuine_test_labeled,
                impostor_df=sampled_impostors_labeled,
                rng=rng,
            )

            adapted_samples = 0
            for _, row in chronological_stream.iterrows():
                expected_label = int(row["_label"])
                row_features = pd.DataFrame([row.drop(labels=self._dataset.get_drop_columns() + ["_label"]).astype(float)])
                predicted_label = int(estimator.predict(row_features)[0])

                if predicted_label == GENUINE_LABEL and hasattr(estimator, "adapt"):
                    can_include = (
                        estimator.should_include_sample(row_features)
                        if hasattr(estimator, "should_include_sample")
                        else True
                    )
                    if can_include:
                        estimator.adapt(row_features)
                        adapted_samples += 1
                        if (
                            self._dynamic_threshold
                            and self._threshold_method is not None
                            and hasattr(estimator, "calibrate_threshold")
                        ):
                            estimator.calibrate_threshold(method=self._threshold_method)

                pred = UserModelPrediction(
                    user_id=uk,
                    expected=expected_label,
                    predicted=predicted_label,
                    session=int(row[session_col]),
                    repetition=int(row[rep_col]),
                )
                pred_dict = pred.to_dict()
                pred_dict["seed"] = seed
                pred_frames.append(pd.Series(pred_dict))

            self._log_user_completion(
                uk=uk,
                user_idx=user_idx,
                total_users=total_users,
                seed=seed,
                user_start_time=user_start_time,
                predictions_count=len(chronological_stream),
                adapted_samples=adapted_samples,
            )

        return pred_frames
