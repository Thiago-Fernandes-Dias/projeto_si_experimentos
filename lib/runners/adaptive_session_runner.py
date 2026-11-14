from datetime import datetime
import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from lib.datasets.dataset import Dataset
from lib.repositories.results_repository import ResultsRepository
from lib.runners.experiment_runner import ExperimentRunner
from lib.user_model_prediction import UserModelPrediction
from lib.utils import GENUINE_LABEL, IMPOSTOR_LABEL, interleave_preserving_chronology


class AdaptiveSessionRunner(ExperimentRunner):
    """Runner that evaluates keystroke models in a discrete session context (Mhenni et al., 2016).

    Delayed semi-supervised template update with double thresholding:
      - During a session, queries are verified against estimator.threshold.
      - Accepted queries that also satisfy estimator.should_include_sample (inclusion_threshold)
        are buffered for end-of-session template adaptation (no ground-truth oracle).
      - At the end of the session, buffered queries update the reference template (sliding/growing).
      - Threshold policies:
          * 'fixed': global fixed threshold across all users and sessions.
          * 'individual': user-specific threshold calibrated at enrollment from user gallery.
          * 'variable': threshold tightened after each session via Mhenni et al. (2016) eq. (2):
            T_{i+1} = max(minimum_threshold, T_i - exp(-mu / sigma))
    """

    _estimator_factory: Callable[[int], BaseEstimator]
    _window_strategy: str
    _threshold_policy: str
    _minimum_threshold: float
    _inclusion_threshold_ratio: float | None

    def __init__(
        self,
        dataset: Dataset,
        estimator_factory: Callable[[int], BaseEstimator],
        exp_name: str,
        results_repo: ResultsRepository,
        window_strategy: str = "sliding",
        threshold_policy: str = "fixed",
        minimum_threshold: float = 0.05,
        inclusion_threshold_ratio: float | None = 0.85,
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
        self._window_strategy = window_strategy
        self._threshold_policy = threshold_policy
        self._minimum_threshold = minimum_threshold
        self._inclusion_threshold_ratio = inclusion_threshold_ratio
        self.logger = logging.getLogger(__name__)

    def exec(self) -> None:
        start_time = datetime.now()
        self.logger.info(f"Starting Adaptive Session Experiment: {self._exp_name} at {start_time}")

        for seed in self._seeds_range:
            pred_df = pd.DataFrame(self._train_and_predict_sessions(seed=seed, start_time=start_time))
            self._results_repository.add_predictions_frame(
                predictions_frame=pred_df,
                date=start_time,
                exp_name=self._exp_name,
                seed=seed,
            )

        self._log_experiment_completion(start_time)

    def _train_and_predict_sessions(self, start_time: datetime, seed: int = 0) -> list[pd.Series]:
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

        for uk in self._dataset.user_keys():
            estimator = self._estimator_factory(seed)
            x_training, y_training = self._get_user_training_vectors(uk)
            clean_training = x_training.drop(columns=self._dataset.get_drop_columns())
            estimator.fit(clean_training, y_training)

            if self._threshold_policy == "individual" and hasattr(estimator, "calibrate_user_threshold"):
                estimator.calibrate_user_threshold(inclusion_ratio=self._inclusion_threshold_ratio)
            elif (
                self._inclusion_threshold_ratio is not None
                and hasattr(estimator, "inclusion_threshold")
                and getattr(estimator, "inclusion_threshold") is None
            ):
                estimator.inclusion_threshold = estimator.threshold * self._inclusion_threshold_ratio

            genuine_test = self._X_genuine_test[uk]
            test_sessions = sorted(genuine_test[session_col].unique())
            impostor_test = self._X_impostors_test[uk]

            for session_num in test_sessions:
                session_genuine = genuine_test[genuine_test[session_col] == session_num].sort_values(
                    rep_col, kind="stable"
                )
                sample_count = len(session_genuine)
                session_impostors = impostor_test.sample(
                    n=min(sample_count, len(impostor_test)),
                    random_state=seed + int(session_num),
                ).sort_values([user_col, session_col, rep_col], kind="stable")

                session_genuine_labeled = session_genuine.copy()
                session_genuine_labeled["_label"] = GENUINE_LABEL
                session_impostors_labeled = session_impostors.copy()
                session_impostors_labeled["_label"] = IMPOSTOR_LABEL

                combined_session = interleave_preserving_chronology(
                    genuine_df=session_genuine_labeled,
                    impostor_df=session_impostors_labeled,
                    rng=rng,
                )

                accepted_queries_for_update = []

                for _, row in combined_session.iterrows():
                    expected_label = int(row["_label"])
                    row_features = pd.DataFrame([row.drop(labels=self._dataset.get_drop_columns() + ["_label"]).astype(float)])
                    predicted_label = int(estimator.predict(row_features)[0])

                    if predicted_label == GENUINE_LABEL:
                        can_include = (
                            estimator.should_include_sample(row_features)
                            if hasattr(estimator, "should_include_sample")
                            else True
                        )
                        if can_include:
                            accepted_queries_for_update.append(row_features)

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

                if accepted_queries_for_update and hasattr(estimator, "adapt") and self._window_strategy != "none":
                    batch_accepted = pd.concat(accepted_queries_for_update)
                    estimator.adapt(batch_accepted, window=self._window_strategy)

                if self._threshold_policy == "variable" and hasattr(estimator, "calculate_variable_threshold_decrement"):
                    decrement = estimator.calculate_variable_threshold_decrement()
                    new_threshold = max(self._minimum_threshold, estimator.threshold - decrement)
                    estimator.threshold = new_threshold
                    if self._inclusion_threshold_ratio is not None and hasattr(estimator, "inclusion_threshold"):
                        estimator.inclusion_threshold = new_threshold * self._inclusion_threshold_ratio

        return pred_frames
