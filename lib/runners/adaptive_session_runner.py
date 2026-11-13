from datetime import datetime
import logging
from typing import Callable

import pandas as pd
from sklearn.base import BaseEstimator

from lib.datasets.dataset import Dataset
from lib.repositories.results_repository import ResultsRepository
from lib.runners.experiment_runner import ExperimentRunner
from lib.user_model_prediction import UserModelPrediction
from lib.utils import GENUINE_LABEL, IMPOSTOR_LABEL


class AdaptiveSessionRunner(ExperimentRunner):
    """Runner that evaluates keystroke models in a discrete session context (Mhenni et al., 2016).

    Delayed template update:
      - During a session, queries are evaluated using the current reference and threshold.
      - At the end of the session, accepted queries update the reference template (sliding/growing).
      - If threshold_policy == 'variable', the threshold is updated via eq. (2):
        T_{i+1} = max(minimum_threshold, T_i - exp(-mu / sigma))
    """

    _estimator_factory: Callable[[int], BaseEstimator]
    _window_strategy: str
    _threshold_policy: str
    _minimum_threshold: float

    def __init__(
        self,
        dataset: Dataset,
        estimator_factory: Callable[[int], BaseEstimator],
        exp_name: str,
        results_repo: ResultsRepository,
        window_strategy: str = "sliding",
        threshold_policy: str = "fixed",
        minimum_threshold: float = 0.05,
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

        base_estimator = self._estimator_factory(seed)
        self._results_repository.add_hp(
            hp=base_estimator.get_params(),
            exp_name=self._exp_name,
            date=start_time,
            seed=seed,
        )

        session_col = self._dataset.get_session_key_name()
        rep_col = self._dataset.get_repetition_key_name()

        for uk in self._dataset.user_keys():
            estimator = self._estimator_factory(seed)
            x_training, y_training = self._get_user_training_vectors(uk)
            clean_training = x_training.drop(columns=self._dataset.get_drop_columns())
            estimator.fit(clean_training, y_training)

            # Available test sessions (sessions >= 2)
            genuine_test = self._X_genuine_test[uk]
            test_sessions = sorted(genuine_test[session_col].unique())

            impostor_test = self._X_impostors_test[uk]

            for session_num in test_sessions:
                session_genuine = genuine_test[genuine_test[session_col] == session_num].sort_values(
                    rep_col, kind="stable"
                )
                # Sample equal amount of impostor attacks for this session
                sample_count = len(session_genuine)
                session_impostors = impostor_test.sample(n=min(sample_count, len(impostor_test)), random_state=seed)

                session_genuine_labeled = session_genuine.copy()
                session_genuine_labeled["_label"] = GENUINE_LABEL
                session_impostors_labeled = session_impostors.copy()
                session_impostors_labeled["_label"] = IMPOSTOR_LABEL

                combined_session = pd.concat([session_genuine_labeled, session_impostors_labeled]).sample(
                    frac=1.0, random_state=seed
                )

                accepted_genuine_queries = []

                for _, row in combined_session.iterrows():
                    expected_label = int(row["_label"])
                    row_features = pd.DataFrame([row.drop(labels=self._dataset.get_drop_columns() + ["_label"])])
                    predicted_label = int(estimator.predict(row_features)[0])

                    if predicted_label == GENUINE_LABEL and expected_label == GENUINE_LABEL:
                        accepted_genuine_queries.append(row_features)

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

                # End of session: delayed update with accepted queries
                if accepted_genuine_queries and hasattr(estimator, "adapt") and self._window_strategy != "none":
                    batch_accepted = pd.concat(accepted_genuine_queries)
                    estimator.adapt(batch_accepted, window=self._window_strategy)

                # Variable threshold adjustment
                if self._threshold_policy == "variable" and hasattr(estimator, "calculate_variable_threshold_decrement"):
                    dec = estimator.calculate_variable_threshold_decrement()
                    new_t = max(self._minimum_threshold, estimator.threshold - dec)
                    estimator.threshold = new_t

        return pred_frames
