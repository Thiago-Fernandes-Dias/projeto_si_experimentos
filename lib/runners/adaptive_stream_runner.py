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
from lib.utils import GENUINE_LABEL, IMPOSTOR_LABEL


class AdaptiveStreamRunner(ExperimentRunner):
    """Runner that evaluates keystroke models in a sequential data stream context (Pisani et al., 2015).

    Test queries arrive sequentially in chronological order. When a query is predicted
    as GENUINE_LABEL, the estimator is adapted online (semi-supervised concept drift adaptation).
    """

    _estimator_factory: Callable[[int], BaseEstimator]
    _impostor_ratio: float

    def __init__(
        self,
        dataset: Dataset,
        estimator_factory: Callable[[int], BaseEstimator],
        exp_name: str,
        results_repo: ResultsRepository,
        impostor_ratio: float = 0.3,
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
        self.logger = logging.getLogger(__name__)

    def exec(self) -> None:
        start_time = datetime.now()
        self.logger.info(f"Starting Adaptive Stream Experiment: {self._exp_name} at {start_time}")

        for seed in self._seeds_range:
            pred_df = pd.DataFrame(self._train_and_predict_stream(seed=seed, start_time=start_time))
            self._results_repository.add_predictions_frame(
                predictions_frame=pred_df,
                date=start_time,
                exp_name=self._exp_name,
                seed=seed,
            )

        self._log_experiment_completion(start_time)

    def _train_and_predict_stream(self, start_time: datetime, seed: int = 0) -> list[pd.Series]:
        pred_frames: list[pd.Series] = []
        self._dataset.set_seed(seed)
        rng = np.random.default_rng(seed)

        # Record estimator parameters
        base_estimator = self._estimator_factory(seed)
        self._results_repository.add_hp(
            hp=base_estimator.get_params(),
            exp_name=self._exp_name,
            date=start_time,
            seed=seed,
        )

        for uk in self._dataset.user_keys():
            estimator = self._estimator_factory(seed)
            x_training, y_training = self._get_user_training_vectors(uk)
            clean_training = x_training.drop(columns=self._dataset.get_drop_columns())
            estimator.fit(clean_training, y_training)

            # Build chronological test stream: genuine test queries + interleaved impostors
            genuine_test = self._X_genuine_test[uk].sort_values(
                [self._dataset.get_session_key_name(), self._dataset.get_repetition_key_name()],
                kind="stable",
            )
            impostor_test = self._X_impostors_test[uk]

            total_genuine = len(genuine_test)
            target_impostors = int(round(self._impostor_ratio / (1.0 - self._impostor_ratio) * total_genuine))
            target_impostors = min(target_impostors, len(impostor_test))

            sampled_impostors = impostor_test.sample(n=target_impostors, random_state=seed)

            # Assign labels
            genuine_test_labeled = genuine_test.copy()
            genuine_test_labeled["_label"] = GENUINE_LABEL
            sampled_impostors_labeled = sampled_impostors.copy()
            sampled_impostors_labeled["_label"] = IMPOSTOR_LABEL

            combined = pd.concat([genuine_test_labeled, sampled_impostors_labeled])
            # Interleave randomly while preserving relative order
            shuffled_stream = combined.sample(frac=1.0, random_state=seed)

            for _, row in shuffled_stream.iterrows():
                expected_label = int(row["_label"])
                row_features = pd.DataFrame([row.drop(labels=self._dataset.get_drop_columns() + ["_label"])])
                predicted_label = int(estimator.predict(row_features)[0])

                # If accepted as genuine, adapt model
                if predicted_label == GENUINE_LABEL and hasattr(estimator, "adapt"):
                    estimator.adapt(row_features)

                pred = UserModelPrediction(
                    user_id=uk,
                    expected=expected_label,
                    predicted=predicted_label,
                    session=int(row[self._dataset.get_session_key_name()]),
                    repetition=int(row[self._dataset.get_repetition_key_name()]),
                )
                pred_dict = pred.to_dict()
                pred_dict["seed"] = seed
                pred_frames.append(pd.Series(pred_dict))

        return pred_frames
