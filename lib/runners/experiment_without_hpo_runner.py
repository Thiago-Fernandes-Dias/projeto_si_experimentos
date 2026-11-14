from sklearn.base import BaseEstimator

from lib.datasets.dataset import *
from lib.logging_config import experiment_log_context
from lib.repositories.results_repository import ResultsRepository
from lib.runners.experiment_runner import ExperimentRunner


class ExperimentWithoutHPORunner(ExperimentRunner):
    _estimator_factory: Callable[[int], BaseEstimator]

    def __init__(self, dataset: Dataset, estimator_factory: Callable[[int], BaseEstimator], exp_name: str, results_repo: ResultsRepository,
                 use_impostor_samples: bool, seeds_range: list[int] | None = None) -> None:
        super().__init__(dataset=dataset, use_impostor_samples=use_impostor_samples, exp_name=exp_name,
                         results_repo=results_repo, seeds_range=seeds_range)
        self._estimator_factory = estimator_factory
        self.logger = logging.getLogger(__name__)

    def exec(self) -> None:
        start_time = datetime.now()
        seeds_to_run = list(self._seeds_range) if self._use_impostor_samples else [0]
        total_seeds = len(seeds_to_run)

        with experiment_log_context(self._exp_name, start_time):
            self._log_experiment_start(start_time, total_seeds=total_seeds)

            for seed_idx, seed in enumerate(seeds_to_run, start=1):
                seed_start_time = datetime.now()
                self._log_seed_start(seed=seed, seed_idx=seed_idx, total_seeds=total_seeds)
                pred_df = pd.DataFrame(self._train_and_predict(seed=seed, start_time=start_time))
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

    def _train_and_predict(self, start_time: datetime, seed: int = 0) -> list[pd.Series]:
        pred_frames: list[pd.Series] = []
        self._dataset.set_seed(seed)
        estimator = self._estimator_factory(seed)
        self._results_repository.add_hp(
            hp=estimator.get_params(),
            exp_name=self._exp_name,
            date=start_time,
            seed=seed,
        )
        ordered_users = self._ordered_user_keys()
        total_users = len(ordered_users)
        for user_idx, uk in enumerate(ordered_users, start=1):
            user_start_time = datetime.now()
            x_training, y_training = self._get_user_training_vectors(uk)
            estimator.fit(x_training.drop(columns=self._dataset.get_drop_columns()), y_training)
            user_preds = self._test_user_model(estimator=estimator, uk=uk, seed=seed)
            pred_frames += user_preds
            self._log_user_completion(
                uk=uk,
                user_idx=user_idx,
                total_users=total_users,
                seed=seed,
                user_start_time=user_start_time,
                predictions_count=len(user_preds),
            )
        return pred_frames
