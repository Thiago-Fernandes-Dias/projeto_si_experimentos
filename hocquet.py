import logging

from lib.estimators.hocquet_alg import HocquetStatisticalAlg
from lib.experiment_executor import ExperimentExecutor
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_without_hpo_runner import ExperimentWithoutHPORunner
from download_cmu_dataset import download_cmu_if_missing
from download_keyrecs_dataset import download_keyrecs_if_missing


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    download_cmu_if_missing()
    download_keyrecs_if_missing()

    executor = ExperimentExecutor(
        runner_factory=lambda ds: ExperimentWithoutHPORunner(
            results_repo=results_repository_factory(),
            dataset=ds,
            estimator_factory=lambda s: HocquetStatisticalAlg(threshold=0.5),
            exp_name="Hocquet",
            use_impostor_samples=False,
            seeds_range=[0],
        ),
    )
    executor.execute()


if __name__ == "__main__":
    main()
