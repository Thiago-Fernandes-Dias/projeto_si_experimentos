import logging

from lib.estimators.hocquet_alg import HocquetStatisticalAlg
from lib.experiment_executor import ExperimentExecutor
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_without_hpo_runner import ExperimentWithoutHPORunner


def main() -> None:
    logging.basicConfig(level=logging.INFO)

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
