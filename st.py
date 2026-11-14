import logging
import os

from lib.estimators.m2005_alg import M2005Alg
from lib.experiment_executor import ExperimentExecutor
from lib.logging_config import configure_logging
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_without_hpo_runner import ExperimentWithoutHPORunner


def main() -> None:
    configure_logging()

    executor = ExperimentExecutor(
        runner_factory=lambda ds: ExperimentWithoutHPORunner(
            results_repo=results_repository_factory(),
            dataset=ds, estimator_factory=lambda s: M2005Alg(),
            exp_name="Magalhães",
            use_impostor_samples=False
        ),
    )
    executor.execute()


if __name__ == "__main__":
    main()
