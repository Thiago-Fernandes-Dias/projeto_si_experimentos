import logging

from lib.estimators.self_detector_alg import SelfDetectorAlg
from lib.experiment_executor import ExperimentExecutor
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_without_hpo_runner import ExperimentWithoutHPORunner
from download_cmu_dataset import download_cmu_if_missing


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    download_cmu_if_missing()

    executor = ExperimentExecutor(
        runner_factory=lambda ds: ExperimentWithoutHPORunner(
            results_repo=results_repository_factory(),
            dataset=ds,
            estimator_factory=lambda s: SelfDetectorAlg(radius=0.02, strategy="none"),
            exp_name="Self-Detector",
            use_impostor_samples=False,
            seeds_range=[0],
        ),
    )
    executor.execute()


if __name__ == "__main__":
    main()
