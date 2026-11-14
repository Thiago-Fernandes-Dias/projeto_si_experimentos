import logging

from lib.estimators.self_detector_alg import SelfDetectorAlg
from lib.estimators.m2005_adaptive_alg import AdaptiveM2005Alg
from lib.estimators.m2005_alg import M2005Alg
from lib.experiment_executor import ExperimentExecutor
from lib.logging_config import configure_logging
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.adaptive_stream_runner import AdaptiveStreamRunner


def main() -> None:
    configure_logging()

    repo = results_repository_factory()

    configurations = [
        ("SelfDet_NoAdapt", lambda s: SelfDetectorAlg(radius=0.02, strategy="none")),
        ("SelfDet_Sliding", lambda s: SelfDetectorAlg(radius=0.02, strategy="sliding")),
        ("SelfDet_UsageControlR", lambda s: SelfDetectorAlg(radius=0.02, strategy="usage_control_r")),
        ("SelfDet_UsageControl2", lambda s: SelfDetectorAlg(radius=0.02, strategy="usage_control_2")),
        ("M2005_IDB", lambda s: AdaptiveM2005Alg(threshold=0.7, variant="idb")),
        ("M2005_DB", lambda s: AdaptiveM2005Alg(threshold=0.7, variant="db")),
        (
            "M2005_Pisani2024_NoAdapt",
            lambda s: M2005Alg(window_strategy="none", threshold_method="min_non_outlier", dynamic_threshold=False),
        ),
        (
            "M2005_Pisani2024_Growing_Fixed",
            lambda s: M2005Alg(window_strategy="growing", threshold_method="min_non_outlier", dynamic_threshold=False),
        ),
        (
            "M2005_Pisani2024_Sliding_Fixed",
            lambda s: M2005Alg(window_strategy="sliding", threshold_method="min_non_outlier", dynamic_threshold=False),
        ),
        (
            "M2005_Pisani2024_Growing_Dynamic",
            lambda s: M2005Alg(window_strategy="growing", threshold_method="min_non_outlier", dynamic_threshold=True),
        ),
        (
            "M2005_Pisani2024_Sliding_Dynamic",
            lambda s: M2005Alg(window_strategy="sliding", threshold_method="min_non_outlier", dynamic_threshold=True),
        ),
    ]

    for exp_title, factory in configurations:
        logging.info(f"--- Running: {exp_title} ---")
        executor = ExperimentExecutor(
            runner_factory=lambda ds, title=exp_title, fac=factory: AdaptiveStreamRunner(
                dataset=ds,
                estimator_factory=fac,
                exp_name=title,
                results_repo=repo,
                impostor_ratio=0.3,
                seeds_range=[0],
            )
        )
        executor.execute()


if __name__ == "__main__":
    main()
