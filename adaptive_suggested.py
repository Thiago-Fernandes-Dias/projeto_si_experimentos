import logging

from lib.estimators.scaled_manhattan_alg import ScaledManhattanAlg
from lib.estimators.knn_instance_alg import KNNInstanceAlg
from lib.experiment_executor import ExperimentExecutor
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.adaptive_stream_runner import AdaptiveStreamRunner
from download_cmu_dataset import download_cmu_if_missing
from download_keyrecs_dataset import download_keyrecs_if_missing


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    download_cmu_if_missing()
    download_keyrecs_if_missing()

    repo = results_repository_factory()

    configurations = [
        ("ScaledManhattan_Static", lambda s: ScaledManhattanAlg(threshold=2.5, window_strategy="none")),
        (
            "ScaledManhattan_Sliding",
            lambda s: ScaledManhattanAlg(threshold=2.5, sliding_window_size=10, window_strategy="sliding"),
        ),
        ("ScaledManhattan_Growing", lambda s: ScaledManhattanAlg(threshold=2.5, window_strategy="growing")),
        ("KNN_Static", lambda s: KNNInstanceAlg(k=1, threshold=1.2, sliding_window_size=None)),
        ("KNN_Sliding", lambda s: KNNInstanceAlg(k=1, threshold=1.2, sliding_window_size=10)),
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
