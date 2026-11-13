import logging
import os

from lib.datasets.cmu_dataset import CMUDataset
from lib.estimators.scaled_manhattan_alg import ScaledManhattanAlg
from lib.estimators.knn_instance_alg import KNNInstanceAlg
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.adaptive_stream_runner import AdaptiveStreamRunner
from lib.utils import CMU_PATH, cmu_split
from download_cmu_dataset import download_cmu_if_missing


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    download_cmu_if_missing()

    cmu = CMUDataset(CMU_PATH, cmu_split)
    repo = results_repository_factory()

    configurations = [
        ("ScaledManhattan_Static", lambda s: ScaledManhattanAlg(threshold=2.5, sliding_window_size=None)),
        ("ScaledManhattan_Sliding", lambda s: ScaledManhattanAlg(threshold=2.5, sliding_window_size=10)),
        ("KNN_Static", lambda s: KNNInstanceAlg(k=1, threshold=1.2, sliding_window_size=None)),
        ("KNN_Sliding", lambda s: KNNInstanceAlg(k=1, threshold=1.2, sliding_window_size=10)),
    ]

    for exp_title, factory in configurations:
        logging.info(f"--- Running: {exp_title} ---")
        runner = AdaptiveStreamRunner(
            dataset=cmu,
            estimator_factory=factory,
            exp_name=f"{exp_title} (CMU)",
            results_repo=repo,
            impostor_ratio=0.3,
            seeds_range=[0],
        )
        runner.exec()


if __name__ == "__main__":
    main()
