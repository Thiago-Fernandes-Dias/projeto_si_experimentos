import logging
import os

from lib.datasets.cmu_dataset import CMUDataset
from lib.estimators.self_detector_alg import SelfDetectorAlg
from lib.estimators.m2005_adaptive_alg import AdaptiveM2005Alg
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
        ("SelfDet_NoAdapt", lambda s: SelfDetectorAlg(radius=0.02, strategy="none")),
        ("SelfDet_Sliding", lambda s: SelfDetectorAlg(radius=0.02, strategy="sliding")),
        ("SelfDet_UsageControlR", lambda s: SelfDetectorAlg(radius=0.02, strategy="usage_control_r")),
        ("SelfDet_UsageControl2", lambda s: SelfDetectorAlg(radius=0.02, strategy="usage_control_2")),
        ("M2005_IDB", lambda s: AdaptiveM2005Alg(threshold=0.7, variant="idb", window_size=10)),
        ("M2005_DB", lambda s: AdaptiveM2005Alg(threshold=0.7, variant="db", window_size=10)),
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
