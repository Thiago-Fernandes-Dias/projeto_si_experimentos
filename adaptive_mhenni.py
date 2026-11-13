import logging
import os

from lib.datasets.cmu_dataset import CMUDataset
from lib.estimators.hocquet_alg import HocquetStatisticalAlg
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.adaptive_session_runner import AdaptiveSessionRunner
from lib.utils import CMU_PATH, cmu_split
from download_cmu_dataset import download_cmu_if_missing


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    download_cmu_if_missing()

    cmu = CMUDataset(CMU_PATH, cmu_split)
    repo = results_repository_factory()

    configurations = [
        ("Mhenni_None_Fixed", "none", "fixed", 0.5),
        ("Mhenni_Sliding_Fixed", "sliding", "fixed", 0.5),
        ("Mhenni_Sliding_Variable", "sliding", "variable", 0.5),
        ("Mhenni_Growing_Fixed", "growing", "fixed", 0.5),
        ("Mhenni_Growing_Variable", "growing", "variable", 0.5),
    ]

    for exp_title, win_strat, thresh_policy, init_t in configurations:
        logging.info(f"--- Running: {exp_title} ---")
        runner = AdaptiveSessionRunner(
            dataset=cmu,
            estimator_factory=lambda s: HocquetStatisticalAlg(threshold=init_t),
            exp_name=f"{exp_title} (CMU)",
            results_repo=repo,
            window_strategy=win_strat,
            threshold_policy=thresh_policy,
            minimum_threshold=0.05,
            seeds_range=[0],
        )
        runner.exec()


if __name__ == "__main__":
    main()
