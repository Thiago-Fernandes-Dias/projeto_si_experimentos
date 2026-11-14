import logging

from lib.estimators.hocquet_alg import HocquetStatisticalAlg
from lib.experiment_executor import ExperimentExecutor
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.adaptive_session_runner import AdaptiveSessionRunner


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    repo = results_repository_factory()

    configurations = [
        ("Mhenni_None_Fixed", "none", "fixed", 0.5),
        ("Mhenni_Sliding_Fixed", "sliding", "fixed", 0.5),
        ("Mhenni_Sliding_Individual", "sliding", "individual", 0.5),
        ("Mhenni_Sliding_Variable", "sliding", "variable", 0.5),
        ("Mhenni_Growing_Fixed", "growing", "fixed", 0.5),
        ("Mhenni_Growing_Individual", "growing", "individual", 0.5),
        ("Mhenni_Growing_Variable", "growing", "variable", 0.5),
    ]

    for exp_title, win_strat, thresh_policy, init_t in configurations:
        logging.info(f"--- Running: {exp_title} ---")
        executor = ExperimentExecutor(
            runner_factory=lambda ds, title=exp_title, ws=win_strat, tp=thresh_policy, t=init_t: AdaptiveSessionRunner(
                dataset=ds,
                estimator_factory=lambda s, thresh=t: HocquetStatisticalAlg(threshold=thresh),
                exp_name=title,
                results_repo=repo,
                window_strategy=ws,
                threshold_policy=tp,
                minimum_threshold=0.05,
                seeds_range=[0],
            )
        )
        executor.execute()


if __name__ == "__main__":
    main()
