import logging

from sklearn.ensemble import RandomForestClassifier

from lib.datasets.keyrecs_dataset import KeyrecsDataset
from lib.hp_grids import rf_params_grid
from lib.logging_config import configure_logging
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_without_hpo_runner import ExperimentWithoutHPORunner
from lib.utils import KEYRECS_PATH, keyrecs_default_split


def main() -> None:
    configure_logging()

    keyrecs = KeyrecsDataset(KEYRECS_PATH, keyrecs_default_split)
    keyrecs_runner = ExperimentWithoutHPORunner(
        estimator_factory=lambda s: RandomForestClassifier(random_state=s),
        dataset=keyrecs,
        results_repo=results_repository_factory(),
        exp_name="Random Forest (Keyrecs)",
        use_impostor_samples=True,
        seeds_range=list(range(21, 30))
    )
    keyrecs_runner.exec()


if __name__ == "__main__":
    main()
