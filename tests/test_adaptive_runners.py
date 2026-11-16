import json
import os
import pandas as pd
import pytest

from lib.datasets.dataset import Dataset
from lib.estimators.hocquet_alg import HocquetStatisticalAlg
from lib.estimators.self_detector_alg import SelfDetectorAlg
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.adaptive_session_runner import AdaptiveSessionRunner
from lib.runners.adaptive_stream_runner import AdaptiveStreamRunner
from lib.runners.experiment_without_hpo_runner import ExperimentWithoutHPORunner
from lib.utils import GENUINE_LABEL, IMPOSTOR_LABEL


class SyntheticTestDataset(Dataset):
    """Mock dataset with 3 subjects, 3 sessions, 5 repetitions per session."""

    def __init__(self, file_path: str):
        super().__init__(file_path, test_train_split=self._split)

    def _split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = df[df["session"] == 1]
        test = df[df["session"] > 1]
        return train, test

    def get_drop_columns(self) -> list[str]:
        return ["session", "rep", "subject"]

    def user_key_name(self) -> str:
        return "subject"

    def get_session_key_name(self) -> str:
        return "session"

    def get_repetition_key_name(self) -> str:
        return "rep"


@pytest.fixture
def mock_dataset_csv(tmp_path):
    csv_file = tmp_path / "mock_dataset.csv"
    rows = []
    for s_idx in range(1, 4):
        for rep in range(1, 6):
            for sub in ["s01", "s02", "s03"]:
                base = 0.2 if sub == "s01" else (0.4 if sub == "s02" else 0.6)
                rows.append({
                    "subject": sub,
                    "session": s_idx,
                    "rep": rep,
                    "H.a": base + 0.01 * rep,
                    "H.b": base + 0.02 * rep,
                    "DD.a.b": base + 0.03 * rep,
                })
    pd.DataFrame(rows).to_csv(csv_file, index=False)
    return str(csv_file)


def test_adaptive_stream_runner(mock_dataset_csv):
    repo = results_repository_factory()
    dataset = SyntheticTestDataset(mock_dataset_csv)

    runner = AdaptiveStreamRunner(
        dataset=dataset,
        estimator_factory=lambda s: SelfDetectorAlg(radius=0.1, inclusion_radius=0.05, strategy="sliding"),
        exp_name="Test_SelfDet_Stream",
        results_repo=repo,
        impostor_ratio=0.3,
        seeds_range=[0],
    )
    runner.exec()

    results = repo.read_results("Test_SelfDet_Stream")
    assert results is not None
    assert len(results.model_predictions_per_seed) > 0

    pred_df = results.model_predictions_per_seed[0]
    for user_id in dataset.user_keys():
        user_genuine_preds = pred_df[(pred_df["user_id"] == user_id) & (pred_df["expected"] == GENUINE_LABEL)]
        chronological_keys = list(zip(user_genuine_preds["session"], user_genuine_preds["repetition"]))
        assert chronological_keys == sorted(chronological_keys)


def test_adaptive_session_runner(mock_dataset_csv):
    repo = results_repository_factory()
    dataset = SyntheticTestDataset(mock_dataset_csv)

    runner = AdaptiveSessionRunner(
        dataset=dataset,
        estimator_factory=lambda s: HocquetStatisticalAlg(threshold=0.5, inclusion_threshold=0.35),
        exp_name="Test_Hocquet_Session",
        results_repo=repo,
        window_strategy="sliding",
        threshold_policy="variable",
        minimum_threshold=0.05,
        seeds_range=[0],
    )
    runner.exec()

    results = repo.read_results("Test_Hocquet_Session")
    assert results is not None
    assert len(results.model_predictions_per_seed) > 0


def test_standardized_results_format_across_all_runners(mock_dataset_csv):
    repo = results_repository_factory()
    dataset = SyntheticTestDataset(mock_dataset_csv)
    expected_columns = ["user_id", "expected", "predicted", "session", "repetition", "seed"]

    runners = [
        (
            "Test_Std_NoHPO",
            ExperimentWithoutHPORunner(
                dataset=dataset,
                estimator_factory=lambda s: HocquetStatisticalAlg(threshold=0.5),
                exp_name="Test_Std_NoHPO",
                use_impostor_samples=False,
                results_repo=repo,
                seeds_range=[0],
            ),
        ),
        (
            "Test_Std_Stream",
            AdaptiveStreamRunner(
                dataset=dataset,
                estimator_factory=lambda s: SelfDetectorAlg(radius=0.1, inclusion_radius=0.05),
                exp_name="Test_Std_Stream",
                results_repo=repo,
                impostor_ratio=0.5,
                seeds_range=[0],
            ),
        ),
        (
            "Test_Std_Session",
            AdaptiveSessionRunner(
                dataset=dataset,
                estimator_factory=lambda s: HocquetStatisticalAlg(threshold=0.5, inclusion_threshold=0.35),
                exp_name="Test_Std_Session",
                results_repo=repo,
                window_strategy="sliding",
                threshold_policy="fixed",
                seeds_range=[0],
            ),
        ),
    ]

    reference_dtypes = None
    for exp_name, runner in runners:
        runner.exec()
        results = repo.read_results(exp_name)
        assert results is not None, f"Results missing for {exp_name}"
        assert len(results.model_predictions_per_seed) == 1

        pred_df = results.model_predictions_per_seed[0]
        assert list(pred_df.columns) == expected_columns, f"Columns mismatch in {exp_name}: {list(pred_df.columns)}"

        if reference_dtypes is None:
            reference_dtypes = pred_df.dtypes.to_dict()
        else:
            assert pred_df.dtypes.to_dict() == reference_dtypes, (
                f"Dtype mismatch in {exp_name}: {pred_df.dtypes.to_dict()} != {reference_dtypes}"
            )

        assert set(pred_df["expected"].unique()).issubset({GENUINE_LABEL, IMPOSTOR_LABEL})
        assert set(pred_df["predicted"].unique()).issubset({GENUINE_LABEL, IMPOSTOR_LABEL})

        metrics_per_user = results.get_metrics_per_user()
        assert set(metrics_per_user.keys()) == set(dataset.user_keys())
        for user_metrics in metrics_per_user.values():
            assert 0.0 <= user_metrics.far <= 1.0
            assert 0.0 <= user_metrics.frr <= 1.0

        exp_runs = sorted(os.listdir(os.path.join("results", exp_name)), reverse=True)
        log_file = os.path.join("results", exp_name, exp_runs[0], "execution.log")
        assert os.path.isfile(log_file), f"Missing execution.log for {exp_name}"
        with open(log_file, "r", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        events = [r.get("event") for r in records]
        for required_event in (
            "experiment_started",
            "seed_started",
            "user_completed",
            "seed_completed",
            "experiment_completed",
        ):
            assert required_event in events, f"Missing {required_event} in {exp_name} execution.log"
        user_records = [r for r in records if r.get("event") == "user_completed"]
        assert len(user_records) == len(dataset.user_keys())
        assert all("elapsed_seconds" in r for r in user_records)
