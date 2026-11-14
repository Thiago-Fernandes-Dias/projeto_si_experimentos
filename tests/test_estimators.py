import numpy as np
import pandas as pd
import pytest

from lib.estimators.hocquet_alg import HocquetStatisticalAlg
from lib.estimators.self_detector_alg import SelfDetectorAlg, rank_transform_vector
from lib.estimators.scaled_manhattan_alg import ScaledManhattanAlg
from lib.estimators.knn_instance_alg import KNNInstanceAlg
from lib.estimators.m2005_adaptive_alg import AdaptiveM2005Alg
from lib.estimators.m2005_alg import M2005Alg, M2005
from lib.utils import GENUINE_LABEL, IMPOSTOR_LABEL


def create_sample_df(n_samples: int = 10, n_features: int = 5, loc: float = 0.2, scale: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = rng.normal(loc=loc, scale=scale, size=(n_samples, n_features))
    cols = [f"H.f{i}" for i in range(n_features)]
    return pd.DataFrame(data, columns=cols)


def test_hocquet_statistical_alg():
    df = create_sample_df(n_samples=10)
    model = HocquetStatisticalAlg(threshold=0.5)
    model.fit(df)

    # Identical mean vector should be predicted as genuine
    mean_vec = df.mean()
    assert model.predict_vec(mean_vec) == GENUINE_LABEL

    # Extreme vector should be predicted as impostor
    far_vec = mean_vec + 10.0 * df.std()
    assert model.predict_vec(far_vec) == IMPOSTOR_LABEL

    # Test adaptation and threshold decrement
    dec = model.calculate_variable_threshold_decrement()
    assert dec > 0.0

    model.adapt(mean_vec, window="sliding", max_size=10)
    assert len(model._samples) == 10


def test_self_detector_alg():
    df = create_sample_df(n_samples=10)
    model = SelfDetectorAlg(radius=0.05, strategy="sliding")
    model.fit(df)

    # Training sample should match genuine
    assert model.predict_vec(df.iloc[0]) == GENUINE_LABEL

    # Adaptation
    adapted = model.adapt(df.iloc[0])
    assert adapted is True
    assert len(model._detectors) == 10


def test_scaled_manhattan_alg():
    df = create_sample_df(n_samples=10)
    model = ScaledManhattanAlg(threshold=2.0, sliding_window_size=10)
    model.fit(df)

    # Center sample should have distance close to 0
    assert model.predict_vec(df.mean()) == GENUINE_LABEL

    # Far sample
    far_vec = df.mean() + 15.0 * df.std()
    assert model.predict_vec(far_vec) == IMPOSTOR_LABEL

    model.adapt(df.mean())
    assert len(model._samples) == 10

    static_model = ScaledManhattanAlg(threshold=2.0, sliding_window_size=None)
    static_model.fit(df)
    static_model.adapt(df.mean())
    assert len(static_model._samples) == 10


def test_knn_instance_alg():
    df = create_sample_df(n_samples=10)
    model = KNNInstanceAlg(k=1, threshold=1.0, sliding_window_size=10)
    model.fit(df)

    # Existing sample should have distance 0 to itself
    assert model.predict_vec(df.iloc[0]) == GENUINE_LABEL

    model.adapt(df.iloc[0])
    assert len(model._samples) == 10

    static_model = KNNInstanceAlg(k=1, threshold=1.0, sliding_window_size=None)
    static_model.fit(df)
    static_model.adapt(df.iloc[0])
    assert len(static_model._samples) == 10


def test_m2005_alg():
    df = create_sample_df(n_samples=10)
    model = M2005Alg(threshold=0.7, inclusion_threshold=0.85, window_strategy="sliding")
    model.fit(df)
    mean_vec = df.mean()
    assert model.predict_vec(mean_vec) == GENUINE_LABEL
    assert model.should_include_sample(mean_vec) is True
    assert M2005 is M2005Alg

    calibrated_t = model.calibrate_threshold(method="min_non_outlier")
    assert 0.0 < calibrated_t <= 1.0


def test_adaptive_m2005_alg():
    df = create_sample_df(n_samples=15)
    model = AdaptiveM2005Alg(threshold=0.7, variant="idb", window_size=10)
    model.fit(df)

    # Verify Welford online sample std matches pandas sample std (ddof=1)
    for col in df.columns:
        expected_std = float(df[col].std(ddof=1))
        actual_std = model._growing_estimator._digraphs_metrics[col].std_dev
        assert np.isclose(actual_std, expected_std, atol=1e-9)

    mean_vec = df.mean()
    assert model.predict_vec(mean_vec) == GENUINE_LABEL

    model.adapt(mean_vec)
    assert len(model._sliding_gallery) == 10
