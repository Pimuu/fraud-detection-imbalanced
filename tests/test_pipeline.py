"""Unit tests: run with ``python -m pytest -q`` from the repository root."""
import numpy as np
import pandas as pd
import pytest

from src.data_generation import generate_transactions
from src.evaluate import best_f1_threshold, ranking_metrics, threshold_metrics
from src.features import build_features, feature_columns
from src.imbalance import HAVE_IMBLEARN, resample


@pytest.fixture(scope="module")
def small():
    return generate_transactions(seed=7, n_customers=300)


def test_fraud_rate_is_about_one_percent(small):
    assert abs(small.is_fraud.mean() - 0.01) < 0.001
    assert set(small.fraud_scenario[small.is_fraud == 1]) <= {
        "cnp_spree", "card_testing", "stolen_card", "mimic"}


def test_generation_is_deterministic():
    a = generate_transactions(seed=3, n_customers=100)
    b = generate_transactions(seed=3, n_customers=100)
    pd.testing.assert_frame_equal(a, b)


def test_features_are_causal(small):
    """Features of past rows must not change when future rows are removed."""
    full = build_features(small)
    cutoff = small.t_seconds.quantile(0.5)
    past = small[small.t_seconds <= cutoff]
    trunc = build_features(past)
    cols = feature_columns(full)
    np.testing.assert_allclose(full.loc[past.index, cols].to_numpy(),
                               trunc[cols].to_numpy(), rtol=1e-9, atol=1e-9)


def test_feature_matrix_is_finite(small):
    f = build_features(small)
    assert np.isfinite(f[feature_columns(f)].to_numpy()).all()


@pytest.mark.parametrize("method", ["random_under", "random_over", "smote"])
@pytest.mark.parametrize("ratio", [0.1, 1.0])
def test_manual_resampling_matches_target_ratio(method, ratio):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2_000, 4))
    y = (rng.random(2_000) < 0.02).astype(int)
    Xr, yr = resample(X, y, method, ratio=ratio, seed=0, force_manual=True)
    assert abs(yr.sum() / (yr == 0).sum() - ratio) < 0.02
    if HAVE_IMBLEARN:   # both implementations should give identical class counts
        Xi, yi = resample(X, y, method, ratio=ratio, seed=0)
        assert abs(int(yi.sum()) - int(yr.sum())) <= 1
        assert abs(int((yi == 0).sum()) - int((yr == 0).sum())) <= 1


def test_manual_smote_points_lie_inside_minority_hull():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1, (500, 2)), rng.uniform(5, 6, (20, 2))])
    y = np.r_[np.zeros(500, int), np.ones(20, int)]
    Xr, yr = resample(X, y, "smote", ratio=1.0, seed=0, force_manual=True)
    syn = Xr[len(X):]
    assert len(syn) == 480 and ((syn >= 5) & (syn <= 6)).all()


def test_metrics_on_perfect_and_trivial_scores():
    y = np.r_[np.zeros(990, int), np.ones(10, int)]
    amt = np.ones(1_000)
    perfect = y.astype(float)
    assert ranking_metrics(y, perfect)["pr_auc"] == pytest.approx(1.0)
    m = threshold_metrics(y, np.zeros(1_000), 0.5, amt)
    assert m["accuracy"] == pytest.approx(0.99) and m["recall"] == 0   # accuracy paradox
    assert 0 < best_f1_threshold(y, perfect) <= 1
