"""Causal behavioural feature engineering.

Every feature for transaction *i* of customer *c* uses only transaction *i*
itself and transactions of *c* that happened **strictly before** *i*. This is
exactly the information a real-time scoring system has at authorisation time,
so there is no look-ahead leakage even though features are computed on the
full stream before the chronological split.

Feature groups
--------------
* raw / contextual : log amount, hour, night flag, weekend flag, channel,
                     log geolocation distance, merchant category (one-hot)
* velocity         : #txns and total spend in the previous 1 h and 24 h,
                     hours since the previous transaction
* deviation        : amount vs. customer's historical mean (ratio, z-score),
                     distance vs. historical mean, first-time category,
                     share of history in this category, historical night /
                     online share, history length
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_generation import CAT_NAMES

# Key multiplier: customer_id * KEY + t_seconds gives a globally sortable key
# whose order is (customer, time). 90 days < 1e8 seconds.
_KEY = 1e8
COLD_START_LOG_AMOUNT = float(np.log1p(50.0))
COLD_START_LOG_STD = 1.0


def _prior_window(key: np.ndarray, values: np.ndarray, window_s: float):
    """Count and sum of ``values`` in the window [t - w, t) for each row.

    ``key`` must be sorted ascending (customer-major, time-minor). Because
    customers are separated by ``_KEY`` seconds, a window never crosses into
    another customer's history.
    """
    idx = np.arange(len(key))
    start = np.searchsorted(key, key - window_s, side="left")
    csum = np.concatenate([[0.0], np.cumsum(values)])
    count = idx - start                       # rows strictly before i in window
    total = csum[idx] - csum[start]
    return count, total


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature matrix (one row per transaction, same index as ``df``).

    The returned frame also carries ``is_fraud``, ``t_seconds`` and
    ``fraud_scenario`` for splitting/analysis; they are *not* model inputs
    (see :data:`feature_columns`).
    """
    d = df.sort_values(["customer_id", "t_seconds"], kind="mergesort").copy()
    key = d["customer_id"].to_numpy() * _KEY + d["t_seconds"].to_numpy()
    amt = d["amount"].to_numpy()
    log_amt = np.log1p(amt)
    log_dist = np.log1p(d["distance_km"].to_numpy())
    hour = d["timestamp"].dt.hour.to_numpy() + d["timestamp"].dt.minute.to_numpy() / 60
    is_night = (hour < 6).astype(float)

    g = d.groupby("customer_id", sort=False)
    n_prior = g.cumcount().to_numpy()                       # history length
    safe_n = np.maximum(n_prior, 1)

    def prior_mean(x):
        """Expanding mean of x over the customer's *previous* rows."""
        s = pd.Series(x, index=d.index).groupby(d["customer_id"]).cumsum().to_numpy() - x
        return s / safe_n

    f = pd.DataFrame(index=d.index)
    # ---- raw / contextual --------------------------------------------------
    f["log_amount"] = log_amt
    f["hour"] = hour
    f["is_night"] = is_night
    f["is_weekend"] = (d["timestamp"].dt.dayofweek >= 5).astype(float).to_numpy()
    f["is_online"] = d["is_online"].to_numpy().astype(float)
    f["log_distance"] = log_dist

    # ---- velocity ----------------------------------------------------------
    for name, w in [("1h", 3_600), ("24h", 86_400)]:
        cnt, tot = _prior_window(key, amt, w)
        f[f"txn_count_{name}"] = cnt
        f[f"log_amount_sum_{name}"] = np.log1p(tot)
    gap_h = g["t_seconds"].diff().to_numpy() / 3_600
    f["log_hours_since_last"] = np.log1p(np.nan_to_num(gap_h, nan=24 * 30))

    # ---- deviation from the customer's own history -------------------------
    mean_log_amt = prior_mean(log_amt)
    mean_sq = prior_mean(log_amt ** 2)
    std_log_amt = np.sqrt(np.maximum(mean_sq - mean_log_amt ** 2, 0))
    has_hist = n_prior >= 3                                  # need some history
    # Cold-start prior for customers with < 3 past transactions. These are
    # fixed constants (≈ log of a $50 ticket, 1 log-unit spread), NOT statistics
    # of the dataset — a data-derived global mean would leak future rows.
    glob_mean, glob_std = COLD_START_LOG_AMOUNT, COLD_START_LOG_STD
    mu = np.where(has_hist, mean_log_amt, glob_mean)
    sd = np.where(has_hist, np.maximum(std_log_amt, 0.25), glob_std)
    f["amount_zscore"] = (log_amt - mu) / sd
    f["log_amount_ratio"] = log_amt - mu                     # log(amount / typical)
    mean_log_dist = prior_mean(log_dist)
    f["distance_dev"] = np.where(n_prior >= 1, log_dist - mean_log_dist, 0.0)

    cat_prior = d.groupby(["customer_id", "merchant_category"], sort=False).cumcount().to_numpy()
    f["is_new_category"] = ((cat_prior == 0) & (n_prior >= 1)).astype(float)
    f["category_share_prior"] = np.where(n_prior >= 1, cat_prior / safe_n, 0.0)
    f["night_share_prior"] = np.where(n_prior >= 1, prior_mean(is_night), 0.0)
    f["online_share_prior"] = np.where(n_prior >= 1, prior_mean(f["is_online"].to_numpy()), 0.0)
    f["log_n_prior"] = np.log1p(n_prior)

    # ---- merchant category one-hot ----------------------------------------
    for c in CAT_NAMES:
        f[f"cat_{c}"] = (d["merchant_category"].to_numpy() == c).astype(float)

    # ---- bookkeeping columns (not model inputs) ---------------------------
    f["t_seconds"] = d["t_seconds"].to_numpy()
    f["amount"] = amt
    f["fraud_scenario"] = d["fraud_scenario"].to_numpy()
    f["is_fraud"] = d["is_fraud"].to_numpy()
    return f.loc[df.index]                                   # restore input order


NON_FEATURES = {"t_seconds", "amount", "fraud_scenario", "is_fraud"}


def feature_columns(f: pd.DataFrame) -> list[str]:
    """Names of the model-input columns of a frame built by build_features."""
    return [c for c in f.columns if c not in NON_FEATURES]
