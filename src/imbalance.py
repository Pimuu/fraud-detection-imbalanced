"""Class-imbalance handling strategies.

Two families are compared:

* **Algorithm-level** — ``class_weight``: keep the data, re-weight the loss so
  that each class contributes equally (sklearn's ``"balanced"`` heuristic,
  w_k = n / (2 * n_k)).
* **Data-level** — resample the *training set only*:
  ``random_under`` (drop majority rows), ``random_over`` (duplicate minority
  rows) and ``smote`` (synthesise minority rows by interpolating between a
  minority point and one of its k nearest minority neighbours; Chawla et al.,
  2002).

``imbalanced-learn`` is used when installed. If it is missing, equivalent
NumPy / scikit-learn implementations below are used instead, so the pipeline
never hard-depends on it. Both paths take the same ``sampling_ratio`` =
desired n_minority / n_majority after resampling.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

try:  # optional dependency
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    HAVE_IMBLEARN = True
except ImportError:  # pragma: no cover - exercised only without imblearn
    HAVE_IMBLEARN = False

# name -> (uses class weights?, resampler name or None)
STRATEGIES = {
    "none":         (False, None),
    "class_weight": (True,  None),
    "random_under": (False, "random_under"),
    "random_over":  (False, "random_over"),
    "smote":        (False, "smote"),
}
STRATEGY_LABELS = {
    "none": "No handling",
    "class_weight": "Class weights",
    "random_under": "Random under-sampling",
    "random_over": "Random over-sampling",
    "smote": "SMOTE",
}


# --------------------------------------------------------------------------
# Manual fallbacks (used only when imbalanced-learn is unavailable)
# --------------------------------------------------------------------------
def _manual_random_under(X, y, ratio, rng):
    min_idx, maj_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_keep = min(len(maj_idx), int(round(len(min_idx) / ratio)))
    keep = np.concatenate([min_idx, rng.choice(maj_idx, n_keep, replace=False)])
    return X[keep], y[keep]


def _manual_random_over(X, y, ratio, rng):
    min_idx, maj_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_new = max(0, int(round(ratio * len(maj_idx))) - len(min_idx))
    extra = rng.choice(min_idx, n_new, replace=True)
    keep = np.concatenate([maj_idx, min_idx, extra])
    return X[keep], y[keep]


def _manual_smote(X, y, ratio, rng, k=5):
    """Plain SMOTE: x_new = x_i + u * (x_nn - x_i), u ~ U(0, 1)."""
    X_min = X[y == 1]
    n_maj = int((y == 0).sum())
    n_new = max(0, int(round(ratio * n_maj)) - len(X_min))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    neigh = nn.kneighbors(X_min, return_distance=False)[:, 1:]   # drop self
    base = rng.integers(0, len(X_min), n_new)
    partner = neigh[base, rng.integers(0, k, n_new)]
    gap = rng.random((n_new, 1))
    X_syn = X_min[base] + gap * (X_min[partner] - X_min[base])
    return np.vstack([X, X_syn]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def resample(X: np.ndarray, y: np.ndarray, method: str | None,
             ratio: float = 1.0, seed: int = 0, force_manual: bool = False):
    """Resample (X, y) with ``method``; returns (X_res, y_res).

    ``method=None`` returns the data untouched. ``force_manual=True`` uses the
    built-in implementations even if imbalanced-learn is installed (used by
    the unit tests to check both paths agree).
    """
    if method is None:
        return X, y
    if HAVE_IMBLEARN and not force_manual:
        sampler = {
            "random_under": RandomUnderSampler(sampling_strategy=ratio, random_state=seed),
            "random_over": RandomOverSampler(sampling_strategy=ratio, random_state=seed),
            "smote": SMOTE(sampling_strategy=ratio, k_neighbors=5, random_state=seed),
        }[method]
        return sampler.fit_resample(X, y)
    rng = np.random.default_rng(seed)
    fn = {"random_under": _manual_random_under,
          "random_over": _manual_random_over,
          "smote": _manual_smote}[method]
    return fn(X, y, ratio, rng)
