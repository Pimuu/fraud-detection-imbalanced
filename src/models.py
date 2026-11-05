"""Model zoo.

Three classifiers of increasing capacity are compared:

* ``logreg`` — L2 logistic regression: linear, calibrated-by-design baseline.
* ``random_forest`` — bagged deep trees; robust non-linear baseline.
* ``hist_gb`` — histogram gradient boosting (LightGBM-style, built into
  scikit-learn); usually the strongest model on tabular data.

All three accept ``class_weight="balanced"``, so the *same* model can be
trained with algorithm-level imbalance handling or without it.
Hyper-parameters are fixed (not tuned per strategy) so that differences
between imbalance strategies are not confounded with tuning effort.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODEL_LABELS = {
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    "hist_gb": "Hist. gradient boosting",
}


def make_model(name: str, class_weight: str | None, seed: int):
    """Instantiate an unfitted model; ``class_weight`` is None or 'balanced'."""
    if name == "logreg":
        return LogisticRegression(C=1.0, max_iter=2_000, class_weight=class_weight,
                                  random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                      max_features="sqrt", n_jobs=-1,
                                      class_weight=class_weight, random_state=seed)
    if name == "hist_gb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                              max_leaf_nodes=31, l2_regularization=1.0,
                                              early_stopping=False,
                                              class_weight=class_weight, random_state=seed)
    raise ValueError(f"unknown model {name!r}")
