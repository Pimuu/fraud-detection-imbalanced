"""Imbalance-aware evaluation utilities.

Why not accuracy? With 1% fraud, the trivial "everything is legitimate"
classifier scores 99% accuracy while catching nothing. We therefore report:

* threshold-free ranking metrics — **PR-AUC** (average precision; its
  chance level equals the fraud prevalence) and ROC-AUC (chance = 0.5);
* threshold-dependent metrics — precision, recall, F1, MCC, balanced
  accuracy, the confusion matrix and the share of fraudulent *dollars*
  caught — at two operating points:
    - ``default``: probability >= 0.5 (what you get "out of the box");
    - ``tuned``: the threshold maximising F1 on the *validation* period,
      then frozen and applied to the test period;
* Brier score — probability calibration, which resampling distorts;
* bootstrap 95% confidence intervals for PR-AUC / ROC-AUC / F1, plus paired
  bootstrap CIs for the PR-AUC difference vs. the no-handling baseline.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold that maximises F1 on (y_true, scores)."""
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    # precision_recall_curve returns len(thr) + 1 points; the last has no threshold
    return float(thr[np.argmax(f1[:-1])])


def recall_at_fpr(y_true, scores, max_fpr=0.01) -> float:
    """Highest recall achievable while keeping the false-positive rate <= max_fpr."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(tpr[fpr <= max_fpr].max())


def threshold_metrics(y_true, scores, threshold, amounts) -> dict:
    """Metrics that depend on a decision threshold."""
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fraud = y_true == 1
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "amount_recall": float(amounts[fraud & (y_pred == 1)].sum() / amounts[fraud].sum()),
        "alerts": int(tp + fp),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def ranking_metrics(y_true, scores) -> dict:
    """Threshold-free metrics."""
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, scores, 0.01),
        "brier": brier_score_loss(y_true, scores),
    }


def bootstrap_ci(y_true, scores, threshold, n_boot, seed, ref_scores=None):
    """Percentile-bootstrap 95% CIs on the test set.

    The same ``seed`` yields the same resampled indices for every model, so
    passing ``ref_scores`` (the baseline model's scores) gives a *paired*
    CI for ΔPR-AUC = PR-AUC(model) - PR-AUC(ref).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    pr, roc, f1, dpr = [], [], [], []
    y_pred = (scores >= threshold).astype(int)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.sum() == 0:
            continue
        pr.append(average_precision_score(yt, scores[idx]))
        roc.append(roc_auc_score(yt, scores[idx]))
        f1.append(f1_score(yt, y_pred[idx], zero_division=0))
        if ref_scores is not None:
            dpr.append(pr[-1] - average_precision_score(yt, ref_scores[idx]))

    def ci(v):
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    out = {}
    out["pr_auc_lo"], out["pr_auc_hi"] = ci(pr)
    out["roc_auc_lo"], out["roc_auc_hi"] = ci(roc)
    out["f1_tuned_lo"], out["f1_tuned_hi"] = ci(f1)
    if ref_scores is not None:
        out["delta_pr_auc_lo"], out["delta_pr_auc_hi"] = ci(dpr)
    return out
