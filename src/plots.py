"""All figures. Every function writes one PNG (dpi 160) and closes the figure.

Visual conventions: one fixed colour per imbalance strategy (and per model)
across all figures, thin 2 px lines, recessive grid, text in neutral ink,
and log-scaled false-positive-rate axes where the interesting region of the
ROC curve (FPR << 1%) would otherwise be invisible.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (average_precision_score, precision_recall_curve,  # noqa: E402
                             roc_curve)

from .imbalance import STRATEGIES, STRATEGY_LABELS  # noqa: E402
from .models import MODEL_LABELS  # noqa: E402

# Fixed categorical order (validated colour-blind-safe for adjacent pairs).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
STRAT_COLOR = dict(zip(STRATEGIES, PALETTE))
CLASS_COLOR = {0: PALETTE[0], 1: PALETTE[1]}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
MARKERS = {"none": "o", "class_weight": "s", "random_under": "v",
           "random_over": "^", "smote": "D"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "lines.linewidth": 2, "font.size": 10,
    "axes.titlesize": 11, "legend.frameon": False, "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def plot_data_overview(raw, path):
    """Class balance, amount, hour-of-day and fraud-scenario distributions."""
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    counts = raw.is_fraud.value_counts().sort_index()
    a = ax[0, 0]
    a.bar(["Legitimate", "Fraud"], counts.values, color=[CLASS_COLOR[0], CLASS_COLOR[1]], width=0.55)
    a.set_yscale("log")
    a.set_title("(a) Class counts (log scale)")
    for i, v in enumerate(counts.values):
        a.text(i, v * 1.15, f"{v:,}\n({v / counts.sum():.2%})", ha="center", va="bottom", color=INK)
    a.set_ylim(top=counts.max() * 12)

    a = ax[0, 1]
    bins = np.logspace(np.log10(0.5), np.log10(raw.amount.max()), 60)
    for k, lab in [(0, "Legitimate"), (1, "Fraud")]:
        v = raw.amount[raw.is_fraud == k]
        a.hist(v, bins=bins, weights=np.full(len(v), 1 / len(v)), histtype="step",
               linewidth=2, color=CLASS_COLOR[k], label=lab)
    a.set_xscale("log")
    a.set_xlabel("Amount (USD, log scale)")
    a.set_ylabel("Share of class per bin")
    a.set_title("(b) Transaction amount")
    a.legend()

    a = ax[1, 0]
    hrs = raw.timestamp.dt.hour
    for k, lab in [(0, "Legitimate"), (1, "Fraud")]:
        h = np.bincount(hrs[raw.is_fraud == k], minlength=24)
        a.plot(np.arange(24), h / h.sum(), color=CLASS_COLOR[k], marker="o", markersize=4, label=lab)
    a.set_xlabel("Hour of day")
    a.set_ylabel("Share of class")
    a.set_xticks(range(0, 24, 3))
    a.set_title("(c) Hour of day")
    a.legend()

    a = ax[1, 1]
    sc = raw[raw.is_fraud == 1].fraud_scenario.value_counts()
    a.barh(sc.index[::-1], sc.values[::-1], color=CLASS_COLOR[1], height=0.55)
    for i, v in enumerate(sc.values[::-1]):
        a.text(v + sc.max() * 0.01, i, f"{v:,}", va="center", color=INK)
    a.set_xlabel("Fraudulent transactions")
    a.set_title("(d) Fraud scenarios")
    a.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, path)


def plot_pr_curves(y, scores, models, prevalence, path):
    """Precision-recall curves, one panel per model, one line per strategy."""
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.2), sharey=True)
    for a, m in zip(axes, models):
        for s in STRATEGIES:
            p, r, _ = precision_recall_curve(y, scores[(m, s)])
            ap = average_precision_score(y, scores[(m, s)])
            a.plot(r, p, color=STRAT_COLOR[s], label=f"{STRATEGY_LABELS[s]} (AP={ap:.3f})")
        a.axhline(prevalence, color=INK2, linestyle="--", linewidth=1)
        a.text(0.02, prevalence + 0.015, f"chance = prevalence ({prevalence:.3f})", color=INK2, fontsize=8)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1.02)
        a.set_xlabel("Recall")
        a.set_title(MODEL_LABELS[m])
        a.legend(loc="lower left", fontsize=8, bbox_to_anchor=(0.0, 0.06))
    axes[0].set_ylabel("Precision")
    fig.tight_layout()
    _save(fig, path)


def plot_roc_curves(y, scores, models, path):
    """ROC curves with a log-scaled FPR axis (operational region is FPR < 1%)."""
    from sklearn.metrics import roc_auc_score
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.2), sharey=True)
    for a, m in zip(axes, models):
        for s in STRATEGIES:
            fpr, tpr, _ = roc_curve(y, scores[(m, s)])
            auc = roc_auc_score(y, scores[(m, s)])
            a.plot(np.maximum(fpr, 1e-5), tpr, color=STRAT_COLOR[s],
                   label=f"{STRATEGY_LABELS[s]} (AUC={auc:.3f})")
        grid = np.logspace(-5, 0, 100)
        a.plot(grid, grid, color=INK2, linestyle="--", linewidth=1)
        a.axvline(0.01, color=INK2, linestyle=":", linewidth=1)
        a.text(0.0105, 0.03, "FPR = 1%", color=INK2, fontsize=8)
        a.set_xscale("log")
        a.set_xlim(1e-4, 1)
        a.set_ylim(0, 1.02)
        a.set_xlabel("False-positive rate (log scale)")
        a.set_title(MODEL_LABELS[m])
        a.legend(loc="lower left", fontsize=8, bbox_to_anchor=(0.0, 0.07))
    axes[0].set_ylabel("True-positive rate (recall)")
    fig.tight_layout()
    _save(fig, path)


def plot_confusion_matrices(metrics, model, path):
    """2 x 5 grid: rows = threshold (0.5 vs. tuned), cols = strategy."""
    strategies = list(STRATEGIES)
    fig, axes = plt.subplots(2, len(strategies), figsize=(2.7 * len(strategies), 6.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#2a78d6", "#0d366b"])
    for j, s in enumerate(strategies):
        r = metrics.query("model == @model and strategy == @s").iloc[0]
        for i, tag in enumerate(["default", "tuned"]):
            cm = np.array([[r[f"tn_{tag}"], r[f"fp_{tag}"]],
                           [r[f"fn_{tag}"], r[f"tp_{tag}"]]]).astype(int)
            norm = cm / cm.sum(axis=1, keepdims=True)          # row-normalised colour
            a = axes[i, j]
            a.imshow(norm, cmap=cmap, vmin=0, vmax=1)
            for (u, v), val in np.ndenumerate(cm):
                a.text(v, u, f"{val:,}\n{norm[u, v]:.1%}", ha="center", va="center",
                       color="white" if norm[u, v] > 0.55 else INK, fontsize=9)
            a.set_xticks([0, 1], ["Legit", "Fraud"])
            a.set_yticks([0, 1], ["Legit", "Fraud"])
            a.grid(False)
            for sp in a.spines.values():
                sp.set_visible(False)
            thr = 0.5 if tag == "default" else r["threshold_tuned"]
            a.set_title(f"{STRATEGY_LABELS[s]}\nthr={thr:.3f}  F1={r[f'f1_{tag}']:.3f}", fontsize=9)
            if j == 0:
                a.set_ylabel(("Threshold 0.5" if tag == "default" else "Tuned threshold")
                             + "\nTrue class")
            if i == 1:
                a.set_xlabel("Predicted class")
    fig.suptitle(f"Confusion matrices on the test period — {MODEL_LABELS[model]} "
                 "(cell: count and row %)", y=1.0)
    fig.tight_layout(h_pad=2.5)
    _save(fig, path)


def _grouped_bars(a, metrics, models, col, lo=None, hi=None):
    strategies = list(STRATEGIES)
    w = 0.8 / len(strategies)
    x = np.arange(len(models))
    for k, s in enumerate(strategies):
        sub = metrics[metrics.strategy == s].set_index("model").loc[models]
        pos = x - 0.4 + w * (k + 0.5)
        yerr = None
        if lo is not None:
            yerr = np.vstack([sub[col] - sub[lo], sub[hi] - sub[col]])
        a.bar(pos, sub[col], width=w * 0.9, color=STRAT_COLOR[s], label=STRATEGY_LABELS[s],
              yerr=yerr, error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
    a.set_xticks(x, [MODEL_LABELS[m] for m in models])
    a.grid(axis="x", visible=False)


def plot_resampling_effect(metrics, models, path):
    """Effect of each imbalance strategy on PR-AUC and on F1 at both thresholds."""
    m = metrics[metrics.model.isin(models)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    _grouped_bars(axes[0], m, models, "pr_auc", "pr_auc_lo", "pr_auc_hi")
    axes[0].set_title("(a) PR-AUC (threshold-free), 95% bootstrap CI")
    axes[0].set_ylabel("PR-AUC")
    _grouped_bars(axes[1], m, models, "f1_default")
    axes[1].set_title("(b) F1 at default threshold 0.5")
    axes[1].set_ylabel("F1")
    _grouped_bars(axes[2], m, models, "f1_tuned", "f1_tuned_lo", "f1_tuned_hi")
    axes[2].set_title("(c) F1 at validation-tuned threshold, 95% CI")
    axes[2].set_ylabel("F1")
    for a in axes:
        a.set_ylim(0, 1)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    _save(fig, path)


def plot_operating_points(metrics, models, path):
    """Precision vs. recall at threshold 0.5 (hollow) and tuned threshold (filled)."""
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.2), sharey=True)
    for a, mod in zip(axes, models):
        for s in STRATEGIES:
            r = metrics.query("model == @mod and strategy == @s").iloc[0]
            c = STRAT_COLOR[s]
            a.annotate("", xy=(r.recall_tuned, r.precision_tuned),
                       xytext=(r.recall_default, r.precision_default),
                       arrowprops={"arrowstyle": "->", "color": c, "lw": 1.2})
            a.scatter(r.recall_default, r.precision_default, s=60, facecolors="white",
                      edgecolors=c, linewidths=2, marker=MARKERS[s], zorder=3)
            a.scatter(r.recall_tuned, r.precision_tuned, s=60, color=c, marker=MARKERS[s],
                      edgecolors="white", linewidths=1, zorder=4, label=STRATEGY_LABELS[s])
        a.set_xlim(0, 1)
        a.set_ylim(0, 1.02)
        a.set_xlabel("Recall")
        a.set_title(MODEL_LABELS[mod])
    axes[0].set_ylabel("Precision")
    axes[0].legend(loc="lower left", fontsize=8, title="hollow: thr=0.5 → filled: tuned",
                   title_fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_ratio_ablation(ab, path):
    """PR-AUC, F1@tuned and precision/recall@0.5 as the resampling ratio varies."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    for i, m in enumerate(["logreg", "hist_gb"]):
        for s in ["random_under", "random_over", "smote"]:
            sub = ab[(ab.model == m) & (ab.strategy == s)].sort_values("ratio")
            c = STRAT_COLOR[s]
            kw = dict(color=c, marker=MARKERS[s], markersize=6)
            axes[i, 0].plot(sub.ratio, sub.pr_auc, label=STRATEGY_LABELS[s], **kw)
            axes[i, 1].plot(sub.ratio, sub.f1_tuned, **kw)
            axes[i, 2].plot(sub.ratio, sub.precision_default, linestyle="-", **kw)
            axes[i, 2].plot(sub.ratio, sub.recall_default, linestyle="--", **kw)
        axes[i, 0].set_ylabel(f"{MODEL_LABELS[m]}")
        for a in axes[i]:
            a.set_xscale("log")
            a.set_ylim(0, 1)
    axes[0, 0].set_title("PR-AUC")
    axes[0, 1].set_title("F1 at tuned threshold")
    axes[0, 2].set_title("At threshold 0.5: precision (—) / recall (- -)")
    for a in axes[1]:
        a.set_xlabel("Minority : majority ratio after resampling (log)")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    _save(fig, path)


def plot_calibration(y, scores, model, path):
    """Reliability diagram on log-spaced probability bins.

    Quantile bins are useless at 1% prevalence (almost every bin is ~0), so
    bins are log-spaced over [1e-4, 1] and bins with < 20 samples are dropped.
    Resampling / re-weighting shifts curves to the right (over-confidence).
    """
    from sklearn.metrics import brier_score_loss
    edges = np.concatenate([[0], np.logspace(-4, 0, 13)])
    fig, a = plt.subplots(figsize=(5.6, 5.2))
    a.plot([1e-4, 1], [1e-4, 1], color=INK2, linestyle="--", linewidth=1, label="Perfect calibration")
    for s in STRATEGIES:
        sc = np.clip(scores[(model, s)], 0, 1)
        b = np.clip(np.digitize(sc, edges[1:-1]), 0, len(edges) - 2)
        xs, ys = [], []
        for k in range(len(edges) - 1):
            m = b == k
            if m.sum() >= 20:
                xs.append(sc[m].mean())
                ys.append(max(y[m].mean(), 1e-4))     # floor so empty bins show on log axis
        a.plot(xs, ys, color=STRAT_COLOR[s], marker=MARKERS[s], markersize=5,
               label=f"{STRATEGY_LABELS[s]} (Brier={brier_score_loss(y, sc):.4f})")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(1e-4, 1.2)
    a.set_ylim(1e-4, 1.2)
    a.set_xlabel("Mean predicted fraud probability (log)")
    a.set_ylabel("Observed fraud frequency (log)")
    a.set_title(f"Calibration on test — {MODEL_LABELS[model]}")
    a.legend(fontsize=8, loc="upper left")
    _save(fig, path)


def plot_importance(imp, path, top=15):
    """Permutation importance (drop in PR-AUC when a feature is shuffled)."""
    d = imp.head(top).iloc[::-1]
    fig, a = plt.subplots(figsize=(6.5, 5.2))
    a.barh(d.feature, d.importance_mean, xerr=d.importance_std, color=PALETTE[0], height=0.6,
           error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
    a.set_xlabel("Mean drop in test PR-AUC when shuffled")
    a.set_title("Permutation feature importance (best model)")
    a.grid(axis="y", visible=False)
    _save(fig, path)


def plot_scenario_recall(scen, model, path):
    """Recall per fraud scenario for each strategy (best model, tuned threshold)."""
    sub = scen[scen.model == model]
    names = sorted(sub.scenario.unique())
    strategies = list(STRATEGIES)
    w = 0.8 / len(strategies)
    x = np.arange(len(names))
    fig, a = plt.subplots(figsize=(8.5, 4.2))
    for k, s in enumerate(strategies):
        v = sub[sub.strategy == s].set_index("scenario").loc[names]
        a.bar(x - 0.4 + w * (k + 0.5), v.recall_tuned, width=w * 0.9, color=STRAT_COLOR[s],
              label=STRATEGY_LABELS[s])
    n = sub.drop_duplicates("scenario").set_index("scenario").loc[names].n_test
    a.set_xticks(x, [f"{nm}\n(n={c})" for nm, c in zip(names, n)])
    a.set_ylim(0, 1.05)
    a.set_ylabel("Recall at tuned threshold")
    a.set_title(f"Recall by fraud scenario (test) — {MODEL_LABELS[model]}")
    a.grid(axis="x", visible=False)
    a.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _save(fig, path)


def plot_seed_robustness(rob, models, path):
    """ΔPR-AUC and ΔF1@tuned vs. no handling, one dot per independent dataset."""
    strategies = [s for s in STRATEGIES if s != "none"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=False)
    for a, col, title in [(axes[0], "delta_pr_auc", "(a) ΔPR-AUC vs. no handling"),
                          (axes[1], "delta_f1_tuned", "(b) ΔF1 at tuned threshold vs. no handling")]:
        w = 0.8 / len(strategies)
        for k, s in enumerate(strategies):
            for i, m in enumerate(models):
                d = rob[(rob.model == m) & (rob.strategy == s)][col]
                x = i - 0.4 + w * (k + 0.5)
                jitter = np.linspace(-w * 0.25, w * 0.25, len(d))
                a.scatter(x + jitter, d, s=14, color=STRAT_COLOR[s], alpha=0.8,
                          marker=MARKERS[s], edgecolors="none")
                a.hlines(d.mean(), x - w * 0.4, x + w * 0.4, color=STRAT_COLOR[s], linewidth=2.5,
                         label=STRATEGY_LABELS[s] if i == 0 else None)
        a.axhline(0, color=INK2, linewidth=1)
        a.set_xticks(range(len(models)), [MODEL_LABELS[m] for m in models])
        a.grid(axis="x", visible=False)
        a.set_title(title)
    axes[0].set_ylabel("Difference (dots: datasets, bar: mean)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    _save(fig, path)
