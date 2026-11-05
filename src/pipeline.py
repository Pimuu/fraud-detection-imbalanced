"""End-to-end experiment pipeline.

Run from the repository root:

    python -m src.pipeline            # full run (~ several minutes on 4 cores)
    python -m src.pipeline --quick    # small smoke run for CI / sanity checks

Steps
-----
1. Generate the synthetic transaction stream (``data/transactions.csv``).
2. Build causal behavioural features (``src/features.py``).
3. Chronological split: first 60% of time-ordered rows = train, next 20% =
   validation (threshold tuning only), last 20% = test (reported numbers).
4. Standardise features with statistics from the *training* period only.
5. For every (model, imbalance strategy) pair: resample the training set if
   required, fit, score validation and test, tune the threshold on
   validation, and compute metrics + bootstrap CIs on test.
6. Ablation: vary the post-resampling minority:majority ratio.
7. Extra analyses: recall per fraud scenario, permutation importance.
8. Write CSV/JSON/Markdown results and all figures to ``results/``.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from . import config, plots
from .data_generation import generate_transactions
from .evaluate import (best_f1_threshold, bootstrap_ci, ranking_metrics,
                       threshold_metrics)
from .features import build_features, feature_columns
from .imbalance import HAVE_IMBLEARN, STRATEGIES, resample
from .models import make_model

MODELS = ["logreg", "random_forest", "hist_gb"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------
def prepare_data(seed: int, n_customers: int):
    """Generate data, build features, split chronologically, scale."""
    raw = generate_transactions(seed=seed, n_customers=n_customers)
    feats = build_features(raw)
    cols = feature_columns(feats)

    # Chronological split on the time-ordered stream (raw is sorted by time).
    order = np.argsort(feats["t_seconds"].to_numpy(), kind="mergesort")
    n = len(order)
    n_tr, n_va = int(config.TRAIN_FRAC * n), int(config.VAL_FRAC * n)
    parts = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va], "test": order[n_tr + n_va:]}

    scaler = StandardScaler().fit(feats.iloc[parts["train"]][cols].to_numpy())
    split = {}
    for name, idx in parts.items():
        sub = feats.iloc[idx]
        split[name] = {
            "X": scaler.transform(sub[cols].to_numpy()),
            "y": sub["is_fraud"].to_numpy(),
            "amount": sub["amount"].to_numpy(),
            "scenario": sub["fraud_scenario"].to_numpy(),
            "t": sub["t_seconds"].to_numpy(),
        }
    return raw, feats, cols, split


def fit_and_score(model_name, strategy, split, seed, ratio=1.0):
    """Train one configuration; return fitted model + val/test scores."""
    use_cw, method = STRATEGIES[strategy]
    X_tr, y_tr = resample(split["train"]["X"], split["train"]["y"], method,
                          ratio=ratio, seed=seed)
    model = make_model(model_name, "balanced" if use_cw else None, seed)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_s = time.time() - t0
    s_val = model.predict_proba(split["val"]["X"])[:, 1]
    s_test = model.predict_proba(split["test"]["X"])[:, 1]
    return model, s_val, s_test, {"n_train_after": len(y_tr),
                                  "train_fraud_rate_after": float(y_tr.mean()),
                                  "fit_seconds": fit_s}


def evaluate_config(split, s_val, s_test, n_boot, seed, ref_scores=None):
    """All test metrics for one configuration at both operating points."""
    y_te, amt = split["test"]["y"], split["test"]["amount"]
    thr = best_f1_threshold(split["val"]["y"], s_val)
    row = ranking_metrics(y_te, s_test)
    for tag, t in [("default", 0.5), ("tuned", thr)]:
        for k, v in threshold_metrics(y_te, s_test, t, amt).items():
            row[f"{k}_{tag}"] = v
    if n_boot:
        row.update(bootstrap_ci(y_te, s_test, thr, n_boot, seed, ref_scores))
    return row


# --------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------
def run(seed: int = config.SEED, quick: bool = False) -> None:
    np.random.seed(seed)
    # Quick runs write to a separate (git-ignored) folder so they never
    # overwrite the committed full results.
    res = config.ROOT / "results_quick" if quick else config.RESULTS_DIR
    fig = res / "figures"
    data_dir = res if quick else config.DATA_DIR
    fig.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    n_customers = 600 if quick else config.N_CUSTOMERS
    n_boot = 100 if quick else config.N_BOOTSTRAP
    log(f"imbalanced-learn available: {HAVE_IMBLEARN}")

    # ---- 1-4: data ----------------------------------------------------------
    raw, feats, cols, split = prepare_data(seed, n_customers)
    raw.to_csv(data_dir / "transactions.csv", index=False)
    raw.sample(1_000, random_state=seed).sort_values("t_seconds").to_csv(
        data_dir / "sample_transactions.csv", index=False)
    data_summary = {
        "n_transactions": int(len(raw)),
        "n_customers": int(raw.customer_id.nunique()),
        "n_fraud": int(raw.is_fraud.sum()),
        "fraud_rate": float(raw.is_fraud.mean()),
        "n_features": len(cols),
        "fraud_by_scenario": raw[raw.is_fraud == 1].fraud_scenario.value_counts().to_dict(),
        "splits": {k: {"n": int(len(v["y"])), "n_fraud": int(v["y"].sum()),
                       "fraud_rate": float(v["y"].mean()),
                       "day_from": float(v["t"].min() / 86_400),
                       "day_to": float(v["t"].max() / 86_400)}
                   for k, v in split.items()},
        "median_amount": raw.groupby("is_fraud").amount.median().to_dict(),
        "night_share": raw.assign(n=raw.timestamp.dt.hour < 6).groupby("is_fraud").n.mean().to_dict(),
        "online_share": raw.groupby("is_fraud").is_online.mean().to_dict(),
        "median_distance_km": raw.groupby("is_fraud").distance_km.median().to_dict(),
    }
    log(f"data: {data_summary['n_transactions']:,} txns, fraud rate "
        f"{data_summary['fraud_rate']:.3%}, {len(cols)} features")
    plots.plot_data_overview(raw, fig / "data_overview.png")

    # ---- 5: main grid --------------------------------------------------------
    rows, scores, fitted = [], {}, {}
    y_te = split["test"]["y"]
    # Trivial reference: predict "legitimate" for everything.
    triv = threshold_metrics(y_te, np.zeros(len(y_te)), 0.5, split["test"]["amount"])
    rows.append({"model": "always_legit", "strategy": "none",
                 "roc_auc": 0.5, "pr_auc": float(y_te.mean()),
                 **{f"{k}_default": v for k, v in triv.items()}})
    for m in MODELS:
        for s in STRATEGIES:
            model, s_val, s_test, info = fit_and_score(m, s, split, seed)
            ref = scores.get((m, "none"))
            row = {"model": m, "strategy": s, **info,
                   **evaluate_config(split, s_val, s_test, n_boot, seed, ref)}
            rows.append(row)
            scores[(m, s)] = s_test
            fitted[(m, s)] = model
            log(f"{m:13s} {s:12s} PR-AUC={row['pr_auc']:.3f} ROC-AUC={row['roc_auc']:.3f} "
                f"F1@0.5={row['f1_default']:.3f} F1@tuned={row['f1_tuned']:.3f} "
                f"({info['fit_seconds']:.1f}s)")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(res / "metrics_all.csv", index=False)

    # Compact with/without-handling comparison table (the headline CSV).
    comp_cols = ["model", "strategy", "pr_auc", "pr_auc_lo", "pr_auc_hi",
                 "delta_pr_auc_lo", "delta_pr_auc_hi", "roc_auc", "recall_at_1pct_fpr",
                 "precision_default", "recall_default", "f1_default", "accuracy_default",
                 "threshold_tuned", "precision_tuned", "recall_tuned", "f1_tuned",
                 "mcc_tuned", "amount_recall_tuned", "brier"]
    comp = metrics[metrics.model != "always_legit"][comp_cols].copy()
    base = comp[comp.strategy == "none"].set_index("model")
    comp["delta_pr_auc_vs_none"] = comp.pr_auc - comp.model.map(base.pr_auc)
    comp["delta_f1_tuned_vs_none"] = comp.f1_tuned - comp.model.map(base.f1_tuned)
    comp["imbalance_handling"] = np.where(comp.strategy == "none", "without", "with")
    comp.round(4).to_csv(res / "metrics_with_vs_without_handling.csv", index=False)

    cms = metrics[metrics.model != "always_legit"][
        ["model", "strategy"] + [f"{k}_{t}" for t in ("default", "tuned")
                                 for k in ("tp", "fp", "fn", "tn")]]
    cms.to_csv(res / "confusion_matrices.csv", index=False)

    # ---- 6: sampling-ratio ablation ---------------------------------------
    ab_rows = []
    ratios = [0.05, 0.25, 1.0] if quick else config.SAMPLING_RATIOS
    for m in ["logreg", "hist_gb"]:
        for s in ["random_under", "random_over", "smote"]:
            for r in ratios:
                _, s_val, s_test, info = fit_and_score(m, s, split, seed, ratio=r)
                row = evaluate_config(split, s_val, s_test, 0, seed)
                ab_rows.append({"model": m, "strategy": s, "ratio": r, **info, **row})
        log(f"ablation done for {m}")
    ablation = pd.DataFrame(ab_rows)
    ablation.to_csv(res / "sampling_ratio_ablation.csv", index=False)

    # ---- 7: per-scenario recall + importance -------------------------------
    best = metrics[metrics.model != "always_legit"].sort_values("pr_auc").iloc[-1]
    best_key = (best.model, best.strategy)
    log(f"best PR-AUC configuration: {best_key}")
    scen_rows = []
    sc = split["test"]["scenario"]
    for (m, s), sco in scores.items():
        thr = metrics.query("model == @m and strategy == @s").threshold_tuned.iloc[0]
        pred = sco >= thr
        for name in sorted(set(sc[y_te == 1])):
            mask = sc == name
            scen_rows.append({"model": m, "strategy": s, "scenario": name,
                              "n_test": int(mask.sum()), "recall_tuned": float(pred[mask].mean())})
    scen = pd.DataFrame(scen_rows)
    scen.to_csv(res / "recall_by_scenario.csv", index=False)

    pi = permutation_importance(fitted[best_key], split["test"]["X"], y_te,
                                scoring="average_precision",
                                n_repeats=3 if quick else 5, random_state=seed, n_jobs=1)
    imp = pd.DataFrame({"feature": cols, "importance_mean": pi.importances_mean,
                        "importance_std": pi.importances_std}).sort_values(
                            "importance_mean", ascending=False)
    imp.to_csv(res / "permutation_importance.csv", index=False)

    # ---- 7b: seed robustness (independent synthetic datasets) --------------
    rob = run_seed_robustness([seed] + ([] if quick else config.ROBUSTNESS_SEEDS),
                              n_customers, main=metrics)
    rob.to_csv(res / "seed_robustness.csv", index=False)
    plots.plot_seed_robustness(rob, MODELS, fig / "seed_robustness.png")

    # ---- 8: figures + summary ---------------------------------------------
    prevalence = float(y_te.mean())
    plots.plot_pr_curves(y_te, scores, MODELS, prevalence, fig / "pr_curves.png")
    plots.plot_roc_curves(y_te, scores, MODELS, fig / "roc_curves.png")
    plots.plot_confusion_matrices(metrics, best.model, fig / "confusion_matrices.png")
    plots.plot_resampling_effect(metrics, MODELS, fig / "resampling_effect.png")
    plots.plot_operating_points(metrics, MODELS, fig / "operating_points.png")
    plots.plot_ratio_ablation(ablation, fig / "sampling_ratio_ablation.png")
    plots.plot_calibration(y_te, scores, best.model, fig / "calibration.png")
    plots.plot_importance(imp, fig / "feature_importance.png")
    plots.plot_scenario_recall(scen, best.model, fig / "recall_by_scenario.png")

    summary = {"seed": seed, "quick": quick, "imblearn": HAVE_IMBLEARN,
               "data": data_summary, "best_config": {"model": best.model, "strategy": best.strategy,
                                                     "pr_auc": float(best.pr_auc)}}
    with open(res / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    write_markdown_tables(res, metrics, ablation, scen, imp, data_summary, rob)
    log(f"done — see {res.relative_to(config.ROOT)}/")


def run_seed_robustness(seeds, n_customers, main):
    """Repeat the main grid on independently generated datasets.

    Each seed regenerates the data *and* re-seeds resampling / models, so the
    spread captures dataset-to-dataset variation, which the single-test-set
    bootstrap cannot. No bootstrap here (n_boot=0) to keep the cost down.
    The main run's seed is reused from ``main`` instead of being recomputed.
    """
    rows = []
    keep = ["pr_auc", "roc_auc", "f1_default", "f1_tuned", "precision_tuned",
            "recall_tuned", "brier"]
    for r in main[main.model != "always_legit"].itertuples():
        rows.append({"seed": seeds[0], "model": r.model, "strategy": r.strategy,
                     **{k: getattr(r, k) for k in keep}})
    for sd in seeds[1:]:
        _, _, _, split = prepare_data(sd, n_customers)
        for m in MODELS:
            for s in STRATEGIES:
                _, s_val, s_test, _ = fit_and_score(m, s, split, sd)
                row = evaluate_config(split, s_val, s_test, 0, sd)
                rows.append({"seed": sd, "model": m, "strategy": s,
                             **{k: row[k] for k in keep}})
        log(f"robustness seed {sd} done")
    rob = pd.DataFrame(rows)
    # Within-seed difference to the same model without handling (paired by seed).
    base = rob[rob.strategy == "none"].set_index(["seed", "model"])
    for k in ["pr_auc", "f1_tuned", "f1_default"]:
        rob[f"delta_{k}"] = rob[k].to_numpy() - base.loc[
            list(zip(rob.seed, rob.model)), k].to_numpy()
    return rob


def write_markdown_tables(res, metrics, ablation, scen, imp, data_summary, rob):
    """Human-readable results/summary.md (numbers referenced by the paper)."""
    m = metrics[metrics.model != "always_legit"].copy()
    out = ["# Results summary (auto-generated by `python -m src.pipeline`)\n"]
    out.append(f"Test period: {data_summary['splits']['test']['n']:,} transactions, "
               f"{data_summary['splits']['test']['n_fraud']} fraudulent "
               f"({data_summary['splits']['test']['fraud_rate']:.3%}).\n")
    out.append("## Main grid (test set)\n")
    out.append("| Model | Strategy | PR-AUC [95% CI] | ROC-AUC | Rec@1%FPR | P@0.5 | R@0.5 | F1@0.5 "
               "| thr* | P@thr* | R@thr* | F1@thr* | $-recall@thr* | Brier |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in m.itertuples():
        out.append(f"| {r.model} | {r.strategy} | {r.pr_auc:.3f} [{r.pr_auc_lo:.3f}, {r.pr_auc_hi:.3f}] "
                   f"| {r.roc_auc:.3f} | {r.recall_at_1pct_fpr:.3f} | {r.precision_default:.3f} "
                   f"| {r.recall_default:.3f} | {r.f1_default:.3f} | {r.threshold_tuned:.3f} "
                   f"| {r.precision_tuned:.3f} | {r.recall_tuned:.3f} | {r.f1_tuned:.3f} "
                   f"| {r.amount_recall_tuned:.3f} | {r.brier:.4f} |")
    t = metrics[metrics.model == "always_legit"].iloc[0]
    out.append(f"\nTrivial `always_legit` baseline: accuracy = {t.accuracy_default:.4f}, "
               f"recall = 0, PR-AUC (= prevalence) = {t.pr_auc:.4f}.\n")
    out.append("## Paired bootstrap ΔPR-AUC vs. no handling (same model)\n")
    out.append("| Model | Strategy | ΔPR-AUC | 95% CI |")
    out.append("|---|---|---|---|")
    base = m[m.strategy == "none"].set_index("model").pr_auc
    for r in m[m.strategy != "none"].itertuples():
        out.append(f"| {r.model} | {r.strategy} | {r.pr_auc - base[r.model]:+.4f} "
                   f"| [{r.delta_pr_auc_lo:+.4f}, {r.delta_pr_auc_hi:+.4f}] |")
    out.append("\n## Sampling-ratio ablation (test PR-AUC / F1@thr*)\n")
    piv = ablation.pivot_table(index=["model", "strategy"], columns="ratio",
                               values=["pr_auc", "f1_tuned"])
    out.append(piv.round(3).to_markdown() if _has_tabulate() else piv.round(3).to_string())
    out.append("\n## Recall by fraud scenario (tuned threshold)\n")
    sp = scen.pivot_table(index=["model", "strategy"], columns="scenario", values="recall_tuned")
    out.append(sp.round(3).to_markdown() if _has_tabulate() else sp.round(3).to_string())
    out.append("\n## Seed robustness: mean ± sd over independent datasets\n")
    g = rob.groupby(["model", "strategy"], sort=False)
    out.append(f"Seeds: {sorted(rob.seed.unique().tolist())}. ΔX = X(strategy) − X(none), "
               "paired within seed; 'wins' = #seeds with Δ > 0.\n")
    out.append("| Model | Strategy | PR-AUC | ΔPR-AUC | wins | F1@0.5 | F1@thr* | ΔF1@thr* | wins |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for (mm, ss), d in g:
        out.append(f"| {mm} | {ss} | {d.pr_auc.mean():.3f} ± {d.pr_auc.std():.3f} "
                   f"| {d.delta_pr_auc.mean():+.4f} ± {d.delta_pr_auc.std():.4f} "
                   f"| {(d.delta_pr_auc > 0).sum()}/{len(d)} "
                   f"| {d.f1_default.mean():.3f} ± {d.f1_default.std():.3f} "
                   f"| {d.f1_tuned.mean():.3f} ± {d.f1_tuned.std():.3f} "
                   f"| {d.delta_f1_tuned.mean():+.4f} ± {d.delta_f1_tuned.std():.4f} "
                   f"| {(d.delta_f1_tuned > 0).sum()}/{len(d)} |")
    out.append("\n## Permutation importance (best model, ΔPR-AUC when shuffled)\n")
    out.append("| Feature | Mean | Std |")
    out.append("|---|---|---|")
    for r in imp.head(15).itertuples():
        out.append(f"| {r.feature} | {r.importance_mean:.4f} | {r.importance_std:.4f} |")
    (res / "summary.md").write_text("\n".join(out) + "\n")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--quick", action="store_true", help="small, fast smoke run")
    args = ap.parse_args()
    run(seed=args.seed, quick=args.quick)
