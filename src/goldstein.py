"""
Replication of Goldstein & Cohen (2023) and a label-contamination experiment.

Two questions:

  1. Can the published AUC of 0.94 be reproduced? (sanity check)

  2. The authors write: "as the percentage of endometriosis is estimated
     between 5 and 10%, we expect such bias to be relatively small." That is a
     testable claim, and their own data can test it.

Protocol: move a known fraction x of the cases into the control group, retrain,
then evaluate TWO ways:

  - against the corrupted labels -> what a researcher would observe
  - against the true labels      -> the model's actual performance

The gap between the two is the quantity of interest.

Caveat: their controls already contain an unknown baseline contamination, so we
are not measuring an absolute level but a SLOPE — how fast performance degrades
per unit of ADDED contamination. Absolute levels require synthetic data, where
the truth is known by construction (see contamination.py).

    python goldstein.py --input ../data/external/Dataset.xlsx --repeats 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

PLUM, TEAL, AMBER = "#5B2149", "#0F7B72", "#B8860B"
SEED = 42
DROP = ["Unnamed: 0", "row", "label"]
LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]


def hr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def load(path):
    df = pd.read_excel(path)
    sym = [c for c in df.columns if c not in DROP]
    return df[sym].astype(int), df["label"].astype(int), sym


def models():
    return {
        "Decision tree": DecisionTreeClassifier(random_state=SEED),
        "Random forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        "Gradient boosting": GradientBoostingClassifier(random_state=SEED),
        "AdaBoost": AdaBoostClassifier(random_state=SEED),
    }


def cv_predict(X, y, mdl, n_splits, seed):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(mdl, X, y, cv=cv, method="predict_proba")[:, 1]


def sens_spec(pred, ref):
    tp = int(((pred == 1) & (ref == 1)).sum())
    tn = int(((pred == 0) & (ref == 0)).sum())
    fp = int(((pred == 1) & (ref == 0)).sum())
    fn = int(((pred == 0) & (ref == 1)).sum())
    return (tp / (tp + fn) if tp + fn else np.nan,
            tn / (tn + fp) if tn + fp else np.nan)


def contaminate(y_true: np.ndarray, frac: float, rng) -> np.ndarray:
    """Move a known fraction of cases into the control group."""
    y = y_true.copy()
    idx = np.flatnonzero(y_true == 1)
    n = int(round(frac * len(idx)))
    if n:
        y[rng.choice(idx, n, replace=False)] = 0
    return y


def top_features(X, y, mdl, k=10):
    mdl.fit(X, y)
    imp = getattr(mdl, "feature_importances_", None)
    if imp is None:
        return []
    return list(pd.Series(imp, index=X.columns).sort_values(ascending=False).head(k).index)


def band(ax, x, m, lo, hi, color, label, marker="o"):
    ax.plot(x, m, marker + "-", color=color, label=label, ms=5, lw=1.8)
    ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/external/Dataset.xlsx")
    ap.add_argument("--out", default="../out")
    ap.add_argument("--repeats", type=int, default=100,
                    help="random contamination draws per level")
    ap.add_argument("--folds", type=int, default=5,
                    help="CV folds inside the sweep (replication always uses 10)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X, y, sym = load(args.input)

    hr("1. DATA")
    print(f"file       : {args.input}")
    print(f"respondents: {len(X)}")
    print(f"symptoms   : {len(sym)}")
    print(f"diagnosed      : {int(y.sum())}")
    print(f"not diagnosed  : {int((1 - y).sum())}")
    print(f"missing values : {int(X.isna().sum().sum())}")
    print("\nPublished paper: 886 respondents, 474 diagnosed, 412 not diagnosed.")

    hr("2. REPLICATION (10-fold cross-validation, as in the paper)")
    ref = {"Decision tree": 0.875, "Random forest": 0.930,
           "Gradient boosting": 0.928, "AdaBoost": 0.937}
    rows = []
    for name, mdl in models().items():
        p = cv_predict(X, y, mdl, 10, SEED)
        auc = roc_auc_score(y, p)
        se, sp = sens_spec((p >= 0.5).astype(int), y.to_numpy())
        rows.append({"model": name, "AUC": auc, "sensitivity": se, "specificity": sp,
                     "AUC_paper": ref[name], "difference": auc - ref[name]})
    rep = pd.DataFrame(rows)
    print(rep.round(3).to_string(index=False))
    rep.to_csv(out / "goldstein_replication.csv", index=False)

    hr(f"3. CONTAMINATION EXPERIMENT ({args.repeats} draws per level)")
    print("A known fraction of cases is moved into the control group, the model is")
    print("retrained, and performance is measured against both label sets.\n")

    base_top = top_features(X, y, AdaBoostClassifier(random_state=SEED))
    yt = y.to_numpy()
    recs = []

    for frac in LEVELS:
        for r in range(args.repeats):
            rng = np.random.default_rng(SEED * 1000 + r)
            yc = contaminate(yt, frac, rng)
            if yc.sum() < 20:
                continue
            p = cv_predict(X, pd.Series(yc), AdaBoostClassifier(random_state=SEED),
                           args.folds, SEED + r)
            se_t, sp_t = sens_spec((p >= 0.5).astype(int), yt)
            tf = top_features(X, pd.Series(yc), AdaBoostClassifier(random_state=SEED))
            recs.append({
                "contamination": frac, "draw": r,
                "observed_AUC": roc_auc_score(yc, p),
                "true_AUC": roc_auc_score(yt, p),
                "true_sensitivity": se_t, "true_specificity": sp_t,
                "cases_remaining": int(yc.sum()),
                "top10_overlap": len(set(tf) & set(base_top)) / 10,
            })
        print(f"  {frac:.0%} done")

    d = pd.DataFrame(recs)
    d.to_csv(out / "goldstein_contamination_raw.csv", index=False)

    metrics = ["cases_remaining", "observed_AUC", "true_AUC", "true_sensitivity",
               "true_specificity", "top10_overlap"]
    g = d.groupby("contamination")[metrics]
    s, sd = g.mean().reset_index(), g.std().reset_index()
    q05, q95 = g.quantile(0.05).reset_index(), g.quantile(0.95).reset_index()

    hr("RESULTS (mean ± standard deviation across draws)")
    show = s.copy()
    for c in ["observed_AUC", "true_AUC", "true_sensitivity", "top10_overlap"]:
        show[c] = [f"{m:.3f} ± {v:.3f}" for m, v in zip(s[c], sd[c])]
    show["cases_remaining"] = s["cases_remaining"].round(0).astype(int)
    show["true_specificity"] = s["true_specificity"].round(3)
    print(show.to_string(index=False))
    s.to_csv(out / "goldstein_contamination.csv", index=False)
    sd.to_csv(out / "goldstein_contamination_sd.csv", index=False)

    b, w = s.iloc[0], s.iloc[-1]
    print(f"\nFrom 0% to {w['contamination']:.0%} added contamination:")
    print(f"  observed AUC     {b['observed_AUC']:.3f} -> {w['observed_AUC']:.3f}"
          f"  ({w['observed_AUC'] - b['observed_AUC']:+.3f})")
    print(f"  true AUC         {b['true_AUC']:.3f} -> {w['true_AUC']:.3f}"
          f"  ({w['true_AUC'] - b['true_AUC']:+.3f})")
    print(f"  true sensitivity {b['true_sensitivity']:.3f} -> {w['true_sensitivity']:.3f}"
          f"  ({w['true_sensitivity'] - b['true_sensitivity']:+.3f})")
    print(f"  top-10 symptoms retained: {w['top10_overlap']:.0%}")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    fig.suptitle("Known contamination of the control group — Goldstein & Cohen (2023) data"
                 f"   ·   {args.repeats} draws per level, band = 5th–95th percentile",
                 fontsize=12.5, color=PLUM, fontweight="bold")
    xs = s["contamination"] * 100

    ax = axes[0]
    band(ax, xs, s["observed_AUC"], q05["observed_AUC"], q95["observed_AUC"],
         PLUM, "Observed AUC\n(corrupted labels)")
    band(ax, xs, s["true_AUC"], q05["true_AUC"], q95["true_AUC"],
         TEAL, "True AUC\n(ground truth)", "s")
    ax.axhline(0.937, ls=":", color="grey", lw=1)
    ax.text(1, 0.939, "0.937 published", fontsize=8, color="grey")
    ax.set_xlabel("cases moved into controls (%)", fontsize=9)
    ax.set_ylabel("AUC", fontsize=9)
    ax.set_title("Discrimination", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    band(ax, xs, s["true_sensitivity"], q05["true_sensitivity"], q95["true_sensitivity"],
         PLUM, "Sensitivity")
    band(ax, xs, s["true_specificity"], q05["true_specificity"], q95["true_specificity"],
         TEAL, "Specificity", "s")
    ax.set_xlabel("cases moved into controls (%)", fontsize=9)
    ax.set_ylabel("value at threshold 0.50", fontsize=9)
    ax.set_title("Measured against the truth", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    band(ax, xs, s["top10_overlap"] * 100, q05["top10_overlap"] * 100,
         q95["top10_overlap"] * 100, AMBER, "top-10 retained")
    ax.set_ylim(0, 105)
    ax.set_xlabel("cases moved into controls (%)", fontsize=9)
    ax.set_ylabel("% retained", fontsize=9)
    ax.set_title("Stability of the top-10 symptoms", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "goldstein_contamination.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure: {out / 'goldstein_contamination.png'}")
    hr("DONE")


if __name__ == "__main__":
    main()
