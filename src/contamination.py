"""
Contamination experiment on synthetic data.

Companion to goldstein.py. The difference is what each can measure:

  goldstein.py     real data, UNKNOWN baseline contamination
                   -> measures a SLOPE (degradation per unit added)

  contamination.py synthetic data, ZERO baseline contamination by construction
                   -> measures an ABSOLUTE level

Both are needed. The slope shows the effect exists on real data; the absolute
level says how large it is.

A correction is also tested: recalibrating the decision threshold to an assumed
positive rate instead of leaving it at 0.50. This is the "how do we evaluate"
half of the question, and it is the only one of the three degradations that
recovers without changing the model.

Note on what recalibration needs: an estimate of how many positives the sample
holds, not which individuals they are. In a population sample that is close to
the published prevalence; in a case-control sample like ours it is not, and has
to be reasoned from the recruitment design.

Variability is separated into two sources:
  --datasets   how many synthetic datasets are generated (sampling variability)
  --draws      how many contamination draws per dataset (which cases get moved)
Total repeats per level = datasets x draws.

    python contamination.py --datasets 10 --draws 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from config import load_schema
from encode import encode
from synth import generate
from validate import run_all

PLUM, TEAL, AMBER = "#5B2149", "#0F7B72", "#B8860B"
SEED = 42
ROOT = Path(__file__).resolve().parents[1]
LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]


def hr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def rf(n_estimators=200):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(n_estimators=n_estimators,
                                                    min_samples_leaf=3,
                                                    class_weight="balanced",
                                                    random_state=SEED, n_jobs=-1))])


def contaminate(y_true, frac, rng):
    """Move a known fraction of cases into the control group."""
    y = y_true.copy()
    idx = np.flatnonzero(y_true == 1)
    n = int(round(frac * len(idx)))
    if n:
        y[rng.choice(idx, n, replace=False)] = 0
    return y


def sens_spec(pred, ref):
    tp = int(((pred == 1) & (ref == 1)).sum())
    tn = int(((pred == 0) & (ref == 0)).sum())
    fp = int(((pred == 1) & (ref == 0)).sum())
    fn = int(((pred == 0) & (ref == 1)).sum())
    return (tp / (tp + fn) if tp + fn else np.nan,
            tn / (tn + fp) if tn + fp else np.nan)


def evaluate(X, y_obs, y_true, positive_rate, folds, seed):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    p = cross_val_predict(rf(), X, y_obs, cv=cv, method="predict_proba")[:, 1]

    se50, sp50 = sens_spec((p >= 0.50).astype(int), y_true)

    # recalibrated: flag the top-k scores, where k matches the assumed positive rate
    k = int(round(positive_rate * len(p)))
    thr = np.sort(p)[::-1][min(k, len(p) - 1)] if k else 1.0
    seR, spR = sens_spec((p >= thr).astype(int), y_true)

    return {"observed_AUC": roc_auc_score(y_obs, p),
            "true_AUC": roc_auc_score(y_true, p),
            "sens_0.50": se50, "spec_0.50": sp50,
            "sens_recal": seR, "spec_recal": spR}


def top_features(X, y, k=10):
    m = rf().fit(X, y)
    imp = m.named_steps["clf"].feature_importances_
    return list(pd.Series(imp, index=X.columns).sort_values(ascending=False).head(k).index)


def band(ax, x, m, lo, hi, color, label, marker="o"):
    ax.plot(x, m, marker + "-", color=color, label=label, ms=5, lw=1.8)
    ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--datasets", type=int, default=10)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="../out")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    schema = load_schema()
    total = args.datasets * args.draws

    hr("CONTAMINATION EXPERIMENT — SYNTHETIC DATA")
    print("Here the truth is known by construction: the control group starts with")
    print("no cases at all. The levels below are therefore absolute, unlike the")
    print("Goldstein & Cohen experiment where the baseline contamination is unknown.")
    print(f"\n{args.datasets} datasets x {args.draws} draws = {total} repeats per level\n")

    recs = []
    for dset in range(args.datasets):
        df, _ = generate(n=args.n, seed=SEED + dset, schema=schema,
                         polish_path=ROOT / "data" / "polish_correlations.xlsx")
        df, _ = run_all(df, schema, strict=False)
        X, y = encode(df, schema)
        y_true = y.to_numpy()
        rate = float(y_true.mean())
        base_top = top_features(X, y)

        for frac in LEVELS:
            for dr in range(args.draws):
                rng = np.random.default_rng(SEED * 1000 + dset * 100 + dr)
                y_obs = contaminate(y_true, frac, rng)
                if y_obs.sum() < 20:
                    continue
                m = evaluate(X, pd.Series(y_obs, index=X.index), y_true, rate,
                             args.folds, SEED + dr)
                tf = top_features(X, pd.Series(y_obs, index=X.index))
                recs.append({"contamination": frac, "dataset": dset, "draw": dr,
                             "observed_cases": int(y_obs.sum()),
                             "top10_overlap": len(set(tf) & set(base_top)) / 10,
                             **m})
        print(f"  dataset {dset + 1}/{args.datasets} done "
              f"(n={len(X)}, true positive rate {rate:.1%})")

    d = pd.DataFrame(recs)
    d.to_csv(out / "contamination_synth_raw.csv", index=False)

    metrics = ["observed_cases", "observed_AUC", "true_AUC", "sens_0.50",
               "spec_0.50", "sens_recal", "spec_recal", "top10_overlap"]
    g = d.groupby("contamination")[metrics]
    s, sd = g.mean().reset_index(), g.std().reset_index()
    q05, q95 = g.quantile(0.05).reset_index(), g.quantile(0.95).reset_index()

    hr("RESULTS (mean ± standard deviation across repeats)")
    show = s.copy()
    for c in ["observed_AUC", "true_AUC", "sens_0.50", "sens_recal", "top10_overlap"]:
        show[c] = [f"{m:.3f} ± {v:.3f}" for m, v in zip(s[c], sd[c])]
    show["observed_cases"] = s["observed_cases"].round(0).astype(int)
    for c in ["spec_0.50", "spec_recal"]:
        show[c] = s[c].round(3)
    print(show.to_string(index=False))
    s.to_csv(out / "contamination_synth.csv", index=False)
    sd.to_csv(out / "contamination_synth_sd.csv", index=False)

    b, w = s.iloc[0], s.iloc[-1]
    hr("READING")
    print(f"From 0% to {w['contamination']:.0%} contamination (absolute levels):\n")
    print(f"  Observed AUC (what you think you see) : {b['observed_AUC']:.3f} -> {w['observed_AUC']:.3f}"
          f"   ({w['observed_AUC'] - b['observed_AUC']:+.3f})")
    print(f"  True AUC     (ground truth)           : {b['true_AUC']:.3f} -> {w['true_AUC']:.3f}"
          f"   ({w['true_AUC'] - b['true_AUC']:+.3f})")
    print(f"  Sensitivity at threshold 0.50         : {b['sens_0.50']:.3f} -> {w['sens_0.50']:.3f}"
          f"   ({w['sens_0.50'] - b['sens_0.50']:+.3f})")
    print(f"  Sensitivity at recalibrated threshold : {b['sens_recal']:.3f} -> {w['sens_recal']:.3f}"
          f"   ({w['sens_recal'] - b['sens_recal']:+.3f})")
    print(f"  Top-10 variables retained             : {b['top10_overlap']:.0%} -> {w['top10_overlap']:.0%}")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    fig.suptitle("Control-group contamination — synthetic data, ground truth known"
                 f"   ·   {total} repeats per level, band = 5th–95th percentile",
                 fontsize=12.5, color=PLUM, fontweight="bold")
    xs = s["contamination"] * 100

    ax = axes[0]
    band(ax, xs, s["observed_AUC"], q05["observed_AUC"], q95["observed_AUC"],
         PLUM, "Observed AUC\n(corrupted labels)")
    band(ax, xs, s["true_AUC"], q05["true_AUC"], q95["true_AUC"],
         TEAL, "True AUC\n(ground truth)", "s")
    ax.set_xlabel("cases moved into controls (%)", fontsize=9)
    ax.set_ylabel("AUC", fontsize=9)
    ax.set_title("Discrimination", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    band(ax, xs, s["sens_0.50"], q05["sens_0.50"], q95["sens_0.50"],
         PLUM, "fixed threshold 0.50")
    band(ax, xs, s["sens_recal"], q05["sens_recal"], q95["sens_recal"],
         TEAL, "recalibrated threshold", "s")
    ax.set_xlabel("cases moved into controls (%)", fontsize=9)
    ax.set_ylabel("sensitivity (against the truth)", fontsize=9)
    ax.set_title("Recalibration recovers sensitivity", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    band(ax, xs, s["top10_overlap"] * 100, q05["top10_overlap"] * 100,
         q95["top10_overlap"] * 100, AMBER, "top-10 retained")
    ax.set_ylim(0, 105)
    ax.set_xlabel("cases moved into controls (%)", fontsize=9)
    ax.set_ylabel("% retained", fontsize=9)
    ax.set_title("Stability of the top-10 variables", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "contamination_synth.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure: {out / 'contamination_synth.png'}")


if __name__ == "__main__":
    main()
