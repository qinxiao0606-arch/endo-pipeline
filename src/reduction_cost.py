"""
What does it cost to cut Goldstein's 58 variables down to the 9 we share?

Julia, August (verbatim, PROGRESS §8):

  "if you are adjusting the model / retraining on the same data but using only
   that defined / shared feature subset you need to validate it on original data
   first, so we know how much performance is lost if any (also good exp+result on
   its own). then apply exactly that reduced model to our data"

Her instruction is three measurements in a fixed order. This script does the
first two; the third needs our own 191 responses and is blocked on the control
group problem (PROGRESS §2, B3). Tracked as A3 steps 1-3 in PROGRESS §1.

  1. train on all 58 Goldstein variables            -> baseline
  2. retrain on the 9 shared variables, SAME data   -> cost of the reduction
  3. apply that reduced model unchanged to our data -> dataset difference

Keeping 1 and 2 on the same data is the whole point: it separates "we threw away
49 columns" from "our respondents are different people". Skip the middle one and
the two causes are confounded and neither number means anything.

Why this is not just re-reading themes_vs_count.py
--------------------------------------------------
That script already gives the two point estimates (0.977 for all 58, 0.947 for
the 9 shared). But Julia wrote "how much performance is lost IF ANY", and a
single 5-fold CV run cannot answer "if any": it gives one number with no spread,
so there is no way to tell 0.030 from zero. Three things are added here.

1. REPEATED CV WITH PAIRED DIFFERENCES. The same fold assignment is handed to
   both feature sets in every repeat, so the difference is measured within a
   repeat and the fold-assignment noise cancels. Reporting the spread of that
   paired difference is what turns "0.030" into a claim.

2. A RANDOM 9-VARIABLE BASELINE. The 9 shared variables were chosen by wording
   match against our questionnaire, pre-registered in A0, NOT by importance. So
   the honest question is not "do 9 variables do worse than 58" (obviously they
   might) but "did OUR 9 do better than 9 arbitrary ones". §10 material P already
   found that selection stops doing work once k is large; here k=9 is small, so
   it should still be doing something. If it is not, step 2's number is not about
   our mapping at all, it is just about the number 9.

3. A PERSISTED MODEL. The third measurement applies "exactly that reduced
   model", so the second one has to leave behind the fitted object itself, not
   just a score. Saved with its threshold, because a probability with no cut-off
   cannot produce the sensitivity and specificity Julia asked to see.
   ⚠ The .pkl is a Python pickle and pickles break across scikit-learn versions.
   It is a convenience, never the record: everything in it that matters (feature
   list, thresholds, prevalence) is also written to reduced_model_info.csv in
   plain text, and the fit is deterministic given the seed, so anything that
   loads this file can equally well just rerun this script.

Protocol is copied from themes_vs_count.py on purpose - same folds, same seed,
same models, same prevalence-matched cut-off. Any deviation would make the new
numbers incomparable with A0-2, which is the one thing that must not happen.

    cd src && python3 reduction_cost.py
    cd src && python3 reduction_cost.py --repeats 20 --random-draws 50
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from groups import ALIGNMENT, SHARED_SUBSET, check_goldstein_coverage

warnings.filterwarnings("ignore")
SEED = 42
DROP = ["Unnamed: 0", "row", "label"]


def hr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def models():
    return [("ada", AdaBoostClassifier(random_state=SEED)),
            ("lr", LogisticRegression(max_iter=2000))]


def evaluate(M, y, folds, rate, seed):
    """AUC, sensitivity, specificity at a prevalence-matched cut-off.

    Identical to themes_vs_count.evaluate except the CV seed is a parameter, so
    the same seed can be handed to every feature set inside one repeat.
    """
    out = {}
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tag, mdl in models():
        p = cross_val_predict(mdl, M, y, cv=cv, method="predict_proba")[:, 1]
        k = max(1, int(round(rate * len(p))))
        thr = np.sort(p)[::-1][min(k, len(p) - 1)]
        pred = (p >= thr).astype(int)
        out[tag] = (roc_auc_score(y, p),
                    float(pred[y == 1].mean()),
                    float(1.0 - pred[y == 0].mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/external/Dataset.xlsx")
    ap.add_argument("--out", default="../out")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20,
                    help="CV repeats with different fold assignments")
    ap.add_argument("--random-draws", type=int, default=50,
                    help="random 9-variable subsets for the selection baseline")
    ap.add_argument("--random-repeats", type=int, default=3,
                    help="CV repeats per random subset")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(a.input)
    sym = [c for c in df.columns if c not in DROP]
    n_var, n_grp = check_goldstein_coverage(sym)
    X = df[sym].astype(float)
    y = df["label"].astype(int).to_numpy()
    rate = float(y.mean())

    missing = [v for v in SHARED_SUBSET if v not in X.columns]
    if missing:
        raise ValueError(f"shared subset not in data: {missing}")

    hr("cost of reducing 58 Goldstein variables to the 9 shared ones")
    print(f"  data        {a.input}   {X.shape[0]} rows x {X.shape[1]} variables")
    print(f"  labels      {int(y.sum())} cases / {int((y == 0).sum())} controls, "
          f"prevalence {rate:.3f}")
    print(f"  groups      {n_var} variables in {n_grp} groups (from groups.py)")
    print(f"  protocol    {a.folds}-fold stratified CV x {a.repeats} repeats, "
          f"prevalence-matched cut-off")
    print(f"  shared set  {len(SHARED_SUBSET)} variables, chosen by wording match "
          f"(A0, pre-registered)")

    sets = {"all58": X, "shared9": X[SHARED_SUBSET]}

    # ---- steps 1 and 2, paired across repeats ---------------------------
    rows = []
    for rep in range(a.repeats):
        seed = SEED + rep
        for name, M in sets.items():
            r = evaluate(M, y, a.folds, rate, seed)
            for tag in ("ada", "lr"):
                auc, sens, spec = r[tag]
                rows.append({"step": name, "k": M.shape[1], "model": tag,
                             "repeat": rep, "cv_seed": seed,
                             "AUC": auc, "sens": sens, "spec": spec})
        print(f"    repeat {rep + 1:>3}/{a.repeats}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "reduction_cost_repeats.csv", index=False)

    hr("1. all 58 vs shared 9  (mean over repeats, same folds within a repeat)")
    summ = (res.groupby(["step", "k", "model"])[["AUC", "sens", "spec"]]
            .agg(["mean", "std"]).round(4))
    print(summ.to_string())

    hr("2. THE ANSWER TO 'how much is lost, if any' — paired difference")
    print("   paired per repeat, so fold-assignment noise cancels\n")
    diff_rows = []
    for tag in ("ada", "lr"):
        w = res[res["model"] == tag].pivot_table(
            index="repeat", columns="step", values=["AUC", "sens", "spec"])
        for metric in ("AUC", "sens", "spec"):
            d = w[(metric, "all58")] - w[(metric, "shared9")]
            lo, hi = np.percentile(d, [2.5, 97.5])
            diff_rows.append({"model": tag, "metric": metric,
                              "mean_loss": d.mean(), "sd": d.std(ddof=1),
                              "pct2.5": lo, "pct97.5": hi,
                              "n_repeats_with_loss": int((d > 0).sum()),
                              "n_repeats": len(d)})
    diff = pd.DataFrame(diff_rows)
    diff.to_csv(out / "reduction_cost_loss.csv", index=False)
    print(diff.round(4).to_string(index=False))
    print("\n   mean_loss > 0 means the full 58 did better, i.e. reduction cost something.")
    print("   n_repeats_with_loss tells you whether it is consistent or noise.")

    # ---- the selection baseline -----------------------------------------
    hr(f"3. did OUR 9 do better than 9 arbitrary ones?  "
       f"({a.random_draws} random subsets)")
    print("   the 9 were chosen by wording match, not importance — so this is the\n"
          "   question that decides whether step 2's number is about our mapping\n"
          "   or just about the number 9\n")
    rng = np.random.default_rng(SEED)
    rnd = []
    for d in range(a.random_draws):
        cols = list(rng.choice(sym, size=len(SHARED_SUBSET), replace=False))
        aucs = [evaluate(X[cols], y, a.folds, rate, SEED + r)["ada"][0]
                for r in range(a.random_repeats)]
        rnd.append({"draw": d, "AUC": float(np.mean(aucs)),
                    "variables": " | ".join(cols)})
        if (d + 1) % 10 == 0:
            print(f"    draw {d + 1:>3}/{a.random_draws}", flush=True)
    rnd = pd.DataFrame(rnd)
    rnd.to_csv(out / "random9_baseline.csv", index=False)

    ours = res[(res["step"] == "shared9") & (res["model"] == "ada")
               & (res["repeat"] < a.random_repeats)]["AUC"].mean()
    beat = int((rnd["AUC"] >= ours).sum())
    print(f"\n   our 9 (wording match)     AUC = {ours:.4f}")
    print(f"   random 9, mean            AUC = {rnd['AUC'].mean():.4f}")
    print(f"   random 9, min / max       AUC = {rnd['AUC'].min():.4f} / "
          f"{rnd['AUC'].max():.4f}")
    print(f"   random 9, 95th pct        AUC = {np.percentile(rnd['AUC'], 95):.4f}")
    print(f"   random subsets matching or beating ours: "
          f"{beat}/{a.random_draws}  (empirical p = {(beat + 1) / (a.random_draws + 1):.3f})")

    # ---- persist the step-2 model for step 3 ----------------------------
    hr("4. fitting and saving the reduced model, for the step onto our data")
    cv = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=SEED)
    bundle = {}
    for tag, mdl in models():
        p = cross_val_predict(mdl, X[SHARED_SUBSET], y, cv=cv,
                              method="predict_proba")[:, 1]
        k = max(1, int(round(rate * len(p))))
        thr = float(np.sort(p)[::-1][min(k, len(p) - 1)])
        mdl.fit(X[SHARED_SUBSET], y)
        bundle[tag] = {"model": mdl, "threshold": thr}
        print(f"  {tag:<4} fitted on all {len(y)} rows, "
              f"CV-derived cut-off {thr:.4f}")
    info = pd.DataFrame([{"field": "feature", "value": v} for v in SHARED_SUBSET]
                        + [{"field": f"threshold_{k}", "value": v["threshold"]}
                           for k, v in bundle.items()]
                        + [{"field": "goldstein_prevalence", "value": rate},
                           {"field": "n_rows_fitted_on", "value": len(y)},
                           {"field": "seed", "value": SEED}])
    info.to_csv(out / "reduced_model_info.csv", index=False)
    print(f"  -> {out}/reduced_model_info.csv  (plain-text twin of the pickle)")

    bundle["features"] = list(SHARED_SUBSET)
    bundle["goldstein_prevalence"] = rate
    bundle["alignment"] = {t: dict(v) for t, v in ALIGNMENT.items()}
    with open(out / "reduced_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  -> {out}/reduced_model.pkl")
    print("  ⚠ the saved cut-off is matched to GOLDSTEIN prevalence "
          f"({rate:.3f}). Our sample")
    print("    has a different case mix, so step 3 must decide explicitly whether to")
    print("    carry this cut-off over or re-derive one. Carrying it over unchanged is")
    print("    the literal reading of 'apply exactly that reduced model' and is the")
    print("    default, but the choice has to be stated, not left implicit.")

    hr("the 9 shared variables (wording match, pre-registered in A0)")
    for tid, t in ALIGNMENT.items():
        print(f"  {tid}  {t['theme']:<26} {t['rep']}")

    hr("written")
    for f in ["reduction_cost_repeats.csv", "reduction_cost_loss.csv",
              "random9_baseline.csv", "reduced_model.pkl",
              "reduced_model_info.csv"]:
        q = out / f
        if q.exists():
            print(f"  {f:<32} {q.stat().st_size / 1024:>8.1f} KB")


if __name__ == "__main__":
    main()
