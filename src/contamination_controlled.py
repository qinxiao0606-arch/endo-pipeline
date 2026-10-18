"""
A1 — Random vs controlled (phenotype-dependent) contamination on Goldstein data.

Julia's point 1: "Compare random vs controlled/phenotype-dependent contamination.
Controlled = relabel as negative only cases that already look similar to
negatives / milder / atypical."

Three arms:

  random      cases moved at random                      -> baseline / null model
  mild        cases with the lowest symptom burden        -> group-survival pattern
                                                             is a genuine empirical
                                                             result (severity is a
                                                             sum over ALL groups and
                                                             favours none of them)
  atypical    cases with organ involvement but little     -> direction is partly
              menstrual pain                                 constructed; use it to
                                                             quantify magnitude, and
                                                             say so in the paper

Why the scores are built the way they are
-----------------------------------------
Severity is NOT the raw tick count. In this dataset the tick count alone reaches
AUC 0.905, so selecting on it would make the contamination rule nearly collinear
with the model itself, which comes close to the circularity we have ruled out
(using a model's own output to define its contamination). Instead severity is the
mean of the 15 group means, so every theme contributes equally and the 10-variable
systemic group cannot dominate by size alone.

This mitigation FAILED and the script says so: balanced severity correlates with
the model's prediction at r = +0.82, exactly as strongly as the raw tick count.
In this dataset there is no version of "symptom burden" that is not essentially
the tick count. The 'mild' arm is therefore near-circular and must be reported as
such. The 'atypical' arm is not (r = -0.13), which is why it carries the weight.

Two groups are excluded from both scores:
  P4  unlocalised pain descriptors  — pure intensity, would make severity circular
  X1  requires prior investigation  — encodes healthcare access, not symptom burden

The collinearity that remains is measured, not assumed: the script reports the
correlation between each selection score and the uncontaminated model's predicted
probability. Read that number before trusting the arm.

Selection is soft, not a hard cut-off: the probability of being relabelled rises
with rank on the selection score. This gives error bars and is closer to reality
than a sharp threshold.

Outputs (to --out):
  contamination_controlled_raw.csv       one row per (arm, level, repeat)
  contamination_controlled_summary.csv   means per (arm, level)
  contamination_controlled_importance.csv      per-feature importance, for the A2 analysis
  contamination_controlled_scores.csv                  per-respondent severity / atypicality scores

    cd src && python3 contamination_controlled.py --repeats 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

SEED = 42
DROP = ["Unnamed: 0", "row", "label"]
LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
ARMS = ["random", "mild", "atypical"]

# Group definitions live in groups.py and nowhere else. Importing them rather
# than restating them here is what stops the two copies from drifting apart.
from groups import (GOLDSTEIN as G, EXCLUDE_FROM_SCORES, AX_TYP, AX_ORG,
                    check_goldstein_coverage)


def hr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def load(path):
    df = pd.read_excel(path)
    sym = [c for c in df.columns if c not in DROP]
    return df[sym].astype(int), df["label"].astype(int).to_numpy(), sym


def check_groups(sym):
    n, k = check_goldstein_coverage(sym)
    print(f"group check ok: {n} variables in {k} groups (definitions from groups.py)")


def pct_rank(a):
    """Percentile rank in [0, 1]; ties get their average rank."""
    s = pd.Series(a)
    return (s.rank(method="average") - 1) / (len(s) - 1)


def build_scores(X):
    """Severity and atypicality, both as percentile ranks in [0, 1]."""
    gm = pd.DataFrame({k: X[v].mean(axis=1) for k, v in G.items()})
    used = [k for k in G if k not in EXCLUDE_FROM_SCORES]

    severity = gm[used].mean(axis=1)                       # every theme equal weight
    typ = pct_rank(gm[AX_TYP].mean(axis=1).to_numpy())
    org = pct_rank(gm[AX_ORG].mean(axis=1).to_numpy())
    atypicality = org - typ                                # high = organ, little period pain

    return (pd.Series(pct_rank(severity.to_numpy()), index=X.index),
            pd.Series(pct_rank(atypicality.to_numpy()), index=X.index),
            gm)


def select(y_true, frac, arm, severity, atypicality, rng, sharpness=3.0):
    """Choose which cases to relabel as negative. Soft rank-weighted sampling."""
    y = y_true.copy()
    idx = np.flatnonzero(y_true == 1)
    n = int(round(frac * len(idx)))
    if n == 0:
        return y, np.array([], dtype=int)

    if arm == "random":
        w = np.ones(len(idx))
    elif arm == "mild":                       # low severity -> high weight
        w = (1.0 - severity.to_numpy()[idx]) ** sharpness
    elif arm == "atypical":                   # high atypicality -> high weight
        w = (atypicality.to_numpy()[idx] + 1.0) ** sharpness   # shift to [0, 2]
    else:
        raise ValueError(arm)

    w = w + 1e-9
    picked = rng.choice(idx, n, replace=False, p=w / w.sum())
    y[picked] = 0
    return y, picked


def sens_spec(pred, ref):
    tp = int(((pred == 1) & (ref == 1)).sum())
    tn = int(((pred == 0) & (ref == 0)).sum())
    fp = int(((pred == 1) & (ref == 0)).sum())
    fn = int(((pred == 0) & (ref == 1)).sum())
    return (tp / (tp + fn) if tp + fn else np.nan,
            tn / (tn + fp) if tn + fp else np.nan)


def cv_proba(X, y, folds, seed):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(AdaBoostClassifier(random_state=SEED), X, y,
                             cv=cv, method="predict_proba")[:, 1]


def evaluate(X, y_obs, y_true, rate, folds, seed, vuln_mild, vuln_atyp, picked,
             cutoff="observed"):
    p = cv_proba(X, y_obs, folds, seed)

    se50, sp50 = sens_spec((p >= 0.50).astype(int), y_true)
    # 2026-10-17 fix (PROGRESS P-10, same bug Julia found on 09-16 in the overall
    # numbers): the cut-off flags the top `rate` fraction. `rate` used to be the CLEAN
    # prevalence, which no analyst can know once labels are contaminated; it is now
    # the OBSERVED prevalence of y_obs, recomputed per draw. cutoff="clean" keeps the
    # old behaviour so the pre-fix numbers can be reproduced.
    if cutoff == "observed":
        rate = float(np.mean(y_obs))
    k = int(round(rate * len(p)))
    thr = np.sort(p)[::-1][min(k, len(p) - 1)] if k else 1.0
    pred_r = (p >= thr).astype(int)
    seR, spR = sens_spec(pred_r, y_true)

    # the metric that matters: sensitivity among the true cases the model is
    # most likely to abandon -- the mildest and the most atypical thirds
    def sens_sub(mask):
        m = mask & (y_true == 1)
        return float(pred_r[m].mean()) if m.sum() else np.nan

    # A case that was relabelled is trained as a negative, so of course the model
    # scores it low -- measuring sensitivity on it is partly tautological. The
    # honest quantity is the SPILLOVER: sensitivity among vulnerable true cases
    # that were NOT relabelled in this draw. Nothing about them was touched.
    keep = np.ones(len(y_true), dtype=bool)
    keep[picked] = False

    return {"observed_AUC": roc_auc_score(y_obs, p),
            "true_AUC": roc_auc_score(y_true, p),
            "sens_0.50": se50, "spec_0.50": sp50,
            "sens_recal": seR, "spec_recal": spR,
            "sens_mildest_third": sens_sub(vuln_mild),
            "sens_most_atypical_third": sens_sub(vuln_atyp),
            "sens_mildest_unflipped": sens_sub(vuln_mild & keep),
            "sens_atypical_unflipped": sens_sub(vuln_atyp & keep)}


def importances(X, y):
    m = AdaBoostClassifier(random_state=SEED).fit(X, y)
    return pd.Series(m.feature_importances_, index=X.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/external/Dataset.xlsx")
    ap.add_argument("--out", default="../out")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--sharpness", type=float, default=3.0,
                    help="how strongly selection favours the target phenotype")
    ap.add_argument("--cutoff", choices=["observed", "clean"], default="observed",
                    help="prevalence used for the cut-off: observed labels (fixed "
                         "2026-10-17) or the clean prevalence (pre-fix behaviour)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X, y_true, sym = load(args.input)
    check_groups(sym)
    rate = float(y_true.mean())

    severity, atypicality, gm = build_scores(X)
    vuln_mild = (severity <= severity.quantile(1 / 3)).to_numpy()
    vuln_atyp = (atypicality >= atypicality.quantile(2 / 3)).to_numpy()

    hr("1. SELECTION SCORES — collinearity check")
    p0 = cv_proba(X, y_true, args.folds, SEED)
    tick = X.sum(axis=1)
    print("Read these before trusting any arm. A selection score that correlates")
    print("strongly with the uncontaminated model's own prediction makes the")
    print("contamination rule nearly circular.\n")
    print(f"  tick count           vs model p : r = {np.corrcoef(tick, p0)[0, 1]:+.3f}"
          "   <- why severity is NOT the tick count")
    print(f"  severity (balanced)  vs model p : r = {np.corrcoef(severity, p0)[0, 1]:+.3f}")
    print(f"  atypicality          vs model p : r = {np.corrcoef(atypicality, p0)[0, 1]:+.3f}")
    print(f"  severity vs atypicality         : r = {np.corrcoef(severity, atypicality)[0, 1]:+.3f}")
    print(f"\n  baseline AUC (no contamination) : {roc_auc_score(y_true, p0):.3f}")

    pd.DataFrame({"label": y_true, "severity": severity.to_numpy(),
                  "atypicality": atypicality.to_numpy(),
                  "tick_count": tick.to_numpy()}).to_csv(
        out / "contamination_controlled_scores.csv", index=False)

    base_imp = importances(X, pd.Series(y_true, index=X.index))

    hr("2. SWEEP")
    print(f"{len(ARMS)} arms x {len(LEVELS)} levels x {args.repeats} repeats, "
          f"AdaBoost, {args.folds}-fold CV\n")

    recs, imps = [], []
    for arm in ARMS:
        for frac in LEVELS:
            for r in range(args.repeats if frac > 0 else 1):
                rng = np.random.default_rng(SEED * 977 + r)
                y_obs, picked = select(y_true, frac, arm, severity, atypicality,
                                       rng, args.sharpness)
                if y_obs.sum() < 20:
                    continue
                ys = pd.Series(y_obs, index=X.index)
                m = evaluate(X, ys, y_true, rate, args.folds, SEED + r,
                             vuln_mild, vuln_atyp, picked, cutoff=args.cutoff)
                imp = importances(X, ys)
                recs.append({"arm": arm, "contamination": frac, "repeat": r,
                             "observed_cases": int(y_obs.sum()),
                             "flipped_mean_severity": float(severity.to_numpy()[picked].mean())
                             if len(picked) else np.nan,
                             "flipped_mean_atypicality": float(atypicality.to_numpy()[picked].mean())
                             if len(picked) else np.nan,
                             **m})
                imps.append({"arm": arm, "contamination": frac, "repeat": r,
                             **imp.to_dict()})
        print(f"  arm '{arm}' done")

    d = pd.DataFrame(recs)
    d.to_csv(out / "contamination_controlled_raw.csv", index=False)
    pd.DataFrame(imps).to_csv(out / "contamination_controlled_importance.csv", index=False)
    pd.DataFrame({"baseline_importance": base_imp}).to_csv(out / "contamination_controlled_baseline_importance.csv")

    metrics = ["observed_AUC", "true_AUC", "sens_0.50", "sens_recal",
               "sens_mildest_third", "sens_most_atypical_third",
               "sens_mildest_unflipped", "sens_atypical_unflipped",
               "flipped_mean_severity", "flipped_mean_atypicality"]
    s = d.groupby(["arm", "contamination"])[metrics].mean().reset_index()
    s.to_csv(out / "contamination_controlled_summary.csv", index=False)

    hr("3. RESULTS")
    print("flipped_mean_* confirms the arms actually select what they claim to")
    print("(severity/atypicality are percentile ranks; random should sit near 0.5).\n")
    for arm in ARMS:
        sub = s[s.arm == arm]
        print(f"--- {arm} ---")
        show = sub.drop(columns="arm").copy()
        show["contamination"] = (show["contamination"] * 100).astype(int).astype(str) + "%"
        print(show.round(3).to_string(index=False))
        print()

    hr("4. READING")
    for arm in ARMS:
        b = s[(s.arm == arm) & (s.contamination == 0.0)].iloc[0]
        w = s[(s.arm == arm) & (s.contamination == LEVELS[-1])].iloc[0]
        print(f"{arm:<9} 0% -> {LEVELS[-1]:.0%}   "
              f"obsAUC {b.observed_AUC:.3f}->{w.observed_AUC:.3f} ({w.observed_AUC - b.observed_AUC:+.3f}) | "
              f"trueAUC {b.true_AUC:.3f}->{w.true_AUC:.3f} ({w.true_AUC - b.true_AUC:+.3f}) | "
              f"sens@recal {b.sens_recal:.3f}->{w.sens_recal:.3f} ({w.sens_recal - b.sens_recal:+.3f})")
    print()
    for arm in ARMS:
        b = s[(s.arm == arm) & (s.contamination == 0.0)].iloc[0]
        w = s[(s.arm == arm) & (s.contamination == LEVELS[-1])].iloc[0]
        print(f"{arm:<9} vulnerable thirds, ALL true cases (partly tautological): "
              f"mildest {b.sens_mildest_third:.3f}->{w.sens_mildest_third:.3f} "
              f"({w.sens_mildest_third - b.sens_mildest_third:+.3f}) | "
              f"atypical {b.sens_most_atypical_third:.3f}->{w.sens_most_atypical_third:.3f} "
              f"({w.sens_most_atypical_third - b.sens_most_atypical_third:+.3f})")
    print()
    for arm in ARMS:
        b = s[(s.arm == arm) & (s.contamination == 0.0)].iloc[0]
        w = s[(s.arm == arm) & (s.contamination == LEVELS[-1])].iloc[0]
        print(f"{arm:<9} SPILLOVER — never-relabelled cases only (the honest number): "
              f"mildest {b.sens_mildest_unflipped:.3f}->{w.sens_mildest_unflipped:.3f} "
              f"({w.sens_mildest_unflipped - b.sens_mildest_unflipped:+.3f}) | "
              f"atypical {b.sens_atypical_unflipped:.3f}->{w.sens_atypical_unflipped:.3f} "
              f"({w.sens_atypical_unflipped - b.sens_atypical_unflipped:+.3f})")

    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
