"""
Two things Julia asked for on 2026-09-16, in one script.

  "In the current implementation, the classification threshold seems to be defined
   by forcing approximately 53.5% of observations to be predicted positive, based
   on the original class prevalence. [...] that could mechanically produce part of
   the large sensitivity/specificity shift once cases are relabelled as controls.
   Could you please check this and rerun sensitivity/specificity?"

  "i think you are not calculating auc exactly like in the paper, Could you check
   the paper-style calculation as well? I want to make sure we are comparing
   apples to apples"

She is right on the first one. Confirmed in the source:

  contamination_controlled.py:203   rate = float(y_true.mean())
  contamination_split.py:109        rate = float(y_true.mean())

`rate` is computed once, from the CLEAN labels, before any contamination, and the
same value is then passed to every arm and every level. The cut-off is "flag the
top `rate` fraction by score", so every arm flags ~53.5% of rows as positive no
matter how many cases were relabelled.

Why that inflates the shift. Take mild 40%: 190 of 474 cases are relabelled, so
the analyst now observes 284 cases (32%) and 602 controls (68%). The code still
flags 474 rows. Almost every observed case is inside those 474, so reported
sensitivity is pushed towards 1. The leftover ~190 flags land on observed
controls, so reported specificity is pushed towards 1 - 190/602 ~ 0.68. Both
movements follow from the count, before the model has said anything.

A real analyst cannot do this. They do not know the clean prevalence; they only
see their own contaminated labels. So the honest cut-off is the one derived from
the OBSERVED prevalence, recomputed per arm. That is policy `observed` below.

Three policies are reported side by side so the size of the artefact is visible
rather than argued about:

  clean_prevalence   the current behaviour: top 53.5% always (what Julia spotted)
  observed           top `observed prevalence` fraction, recomputed per arm (fix)
  fixed_0.50         p >= 0.50, which is what the original paper does

On the paper-style question
---------------------------
The published code is in the repo (ANN.py, MLP_model.py). What it actually does:

  ANN.py:107   raw_data.sample(frac=1)              shuffle, NOT stratified
  ANN.py:113   TRAIN_SIZE = int(data_len * 0.7)     single 70/30 split, no CV
  ANN.py:166   tf.argmax(hyp, 1)                    2-class softmax + argmax,
                                                    i.e. a fixed 0.50 cut-off
  ANN.py:173   tf.confusion_matrix(...)             metrics from that one split

So the paper differs from us in three ways at once: a single unstratified 70/30
split instead of 5-fold stratified CV, a TensorFlow MLP instead of AdaBoost, and
argmax instead of a prevalence-matched cut-off. AUC itself is threshold-free, so
only the split and the model can move it; sensitivity and specificity are moved
by all three.

Section 3 runs our model under the paper's evaluation geometry (single 70/30
unstratified split, fixed 0.50) so the two can be compared without the protocol
difference in the way. It does not reimplement their MLP: the point is to isolate
the evaluation protocol, not to re-run their model.

    cd src && python3 threshold_audit.py
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from contamination_controlled import build_scores, load, select

warnings.filterwarnings("ignore")
SEED = 42
ARMS = ["random", "mild", "atypical"]
LEVELS = [0.20, 0.40]


def hr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def sens_spec(pred, y_ref):
    return (float(pred[y_ref == 1].mean()) if (y_ref == 1).any() else np.nan,
            float(1.0 - pred[y_ref == 0].mean()) if (y_ref == 0).any() else np.nan)


def at_policy(p, y_ref, policy, clean_rate):
    """Threshold the scores under one of the three policies."""
    if policy == "fixed_0.50":
        return (p >= 0.50).astype(int)
    rate = clean_rate if policy == "clean_prevalence" else float((y_ref == 1).mean())
    k = max(1, int(round(rate * len(p))))
    thr = np.sort(p)[::-1][min(k, len(p) - 1)]
    return (p >= thr).astype(int)


def cv_proba(X, y, folds, seed):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(AdaBoostClassifier(random_state=SEED), X, y,
                             cv=cv, method="predict_proba")[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/external/Dataset.xlsx")
    ap.add_argument("--out", default="../out")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    X, y_true, sym = load(a.input)
    sev, atyp, _ = build_scores(X)
    clean_rate = float(y_true.mean())

    hr("threshold audit")
    print(f"  {len(y_true)} rows, {int(y_true.sum())} cases, clean prevalence {clean_rate:.3f}")
    print(f"  {a.repeats} repeats x {a.folds}-fold CV per cell")

    # ---- 1. how many rows does each policy flag? --------------------------
    hr("1. WHAT EACH POLICY ACTUALLY FLAGS  (before any model quality enters)")
    rows = []
    rng = np.random.default_rng(SEED)
    for arm in ARMS:
        for lv in LEVELS:
            y_obs, _ = select(y_true, lv, arm, sev, atyp, rng)
            obs_rate = float(y_obs.mean())
            rows.append({"arm": arm, "level": lv,
                         "observed_cases": int(y_obs.sum()),
                         "observed_prevalence": obs_rate,
                         "flagged_clean_prevalence": int(round(clean_rate * len(y_obs))),
                         "flagged_observed": int(round(obs_rate * len(y_obs)))})
    fl = pd.DataFrame(rows)
    print(fl.round(3).to_string(index=False))
    print("\n   The gap between the last two columns is the artefact, in rows.")
    print("   Those extra flags have to land somewhere, and the only place left is")
    print("   the observed controls, which is what drives specificity down.")

    # ---- 2. rerun sens/spec under all three policies ----------------------
    hr("2. SENSITIVITY AND SPECIFICITY RERUN UNDER EACH POLICY")
    res = []
    for rep in range(a.repeats):
        rng = np.random.default_rng(SEED + rep)
        tr, te = train_test_split(np.arange(len(y_true)), test_size=0.30,
                                  random_state=SEED + rep, stratify=y_true)
        # clean baseline
        p_clean = cv_proba(X, y_true, a.folds, SEED + rep)
        for pol in ("clean_prevalence", "observed", "fixed_0.50"):
            se, sp = sens_spec(at_policy(p_clean, y_true, pol, clean_rate), y_true)
            res.append({"arm": "clean", "level": 0.0, "policy": pol, "rep": rep,
                        "reported_AUC": roc_auc_score(y_true, p_clean),
                        "reported_sens": se, "reported_spec": sp})
        for arm in ARMS:
            for lv in LEVELS:
                y_obs, _ = select(y_true, lv, arm, sev, atyp, rng)
                p_dirty = cv_proba(X, y_obs, a.folds, SEED + rep)
                for pol in ("clean_prevalence", "observed", "fixed_0.50"):
                    # the analyst's own view: dirty model, dirty yardstick
                    se, sp = sens_spec(at_policy(p_dirty, y_obs, pol, clean_rate), y_obs)
                    res.append({"arm": arm, "level": lv, "policy": pol, "rep": rep,
                                "reported_AUC": roc_auc_score(y_obs, p_dirty),
                                "reported_sens": se, "reported_spec": sp})
        print(f"    repeat {rep + 1}/{a.repeats}", flush=True)
    df = pd.DataFrame(res)
    df.to_csv(out / "threshold_audit_raw.csv", index=False)

    s = (df.groupby(["policy", "arm", "level"])[["reported_AUC", "reported_sens", "reported_spec"]]
         .mean().reset_index())
    s.to_csv(out / "threshold_audit_summary.csv", index=False)
    for pol in ("clean_prevalence", "observed", "fixed_0.50"):
        print(f"\n-- policy: {pol} --")
        print(s[s["policy"] == pol].drop(columns="policy").round(4).to_string(index=False))

    hr("2b. THE HEADLINE NUMBERS, OLD VS FIXED  (mild 40%, reported view)")
    base = s[(s.arm == "clean")].set_index("policy")
    m40 = s[(s.arm == "mild") & (s.level == 0.40)].set_index("policy")
    for pol in ("clean_prevalence", "observed", "fixed_0.50"):
        b, m = base.loc[pol], m40.loc[pol]
        print(f"  {pol:<18} clean  sens {b.reported_sens:.3f}  spec {b.reported_spec:.3f}"
              f"   ->  mild40  sens {m.reported_sens:.3f}  spec {m.reported_spec:.3f}"
              f"   (d_sens {m.reported_sens - b.reported_sens:+.3f},"
              f" d_spec {m.reported_spec - b.reported_spec:+.3f})")

    # ---- 3. paper-style evaluation geometry -------------------------------
    hr("3. PAPER-STYLE EVALUATION  (single unstratified 70/30 split, cut-off 0.50)")
    print("  ANN.py: sample(frac=1) -> no stratification; TRAIN_SIZE = 0.7*n -> one split;")
    print("  argmax on a 2-class softmax -> a fixed 0.50 cut-off. AUC is threshold-free,")
    print("  so only the split geometry can move it.\n")
    prows = []
    for rep in range(a.repeats):
        idx = np.random.default_rng(SEED + rep).permutation(len(y_true))  # unstratified
        cut = int(len(idx) * 0.7)
        tr, te = idx[:cut], idx[cut:]
        m = AdaBoostClassifier(random_state=SEED).fit(X.iloc[tr], y_true[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        se, sp = sens_spec((p >= 0.50).astype(int), y_true[te])
        prows.append({"protocol": "paper_style_70_30_split_cut0.50",
                      "AUC": roc_auc_score(y_true[te], p), "sens": se, "spec": sp})
    ours = df[(df.arm == "clean") & (df.policy == "fixed_0.50")]
    oursr = df[(df.arm == "clean") & (df.policy == "clean_prevalence")]
    pp = pd.DataFrame(prows)
    comp = pd.DataFrame([
        {"protocol": "paper style: 70/30 unstratified, cut 0.50",
         "AUC": pp.AUC.mean(), "sens": pp.sens.mean(), "spec": pp.spec.mean()},
        {"protocol": "ours: 5-fold stratified CV, cut 0.50",
         "AUC": ours.reported_AUC.mean(), "sens": ours.reported_sens.mean(),
         "spec": ours.reported_spec.mean()},
        {"protocol": "ours: 5-fold stratified CV, prevalence-matched cut",
         "AUC": oursr.reported_AUC.mean(), "sens": oursr.reported_sens.mean(),
         "spec": oursr.reported_spec.mean()},
    ])
    comp.to_csv(out / "threshold_audit_paper_style.csv", index=False)
    print(comp.round(4).to_string(index=False))
    print("\n  Same model, same data, clean labels throughout. Any difference here is")
    print("  the evaluation protocol alone, which is what 'apples to apples' means.")

    hr("written")
    for f in ["threshold_audit_raw.csv", "threshold_audit_summary.csv",
              "threshold_audit_paper_style.csv"]:
        q = out / f
        if q.exists():
            print(f"  {f:<36} {q.stat().st_size / 1024:>8.1f} KB")


if __name__ == "__main__":
    main()
