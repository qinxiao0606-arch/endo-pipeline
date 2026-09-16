"""
Separating training-label contamination from evaluation-label contamination.

Julia, September:

  "separate before we build the story: training-label contamination versus
   evaluation-label contamination. could you run/describe these separately:
   a] contaminate the training labels only, then evaluate on a completely
   untouched test set against the original labels; b] contaminate the labels that
   would also serve as the study's apparent reference standard, and compare
   'observed' versus original/latent truth. then we answer two different
   questions: does contamination damage the learned model? versus can a
   contaminated reference standard make that damage invisible in the reported
   AUC?"

Why this was needed
-------------------
contamination_controlled.py corrupts every label and then cross-validates, so the
same corrupted labels train the model AND score it. When the reported AUC barely
moves, two explanations are indistinguishable: the model was not damaged, or the
model was damaged and the corrupted answer key hid it. This script pulls a fixed
untouched test set out first, so each effect can be measured on its own.

Four conditions, a 2x2 over what is corrupted:

  CLEAN     train clean,  score against untouched test labels
            the reference point

  (a) TRAIN train dirty,  score against untouched test labels
            -> "does contamination damage the learned model?"
            The yardstick is clean, so any drop here is real model damage.

  (b) EVAL  train clean,  score against corrupted test labels
            -> "can a corrupted reference standard hide damage?"
            The model is identical to CLEAN by construction, so any movement here
            is pure measurement distortion. If reported AUC RISES above CLEAN, a
            corrupted reference standard is not merely noisy, it flatters.

  BOTH      train dirty,  score against corrupted test labels, and also against
            the original ones
            What a real study actually experiences: the reported number and the
            latent truth at the same time.

    cd src && python3 contamination_split.py --repeats 25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from contamination_controlled import build_scores, load, select
from groups import check_goldstein_coverage

SEED = 42
LEVELS = [0.0, 0.10, 0.20, 0.30, 0.40]
ARMS = ["random", "mild", "atypical"]
TEST_SIZE = 0.35


def hr(t):
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}")


def fit_score(Xtr, ytr, Xte):
    return AdaBoostClassifier(random_state=SEED).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def auc(y, p):
    return roc_auc_score(y, p) if len(set(y)) > 1 else np.nan


def sens_spec_at(p, y_ref, rate):
    """Sensitivity and specificity when the top `rate` fraction by score is flagged.

    AUC only reads the ordering, so it is stubborn under contamination. Sensitivity
    and specificity depend on where each score falls relative to a fixed cut, which
    is why Julia asked for them: they move when AUC will not.
    """
    k = max(1, int(round(rate * len(p))))
    thr = np.sort(p)[::-1][min(k, len(p) - 1)]
    pred = (p >= thr).astype(int)
    pos, neg = y_ref == 1, y_ref == 0
    se = float(pred[pos].mean()) if pos.sum() else np.nan
    sp = float(1.0 - pred[neg].mean()) if neg.sum() else np.nan
    return se, sp, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/external/Dataset.xlsx")
    ap.add_argument("--out", default="../out")
    ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--sharpness", type=float, default=3.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X, y_true, sym = load(args.input)
    n, k = check_goldstein_coverage(sym)
    print(f"group check ok: {n} variables in {k} groups (definitions from groups.py)")

    severity, atypicality, _ = build_scores(X)
    rate = float(y_true.mean())
    idx_all = np.arange(len(y_true))

    hr("SETUP")
    print(f"{len(y_true)} rows, {int(y_true.sum())} cases, prevalence {rate:.3f}")
    print(f"test set {TEST_SIZE:.0%}, held out and never trained on")
    print(f"{len(ARMS)} arms x {len(LEVELS)} levels x {args.repeats} splits, AdaBoost")

    recs = []
    for r in range(args.repeats):
        tr, te = train_test_split(idx_all, test_size=TEST_SIZE, random_state=SEED + r,
                                  stratify=y_true)
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr_clean, yte_clean = y_true[tr], y_true[te]

        # CLEAN: the model every other condition is compared against. Trained once
        # per split and reused, so condition (b) is guaranteed to hold the model
        # fixed and vary only the yardstick.
        p_clean = fit_score(Xtr, ytr_clean, Xte)
        auc_clean = auc(yte_clean, p_clean)
        sens_clean, spec_clean, _ = sens_spec_at(p_clean, yte_clean, rate)

        sev_tr = severity.iloc[tr].reset_index(drop=True)
        aty_tr = atypicality.iloc[tr].reset_index(drop=True)
        sev_te = severity.iloc[te].reset_index(drop=True)
        aty_te = atypicality.iloc[te].reset_index(drop=True)

        for arm in ARMS:
            for frac in LEVELS:
                rng = np.random.default_rng(SEED * 7919 + r * 131 + int(frac * 100))

                ytr_dirty, _ = select(ytr_clean, frac, arm, sev_tr, aty_tr,
                                      rng, args.sharpness)
                yte_dirty, _ = select(yte_clean, frac, arm, sev_te, aty_te,
                                      rng, args.sharpness)
                if ytr_dirty.sum() < 15:
                    continue

                p_dirty = fit_score(Xtr, ytr_dirty, Xte)

                # (a) dirty model, clean yardstick -> pure model damage
                a_auc = auc(yte_clean, p_dirty)
                a_sens, a_spec, _ = sens_spec_at(p_dirty, yte_clean, rate)

                # (b) clean model, dirty yardstick -> pure measurement distortion
                b_auc = auc(yte_dirty, p_clean)
                b_sens, b_spec, _ = sens_spec_at(p_clean, yte_dirty, rate)

                # BOTH, scored the way a real study would score it
                both_sens, both_spec, _ = sens_spec_at(p_dirty, yte_dirty, rate)

                # BOTH: what a real study sees, and what is actually true
                both_reported = auc(yte_dirty, p_dirty)
                both_latent = a_auc

                recs.append({
                    "arm": arm, "contamination": frac, "split": r,
                    "clean_AUC": auc_clean, "clean_sens": sens_clean,
                    "clean_spec": spec_clean,
                    "a_model_damage_AUC": a_auc, "a_model_damage_sens": a_sens,
                    "a_model_damage_spec": a_spec,
                    "b_measurement_AUC": b_auc, "b_measurement_sens": b_sens,
                    "b_measurement_spec": b_spec,
                    "both_reported_AUC": both_reported,
                    "both_latent_AUC": both_latent,
                    "both_reported_sens": both_sens, "both_reported_spec": both_spec,
                    "both_latent_sens": a_sens, "both_latent_spec": a_spec,
                })

    d = pd.DataFrame(recs)
    d.to_csv(out / "contamination_split_raw.csv", index=False)
    cols = ["clean_AUC", "clean_sens", "clean_spec",
            "a_model_damage_AUC", "a_model_damage_sens", "a_model_damage_spec",
            "b_measurement_AUC", "b_measurement_sens", "b_measurement_spec",
            "both_reported_AUC", "both_latent_AUC",
            "both_reported_sens", "both_latent_sens",
            "both_reported_spec", "both_latent_spec"]
    s = d.groupby(["arm", "contamination"])[cols].mean().reset_index()

    s["a_delta"] = s.a_model_damage_AUC - s.clean_AUC
    s["b_delta"] = s.b_measurement_AUC - s.clean_AUC
    s["both_hidden"] = s.both_reported_AUC - s.both_latent_AUC
    s.to_csv(out / "contamination_split_summary.csv", index=False)

    hr("(a) DOES CONTAMINATION DAMAGE THE LEARNED MODEL?")
    print("Training labels corrupted. Test set never touched, scored against the")
    print("original labels. A clean yardstick, so every drop below is real damage.\n")
    print(f"{'arm':<10}{'level':>7}{'AUC':>8}{'d':>8}{'sens':>8}{'d':>8}{'spec':>8}{'d':>8}")
    print("-" * 64)
    for arm in ARMS:
        for _, x in s[s.arm == arm].iterrows():
            print(f"{arm:<10}{x.contamination:>6.0%}{x.a_model_damage_AUC:>8.3f}"
                  f"{x.a_delta:>+8.3f}{x.a_model_damage_sens:>8.3f}"
                  f"{x.a_model_damage_sens - x.clean_sens:>+8.3f}"
                  f"{x.a_model_damage_spec:>8.3f}"
                  f"{x.a_model_damage_spec - x.clean_spec:>+8.3f}")
        print()

    hr("(b) CAN A CORRUPTED REFERENCE STANDARD HIDE THE DAMAGE?")
    print("The model is the CLEAN one, byte for byte identical across every row")
    print("below. Only the yardstick changes. Any movement is measurement alone.\n")
    print(f"{'arm':<10}{'level':>7}{'AUC':>8}{'d':>8}{'sens':>8}{'d':>8}{'spec':>8}{'d':>8}")
    print("-" * 64)
    for arm in ARMS:
        for _, x in s[s.arm == arm].iterrows():
            print(f"{arm:<10}{x.contamination:>6.0%}{x.b_measurement_AUC:>8.3f}"
                  f"{x.b_delta:>+8.3f}{x.b_measurement_sens:>8.3f}"
                  f"{x.b_measurement_sens - x.clean_sens:>+8.3f}"
                  f"{x.b_measurement_spec:>8.3f}"
                  f"{x.b_measurement_spec - x.clean_spec:>+8.3f}")
        print()

    hr("BOTH AT ONCE: WHAT A REAL STUDY WOULD LIVE THROUGH")
    print("Training and reference standard both corrupted. 'reported' is the number")
    print("that would appear in the paper, 'latent' is the truth. The gap is how")
    print("much of the damage never becomes visible.\n")
    print(f"{'arm':<10}{'level':>7}{'AUC rep':>9}{'AUC lat':>9}{'gap':>8}"
          f"{'sens rep':>10}{'sens lat':>10}{'spec rep':>10}{'spec lat':>10}")
    print("-" * 76)
    for arm in ARMS:
        for _, x in s[s.arm == arm].iterrows():
            print(f"{arm:<10}{x.contamination:>6.0%}{x.both_reported_AUC:>9.3f}"
                  f"{x.both_latent_AUC:>9.3f}{x.both_hidden:>+8.3f}"
                  f"{x.both_reported_sens:>10.3f}{x.both_latent_sens:>10.3f}"
                  f"{x.both_reported_spec:>10.3f}{x.both_latent_spec:>10.3f}")
        print()

    hr("ANSWERS")
    top = LEVELS[-1]
    for arm in ARMS:
        x = s[(s.arm == arm) & (s.contamination == top)].iloc[0]
        print(f"{arm:<10} at {top:.0%}")
        print(f"    (a) model damage        {x.a_delta:+.3f} AUC, "
              f"{x.a_model_damage_sens - x.clean_sens:+.3f} sens, "
              f"{x.a_model_damage_spec - x.clean_spec:+.3f} spec")
        print(f"    (b) measurement effect  {x.b_delta:+.3f} AUC, "
              f"{x.b_measurement_sens - x.clean_sens:+.3f} sens, "
              f"{x.b_measurement_spec - x.clean_spec:+.3f} spec "
              f"(same model, only the yardstick moved)")
        print(f"    hidden when both apply  {x.both_hidden:+.3f} AUC")
        print()
    print(f"written to {out}/")


if __name__ == "__main__":
    main()
