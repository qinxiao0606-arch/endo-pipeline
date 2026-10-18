"""
Other models, fitted on OUR questionnaire, compared with the carried-over Goldstein model.

Why this exists
---------------
transport.py applies the Goldstein 9-variable model to our answers unchanged. On all
our controls it ranks cases above controls only modestly (AUC about 0.6-0.7). Julia
(2026-10): see whether other models do better on our data.

"Better" can mean two different things, and the script keeps them apart:
  1. a model FITTED ON OUR ANSWERS (cross-validated, so every score comes from a model
     that never saw that person) instead of one fitted on Goldstein's;
  2. a different ALGORITHM (logistic regression, random forest, gradient boosting,
     AdaBoost = Goldstein's choice).

Every model is scored on the same four comparisons as transport.py section 6, because
on our data the answer depends far more on WHICH controls than on which model:
  E1 all cases vs all controls
  E2 all cases vs clean-channel controls
  E3 channel-matched: clean-channel cases vs clean-channel controls  (the real test)
  E4 all cases vs community / unknown-channel controls

Two training sets:
  R1 every row, labels as given
  R2 cases + clean-channel controls only (community controls left out of training,
     because about 40% of them behave like patients on one of two independent signs)
Rows outside the training set (R2: the community controls) are scored by the fold
models and averaged, so they are out-of-sample too.

Three feature sets:
  F9   the 9 shared themes, binary, exactly what the Goldstein model reads
  F22  all 22 questionnaire features (encode.py: ordinal scales kept, missing flagged)
  F20  F22 without age band and hormonal contraception. Age is a pseudo-predictor here
       (the student link brought almost only under-25 controls); current hormonal
       treatment can be a consequence of the diagnosis rather than a sign of it.

Reported per comparison: AUC with a stratified-bootstrap 95% CI, and specificity at
80% sensitivity (threshold-free across models, so models are compared on one footing).

Also: each F9 model fitted on all our rows is applied to the Goldstein data, i.e. the
transport run in the other direction.

What it can NOT do: cross-validation on our own data is internal validation. It is
the optimistic counterpart of transport, not the same question. And no model can
learn past labels that are wrong: if a control is an undiagnosed patient, a model
that scores her high is counted as wrong.

    cd src && python3 other_models.py                  # 5 CV repeats, 1000 bootstrap draws
    cd src && python3 other_models.py --repeats 2 --boot 200     # quick look
"""
from __future__ import annotations

import argparse
import contextlib
import io
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from config import load_schema
from contamination_controlled import load
from encode import encode
from ingest import ingest
from model import build_models, logistic_coefficients
from transport import (CLEAN_CHANNELS, SEED, auc_boot, binarise, hr,
                       resubmission_mask)
from validate import run_all

warnings.filterwarnings("ignore")

MODEL_NAMES = {"Régression logistique": "logistic regression",
               "Forêt aléatoire": "random forest",
               "Gradient boosting": "gradient boosting"}
SENS_TARGET = 0.80


def models():
    m = {MODEL_NAMES[k]: v for k, v in build_models().items()}
    m["AdaBoost (Goldstein's algorithm)"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", AdaBoostClassifier(random_state=SEED))])
    return m


def spec_at_sens(y, p, target=SENS_TARGET):
    """Specificity at the first cut-off that reaches the target sensitivity."""
    fpr, tpr, _ = roc_curve(y, p)
    i = int(np.argmax(tpr >= target))
    return float(1.0 - fpr[i])


def oof_scores(mdl, X, y, train, repeats, folds=5):
    """Out-of-sample score for every row.

    Rows in `train` get the score of the fold model that held them out; rows outside
    `train` get the mean score of all fold models (none of which saw them). Averaged
    over `repeats` different fold assignments.
    """
    X = X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else X
    idx_tr = np.flatnonzero(train)
    idx_out = np.flatnonzero(~train)
    total = np.zeros(len(y))
    for r in range(repeats):
        p = np.zeros(len(y))
        out_sum = np.zeros(len(idx_out))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED + r)
        for a, b in cv.split(idx_tr, y[idx_tr]):
            m = clone(mdl).fit(X[idx_tr[a]], y[idx_tr[a]])
            p[idx_tr[b]] = m.predict_proba(X[idx_tr[b]])[:, 1]
            if len(idx_out):
                out_sum += m.predict_proba(X[idx_out])[:, 1]
        if len(idx_out):
            p[idx_out] = out_sum / folds
        total += p
    return total / repeats


def score(y, p, mask, n_boot):
    ym, pm = y[mask], p[mask]
    n1, n0 = int(ym.sum()), int((ym == 0).sum())
    if min(n1, n0) < 10:
        return {"n_cases": n1, "n_controls": n0}
    lo, hi = auc_boot(ym, pm, n_boot, SEED)
    return {"n_cases": n1, "n_controls": n0, "AUC": roc_auc_score(ym, pm),
            "AUC_lo": lo, "AUC_hi": hi, "spec_at_80sens": spec_at_sens(ym, pm)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../out/reduced_model.pkl")
    ap.add_argument("--goldstein", default="../data/external/Dataset.xlsx")
    ap.add_argument("--out", default="../out")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # ---- data: identical to transport.py (ingest, checks, same-link resubmissions) ----
    with contextlib.redirect_stdout(io.StringIO()):
        schema = load_schema()
        df, _ = run_all(ingest(schema=schema), schema, strict=False)
    keep, _ = resubmission_mask(df, [c for c in schema.data_vars if c in df.columns])
    df = df[keep].reset_index(drop=True)
    y = (df["Groupe"] == "Cas").astype(int).to_numpy()
    ch = df["Source_canal"].to_numpy()
    clean = np.isin(ch, CLEAN_CHANNELS)
    is_case, is_ctrl = y == 1, y == 0

    hr("OTHER MODELS ON OUR DATA  (cross-validated) vs THE CARRIED-OVER GOLDSTEIN MODEL")
    print(f"  {len(y)} rows | {int(is_case.sum())} cases / {int(is_ctrl.sum())} controls | "
          f"clean channels {', '.join(CLEAN_CHANNELS)}: "
          f"{int((is_case & clean).sum())} cases / {int((is_ctrl & clean).sum())} controls")
    print(f"  {a.repeats} CV repeats x 5 folds, {a.boot} bootstrap draws per interval")

    evals = {"E1 all controls": is_case | is_ctrl,
             "E2 clean controls": is_case | (is_ctrl & clean),
             "E3 channel-matched": clean,
             "E4 community controls": is_case | (is_ctrl & ~clean)}

    bundle = pickle.load(open(a.model, "rb"))
    F9 = binarise(df, "preregistered")[bundle["features"]]
    with contextlib.redirect_stdout(io.StringIO()):
        F22, _ = encode(df, schema)
    drop = [c for c in F22.columns
            if c.startswith("Tranche_age") or c.startswith("Contraception_hormonale")]
    F20 = F22.drop(columns=drop)
    fsets = {"F9 themes": F9, "F22 all items": F22, "F20 no age/contraception": F20}
    print(f"  feature columns: F9 {F9.shape[1]}, F22 {F22.shape[1]}, F20 {F20.shape[1]}")

    rows = []
    scores = {}

    def add(regime, fset, name, p):
        scores[(regime, fset, name)] = p
        for e, m in evals.items():
            rows.append({"regime": regime, "features": fset, "model": name,
                         "comparison": e, **score(y, p, m, a.boot)})

    # ---- references: no fitting on our data ---------------------------------------
    p_gs = bundle["ada"]["model"].predict_proba(F9)[:, 1]
    add("reference", "F9 themes", "Goldstein model, carried over (transport.py)", p_gs)
    add("reference", "F9 themes", "count of themes reported (no model)",
        F9.sum(axis=1).to_numpy().astype(float))

    # ---- fitted on our data ----------------------------------------------------------
    regimes = {"R1 trained on all rows": np.ones(len(y), bool),
               "R2 trained on cases + clean controls": is_case | (is_ctrl & clean)}
    XG, yG, _ = load(a.goldstein)
    rev = []
    for rname, train in regimes.items():
        for fname, X in fsets.items():
            for mname, mdl in models().items():
                p = oof_scores(mdl, X, y, train, a.repeats)
                add(rname, fname, mname, p)
                if fname == "F9 themes":      # the other direction: ours -> Goldstein
                    full = clone(mdl).fit(X.to_numpy(float)[train], y[train])
                    pg = full.predict_proba(XG[bundle["features"]].to_numpy(float))[:, 1]
                    rev.append({"regime": rname, "model": mname,
                                "AUC_on_Goldstein": roc_auc_score(yG, pg)})
            print(f"  done: {rname} / {fname}")

    res = pd.DataFrame(rows)
    res.to_csv(out / "other_models.csv", index=False)
    rev = pd.DataFrame(rev)
    rev.to_csv(out / "other_models_on_goldstein.csv", index=False)

    # ---- report ------------------------------------------------------------------------
    def table(comp):
        t = res[res.comparison == comp].copy()
        t["AUC [95% CI]"] = t.apply(
            lambda r: (f"{r.AUC:.3f} [{r.AUC_lo:.3f}-{r.AUC_hi:.3f}]"
                       if pd.notna(r.get("AUC")) else "not scored"), axis=1)
        t["spec@80%sens"] = t["spec_at_80sens"].map(
            lambda v: f"{v:.2f}" if pd.notna(v) else "")
        return t[["regime", "features", "model", "n_cases", "n_controls",
                  "AUC [95% CI]", "spec@80%sens"]]

    for comp in evals:
        hr(comp)
        print(table(comp).to_string(index=False))

    hr("BEST FITTED MODEL PER COMPARISON vs THE CARRIED-OVER GOLDSTEIN MODEL")
    fitted = res[res.regime != "reference"].dropna(subset=["AUC"])
    for comp in evals:
        ref = res[(res.comparison == comp)
                  & res.model.str.startswith("Goldstein")].iloc[0]
        sub = fitted[fitted.comparison == comp]
        if sub.empty or pd.isna(ref.get("AUC")):
            continue
        best = sub.loc[sub.AUC.idxmax()]
        print(f"  {comp:24s} Goldstein {ref.AUC:.3f} [{ref.AUC_lo:.3f}-{ref.AUC_hi:.3f}]"
              f"   best fitted {best.AUC:.3f} [{best.AUC_lo:.3f}-{best.AUC_hi:.3f}]"
              f"  ({best.model}, {best.features}, {best.regime[:2]})")
    print("\n   'Best of many' is itself optimistic: with 24 fitted variants the top one")
    print("   is partly picked by luck. Read the spread across models, not only the top.")

    hr("PAIRED DIFFERENCE: fitted on our data minus the carried-over Goldstein model")
    print("   Same people scored by both models, so the difference gets its own interval")
    print("   (stratified bootstrap of the pairs). Comparators: the SIMPLEST fitted model,")
    print("   logistic regression trained on all rows, once with the same 9 themes and once")
    print("   with all items minus age / contraception. Named by that rule, but after the")
    print("   tables above had been seen, so treat them as descriptive, not confirmatory.")
    ref = scores[("reference", "F9 themes", "Goldstein model, carried over (transport.py)")]
    pairs = [("R1 trained on all rows", "F9 themes", "logistic regression"),
             ("R1 trained on all rows", "F20 no age/contraception", "logistic regression")]
    rng = np.random.default_rng(SEED)
    diff_rows = []
    for key in pairs:
        p = scores[key]
        for comp, m in evals.items():
            ym, a1, a0 = y[m], p[m], ref[m]
            pos, neg = np.flatnonzero(ym == 1), np.flatnonzero(ym == 0)
            if min(len(pos), len(neg)) < 10:
                continue
            d = np.empty(a.boot)
            for b in range(a.boot):
                i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
                d[b] = roc_auc_score(ym[i], a1[i]) - roc_auc_score(ym[i], a0[i])
            est = roc_auc_score(ym, a1) - roc_auc_score(ym, a0)
            lo, hi = np.percentile(d, [2.5, 97.5])
            diff_rows.append({"fitted": f"{key[2]}, {key[1]}", "comparison": comp,
                              "AUC_diff": est, "lo": lo, "hi": hi})
            print(f"   {key[1]:26s} {comp:22s} {est:+.3f}  95% CI {lo:+.3f} to {hi:+.3f}")
    pd.DataFrame(diff_rows).to_csv(out / "other_models_paired_diff.csv", index=False)

    hr("THE OTHER DIRECTION: F9 models fitted on OUR rows, applied to Goldstein")
    print(rev.round(3).to_string(index=False))
    print("\n   The carried-over direction (Goldstein -> ours) gives about 0.6-0.7 on all")
    print("   our controls. If models fitted on our noisy labels still rank Goldstein's")
    print("   cases well, Goldstein's data is the easy one, not our model the bad one.")

    hr("WHAT A MODEL FITTED ON ALL OUR ROWS LEANS ON  (logistic regression, F22)")
    with contextlib.redirect_stdout(io.StringIO()):
        coef = logistic_coefficients(F22, pd.Series(y), top=12)
    print(coef.round(3).to_string(index=False))

    hr("written")
    for f in ["other_models.csv", "other_models_paired_diff.csv",
              "other_models_on_goldstein.csv"]:
        print(f"  {f:34s} {(out / f).stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
