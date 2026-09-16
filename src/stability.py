"""
A2 — Feature stability: how many of a published top-k list actually survive.

Julia, August (verbatim, PROGRESS §8):

  "Not top-10: look at top-5/10/20 for example and determine how many features are
   actually robust ('insights'). Also check which symptom groups/features survive
   vs collapse (between random and controlled scenarios)"

Julia, WhatsApp 2026-09-06 (verbatim, her spelling):

  "and then for the feature part check pls Top-5 / Top-10 / Top-20 smptoms
   stability."

The question in one sentence
---------------------------
If a paper publishes a top-10 list of predictive symptoms, and the negatives in
that paper's data are undiagnosed patients rather than healthy people, how many of
those ten entries are still there afterwards?

Two different meanings of "robust", reported separately
-------------------------------------------------------
These get conflated constantly and they are not the same claim.

  1. RESAMPLING robustness (the floor). Labels are perfectly clean. The only thing
     that changes between replicates is which rows the bootstrap happened to draw.
     A ranking that cannot reproduce itself under clean resampling was never a
     finding in the first place, whatever contamination does to it later. This is
     the `clean` arm and it is the ceiling every other arm is measured against.

  2. CONTAMINATION robustness. Labels are corrupted in a realistic shape on top of
     the resampling. This is the `random` / `mild` / `atypical` arms.

Reporting only (2) would make contamination look responsible for instability that
is really just sampling noise.

Why there is a `shuffled` arm
-----------------------------
An overlap number means nothing without a floor. Two independent uniform random
top-k draws out of p=58 features already share k/58 of their slots on average
(k=10 -> 0.172). So 0.6 overlap is not "fairly stable", it is 0.6 against a floor
of 0.172 that must be printed next to it.

But k/58 is the WRONG floor here, and this script says so loudly. AdaBoost with 50
stumps can only ever use at most 50 features, and on this data it uses about 30:
the other ~28 get importance exactly 0.0 under BOTH measures (a feature no stump
ever looks at cannot change predictions when you permute it, so its permutation
importance is a deterministic zero, not noise). The effective candidate pool is
therefore ~30, not 58, and the true chance floor is higher than k/58 -- possibly
much higher, because which ~30 get used is itself driven by prevalence and is
partly repeatable even with no signal at all.

The `shuffled` arm measures that empirically: labels are randomly permuted, so
there is no signal by construction, and whatever consistency the procedure still
shows is pure floor. Read the empirical floor, not the analytic one. The analytic
one is printed only so the gap between them is visible.

Two importance measures
-----------------------
  builtin      AdaBoost's own feature_importances_ (how much each feature was used
               to split, weighted)
  permutation  how much roc_auc falls when that one column is shuffled (the
               pipeline convention, standing in for SHAP)

Disagreement between them would itself be a result. Expect little disagreement
here and understand why before reporting it as agreement: for a stump ensemble the
two measures share most of their machinery, because both are pinned to exactly
zero for every feature the ensemble never used.

Design choices worth knowing about before reading the numbers
-------------------------------------------------------------
* Bootstrap is STRATIFIED (resampled within each class), so prevalence is held
  fixed. Otherwise class balance wobbles between replicates and adds a second,
  uninteresting source of rank movement on top of the one being measured.

* Order is RESAMPLE FIRST, THEN CONTAMINATE, with a fresh contamination draw per
  replicate. A real study gets one unknown draw of which cases are undiagnosed, so
  that draw's randomness belongs inside the error bars, not outside them.

* Severity and atypicality are recomputed inside each bootstrap sample rather than
  once on the full data, because they are percentile ranks and a percentile rank
  is a statement about the sample you are in.

* Importance is computed IN-SAMPLE, on the same rows the model was fitted to, with
  the OBSERVED (contaminated) labels. This is deliberate. Each bootstrap replicate
  is standing in for one hypothetical parallel study, and a parallel study would
  fit on its own data and report importance from that fit -- it has no clean
  labels and no untouched holdout to use instead. Measuring on held-out rows would
  be better statistics and worse mimicry, and mimicry is the point here.

* Selection logic is imported from contamination_controlled.py, group definitions
  from groups.py. Nothing is redefined locally. This is what stops the copies from
  drifting apart.

Two stages, because stage 1 is the expensive one
------------------------------------------------
  stage 1   fit + importance for every (arm, level, replicate). Writes
            stability_raw.csv incrementally, ONE ARM-LEVEL AT A TIME. Killed
            halfway = keep everything finished so far; rerunning skips what is
            already on disk. permutation_importance runs with n_jobs=1 on purpose:
            n_jobs=-1 makes joblib copy the dataset once per core and is what
            killed the previous attempt. This is an implementation fix, not a
            method change.
  stage 2   all summaries, read back off stability_raw.csv. Cheap, rerunnable,
            safe to iterate on without refitting anything.

    cd src && python3 stability.py                    # both stages
    cd src && python3 stability.py --stage 1           # just the fits
    cd src && python3 stability.py --stage 2           # just the summaries
    cd src && python3 stability.py --stage 1 --combos mild@0.4 atypical@0.4
"""
from __future__ import annotations

import argparse
import itertools
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.inspection import permutation_importance

from contamination_controlled import build_scores, load, select
from groups import GOLDSTEIN as G, GOLDSTEIN_LABELS, check_goldstein_coverage

SEED = 42
KS = [5, 10, 20]
METHODS = ["builtin", "permutation"]

# (arm, contamination level). clean and shuffled are the two reference arms.
COMBOS = [("clean", 0.00),
          ("shuffled", 0.00),
          ("random", 0.20), ("random", 0.40),
          ("mild", 0.20), ("mild", 0.40),
          ("atypical", 0.20), ("atypical", 0.40)]

# "appears in top-k in at least this fraction of replicates" = robust
FREQ_THRESHOLDS = [0.60, 0.80, 0.90]
HEADLINE_THRESHOLD = 0.80

RAW = "stability_raw.csv"


def hr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def combo_id(arm, level):
    return f"{arm}@{level:.2f}"


# --------------------------------------------------------------------------
# stage 1 : fits
# --------------------------------------------------------------------------

def strat_boot(y, rng):
    """Bootstrap indices, resampled within each class so prevalence is fixed."""
    out = []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        out.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(out)


def one_replicate(X, y, arm, level, rng, perm_repeats):
    """One hypothetical parallel study: resample, contaminate, fit, rank."""
    boot = strat_boot(y, rng)
    Xb = X.iloc[boot].reset_index(drop=True)
    yb_true = y[boot]

    if arm == "clean":
        yb_obs, n_flipped = yb_true, 0
    elif arm == "shuffled":
        yb_obs, n_flipped = rng.permutation(yb_true), -1
    else:
        sev, atyp, _ = build_scores(Xb)
        yb_obs, picked = select(yb_true, level, arm, sev, atyp, rng)
        n_flipped = len(picked)

    m = AdaBoostClassifier(random_state=SEED).fit(Xb, yb_obs)

    imp = {"builtin": np.asarray(m.feature_importances_, dtype=float)}
    r = permutation_importance(m, Xb, yb_obs, scoring="roc_auc",
                               n_repeats=perm_repeats, n_jobs=1,
                               random_state=int(rng.integers(0, 2**31 - 1)))
    imp["permutation"] = np.asarray(r.importances_mean, dtype=float)
    return imp, n_flipped


def rank_frame(imp, cols, arm, level, rep, n_flipped):
    rows = []
    for method, v in imp.items():
        s = pd.Series(v, index=cols)
        # rank 1 = most important. method="first" breaks ties by column order,
        # which is deterministic; n_zero below tells you how much that matters.
        rk = s.rank(ascending=False, method="first").astype(int)
        rows.append(pd.DataFrame({"arm": arm, "level": level, "rep": rep,
                                  "method": method, "feature": cols,
                                  "importance": s.to_numpy(),
                                  "rank": rk.to_numpy(),
                                  "n_zero": int((s == 0).sum()),
                                  "n_flipped": n_flipped}))
    return pd.concat(rows, ignore_index=True)


def stage1(X, y, cols, out, repeats, perm_repeats, combos):
    path = out / RAW
    done = set()
    if path.exists():
        prev = pd.read_csv(path)
        for (a, l), g in prev.groupby(["arm", "level"]):
            if g["rep"].nunique() >= repeats:
                done.add(combo_id(a, float(l)))
        if done:
            print(f"resuming: {len(done)} arm-level(s) already complete -> "
                  f"{sorted(done)}")

    for arm, level in combos:
        cid = combo_id(arm, level)
        if cid in done:
            print(f"  skip {cid} (already on disk)")
            continue
        hr(f"stage 1 : {cid}   B={repeats}  perm n_repeats={perm_repeats}  n_jobs=1")
        # One seed per arm-level, so rerunning a single arm reproduces it exactly.
        # zlib.crc32, NOT hash(): Python randomises string hashing per process, so
        # hash() would silently give different results on every run.
        rng = np.random.default_rng(SEED + zlib.crc32(cid.encode()))
        chunks = []
        for rep in range(repeats):
            imp, nf = one_replicate(X, y, arm, level, rng, perm_repeats)
            chunks.append(rank_frame(imp, cols, arm, level, rep, nf))
            print(f"    rep {rep + 1:>3}/{repeats}   flipped={nf}", flush=True)
        df = pd.concat(chunks, ignore_index=True)
        # append as soon as this arm-level finishes: a kill costs at most one arm
        df.to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"  -> appended {len(df):,} rows to {path.name}")


# --------------------------------------------------------------------------
# stage 2 : summaries
# --------------------------------------------------------------------------

def reference_importances(X, y, perm_repeats):
    """The 'published list': one fit on the full clean data, no bootstrap."""
    m = AdaBoostClassifier(random_state=SEED).fit(X, y)
    out = {"builtin": pd.Series(m.feature_importances_, index=X.columns)}
    r = permutation_importance(m, X, y, scoring="roc_auc",
                               n_repeats=perm_repeats, n_jobs=1, random_state=SEED)
    out["permutation"] = pd.Series(r.importances_mean, index=X.columns)
    return out


def topk_sets(df, k):
    """{rep: set of features ranked 1..k} for one (arm, level, method) slice."""
    sel = df[df["rank"] <= k]
    return {rep: set(g["feature"]) for rep, g in sel.groupby("rep")}


def pairwise_overlap(sets, k):
    """Mean |A n B| / k over all replicate pairs. Does the arm reproduce itself?"""
    reps = sorted(sets)
    vals = [len(sets[a] & sets[b]) / k for a, b in itertools.combinations(reps, 2)]
    return float(np.mean(vals)) if vals else np.nan


def recovery(sets, ref_set, k):
    """Mean |replicate n published list| / k. How much of the paper survives."""
    vals = [len(s & ref_set) / k for s in sets.values()]
    return float(np.mean(vals)) if vals else np.nan


def stage2(X, y, cols, out, perm_repeats):
    path = out / RAW
    if not path.exists():
        sys.exit(f"{path} not found — run stage 1 first")
    raw = pd.read_csv(path)
    p = len(cols)

    hr("stage 2 : reference fit on full clean data (the 'published list')")
    ref = reference_importances(X, y, perm_repeats)
    ref_rows, ref_sets = [], {}
    for method, s in ref.items():
        order = s.sort_values(ascending=False)
        for k in KS:
            ref_sets[(method, k)] = set(order.index[:k])
        ref_rows.append(pd.DataFrame({"method": method, "feature": order.index,
                                      "importance": order.to_numpy(),
                                      "rank": np.arange(1, p + 1)}))
        print(f"  {method:<12} features with importance exactly 0.0: "
              f"{int((s == 0).sum())}/{p}  -> effective pool ≈ {int((s != 0).sum())}")
    pd.concat(ref_rows, ignore_index=True).to_csv(
        out / "stability_reference.csv", index=False)

    slices = {(a, float(l), m): g
              for (a, l, m), g in raw.groupby(["arm", "level", "method"])}

    # ---- consistency ------------------------------------------------------
    rows = []
    for (arm, level, method), g in slices.items():
        for k in KS:
            sets = topk_sets(g, k)
            rows.append({
                "arm": arm, "level": level, "method": method, "k": k,
                "n_rep": len(sets),
                "pairwise_overlap": pairwise_overlap(sets, k),
                "recovery_vs_published": recovery(sets, ref_sets[(method, k)], k),
                "baseline_analytic": k / p,
            })
    con = pd.DataFrame(rows)

    # empirical floor = what the shuffled-label arm still manages
    floor = (con[con["arm"] == "shuffled"]
             .set_index(["method", "k"])["pairwise_overlap"])
    con["baseline_empirical"] = [floor.get((m, k), np.nan)
                                 for m, k in zip(con["method"], con["k"])]
    con["above_empirical_floor"] = con["pairwise_overlap"] - con["baseline_empirical"]
    con = con.sort_values(["method", "k", "arm", "level"])
    con.to_csv(out / "stability_consistency.csv", index=False)

    # ---- selection frequency + robust counts ------------------------------
    freq_rows, count_rows = [], []
    for (arm, level, method), g in slices.items():
        n_rep = g["rep"].nunique()
        for k in KS:
            f = (g[g["rank"] <= k].groupby("feature").size() / n_rep)
            f = f.reindex(cols, fill_value=0.0)
            freq_rows.append(pd.DataFrame({
                "arm": arm, "level": level, "method": method, "k": k,
                "feature": f.index, "selection_freq": f.to_numpy()}))
            row = {"arm": arm, "level": level, "method": method, "k": k,
                   "n_rep": n_rep}
            for t in FREQ_THRESHOLDS:
                row[f"n_robust_ge{int(t * 100)}"] = int((f >= t).sum())
            count_rows.append(row)
    freq = pd.concat(freq_rows, ignore_index=True)
    freq.to_csv(out / "stability_selection_freq.csv", index=False)
    counts = pd.DataFrame(count_rows).sort_values(["method", "k", "arm", "level"])
    counts.to_csv(out / "stability_robust_counts.csv", index=False)

    # ---- feature-level survive vs collapse, against the clean arm ---------
    base = (freq[(freq["arm"] == "clean")]
            .set_index(["method", "k", "feature"])["selection_freq"])
    freq["clean_freq"] = [base.get((m, k, f), np.nan) for m, k, f
                          in zip(freq["method"], freq["k"], freq["feature"])]
    freq["delta_vs_clean"] = freq["selection_freq"] - freq["clean_freq"]
    g2f = {v: g for g, lst in G.items() for v in lst}
    freq["group"] = freq["feature"].map(g2f)
    freq["group_label"] = freq["group"].map(GOLDSTEIN_LABELS)
    delta = freq[~freq["arm"].isin(["clean", "shuffled"])].sort_values(
        ["method", "k", "arm", "level", "delta_vs_clean"])
    delta.to_csv(out / "stability_feature_delta.csv", index=False)

    # ---- group-level shares of the top-k slots ---------------------------
    # NOTE the A0-2 caveat printed at the end before reading these clinically.
    gr = (freq.groupby(["arm", "level", "method", "k", "group", "group_label"],
                       dropna=False)["selection_freq"].sum().reset_index()
          .rename(columns={"selection_freq": "expected_slots"}))
    gr["share_of_topk"] = gr["expected_slots"] / gr["k"]
    gbase = (gr[gr["arm"] == "clean"]
             .set_index(["method", "k", "group"])["share_of_topk"])
    gr["clean_share"] = [gbase.get((m, k, g), np.nan) for m, k, g
                         in zip(gr["method"], gr["k"], gr["group"])]
    gr["delta_vs_clean"] = gr["share_of_topk"] - gr["clean_share"]
    gr = gr.sort_values(["method", "k", "arm", "level", "share_of_topk"],
                        ascending=[True, True, True, True, False])
    gr.to_csv(out / "stability_group_shares.csv", index=False)

    report(con, counts, freq, gr, ref_sets, p, out)


def report(con, counts, freq, gr, ref_sets, p, out):
    pd.set_option("display.width", 200, "display.max_columns", 40)

    hr("1. TOP-K CONSISTENCY  (pairwise overlap between replicates, 0-1)")
    print("   clean    = resampling floor: can the ranking reproduce itself at all\n"
          "   shuffled = empirical chance floor: labels carry no signal\n")
    for method in METHODS:
        t = con[con["method"] == method].pivot_table(
            index=["arm", "level"], columns="k", values="pairwise_overlap")
        print(f"\n-- {method} --")
        print(t.round(3))
    print("\n   analytic chance floor k/58:",
          {k: round(k / p, 3) for k in KS})
    emp = con.drop_duplicates(["method", "k"]).set_index(["method", "k"])
    print("   EMPIRICAL chance floor (shuffled arm):")
    for method in METHODS:
        print(f"     {method:<12}",
              {k: round(float(emp.loc[(method, k), "baseline_empirical"]), 3)
               for k in KS})

    hr("2. HOW MUCH OF THE PUBLISHED LIST SURVIVES  (overlap with clean full fit)")
    for method in METHODS:
        t = con[con["method"] == method].pivot_table(
            index=["arm", "level"], columns="k", values="recovery_vs_published")
        print(f"\n-- {method} --")
        print(t.round(3))

    hr(f"3. HOW MANY FEATURES ARE ACTUALLY ROBUST  "
       f"(in top-k in >= {int(HEADLINE_THRESHOLD * 100)}% of replicates)")
    print("   This is Julia's 'insights' count. Read as n_robust / k.\n")
    col = f"n_robust_ge{int(HEADLINE_THRESHOLD * 100)}"
    for method in METHODS:
        t = counts[counts["method"] == method].pivot_table(
            index=["arm", "level"], columns="k", values=col)
        print(f"\n-- {method} --")
        print(t.astype(int))

    hr("4. WHICH SYMPTOM GROUPS COLLAPSE  (share of top-10 slots vs clean arm)")
    print("   ⚠ A0-2: at theme granularity 'which symptoms' is worth only 0.007 AUC\n"
          "   (9 themes as features 0.947 vs count alone 0.940). Group-level\n"
          "   movement is therefore mostly a reallocation of tick-count, NOT a\n"
          "   clinical statement. Do not read these as clinical findings.\n")
    sub = gr[(gr["method"] == "permutation") & (gr["k"] == 10)
             & (~gr["arm"].isin(["clean", "shuffled"]))]
    for (arm, level), g in sub.groupby(["arm", "level"]):
        g = g.sort_values("delta_vs_clean")
        print(f"\n-- {arm} @ {level:.0%} --  biggest losses then biggest gains")
        show = pd.concat([g.head(3), g.tail(3)])
        print(show[["group", "group_label", "clean_share",
                    "share_of_topk", "delta_vs_clean"]].round(3).to_string(index=False))

    hr("5. INDIVIDUAL FEATURES THAT COLLAPSE  (permutation, k=10)")
    sub = freq[(freq["method"] == "permutation") & (freq["k"] == 10)
               & (~freq["arm"].isin(["clean", "shuffled"]))
               & (freq["clean_freq"] >= 0.50)]
    for (arm, level), g in sub.groupby(["arm", "level"]):
        g = g.sort_values("delta_vs_clean").head(5)
        print(f"\n-- {arm} @ {level:.0%} --  features the clean arm kept >=50% of the time")
        print(g[["feature", "group", "clean_freq", "selection_freq",
                 "delta_vs_clean"]].round(3).to_string(index=False))

    hr("written")
    for f in ["stability_raw.csv", "stability_reference.csv",
              "stability_consistency.csv", "stability_selection_freq.csv",
              "stability_robust_counts.csv", "stability_feature_delta.csv",
              "stability_group_shares.csv"]:
        q = out / f
        if q.exists():
            print(f"  {f:<36} {q.stat().st_size / 1024:>8.1f} KB")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/external/Dataset.xlsx")
    ap.add_argument("--out", default="../out")
    ap.add_argument("--repeats", type=int, default=30,
                    help="bootstrap replicates per arm-level (B)")
    ap.add_argument("--perm-repeats", type=int, default=5,
                    help="n_repeats inside permutation_importance")
    ap.add_argument("--stage", default="all", choices=["1", "2", "all"])
    ap.add_argument("--combos", nargs="*", default=None,
                    help="subset like: mild@0.4 atypical@0.4")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y, cols = load(a.data)
    n_var, n_grp = check_goldstein_coverage(list(X.columns))
    hr("A2 — feature stability")
    print(f"  data      {a.data}   {X.shape[0]} rows x {X.shape[1]} features")
    print(f"  labels    {int(y.sum())} positive / {int((y == 0).sum())} negative")
    print(f"  groups    {n_var} variables in {n_grp} groups (from groups.py)")
    print(f"  k values  {KS}")

    combos = COMBOS
    if a.combos:
        want = set(a.combos)
        combos = [(arm, lv) for arm, lv in COMBOS
                  if combo_id(arm, lv) in want or f"{arm}@{lv:g}" in want]
        if not combos:
            sys.exit(f"no combo matched {sorted(want)}; "
                     f"available: {[combo_id(*c) for c in COMBOS]}")

    if a.stage in ("1", "all"):
        stage1(X, y, cols, out, a.repeats, a.perm_repeats, combos)
    if a.stage in ("2", "all"):
        stage2(X, y, cols, out, a.perm_repeats)


if __name__ == "__main__":
    main()
