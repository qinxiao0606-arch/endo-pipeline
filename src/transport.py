"""
Apply the reduced 9-variable model, unchanged, to our own questionnaire responses.

Julia, 2026-09-16 (verbatim, PROGRESS §8):

  "The main thing is to transport performance to our data using your 9 theme
   representation and same model ."

Her earlier instruction fixed the order (PROGRESS §1 A3): train on all 58
Goldstein variables, retrain on the 9 shared ones to price the reduction, then
apply THAT model to our data. reduction_cost.py did the first two and saved the
fitted model. This script is the third.

Why the diagnostic comes first
------------------------------
A transported AUC on its own is uninterpretable here. If it comes out low there
are two completely different explanations and the number cannot tell them apart:

  (a) the model does not transport - different population, different measurement
  (b) our controls are not controls

(b) is a live risk, not a hypothetical. Our controls were recruited through
patient-association channels and through personal networks, and the form has no
"I don't know" option for the diagnosis question, so anyone unsure had to pick a
side. Validation rule C-12 (pre-existing, not invented for this script) flags
controls reporting BOTH severe dysmenorrhea AND severe pelvic pain. Those people
are, on the face of it, exactly the undiagnosed patients this whole project has
spent the year simulating.

So section 1 measures how case-like our controls are, per channel, before any
model is applied. Only then does the transported number mean something.

The de-contamination counterfactual
-----------------------------------
Section 4 is the one thing our own data can do that no simulation can. Every
contamination experiment so far took clean labels and damaged them on purpose.
Here the damage is already in the data and was not put there by us, so we can run
the experiment in the other direction: drop the C-12-flagged controls and see
whether performance moves the way the simulations predicted it would. Agreement
is evidence the simulated mechanism is real outside the simulation. This is a
diagnostic, not a corrected result: C-12 is a conservative proxy, dropping rows
on a symptom criterion is itself a selection, and the headline number stays the
one computed on all rows.

Binarisation
------------
Their items are yes/no, ours are graded, so the mapping needs cut-points. The
cuts come from groups.ALIGNMENT (field `rule`), pre-registered in A0 before any
of this was run. Section 3 varies them, because a cut-point is a free parameter
and PROGRESS §10 material L committed us to showing how much it matters.

Clean-channel controls (section 6, added 2026-09-28)
---------------------------------------------------
Sections 1-5 show the transported number is dominated by who our controls are:
of 236 controls, only the 30 from the friends/family link (597) and the English
link (178) report symptoms at anything like a control level. Section 6 re-runs
the transport with ALL cases against ONLY those controls, and is written to be
re-run after every new export, because recruitment on those two links is still
going on and the answer is expected to firm up as they grow.

Three things make the small number usable rather than misleading:

  1. every AUC carries a stratified-bootstrap 95% CI (DeLong as a cross-check),
     and every sensitivity / specificity a Wilson 95% interval
  2. the run is labelled FIRST LOOK until the clean controls reach the B3
     target (PROGRESS §2), and a precision table shows how wide the interval
     would be at larger n, so nobody has to guess when it stops being a glance
  3. each run appends one row to transport_clean_history.csv, keyed on the data
     snapshot, so successive exports can be compared without re-running old data

Resubmissions (added 2026-10-05). Before anything is computed, a row is dropped
when an earlier row from the SAME link has exactly the same answers on every
questionnaire item and came in at most RESUBMIT_WINDOW_MIN minutes before it; the
earliest one is kept. Every dropped sid is printed and written to
transport_dropped_resubmissions.csv. --keep-resubmissions reproduces the older
numbers. TODO: this rule belongs in the dictionary's Regles_controle sheet.

Channels whose link was distributed outside endometriosis communities but whose
membership is not yet confirmed are CANDIDATES (CANDIDATE_CHANNELS). They are
scored in rows of their own and never enter the headline (added 2026-10-05).
The first candidate, 497 (social), was confirmed the same day and moved into
CLEAN_CHANNELS. The headline therefore changes definition from the 10-05
snapshot on; the 09-28 two-channel list is still scored in rows marked
[continuity], and the history file records which list each row used.

What it can NOT fix, and prints every time: cases still come mostly from
community channels and controls do not. Part of any gap between them is
channel, not disease. The channel-matched comparison is shown next to the
headline so the size of that problem stays visible.

    cd src && python3 transport.py
    cd src && python3 transport.py --model ../out/reduced_model.pkl
    cd src && python3 transport.py --boot 5000      # more bootstrap draws
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

from config import load_schema
from contamination_controlled import load
from groups import ALIGNMENT, SHARED_SUBSET
from ingest import ingest
from validate import run_all

warnings.filterwarnings("ignore")

# cut-points, pre-registered in A0 (groups.ALIGNMENT `rule`), plus the
# alternatives section 3 sweeps. "strict" = one notch harder, "loose" = one
# notch easier, wherever the scale allows it.
CUTS = {
    "Dysmenorrhee":       {"preregistered": ["Modérée", "Sévère"],
                           "strict": ["Sévère"],
                           "loose": ["Légère", "Modérée", "Sévère"]},
    "Douleur_pelvienne":  {"preregistered": ["Modérée", "Sévère"],
                           "strict": ["Sévère"],
                           "loose": ["Légère", "Modérée", "Sévère"]},
    "Dyspareunie":        {"preregistered": ["Oui"], "strict": ["Oui"], "loose": ["Oui"]},
    "Dyschesie":          {"preregistered": ["Oui"], "strict": ["Oui"], "loose": ["Oui"]},
    "Symptomes_cycliques": {"preregistered": ["Oui"], "strict": ["Oui"], "loose": ["Oui"]},
    "Abondance_regles":   {"preregistered": ["Abondante"],
                           "strict": ["Abondante"],
                           "loose": ["Normale", "Abondante"]},
    "Duree_regles":       {"preregistered": [">7 jours"],
                           "strict": [">7 jours"],
                           "loose": ["4-7 jours", ">7 jours"]},
    "Regularite_cycle":   {"preregistered": ["Irrégulier"],
                           "strict": ["Irrégulier"], "loose": ["Irrégulier"]},
    "Infertilite":        {"preregistered": ["Oui"], "strict": ["Oui"], "loose": ["Oui"]},
}
# ours -> the Goldstein column the saved model expects, in the model's order
OURS_BY_THEME = {t: v["ours"] for t, v in ALIGNMENT.items()}
# labels that mean "no answer", not "no symptom"
MISSING = {"Non renseigné", "Non concernée", "Inconnu"}
SEED = 42
GS_BY_OURS = {v["ours"]: v["rep"] for v in ALIGNMENT.values()}

# ---- section 6 settings --------------------------------------------------
# The two links whose controls behave like controls (PROGRESS §2 B3, 2026-09-16).
# A priori reason, independent of the data: neither link was distributed through
# endometriosis communities. 597 went to friends, family, forwards and mailing
# lists; 178 (English) went to non-French speakers among family, friends,
# colleagues and people in the same group (confirmed by Xiao, 2026-09-28).
# ⚠ The list was nonetheless picked after looking at how many of the nine themes
#   the controls report, i.e. on the same items the model reads. On the
#   421-response snapshot it is therefore partly post hoc. Two rules keep it
#   honest from here on:
#   (a) the list is FROZEN as of 2026-09-28. A new channel joins it because of
#       how its link was distributed, never because of what its controls report.
#   (b) controls submitted after SELECTION_CUTOFF were not seen when the list was
#       chosen, so they are scored separately as an out-of-sample check.
# TODO: PROGRESS §5 rule 6 says channel names belong in the dictionary. Move this
#       to a column of the Formulaires sheet at the next dictionary revision.
# 2026-10-05: "social" (497) added under rule (a) and the candidate rule below.
# Xiao, 10-05: the link went to student groups and student social media; the
# people posting it were told to avoid endometriosis keywords and never to put
# it in endometriosis-related groups. That is the distributors' instruction,
# not a check of every post, and it is recorded as such.
CLEAN_CHANNELS = ("autre", "communaute_intl", "social")
# The list as frozen on 2026-09-28, still scored so the series stays comparable.
CLEAN_CHANNELS_0928 = ("autre", "communaute_intl")
# Candidate channels: links that, by how they were DISTRIBUTED, look like clean
# channels, but whose membership is not confirmed yet. Added 2026-10-05, before
# any of their responses were analysed (497 had 0 responses until then).
# 497 = social media of a student association (dictionary, Formulaires sheet:
# "Réseaux sociaux (association étudiante)"). Rule, fixed in advance: it joins
# CLEAN_CHANNELS if and only if Xiao confirms the link was not posted in
# endometriosis-focused groups or pages. What its controls report plays no part.
# Until then it is scored only in its own rows and never enters the headline.
# 497 was confirmed on 2026-10-05 (see CLEAN_CHANNELS); no candidates remain.
CANDIDATE_CHANNELS = ()
SELECTION_CUTOFF = pd.Timestamp("2026-09-16 17:16")   # last submission in the 421 snapshot
FIRST_LOOK_TARGET = 150    # B3 target for clean controls, set in PROGRESS §2 before this ran
MIN_SCORED = 10            # a comparison with fewer cases OR controls is counted, not scored
PROJECT_N0 = (30, 60, 100, 150, 300)
RESUBMIT_WINDOW_MIN = 10  # identical answers, same link, this close in time = one person twice


def hr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def wilson(k, n, z=1.96):
    """95% interval for a proportion k/n. Unlike p ± 1.96·SE it stays inside
    [0, 1] and does not collapse to zero width when k = 0 or k = n."""
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def auc_boot(y, p, n_boot, seed):
    """Stratified bootstrap: redraw the cases from the cases and the controls from
    the controls, each with replacement and at their own size, then recompute AUC.
    Stratified so every redraw keeps the observed group sizes (185 cases and 30
    controls on the 09-16 data): an unstratified redraw would let the small group
    shrink or grow, which is not the uncertainty we actually have."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        draws[b] = roc_auc_score(y[i], p[i])
    return np.percentile(draws, [2.5, 97.5])


def auc_delong(y, p):
    """AUC and its DeLong standard error, as a cross-check on the bootstrap.
    AUC = share of (case, control) pairs where the case scores higher, ties 1/2.
    Its variance comes from how much each case's, and each control's, own share
    of won pairs varies."""
    pos, neg = p[y == 1], p[y == 0]
    won = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    auc = won.mean()
    se = np.sqrt(won.mean(axis=1).var(ddof=1) / len(pos)
                 + won.mean(axis=0).var(ddof=1) / len(neg))
    return auc, se


def auc_var_parts(y, p):
    """The two halves of the DeLong variance, kept apart: the part that shrinks
    when more CASES arrive and the part that shrinks when more CONTROLS arrive.
    Used to project precision at larger n, never as a result."""
    pos, neg = p[y == 1], p[y == 0]
    won = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    return won.mean(axis=1).var(ddof=1), won.mean(axis=0).var(ddof=1)


def resubmission_mask(df, cols, window_min=RESUBMIT_WINDOW_MIN):
    """Which rows to keep (True) once same-link resubmissions are removed.

    A row is dropped when an EARLIER row has exactly the same answers on every
    column in `cols` (Source_canal is one of them, so both came through the same
    link) and was submitted at most `window_min` minutes before it. Rows are
    visited by submission time, then sid, so the earliest copy is the one kept.

    Why a time window: two different people can genuinely give identical answers,
    low-symptom controls especially. Identical answers on every item, on the same
    link, a few minutes apart, are far more likely one person sending the form
    twice (or a test). Without the window the rule would delete real people.
    Returns (keep as a boolean array, sid of the kept copy for each dropped row).
    """
    key = df[cols].astype(str)
    when = pd.to_datetime(df["_date"])
    order = (pd.DataFrame({"t": when, "s": pd.to_numeric(df["_sid"], errors="coerce")})
             .sort_values(["t", "s"]).index)
    keep = pd.Series(True, index=df.index)
    kept_of = pd.Series("", index=df.index, dtype=object)
    last_kept = {}
    for i in order:
        k = tuple(key.loc[i])
        t = when.loc[i]
        if k in last_kept:
            t0, sid0 = last_kept[k]
            if pd.notna(t) and pd.notna(t0) and t - t0 <= pd.Timedelta(minutes=window_min):
                keep.loc[i] = False
                kept_of.loc[i] = sid0
                continue
        last_kept[k] = (t, df.at[i, "_sid"])
    return keep.to_numpy(), kept_of


def fit_eight(bundle, goldstein_path):
    """The 8-variable variant (fertility dropped), refitted on Goldstein with the
    same protocol as the 9-variable model. Shared by sections 5 and 6."""
    XG, yG, _ = load(goldstein_path)
    eight = [c for c in bundle["features"] if c != GS_BY_OURS["Infertilite"]]
    rateG = float(yG.mean())
    fitted = {}
    for tag, mdl in (("ada", AdaBoostClassifier(random_state=SEED)),
                     ("lr", LogisticRegression(max_iter=2000))):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        pg = cross_val_predict(mdl, XG[eight], yG, cv=cv, method="predict_proba")[:, 1]
        k = max(1, int(round(rateG * len(pg))))
        thr8 = float(np.sort(pg)[::-1][min(k, len(pg) - 1)])
        mdl.fit(XG[eight], yG)
        fitted[tag] = {"model": mdl, "threshold": thr8}
    fitted["features"] = eight
    return fitted


def binarise(df, scheme, missing="zero"):
    """Our graded answers -> the binary columns the saved model expects.

    Non-response becomes 0, i.e. "symptom not reported". That is the only choice
    consistent with how the Goldstein data was built (absence of a tick is a 0
    there too), but it is a real assumption and it matters most for Infertilite,
    which C-11 flags at ~26% non-response.
    """
    out = {}
    for ours, opts in CUTS.items():
        pos = df[ours].isin(opts[scheme])
        if missing == "one":
            pos = pos | df[ours].isin(MISSING)
        out[GS_BY_OURS[ours]] = pos.astype(int)
    return pd.DataFrame(out)[[GS_BY_OURS[o] for o in CUTS]]


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    return (roc_auc_score(y, p),
            float(pred[y == 1].mean()),
            float(1.0 - pred[y == 0].mean()))


def clean_channel_section(df, y, suspect, bundle, eight_fit, goldstein_path, out,
                          n_boot):
    """Section 6: all cases against the clean-channel controls only."""
    hr("6. ALL CASES vs CLEAN-CHANNEL CONTROLS ONLY  (re-run after every export)")
    ch = df["Source_canal"].to_numpy()
    when = pd.to_datetime(df["_date"])
    is_case, is_ctrl = y == 1, y == 0
    clean = np.isin(ch, CLEAN_CHANNELS)
    cand = np.isin(ch, CANDIDATE_CHANNELS)
    clean0928 = np.isin(ch, CLEAN_CHANNELS_0928)
    community = ~clean & ~cand          # endometriosis-community and untracked links
    new = (when > SELECTION_CUTOFF).fillna(False).to_numpy()
    snapshot = when.max()
    n_clean = int((is_ctrl & clean).sum())
    status = "FIRST LOOK" if n_clean < FIRST_LOOK_TARGET else "RESULT"

    print(f"  data snapshot     latest submission {snapshot:%Y-%m-%d %H:%M}, {len(y)} rows")
    print(f"  clean channels    {', '.join(CLEAN_CHANNELS)}  (frozen list, see top of file)")
    if CANDIDATE_CHANNELS:
        print(f"  candidate         {', '.join(CANDIDATE_CHANNELS)}  "
              f"({int((is_ctrl & cand).sum())} controls; own rows only, pending confirmation)")
    if set(CLEAN_CHANNELS) != set(CLEAN_CHANNELS_0928):
        print(f"  09-28 list        {', '.join(CLEAN_CHANNELS_0928)}  "
              f"({int((is_ctrl & clean0928).sum())} controls; rows marked [continuity])")
    print(f"  clean controls    {n_clean}  (target {FIRST_LOOK_TARGET})   ->  status: {status}")
    n28 = int((is_ctrl & clean0928).sum())
    if status == "RESULT" and n28 < FIRST_LOOK_TARGET:
        print(f"  ⚠ The target is reached only with the channels added after 09-28; the")
        print(f"    09-28 list alone has {n28} controls and would still be a FIRST LOOK.")
    if status == "FIRST LOOK":
        print(f"  ⚠ Too few controls to be a result. Everything below is a first look and")
        print(f"    must be quoted with its interval, never as a bare number.")

    # ---- 6a. who ends up in the comparison -------------------------------
    print("\n  6a. who is in it  (mean_themes = how many of the 9 themes reported, 0-9)")
    X = binarise(df, "preregistered")[bundle["features"]]
    n_pos = X.sum(axis=1).to_numpy()
    who = []
    for c in sorted(set(ch)):
        m = ch == c
        last = when[m].max()
        who.append({"channel": c, "clean": c in CLEAN_CHANNELS,
                    "candidate": c in CANDIDATE_CHANNELS,
                    "latest_submission": (f"{last:%Y-%m-%d}" if pd.notna(last) else ""),
                    "n_cases": int((m & is_case).sum()),
                    "n_controls": int((m & is_ctrl).sum()),
                    "controls_after_cutoff": int((m & is_ctrl & new).sum()),
                    "controls_C12": int((m & is_ctrl & suspect).sum()),
                    "mean_themes_cases": (float(n_pos[m & is_case].mean())
                                          if (m & is_case).any() else np.nan),
                    "mean_themes_controls": (float(n_pos[m & is_ctrl].mean())
                                             if (m & is_ctrl).any() else np.nan)})
    XG, yG, _ = load(goldstein_path)
    tG = XG[bundle["features"]].sum(axis=1).to_numpy()
    who = pd.DataFrame(who).sort_values(["clean", "candidate", "n_controls"],
                                         ascending=[False, False, False])
    who = pd.concat([who, pd.DataFrame([{
        "channel": "GOLDSTEIN (training data)", "clean": np.nan, "candidate": np.nan,
        "n_cases": int((yG == 1).sum()), "n_controls": int((yG == 0).sum()),
        "mean_themes_cases": float(tG[yG == 1].mean()),
        "mean_themes_controls": float(tG[yG == 0].mean())}])], ignore_index=True)
    for c in ("n_cases", "n_controls", "controls_after_cutoff", "controls_C12"):
        who[c] = who[c].astype("Int64")
    print(who.round(2).to_string(index=False))
    who.insert(0, "snapshot", f"{snapshot:%Y-%m-%d %H:%M}")
    who.to_csv(out / "transport_clean_composition.csv", index=False)
    gap_gs = float(tG[yG == 1].mean() - tG[yG == 0].mean())
    mc = n_pos[is_case & community].mean()
    mcc = n_pos[is_case & clean].mean() if (is_case & clean).any() else np.nan
    print(f"\n   Cases from community/unknown channels report {mc:.2f} themes on average,")
    print(f"   cases from the clean channels {mcc:.2f}. The headline below pairs mostly the")
    print(f"   former with controls from the latter, so part of any separation is CHANNEL,")
    print(f"   not disease (two-gate design, known to overstate accuracy). The")
    print(f"   channel-matched row is the check on this; it is scored once both groups")
    print(f"   reach {MIN_SCORED}.")
    print("   latest_submission is per channel. If it differs a lot between channels,")
    print("   the links were probably exported on different days: export all seven")
    print("   together, or the 'snapshot' below mixes dates.")

    # ---- 6b. the comparisons ---------------------------------------------
    comps = [("all controls (= section 2)", is_case | is_ctrl),
             ("clean-channel controls  [HEADLINE]", is_case | (is_ctrl & clean)),
             ("community/unknown-channel controls", is_case | (is_ctrl & community))]
    for c in CLEAN_CHANNELS:
        comps.append((f"clean, {c} controls only", is_case | (is_ctrl & (ch == c))))
    for c in CANDIDATE_CHANNELS:
        comps.append((f"candidate, {c} controls only  [pending]",
                      is_case | (is_ctrl & (ch == c))))
    if CANDIDATE_CHANNELS:
        comps.append(("clean + candidate controls  [pending]",
                      is_case | (is_ctrl & (clean | cand))))
    comps += [("channel-matched: clean cases vs clean controls", clean),
              ("clean controls submitted after channel choice",
               is_case | (is_ctrl & clean & new))]
    if set(CLEAN_CHANNELS) != set(CLEAN_CHANNELS_0928):
        comps += [("09-28 list controls  [continuity]", is_case | (is_ctrl & clean0928)),
                  ("09-28 list, channel-matched  [continuity]", clean0928),
                  ("09-28 list, controls after channel choice  [continuity]",
                   is_case | (is_ctrl & clean0928 & new))]

    rows = []
    for label, m in comps:
        ym = y[m]
        n1, n0 = int(ym.sum()), int((ym == 0).sum())
        for tag in ("ada", "lr"):
            r = {"comparison": label, "model": tag, "n_cases": n1, "n_controls": n0,
                 "theme_gap": (float(n_pos[m & is_case].mean() - n_pos[m & is_ctrl].mean())
                               if n1 and n0 else np.nan)}
            if min(n1, n0) < MIN_SCORED:
                r["note"] = f"not scored: fewer than {MIN_SCORED} in a group"
                rows.append(r)
                continue
            mdl, thr = bundle[tag]["model"], bundle[tag]["threshold"]
            p = mdl.predict_proba(X[m])[:, 1]
            auc, sens, spec = metrics(ym, p, thr)
            lo, hi = auc_boot(ym, p, n_boot, SEED)
            _, se = auc_delong(ym, p)
            k_sens = int(round(sens * n1))
            k_spec = int(round(spec * n0))
            r.update({"AUC": auc, "AUC_boot_lo": lo, "AUC_boot_hi": hi,
                      "AUC_delong_lo": max(0.0, auc - 1.96 * se),
                      "AUC_delong_hi": min(1.0, auc + 1.96 * se),
                      "sens": sens, "sens_lo": wilson(k_sens, n1)[0],
                      "sens_hi": wilson(k_sens, n1)[1],
                      "spec": spec, "spec_lo": wilson(k_spec, n0)[0],
                      "spec_hi": wilson(k_spec, n0)[1],
                      "threshold": thr, "note": ""})
            rows.append(r)
    res = pd.DataFrame(rows)

    print("\n  6b. transport, Goldstein cut-off carried over unchanged")
    for _, r in res[res.model == "ada"].iterrows():
        print(f"\n   {r.comparison}")
        gap = f"   theme gap {r.theme_gap:.2f}" if pd.notna(r.theme_gap) else ""
        print(f"     {r.n_cases} cases / {r.n_controls} controls{gap}")
        if isinstance(r.note, str) and r.note:
            print(f"     {r.note}")
            continue
        print(f"     AUC  {r.AUC:.3f}   95% CI {r.AUC_boot_lo:.3f}-{r.AUC_boot_hi:.3f} "
              f"(bootstrap)   {r.AUC_delong_lo:.3f}-{r.AUC_delong_hi:.3f} (DeLong)")
        print(f"     sens {r.sens:.3f}   95% CI {r.sens_lo:.3f}-{r.sens_hi:.3f}")
        print(f"     spec {r.spec:.3f}   95% CI {r.spec_lo:.3f}-{r.spec_hi:.3f}")
    print(f"\n   theme gap = cases' mean themes minus controls' mean themes in that row.")
    print(f"   Goldstein's own gap is {gap_gs:.2f}. A model trained on one gap scores well")
    print(f"   wherever a similar gap is recreated, whatever the reason for the gap.")
    print("   (ada shown; lr in the csv.) Sensitivity is the same on the first rows BY")
    print("   CONSTRUCTION: same cases, same cut-off. Only AUC and specificity carry news.")
    print("   The cut-off is not re-derived here: in a case-control subset the 'prevalence'")
    print("   is whatever ratio we sampled, so a prevalence-matched cut-off means nothing.")

    # ---- 6c. is clean vs community a real difference? --------------------
    print("\n  6c. clean-channel minus community-channel controls, same cases (bootstrap)")
    rng = np.random.default_rng(SEED)
    pos = np.flatnonzero(is_case)
    neg_c, neg_d = np.flatnonzero(is_ctrl & clean), np.flatnonzero(is_ctrl & community)
    diff_rows = []
    if min(len(neg_c), len(neg_d)) >= MIN_SCORED:
        for tag in ("ada", "lr"):
            p = bundle[tag]["model"].predict_proba(X)[:, 1]
            d = np.empty(n_boot)
            for b in range(n_boot):
                i1 = rng.choice(pos, len(pos))
                ic, idd = rng.choice(neg_c, len(neg_c)), rng.choice(neg_d, len(neg_d))
                a_c = roc_auc_score(np.r_[np.ones(len(i1)), np.zeros(len(ic))],
                                    np.r_[p[i1], p[ic]])
                a_d = roc_auc_score(np.r_[np.ones(len(i1)), np.zeros(len(idd))],
                                    np.r_[p[i1], p[idd]])
                d[b] = a_c - a_d
            a_c = roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg_c))],
                                np.r_[p[pos], p[neg_c]])
            a_d = roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg_d))],
                                np.r_[p[pos], p[neg_d]])
            lo, hi = np.percentile(d, [2.5, 97.5])
            diff_rows.append({"comparison": "difference: clean minus community controls",
                              "model": tag, "n_cases": len(pos),
                              "n_controls": len(neg_c) + len(neg_d),
                              "AUC": a_c - a_d, "AUC_boot_lo": lo, "AUC_boot_hi": hi,
                              "note": "AUC difference, cases shared"})
            print(f"     {tag:<4} {a_c - a_d:+.3f}   95% CI {lo:+.3f} to {hi:+.3f}")
        print("   An interval that excludes 0 means the two control groups really do")
        print("   behave differently under the same model, not just by sampling luck.")
    else:
        print(f"     not scored: fewer than {MIN_SCORED} controls on one side")
    res = pd.concat([res, pd.DataFrame(diff_rows)], ignore_index=True)
    res.insert(0, "snapshot", f"{snapshot:%Y-%m-%d %H:%M}")
    res.to_csv(out / "transport_clean_controls.csv", index=False)

    # ---- 6d. analytic choices on the headline comparison -----------------
    print("\n  6d. headline under the other analytic choices (sections 3 and 5 repeated)")
    head = is_case | (is_ctrl & clean)
    complete = ~df[[v["ours"] for v in ALIGNMENT.values()]].isin(MISSING).any(axis=1)
    complete = complete.to_numpy()
    variants = [("cuts preregistered, missing as 0 (current)", "preregistered", "zero",
                 head, bundle),
                ("cuts strict, missing as 0", "strict", "zero", head, bundle),
                ("cuts loose, missing as 0", "loose", "zero", head, bundle),
                ("missing as 1", "preregistered", "one", head, bundle),
                ("complete cases only", "preregistered", "zero", head & complete, bundle),
                ("8 variables, fertility dropped", "preregistered", "zero", head, eight_fit)]
    ch_rows = []
    for label, scheme, miss, m, mb in variants:
        Xv = binarise(df, scheme, missing=miss)[mb["features"]][m]
        ym = y[m]
        for tag in ("ada", "lr"):
            p = mb[tag]["model"].predict_proba(Xv)[:, 1]
            auc, se_, sp = metrics(ym, p, mb[tag]["threshold"])
            ch_rows.append({"snapshot": f"{snapshot:%Y-%m-%d %H:%M}", "choice": label,
                            "model": tag, "n_cases": int(ym.sum()),
                            "n_controls": int((ym == 0).sum()),
                            "AUC": auc, "sens": se_, "spec": sp})
    chs = pd.DataFrame(ch_rows)
    chs.to_csv(out / "transport_clean_choices.csv", index=False)
    print(chs[chs.model == "ada"][["choice", "n_cases", "n_controls", "AUC", "spec"]]
          .round(3).to_string(index=False))
    miss_rows = chs[(chs.model == "ada") & ~chs.choice.str.startswith("cuts s")
                    & ~chs.choice.str.startswith("cuts l")]
    cut_rows = chs[(chs.model == "ada") & chs.choice.str.startswith("cuts")]
    print(f"\n   AUC range over the four missingness treatments  "
          f"{miss_rows.AUC.min():.3f}-{miss_rows.AUC.max():.3f}")
    print(f"   AUC range over the three cut-point schemes       "
          f"{cut_rows.AUC.min():.3f}-{cut_rows.AUC.max():.3f}")
    print("   These ranges are a different kind of uncertainty from the CI: the CI is")
    print(f"   'which {int((is_ctrl & clean).sum())} controls we happened to get', the range is 'which defensible")
    print("   analysis choice we made'. Report both.")

    # ---- 6e. how precise is it now, how precise would it be --------------
    print("\n  6e. precision now and at larger n  (keeps today's spread per person;")
    print("      assumes new controls resemble today's, which is exactly what is unknown)")
    hl = res[(res.comparison.str.contains("HEADLINE")) & (res.model == "ada")].iloc[0]
    if pd.notna(hl.get("AUC")):
        p_h = bundle["ada"]["model"].predict_proba(X[head])[:, 1]
        v_case, v_ctrl = auc_var_parts(y[head], p_h)
        n1 = int(hl.n_cases)
        share = (v_ctrl / hl.n_controls) / (v_case / n1 + v_ctrl / hl.n_controls)
        prec = []
        for n0 in sorted(set(PROJECT_N0) | {int(hl.n_controls)}):
            k = int(round(hl.spec * n0))
            wl, wh = wilson(k, n0)
            prec.append({"n_controls": n0, "n_cases": n1,
                         "AUC_ci_halfwidth": 1.96 * np.sqrt(v_case / n1 + v_ctrl / n0),
                         "spec_ci_width": wh - wl, "is_now": n0 == int(hl.n_controls)})
        prec = pd.DataFrame(prec)
        prec.to_csv(out / "transport_clean_precision.csv", index=False)
        print(prec.round(3).to_string(index=False))
        floor = 1.96 * np.sqrt(v_case / n1)
        print(f"   Controls carry {share:.0%} of today's AUC variance. More controls help,")
        print(f"   but with these {n1} cases the AUC interval cannot get narrower than")
        print(f"   about ±{floor:.3f}. Specificity depends on the controls alone, so that is")
        print(f"   the number that gains most from recruitment.")
        print("   ⚠ An interval only covers 'which of these controls we happened to draw'.")
        print("     It cannot cover 'friends, English speakers and students are not")
        print("     the population'. That limitation is bigger than any interval here.")

    # ---- 6f. history, one row per data snapshot --------------------------
    hist_path = out / "transport_clean_history.csv"
    new_ctrl = is_ctrl & clean & new
    ext_rows = res[res.comparison.str.startswith("clean + candidate") & (res.model == "ada")]
    ext = ext_rows.iloc[0] if len(ext_rows) else pd.Series(dtype=object)
    c28_rows = res[res.comparison.str.startswith("09-28 list controls") & (res.model == "ada")]
    c28 = c28_rows.iloc[0] if len(c28_rows) else hl     # same list -> headline is the series
    row = {"snapshot": f"{snapshot:%Y-%m-%d %H:%M}", "n_rows": len(y),
           "clean_channels": "+".join(CLEAN_CHANNELS),
           "n_cases": int(is_case.sum()), "n_clean_controls": n_clean,
           **{f"n_controls_{c}": int((is_ctrl & (ch == c)).sum())
              for c in dict.fromkeys(CLEAN_CHANNELS + CANDIDATE_CHANNELS
                                     + CLEAN_CHANNELS_0928)},
           "n_clean_controls_after_cutoff": int(new_ctrl.sum()),
           "AUC_ada": hl.get("AUC"), "AUC_ada_lo": hl.get("AUC_boot_lo"),
           "AUC_ada_hi": hl.get("AUC_boot_hi"),
           "spec_ada": hl.get("spec"), "spec_ada_lo": hl.get("spec_lo"),
           "spec_ada_hi": hl.get("spec_hi"),
           "AUC_ada_range_missingness_lo": float(miss_rows.AUC.min()),
           "AUC_ada_range_missingness_hi": float(miss_rows.AUC.max()),
           "AUC_ada_0928list": c28.get("AUC"), "AUC_ada_0928list_lo": c28.get("AUC_boot_lo"),
           "AUC_ada_0928list_hi": c28.get("AUC_boot_hi"),
           "AUC_ada_with_candidates": ext.get("AUC"),
           "AUC_ada_with_candidates_lo": ext.get("AUC_boot_lo"),
           "AUC_ada_with_candidates_hi": ext.get("AUC_boot_hi"),
           "latest_by_channel": "; ".join(f"{r.channel}:{r.latest_submission}"
                                          for r in who.itertuples()
                                          if isinstance(r.latest_submission, str)
                                          and r.latest_submission),
           "status": status, "run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
    hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    if len(hist) and not {"snapshot", "n_rows"} <= set(hist.columns):
        old = hist_path.with_name("transport_clean_history_oldlayout.csv")
        hist_path.rename(old)
        print(f"  history file had an older column layout; moved to {old.name}")
        hist = pd.DataFrame()
    if len(hist):
        hist = hist[~((hist["snapshot"] == row["snapshot"])
                      & (hist["n_rows"] == row["n_rows"]))]
        # rows written before 10-05 used the 09-28 list for their headline
        if "clean_channels" not in hist.columns:
            hist["clean_channels"] = np.nan
        hist["clean_channels"] = hist["clean_channels"].astype(object)
        old_def = hist["clean_channels"].isna()
        hist.loc[old_def, "clean_channels"] = "+".join(CLEAN_CHANNELS_0928)
        for k in ("", "_lo", "_hi"):
            if f"AUC_ada_0928list{k}" not in hist.columns:
                hist[f"AUC_ada_0928list{k}"] = np.nan
            fill = old_def & hist[f"AUC_ada_0928list{k}"].isna()
            hist.loc[fill, f"AUC_ada_0928list{k}"] = hist.loc[fill, f"AUC_ada{k}"]
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(hist_path, index=False)
    print(f"\n  6f. history: {len(hist)} snapshot(s) in {hist_path.name} "
          f"(same snapshot re-run replaces its row)")
    print(hist[["snapshot", "clean_channels", "n_clean_controls",
                "n_clean_controls_after_cutoff", "AUC_ada", "AUC_ada_lo", "AUC_ada_hi",
                "spec_ada", "AUC_ada_0928list", "status"]]
          .round(3).to_string(index=False))
    print("   AUC_ada follows the clean list in force at that snapshot (clean_channels);")
    print("   AUC_ada_0928list keeps the 09-28 two-channel list, for a like-for-like series.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../out/reduced_model.pkl")
    ap.add_argument("--out", default="../out")
    ap.add_argument("--goldstein", default="../data/external/Dataset.xlsx")
    ap.add_argument("--boot", type=int, default=2000,
                    help="bootstrap draws for the section 6 AUC intervals")
    ap.add_argument("--keep-resubmissions", action="store_true",
                    help="do not drop same-link resubmissions (reproduces pre-10-05 numbers)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    schema = load_schema()
    raw = ingest(schema=schema)
    df, rep = run_all(raw, schema, strict=False)
    suspect = np.asarray(rep.flags["flag_temoin_suspect"], dtype=bool)

    hr(f"0. SAME-LINK RESUBMISSIONS  (identical answers, <= {RESUBMIT_WINDOW_MIN} min apart)")
    cols = [c for c in schema.data_vars if c in df.columns]
    keep, kept_of = resubmission_mask(df, cols)
    dropped = df.loc[~keep, ["_sid", "_date", "Source_canal", "Groupe"]].copy()
    dropped["kept_sid"] = kept_of[~keep].to_numpy()
    dropped.to_csv(out / "transport_dropped_resubmissions.csv", index=False)
    if len(dropped):
        print(dropped.to_string(index=False))
    print(f"\n  {len(dropped)} row(s) are later copies of an earlier submission.")
    if a.keep_resubmissions:
        print("  --keep-resubmissions: kept anyway, numbers match the pre-10-05 runs.")
    else:
        df = df[keep].reset_index(drop=True)
        suspect = suspect[keep]
        print("  Dropped before any analysis; the earliest copy of each is kept.")
    y = (df["Groupe"] == "Cas").astype(int).to_numpy()

    hr("our data after ingestion and C-13 normalisation")
    print(f"  {len(df)} responses | {int(y.sum())} cases / {int((y == 0).sum())} controls "
          f"| prevalence {y.mean():.3f}")
    print(f"  C-12 flags {int(suspect.sum())} controls reporting severe dysmenorrhea AND "
          f"severe pelvic pain")
    print(f"  = {suspect.sum() / max(1, (y == 0).sum()):.1%} of our control group")

    # ---- 1. is our control group a control group? ------------------------
    hr("1. HOW CASE-LIKE ARE OUR CONTROLS  (before any model is applied)")
    X = binarise(df, "preregistered")
    n_pos = X.sum(axis=1)
    rows = []
    for ch, g in df.groupby("Source_canal"):
        idx = g.index
        m = df.index.isin(idx)
        cas, tem = m & (y == 1), m & (y == 0)
        rows.append({
            "channel": ch, "n": int(m.sum()),
            "n_cases": int(cas.sum()), "n_controls": int(tem.sum()),
            "mean_themes_cases": float(n_pos[cas].mean()) if cas.sum() else np.nan,
            "mean_themes_controls": float(n_pos[tem].mean()) if tem.sum() else np.nan,
            "pct_controls_flagged_C12": (float(suspect[tem].mean())
                                         if tem.sum() else np.nan)})
    ch = pd.DataFrame(rows).sort_values("n", ascending=False)
    ch["gap"] = ch["mean_themes_cases"] - ch["mean_themes_controls"]
    ch.to_csv(out / "transport_channels.csv", index=False)
    print(ch.round(3).to_string(index=False))
    print("\n   mean_themes_* = average number of the 9 themes reported positive (0-9).")
    print("   gap = how far apart cases and controls are. A small gap means the")
    print("   control group is not doing its job, whatever the model then says.")

    # ---- 2. transport ----------------------------------------------------
    hr("2. TRANSPORT — the saved model applied unchanged")
    with open(a.model, "rb") as f:
        bundle = pickle.load(f)
    X = X[bundle["features"]]
    tr_rows = []
    for tag in ("ada", "lr"):
        mdl, thr = bundle[tag]["model"], bundle[tag]["threshold"]
        p = mdl.predict_proba(X)[:, 1]
        auc, sens, spec = metrics(y, p, thr)
        tr_rows.append({"model": tag, "set": "ours_all", "n": len(y),
                        "AUC": auc, "sens": sens, "spec": spec,
                        "threshold": thr, "threshold_source": "goldstein_prevalence"})
        # what the cut-off would be if re-derived on our prevalence
        k = max(1, int(round(y.mean() * len(p))))
        thr2 = float(np.sort(p)[::-1][min(k, len(p) - 1)])
        auc2, sens2, spec2 = metrics(y, p, thr2)
        tr_rows.append({"model": tag, "set": "ours_all", "n": len(y),
                        "AUC": auc2, "sens": sens2, "spec": spec2,
                        "threshold": thr2, "threshold_source": "our_prevalence"})
    tr = pd.DataFrame(tr_rows)
    print(tr.round(4).to_string(index=False))
    print(f"\n   Goldstein prevalence {bundle['goldstein_prevalence']:.3f} vs ours {y.mean():.3f}.")
    print("   AUC is threshold-free and identical on both rows; only sens/spec move.")
    print("   The carried-over cut-off is the literal reading of \"exactly that model\"")
    print("   and stays the headline. The re-derived one is shown so the choice is visible.")

    # ---- 3. cut-point sensitivity ----------------------------------------
    hr("3. CUT-POINT SENSITIVITY  (the free parameter, PROGRESS §10 material L)")
    sens_rows = []
    for scheme in ("preregistered", "strict", "loose"):
        Xs = binarise(df, scheme)[bundle["features"]]
        for tag in ("ada", "lr"):
            mdl, thr = bundle[tag]["model"], bundle[tag]["threshold"]
            p = mdl.predict_proba(Xs)[:, 1]
            auc, se, sp = metrics(y, p, thr)
            sens_rows.append({"cuts": scheme, "model": tag, "AUC": auc,
                              "sens": se, "spec": sp,
                              "mean_themes": float(Xs.sum(axis=1).mean())})
    sw = pd.DataFrame(sens_rows)
    sw.to_csv(out / "transport_cut_sensitivity.csv", index=False)
    print(sw.round(4).to_string(index=False))
    print("\n   If AUC barely moves across the three schemes, the cut-points are not")
    print("   driving the result and the pre-registration did its job.")

    # ---- 4. the de-contamination counterfactual --------------------------
    hr("4. DROPPING THE C-12-FLAGGED CONTROLS  (diagnostic, not a corrected result)")
    keep = ~suspect
    Xk, yk = X[keep], y[keep]
    for tag in ("ada", "lr"):
        mdl, thr = bundle[tag]["model"], bundle[tag]["threshold"]
        p_all = mdl.predict_proba(X)[:, 1]
        p_k = mdl.predict_proba(Xk)[:, 1]
        a_all, se_all, sp_all = metrics(y, p_all, thr)
        a_k, se_k, sp_k = metrics(yk, p_k, thr)
        tr_rows.append({"model": tag, "set": "ours_minus_C12", "n": int(keep.sum()),
                        "AUC": a_k, "sens": se_k, "spec": sp_k,
                        "threshold": thr, "threshold_source": "goldstein_prevalence"})
        print(f"  {tag:<4} all rows        n={len(y):>4}  AUC {a_all:.4f}  "
              f"sens {se_all:.4f}  spec {sp_all:.4f}")
        print(f"  {tag:<4} minus {int(suspect.sum())} flagged n={int(keep.sum()):>4}  "
              f"AUC {a_k:.4f}  sens {se_k:.4f}  spec {sp_k:.4f}")
        print(f"       delta            AUC {a_k - a_all:+.4f}  "
              f"sens {se_k - se_all:+.4f}  spec {sp_k - sp_all:+.4f}\n")
    pd.DataFrame(tr_rows).to_csv(out / "transport_performance.csv", index=False)
    print("   The simulations predict specificity should be what recovers most when")
    print("   this kind of contamination is removed. Check that against the deltas.")

    # ---- 5. how much does "not reported = 0" decide the answer? ----------
    hr("5. MISSINGNESS SENSITIVITY  (the other free parameter, alongside the cuts)")
    print("  Four of the nine items carry non-response: Infertilite 25.9%, Dyspareunie")
    print("  12.6%, Symptomes_cycliques 7.4%, Dyschesie 5.2%. binarise() codes a missing")
    print("  answer as 0, i.e. 'symptom not reported'. That matches how the Goldstein")
    print("  data was built, where an unticked box is also a 0, but it is an assumption,")
    print("  and non-response is NOT balanced between the groups: on the fertility item")
    print("  32% of controls skipped against 18% of cases. Coding those skips as 0 pushes")
    print("  controls towards the negative class, so it flatters the result rather than")
    print("  hurting it. Four treatments are compared so the size of that is visible.\n")

    ms_rows = []

    def score(Xm, ym, tag, label, n):
        mdl, thr = bundle[tag]["model"], bundle[tag]["threshold"]
        p = mdl.predict_proba(Xm)[:, 1]
        auc, se, sp = metrics(ym, p, thr)
        ms_rows.append({"treatment": label, "model": tag, "n": n,
                        "AUC": auc, "sens": se, "spec": sp})

    # (a) baseline, and (b) the opposite extreme
    for label, mode in [("missing as 0 (current)", "zero"), ("missing as 1", "one")]:
        Xm = binarise(df, "preregistered", missing=mode)[bundle["features"]]
        for tag in ("ada", "lr"):
            score(Xm, y, tag, label, len(y))

    # (c) drop the rows with any missing answer - no assumption at all
    complete = ~df[[v["ours"] for v in ALIGNMENT.values()]].isin(MISSING).any(axis=1)
    Xc = binarise(df[complete], "preregistered")[bundle["features"]]
    yc = y[complete.to_numpy()]
    for tag in ("ada", "lr"):
        score(Xc, yc, tag, "complete cases only", int(complete.sum()))

    # (d) remove the worst offender from the model entirely, refitting on Goldstein
    #     so it is still "the same model", just an 8-variable version of it
    eight_fit = fit_eight(bundle, a.goldstein)
    X8 = binarise(df, "preregistered")[eight_fit["features"]]
    for tag in ("ada", "lr"):
        mdl, thr8 = eight_fit[tag]["model"], eight_fit[tag]["threshold"]
        p = mdl.predict_proba(X8)[:, 1]
        auc, se, sp = metrics(y, p, thr8)
        ms_rows.append({"treatment": "8 variables, fertility dropped", "model": tag,
                        "n": len(y), "AUC": auc, "sens": se, "spec": sp})

    ms = pd.DataFrame(ms_rows)
    ms.to_csv(out / "transport_missingness.csv", index=False)
    print(ms.round(4).to_string(index=False))
    base = ms[(ms.treatment == "missing as 0 (current)") & (ms.model == "ada")].AUC.iloc[0]
    print(f"\n   spread around the current AUC of {base:.4f} (ada):")
    for _, r in ms[ms.model == "ada"].iterrows():
        print(f"     {r.treatment:<34} {r.AUC:.4f}  ({r.AUC - base:+.4f})")
    print("\n   If the spread is small the assumption is not deciding the answer and can")
    print("   be declared as a limitation. If it is large, the headline number needs a")
    print("   different treatment of missingness, not a footnote.")

    clean_channel_section(df, y, suspect, bundle, eight_fit, a.goldstein, out, a.boot)

    hr("written")
    for f in ["transport_dropped_resubmissions.csv",
              "transport_channels.csv", "transport_performance.csv",
              "transport_cut_sensitivity.csv", "transport_missingness.csv",
              "transport_clean_composition.csv", "transport_clean_controls.csv",
              "transport_clean_choices.csv",
              "transport_clean_precision.csv", "transport_clean_history.csv"]:
        q = out / f
        if q.exists():
            print(f"  {f:<34} {q.stat().st_size / 1024:>8.1f} KB")


if __name__ == "__main__":
    main()
