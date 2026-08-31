"""
Chaîne complète : chargement -> normalisation -> contrôles -> encodage -> EDA -> modèles.

    python run.py                        # données synthétiques
    python run.py --input export.csv     # données réelles (seul changement nécessaire)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_schema
from encode import encode
from eda import group_comparison, plot_overview
from model import (cross_validate, feature_importance, full_metrics,
                   logistic_coefficients, select_features, sensitivity_by_channel,
                   sensitivity_drop_suspect, tune_hyperparameters)
from unsupervised import (clustering_analysis, cluster_profile,
                          compare_clustering_methods, pca_analysis,
                          plot_unsupervised)
from synth import generate
from validate import Report, ValidationError, run_all

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DATA = ROOT / "data"


def hr(title: str):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None,
                    help="CSV réel. Par défaut : données synthétiques.")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--prop-cas", type=float, default=0.42)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inject-errors", type=int, default=0)
    ap.add_argument("--no-strict", action="store_true",
                    help="ne pas interrompre sur contrôle bloquant")
    ap.add_argument("--fast", action="store_true",
                    help="sauter le réglage des hyperparamètres et la sélection de variables")
    args = ap.parse_args(argv)

    OUT.mkdir(exist_ok=True)
    schema = load_schema()

    # ---------------------------------------------------------------- 1. données
    hr("1. LOADING")
    if args.input:
        df = pd.read_csv(args.input, dtype=str)
        print(f"source: {args.input}  ({len(df)} lignes, {df.shape[1]} colonnes)")
    else:
        df, meta = generate(n=args.n, prop_cas=args.prop_cas, seed=args.seed,
                            schema=schema, polish_path=DATA / "polish_correlations.xlsx",
                            inject_errors=args.inject_errors)
        print(f"source: SYNTHETIC data ({meta['n']} rows, {meta['prop_cas']:.1%} cases)")
        print(f"correlations: {meta['n_polish_pairs']} pairs from Zielinski 2023, "
              f"{meta['n_hyp_pairs']} hypothesised pairs")
        df.to_csv(OUT / "synthetic_raw.csv", index=False)

    # ---------------------------------------------------- 2. contrôles qualité
    hr("2. QUALITY CONTROLS (C-01 to C-13)")
    try:
        df, rep = run_all(df, schema, strict=not args.no_strict)
    except ValidationError as e:
        print(e)
        print("\nPipeline stopped. Fix the export and rerun.")
        return 1
    print(rep.to_frame().to_string(index=False))
    print(f"\n{rep.summary()}")
    rep.to_frame().to_csv(OUT / "validation_report.csv", index=False)

    # ------------------------------------------------------------ 3. encodage
    hr("3. ENCODING")
    enc_rep = Report()
    X, y = encode(df, schema, report=enc_rep)
    for f in enc_rep.findings:
        print(f"  {f.rule} [{f.severity}] {'OK' if f.passed else 'FAIL'}: {f.message}")
    if enc_rep.blocking_failures:
        print("\nLEAKAGE DETECTED — stopping.")
        return 1
    print(f"\nX: {X.shape[0]} rows x {X.shape[1]} columns")
    print(f"y: {int(y.sum())} cases / {int((1 - y).sum())} controls")
    print(f"excluded from X: {schema.leaky}")

    # ----------------------------------------------------------------- 4. EDA
    hr("4. EXPLORATORY ANALYSIS")
    comp = group_comparison(df, schema)
    print(comp.head(12).to_string(index=False))
    comp.to_csv(OUT / "eda_group_comparison.csv", index=False)
    plot_overview(df, schema, OUT / "eda_overview.png")
    print(f"\nfigure: {OUT / 'eda_overview.png'}")

    # -------------------------------------------------------------- 5. modèles
    hr("5. BASELINE MODELS (5-fold cross-validation x 3 repeats)")
    res = cross_validate(X, y)
    print(res.round(3).to_string(index=False))
    res.to_csv(OUT / "model_scores.csv", index=False)

    best = res.iloc[0]["AUC"]
    if best > 0.97:
        print("\n  WARNING: AUC > 0.97 on a sample this size.")
        print("  That is the classic symptom of leakage. Check X before going further.")

    hr("6. VARIABLE IMPORTANCE")
    imp = feature_importance(X, y, top=15)
    print(imp.round(4).to_string(index=False))
    imp.to_csv(OUT / "feature_importance.csv", index=False)
    print("\nLogistic regression coefficients (odds ratios):")
    print(logistic_coefficients(X, y, top=10).round(3).to_string(index=False))

    hr("7. DETAILED METRICS (best model)")
    fm = full_metrics(X, y)
    print(fm.round(3).to_string(index=False))
    fm.to_csv(OUT / "metrics_detailed.csv", index=False)
    print(f"\nAUC = {fm.attrs['auc']:.3f}")
    print("\n   Sensitivity: share of true cases correctly identified.")
    print("   Specificity: share of true controls correctly identified.")
    print("   Precision (PPV): among those flagged at risk, the share of true cases.")
    print("   The Youden threshold favours sensitivity, which suits a triage tool")
    print("   better than a fixed 0.50 cut-off.")

    if not args.fast:
        hr("8. HYPERPARAMETER TUNING (grid search)")
        tun = tune_hyperparameters(X, y)
        print(tun.round(3).to_string(index=False))
        tun.to_csv(OUT / "hyperparameters.csv", index=False)

        hr("9. FEATURE SELECTION (RFECV)")
        kept, curve = select_features(X, y)
        print(f"variables kept: {len(kept)} / {X.shape[1]}")
        for v in kept:
            print(f"  - {v}")
        best = curve.loc[curve["AUC"].idxmax()]
        print(f"\nbest score: AUC {best['AUC']:.3f} with {int(best['n_variables'])} variables")
        curve.to_csv(OUT / "feature_selection_curve.csv", index=False)
    else:
        print("\n(sections 8 and 9 skipped: --fast)")

    # ---------------------------------------------------- 10. non supervisé
    hr("10. UNSUPERVISED ANALYSIS (PCA + CLUSTERING)")
    coords, var = pca_analysis(X)
    print("Variance explained by the first components:")
    print(var.head(5).round(3).to_string(index=False))
    n80 = int((var["variance_cumulée"] < 0.80).sum()) + 1
    print(f"\n{n80} components explain 80% of the variance (out of {X.shape[1]} variables).")
    print("\nVariables contributing most to the first two axes:")
    print(var.attrs["loadings"].head(6).round(3).to_string())
    var.to_csv(OUT / "pca_variance.csv", index=False)

    print("\nK-means clustering for different values of k:")
    cl = clustering_analysis(X, y)
    print(cl.round(3).to_string(index=False))
    cl.to_csv(OUT / "clustering_k.csv", index=False)

    print("\nComparison of two methods at k=2:")
    print(compare_clustering_methods(X, y).round(3).to_string(index=False))

    print("\nCluster composition:")
    print(cluster_profile(X, y, df, schema).to_string(index=False))

    plot_unsupervised(X, y, OUT / "unsupervised_overview.png")
    print(f"\nfigure: {OUT / 'unsupervised_overview.png'}")
    print("\n   ARI near 0 = the clusters do not correspond to the groups.")
    print("   This is not a failure: it means general population variability")
    print("   dominates the disease signal. That is expected, and it is precisely")
    print("   why a supervised approach is required.")

    # ------------------------------------------------- 11. analyses de sensibilité
    hr("11. SENSITIVITY ANALYSES")
    if "Source_canal" in df.columns:
        print("a) by recruitment channel")
        s1 = sensitivity_by_channel(X, y, df["Source_canal"])
        print(s1.round(3).to_string(index=False))
        s1.to_csv(OUT / "sensitivity_channel.csv", index=False)
        print("\n   Reading: if the AUC holds up within the social-media channel alone,")
        print("   where cases and controls coexist, the signal is not merely a")
        print("   recruitment artefact.")
    else:
        print("a) by channel: Source_canal missing from the export — analysis not possible.")

    if "flag_temoin_suspect" in df.columns:
        print("\nb) excluding probably under-diagnosed controls (C-12)")
        s2 = sensitivity_drop_suspect(X, y, df["flag_temoin_suspect"])
        print(s2.round(3).to_string(index=False))
        s2.to_csv(OUT / "sensitivity_suspect.csv", index=False)

    hr("DONE")
    print(f"outputs: {OUT}")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
