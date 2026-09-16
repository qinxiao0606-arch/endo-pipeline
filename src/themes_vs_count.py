"""
Does the model need to know WHICH symptoms, or only HOW MANY?

Julia, September:

  "try to train the 9 shared themes as separate features vs only the total number
   of positive themes -> model should know the symptoms or it is enough just to
   know the number of them :)"

A direct test of the count dominance noted in A3-0. Two feature sets, same data,
same model, same folds:

  A   the 9 shared themes as 9 separate binary features
  B   one single feature: how many of those 9 are positive

If B lands on A, the identity of the symptoms adds nothing beyond their number,
and any "which symptoms matter most" ranking built on this data is largely a
restatement of how many boxes each respondent ticked.

The same comparison is repeated on the raw 58 variables, because the answer turns
out to depend on the level of aggregation, and that contrast is the actual result.

Reports sensitivity and specificity alongside AUC. AUC only reads the ordering of
the scores, so it is insensitive to a great deal; sensitivity and specificity
depend on where scores sit relative to a fixed cut and move when AUC will not.

    cd src && python3 themes_vs_count.py
"""
from __future__ import annotations

import argparse
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


def evaluate(M, y, folds, rate):
    """AUC, sensitivity and specificity at a prevalence matched cut-off."""
    out = {}
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for tag, mdl in [("ada", AdaBoostClassifier(random_state=SEED)),
                     ("lr", LogisticRegression(max_iter=2000))]:
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
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(args.input)
    sym = [c for c in df.columns if c not in DROP]
    n, k = check_goldstein_coverage(sym)
    print(f"group check ok: {n} variables in {k} groups (definitions from groups.py)")

    X = df[sym].astype(float)
    y = df["label"].astype(int).to_numpy()
    rate = float(y.mean())

    sets = {
        "A  9 shared themes, separate features":
            X[SHARED_SUBSET],
        "B  9 shared themes, count only":
            pd.DataFrame({"n_positive_themes": X[SHARED_SUBSET].sum(axis=1)}),
        "C  all 58 variables, separate features":
            X,
        "D  all 58 variables, count only":
            pd.DataFrame({"n_ticked": X.sum(axis=1)}),
    }

    print(f"\n{len(y)} rows, {int(y.sum())} cases, prevalence {rate:.3f}, "
          f"{args.folds}-fold CV\n")
    print(f"{'feature set':<40}{'k':>3}{'AUC':>8}{'sens':>7}{'spec':>7}{'lrAUC':>8}")
    print("-" * 73)

    res, recs = {}, []
    for name, M in sets.items():
        r = evaluate(M, y, args.folds, rate)
        a, l = r["ada"], r["lr"]
        res[name[0]] = a[0]
        print(f"{name:<40}{M.shape[1]:>3}{a[0]:>8.3f}{a[1]:>7.3f}{a[2]:>7.3f}{l[0]:>8.3f}")
        recs.append({"feature_set": name, "k": M.shape[1],
                     "ada_AUC": a[0], "ada_sens": a[1], "ada_spec": a[2],
                     "lr_AUC": l[0], "lr_sens": l[1], "lr_spec": l[2]})

    pd.DataFrame(recs).to_csv(out / "themes_vs_count.csv", index=False)

    print("-" * 73)
    print(f"\nat theme level      A - B = {res['A'] - res['B']:+.3f} AUC")
    print(f"at variable level   C - D = {res['C'] - res['D']:+.3f} AUC")
    print("""
Knowing which of the 9 themes are positive is worth almost nothing over knowing
how many of them are. On the raw 58 variables the identity is worth considerably
more. The information sits in the fine grain and aggregation destroys it, which
is a reason to be careful about reading theme level importance rankings as
clinical statements.""")

    print("\nthe 9 shared themes:")
    for tid, t in ALIGNMENT.items():
        print(f"  {tid}  {t['theme']:<26} {t['rep']}")
    print(f"\nwritten to {out}/themes_vs_count.csv")


if __name__ == "__main__":
    main()
