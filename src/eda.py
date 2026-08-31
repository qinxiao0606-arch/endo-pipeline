"""Exploratory analysis: case/control comparison, variable by variable."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import Schema

PLUM, TEAL = "#5B2149", "#0F7B72"


def cramers_v(ct: pd.DataFrame) -> float:
    """Effect size for a contingency table (0 = independence)."""
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.to_numpy().sum()
    r, k = ct.shape
    if n == 0 or min(r - 1, k - 1) == 0:
        return np.nan
    phi2 = chi2 / n
    phi2 = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / min(kc - 1, rc - 1)))


def group_comparison(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Association of each variable with the group, sorted by effect size."""
    rows = []
    for var in schema.features:
        if var not in df.columns:
            continue
        ct = pd.crosstab(df[var], df[schema.target])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        try:
            chi2, p, _, _ = stats.chi2_contingency(ct)
            v = cramers_v(ct)
        except ValueError:
            continue
        pct = pd.crosstab(df[var], df[schema.target], normalize="columns") * 100
        top = pct["Cas"].idxmax() if "Cas" in pct.columns else ""
        rows.append({
            "variable": var,
            "cramers_v": round(v, 3),
            "p": f"{p:.1e}",
            "top_category_cases": top,
            "pct_cases": round(float(pct.loc[top, "Cas"]), 1) if top in pct.index else np.nan,
            "pct_controls": round(float(pct.loc[top, "Témoin"]), 1) if top in pct.index else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("cramers_v", ascending=False)
    return out.reset_index(drop=True)


def plot_overview(df: pd.DataFrame, schema: Schema, path, top_n=6):
    comp = group_comparison(df, schema)
    vars_ = comp["variable"].head(top_n).tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Cases vs controls — most discriminating variables",
                 fontsize=14, color=PLUM, fontweight="bold")

    for ax, var in zip(axes.ravel(), vars_):
        pct = pd.crosstab(df[var], df[schema.target], normalize="columns") * 100
        order = [v for v in schema.values_of(var) if v in pct.index]
        pct = pct.loc[order]
        x = np.arange(len(pct))
        w = 0.38
        ax.bar(x - w / 2, pct.get("Cas", 0), w, label="Cases", color=PLUM)
        ax.bar(x + w / 2, pct.get("Témoin", 0), w, label="Controls", color=TEAL)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i)[:14] for i in pct.index], rotation=30,
                           ha="right", fontsize=8)
        v = comp.loc[comp["variable"] == var, "cramers_v"].iloc[0]
        ax.set_title(f"{var}  (V={v})", fontsize=10)
        ax.set_ylabel("% of group", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes.ravel()[len(vars_):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
