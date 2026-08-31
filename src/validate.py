"""
Contrôles C-01 à C-13 du dictionnaire.

BLOQUANT -> lève ValidationError (le pipeline s'arrête)
ALERTE   -> journalisé, la ligne concernée est marquée d'un flag
AUTO     -> correction automatique tracée

Chaque contrôle renvoie un Finding. Rien n'est corrigé silencieusement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config import META_COLS, NA_APPLICABLE, NA_INCONNU, NA_RENSEIGNE, Schema

BLOCKING, ALERT, AUTO = "BLOCKING", "ALERT", "AUTO"


class ValidationError(RuntimeError):
    pass


@dataclass
class Finding:
    rule: str
    severity: str
    passed: bool
    message: str
    n_rows: int = 0
    details: object = None


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    flags: dict[str, pd.Series] = field(default_factory=dict)

    def add(self, f: Finding):
        self.findings.append(f)

    @property
    def blocking_failures(self):
        return [f for f in self.findings if not f.passed and f.severity == BLOCKING]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "rule": f.rule, "severity": f.severity,
            "status": "OK" if f.passed else "FAIL",
            "rows": f.n_rows, "message": f.message,
        } for f in self.findings])

    def summary(self) -> str:
        ok = sum(f.passed for f in self.findings)
        return f"{ok}/{len(self.findings)} controls passed"


# ---------------------------------------------------------------------------
# C-13 : normalisation des libellés (première étape, avant tout autre contrôle)
# ---------------------------------------------------------------------------
def c13_normalise(df: pd.DataFrame, schema: Schema, rep: Report) -> pd.DataFrame:
    df = df.copy()
    n_fixed, detail = 0, {}
    for var in schema.data_vars:
        if var not in df.columns:
            continue
        amap = schema.alias_map(var)
        canon = set(schema.values_of(var))
        col = df[var].astype(object)
        hits = col.map(lambda v: isinstance(v, str) and v not in canon and v in amap)
        if hits.any():
            for bad in col[hits].unique():
                detail[f"{var}: {bad!r} -> {amap[bad]!r}"] = int((col == bad).sum())
            df.loc[hits, var] = col[hits].map(amap)
            n_fixed += int(hits.sum())
    rep.add(Finding("C-13", AUTO, True,
                    f"{n_fixed} value(s) normalised from the alias table",
                    n_fixed, detail))
    return df


# ---------------------------------------------------------------------------
def c01_columns(df, schema, rep):
    expected = set(schema.data_vars)
    missing = sorted(expected - set(df.columns))
    extra = sorted(set(df.columns) - expected - {schema.identifier} - set(META_COLS))
    rep.add(Finding("C-01", BLOCKING, not missing,
                    "all dictionary columns present" if not missing
                    else f"missing columns: {missing}",
                    len(missing), {"missing": missing, "not_in_dictionary": extra}))


def c05_modalities(df, schema, rep):
    bad = {}
    for var in schema.data_vars:
        if var not in df.columns:
            continue
        allowed = set(schema.values_of(var))
        seen = set(df[var].dropna().astype(str).unique())
        unknown = seen - allowed
        if unknown:
            bad[var] = sorted(unknown)
    n = sum(len(v) for v in bad.values())
    rep.add(Finding("C-05", BLOCKING, not bad,
                    "all values appear in the Modalites sheet" if not bad
                    else f"unknown values in {len(bad)} variable(s)", n, bad))


def c02_stade(df, schema, rep):
    if not {"Stade", "Groupe"} <= set(df.columns):
        return
    bad = df[(df["Groupe"] == "Témoin") & (~df["Stade"].isin(["Non applicable"]))]
    rep.add(Finding("C-02", BLOCKING, len(bad) == 0,
                    "Stage / Group consistency holds" if len(bad) == 0
                    else f"{len(bad)} control(s) with a stage recorded",
                    len(bad), bad.index.tolist()[:20]))


def c03_delai(df, schema, rep):
    if not {"Delai_diagnostic", "Groupe"} <= set(df.columns):
        return
    bad = df[(df["Groupe"] == "Témoin") & (df["Delai_diagnostic"] != "Non concernée")]
    rep.add(Finding("C-03", ALERT, len(bad) == 0,
                    "Delay / Group consistency holds" if len(bad) == 0
                    else f"{len(bad)} control(s) with a delay recorded -> recoded",
                    len(bad), bad.index.tolist()[:20]))
    if len(bad):
        rep.flags["flag_incoherent_delai"] = df.index.isin(bad.index)


def c04_cycle(df, schema, rep):
    if not {"Longueur_cycle", "Regularite_cycle"} <= set(df.columns):
        return
    bad = df[((df["Longueur_cycle"] == "Irrégulier") & (df["Regularite_cycle"] == "Régulier"))
             | ((df["Longueur_cycle"] != "Irrégulier") & (df["Regularite_cycle"] == "Irrégulier")
                & (df["Longueur_cycle"].isin(["<21 jours", "21-35 jours", ">35 jours"])) & False)]
    rep.add(Finding("C-04", ALERT, len(bad) == 0,
                    "cycle consistency holds" if len(bad) == 0
                    else f"{len(bad)} row(s) inconsistent on cycle length/regularity",
                    len(bad), bad.index.tolist()[:20]))
    if len(bad):
        rep.flags["flag_incoherent_cycle"] = df.index.isin(bad.index)


def c06_na_semantics(df, schema, rep):
    """« Non applicable » ne doit apparaître que là où la question n'était pas affichée."""
    problems = {}
    if {"Stade", "Groupe"} <= set(df.columns):
        n = int(((df["Groupe"] == "Cas") & (df["Stade"] == "Non applicable")).sum())
        if n:
            problems["Stage='Non applicable' among cases"] = n
    rep.add(Finding("C-06", BLOCKING, not problems,
                    "missing-value semantics consistent" if not problems
                    else f"structural NA confused with response NA: {problems}",
                    sum(problems.values()), problems))


def c07_leakage(X_cols, schema, rep):
    leak = sorted(set(X_cols) & set(schema.leaky))
    rep.add(Finding("C-07", BLOCKING, not leak,
                    "no leaky variable in X" if not leak
                    else f"LEAKAGE: {leak} present in matrix X",
                    len(leak), leak))


def c08_k_anonymity(df, schema, rep, k=5):
    qi = [c for c in ["Tranche_age", "IMC_categorie", "Parite", "Tabagisme",
                      "Consommation_alcool", "Activite_physique"] if c in df.columns]
    if not qi:
        return
    counts = df.groupby(qi, observed=True).size()
    thin = counts[counts < k]
    n_rows = int(thin.sum())
    rep.add(Finding("C-08", BLOCKING, n_rows == 0,
                    f"k-anonymity k>={k} satisfied on {len(qi)} quasi-identifiers"
                    if n_rows == 0 else
                    f"{len(thin)} combination(s) shared by < {k} people, i.e. {n_rows} rows "
                    f"(blocking before RELEASE only)",
                    n_rows, {"quasi_identifiers": qi, "n_thin_combinations": len(thin)}))


def c09_duplicates(df, schema, rep):
    cols = [c for c in schema.data_vars if c in df.columns]
    dup = df.duplicated(subset=cols, keep=False)
    rep.add(Finding("C-09", ALERT, not dup.any(),
                    "no exact duplicates" if not dup.any()
                    else f"{int(dup.sum())} exactly identical row(s)",
                    int(dup.sum()), df.index[dup].tolist()[:20]))
    if dup.any():
        rep.flags["flag_doublon"] = dup.to_numpy()


def c10_balance(df, schema, rep, min_n=30):
    if "Groupe" not in df.columns:
        return
    vc = df["Groupe"].value_counts()
    n_min = int(vc.min()) if len(vc) else 0
    ratio = float(vc.max() / vc.min()) if len(vc) == 2 and vc.min() else float("inf")
    ok = n_min >= min_n and ratio <= 3
    rep.add(Finding("C-10", ALERT, ok,
                    f"groups balanced ({dict(vc)}, ratio {ratio:.2f})" if ok
                    else f"imbalance: {dict(vc)}, ratio {ratio:.2f} "
                         f"-> prefer PR-AUC and class_weight",
                    n_min, dict(vc)))


def c11_missingness(df, schema, rep, thr=0.20):
    rates = {}
    for var in schema.features:
        if var not in df.columns:
            continue
        r = float((df[var] == "Non renseigné").mean())
        if r > thr:
            rates[var] = round(r, 3)
    rep.add(Finding("C-11", ALERT, not rates,
                    f"no variable above {thr:.0%} non-response" if not rates
                    else f"high non-response: {rates}", len(rates), rates))


def c12_control_contamination(df, schema, rep):
    need = {"Groupe", "Dysmenorrhee", "Douleur_pelvienne"}
    if not need <= set(df.columns):
        return
    susp = ((df["Groupe"] == "Témoin") & (df["Dysmenorrhee"] == "Sévère")
            & (df["Douleur_pelvienne"] == "Sévère"))
    rep.add(Finding("C-12", ALERT, not susp.any(),
                    "no suspect controls" if not susp.any()
                    else f"{int(susp.sum())} control(s) probably under-diagnosed "
                         f"-> kept and flagged for the sensitivity analysis",
                    int(susp.sum()), df.index[susp].tolist()[:20]))
    rep.flags["flag_temoin_suspect"] = susp.to_numpy()


# ---------------------------------------------------------------------------
def run_all(df: pd.DataFrame, schema: Schema, *, k=5, strict=True) -> tuple[pd.DataFrame, Report]:
    rep = Report()
    df = c13_normalise(df, schema, rep)
    c01_columns(df, schema, rep)
    c05_modalities(df, schema, rep)
    c02_stade(df, schema, rep)
    c03_delai(df, schema, rep)
    c04_cycle(df, schema, rep)
    c06_na_semantics(df, schema, rep)
    c08_k_anonymity(df, schema, rep, k=k)
    c09_duplicates(df, schema, rep)
    c10_balance(df, schema, rep)
    c11_missingness(df, schema, rep)
    c12_control_contamination(df, schema, rep)

    # corrections tracées des ALERTES
    if "Delai_diagnostic" in df.columns and "Groupe" in df.columns:
        m = (df["Groupe"] == "Témoin") & (df["Delai_diagnostic"] != "Non concernée")
        df.loc[m, "Delai_diagnostic"] = "Non concernée"

    for name, mask in rep.flags.items():
        df[name] = mask

    if strict:
        fatal = [f for f in rep.blocking_failures if f.rule != "C-08"]
        if fatal:
            msg = "\n".join(f"  {f.rule}: {f.message}" for f in fatal)
            raise ValidationError(f"blocking controls failed:\n{msg}")
    return df, rep
