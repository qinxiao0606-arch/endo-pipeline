"""
Encodage des réponses en matrice X.

Principes :
  - ordinales      -> entier respectant l'ordre du dictionnaire
  - nominales/bin. -> one-hot
  - NA_inconnu et NA_non_renseigne -> NaN + colonne indicatrice (l'absence de réponse
    est elle-même informative, on ne l'efface pas)
  - sentinelle 99 (« Non concernée ») -> colonne binaire dédiée, PAS un NaN :
    « aucun symptôme » est une réponse, pas une donnée manquante
  - les variables de fuite ne sont jamais dans X (vérifié par C-07)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import Schema
from validate import c07_leakage

NA_LABELS = {"Non renseigné", "Inconnu", "Non applicable"}
SENTINEL_LABEL = "Non concernée"


def encode(df: pd.DataFrame, schema: Schema, report=None) -> tuple[pd.DataFrame, pd.Series]:
    """Renvoie (X, y). y = 1 pour les Cas."""
    y = (df[schema.target] == "Cas").astype(int)
    parts: list[pd.DataFrame] = []

    for var in schema.features:
        if var not in df.columns:
            continue
        col = df[var].astype(object)
        typ = schema.type_of(var)
        order = schema.order_of(var)          # valeur -> rang (hors NA et hors 99)

        has_sentinel = SENTINEL_LABEL in schema.values_of(var)
        if has_sentinel:
            parts.append(pd.DataFrame(
                {f"{var}__non_concernee": (col == SENTINEL_LABEL).astype(int)},
                index=df.index))

        is_na = col.isin(NA_LABELS) | col.isna()
        if is_na.any():
            parts.append(pd.DataFrame({f"{var}__manquant": is_na.astype(int)},
                                      index=df.index))

        if typ == "ordinal":
            vals = col.map(order)                       # non mappé -> NaN
            parts.append(pd.DataFrame({var: vals.astype(float)}, index=df.index))
        else:
            usable = col.where(~is_na & (col != SENTINEL_LABEL))
            cats = [v for v in schema.values_of(var)
                    if v not in NA_LABELS and v != SENTINEL_LABEL]
            dummies = pd.get_dummies(
                pd.Categorical(usable, categories=cats), prefix=var, dtype=int)
            dummies.index = df.index
            if len(cats) == 2:                          # binaire : une seule colonne suffit
                dummies = dummies.iloc[:, [0]]
            parts.append(dummies)

    X = pd.concat(parts, axis=1)
    X = X.loc[:, X.nunique(dropna=False) > 1]           # colonnes constantes inutiles

    if report is not None:
        c07_leakage(list(X.columns), schema, report)
    else:
        leak = set(X.columns) & set(schema.leaky)
        assert not leak, f"C-07 : fuite de cible dans X : {leak}"

    return X, y


def describe(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "colonne": X.columns,
        "type": ["ordinal" if X[c].dtype == float else "binaire" for c in X.columns],
        "manquants_%": [round(float(X[c].isna().mean()) * 100, 1) for c in X.columns],
    })
