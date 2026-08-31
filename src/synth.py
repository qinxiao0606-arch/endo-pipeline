"""
Générateur de données synthétiques conforme au dictionnaire.

Modèle génératif (gaussienne conditionnelle / copule) :

  1. y ~ Bernoulli(p_cas)
  2. Z | y ~ N(delta * y, R)      Z = variables latentes continues
  3. chaque Z_i est discrétisé en modalités selon des fréquences marginales cibles

Autrement dit : chaque variable a une variable latente normale ; le groupe Cas décale
la moyenne de delta_i écarts-types ; R contrôle les corrélations ENTRE variables.
Le découpage en tranches reproduit ensuite les modalités du questionnaire.

Deux sources pour les paramètres, volontairement séparées :
  - R      : matrice de corrélation, partiellement issue de Zieliński 2023 (données réelles)
  - delta  : tailles d'effet, issues d'ordres de grandeur de la littérature (HYPOTHÈSES)

delta n'est PAS mesuré, c'est un choix explicite documenté ci-dessous. Le but est que le
pipeline soit testé sur des données qui se comportent de façon plausible, pas de simuler
la vérité épidémiologique.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import NA_APPLICABLE, Schema, load_schema

# ---------------------------------------------------------------------------
# 1. Fréquences marginales cibles (population féminine, hypothèses documentées)
# ---------------------------------------------------------------------------
# Chaque liste suit l'ordre des modalités du dictionnaire (hors NA et sentinelles).
MARGINALS = {
    "Tranche_age":             [.06, .15, .20, .20, .17, .12, .10],
    "IMC_categorie":           [.07, .55, .26, .12],
    "Age_menarche":            [.07, .38, .40, .15],
    "Longueur_cycle":          [.08, .70, .10, .12],
    "Regularite_cycle":        [.75, .25],
    "Abondance_regles":        [.20, .55, .25],
    "Duree_regles":            [.18, .67, .15],
    "Dysmenorrhee":            [.18, .32, .30, .20],
    "Douleur_pelvienne":       [.45, .28, .18, .09],
    "Dyspareunie":             [.25, .75],
    "Dyschesie":               [.18, .82],
    "Symptomes_cycliques":     [.22, .78],
    "Age_apparition_symptomes": [.24, .32, .22, .13, .09],
    "Parite":                  [.45, .22, .23, .10],
    "Infertilite":             [.14, .86],
    "Contraception_hormonale": [.28, .40, .32],
    "Antecedent_familial":     [.12, .88],
    "Tabagisme":               [.58, .22, .20],
    "Activite_physique":       [.35, .45, .20],
    "Consommation_alcool":     [.25, .55, .20],
    "SII":                     [.12, .88],
    "Maladie_auto_immune":     [.08, .92],
}

# Sens de la modalité "positive" pour les binaires : ordre du dictionnaire = Oui, Non
# donc un Z élevé doit tomber sur "Oui" -> on inverse ces variables au découpage.
REVERSED = {"Dyspareunie", "Dyschesie", "Symptomes_cycliques", "Infertilite",
            "SII", "Maladie_auto_immune", "Antecedent_familial"}

# ---------------------------------------------------------------------------
# 2. Tailles d'effet : décalage de moyenne latente chez les Cas (en écarts-types)
# ---------------------------------------------------------------------------
# Conversion approximative depuis des odds ratios de la littérature : d ~ ln(OR) * 0.55.
# Ce sont des ORDRES DE GRANDEUR, pas des estimations. Documentées pour être discutables.
DELTA = {
    "Dysmenorrhee":             1.15,   # OR ~8   prédicteur le plus fort
    "Douleur_pelvienne":        0.90,   # OR ~5
    "Dyschesie":                0.70,   # OR ~3.5 symptôme assez spécifique
    "Dyspareunie":              0.55,   # OR ~2.7
    "Symptomes_cycliques":      0.50,   # OR ~2.5
    "Antecedent_familial":      0.95,   # OR ~5.5 facteur héréditaire bien documenté
    "Infertilite":              0.55,   # OR ~2.7
    "SII":                      0.50,   # OR ~2.5 comorbidité (SHAP n°1 UK Biobank 2022)
    "Maladie_auto_immune":      0.35,   # OR ~1.9
    "Age_menarche":            -0.25,   # ménarche précoce = risque -> décalage négatif
    "Longueur_cycle":          -0.30,   # cycles courts = risque
    "Abondance_regles":         0.35,
    "Duree_regles":             0.30,
    "Parite":                  -0.35,   # nulliparité associée (causalité inverse possible)
    "Age_apparition_symptomes": -0.45,  # symptômes plus précoces
    "Tranche_age":              0.20,   # léger, biais de recrutement associatif
    "IMC_categorie":           -0.15,   # association inverse faible
    "Contraception_hormonale":  0.25,   # traitement -> causalité inverse
    "Regularite_cycle":         0.20,
    "Tabagisme":                0.00,   # littérature contradictoire
    "Activite_physique":       -0.10,
    "Consommation_alcool":      0.10,
}

# ---------------------------------------------------------------------------
# 3. Corrélations issues de Zieliński 2023 (Additional file 2)
# ---------------------------------------------------------------------------
# Correspondances honnêtes uniquement. Les variables polonaises sans équivalent
# dans notre questionnaire sont ignorées.
POLISH_MAP = {
    "Menstrual pains": "Dysmenorrhee",
    "Bmi": "IMC_categorie",
    "Patient age": "Tranche_age",
    "Average menstrual cycle lenght": "Longueur_cycle",
    "Number of pregnancies": "Parite",
    "First menarche": "Age_menarche",
    "Diagnosed infertility": "Infertilite",
    "Oral contraceptives": "Contraception_hormonale",
    "Urinary-genital system symptoms": "Symptomes_cycliques",
}

# Corrélations complémentaires (hypothèses cliniques, non issues des données polonaises).
# Les grappes de douleur sont fortement corrélées entre elles ; l'étude polonaise ne
# couvre pas ces variables.
EXTRA_CORR = {
    ("Dysmenorrhee", "Douleur_pelvienne"): 0.45,
    ("Dysmenorrhee", "Dyschesie"): 0.30,
    ("Dysmenorrhee", "Abondance_regles"): 0.20,
    ("Douleur_pelvienne", "Dyspareunie"): 0.35,
    ("Douleur_pelvienne", "Dyschesie"): 0.30,
    ("Dyspareunie", "Dyschesie"): 0.25,
    ("Symptomes_cycliques", "SII"): 0.30,
    ("Longueur_cycle", "Regularite_cycle"): 0.55,
    ("Age_apparition_symptomes", "Dysmenorrhee"): -0.25,
    ("Infertilite", "Parite"): -0.40,
    ("SII", "Maladie_auto_immune"): 0.15,
}


def load_polish_corr(path) -> pd.DataFrame:
    """Extrait le sous-bloc de la matrice polonaise correspondant à nos variables."""
    frames = []
    for sheet in ["Spearman", "Matthews_correlation_coefficien"]:
        m = pd.read_excel(path, sheet_name=sheet, index_col=0)
        m.index = [str(i).strip() for i in m.index]
        m.columns = [str(c).strip() for c in m.columns]
        keep = [c for c in m.columns if c in POLISH_MAP]
        if keep:
            sub = m.loc[keep, keep].rename(index=POLISH_MAP, columns=POLISH_MAP)
            frames.append(sub)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.combine_first(f)
    return out


def build_corr(schema: Schema, polish: pd.DataFrame | None, seed=0) -> pd.DataFrame:
    """Matrice de corrélation latente : polonaise si disponible, sinon hypothèses."""
    vars_ = [v for v in schema.features]
    R = pd.DataFrame(np.eye(len(vars_)), index=vars_, columns=vars_)
    source = pd.DataFrame("", index=vars_, columns=vars_)

    if polish is not None and not polish.empty:
        for a in polish.index:
            for b in polish.columns:
                if a in R.index and b in R.columns and a != b:
                    v = polish.loc[a, b]
                    # combine_first laisse des NaN pour les paires inter-feuilles
                    if pd.notna(v):
                        R.loc[a, b] = float(v)
                        source.loc[a, b] = "PL"

    for (a, b), v in EXTRA_CORR.items():
        if a in R.index and b in R.columns:
            R.loc[a, b] = R.loc[b, a] = v
            source.loc[a, b] = source.loc[b, a] = "hyp"

    Rv = ((R + R.T) / 2).to_numpy(copy=True)
    np.fill_diagonal(Rv, 1.0)
    R = pd.DataFrame(Rv, index=R.index, columns=R.columns)
    return nearest_pd(R), source


def nearest_pd(R: pd.DataFrame) -> pd.DataFrame:
    """Projection sur la matrice définie positive la plus proche (clip des valeurs propres)."""
    vals, vecs = np.linalg.eigh(R.to_numpy(copy=True))
    vals = np.clip(vals, 1e-6, None)
    A = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(A))
    A = A / np.outer(d, d)
    np.fill_diagonal(A, 1.0)
    return pd.DataFrame(A, index=R.index, columns=R.columns)


def _cut(z: np.ndarray, probs: list[float], labels: list[str], reverse=False) -> np.ndarray:
    """Découpe une latente normale en modalités selon des fréquences marginales.

    reverse=True : on veut qu'un Z élevé tombe sur la PREMIÈRE modalité (cas des
    binaires écrites « Oui / Non »). On inverse alors labels ET probabilités, sinon
    la distribution marginale serait inversée elle aussi.
    """
    from scipy.stats import norm
    if reverse:
        probs = list(probs)[::-1]
        labels = list(labels)[::-1]
    edges = norm.ppf(np.cumsum(probs)[:-1])
    idx = np.digitize(z, edges)
    return np.array(labels, dtype=object)[idx]


def generate(n=400, prop_cas=0.42, seed=42, schema: Schema | None = None,
             polish_path=None, inject_errors=0) -> tuple[pd.DataFrame, dict]:
    """Génère un export synthétique au format brut du formulaire."""
    rng = np.random.default_rng(seed)
    schema = schema or load_schema()

    polish = load_polish_corr(polish_path) if polish_path else None
    R, source = build_corr(schema, polish, seed)
    vars_ = list(R.index)

    y = rng.random(n) < prop_cas                       # True = Cas
    delta = np.array([DELTA.get(v, 0.0) for v in vars_])
    L = np.linalg.cholesky(R.values)
    Z = rng.standard_normal((n, len(vars_))) @ L.T
    Z = Z + np.outer(y.astype(float), delta)

    df = pd.DataFrame(index=range(n))
    df[schema.identifier] = [f"E{i+1:04d}" for i in range(n)]
    df["Groupe"] = np.where(y, "Cas", "Témoin")

    for j, v in enumerate(vars_):
        labels = [m.valeur for m in schema.mods
                  if m.variable == v and not (isinstance(m.code, str) and m.code.startswith("NA"))
                  and m.valeur != "Non concernée"]
        probs = MARGINALS.get(v)
        if probs is None or len(probs) != len(labels):
            raise ValueError(f"marginales incohérentes pour {v}: "
                             f"{len(probs) if probs else 0} vs {len(labels)} modalités")
        df[v] = _cut(Z[:, j], probs, labels, reverse=(v in REVERSED))

    # cohérence interne : « cycle irrégulier » doit concorder entre Q6 et Q7.
    # Les vraies réponses en contiendront naturellement quelques-unes d'incohérentes ;
    # ici on part propre, et _inject() en réintroduit pour tester C-04.
    df.loc[df["Longueur_cycle"] == "Irrégulier", "Regularite_cycle"] = "Irrégulier"

    # --- variables conditionnelles / structurellement absentes ---------------
    stade_lab = ["I", "II", "III", "IV", "Inconnu"]
    stade_p = [.22, .26, .22, .12, .18]
    df["Stade"] = np.where(y, rng.choice(stade_lab, n, p=stade_p), "Non applicable")

    delai_lab = ["<2 ans", "2-5 ans", "6-10 ans", ">10 ans", "Inconnu"]
    delai_p = [.16, .30, .28, .20, .06]
    df["Delai_diagnostic"] = np.where(y, rng.choice(delai_lab, n, p=delai_p), "Non concernée")

    # asymptomatiques : surtout des témoins
    asympt = (~y) & (df["Dysmenorrhee"] == "Aucune") & (df["Douleur_pelvienne"] == "Aucune")
    df.loc[asympt, "Age_apparition_symptomes"] = "Non concernée"

    # non-réponses sur les questions intimes ("Non renseigné")
    for v, p in [("Dyspareunie", .08), ("Infertilite", .05), ("SII", .04),
                 ("Maladie_auto_immune", .04), ("Dyschesie", .05), ("Symptomes_cycliques", .04)]:
        mask = rng.random(n) < p
        df.loc[mask, v] = "Non renseigné"
    for v, p in [("Age_menarche", .05), ("Antecedent_familial", .12)]:
        mask = rng.random(n) < p
        df.loc[mask, v] = "Inconnu"

    # --- canal de recrutement (métadonnée, hors modélisation) ----------------
    # Reproduit le problème réel : les Cas viennent des associations, les Témoins
    # des réseaux sociaux. C'est ce qui rend l'analyse de sensibilité nécessaire.
    canal = np.empty(n, dtype=object)
    asso = ["endoespoir", "endomind", "endoaction"]
    for i in range(n):
        if y[i]:
            canal[i] = rng.choice(asso + ["social", "autre"], p=[.28, .30, .22, .15, .05])
        else:
            canal[i] = rng.choice(asso + ["social", "autre"], p=[.02, .02, .01, .85, .10])
    df["Source_canal"] = canal
    # Le formulaire synthétique est le formulaire français : la version anglaise
    # n'existe que dans les données réelles.
    df["Langue_formulaire"] = "fr"

    # --- ordre des colonnes conforme au dictionnaire -------------------------
    # dict.fromkeys : depuis la v4 du dictionnaire, Source_canal et
    # Langue_formulaire sont des variables déclarées, donc déjà dans data_vars.
    # Sans déduplication, df[cols] renverrait des colonnes en double.
    cols = list(dict.fromkeys(
        [schema.identifier] + schema.data_vars + ["Source_canal", "Langue_formulaire"]))
    df = df[[c for c in cols if c in df.columns]]

    meta = {"n": n, "prop_cas": float(y.mean()), "seed": seed,
            "corr_source": source, "R": R,
            "n_polish_pairs": int((source == "PL").values.sum() // 2),
            "n_hyp_pairs": int((source == "hyp").values.sum() // 2)}

    if inject_errors:
        df = _inject(df, rng, inject_errors)
        meta["errors_injected"] = inject_errors
    return df, meta


def _inject(df: pd.DataFrame, rng, k: int) -> pd.DataFrame:
    """Introduit délibérément des anomalies pour vérifier que les contrôles se déclenchent."""
    df = df.copy()
    n = len(df)
    pick = lambda: int(rng.integers(0, n))

    # C-02 : Stade renseigné chez un Témoin
    i = df.index[df["Groupe"] == "Témoin"][0]
    df.loc[i, "Stade"] = "III"
    # C-03 : délai renseigné chez un Témoin
    i = df.index[df["Groupe"] == "Témoin"][1]
    df.loc[i, "Delai_diagnostic"] = "2-5 ans"
    # C-04 : incohérence cycle
    df.loc[pick(), ["Longueur_cycle", "Regularite_cycle"]] = ["Irrégulier", "Régulier"]
    # C-05 : modalité inconnue
    df.loc[pick(), "Abondance_regles"] = "Très abondante"
    # C-13 : alias non normalisé
    df.loc[pick(), "Longueur_cycle"] = "<21 j"
    df.loc[pick(), "Consommation_alcool"] = "Non"
    df.loc[pick(), "Activite_physique"] = "Modérée"
    # C-09 : doublon strict
    df = pd.concat([df, df.iloc[[pick()]]], ignore_index=True)
    df.loc[df.index[-1], "ID_anonyme"] = f"E{len(df):04d}"
    return df


if __name__ == "__main__":
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    df, meta = generate(n=400, polish_path=root / "data" / "polish_correlations.xlsx",
                        inject_errors=0)
    print(df.head(3).T)
    print(f"\nn={meta['n']}  Cas={meta['prop_cas']:.1%}")
    print(f"paires de corrélation issues de Zieliński 2023 : {meta['n_polish_pairs']}")
    print(f"paires hypothétiques                           : {meta['n_hyp_pairs']}")
