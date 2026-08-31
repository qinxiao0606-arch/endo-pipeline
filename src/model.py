"""
Modèles de référence, validation croisée, importance des variables,
et analyse de sensibilité sur le canal de recrutement.

Trois modèles volontairement simples. L'objectif à ce stade n'est pas la performance
mais de vérifier que la chaîne produit des résultats cohérents et interprétables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def build_models(class_weight="balanced") -> dict:
    return {
        "Régression logistique": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight=class_weight,
                                       random_state=RANDOM_STATE)),
        ]),
        "Forêt aléatoire": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                           class_weight=class_weight,
                                           random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        # gère les NaN nativement, pas d'imputation
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=RANDOM_STATE),
    }


def cross_validate(X: pd.DataFrame, y: pd.Series, models=None, n_splits=5,
                   n_repeats=3) -> pd.DataFrame:
    """AUC et PR-AUC en validation croisée répétée."""
    models = models or build_models()
    rows = []
    for name, mdl in models.items():
        aucs, aps = [], []
        for rep in range(n_repeats):
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                 random_state=RANDOM_STATE + rep)
            p = cross_val_predict(mdl, X, y, cv=cv, method="predict_proba")[:, 1]
            aucs.append(roc_auc_score(y, p))
            aps.append(average_precision_score(y, p))
        rows.append({
            "modèle": name,
            "AUC": np.mean(aucs), "AUC_sd": np.std(aucs),
            "PR-AUC": np.mean(aps), "PR-AUC_sd": np.std(aps),
        })
    return pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)


def feature_importance(X: pd.DataFrame, y: pd.Series, model=None, n_repeats=10,
                       top=20) -> pd.DataFrame:
    """Importance par permutation (substitut de SHAP, non installé dans cet environnement)."""
    model = model or build_models()["Gradient boosting"]
    model.fit(X, y)
    r = permutation_importance(model, X, y, n_repeats=n_repeats,
                               random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1)
    out = (pd.DataFrame({"variable": X.columns,
                         "importance": r.importances_mean,
                         "sd": r.importances_std})
           .sort_values("importance", ascending=False)
           .reset_index(drop=True))
    return out.head(top)


def logistic_coefficients(X: pd.DataFrame, y: pd.Series, top=20) -> pd.DataFrame:
    mdl = build_models()["Régression logistique"]
    mdl.fit(X, y)
    coef = mdl.named_steps["clf"].coef_[0]
    out = pd.DataFrame({"variable": X.columns, "coef": coef,
                        "odds_ratio": np.exp(coef)})
    out["abs"] = out["coef"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").head(top).reset_index(drop=True)


def sensitivity_by_channel(X: pd.DataFrame, y: pd.Series, canal: pd.Series,
                           min_n=60) -> pd.DataFrame:
    """
    Analyse de sensibilité centrale du projet.

    Le modèle principal utilise tout l'échantillon, mais les Cas viennent surtout des
    associations et les Témoins des réseaux sociaux. On refait donc tourner le modèle
    à l'intérieur du seul canal « réseaux sociaux », où les deux groupes coexistent.
    Si l'AUC s'y maintient, le signal n'est pas qu'un artefact de recrutement.
    """
    rows = []
    full = cross_validate(X, y, n_repeats=2).iloc[0]
    rows.append({"échantillon": "complet", "n": len(X), "n_cas": int(y.sum()),
                 "modèle": full["modèle"], "AUC": full["AUC"], "PR-AUC": full["PR-AUC"]})

    # Les canaux associatifs sont lus dans le dictionnaire (préfixe « endo »),
    # pas écrits en dur : un partenariat de plus ne doit pas exiger un patch ici.
    asso = [c for c in canal.dropna().unique() if str(c).startswith("endo")]

    for grp, label in [(canal == "social", "réseaux sociaux uniquement"),
                       (canal.isin(asso), "associations uniquement")]:
        Xi, yi = X[grp.to_numpy()], y[grp.to_numpy()]
        if len(Xi) < min_n or yi.nunique() < 2 or min(yi.sum(), len(yi) - yi.sum()) < 15:
            rows.append({"échantillon": label, "n": int(grp.sum()),
                         "n_cas": int(yi.sum()), "modèle": "—",
                         "AUC": np.nan, "PR-AUC": np.nan})
            continue
        r = cross_validate(Xi, yi, n_repeats=2).iloc[0]
        rows.append({"échantillon": label, "n": len(Xi), "n_cas": int(yi.sum()),
                     "modèle": r["modèle"], "AUC": r["AUC"], "PR-AUC": r["PR-AUC"]})
    return pd.DataFrame(rows)


def sensitivity_drop_suspect(X: pd.DataFrame, y: pd.Series,
                             suspect: pd.Series) -> pd.DataFrame:
    """Résultats avec et sans les témoins probablement sous-diagnostiqués (C-12)."""
    rows = []
    for keep, label in [(np.ones(len(X), bool), "tous les témoins"),
                        (~suspect.to_numpy(), "témoins suspects exclus")]:
        r = cross_validate(X[keep], y[keep], n_repeats=2).iloc[0]
        rows.append({"échantillon": label, "n": int(keep.sum()),
                     "AUC": r["AUC"], "PR-AUC": r["PR-AUC"]})
    return pd.DataFrame(rows)


# ===========================================================================
# Métriques complètes, réglage des hyperparamètres, sélection de variables
# ===========================================================================
from sklearn.feature_selection import RFECV
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_curve)
from sklearn.model_selection import GridSearchCV


def full_metrics(X: pd.DataFrame, y: pd.Series, model=None, n_splits=5) -> pd.DataFrame:
    """
    Sensibilité, spécificité, précision, rappel, F1 — à deux seuils.

    L'AUC résume la performance sur tous les seuils possibles. Mais dès qu'on
    veut décider « cette personne est à risque ou non », il faut choisir UN seuil,
    et ce choix change tout.

    seuil 0,50        : le seuil par défaut, celui que la plupart des outils
                        appliquent sans le dire.
    seuil de Youden   : celui qui maximise sensibilité + spécificité - 1.
                        Pour un outil d'orientation, on préfère généralement
                        privilégier la sensibilité : rater une patiente coûte
                        plus cher qu'une fausse alerte qui mène à une consultation.

    Rappel = sensibilité. Les deux termes désignent la même quantité, l'un vient
    de la recherche d'information, l'autre de l'épidémiologie.
    """
    model = model or build_models()["Forêt aléatoire"]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    fpr, tpr, thr = roc_curve(y, p)
    youden = thr[np.argmax(tpr - fpr)]

    rows = []
    for seuil, nom in [(0.50, "0,50 (défaut)"), (float(youden), f"{youden:.2f} (Youden)")]:
        pred = (p >= seuil).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        rows.append({
            "seuil": nom,
            "sensibilité (rappel)": tp / (tp + fn) if tp + fn else np.nan,
            "spécificité": tn / (tn + fp) if tn + fp else np.nan,
            "précision (VPP)": precision_score(y, pred, zero_division=0),
            "VPN": tn / (tn + fn) if tn + fn else np.nan,
            "F1": f1_score(y, pred, zero_division=0),
            "vrais positifs": int(tp), "faux positifs": int(fp),
            "faux négatifs": int(fn), "vrais négatifs": int(tn),
        })
    out = pd.DataFrame(rows)
    out.attrs["auc"] = roc_auc_score(y, p)
    return out


def tune_hyperparameters(X: pd.DataFrame, y: pd.Series, n_splits=5) -> pd.DataFrame:
    """
    Recherche exhaustive sur grille, évaluée en validation croisée.

    Les grilles restent volontairement petites : avec quelques centaines de lignes,
    explorer trop de combinaisons revient à sur-ajuster le choix des paramètres
    lui-même. Le gain attendu est modeste, l'objectif est surtout de vérifier que
    les valeurs par défaut ne sont pas absurdes.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    grids = {
        "Régression logistique": (
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                 random_state=RANDOM_STATE))]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Forêt aléatoire": (
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("clf", RandomForestClassifier(n_estimators=400,
                                                     class_weight="balanced",
                                                     random_state=RANDOM_STATE, n_jobs=-1))]),
            {"clf__max_depth": [None, 6, 10],
             "clf__min_samples_leaf": [1, 3, 8]},
        ),
        "Gradient boosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {"learning_rate": [0.03, 0.06, 0.12],
             "max_leaf_nodes": [7, 15, 31],
             "l2_regularization": [0.0, 1.0]},
        ),
    }

    rows = []
    for name, (est, grid) in grids.items():
        gs = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        gs.fit(X, y)
        base = cross_validate(X, y, {name: est}, n_repeats=1).iloc[0]["AUC"]
        rows.append({
            "modèle": name,
            "AUC_par_défaut": base,
            "AUC_optimisée": gs.best_score_,
            "gain": gs.best_score_ - base,
            "meilleurs_paramètres": str(gs.best_params_),
        })
    return pd.DataFrame(rows).sort_values("AUC_optimisée", ascending=False).reset_index(drop=True)


def select_features(X: pd.DataFrame, y: pd.Series, n_splits=5) -> tuple[list, pd.DataFrame]:
    """
    Élimination récursive de variables, le nombre optimal étant choisi en
    validation croisée (RFECV).

    On entraîne, on retire la variable la moins utile, on recommence. La
    validation croisée décide où s'arrêter. Utile ici pour deux raisons : montrer
    que le modèle ne repose pas sur 22 variables dont la moitié seraient du bruit,
    et produire une version courte du questionnaire si on devait le raccourcir.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    est = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                               random_state=RANDOM_STATE))])
    Xi = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),
                      columns=X.columns, index=X.index)
    sel = RFECV(LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=RANDOM_STATE),
                step=1, cv=cv, scoring="roc_auc", min_features_to_select=3, n_jobs=-1)
    sel.fit(StandardScaler().fit_transform(Xi), y)

    kept = [c for c, k in zip(X.columns, sel.support_) if k]
    curve = pd.DataFrame({
        "n_variables": range(3, 3 + len(sel.cv_results_["mean_test_score"])),
        "AUC": sel.cv_results_["mean_test_score"],
    })
    return kept, curve
