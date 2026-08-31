"""
Volet non supervisé : réduction de dimension et clustering.

Question posée ici : sans jamais montrer l'étiquette Cas/Témoin à l'algorithme,
retrouve-t-on quand même une structure qui correspond aux deux groupes ?

C'est une vérification indépendante du modèle supervisé. Si oui, le signal est
suffisamment fort pour émerger tout seul. Si non, cela ne veut pas dire que le
modèle supervisé est faux : cela veut dire que le signal existe mais qu'il est
minoritaire par rapport à la variabilité générale de la population.

Réserve méthodologique : l'ACP suppose des variables continues. Nos données sont
ordinales et binaires. Après standardisation, l'ACP reste utilisable comme outil
exploratoire, mais une analyse des correspondances multiples (ACM) serait plus
rigoureuse. Nous le signalons plutôt que de le passer sous silence.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PLUM, TEAL = "#5B2149", "#0F7B72"
RANDOM_STATE = 42


def _prepare(X: pd.DataFrame) -> np.ndarray:
    """Imputation médiane puis standardisation : indispensable avant ACP."""
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]).fit_transform(X)


def pca_analysis(X: pd.DataFrame, n_components=10) -> tuple[np.ndarray, pd.DataFrame]:
    """Réduction de dimension. Renvoie les coordonnées et la variance expliquée."""
    Z = _prepare(X)
    n = min(n_components, Z.shape[1])
    pca = PCA(n_components=n, random_state=RANDOM_STATE)
    coords = pca.fit_transform(Z)

    var = pd.DataFrame({
        "composante": [f"PC{i+1}" for i in range(n)],
        "variance_expliquée": pca.explained_variance_ratio_,
        "variance_cumulée": np.cumsum(pca.explained_variance_ratio_),
    })

    # variables contribuant le plus aux deux premiers axes
    load = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"])
    var.attrs["loadings"] = load.reindex(load["PC1"].abs().sort_values(ascending=False).index)
    return coords, var


def clustering_analysis(X: pd.DataFrame, y: pd.Series, k_range=range(2, 7)) -> pd.DataFrame:
    """
    K-moyennes pour plusieurs valeurs de k.

    silhouette   : qualité intrinsèque des groupes (proche de 1 = groupes nets,
                   proche de 0 = groupes qui se chevauchent). Ne regarde jamais y.
    ARI          : correspondance entre les clusters trouvés et les vrais groupes
                   (0 = hasard, 1 = correspondance parfaite). Calculé APRÈS coup,
                   uniquement pour évaluer, jamais pour guider le clustering.
    """
    Z = _prepare(X)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        lab = km.fit_predict(Z)
        rows.append({
            "k": k,
            "silhouette": silhouette_score(Z, lab),
            "ARI_vs_group": adjusted_rand_score(y, lab),
            "inertia": km.inertia_,
        })
    return pd.DataFrame(rows)


def compare_clustering_methods(X: pd.DataFrame, y: pd.Series, k=2) -> pd.DataFrame:
    """K-moyennes contre classification ascendante hiérarchique, à k fixé."""
    Z = _prepare(X)
    out = []
    for name, algo in [
        ("K-means", KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)),
        ("Hierarchical (Ward)", AgglomerativeClustering(n_clusters=k, linkage="ward")),
    ]:
        lab = algo.fit_predict(Z)
        out.append({
            "method": name,
            "silhouette": silhouette_score(Z, lab),
            "ARI_vs_group": adjusted_rand_score(y, lab),
        })
    return pd.DataFrame(out)


def cluster_profile(X: pd.DataFrame, y: pd.Series, df_raw: pd.DataFrame,
                    schema, k=2) -> pd.DataFrame:
    """Composition de chaque cluster : part de Cas, effectif, variables marquantes."""
    Z = _prepare(X)
    lab = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(Z)
    rows = []
    for c in range(k):
        m = lab == c
        rows.append({
            "cluster": c,
            "n": int(m.sum()),
            "pct_cases": round(float(y[m].mean()) * 100, 1),
            "severe_dysmenorrhea_pct": round(
                float((df_raw.loc[m, "Dysmenorrhee"] == "Sévère").mean()) * 100, 1)
            if "Dysmenorrhee" in df_raw.columns else np.nan,
            "family_history_pct": round(
                float((df_raw.loc[m, "Antecedent_familial"] == "Oui").mean()) * 100, 1)
            if "Antecedent_familial" in df_raw.columns else np.nan,
        })
    return pd.DataFrame(rows)


def plot_unsupervised(X: pd.DataFrame, y: pd.Series, path, k=2):
    """Trois vues : variance expliquée, plan factoriel par groupe réel, par cluster."""
    Z = _prepare(X)
    pca = PCA(n_components=min(10, Z.shape[1]), random_state=RANDOM_STATE)
    coords = pca.fit_transform(Z)
    lab = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(Z)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle("Unsupervised analysis — PCA and clustering",
                 fontsize=13, color=PLUM, fontweight="bold")

    ax = axes[0]
    ev = pca.explained_variance_ratio_ * 100
    ax.bar(range(1, len(ev) + 1), ev, color=PLUM, alpha=.85)
    ax.plot(range(1, len(ev) + 1), np.cumsum(ev), color=TEAL, marker="o", ms=4)
    ax.set_xlabel("principal component", fontsize=9)
    ax.set_ylabel("% of variance", fontsize=9)
    ax.set_title(f"Explained variance\n(PC1+PC2 = {ev[0]+ev[1]:.1f}%)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for val, lb, col in [(1, "Cases", PLUM), (0, "Controls", TEAL)]:
        m = (y == val).to_numpy()
        ax.scatter(coords[m, 0], coords[m, 1], s=14, alpha=.55, label=lb, color=col)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f} %)", fontsize=9)
    ax.set_ylabel(f"PC2 ({ev[1]:.1f} %)", fontsize=9)
    ax.set_title("Factorial plane — true groups", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    for c, col in zip(range(k), [PLUM, TEAL, "#B8860B", "#4A6FA5"]):
        m = lab == c
        ax.scatter(coords[m, 0], coords[m, 1], s=14, alpha=.55,
                   label=f"cluster {c}", color=col)
    ari = adjusted_rand_score(y, lab)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f} %)", fontsize=9)
    ax.set_ylabel(f"PC2 ({ev[1]:.1f} %)", fontsize=9)
    ax.set_title(f"Clusters found without labels\n(ARI = {ari:.3f})", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
