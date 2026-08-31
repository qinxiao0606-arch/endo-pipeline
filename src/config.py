"""
Lecture du dictionnaire de données. C'est la seule source de vérité du pipeline :
aucun nom de variable, aucune modalité et aucune règle n'est codée en dur ici.

Si le dictionnaire change, le pipeline suit automatiquement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = ROOT / "data" / "Dictionnaire_donnees_endometriose_v5.xlsx"

# Valeurs manquantes : trois sémantiques distinctes, jamais fusionnées avant modélisation
NA_APPLICABLE = "NA_non_applicable"   # la question n'a pas été affichée
NA_INCONNU = "NA_inconnu"             # question posée, réponse inconnue
NA_RENSEIGNE = "NA_non_renseigne"     # refus de répondre
NA_TOKENS = {NA_APPLICABLE, NA_INCONNU, NA_RENSEIGNE}

# Marqueur du dictionnaire : libellé anglais pas encore relevé sur le formulaire
TODO_EN = "À VÉRIFIER"

# Colonnes présentes dans l'export mais hors modélisation
META_COLS = ["Source_canal", "Langue_formulaire"]


@dataclass
class Modality:
    variable: str
    valeur: str          # valeur canonique (libellé du formulaire en ligne)
    code: object         # int, ou un token NA_*
    ordre: object
    alias: list[str] = field(default_factory=list)
    libelle_en: str | None = None   # libellé du formulaire anglais, None si non relevé


@dataclass
class Schema:
    dic: pd.DataFrame
    mods: list[Modality]
    forms: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ---- listes de variables -------------------------------------------------
    @property
    def target(self) -> str:
        return self.dic.loc[self.dic["Rôle"] == "cible", "Variable"].iloc[0]

    @property
    def identifier(self) -> str:
        return self.dic.loc[self.dic["Type"] == "identifiant", "Variable"].iloc[0]

    @property
    def features(self) -> list[str]:
        return self.dic.loc[self.dic["Rôle"] == "feature", "Variable"].tolist()

    @property
    def excluded(self) -> list[str]:
        """Variables exclues de X pour cause de fuite (hors identifiant technique)."""
        d = self.dic
        return d.loc[(d["Rôle"] == "exclu") & (d["Type"] != "identifiant"), "Variable"].tolist()

    @property
    def leaky(self) -> list[str]:
        """Tout ce qui ne doit jamais entrer dans la matrice X."""
        return sorted(set(self.excluded) | {self.target})

    @property
    def all_vars(self) -> list[str]:
        return self.dic["Variable"].tolist()

    @property
    def data_vars(self) -> list[str]:
        """Variables attendues dans l'export (hors identifiant technique)."""
        return [v for v in self.all_vars if v != self.identifier]

    def type_of(self, var: str) -> str:
        return self.dic.set_index("Variable").loc[var, "Type"]

    @property
    def ordinals(self) -> list[str]:
        d = self.dic
        return d.loc[(d["Type"] == "ordinal") & (d["Rôle"] == "feature"), "Variable"].tolist()

    @property
    def nominals(self) -> list[str]:
        d = self.dic
        return d.loc[(d["Type"].isin(["nominal", "binaire"])) & (d["Rôle"] == "feature"),
                     "Variable"].tolist()

    # ---- modalités -----------------------------------------------------------
    def values_of(self, var: str) -> list[str]:
        return [m.valeur for m in self.mods if m.variable == var]

    def codes_of(self, var: str) -> dict:
        return {m.valeur: m.code for m in self.mods if m.variable == var}

    def order_of(self, var: str) -> dict:
        """Valeur -> rang, uniquement pour les modalités non-NA et non-sentinelles."""
        out = {}
        for m in self.mods:
            if m.variable == var and isinstance(m.code, (int, float)) and m.code != 99:
                out[m.valeur] = int(m.code)
        return out

    def alias_map(self, var: str | None = None) -> dict[str, str]:
        """Alias -> valeur canonique. Base du contrôle C-13.

        TOUJOURS appeler avec `var`. Un même libellé est ambigu hors contexte :
        « Non » est un alias de Témoin, de Aucune et de Jamais, tout en étant la
        valeur canonique de sept variables binaires. Une table globale écrase donc
        silencieusement deux cibles sur trois. Le paramètre `var` lève l'ambiguïté.
        """
        mods = self.mods if var is None else [m for m in self.mods if m.variable == var]
        out = {}
        for m in mods:
            for a in m.alias:
                out[a] = m.valeur
        return out

    def en_map(self, var: str) -> dict[str, str]:
        """Libellé du formulaire anglais -> valeur canonique française, pour UNE variable.

        Indexé par variable pour la même raison que alias_map : « Normal » vaut
        « Normale » pour Abondance_regles et « Normal (18,5-24,9) » pour IMC_categorie.
        Les modalités non encore relevées (« À VÉRIFIER ») sont ignorées.
        """
        out = {}
        for m in self.mods:
            if m.variable == var and m.libelle_en and m.libelle_en != TODO_EN:
                out[m.libelle_en] = m.valeur
        return out

    # ---- registre des formulaires -------------------------------------------
    def export_keys(self, lang: str) -> dict[str, str]:
        """Clé d'export Framaforms -> nom de variable, pour une version linguistique."""
        col = f"Cle_export_{lang}"
        d = self.dic
        return {str(r[col]).strip(): r["Variable"]
                for _, r in d.iterrows()
                if isinstance(r.get(col), str) and r[col].strip() not in ("", "—")}

    def form(self, suffix: str) -> pd.Series:
        """Ligne du registre Formulaires correspondant à un suffixe de lien."""
        f = self.forms
        hit = f[f["Suffixe"].astype(str).str.strip() == str(suffix)]
        if hit.empty:
            raise KeyError(f"suffixe {suffix!r} absent de la feuille Formulaires")
        return hit.iloc[0]

    def conditional_rules(self) -> dict[str, str]:
        """Variable -> texte de la condition d'affichage (vide si inconditionnelle)."""
        d = self.dic
        return {r["Variable"]: r["Conditionnel"]
                for _, r in d.iterrows()
                if isinstance(r["Conditionnel"], str) and r["Conditionnel"] not in ("—", "")}


def _read(path, sheet: str) -> pd.DataFrame:
    """Lecture d'une feuille SANS interprétation des valeurs manquantes.

    pandas convertit par défaut « None », « N/A », « NA » et « null » en NaN.
    Or ce sont des libellés légitimes de notre dictionnaire : « None » est la
    modalité anglaise de Douleur_pelvienne, « N/A » un alias de Stade. Les
    laisser filer supprimerait silencieusement des correspondances.
    """
    df = pd.read_excel(path, sheet_name=sheet, header=3,
                       keep_default_na=False, na_values=[])
    return df.replace({"": None})


def load_schema(path: Path | str = DICT_PATH) -> Schema:
    dic = _read(path, "Dictionnaire")
    dic = dic[dic["Variable"].notna()].copy()
    dic = dic[dic["Rôle"].isin(["identifiant", "cible", "feature", "exclu"])]
    dic["Variable"] = dic["Variable"].str.strip()

    raw = _read(path, "Modalites")
    raw = raw[raw["Variable"].notna()]

    mods: list[Modality] = []
    for _, r in raw.iterrows():
        alias_raw = r.get("Alias acceptés")
        alias = []
        if isinstance(alias_raw, str) and alias_raw.strip():
            alias = [a.strip() for a in alias_raw.split("|")
                     if a.strip() and a.strip() != "(vide)"]
        en = r.get("Libellé EN")
        mods.append(Modality(
            variable=str(r["Variable"]).strip(),
            valeur=str(r["Valeur canonique"]).strip(),
            code=r["Code"],
            ordre=r["Ordre"],
            alias=alias,
            libelle_en=en.strip() if isinstance(en, str) and en.strip() else None,
        ))

    forms = _read(path, "Formulaires")
    # les lignes de notes en bas de feuille n'ont pas de langue : on les écarte
    forms = forms[forms["Suffixe"].notna() & forms["Langue"].notna()].copy()
    forms["Suffixe"] = forms["Suffixe"].astype(str).str.strip()

    return Schema(dic=dic.reset_index(drop=True), mods=mods, forms=forms.reset_index(drop=True))


if __name__ == "__main__":
    s = load_schema()
    print(f"cible      : {s.target}")
    print(f"identifiant: {s.identifier}")
    print(f"exclues    : {s.excluded}")
    print(f"features   : {len(s.features)}")
    print(f"  ordinales: {len(s.ordinals)} | nominales/binaires: {len(s.nominals)}")
    print(f"modalités  : {len(s.mods)} sur {len(set(m.variable for m in s.mods))} variables")
    print(f"alias      : {len(s.alias_map())}")
    print(f"conditions : {s.conditional_rules()}")
    print(f"formulaires: {len(s.forms)} | clés FR {len(s.export_keys('fr'))} "
          f"| clés EN {len(s.export_keys('en'))}")
    todo = sum(1 for m in s.mods if m.libelle_en == TODO_EN)
    print(f"libellés EN restant à relever : {todo}")
