"""
Import des exports Framaforms réels.

Sept formulaires en ligne collectent en parallèle : six en français (le lien
initial 989 plus cinq clones par canal) et un en anglais. Chaque export est un
TSV dont les en-têtes sont des clés opaques (`new_<horodatage>`) propres au
formulaire. Ce module les ramène tous au vocabulaire du dictionnaire.

Le dictionnaire reste la seule source de vérité : correspondance clé -> variable
(colonnes Cle_export_fr / Cle_export_en), rattachement fichier -> canal et sid de
test à écarter (feuille Formulaires), libellés anglais (colonne Libellé EN).
Rien n'est codé en dur ici, pas même les noms de fichiers.

Sortie : un DataFrame au format attendu par validate.py, prêt pour C-13 et la suite.

    python src/ingest.py            # importe data/real/*.tsv -> out/real_raw.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from config import NA_APPLICABLE, ROOT, TODO_EN, Schema, load_schema

REAL_DIR = ROOT / "data" / "real"
OUT_DIR = ROOT / "out"

# Les deux premières lignes de l'export Webform sont un cartouche, pas des données.
SKIPROWS = 2

# Colonnes techniques ajoutées par Webform. L'adresse IP et l'identité de compte
# ne doivent jamais arriver jusqu'ici : elles sont décochées à l'export.
WEBFORM_META = ["webform_serial", "webform_sid", "webform_time",
                "webform_completed_time", "webform_modified_time",
                "webform_draft", "webform_uid", "webform_username"]
FORBIDDEN = ["webform_ip_address", "webform_remote_addr", "adresse_ip"]


class IngestError(RuntimeError):
    pass


def _suffix(path: Path) -> str:
    """Suffixe du lien, déduit du nom de fichier (`526.tsv`, `flyer_endomind_559.tsv`)."""
    hits = re.findall(r"(\d{3})(?=\D*$)", path.stem)
    if not hits:
        raise IngestError(
            f"{path.name} : impossible d'en déduire un suffixe de lien. "
            f"Nommer le fichier d'après les trois derniers chiffres du lien."
        )
    return hits[-1]


def read_export(path: Path) -> pd.DataFrame:
    """Lit un TSV Framaforms brut, sans rien interpréter.

    dtype=str et keep_default_na=False sont indispensables : sans eux pandas
    convertit « 0 » en entier (Parite est une chaîne canonique) et fabrique des
    NaN, ce qui écraserait la distinction entre nos trois valeurs manquantes.
    """
    df = pd.read_csv(path, sep="\t", skiprows=SKIPROWS, dtype=str,
                     keep_default_na=False, encoding="utf-8")
    leaked = [c for c in df.columns if c.lower() in FORBIDDEN]
    if leaked:
        raise IngestError(
            f"{path.name} : colonnes identifiantes présentes {leaked}. "
            f"Réexporter en décochant « Adresse IP » et les identifiants de compte."
        )
    return df


def c14_structure(df: pd.DataFrame, schema: Schema, suffix: str) -> dict[str, str]:
    """C-14 — intégrité structurelle : les clés du fichier sont-elles celles attendues ?

    BLOQUANT. C'est la seule défense contre le décalage silencieux de colonnes :
    un formulaire modifié en cours de collecte produit un export qui se lit sans
    la moindre erreur, mais dont les colonnes ne portent plus le bon contenu.
    Rien dans le fichier ne signale la bascule.
    """
    form = schema.form(suffix)
    lang = str(form["Langue"]).strip()
    keys = schema.export_keys(lang)

    present = [c for c in df.columns if c.startswith("new_")]
    missing = sorted(set(keys) - set(present))
    if missing:
        raise IngestError(
            f"{suffix} : {len(missing)} clé(s) du dictionnaire absente(s) de l'export "
            f"{missing[:3]}{'…' if len(missing) > 3 else ''}. "
            f"Le formulaire a probablement été modifié — NE PAS importer ce fichier "
            f"avant d'avoir tranché quelles lignes suivent quelle structure."
        )
    return keys


def load_form(path: Path, schema: Schema) -> pd.DataFrame:
    """Un fichier -> un DataFrame au vocabulaire du dictionnaire."""
    suffix = _suffix(path)
    form = schema.form(suffix)
    lang = str(form["Langue"]).strip()
    canal = str(form["Source_canal"]).strip()

    df = read_export(path)
    keys = c14_structure(df, schema, suffix)

    # Renommage par CLÉ, jamais par position : l'ordre des colonnes n'est pas un contrat.
    out = df[list(keys)].rename(columns=keys).copy()

    # Framaforms restitue certains libellés avec une espace parasite (« ≥45 »).
    # On normalise ici plutôt que par un alias : une espace de bord ne survit pas
    # au découpage de la colonne « Alias acceptés », et c'est de toute façon du
    # nettoyage de lecture, pas une équivalence de vocabulaire.
    for c in out.columns:
        out[c] = out[c].str.strip()

    out = fill_structural_blanks(out, schema)

    # Traçabilité : conservés jusqu'à la déduplication, écartés avant modélisation.
    out["_sid"] = df["webform_sid"]
    out["_date"] = pd.to_datetime(df["webform_time"], format="%d/%m/%Y - %H:%M",
                                  errors="coerce")
    out["Source_canal"] = canal
    out["Langue_formulaire"] = lang

    # Soumissions de test faites pour valider les liens : identifiées par sid,
    # jamais par leur contenu (une vraie réponse peut ressembler à un test).
    test_sids = {s.strip() for s in str(form["sid de test à exclure"]).split(",")
                 if s.strip() and s.strip() != "—"}
    if test_sids:
        n = int(out["_sid"].isin(test_sids).sum())
        out = out[~out["_sid"].isin(test_sids)]
        if n:
            print(f"    {suffix} : {n} soumission(s) de test écartée(s)")

    if lang == "en":
        out = translate_en(out, schema, suffix)

    return out


def fill_structural_blanks(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Une question non affichée ressort vide de l'export : on lui rend son sens.

    Une cellule vide sur une variable conditionnelle ne veut pas dire « pas de
    réponse » mais « la question n'a jamais été posée » — c'est NA_non_applicable,
    et c'est ce que C-06 exige de distinguer des deux autres valeurs manquantes.
    La valeur de remplacement est lue dans le dictionnaire (modalité dont le code
    vaut NA_non_applicable), jamais écrite en dur.
    """
    df = df.copy()
    for var in schema.conditional_rules():
        if var not in df.columns:
            continue
        label = next((m.valeur for m in schema.mods
                      if m.variable == var and m.code == NA_APPLICABLE), None)
        if label is None:
            raise IngestError(
                f"{var} est conditionnelle mais aucune de ses modalités ne porte le "
                f"code {NA_APPLICABLE} : compléter la feuille Modalites."
            )
        df.loc[df[var] == "", var] = label
    return df


def translate_en(df: pd.DataFrame, schema: Schema, suffix: str) -> pd.DataFrame:
    """Ramène les libellés du formulaire anglais aux valeurs canoniques françaises.

    Variable par variable : hors contexte, « Normal » vaut « Normale » pour
    Abondance_regles et « Normal (18,5-24,9) » pour IMC_categorie. Une table
    globale confondrait les deux.

    Un libellé anglais inconnu n'est PAS traduit : il est laissé tel quel et
    signalé, pour que C-05 le rejette plus loin. Deviner une correspondance
    reviendrait à inventer de la donnée.
    """
    df = df.copy()
    unknown: dict[str, list[str]] = {}
    for var in df.columns:
        emap = schema.en_map(var)
        if not emap:
            continue
        canon = set(schema.values_of(var))
        col = df[var].astype(object)
        bad = sorted({v for v in col.unique()
                      if isinstance(v, str) and v and v not in canon and v not in emap})
        if bad:
            unknown[var] = bad
        df[var] = col.map(lambda v: emap.get(v, v))

    if unknown:
        todo = sum(1 for m in schema.mods if m.libelle_en == TODO_EN)
        detail = "; ".join(f"{v}: {vals}" for v, vals in unknown.items())
        print(f"    {suffix} : libellés EN non traduits -> {detail}")
        print(f"    ({todo} modalités encore marquées « {TODO_EN} » dans le dictionnaire — "
              f"les relever sur la page publique du formulaire, ne pas les deviner)")
    return df


def assign_ids(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Attribue ID_anonyme (E001, E002…).

    Tri par (Source_canal, sid). Le sid est l'identifiant auto-incrémenté de
    Framaforms : stable, indépendant de l'ordre de lecture des fichiers, et sans
    lien avec l'identité de la répondante. Le même jeu de fichiers redonne donc
    toujours les mêmes ID — condition nécessaire pour qu'un résultat soit
    reproductible et qu'une ligne signalée reste retrouvable d'une exécution à
    l'autre.

    Volontairement PAS dérivé d'un horodatage : ce serait réintroduire une
    information temporelle dans un champ censé n'en porter aucune.
    """
    df = df.sort_values(["Source_canal", "_sid"], kind="mergesort").reset_index(drop=True)
    df.insert(0, schema.identifier, [f"E{i:03d}" for i in range(1, len(df) + 1)])
    return df


def ingest(directory: Path | str = REAL_DIR, schema: Schema | None = None) -> pd.DataFrame:
    """Importe tous les exports d'un répertoire et les empile."""
    schema = schema or load_schema()
    directory = Path(directory)

    files = sorted(p for p in directory.glob("*.tsv"))
    if not files:
        raise IngestError(f"aucun .tsv dans {directory}")

    known = set(schema.forms["Suffixe"])
    frames, seen = [], []
    for path in files:
        suffix = _suffix(path)
        if suffix not in known:
            raise IngestError(
                f"{path.name} : suffixe {suffix!r} inconnu du registre. "
                f"Ajouter une ligne à la feuille Formulaires du dictionnaire "
                f"AVANT d'importer — un canal non déclaré est un canal non traçable."
            )
        if suffix in seen:
            raise IngestError(f"suffixe {suffix!r} présent deux fois dans {directory}")
        seen.append(suffix)

        part = load_form(path, schema)
        form = schema.form(suffix)
        print(f"  {path.name:24s} -> {suffix} | {str(form['Source_canal']):16s} "
              f"| {form['Langue']} | {len(part):4d} réponses")
        if len(part):
            frames.append(part)

    df = pd.concat(frames, ignore_index=True)

    absent = sorted(known - set(seen))
    if absent:
        print(f"  formulaires du registre non importés : {absent}")

    return assign_ids(df, schema)


def main():
    schema = load_schema()
    print(f"registre : {len(schema.forms)} formulaires déclarés")
    df = ingest(schema=schema)

    OUT_DIR.mkdir(exist_ok=True)
    dest = OUT_DIR / "real_raw.csv"
    df.to_csv(dest, index=False, encoding="utf-8")

    print(f"\n{len(df)} réponses importées -> {dest}")
    print("\nrépartition par canal :")
    print(pd.crosstab(df["Source_canal"], df[schema.target],
                      margins=True).to_string())
    print("\nLes libellés ne sont pas encore normalisés : C-13 s'en charge, "
          "puis C-05 rejette ce qui reste hors dictionnaire.")


if __name__ == "__main__":
    main()
