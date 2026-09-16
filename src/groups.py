"""
Single source of truth for the thematic groupings (A0).

Nothing else in the pipeline is allowed to define groups. Import from here, so a
change to a grouping propagates everywhere instead of silently disagreeing with a
second copy. `check_goldstein_coverage` is called by the scripts that use these
definitions and fails loudly on a mismatch.

Design rationale, the members of each group and the open questions are in
docs/A0_goldstein_grouping_v1.md and docs/A0_our22_grouping_v1.md.

TODO (agreed in PROGRESS): OURS_SYMPTOM / OURS_CONTEXT should be read from the
data dictionary via new Groupe_L1 / Groupe_L2 columns rather than hardcoded here.
The columns do not exist in v5 yet. GOLDSTEIN stays hardcoded because it describes
an external dataset that our dictionary does not cover.
"""
from __future__ import annotations

# --- Goldstein 2023, 58 variables into 15 themes -------------------------
GOLDSTEIN = {
    "P1": ["Menstrual pain (Dysmenorrhea)", "Painful cramps during period",
           "Cramping", "Stomach cramping"],
    "P2": ["Pelvic pain", "Abdominal pain / pressure", "Painful ovulation",
           "Vaginal Pain/Pressure"],
    "P3": ["Painful / Burning pain during sex (Dyspareunia)", "Pain after Intercourse",
           "Abdominal Cramps during Intercourse", "Painful bowel movements", "Bowel pain"],
    "P4": ["Pain / Chronic pain", "Extreme / Severe pain", "Sharp / Stabbing pain"],
    "O1": ["Diarrhea", "Constipation / Chronic constipation", "Digestive / GI problems",
           "IBS-like symptoms", "Bloating"],
    "O2": ["Painful urination"],
    "M1": ["Heavy / Extreme menstrual bleeding", "Bleeding", "Excessive bleeding",
           "Constant bleeding", "Abnormal uterine bleeding", "Menstrual clots"],
    "M2": ["Long menstruation", "Irregular / Missed periods"],
    "R1": ["Infertility", "Fertility Issues"],
    "X1": ["Ovarian cysts", "Cysts (unspecified)", "Irritable Bowel Syndrome (IBS)",
           "Anaemia / Iron deficiency"],
    "N1": ["Back pain", "Lower back pain", "Leg pain", "Hip pain"],
    "N2": ["Migraines", "Headaches"],
    "C1": ["Fatigue / Chronic fatigue", "Decreased energy / Exhaustion",
           "Malaise / Sickness", "Feeling sick", "Dizziness",
           "Syncope (fainting, passing out)", "Fever", "Nausea",
           "Vomiting / constant vomiting", "Loss of appetite"],
    "C2": ["Depression", "Anxiety", "Mood swings", "Insomnia / Sleeplessness"],
    "Z1": ["Hormonal problems", "Acne / pimples"],
}

GOLDSTEIN_LABELS = {
    "P1": "menstrual pain (typical core)",
    "P2": "pelvic / abdominal pain",
    "P3": "deep provoked pain (DIE marker)",
    "P4": "unlocalised pain descriptors",
    "O1": "digestive function",
    "O2": "urinary",
    "M1": "bleeding volume and pattern",
    "M2": "cycle timing",
    "R1": "reproductive outcome",
    "X1": "requires prior medical investigation",
    "N1": "referred / musculoskeletal pain",
    "N2": "neurological / headache",
    "C1": "systemic / constitutional",
    "C2": "psychological / sleep",
    "Z1": "endocrine / skin",
}

# Excluded from severity and atypicality scores.
#   P4 is pure pain intensity, so including it would make severity circular.
#   X1 encodes healthcare access rather than symptom burden.
EXCLUDE_FROM_SCORES = ["P4", "X1"]

AX_TYP = ["P1"]                  # typical presentation
AX_ORG = ["P3", "O1", "O2"]      # organ involvement: deep, digestive, urinary

# --- our questionnaire, 22 features into 11 groups -----------------------
# Symptom side only feeds the severity / atypicality scores. Background variables
# sit before the disease in the causal chain, so they take no part in deciding
# which cases "already look similar to negatives".
OURS_SYMPTOM = {
    "S1": ["Dysmenorrhee", "Douleur_pelvienne"],
    "S2": ["Dyspareunie", "Dyschesie"],
    "S3": ["Symptomes_cycliques"],
    "S4": ["Abondance_regles", "Duree_regles", "Longueur_cycle", "Regularite_cycle"],
    "S5": ["Infertilite"],
    "S6": ["Age_apparition_symptomes"],
}
OURS_CONTEXT = {
    "B1": ["Tranche_age", "IMC_categorie"],
    "B2": ["Age_menarche", "Parite", "Contraception_hormonale"],
    "B3a": ["Antecedent_familial"],
    "B3b": ["Tabagisme", "Activite_physique", "Consommation_alcool"],
    "B4": ["SII", "Maladie_auto_immune"],
}
OURS_LABELS = {
    "S1": "core pelvic pain (typical)", "S2": "deep provoked pain",
    "S3": "cyclic organ symptoms", "S4": "menstrual pattern",
    "S5": "reproductive outcome", "S6": "symptom onset age",
    "B1": "demographics / body", "B2": "gynaecological & reproductive history",
    "B3a": "genetic predisposition", "B3b": "lifestyle", "B4": "comorbidity",
}

# --- cross-dataset alignment, 9 shared themes ----------------------------
# `rep` is chosen by closeness of wording to our question, NOT by discriminative
# power. This rule is pre-registered: see docs/A0_goldstein_grouping_v1.md. In 8
# of 9 themes the word-match happens to also be the strongest member; T5 is the
# counter-example (we take the 4th strongest), which is the evidence that the
# selection was not made on the outcome.
ALIGNMENT = {
    "T1": dict(theme="menstrual pain", gs_group="P1",
               rep="Menstrual pain (Dysmenorrhea)",
               ours="Dysmenorrhee", rule=">=2 (Moderee)"),
    "T2": dict(theme="pelvic / abdominal pain", gs_group="P2",
               rep="Pelvic pain",
               ours="Douleur_pelvienne", rule=">=2 (Moderee)"),
    "T3": dict(theme="pain during sex", gs_group="P3",
               rep="Painful / Burning pain during sex (Dyspareunia)",
               ours="Dyspareunie", rule="Oui"),
    "T4": dict(theme="painful defecation", gs_group="P3",
               rep="Painful bowel movements",
               ours="Dyschesie", rule="Oui"),
    "T5": dict(theme="digestive / urinary", gs_group="O1+O2",
               rep="Digestive / GI problems",
               ours="Symptomes_cycliques", rule="Oui"),
    "T6": dict(theme="bleeding volume", gs_group="M1",
               rep="Heavy / Extreme menstrual bleeding",
               ours="Abondance_regles", rule=">=2 (Abondante)"),
    "T7": dict(theme="menstruation length", gs_group="M2",
               rep="Long menstruation",
               ours="Duree_regles", rule=">=2 (>7 jours)"),
    "T8": dict(theme="cycle regularity", gs_group="M2",
               rep="Irregular / Missed periods",
               ours="Regularite_cycle", rule="Irregulier"),
    "T9": dict(theme="infertility", gs_group="R1",
               rep="Infertility",
               ours="Infertilite", rule="Oui"),
}

# Goldstein variables covered by the alignment. The 29 that are not covered are
# referred pain, systemic, psychological, neurological, endocrine, unlocalised
# descriptors and requires-prior-investigation.
ALIGNED_GOLDSTEIN = (GOLDSTEIN["P1"] + GOLDSTEIN["P2"] + GOLDSTEIN["P3"]
                     + GOLDSTEIN["O1"] + GOLDSTEIN["O2"] + GOLDSTEIN["M1"]
                     + GOLDSTEIN["M2"] + GOLDSTEIN["R1"])
SHARED_SUBSET = [t["rep"] for t in ALIGNMENT.values()]


def check_goldstein_coverage(columns):
    """Fail loudly if the group definitions and the dataset disagree."""
    assigned = [v for lst in GOLDSTEIN.values() for v in lst]
    missing = [c for c in columns if c not in assigned]
    extra = [c for c in assigned if c not in columns]
    dup = len(assigned) - len(set(assigned))
    if missing or extra or dup:
        raise ValueError(f"group definition mismatch: missing={missing} "
                         f"extra={extra} duplicates={dup}")
    return len(assigned), len(GOLDSTEIN)


def check_ours_coverage(features):
    assigned = ([v for lst in OURS_SYMPTOM.values() for v in lst]
                + [v for lst in OURS_CONTEXT.values() for v in lst])
    missing = [c for c in features if c not in assigned]
    extra = [c for c in assigned if c not in features]
    if missing or extra:
        raise ValueError(f"group definition mismatch: missing={missing} extra={extra}")
    return len(assigned), len(OURS_SYMPTOM) + len(OURS_CONTEXT)


if __name__ == "__main__":
    g = sum(len(v) for v in GOLDSTEIN.values())
    o = (sum(len(v) for v in OURS_SYMPTOM.values())
         + sum(len(v) for v in OURS_CONTEXT.values()))
    print(f"Goldstein : {g} variables in {len(GOLDSTEIN)} groups")
    print(f"ours      : {o} variables in "
          f"{len(OURS_SYMPTOM) + len(OURS_CONTEXT)} groups "
          f"({sum(len(v) for v in OURS_SYMPTOM.values())} symptom, "
          f"{sum(len(v) for v in OURS_CONTEXT.values())} context)")
    print(f"alignment : {len(ALIGNMENT)} shared themes, "
          f"{len(ALIGNED_GOLDSTEIN)}/{g} Goldstein variables covered")
