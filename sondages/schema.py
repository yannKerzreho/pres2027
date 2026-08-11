"""Contrat de sortie du package `sondages` (schéma stable, documenté).

Une ligne = un candidat × une hypothèse × un sondage. Données **brutes et
riches** (niveau candidat, jamais agrégées) + métadonnées de source pour la
traçabilité. C'est l'interface publique que les consommateurs (modèles, front,
tiers) peuvent tenir pour stable.
"""

from __future__ import annotations

import pandas as pd

# Ordre et signification des colonnes (contrat).
COLUMNS: dict[str, str] = {
    "notice_id":     "identifiant de la notice (préfixe numérique du nom)",
    "notice":        "nom de la notice (Commission des sondages)",
    "notice_url":    "URL du PDF de la notice (source vérifiable)",
    "institut":      "institut de sondage (normalisé)",
    "commanditaire": "commanditaire (média), si extrait",
    "date_debut":    "début du terrain (AAAA-MM-JJ), si extrait",
    "date_fin":      "fin du terrain (AAAA-MM-JJ) ; repli sur la date de notice",
    "date_notice":   "date de la notice (index NSPPolls), repli fiable",
    "echantillon":   "taille d'échantillon (inscrits), si extraite",
    "methode":       "mode de recueil (internet / téléphone), si extrait",
    "tour":          "'Premier tour' ou 'Deuxième tour'",
    "hypothese":     "hypothèse de candidatures testée (ou None)",
    "candidat":      "nom du candidat",
    "intention":     "intention de vote en % des suffrages exprimés",
}

TOURS = {"Premier tour", "Deuxième tour"}


def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=list(COLUMNS))


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Valide le dataset contre le contrat (lève AssertionError sinon)."""
    missing = set(COLUMNS) - set(df.columns)
    assert not missing, f"colonnes manquantes : {sorted(missing)}"
    if df.empty:
        return df
    assert df["candidat"].notna().all(), "candidat manquant"
    assert df["institut"].notna().all(), "institut manquant"
    assert df["notice_url"].notna().all(), "notice_url manquant (traçabilité)"
    assert df["tour"].isin(TOURS).all(), "valeur de 'tour' invalide"
    assert df["intention"].between(0, 100).all(), "intention hors [0, 100]"
    # cohérence : chaque (sondage, hypothèse, tour) somme à ~100 %
    g = df.groupby(["notice", "hypothese", "tour"], dropna=False)["intention"].sum()
    bad = g[(g < 90) | (g > 110)]
    assert bad.empty, f"sommes d'intentions non ~100 % : {bad.to_dict()}"
    return df
