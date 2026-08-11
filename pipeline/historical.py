"""Chargement et mise en forme des données historiques pour la calibration.

Ce module produit le jeu de données central décrit au §4 du plan projet : pour
chaque observation historique, le triplet

    (institut, horizon = nb de jours avant le scrutin, écart = intention − résultat)

enrichi du bloc politique du candidat (via ``candidat_blocs.csv``) et de la
taille d'échantillon.

Source des sondages 2017/2022 : `data/parsed/intentions_{annee}_wiki.csv`
(sortie de `sondages.wiki`, la même source primaire que pour 2027 — cf. son
docstring pour pourquoi Wikipedia plutôt que pollsposition/nsppolls : bien plus
large couverture d'instituts, wikitext déjà normalisé). Les résultats officiels
et le mapping candidat -> bloc restent dans `data/historical/` (ne bougent pas,
ce sont des faits, pas des sondages).

Piège propre à la source Wikipedia : l'en-tête de colonne d'un sondage ne porte
souvent que le **nom de famille** du candidat ("Le Pen", "Dupont-Aignan"), pas
le nom complet utilisé par `results_XXXX.csv`/`candidat_blocs.csv` ("Marine Le
Pen"). `_resolve_candidate` résout ça par correspondance suffixe sur les tokens
(nom de famille = fin du nom complet) — un nom qui ne matche aucun candidat réel
de l'élection (colonne parasite, ligne « Autres candidats », etc.) reste
inchangé et sera naturellement écarté par `_select_best_hypothesis` (candidat
non réel).
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "historical"
PARSED_DIR = Path(__file__).resolve().parent.parent / "data" / "parsed"

# Dates officielles du premier tour, par élection. Sert au calcul de l'horizon.
ELECTION_DATES = {
    2017: "2017-04-23",
    2022: "2022-04-10",
}

# Sondages 2017/2022, sortie de `sondages.wiki` (schéma `sondages.schema.COLUMNS`).
POLL_FILES = {
    2017: "intentions_2017_wiki.csv",
    2022: "intentions_2022_wiki.csv",
}

# Résultats officiels finaux, par élection.
RESULT_FILES = {
    2017: "results_2017.csv",  # Conseil constitutionnel (via pollsposition)
    2022: "results_2022.csv",  # Ministère de l'Intérieur
}


def load_results() -> pd.DataFrame:
    """Résultats officiels finaux, toutes élections empilées."""
    return pd.concat(
        [pd.read_csv(DATA_DIR / f) for f in RESULT_FILES.values()],
        ignore_index=True,
    )


def load_blocs() -> pd.DataFrame:
    """Table de mapping candidat -> bloc politique, par élection."""
    return pd.read_csv(DATA_DIR / "candidat_blocs.csv")


def _fold(s: str) -> str:
    """Clé insensible aux accents / casse / traits d'union pour apparier les noms."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("-", " ").replace(".", " ")
    return " ".join(s.split())


def _resolve_candidate(raw: str, real_candidates: set[str]) -> str:
    """Résout un nom brut (parfois juste le nom de famille) vers le nom complet
    canonique d'un candidat réel de l'élection, par correspondance suffixe sur
    les tokens folés. Renvoie `raw` inchangé si aucune correspondance unique —
    la ligne sera écartée en aval, pas devinée."""
    toks = _fold(raw).split()
    if not toks:
        return raw
    matches = {c for c in real_candidates if _fold(c).split()[-len(toks):] == toks}
    return matches.pop() if len(matches) == 1 else raw


def _select_best_hypothesis(t1: pd.DataFrame, real_candidates: set[str]) -> pd.DataFrame:
    """Sélectionne, pour chaque sondage, l'hypothèse la plus proche du réel.

    Les sondages 2022 testent souvent plusieurs hypothèses de candidatures
    (Bertrand vs Pécresse, avec/sans Zemmour, ...). Pour comparer une intention
    à un résultat *réel*, on retient par sondage l'hypothèse qui contient le plus
    de candidats effectivement présents au scrutin — c.-à-d. le scénario le plus
    proche de ce qui s'est réellement passé — ce qui limite la contamination par
    des candidats putatifs. En cas d'égalité, on préfère l'hypothèse « par
    défaut » (``hypothese`` vide).
    """
    t1 = t1.copy()
    t1["_is_real"] = t1["candidat"].isin(real_candidates)
    # clé d'hypothèse robuste au NaN (hypothèse par défaut du sondage)
    t1["_hyp"] = t1["hypothese"].fillna("__defaut__")

    # nombre de candidats réels présents dans chaque (sondage, hypothèse)
    score = (
        t1[t1["_is_real"]]
        .groupby(["notice", "_hyp"])["candidat"]
        .nunique()
        .rename("n_real")
        .reset_index()
    )
    # tie-break : l'hypothèse par défaut d'abord
    score["_default_first"] = (score["_hyp"] != "__defaut__").astype(int)
    score = score.sort_values(
        ["notice", "n_real", "_default_first"], ascending=[True, False, True]
    )
    best = score.groupby("notice").first().reset_index()[["notice", "_hyp"]]

    keep = t1.merge(best, on=["notice", "_hyp"], how="inner")
    return keep[keep["_is_real"]].drop(columns=["_is_real"])


def load_calibration_frame(elections: list[int] | None = None) -> pd.DataFrame:
    """Construit le jeu de calibration (un écart par sondage × candidat réel).

    Colonnes de sortie :
        election, id (sondage), institut, commanditaire, fin_enquete,
        horizon (jours avant le 1er tour), echantillon, candidat, bloc,
        intention (%), resultat (%), ecart (= intention − resultat, en points).
    """
    elections = elections or sorted(POLL_FILES)
    results = load_results()
    blocs = load_blocs()

    frames = []
    for elec in elections:
        polls = pd.read_csv(PARSED_DIR / POLL_FILES[elec])
        polls["election"] = elec

        res_e = results[(results["election"] == elec) & (results["tour"] == "Premier tour")]
        real = set(res_e["candidat"])

        t1 = polls[polls["tour"] == "Premier tour"].copy()
        # Wikipedia n'expose souvent que le nom de famille -> résolution vers le
        # nom complet canonique AVANT le filtrage "candidat réel", sinon on perd
        # silencieusement toutes les lignes (aucun nom de famille ne matche un
        # nom complet par simple isin()).
        t1["candidat"] = t1["candidat"].apply(lambda raw: _resolve_candidate(raw, real))
        t1 = _select_best_hypothesis(t1, real)

        # date_fin manquante -> repli sur date_notice (cf. sondages/schema.py :
        # "repli fiable") ; le terrain n'est pas toujours détaillé sur Wikipedia.
        date_fin = pd.to_datetime(t1["date_fin"], errors="coerce")
        date_notice = pd.to_datetime(t1.get("date_notice"), errors="coerce")
        t1["fin_enquete"] = date_fin.fillna(date_notice)

        elec_date = pd.Timestamp(ELECTION_DATES[elec])
        t1["horizon"] = (elec_date - t1["fin_enquete"]).dt.days

        t1 = t1.merge(
            res_e[["candidat", "resultat"]], on="candidat", how="left"
        )
        t1 = t1.merge(
            blocs[blocs["election"] == elec][["candidat", "bloc"]],
            on="candidat",
            how="left",
        )
        t1["ecart"] = t1["intention"] - t1["resultat"]
        t1 = t1.rename(columns={"notice": "id"})

        frames.append(
            t1[
                [
                    "election",
                    "id",
                    "institut",
                    "commanditaire",
                    "fin_enquete",
                    "horizon",
                    "echantillon",
                    "candidat",
                    "bloc",
                    "intention",
                    "resultat",
                    "ecart",
                ]
            ]
        )

    out = pd.concat(frames, ignore_index=True)
    # on écarte les horizons négatifs/inconnus (dates manquantes ou sondages
    # post-scrutin) et les lignes sans bloc renseigné (candidat non mappé)
    out = out[(out["horizon"] >= 0) & out["bloc"].notna()].reset_index(drop=True)
    return out


if __name__ == "__main__":
    df = load_calibration_frame()
    print(f"{len(df)} observations de calibration")
    print(df.groupby(["institut", "bloc"]).size().unstack(fill_value=0))
