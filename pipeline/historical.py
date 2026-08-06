"""Chargement et mise en forme des données historiques pour la calibration.

Ce module produit le jeu de données central décrit au §4 du plan projet : pour
chaque observation historique, le triplet

    (institut, horizon = nb de jours avant le scrutin, écart = intention − résultat)

enrichi du bloc politique du candidat (via ``candidat_blocs.csv``) et de la
taille d'échantillon. Il est consommé à la fois par le notebook exploratoire
(Phase 0) et par le fit hiérarchique de calibration (Phase 1).

Limite connue : seule l'élection 2022 est disponible dans ``nsppolls/nsppolls``
(le CSV présidentielle ne remonte pas à 2017). Le code est écrit pour empiler
plusieurs élections dès que les données 2017 seront retrouvées — il suffira
d'ajouter les fichiers correspondants et une entrée dans ``ELECTION_DATES``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "historical"

# Dates officielles du premier tour, par élection. Sert au calcul de l'horizon.
ELECTION_DATES = {
    2017: "2017-04-23",
    2022: "2022-04-10",
}

# Fichiers de sondages historiques déjà parsés, par élection.
POLL_FILES = {
    2017: "presidentielle_2017_pollsposition.csv",  # source : pollsposition/data
    2022: "presidentielle_2022_nsppolls.csv",       # source : nsppolls/nsppolls
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
        .groupby(["id", "_hyp"])["candidat"]
        .nunique()
        .rename("n_real")
        .reset_index()
    )
    # tie-break : l'hypothèse par défaut d'abord
    score["_default_first"] = (score["_hyp"] != "__defaut__").astype(int)
    score = score.sort_values(
        ["id", "n_real", "_default_first"], ascending=[True, False, True]
    )
    best = score.groupby("id").first().reset_index()[["id", "_hyp"]]

    keep = t1.merge(best, on=["id", "_hyp"], how="inner")
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
        polls = pd.read_csv(DATA_DIR / POLL_FILES[elec])
        polls["election"] = elec

        res_e = results[(results["election"] == elec) & (results["tour"] == "Premier tour")]
        real = set(res_e["candidat"])

        t1 = polls[polls["tour"] == "Premier tour"].copy()
        t1 = _select_best_hypothesis(t1, real)

        elec_date = pd.Timestamp(ELECTION_DATES[elec])
        t1["fin_enquete"] = pd.to_datetime(t1["fin_enquete"])
        t1["horizon"] = (elec_date - t1["fin_enquete"]).dt.days

        t1 = t1.merge(
            res_e[["candidat", "resultat"]], on="candidat", how="left"
        )
        t1 = t1.merge(
            blocs[blocs["election"] == elec][["candidat", "bloc"]],
            on="candidat",
            how="left",
        )
        t1["ecart"] = t1["intentions"] - t1["resultat"]
        t1 = t1.rename(columns={"nom_institut": "institut", "intentions": "intention"})

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
    # on écarte les horizons négatifs éventuels (sondages post-scrutin) et les
    # lignes sans bloc renseigné (candidat non mappé)
    out = out[(out["horizon"] >= 0) & out["bloc"].notna()].reset_index(drop=True)
    return out


if __name__ == "__main__":
    df = load_calibration_frame()
    print(f"{len(df)} observations de calibration")
    print(df.groupby(["institut", "bloc"]).size().unstack(fill_value=0))
