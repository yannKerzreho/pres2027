"""Ingestion des données 2017 (source : pollsposition/data).

Convertit :
  - sondages   : sources/presidentielles_2017_pollsposition.json
  - résultats  : sources/resultats_pollsposition.json
vers le même schéma que le fichier nsppolls 2022, pour que
`pipeline/historical.py` empile 2017 et 2022 sans traitement spécial.

Points d'attention sur la source :
  - Le champ `date_fin` du JSON 2017 est corrompu (année 2021). On utilise donc
    la date encodée dans la clé du sondage (AAAAMMJJ), fiable, comme fin
    d'enquête.
  - Les noms d'instituts sont normalisés sur l'orthographe 2022 (« Opinionway »
    -> « Opinion Way », etc.) pour que le même institut soit mutualisé d'une
    élection à l'autre dans le modèle hiérarchique.
  - Couverture : ~5 dernières semaines avant le 1er tour (16 mars -> 5 mai 2017).
    Peu d'horizons longs : 2017 informe surtout le biais près du scrutin.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "historical"
SRC = DATA_DIR / "sources"

# Alignement des noms d'instituts sur l'orthographe du fichier 2022 (nsppolls).
INSTITUT_RENAME = {
    "Opinionway": "Opinion Way",
    "Harris interactive": "Harris Interactive",
    "Kantar": "Kantar Public",
}


def convert_polls() -> pd.DataFrame:
    raw = json.loads((SRC / "presidentielles_2017_pollsposition.json").read_text())["sondages"]
    rows = []
    tour_map = {"premier_tour": "Premier tour", "second_tour": "Deuxième tour"}
    for key, poll in raw.items():
        institut = INSTITUT_RENAME.get(poll["institut"], poll["institut"])
        date_str = key.split("_")[0]  # AAAAMMJJ, fiable (contrairement à date_fin)
        fin = pd.to_datetime(date_str, format="%Y%m%d").date().isoformat()
        n = poll.get("interroges")
        for tkey, tour in tour_map.items():
            hyps = poll.get(tkey) or []
            for h_idx, hyp in enumerate(hyps):
                # hypothèse : None si une seule, sinon index (comme nsppolls)
                hypothese = None if len(hyps) == 1 else f"hyp_{h_idx}"
                for cand, pct in hyp["intentions"].items():
                    rows.append(
                        {
                            "candidat": cand,
                            "parti": None,
                            "intentions": float(pct),
                            "id": key,
                            "nom_institut": institut,
                            "commanditaire": ", ".join(poll.get("commanditaires", [])),
                            "fin_enquete": fin,
                            "echantillon": n,
                            "tour": tour,
                            "hypothese": hypothese,
                        }
                    )
    return pd.DataFrame(rows)


def convert_results() -> pd.DataFrame:
    res = json.loads((SRC / "resultats_pollsposition.json").read_text())["2017"]
    rows = []
    for tkey, tour in (("premier_tour", "Premier tour"), ("second_tour", "Deuxième tour")):
        block = res[tkey]
        exprimes = block["exprimes"]
        for cand, votes in block["resultats"].items():
            rows.append(
                {
                    "election": 2017,
                    "tour": tour,
                    "candidat": cand,
                    "parti": None,
                    "resultat": round(100 * votes / exprimes, 2),
                }
            )
    # partis depuis la liste candidats
    partis = {c["nom"]: c["parti"] for c in res["candidats"]}
    df = pd.DataFrame(rows)
    df["parti"] = df["candidat"].map(partis).fillna("")
    return df


def main() -> None:
    polls = convert_polls()
    polls.to_csv(DATA_DIR / "presidentielle_2017_pollsposition.csv", index=False)
    print(f"Sondages 2017 : {polls['id'].nunique()} sondages, {len(polls)} lignes "
          f"-> presidentielle_2017_pollsposition.csv")

    results = convert_results()
    results.to_csv(DATA_DIR / "results_2017.csv", index=False)
    print("Résultats 2017 (1er tour, % exprimés) :")
    t1 = results[results.tour == "Premier tour"].sort_values("resultat", ascending=False)
    print(t1[["candidat", "resultat"]].to_string(index=False))


if __name__ == "__main__":
    main()
