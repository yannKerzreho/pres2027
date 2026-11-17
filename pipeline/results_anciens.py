"""Résultats officiels 2002, 2007, 2012 (% des exprimés, T1 et T2) depuis
`data/historical/sources/resultats_pollsposition.json` (voix et exprimés du
Conseil constitutionnel, URL source dans le JSON) -> `results_{annee}.csv`,
au schéma de `results_2017.csv`.

Le JSON a des coquilles de noms (corrigées ici pour coller aux noms des
sondages Wikipedia) et de dates (non reprises : le CSV n'a pas de date).
Contrôle : la somme des voix égale les exprimés publiés.

    python -m pipeline.results_anciens
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "historical"
NOMS = {"Brunot Mégret": "Bruno Mégret", "Corrine Lepage": "Corinne Lepage",
        "Christine Taubira": "Christiane Taubira", "Marie-Georges Buffet": "Marie-George Buffet"}
PARTIS = {"Bruno Mégret": "Mouvement national républicain", "Christiane Taubira": "Parti radical de gauche",
          "Corinne Lepage": "Cap21", "Philippe Poutou": "Nouveau parti anticapitaliste"}
TOURS = (("premier_tour", "Premier tour"), ("second_tour", "Deuxième tour"))


def results(annee: str, src: dict) -> pd.DataFrame:
    e = src[annee]
    partis = {NOMS.get(c["nom"], c["nom"]): c["parti"] for c in e["candidats"]}
    rows = []
    for cle, tour in TOURS:
        t = e[cle]
        assert sum(t["resultats"].values()) == t["exprimes"], (annee, cle)
        for nom, voix in sorted(t["resultats"].items(), key=lambda kv: -kv[1]):
            nom = NOMS.get(nom, nom)
            rows.append({"election": int(annee), "tour": tour, "candidat": nom,
                         "parti": partis.get(nom) or PARTIS.get(nom, ""),
                         "resultat": round(100 * voix / t["exprimes"], 2)})
    return pd.DataFrame(rows)


def main():
    src = json.loads((DATA / "sources" / "resultats_pollsposition.json").read_text())
    for annee in ("2002", "2007", "2012"):
        out = DATA / f"results_{annee}.csv"
        results(annee, src).to_csv(out, index=False)
        print(out)


if __name__ == "__main__":
    main()
