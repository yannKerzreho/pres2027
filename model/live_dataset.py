"""Phase 3 — mise en forme des sondages 2027 pour le modèle live.

Transforme `data/parsed/intentions_2027.csv` (sortie du parsing PDF) en
observations exploitables par le modèle bayésien :

  - **canonicalisation** des noms (accents / traits d'union hétérogènes) ;
  - projection candidat -> **slot de candidature** : on fusionne les alternatives
    mutuellement exclusives (le RN présente Le Pen *ou* Bardella, le centre Attal
    *ou* Philippe, ...) en un seul slot, ce qui rend tous les sondages comparables
    quelle que soit l'hypothèse testée ;
  - agrégation par (sondage × hypothèse) puis moyenne par sondage → un vecteur de
    parts par slot et par sondage ;
  - calcul de l'horizon (jours avant le 1er tour du 18 avril 2027).

Chaque slot porte son **bloc politique**, ce qui permet d'appliquer les house
effects calibrés (biais/variance par institut × bloc) de la Phase 1.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

PARSED = Path(__file__).resolve().parent.parent / "data" / "parsed" / "intentions_2027.csv"
ELECTION_T1 = pd.Timestamp("2027-04-18")

# slot -> (bloc, liste de candidats fusionnés). Les alternatives qui ne coexistent
# jamais dans une même hypothèse sont regroupées en un slot unique.
SLOTS = {
    "Arthaud (LO)":        ("gauche_radicale", ["Nathalie Arthaud"]),
    "Poutou (NPA)":        ("gauche_radicale", ["Philippe Poutou"]),
    "Mélenchon (LFI)":     ("gauche_radicale", ["Jean-Luc Mélenchon"]),
    "Ruffin":              ("gauche_radicale", ["François Ruffin"]),
    "Roussel (PCF)":       ("gauche", ["Fabien Roussel"]),
    "PS-PP (gauche)":      ("gauche", ["Raphaël Glucksmann", "Olivier Faure", "François Hollande"]),
    "Tondelier (EELV)":    ("ecologistes", ["Marine Tondelier"]),
    "Centre":              ("centre", ["Gabriel Attal", "Édouard Philippe",
                                        "Sébastien Lecornu", "Gérald Darmanin"]),
    "LR (droite)":         ("droite", ["Bruno Retailleau", "Laurent Wauquiez"]),
    "Dupont-Aignan":       ("droite_radicale", ["Nicolas Dupont-Aignan"]),
    "RN":                  ("droite_radicale", ["Marine Le Pen", "Jordan Bardella"]),
    "Zemmour (Reconquête)": ("droite_radicale", ["Éric Zemmour"]),
}


def _fold(s: str) -> str:
    """Clé insensible aux accents / casse / traits d'union pour apparier les noms."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("-", " ").replace(".", " ")
    return " ".join(s.split())


# index : nom canonique folé -> (slot, bloc)
_CANDIDATE_INDEX = {}
for slot, (bloc, names) in SLOTS.items():
    for nm in names:
        _CANDIDATE_INDEX[_fold(nm)] = (slot, bloc)


def map_candidate(name: str) -> tuple[str | None, str | None]:
    return _CANDIDATE_INDEX.get(_fold(name), (None, None))


def load_live_observations(path: Path | None = None,
                           tour: str = "Premier tour") -> pd.DataFrame:
    """Retourne un vecteur de parts par slot et par sondage (1 ligne = 1 slot ×
    1 sondage), avec institut, horizon et bloc — prêt pour le modèle."""
    df = pd.read_csv(path or PARSED)
    df = df[df["tour"] == tour].copy()

    slot_bloc = df["candidat"].apply(map_candidate)
    df["slot"] = slot_bloc.apply(lambda x: x[0])
    df["bloc"] = slot_bloc.apply(lambda x: x[1])
    unmapped = sorted(df.loc[df["slot"].isna(), "candidat"].unique())
    if unmapped:
        print(f"[live_dataset] candidats non mappés (ignorés) : {unmapped}")
    df = df[df["slot"].notna()]

    # 1) somme par (sondage, hypothèse, slot) — fusionne les alternatives
    g = (df.groupby(["notice", "institut", "date_fin", "echantillon",
                     "hypothese", "slot", "bloc"], dropna=False)["intention"]
           .sum().reset_index())
    # 2) moyenne sur les hypothèses d'un même sondage -> un vecteur par sondage
    obs = (g.groupby(["notice", "institut", "date_fin", "echantillon", "slot", "bloc"],
                     dropna=False)["intention"].mean().reset_index())

    # horizon (jours avant le 1er tour)
    obs["date_fin"] = pd.to_datetime(obs["date_fin"], errors="coerce")
    obs["horizon"] = (ELECTION_T1 - obs["date_fin"]).dt.days
    obs = obs[obs["horizon"].notna() & (obs["horizon"] >= 0)]
    obs["echantillon"] = obs["echantillon"].fillna(obs["echantillon"].median())
    return obs.rename(columns={"intention": "part"}).reset_index(drop=True)


if __name__ == "__main__":
    obs = load_live_observations()
    print(f"{obs['notice'].nunique()} sondages, {len(obs)} observations slot×sondage")
    print("\nMoyenne des parts par slot (brut, non débiaisé) :")
    m = obs.groupby("slot")["part"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    print(m.round(1).to_string())
    print(f"\nHorizons (jours avant T1) : {obs['horizon'].min()} → {obs['horizon'].max()}")
