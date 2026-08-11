"""Phase 3 — mise en forme des sondages 2027 pour le modèle live.

Transforme `data/parsed/intentions_2027_wiki.csv` (sortie de `sondages.wiki`,
source primaire — cf. son docstring) en observations exploitables par le
modèle bayésien :

  - **canonicalisation** des noms (accents / traits d'union hétérogènes) ;
  - projection candidat -> **slot de candidature** : là où plusieurs personnalités
    sont testées en alternative dans les sondages (le RN teste Le Pen *ou* Bardella,
    le centre Attal *ou* Philippe, ...), le modèle de base n'en retient QU'UNE —
    l'hypothèse la plus probable actuellement (cf. `SLOTS`) — plutôt que de les
    fusionner, ce qui diluerait des profils de vote différents ;
  - agrégation par (sondage × hypothèse) puis moyenne par sondage → un vecteur de
    parts par slot et par sondage ;
  - calcul de l'horizon (jours avant le 1er tour du 18 avril 2027).

Chaque slot porte son **bloc politique**, ce qui permet d'appliquer les house
effects calibrés (biais/variance par institut × bloc) de la Phase 1.
"""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

PARSED = Path(__file__).resolve().parents[2] / "data" / "parsed" / "intentions_2027_wiki.csv"
ELECTION_T1 = pd.Timestamp("2027-04-18")

# slot -> (bloc, liste de candidats fusionnés). Les alternatives qui ne coexistent
# jamais dans une même hypothèse sont regroupées en un slot unique.
#
# Choix du modèle DE BASE (décidé le 2026-08-09, cf. session de conception) :
# un seul candidat par bloc là où plusieurs personnalités sont testées en
# alternative (Centre, PS-PP, LR, RN) — l'hypothèse actuellement la plus
# probable, PAS une fusion qui dilue des profils de vote différents (Attal et
# Philippe n'ont pas du tout le même niveau). Modéliser explicitement "qui se
# présente" (reports de voix si un candidat change) est un chantier séparé,
# volontairement hors de ce modèle de suivi. Pour changer l'hypothèse
# retenue : modifier UNIQUEMENT la liste ci-dessous, rien d'autre dans le
# code n'a besoin de changer (`map_candidate` référence directement `SLOTS`).
SLOTS = {
    "Arthaud (LO)":        ("gauche_radicale", ["Nathalie Arthaud"]),
    "Poutou (NPA)":        ("gauche_radicale", ["Philippe Poutou"]),
    "Mélenchon (LFI)":     ("gauche_radicale", ["Jean-Luc Mélenchon"]),
    "Ruffin":              ("gauche_radicale", ["François Ruffin"]),
    "Roussel (PCF)":       ("gauche", ["Fabien Roussel"]),
    "Glucksmann (PP)":     ("gauche", ["Raphaël Glucksmann"]),          # alternatives écartées : Olivier Faure, François Hollande
    "Tondelier (EELV)":    ("ecologistes", ["Marine Tondelier"]),
    "Philippe (Horizons)": ("centre", ["Édouard Philippe"]),           # alternatives écartées : Gabriel Attal, Sébastien Lecornu, Gérald Darmanin
    "Retailleau (LR)":     ("droite", ["Bruno Retailleau"]),           # alternative écartée : Laurent Wauquiez (jamais testé en pratique)
    "Dupont-Aignan":       ("droite_radicale", ["Nicolas Dupont-Aignan"]),
    "RN":                  ("droite_radicale", ["Marine Le Pen"]),     # alternative écartée : Jordan Bardella (scénario d'inéligibilité)
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
# tokens (dans l'ordre) de chaque nom complet folé, pour le repli nom-de-famille
_CANDIDATE_TOKENS = [(key.split(), val) for key, val in _CANDIDATE_INDEX.items()]


def map_candidate(name: str) -> tuple[str | None, str | None]:
    """Résout un nom brut vers (slot, bloc). Match exact d'abord (nom complet
    folé) ; à défaut, repli **nom de famille** : la source Wikipedia des
    sondages ne publie que le nom de famille en en-tête de colonne ("Le Pen",
    "Dupont-Aignan"), pas le nom complet — on matche si les tokens du nom brut
    sont un SUFFIXE des tokens du nom complet (gère les noms composés)."""
    folded = _fold(name)
    hit = _CANDIDATE_INDEX.get(folded)
    if hit is not None:
        return hit
    toks = folded.split()
    if not toks:
        return (None, None)
    matches = {val for full_toks, val in _CANDIDATE_TOKENS if full_toks[-len(toks):] == toks}
    return matches.pop() if len(matches) == 1 else (None, None)


def load_live_observations(path: Path | None = None,
                           tour: str = "Premier tour",
                           as_of: str | None = None) -> pd.DataFrame:
    """Raccourci : sondages bruts -> agrégés en slots (compat)."""
    return aggregate_to_slots(load_raw_polls(path, tour, as_of))


def load_raw_polls(path: Path | None = None, tour: str = "Premier tour",
                   as_of: str | None = None) -> pd.DataFrame:
    """Sondages 2027 **bruts et riches** passés aux modèles : niveau candidat, avec
    institut, commanditaire, echantillon, methode, hypothese, date_fin et horizon.

    On ne fusionne PAS les slots ici : un modèle peut avoir besoin de l'info brute
    (taille d'échantillon, institut, méthode, candidat exact, hypothèse) pour son
    propre algorithme. L'agrégation en slots est un helper séparé (`aggregate_to_slots`).

    as_of : ne garde que les sondages terminés à cette date ou avant (backfill)."""
    df = pd.read_csv(path or PARSED)
    df = df[df["tour"] == tour].copy()
    # date_fin manquante -> repli sur date_notice (cf. sondages/schema.py : "repli
    # fiable"), sinon une bonne partie des sondages wiki (terrain non détaillé)
    # serait perdue faute de date exploitable.
    date_fin = pd.to_datetime(df["date_fin"], errors="coerce")
    date_notice = pd.to_datetime(df.get("date_notice"), errors="coerce")
    df["date_fin"] = date_fin.fillna(date_notice)
    if as_of is not None:
        df = df[df["date_fin"] <= pd.Timestamp(as_of)]
    df["horizon"] = (ELECTION_T1 - df["date_fin"]).dt.days
    df = df[df["horizon"].notna() & (df["horizon"] >= 0)]
    # slot/bloc ajoutés comme colonnes (utiles) mais SANS agréger — le brut reste
    sb = df["candidat"].apply(map_candidate)
    df["slot"] = sb.apply(lambda x: x[0])
    df["bloc"] = sb.apply(lambda x: x[1])
    # identifiant stable du scénario testé (institut × date × hypothèse) : un
    # modèle qui veut raisonner par scénario (masque de présence/absence des
    # candidats, reports de voix...) fait un simple groupby("scenario_id")
    # au lieu de reconstruire une clé composite à partir de plusieurs colonnes.
    scenario_key = (df["institut"].astype(str) + "|" + df["date_fin"].astype(str)
                    + "|" + df["hypothese"].astype(str))
    df["scenario_id"] = scenario_key.apply(lambda s: hashlib.sha1(s.encode()).hexdigest()[:12])
    return df.reset_index(drop=True)


def aggregate_to_slots(raw: pd.DataFrame) -> pd.DataFrame:
    """Helper PARTAGÉ (optionnel) : agrège les sondages bruts en un vecteur de
    parts par slot et par (sondage, hypothèse). Les modèles qui veulent le
    niveau slot appellent ceci ; les autres travaillent directement sur `raw`.

    Un même sondage teste souvent PLUSIEURS hypothèses de champ (variante_1,
    variante_2... — Attal *ou* Philippe, avec ou sans Villepin/Ruffin...),
    chacune sommant à ~100% en interne mais sur un ensemble de candidats
    DIFFÉRENT. Deux approches essayées et rejetées (session du 2026-08-09) :
    moyenner candidat par candidat gonfle la somme totale (mesuré à 104% sur
    Elabe 9-10 juillet 2026, un candidat testé dans peu d'hypothèses ressort
    gonflé, conditionné à l'absence de son alternative) ; ne garder que la
    "meilleure" hypothèse (couverture maximale du roster) PERD des candidats
    entiers testés seulement dans une hypothèse minoritaire — mesuré à 0/3
    sondages retenus pour Ruffin, pourtant bien testé 3 fois, alors qu'un
    concurrent couvrant 1 slot de plus gagnait systématiquement le
    départage. On garde donc TOUTES les hypothèses comme autant
    d'observations SÉPARÉES du même sondage (même date, delta_t=0 — déjà géré
    par le SSM comme deux nœuds au même jour) : chacune sature déjà
    `poll_plan`/le padding conçus pour une observation partielle du champ, pas
    besoin d'inventer un mécanisme dédié. Pour ne pas laisser un sondage à N
    hypothèses peser N fois plus qu'un sondage à 1 hypothèse dans le filtre
    (mesures fortement corrélées, même terrain — vérifié : la part RN varie de
    <1pt d'une hypothèse à l'autre du même sondage), `n_hypotheses` est
    exposé pour que le consommateur déflate l'échantillon effectif
    (`echantillon / n_hypotheses`, cf. `NowcastData.from_dataframe`)."""
    df = raw[raw["slot"].notna()].copy()
    if "notice_url" not in df.columns:            # compat si champ absent
        df["notice_url"] = None
    df["hypothese"] = df["hypothese"].fillna("__unique__")

    # somme par (sondage, hypothèse, slot) — fusionne, au sein d'une même
    # hypothèse, plusieurs lignes candidat qui retomberaient sur le même slot.
    g = (df.groupby(["notice", "notice_url", "institut", "date_fin", "echantillon",
                     "hypothese", "slot", "bloc"])["intention"]
           .sum().reset_index())
    g["n_hypotheses"] = g.groupby("notice")["hypothese"].transform("nunique")

    obs = g.rename(columns={"intention": "part"})
    obs["horizon"] = (ELECTION_T1 - pd.to_datetime(obs["date_fin"], errors="coerce")).dt.days
    obs["echantillon"] = obs["echantillon"].fillna(obs["echantillon"].median())
    return obs.reset_index(drop=True)


if __name__ == "__main__":
    obs = load_live_observations()
    print(f"{obs['notice'].nunique()} sondages, {len(obs)} observations slot×sondage")
    print("\nMoyenne des parts par slot (brut, non débiaisé) :")
    m = obs.groupby("slot")["part"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    print(m.round(1).to_string())
    print(f"\nHorizons (jours avant T1) : {obs['horizon'].min()} → {obs['horizon'].max()}")
