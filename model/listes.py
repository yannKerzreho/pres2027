"""Courbes gp-pooling par LISTE de candidats : les champs les plus testés.

    python -m model.listes                  # aujourd'hui
    python -m model.listes --as-of 2026-10-01

Pourquoi
--------
`gp-pooling` ne consomme que les hypothèses testant EXACTEMENT sa liste
(`filter_scenarios_by_exact_slots`) : deux champs différents ne mesurent pas la
même quantité. Les instituts testent pourtant d'autres listes, aussi souvent que
la nôtre (Attal au lieu de Philippe, Attal ET Philippe…). Ce module ajuste le
MÊME modèle, sans rien y changer, sur chacune des listes les plus testées, pour
que le lecteur voie aussi ces mesures-là.

Définition d'une liste (« champ »)
-----------------------------------
L'ensemble EXACT des candidats soumis aux sondés dans une hypothèse (institut ×
date × hypothèse = `scenario_id`), petits candidats compris. C'est la définition
qu'emploie déjà `gp-pooling` — aucune équivalence entre champs, aucun alias : un
champ qui diffère d'un seul nom est un autre champ. Un seuil sur les parts (« ne
garder que les candidats à 3 % ») n'est PAS un simple découpage : la même liste
se scinderait selon qu'un candidat est mesuré à 2,9 ou 3,1 %, et le champ
dépendrait de la mesure. Le choisir serait une décision de modélisation.

Règle de sélection (validée par l'utilisateur)
----------------------------------------------
- On regarde les `N_RECENTS` sondages les plus récents à la date `as_of`.
- Une liste y compte pour 1 par sondage qui la teste (au moins une hypothèse).
- On retient au plus `N_MAX` listes, par nombre décroissant, puis par nombre total
  de sondages dans l'historique, puis par clé (ordre déterministe).
- Planchers : `MIN_RECENTS` sondages récents (une liste testée une seule fois
  n'est pas « la plus testée ») et `MIN_TOTAL` sondages au total (le seuil de
  publication de `model/backfill.py` : en dessous, la courbe serait trompeuse).

La règle choisit QUELLES listes on affiche, pas SUR QUOI on ajuste : chaque
courbe utilise tout l'historique des sondages de sa liste. Elle garde ainsi une
définition fixe, et son backtest reste valable.

Sorties (`docs/data/gp-listes/`)
-------------------------------
- `listes.json` : le classement du jour, la règle, et l'historique des entrées et
  sorties du classement (recalculé à chaque passage sur les dates de sondage —
  rien à mémoriser, comme le rejeu de `model/run.py`).
- `<id>.json` : une courbe par liste retenue (trajectoire lissée + sondages),
  sauf la liste de la prévision, dont la courbe est le snapshot `gp-pooling`,
  SANS prévision au scrutin ni archive datée — la trajectoire lissée d'aujourd'hui
  contient déjà toute la courbe. Les fichiers des listes sorties du classement sont
  supprimés : le dossier ne grossit pas avec le temps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

from model.core.base import SITE_DATA
from model.core.live_dataset import (
    SLOTS, _fold, filter_scenarios_by_exact_slots, load_raw_polls, map_candidate,
)
from model.models.gp_pooling.model import GPPooling

DOSSIER = "gp-listes"
N_RECENTS = 10
N_MAX = 3
MIN_RECENTS = 2
MIN_TOTAL = 3
# Premier jour où l'on calcule le classement pour l'historique des entrées et
# sorties : même borne que les courbes publiées (`model/backfill.py`).
DEPUIS = "2026-01-01"


# --- champs -------------------------------------------------------------------
def cle_champ(noms) -> str:
    """Clé canonique d'un champ : noms folés (accents, casse, tirets), triés."""
    return "|".join(sorted({_fold(n) for n in noms}))


def id_champ(cle: str) -> str:
    """Identifiant court et STABLE d'un champ : sert de nom de fichier, et ne
    change donc pas d'un jour à l'autre pour une même liste."""
    return hashlib.sha1(cle.encode()).hexdigest()[:8]


def champs(raw: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par hypothèse : `scenario_id, notice, date_fin, cle, candidats`."""
    rows = []
    for sid, sub in raw.groupby("scenario_id", sort=False):
        noms = sorted(set(sub["candidat"].astype(str)), key=_fold)
        rows.append({"scenario_id": sid, "notice": sub["notice"].iloc[0],
                     "date_fin": pd.Timestamp(sub["date_fin"].max()),
                     "cle": cle_champ(noms), "candidats": noms})
    return pd.DataFrame(rows, columns=["scenario_id", "notice", "date_fin", "cle", "candidats"])


def cle_reference(raw: pd.DataFrame) -> str | None:
    """Champ du modèle de base `gp-pooling` (celui de la prévision affichée)."""
    ref = filter_scenarios_by_exact_slots(raw, set(SLOTS))
    if ref.empty:
        return None
    cles = champs(ref)["cle"].unique()
    return cles[0] if len(cles) == 1 else None


def classement(ch: pd.DataFrame, as_of, n_recents=N_RECENTS, n_max=N_MAX,
               min_recents=MIN_RECENTS, min_total=MIN_TOTAL) -> list[dict]:
    """Les listes les plus testées parmi les `n_recents` derniers sondages à `as_of`."""
    ch = ch[ch["date_fin"] <= pd.Timestamp(as_of)]
    if ch.empty:
        return []
    par_sondage = ch.groupby("notice")["date_fin"].max().reset_index()
    recents = set(par_sondage.sort_values(["date_fin", "notice"])["notice"].tail(n_recents))
    total = ch.groupby("cle")["notice"].nunique()
    rec = ch[ch["notice"].isin(recents)].groupby("cle")["notice"].nunique()
    noms = ch.drop_duplicates("cle").set_index("cle")["candidats"]
    t = pd.DataFrame({"n_recents": rec}).join(total.rename("n_total"))
    t = t[(t["n_recents"] >= min_recents) & (t["n_total"] >= min_total)]
    t = t.reset_index().sort_values(["n_recents", "n_total", "cle"],
                                    ascending=[False, False, True]).head(n_max)
    return [{"cle": r.cle, "id": id_champ(r.cle), "candidats": list(noms[r.cle]),
             "n_recents": int(r.n_recents), "n_total": int(r.n_total)}
            for r in t.itertuples()]


def historique(ch: pd.DataFrame, as_of, depuis=DEPUIS) -> tuple[list[dict], dict]:
    """Entrées et sorties du classement, évaluées à chaque date de sondage.

    Le classement ne change qu'à l'arrivée d'un sondage : les dates de terrain
    suffisent. Renvoie `(evenements, entree_courante)` où `entree_courante[cle]`
    est la date de la DERNIÈRE entrée de chaque liste présente aujourd'hui."""
    jours = sorted({d for d in ch["date_fin"] if pd.Timestamp(depuis) <= d <= pd.Timestamp(as_of)})
    jours.append(pd.Timestamp(as_of))
    precedent: dict[str, dict] = {}
    evenements, entree = [], {}
    for j in jours:
        courant = {x["cle"]: x for x in classement(ch, j)}
        jour = str(j.date())
        for cle in courant.keys() - precedent.keys():
            evenements.append({"date": jour, "type": "entree", "id": id_champ(cle),
                               "candidats": courant[cle]["candidats"]})
            entree[cle] = jour
        for cle in precedent.keys() - courant.keys():
            evenements.append({"date": jour, "type": "sortie", "id": id_champ(cle),
                               "candidats": precedent[cle]["candidats"]})
            entree.pop(cle, None)
        precedent = courant
    # dédoublonne le dernier jour s'il coïncide avec une date de sondage
    vus, uniques = set(), []
    for e in evenements:
        k = (e["date"], e["type"], e["id"])
        if k not in vus:
            vus.add(k); uniques.append(e)
    return uniques, entree


# --- libellés -----------------------------------------------------------------
def libelle_candidat(nom: str) -> str:
    """Libellé d'affichage : celui du slot du modèle de base quand le candidat en
    a un (« RN », « Philippe (Horizons) »), sinon son nom tel que publié. Aucun
    alias : Bardella reste « Bardella », il n'est pas versé dans « RN »."""
    slot, _ = map_candidate(nom)
    return slot or nom


def libelle_liste(candidats: list[str], reference: list[str] | None) -> str:
    """Comment la liste diffère de celle de la prévision (« avec Attal, sans
    Philippe ») — c'est ce qui distingue les listes à l'œil, pas leurs 10 noms."""
    if reference is None:
        return ", ".join(candidats)
    a = {_fold(c): c for c in candidats}
    r = {_fold(c): c for c in reference}
    plus = [a[k] for k in sorted(a.keys() - r.keys())]
    moins = [r[k] for k in sorted(r.keys() - a.keys())]
    if not plus and not moins:
        return "liste de la prévision"
    morceaux = (["avec " + ", ".join(plus)] if plus else []) + \
               (["sans " + ", ".join(moins)] if moins else [])
    return ", ".join(morceaux)


# --- ajustement ---------------------------------------------------------------
class GPListe(GPPooling):
    """`gp-pooling` à l'identique, sur un autre champ. Non enregistré : il n'a ni
    prévision publiée ni archive datée, seulement une courbe."""

    def __init__(self, cle: str, candidats: list[str]):
        self.cle = cle
        self.id = f"{DOSSIER}/{id_champ(cle)}"
        self.labels = {c: libelle_candidat(c) for c in candidats}
        if len(set(self.labels.values())) != len(self.labels):
            raise ValueError(f"libellés en collision dans le champ {cle}")

    def slots_modele(self) -> list[str]:
        return list(self.labels.values())

    def used_polls(self, raw_polls):
        ch = champs(raw_polls)
        sids = set(ch.loc[ch["cle"] == self.cle, "scenario_id"])
        sub = raw_polls[raw_polls["scenario_id"].isin(sids)].copy()
        # Le vocabulaire de CETTE liste : chaque candidat est son propre slot.
        sub["slot"] = sub["candidat"].map(self.labels)
        return sub.reset_index(drop=True)


def courbe(raw: pd.DataFrame, entree: dict, as_of: str) -> dict:
    mdl = GPListe(entree["cle"], entree["candidats"])
    mdl.load_artifacts()
    used = mdl.used_polls(raw)
    nc = mdl.nowcast(raw, as_of)
    return {
        "id": entree["id"], "cle": entree["cle"], "as_of": as_of,
        "candidats": mdl.slots_modele(),
        "n_sondages": int(used["notice"].nunique()),
        "instituts": sorted(used["institut"].unique().tolist()),
        "nowcast": nc.summary(),
        "trajectoire": nc.trajectoire,
        "polls": mdl.display_polls(used),
    }


def produire(raw: pd.DataFrame, as_of: str, dossier: Path | None = None) -> dict:
    """Écrit `listes.json` et une courbe par liste retenue ; renvoie le manifeste."""
    dossier = dossier or SITE_DATA / DOSSIER
    dossier.mkdir(parents=True, exist_ok=True)
    raw = raw[raw["date_fin"] <= pd.Timestamp(as_of)]
    ch = champs(raw)
    top = classement(ch, as_of)
    evenements, depuis = historique(ch, as_of)
    ref = cle_reference(raw)
    ref_noms = (ch.loc[ch["cle"] == ref, "candidats"].iloc[0] if ref else None)

    listes, gardes = [], {"listes.json"}
    for rang, e in enumerate(top, 1):
        est_ref = e["cle"] == ref
        info = {**e, "rang": rang, "reference": est_ref,
                "libelle": libelle_liste(e["candidats"], ref_noms),
                "dans_le_classement_depuis": depuis.get(e["cle"])}
        if est_ref:
            # La liste de la prévision EST `gp-pooling` : même champ, même
            # modèle, même courbe au bruit de Monte-Carlo près (~0,1 pt). Le
            # front a déjà son snapshot en main ; l'écrire une seconde fois
            # doublerait un fichier réécrit chaque jour, pour rien.
            info["fichier"] = None
        else:
            c = courbe(raw, e, as_of)
            (dossier / f"{e['id']}.json").write_text(json.dumps(c, ensure_ascii=False))
            gardes.add(f"{e['id']}.json")
            info["fichier"] = f"{DOSSIER}/{e['id']}.json"
            info["n_sondages_ajustes"] = c["n_sondages"]
        listes.append(info)
    for e in evenements:
        e["libelle"] = libelle_liste(e["candidats"], ref_noms)
    # Le dossier ne garde que les courbes du classement du jour.
    for f in dossier.glob("*.json"):
        if f.name not in gardes:
            f.unlink()

    manifeste = {
        "as_of": as_of, "genere_le": date.today().isoformat(),
        "regle": {"n_recents": N_RECENTS, "n_max": N_MAX, "min_recents": MIN_RECENTS,
                  "min_total": MIN_TOTAL,
                  "champ": "ensemble exact des candidats testés dans l'hypothèse"},
        "reference": {"cle": ref, "id": id_champ(ref) if ref else None,
                      "candidats": ref_noms},
        "listes": listes,
        "evenements": evenements,
    }
    (dossier / "listes.json").write_text(json.dumps(manifeste, ensure_ascii=False, indent=1))
    return manifeste


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None)
    args = ap.parse_args()
    as_of = args.as_of or date.today().isoformat()
    m = produire(load_raw_polls(as_of=as_of), as_of)
    for x in m["listes"]:
        print(f"#{x['rang']} {x['libelle']} : {x['n_recents']}/{N_RECENTS} récents, "
              f"{x['n_total']} au total -> {x['fichier']}")
    print(f"{len(m['evenements'])} entrée(s)/sortie(s) depuis {DEPUIS}")


if __name__ == "__main__":
    main()
