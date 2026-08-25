"""Point d'entrée du modèle live : produit les snapshots datés.

    python -m model.run                      # tout ce que le site affiche, aujourd'hui
    python -m model.run --model bayesian-nowcast
    python -m model.run --all                # + variantes de comparaison
    python -m model.run --as-of 2026-03-01   # rejoue une date passée (backfill)

Le job quotidien lance `python -m model.run` sans argument.

Portée par défaut : tout modèle rattaché à une rubrique du site
(`ForecastModel.surface`, donc « suivi » ET « scénarios ») — c'est-à-dire tout ce
dont une page a besoin d'une version fraîche chaque jour. Le registre contient
aussi des variantes de diagnostic (`surface=None`) qui n'ont pas vocation à
produire un point par jour : chacune est un run NUTS complet, et les empiler dans
le cron quotidien allonge le job sans que personne ne lise le résultat. Elles
restent lançables explicitement, par `--model <id>` pour l'une ou `--all` pour
toutes.

Rejeu des dates touchées par un sondage arrivé en retard
--------------------------------------------------------
Un sondage est publié sur Wikipedia PLUSIEURS JOURS après son terrain (mesuré :
Harris, terrain des 18-19 août, saisi le 24 à 08h06 UTC — 11 minutes après le
passage du job). Sans rejeu, ce sondage n'entrait que dans le snapshot du jour
de son ARRIVÉE : la courbe faisait sa marche à cette date-là, alors que le point
de sondage, lui, est tracé à sa date de TERRAIN (`docs/index.html`, `scatter`).
Le lecteur voyait un décrochage à une date où il ne s'est rien passé, et rien du
tout à la date où l'opinion a été mesurée.

On rejoue donc, à chaque passage, TOUTE date dont le jeu de sondages a changé —
pas seulement la date du sondage. C'est imposé par la construction du nowcast :
θ est estimé en `as_of` à partir de tous les sondages antérieurs, donc un
sondage daté D modifie tous les snapshots de D à aujourd'hui, pas seulement
celui de D. Ne rejouer que D produirait DEUX décrochages au lieu d'un (un vrai
en D, puis un retour en arrière en D+1 tant que les jours suivants ignorent le
sondage).

Détection : on compare, date par date, les notices que le modèle CONSOMMERAIT
aujourd'hui (`used_polls` sur les sondages ≤ cette date) à celles que le
snapshot publié déclare avoir consommées (son champ `polls`). C'est un
auto-diagnostic, pas un journal à tenir : rien à mémoriser entre deux passages,
et une correction de la page wiki (valeur rectifiée, sondage retiré) est
rattrapée au même titre qu'un ajout.

Contrepartie assumée : la courbe devient RÉVISABLE. Un point publié hier peut
changer aujourd'hui si un sondage antérieur est arrivé entre-temps. C'est le
comportement correct d'une estimation qui date l'opinion plutôt que la croyance
— mais deux captures d'écran prises à deux jours d'intervalle ne sont plus
superposables.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

import model.core.registered  # noqa: F401  (peuple le registre)
from model.core.base import SITE_DATA, all_models, get_model, write_snapshot
from model.core.live_dataset import load_raw_polls

# Plafond de rejeu par passage. Le régime de croisière est de 0 à quelques dates
# (un sondage a rarement plus d'une semaine de retard), mais le PREMIER passage
# après une évolution du roster ou du modèle peut en trouver des dizaines : un
# run `spatial-pooling` est une inférence NUTS d'environ 100 s, de quoi faire
# sauter le `timeout-minutes` du job et perdre le snapshot du jour pour une
# raison sans rapport avec le jour. On rattrape donc les plus RÉCENTES d'abord
# (celles que le lecteur regarde) et le reste au fil des jours suivants — le
# rattrapage converge tout seul puisque la détection est un auto-diagnostic.
MAX_REJEU_DEFAUT = 6


def _selection(model_id: str, include_private: bool) -> dict:
    if model_id != "all":
        # Un id demandé explicitement est toujours honoré, affiché ou non.
        return {model_id: get_model(model_id)}
    models = all_models()
    if include_private:
        return models
    return {mid: m for mid, m in models.items() if m.surface is not None}


def _notices_publiees(path: Path) -> set[str] | None:
    """Notices qu'un snapshot DÉJÀ PUBLIÉ déclare avoir consommées, ou None si
    le fichier est absent/illisible (donc à produire). Un snapshot présent mais
    dont aucun point ne porte de `notice` renvoie un ensemble vide : il sera
    rejoué, ce qui est le bon repli — au pire on recalcule à l'identique."""
    if not path.exists():
        return None
    try:
        snap = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return {p["notice"] for p in snap.get("polls", []) if p.get("notice")}


def _notices_attendues(mdl, raw: pd.DataFrame, jour: str) -> set[str]:
    """Notices que `mdl` consommerait s'il tournait à cette date avec les
    données d'aujourd'hui. `used_polls` (et pas `raw`) : c'est le filtre PROPRE
    au modèle — roster exact pour `gp-pooling`, date plancher + roster pour
    `spatial-pooling` — donc un sondage ingéré mais écarté par le modèle ne
    déclenche pas un rejeu perpétuel."""
    en_date = raw[raw["date_fin"] <= pd.Timestamp(jour)]
    if en_date.empty:
        return set()
    used = mdl.used_polls(en_date)
    return set() if used.empty else set(used["notice"].unique())


def dates_a_produire(mdl, raw: pd.DataFrame, as_of: str,
                     max_rejeu: int = MAX_REJEU_DEFAUT) -> tuple[list[str], list[str]]:
    """(dates à calculer dans l'ordre chronologique, dates écartées par le plafond).

    `as_of` en fait TOUJOURS partie, même si son snapshot est déjà à jour : le
    graphe des probabilités de qualification bouge chaque jour sans nouveau
    sondage (l'horizon jusqu'au scrutin se réduit), le point quotidien a donc
    du sens en propre. Les dates PASSÉES, elles, ne sont recalculées que si
    leur jeu de sondages a changé.

    Les dates candidates sont les snapshots déjà publiés (à corriger) ET les
    dates de terrain des sondages (à créer) : c'est ce second ensemble qui fait
    apparaître un point de courbe le jour même de la mesure, là où le nuage de
    sondages du site place déjà son point.
    """
    mdir = SITE_DATA / mdl.id
    idx = mdir / "index.json"
    publiees = set(json.loads(idx.read_text())) if idx.exists() else set()

    used = mdl.used_polls(raw)
    terrains = set() if used.empty else {
        str(d)[:10] for d in pd.to_datetime(used["date_fin"]).dropna()}

    candidates = sorted({d for d in publiees | terrains if d < as_of})
    perimees = []
    for d in candidates:
        attendues = _notices_attendues(mdl, raw, d)
        if not attendues:
            # Aucun sondage exploitable à cette date : le modèle ne peut pas
            # tourner. On n'invente pas un point sur une courbe.
            continue
        if _notices_publiees(mdir / f"{d}.json") != attendues:
            perimees.append(d)

    # Les plus récentes d'abord quand le plafond mord, puis retour à l'ordre
    # chronologique : `spatial-pooling` réécrit `scenarios.json` à CHAQUE
    # nowcast (cf. `_export_scenarios`), donc la dernière date calculée est
    # celle qui reste dans l'artefact — ce doit être `as_of`.
    ecartees = sorted(perimees[:-max_rejeu]) if len(perimees) > max_rejeu else []
    retenues = sorted(perimees[len(ecartees):] + [as_of])
    return retenues, ecartees


def run(model_id: str = "all", as_of: str | None = None, include_private: bool = False,
        rejeu: bool | None = None, max_rejeu: int = MAX_REJEU_DEFAUT) -> None:
    as_of_explicite = as_of is not None
    as_of = as_of or date.today().isoformat()
    # Un `--as-of` explicite est un backfill ponctuel : on produit CETTE date et
    # rien d'autre, sauf demande contraire. Le rejeu automatique n'est le défaut
    # que pour le passage quotidien.
    if rejeu is None:
        rejeu = not as_of_explicite

    raw = load_raw_polls(as_of=as_of)
    if raw.empty:
        raise SystemExit(f"Aucun sondage disponible à la date {as_of}.")

    models = _selection(model_id, include_private)
    if not models:
        raise SystemExit("Aucun modèle à lancer (registre vide ou aucun rattaché au site).")
    for mid, mdl in models.items():
        if rejeu:
            dates, ecartees = dates_a_produire(mdl, raw, as_of, max_rejeu)
            if len(dates) > 1:
                print(f"[{mid}] rejeu de {len(dates) - 1} date(s) dont le jeu de "
                      f"sondages a changé : {', '.join(d for d in dates if d != as_of)}")
            if ecartees:
                print(f"[{mid}] {len(ecartees)} date(s) plus anciennes reportées au "
                      f"prochain passage (plafond --max-rejeu={max_rejeu}) : "
                      f"{ecartees[0]} → {ecartees[-1]}")
        else:
            dates = [as_of]

        for jour in dates:
            raw_jour = raw if jour == as_of else raw[raw["date_fin"] <= pd.Timestamp(jour)]
            snap = mdl.run(raw_jour, jour)
            path = write_snapshot(snap)
            rn = snap["forecast_scrutin"].get("RN", {}).get("p_qualifie_top2")
            print(f"[{mid}] {jour} : {snap['meta']['n_sondages']} sondages, "
                  f"P(RN top2)={rn} -> {path.name}")
            diag = snap.get("diagnostics")
            if diag and "hyperparametres" in diag:
                # Schéma bayesian-nowcast (NUTS) — PAS imposé par le framework
                # (cf. Nowcast.diagnostics, model/core/base.py) : un modèle sans
                # inférence MCMC (ex. gp-pooling) a son propre schéma, affiché
                # tel quel ci-dessous plutôt que de supposer ces clés.
                hp = ", ".join(f"{k}={v['mean']:.4f}±{v['sd']:.4f}"
                              for k, v in diag["hyperparametres"].items() if v)
                print(f"    diagnostics : R-hat max={diag['rhat_max']}, "
                      f"ESS min={diag['ess_min']}, divergences={diag['n_divergences']} | {hp}")
            elif diag and jour == as_of:
                print(f"    diagnostics : {diag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", help="id du modèle ou 'all'")
    ap.add_argument("--as-of", default=None, help="date AAAA-MM-JJ (défaut : aujourd'hui)")
    ap.add_argument("--all", dest="include_private", action="store_true",
                    help="inclure les modèles hors site (surface=None : variantes de comparaison)")
    ap.add_argument("--rejeu", dest="rejeu", action="store_true", default=None,
                    help="recalculer les dates passées dont le jeu de sondages a changé "
                         "(défaut quand --as-of est absent)")
    ap.add_argument("--no-rejeu", dest="rejeu", action="store_false",
                    help="ne produire que la date demandée")
    ap.add_argument("--max-rejeu", type=int, default=MAX_REJEU_DEFAUT,
                    help=f"plafond de dates passées rejouées par passage (défaut {MAX_REJEU_DEFAUT})")
    args = ap.parse_args()
    run(args.model, args.as_of, args.include_private, args.rejeu, args.max_rejeu)


if __name__ == "__main__":
    main()
