"""Backfill de l'historique : rejoue TOUS les modèles "as-of" chaque date de sondage.

Peuple les courbes d'évolution dès le jour 1. Le job quotidien n'ajoute ensuite
qu'un point par jour.

Usage : python -m model.backfill [--since 2026-01-01] [--min-sondages 3]
                                 [--models linear-pooling] [--skip-existing]
"""

from __future__ import annotations

import argparse

import pandas as pd

import model.core.registered  # noqa: F401
from model.core.base import SITE_DATA, all_models, write_snapshot
from model.core.live_dataset import load_raw_polls

# Première date de snapshot publiée. Avant 2026, les sondages d'intentions 2027
# sont trop rares et trop espacés (3 à 4 par an, séparés de plusieurs mois) pour
# qu'un nowcast soit défendable : le modèle produirait des points isolés au
# faisceau d'incertitude trompeusement étroit (mesuré : RN à 46 % sur 3 sondages
# en mars 2023). Ce n'est PAS un filtre sur les sondages que le modèle voit —
# `load_raw_polls` lui donne toujours tout l'historique disponible à `as_of` —
# seulement sur les dates auxquelles on publie un point de courbe.
DEFAULT_SINCE = "2026-01-01"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=DEFAULT_SINCE,
                    help=f"première date de snapshot AAAA-MM-JJ (défaut : {DEFAULT_SINCE})")
    ap.add_argument("--min-sondages", type=int, default=3)
    ap.add_argument("--draws", type=int, default=400, help="draws réduits pour le backfill")
    ap.add_argument("--models", default=None,
                    help="sous-ensemble d'ids de modèles, séparés par des virgules "
                        "(défaut : les modèles publics, cf. --all)")
    ap.add_argument("--all", dest="include_private", action="store_true",
                    help="inclure les modèles non publics (variantes de comparaison)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="ne recalcule pas un snapshot déjà écrit sur disque "
                        "(site/data/<model>/<as_of>.json) — reprise après interruption")
    args = ap.parse_args()

    all_polls = load_raw_polls()
    dates = sorted(pd.to_datetime(all_polls["date_fin"]).dropna().dt.date.unique())
    since = pd.Timestamp(args.since).date()
    dates = [d for d in dates if d >= since]
    models = all_models()
    if args.models:
        wanted = set(args.models.split(","))
        models = {mid: m for mid, m in models.items() if mid in wanted}
    elif not args.include_private:
        # Même portée par défaut que `model.run` : rejouer 23 dates × 8
        # variantes NUTS de diagnostic coûte des heures pour des courbes que
        # le site n'affiche pas.
        models = {mid: m for mid, m in models.items() if m.public}
    for m in models.values():                        # backfill rapide (draws réduits)
        for attr in ("draws", "tune"):
            if hasattr(m, attr):
                setattr(m, attr, args.draws)
        if hasattr(m, "chains"):
            m.chains = 2
    print(f"{len(dates)} dates × {len(models)} modèles → backfill…")
    for d in dates:
        as_of = d.isoformat()
        todo = {mid: mdl for mid, mdl in models.items()
               if not (args.skip_existing and (SITE_DATA / mid / f"{as_of}.json").exists())}
        if not todo:
            continue
        raw = load_raw_polls(as_of=as_of)
        if raw["notice"].nunique() < args.min_sondages:
            continue
        for mid, mdl in todo.items():
            try:
                snap = mdl.run(raw, as_of)
            except (IndexError, ValueError) as e:
                # un modèle peut filtrer les sondages en interne (ex.
                # bayesian-nowcast écarte tout ce qui précède MIN_POLL_DATE,
                # cf. nowcast.py) et se retrouver sans donnée du tout à cette
                # date, même si `raw` (non filtré) en a assez — pas une
                # erreur de backfill, cette date est juste hors du scope de
                # CE modèle.
                print(f"  {as_of} [{mid}] : ignoré ({e})")
                continue
            write_snapshot(snap)
        print(f"  {as_of} : {raw['notice'].nunique()} sondages ({', '.join(todo)})")


if __name__ == "__main__":
    main()
