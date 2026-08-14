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
"""

from __future__ import annotations

import argparse
from datetime import date

import model.core.registered  # noqa: F401  (peuple le registre)
from model.core.base import all_models, get_model, write_snapshot
from model.core.live_dataset import load_raw_polls


def _selection(model_id: str, include_private: bool) -> dict:
    if model_id != "all":
        # Un id demandé explicitement est toujours honoré, affiché ou non.
        return {model_id: get_model(model_id)}
    models = all_models()
    if include_private:
        return models
    return {mid: m for mid, m in models.items() if m.surface is not None}


def run(model_id: str = "all", as_of: str | None = None, include_private: bool = False) -> None:
    as_of = as_of or date.today().isoformat()
    raw = load_raw_polls(as_of=as_of)
    if raw.empty:
        raise SystemExit(f"Aucun sondage disponible à la date {as_of}.")

    models = _selection(model_id, include_private)
    if not models:
        raise SystemExit("Aucun modèle à lancer (registre vide ou aucun rattaché au site).")
    for mid, mdl in models.items():
        snap = mdl.run(raw, as_of)
        path = write_snapshot(snap)
        rn = snap["forecast_scrutin"].get("RN", {}).get("p_qualifie_top2")
        print(f"[{mid}] {as_of} : {snap['meta']['n_sondages']} sondages, "
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
        elif diag:
            print(f"    diagnostics : {diag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", help="id du modèle ou 'all'")
    ap.add_argument("--as-of", default=None, help="date AAAA-MM-JJ (défaut : aujourd'hui)")
    ap.add_argument("--all", dest="include_private", action="store_true",
                    help="inclure les modèles hors site (surface=None : variantes de comparaison)")
    args = ap.parse_args()
    run(args.model, args.as_of, args.include_private)


if __name__ == "__main__":
    main()
