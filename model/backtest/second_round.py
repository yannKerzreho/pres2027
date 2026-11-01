"""Backtest T2 du spatial pooling : option extérieure uniforme ou non.

Exemples :
  python -m model.backtest.second_round --fast
  python -m model.backtest.second_round

Chaque élection est évaluée une semaine avant le second tour, avec seulement
les sondages publiés jusque-là. Les paramètres NUTS du mode complet sont ceux
du modèle publié ; ``--fast`` est un écran de fumée utile pour le développement,
pas une validation de production.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline.historical import (PARSED_DIR, POLL_FILES, _resolve_candidate,
                                 _select_best_hypothesis, load_results)
from model.models.spatial_pooling.fit import fit_spatial_pooling, second_round_from_fit
from model.models.spatial_pooling.roster import ORDER_GROUPS_HISTORIQUES

T2_DATES = {2017: pd.Timestamp("2017-05-07"), 2022: pd.Timestamp("2022-04-24")}


def polls_historiques(election: int, as_of: pd.Timestamp,
                      history_days: int = 180) -> pd.DataFrame:
    """Sondages wiki dans le vocabulaire des rosters historiques."""
    raw = pd.read_csv(Path(PARSED_DIR) / POLL_FILES[election])
    results = load_results()
    results_e = results.loc[results.election.eq(election)]
    names = set(results_e["candidat"])
    names_t1 = set(results_e.loc[results_e.tour.eq("Premier tour"), "candidat"])
    names_t2 = set(results_e.loc[results_e.tour.eq("Deuxième tour"), "candidat"])
    raw["candidat"] = raw["candidat"].apply(lambda x: _resolve_candidate(x, names))
    dates = pd.to_datetime(raw["date_fin"], errors="coerce")
    raw["date_fin"] = dates.fillna(pd.to_datetime(raw.get("date_notice"), errors="coerce"))
    raw = raw[raw["date_fin"].notna() & raw["date_fin"].le(as_of)].copy()
    # Les rosters historiques décrivent les candidats effectivement présents
    # au scrutin. Les anciennes hypothèses (primaires, retraits) doivent être
    # retirées, sinon elles rendent le fit impossible sans améliorer le test.
    raw = raw.loc[
        (raw.tour.eq("Premier tour") & raw.candidat.isin(names_t1))
        | (raw.tour.eq("Deuxième tour") & raw.candidat.isin(names_t2))
    ].copy()
    # Une seule hypothèse T1 par notice, celle qui ressemble le plus au champ
    # finalement observé. Sans cela, on compte plusieurs fois le même terrain.
    t1 = _select_best_hypothesis(raw[raw.tour.eq("Premier tour")], names_t1)
    # En T2, ne conserver que les mesures du duel réellement arrivé : filtrer
    # seulement les lignes laisserait des demi-duels (une ligne survivante) et
    # injecterait de fausses observations dans la vraisemblance.
    t2 = raw[raw.tour.eq("Deuxième tour")].copy()
    t2["_hyp"] = t2["hypothese"].fillna("__defaut__")
    actual_pairs = (
        t2.groupby(["notice", "_hyp"])["candidat"].agg(lambda x: set(x))
        .eq(names_t2)
    )
    good = actual_pairs[actual_pairs].index
    t2 = t2.set_index(["notice", "_hyp"]).loc[good].reset_index()
    t2 = t2.drop(columns="_hyp")
    campaign_start = as_of - pd.Timedelta(days=history_days)
    return pd.concat([t1, t2], ignore_index=True).loc[lambda x: x.date_fin.ge(campaign_start)].copy()


def evaluate(election: int, omega: float | None, days_before: int, draws: int, tune: int,
             chains: int) -> dict:
    as_of = T2_DATES[election] - pd.Timedelta(days=days_before)
    raw = polls_historiques(election, as_of)
    fit = fit_spatial_pooling(
        raw, str(as_of.date()), draws=draws, tune=tune, chains=chains,
        min_poll_date="1900-01-01", order_groups=ORDER_GROUPS_HISTORIQUES[election],
        couverture_min=None, outside_omega=omega,
    )
    results = load_results()
    truth = results[(results.election.eq(election)) & (results.tour.eq("Deuxième tour"))]
    duel_names = sorted(truth.candidat.tolist())
    duels = second_round_from_fit(fit, raw)
    duel = next((d for d in duels if d["candidats"] == duel_names), None)
    if duel is None:
        raise RuntimeError(f"{election}: duel final absent du fit ({duel_names})")
    a, b = duel_names
    actual = float(truth.loc[truth.candidat.eq(a), "resultat"].iloc[0]) / 100.0
    pred = duel["intentions"][a]
    return {
        "election": election, "jours_avant_t2": days_before,
        "option_exterieure": "non" if omega is None else f"omega={omega:g}",
        "candidat": a, "observe": actual, "pred": pred["moyenne"],
        "erreur_abs": abs(pred["moyenne"] - actual),
        "couvre_ic90": pred["ic90"][0] <= actual <= pred["ic90"][1],
        "p_gagne": pred["p_gagne"], "rhat_max": fit.diagnostics["rhat_max"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--days-before", type=int, default=7)
    args = ap.parse_args()
    draws, tune, chains = (40, 40, 1) if args.fast else (400, 600, 4)
    rows = []
    for y in (2017, 2022):
        for omega in (None, 0.0):
            print(f"Backtest {y}, omega={omega}...", flush=True)
            rows.append(evaluate(y, omega, args.days_before, draws, tune, chains))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nErreur absolue moyenne / couverture IC90")
    print(out.groupby("option_exterieure").agg(erreur_abs=("erreur_abs", "mean"),
                                                  couverture=("couvre_ic90", "mean")).to_string())


if __name__ == "__main__":
    main()
