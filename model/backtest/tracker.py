"""Tracker par candidat (spec_spatial_pooling §10) : trajectoire de la part de
chaque candidat dans un champ de référence, tirée du chemin w(t) du fit.

    python -m model.backtest.tracker --chaine k        # 4 processus, un par cœur
    python -m model.backtest.tracker --fusionner       # JSON des trajectoires

Configuration de PRODUCTION (T1 seul, comme `PUBLIER_T2 = False`). Champ de
référence par défaut : le champ le plus testé (même règle que l'onglet
Scénarios) ; les listes du site viendront avec le backtest de suivi.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

from model.backtest import second_round as sr
from model.core.live_dataset import load_raw_polls
from model.models.spatial_pooling.fit import fit_spatial_pooling, trajectoire
from model.models.spatial_pooling.roster import build_poll_arrays, split_tours


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chaine", type=int, default=None)
    ap.add_argument("--fusionner", action="store_true")
    a = ap.parse_args()
    raw = load_raw_polls(tour=None)
    as_of = str(pd.to_datetime(raw["date_fin"]).max().date())
    out = sr.OUT_DIR / f"tracker_2027_{as_of}.json"
    base = sr.CHAINES_DIR / out.stem
    nuts = dict(draws=400, tune=600, chains=4)
    if a.chaine is not None:
        nuts["mcmc_runner"] = sr._runner_chaine(base.with_name(f"{out.stem}_c{a.chaine}.npz"), a.chaine)
    elif a.fusionner:
        nuts["mcmc_runner"] = sr._runner_fusion([base.with_name(f"{out.stem}_c{k}.npz")
                                                 for k in range(sr.N_CHAINES)])
    else:
        ap.error("--chaine k ou --fusionner")
    fit = fit_spatial_pooling(raw, as_of=as_of, second_tour=False, **nuts)
    if a.chaine is not None:
        return
    tm = fit.tested_mask > 0
    modal = Counter(tuple(np.flatnonzero(r)) for r in tm).most_common(1)[0][0]
    mask = np.zeros(len(fit.candidates)); mask[list(modal)] = 1
    dates, pi = trajectoire(fit, mask)
    noms = [fit.candidates[i] for i in modal]
    # Sondages bruts du MÊME champ exact, pour comparer à l'œil.
    t1 = split_tours(raw)[0]
    t1 = t1.assign(_h=t1["hypothese"].fillna("__u__"))
    champ = t1.groupby(["notice", "_h"]).candidat.transform(lambda x: frozenset(x) == frozenset(noms))
    bruts = (t1[champ].groupby(["notice", "_h", "date_fin", "candidat"]).intention.sum()
             .reset_index())
    res = {"as_of": as_of, "champ": noms, "diagnostics": {k: fit.diagnostics.get(k) for k in
           ("rhat_max", "ess_min", "n_divergences", "n_dynamiques", "sigma_w")},
           "chaines_retenues": list(sr.CHAINES_RETENUES),
           "dynamiques": [c for c, d in zip(fit.candidates, fit.dynamic) if d],
           "trajectoires": {c: {"dates": [str(pd.Timestamp.fromordinal(int(d)).date()) for d in dates],
                                "moyenne": [round(float(x), 4) for x in pi[:, :, j].mean(0)],
                                "ic90": [[round(float(x), 4) for x in np.percentile(pi[:, t, j], [5, 95])]
                                         for t in range(len(dates))]}
                            for j, c in enumerate(noms)},
           "sondages_meme_champ": [{"date": str(pd.Timestamp(r.date_fin).date()), "candidat": r.candidat,
                                    "intention": float(r.intention)} for r in bruts.itertuples()]}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
