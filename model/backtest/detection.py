"""Étape 3 du backtest de suivi : détection de la montée Zemmour 2021
(spec_spatial_pooling §10). Lit les fits fusionnés de `model.backtest.suivi`
(aucun NUTS : `--fusionner` rejoue la fusion des chaînes sauvées).

    python -m model.backtest.detection
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from model.backtest import second_round as sr
from model.backtest import suivi as su
from model.listes import cle_champ
from model.models.spatial_pooling.fit import fit_spatial_pooling, pi_draws_for_mask
from model.models.spatial_pooling.roster import ORDER_GROUPS_HISTORIQUES

E = 2022
HS = [215, 208, 201, 194, 187, 180, 173, 166]
R = ["Anne Hidalgo", "Bertrand", "Emmanuel Macron", "Eric Zemmour", "Fabien Roussel", "Jean Lassalle",
     "Jean-Luc Mélenchon", "Marine Le Pen", "Montebourg", "Nathalie Arthaud", "Nicolas Dupont-Aignan",
     "Philippe Poutou", "Yannick Jadot"]
Z = "Eric Zemmour"
SEUIL, RECUL = 0.03, 28


def _brut(c: pd.Timestamp) -> float:
    raw = sr.polls_historiques(E, c, history_days=14)
    t1 = raw[raw.tour.eq("Premier tour") & raw.candidat.eq(Z)]
    return float(t1.intention.mean()) / 100 if len(t1) else np.nan


def _spatial(h: int, c: pd.Timestamp) -> tuple[float, list[float], list[str]]:
    raw = sr.polls_historiques(E, c)
    stem = f"suivi_{E}_h{h:03d}"
    runner = sr._runner_fusion([sr.CHAINES_DIR / f"{stem}_c{k}.npz" for k in range(sr.N_CHAINES)])
    fit = fit_spatial_pooling(raw, str(c.date()), min_poll_date="1900-01-01",
                              order_groups=ORDER_GROUPS_HISTORIQUES[E], couverture_min=None,
                              second_tour=False, draws=400, tune=600, chains=4, mcmc_runner=runner)
    idx = {x: i for i, x in enumerate(fit.candidates)}
    absents = [x for x in R if x not in idx]
    mask = np.zeros(len(fit.candidates)); mask[[idx[x] for x in R if x in idx]] = 1
    d = pi_draws_for_mask(fit, mask)[:, idx[Z]]
    return float(d.mean()), [float(v) for v in np.percentile(d, [5, 95])], absents


def _gp(c: pd.Timestamp) -> tuple[float | None, int]:
    raw = sr.polls_historiques(E, c)
    ch = su._champs_t1(raw)
    n = int(ch.loc[ch.cle.eq(cle_champ(R)), "notice"].nunique())
    if n < 3:
        return None, n
    L = [{"cle": cle_champ(R), "id": "R", "candidats": R}]
    # Réutilise gp_listes sans cible : on lit directement le nowcast.
    from model.core.base import ELECTION_T1
    from model.listes import GPListe
    delta = ELECTION_T1 - sr.T1_DATES[E]
    t1 = raw[raw.tour.eq("Premier tour")].copy()
    t1["date_fin"] = pd.to_datetime(t1["date_fin"]) + delta
    t1["scenario_id"] = t1["notice"].astype(str) + "|" + t1["hypothese"].fillna("__u__").astype(str)
    t1["bloc"] = "?"
    mdl = GPListe(L[0]["cle"], R)
    mdl.load_artifacts()
    nc = mdl.nowcast(t1, str((c + delta).date()))
    return float(np.asarray(nc.draws.sel(candidat=mdl.labels[Z])).mean()), n


def _detection(dates, serie) -> str | None:
    s = pd.Series(serie, index=pd.to_datetime(dates)).dropna()
    for t, v in s.items():
        avant = s[s.index <= t - pd.Timedelta(days=RECUL)]
        if len(avant) and v - avant.iloc[-1] >= SEUIL:
            return str(t.date())
    return None


def main() -> None:
    lignes = []
    for h in HS:
        c = sr.T1_DATES[E] - pd.Timedelta(days=h)
        sp, ic, absents = _spatial(h, c)
        g, n_r = _gp(c)
        lignes.append({"h": h, "coupure": str(c.date()), "brut_14j": round(_brut(c), 4),
                       "spatial": round(sp, 4), "spatial_ic90": [round(x, 4) for x in ic],
                       "absents_du_roster": absents, "gp": None if g is None else round(g, 4),
                       "n_sondages_R": n_r})
        print(lignes[-1], flush=True)
    dates = [l["coupure"] for l in lignes]
    res = {"liste_R": R, "seuil_pt": 100 * SEUIL, "recul_j": RECUL, "lignes": lignes,
           "detection": {k: _detection(dates, [l[k] for l in lignes])
                         for k in ("brut_14j", "spatial", "gp")}}
    out = sr.OUT_DIR / "detection_zemmour_2022.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(res["detection"], f"-> {out}")


if __name__ == "__main__":
    main()
