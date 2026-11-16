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
# 243-222 : ajoutées après la revue du 16/11 (point 4) pour qu'une détection
# soit possible AVANT le 05/10 (recul 28 j ; la montée commence fin août).
HS = [243, 236, 229, 222, 215, 208, 201, 194, 187, 180, 173, 166]
R = ["Anne Hidalgo", "Bertrand", "Emmanuel Macron", "Eric Zemmour", "Fabien Roussel", "Jean Lassalle",
     "Jean-Luc Mélenchon", "Marine Le Pen", "Montebourg", "Nathalie Arthaud", "Nicolas Dupont-Aignan",
     "Philippe Poutou", "Yannick Jadot"]
Z = "Eric Zemmour"
SEUIL, RECUL = 0.03, 28


def _brut(c: pd.Timestamp) -> float:
    raw = sr.polls_historiques(E, c, history_days=14)
    t1 = raw[raw.tour.eq("Premier tour") & raw.candidat.eq(Z)]
    return float(t1.intention.mean()) / 100 if len(t1) else np.nan


def _fit(h: int, c: pd.Timestamp):
    raw = sr.polls_historiques(E, c)
    stem = f"suivi_{E}_h{h:03d}"
    runner = sr._runner_fusion([sr.CHAINES_DIR / f"{stem}_c{k}.npz" for k in range(sr.N_CHAINES)])
    return fit_spatial_pooling(raw, str(c.date()), min_poll_date="1900-01-01",
                               order_groups=ORDER_GROUPS_HISTORIQUES[E], couverture_min=None,
                               second_tour=False, draws=400, tune=600, chains=4, mcmc_runner=runner)


def _spatial(fit, liste: list[str]) -> tuple[float, list[float], list[str]]:
    """Part de Zemmour dans `liste` (candidats hors roster omis, et listés)."""
    idx = {x: i for i, x in enumerate(fit.candidates)}
    absents = [x for x in liste if x not in idx]
    if Z not in idx:
        return np.nan, [np.nan, np.nan], absents
    mask = np.zeros(len(fit.candidates)); mask[[idx[x] for x in liste if x in idx]] = 1
    d = pi_draws_for_mask(fit, mask)[:, idx[Z]]
    return float(d.mean()), [float(v) for v in np.percentile(d, [5, 95])], absents


def _liste_zemmour(c: pd.Timestamp) -> tuple[list[str] | None, int]:
    """Liste contenant Zemmour la mieux fournie (notices distinctes) avant c."""
    ch = su._champs_t1(sr.polls_historiques(E, c))
    ch = ch[ch.candidats.map(lambda l: Z in l)]
    if ch.empty:
        return None, 0
    n = ch.groupby("cle")["notice"].nunique().sort_index().sort_values(ascending=False, kind="stable")
    cle = n.index[0]
    return list(ch.loc[ch.cle.eq(cle), "candidats"].iloc[0]), int(n.iloc[0])


def _brut_liste(c: pd.Timestamp, liste: list[str]) -> float:
    """Moyenne 14 j de Zemmour sur les sondages de la liste EXACTE (parts renormalisées)."""
    raw = sr.polls_historiques(E, c, history_days=14)
    t1 = raw[raw.tour.eq("Premier tour")].copy()
    t1["sid"] = t1["notice"].astype(str) + "|" + t1["hypothese"].fillna("__u__").astype(str)
    v = []
    for _, g in t1.groupby("sid"):
        if cle_champ(sorted(set(g.candidat))) == cle_champ(liste):
            v.append(float(g.loc[g.candidat.eq(Z), "intention"].sum() / g.intention.sum()))
    return float(np.mean(v)) if v else np.nan


def _gp(c: pd.Timestamp, liste: list[str] = R) -> tuple[float | None, int]:
    raw = sr.polls_historiques(E, c)
    ch = su._champs_t1(raw)
    n = int(ch.loc[ch.cle.eq(cle_champ(liste)), "notice"].nunique())
    if n < 3:
        return None, n
    L = [{"cle": cle_champ(liste), "id": "L", "candidats": liste}]
    # Réutilise gp_listes sans cible : on lit directement le nowcast.
    from model.core.base import ELECTION_T1
    from model.listes import GPListe
    delta = ELECTION_T1 - sr.T1_DATES[E]
    t1 = raw[raw.tour.eq("Premier tour")].copy()
    t1["date_fin"] = pd.to_datetime(t1["date_fin"]) + delta
    t1["scenario_id"] = t1["notice"].astype(str) + "|" + t1["hypothese"].fillna("__u__").astype(str)
    t1["bloc"] = "?"
    mdl = GPListe(L[0]["cle"], liste)
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
        fit = _fit(h, c)
        sp, ic, absents = _spatial(fit, R)
        g, n_r = _gp(c)
        lc, n_lc = _liste_zemmour(c)
        if lc is not None:
            spl, icl, _ = _spatial(fit, lc)
            gl, _ = _gp(c, lc)
            bl = _brut_liste(c, lc)
        else:
            spl, icl, gl, bl = np.nan, [np.nan, np.nan], None, np.nan
        r4 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)
        lignes.append({"h": h, "coupure": str(c.date()), "brut_14j": r4(_brut(c)),
                       "spatial": r4(sp), "spatial_ic90": [r4(x) for x in ic],
                       "absents_du_roster": absents, "gp": r4(g), "n_sondages_R": n_r,
                       "liste_Lc": lc, "n_sondages_Lc": n_lc, "brut_14j_Lc": r4(bl),
                       "spatial_Lc": r4(spl), "spatial_Lc_ic90": [r4(x) for x in icl],
                       "gp_Lc": r4(gl)})
        print(lignes[-1], flush=True)
    dates = [l["coupure"] for l in lignes]
    series = ("brut_14j", "spatial", "gp", "brut_14j_Lc", "spatial_Lc", "gp_Lc")
    res = {"liste_R": R, "seuil_pt": 100 * SEUIL, "recul_j": RECUL, "lignes": lignes,
           "detection": {k: _detection(dates, [np.nan if l[k] is None else l[k] for l in lignes])
                         for k in series}}
    out = sr.OUT_DIR / "detection_zemmour_2022_v2.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(res["detection"], f"-> {out}")


if __name__ == "__main__":
    main()
