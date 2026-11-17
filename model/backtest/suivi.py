"""Backtest de suivi de tendance du spatial (spec_spatial_pooling §10, étape 1).

    python -m model.backtest.suivi --lister --election 2022
    python -m model.backtest.suivi --election 2022 --h 180 --chaine k
    python -m model.backtest.suivi --election 2022 --h 180 --fusionner

Un fit par coupure, sur les seuls sondages connus à la coupure ; listes =
règle du site (`model.listes.classement`) ; cible = sondages de la liste
exacte des 7 jours suivants (hors échantillon).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from model.backtest import second_round as sr
from model.listes import classement, cle_champ
from model.models.spatial_pooling.fit import fit_spatial_pooling, pi_draws_for_mask
from model.models.spatial_pooling.roster import ORDER_GROUPS_HISTORIQUES
from model.core.live_dataset import _fold

HORIZONS = list(range(159, 11, -21))     # 159, ..., 12 (180 : aucune cible)
H_PROJ = 7.0     # jours : milieu de la fenêtre cible
APRES = 14       # 7 j ne laissait que 0-2 sondages cibles par liste (spec §10, avant résultat)


def _champs_t1(raw: pd.DataFrame) -> pd.DataFrame:
    t1 = raw[raw.tour.eq("Premier tour")].copy()
    t1["scenario_id"] = t1["notice"].astype(str) + "|" + t1["hypothese"].fillna("__u__").astype(str)
    rows = []
    for sid, sub in t1.groupby("scenario_id", sort=False):
        noms = sorted(set(sub["candidat"].astype(str)), key=_fold)
        rows.append({"scenario_id": sid, "notice": sub["notice"].iloc[0],
                     "date_fin": pd.Timestamp(sub["date_fin"].max()),
                     "cle": cle_champ(noms), "candidats": noms})
    return pd.DataFrame(rows)


def cibles(election: int, c: pd.Timestamp, cle: str) -> tuple[dict, int]:
    """Moyenne par candidat des sondages de la liste exacte dans (c, c+7]."""
    fut = sr.polls_historiques(election, c + pd.Timedelta(days=APRES), history_days=APRES)
    fut = fut[fut.tour.eq("Premier tour") & fut.date_fin.gt(c)]
    if fut.empty:
        return {}, 0
    ch = _champs_t1(fut)
    sids = set(ch.loc[ch.cle.eq(cle), "scenario_id"])
    fut = fut.assign(scenario_id=fut["notice"].astype(str) + "|" + fut["hypothese"].fillna("__u__").astype(str))
    sel = fut[fut.scenario_id.isin(sids)]
    return ({k: float(v) / 100 for k, v in sel.groupby("candidat").intention.mean().items()},
            int(sel.scenario_id.nunique()))


GP_PARAMS: dict | None = None   # revue 16/11 : paramètres gp hors échantillon (REML sans 2022)
GP_DECALAGE_J = 0               # lecture de gp à c + décalage (date égale au spatial prolongé)


def gp_listes(election: int, c: pd.Timestamp, listes: list[dict], raw: pd.DataFrame) -> list[dict]:
    """Étape 2 : `gp-pooling` (GPListe, inchangé) sur les mêmes listes. Le modèle
    compte les horizons depuis le T1 2027 ; on décale les dates de
    Δ = T1_2027 − T1_e, ce qui redonne exactement les horizons au scrutin
    historique (noyau OU stationnaire, σ_h fonction de l'horizon)."""
    from model.core.base import ELECTION_T1
    from model.listes import GPListe
    delta = ELECTION_T1 - sr.T1_DATES[election]
    t1 = raw[raw.tour.eq("Premier tour")].copy()
    t1["date_fin"] = pd.to_datetime(t1["date_fin"]) + delta
    t1["scenario_id"] = t1["notice"].astype(str) + "|" + t1["hypothese"].fillna("__u__").astype(str)
    t1["bloc"] = "?"                      # simple clé de regroupement d aggregate_to_slots
    for col in ("commanditaire", "methode"):
        if col not in t1:
            t1[col] = None
    rng = np.random.default_rng(3)
    lignes = []
    for L in listes:
        try:
            mdl = GPListe(L["cle"], L["candidats"])
            mdl.load_artifacts()
            if GP_PARAMS is not None:
                mdl.params = {**mdl.params, **GP_PARAMS}
            nc = mdl.nowcast(t1, str((c + delta + pd.Timedelta(days=GP_DECALAGE_J)).date()))
        except Exception as exc:          # liste sans sondage exploitable : comptée
            lignes.append({"liste": L["id"], "erreur": f"{type(exc).__name__}: {exc}"})
            continue
        obs, n = cibles(election, c, L["cle"])
        lab = {x: mdl.labels[x] for x in L["candidats"]}
        draws = nc.draws
        for x in L["candidats"]:
            if x not in obs:
                continue
            d = np.asarray(draws.sel(candidat=lab[x]))
            dp = d + rng.standard_normal(d.shape) * np.sqrt(d * (1 - d) / 1000 / max(n, 1))
            lo, hi = np.percentile(dp, [5, 95])
            lignes.append({"liste": L["id"], "candidat": x, "observe": round(obs[x], 4), "n_cible": n,
                           "modele": round(float(d.mean()), 4), "ic90": [round(lo, 4), round(hi, 4)],
                           "erreur_pt": round(100 * abs(float(d.mean()) - obs[x]), 2),
                           "signe_pt": round(100 * (float(d.mean()) - obs[x]), 2),
                           "couvre": bool(lo <= obs[x] <= hi)})
    return lignes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--election", type=int, choices=[2017, 2022], default=2022)
    ap.add_argument("--h", type=int, default=None)
    ap.add_argument("--lister", action="store_true")
    ap.add_argument("--chaine", type=int, default=None)
    ap.add_argument("--fusionner", action="store_true")
    ap.add_argument("--gp", action="store_true", help="étape 2 : gp-pooling sur les mêmes listes")
    ap.add_argument("--gp-params", default=None, help="JSON de paramètres gp (REML hors échantillon)")
    ap.add_argument("--gp-facteur", type=float, default=3.0, help="facteur de diffusion appliqué à sigma2 et tau")
    ap.add_argument("--gp-decalage", type=int, default=0, help="lecture de gp à c + N jours")
    ap.add_argument("--suffixe", default="")
    ap.add_argument("--sigma-w-candidat", action="store_true", help="piste C : σ_w par candidat")
    a = ap.parse_args()
    if a.lister:
        for h in HORIZONS:
            print(f"--election {a.election} --h {h}")
        return
    c = sr.T1_DATES[a.election] - pd.Timedelta(days=a.h)
    raw = sr.polls_historiques(a.election, c)
    global GP_PARAMS, GP_DECALAGE_J
    GP_DECALAGE_J = a.gp_decalage
    if a.gp_params:
        q = json.loads(open(a.gp_params).read())
        GP_PARAMS = {"sigma_h": q["sigma_h"], "sigma_p": q["sigma_p"],
                     "sigma2": q["sigma2"] * a.gp_facteur, "tau": q["tau"] * a.gp_facteur}
    if a.gp:
        listes = classement(_champs_t1(raw), c)
        res = {"election": a.election, "h": a.h, "coupure": str(c.date()), "modele": "gp-pooling",
               "listes": listes, "lignes": gp_listes(a.election, c, listes, raw)}
        out = sr.OUT_DIR / f"suivi_gp{a.suffixe}_{a.election}_h{a.h:03d}.json"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str) + "\n")
        print(f"-> {out}")
        return
    swc = "_swc" if a.sigma_w_candidat else ""
    out = sr.OUT_DIR / f"suivi{swc}_{a.election}_h{a.h:03d}.json"
    base = sr.CHAINES_DIR / out.stem
    stem = out.stem                    # nom des fichiers de chaînes (inchangé)
    nuts = dict(draws=400, tune=600, chains=4)
    if a.chaine is not None:
        nuts["mcmc_runner"] = sr._runner_chaine(base.with_name(f"{stem}_c{a.chaine}.npz"), a.chaine)
    elif a.fusionner:
        nuts["mcmc_runner"] = sr._runner_fusion([base.with_name(f"{stem}_c{k}.npz")
                                                 for k in range(sr.N_CHAINES)])
    else:
        ap.error("--chaine k ou --fusionner")
    if H_PROJ:
        out = out.with_name(f"suivi_proj{swc}_{a.election}_h{a.h:03d}.json")
    fit = fit_spatial_pooling(raw, str(c.date()), min_poll_date="1900-01-01",
                              order_groups=ORDER_GROUPS_HISTORIQUES[a.election],
                              couverture_min=None, second_tour=False,
                              sigma_w_par_candidat=a.sigma_w_candidat, **nuts)
    if a.chaine is not None:
        return
    listes = classement(_champs_t1(raw), c)
    idx = {x: i for i, x in enumerate(fit.candidates)}
    lignes, ecartees = [], []
    rng = np.random.default_rng(3)
    for L in listes:
        if not all(x in idx for x in L["candidats"]):
            ecartees.append({"candidats": L["candidats"],
                             "hors_roster": [x for x in L["candidats"] if x not in idx]})
            continue
        mask = np.zeros(len(fit.candidates)); mask[[idx[x] for x in L["candidats"]]] = 1
        # Cible étalée sur (c, c+14] : on prolonge le chemin OU de h = 7 j (même
        # processus que la projection du modèle), sinon l'IC ignore toute dérive
        # après la coupure (couverture 0,73 mesurée sans prolongement).
        from model.models.spatial_pooling.fit import projeter_w
        from model.models.spatial_pooling.geometry import spatial_shares
        import jax.numpy as jnp
        pi = np.asarray(spatial_shares(jnp.asarray(fit.mu_draws), jnp.asarray(fit.sigma_draws),
                                       jnp.asarray(projeter_w(fit, H_PROJ)), jnp.asarray(mask)))
        obs, n = cibles(a.election, c, L["cle"])
        for x in L["candidats"]:
            if x not in obs:
                continue
            d = pi[:, idx[x]]
            # IC prédictif de la MOYENNE de n sondages (n ≈ 1000 répondants).
            dp = d + rng.standard_normal(d.shape) * np.sqrt(d * (1 - d) / 1000 / max(n, 1))
            lo, hi = np.percentile(dp, [5, 95])
            lignes.append({"liste": L["id"], "candidat": x, "observe": round(obs[x], 4), "n_cible": n,
                           "modele": round(float(d.mean()), 4), "ic90": [round(lo, 4), round(hi, 4)],
                           "erreur_pt": round(100 * abs(float(d.mean()) - obs[x]), 2),
                           "signe_pt": round(100 * (float(d.mean()) - obs[x]), 2),
                           "couvre": bool(lo <= obs[x] <= hi)})
    res = {"election": a.election, "h": a.h, "coupure": str(c.date()),
           "listes": listes, "listes_ecartees": ecartees,
           "chaines_retenues": list(sr.CHAINES_RETENUES), "lp_par_chaine": list(sr.LP_CHAINES),
           "diagnostics": {k: fit.diagnostics.get(k) for k in
                           ("rhat_max", "ess_min", "n_divergences", "n_noeuds", "n_champs_distincts")},
           "lignes": lignes}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str) + "\n")
    print(f"-> {out}  ({len(lignes)} lignes, {len(ecartees)} listes écartées)")


if __name__ == "__main__":
    main()
