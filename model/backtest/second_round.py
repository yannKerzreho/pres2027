"""Backtest du second tour de spatial-pooling (spec_second_tour.md §8).

    python -m model.backtest.second_round                 # tout (≈ 13 fits NUTS : Colab)
    python -m model.backtest.second_round --fast          # écran de fumée (1 chaîne)
    python -m model.backtest.second_round --only v2 --election 2022

V1 « paire retirée » : à T1−1 j, retirer TOUS les sondages d'une paire, ajuster,
prédire la moyenne de ces sondages sur les 30 derniers jours. C'est le test des
reports : la paire n'est vue qu'à travers la géométrie et les autres duels.
Comparé, sur les MÊMES tirages, à la géométrie seule (κ = 1, η = 0) et à un
Bradley-Terry pur (η seul, ajusté sur les autres paires).

V2 « résultat » : à T1−30 j, T1−1 j et T2−7 j, prédire le résultat réel du duel
final (dérive OU jusqu'au T2 + erreur de scrutin). Couverture IC90, erreur,
log-score, contre la moyenne brute des 7 derniers jours de sondages de la paire
± la même erreur de scrutin.

Limites connues (relecture du 2026-11-14) : TAU_T2_SCRUTIN est calibré sur
les erreurs 2017/2022 elles-mêmes (couverture V2 DANS l'échantillon) ; le
champ T1 réel est connu a posteriori (`names_t1`) ; les priors κ/η ont été
choisis en regardant 2027 ; 2017 est presque sans dates de terrain.

Aucun verdict automatique : deux élections ne valident rien statistiquement.
Le rapport est écrit dans `model/backtest/results/` pour être lu.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pipeline.historical import (PARSED_DIR, POLL_FILES, _resolve_candidate,
                                 _select_best_hypothesis, load_results)
from model.core.inference import run_numpyro_mcmc
from model.models.spatial_pooling.fit import (TAU_T2_SCRUTIN, duel_draws,
                                              fit_spatial_pooling)
from model.models.spatial_pooling.geometry import duel_shares, log_attractivite
from model.models.spatial_pooling.roster import ORDER_GROUPS_HISTORIQUES

T1_DATES = {2017: pd.Timestamp("2017-04-23"), 2022: pd.Timestamp("2022-04-10")}
T2_DATES = {2017: pd.Timestamp("2017-05-07"), 2022: pd.Timestamp("2022-04-24")}
FINALE = {2017: ("Emmanuel Macron", "Marine Le Pen"), 2022: ("Emmanuel Macron", "Marine Le Pen")}
MIN_SONDAGES_PAIRE = 3
OUT_DIR = Path(__file__).resolve().parent / "results"
# Tirages bruts par chaîne (hors git) : un fit complet en 4 chaînes
# séquentielles dépasse un réveil de la routine ; on lance 4 processus d'une
# chaîne (un par cœur), puis on fusionne. `chain_method="parallel"` dans un seul
# processus a été essayé : fuite mémoire JAX/CPU (~100 Mo/s, OOM à 15 Go en
# 2 min), le séquentiel reste plat à 1,4 Go.
CHAINES_DIR = OUT_DIR / "chaines"
# Surchargeable (revue locale du 20/11 PM, point 1.5) : runs à 8 chaînes (§8.50) ;
# sans cela `--fusionner` et `eval_r2` ne lisaient que c0..c3 sans prévenir.
N_CHAINES = int(__import__("os").environ.get("PRES2027_N_CHAINES", "4"))


def _runner_chaine(path: Path, k: int, init: dict | None = None):
    """NUTS d'UNE chaîne (graine `seed + k`), tirages sauvés dans `path`.
    `init` : valeurs des sites latents (départ commun, spec §8.8 phase 2)."""
    def run(model_fn, model_kwargs, **kw):
        kw = {**kw, "chains": 1, "seed": kw.get("seed", 27) + k}
        if init is not None:
            from numpyro.infer import init_to_value
            kw["init_strategy"] = init_to_value(values=init)
        samples, extra = run_numpyro_mcmc(model_fn, model_kwargs, **kw)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f"s__{n}": np.asarray(v) for n, v in samples.items()},
                            **{f"e__{n}": np.asarray(v).reshape(-1) for n, v in extra.items()})
        return samples, extra
    return run


def _runner_fusion(paths: list[Path]):
    """Aucun NUTS : concatène les chaînes sauvées (axe chaîne) ; R-hat/ESS sont
    ensuite calculés par `fit_spatial_pooling` exactement comme en 4 chaînes."""
    manquants = [str(p) for p in paths if not p.exists()]
    if manquants:
        raise FileNotFoundError(f"chaînes manquantes : {manquants}")

    def run(model_fn, model_kwargs, **kw):
        zs = [np.load(p) for p in paths]
        noms = zs[0].files
        samples = {n[3:]: np.concatenate([z[n] for z in zs], axis=0) for n in noms if n.startswith("s__")}
        extra = {n[3:]: np.concatenate([z[n] for z in zs], axis=0) for n in noms if n.startswith("e__")}
        lp, se = _log_densite_par_chaine(model_fn, model_kwargs, samples)
        keep = _chaines_retenues(lp, se)
        if STACKING:
            w = poids_stacking(_lpd_loo_par_chaine(model_kwargs, samples))
            POIDS_STACKING[:] = [round(float(x), 4) for x in w]
            C, D = samples[next(iter(samples))].shape[:2]
            k = np.round(w * C * D).astype(int)
            idx = [(c, t) for c in range(C) for t in np.linspace(0, D - 1, k[c]).astype(int)] if k.sum() else []
            ci, ti = np.array([i for i, _ in idx]), np.array([t for _, t in idx])
            LP_CHAINES[:] = lp
            LP_SE[:] = se
            CHAINES_RETENUES[:] = []
            samples = {n: v[ci, ti][None] for n, v in samples.items()}
            extra = {n: v.reshape(C, D, *v.shape[1:])[ci, ti] for n, v in extra.items()}
            return samples, extra
        LP_CHAINES[:] = lp
        LP_SE[:] = se
        CHAINES_RETENUES[:] = keep.tolist()
        # Revue 2026-11-15, point 4 : TOUT l'aval (IC, couverture, log-score,
        # géométrie seule, κ/ω/η, R-hat) ne voit que les chaînes retenues.
        D = samples[next(iter(samples))].shape[1]
        samples = {n: v[keep] for n, v in samples.items()}
        extra = {n: v.reshape(len(zs), D, *v.shape[1:])[keep].reshape(-1, *v.shape[1:])
                 for n, v in extra.items()}
        return samples, extra
    return run


def _init_mode_dominant(json_phase1: str) -> dict:
    """Dernier tirage de la chaîne de plus grand L̄_c d'une phase 1 fusionnée."""
    p = Path(json_phase1)
    p = p if p.exists() else OUT_DIR / p
    r = json.loads(p.read_text())
    ligne = (r.get("v1") or r.get("v2"))[0]
    best = int(np.argmax(ligne["lp_par_chaine"]))
    z = np.load(CHAINES_DIR / f"{p.stem}_c{best}.npz")
    return {n[3:]: np.asarray(z[n][0, -1]) for n in z.files if n.startswith("s__")}


# Log-densité moyenne par chaîne de la dernière fusion (spec §8.8, révisée
# après la revue du 2026-11-15 : L̄ sur TOUS les tirages, seuil tenant compte
# de l'erreur-type ; 20 tirages donnaient ~4 nats d'erreur sur un écart).
LP_CHAINES: list[float] = []
LP_SE: list[float] = []
CHAINES_RETENUES: list[int] = []
DELTA_LP = 10.0          # nats : seuil minimal de rétention des chaînes
N_LOTS_SE = 10           # lots pour l'erreur-type de L̄ (moyennes par lots)


def _log_densite_par_chaine(model_fn, model_kwargs, samples) -> tuple[list[float], list[float]]:
    """(L̄_c, SE_c) : moyenne de log p(θ, y) sur tous les tirages de chaque chaîne,
    erreur-type par moyennes de `N_LOTS_SE` lots (robuste à l'autocorrélation)."""
    import jax
    import jax.numpy as jnp
    from numpyro import handlers
    from numpyro.infer.util import log_density
    tr = handlers.trace(handlers.seed(model_fn, 0)).get_trace(**model_kwargs)
    sites = [n for n, v in tr.items() if v["type"] == "sample" and not v["is_observed"]]
    f = jax.jit(lambda prm: log_density(model_fn, (), model_kwargs, prm)[0])
    C, D = samples[sites[0]].shape[:2]
    moy, se = [], []
    for c in range(C):
        L = np.array([float(f({n: jnp.asarray(samples[n][c, t]) for n in sites})) for t in range(D)])
        lots = np.array_split(L, N_LOTS_SE)
        moy.append(float(L.mean()))
        se.append(float(np.std([b.mean() for b in lots], ddof=1) / np.sqrt(len(lots))))
    return moy, se


def _lpd_loo_par_chaine(model_kwargs, samples) -> np.ndarray:
    """(C, n) log densité prédictive LOO (IS tronqué) de chaque observation —
    nœud T1 ou duel — sous chaque chaîne (spec §8.26)."""
    from scipy.special import logsumexp
    from scipy.stats import norm
    Y = np.asarray(model_kwargs["Y"]); Np = np.asarray(model_kwargs["Np"])
    tm = np.asarray(model_kwargs["tested_mask"])
    ev = model_kwargs.get("excess_var")
    ev = 0.0 if ev is None else np.asarray(ev)[:, None]
    pi = samples["pi"]                                          # (C, D, nœuds, N)
    var = pi * (1 - pi) / np.clip(Np[None, None, :, None], 1, None) + ev
    ll = (norm.logpdf(Y, pi, np.sqrt(var + 1e-8)) * tm).sum(-1)  # (C, D, nœuds)
    if "duel_s" in samples and model_kwargs.get("duel_y") is not None:
        s = samples["duel_s"]; y = np.asarray(model_kwargs["duel_y"])
        n = np.asarray(model_kwargs["duel_n"]); st = samples["sigma_t2"][..., None]
        vd = s * (1 - s) / np.clip(n, 1, None) + st ** 2
        ll = np.concatenate([ll, norm.logpdf(y, s, np.sqrt(vd + 1e-8))], axis=-1)
    C, D = ll.shape[:2]
    lr = -ll                                                    # log r_s
    lr_max = logsumexp(lr, axis=1, keepdims=True) - np.log(D) + 0.5 * np.log(D)
    lr = np.minimum(lr, lr_max)
    return logsumexp(lr + ll, axis=1) - logsumexp(lr, axis=1)   # (C, n)


def poids_stacking(lpd: np.ndarray) -> np.ndarray:
    """argmax_w Σ_i log Σ_c w_c exp(lpd_ci), w sur le simplexe (spec §8.26)."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp, softmax
    C = lpd.shape[0]
    if C == 1:
        return np.ones(1)
    f = lambda a: -logsumexp(lpd + np.log(softmax(np.r_[0.0, a]))[:, None], axis=0).sum()
    a = minimize(f, np.zeros(C - 1), method="L-BFGS-B").x
    return softmax(np.r_[0.0, a])


STACKING = False                 # --stacking (fusion seulement, spec §8.26)
POIDS_STACKING: list = []


def _chaines_retenues(lp, se) -> np.ndarray:
    """Chaînes c avec L̄_c ≥ L̄_best − max(Δ, 4·SE de l'écart) (spec §8.8)."""
    lp, se = np.asarray(lp, float), np.asarray(se, float)
    b = int(np.argmax(lp))
    seuil = np.maximum(DELTA_LP, 4.0 * np.sqrt(se ** 2 + se[b] ** 2))
    return np.flatnonzero(lp >= lp[b] - seuil)


def polls_historiques(election: int, as_of: pd.Timestamp, history_days: int = 180,
                      toutes_hypotheses: bool = True) -> pd.DataFrame:
    """Sondages wiki ≤ as_of dans le vocabulaire des résultats officiels.
    T1 : TOUTES les hypothèses de toutes les notices, candidats putatifs
    compris (spec_second_tour.md §8.3) — comme le roster 2027, c'est la
    variation des champs qui identifie les positions. T2 : TOUTES les paires
    de deux candidats du scrutin (les duels hypothétiques sont ceux qui
    renseignent les reports)."""
    raw = pd.read_csv(Path(PARSED_DIR) / POLL_FILES[election])
    res = load_results()
    res_e = res[res.election.eq(election)]
    names = set(res_e["candidat"])
    names_t1 = set(res_e.loc[res_e.tour.eq("Premier tour"), "candidat"])
    raw["candidat"] = raw["candidat"].apply(lambda x: _resolve_candidate(x, names))
    dates = pd.to_datetime(raw["date_fin"], errors="coerce")
    raw["date_fin"] = dates.fillna(pd.to_datetime(raw.get("date_notice"), errors="coerce"))
    raw = raw[raw["date_fin"].notna() & raw["date_fin"].le(as_of)
              & raw["date_fin"].ge(as_of - pd.Timedelta(days=history_days))].copy()
    t1 = raw[raw.tour.eq("Premier tour")]
    if not toutes_hypotheses:
        # Ancien cadre : une hypothèse par notice, candidats du scrutin seuls.
        # Géométrie NON identifiée (100/108 nœuds au même champ, spec §8.2).
        t1 = _select_best_hypothesis(t1[t1.candidat.isin(names_t1)], names_t1)
    t2 = raw[raw.tour.eq("Deuxième tour")].copy()
    t2["_h"] = t2["hypothese"].fillna("__u__")
    ok = t2.groupby(["notice", "_h"])["candidat"].transform(
        lambda x: len(x) == 2 and x.isin(names_t1).all() and x.nunique() == 2)
    t2 = t2[ok.astype(bool)].drop(columns="_h")
    return pd.concat([t1, t2], ignore_index=True)


N_NOEUD_RESULTAT = 20000   # spec §8.25 : résultat T1 comme nœud quasi exact
NOEUD_RESULTAT = False     # surchargé par --noeud-resultat


def avec_noeud_resultat(raw: pd.DataFrame, election: int) -> pd.DataFrame:
    """Ajoute le résultat officiel du T1 comme un « sondage » de N = 20 000 au
    jour du T1, champ exact des candidats du scrutin (spec §8.25). N'a de sens
    qu'après le T1 (coupures T2−k)."""
    res = load_results()
    r = res[res.election.eq(election) & res.tour.eq("Premier tour")]
    lignes = pd.DataFrame({"notice": f"RESULTAT_T1_{election}", "institut": "Résultat officiel",
                           "tour": "Premier tour", "hypothese": "resultat", "candidat": r.candidat.to_numpy(),
                           "intention": r.resultat.to_numpy(float), "echantillon": N_NOEUD_RESULTAT,
                           "date_fin": T1_DATES[election]})
    return pd.concat([raw, lignes], ignore_index=True)


def _duels_moyens(raw: pd.DataFrame, depuis: pd.Timestamp | None = None) -> pd.DataFrame:
    """Une ligne par duel sondé : paire triée, part du premier nom, date."""
    t2 = raw[raw.tour.eq("Deuxième tour")].copy()
    if depuis is not None:
        t2 = t2[t2.date_fin.ge(depuis)]
    rows = []
    for (_, _), g in t2.groupby(["notice", t2["hypothese"].fillna("__u__")]):
        g = g.sort_values("candidat")
        v = g.intention.to_numpy(float)
        rows.append((g.candidat.iloc[0], g.candidat.iloc[1], v[0] / v.sum(), g.date_fin.iloc[0],
                     g["institut"].iloc[0] if "institut" in g else "?"))
    return pd.DataFrame(rows, columns=["a", "b", "y", "date", "institut"])


def _sans_paire(raw: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    t2 = raw.tour.eq("Deuxième tour")
    key = raw["notice"].astype(str) + "|" + raw["hypothese"].fillna("__u__").astype(str)
    paires = raw[t2].groupby(key[t2])["candidat"].agg(lambda x: frozenset(x))
    drop = set(paires[paires.eq(frozenset((a, b)))].index)
    return raw[~(t2 & key.isin(drop))]


def _geometrie_seule(fit, a: str, b: str) -> np.ndarray:
    i = {c: k for k, c in enumerate(fit.candidates)}
    A = np.asarray(log_attractivite(np.asarray(fit.mu_draws), np.asarray(fit.sigma_draws),
                                    np.asarray(fit.w_draws),
                                    None if getattr(fit, "nu_t_draws", None) is None else np.asarray(fit.nu_t_draws)))
    return np.asarray(duel_shares(A[:, i[a]], A[:, i[b]], 1.0, 0.0, 0.0))


def _bradley_terry(train: pd.DataFrame, a: str, b: str, sd_eta: float = 0.5) -> float:
    """η seul, ridge (même prior que le modèle), sur la moyenne logit par paire."""
    m = train.assign(z=np.log(train.y / (1 - train.y))).groupby(["a", "b"]).z.agg(["mean", "count"])
    cands = sorted(set(m.index.get_level_values(0)) | set(m.index.get_level_values(1)) | {a, b})
    ci = {c: k for k, c in enumerate(cands)}

    def loss(e):
        r = sum(n * (e[ci[x]] - e[ci[y]] - z) ** 2 for (x, y), (z, n) in m.iterrows())
        return r / (2 * 0.1 ** 2) + (e ** 2).sum() / (2 * sd_eta ** 2)
    e = minimize(loss, np.zeros(len(cands)), method="L-BFGS-B").x
    return float(1 / (1 + np.exp(-(e[ci[a]] - e[ci[b]]))))


ABSTENTION = True   # surchargé par --variante
VARIANTE_KW: dict = {}   # geo / geo1 (spec §8.9)
SECOND_TOUR = True  # False = --sans-t2 : diagnostic de convergence du T1 seul
POS_SCALE_BT = None  # --pos-scale : EXPÉRIENCE de diagnostic (spec §8.7), pas un réglage


def _fit(raw, as_of, election, nuts):
    t = time.time()
    kw = dict(VARIANTE_KW)
    reps = []
    if kw.pop("_reports", False):
        # R2 (spec_second_tour §8.44.4) : reports T1→T2 en vraisemblance ; seuls
        # ceux datés au plus `as_of` entrent (filtre dans `arrays_reports`).
        reps.append(pd.read_csv(Path(PARSED_DIR) / f"reports_t2_{election}.csv"))
    if kw.pop("_reports_avant", False):
        # R3b (spec_second_tour §8.47.5) : reports publiés AVANT le T1 (intention T1),
        # dans la fenêtre des sondages T1 [as_of − 180 j ; as_of].
        ra = pd.read_csv(Path(PARSED_DIR) / f"reports_avant_t1_{election}.csv")
        ra = ra[pd.to_datetime(ra["date_fin"]) >= as_of - pd.Timedelta(days=180)]
        reps.append(ra.drop(columns=[c for c in ("nature_t1", "page") if c in ra.columns]))
    if kw.pop("_lse_officiel", False):
        # §8.49 : valeur inclusive sur le champ officiel du T1 (--lse-officiel)
        res = load_results()
        kw["lse_officiel"] = res[(res.election == election) & (res.tour == "Premier tour")].candidat.tolist()
    if reps:
        res = load_results()
        kw["reports"] = pd.concat(reps, ignore_index=True)
        kw["champ_reports"] = res[(res.election == election) & (res.tour == "Premier tour")].candidat.tolist()
    fit = fit_spatial_pooling(raw, str(as_of.date()), min_poll_date="1900-01-01",
                              order_groups=ORDER_GROUPS_HISTORIQUES[election],
                              couverture_min=None, abstention=ABSTENTION,
                              second_tour=SECOND_TOUR, **kw,
                              **({"pos_scale": POS_SCALE_BT} if POS_SCALE_BT else {}), **nuts)
    return fit, round(time.time() - t, 1)


def _diag(fit, dt):
    d = fit.diagnostics
    return {"rhat_max": d["rhat_max"], "ess_min": d["ess_min"], "divergences": d["n_divergences"],
            "kappa": d.get("kappa"), "omega": d.get("omega"), "rhat_t2": d.get("rhat_t2"), "sigma_t2": d.get("sigma_t2"), "secondes": dt,
            "sigma_maison_t2": d.get("sigma_maison_t2"), "sigma_notice_t2": d.get("sigma_notice_t2"),
            "eta": {c: round(float(x), 3) for c, x in zip(fit.candidates, fit.eta_draws.mean(0))}
            if fit.eta_draws is not None else None,
            # carte 2D (spec §8.44) : positions en u, λ, rayon moyen en v
            **({"nu": {c: round(float(x), 3) for c, x in zip(fit.candidates, fit.nu_draws.mean(0))},
                "lambda_u": round(float(fit.lambda_draws.mean()), 4)} if fit.nu_draws is not None else {}),
            **({"arc": round(float(fit.arc_draws.mean()), 4),
                "f_arriere": round(float(1 - fit.arc_draws.mean() / (2 * np.pi)), 4)} if fit.arc_draws is not None else {}),
            "sigma_v_moyen": round(float(fit.sigma_draws.mean()), 4)}


def v1_paire_retiree(election: int, nuts: dict, seule: tuple | None = None) -> list[dict]:
    as_of = T1_DATES[election] - pd.Timedelta(days=1)
    raw = polls_historiques(election, as_of)
    duels = _duels_moyens(raw, as_of - pd.Timedelta(days=30))
    tous = _duels_moyens(raw)
    paires = tous.groupby(["a", "b"]).size()
    out = []
    for (a, b), n in paires.items():
        if n < MIN_SONDAGES_PAIRE or (seule is not None and (a, b) != tuple(sorted(seule))):
            continue
        cible = duels[(duels.a == a) & (duels.b == b)]
        if cible.empty:
            continue
        print(f"[V1 {election}] paire retirée {a} / {b} ({n} sondages)", flush=True)
        fit, dt = _fit(_sans_paire(raw, a, b), as_of, election, nuts)
        obs = float(cible.y.mean())
        mod = duel_draws(fit, a, b, 0, scrutin=False)
        cible = _cible_par_chaine(mod, nuts.get("chains", 1))
        cible_df = duels[(duels.a == a) & (duels.b == b)]
        mod_cible, inst_inconnus = recentre_maisons(fit, mod, a, b, cible_df.institut)
        dp = predictif_moyenne(mod_cible, len(cible_df), 1000.0, fit.diagnostics.get("sigma_t2", 0.0),
                               var_effets=var_effets_moyenne(fit, cible_df.institut, inst_inconnus))
        cible.update({"ic90_predictif": [round(float(v), 4) for v in np.percentile(dp, [5, 95])],
                      "couvre_predictif": bool(np.percentile(dp, 5) <= obs <= np.percentile(dp, 95)),
                      "log_score_predictif": round(_log_score(np.log(dp / (1 - dp)), obs), 3),
                      "tirages_predictifs": [round(float(v), 4) for v in dp],  # §8.20 (KDE)
                      "n_cible": int(len(cible_df))})
        geo = _geometrie_seule(fit, a, b)
        bt = _bradley_terry(tous[~((tous.a == a) & (tous.b == b))], a, b)
        autres = tous[((tous.a.isin([a, b])) | (tous.b.isin([a, b]))) & ~((tous.a == a) & (tous.b == b))]
        out.append({"election": election, "paire": [a, b], "n_sondages_retires": int(n),
                    "observe_30j": round(obs, 4), "modele": round(float(mod.mean()), 4),
                    "modele_recentre_maisons": round(float(mod_cible.mean()), 4),
                    "modele_ic90": [round(float(x), 4) for x in np.percentile(mod, [5, 95])],
                    "geometrie_seule": round(float(geo.mean()), 4), "bradley_terry": round(bt, 4),
                    "autres_duels_des_deux": int(len(autres)), **cible, **_diag(fit, dt)})
    return out


def _cible_par_chaine(d: np.ndarray, n_chaines: int) -> dict:
    """Convergence DE LA QUANTITÉ PRÉDITE (spec §8.4) sur les chaînes du fit (après
    fusion : les seules chaînes retenues, §8.8). Tirages rangés chaîne par chaîne."""
    if CHAINES_RETENUES:
        n_chaines = len(CHAINES_RETENUES)
    out = {}
    if LP_CHAINES:
        out.update({"lp_par_chaine": [round(float(v), 1) for v in LP_CHAINES],
                    "lp_se": [round(float(v), 2) for v in LP_SE],
                    "chaines_retenues": list(CHAINES_RETENUES)})
    if n_chaines < 1 or len(d) % n_chaines:
        return out
    from numpyro.diagnostics import split_gelman_rubin
    x = np.log(d / (1 - d)).reshape(n_chaines, -1)
    m = (1 / (1 + np.exp(-x))).mean(1)
    out.update({"modele_par_chaine": [round(float(v), 4) for v in m],
                "ecart_chaines_pt": round(100 * float(m.max() - m.min()), 2)})
    if n_chaines >= 2:
        out["rhat_cible"] = round(float(split_gelman_rubin(x[..., None])[0]), 3)
    out["concluant"] = bool(n_chaines >= 2 and out["rhat_cible"] <= 1.1
                            and out["ecart_chaines_pt"] <= 1.0)
    return out


def recentre_maisons(fit, d: np.ndarray, a: str, b: str, instituts):
    """§8.28 / revue 17/11 point 3 : la cible est une moyenne de sondages
    d'instituts connus ; on la recentre tirage par tirage par leurs effets
    maison a posteriori, u_{h,a} − u_{h,b}. Instituts absents du fit : effet
    non appris, laissé en variance. Renvoie (tirages, liste des inconnus)."""
    if fit.maison_draws is None or not fit.maisons or len(instituts) == 0:
        return d, list(instituts)
    ci = {c: k for k, c in enumerate(fit.candidates)}
    hi = {h: k for k, h in enumerate(fit.maisons)}
    U = fit.maison_draws                                        # (S, H, N)
    dec, inconnus = np.zeros(len(d)), []
    for h in instituts:
        if str(h) in hi:
            dec += U[:, hi[str(h)], ci[a]] - U[:, hi[str(h)], ci[b]]
        else:
            inconnus.append(h)
    return np.clip(d + dec / len(instituts), 1e-4, 1 - 1e-4), inconnus


def var_effets_moyenne(fit, instituts, inconnus=None) -> float:
    """Var. des effets maison/notice d'une moyenne de n duels d'une même paire
    (§8.23 bis) : [2σ_h² Σ_h n_h² + 2σ_n² n] / n² (σ_h = 0 : effet notice seul).
    `inconnus` : si donné, seuls ces instituts portent une variance maison (les
    autres ont été recentrés par `recentre_maisons`)."""
    sn = fit.diagnostics.get("sigma_notice_t2")
    if sn is None or len(instituts) == 0:
        return 0.0
    sh = fit.diagnostics.get("sigma_maison_t2") or 0.0
    n = len(instituts)
    base = list(instituts) if inconnus is None or fit.maison_draws is None else list(inconnus)
    nh = pd.Series(base).value_counts().to_numpy(float) if base else np.zeros(0)
    return float((2 * sh ** 2 * (nh ** 2).sum() + 2 * sn ** 2 * n) / n ** 2)


def predictif_moyenne(d: np.ndarray, n: int, n_ech: float, sigma_t2: float,
                      seed: int = 11, var_effets: float = 0.0) -> np.ndarray:
    """Tirages prédictifs de la MOYENNE de n sondages d'un duel (spec §8.14) ;
    `var_effets` : variance des effets maison/notice de cette moyenne (§8.23 bis)."""
    rng = np.random.default_rng(seed)
    sd = np.sqrt((d * (1 - d) / max(n_ech, 1.0) + sigma_t2 ** 2) / max(n, 1) + var_effets)
    return np.clip(d + sd * rng.standard_normal(d.shape), 1e-4, 1 - 1e-4)


def _log_score(draws_logit: np.ndarray, y: float) -> float:
    m, s = draws_logit.mean(), draws_logit.std()
    z = np.log(y / (1 - y))
    return float(-0.5 * np.log(2 * np.pi * s ** 2) - (z - m) ** 2 / (2 * s ** 2) - np.log(y * (1 - y)))


def v2_resultat(election: int, nuts: dict, coupures=("T1-30", "T1-1", "T2-7")) -> list[dict]:
    a, b = sorted(FINALE[election])
    res = load_results()
    vrai = res[(res.election == election) & (res.tour == "Deuxième tour")].set_index("candidat")
    y = float(vrai.loc[a, "resultat"]) / (vrai.loc[a, "resultat"] + vrai.loc[b, "resultat"])
    out = []
    for c in coupures:
        ref = T1_DATES[election] if c.startswith("T1") else T2_DATES[election]
        as_of = ref - pd.Timedelta(days=int(c.split("-")[1]))
        print(f"[V2 {election}] {c} ({as_of.date()})", flush=True)
        raw = polls_historiques(election, as_of)
        if NOEUD_RESULTAT:
            if as_of <= T1_DATES[election]:
                raise ValueError("--noeud-resultat : coupure postérieure au T1 seulement")
            raw = avec_noeud_resultat(raw, election)
        n_t1 = int(raw.loc[raw.tour.eq("Premier tour"), "notice"].nunique())
        try:
            fit, dt = _fit(raw, as_of, election, nuts)
        except (IndexError, ValueError) as exc:
            # 2017 : la plupart des sondages T1 n'ont aucune date (cf. note
            # données) -- une coupure peut n'avoir aucun nœud T1.
            out.append({"election": election, "coupure": c, "as_of": str(as_of.date()),
                        "erreur": f"{type(exc).__name__}: {exc}", "n_notices_t1": n_t1})
            continue
        if fit.kappa_draws is None:
            # --sans-t2 : pas de duel, on ne garde que la convergence du T1.
            d = fit.diagnostics
            out.append({"election": election, "coupure": c, "as_of": str(as_of.date()),
                        "sans_t2": True, "n_noeuds": d["n_noeuds"],
                        "n_champs_distincts": d["n_champs_distincts"],
                        "rhat_max": d["rhat_max"], "rhat_pires_sites": d["rhat_pires_sites"],
                        "ess_min": d["ess_min"], "divergences": d["n_divergences"],
                        "pas_leapfrog_median": d["pas_leapfrog_median"],
                        "mu": {k: round(float(x), 3) for k, x in zip(fit.candidates, fit.mu_draws.mean(0))}})
            continue
        h = int((T2_DATES[election] - as_of).days)
        d = duel_draws(fit, a, b, h, scrutin=True)
        # Cible SANS l'aléa de scrutin (tiré indépendamment, il ne dit rien des chaînes).
        cible = _cible_par_chaine(duel_draws(fit, a, b, h, scrutin=False), nuts.get("chains", 1))
        recents = _duels_moyens(raw, as_of - pd.Timedelta(days=7))
        recents = recents[(recents.a == a) & (recents.b == b)]
        if recents.empty:
            recents = _duels_moyens(raw).query("a == @a and b == @b").tail(5)
        p0 = float(recents.y.mean())
        rng = np.random.default_rng(0)
        # MÊME variance totale que le modèle (logit) : on ne compare que les centres.
        zd = np.log(d / (1 - d))
        base = np.log(p0 / (1 - p0)) + rng.normal(0, zd.std(), 4000)
        out.append({"election": election, "coupure": c, "as_of": str(as_of.date()),
                    "candidat": a, "resultat": round(y, 4), "n_notices_t1": n_t1,
                    "modele": round(float(d.mean()), 4),
                    "modele_ic90": [round(float(x), 4) for x in np.percentile(d, [5, 95])],
                    "couvre_ic90": bool(np.percentile(d, 5) <= y <= np.percentile(d, 95)),
                    "log_score": round(_log_score(np.log(d / (1 - d)), y), 3),
                    "sondages_7j": round(p0, 4), "n_sondages_7j": int(len(recents)),
                    "log_score_sondages": round(_log_score(base, y), 3),
                    "couvre_ic90_sondages": bool(np.percentile(1 / (1 + np.exp(-base)), 5) <= y
                                                 <= np.percentile(1 / (1 + np.exp(-base)), 95)),
                    **cible, **_diag(fit, dt)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--only", choices=["v1", "v2"], default=None)
    ap.add_argument("--election", type=int, choices=[2017, 2022], default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--paire", default=None, help="V1 : une seule paire, « Nom A|Nom B »")
    ap.add_argument("--coupure", default=None, choices=["T1-30", "T1-1", "T2-7"],
                    help="V2 : une seule coupure")
    ap.add_argument("--lister", action="store_true", help="liste les tâches unitaires et sort")
    ap.add_argument("--variante", choices=["abst", "eta", "geo", "geo1", "hier", "hier_maison", "hier_notice", "hier2d", "hier_rep", "hiercercle", "hiercercle_sr", "hiercercle_ra", "hier_ra", "hiercercle_ra_abs", "hier_ra_t", "hier_ra_lie", "hiercercle_ra_ferme", "hier_ra_fid", "hiercercle_ra_fid", "hier_ra_fid_omk", "hiercercle_ra_fid_omk"], default="abst",
                    help="abst = abstention + valence ; eta = valence seule ; geo = κ + abstention, "
                         "sans valence ; geo1 = abstention seule, κ = 1 (spec §8.9) ; hier = abst à échelle de "
                         "valence apprise (§8.10)")
    ap.add_argument("--chaine", type=int, default=None,
                    help="ne lance QUE la chaîne k (0-3, graine 27+k) et sauve ses tirages")
    ap.add_argument("--fusionner", action="store_true",
                    help="assemble les 4 chaînes sauvées (aucun NUTS) et écrit le JSON final")
    ap.add_argument("--sans-t2", action="store_true",
                    help="V2 : T1 seul (aucun duel), pour diagnostiquer la convergence")
    ap.add_argument("--init-depuis", default=None,
                    help="phase 2 (spec §8.8) : nom du JSON fusionné d'une phase 1 ; les chaînes "
                         "partent du dernier tirage de sa chaîne la plus dense")
    ap.add_argument("--ancres-famille", action="store_true",
                    help="ancres cardinales par famille, écart minimal intra-famille (piste B)")
    ap.add_argument("--lse-officiel", action="store_true",
                    help="valeur inclusive sur le champ officiel du T1 (spec §8.49)")
    ap.add_argument("--stacking", action="store_true",
                    help="fusion : mélange des chaînes par stacking au lieu de la sélection (spec §8.26)")
    ap.add_argument("--noeud-resultat", action="store_true",
                    help="V2 après le T1 : résultat T1 comme nœud quasi exact (spec §8.25)")
    ap.add_argument("--pos-scale", type=float, default=None,
                    help="diagnostic : liberté de position autour de l'ancre (défaut du modèle 0,4)")
    args = ap.parse_args()
    global ABSTENTION, SECOND_TOUR, POS_SCALE_BT, NOEUD_RESULTAT
    NOEUD_RESULTAT = args.noeud_resultat
    global STACKING
    STACKING = args.stacking
    if args.lse_officiel:
        VARIANTE_KW["_lse_officiel"] = True
    if args.ancres_famille:
        VARIANTE_KW["ancres_famille"] = True
    POS_SCALE_BT = args.pos_scale
    ABSTENTION = args.variante in ("abst", "geo", "geo1", "hier", "hier_maison", "hier_notice", "hier2d", "hier_rep", "hiercercle", "hiercercle_sr", "hiercercle_ra", "hier_ra", "hiercercle_ra_abs", "hier_ra_t", "hier_ra_lie", "hiercercle_ra_ferme", "hier_ra_fid", "hiercercle_ra_fid", "hier_ra_fid_omk", "hiercercle_ra_fid_omk")
    VARIANTE_KW.update({"geo": {"eta_scale": 0.0},
                        "geo1": {"eta_scale": 0.0, "kappa_log_scale": 1e-3},
                        "hier": {"eta_hier": True},
                        "hier_maison": {"eta_hier": True, "effets_maison": True},
                        "hier_notice": {"eta_hier": True, "effet_notice_seul": True},
                        # R2 (spec §8.44) : carte 2D + reports en vraisemblance (drapeau éteint ailleurs)
                        "hier2d": {"eta_hier": True, "carte_2d": True, "_reports": True},
                        # témoin équitable de R2 (revue 19/11, point 3) : reports en vraisemblance, 1D
                        "hier_rep": {"eta_hier": True, "_reports": True},
                        # R3 (spec §8.46) : votants sur un cercle, avec / sans reports
                        "hiercercle": {"eta_hier": True, "carte_cercle": True, "_reports": True},
                        "hiercercle_sr": {"eta_hier": True, "carte_cercle": True},
                        "hiercercle_ra": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True},
                        "hier_ra": {"eta_hier": True, "_reports_avant": True},
                        "hiercercle_ra_abs": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True,
                                              "abst_situee": True},
                        "hier_ra_t": {"eta_hier": True, "_reports_avant": True, "noyau_t": True},
                        "hier_ra_lie": {"eta_hier": True, "_reports_avant": True, "sigma_lie": True},
                        "hiercercle_ra_ferme": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True, "arc_ferme": True},
                        "hier_ra_fid": {"eta_hier": True, "_reports_avant": True, "fidelite": True},
                        "hiercercle_ra_fid": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True, "fidelite": True},
                        "hier_ra_fid_omk": {"eta_hier": True, "_reports_avant": True, "fidelite": True, "omega_electorat": True},
                        "hiercercle_ra_fid_omk": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True, "fidelite": True, "omega_electorat": True}}.get(args.variante, {}))
    SECOND_TOUR = not args.sans_t2
    if args.sans_t2 and args.only != "v2":
        ap.error("--sans-t2 n'a de sens qu'en V2 (V1 prédit un duel)")
    nuts = dict(draws=60, tune=60, chains=1) if args.fast else dict(draws=400, tune=600, chains=4)
    if args.lister:
        for e in (2017, 2022):
            tous = _duels_moyens(polls_historiques(e, T1_DATES[e] - pd.Timedelta(days=1)))
            for (a, b), n in tous.groupby(["a", "b"]).size().items():
                if n >= MIN_SONDAGES_PAIRE:
                    print(f"--only v1 --election {e} --paire '{a}|{b}'")
            for c in ("T1-30", "T1-1", "T2-7"):
                print(f"--only v2 --election {e} --coupure {c}")
        return
    OUT_DIR.mkdir(exist_ok=True)
    out = args.out or OUT_DIR / (f"second_round_{args.variante}_{'fast_' if args.fast else ''}"
                                 f"{args.only or 'all'}_{args.election or 'both'}"
                                 + (f"_{args.paire.replace('|', '-').replace(' ', '_')}" if args.paire else "")
                                 + (f"_{args.coupure}" if args.coupure else "")
                                 + ("_sans_t2" if args.sans_t2 else "")
                                 + (f"_pos{args.pos_scale:g}" if args.pos_scale else "")
                                 + ("_noeud" if args.noeud_resultat else "")
                                 + ("_famille" if args.ancres_famille else "")
                                 + ("_lseo" if args.lse_officiel else "")
                                 + ("_init" if args.init_depuis else "") + ".json")
    nuts_json = dict(nuts)
    if args.chaine is not None or args.fusionner:
        # Une tâche unitaire = UN fit : sinon les fichiers de chaîne se mélangeraient.
        if args.fast or not args.election or not (args.paire or args.coupure) or not args.only:
            ap.error("--chaine/--fusionner : une tâche unitaire (--only, --election, --paire ou --coupure)")
        base = CHAINES_DIR / out.stem
        if args.chaine is not None:
            init = _init_mode_dominant(args.init_depuis) if args.init_depuis else None
            nuts["mcmc_runner"] = _runner_chaine(base.with_name(f"{out.stem}_c{args.chaine}.npz"),
                                                 args.chaine, init)
            out = base.with_name(f"{out.stem}_c{args.chaine}.json")    # JSON 1 chaîne : brouillon
            nuts_json = {**nuts_json, "chains": 1, "chaine": args.chaine}
        else:
            nuts["mcmc_runner"] = _runner_fusion([base.with_name(f"{out.stem}_c{k}.npz")
                                                  for k in range(N_CHAINES)])
            nuts_json["mode"] = f"{N_CHAINES} processus d'une chaîne (graines 27..{26 + N_CHAINES}), fusionnés"
            if args.stacking:
                out = out.with_name(f"{out.stem}_stacking.json")      # chaînes : même nom
    rows = {"v1": [], "v2": []}
    for e in ([args.election] if args.election else [2017, 2022]):
        if args.only in (None, "v1"):
            rows["v1"] += v1_paire_retiree(e, nuts, tuple(args.paire.split("|")) if args.paire else None)
        if args.only in (None, "v2"):
            rows["v2"] += v2_resultat(e, nuts, (args.coupure,) if args.coupure else ("T1-30", "T1-1", "T2-7"))
    out.parent.mkdir(parents=True, exist_ok=True)
    if STACKING:
        nuts_json["poids_stacking"] = POIDS_STACKING
    out.write_text(json.dumps({"variante": args.variante, "nuts": nuts_json,
                               "pos_scale": args.pos_scale, "tau_t2_scrutin": TAU_T2_SCRUTIN, **rows},
                              ensure_ascii=False, indent=1) + "\n")
    for k, r in rows.items():
        if r:
            cols = [c for c in pd.DataFrame(r).columns if c != "eta"]
            print(f"\n== {k} ==\n" + pd.DataFrame(r)[cols].to_string(index=False))
    if rows["v1"]:
        t = pd.DataFrame(rows["v1"])
        for c in ("modele", "geometrie_seule", "bradley_terry"):
            print(f"V1 MAE {c}: {100 * (t[c] - t.observe_30j).abs().mean():.2f} pt")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
