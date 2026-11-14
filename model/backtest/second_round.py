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
N_CHAINES = 4


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
        LP_CHAINES[:] = _log_densite_par_chaine(model_fn, model_kwargs, samples)
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


# Log-densité moyenne par chaîne de la dernière fusion (spec §8.8).
LP_CHAINES: list[float] = []
DELTA_LP = 10.0          # nats : seuil de rétention des chaînes (spec §8.8)
N_TIRAGES_LP = 20        # tirages amincis par chaîne pour estimer L̄_c


def _log_densite_par_chaine(model_fn, model_kwargs, samples) -> list[float]:
    """L̄_c = moyenne de log p(θ, y) sur `N_TIRAGES_LP` tirages de chaque chaîne."""
    import jax
    import jax.numpy as jnp
    from numpyro import handlers
    from numpyro.infer.util import log_density
    tr = handlers.trace(handlers.seed(model_fn, 0)).get_trace(**model_kwargs)
    sites = [n for n, v in tr.items() if v["type"] == "sample" and not v["is_observed"]]
    f = jax.jit(lambda prm: log_density(model_fn, (), model_kwargs, prm)[0])
    C, D = samples[sites[0]].shape[:2]
    out = []
    for c in range(C):
        idx = np.linspace(0, D - 1, min(N_TIRAGES_LP, D)).astype(int)
        out.append(float(np.mean([float(f({n: jnp.asarray(samples[n][c, t]) for n in sites}))
                                  for t in idx])))
    return out


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


def _duels_moyens(raw: pd.DataFrame, depuis: pd.Timestamp | None = None) -> pd.DataFrame:
    """Une ligne par duel sondé : paire triée, part du premier nom, date."""
    t2 = raw[raw.tour.eq("Deuxième tour")].copy()
    if depuis is not None:
        t2 = t2[t2.date_fin.ge(depuis)]
    rows = []
    for (_, _), g in t2.groupby(["notice", t2["hypothese"].fillna("__u__")]):
        g = g.sort_values("candidat")
        v = g.intention.to_numpy(float)
        rows.append((g.candidat.iloc[0], g.candidat.iloc[1], v[0] / v.sum(), g.date_fin.iloc[0]))
    return pd.DataFrame(rows, columns=["a", "b", "y", "date"])


def _sans_paire(raw: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    t2 = raw.tour.eq("Deuxième tour")
    key = raw["notice"].astype(str) + "|" + raw["hypothese"].fillna("__u__").astype(str)
    paires = raw[t2].groupby(key[t2])["candidat"].agg(lambda x: frozenset(x))
    drop = set(paires[paires.eq(frozenset((a, b)))].index)
    return raw[~(t2 & key.isin(drop))]


def _geometrie_seule(fit, a: str, b: str) -> np.ndarray:
    i = {c: k for k, c in enumerate(fit.candidates)}
    A = np.asarray(log_attractivite(np.asarray(fit.mu_draws), np.asarray(fit.sigma_draws),
                                    np.asarray(fit.w_draws)))
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
SECOND_TOUR = True  # False = --sans-t2 : diagnostic de convergence du T1 seul
POS_SCALE_BT = None  # --pos-scale : EXPÉRIENCE de diagnostic (spec §8.7), pas un réglage


def _fit(raw, as_of, election, nuts):
    t = time.time()
    fit = fit_spatial_pooling(raw, str(as_of.date()), min_poll_date="1900-01-01",
                              order_groups=ORDER_GROUPS_HISTORIQUES[election],
                              couverture_min=None, abstention=ABSTENTION,
                              second_tour=SECOND_TOUR,
                              **({"pos_scale": POS_SCALE_BT} if POS_SCALE_BT else {}), **nuts)
    return fit, round(time.time() - t, 1)


def _diag(fit, dt):
    d = fit.diagnostics
    return {"rhat_max": d["rhat_max"], "ess_min": d["ess_min"], "divergences": d["n_divergences"],
            "kappa": d.get("kappa"), "omega": d.get("omega"), "rhat_t2": d.get("rhat_t2"), "sigma_t2": d.get("sigma_t2"), "secondes": dt,
            "eta": {c: round(float(x), 3) for c, x in zip(fit.candidates, fit.eta_draws.mean(0))}
            if fit.eta_draws is not None else None}


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
        geo = _geometrie_seule(fit, a, b)
        bt = _bradley_terry(tous[~((tous.a == a) & (tous.b == b))], a, b)
        autres = tous[((tous.a.isin([a, b])) | (tous.b.isin([a, b]))) & ~((tous.a == a) & (tous.b == b))]
        out.append({"election": election, "paire": [a, b], "n_sondages_retires": int(n),
                    "observe_30j": round(obs, 4), "modele": round(float(mod.mean()), 4),
                    "modele_ic90": [round(float(x), 4) for x in np.percentile(mod, [5, 95])],
                    "geometrie_seule": round(float(geo.mean()), 4), "bradley_terry": round(bt, 4),
                    "autres_duels_des_deux": int(len(autres)), **cible, **_diag(fit, dt)})
    return out


def _cible_par_chaine(d: np.ndarray, n_chaines: int) -> dict:
    """Convergence DE LA QUANTITÉ PRÉDITE (spec §8.4) : R-hat et écart entre
    moyennes de chaînes du duel cible. Les tirages sont rangés chaîne par chaîne
    (`reshape(-1, ...)` de (chaînes, tirages, ...))."""
    if n_chaines < 2 or len(d) % n_chaines:
        return {}
    from numpyro.diagnostics import split_gelman_rubin
    x = np.log(d / (1 - d)).reshape(n_chaines, -1)
    m = (1 / (1 + np.exp(-x))).mean(1)
    out = {"modele_par_chaine": [round(float(v), 4) for v in m],
           "ecart_chaines_pt": round(100 * float(m.max() - m.min()), 2),
           "rhat_cible": round(float(split_gelman_rubin(x[..., None])[0]), 3)}
    if len(LP_CHAINES) == n_chaines:
        lp = np.array(LP_CHAINES)
        keep = np.flatnonzero(lp >= lp.max() - DELTA_LP)
        xs = x[keep]
        dr = (1 / (1 + np.exp(-xs))).ravel()
        out.update({"lp_par_chaine": [round(float(v), 1) for v in lp],
                    "chaines_retenues": keep.tolist(),
                    "modele_retenu": round(float(dr.mean()), 4),
                    "modele_ic90_retenu": [round(float(v), 4) for v in np.percentile(dr, [5, 95])],
                    "ecart_chaines_retenues_pt": round(100 * float(m[keep].max() - m[keep].min()), 2)})
        if len(keep) >= 2:
            out["rhat_cible_retenu"] = round(float(split_gelman_rubin(xs[..., None])[0]), 3)
        out["concluant"] = bool(len(keep) >= 2 and out["rhat_cible_retenu"] <= 1.1
                                and out["ecart_chaines_retenues_pt"] <= 1.0)
    return out


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
    ap.add_argument("--variante", choices=["abst", "eta"], default="abst",
                    help="abst = abstention (valeur inclusive) + valence ; eta = valence seule")
    ap.add_argument("--chaine", type=int, default=None,
                    help="ne lance QUE la chaîne k (0-3, graine 27+k) et sauve ses tirages")
    ap.add_argument("--fusionner", action="store_true",
                    help="assemble les 4 chaînes sauvées (aucun NUTS) et écrit le JSON final")
    ap.add_argument("--sans-t2", action="store_true",
                    help="V2 : T1 seul (aucun duel), pour diagnostiquer la convergence")
    ap.add_argument("--init-depuis", default=None,
                    help="phase 2 (spec §8.8) : nom du JSON fusionné d'une phase 1 ; les chaînes "
                         "partent du dernier tirage de sa chaîne la plus dense")
    ap.add_argument("--pos-scale", type=float, default=None,
                    help="diagnostic : liberté de position autour de l'ancre (défaut du modèle 0,4)")
    args = ap.parse_args()
    global ABSTENTION, SECOND_TOUR, POS_SCALE_BT
    POS_SCALE_BT = args.pos_scale
    ABSTENTION = args.variante == "abst"
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
    rows = {"v1": [], "v2": []}
    for e in ([args.election] if args.election else [2017, 2022]):
        if args.only in (None, "v1"):
            rows["v1"] += v1_paire_retiree(e, nuts, tuple(args.paire.split("|")) if args.paire else None)
        if args.only in (None, "v2"):
            rows["v2"] += v2_resultat(e, nuts, (args.coupure,) if args.coupure else ("T1-30", "T1-1", "T2-7"))
    out.parent.mkdir(parents=True, exist_ok=True)
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
