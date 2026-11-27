"""Essai 2027, cercle contre segment, modèle unique (spec_second_tour §8.92).

    JAX_ENABLE_X64=1 PRES2027_B_QUAD=100 python -m model.backtest.essai_2027 --geo cercle \\
        --epreuve complet | "v2:Le Pen|Philippe" | "v1:Philippe – Le Pen"   [--inits 4] [--pas 4000]
    python -m model.backtest.essai_2027 --lister

MAP (Adam puis L-BFGS) ; rien n'est écrit hors de `model/backtest/results/essai2027_*.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro import handlers
from numpyro.infer import init_to_median
from numpyro.infer.util import initialize_model
from scipy.special import logsumexp

import model.backtest.eval_r2 as E
import model.backtest.second_round as sr
from model.backtest.polir import polir
from model.core.live_dataset import load_raw_polls
from model.models.spatial_pooling.fit import duel_draws, fit_spatial_pooling
from model.models.spatial_pooling.geometry import reports_cellules

AS_OF = "2026-11-25"
CACHES_V1 = ("Philippe – Le Pen", "Attal – Le Pen", "Glucksmann – Le Pen")


def court(nom: str) -> str:
    return "Le Pen" if "Le Pen" in nom else nom.split()[-1]


def donnees():
    raw = load_raw_polls(tour=None, as_of=AS_OF)
    raw = raw[raw.date_fin >= pd.Timestamp("2026-01-01")].copy()
    rep = pd.read_csv(Path(sr.PARSED_DIR) / "reports_avant_t1_2027.csv")
    rep["hypothese"] = rep["hypothese"].str.replace("—", "–")
    rep = rep[pd.to_datetime(rep.date_fin) <= pd.Timestamp(AS_OF)]
    t1 = raw[raw.tour.eq("Premier tour") & (raw.date_fin >= pd.Timestamp(AS_OF) - pd.Timedelta(days=60))]
    champs = t1.groupby(["notice", "hypothese"], dropna=False).candidat.agg(lambda x: tuple(sorted(x)))
    return raw, rep, list(champs.value_counts().index[0])


def duels(raw: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par duel : a, b (triés par nom court), part de a, date."""
    t2 = raw[raw.tour.eq("Deuxième tour")]
    out = []
    for _, g in t2.groupby(["notice", "hypothese"], dropna=False):
        if len(g) != 2:
            continue
        g = g.assign(c=g.candidat.map(court)).sort_values("c")
        out.append({"a": g.c.iloc[0], "b": g.c.iloc[1], "date": g.date_fin.iloc[0],
                    "y": float(g.intention.iloc[0] / g.intention.sum())})
    return pd.DataFrame(out)


def cible_v2(d: pd.DataFrame, a: str, b: str) -> tuple[float, int]:
    x = d[(d.a == a) & (d.b == b)].sort_values("date")
    r = x[x.date >= pd.Timestamp(AS_OF) - pd.Timedelta(days=60)]
    r = r if len(r) else x.tail(3)
    return float(100 * r.y.mean()), len(r)


def retirer(raw, rep, epreuve: str):
    if epreuve.startswith("v1:"):
        return raw, rep[rep.hypothese != epreuve[3:]]
    if epreuve.startswith("v2:"):
        paire = set(epreuve[3:].split("|"))
        t2 = raw.tour.eq("Deuxième tour")
        cle = raw.notice.astype(str) + "|" + raw.hypothese.fillna("__u__").astype(str)
        p = raw[t2].groupby(cle[t2]).candidat.agg(lambda x: {court(c) for c in x})
        raw = raw[~(t2 & cle.isin(p[p.map(lambda s: s == paire)].index))]
        rep = rep[rep.hypothese.map(lambda h: {x.strip() for x in h.split("–")} != paire)]
    return raw, rep


def appel(raw, rep, champ, geo, runner):
    return fit_spatial_pooling(raw, AS_OF, second_tour=True, eta_hier=True, fidelite=True,
                               sigma_dynamiques_30j=True, carte_cercle=geo in ("cercle", "cercle2pi"),
                               arc_ferme=(geo == "cercle2pi"),                  # §8.113 e5 : arc fixé a priori à 2π
                               lse_officiel=champ, reports=rep, champ_reports=champ, mcmc_runner=runner)


class _Stop(Exception):
    pass


def cellules(fit, champ, ia, ib, k, w=None, kappa_rho=None, omega_rho=None, fidelite=True):
    """`w` (S, N) : attractivités à une date donnée (revue du 26/11) ; défaut : as_of.
    `ia`, `ib`, `k` : entiers, ou tableaux (S,) — une ligne par tirage (§8.100 e2, lecture vectorisée).
    `kappa_rho`, `omega_rho` (§8.100 e2) : pente et ω de l'expression ρ des cellules seulement."""
    A, poids = E._A(fit, fit.w_draws if w is None else w)
    S = A.shape[0]
    ia, ib, k = (np.broadcast_to(np.asarray(x), (S,)) for x in (ia, ib, k))
    rows = np.arange(S)
    ml = None if fit.masque_lse is None else np.tile(np.asarray(fit.masque_lse, dtype=float), (S, 1))
    if ml is not None:                                  # revue du 26/11 : finalistes et électorat dans le lse
        ml[rows, ia] = ml[rows, ib] = ml[rows, k] = 1.0
    lse = logsumexp(A if ml is None else A + np.log(ml)[:, :, None], axis=1)
    base = np.array([float(c in champ) for c in fit.candidates])
    mask = np.tile(base, (S, 1))
    mask[rows, ia] = mask[rows, ib] = mask[rows, k] = 1.0
    if w is None and np.all(ia == ia[0]) and np.all(ib == ib[0]):
        s = duel_draws(fit, fit.candidates[ia[0]], fit.candidates[ib[0]], 0.0, scrutin=False)
    else:                                               # revue du 26/11 ~14:40Z (§8.98 c) : part du duel À LA MÊME DATE
        from model.models.spatial_pooling.fit import _duel_par_tirage
        s = _duel_par_tirage(fit, fit.w_draws if w is None else w, ia, ib)
    om = fit.omega_draws if omega_rho is None else np.full(S, omega_rho)
    kr = None if kappa_rho is None else jnp.full(S, kappa_rho)
    th, _ = reports_cellules(jnp.asarray(A), jnp.asarray(mask), jnp.asarray(ia), jnp.asarray(ib), jnp.asarray(k),
                             jnp.asarray(fit.kappa_draws), jnp.asarray(fit.eta_draws[rows, ia] - fit.eta_draws[rows, ib]),
                             jnp.asarray(lse), jnp.asarray(om), jnp.asarray(s), poids, n_newton=8,
                             fidelite=fidelite, kappa_rho=kr)              # §8.105 e3 : R7 en lecture
    lignes = not (np.all(ia == ia[0]) and np.all(ib == ib[0]) and np.all(k == k[0]))
    if np.ndim(th) == 2 and (lignes or (w is not None and np.ndim(w) == 2 and not np.allclose(w, w[0]))):
        return 100 * np.asarray(th)                     # une cellule par ligne (§8.105 a : aussi si w varie)
    return 100 * np.asarray(th).mean(0)


def lire(fit, champ, rep_tout, d) -> dict:
    idx = {court(c): i for i, c in enumerate(fit.candidates)}
    out = {"duels": {}, "cellules": []}
    for a, b in {(r.a, r.b) for r in d.itertuples()}:
        if a in idx and b in idx:
            out["duels"][f"{a}|{b}"] = float(100 * duel_draws(fit, fit.candidates[idx[a]], fit.candidates[idx[b]],
                                                              0.0, scrutin=False).mean())
    # revue du 26/11 : la même part évaluée AUX DATES DE LA CIBLE V2 (chemin de w), et non à as_of
    from model.models.spatial_pooling.fit import _duel_par_tirage
    out["duels_date"] = {}
    if fit.w_path_draws is not None:
        for a, b in {(r.a, r.b) for r in d.itertuples()}:
            if a in idx and b in idx:
                x = d[(d.a == a) & (d.b == b)].sort_values("date")
                r = x[x.date >= pd.Timestamp(AS_OF) - pd.Timedelta(days=60)]
                r = r if len(r) else x.tail(3)
                v = []
                for t in r.date:
                    j = int(np.clip(np.searchsorted(fit.w_dates, float(t.toordinal())), 0, len(fit.w_dates) - 1))
                    S = fit.w_path_draws.shape[0]
                    v.append(100 * float(_duel_par_tirage(fit, fit.w_path_draws[:, :, j], np.full(S, idx[a]),
                                                          np.full(S, idx[b])).mean()))
                out["duels_date"][f"{a}|{b}"] = float(np.mean(v))
    for (h, e), _ in rep_tout.groupby(["hypothese", "electorat_t1"]):
        a, b = [x.strip() for x in h.split("–")]
        if {a, b, court(e)} <= set(idx):
            th = cellules(fit, champ, idx[a], idx[b], idx[court(e)])
            c = {"hypothese": h, "electorat": e, "pred": [round(float(x), 2) for x in th]}
            if fit.w_path_draws is not None:            # revue du 26/11 : à la date de CHAQUE notice
                v = []
                g = rep_tout[(rep_tout.hypothese == h) & (rep_tout.electorat_t1 == e)]   # §8.100 c : notices de CET électorat
                t_ = g.pivot_table(index="notice", columns="candidat_t2", values="part")
                ok = t_[[a, b, "Aucun"]].dropna().index if {a, b, "Aucun"} <= set(t_.columns) else []
                dates = g.drop_duplicates("notice").set_index("notice").loc[list(ok), "date_fin"]
                for t in pd.to_datetime(dates):                # une date par notice complète, comme `mesure()`
                    j = int(np.clip(np.searchsorted(fit.w_dates, float(t.toordinal())), 0, len(fit.w_dates) - 1))
                    v.append(cellules(fit, champ, idx[a], idx[b], idx[court(e)], w=fit.w_path_draws[:, :, j]))
                c["pred_date"] = [round(float(x), 2) for x in np.mean(v, axis=0)]
            out["cellules"].append(c)
    # concentration R de l'électorat T1 (§8.90 b), cercle seulement
    if fit.arc_draws is not None:
        from model.models.spatial_pooling.geometry import THETA
        A, poids = E._A(fit, fit.w_draws)
        m = np.array([c in champ for c in fit.candidates])
        p = np.exp(A[0][m] - logsumexp(A[0][m], axis=0)) * np.asarray(poids)
        z = (p * np.exp(1j * np.asarray(THETA))).sum(1) / p.sum(1)
        out["R"] = {c: round(float(abs(v)), 3) for c, v in zip(np.array(fit.candidates)[m], z)}
    return out


def lancer(geo: str, epreuve: str, inits: int, pas: int, lr: float = 0.01) -> dict:
    raw0, rep0, champ = donnees()
    raw, rep = retirer(raw0, rep0, epreuve)
    cap = {}

    def capture(fn, kw, **k):
        cap.update(fn=fn, kw=kw)
        raise _Stop
    try:
        appel(raw, rep, champ, geo, capture)
    except _Stop:
        pass
    fn, kw = cap["fn"], cap["kw"]
    d = duels(raw0)
    res = {"spec": "§8.92", "geo": geo, "epreuve": epreuve, "champ": champ, "n_reports_fit": int(len(kw["rep_y"])),
           "n_duels_fit": int(len(kw["duel_y"])), "x64": bool(jax.config.jax_enable_x64), "inits": []}
    for s in range(inits):
        t = time.time()
        info = initialize_model(jax.random.PRNGKey(s), fn, model_kwargs=kw, init_strategy=init_to_median)
        z = dict(info.param_info.z)
        rng = np.random.default_rng(s)
        if s:
            z = {k: v + 0.5 * jnp.asarray(rng.standard_normal(np.shape(v)), dtype=jnp.asarray(v).dtype)
                 for k, v in z.items()}
        vg = jax.jit(jax.value_and_grad(info.potential_fn))

        @jax.jit
        def pas_adam(i, etat):
            z, m, v = etat
            g = jax.grad(info.potential_fn)(z)
            m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
            v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
            l = lr * 0.05 ** (i / pas)
            z = jax.tree_util.tree_map(lambda x, a, b: x - l * (a / (1 - 0.9 ** (i + 1.0))) /
                                       (jnp.sqrt(b / (1 - 0.999 ** (i + 1.0))) + 1e-8), z, m, v)
            return z, m, v
        zero = jax.tree_util.tree_map(jnp.zeros_like, z)
        z, _, _ = jax.lax.fori_loop(0, pas, pas_adam, (z, zero, zero))
        U0 = float(vg(z)[0])
        z, U, gn, nit, msg = polir(info.potential_fn, z, maxiter=3000)
        tr = handlers.trace(handlers.substitute(handlers.seed(fn, 0), data=info.postprocess_fn(z))).get_trace(**kw)
        vals = {n: np.asarray(x["value"]) for n, x in tr.items()
                if x["type"] in ("sample", "deterministic") and not x.get("is_observed", False)}

        def rejoue(f, k2, **k):
            S = 4
            return ({n: np.repeat(v[None, None], S, axis=1) for n, v in vals.items()},
                    {"diverging": np.zeros((1, S), bool), "num_steps": np.ones((1, S), int)})
        fit = appel(raw, rep, champ, geo, rejoue)
        r = {"init": s, "U_adam": round(U0, 2), "U": round(U, 3), "gnorm": gn, "it": nit, "arret": msg,
             "secondes": round(time.time() - t, 1), "arc": float(vals["arc"]) if "arc" in vals else None,
             "candidats": list(fit.candidates), "mu": vals["mu"].tolist(), "sigma": vals["sigma"].tolist(),
             "z_pos": vals["z_pos"].tolist(), "z_sigma": vals["z_sigma"].tolist(), "eta": vals["eta"].tolist(),
             "kappa": float(vals["kappa"]), "omega": float(vals["omega"]), **lire(fit, champ, rep0, d),
             # revue du 26/11 : traçabilité (prédiction de chaque duel À SA DATE, z, quadrature)
             "duel_s": np.asarray(vals["duel_s"]).tolist(), "duel_y": np.asarray(kw["duel_y"]).tolist(),
             "duel_a": np.asarray(kw["duel_a"]).tolist(), "duel_b": np.asarray(kw["duel_b"]).tolist(),
             "z": {n: np.asarray(v).tolist() for n, v in vals.items() if n.startswith(("z_", "s_", "sigma_", "a_arc", "log_"))},
             "B_quad": int(__import__("os").environ.get("PRES2027_B_QUAD", "25"))}
        res["inits"].append(r)
        print(json.dumps({k: r[k] for k in ("init", "U_adam", "U", "gnorm", "it", "secondes", "arc")}), flush=True)
    nom = "".join(ch for ch in epreuve if ch.isalnum())
    (sr.OUT_DIR / f"essai2027_{geo}_{nom}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    return res


def rejouer(geo: str, epreuve: str) -> dict:
    """§8.98 c : relit les cellules d'un fit sauvegardé SANS refit (valeurs des sites rejouées), et réécrit le JSON."""
    nom = "".join(ch for ch in epreuve if ch.isalnum())
    f = sr.OUT_DIR / f"essai2027_{geo}_{nom}.json"
    res = json.loads(f.read_text())
    raw0, rep0, champ = donnees()
    raw, rep = retirer(raw0, rep0, epreuve)
    cap = {}

    def capture(fn, kw, **k):
        cap.update(fn=fn, kw=kw)
        raise _Stop
    try:
        appel(raw, rep, champ, geo, capture)
    except _Stop:
        pass
    fn, kw = cap["fn"], cap["kw"]
    d = duels(raw0)
    for r in res["inits"]:
        data = {n: jnp.asarray(v) for n, v in r["z"].items()}
        data.update(kappa=jnp.asarray(r["kappa"]), omega=jnp.asarray(r["omega"]))
        tr = handlers.trace(handlers.substitute(handlers.seed(fn, 0), data=data)).get_trace(**kw)
        manque = [n for n, x in tr.items() if x["type"] == "sample" and not x.get("is_observed", False) and n not in data]
        assert not manque, manque
        vals = {n: np.asarray(x["value"]) for n, x in tr.items()
                if x["type"] in ("sample", "deterministic") and not x.get("is_observed", False)}
        ecart = float(np.abs(np.asarray(vals["duel_s"]) - np.asarray(r["duel_s"])).max())
        assert ecart < 1e-6, ecart                       # le rejeu reproduit le fit sauvegardé

        def rejoue(f, k2, **k):
            S = 4
            return ({n: np.repeat(v[None, None], S, axis=1) for n, v in vals.items()},
                    {"diverging": np.zeros((1, S), bool), "num_steps": np.ones((1, S), int)})
        fit = appel(raw, rep, champ, geo, rejoue)
        L = lire(fit, champ, rep0, d)
        r.setdefault("cellules_s_asof", r["cellules"])   # lecture de §8.97, gardée pour comparaison
        r["cellules"] = L["cellules"]
        r["rejeu"] = {"spec": "§8.98 c, §8.100 c", "ecart_duel_s": ecart}
        print(json.dumps({"init": r["init"], "ecart": ecart}), flush=True)
    f.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    return res


def epreuves() -> list[str]:
    raw, _, _ = donnees()
    n = duels(raw).groupby(["a", "b"]).size()
    return ["complet"] + [f"v2:{a}|{b}" for (a, b), k in n.items() if k >= 3] + [f"v1:{h}" for h in CACHES_V1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geo", choices=["cercle", "segment", "cercle2pi"])
    ap.add_argument("--epreuve")
    ap.add_argument("--inits", type=int, default=4)
    ap.add_argument("--pas", type=int, default=4000)
    ap.add_argument("--lister", action="store_true")
    ap.add_argument("--rejouer", action="store_true", help="§8.98 c : relire un fit sauvegardé, sans refit")
    a = ap.parse_args()
    if a.lister:
        print("\n".join(epreuves()))
    elif a.rejouer:
        rejouer(a.geo, a.epreuve)
    else:
        lancer(a.geo, a.epreuve, a.inits, a.pas)
