"""Test β sans refit (spec_second_tour §8.100 e2) sur les fits V1 de l'essai 2027.

Pour chaque fit V1 sauvegardé (meilleur U), les valeurs des sites sont rejouées. Seule l'expression ρ des
cellules de reports change : pente β au lieu de κ ; `q`, `s` et le Newton gardent les valeurs ajustées.
ω est re-profilé (1-D) pour chaque β sur la seule vraisemblance Dirichlet des reports D'ENTRAÎNEMENT
(`log_c` fixé). β* est choisi sur la grille par cette vraisemblance, jamais sur les cellules cachées.

Usage : JAX_ENABLE_X64=1 PRES2027_B_QUAD=100 python -m model.backtest.beta_essai_2027
"""
from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro import distributions as dist
from numpyro import handlers
from scipy.optimize import minimize_scalar

from model.backtest import essai_2027 as X
from model.backtest import second_round as sr
from model.backtest.lire_essai_2027 import mesure

GRILLE_BETA = (0.0, 0.4, 0.56, 0.75, None)       # None = κ du fit (référence)


def reconstruire(geo, epreuve, r, S):
    """Fit rejoué à S lignes identiques, avec kw du modèle (reports d'entraînement)."""
    raw0, rep0, champ = X.donnees()
    raw, rep = X.retirer(raw0, rep0, epreuve)
    cap = {}

    def capture(fn, kw, **k):
        cap.update(fn=fn, kw=kw)
        raise X._Stop
    try:
        X.appel(raw, rep, champ, geo, capture)
    except X._Stop:
        pass
    fn, kw = cap["fn"], cap["kw"]
    data = {n: jnp.asarray(v) for n, v in r["z"].items()}
    data.update(kappa=jnp.asarray(r["kappa"]), omega=jnp.asarray(r["omega"]))
    tr = handlers.trace(handlers.substitute(handlers.seed(fn, 0), data=data)).get_trace(**kw)
    vals = {n: np.asarray(x["value"]) for n, x in tr.items()
            if x["type"] in ("sample", "deterministic") and not x.get("is_observed", False)}
    assert float(np.abs(vals["duel_s"] - np.asarray(r["duel_s"])).max()) < 1e-9

    def rejoue(f, k2, **k):
        return ({n: np.repeat(v[None, None], S, axis=1) for n, v in vals.items()},
                {"diverging": np.zeros((1, S), bool), "num_steps": np.ones((1, S), int)})
    fit = X.appel(raw, rep, champ, geo, rejoue)
    return fit, kw, vals, champ, rep0


def theta_entrainement(fit, kw, champ, beta, omega):
    R = len(kw["rep_y"])
    j = np.asarray(kw["rep_date_idx"])
    w = fit.w_path_draws[np.arange(R), :, j]
    return X.cellules(fit, champ, np.asarray(kw["rep_a"]), np.asarray(kw["rep_b"]), np.asarray(kw["rep_k"]),
                      w=w, kappa_rho=beta, omega_rho=omega) / 100


def ll(theta, y, c):
    return float(dist.Dirichlet(c * jnp.clip(jnp.asarray(theta), 1e-4, None)).log_prob(jnp.asarray(y)).sum())


def cellules_cachees(geo, epreuve, r, champ, rep0, beta, omega):
    """Cellules cachées à la date de chaque notice complète de l'électorat (comme `pred_date`, §8.100 c)."""
    h = epreuve[3:]
    a, b = [x.strip() for x in h.split("–")]
    lignes = []
    for e, g in rep0[rep0.hypothese == h].groupby("electorat_t1"):
        t_ = g.pivot_table(index="notice", columns="candidat_t2", values="part")
        ok = t_[[a, b, "Aucun"]].dropna().index if {a, b, "Aucun"} <= set(t_.columns) else []
        for nt in ok:
            lignes.append((X.court(e), pd.Timestamp(g[g.notice == nt].date_fin.iloc[0])))
    fit, _, _, _, _ = reconstruire(geo, epreuve, r, len(lignes))
    idx = {X.court(c): i for i, c in enumerate(fit.candidates)}
    garde = [i for i, (e, _) in enumerate(lignes) if e in idx]
    lignes = [lignes[i] for i in garde]
    S = fit.w_path_draws.shape[0]
    jj = [int(np.clip(np.searchsorted(fit.w_dates, float(t.toordinal())), 0, len(fit.w_dates) - 1)) for _, t in lignes]
    n = len(lignes)
    w = fit.w_path_draws[np.arange(S), :, np.array(jj + [jj[-1]] * (S - n))]
    k = np.array([idx[e] for e, _ in lignes] + [idx[lignes[-1][0]]] * (S - n))
    th = X.cellules(fit, champ, np.full(S, idx[a]), np.full(S, idx[b]), k, w=w, kappa_rho=beta, omega_rho=omega)[:n]
    out = {}
    for (e, _), v in zip(lignes, th):
        out.setdefault(e, []).append(v)
    return {e: np.mean(v, axis=0) for e, v in out.items()}


def main() -> dict:
    res = {"spec": "§8.100 e2", "grille_beta": [b if b is not None else "kappa" for b in GRILLE_BETA], "fits": {}}
    for geo in ("segment", "cercle"):
        for h in X.CACHES_V1:
            ep = f"v1:{h}"
            nom = "".join(ch for ch in ep if ch.isalnum())
            J = json.loads((sr.OUT_DIR / f"essai2027_{geo}_{nom}.json").read_text())
            r = min(J["inits"], key=lambda i: i["U"])
            R = None
            fit0, kw, vals, champ, rep0 = reconstruire(geo, ep, r, 4)
            R = len(kw["rep_y"])
            fit, kw, vals, champ, rep0 = reconstruire(geo, ep, r, R)
            y, c = np.asarray(kw["rep_y"]), float(np.exp(vals["log_c"]))
            th0 = theta_entrainement(fit, kw, champ, None, None)
            ecart = float(np.abs(th0 - np.asarray(vals["reports_theta"])).max())
            assert ecart < 1e-8, ecart                       # non-régression : β = κ, ω du fit
            prof = {}
            for beta in GRILLE_BETA:
                f = lambda om: -ll(theta_entrainement(fit, kw, champ, beta, om), y, c)  # noqa: E731
                o = minimize_scalar(f, bounds=(r["omega"] - 6, r["omega"] + 6), method="bounded",
                                    options={"xatol": 1e-3})
                prof["kappa" if beta is None else beta] = {"omega": float(o.x), "ll": float(-o.fun)}
            best = max(prof, key=lambda b: prof[b]["ll"])
            bb = None if best == "kappa" else best
            cach = cellules_cachees(geo, ep, r, champ, rep0, bb, prof[best]["omega"])
            res["fits"][f"{geo}|{h}"] = {"R_entrainement": R, "ecart_non_regression": ecart, "kappa": r["kappa"],
                                         "omega_fit": r["omega"], "ll_fit": ll(th0, y, c), "profil": prof,
                                         "beta_etoile": best,
                                         "cellules": {e: [round(float(x), 2) for x in v] for e, v in cach.items()}}
            print(json.dumps({"fit": f"{geo}|{h}", "beta*": best, "profil": prof}), flush=True)
    # lecture, critères de §8.100 e2
    _, rep, _ = X.donnees()
    S = {}
    for geo in ("segment", "cercle"):
        err, lfi, expr = [], [], []
        err8 = []
        for h in X.CACHES_V1:
            cible = mesure(rep, h)
            fin = {x.strip() for x in h.split("–")}
            pr = res["fits"][f"{geo}|{h}"]["cellules"]
            for e, m in cible.items():
                if e in fin or e not in pr:
                    continue
                p = np.array(pr[e])
                err.append(np.abs(p - m).mean())
                if not (h.startswith("Glucksmann") and e == "Attal"):
                    err8.append(np.abs(p - m).mean())
                if e == "Mélenchon":
                    lfi.append(np.abs(p[1:] - m[1:]).mean())
                    expr.append(abs(100 * p[1] / (p[0] + p[1]) - 100 * m[1] / (m[0] + m[1])))
        S[geo] = {"V1": round(float(np.mean(err)), 2), "V1_8": round(float(np.mean(err8)), 2),
                  "LFI_RN": round(float(np.mean(lfi)), 2), "LFI_exprimes": round(float(np.mean(expr)), 2)}
    lec = json.loads((sr.OUT_DIR / "essai2027_lecture.json").read_text())["synthese"]
    for geo in ("segment", "cercle"):
        ref = lec[f"LFI_RN_exprimes_{geo}_date"]
        S[geo]["critere"] = bool(S[geo]["LFI_RN"] < 6 and S[geo]["LFI_exprimes"] <= ref + 1
                                 and S[geo]["V1_8"] <= lec["V1_sans_pollue_K_s"][0])
        S[geo]["reference_exprimes"] = ref
    S["K_s_8"] = lec["V1_sans_pollue_K_s"][0]
    res["synthese"] = S
    (sr.OUT_DIR / "essai2027_beta.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(S, ensure_ascii=False, indent=1))
    return res


if __name__ == "__main__":
    main()
