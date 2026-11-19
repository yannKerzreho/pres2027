"""(D-b) Test d'existence sans fit des votants en 2D (spec_second_tour §8.41).

Sur 40 tirages des fits V2 T2−7 (§8.38), ajoute une coordonnée u avec un
noyau séparable N(u; ν_i, λ), re-résout w pour garder les parts T1 du champ
réel, puis cherche les ν (9 libres, communs à 2017 et 2022) qui collent aux
tableaux de reports.

    python -m model.backtest.reports_2d [--variantes hier abst] [--lambdas 0.1 0.2 0.3]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

import model.backtest.second_round as sr
from model.backtest.reports_t2 import OUT, _fit_v2, _norm, mesures, reports_predits
from model.models.spatial_pooling.fit import duel_draws
from model.models.spatial_pooling.geometry import V, W

BU = 7
_x, _w = np.polynomial.legendre.leggauss(BU)
U = 0.01 + 0.98 * (_x + 1) / 2
WU = _w / 2
LIBRES = ["Emmanuel Macron", "Marine Le Pen", "Jean-Luc Mélenchon", "Benoît Hamon", "François Fillon",
          "Nicolas Dupont-Aignan", "Yannick Jadot", "Valérie Pécresse", "Éric Zemmour"]
N_TIRAGES = 40


def attractivite_2d(mu, sigma, w, nu, lam):
    """A (S, N, Bv·Bu) et poids (Bv·Bu,) : noyau séparable normalisé sur la grille."""
    Dv = -((np.asarray(V)[None, None, :] - mu[..., None]) ** 2) / (2 * sigma[..., None] ** 2)   # S,N,Bv
    Du = -((U[None, :] - nu[:, None]) ** 2) / (2 * lam ** 2)                                     # N,Bu
    D = Dv[..., :, None] + Du[None, :, None, :]                                                  # S,N,Bv,Bu
    poids = (np.asarray(W)[:, None] * WU[None, :]).ravel()
    D = D.reshape(*D.shape[:2], -1)
    return w[..., None] + D - logsumexp(D, axis=-1, b=poids, keepdims=True), poids


def parts_t1(A, champ, poids):
    Ac = A[:, champ, :]
    return np.einsum("skb,b->sk", np.exp(Ac - logsumexp(Ac, axis=1, keepdims=True)), poids)


def resoudre_w(mu, sigma, w, nu, lam, champ, cible, n_iter=40):
    """Point fixe w <- w + log(cible/π) sur le champ réel (§8.41)."""
    w = w.copy()
    for _ in range(n_iter):
        A, poids = attractivite_2d(mu, sigma, w, nu, lam)
        pi = parts_t1(A, champ, poids)
        w[:, champ] += np.log(cible / np.clip(pi, 1e-12, None))
    A, poids = attractivite_2d(mu, sigma, w, nu, lam)
    return A, poids, float(np.abs(parts_t1(A, champ, poids) - cible).max())


class Annee:
    """Tout ce qui est fixe pour une élection et une variante."""

    def __init__(self, election, variante):
        fit = _fit_v2(election, variante)
        rng = np.random.default_rng(0)
        idx_s = rng.choice(len(fit.mu_draws), N_TIRAGES, replace=False)
        self.election = election
        self.cands = list(fit.candidates)
        self.idx = {_norm(c): i for i, c in enumerate(self.cands)}
        res = sr.load_results()
        t1 = res[(res.election == election) & (res.tour == "Premier tour")].candidat
        self.champ = np.array([self.idx[_norm(c)] for c in t1 if _norm(c) in self.idx])
        self.mu, self.sigma = fit.mu_draws[idx_s], fit.sigma_draws[idx_s]
        self.w = fit.w_draws[idx_s]
        self.kappa, self.omega = fit.kappa_draws[idx_s], fit.omega_draws[idx_s]
        a, b = sr.FINALE[election]
        self.ia, self.ib = self.idx[_norm(a)], self.idx[_norm(b)]
        self.s_ab = duel_draws(fit, a, b, 0.0, scrutin=False)[idx_s]
        from model.models.spatial_pooling.geometry import log_attractivite
        A1 = np.asarray(log_attractivite(self.mu, self.sigma, self.w))
        self.cible = parts_t1(A1, self.champ, np.asarray(W))
        m = mesures(pd.read_csv(sr.PARSED_DIR / f"reports_t2_{election}.csv"))
        self.m = m[m.cellule != "partage"]
        self.elect = sorted(self.m.electorat_t1.unique())

    def predire(self, nu_par_nom, lam):
        nu = np.full(len(self.cands), 0.5)
        for n, x in nu_par_nom.items():
            if _norm(n) in self.idx:
                nu[self.idx[_norm(n)]] = x
        A, poids, ecart = resoudre_w(self.mu, self.sigma, self.w, nu, lam, self.champ, self.cible)
        lse = logsumexp(A, axis=1)
        r = reports_predits(A, self.champ, self.ia, self.ib, self.kappa, self.omega, lse, self.s_ab, poids=poids)
        out = {}
        champ = list(self.champ)
        for e in self.elect:
            j = champ.index(self.idx[_norm(e)])
            M, LP, Au = (100 * r[k][:, j].mean() for k in ("a", "b", "aucun"))
            out[e] = {"Macron": M, "Le Pen": LP, "Aucun": Au, "partage": 100 * M / (M + LP)}
        return out, ecart

    def erreurs(self, pred):
        rows = []
        for _, r in mesures(pd.read_csv(sr.PARSED_DIR / f"reports_t2_{self.election}.csv")).iterrows():
            if r.electorat_t1 in pred:
                rows.append({"election": self.election, "electorat": r.electorat_t1, "cellule": r.cellule,
                             "mesure": round(r["mean"], 1), "modele": round(pred[r.electorat_t1][r.cellule], 1),
                             "erreur": round(pred[r.electorat_t1][r.cellule] - r["mean"], 1)})
        return rows


def ajuster(annees, lam, departs=2):
    def perte(z):
        nu = dict(zip(LIBRES, expit(z)))
        tot = 0.0
        for an in annees:
            pred, _ = an.predire(nu, lam)
            for _, r in an.m.iterrows():
                tot += (pred[r.electorat_t1][r.cellule] - r["mean"]) ** 2
        return tot
    rng = np.random.default_rng(1)
    best = None
    for d in range(departs):
        z0 = np.zeros(len(LIBRES)) if d == 0 else rng.normal(0, 1, len(LIBRES))
        r = minimize(perte, z0, method="Powell", options={"maxiter": 3000, "xtol": 1e-2, "ftol": 1e-3})
        if best is None or r.fun < best.fun:
            best = r
    return dict(zip(LIBRES, expit(best.x))), float(best.fun)


def criteres(err1d, err2d, nu):
    """§8.41 : Mélenchon < 10 pt les deux années ; autres non dégradés > 1 pt ; ν interprétable."""
    t1, t2 = pd.DataFrame(err1d), pd.DataFrame(err2d)
    part = lambda t, mel: t[(t.cellule == "partage") & ((t.electorat == "Jean-Luc Mélenchon") == mel)]  # noqa: E731
    mel_ok = bool((part(t2, True).erreur.abs() < 10).all() and len(part(t2, True)) == 2)
    c = lambda t: t[(t.cellule != "partage") & (t.electorat != "Jean-Luc Mélenchon")].erreur.abs().mean()  # noqa: E731
    autres_ok = bool(c(t2) <= c(t1) + 1.0)
    M, LP, J = nu["Emmanuel Macron"], nu["Marine Le Pen"], nu["Jean-Luc Mélenchon"]
    interp = bool((J - M) * (LP - M) > 0 and all(abs(J - LP) < abs(nu[k] - LP) for k in ("Benoît Hamon", "Yannick Jadot")))
    return {"melenchon_lt10": mel_ok, "autres_non_degrades": autres_ok, "interpretable": interp,
            "mae_autres_1d": round(float(c(t1)), 1), "mae_autres_2d": round(float(c(t2)), 1),
            "passe": mel_ok and autres_ok and interp}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variantes", nargs="+", default=["hier", "abst"])
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    args = ap.parse_args()
    out = []
    for v in args.variantes:
        annees = [Annee(e, v) for e in (2017, 2022)]
        # Référence 1D : mêmes 40 tirages, quadrature 1D (λ → ∞ équivaut à ignorer u).
        err1d = []
        for an in annees:
            A1 = np.asarray(__import__("model.models.spatial_pooling.geometry", fromlist=["x"])
                            .log_attractivite(an.mu, an.sigma, an.w))
            r = reports_predits(A1, an.champ, an.ia, an.ib, an.kappa, an.omega, logsumexp(A1, axis=1), an.s_ab)
            champ = list(an.champ)
            pred = {}
            for e in an.elect:
                j = champ.index(an.idx[_norm(e)])
                M, LP, Au = (100 * r[k][:, j].mean() for k in ("a", "b", "aucun"))
                pred[e] = {"Macron": M, "Le Pen": LP, "Aucun": Au, "partage": 100 * M / (M + LP)}
            err1d += an.erreurs(pred)
        for lam in args.lambdas:
            nu, f = ajuster(annees, lam)
            err2d, ecarts = [], []
            for an in annees:
                pred, ec = an.predire(nu, lam)
                err2d += an.erreurs(pred)
                ecarts.append(ec)
            crit = criteres(err1d, err2d, nu)
            ligne = {"variante": v, "lambda": lam, "perte": round(f, 1), "nu": {k: round(float(x), 3) for k, x in nu.items()},
                     "ecart_t1_max": round(max(ecarts), 5), "criteres": crit, "cellules_2d": err2d, "cellules_1d": err1d}
            out.append(ligne)
            print(json.dumps({k: ligne[k] for k in ("variante", "lambda", "perte", "nu", "ecart_t1_max", "criteres")},
                             ensure_ascii=False), flush=True)
            print(pd.DataFrame(err2d).query("cellule == 'partage'").to_string(index=False), flush=True)
    suff = "_".join(args.variantes) + "_" + "_".join(f"{x:g}" for x in args.lambdas)
    dest = OUT.with_name(f"reports_t2_2d_{suff}.json")
    dest.write_text(json.dumps({"spec": "spec_second_tour §8.41", "essais": out}, ensure_ascii=False, indent=1,
                               default=float) + "\n")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
