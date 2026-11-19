"""(R1) Part anti-système par électorat, par famille (spec_second_tour §8.43). 0 fit.

    python -m model.backtest.reports_r1 [--variantes hier abst]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

import model.backtest.second_round as sr
from model.backtest.reports_t2 import OUT, _fit_v2, _norm, composantes, melange, mesures
from model.models.spatial_pooling.fit import duel_draws
from model.models.spatial_pooling.geometry import log_attractivite

FAMILLES = ["GR", "GE", "D", "DR"]
FAMILLE = {"jean-luc melenchon": "GR", "philippe poutou": "GR", "nathalie arthaud": "GR", "fabien roussel": "GR",
           "benoit hamon": "GE", "yannick jadot": "GE", "anne hidalgo": "GE",
           "francois fillon": "D", "valerie pecresse": "D",
           "nicolas dupont-aignan": "DR", "eric zemmour": "DR"}
CELLULES = ("Macron", "Le Pen", "Aucun")


class Annee:
    def __init__(self, election, variante):
        fit = _fit_v2(election, variante)
        self.election = election
        idx = {_norm(c): i for i, c in enumerate(fit.candidates)}
        res = sr.load_results()
        t1 = res[(res.election == election) & (res.tour == "Premier tour")].candidat
        champ = [idx[_norm(c)] for c in t1 if _norm(c) in idx]
        self.noms = [_norm(fit.candidates[i]) for i in champ]
        a, b = sr.FINALE[election]
        A = np.asarray(log_attractivite(fit.mu_draws, fit.sigma_draws, fit.w_draws))
        lse = logsumexp(A, axis=1)
        self.s_ab = duel_draws(fit, a, b, 0.0, scrutin=False)
        self.E, self.R, self.G = composantes(A, np.array(champ), idx[_norm(a)], idx[_norm(b)],
                                             fit.kappa_draws, fit.omega_draws, lse)
        m = mesures(pd.read_csv(sr.PARSED_DIR / f"reports_t2_{election}.csv"))
        self.m = m
        self.fam = np.array([FAMILLE.get(n, "") for n in self.noms])

    def cellules(self, pi_as: dict, pi_aucun: dict | None = None) -> tuple[dict, dict]:
        pa = np.array([pi_as.get(f, 0.0) for f in self.fam])
        pn = None if pi_aucun is None else np.array([pi_aucun.get(f, 0.0) for f in self.fam])
        r = melange(self.E, self.R, self.G, self.s_ab, pa, pn)
        out = {}
        for e in self.m.electorat_t1.unique():
            j = self.noms.index(_norm(e))
            M, LP, Au = (100 * r[k][:, j].mean() for k in ("a", "b", "aucun"))
            out[e] = {"Macron": M, "Le Pen": LP, "Aucun": Au, "partage": 100 * M / (M + LP)}
        info = {"delta_moyen": float(r["delta"].mean()),
                "decalage_duel_pt": float(100 * r["decalage_duel"].mean())}
        return out, info

    def erreurs(self, pred):
        return [{"election": self.election, "electorat": r.electorat_t1, "cellule": r.cellule,
                 "mesure": round(float(r["mean"]), 1), "modele": round(float(pred[r.electorat_t1][r.cellule]), 1),
                 "erreur": round(float(pred[r.electorat_t1][r.cellule] - r["mean"]), 1)}
                for _, r in self.m.iterrows()]


def ajuster(annees, aucun: bool):
    n = len(FAMILLES)

    def unpack(z):
        pa = expit(z[:n])
        pn = expit(z[n:]) if aucun else None
        if aucun:   # garder pa + pn < 1
            tot = np.maximum(pa + pn, 1.0)
            pa, pn = pa / tot, pn / tot
        return dict(zip(FAMILLES, pa)), (dict(zip(FAMILLES, pn)) if aucun else None)

    def perte(z):
        pa, pn = unpack(z)
        tot = 0.0
        for an in annees:
            pred, _ = an.cellules(pa, pn)
            for _, r in an.m[an.m.cellule.isin(CELLULES)].iterrows():
                tot += (pred[r.electorat_t1][r.cellule] - r["mean"]) ** 2
        return tot

    z0 = np.full(2 * n if aucun else n, -2.0)
    r = minimize(perte, z0, method="Powell", options={"maxiter": 4000, "xtol": 1e-3, "ftol": 1e-4})
    pa, pn = unpack(r.x)
    return pa, pn, float(r.fun)


def _r(d):
    return None if d is None else {k: round(float(v), 3) for k, v in d.items()}


def criteres(err0, errj, pa_sep):
    t0, tj = pd.DataFrame(err0), pd.DataFrame(errj)
    mel = tj[(tj.cellule == "partage") & (tj.electorat == "Jean-Luc Mélenchon")]
    c1 = bool(len(mel) == 2 and (mel.erreur.abs() < 10).all())
    autres = lambda t: t[t.cellule.isin(CELLULES) & (t.electorat != "Jean-Luc Mélenchon")].erreur.abs().mean()  # noqa: E731
    c2 = bool(autres(tj) <= autres(t0) + 1.0)
    ecarts = {f: abs(pa_sep[2017][f] - pa_sep[2022][f]) for f in FAMILLES}
    c3 = bool(all(v < 0.1 for v in ecarts.values()))
    return {"melenchon_lt10": c1, "partage_melenchon_erreur": mel.erreur.tolist(),
            "autres_non_degrades": c2, "mae_autres_sans_melange": round(float(autres(t0)), 1),
            "mae_autres_melange": round(float(autres(tj)), 1),
            "stabilite_pi": c3, "ecart_pi_2017_2022": {k: round(v, 3) for k, v in ecarts.items()},
            "passe": c1 and c2 and c3}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variantes", nargs="+", default=["hier", "abst"])
    args = ap.parse_args()
    sortie = {"spec": "spec_second_tour §8.43", "familles": FAMILLE, "resultats": []}
    for v in args.variantes:
        annees = {e: Annee(e, v) for e in (2017, 2022)}
        err0 = sum((an.erreurs(an.cellules({})[0]) for an in annees.values()), [])
        for aucun in (False, True):
            nom = "R1b" if aucun else "R1a"
            pa, pn, f = ajuster(list(annees.values()), aucun)
            errj, infos = [], {}
            for e, an in annees.items():
                pred, info = an.cellules(pa, pn)
                errj += an.erreurs(pred)
                infos[e] = {k: round(x, 3) for k, x in info.items()}
            sep = {e: ajuster([an], aucun)[:2] for e, an in annees.items()}
            loyo = []
            for e, an in annees.items():
                autre = 2022 if e == 2017 else 2017
                pred, _ = an.cellules(*sep[autre])
                loyo += an.erreurs(pred)
            ligne = {"variante": v, "forme": nom, "perte": round(f, 1), "pi_as": _r(pa), "pi_aucun": _r(pn),
                     "duel": infos,
                     "pi_as_par_annee": {e: _r(sep[e][0]) for e in sep},
                     "pi_aucun_par_annee": {e: _r(sep[e][1]) for e in sep},
                     "criteres": criteres(err0, errj, {e: sep[e][0] for e in sep}),
                     "cellules": errj, "cellules_annee_retiree": loyo, "cellules_sans_melange": err0}
            sortie["resultats"].append(ligne)
            print(json.dumps({k: ligne[k] for k in ("variante", "forme", "perte", "pi_as", "pi_aucun", "duel",
                                                    "pi_as_par_annee", "criteres")}, ensure_ascii=False), flush=True)
            t = pd.DataFrame(errj).merge(pd.DataFrame(loyo)[["election", "electorat", "cellule", "erreur"]]
                                         .rename(columns={"erreur": "erreur_annee_retiree"}),
                                         on=["election", "electorat", "cellule"])
            print(t[t.cellule == "partage"].to_string(index=False), flush=True)
    dest = OUT.with_name("reports_t2_r1.json")
    dest.write_text(json.dumps(sortie, ensure_ascii=False, indent=1, default=float) + "\n")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
