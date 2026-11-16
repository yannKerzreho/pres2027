"""Effets maison au second tour par REML (spec_second_tour §8.23), sans fit.

Par campagne : y_d = part du premier nommé du duel d (points), modèle mixte
    y_d = β_{cellule(d)} + u_{institut(d)} + v_{notice(d)} + e_d,
    u ~ N(0, σ_h²), v ~ N(0, σ_n²), e_d ~ N(0, b_d + σ²), b_d = y(1-y)/N_d,
y orientée vers le candidat de référence (REFERENCE),
cellule = (paire, fenêtre de 14 j) en effet fixe (absorbe la tendance et la
paire). REML ; test du rapport de vraisemblance σ_h = 0 (mélange ½χ²₀ + ½χ²₁).

    python -m model.backtest.maison_t2
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from model.backtest import second_round as sr
from model.backtest import v1_2027 as v
from model.core.live_dataset import load_raw_polls

FENETRE_J = 14
# Orientation : part du candidat de référence (un effet maison sur « le premier
# par ordre alphabétique » n'a pas de sens quand l'ordre change d'une paire à
# l'autre). 2017/2022 : Macron ; 2027 : le candidat RN. Duels sans lui écartés.
REFERENCE = {2017: {"Emmanuel Macron"}, 2022: {"Emmanuel Macron"}, 2027: {"Le Pen", "Bardella"}}


def duels_campagne(e: int) -> pd.DataFrame:
    if e == 2027:
        raw = load_raw_polls(tour=None)
    else:
        raw = sr.polls_historiques(e, sr.T2_DATES[e] - pd.Timedelta(days=1), history_days=400)
    d = v._duels(raw)
    ref = REFERENCE[e]
    d = d[d.a.isin(ref) ^ d.b.isin(ref)].copy()
    inv = d.b.isin(ref)
    d.loc[inv, "y"] = 1 - d.loc[inv, "y"]
    d.loc[inv, ["a", "b"]] = d.loc[inv, ["b", "a"]].to_numpy()
    d["institut"] = d.notice.map(raw.drop_duplicates("notice").set_index("notice")["institut"])
    d["cellule"] = (d.a + "/" + d.b + "/" +
                    ((d.date - d.date.min()).dt.days // FENETRE_J).astype(str))
    return d


def reml(d: pd.DataFrame, avec_maison: bool = True) -> dict:
    y = 100 * d.y.to_numpy()
    b = 1e4 * d.y.to_numpy() * (1 - d.y.to_numpy()) / d.n_ech.to_numpy()
    X = pd.get_dummies(d.cellule).to_numpy(float)
    Zh = pd.get_dummies(d.institut).to_numpy(float)
    Zn = pd.get_dummies(d.notice).to_numpy(float)
    Kh, Kn = Zh @ Zh.T, Zn @ Zn.T
    n, p = X.shape

    def nll(th):
        s2h = np.exp(th[0]) if avec_maison else 0.0
        s2n, s2 = np.exp(th[-2]), np.exp(th[-1])
        V = s2h * Kh + s2n * Kn + np.diag(b + s2)
        L = np.linalg.cholesky(V)
        Li = np.linalg.solve(L, np.eye(n))
        Vi = Li.T @ Li
        XtViX = X.T @ Vi @ X
        beta = np.linalg.solve(XtViX, X.T @ Vi @ y)
        r = y - X @ beta
        return 0.5 * (2 * np.log(np.diag(L)).sum() + np.linalg.slogdet(XtViX)[1] + r @ Vi @ r)

    th0 = np.log([0.5, 0.5, 0.5]) if avec_maison else np.log([0.5, 0.5])
    best = min((minimize(nll, th0 + dlt, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-6})
                for dlt in (0.0, -3.0, 2.0)), key=lambda o: o.fun)
    th = np.exp(best.x)
    out = {"n_duels": int(n), "n_cellules": int(p), "n_notices": int(Zn.shape[1]),
           "n_instituts": int(Zh.shape[1]), "reml_nll": float(best.fun),
           "sigma_notice_pt": round(float(np.sqrt(th[-2])), 3), "sigma_pt": round(float(np.sqrt(th[-1])), 3)}
    if avec_maison:
        out["sigma_maison_pt"] = round(float(np.sqrt(th[0])), 3)
    return out


def main() -> None:
    global FENETRE_J
    res = {}
    for e in (2017, 2022, 2027):
        res[e] = {}
        for f in (7, 14, 30):
            FENETRE_J = f
            d = duels_campagne(e)
            h1, h0 = reml(d, True), reml(d, False)
            lr = max(0.0, 2 * (h0["reml_nll"] - h1["reml_nll"]))
            res[e][f"fenetre_{f}j"] = {"avec_maison": h1, "sans_maison": h0, "lr": round(lr, 3),
                                       "p_valeur": round(float(0.5 * chi2.sf(lr, 1)) if lr > 0 else 0.5, 4)}
            print(e, f, json.dumps(res[e][f"fenetre_{f}j"], ensure_ascii=False), flush=True)
    out = sr.OUT_DIR / "maison_t2_reml.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
