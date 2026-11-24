"""Témoin R1 de V2 (spec_second_tour §8.77 c) : régression sur les parts T1, 0 fit.

    logit y_ab = β (log t_a − log t_b)

MCO sans constante, pondérés par le nombre de duels, ajustés sur les AUTRES paires éligibles
(≥ `n_min` duels dans la fenêtre de 30 jours avant `as_of`). t_x : part T1 de x, moyenne par notice
sur l'hypothèse la plus large, puis moyenne des notices sur la même fenêtre.

    python -m model.backtest.temoin_r1 --election 2022
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import model.backtest.second_round as sr


def tableau(raw: pd.DataFrame, as_of: pd.Timestamp, jours: int = 30):
    dep = as_of - pd.Timedelta(days=jours)
    d = sr._duels_moyens(raw, dep)
    g = d.groupby(["a", "b"]).y.agg(["mean", "count"]).reset_index()
    t1 = raw[raw.tour.eq("Premier tour") & raw.date_fin.ge(dep)]
    t1 = t1.assign(_h=t1["hypothese"].fillna("__u__"))
    larg = t1.groupby(["notice", "_h"]).candidat.transform("count")
    t1 = t1[larg.eq(t1.assign(l=larg).groupby("notice").l.transform("max"))]
    p = t1.groupby(["notice", "candidat"]).intention.mean().groupby("candidat").mean()
    return g, p


def predire(train: pd.DataFrame, p: pd.Series, a: str, b: str) -> float:
    x = np.log(p[train.a].values) - np.log(p[train.b].values)
    y = np.log(train["mean"].values / (1 - train["mean"].values))
    w = train["count"].values
    beta = (w * x * y).sum() / (w * x * x).sum()
    return float(1 / (1 + np.exp(-beta * (np.log(p[a]) - np.log(p[b])))))


def evaluer(election: int, coupure: str = "T1-30", n_min: int = 3) -> dict:
    as_of = sr.T1_DATES[election] - pd.Timedelta(days=int(coupure.split("-")[1]))
    g, p = tableau(sr.polls_historiques(election, as_of), as_of)
    el = g[g["count"] >= n_min]
    out = []
    for _, r in el.iterrows():
        tr = el[~((el.a == r.a) & (el.b == r.b))]
        pred = predire(tr, p, r.a, r.b)
        out.append({"paire": [r.a, r.b], "observe": round(100 * r["mean"], 2),
                    "R1": round(100 * pred, 2), "ecart": round(100 * (pred - r["mean"]), 2)})
    return {"spec": "§8.77 c", "election": election, "coupure": coupure, "paires": out,
            "mae": round(float(np.mean([abs(o["ecart"]) for o in out])), 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--election", type=int, default=2022)
    ap.add_argument("--coupure", default="T1-30")
    a = ap.parse_args()
    res = evaluer(a.election, a.coupure)
    dest = sr.OUT_DIR / f"temoin_r1_{a.election}_{a.coupure}.json"
    dest.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1))
