"""Idée B (spec_spatial_pooling §10) : combiner spatial et gp-pooling, 0 fit.

    python -m model.backtest.combinaison
"""
from __future__ import annotations

import glob
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

from model.backtest import second_round as sr

Z90 = 1.6449
ALPHA = 0.1


def charge() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(sr.OUT_DIR / "suivi_gp_r17f3_c7_2022_h*.json"))):
        h = int(re.search(r"_h(\d+)\.json", f).group(1))
        g = json.load(open(f))
        sp = json.load(open(sr.OUT_DIR / f"suivi_proj_2022_h{h:03d}.json"))
        S = {(l["liste"], l["candidat"]): l for l in sp["lignes"]}
        for l in g["lignes"]:
            k = (l["liste"], l["candidat"])
            if k not in S or l.get("modele") is None:
                continue
            s = S[k]
            rows.append({"h": h, "liste": l["liste"], "candidat": l["candidat"], "y": l["observe"],
                         "m_gp": l["modele"], "s_gp": (l["ic90"][1] - l["ic90"][0]) / (2 * Z90),
                         "m_sp": s["modele"], "s_sp": (s["ic90"][1] - s["ic90"][0]) / (2 * Z90)})
    d = pd.DataFrame(rows)
    d["unite"] = d.h.astype(str) + "|" + d.liste
    return d


def _quantiles_melange(w, m1, s1, m2, s2, q):
    """Quantiles d'un mélange de deux normales (bissection vectorisée)."""
    lo = np.minimum(m1 - 6 * s1, m2 - 6 * s2); hi = np.maximum(m1 + 6 * s1, m2 + 6 * s2)
    for _ in range(60):
        mid = (lo + hi) / 2
        F = w * norm.cdf(mid, m1, s1) + (1 - w) * norm.cdf(mid, m2, s2)
        lo, hi = np.where(F < q, mid, lo), np.where(F < q, hi, mid)
    return (lo + hi) / 2


def scores(y, m, lo, hi) -> dict:
    is_ = (hi - lo) + 2 / ALPHA * np.clip(lo - y, 0, None) + 2 / ALPHA * np.clip(y - hi, 0, None)
    return {"ae": 100 * np.abs(m - y), "is": 100 * is_, "cov": ((lo <= y) & (y <= hi)).astype(float)}


def normale(m, s):
    return m, m - Z90 * s, m + Z90 * s


def melange(d, w):
    lo = _quantiles_melange(w, d.m_gp.values, d.s_gp.values, d.m_sp.values, d.s_sp.values, 0.05)
    hi = _quantiles_melange(w, d.m_gp.values, d.s_gp.values, d.m_sp.values, d.s_sp.values, 0.95)
    return w * d.m_gp.values + (1 - w) * d.m_sp.values, lo, hi


def ls_melange(d, w):
    p = w * norm.pdf(d.y, d.m_gp, d.s_gp) + (1 - w) * norm.pdf(d.y, d.m_sp, d.s_sp)
    return np.log(np.clip(p, 1e-300, None)).sum()


def ls_c3(d, w, k):
    m = w * d.m_gp + (1 - w) * d.m_sp
    return norm.logpdf(d.y, m, k * d.s_sp).sum()


W = np.round(np.arange(0, 1.0001, 0.05), 2)
K = np.round(np.arange(0.8, 2.0001, 0.1), 2)


def evalue(d: pd.DataFrame) -> pd.DataFrame:
    out = {}
    y = d.y.values
    out["gp"] = scores(y, *normale(d.m_gp.values, d.s_gp.values))
    out["spatial"] = scores(y, *normale(d.m_sp.values, d.s_sp.values))
    out["C1"] = scores(y, *normale(d.m_gp.values, d.s_sp.values))
    c2 = {k: np.zeros(len(d)) for k in ("ae", "is", "cov")}
    c3 = {k: np.zeros(len(d)) for k in ("ae", "is", "cov")}
    choix = []
    for h in sorted(d.h.unique()):
        tr, te = d[d.h != h], d.h == h
        w2 = W[np.argmax([ls_melange(tr, w) for w in W])]
        m, lo, hi = melange(d[te], w2)
        for k, v in scores(y[te], m, lo, hi).items():
            c2[k][te.values] = v
        g = [(ls_c3(tr, w, kk), w, kk) for w in W for kk in K]
        _, w3, k3 = max(g)
        m = w3 * d[te].m_gp.values + (1 - w3) * d[te].m_sp.values
        for k, v in scores(y[te], *normale(m, k3 * d[te].s_sp.values)).items():
            c3[k][te.values] = v
        choix.append({"h": int(h), "w_C2": float(w2), "w_C3": float(w3), "k_C3": float(k3)})
    out["C2"], out["C3"] = c2, c3
    rows = []
    for nom, sc in out.items():
        for i in range(len(d)):
            rows.append({"methode": nom, "unite": d.unite.iloc[i], **{k: float(v[i]) for k, v in sc.items()}})
    return pd.DataFrame(rows), choix


def bootstrap_delta(t: pd.DataFrame, a: str, b: str, col: str, n=10_000, seed=4):
    u = t.groupby(["methode", "unite"])[col].mean().unstack(0)
    dlt = (u[a] - u[b]).to_numpy()
    rng = np.random.default_rng(seed)
    m = dlt[rng.integers(0, len(dlt), (n, len(dlt)))].mean(1)
    return round(float(dlt.mean()), 3), [round(float(x), 3) for x in np.percentile(m, [5, 95])]


def main() -> None:
    d = charge()
    t, choix = evalue(d)
    res = {"n_lignes": int(len(d)), "n_unites": int(d.unite.nunique()), "choix_loco": choix,
           "methodes": {m: {"mae": round(g.ae.mean(), 3), "score_intervalle": round(g["is"].mean(), 3),
                            "couverture": round(g["cov"].mean(), 3)} for m, g in t.groupby("methode")},
           "deltas": {}}
    for c in ("C1", "C2", "C3"):
        for ref in ("gp", "spatial"):
            res["deltas"][f"{c}-{ref}"] = {"mae": bootstrap_delta(t, c, ref, "ae"),
                                          "score_intervalle": bootstrap_delta(t, c, ref, "is")}
    out = sr.OUT_DIR / "combinaison_spatial_gp_2022.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
