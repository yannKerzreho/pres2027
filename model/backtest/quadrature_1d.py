"""Précision de la quadrature 1D aux petits σ (spec_second_tour §8.48). 0 fit.

    python -m model.backtest.quadrature_1d

Réévalue les mêmes tirages des fits témoins 1D 2022 avec B = 25 (production) et
B = 400 (référence), en remplaçant la grille de `geometry` le temps du calcul.
"""
from __future__ import annotations

import json
from itertools import combinations

import jax.numpy as jnp
import numpy as np

import model.backtest.second_round as sr
import model.models.spatial_pooling.geometry as G
from model.backtest.eval_r2 import DUELS_CLES, ELECTION, _parts, reconstruire
from model.backtest.reports_t2 import _norm
from model.models.spatial_pooling.fit import duel_draws

CAS = [("r2_temoin_hier_v2_2022_T2-7", "T2-7"), ("r2_temoin_hier_v2_2022_T1-30", "T1-30")]


def grille(B: int):
    x, w = np.polynomial.legendre.leggauss(B)
    return jnp.asarray(0.01 + 0.98 * (x + 1.0) / 2.0), jnp.asarray(w / 2.0)


def mesurer(fit) -> dict:
    duels = {f"{a} / {b}": round(100 * float(duel_draws(fit, a, b, 0.0, scrutin=False).mean()), 3)
             for a, b in combinations(DUELS_CLES, 2)}
    res = sr.load_results()
    t1 = res[(res.election == ELECTION) & (res.tour == "Premier tour")]
    idx = {_norm(c): i for i, c in enumerate(fit.candidates)}
    vrai = t1.assign(i=t1.candidat.map(lambda c: idx.get(_norm(c)))).dropna(subset=["i"])
    mask = np.zeros(len(fit.candidates)); mask[vrai.i.astype(int)] = 1.0
    col = int(np.clip(np.searchsorted(fit.w_dates, float(sr.T1_DATES[ELECTION].toordinal())), 0,
                      fit.w_path_draws.shape[2] - 1))
    pm = _parts(fit, fit.w_path_draws[:, :, col], mask)[:, vrai.i.astype(int)].mean(0)
    y = vrai.resultat.to_numpy() / vrai.resultat.sum()
    return {"duels": duels, "t1_mae_pt": round(100 * float(np.abs(pm - y).mean()), 3),
            "parts_t1": {c: round(100 * float(p), 3) for c, p in zip(vrai.candidat, pm)}}


def main() -> None:
    out = {"spec": "spec_second_tour §8.48", "cas": []}
    V0, W0 = G.V, G.W
    for stem, coupure in CAS:
        fit, _, _ = reconstruire("hier", stem, coupure)
        r = {"stem": stem, "coupure": coupure,
             "sigma_min_mediane": round(float(np.median(fit.sigma_draws, 0).min()), 4)}
        for B in (25, 400):
            G.V, G.W = (V0, W0) if B == 25 else grille(B)
            r[f"B{B}"] = mesurer(fit)
        G.V, G.W = V0, W0
        r["ecart_duels_max_pt"] = round(max(abs(r["B25"]["duels"][k] - r["B400"]["duels"][k]) for k in r["B25"]["duels"]), 3)
        r["ecart_parts_t1_max_pt"] = round(max(abs(r["B25"]["parts_t1"][k] - r["B400"]["parts_t1"][k])
                                               for k in r["B25"]["parts_t1"]), 3)
        r["ecart_mae_t1_pt"] = round(r["B25"]["t1_mae_pt"] - r["B400"]["t1_mae_pt"], 3)
        out["cas"].append(r)
        print({k: r[k] for k in ("coupure", "sigma_min_mediane", "ecart_duels_max_pt", "ecart_parts_t1_max_pt",
                                 "ecart_mae_t1_pt")}, flush=True)
        print({k: (r["B25"]["duels"][k], r["B400"]["duels"][k]) for k in r["B25"]["duels"]}, flush=True)
    dest = sr.OUT_DIR / "quadrature_1d_2022.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
