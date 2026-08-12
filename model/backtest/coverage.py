"""Backtest de couverture au JOUR DU SCRUTIN : les IC 90 % publiés tiennent-ils ?

Un intervalle à 90 % qui ne contient le résultat réel que 60 % du temps est un
mensonge chiffré, et rien dans les sorties du modèle ne le signale.

On rejoue chaque modèle sur 2017 et 2022, à plusieurs horizons, et on compte
combien de fois le résultat officiel tombe dans l'intervalle annoncé. Les blocs sont ceux de `history_blocks` : même élection, même champ exact de
candidats — le CLR n'a de sens qu'à champ constant.

Complémentaire de `predictive_coverage.py`, qui teste le nowcast. Ici la dérive
jusqu'au scrutin domine ; là-bas c'est la variance de mesure. Un modèle doit
passer les deux.

Exécution : .venv/bin/python -m model.backtest.coverage
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from model.core.bank import Bank
from model.core.simulate import forecast_from_draws
from model.core.opinion import history_blocks, load_law
from model.core.gp_math import gp_posterior
from model.core.projection import delta_draws
from pipeline.historical import load_results

HORIZONS = [30, 60, 90, 120, 180, 250]
N_DRAWS = 4000
SEED = 27
BLOC_FALLBACK = "droite"      # les rosters historiques n'ont pas de bloc politique par slot


def _softmax(a):
    e = np.exp(a - a.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def scrutin_gp(bloc, pool, params, rng) -> np.ndarray:
    """Parts au scrutin selon `gp-pooling` : le postérieur prolongé jusqu'à
    `h = 0` (diffusion), puis l'écart de traduction δ (`terminal.py`). Les deux
    mécanismes sont séparés, contrairement au saut terminal historique."""
    post0 = gp_posterior(bloc.Z[pool], bloc.V[pool] + params["sigma_p"] ** 2,
                         bloc.h[pool], bloc.inst_idx[pool], h_star=0.0,
                         tau=params["tau"], sigma2=params["sigma2"],
                         sigma_h=params["sigma_h"])
    theta = rng.normal(post0.mean, np.sqrt(np.maximum(post0.var_latent, 0.0)),
                       size=(N_DRAWS, bloc.K))
    return _softmax(theta + delta_draws(params, theta.shape, SEED + 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--niveau", type=float, default=0.90)
    args = ap.parse_args()
    lo_q, hi_q = 50 * (1 - args.niveau), 100 - 50 * (1 - args.niveau)

    params = load_law()
    res = load_results(); res = res[res["tour"] == "Premier tour"]
    blocs = history_blocks(min_polls=3)
    rng = np.random.default_rng(SEED)

    lignes = []
    for b in blocs:
        verite = dict(zip(res.loc[res.election == b.election, "candidat"],
                          res.loc[res.election == b.election, "resultat"]))
        if not set(b.roster) & set(verite):
            continue
        for h in HORIZONS:
            pool = np.where(b.h >= h)[0]
            if len(pool) < 3:
                continue
            pi = scrutin_gp(b, pool, params, rng)
            fc = forecast_from_draws(pi, list(b.roster), h, drift_sd=0.0)
            for c in b.roster:
                if c not in verite:
                    continue
                lo, hi = np.array(fc["forecast_scrutin"][c]["ic90"]) * 100
                lignes.append({"horizon": h, "election": b.election,
                               "dans": lo <= verite[c] <= hi, "largeur": hi - lo})
    t = pd.DataFrame(lignes)
    if t.empty:
        raise SystemExit("aucune observation testable")

    n = int(args.niveau * 100)
    print(f"Couverture au jour du scrutin (IC {n} %) — {len(t)} observations\n")
    print(f"{'horizon':>8s} {'obs':>6s} {'couverture':>11s} {'largeur med':>12s}")
    for h, g in t.groupby("horizon"):
        print(f"{h:8d} {len(g):6d} {g['dans'].mean():11.3f} {g['largeur'].median():11.1f}pt")
    print(f"\nglobal : {t['dans'].mean():.3f}  (nominal {args.niveau:.2f}) — "
          f"largeur médiane {t['largeur'].median():.1f} pt")


if __name__ == "__main__":
    main()
