"""Backtest de couverture PRÉDICTIVE du nowcast — le test que le backtest au
jour du scrutin ne sait pas faire.

`model/backtest/coverage.py` vérifie les intervalles à J-0. À 250 jours, la
dérive écrase tout : ce test valide surtout la loi terminale et reste quasiment
aveugle à la variance de MESURE — or c'est elle qui produit le bloc `nowcast`,
celui qu'affiche le site sous « intentions ».

Protocole : on retire un sondage, on le prédit à partir des seuls sondages
ANTÉRIEURS du même bloc (même élection, même champ exact de candidats), et on
regarde si sa part observée tombe dans l'intervalle prédictif à 90 %. Celui-ci
inclut le bruit d'échantillonnage du sondage retiré — on prédit une observation,
pas l'opinion latente.

La couverture est ventilée par **taille du pool** : c'est le régime à peu de
sondages qui discrimine les modèles, et c'est celui où nous sommes.

Exécution : .venv/bin/python -m model.backtest.predictive_coverage
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from model.models.gp_pooling.calibration import history_blocks, load_params
from model.models.gp_pooling.gp import gp_posterior

N_DRAWS = 4000
SEED = 27


def _softmax(a: np.ndarray) -> np.ndarray:
    e = np.exp(a - a.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def predire(bloc, cible: int, pool: np.ndarray, params: dict,
            rng: np.random.Generator) -> np.ndarray:
    """Tirages prédictifs `(S, K)` de la composition qu'observerait le sondage
    `cible` : postérieur à sa date, avec la variance d'OBSERVATION (house effect
    du sondage retiré + son bruit d'échantillonnage)."""
    post = gp_posterior(
        bloc.Z[pool], bloc.V[pool] + params["sigma_p"] ** 2, bloc.h[pool],
        bloc.inst_idx[pool], h_star=float(bloc.h[cible]),
        tau=params["tau"], sigma2=params["sigma2"], sigma_h=params["sigma_h"],
        target_inst=int(bloc.inst_idx[cible]),
        target_v=bloc.V[cible] + params["sigma_p"] ** 2)
    theta = rng.normal(post.mean, np.sqrt(np.maximum(post.var_obs, 0.0)),
                       size=(N_DRAWS, bloc.K))
    return _softmax(theta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--niveau", type=float, default=0.90)
    args = ap.parse_args()
    lo_q, hi_q = 50 * (1 - args.niveau), 100 - 50 * (1 - args.niveau)

    params = load_params()
    rng = np.random.default_rng(SEED)

    lignes = []
    for b in history_blocks(min_polls=2):
        ordre = np.argsort(-b.h)                       # du plus ancien au plus récent
        for pos, cible in enumerate(ordre):
            pool = ordre[:pos]
            if len(pool) < 1:
                continue
            P = predire(b, cible, pool, params, rng)
            for s in range(b.K):
                lo, hi = np.percentile(P[:, s], [lo_q, hi_q]) * 100
                lignes.append({"n_pool": len(pool),
                               "dans": lo <= b.Y[cible, s] * 100 <= hi,
                               "largeur": hi - lo})
    t = pd.DataFrame(lignes)
    if t.empty:
        raise SystemExit("aucune observation testable")
    t["tranche"] = pd.cut(t["n_pool"], [0, 2, 4, 9, 10**6],
                          labels=["1-2", "3-4", "5-9", "10+"])

    n = int(args.niveau * 100)
    print(f"Couverture prédictive du nowcast (IC {n} %) — {len(t)} observations\n")
    print(f"{'tranche':>8s} {'obs':>6s} {'couverture':>11s} {'largeur med':>12s}")
    for tr, g in t.groupby("tranche", observed=True):
        print(f"{str(tr):>8s} {len(g):6d} {g['dans'].mean():11.3f} {g['largeur'].median():11.1f}pt")
    print(f"\nglobal : {t['dans'].mean():.3f}  (nominal {args.niveau:.2f})")


if __name__ == "__main__":
    main()
