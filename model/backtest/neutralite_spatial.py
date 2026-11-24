"""Neutralité du spatial de production, main contre branche (spec_second_tour §8.91).

    PYTHONPATH=<arbre> python model/backtest/neutralite_spatial.py fit <sortie.json> [--seed 27]
    python model/backtest/neutralite_spatial.py comparer <main27> <main28> <branche27> <branche28>

`fit` s'exécute DANS l'arbre désigné par PYTHONPATH (main extrait par `git archive`, ou la branche),
réglages de production, premier tour seul ; il appelle `fit_spatial_pooling` directement et n'écrit
rien d'autre que <sortie.json> (pas de docs/, pas de scenarios.json).
"""
from __future__ import annotations

import inspect
import json
import sys

import numpy as np


def ajuster(sortie: str, seed: int) -> None:
    from model.core.live_dataset import load_raw_polls
    from model.models.spatial_pooling import fit as F
    raw = load_raw_polls(as_of="2026-11-24")
    kw = {"seed": seed}
    if "second_tour" in inspect.signature(F.fit_spatial_pooling).parameters:
        kw.update(second_tour=False, eta_hier=True)          # chemin de production (PUBLIER_T2 False)
    fit = F.fit_spatial_pooling(raw, as_of="2026-11-24", **kw)
    pi = np.asarray(F.pi_draws_for_mask(fit, np.ones(len(fit.candidates))))
    d = fit.diagnostics
    json.dump({"module": F.__file__, "seed": seed, "rhat_max": d["rhat_max"], "ess_min": d["ess_min"],
               "n_div": d.get("n_divergences"), "candidats": list(fit.candidates),
               "mean": pi.mean(0).tolist(), "sd": pi.std(0).tolist()}, open(sortie, "w"), ensure_ascii=False)


def comparer(m27, m28, b27, b28) -> None:
    M, N, B, C = (json.load(open(p)) for p in (m27, m28, b27, b28))
    assert M["candidats"] == B["candidats"], "candidats différents"
    sd = np.maximum(np.array(M["sd"]), 1e-6)
    e = lambda x, y: np.abs(np.array(x["mean"]) - np.array(y["mean"])) / sd
    r = {"graines main/main": e(M, N), "graines branche/branche": e(B, C),
         "main/branche graine 27": e(M, B), "main/branche graine 28": e(N, C)}
    for k, v in r.items():
        print(f"{k:28s} max {v.max():.3f} sd   moyenne {v.mean():.3f} sd")
    bruit = max(r["graines main/main"].max(), r["graines branche/branche"].max())
    ecart = max(r["main/branche graine 27"].max(), r["main/branche graine 28"].max())
    print("rhat", [x["rhat_max"] for x in (M, N, B, C)], "ess", [x["ess_min"] for x in (M, N, B, C)])
    print("VERDICT", "neutre au bruit près" if ecart <= 1.5 * bruit and
          max(x["rhat_max"] for x in (M, N, B, C)) <= 1.2 else "NON conclu")


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        ajuster(sys.argv[2], int(sys.argv[4]) if len(sys.argv) > 4 else 27)
    else:
        comparer(*sys.argv[2:6])
