"""Phase 3 — modèle bayésien live du 1er tour 2027.

On estime la répartition des voix au 1er tour (`π`, simplexe sur les slots de
candidature) à partir des sondages 2027 parsés, en appliquant les **house
effects calibrés** de la Phase 1 :

  - chaque part sondée est **débiaisée** avec le biais `biais[institut, bloc]`
    calibré (soustraction du biais à J-0) ;
  - le bruit d'observation de chaque part est l'écart-type calibré
    `sqrt(échantillonnage + excès(horizon)²)` — **indépendant** par sondage,
    donc réductible en accumulant les sondages.

`π` est un *nowcast* (état de l'opinion au moment des sondages). L'incertitude
**irréductible** de dérive d'ici au scrutin (commune à tous les sondages) est
ajoutée ensuite en simulation (`simulate.py`), pas ici.

Modèle :
    π           ~ Dirichlet(α)                      (simplexe sur les slots)
    part_débiaisée_o ~ Normal(100·π[slot_o], σ_o)   (σ_o connu, par observation)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calibration.priors_utils import load_priors, expected_bias, observation_sigma
from model.live_dataset import load_live_observations, SLOTS


def prepare(obs: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """Ajoute par observation : biais calibré, part débiaisée, sigma d'obs."""
    obs = obs.copy()
    obs["bias"] = [expected_bias(priors, r.institut, r.bloc)["mean"]
                   for r in obs.itertuples()]
    obs["part_debiais"] = obs["part"] - obs["bias"]
    obs["sigma"] = [observation_sigma(priors, r.institut, int(r.horizon),
                                      max(r.part, 0.5), int(r.echantillon))
                    for r in obs.itertuples()]
    return obs


def build_and_fit(obs: pd.DataFrame, priors: dict, draws: int = 1000,
                  tune: int = 1000, chains: int = 4, seed: int = 27):
    obs = prepare(obs, priors)
    slots = list(SLOTS)
    slot_idx = obs["slot"].map({s: i for i, s in enumerate(slots)}).to_numpy()

    coords = {"slot": slots, "obs": np.arange(len(obs))}
    with pm.Model(coords=coords) as model:
        pi = pm.Dirichlet("pi", a=np.ones(len(slots)), dims="slot")
        mu = pi * 100.0
        pm.Normal("y", mu=mu[slot_idx], sigma=obs["sigma"].to_numpy(),
                  observed=obs["part_debiais"].to_numpy(), dims="obs")
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=0.95, random_seed=seed,
                          nuts_sampler="numpyro", progressbar=False)
    return idata, slots, obs


if __name__ == "__main__":
    priors = load_priors()
    obs = load_live_observations()
    idata, slots, obs = build_and_fit(obs, priors)
    import arviz as az
    s = az.summary(idata, var_names=["pi"], round_to=3)
    s.index = slots
    print(s[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat"]].sort_values("mean", ascending=False).to_string())
    print("\nR-hat max:", float(az.rhat(idata)["pi"].max()))
