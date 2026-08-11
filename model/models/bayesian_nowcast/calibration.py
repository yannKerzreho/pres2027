"""Calibration du modèle « bayesian-nowcast » : house effects (biais + dérive).

Apprend, sur l'historique 2017/2022, le biais d'un institut sur un bloc
(house effect à J-0) et la dérive d'opinion propre à chaque campagne —
partial pooling, non centré. Composé via `run_numpyro_mcmc`
(`model/core/inference.py`), sans ABC dédiée : `HouseEffectsCalibration` est
une classe normale. Produit une `Bank` (`model/core/bank.py`, générique, ne
connaît rien du domaine) consommée par `nowcast.py`.

`horizon_diffusion` (la loi de diffusion horizon -> variance/dérive) est
définie une seule fois, ici, réutilisée en JAX (`house_effects_model`) et en
NumPy (les sigmas dérivées ci-dessous, pour le backtest/les notebooks) — un
autre modèle est libre de choisir une autre forme (log1p, sinh-arcsinh
calibré...), rien dans le framework n'impose une forme.

Le **saut terminal** (`TerminalJumpCalibration` et son jeu de données
`movement_pool`) ne vit PAS ici mais dans `model/core/terminal_jump.py` : il
répond à une question — « de combien l'opinion peut-elle encore bouger d'ici
au scrutin ? » — indépendante de la façon dont on estime l'opinion courante,
et `linear-pooling` s'en sert autant que ce modèle. Ré-exporté ci-dessous pour
que `model.models.bayesian_nowcast.<nom>` continue de fonctionner (backtest,
tests), avec une seule source de vérité.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from model.core.bank import Bank
from model.core.inference import run_numpyro_mcmc
from model.core.terminal_jump import (  # noqa: F401  (ré-export, cf. docstring)
    DEFAULT_JUMP_PRIORS_CFG, TerminalJumpCalibration, institut_bias_prior, movement_pool,
    terminal_jump_model,
)
from model.core.utils import SinhArcsinh, clr
from pipeline.historical import load_calibration_frame, load_results

BANK_PATH = Path(__file__).parent / "bank.json"
JUMP_BANK_PATH = Path(__file__).parent / "bank_jump.json"


def horizon_diffusion(horizon_days):
    """LA transformation de diffusion de CE modèle : sqrt(h), nulle à J-0
    (cohérente avec une marche aléatoire de l'opinion, variance ∝ temps).
    Définie une seule fois, réutilisée partout où ce modèle a besoin de
    l'horizon transformé (build_model NumPyro, sigmas dérivées hors-MCMC) —
    un autre modèle est libre de choisir log1p, sinh-arcsinh calibré, etc.,
    rien dans le framework n'impose une forme. Marche aussi bien sur des
    tableaux JAX que NumPy/Python (jnp.sqrt accepte les deux)."""
    return jnp.sqrt(horizon_days)


@dataclass
class HouseEffectsData:
    n_instituts: int
    n_blocs: int
    n_elections: int
    institut_idx: jnp.ndarray
    bloc_idx: jnp.ndarray
    election_idx: jnp.ndarray
    horizon: jnp.ndarray            # BRUT (jours), transformé dans build_model
    design_effect: float
    sampling_variance: jnp.ndarray
    observed_errors: jnp.ndarray
    institut_names: list[str]
    bloc_names: list[str]
    election_labels: list[int]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, deff: float = 1.0) -> "HouseEffectsData":
        """Encode un DataFrame calibration (institut, bloc, election, horizon,
        intention, echantillon, ecart) en tableaux JAX indexés."""
        institut_codes, institut_names = pd.factorize(df["institut"], sort=True)
        bloc_codes, bloc_names = pd.factorize(df["bloc"], sort=True)
        elec_codes, elec_names = pd.factorize(df["election"], sort=True)

        p = df["intention"].to_numpy(dtype=float)
        n = df["echantillon"].to_numpy(dtype=float)
        n_med = np.nanmedian(n[n > 0])
        n = np.where((~np.isfinite(n)) | (n <= 0), n_med, n)
        sampling_variance = p * (100.0 - p) / n

        return cls(
            n_instituts=len(institut_names), n_blocs=len(bloc_names),
            n_elections=len(elec_names),
            institut_idx=jnp.array(institut_codes), bloc_idx=jnp.array(bloc_codes),
            election_idx=jnp.array(elec_codes),
            horizon=jnp.array(df["horizon"].to_numpy(dtype=float)),
            design_effect=deff,
            sampling_variance=jnp.array(sampling_variance),
            observed_errors=jnp.array(df["ecart"].to_numpy(dtype=float)),
            institut_names=list(institut_names), bloc_names=list(bloc_names),
            election_labels=[int(e) for e in elec_names],
        )


# Hyperparamètres des priors (échelles) — surchargeables via `priors_cfg`
# (cf. HouseEffectsCalibration.priors_cfg), pour l'analyse de sensibilité
# (model/backtest/prior_sensitivity.py). Valeurs par défaut = config de référence.
DEFAULT_PRIORS_CFG = {
    "bloc_bias_mean_sd": 5.0, "bloc_bias_sd_sd": 3.0, "campaign_drift_sd_sd": 1.0,
    "excess_log_scale_mu": float(jnp.log(2.0)), "excess_log_scale_sd": 1.0,
    "excess_institut_sd_sd": 0.5, "excess_horizon_slope_sd": 1.0,
}


def house_effects_model(data: HouseEffectsData, priors_cfg: dict | None = None):
    """NumPyro pur. ecart_i ~ Normal(mu_i, sigma_i) avec :

        mu_i = institut_bias[institut,bloc] + campaign_drift[election,bloc] * base(horizon)
        var_i = deff * sampling_variance_i + excess_i^2
        excess_i = exp(excess_log_scale + excess_institut_offset[institut]
                       + excess_horizon_slope * z(horizon))

    Partial pooling non centré sur le biais (transférable entre élections) et
    sur la dérive (spécifique à chaque campagne, moyenne nulle — jamais
    calibrée sur une élection en cours)."""
    cfg = {**DEFAULT_PRIORS_CFG, **(priors_cfg or {})}
    n_bloc, n_institut, n_election = data.n_blocs, data.n_instituts, data.n_elections

    base = horizon_diffusion(data.horizon)
    h_mean, h_std = base.mean(), base.std()
    numpyro.deterministic("horizon_scale_mean", h_mean)
    numpyro.deterministic("horizon_scale_std", h_std)
    h_z = (base - h_mean) / h_std

    bloc_bias_mean = numpyro.sample("bloc_bias_mean",
                                    dist.Normal(0.0, cfg["bloc_bias_mean_sd"]).expand([n_bloc]))
    bloc_bias_sd = numpyro.sample("bloc_bias_sd",
                                  dist.HalfNormal(cfg["bloc_bias_sd_sd"]).expand([n_bloc]))
    institut_bias_z = numpyro.sample("institut_bias_z",
                                     dist.Normal(0.0, 1.0).expand([n_institut, n_bloc]))
    institut_bias = numpyro.deterministic(
        "institut_bias", bloc_bias_mean[None, :] + institut_bias_z * bloc_bias_sd[None, :])

    campaign_drift_sd = numpyro.sample("campaign_drift_sd", dist.HalfNormal(cfg["campaign_drift_sd_sd"]))
    campaign_drift_z = numpyro.sample("campaign_drift_z",
                                      dist.Normal(0.0, 1.0).expand([n_election, n_bloc]))
    campaign_drift = numpyro.deterministic("campaign_drift", campaign_drift_z * campaign_drift_sd)

    excess_log_scale = numpyro.sample("excess_log_scale",
                                      dist.Normal(cfg["excess_log_scale_mu"], cfg["excess_log_scale_sd"]))
    excess_institut_sd = numpyro.sample("excess_institut_sd", dist.HalfNormal(cfg["excess_institut_sd_sd"]))
    excess_institut_offset_z = numpyro.sample("excess_institut_offset_z",
                                              dist.Normal(0.0, 1.0).expand([n_institut]))
    excess_institut_offset = numpyro.deterministic(
        "excess_institut_offset", excess_institut_offset_z * excess_institut_sd)
    excess_horizon_slope = numpyro.sample("excess_horizon_slope",
                                          dist.Normal(0.0, cfg["excess_horizon_slope_sd"]))

    mu_i = (institut_bias[data.institut_idx, data.bloc_idx]
            + campaign_drift[data.election_idx, data.bloc_idx] * base)
    excess_i = jnp.exp(excess_log_scale + excess_institut_offset[data.institut_idx]
                       + excess_horizon_slope * h_z)
    var_i = data.design_effect * data.sampling_variance + excess_i ** 2

    numpyro.sample("y", dist.Normal(mu_i, jnp.sqrt(var_i)), obs=data.observed_errors)


class HouseEffectsCalibration:
    """Pas d'ABC : une classe normale qui compose run_numpyro_mcmc + Bank."""
    draws = 1000
    tune = 1000
    chains = 4
    target_accept = 0.9
    seed = 27
    design_effect = 1.0
    priors_cfg: dict | None = None    # surcharge de DEFAULT_PRIORS_CFG (sensibilité aux priors)

    def calibrate(self, df: pd.DataFrame) -> Bank:
        data = HouseEffectsData.from_dataframe(df, deff=self.design_effect)
        samples, _ = run_numpyro_mcmc(
            house_effects_model, {"data": data, "priors_cfg": self.priors_cfg},
            draws=self.draws, tune=self.tune, chains=self.chains,
            seed=self.seed, target_accept=self.target_accept)
        return Bank(
            samples,
            dims={"institut_bias": ["institut", "bloc"],
                 "excess_institut_offset": ["institut"],
                 "campaign_drift": ["election", "bloc"],
                 "bloc_bias_mean": ["bloc"], "bloc_bias_sd": ["bloc"]},
            coords={"institut": data.institut_names, "bloc": data.bloc_names,
                   "election": data.election_labels})


# --- Quantités dérivées (pas des sites bruts du postérieur) ----------------

def sampling_sigma(intention: float, n: int, deff: float = 1.0) -> float:
    """Écart-type d'échantillonnage : sqrt(deff * p(100-p)/n)."""
    return math.sqrt(deff * intention * (100.0 - intention) / max(int(n), 1))


def excess_sigma(bank: Bank, institut: str, horizon) -> float:
    """Écart-type d'excès (hors échantillonnage) : bruit propre à l'institut +
    volatilité croissante avec l'horizon — réutilise horizon_diffusion, la
    même transformation que celle calibrée."""
    log_scale, _ = bank.excess_log_scale.item()
    slope, _ = bank.excess_horizon_slope.item()
    h_mean, _ = bank.horizon_scale_mean.item()
    h_std, _ = bank.horizon_scale_std.item()
    try:
        offset, _ = bank.excess_institut_offset.at(institut=institut)
    except KeyError:
        offset = 0.0
    z = (float(horizon_diffusion(horizon)) - h_mean) / h_std
    return math.exp(log_scale + offset + slope * z)


def forecast_drift_sigma(bank: Bank, horizon) -> float:
    """Incertitude a priori sur la dérive d'ici au scrutin à l'horizon h :
    campaign_drift_sd * horizon_diffusion(h) — nulle à J-0, croissante avec
    l'horizon."""
    tau, _ = bank.campaign_drift_sd.item()
    return tau * float(horizon_diffusion(horizon))


def measurement_sigma(bank: Bank, institut: str, intention: float, n: int) -> float:
    """Bruit de MESURE du nowcast : échantillonnage + plancher house-effect de
    l'institut à horizon 0 (pas de croissance en horizon — la dérive jusqu'au
    scrutin appartient à la prévision, pas à la mesure de l'opinion courante)."""
    return math.hypot(sampling_sigma(intention, n), excess_sigma(bank, institut, 0))


def main() -> None:
    """Calibre le modèle et exporte la Bank. `.venv/bin/python -m model.models.bayesian_nowcast`."""
    from numpyro.diagnostics import summary as numpyro_summary

    df = load_calibration_frame()
    print(f"Calibration sur {len(df)} observations "
          f"({df['institut'].nunique()} instituts × {df['bloc'].nunique()} blocs).")

    cal = HouseEffectsCalibration()
    data = HouseEffectsData.from_dataframe(df, deff=cal.design_effect)
    samples, _ = run_numpyro_mcmc(house_effects_model, {"data": data},
                                  draws=cal.draws, tune=cal.tune, chains=cal.chains,
                                  seed=cal.seed, target_accept=cal.target_accept)
    # horizon_scale_mean/std sont des constantes déterministes (fonctions des
    # données, pas des sites échantillonnés) : variance nulle -> R-hat/ESS
    # dégénérés, exclues du diagnostic de convergence.
    diag = numpyro_summary(samples, prob=0.9)
    diag = {k: v for k, v in diag.items() if k not in ("horizon_scale_mean", "horizon_scale_std")}
    rhat_max = max(float(np.max(d["r_hat"])) for d in diag.values())
    ess_min = min(float(np.min(d["n_eff"])) for d in diag.values())
    print(f"R-hat max : {rhat_max:.4f} | ESS min : {ess_min:.0f}")

    bank = Bank(samples,
               dims={"institut_bias": ["institut", "bloc"],
                    "excess_institut_offset": ["institut"],
                    "campaign_drift": ["election", "bloc"],
                    "bloc_bias_mean": ["bloc"], "bloc_bias_sd": ["bloc"]},
               coords={"institut": data.institut_names, "bloc": data.bloc_names,
                      "election": data.election_labels})
    bank.save(BANK_PATH)
    print(f"Bank écrite dans {BANK_PATH}")

    print("\nFit du saut terminal (sinh-arcsinh sur le movement pool 2017/2022)...")
    jump_bank = TerminalJumpCalibration().calibrate(bank)
    jump_bank.save(JUMP_BANK_PATH)
    print(f"Bank (saut terminal) écrite dans {JUMP_BANK_PATH}")
