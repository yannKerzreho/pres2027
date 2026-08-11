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
from model.core.utils import SinhArcsinh
from pipeline.historical import load_calibration_frame, load_results

from .latent import clr

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


def institut_bias_prior(bank: Bank | None, institut: str, bloc: str) -> tuple[float, float]:
    """(mean, sd) du biais institut×bloc. Repli sur la moyenne de population du
    bloc si l'institut est inconnu (shrinkage) ; prior faible si pas de Bank."""
    if bank is None:
        return 0.0, 5.0
    try:
        return bank.institut_bias.at(institut=institut, bloc=bloc)
    except KeyError:
        return bank.bloc_bias_mean.at(bloc=bloc)


def movement_pool(bank: Bank, hmin: float = 45.0) -> pd.DataFrame:
    """Mouvements RÉELS d'opinion 2017/2022, en espace log-ratio (CLR),
    par candidat×élection×fenêtre d'horizon — jeu de données du fit
    paramétrique du saut terminal (TerminalJumpCalibration, sinh-arcsinh,
    docs/spec_ssm_nowcast.md §2.2). Débiaise chaque sondage historique du
    biais calibré, moyenne les sondages proches (même candidat, fenêtre
    d'horizon ≥ hmin) pour annuler le bruit d'échantillon, et calcule le
    mouvement `clr(résultat) − clr(sondage_débiaisé)`. Bloc ET horizon (moyen
    de la fenêtre) conservés par mouvement : loc/scale du saut sont fittés par
    bloc (§7), et TerminalJumpCalibration normalise par horizon avant de
    fitter (un mouvement mesuré à J-400 et un mouvement à J-50 ne sont PAS la
    même quantité — cf. sa docstring)."""
    df = load_calibration_frame()
    df["bias"] = [institut_bias_prior(bank, r.institut, r.bloc)[0] for r in df.itertuples()]
    df["deb"] = df["intention"] - df["bias"]
    res = load_results()
    res = res[res["tour"] == "Premier tour"]

    df["hbin"] = pd.cut(df["horizon"], [0, 45, 90, 180, 400])
    moves: list[float] = []
    blocs: list[str] = []
    horizons: list[float] = []
    for elec in df["election"].unique():
        de = df[df["election"] == elec]
        cand = sorted(de["candidat"].unique())
        bloc_of = de.drop_duplicates("candidat").set_index("candidat")["bloc"]
        rmap = dict(zip(res.loc[res.election == elec, "candidat"],
                        res.loc[res.election == elec, "resultat"]))
        r = np.array([rmap.get(c, np.nan) for c in cand])
        for _, grp in de.groupby("hbin", observed=True):
            h = grp["horizon"].mean()
            if h < hmin:
                continue
            smap = grp.groupby("candidat")["deb"].mean()
            s = np.array([smap.get(c, np.nan) for c in cand])
            mask = ~np.isnan(s) & ~np.isnan(r) & (s > 0)
            if mask.sum() < 4:
                continue
            cand_kept = [c for c, keep in zip(cand, mask) if keep]
            moves.extend((clr(r[mask]) - clr(s[mask])).tolist())
            blocs.extend(bloc_of.loc[cand_kept].tolist())
            horizons.extend([float(h)] * int(mask.sum()))
    return pd.DataFrame({"bloc": blocs, "move": moves, "horizon": horizons})


# Hyperparamètres des priors du fit sinh-arcsinh (saut terminal).
DEFAULT_JUMP_PRIORS_CFG = {
    "loc_mean_sd": 5.0, "loc_sd_sd": 3.0,
    "log_scale_mu": float(np.log(3.0)), "log_scale_sd": 1.0,
    "skew_sd": 1.0, "tail_log_sd": 0.5,
}


def terminal_jump_model(bloc_idx: jnp.ndarray, move_at_href: jnp.ndarray, n_bloc: int,
                        horizon_ref: float, priors_cfg: dict | None = None):
    """NumPyro pur : move_at_href_i ~ SinhArcsinh(loc[bloc_i], scale[bloc_i], skew, tail)
    (TFP, substrate JAX — cf. docs/spec_ssm_nowcast.md §2.2 ; `tailweight` de
    TFP joue le rôle de `tail`, convention multiplicative légèrement différente
    du pseudo-code de la spec mais même interprétation : 1 = pas de déformation
    par rapport à une gaussienne).

    `move_at_href` : mouvements RENORMALISÉS à l'horizon de référence
    `horizon_ref` AVANT ce modèle (cf. TerminalJumpCalibration.calibrate) — un
    mouvement mesuré à J-400 et un mouvement à J-50 ne sont pas la même
    quantité, la dérive continue jusqu'au scrutin (loi en sqrt(horizon), même
    `horizon_diffusion` que `campaign_drift` ailleurs dans ce module) ; le fit
    lui-même reste sur l'horizon de référence, seul le TIRAGE à `forecast()`
    est rescalé par horizon (`model/core/simulate.py`). `horizon_ref` est
    stocké tel quel dans la Bank (constante, comme `horizon_scale_mean` dans
    house_effects_model) pour que ce rescaling reste cohérent sans dupliquer
    la valeur en dur ailleurs.

    skew/tail GLOBAUX (spec_ssm_nowcast.md §7 : seulement 2 élections
    historiques, un ajustement par bloc de 4 paramètres de forme risquerait de
    capturer les particularités de CES deux campagnes plutôt que des
    fondamentaux généralisables). `scale` GLOBAL aussi, pour la même raison
    (décidé le 2026-08-09, cf. session de conception) : avec seulement 2
    élections, le partial pooling hiérarchique par bloc n'a aucun moyen de
    distinguer une vraie hétérogénéité de volatilité entre blocs politiques
    d'un simple écart d'échantillon entre CES deux campagnes précises — observé
    concrètement sur `droite_radicale` (0.55) vs `centre` (0.30, quasi 2x) à
    partir de n=2, qui produisait une inversion de `p_qualifie_top2` non
    justifiée (RN sous Horizons au 2026-03-26 malgré une intention nettement
    supérieure). `loc` reste par bloc (partial pooling non centré, même
    logique que bloc_bias_mean/excess_log_scale dans house_effects_model) : la
    direction moyenne du saut (qui dérive vers le haut/bas d'ici au scrutin)
    est mieux identifiée que son amplitude à si peu de données."""
    cfg = {**DEFAULT_JUMP_PRIORS_CFG, **(priors_cfg or {})}
    numpyro.deterministic("jump_horizon_ref", horizon_ref)

    loc_mean = numpyro.sample("jump_loc_mean", dist.Normal(0.0, cfg["loc_mean_sd"]))
    loc_sd = numpyro.sample("jump_loc_sd", dist.HalfNormal(cfg["loc_sd_sd"]))
    loc_z = numpyro.sample("jump_loc_z", dist.Normal(0.0, 1.0).expand([n_bloc]))
    loc = numpyro.deterministic("jump_loc", loc_mean + loc_z * loc_sd)

    log_scale = numpyro.sample("jump_log_scale", dist.Normal(cfg["log_scale_mu"], cfg["log_scale_sd"]))
    # broadcast en (n_bloc,) : garde le schéma Bank (dims=["bloc"]) et les
    # lookups `.at(bloc=...)` de model/core/simulate.py inchangés, avec la
    # MÊME valeur pour chaque bloc.
    scale = numpyro.deterministic("jump_scale", jnp.exp(log_scale) * jnp.ones(n_bloc))

    skew = numpyro.sample("jump_skew", dist.Normal(0.0, cfg["skew_sd"]))
    tail = numpyro.sample("jump_tail", dist.LogNormal(0.0, cfg["tail_log_sd"]))

    d = SinhArcsinh(loc=loc[bloc_idx], scale=scale[bloc_idx], skewness=skew, tailweight=tail)
    numpyro.sample("move_obs", d, obs=move_at_href)


class TerminalJumpCalibration:
    """Fit paramétrique du saut terminal (nowcast -> jour du scrutin) : pas
    d'ABC, une classe normale qui compose run_numpyro_mcmc + Bank — même
    pattern que HouseEffectsCalibration, jeu de données et Bank séparés (deux
    modèles NumPyro distincts, pas de raison de les forcer dans un seul run).

    Dépendance à l'horizon : chaque mouvement historique est mesuré à un
    horizon différent (fenêtres 45-90/90-180/180-400 jours) — les renormaliser
    à un horizon de référence commun (`horizon_ref`, moyenne du pool) avant de
    fitter rend le fit interne cohérent ; `forecast_from_draws`
    (model/core/simulate.py) rescale ensuite le TIRAGE à l'horizon réel de la
    prévision, dans le même sens (sqrt(horizon)) que `campaign_drift`."""
    draws = 1000
    tune = 1000
    chains = 4
    target_accept = 0.9
    seed = 27
    hmin = 45.0
    priors_cfg: dict | None = None

    def calibrate(self, bank: Bank) -> Bank:
        moves = movement_pool(bank, hmin=self.hmin)
        horizon_ref = float(moves["horizon"].mean())
        scale_to_ref = np.sqrt(horizon_ref / moves["horizon"].to_numpy())
        move_at_href = moves["move"].to_numpy() * scale_to_ref

        bloc_codes, bloc_names = pd.factorize(moves["bloc"], sort=True)
        data = {"bloc_idx": jnp.array(bloc_codes), "move_at_href": jnp.array(move_at_href),
                "n_bloc": len(bloc_names), "horizon_ref": horizon_ref, "priors_cfg": self.priors_cfg}
        samples, _ = run_numpyro_mcmc(
            terminal_jump_model, data, draws=self.draws, tune=self.tune, chains=self.chains,
            seed=self.seed, target_accept=self.target_accept)
        return Bank(samples, dims={"jump_loc": ["bloc"], "jump_scale": ["bloc"]},
                   coords={"bloc": list(bloc_names)})


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
