"""Écart terminal δ : de l'opinion du jour du scrutin au résultat des urnes.

Sépare deux choses que le saut terminal historique confondait :

- la **diffusion d'opinion** entre la date d'un sondage et le jour J, qui grandit
  avec l'horizon — c'est le noyau OU du GP (`calibration.py`) ;
- l'**écart de traduction** δ entre une intention de vote mesurée et un bulletin
  déposé (abstention différentielle, indécis, vote utile), qui ne grandit PAS
  avec l'horizon.

La version précédente ajustait une seule loi `scale²·(1−e^(−h/τ))` sur la somme
des deux. Cette forme vaut **zéro en h=0** : elle interdisait structurellement à
δ d'exister et forçait `τ` à rabougrir pour compenser. Mesuré : à J-14,
`sondage → résultat` vaut déjà 0,111 alors que `sondage → sondage` sur 14 jours
ne vaut que 0,018.

Ici : `résultat = θ(0) + δ`, où θ(0) vient du postérieur GP prolongé jusqu'à J-0
et δ est ajusté comme le **résidu**, à diffusion FIXÉE par le noyau du GP. Chaque
mécanisme est estimé sur la quantité qui le mesure.

Limite assumée : le pool de mouvements ne descend pas sous 59 jours d'horizon,
donc l'ordonnée à l'origine n'y est pas directement identifiable (à diffusion
libre, les deux structures s'ajustent aussi bien). Ce qui départage n'est pas la
vraisemblance sur le pool mais la couverture au scrutin.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import numpyro
import numpyro.distributions as dist

from model.core.bank import Bank
from model.core.inference import run_numpyro_mcmc
from model.core.movements import movement_pool
from model.core.utils import SinhArcsinh
from model.models.gp_pooling.calibration import load_params

BANK_PATH = Path(__file__).parent / "bank_terminal.json"

DEFAULT_PRIORS = {
    "log_scale_mu": float(np.log(0.4)), "log_scale_sd": 1.0,
    "skew_sd": 1.0, "tail_log_sd": 0.5,
}


def diffusion_var(horizon, sigma2: float, tau: float):
    """Variance de la diffusion d'opinion sur `horizon` jours : `2σ²(1−e^(−h/τ))`."""
    h = np.clip(np.asarray(horizon, dtype=float), 0.0, None)
    return 2.0 * sigma2 * (1.0 - np.exp(-h / tau))


def terminal_model(move, var_diff, priors_cfg: dict | None = None):
    """`move_i ~ SinhArcsinh(0, √(var_diff_i + scale_δ²), skew, tail)`.

    `var_diff` est **fixé** (noyau du GP, calibré sur le mouvement sondage à
    sondage) : le seul paramètre d'échelle libre est celui de δ. C'est ce qui
    fait de δ un résidu et non un fourre-tout.

    Moyenne nulle : les mouvements sont en log-ratio centré, donc leur moyenne
    est exactement 0 par construction.
    `skew`/`tail` restent libres — le pool est asymétrique et à queues épaisses,
    et c'est précisément ce qui évite des probabilités de qualification
    surconfiantes.
    """
    import jax.numpy as jnp
    cfg = {**DEFAULT_PRIORS, **(priors_cfg or {})}
    log_scale = numpyro.sample("delta_log_scale",
                               dist.Normal(cfg["log_scale_mu"], cfg["log_scale_sd"]))
    scale_d = numpyro.deterministic("delta_scale", jnp.exp(log_scale))
    skew = numpyro.sample("delta_skew", dist.Normal(0.0, cfg["skew_sd"]))
    tail = numpyro.sample("delta_tail", dist.LogNormal(0.0, cfg["tail_log_sd"]))
    s = jnp.sqrt(jnp.asarray(var_diff) + scale_d ** 2)
    numpyro.sample("move_obs", SinhArcsinh(loc=0.0, scale=s, skewness=skew, tailweight=tail),
                   obs=jnp.asarray(move))


class TerminalGapCalibration:
    """Ajuste δ sur le pool de mouvements, diffusion fixée par le noyau du GP."""
    draws = 1000
    tune = 1000
    chains = 4
    target_accept = 0.9
    seed = 27
    hmin = 45.0

    def calibrate(self) -> Bank:
        p = load_params()
        moves = movement_pool(None, hmin=self.hmin)
        var_diff = diffusion_var(moves["horizon"].to_numpy(), p["sigma2"], p["tau"])
        samples, _ = run_numpyro_mcmc(
            terminal_model, {"move": moves["move"].to_numpy(), "var_diff": var_diff},
            draws=self.draws, tune=self.tune, chains=self.chains,
            seed=self.seed, target_accept=self.target_accept)
        return Bank(samples, dims={}, coords={})


def delta_draws(bank: Bank, size: tuple[int, int], seed: int = 27) -> np.ndarray:
    """Tirages `(S, K)` de δ en log-ratio, indépendants entre candidats."""
    scale, _ = bank.delta_scale.item()
    skew, _ = bank.delta_skew.item()
    tail, _ = bank.delta_tail.item()
    d = SinhArcsinh(loc=0.0, scale=scale, skewness=skew, tailweight=tail)
    return np.asarray(d.sample(jax.random.PRNGKey(seed), sample_shape=size))


def main() -> None:
    print("Calibration de l'écart terminal δ (diffusion fixée par le noyau GP)...")
    bank = TerminalGapCalibration().calibrate()
    bank.save(BANK_PATH)
    print(f"  delta_scale = {bank.delta_scale.item()[0]:.4f}")
    print(f"  delta_skew  = {bank.delta_skew.item()[0]:+.4f}")
    print(f"  delta_tail  = {bank.delta_tail.item()[0]:.4f}")
    print(f"Bank écrite dans {BANK_PATH}")


if __name__ == "__main__":
    main()
