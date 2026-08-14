"""Filtre de Kalman linéaire-gaussien, à paramètres variables dans le temps.

Remplace `dynamax.linear_gaussian_ssm.inference.lgssm_filter`, dont c'était le
SEUL usage du dépôt. Le motif justifiant la dépendance a disparu : dynamax tire
tensorflow-probability, dont la dernière version (0.25) casse à partir de
jax 0.7, alors que numpyro >= 0.20 EXIGE jax >= 0.7 — deux bornes inconciliables
qui enfermaient le nowcast SSM dans un environnement figé (jax 0.4, numpyro
0.19) pendant que le reste du dépôt avançait. Cinquante lignes de `lax.scan`
valaient mieux que ce blocage.

Conventions reprises telles quelles de dynamax, pour que la substitution soit
numériquement neutre (cf. `tests/test_lgssm.py`, qui compare à la marginale
gaussienne exacte) :

  - `init_mean` / `init_cov` décrivent l'état AVANT toute observation, c'est-à-
    dire la loi prédictive en t=0 — surtout pas l'état filtré.
  - à l'instant t : on accumule log p(y_t | y_{<t}), on conditionne sur y_t,
    PUIS on propage vers t+1. `F[t]` et `Q[t]` sont donc la transition t -> t+1,
    et le dernier pas de transition est calculé mais jamais lu (l'appelant
    extrapole lui-même au-delà du dernier nœud).
  - `filtered_means[t]` est la moyenne a posteriori en t sachant y_0..y_t.

Chaque paramètre s'accepte statique ou daté : un tableau qui porte un axe temps
supplémentaire est indexé par t, sinon il est réutilisé à chaque pas.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular

# Même amorce diagonale que `dynamax.utils.psd_solve` : le gain de Kalman passe
# par une factorisation de Cholesky de S, qui devient singulière quand deux
# sondages du même jour portent une information quasi identique.
_DIAG_BOOST = 1e-9

_LOG_2PI = float(jnp.log(2.0 * jnp.pi))


class KalmanFilterResult(NamedTuple):
    """Champs nommés comme ceux de `PosteriorGSSMFiltered` (dynamax)."""

    marginal_loglik: jnp.ndarray
    filtered_means: jnp.ndarray
    filtered_covariances: jnp.ndarray


def _symmetrize(A):
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def _at(x, t, ndim: int):
    """`x[t]` si `x` porte un axe temps en plus de `ndim`, `x` sinon."""
    return x[t] if jnp.ndim(x) == ndim + 1 else x


def _psd_solve(A, b):
    A = _symmetrize(A) + _DIAG_BOOST * jnp.eye(A.shape[-1])
    return cho_solve(cho_factor(A, lower=True), b)


def _mvn_logpdf(y, mean, cov):
    """log N(y | mean, cov) par Cholesky — pas d'amorce diagonale ici : cette
    covariance-là contient déjà R, le bruit de mesure, donc elle est
    strictement définie positive dès qu'un sondage a une taille finie."""
    L = jnp.linalg.cholesky(cov)
    z = solve_triangular(L, y - mean, lower=True)
    return -0.5 * (z @ z + y.shape[-1] * _LOG_2PI) - jnp.sum(jnp.log(jnp.diag(L)))


def kalman_filter(init_mean, init_cov, F, Q, H, R, emissions) -> KalmanFilterResult:
    """Filtre un modèle d'état linéaire-gaussien sans biais ni entrée exogène.

        z_0     ~ N(init_mean, init_cov)
        z_{t+1} = F[t] z_t + w_t,   w_t ~ N(0, Q[t])
        y_t     = H[t] z_t + v_t,   v_t ~ N(0, R[t])

    `F`/`Q` sont des matrices (D, D) ou des piles (T, D, D) ; `H` est (K, D) ou
    (T, K, D) ; `R` est (K, K) ou (T, K, K) ; `emissions` est (T, K).

    Renvoie la log-vraisemblance marginale des observations et la trajectoire
    filtrée.
    """
    # Conversion explicite : indexer un `np.ndarray` par le compteur de `scan`
    # (un traceur) lève `TracerArrayConversionError`. L'appelant a le droit de
    # passer du numpy — c'est le cas des tests et de tout appel hors modèle.
    init_mean, init_cov = jnp.asarray(init_mean), jnp.asarray(init_cov)
    F, Q, H, R = (jnp.asarray(a) for a in (F, Q, H, R))
    emissions = jnp.asarray(emissions)
    n_steps = emissions.shape[0]

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry
        F_t, Q_t = _at(F, t, 2), _at(Q, t, 2)
        H_t, R_t = _at(H, t, 2), _at(R, t, 2)
        y = emissions[t]

        # Innovation : covariance de y_t sachant y_{<t}.
        S = R_t + H_t @ pred_cov @ H_t.T
        ll = ll + _mvn_logpdf(y, H_t @ pred_mean, S)

        # Conditionnement sur y_t. La forme K S K' (plutôt que (I - KH) P) est
        # celle de dynamax : elle garde la covariance symétrique par
        # construction, ce que `_symmetrize` finit de garantir contre la dérive
        # d'arrondi accumulée sur une longue série.
        K = _psd_solve(S, H_t @ pred_cov).T
        filt_mean = pred_mean + K @ (y - H_t @ pred_mean)
        filt_cov = _symmetrize(pred_cov - K @ S @ K.T)

        # Propagation vers t+1.
        return (ll, F_t @ filt_mean, F_t @ filt_cov @ F_t.T + Q_t), (filt_mean, filt_cov)

    carry = (jnp.zeros(()), init_mean, init_cov)
    (ll, _, _), (means, covs) = lax.scan(_step, carry, jnp.arange(n_steps))
    return KalmanFilterResult(marginal_loglik=ll, filtered_means=means, filtered_covariances=covs)
