"""Petits utilitaires mathématiques partagés du projet.

Ce module reste volontairement léger : fonctions JAX pures et distributions
NumPyro réutilisables sans dépendre de bibliothèques externes fragiles côté
versions (ex. TFP substrate JAX).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes


def sinh_arcsinh_standard(z, skewness, tailweight):
    """Déformation sinh-arcsinh d'une normale standard.

    Convention identique à la littérature et à l'ancienne utilisation TFP :
        x = sinh((asinh(z) + skewness) / tailweight)
    avec `tailweight > 0`.
    """
    return jnp.sinh((jnp.arcsinh(z) + skewness) / tailweight)


def sinh_arcsinh_loc_scale(z, loc, scale, skewness, tailweight):
    """Version affine de `sinh_arcsinh_standard`."""
    return loc + scale * sinh_arcsinh_standard(z, skewness, tailweight)


class SinhArcsinh(dist.Distribution):
    """Distribution sinh-arcsinh paramétrée par `loc`, `scale`, `skewness`, `tailweight`.

    Construction par transformation monotone d'une normale standard :
        z ~ Normal(0, 1)
        x = loc + scale * sinh((asinh(z) + skewness) / tailweight)
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "skewness": constraints.real,
        "tailweight": constraints.positive,
    }
    support = constraints.real
    reparametrized_params = ["loc", "scale", "skewness", "tailweight"]

    def __init__(self, loc=0.0, scale=1.0, skewness=0.0, tailweight=1.0, *, validate_args=None):
        self.loc, self.scale, self.skewness, self.tailweight = promote_shapes(
            loc, scale, skewness, tailweight
        )
        batch_shape = jnp.broadcast_shapes(
            jnp.shape(self.loc),
            jnp.shape(self.scale),
            jnp.shape(self.skewness),
            jnp.shape(self.tailweight),
        )
        super().__init__(batch_shape=batch_shape, event_shape=(), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        z = jr.normal(key, shape=shape)
        return sinh_arcsinh_loc_scale(
            z, self.loc, self.scale, self.skewness, self.tailweight
        )

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        w = (value - self.loc) / self.scale
        z = jnp.sinh(self.tailweight * jnp.arcsinh(w) - self.skewness)
        log_dz_dx = (
            jnp.log(self.tailweight)
            + jnp.log(jnp.cosh(self.tailweight * jnp.arcsinh(w) - self.skewness))
            - jnp.log(self.scale)
            - 0.5 * jnp.log1p(w ** 2)
        )
        return dist.Normal(0.0, 1.0).log_prob(z) + log_dz_dx
