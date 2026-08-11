"""Moteur d'inférence NumPyro/JAX partagé, par composition (pas d'héritage).

N'importe quelle fonction NumPyro pure + son dict de données suffit à
l'utiliser — aucune classe de base à sous-classer. Un modèle (calibration ou
nowcast) l'appelle depuis sa propre méthode quand il en a besoin.
"""

from __future__ import annotations

import jax
from numpyro.infer import MCMC, NUTS, init_to_median


def run_numpyro_mcmc(model_fn, model_kwargs: dict, draws: int = 1000, tune: int = 1000,
                     chains: int = 4, seed: int = 27, target_accept: float = 0.9,
                     extra_fields: tuple[str, ...] = (), init_strategy=init_to_median
                     ) -> tuple[dict, dict]:
    """Lance NUTS sur `model_fn(**model_kwargs)`.

    `init_strategy=init_to_median` (défaut NumPyro : `init_to_uniform`, un
    point tiré uniformément dans l'espace non contraint) : pour un paramètre
    d'échelle (HalfNormal — `tau`, les `*_sd` de `house_effects_model`...),
    `init_to_uniform` peut occasionnellement initialiser une chaîne TRÈS loin
    dans la queue, dans une région où le gradient de la vraisemblance
    (notamment à travers un filtre de Kalman, cf. un modèle à état latent)
    s'annule numériquement — la chaîne reste alors bloquée EXACTEMENT à sa
    valeur d'initialisation tout le run (observé concrètement : 2 chaînes sur
    4 figées à tau=5.09/3.15, écart-type 0.0000, sur le nowcast 2026-07-08).
    `init_to_median` démarre au médian du prior — évite cette queue par
    construction, sans changer la géométrie explorée une fois démarré.

    Retourne `(samples, extra)` : `samples` est le dict `{site: array (chains,
    draws, ...)}` du postérieur (chaînes séparées — nécessaire pour R-hat/ESS
    et pour construire une `Bank` avec dims `("chain", "draw", ...)`), `extra`
    les champs demandés dans `extra_fields` (ex. `"diverging"`)."""
    kernel = NUTS(model_fn, target_accept_prob=target_accept, init_strategy=init_strategy)
    mcmc = MCMC(kernel, num_warmup=tune, num_samples=draws, num_chains=chains,
                chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), **model_kwargs, extra_fields=extra_fields)
    extra = {f: mcmc.get_extra_fields()[f] for f in extra_fields}
    return mcmc.get_samples(group_by_chain=True), extra
