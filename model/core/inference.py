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
                     extra_fields: tuple[str, ...] = (), init_strategy=init_to_median,
                     chain_method: str = "sequential",
                     dense_mass: bool = False) -> tuple[dict, dict]:
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

    `chain_method` reste `"sequential"` par défaut : c'est le comportement
    historique de tous les modèles du dépôt, et le changer globalement
    modifierait leurs temps et leurs empreintes mémoire sans qu'on l'ait
    demandé. `"vectorized"` exécute les chaînes en un seul `scan` vmappé, au
    prix d'une empreinte mémoire multipliée par le nombre de chaînes.

    Attention à deux choses mesurées sur `spatial_pooling` (spec §12.22) :

    - ce n'est PAS plus rapide sur CPU (377 s contre 378 s sur le roster 2027).
      L'intuition « un produit matriciel groupé au lieu de N séparés » ne se
      vérifie pas ici, le coût étant déjà dominé par un grand tenseur par pas ;
    - ça change les VERDICTS de convergence : mêmes données, même graine,
      R-hat 2,417 en `"sequential"` contre 1,032 en `"vectorized"`. Un
      diagnostic de non-convergence n'est donc pas comparable d'un
      `chain_method` à l'autre — le noter avec le chiffre.

    Retourne `(samples, extra)` : `samples` est le dict `{site: array (chains,
    draws, ...)}` du postérieur (chaînes séparées — nécessaire pour R-hat/ESS
    et pour construire une `Bank` avec dims `("chain", "draw", ...)`), `extra`
    les champs demandés dans `extra_fields` (ex. `"diverging"`).

    `dense_mass` (défaut `False`, comportement NumPyro d'origine) : estime une
    matrice de masse PLEINE au lieu d'une diagonale. À essayer quand le nombre
    de pas de leapfrog colle à `2^k - 1` presque à chaque itération (255, 511) :
    ça veut dire que la trajectoire ne fait jamais demi-tour, ce qui est la
    signature d'un postérieur corrélé qu'une masse diagonale ne peut pas
    précondionner. Coût : `dim²` en mémoire et une covariance à estimer pendant
    le warmup — inutilisable si `dim` dépasse quelques centaines.
    """
    kernel = NUTS(model_fn, target_accept_prob=target_accept, init_strategy=init_strategy,
                  dense_mass=dense_mass)
    mcmc = MCMC(kernel, num_warmup=tune, num_samples=draws, num_chains=chains,
                chain_method=chain_method, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), **model_kwargs, extra_fields=extra_fields)
    extra = {f: mcmc.get_extra_fields()[f] for f in extra_fields}
    return mcmc.get_samples(group_by_chain=True), extra
