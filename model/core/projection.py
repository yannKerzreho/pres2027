"""Projection des tirages d'un nowcast jusqu'au jour du scrutin.

Machinerie **partagée** : elle répond à une question — « que devient une
estimation de l'opinion d'aujourd'hui le jour du vote ? » — que tout modèle de
suivi doit trancher, quelle que soit la façon dont il estime l'opinion courante.
Elle vit donc dans `core`, pas dans un modèle : loger une loi partagée à
l'intérieur d'un modèle est précisément ce qui a cassé la branche `main` quand
ce modèle a été retiré.

Deux quantités **distinctes**, à ne pas confondre :

1. la **diffusion d'opinion** entre `as_of` et J-0, de variance
   `2σ²(1−e^(−h/τ))` — un Ornstein-Uhlenbeck, calibré sur le mouvement
   *sondage à sondage* ;
2. l'**écart de traduction** δ entre une intention de vote mesurée et un
   bulletin déposé (participation différentielle, indécis, vote utile), qui ne
   dépend **pas** de l'horizon.

L'ancien saut terminal ajustait une seule loi sur la somme des deux, avec une
forme nulle en `h = 0` qui interdisait structurellement à δ d'exister et forçait
`τ` à rabougrir. Mesuré : à J-14, `sondage → résultat` vaut déjà 0,111 quand
`sondage → sondage` sur la même durée ne vaut que 0,018.

Ce module ne calibre rien : il applique. Les paramètres viennent de la banque
commune `model/core/opinion.py` (`load_law()`), que tous les modèles partagent.
"""

from __future__ import annotations

import jax
import numpy as np

from model.core.utils import SinhArcsinh


def diffusion_var(horizon_days, sigma2: float, tau: float):
    """Variance de la diffusion d'opinion sur `horizon_days` jours :
    `2σ²(1−e^(−h/τ))`. Nulle à J-0, bornée par `2σ²` — la dispersion sature au
    lieu de croître indéfiniment comme le ferait une marche aléatoire."""
    h = np.clip(np.asarray(horizon_days, dtype=float), 0.0, None)
    return 2.0 * sigma2 * (1.0 - np.exp(-h / tau))


def delta_draws(law: dict, size: tuple[int, int], seed: int = 27) -> np.ndarray:
    """Tirages `(S, K)` de l'écart sondages-urne δ, en log-ratio, indépendants
    entre candidats. Loi sinh-arcsinh : asymétrique et à queues ajustables,
    parce que les mouvements historiques le sont — une gaussienne
    sous-estimerait les surprises et produirait des probabilités de
    qualification surconfiantes."""
    d = SinhArcsinh(loc=0.0, scale=law["delta_scale"], skewness=law["delta_skew"],
                    tailweight=law["delta_tail"])
    return np.asarray(d.sample(jax.random.PRNGKey(seed), sample_shape=size))


def projeter_au_scrutin(pi: np.ndarray, horizon_days: int, law: dict,
                        seed: int = 27) -> np.ndarray:
    """Tirages `(S, K)` des parts AU SCRUTIN à partir des tirages du nowcast.

    Enchaîne les deux mécanismes du docstring du module, en espace log-ratio
    (softmax final → simplexe garanti). Un modèle qui a déjà projeté son état
    jusqu'à J-0 par ses propres moyens n'appelle que `delta_draws`.
    """
    rng = np.random.default_rng(seed)
    alpha = np.log(np.clip(pi, 1e-9, None))
    sd = np.sqrt(diffusion_var(max(horizon_days, 0), law["sigma2"], law["tau"]))
    alpha = alpha + rng.normal(0.0, sd, size=alpha.shape)
    alpha = alpha + delta_draws(law, alpha.shape, seed + 1)
    e = np.exp(alpha - alpha.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
