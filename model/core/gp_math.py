"""Postérieur gaussien en forme close sur un état latent à noyau Ornstein-Uhlenbeck.

Cœur mathématique de `gp-pooling` (cf. spec §3.4). Aucune dépendance au reste du
dépôt : que du numpy/scipy, pour que les mesures du backtest et le modèle live
partagent EXACTEMENT le même code — un écart entre les deux invaliderait les
critères d'acceptation.

Pourquoi une forme close plutôt qu'un échantillonneur : tout est gaussien
(prior GP, house effect, approximation gaussienne du bruit d'échantillonnage),
donc le postérieur est une gaussienne dont la moyenne et la variance
s'obtiennent par une seule résolution linéaire `P×P`. Pas de NUTS, pas de
divergences, pas de R-hat à surveiller — les trois reproches faits au SSM
tombent d'eux-mêmes.

Le paramètre `mu` est **inconnu et marginalisé** avec un prior plat (krigeage
universel) : c'est ce qui donne, sans montage supplémentaire, le retour au
niveau moyen estimé et la saturation de la variance vers `sigma2` aux horizons
lointains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

# Le prior plat sur `mu` rend `Sigma` singulier dans la direction constante si
# aucune observation ne la contraint ; ce plancher (variance CLR) évite un
# Cholesky qui échoue sur un pool d'un seul sondage.
JITTER = 1e-9


def ou_kernel(h_a: np.ndarray, h_b: np.ndarray, tau: float, sigma2: float) -> np.ndarray:
    """Noyau exponentiel `sigma2 · exp(-|h_a - h_b| / tau)` — la covariance d'un
    Ornstein-Uhlenbeck stationnaire.

    C'est la MÊME hypothèse que celle qui justifie déjà
    `model/core/terminal_jump.saturation` : `Var(θ(t)-θ(s)) = 2·sigma2·(1 -
    exp(-|t-s|/tau))`. Voir `calibration.py` pour pourquoi `sigma2` ne peut
    PAS être repris tel quel du saut terminal.
    """
    return sigma2 * np.exp(-np.abs(np.asarray(h_a)[:, None] - np.asarray(h_b)[None, :]) / tau)


def clr_rows(shares: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    """CLR ligne par ligne d'une matrice `(P, K)` de compositions.

    Version matricielle de `model.core.utils.clr` (qui ne traite qu'un vecteur) :
    même définition, mêmes conventions de plancher.
    """
    v = np.clip(np.asarray(shares, dtype=float), floor, None)
    v = v / v.sum(axis=1, keepdims=True)
    lg = np.log(v)
    return lg - lg.mean(axis=1, keepdims=True)


def sampling_cov_clr_diag(shares: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Diagonale de `P diag(1/(n·y)) P` avec `P = I - 11ᵀ/K` : variance
    d'échantillonnage de chaque composante CLR, par la méthode delta sur un
    multinomial.

    Développement : `[P D P]_ss = d_s(1 - 2/K) + (1/K²)·Σ_j d_j`, `d_j = 1/(n y_j)`.

    On ne garde que la diagonale (spec §7.1) : les slots sont traités
    indépendamment, une covariance complète n'aurait nulle part où agir. Le
    terme `(1/K²)Σ_j d_j` conserve tout de même l'essentiel de l'effet
    simplexe — un petit candidat très bruité gonfle la variance de TOUS les
    autres via la contrainte de somme.
    """
    shares = np.atleast_2d(np.asarray(shares, dtype=float))
    n = np.asarray(n, dtype=float).reshape(-1, 1)
    K = shares.shape[1]
    d = 1.0 / (n * shares)
    return d * (1.0 - 2.0 / K) + d.sum(axis=1, keepdims=True) / K**2


@dataclass
class GPPosterior:
    """Sortie du postérieur, par slot (tableaux `(K,)`).

    `mean` / `var_latent` décrivent θ(h*) — l'opinion elle-même.
    `var_obs` (si un sondage cible est fourni) y ajoute le house effect et le
    bruit d'échantillonnage de CE sondage : c'est la loi de ce qu'on
    OBSERVERAIT, la seule quantité qu'un test de couverture puisse confronter à
    une donnée.
    """
    mean: np.ndarray
    var_latent: np.ndarray
    var_obs: np.ndarray | None = None
    mu_hat: np.ndarray | None = None


def gp_posterior(Z: np.ndarray, Vsamp: np.ndarray, h: np.ndarray, inst_idx: np.ndarray,
                 h_star: float, tau: float, sigma2: float, sigma_h: float,
                 target_inst: int | None = None,
                 target_v: np.ndarray | None = None) -> GPPosterior:
    """Postérieur du §3.4, calculé slot par slot (les slots sont indépendants
    a priori, cf. §3.2 ; ils ne se recouplent qu'au softmax final).

    - `Z` `(P, K)` : sondages en CLR ; `Vsamp` `(P, K)` : leurs variances
      d'échantillonnage ; `h` `(P,)` : horizons en jours avant le scrutin ;
      `inst_idx` `(P,)` : code entier de l'institut de chaque sondage.
    - `h_star` : horizon où l'on veut l'état (nowcast : `h_as_of` ; scrutin : 0).
    - `target_inst` / `target_v` : institut et variance d'échantillonnage d'un
      sondage CIBLE dont on veut la loi prédictive (protocole du §8). `None`
      pour un simple nowcast.

    La matrice `Σ = K_OU + σ_h²Δ + diag(v)` ne dépend du slot que par `diag(v)`,
    donc une factorisation de Cholesky par slot — `P` est de l'ordre de la
    centaine, c'est instantané.
    """
    Z = np.atleast_2d(Z)
    Vsamp = np.atleast_2d(Vsamp)
    h = np.asarray(h, dtype=float)
    inst_idx = np.asarray(inst_idx)
    P, K = Z.shape

    Kou = ou_kernel(h, h, tau, sigma2)
    same = (inst_idx[:, None] == inst_idx[None, :]).astype(float)
    base = Kou + sigma_h**2 * same

    k_lat = ou_kernel(np.array([h_star]), h, tau, sigma2)[0]        # (P,)
    if target_inst is not None:
        # Covariance entre le sondage cible et les sondages du pool : dérive
        # commune + house effect PARTAGÉ si c'est le même institut. C'est ce
        # terme qui empêche deux sondages d'une même maison de se confirmer
        # mutuellement à tort.
        k_obs = k_lat + sigma_h**2 * (inst_idx == target_inst).astype(float)
    one = np.ones(P)

    mean = np.empty(K)
    var_lat = np.empty(K)
    var_obs = np.empty(K) if target_inst is not None else None
    mu_hat = np.empty(K)

    for s in range(K):
        Sigma = base + np.diag(Vsamp[:, s]) + JITTER * np.eye(P)
        c = cho_factor(Sigma, lower=True)
        Si1 = cho_solve(c, one)
        Siz = cho_solve(c, Z[:, s])
        a = float(one @ Si1)                       # 1ᵀΣ⁻¹1
        m = float(one @ Siz) / a                   # mu chapeau (krigeage universel)
        mu_hat[s] = m

        # Variance latente : on utilise TOUJOURS k_lat, même quand un sondage
        # cible est fourni — θ(h*) ne dépend pas de qui l'observe.
        Sik = cho_solve(c, k_lat)
        var_lat[s] = max(sigma2 - float(k_lat @ Sik)
                         + (1.0 - float(one @ Sik)) ** 2 / a, 0.0)

        if target_inst is None:
            mean[s] = m + float(k_lat @ (Siz - m * Si1))
        else:
            Sik_obs = cho_solve(c, k_obs)
            mean[s] = m + float(k_obs @ (Siz - m * Si1))
            prior_v = sigma2 + sigma_h**2 + float(target_v[s])
            var_obs[s] = max(prior_v - float(k_obs @ Sik_obs)
                             + (1.0 - float(one @ Sik_obs)) ** 2 / a, 0.0)

    return GPPosterior(mean=mean, var_latent=var_lat, var_obs=var_obs, mu_hat=mu_hat)


def draws_from_posterior(mean: np.ndarray, var: np.ndarray, n_draws: int,
                         seed: int) -> np.ndarray:
    """Tirages de compositions `(n_draws, K)` : gaussienne par slot en CLR, puis
    softmax.

    `softmax(clr(y)) == y` exactement, donc ce softmax n'est pas une
    déformation ajoutée après coup : c'est l'inverse exact de la transformation
    par laquelle les sondages sont entrés. Les slots sont tirés indépendamment
    (§7.2) ; le softmax les recouple sur le simplexe, ce qui suffit à garantir
    le contrat `Nowcast.draws` (chaque tirage somme à 1).
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(mean, np.sqrt(np.maximum(var, 0.0)), size=(n_draws, len(mean)))
    e = np.exp(theta - theta.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
