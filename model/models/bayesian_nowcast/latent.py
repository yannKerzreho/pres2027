"""Géométrie compositionnelle (ILR) du chemin SSM.

Justification mathématique complète : `docs/spec_ssm_espace_latent.md`.
Décision actée (`docs/spec_ssm_nowcast.md` §3) : ILR, base de Helmert — marche
gaussienne isotrope en `alpha` = marche isotrope au sens d'Aitchison sur le
simplexe, cohérent avec `campaign_drift_sd`/`movement_pool` (déjà en CLR
centré, `calibration.py`).

Un sondage 2027 ne teste JAMAIS les K slots simultanément (7 à 11 sur 12 dans
les données actuelles, aucun sondage complet) : `poll_observation` encode
chaque sondage comme une observation linéaire de l'état complet, de taille
FIXE K-1 (padding H=0/R=variance énorme sur les dimensions non testées —
contribution EXACTEMENT constante donc neutre à la vraisemblance, pas une
approximation, cf. docstring de la fonction), pour satisfaire la forme
rectangulaire qu'exige `model.core.lgssm.kalman_filter` (un array
`(ntime, emission_dim)`, `emission_dim` fixe)."""

from __future__ import annotations

from dataclasses import dataclass

import jax.nn
import jax.numpy as jnp
import numpy as np

# `clr` (log-ratio centré) est de la géométrie compositionnelle générique, pas
# propre au SSM : elle vit dans model/core/utils.py, d'où `movement_pool` la
# tire aussi. Ré-exportée ici pour que `from .latent import clr` reste valide.
from model.core.utils import FLOOR, clr  # noqa: F401

PADDING_VAR = 1e6      # variance de padding pour les dimensions non observées d'un sondage partiel


def helmert_basis(K: int) -> np.ndarray:
    """Base orthonormée (K x K-1) de l'hyperplan à somme nulle `H = {v : Σv=0}`
    — cf. docs/spec_ssm_espace_latent.md §8 pour la dérivation."""
    V = np.zeros((K, K - 1))
    for k in range(1, K):
        V[:k, k - 1] = 1.0 / np.sqrt(k * (k + 1))
        V[k, k - 1] = -k / np.sqrt(k * (k + 1))
    return V


def ilr_encode(shares: np.ndarray, V: np.ndarray, floor: float = FLOOR) -> np.ndarray:
    """pi (K,) -> alpha (K-1,) : alpha = V^T . clr(pi)."""
    return V.T @ clr(shares, floor)


def ilr_decode(alpha, V):
    """alpha (K-1,) -> pi (K,) : pi = softmax(V . alpha). JAX pur (utilisable
    dans un modèle NumPyro, différentiable) ; marche aussi sur des arrays
    NumPy en entrée (retourne alors un array JAX)."""
    return jax.nn.softmax(jnp.asarray(V) @ jnp.asarray(alpha))


@dataclass
class PollPlan:
    """Plan STATIQUE d'encodage d'un sondage — ne dépend que de quels slots
    sont testés, de la base V et du MODE choisi, JAMAIS d'une valeur
    échantillonnée (biais...). Calculé une fois par sondage à la
    construction de `NowcastData` (`nowcast.py`) ; réutilisé à chaque
    itération NUTS avec les parts (potentiellement débiaisées) courantes
    via `poll_observation_values`. `r_local` porte le MODE : `None` =
    encodage ILR local (`full_covariance=False`), sinon = index (dans
    `tested`) de la référence ALR (`full_covariance=True`) — cf.
    `poll_plan` pour le choix, `poll_observation_values` doit lire le MÊME
    plan pour rester cohérent (pas de mode passé séparément, pour éviter
    un désaccord H/y/R)."""
    tested: np.ndarray      # indices (0..K-1) des slots testés par ce sondage
    H: np.ndarray           # (K-1, K-1), paddé — ne dépend QUE de V/tested/mode
    r_local: int | None     # cf. docstring — porte le mode ALR (int) vs ILR local (None)


def poll_plan(shares: np.ndarray, V: np.ndarray, full_covariance: bool = False) -> PollPlan:
    """Construit le plan statique d'un sondage à partir de ses parts BRUTES
    (`shares`, NaN pour les slots non testés) — détermine QUELS slots sont
    testés et QUEL encodage utiliser, cf. docs/spec_ssm_encodage_partiel.md.

    `full_covariance=False` (par défaut, modèle de base, session du
    2026-08-10) : encodage ILR LOCAL (base de Helmert sur le sous-ensemble
    testé, PAS de référence privilégiée) — nécessaire pour que le bruit
    d'observation simplifié (Id, `poll_observation_values`) reste
    symétrique entre candidats testés (un `R=Id` n'est neutre entre
    candidats que sous une base ORTHONORMÉE ; une transformation ALR
    -changement de référence- ne l'est pas).

    `full_covariance=True` : ancien encodage ALR local, référence = part la
    plus large du sous-ensemble testé (choix numériquement le plus stable,
    cf. la note §6.1), à utiliser avec la covariance delta-method complète
    (`poll_observation_values`). Sous cette covariance, le choix de
    référence n'a mathématiquement AUCUNE influence sur le résultat
    (invariance prouvée, note §3-4) — argmax n'est qu'un choix de
    conditionnement, pas une approximation."""
    K = V.shape[0]
    Km1 = K - 1
    tested = np.where(~np.isnan(shares))[0]
    m = len(tested)
    if m < 2:
        raise ValueError("un sondage doit tester au moins 2 slots pour être exploitable")

    H = np.zeros((Km1, Km1))
    if full_covariance:
        r_local = int(np.argmax(shares[tested]))
        others = [i for i in range(m) if i != r_local]
        H[:m - 1, :] = V[tested[others], :] - V[tested[r_local], :]
        return PollPlan(tested=tested, H=H, r_local=r_local)

    W = helmert_basis(m)              # (m, m-1), base locale orthonormée du sous-ensemble testé
    H[:m - 1, :] = W.T @ V[tested, :]  # W^T . CLR_testé(p) = W^T . V[tested,:] . alpha (cf. note §2)
    return PollPlan(tested=tested, H=H, r_local=None)


def poll_observation_values(values_tested, plan: PollPlan, Km1: int, n: float,
                            floor: float = FLOOR, deff: float = 1.0,
                            isotropic: bool = False):
    """Partie DYNAMIQUE de l'encodage : `values_tested` (m,), les parts des
    slots de `plan.tested` — potentiellement débiaisées, donc calculée en JAX
    pur (différentiable, appelée à chaque itération NUTS avec le `biais`
    courant) -> `(y, R)` paddés à taille FIXE K-1, cf. `poll_observation`
    pour la dérivation et la justification du padding H=0/R=PADDING_VAR.
    `plan.H` (statique, déjà dans le MODE choisi par `poll_plan`) complète
    l'observation ; `plan.r_local` indique quel mode reproduire ici pour
    que `y`/`R` restent cohérents avec `H` (cf. `PollPlan`).

    `plan.r_local is None` (défaut, modèle de base) : covariance
    delta-method multinomiale exprimée dans la base ILR LOCALE orthonormée
    `W` (§ci-dessous pour la dérivation) :

        R = (deff/n) . W^T . diag(1/p) . W

    `isotropic=True` remplace ce terme par la SIMPLIFICATION historique
    `R = (deff/n) . I` (session du 2026-08-10) — conservée uniquement comme
    variante de comparaison. Elle revient à supposer `p_i = 1` pour TOUS les
    candidats : elle sous-estime la variance d'observation d'un facteur
    `1/p_i` (x2,9 pour un candidat à 34 %, x100 pour un candidat à 1 %), ce
    qui rend le filtre très largement sur-confiant (couverture IC90 mesurée
    à 35 %). Elle avait été adoptée pour atténuer la dérive du RN, dont la
    cause racine s'est révélée être ailleurs (mélange de rosters
    incohérents + renormalisation, cf. `biais_rn_investigation.md`
    addendum du 2026-08-11) — le motif de la simplification est donc tombé.

    Dérivation de la forme delta-method (identique en ALR et en ILR local,
    cf. le théorème d'invariance de spec_ssm.md §5.2 — ce ne sont que deux
    systèmes de coordonnées du MÊME modèle de bruit) : pour un multinomial,
    `Cov(p_hat) = (diag(p) - p p^T)/n` (donc `Var(p_i) = p_i(1-p_i)/n`) ;
    delta-method sur `log` (`J = diag(1/p)`) donne
    `Cov(log p_hat) = (diag(1/p) - 1 1^T)/n` ; la projection CLR
    `P = I - 1 1^T/m` annule le terme `-1 1^T` (`P.1 = 0`), d'où
    `Cov(clr) = P diag(1/(n p)) P` ; et comme `W^T P = W^T` (`W W^T = P`,
    `W^T W = I`), `Cov(y) = W^T diag(1/(n p)) W`. Le facteur `(1-p_i)` du
    multinomial n'est PAS perdu : il est exactement porté par la projection.

    `deff` (design effect) déflate `n` : les sondages français ne sont pas
    des tirages aléatoires simples et deux sondages appariés (même champ
    exact, même jour) se dispersent nettement plus que `p(1-p)/n` — mesuré
    sur 2017/2022, `deff ~ 2` (cf. `nowcast.py:DEFF_DEFAULT`).

    `plan.r_local` est un entier (`full_covariance=True`) : covariance
    delta-method du multinomial-log-ratio (ALR, référence `r_local`) :
        Sigma[j,j] = deff/n * (1/p_j + 1/p_ref)
        Sigma[j,k] = deff/n * (1/p_ref)   (j != k)
    (dérivation standard, cf. Aitchison — variance d'un log-ratio commun
    ALR ; invariante au choix de référence, cf. la note §3)."""
    m = len(plan.tested)
    p = jnp.clip(values_tested, floor, None)
    p = p / p.sum()
    inv_n = deff / max(n, 1.0)

    if plan.r_local is not None:
        others = [i for i in range(m) if i != plan.r_local]
        other_idx = jnp.array(others)
        p_ref, p_others = p[plan.r_local], p[other_idx]
        y_real = jnp.log(p_others / p_ref)
        Sigma_real = jnp.full((m - 1, m - 1), inv_n / p_ref) + jnp.diag(inv_n / p_others)
    else:
        W = jnp.asarray(helmert_basis(m))
        clr_tested = jnp.log(p) - jnp.log(p).mean()
        y_real = W.T @ clr_tested
        if isotropic:
            Sigma_real = inv_n * jnp.eye(m - 1)
        else:
            Sigma_real = inv_n * (W.T * (1.0 / p)) @ W
            # W^T diag(1/p) W est symétrique par construction ; le produit
            # matriciel en float32 ne l'est qu'à ~1e-7 près. Symétriser
            # explicitement (le filtre de Kalman suppose R symétrique).
            Sigma_real = 0.5 * (Sigma_real + Sigma_real.T)

    y = jnp.zeros(Km1).at[:m - 1].set(y_real)
    R = jnp.zeros((Km1, Km1)).at[:m - 1, :m - 1].set(Sigma_real)
    pad = jnp.arange(m - 1, Km1)
    R = R.at[pad, pad].set(PADDING_VAR)
    return y, R


def poll_observation(shares: np.ndarray, n: float, V: np.ndarray,
                     floor: float = FLOOR, deff: float = 1.0,
                     full_covariance: bool = False, isotropic: bool = False
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode UN sondage (potentiellement partiel) en observation linéaire de
    l'état ILR complet `alpha` (K-1,), de taille FIXE K-1 — combinaison de
    `poll_plan` (statique) + `poll_observation_values` (dynamique) pour un
    usage hors-modèle (tests, diagnostics) sans biais à soustraire.

    `shares` : vecteur (K,) de parts (%), NaN pour les slots non testés par
    ce sondage. `full_covariance` sélectionne le système de COORDONNÉES :
    `False` (défaut) = ILR local orthonormé, `True` = ALR local (référence =
    plus grande part testée). Avec la covariance delta-method (défaut), les
    deux sont mathématiquement ÉQUIVALENTS (même postérieur sur `alpha`,
    théorème d'invariance de spec_ssm.md §5.2) — l'ILR local est gardé par
    défaut parce qu'il reste orthonormé, donc cohérent avec `tau` isotrope
    et avec la variante `isotropic`. `isotropic=True` (comparaison seulement,
    n'a de sens que sous ILR local) rétablit `R = (deff/n)·I` : cf.
    `poll_observation_values` pour pourquoi ce n'est PAS la variance d'un
    multinomial.

    Les dimensions au-delà du sous-ensemble testé sont paddées à H=0, y=0,
    R=PADDING_VAR : H=0 rend ces lignes insensibles à `alpha` (résidu toujours
    nul), donc leur contribution à la log-vraisemblance est une CONSTANTE
    (indépendante de `alpha` et des hyperparamètres) — un décalage neutre du
    marginal_loglik, pas une approximation qui biaiserait l'inférence.

    Retourne `(y, H, R)` de tailles `((K-1,), (K-1,K-1), (K-1,K-1))`."""
    plan = poll_plan(shares, V, full_covariance=full_covariance)
    y, R = poll_observation_values(shares[plan.tested], plan, V.shape[0] - 1, n, floor,
                                   deff, isotropic)
    return np.asarray(y), plan.H, np.asarray(R)
