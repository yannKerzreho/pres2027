"""Modèle « spatial-pooling » (Hotelling-Downs discrétisé) — cf. spec_spatial_pooling.md
pour la dérivation complète et les résultats de validation (synthétique,
backtest half_life 2017/2022, sanity check 2027).

Portage des prototypes `notebooks/03_spatial_prototype.py` à
`04c_spatial_sanity_check.py` — même maths, structuré en module réutilisable.
PAS ENCORE un `ForecastModel` enregistré (`@register`) : décision explicite de
l'utilisateur (le contrat de sortie du site est encore en réflexion, cf.
spec §7/9). Ce module expose les fonctions nécessaires pour qu'un futur
`ForecastModel` (ou un endpoint de scénario interactif) les compose, sans rien
imposer sur la forme du contrat.

Différence clé avec `bayesian_nowcast`/`linear_pooling` : pas de vocabulaire
de "slots" figé (`SLOTS`, `model/core/live_dataset.py`) — chaque candidat réel
(Le Pen ET Bardella, Attal ET Philippe) est un paramètre séparé, tous deux
exploitables simultanément (c'est le point du modèle, cf. spec §0).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.diagnostics import summary as numpyro_summary

from model.core.bank import Bank
from model.core.inference import run_numpyro_mcmc
from model.core.live_dataset import load_raw_polls
from model.core.simulate import forecast_from_draws

# --- Grille spatiale (spec §1) --------------------------------------------------
B = 50
V = jnp.linspace(0.01, 0.99, B)
W = jnp.ones(B) / B

# --- Roster / fenêtre de campagne (spec §6) --------------------------------------
MIN_POLL_DATE = pd.Timestamp("2026-01-01")   # même repli que bayesian_nowcast.nowcast
MIN_POLLS = 5                                 # sondages RÉELS distincts (`notice`), pas hypothèses

# Ordre gauche->droite fourni à la main (session du 2026-08-10, corrigé le
# 2026-08-10 -- retour utilisateur) : un groupe = un point de la séquence,
# plusieurs candidats peuvent le partager (alternates, ou familles politiques
# proches) SANS ordre imposé entre eux -- cf. spec §2.1.
#
# Corrections apportées : (1) le candidat 2027 du groupe le plus à gauche est
# Arthaud, qui représente LO (Lutte Ouvrière), PAS le NPA (Poutou, NPA, ne
# passe pas le filtre >=5 sondages -- listé ici en repli inerte, tant qu'il
# n'atteint pas le seuil ça ne change rien) ; (2) "ecolo_ps_coco" scindé en
# deux points ordonnés, Coco/Écolo (PCF/EELV) < PS (Glucksmann/Hollande/Faure) ;
# (3) Bardella retiré du groupe MLP -- Marine Le Pen est la candidate
# OFFICIELLE du RN, il n'y a plus de scénario alternatif à modéliser sur ce
# point (contrairement à Attal/Philippe, encore ouvert).
ORDER_GROUPS: list[tuple[str, list[str]]] = [
    ("LO",         ["Arthaud", "Poutou"]),
    ("LFI",        ["Mélenchon", "Ruffin"]),
    ("coco_ecolo", ["Roussel", "Tondelier"]),
    ("PS",         ["Glucksmann", "Hollande", "Faure"]),
    ("Attal",      ["Attal"]),
    ("Philippe",   ["Philippe"]),
    ("LR",         ["Retailleau", "Wauquiez"]),
    ("MLP",        ["Le Pen", "Dupont-Aignan"]),
    ("Zemmour",    ["Zemmour"]),
]

# half_life (jours) : calibré par backtest hors-échantillon 2017/2022
# (notebooks/04b_spatial_halflife_backtest.py, cf. spec §5.2) -- 15j nettement
# meilleur que 45j/infini sur les deux élections, mais balayage grossier
# ({15,45,∞}), PAS affiné. Pas échantillonnable par NUTS (vraisemblance
# tempérée, cf. spec §3.1) -- constante calibrée séparément, comme
# `half_life_days` sur linear_pooling (model/models/linear_pooling/model.py).
DEFAULT_HALF_LIFE_DAYS = 15.0


def sample_ordered_slots(name: str, n_slots: int, base_loc=0.0, base_scale=1.2,
                         gap_log_mu=-0.5, gap_log_scale=0.7):
    """n_slots positions STRICTEMENT croissantes dans (0,1) : base + sauts positifs
    (LogNormal) cumulés, écrasés par sigmoid (monotone -> préserve l'ordre sans
    transformation `Ordered` explicite). Cf. spec §2.2."""
    base = numpyro.sample(f"{name}_base", dist.Normal(base_loc, base_scale))
    if n_slots > 1:
        gaps = numpyro.sample(f"{name}_gaps", dist.LogNormal(gap_log_mu, gap_log_scale).expand([n_slots - 1]))
        raw = base + jnp.concatenate([jnp.zeros(1), jnp.cumsum(gaps)])
    else:
        raw = jnp.reshape(base, (1,))
    return numpyro.deterministic(f"{name}_slot_pos", jax.nn.sigmoid(raw))


def spatial_shares(mu, sigma, w, tested_mask) -> jnp.ndarray:
    """Coeur du modèle d'observation (spec §4) : softmax masqué sur la grille +
    agrégation par les poids démographiques W -> pi (P,N) ou (N,) selon la
    forme de tested_mask. `mu`/`sigma`/`w` : (N,) OU (S,N) pour S tirages
    postérieurs poussés en parallèle (cf. `pi_draws_for_mask`, spec §"Incertitude") ;
    dans ce dernier cas `tested_mask` doit être (N,) (un seul scénario) et la
    sortie est (S,N)."""
    D = -((V[None, :] - mu[..., None]) ** 2) / (2 * sigma[..., None] ** 2)   # (...,N,B)
    A = w[..., None] + D
    A = A - jnp.max(A, axis=-2, keepdims=True)
    expA = jnp.exp(A)
    numerator = expA * tested_mask[..., :, None]
    denom = jnp.clip(numerator.sum(axis=-2, keepdims=True), 1e-12, None)
    return jnp.sum((numerator / denom) * W, axis=-1)


def weighted_loglik(mu, sigma, w, tested_mask, Y, Np, kappa, excess_var=None) -> tuple[jnp.ndarray, jnp.ndarray]:
    """`tested_mask`/`Y`/`Np`/`kappa` : (P,N)/(P,N)/(P,)/(P,) -- vraisemblance
    gaussienne (approx. du bruit multinomial, spec §4) pondérée par récence
    (spec §3.1 -- PAS une densité normalisée en `kappa`, cf. mise en garde
    sur pourquoi `half_life` n'est pas échantillonnable par NUTS).

    `excess_var` (P,) optionnel : variance d'excès (house effects) PAR NOEUD,
    en fraction² -- ajoutée à la variance d'échantillonnage pure
    `pi(1-pi)/N_p`, cf. spec section "Variance d'excès" / `excess_var_for_nodes`.
    `None` (défaut) = comportement d'origine (bruit d'échantillonnage seul)."""
    pi = spatial_shares(mu, sigma, w, tested_mask)                          # (P,N)
    var = pi * (1.0 - pi) / jnp.clip(Np[:, None], 1.0, None)
    if excess_var is not None:
        var = var + excess_var[:, None]
    var = var + 1e-8
    ll = dist.Normal(pi, jnp.sqrt(var)).log_prob(Y) * tested_mask
    return pi, jnp.sum(kappa[:, None] * ll)


def make_kappa(dates: np.ndarray, as_of: float, half_life: float) -> np.ndarray:
    if half_life >= 9999:
        return np.ones_like(dates, dtype=float)
    return 2.0 ** (-(as_of - dates) / half_life)


def spatial_pooling_model(slot_of, n_slots, tested_mask, Y, Np, kappa, excess_var=None,
                          sd_delta_mu_scale=0.06, sd_delta_sigma_scale=0.15,
                          w_scale=1.5, sigma_slot_log_mu=None, sigma_slot_log_scale=0.4):
    """`slot_of` : (N,) indice de groupe d'ordre par candidat (spec §2.1).
    Positions ordonnées au niveau des GROUPES ; chaque candidat a son propre
    delta (mu, spec §2.3), son propre sigma (pooling par groupe), et son
    propre w (spec §3 -- pas de hiérarchie partagée sur w, vérifié sans effet
    sur pi dans notebooks/03b_spatial_debug.py).

    `excess_var` (P,) optionnel -- cf. `weighted_loglik`/`excess_var_for_nodes` :
    variance d'excès par institut, réutilisée depuis la Bank déjà calibrée de
    `bayesian_nowcast` (spec section "Variance d'excès"). Corrige la
    sous-couverture mesurée empiriquement (spec §6.4/§6.5) -- SANS ce terme,
    le modèle traite `pi(1-pi)/N` comme le seul bruit d'observation, ce que
    `notebooks/06b_same_day_poll_coherence_matched.py` a démontré insuffisant
    sur 2017/2022 (RMS(z)=1,58 entre sondages de MÊME champ à écart nul)."""
    N = slot_of.shape[0]
    if sigma_slot_log_mu is None:
        sigma_slot_log_mu = jnp.log(0.15)

    slot_pos = sample_ordered_slots("mu", n_slots)
    sigma_slot = numpyro.sample("sigma_slot",
                                dist.LogNormal(sigma_slot_log_mu, sigma_slot_log_scale).expand([n_slots]))

    sd_delta_mu = numpyro.sample("sd_delta_mu", dist.HalfNormal(sd_delta_mu_scale))
    sd_delta_sigma = numpyro.sample("sd_delta_sigma", dist.HalfNormal(sd_delta_sigma_scale))
    z_mu = numpyro.sample("z_mu", dist.Normal(0.0, 1.0).expand([N]))
    z_sigma = numpyro.sample("z_sigma", dist.Normal(0.0, 1.0).expand([N]))
    mu = numpyro.deterministic("mu", slot_pos[slot_of] + z_mu * sd_delta_mu)
    sigma = numpyro.deterministic("sigma", sigma_slot[slot_of] * jnp.exp(z_sigma * sd_delta_sigma))

    w = numpyro.sample("w_now", dist.Normal(0.0, w_scale).expand([N]))

    pi, ll = weighted_loglik(mu, sigma, w, tested_mask, Y, Np, kappa, excess_var=excess_var)
    numpyro.deterministic("pi", pi)
    numpyro.factor("weighted_ll", ll)


# Petit plancher numérique ajouté à CHAQUE Δt de la marche dense (spec §3bis)
# -- même rôle que MIN_DT_EPS dans bayesian_nowcast/nowcast.py : évite une
# variance de transition nulle si deux dates uniques tombent le même jour
# (ne devrait pas arriver après dédoublonnage, mais coûte rien de le garder)
# ou si le dernier sondage tombe le jour même de `as_of`.
MIN_DT_EPS = 1e-6


def spatial_pooling_model_tau(slot_of, n_slots, tested_mask, Y, Np, date_idx, dt_gaps, as_of_dt,
                              sd_delta_mu_scale=0.06, sd_delta_sigma_scale=0.15,
                              tau_prior_scale=0.05, sigma0_prior_scale=0.5,
                              sigma_slot_log_mu=None, sigma_slot_log_scale=0.4):
    """PARKÉE (session du 2026-08-10, cf. spec §3.bis) : converge proprement
    (R-hat 1.005, ESS 1168, 7/4000 divergences) sur le roster 2027 réel mais
    vers un MODE DÉGÉNÉRÉ -- les mu se sont effondrés entre 0,70 et 0,99
    (quasi plus d'écart gauche-droite), le modèle distinguant les candidats
    via `w` plutôt que la position. R-hat/ESS sains n'excluent PAS un mode
    dégénéré (les chaînes peuvent converger ENSEMBLE vers le même mauvais
    mode). Aussi ~7x plus lent que `spatial_pooling_model` (2066s vs ~300s,
    taille comparable). La stratification par âge de la couverture (spec
    §6.4) suggère par ailleurs que la diffusion n'est probablement pas la
    cause dominante de la sous-couverture -- priorité donnée à un terme de
    variance d'excès (§10) à la place. Code gardé pour référence/reprise
    éventuelle, PAS la voie recommandée actuellement.

    Variante avec VRAIE diffusion temporelle (spec §3.bis) -- remplace la
    vraisemblance tempérée de `spatial_pooling_model` (§3.1) : `w_i(t)` suit
    une marche aléatoire gaussienne sur une ligne du temps GLOBALE dédupliquée
    (M dates uniques, partagée par tous les candidats -- DENSE, pas creuse,
    cf. `build_poll_arrays`), avec `tau` échantillonné normalement par NUTS
    (prior faible ici, PAS encore calibré sur 2017/2022 -- cf. spec §3bis
    pour ce qui reste à faire). Contrairement à `spatial_pooling_model`, la
    vraisemblance n'est PAS pondérée (`kappa` disparaît) : le poids de
    récence est maintenant une conséquence de la diffusion elle-même
    (un sondage vieux de 60 jours pèse moins parce que `tau²·60` a eu le
    temps de rendre l'état d'alors moins informatif sur l'état actuel, pas
    parce qu'on a artificiellement réduit sa vraisemblance).

    Dimension ajoutée par rapport à `spatial_pooling_model` : `N * M` (ex.
    13 candidats x 13 dates uniques = 169 sur le roster 2027 réel au
    2026-08-10 -- petit en pratique, pas besoin d'EKF/UKF pour ce volume de
    données)."""
    N = slot_of.shape[0]
    M = dt_gaps.shape[0] + 1
    if sigma_slot_log_mu is None:
        sigma_slot_log_mu = jnp.log(0.15)

    slot_pos = sample_ordered_slots("mu", n_slots)
    sigma_slot = numpyro.sample("sigma_slot",
                                dist.LogNormal(sigma_slot_log_mu, sigma_slot_log_scale).expand([n_slots]))

    sd_delta_mu = numpyro.sample("sd_delta_mu", dist.HalfNormal(sd_delta_mu_scale))
    sd_delta_sigma = numpyro.sample("sd_delta_sigma", dist.HalfNormal(sd_delta_sigma_scale))
    z_mu = numpyro.sample("z_mu", dist.Normal(0.0, 1.0).expand([N]))
    z_sigma = numpyro.sample("z_sigma", dist.Normal(0.0, 1.0).expand([N]))
    mu = numpyro.deterministic("mu", slot_pos[slot_of] + z_mu * sd_delta_mu)
    sigma = numpyro.deterministic("sigma", sigma_slot[slot_of] * jnp.exp(z_sigma * sd_delta_sigma))

    # tau/sigma0 PARTAGÉS (pas par candidat) pour commencer -- même prudence
    # que le SSM historique sur `tau_candidat` (essayé, pas adopté par
    # défaut faute de gain net, cf. docs/biais_rn_investigation.md point 9).
    tau = numpyro.sample("tau", dist.HalfNormal(tau_prior_scale))
    sigma0 = numpyro.sample("sigma0_w", dist.HalfNormal(sigma0_prior_scale))

    step_sd = jnp.concatenate([
        jnp.full((N, 1), sigma0),                                    # (N,1) état initial
        jnp.broadcast_to(tau * jnp.sqrt(dt_gaps + MIN_DT_EPS), (N, M - 1)),  # (N,M-1) pas de diffusion
    ], axis=1)
    z_w = numpyro.sample("z_w_path", dist.Normal(0.0, 1.0).expand([N, M]))
    w_path = jnp.cumsum(z_w * step_sd, axis=1)                        # (N,M)

    # Extrapolation d'un pas au-delà de la dernière date unique, jusqu'à
    # `as_of` -- même logique que l'extrapolation finale du SSM
    # (bayesian_nowcast_ssm_model, `extrap_cov`), pas de nouvelle observation.
    z_extra = numpyro.sample("z_w_extrap", dist.Normal(0.0, 1.0).expand([N]))
    w_now = numpyro.deterministic("w_now", w_path[:, -1] + tau * jnp.sqrt(as_of_dt + MIN_DT_EPS) * z_extra)

    w_at_node = w_path[:, date_idx].T                                 # (P,N) -- w_i au jour de chaque noeud
    D = -((V[None, :] - mu[:, None]) ** 2) / (2 * sigma[:, None] ** 2)  # (N,B)
    A = w_at_node[:, :, None] + D[None, :, :]                          # (P,N,B)
    A = A - jnp.max(A, axis=1, keepdims=True)
    expA = jnp.exp(A) * tested_mask[:, :, None]
    denom = jnp.clip(expA.sum(axis=1, keepdims=True), 1e-12, None)
    pi = jnp.sum((expA / denom) * W[None, None, :], axis=2)           # (P,N)

    var = pi * (1.0 - pi) / jnp.clip(Np[:, None], 1.0, None) + 1e-8
    ll = dist.Normal(pi, jnp.sqrt(var)).log_prob(Y) * tested_mask
    numpyro.deterministic("pi", pi)
    numpyro.factor("ll", jnp.sum(ll))


# --- EKF + Ornstein-Uhlenbeck pour w (spec §3.ter, session du 2026-08-12) --------
# Remplace la vraisemblance tempérée (§3.1) ET la marche brownienne parquée
# (§3.bis, mode dégénéré) par un vrai filtre séquentiel (extended_kalman_filter,
# dynamax -- déjà une dépendance du projet, cf. model/core/gp_math.py qui
# utilise le MÊME noyau OU pour une observation LINÉAIRE en CLR, avec forme
# close). Ici l'observation (softmax masqué sur la grille) est NON-LINÉAIRE en
# w, pas de forme close possible (contrairement à gp_pooling) -- EKF nécessaire,
# comme anticipé en session. `tau_w`/`sigma_w` échantillonnés par NUTS, prior
# faible, PAS calibrés sur 2017/2022 (contrairement à `sigma2`/`tau` de
# model/core/opinion.py, qui vivent dans un autre espace -- CLR agrégé, pas
# w candidat par candidat -- et ne sont donc pas directement transférables).
PADDING_VAR_EKF = 1e6   # même esprit que PADDING_VAR (bayesian_nowcast/latent.py) : neutralise
                        # la contribution des candidats non testés à un nœud, sans les exclure
                        # du vecteur d'émission à taille FIXE qu'exige l'EKF.


def sort_by_date(arrays: dict) -> dict:
    """Trie les nœuds de `build_poll_arrays` par date croissante et ajoute
    `dt_forward[p]` = écart (jours) vers le nœud SUIVANT (dernier = 0,
    inutilisé) -- nécessaire pour un filtre SÉQUENTIEL (EKF), contrairement
    au modèle tempéré/à la marche dense qui n'ont pas besoin d'ordre
    temporel explicite. Même convention que `dt_forward` dans
    `bayesian_nowcast/nowcast.py::NowcastData`."""
    order = np.argsort(arrays["dates"])
    out = dict(arrays)
    out["tested_mask"] = arrays["tested_mask"][order]
    out["Y"] = arrays["Y"][order]
    out["Np"] = arrays["Np"][order]
    out["dates"] = arrays["dates"][order]
    out["instituts"] = [arrays["instituts"][i] for i in order]
    dt_forward = np.diff(out["dates"])
    out["dt_forward"] = np.append(dt_forward, 0.0)
    return out


def spatial_pooling_model_ekf(slot_of, n_slots, tested_mask, Y, Np, dt_forward, as_of_dt, excess_var=None,
                              sd_delta_mu_scale=0.06, sd_delta_sigma_scale=0.15,
                              tau_w_log_mu=None, tau_w_log_sd=0.8, sigma_w_prior_scale=1.0,
                              sigma_slot_log_mu=None, sigma_slot_log_scale=0.4):
    """`w_i(t)` suit un Ornstein-Uhlenbeck (mean-reversion vers 0 -- inoffensif,
    la jauge de `w` n'est pas identifiée en niveau absolu, §3.2), filtré par
    EKF : NUTS n'échantillonne que les hyperparamètres GLOBAUX (`mu`,`sigma`,
    `tau_w`,`sigma_w`), pas un `w` par nœud (évite la dimension `N x P`
    qu'un NUTS-sur-chemin-complet aurait demandée). `tested_mask`/`Y`/`Np`
    DOIVENT être triés par date croissante (`sort_by_date`) : le filtre est
    séquentiel, contrairement aux autres variantes.

    Transition OU exacte (pas une approximation Euler) sur un pas `dt` :
    `w(t+dt) | w(t) ~ N(w(t)·e^{-dt/τ}, σ_w²·(1-e^{-2dt/τ}))`. Observation
    `h(w) = spatial_shares(mu, sigma, w, mask)` -- non-linéaire, d'où l'EKF
    (linéarise `h` à chaque pas via le Jacobien, `jax.jacfwd` dans dynamax).
    `R` construit à partir des valeurs OBSERVÉES `Y` (pas de `pi` prédit par
    l'état) -- même convention que `poll_observation_values`
    (bayesian_nowcast/latent.py) : évite toute circularité (R ne doit pas
    dépendre de l'état filtré)."""
    from dynamax.nonlinear_gaussian_ssm.inference_ekf import extended_kalman_filter
    from dynamax.nonlinear_gaussian_ssm.models import ParamsNLGSSM

    N = slot_of.shape[0]
    P = tested_mask.shape[0]
    if sigma_slot_log_mu is None:
        sigma_slot_log_mu = jnp.log(0.15)
    if tau_w_log_mu is None:
        tau_w_log_mu = jnp.log(45.0)   # ordre de grandeur du `tau` calibré pour la
                                       # diffusion CLR (model/core/opinion.py, médiane ~60j)
                                       # -- repli raisonnable, PAS calibré pour w lui-même

    slot_pos = sample_ordered_slots("mu", n_slots)
    sigma_slot = numpyro.sample("sigma_slot",
                                dist.LogNormal(sigma_slot_log_mu, sigma_slot_log_scale).expand([n_slots]))
    sd_delta_mu = numpyro.sample("sd_delta_mu", dist.HalfNormal(sd_delta_mu_scale))
    sd_delta_sigma = numpyro.sample("sd_delta_sigma", dist.HalfNormal(sd_delta_sigma_scale))
    z_mu = numpyro.sample("z_mu", dist.Normal(0.0, 1.0).expand([N]))
    z_sigma = numpyro.sample("z_sigma", dist.Normal(0.0, 1.0).expand([N]))
    mu = numpyro.deterministic("mu", slot_pos[slot_of] + z_mu * sd_delta_mu)
    sigma = numpyro.deterministic("sigma", sigma_slot[slot_of] * jnp.exp(z_sigma * sd_delta_sigma))

    tau_w = numpyro.sample("tau_w", dist.LogNormal(tau_w_log_mu, tau_w_log_sd))
    sigma_w = numpyro.sample("sigma_w", dist.HalfNormal(sigma_w_prior_scale))
    sigma_w2 = sigma_w ** 2

    def f(w, u):
        dt = u[-1]
        return w * jnp.exp(-dt / tau_w)

    def h(w, u):
        mask = u[:-1]
        return spatial_shares(mu, sigma, w, mask)

    var_obs = Y * (1.0 - Y) / jnp.clip(Np[:, None], 1.0, None)
    if excess_var is not None:
        var_obs = var_obs + excess_var[:, None]
    var_obs = jnp.where(tested_mask > 0, var_obs, PADDING_VAR_EKF)
    R = jax.vmap(jnp.diag)(var_obs)                                   # (P,N,N)

    Q_scale = sigma_w2 * (1.0 - jnp.exp(-2.0 * dt_forward / tau_w))    # (P,) -- vers le nœud SUIVANT
    Q = jax.vmap(lambda q: q * jnp.eye(N))(Q_scale)                    # (P,N,N)

    inputs = jnp.concatenate([tested_mask, dt_forward[:, None]], axis=1)   # (P, N+1)

    params = ParamsNLGSSM(
        initial_mean=jnp.zeros(N), initial_covariance=sigma_w2 * jnp.eye(N),
        dynamics_function=f, dynamics_covariance=Q,
        emission_function=h, emission_covariance=R,
    )
    post = extended_kalman_filter(params, Y, inputs=inputs,
                                  output_fields=["filtered_means", "filtered_covariances", "marginal_loglik"])
    # `post.marginal_loglik` est la trace CUMULATIVE par pas de temps (dynamax
    # accumule `ll` dans le carry du scan et la restitue à chaque pas, cf.
    # source lue en session) -- seul le DERNIER élément est le vrai
    # log-vraisemblance marginal total ; sommer tout le tableau compterait
    # les termes en double (bug trouvé par contrôle direct, valeurs non
    # monotones sinon).
    ll_total = post.marginal_loglik[-1] if P > 0 else 0.0
    numpyro.factor("ekf_ll", ll_total)

    # Extrapolation d'UN pas au-delà du dernier nœud, jusqu'à `as_of` -- même
    # logique que l'extrapolation finale du SSM historique (`extrap_cov`),
    # PAS une nouvelle observation.
    if P > 0:
        last_mean, last_cov = post.filtered_means[-1], post.filtered_covariances[-1]
    else:
        last_mean, last_cov = jnp.zeros(N), sigma_w2 * jnp.eye(N)
    decay = jnp.exp(-as_of_dt / tau_w)
    extrap_mean = last_mean * decay
    extrap_cov = (decay ** 2) * last_cov + sigma_w2 * (1.0 - decay ** 2) * jnp.eye(N)
    L = jnp.linalg.cholesky(extrap_cov + 1e-9 * jnp.eye(N))
    z_extra = numpyro.sample("z_w_extrap", dist.Normal(0.0, 1.0).expand([N]))
    numpyro.deterministic("w_now", extrap_mean + L @ z_extra)


# --- Roster & mise en forme des sondages -----------------------------------------
def build_roster(raw: pd.DataFrame) -> tuple[list[str], np.ndarray, list[str]]:
    """Filtre les candidats à >= MIN_POLLS sondages RÉELS distincts (spec §6),
    puis assigne chacun à son groupe d'ordre (ORDER_GROUPS). Un candidat
    éligible mais absent de ORDER_GROUPS est signalé (à classer), pas planté."""
    counts = raw.groupby("candidat")["notice"].nunique()
    eligible = set(counts[counts >= MIN_POLLS].index)

    candidates, slot_of, slot_names = [], [], []
    for gi, (gname, members) in enumerate(ORDER_GROUPS):
        for m in members:
            if m in eligible:
                candidates.append(m)
                slot_of.append(gi)
        slot_names.append(gname)

    dropped = eligible - set(candidates)
    if dropped:
        import warnings
        warnings.warn(f"candidats éligibles mais absents de ORDER_GROUPS, à classer : {sorted(dropped)}")
    return candidates, np.array(slot_of), slot_names


def build_poll_arrays(df: pd.DataFrame, candidates: list[str], notice_col="notice", hyp_col="hypothese",
                      candidat_col="candidat", date_col="date_fin", intention_col="intention",
                      echantillon_col="echantillon", institut_col="institut") -> dict:
    """DataFrame long (une ligne = un candidat testé dans une hypothèse d'un
    sondage) -> (tested_mask, Y, Np, dates, instituts) restreint à
    `candidates`. Même déflation `echantillon / n_hypotheses` que
    `aggregate_to_slots` (model/core/live_dataset.py) : les hypothèses d'un
    même sondage partagent le terrain, pas des mesures indépendantes.
    `instituts` (P,) -- un nom d'institut par nœud, nécessaire pour
    `excess_var_for_nodes` (variance d'excès par institut)."""
    idx = {c: i for i, c in enumerate(candidates)}
    df = df[df[candidat_col].isin(candidates)].copy()
    df[hyp_col] = df[hyp_col].fillna("__unique__")
    g = (df.groupby([notice_col, hyp_col, date_col, echantillon_col, institut_col, candidat_col])[intention_col]
          .sum().reset_index())
    g["n_hyp"] = g.groupby(notice_col)[hyp_col].transform("nunique")

    N = len(candidates)
    tested_mask, Y, Np, dates, instituts = [], [], [], [], []
    for (notice, hyp), grp in g.groupby([notice_col, hyp_col], sort=False):
        mask = np.zeros(N)
        y = np.zeros(N)
        for r in grp.itertuples(index=False):
            cand = getattr(r, candidat_col)
            if cand in idx:
                mask[idx[cand]] = 1.0
                y[idx[cand]] = getattr(r, intention_col) / 100.0
        if mask.sum() < 2:
            continue
        tested_mask.append(mask)
        Y.append(y)
        Np.append(float(grp[echantillon_col].iloc[0]) / float(grp["n_hyp"].iloc[0]))
        dates.append(pd.Timestamp(grp[date_col].iloc[0]))
        instituts.append(str(grp[institut_col].iloc[0]))

    dates_num = np.array([d.toordinal() for d in dates], dtype=float)

    # Ligne du temps GLOBALE dédupliquée (jours calendaires distincts, pas
    # nœuds) -- nécessaire pour la marche aléatoire dense de
    # `spatial_pooling_model_tau` (spec §3bis) : `date_idx[p]` pointe vers la
    # position du nœud p sur cette ligne, `dt_gaps[m]` l'écart (jours) entre
    # deux dates uniques consécutives. Calculé systématiquement (coût nul
    # pour le modèle tempéré, qui ignore juste ces clés).
    unique_dates = np.sort(np.unique(dates_num))
    date_idx = np.searchsorted(unique_dates, dates_num)
    dt_gaps = np.diff(unique_dates)

    return dict(tested_mask=np.array(tested_mask), Y=np.array(Y), Np=np.array(Np), dates=dates_num,
               unique_dates=unique_dates, date_idx=date_idx, dt_gaps=dt_gaps, instituts=instituts)


# --- Variance d'excès (house effects) -- calibration PROPRE à spatial_pooling ------
def excess_var_for_nodes(instituts: list[str], bank) -> np.ndarray:
    """Variance d'excès PAR NOEUD (P,), en fraction² -- ajoutée à
    `pi(1-pi)/N` dans `weighted_loglik`.

    Historique (spec §6.5) : une première version réutilisait TELLE QUELLE la
    Bank `bayesian_nowcast` (`excess_sigma`, calibrée en espace ILR contre
    l'écart au résultat final) -- mesuré en SUR-correction sévère (IC90
    empirique à 99,6% au lieu de 90% sur 2022) : cette Bank mélange bruit
    d'institut et dérive de campagne dans sa propre décomposition, pas
    directement transférable à la géométrie de `spatial_pooling`.

    Remplacée par une calibration DÉDIÉE (`calibration.py::SpatialExcessCalibration`,
    `excess_sigma_spatial`) -- fit sur le même diagnostic model-free qui a
    révélé le problème (paires de sondages de champ EXACTEMENT identique,
    `notebooks/06b_same_day_poll_coherence_matched.py`), pas emprunté à un
    autre modèle. Vérifié par contrôle postérieur (RMS(z) sur les mêmes
    paires : 1,57 sans excès -> 1,11 avec, cf. spec §6.5)."""
    from model.models.spatial_pooling.calibration import excess_sigma_spatial
    if bank is None:
        return np.zeros(len(instituts))
    return np.array([(excess_sigma_spatial(bank, inst) / 100.0) ** 2 for inst in instituts])


# --- Fit + lecture AVEC incertitude (pas un point estimate) ----------------------
@dataclass
class SpatialPoolingFit:
    """Résultat d'un fit : tirages POSTÉRIEURS complets (mu, sigma, w), pas
    des moyennes -- nécessaire pour propager l'incertitude (cf. spec section
    "Incertitude", et `pi_draws_for_mask` ci-dessous)."""
    candidates: list[str]
    slot_of: np.ndarray
    slot_names: list[str]
    mu_draws: np.ndarray      # (S, N)
    sigma_draws: np.ndarray   # (S, N)
    w_draws: np.ndarray       # (S, N)
    diagnostics: dict


def fit_spatial_pooling(raw_polls: pd.DataFrame, as_of: str, half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
                        draws: int = 1000, tune: int = 1000, chains: int = 4, seed: int = 27,
                        target_accept: float = 0.95, excess_bank=None) -> SpatialPoolingFit:
    """Fit complet : roster -> arrays -> NUTS. Retourne les tirages bruts
    (mu/sigma/w), PAS un pi déjà agrégé -- `pi_draws_for_mask` construit pi
    APRÈS coup pour un scénario donné (le point de cette architecture, cf.
    spec §7 : un scénario personnalisé est un calcul direct sur ces tirages,
    pas une ré-inférence).

    `excess_bank` : Bank de variance d'excès à réutiliser (`excess_var_for_nodes`)
    -- `None` (défaut) charge celle calibrée pour CE modèle
    (`calibration.BANK_EXCESS_PATH`) si présente, repli sans excès sinon (pas
    d'erreur si jamais calibrée). PAS la Bank `bayesian_nowcast` -- mesurée en
    sur-correction sévère, cf. `excess_var_for_nodes`."""
    raw = raw_polls[pd.to_datetime(raw_polls["date_fin"]) >= MIN_POLL_DATE].copy()
    candidates, slot_of, slot_names = build_roster(raw)
    arrays = build_poll_arrays(raw, candidates)
    as_of_ord = pd.Timestamp(as_of).toordinal()
    kappa = make_kappa(arrays["dates"], as_of=as_of_ord, half_life=half_life_days)

    if excess_bank is None:
        from model.models.spatial_pooling.calibration import BANK_EXCESS_PATH
        excess_bank = Bank.load(BANK_EXCESS_PATH)
    excess_var = excess_var_for_nodes(arrays["instituts"], excess_bank)

    kwargs = dict(
        slot_of=jnp.asarray(slot_of), n_slots=len(slot_names),
        tested_mask=jnp.asarray(arrays["tested_mask"]), Y=jnp.asarray(arrays["Y"]),
        Np=jnp.asarray(arrays["Np"]), kappa=jnp.asarray(kappa), excess_var=jnp.asarray(excess_var),
    )
    t0 = time.time()
    samples, extra = run_numpyro_mcmc(spatial_pooling_model, kwargs, draws=draws, tune=tune,
                                      chains=chains, seed=seed, target_accept=target_accept,
                                      extra_fields=("diverging",))
    elapsed = time.time() - t0

    N = len(candidates)
    diag = numpyro_summary(samples, prob=0.9)
    diagnostics = {
        "half_life_days": half_life_days,
        "n_noeuds": int(arrays["tested_mask"].shape[0]),
        "n_candidats": N,
        "temps_secondes": round(elapsed, 1),
        "rhat_max": round(max(float(np.max(d["r_hat"])) for d in diag.values()), 4),
        "ess_min": round(min(float(np.min(d["n_eff"])) for d in diag.values()), 1),
        "n_divergences": int(np.sum(extra["diverging"])),
        "excess_var_moyenne_pt": round(float(np.mean(excess_var)) * 10000, 3),   # fraction² -> pt²
        "excess_bank_chargee": excess_bank is not None,
    }
    return SpatialPoolingFit(
        candidates=candidates, slot_of=slot_of, slot_names=slot_names,
        mu_draws=np.asarray(samples["mu"]).reshape(-1, N),
        sigma_draws=np.asarray(samples["sigma"]).reshape(-1, N),
        w_draws=np.asarray(samples["w_now"]).reshape(-1, N),
        diagnostics=diagnostics,
    )


def pi_draws_for_mask(fit: SpatialPoolingFit, mask: np.ndarray) -> np.ndarray:
    """LE point de cette architecture (spec §7) : pi pour un scénario
    ARBITRAIRE (ex. coché par un utilisateur sur le site), calculé en poussant
    CHAQUE tirage postérieur déjà en mémoire à travers le softmax masqué --
    pas de ré-inférence, O(tirages x B). `mask` : (N,) 0/1, quels candidats
    du scénario sont "en lice". Retourne (S, N) -- une composition (sur les
    candidats du masque) par tirage, comme `Nowcast.draws` (model/core/base.py).

    C'est aussi ce qui porte l'incertitude correctement : chaque tirage vient
    du postérieur JOINT (mu, sigma, w), qui s'élargit déjà tout seul quand les
    sondages récents d'un candidat se contredisent (pas besoin d'un mélange
    explicite comme linear_pooling -- cf. spec section "Incertitude" pour la
    comparaison et ses limites connues)."""
    return np.asarray(spatial_shares(jnp.asarray(fit.mu_draws), jnp.asarray(fit.sigma_draws),
                                     jnp.asarray(fit.w_draws), jnp.asarray(mask)))


def summarize_pi(pi_draws: np.ndarray, labels: list[str]) -> dict:
    """(S,N) -> {label: {mean, ic90}} -- même résumé que `Nowcast.summary()`
    (model/core/base.py), réutilisable indépendamment du contrat final."""
    out = {}
    for i, label in enumerate(labels):
        vals = pi_draws[:, i]
        out[label] = {
            "mean": round(float(vals.mean()), 4),
            "ic90": [round(float(np.percentile(vals, 5)), 4), round(float(np.percentile(vals, 95)), 4)],
        }
    return out


# --- Saut terminal (nowcast -> scrutin) : machinerie PARTAGÉE, plus rien à faire ici --
# `model/core/projection.py::projeter_au_scrutin` + `model/core/opinion.py::load_law()`
# -- diffusion d'opinion (Ornstein-Uhlenbeck) puis écart sondages-urne δ,
# calibrés une seule fois pour tout le dépôt (`bank_opinion.json`), pas par
# modèle. `spatial_pooling` n'a plus de saut/Bank qui lui soit propre (l'ancien
# `TerminalJumpCalibration`/`BANK_JUMP_PATH` par modèle a disparu avec la
# restructuration -- cf. `bayesian_nowcast`/`gp_pooling`, même pattern partout).
def forecast_spatial_pooling(fit: SpatialPoolingFit, mask: np.ndarray, horizon_days: int) -> dict:
    """Projette un scénario (`mask`, ex. coché par un utilisateur) jusqu'au
    scrutin. `pi_draws_for_mask` donne l'incertitude au nowcast (spec section
    "Incertitude") ; `projeter_au_scrutin` ajoute la dérive résiduelle d'ici
    au scrutin, EXACTEMENT comme les autres modèles -- rien de spécifique à
    spatial_pooling au-delà de fournir `pi` dans le bon format."""
    from model.core.opinion import load_law
    from model.core.projection import projeter_au_scrutin

    mask_bool = mask.astype(bool)
    labels = [c for c, keep in zip(fit.candidates, mask_bool) if keep]
    pi = pi_draws_for_mask(fit, mask)[:, mask_bool]
    pi = projeter_au_scrutin(pi, horizon_days, load_law())
    return forecast_from_draws(pi, labels, horizon_days)
