"""Briques partagées du prototype spatial (Hotelling-Downs) — cf.
notebooks/03_spatial_prototype.py pour la justification de design (vraisemblance
tempérée pour w_now, pas d'état séquentiel) et notebooks/03b_spatial_debug.py
pour la découverte que w brut n'est identifié qu'à une constante additive près
(pi, lui, est bien identifié).

Nouveauté ici (session suivante) : positions mu ORDONNÉES par groupe (pas
seulement ancrées sur un analogue historique) — un groupe = un point de la
séquence gauche->droite fournie à la main ; les candidats d'un même groupe
(ex. Le Pen/Bardella, ou Roussel/Tondelier/Glucksmann/Hollande) partagent ce
point d'ancrage avec un petit delta individuel libre, PAS d'ordre imposé
entre eux (seul l'ordre ENTRE groupes est une contrainte dure).

Pas de scan/filtre : tout est vectorisé (P, N, B), réutilisable tel quel pour
un jeu de données réel (P sondages/hypothèses, N candidats) ou synthétique.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

B = 50
V = jnp.linspace(0.01, 0.99, B)
W = jnp.ones(B) / B


def sample_ordered_slots(name: str, n_slots: int, base_loc=0.0, base_scale=1.2,
                         gap_log_mu=-0.5, gap_log_scale=0.7):
    """n_slots positions STRICTEMENT croissantes dans (0,1) : base + sauts positifs
    (LogNormal) cumulés, écrasés par sigmoid (monotone -> préserve l'ordre)."""
    base = numpyro.sample(f"{name}_base", dist.Normal(base_loc, base_scale))
    if n_slots > 1:
        gaps = numpyro.sample(f"{name}_gaps", dist.LogNormal(gap_log_mu, gap_log_scale).expand([n_slots - 1]))
        raw = base + jnp.concatenate([jnp.zeros(1), jnp.cumsum(gaps)])
    else:
        raw = jnp.reshape(base, (1,))
    return numpyro.deterministic(f"{name}_slot_pos", jax.nn.sigmoid(raw))


def weighted_loglik(mu, sigma, w_now, tested_mask, Y, Np, kappa) -> jnp.ndarray:
    """Coeur partagé : softmax masqué sur la grille + vraisemblance gaussienne
    pondérée par récence (kappa). Entièrement vectorisé (P,N,B), voir
    03_spatial_prototype.py pour la dérivation détaillée."""
    D = -((V[None, :] - mu[:, None]) ** 2) / (2 * sigma[:, None] ** 2)      # (N,B)
    A = w_now[:, None] + D
    A = A - jnp.max(A, axis=0, keepdims=True)
    expA = jnp.exp(A)
    numerator = expA[None, :, :] * tested_mask[:, :, None]                  # (P,N,B)
    denom = jnp.clip(numerator.sum(axis=1, keepdims=True), 1e-12, None)
    pi = jnp.sum((numerator / denom) * W[None, None, :], axis=2)            # (P,N)
    var = pi * (1.0 - pi) / jnp.clip(Np[:, None], 1.0, None) + 1e-8
    ll = dist.Normal(pi, jnp.sqrt(var)).log_prob(Y) * tested_mask
    return pi, jnp.sum(kappa[:, None] * ll)


def spatial_model_ordered(slot_of, n_slots, tested_mask, Y, Np, kappa,
                          sd_delta_mu_scale=0.06, sd_delta_sigma_scale=0.15,
                          w_scale=1.5, sigma_slot_log_mu=None, sigma_slot_log_scale=0.4):
    """`slot_of` : (N,) indices dans [0, n_slots) -- groupe d'ordre de chaque
    candidat (cf. docstring module). Positions ordonnées au niveau des
    GROUPES ; chaque candidat a son propre delta (mu) et son propre sigma
    (pooling par groupe), et son propre w_now (récence, pas de pooling —
    03b_spatial_debug.py a montré que ça ne change rien à pi)."""
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

    w_now = numpyro.sample("w_now", dist.Normal(0.0, w_scale).expand([N]))

    pi, ll = weighted_loglik(mu, sigma, w_now, tested_mask, Y, Np, kappa)
    numpyro.deterministic("pi", pi)
    numpyro.factor("weighted_ll", ll)


def make_kappa(dates: np.ndarray, as_of: float, half_life: float) -> np.ndarray:
    if half_life >= 9999:
        return np.ones_like(dates, dtype=float)
    return 2.0 ** (-(as_of - dates) / half_life)


def build_poll_arrays(df, candidates: list[str], notice_col="notice", hyp_col="hypothese",
                      candidat_col="candidat", date_col="date_fin", intention_col="intention",
                      echantillon_col="echantillon"):
    """DataFrame long (une ligne = un candidat testé dans une hypothèse d'un
    sondage) -> (tested_mask, Y, Np, dates) prêts pour le modèle, restreint à
    `candidates`. Même déflation `echantillon / n_hypotheses` que
    `aggregate_to_slots` (model/core/live_dataset.py) : les hypothèses d'un
    même sondage partagent le terrain, pas des mesures indépendantes."""
    import pandas as pd

    idx = {c: i for i, c in enumerate(candidates)}
    df = df[df[candidat_col].isin(candidates)].copy()
    df[hyp_col] = df[hyp_col].fillna("__unique__")
    g = (df.groupby([notice_col, hyp_col, date_col, echantillon_col, candidat_col])[intention_col]
          .sum().reset_index())
    g["n_hyp"] = g.groupby(notice_col)[hyp_col].transform("nunique")

    N = len(candidates)
    tested_mask, Y, Np, dates = [], [], [], []
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

    dates_num = np.array([d.toordinal() for d in dates], dtype=float)
    return dict(tested_mask=np.array(tested_mask), Y=np.array(Y), Np=np.array(Np), dates=dates_num)
