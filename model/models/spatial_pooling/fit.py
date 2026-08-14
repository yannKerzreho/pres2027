"""Point d'entrée : fit complet, lecture d'un scénario arbitraire, projection au
scrutin.

Le fit rend les TIRAGES postérieurs, pas des moyennes — c'est ce qui permet de
répondre à un champ coché par un visiteur sans ré-inférence."""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro.diagnostics import summary as numpyro_summary

from model.core.bank import Bank
from model.core.inference import run_numpyro_mcmc
from model.core.simulate import forecast_from_draws

from .geometry import ancres_ecartees, ou_kl_basis, spatial_shares
from .joint_model import (LEVEL_SCALE, MIN_GAP_ANCRES, POS_SCALE, SEUIL_DYNAMIQUE,
                          SIGMA_JITTER, spatial_pooling_model_ou)
from .roster import (MIN_POLL_DATE, build_poll_arrays, build_roster,
                     excess_var_for_nodes, position_anchors)

# --- Fit + lecture AVEC incertitude (pas un point estimate) ----------------------
@dataclass
class SpatialPoolingFit:
    """Résultat d'un fit : les TIRAGES postérieurs (mu, sigma, w), pas des
    moyennes. C'est ce qui permet de répondre à un champ arbitraire coché par un
    visiteur sans ré-inférence — `pi_draws_for_mask` pousse chaque tirage à
    travers le softmax masqué."""
    candidates: list[str]
    anchors: np.ndarray       # (N,) ancre de position utilisée
    tested_mask: np.ndarray   # (P,N) champs réellement testés
    mu_draws: np.ndarray      # (S, N)
    sigma_draws: np.ndarray   # (S, N)
    w_draws: np.ndarray       # (S, N)
    diagnostics: dict


def fit_spatial_pooling(raw_polls: pd.DataFrame, as_of: str, draws: int = 400, tune: int = 600,
                        chains: int = 4, seed: int = 27, target_accept: float = 0.9,
                        excess_bank=None, pos_scale: float = POS_SCALE,
                        sigma_jitter: float = SIGMA_JITTER, min_gap: float = MIN_GAP_ANCRES,
                        seuil_dynamique: float = SEUIL_DYNAMIQUE) -> SpatialPoolingFit:
    """Roster -> arrays -> NUTS, en une seule inférence jointe.

    `mu`, `sigma` et le chemin de `w` sont estimés ENSEMBLE. La chaîne en deux
    temps qui a précédé (fit de géométrie, puis relecture de `w` par krigeage)
    ne pouvait pas propager l'incertitude de géométrie : mesuré sur données
    simulées, 99,0 % d'IC90 en géométrie variable contre 79,4 % figée, le vrai
    étant encadré sans qu'aucune répartition intermédiaire ne le corrige.

    `excess_bank` : variance d'excès par institut (`excess_var_for_nodes`).
    `None` charge celle calibrée pour CE modèle, repli sans excès si absente.

    Budget : ~90 s sur le roster 2027 (78 nœuds, 4 chaînes, 600+400) en
    `chain_method="sequential"`, qui est à la fois le plus rapide ici et le plus
    sobre en mémoire — le contraire de ce que suggère l'intuition, et de ce que
    la docstring de `run_numpyro_mcmc` a longtemps affirmé.
    """
    raw = raw_polls[pd.to_datetime(raw_polls["date_fin"]) >= MIN_POLL_DATE].copy()
    candidates, _, _ = build_roster(raw, as_of=as_of)
    arrays = build_poll_arrays(raw, candidates)

    anchors = ancres_ecartees(position_anchors(candidates), min_gap)

    if excess_bank is None:
        from model.models.spatial_pooling.calibration import BANK_EXCESS_PATH
        excess_bank = Bank.load(BANK_EXCESS_PATH)
    excess_var = excess_var_for_nodes(arrays["instituts"], excess_bank)

    from model.core.opinion import load_law
    tau_ou = float(load_law()["tau"])
    as_of_num = float(pd.Timestamp(as_of).toordinal())
    kl, K = ou_kl_basis(arrays["unique_dates"], as_of_num, tau_ou, center=True)

    # Chemin temporel réservé aux candidats qui pèsent : la trajectoire d'un
    # candidat à 1 % n'est pas identifiable sous ±1 pt de bruit d'échantillonnage.
    tested = arrays["tested_mask"] > 0
    part = np.array([arrays["Y"][tested[:, i], i].mean() if tested[:, i].any() else 0.0
                     for i in range(len(candidates))])

    kwargs = dict(
        tested_mask=jnp.asarray(arrays["tested_mask"]), Y=jnp.asarray(arrays["Y"]),
        Np=jnp.asarray(arrays["Np"]), date_idx=jnp.asarray(arrays["date_idx"]),
        kl_basis=jnp.asarray(kl), pos_anchor=jnp.asarray(anchors),
        excess_var=jnp.asarray(excess_var), tau_ou=tau_ou, pos_scale=pos_scale,
        sigma_jitter=sigma_jitter, dynamic_mask=part >= seuil_dynamique,
    )
    t0 = time.time()
    samples, extra = run_numpyro_mcmc(spatial_pooling_model_ou, kwargs, draws=draws, tune=tune,
                                      chains=chains, seed=seed, target_accept=target_accept,
                                      extra_fields=("diverging", "num_steps"))
    elapsed = time.time() - t0

    N = len(candidates)
    # `pi` est un déterministe (P, N) : son R-hat n'ajoute rien à celui des
    # paramètres et écraserait le pire cas. On le retire du diagnostic.
    diag = {k: v for k, v in numpyro_summary(samples, prob=0.9).items() if k != "pi"}
    pires = sorted(((float(np.max(v["r_hat"])), k) for k, v in diag.items()), reverse=True)[:3]
    steps = np.asarray(extra["num_steps"]).ravel()
    diagnostics = {
        "n_noeuds": int(arrays["tested_mask"].shape[0]),
        "n_candidats": N,
        "n_champs_distincts": len({tuple(np.flatnonzero(r)) for r in tested}),
        "n_dynamiques": int((part >= seuil_dynamique).sum()),
        "K_composantes": int(K),
        "temps_secondes": round(elapsed, 1),
        "rhat_max": round(max(r for r, _ in pires), 4),
        "rhat_pires_sites": [(k, round(r, 3)) for r, k in pires],
        "ess_min": round(min(float(np.min(v["n_eff"])) for v in diag.values()), 1),
        "n_divergences": int(np.sum(extra["diverging"])),
        # Le temps vaut `(tirages + warmup) x pas x chaînes` : c'est la mesure
        # qui dit si un changement a amélioré le CONDITIONNEMENT ou seulement le
        # coût par pas. 255 = profondeur 8 pleine, 1023 = plafond saturé.
        "pas_leapfrog_median": int(np.median(steps)),
        "sigma_w": round(float(np.mean(np.asarray(samples["sigma_w"]))), 4),
        "sigma_global": round(float(np.mean(np.asarray(samples["sigma_global"]))), 4),
    }
    return SpatialPoolingFit(
        candidates=candidates, anchors=anchors, tested_mask=arrays["tested_mask"],
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
