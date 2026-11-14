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

from .geometry import (ancres_ecartees, duel_shares, log_attractivite, ou_kl_basis,
                       spatial_shares)
from .joint_model import (LEVEL_SCALE, MIN_GAP_ANCRES, POS_SCALE, SEUIL_DYNAMIQUE,
                          SIGMA_JITTER, spatial_pooling_model_ou)
from .roster import (COUVERTURE_MIN, MIN_POLL_DATE, build_duel_arrays, build_poll_arrays,
                     build_roster, candidats_retires, excess_var_for_nodes, position_anchors,
                     split_tours)

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
    # Second tour (spec_second_tour.md). `None` si le fit n'a vu aucun duel :
    # le modèle T1 est alors exactement celui d'avant.
    kappa_draws: np.ndarray | None = None    # (S,)
    eta_draws: np.ndarray | None = None      # (S, N)
    level_draws: np.ndarray | None = None    # (S, N) niveau m_i, centre du processus OU
    sigma_w_draws: np.ndarray | None = None  # (S,)
    dynamic: np.ndarray | None = None        # (N,) bool, candidats à chemin OU
    tau_ou: float | None = None
    as_of: str | None = None
    duel_pairs: list | None = None           # paires effectivement observées


def fit_spatial_pooling(raw_polls: pd.DataFrame, as_of: str, draws: int = 400, tune: int = 600,
                        chains: int = 4, seed: int = 27, target_accept: float = 0.9,
                        excess_bank=None, pos_scale: float = POS_SCALE,
                        sigma_jitter: float = SIGMA_JITTER, min_gap: float = MIN_GAP_ANCRES,
                        seuil_dynamique: float = SEUIL_DYNAMIQUE,
                        min_poll_date=None, order_groups=None,
                        tau_ou: float | None = None,
                        couverture_min: float | None = COUVERTURE_MIN,
                        second_tour: bool = True) -> SpatialPoolingFit:
    """Roster -> arrays -> NUTS, en une seule inférence jointe.

    `mu`, `sigma` et le chemin de `w` sont estimés ENSEMBLE. La chaîne en deux
    temps qui a précédé (fit de géométrie, puis relecture de `w` par krigeage)
    ne pouvait pas propager l'incertitude de géométrie : mesuré sur données
    simulées, 99,0 % d'IC90 en géométrie variable contre 79,4 % figée, le vrai
    étant encadré sans qu'aucune répartition intermédiaire ne le corrige.

    `couverture_min` : cf. COUVERTURE_MIN ; `None` désactive le contrôle
    (backtest : des sondages 2012-2015 du frame 2017 sont couverts à 0,34).

    `excess_bank` : variance d'excès par institut (`excess_var_for_nodes`).
    `None` charge celle calibrée pour CE modèle, repli sans excès si absente.

    Budget : ~90 s sur le roster 2027 (78 nœuds, 4 chaînes, 600+400) en
    `chain_method="sequential"`, qui est à la fois le plus rapide ici et le plus
    sobre en mémoire — le contraire de ce que suggère l'intuition, et de ce que
    la docstring de `run_numpyro_mcmc` a longtemps affirmé.
    """
    # `min_poll_date` / `order_groups` / `tau_ou` sont injectables pour le
    # backtest historique : la fenêtre de campagne et la table d'ordre sont
    # propres à 2027, et `tau_ou` doit pouvoir suivre la banque qu'on teste.
    plancher = MIN_POLL_DATE if min_poll_date is None else pd.Timestamp(min_poll_date)
    raw_all = raw_polls[pd.to_datetime(raw_polls["date_fin"]) >= plancher].copy()
    # Roster, couverture et `part` restent définis par le PREMIER tour seul :
    # un duel à ~50 % n'est pas une intention de premier tour.
    raw, raw_t2 = split_tours(raw_all)
    candidates, _, _ = build_roster(raw, as_of=as_of, order_groups=order_groups)
    arrays = build_poll_arrays(raw, candidates, couverture_min=couverture_min)
    duels = build_duel_arrays(raw_t2 if second_tour else raw_t2.iloc[0:0], candidates)
    has_t2 = len(duels["y"]) > 0
    if has_t2:
        from .roster import TIME_BIN_DAYS
        assert TIME_BIN_DAYS == 1.0, "l'union des dates suppose des pas d'un jour"
        # Ligne du temps commune aux deux tours (TIME_BIN_DAYS = 1 jour).
        unique_dates = np.union1d(arrays["unique_dates"], np.floor(duels["dates"]))
        arrays["date_idx"] = np.searchsorted(unique_dates, np.floor(arrays["dates"]))
        arrays["unique_dates"] = unique_dates
        duel_date_idx = np.searchsorted(unique_dates, np.floor(duels["dates"]))

    anchors = ancres_ecartees(position_anchors(candidates, order_groups), min_gap)

    if excess_bank is None:
        from model.models.spatial_pooling.calibration import BANK_EXCESS_PATH
        excess_bank = Bank.load(BANK_EXCESS_PATH)
    excess_var = excess_var_for_nodes(arrays["instituts"], excess_bank)

    from model.core.opinion import load_law
    tau_ou = float(load_law()["tau"]) if tau_ou is None else float(tau_ou)
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
    if has_t2:
        kwargs.update(duel_a=jnp.asarray(duels["a"]), duel_b=jnp.asarray(duels["b"]),
                      duel_y=jnp.asarray(duels["y"]), duel_n=jnp.asarray(duels["n"]),
                      duel_date_idx=jnp.asarray(duel_date_idx))
    t0 = time.time()
    samples, extra = run_numpyro_mcmc(spatial_pooling_model_ou, kwargs, draws=draws, tune=tune,
                                      chains=chains, seed=seed, target_accept=target_accept,
                                      extra_fields=("diverging", "num_steps"))
    elapsed = time.time() - t0

    N = len(candidates)
    # `pi` est un déterministe (P, N) : son R-hat n'ajoute rien à celui des
    # paramètres et écraserait le pire cas. On le retire du diagnostic.
    diag = {k: v for k, v in numpyro_summary(samples, prob=0.9).items() if k != "pi"}
    if has_t2:
        # `eta` d'un candidat sans duel suit son prior : R-hat valable, mais on
        # le distingue dans le diagnostic pour ne pas le confondre avec un
        # paramètre contraint.
        en_duel = sorted(set(duels["a"]) | set(duels["b"]))
        # R-hat PROPRE au bloc T2, sur les seuls eta contraints par un duel :
        # le `rhat_max` global est celui du T1 et ne garde rien du T2.
        rhat_t2 = max([float(np.max(diag[k]["r_hat"])) for k in ("kappa", "sigma_t2")]
                      + [float(np.max(np.asarray(diag["z_eta"]["r_hat"])[en_duel]))])
    pires = sorted(((float(np.max(v["r_hat"])), k) for k, v in diag.items()), reverse=True)[:3]
    steps = np.asarray(extra["num_steps"]).ravel()
    diagnostics = {
        "n_noeuds": int(arrays["tested_mask"].shape[0]),
        "n_candidats": N,
        "candidats_retires": candidats_retires(raw, candidates, as_of),
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
    if has_t2:
        diagnostics.update({
            "n_duels": int(len(duels["y"])),
            "paires_duel": sorted({f"{x} / {y}" for x, y in duels["pairs"]}),
            "candidats_en_duel": [candidates[i] for i in en_duel],
            "kappa": round(float(np.mean(np.asarray(samples["kappa"]))), 4),
            "rhat_t2": round(rhat_t2, 4),
            "sigma_t2": round(float(np.mean(np.asarray(samples["sigma_t2"]))), 4),
        })
    return SpatialPoolingFit(
        candidates=candidates, anchors=anchors, tested_mask=arrays["tested_mask"],
        mu_draws=np.asarray(samples["mu"]).reshape(-1, N),
        sigma_draws=np.asarray(samples["sigma"]).reshape(-1, N),
        w_draws=np.asarray(samples["w_now"]).reshape(-1, N),
        diagnostics=diagnostics,
        kappa_draws=np.asarray(samples["kappa"]).reshape(-1) if has_t2 else None,
        eta_draws=np.asarray(samples["eta"]).reshape(-1, N) if has_t2 else None,
        level_draws=np.asarray(samples["w_level"]).reshape(-1, N),
        sigma_w_draws=np.asarray(samples["sigma_w"]).reshape(-1),
        dynamic=np.asarray(part >= seuil_dynamique), tau_ou=tau_ou, as_of=as_of,
        duel_pairs=sorted(set(duels["pairs"])) if has_t2 else [],
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


# --- Second tour : lecture d'un duel et P(élu) (spec_second_tour.md §6-§7) -------
# Erreur de scrutin des sondages de 2nd tour, en logit. RMS ~2,4 pt de part sur
# 2002-2022 (2017 +3,8 et 2022 +2,8 mesurés sur les données du dépôt ; 2002,
# 2007, 2012 de mémoire, À SOURCER), soit ~0,10 autour de 60/40. Centrée : un
# terme directionnel anti-RN est un choix éditorial non tranché (spec §6).
TAU_T2_SCRUTIN = 0.10


def projeter_w(fit: SpatialPoolingFit, horizon_days: float, seed: int = 27) -> np.ndarray:
    """`w` au jour du scrutin, en prolongeant le MÊME processus OU que le fit
    (spec §6.1) : retour vers le niveau `m_i` et diffusion saturante. Les
    candidats statiques gardent leur `w`. Approximation assumée : `m_i` est la
    moyenne du chemin sur la fenêtre (base KL centrée), pris pour le centre OU."""
    if horizon_days <= 0 or fit.level_draws is None:
        return fit.w_draws
    rng = np.random.default_rng(seed)
    r = np.exp(-horizon_days / fit.tau_ou)
    # sigma_w²(1-r²) si le centre OU était connu ; `m_i` n'en est qu'une
    # estimation (moyenne du chemin sur la fenêtre), d'incertitude ~sigma_w² :
    # on ajoute (1-r)²·sigma_w² plutôt que de sous-estimer la dérive de ~25 %.
    sd = fit.sigma_w_draws[:, None] * np.sqrt(1.0 - r ** 2 + (1.0 - r) ** 2)
    w = fit.level_draws + (fit.w_draws - fit.level_draws) * r + sd * rng.standard_normal(fit.w_draws.shape)
    return np.where(fit.dynamic[None, :], w, fit.w_draws)


def _duel_par_tirage(fit, w, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    """Part de `ia[s]` contre `ib[s]` au tirage s (indices par tirage)."""
    s_idx = np.arange(len(ia))
    A = np.asarray(log_attractivite(jnp.asarray(fit.mu_draws), jnp.asarray(fit.sigma_draws),
                                    jnp.asarray(w)))                     # (S, N, B)
    return np.asarray(duel_shares(A[s_idx, ia], A[s_idx, ib], fit.kappa_draws,
                                  fit.eta_draws[s_idx, ia], fit.eta_draws[s_idx, ib]))


def duel_draws(fit: SpatialPoolingFit, a: str, b: str, horizon_days: float = 0.0,
               scrutin: bool = True, seed: int = 27) -> np.ndarray:
    """(S,) part de `a` parmi les exprimés du duel a/b. `horizon_days` = jours
    d'ici au second tour ; `scrutin` ajoute l'erreur de scrutin T2."""
    if fit.kappa_draws is None:
        raise ValueError("fit sans duel : le second tour n'est pas identifié")
    i = {c: k for k, c in enumerate(fit.candidates)}
    S = len(fit.mu_draws)
    s = _duel_par_tirage(fit, projeter_w(fit, horizon_days, seed),
                         np.full(S, i[a]), np.full(S, i[b]))
    if scrutin:
        rng = np.random.default_rng(seed + 1)
        z = np.log(s / (1 - s)) + rng.normal(0.0, TAU_T2_SCRUTIN, S)
        s = 1.0 / (1.0 + np.exp(-z))
    return s


def resume_duel(fit: SpatialPoolingFit, a: str, b: str, horizon_days: float,
                n_sondages: int = 0) -> dict:
    d = duel_draws(fit, a, b, horizon_days)
    p = float((d > 0.5).mean())
    q = lambda x: [round(float(v), 4) for v in np.percentile(x, [5, 95])]   # noqa: E731
    return {"candidats": [a, b],
            "intentions": {a: {"moyenne": round(float(d.mean()), 4), "ic90": q(d), "p_gagne": round(p, 4)},
                           b: {"moyenne": round(float(1 - d.mean()), 4), "ic90": q(1 - d),
                               "p_gagne": round(1 - p, 4)}},
            "n_sondages_directs": int(n_sondages),
            "modele": "spatial joint T1+T2 : géométrie + valence de 2nd tour",
            "horizon_jours": int(horizon_days)}


def p_elu(fit: SpatialPoolingFit, mask: np.ndarray, horizon_t1: int, horizon_t2: int,
          seed: int = 27) -> dict:
    """P(élu) pour le champ `mask` (spec §7) : tirage par tirage, T1 projeté au
    scrutin -> deux premiers -> duel projeté au second tour -> vainqueur. Les
    deux étapes partagent le tirage de (mu, sigma, w, kappa, eta)."""
    from model.core.opinion import load_law
    from model.core.projection import projeter_au_scrutin
    if fit.kappa_draws is None:
        raise ValueError("fit sans duel : le second tour n'est pas identifié")
    m = mask.astype(bool)
    cols = np.flatnonzero(m)
    pi = projeter_au_scrutin(pi_draws_for_mask(fit, mask)[:, m], horizon_t1, load_law(), seed=seed)
    top = np.argsort(-pi, axis=1)[:, :2]
    ia, ib = cols[top[:, 0]], cols[top[:, 1]]
    s = _duel_par_tirage(fit, projeter_w(fit, horizon_t2, seed + 2), ia, ib)
    rng = np.random.default_rng(seed + 3)
    s = 1.0 / (1.0 + np.exp(-(np.log(s / (1 - s)) + rng.normal(0.0, TAU_T2_SCRUTIN, len(s)))))
    elu = np.where(s > 0.5, ia, ib)
    S = len(elu)
    paires: dict = {}
    for x, y in zip(np.minimum(ia, ib), np.maximum(ia, ib)):
        paires[(x, y)] = paires.get((x, y), 0) + 1
    return {"p_elu": {fit.candidates[k]: round(float((elu == k).mean()), 4) for k in cols},
            "duels_qualifies": [{"candidats": [fit.candidates[x], fit.candidates[y]],
                                 "p": round(c / S, 4)}
                                for (x, y), c in sorted(paires.items(), key=lambda t: -t[1])[:6]]}
