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
    omega_draws: np.ndarray | None = None    # (S,) abstention/blanc (None = désactivée)
    eta_draws: np.ndarray | None = None      # (S, N)
    level_draws: np.ndarray | None = None    # (S, N) niveau m_i, centre du processus OU
    sigma_w_draws: np.ndarray | None = None  # (S,)
    dynamic: np.ndarray | None = None        # (N,) bool, candidats à chemin OU
    tau_ou: float | None = None
    as_of: str | None = None
    duel_pairs: list | None = None           # paires effectivement observées
    masque_lse: np.ndarray | None = None     # (N,) champ de la valeur inclusive (§8.27)
    maison_draws: np.ndarray | None = None   # (S, H, N) effet maison u = σ_h·z (§8.23 bis)
    maisons: list | None = None              # noms d'instituts, ordre des H
    # Diagnostic : part prédite de chaque duel observé à sa date (S, D) et les
    # duels eux-mêmes (sortie de `build_duel_arrays`). Aucun usage en production.
    w_path_draws: np.ndarray | None = None   # (S, N, T) chemin de w (tracker, spec_spatial_pooling §10)
    w_dates: np.ndarray | None = None        # (T,) ordinaux des dates du chemin
    nu_draws: np.ndarray | None = None       # (S, N) carte 2D (spec_second_tour §8.44)
    lambda_draws: np.ndarray | None = None   # (S,)
    arc_draws: np.ndarray | None = None      # (S,) cercle (spec_second_tour §8.46)
    gamma_draws: np.ndarray | None = None    # (S,) abstention située (§8.48 bis)
    nu_t_draws: np.ndarray | None = None     # (S,) noyau de Student-t (§8.51)
    omega_k_draws: np.ndarray | None = None  # (S, N) abstention par électorat (§8.58)
    lam_mix_draws: np.ndarray | None = None  # (S,) mélange individuel λ (§8.61)
    reports_theta_draws: np.ndarray | None = None   # (S, R, 3) cellules prédites des reports en fit
    reports_rows: list | None = None         # [(notice, électorat, date)] dans l'ordre de R
    duel_s_draws: np.ndarray | None = None
    duels: dict | None = None


def fit_spatial_pooling(raw_polls: pd.DataFrame, as_of: str, draws: int = 400, tune: int = 600,
                        chains: int = 4, seed: int = 27, target_accept: float = 0.9,
                        excess_bank=None, pos_scale: float = POS_SCALE,
                        sigma_jitter: float = SIGMA_JITTER, min_gap: float = MIN_GAP_ANCRES,
                        seuil_dynamique: float = SEUIL_DYNAMIQUE,
                        min_poll_date=None, order_groups=None,
                        tau_ou: float | None = None,
                        couverture_min: float | None = COUVERTURE_MIN,
                        second_tour: bool = True, abstention: bool = True,
                        chain_method: str = "sequential",
                        mcmc_runner=None, eta_scale: float | None = None,
                        kappa_log_scale: float | None = None,
                        eta_hier: bool = False, effets_maison: bool = False,
                        effet_notice_seul: bool = False,
                        lse_champ: bool = False,
                        lse_officiel: list | None = None,
                        sigma_w_par_candidat: bool = False,
                        ancres_famille: bool = False,
                        carte_2d: bool = False, carte_cercle: bool = False, nu_ancre_val: float = 0.3,
                        abst_situee: bool = False, noyau_t: bool = False, sigma_lie: bool = False,
                        arc_ferme: bool = False, fidelite: bool = False, omega_electorat: bool = False,
                        melange_lambda: bool = False,
                        arc_fixe: float | None = None,
                        sigma_egaux: bool = False,
                        sigma_dynamiques: bool = False,
                        sigma_dynamiques_30j: bool = False,
                        eta_extremite: bool = False,
                        reports: pd.DataFrame | None = None,
                        champ_reports: list | None = None) -> SpatialPoolingFit:
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

    rep = None
    if reports is not None and has_t2:
        rep = arrays_reports(reports, candidates, as_of, champ_reports)
        if rep is not None:
            unique_dates = np.union1d(arrays["unique_dates"], rep["dates"])
            arrays["date_idx"] = np.searchsorted(unique_dates, np.floor(arrays["dates"]))
            duel_date_idx = np.searchsorted(unique_dates, np.floor(duels["dates"]))
            arrays["unique_dates"] = unique_dates
            rep["date_idx"] = np.searchsorted(unique_dates, rep["dates"])

    if ancres_famille:
        from .geometry import ancres_par_famille
        anchors = ancres_par_famille(candidates, order_groups, min_gap)
    else:
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
    if sigma_w_par_candidat:
        kwargs["sigma_w_par_candidat"] = True
    if fidelite:
        kwargs["fidelite"] = True                                      # R7 (§8.56)
    if omega_electorat:
        kwargs["omega_electorat"] = True                               # R8 (§8.58)
    if melange_lambda:
        kwargs["melange_lambda"] = True                                # R9 (§8.61)
    if sigma_dynamiques_30j:                                           # P4c (§8.87) : fenêtre de 30 j
        recent = (np.asarray(arrays["dates"], dtype=float) >= as_of_num - 30.0)[:, None] & tested
        part30 = np.array([arrays["Y"][recent[:, i], i].mean() if recent[:, i].any() else 0.0
                           for i in range(len(candidates))])
        kwargs["sigma_idx_libres"] = [int(i) for i in np.flatnonzero(part30 >= seuil_dynamique)]
    if sigma_dynamiques:                                               # P4b (§8.82)
        kwargs["sigma_idx_libres"] = [int(i) for i in np.flatnonzero(part >= seuil_dynamique)]
    if sigma_egaux:
        kwargs["z_sigma_fixe"] = jnp.zeros(len(candidates))            # P4 (§8.80 f) : σ_i ≡ σ_global
    if sigma_lie:
        # R5 (§8.53) : σ_Le Pen ≡ σ_Macron (noms de famille).
        nc = [_nom_court(c) for c in candidates]
        kwargs["sigma_lie"] = (nc.index("pen"), nc.index("macron"))
    if noyau_t:
        if carte_cercle or carte_2d:
            raise ValueError("noyau_t : segment seulement (§8.51)")
        kwargs["noyau_t"] = True
    if carte_cercle:
        kwargs["carte_cercle"] = True
        if arc_ferme:                                                  # R6 (§8.55) : A = 2π
            kwargs["arc_fixe"] = float(2 * np.pi)
        elif arc_fixe is not None:                                     # profil de l'arc (§8.70 d 4)
            kwargs["arc_fixe"] = float(arc_fixe)
        if abst_situee:
            kwargs["abst_situee"] = True
    if carte_2d:
        if not has_t2:
            raise ValueError("carte_2d : l'ancre et l'ordre en u sont posés sur les finalistes d'un duel")
        kwargs.update(carte_2d=True, nu_ancre_val=nu_ancre_val, **ancres_u(candidates, duels))
    if has_t2:
        kwargs.update(duel_a=jnp.asarray(duels["a"]), duel_b=jnp.asarray(duels["b"]),
                      duel_y=jnp.asarray(duels["y"]), duel_n=jnp.asarray(duels["n"]),
                      duel_date_idx=jnp.asarray(duel_date_idx), abstention=abstention)
        # Variantes de backtest (spec §8.9) : η ≡ 0 (`eta_scale=0`), κ ≡ 1.
        if eta_scale is not None:
            kwargs["eta_scale"] = eta_scale
        if kappa_log_scale is not None:
            kwargs["kappa_log_scale"] = kappa_log_scale
        if eta_extremite:                                              # §8.88
            kwargs["eta_extremite"] = True
        elif eta_hier:
            kwargs["eta_hier"] = True
        masque_lse = None
        noms_maisons = None
        if lse_champ and abstention:
            # §8.27 : valeur inclusive sur le champ modal du T1 (pas tout le roster).
            from collections import Counter
            tm = np.asarray(arrays["tested_mask"]) > 0
            modal = Counter(tuple(np.flatnonzero(r)) for r in tm).most_common(1)[0][0]
            masque_lse = np.full(len(candidates), 1e-30); masque_lse[list(modal)] = 1.0
            masque_lse[sorted(set(duels["a"]) | set(duels["b"]))] = 1.0   # les finalistes
            kwargs["masque_lse"] = jnp.asarray(masque_lse)
        if lse_officiel is not None and abstention:
            # §8.49 : valeur inclusive sur le champ OFFICIEL du T1 (backtest) ∪ finalistes,
            # sans les putatifs du roster dont le w n'est plus contraint.
            masque_lse = masque_lse_officiel(candidates, lse_officiel, duels)
            kwargs["masque_lse"] = jnp.asarray(masque_lse)
        if effets_maison or effet_notice_seul:
            kwargs.update(arrays_effets_maison(raw_t2, duels))
            noms_maisons = kwargs.pop("_noms_maisons")
            if effet_notice_seul:
                kwargs["n_maisons"] = 0
    if rep is not None:
        kwargs.update(rep_y=jnp.asarray(rep["y"]), rep_date_idx=jnp.asarray(rep["date_idx"]),
                      rep_k=jnp.asarray(rep["k"]), rep_a=jnp.asarray(rep["a"]), rep_b=jnp.asarray(rep["b"]),
                      rep_champ=jnp.asarray(rep["champ"]))
    t0 = time.time()
    # `mcmc_runner` : injectable (backtest découpé par chaîne, cf.
    # `model.backtest.second_round --chaine/--fusionner`), même signature que
    # `run_numpyro_mcmc`. La préparation est déterministe, donc une fusion
    # rejouée sur les mêmes données retrouve exactement les mêmes sites.
    samples, extra = (mcmc_runner or run_numpyro_mcmc)(spatial_pooling_model_ou, kwargs, draws=draws, tune=tune,
                                      chains=chains, seed=seed, target_accept=target_accept,
                                      extra_fields=("diverging", "num_steps"),
                                      chain_method=chain_method)
    elapsed = time.time() - t0

    N = len(candidates)
    # `pi` est un déterministe (P, N) : son R-hat n'ajoute rien à celui des
    # paramètres et écraserait le pire cas. On le retire du diagnostic.
    # `eta` ≡ 0 en variante sans valence (spec §8.9) : R-hat indéfini (NaN), exclu.
    diag = {k: v for k, v in numpyro_summary(samples, prob=0.9).items()
            if k not in ("pi", "duel_s", "w_path", "reports_theta") and not (k == "eta" and eta_scale == 0)}
    if has_t2:
        # `eta` d'un candidat sans duel suit son prior : R-hat valable, mais on
        # le distingue dans le diagnostic pour ne pas le confondre avec un
        # paramètre contraint.
        en_duel = sorted(set(duels["a"]) | set(duels["b"]))
        # R-hat PROPRE au bloc T2, sur les seuls eta contraints par un duel :
        # le `rhat_max` global est celui du T1 et ne garde rien du T2.
        rhat_t2 = max([float(np.max(diag[k]["r_hat"])) for k in ("kappa", "sigma_t2")]
                      + ([float(np.max(np.asarray(diag["z_eta"]["r_hat"])[en_duel]))]
                         if eta_scale != 0 and "z_eta" in diag else [])
                      + ([float(diag["lam_ext"]["r_hat"])] if "lam_ext" in diag else [])
                      + ([float(diag["omega"]["r_hat"])] if abstention else [])
                      # revue 2026-11-15 : l'échelle apprise est le site fragile de `hier`
                      + ([float(diag["s_eta"]["r_hat"])] if "s_eta" in diag else [])
                      + [float(diag[k]["r_hat"]) for k in ("sigma_maison_t2", "sigma_notice_t2") if k in diag])
    pires = sorted(((float(np.nanmax(v["r_hat"])), k) for k, v in diag.items()), reverse=True)[:3]
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
        "ess_min": round(min(float(np.nanmin(v["n_eff"])) for v in diag.values()), 1),
        "n_divergences": int(np.sum(extra["diverging"])),
        # Le temps vaut `(tirages + warmup) x pas x chaînes` : c'est la mesure
        # qui dit si un changement a amélioré le CONDITIONNEMENT ou seulement le
        # coût par pas. 255 = profondeur 8 pleine, 1023 = plafond saturé.
        "pas_leapfrog_median": int(np.median(steps)),
        "sigma_w": round(float(np.mean(np.asarray(samples["sigma_w"]))), 4),
        "sigma_global": round(float(np.mean(np.asarray(samples["sigma_global"]))), 4),
        # Garde-fou de quadrature (spec_second_tour §8.48) : diagnostic seulement.
        "sigma_min": round(float(np.median(np.asarray(samples["sigma"]).reshape(-1, N), 0).min()), 4),
        "quadrature_fragile": bool(np.median(np.asarray(samples["sigma"]).reshape(-1, N), 0).min() < 0.035),
    }
    if has_t2:
        diagnostics.update({
            "n_duels": int(len(duels["y"])),
            "paires_duel": sorted({f"{x} / {y}" for x, y in duels["pairs"]}),
            "candidats_en_duel": [candidates[i] for i in en_duel],
            "kappa": round(float(np.mean(np.asarray(samples["kappa"]))), 4),
            "omega": (round(float(np.mean(np.asarray(samples["omega"]))), 4)
                      if abstention else None),
            "rhat_t2": round(rhat_t2, 4),
            "sigma_t2": round(float(np.mean(np.asarray(samples["sigma_t2"]))), 4),
        })
        for k in ("sigma_maison_t2", "sigma_notice_t2"):           # §8.23 bis
            if k in samples:
                diagnostics[k] = round(float(np.mean(np.asarray(samples[k]))), 4)
    return SpatialPoolingFit(
        candidates=candidates, anchors=anchors, tested_mask=arrays["tested_mask"],
        mu_draws=np.asarray(samples["mu"]).reshape(-1, N),
        sigma_draws=np.asarray(samples["sigma"]).reshape(-1, N),
        w_draws=np.asarray(samples["w_now"]).reshape(-1, N),
        diagnostics=diagnostics,
        kappa_draws=np.asarray(samples["kappa"]).reshape(-1) if has_t2 else None,
        omega_draws=(np.asarray(samples["omega"]).reshape(-1) if has_t2 and abstention else None),
        eta_draws=np.asarray(samples["eta"]).reshape(-1, N) if has_t2 else None,
        level_draws=np.asarray(samples["w_level"]).reshape(-1, N),
        # (S,) commun, ou (S, N) par candidat (piste C) : `projeter_w` diffuse les deux.
        sigma_w_draws=(np.asarray(samples["sigma_w_i"]).reshape(-1, N) if "sigma_w_i" in samples
                       else np.asarray(samples["sigma_w"]).reshape(-1)),
        dynamic=np.asarray(part >= seuil_dynamique), tau_ou=tau_ou, as_of=as_of,
        duel_pairs=sorted(set(duels["pairs"])) if has_t2 else [],
        masque_lse=masque_lse if has_t2 else None,
        maison_draws=(np.asarray(samples["z_maison_t2"]).reshape(-1, *np.asarray(samples["z_maison_t2"]).shape[-2:])
                      * np.asarray(samples["sigma_maison_t2"]).reshape(-1, 1, 1)
                      if "z_maison_t2" in samples else None),
        maisons=noms_maisons if (has_t2 and (effets_maison or effet_notice_seul)) else None,
        w_path_draws=(np.asarray(samples["w_path"]).reshape(-1, N, np.asarray(samples["w_path"]).shape[-1])
                      if "w_path" in samples else None),
        # La base OU a une colonne de plus que les dates de sondage : `as_of`.
        w_dates=np.concatenate([np.asarray(arrays["unique_dates"], dtype=float), [as_of_num]]),
        duel_s_draws=(np.asarray(samples["duel_s"]).reshape(-1, len(duels["y"]))
                      if has_t2 else None),
        duels=duels if has_t2 else None,
        nu_draws=np.asarray(samples["nu"]).reshape(-1, N) if "nu" in samples else None,
        lambda_draws=np.asarray(samples["lambda_u"]).reshape(-1) if "lambda_u" in samples else None,
        arc_draws=np.asarray(samples["arc"]).reshape(-1) if "arc" in samples else None,
        gamma_draws=np.asarray(samples["gamma_abs"]).reshape(-1) if "gamma_abs" in samples else None,
        nu_t_draws=np.asarray(samples["nu_t"]).reshape(-1) if "nu_t" in samples else None,
        omega_k_draws=(np.asarray(samples["omega_k"]).reshape(-1, N) if "omega_k" in samples else None),
        lam_mix_draws=np.asarray(samples["lam_mix"]).reshape(-1) if "lam_mix" in samples else None,
        reports_theta_draws=(np.asarray(samples["reports_theta"]).reshape(-1, len(rep["y"]), 3)
                             if "reports_theta" in samples else None),
        reports_rows=rep["rows"] if rep is not None else None,
    )


def _nom_court(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()
    return s.split()[-1] if s else s


def masque_lse_officiel(candidates: list[str], champ: list[str], duels: dict) -> np.ndarray:
    """§8.49 : 1 sur le champ officiel du T1 ∪ finalistes des duels observés, 1e-30 ailleurs."""
    noms = {_nom_court(c): i for i, c in enumerate(candidates)}
    complets = {c: i for i, c in enumerate(candidates)}
    m = np.full(len(candidates), 1e-30)
    for c in champ:
        i = complets.get(c, noms.get(_nom_court(c)))
        if i is not None:
            m[i] = 1.0
    m[sorted(set(np.asarray(duels["a"]).tolist()) | set(np.asarray(duels["b"]).tolist()))] = 1.0
    return m


def ancres_u(candidates: list[str], duels: dict) -> dict:
    """Ancres de la carte 2D (spec §8.44.3) : C = finaliste centriste (Macron, ou à
    défaut le candidat en duel le plus fréquent hors RN), RN = Le Pen / Bardella."""
    from collections import Counter
    noms = [_nom_court(c) for c in candidates]
    rn = next((i for i, n in enumerate(noms) if n in ("pen", "bardella")), None)
    freq = Counter(list(np.asarray(duels["a"])) + list(np.asarray(duels["b"])))
    c = next((i for i, n in enumerate(noms) if n == "macron"), None)
    if c is None:
        c = next((int(i) for i, _ in freq.most_common() if int(i) != rn), None)
    return {"nu_ancre": c, "nu_ordre": rn}


def arrays_reports(reports: pd.DataFrame, candidates: list[str], as_of: str,
                   champ_reports: list | None) -> dict | None:
    """Tableaux de reports (data/parsed/reports_t2_*.csv) -> observations de la
    vraisemblance Dirichlet (spec §8.44.4). Une ligne par (notice, électorat,
    HYPOTHÈSE) datée au plus `as_of`, cellules (a, b, aucun) renormalisées, plancher
    0,5 %. §8.50 : une notice d'avant T1 teste plusieurs duels ; les grouper par
    (notice, électorat) seulement mélangeait leurs cellules. Champ PAR LIGNE : champ
    officiel ∪ {a, b, k} de la ligne (un finaliste hypothétique hors champ n'entre que
    dans sa propre ligne)."""
    noms = {_nom_court(c): i for i, c in enumerate(candidates)}
    complets = {c: i for i, c in enumerate(candidates)}
    r = reports[pd.to_datetime(reports["date_fin"]) <= pd.Timestamp(as_of)]
    if r.empty:
        return None
    champ0 = np.zeros(len(candidates))
    for c in (champ_reports or []):
        i = complets.get(c, noms.get(_nom_court(c)))
        if i is not None:
            champ0[i] = 1.0
    Y, K, A, Bf, D, rows, CH = [], [], [], [], [], [], []
    for (notice, elec, _hyp), g in r.groupby(["notice", "electorat_t1", "hypothese"], sort=False):
        a_lab, b_lab = [x.strip() for x in str(g["hypothese"].iloc[0]).replace("—", "–").split("–")]
        ia, ib, k = noms.get(_nom_court(a_lab)), noms.get(_nom_court(b_lab)), complets.get(elec, noms.get(_nom_court(elec)))
        parts = dict(zip(g["candidat_t2"], g["part"].astype(float)))
        if None in (ia, ib, k) or not {a_lab, b_lab, "Aucun"} <= set(parts):
            continue
        y = np.maximum(np.array([parts[a_lab], parts[b_lab], parts["Aucun"]]), 0.5)
        Y.append(y / y.sum()); K.append(k); A.append(ia); Bf.append(ib)
        D.append(float(pd.Timestamp(g["date_fin"].iloc[0]).toordinal()))
        rows.append((notice, elec, str(g["date_fin"].iloc[0]), _hyp))
        ch = champ0.copy()
        ch[[ia, ib, k]] = 1.0
        CH.append(ch)
    if not Y:
        return None
    CH = np.array(CH)
    if not champ_reports:
        # Pas de champ officiel (ex. 2027) : repli sur l'union des candidats des reports,
        # comme avant §8.50, appliquée à toutes les lignes.
        CH = np.repeat(CH.max(axis=0, keepdims=True), len(CH), axis=0)
    return {"y": np.array(Y), "k": np.array(K), "a": np.array(A), "b": np.array(Bf),
            "dates": np.floor(np.array(D)), "champ": CH, "rows": rows}

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
    if fit.arc_draws is not None:                                         # cercle (§8.46)
        from .geometry import spatial_shares_cercle
        return np.asarray(spatial_shares_cercle(jnp.asarray(fit.mu_draws), jnp.asarray(fit.sigma_draws),
                                                jnp.asarray(fit.arc_draws), jnp.asarray(fit.w_draws),
                                                jnp.asarray(mask)))
    if fit.nu_draws is not None:                                          # carte 2D (§8.44)
        from .geometry import spatial_shares_2d
        return np.asarray(spatial_shares_2d(jnp.asarray(fit.mu_draws), jnp.asarray(fit.sigma_draws),
                                            jnp.asarray(fit.nu_draws), jnp.asarray(fit.lambda_draws),
                                            jnp.asarray(fit.w_draws), jnp.asarray(mask)))
    return np.asarray(spatial_shares(jnp.asarray(fit.mu_draws), jnp.asarray(fit.sigma_draws),
                                     jnp.asarray(fit.w_draws), jnp.asarray(mask), nu=_nu(fit)))


def _nu(fit):
    """Degrés du noyau de Student-t par tirage (§8.51), ou None (gaussien)."""
    v = getattr(fit, "nu_t_draws", None)
    return None if v is None else jnp.asarray(v)


def trajectoire(fit: SpatialPoolingFit, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tracker (spec_spatial_pooling §10) : parts du champ `mask` à chaque date du
    chemin de `w`, tirage par tirage. Retourne (dates ordinales (T,), (S, T, |M|))."""
    if fit.w_path_draws is None:
        raise ValueError("fit sans chemin de w (w_path absent)")
    m = jnp.asarray(mask)
    mu, sg = jnp.asarray(fit.mu_draws), jnp.asarray(fit.sigma_draws)
    T = fit.w_path_draws.shape[-1]
    parts = [np.asarray(spatial_shares(mu, sg, jnp.asarray(fit.w_path_draws[:, :, t]), m, nu=_nu(fit)))
             for t in range(T)]
    return fit.w_dates, np.stack(parts, axis=1)[:, :, np.asarray(mask).astype(bool)]


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
# Erreur de scrutin des sondages de 2nd tour, en logit (spec_second_tour §6).
# 0,19 : décision de l'utilisateur du 2026-11-15 (arbitrage 1 ; RMS ≈ 4,3 pt
# mesuré 2017/2022). Centrée : pas de terme anti-RN (arbitrage 2).
TAU_T2_SCRUTIN = 0.19
# Incertitude d'extrapolation d'un duel JAMAIS sondé, en logit (spec §6.3,
# arbitrage 4). 0,10 : borne basse de la fourchette 0,10-0,15 validée par
# l'utilisateur (le calibrage hier donnait 0,08, hors fourchette : revue du
# 18/11, §8.34) ; en attente de sa décision.
TAU_PAIRE = 0.10


def noyau_effets(a, b, groupes) -> np.ndarray:
    """K = Z Zᵀ des effets par (groupe, candidat) sur la part du premier nommé
    (spec_second_tour §8.23 bis) : K_dd' = [g=g'](1[a=a'] − 1[a=b'] − 1[b=a'] + 1[b=b'])."""
    a, b = np.asarray(a), np.asarray(b)
    g = np.asarray(groupes)
    meme = (g[:, None] == g[None, :]).astype(float)
    eq = lambda x, y: (x[:, None] == y[None, :]).astype(float)   # noqa: E731
    return meme * (eq(a, a) - eq(a, b) - eq(b, a) + eq(b, b))


def arrays_effets_maison(raw_t2: pd.DataFrame, duels: dict) -> dict:
    """Entrées de la forme par blocs de §8.23 bis : indice d'institut par duel,
    blocs de duels par notice (complétés par −1) et noyau K_n de chaque bloc."""
    inst = (raw_t2.drop_duplicates("notice").set_index("notice")["institut"]
            if "institut" in raw_t2 else pd.Series(dtype=str))
    noms = [str(inst.get(nt, nt)) for nt in duels["notices"]]
    codes = {h: k for k, h in enumerate(sorted(set(noms)))}
    notices = [str(x) for x in duels["notices"]]
    groupes: dict = {}
    for d, nt in enumerate(notices):
        groupes.setdefault(nt, []).append(d)
    kmax = max(len(g) for g in groupes.values())
    blocs = np.full((len(groupes), kmax), -1, dtype=int)
    Kb = np.zeros((len(groupes), kmax, kmax))
    a, b = np.asarray(duels["a"]), np.asarray(duels["b"])
    for j, g in enumerate(groupes.values()):
        blocs[j, :len(g)] = g
        Kb[j, :len(g), :len(g)] = noyau_effets(a[g], b[g], ["x"] * len(g))
    return {"duel_maison": jnp.asarray(np.array([codes[h] for h in noms])), "n_maisons": len(codes),
            "_noms_maisons": sorted(codes),
            "duel_blocs": jnp.asarray(blocs), "duel_K_blocs": jnp.asarray(Kb)}


def paire_sondee(fit: SpatialPoolingFit, a: str, b: str) -> bool:
    return any({a, b} == set(p) for p in (fit.duel_pairs or []))


def projeter_w(fit: SpatialPoolingFit, horizon_days: float, seed: int = 27,
               w0: np.ndarray | None = None) -> np.ndarray:
    """`w` au jour du scrutin, en prolongeant le MÊME processus OU que le fit
    (spec §6.1) : retour vers le niveau `m_i` et diffusion saturante. Les
    candidats statiques gardent leur `w`. Approximation assumée : `m_i` est la
    moyenne du chemin sur la fenêtre (base KL centrée), pris pour le centre OU."""
    w0 = fit.w_draws if w0 is None else w0          # w0 : départ (§7.1, prolongement)
    if horizon_days <= 0 or fit.level_draws is None:
        return w0
    rng = np.random.default_rng(seed)
    r = np.exp(-horizon_days / fit.tau_ou)
    # sigma_w²(1-r²) si le centre OU était connu ; `m_i` n'en est qu'une
    # estimation (moyenne du chemin sur la fenêtre), d'incertitude ~sigma_w² :
    # on ajoute (1-r)²·sigma_w² plutôt que de sous-estimer la dérive de ~25 %.
    swd = fit.sigma_w_draws if fit.sigma_w_draws.ndim == 2 else fit.sigma_w_draws[:, None]
    sd = swd * np.sqrt(1.0 - r ** 2 + (1.0 - r) ** 2)
    w = fit.level_draws + (w0 - fit.level_draws) * r + sd * rng.standard_normal(w0.shape)
    return np.where(fit.dynamic[None, :], w, w0)


def _lam(fit):
    """§8.61 (R9) : tirages de λ, ou None."""
    v = getattr(fit, "lam_mix_draws", None)
    return None if v is None else jnp.asarray(v)


def _masque_lse(fit, ia, ib, lse_finalistes: bool):
    """(S, N) ou (N,) : masque du lse. `lse_finalistes` (§8.113 e1, défaut False, inchangé) : les deux finalistes
    du duel lu entrent toujours dans le lse, comme dans `cellules()` (un candidat sans duel au fit, ex. Hollande
    en V2, en sortait)."""
    if fit.masque_lse is None or not lse_finalistes:
        return None if fit.masque_lse is None else np.asarray(fit.masque_lse, dtype=float)[None, :]
    ml = np.tile(np.asarray(fit.masque_lse, dtype=float), (len(ia), 1))
    r = np.arange(len(ia))
    ml[r, ia] = ml[r, ib] = 1.0
    return ml


def _duel_par_tirage(fit, w, ia: np.ndarray, ib: np.ndarray, lse_finalistes: bool = False) -> np.ndarray:
    """Part de `ia[s]` contre `ib[s]` au tirage s (indices par tirage)."""
    s_idx = np.arange(len(ia))
    if lse_finalistes and getattr(fit, "omega_k_draws", None) is not None:   # §8.115 : ρ_k lirait encore l'ancien masque
        raise NotImplementedError("lse_finalistes avec ω par électorat (R8) : non pris en charge")
    ml = _masque_lse(fit, np.asarray(ia), np.asarray(ib), lse_finalistes)
    if fit.nu_draws is not None or fit.arc_draws is not None:            # carte 2D (§8.44) / cercle (§8.46)
        from .geometry import W2, W_THETA, duel_shares_poids, log_attractivite_2d, log_attractivite_cercle
        if fit.nu_draws is not None:
            A = np.asarray(log_attractivite_2d(jnp.asarray(fit.mu_draws), jnp.asarray(fit.sigma_draws),
                                               jnp.asarray(fit.nu_draws), jnp.asarray(fit.lambda_draws),
                                               jnp.asarray(w)))
            pq = W2
        else:
            A = np.asarray(log_attractivite_cercle(jnp.asarray(fit.mu_draws), jnp.asarray(fit.sigma_draws),
                                                   jnp.asarray(fit.arc_draws), jnp.asarray(w)))
            pq = W_THETA
        lse = None
        if fit.omega_draws is not None:
            from scipy.special import logsumexp
            Am = A if ml is None else A + np.log(ml)[:, :, None]
            lse = logsumexp(Am, axis=1)
        og = None
        if getattr(fit, "gamma_draws", None) is not None:                 # §8.48 bis
            from .geometry import THETA
            og = -fit.gamma_draws[:, None] * np.cos(np.asarray(THETA))[None, :]
        rho = None
        if getattr(fit, "omega_k_draws", None) is not None and lse is not None:   # §8.58 (R8)
            from .geometry import rho_electorats
            rn, pn = rho_electorats(jnp.asarray(A), jnp.asarray(A[s_idx, ia]), jnp.asarray(A[s_idx, ib]),
                                    jnp.asarray(fit.kappa_draws), jnp.asarray(lse),
                                    jnp.asarray(fit.omega_k_draws), fit.masque_lse)
            rho = (pn * rn).sum(-2)
        return np.asarray(duel_shares_poids(A[s_idx, ia], A[s_idx, ib], fit.kappa_draws,
                                            fit.eta_draws[s_idx, ia], fit.eta_draws[s_idx, ib],
                                            lse, fit.omega_draws, pq, og, rho=rho, lam=_lam(fit)))
    A = np.asarray(log_attractivite(jnp.asarray(fit.mu_draws), jnp.asarray(fit.sigma_draws),
                                    jnp.asarray(w), _nu(fit)))           # (S, N, B)
    extra = ()
    if fit.omega_draws is not None:
        from scipy.special import logsumexp
        Am = A if ml is None else A + np.log(ml)[:, :, None]
        extra = (logsumexp(Am, axis=1), fit.omega_draws)
        if getattr(fit, "omega_k_draws", None) is not None:                # §8.58 (R8)
            from .geometry import W, duel_shares_poids, rho_electorats
            rn, pn = rho_electorats(jnp.asarray(A), jnp.asarray(A[s_idx, ia]), jnp.asarray(A[s_idx, ib]),
                                    jnp.asarray(fit.kappa_draws), jnp.asarray(extra[0]),
                                    jnp.asarray(fit.omega_k_draws), fit.masque_lse)
            return np.asarray(duel_shares_poids(A[s_idx, ia], A[s_idx, ib], fit.kappa_draws,
                                                fit.eta_draws[s_idx, ia], fit.eta_draws[s_idx, ib],
                                                extra[0], fit.omega_draws, W, rho=(pn * rn).sum(-2),
                                                lam=_lam(fit)))
    return np.asarray(duel_shares(A[s_idx, ia], A[s_idx, ib], fit.kappa_draws,
                                  fit.eta_draws[s_idx, ia], fit.eta_draws[s_idx, ib], *extra, lam=_lam(fit)))


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
        # Prédiction (pas la cible d'un backtest) : + extrapolation si la paire
        # n'a jamais été sondée (§6.3), puis erreur de scrutin (§6.2).
        rng = np.random.default_rng(seed + 1)
        z = np.log(s / (1 - s)) + rng.normal(0.0, TAU_T2_SCRUTIN, S)
        if not paire_sondee(fit, a, b):
            z = z + np.random.default_rng(seed + 4).normal(0.0, TAU_PAIRE, S)
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
          seed: int = 27, alea_unique: bool = False, tau_paire: float | None = None,
          decalage_rn_pt: float = 0.0, rn: tuple = ("Le Pen", "Bardella"),
          tau_t2: float | None = None) -> dict:
    """P(élu) pour le champ `mask` (spec §7) : tirage par tirage, T1 projeté au
    scrutin -> deux premiers -> duel projeté au second tour -> vainqueur. Les
    deux étapes partagent le tirage de (mu, sigma, w, kappa, eta)."""
    from model.core.opinion import load_law
    from model.core.projection import projeter_au_scrutin
    if fit.kappa_draws is None:
        raise ValueError("fit sans duel : le second tour n'est pas identifié")
    m = mask.astype(bool)
    cols = np.flatnonzero(m)
    if alea_unique:
        # §7.1 : un seul chemin w — T1 projeté par l'OU du modèle, + δ, puis
        # prolongement conditionnel jusqu'au T2.
        from model.core.projection import delta_draws
        from .geometry import spatial_shares
        w1 = projeter_w(fit, horizon_t1, seed)
        p1 = np.asarray(spatial_shares(jnp.asarray(fit.mu_draws), jnp.asarray(fit.sigma_draws),
                                       jnp.asarray(w1), jnp.asarray(mask.astype(float)), nu=_nu(fit)))[:, m]
        a = np.log(np.clip(p1, 1e-9, None)) + delta_draws(load_law(), p1.shape, seed + 1)
        pi = np.exp(a - a.max(1, keepdims=True))
        w2 = projeter_w(fit, max(horizon_t2 - horizon_t1, 0), seed + 2, w0=w1)
    else:
        pi = projeter_au_scrutin(pi_draws_for_mask(fit, mask)[:, m], horizon_t1, load_law(), seed=seed)
        w2 = projeter_w(fit, horizon_t2, seed + 2)
    top = np.argsort(-pi, axis=1)[:, :2]
    ia, ib = cols[top[:, 0]], cols[top[:, 1]]
    s = _duel_par_tirage(fit, w2, ia, ib)
    rng = np.random.default_rng(seed + 3)
    # `tau_t2` : sensibilité (§8.36) — τ résiduel quand le biais moyen est retiré.
    z = np.log(s / (1 - s)) + rng.normal(0.0, TAU_T2_SCRUTIN if tau_t2 is None else tau_t2, len(s))
    sondees = {frozenset(p) for p in (fit.duel_pairs or [])}
    jamais = np.array([frozenset((fit.candidates[x], fit.candidates[y])) not in sondees
                       for x, y in zip(ia, ib)])
    tp = TAU_PAIRE if tau_paire is None else tau_paire
    z = z + jamais * np.random.default_rng(seed + 4).normal(0.0, tp, len(s))
    if decalage_rn_pt:
        # Sensibilité (§8.33), PAS un réglage : erreur de sondage anti-RN de
        # `decalage_rn_pt` points de duel (≈ pt / 24,75 en logit autour de 45 %),
        # signée du point de vue du candidat RN.
        est_rn_a = np.array([fit.candidates[x] in rn for x in ia])
        est_rn_b = np.array([fit.candidates[x] in rn for x in ib])
        z = z + (decalage_rn_pt / 24.75) * (est_rn_a.astype(float) - est_rn_b.astype(float))
    s = 1.0 / (1.0 + np.exp(-z))
    elu = np.where(s > 0.5, ia, ib)
    S = len(elu)
    paires: dict = {}
    for x, y in zip(np.minimum(ia, ib), np.maximum(ia, ib)):
        paires[(x, y)] = paires.get((x, y), 0) + 1
    return {"p_elu": {fit.candidates[k]: round(float((elu == k).mean()), 4) for k in cols},
            "duels_qualifies": [{"candidats": [fit.candidates[x], fit.candidates[y]],
                                 "p": round(c / S, 4)}
                                for (x, y), c in sorted(paires.items(), key=lambda t: -t[1])[:6]]}
