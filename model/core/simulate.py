"""Utilitaire de prévision à la carte : dérive log-ratio + comptage des
probabilités. PAS un mécanisme automatique — un modèle qui veut cette
stratégie de dérive appelle `forecast_from_draws` explicitement depuis sa
propre méthode `forecast()` (cf. `model/models/bayesian_nowcast/nowcast.py`) ;
un autre modèle est libre de projeter son nowcast jusqu'au scrutin autrement.

À partir des tirages du *nowcast* `π` (fournis par n'importe quel modèle), on
ajoute la dérive d'opinion d'ici au scrutin **dans l'espace log-ratio** (softmax
→ simplexe garanti), puis on compte par tirage :
  - la probabilité que chaque candidature soit **qualifiée** (top 2) / arrive 1re ;
  - la probabilité de chaque **duel** de 2nd tour.

Trois modes de dérive, du plus au moins spécifique — le premier disponible
l'emporte :
  1. `jump_bank` + `candidate_blocs` : saut terminal paramétrique sinh-arcsinh
     (TerminalJumpCalibration, docs/spec_ssm_nowcast.md §2.2) — skew/tail
     globaux, loc/scale par bloc, remplace le bootstrap empirique d'origine.
  2. `drift_pool` : bootstrap log-ratio empirique (mode d'origine, gardé pour
     un modèle qui n'a pas encore de saut calibré).
  3. repli gaussien (`drift_sd`).

Le *gagnant* du 2nd tour n'est pas encore modélisé (matrice de reports) : on
s'arrête aux duels, honnêtement.
"""

from __future__ import annotations

import numpy as np
import jax

from model.core.utils import SinhArcsinh


def _softmax(a: np.ndarray) -> np.ndarray:
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


def _sinh_arcsinh_moves(S: int, slots: list[str], candidate_blocs: dict[str, str],
                        jump_bank, forecast_horizon: int, rng_seed: int
                        ) -> tuple[np.ndarray, str]:
    """Tire un mouvement (S,K) depuis le saut terminal sinh-arcsinh calibré :
    skew/tail globaux, loc/scale par bloc — POINT ESTIMATE (moyenne
    postérieure), même logique que `campaign_drift_sd` ailleurs dans le
    modèle (l'incertitude de calibration elle-même n'est pas propagée, comme
    pour les autres hyperparamètres dérivés de la Bank).

    Le fit (TerminalJumpCalibration) est calibré à un horizon de référence
    `jump_horizon_ref` (moyenne du movement pool 2017/2022) — le tirage brut
    est donc rescalé à l'horizon RÉEL de cette prévision par
    `sqrt(forecast_horizon / horizon_ref)` : un mouvement mesuré à J-400 et un
    mouvement à J-50 ne sont pas la même quantité (la dérive continue jusqu'au
    scrutin), même loi en racine du temps que `campaign_drift`/
    `horizon_diffusion` dans `calibration.py` (indépendamment reproduite ici
    plutôt qu'importée : ce module reste générique, `horizon_diffusion` est un
    choix du modèle `bayesian-nowcast`, pas une règle imposée par le
    framework)."""
    K = len(slots)
    skew, _ = jump_bank.jump_skew.item()
    tail, _ = jump_bank.jump_tail.item()
    horizon_ref, _ = jump_bank.jump_horizon_ref.item()
    loc = np.array([jump_bank.jump_loc.at(bloc=candidate_blocs[s])[0] for s in slots])
    scale = np.array([jump_bank.jump_scale.at(bloc=candidate_blocs[s])[0] for s in slots])

    d = SinhArcsinh(loc=loc, scale=scale, skewness=skew, tailweight=tail)
    move_at_href = np.asarray(d.sample(jax.random.PRNGKey(rng_seed), sample_shape=(S,)))
    move = move_at_href * np.sqrt(max(forecast_horizon, 0.0) / horizon_ref)
    return move, (f"sinh-arcsinh calibré (2017/2022, par bloc, K={K}, "
                 f"horizon_ref={horizon_ref:.0f}j -> {forecast_horizon}j)")


def forecast_from_draws(pi: np.ndarray, slots: list[str], forecast_horizon: int,
                        jump_bank=None, candidate_blocs: dict[str, str] | None = None,
                        drift_pool: np.ndarray | None = None,
                        drift_sd: float = 0.6, rng_seed: int = 27) -> dict:
    """Cœur PARTAGÉ de la prévision : dérive en espace log-ratio + comptage des
    probabilités. Réutilisé par tous les modèles (ils ne diffèrent que par le
    nowcast `pi` et le mode de dérive). `pi` : tirages du nowcast (S, K)."""
    rng = np.random.default_rng(rng_seed)
    S, K = pi.shape
    # dynamique sur alpha = log(pi) ∈ ℝ puis pi = softmax(alpha) → simplexe garanti.
    alpha = np.log(np.clip(pi, 1e-6, None))
    if jump_bank is not None and candidate_blocs is not None:
        move, drift_mode = _sinh_arcsinh_moves(S, slots, candidate_blocs, jump_bank,
                                               forecast_horizon, rng_seed)
        drift_sd = float(np.std(move))
    elif drift_pool is not None and len(drift_pool) >= 20:
        drift_mode = f"bootstrap log-ratio empirique (2017/2022, n={len(drift_pool)})"
        move = rng.choice(drift_pool, size=(S, K))
        drift_sd = float(np.std(drift_pool))
    else:
        drift_mode = f"gaussien log-ratio (sd={drift_sd})"
        move = rng.normal(0.0, drift_sd, size=(S, K))
    theta = _softmax(alpha + move)

    order = np.argsort(-theta, axis=1)
    p_first = np.bincount(order[:, 0], minlength=K) / S
    top2 = order[:, :2]
    p_top2 = np.array([(top2 == k).any(axis=1).mean() for k in range(K)])

    from collections import Counter
    duel_counts = Counter(tuple(sorted(pair)) for pair in top2)
    duels = sorted(
        ({"candidats": [slots[i], slots[j]], "probabilite": round(c / S, 4)}
         for (i, j), c in duel_counts.items()),
        key=lambda d: -d["probabilite"],
    )

    def band(a):
        return [round(float(np.percentile(a, 5)), 4), round(float(np.percentile(a, 95)), 4)]

    return {
        "forecast_scrutin": {
            slots[k]: {
                # Moyenne, pas médiane — cf. model/core/base.py:Nowcast.summary
                # (docs/spec_ssm_implementation.md §11) : seule la moyenne
                # préserve Σ_c part_moyenne_c = 1 (linéarité de l'espérance).
                "part_moyenne": round(float(theta[:, k].mean()), 4),
                "ic90": band(theta[:, k]),
                "p_qualifie_top2": round(float(p_top2[k]), 4),
                "p_arrive_premier": round(float(p_first[k]), 4),
            } for k in range(K)
        },
        "duels_probables": duels[:8],
        "drift_sd_logratio": round(drift_sd, 3),
        "drift_modele": drift_mode,
    }
