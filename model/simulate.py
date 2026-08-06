"""Phase 3 — simulation Monte Carlo : qualification 1er tour + duels 2nd tour.

Part du *nowcast* `π` (posterior du modèle bayésien) et ajoute l'incertitude
**irréductible de dérive d'opinion** d'ici au scrutin — commune à tous les
sondages, calibrée par `tau_derive` (Phase 1) : à l'horizon h,
`sd_dérive = tau_derive · √h` points par candidat. On obtient des tirages de la
répartition des voix **le jour du vote** `θ`, dont on déduit par comptage :

  - la probabilité que chaque candidature soit **qualifiée** (top 2) / arrive 1re ;
  - la probabilité de **chaque duel** possible au 2nd tour.

Le *gagnant* du 2nd tour n'est pas encore modélisé (il faut une matrice de report
de voix — étape suivante) : on s'arrête aux duels, honnêtement.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calibration.priors_utils import load_priors, _horizon_base


def _draws(idata) -> np.ndarray:
    p = idata.posterior["pi"]
    return p.stack(s=("chain", "draw")).transpose("s", "slot").to_numpy()  # (S, K)


def _softmax(a: np.ndarray) -> np.ndarray:
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


def simulate(idata, slots: list[str], priors: dict, forecast_horizon: int,
             rng_seed: int = 27, drift_pool: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(rng_seed)
    pi = _draws(idata)                                   # nowcast draws (S, K)
    S, K = pi.shape

    # --- dérive d'ici au scrutin, dans l'espace LOG-RATIO (softmax) : la dynamique
    #     évolue sur alpha = log(pi) ∈ ℝ, puis pi = softmax(alpha). Le simplexe est
    #     garanti (jamais de part < 0 ou > 1). Le mouvement est un bootstrap du
    #     drift RÉEL 2017/2022 mesuré en log-ratio (queues épaisses + saturation). ---
    alpha = np.log(np.clip(pi, 1e-6, None))
    if drift_pool is not None and len(drift_pool) >= 20:
        drift_mode = f"bootstrap log-ratio empirique (2017/2022, n={len(drift_pool)})"
        move = rng.choice(drift_pool, size=(S, K))       # mouvement en log-ratio
        drift_sd = float(np.std(drift_pool))
    else:  # repli gaussien en log-ratio
        drift_mode = "gaussien log-ratio (repli)"
        drift_sd = 0.6
        move = rng.normal(0.0, drift_sd, size=(S, K))
    theta = _softmax(alpha + move)                       # election-day draws (simplexe garanti)

    order = np.argsort(-theta, axis=1)
    first = order[:, 0]
    top2 = order[:, :2]

    p_first = np.bincount(first, minlength=K) / S
    p_top2 = np.array([(top2 == k).any(axis=1).mean() for k in range(K)])

    # duels (paires qualifiées)
    from collections import Counter
    duel_counts = Counter(tuple(sorted(pair)) for pair in top2)
    duels = sorted(
        ({"candidats": [slots[i], slots[j]], "probabilite": round(c / S, 4)}
         for (i, j), c in duel_counts.items()),
        key=lambda d: -d["probabilite"],
    )

    def band(a):  # 90 %
        return [round(float(np.percentile(a, 5)), 4), round(float(np.percentile(a, 95)), 4)]

    return {
        "nowcast": {slots[k]: {"mean": round(float(pi[:, k].mean()), 4),
                               "ic90": band(pi[:, k])} for k in range(K)},
        "forecast_scrutin": {
            slots[k]: {
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


def export(idata, slots, obs, priors, out_path: Path | None = None) -> dict:
    forecast_h = int(obs["horizon"].min())              # depuis le sondage le plus récent
    try:
        from model.drift_analysis import clr_movement_pool
        pool = clr_movement_pool(priors)
    except Exception:
        pool = None
    res = simulate(idata, slots, priors, forecast_h, drift_pool=pool)
    payload = {
        "meta": {
            "genere_le": date.today().isoformat(),
            "scrutin_t1": "2027-04-18",
            "n_sondages": int(obs["notice"].nunique()),
            "horizon_prevision_jours": forecast_h,
            "instituts": sorted(obs["institut"].unique().tolist()),
            "note": ("Parts au 1er tour par slot de candidature (alternatives "
                     "mutuellement exclusives fusionnées). Dérive appliquée en "
                     "espace log-ratio (softmax) → simplexe respecté ; mouvement "
                     "bootstrapé sur le drift réel 2017/2022. Probabilités = "
                     "probabilités, pas des prédictions."),
        },
        **res,
    }
    out_path = out_path or (ROOT / "model" / "resultats_2027.json")
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    from model.live_dataset import load_live_observations
    from model.bayesian_model import build_and_fit
    priors = load_priors()
    obs = load_live_observations()
    idata, slots, obs = build_and_fit(obs, priors)
    payload = export(idata, slots, obs, priors)

    print(f"=== Prévision 1er tour 2027 ({payload['drift_modele']}, "
          f"horizon {payload['meta']['horizon_prevision_jours']} j) ===")
    fc = payload["forecast_scrutin"]
    for s in sorted(fc, key=lambda s: -fc[s]["p_qualifie_top2"]):
        d = fc[s]
        print(f"  {s:22s} {100*d['part_moyenne']:4.1f}%  "
              f"P(top2)={d['p_qualifie_top2']:.0%}  P(1er)={d['p_arrive_premier']:.0%}")
    print("\nDuels de 2nd tour les plus probables :")
    for duel in payload["duels_probables"][:5]:
        print(f"  {' vs '.join(duel['candidats']):40s} {duel['probabilite']:.0%}")
