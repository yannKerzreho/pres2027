"""Comptage des probabilités à partir des tirages du jour du scrutin.

À partir de tirages `π` de la composition AU SCRUTIN (fournis par le modèle,
qui a déjà appliqué sa propre dérive), on compte par tirage :
  - la probabilité que chaque candidature soit **qualifiée** (top 2) / arrive 1re ;
  - la probabilité de chaque **duel** de 2nd tour.

Ce module ne modélise plus aucune dérive. C'était le cas avant : il proposait un
saut terminal paramétrique et un bootstrap empirique. Les mesures ont montré que
la loi correcte dépend du modèle qui l'emploie — la diffusion d'opinion et
l'écart sondages-urne sont deux quantités distinctes, calibrées séparément
(cf. `model/core/projection.py`). Laisser ce choix ici revenait à
imposer une stratégie de projection à tous les modèles.

Une dérive gaussienne isotrope reste disponible (`drift_sd`) comme repli pour un
modèle qui n'a pas encore calibré la sienne.

Le *gagnant* du 2nd tour n'est pas modélisé (matrice de reports) : on s'arrête
aux duels, honnêtement.
"""

from __future__ import annotations

import numpy as np


def _softmax(a: np.ndarray) -> np.ndarray:
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


# --- Résumés de forme, pour le survol des barres sur le site ---------------------
# Un IC 90 % dit où la masse se trouve, jamais COMMENT elle s'y répartit. Or les
# distributions publiées ici sont franchement asymétriques (softmax d'un latent
# gaussien + saut sinh-arcsinh à queue épaisse) : la moyenne peut sortir du
# triangle central, et l'écart moyenne/médiane est l'information la plus utile
# qu'on puisse donner au lecteur qui survole une barre. On exporte donc la forme
# elle-même, pas des paramètres qu'il faudrait supposer gaussiens.
N_BINS_DENSITE = 24        # assez pour la forme, ~250 octets par candidature


def quantiles(a: np.ndarray) -> list[float]:
    """[q05, q25, q50, q75, q95] — la boîte et les moustaches."""
    return [round(float(x), 4) for x in np.percentile(a, [5, 25, 50, 75, 95])]


def densite(a: np.ndarray) -> dict:
    """Histogramme normalisé sur [q0.5 %, q99.5 %] : `{x0, dx, y}`.

    `y` est entier sur 0-100 (hauteur relative au pic) : c'est un croquis de
    forme destiné à être dessiné, pas une densité à intégrer — l'arrondi divise
    la taille du JSON par trois sans qu'aucun pixel ne bouge. Les 1 % de queues
    exclus évitent qu'un seul tirage extrême n'écrase toute la figure.
    """
    lo, hi = np.percentile(a, [0.5, 99.5])
    if not np.isfinite(lo) or hi <= lo:
        return {"x0": round(float(lo), 4), "dx": 0.0, "y": []}
    y, bords = np.histogram(a, bins=N_BINS_DENSITE, range=(float(lo), float(hi)))
    pic = max(int(y.max()), 1)
    return {"x0": round(float(bords[0]), 4),
            "dx": round(float(bords[1] - bords[0]), 6),
            "y": [int(round(100 * v / pic)) for v in y]}


def forecast_from_draws(pi: np.ndarray, slots: list[str], forecast_horizon: int,
                        drift_sd: float = 0.0, rng_seed: int = 27) -> dict:
    """Comptage des probabilités. `pi` : tirages `(S, K)` de la composition au
    scrutin, dérive déjà appliquée par le modèle.

    `drift_sd > 0` ajoute une dérive gaussienne isotrope en log-ratio — repli
    pour un modèle sans loi propre, pas le chemin nominal."""
    rng = np.random.default_rng(rng_seed)
    S, K = pi.shape
    if drift_sd > 0:
        # dynamique sur alpha = log(pi) ∈ ℝ puis softmax → simplexe garanti.
        alpha = np.log(np.clip(pi, 1e-6, None))
        theta = _softmax(alpha + rng.normal(0.0, drift_sd, size=(S, K)))
        drift_mode = f"gaussien log-ratio (sd={drift_sd})"
    else:
        theta = pi
        drift_mode = "aucune (dérive appliquée par le modèle)"

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
                # : seule la moyenne
                # préserve Σ_c part_moyenne_c = 1 (linéarité de l'espérance).
                "part_moyenne": round(float(theta[:, k].mean()), 4),
                "ic90": band(theta[:, k]),
                "p_qualifie_top2": round(float(p_top2[k]), 4),
                "p_arrive_premier": round(float(p_first[k]), 4),
                "quantiles": quantiles(theta[:, k]),
                "densite": densite(theta[:, k]),
            } for k in range(K)
        },
        "duels_probables": duels[:8],
        "drift_sd_logratio": round(drift_sd, 3),
        "drift_modele": drift_mode,
    }
