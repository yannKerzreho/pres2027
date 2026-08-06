"""Consommation des priors calibrés (calibration/priors.json).

Utilitaires pour que le modèle live 2027 (Phase 3) lise les house effects sans
avoir à re-fitter : biais attendu d'un institut sur un bloc, et écart-type
d'erreur attendu à un horizon donné (avec sa décomposition échantillonnage /
excès / dérive future).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

PRIORS_PATH = Path(__file__).resolve().parent / "priors.json"


def load_priors(path: Path | None = None) -> dict:
    with open(path or PRIORS_PATH, encoding="utf-8") as f:
        return json.load(f)


def expected_bias(priors: dict, institut: str, bloc: str) -> dict:
    """Biais a posteriori (mean, sd, en points) d'un institut sur un bloc.

    Repli hiérarchique : si l'institut est inconnu (nouveau en 2027), on retourne
    le biais de population du bloc (`mu_bloc`) — comportement de *shrinkage*
    attendu, un institut sans historique reçoit la moyenne du groupe.
    """
    inst = priors["institut"].get(institut)
    if inst is None:
        pop = priors["bloc"][bloc]["biais_moyen_population"]
        return {"mean": pop["mean"], "sd": pop["sd"], "source": "population_bloc"}
    b = inst["biais_par_bloc"][bloc]
    return {"mean": b["mean"], "sd": b["sd"], "source": "institut"}


def _horizon_base(priors: dict, horizon: float) -> float:
    """base(horizon) selon la forme calibrée (sqrt par défaut, ou log), nulle à J-0."""
    if priors["meta"].get("horizon_form", "log") == "sqrt":
        return math.sqrt(horizon)
    return math.log1p(horizon)


def _z_horizon(priors: dict, horizon: int) -> float:
    meta = priors["meta"]
    return (_horizon_base(priors, horizon) - meta["log_h_mean"]) / meta["log_h_std"]


def excess_sigma(priors: dict, institut: str, horizon: int) -> float:
    """Écart-type d'*excès* (hors échantillonnage) : bruit propre à l'institut +
    volatilité croissante avec l'horizon. C'est la partie « house effect » de la
    dispersion, indépendante de la taille d'échantillon."""
    vg = priors["variance_globale"]
    inst = priors["institut"].get(institut)
    s_inst = inst["precision_generale_log"]["mean"] if inst else 0.0
    return math.exp(vg["log_excess0"]["mean"] + s_inst
                    + vg["b_horizon"]["mean"] * _z_horizon(priors, horizon))


def sampling_sigma(priors: dict, intention: float, n: int | None = None) -> float:
    """Écart-type d'échantillonnage d'un sondage : sqrt(deff * p(100-p)/n)."""
    n = n or priors["meta"]["median_n"]
    deff = priors["variance_globale"]["design_effect_quotas"]["mean"]
    return math.sqrt(deff * intention * (100.0 - intention) / n)


def observation_sigma(priors: dict, institut: str, horizon: int,
                      intention: float, n: int | None = None) -> float:
    """Écart-type total attendu d'un *sondage observé* : échantillonnage ⊕ excès.

    C'est la dispersion d'une intention publiée autour de la vérité, telle que
    calibrée sur l'historique.
    """
    return math.hypot(sampling_sigma(priors, intention, n),
                      excess_sigma(priors, institut, horizon))


def forecast_drift_sigma(priors: dict, horizon: int) -> float:
    """Incertitude a priori sur la dérive d'opinion 2027 à l'horizon h :
    tau_derive * base(h) (base = sqrt ou log selon la calibration). S'ajoute (en
    variance) aux IC de prévision et s'annule le jour du vote — d'où des IC qui se
    resserrent en approchant."""
    tau = priors["variance_globale"]["tau_derive"]["mean"]
    return tau * _horizon_base(priors, horizon)


if __name__ == "__main__":
    p = load_priors()
    print("Biais Ifop / droite_radicale :", expected_bias(p, "Ifop", "droite_radicale"))
    print(f"{'horizon':>8} | {'échant.':>8} | {'excès':>7} | {'obs tot':>7} | {'dérive2027':>10}")
    for h in (1, 15, 60, 180):
        print(f"J-{h:<6} | {sampling_sigma(p, 25.0, 1000):8.2f} | "
              f"{excess_sigma(p,'Ifop',h):7.2f} | "
              f"{observation_sigma(p,'Ifop',h,25.0,1000):7.2f} | "
              f"{forecast_drift_sigma(p,h):10.2f}")
