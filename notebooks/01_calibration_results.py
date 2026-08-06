"""Visualisation des résultats de calibration (Phase 1).

Lit calibration/priors.json et trace ce que le modèle a appris :
  1. biais_calibres.png    — house effect (biais à J-0) par bloc, avec IC 90 %
  2. ic_horizon.png        — « fan chart » : bande de prévision 90 % autour d'un
                             candidat, en fonction de l'horizon (jours avant le vote)
  3. variance_decomp.png   — décomposition de l'incertitude : échantillonnage /
                             excès house-effect / dérive future

Exécution : .venv/bin/python notebooks/01_calibration_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calibration.priors_utils import (
    load_priors, sampling_sigma, excess_sigma, forecast_drift_sigma,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
Z90 = 1.645  # quantile 95 % de la normale -> demi-bande à 90 %

# Candidat « type » pour les prévisions illustratives.
P_REF = 22.0       # intention (%)
N_REF = 1000       # taille d'échantillon
INST_REF = "Ifop"


def forecast_sigma(priors, institut, horizon, p=P_REF, n=N_REF):
    """Écart-type de prévision 2027 : échantillonnage ⊕ excès ⊕ dérive future."""
    return np.sqrt(
        sampling_sigma(priors, p, n) ** 2
        + excess_sigma(priors, institut, horizon) ** 2
        + forecast_drift_sigma(priors, horizon) ** 2
    )


def fig_biais(priors):
    blocs = list(priors["bloc"])
    m = np.array([priors["bloc"][b]["biais_moyen_population"]["mean"] for b in blocs])
    s = np.array([priors["bloc"][b]["biais_moyen_population"]["sd"] for b in blocs])
    order = np.argsort(m)
    blocs = [blocs[i] for i in order]; m = m[order]; s = s[order]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.errorbar(m, range(len(blocs)), xerr=Z90 * s, fmt="o", color="#2b6cb0",
                ecolor="#a0aec0", capsize=3)
    ax.axvline(0, color="#e53e3e", lw=1, ls="--")
    ax.set_yticks(range(len(blocs))); ax.set_yticklabels(blocs)
    ax.set_xlabel("Biais à J-0 (points) — intention − résultat")
    ax.set_title("House effect calibré par bloc (biais à J-0, IC 90 %)\n"
                 "après séparation de la dérive temporelle — 2017 + 2022")
    fig.tight_layout(); fig.savefig(FIG_DIR / "biais_calibres.png", dpi=110)
    plt.close(fig)


def fig_fan_chart(priors):
    h = np.arange(0, 400)
    sig = np.array([forecast_sigma(priors, INST_REF, x) for x in h])
    band = Z90 * sig

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    # x = jours avant le scrutin, scrutin (J-0) à droite
    ax.fill_between(h, P_REF - band, P_REF + band, color="#2b6cb0", alpha=0.20,
                    label="bande de prévision 90 %")
    ax.plot(h, np.full_like(h, P_REF, dtype=float), color="#2b6cb0", lw=1.5,
            label=f"intention « vraie » supposée ({P_REF:.0f} %)")
    ax.invert_xaxis()  # on approche du scrutin vers la droite
    ax.set_xlabel("Horizon (jours avant le 1er tour)")
    ax.set_ylabel("Intention (%)")
    ax.set_title(f"Courbe d'IC apprise — bande de prévision 90 % vs horizon\n"
                 f"(candidat à {P_REF:.0f} %, sondage {INST_REF} n={N_REF})")
    ax.legend(loc="upper right")
    for x in (7, 30, 90, 180, 365):
        b = Z90 * forecast_sigma(priors, INST_REF, x)
        ax.annotate(f"±{b:.1f}", (x, P_REF + b), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8, color="#2b6cb0")
    fig.tight_layout(); fig.savefig(FIG_DIR / "ic_horizon.png", dpi=110)
    plt.close(fig)


def fig_variance_decomp(priors):
    h = np.arange(0, 400)
    samp = np.full_like(h, sampling_sigma(priors, P_REF, N_REF), dtype=float)
    exc = np.array([excess_sigma(priors, INST_REF, x) for x in h])
    drift = np.array([forecast_drift_sigma(priors, x) for x in h])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(h, samp, label="échantillonnage  √(p(100−p)/n)", color="#718096", lw=2)
    ax.plot(h, exc, label="excès house-effect  (croît avec l'horizon)", color="#38a169", lw=2)
    ax.plot(h, drift, label="dérive future 2027  τ·log(1+h)", color="#d69e2e", lw=2)
    tot = np.sqrt(samp**2 + exc**2 + drift**2)
    ax.plot(h, tot, label="total (⊕ en quadrature)", color="#2b6cb0", lw=2.5, ls="--")
    ax.invert_xaxis()
    ax.set_xlabel("Horizon (jours avant le 1er tour)")
    ax.set_ylabel("Écart-type (points)")
    ax.set_title("Décomposition de l'incertitude de prévision par source")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(FIG_DIR / "variance_decomp.png", dpi=110)
    plt.close(fig)


def main():
    p = load_priors()
    print("Config calibration :", {k: p["meta"][k] for k in
          ("elections_utilisees", "n_obs", "rhat_max", "ess_bulk_min", "n_divergences")
          if k in p["meta"]})
    print(f"tau_derive = {p['variance_globale']['tau_derive']['mean']:.3f} "
          f"| b_horizon = {p['variance_globale']['b_horizon']['mean']:.3f} "
          f"| deff = {p['variance_globale']['design_effect_quotas']['mean']}")
    print("\nDemi-bande 90 % (candidat 22 %, Ifop n=1000) :")
    for x in (0, 7, 30, 90, 180, 365):
        print(f"  J-{x:<4} : ±{Z90*forecast_sigma(p, INST_REF, x):.2f} pts")

    fig_biais(p); fig_fan_chart(p); fig_variance_decomp(p)
    print(f"\nFigures écrites dans {FIG_DIR}/")


if __name__ == "__main__":
    main()
