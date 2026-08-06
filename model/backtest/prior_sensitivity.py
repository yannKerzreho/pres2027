"""Sensibilité aux priors (mode prod, pas recherche).

Pour un jeu de configurations de priors, on refait le backtest hors-échantillon
(fit sur 2017 en √h, prédiction sur 2022) et on regarde bouger :
  - les biais têtes d'affiche  mu_bloc[droite], mu_bloc[droite_radicale] ;
  - tau_derive (amplitude de dérive, pilote la largeur des IC lointains) ;
  - la couverture 90 % hors-échantillon + le log-score.

Objectif : (a) vérifier que les biais sont robustes (dominés par les données,
pas par le prior) ; (b) tester si la sous-couverture du backtest vient de priors
de variance trop serrés — auquel cas les desserrer doit remonter la couverture.

Exécution : .venv/bin/python model/backtest/prior_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.historical import load_calibration_frame
from calibration.fit_house_effects import fit, export_priors
from backtest_loo import predict, coverage_report  # même dossier

# Configurations : nom -> surcharge de priors (le reste = défauts de référence).
CONFIGS = {
    "référence": {},
    "biais_large": {"mu_bloc_sd": 10.0, "tau_bloc_sd": 6.0},
    "biais_serré": {"mu_bloc_sd": 2.0, "tau_bloc_sd": 1.0},
    "dérive_large": {"tau_derive_sd": 2.5},
    "dérive_serrée": {"tau_derive_sd": 0.4},
    "excès_large": {"log_excess0_sd": 2.0, "b_h_sd": 2.0},
}


def evaluate(train, test, name, cfg):
    idata, meta = fit(train, horizon_form="sqrt", priors_cfg=cfg,
                      draws=500, tune=500, chains=2, seed=27, progressbar=False)
    priors = export_priors(idata, meta)
    known = set(priors["bloc"])
    tk = test[test["bloc"].isin(known)].reset_index(drop=True)
    pred = predict(priors, meta, tk)
    z = (tk["ecart"].to_numpy(float) - pred["mean"]) / pred["sd"]
    cov = coverage_report(z)
    logscore = float(np.mean(-0.5 * np.log(2 * np.pi * pred["sd"] ** 2) - 0.5 * z ** 2))
    return {
        "mu_droite": priors["bloc"]["droite"]["biais_moyen_population"]["mean"],
        "mu_dr": priors["bloc"]["droite_radicale"]["biais_moyen_population"]["mean"],
        "tau_derive": priors["variance_globale"]["tau_derive"]["mean"],
        "cov90": cov["90%"], "cov50": cov["50%"], "logscore": logscore,
    }


def main():
    df = load_calibration_frame()
    train = df[df.election == 2017].reset_index(drop=True)
    test = df[df.election == 2022].reset_index(drop=True)

    print(f"{'config':>14} | {'μ_droite':>8} | {'μ_dr':>6} | {'τ_derive':>8} | "
          f"{'cov50':>5} | {'cov90':>5} | {'logscore':>8}")
    print("-" * 78)
    for name, cfg in CONFIGS.items():
        r = evaluate(train, test, name, cfg)
        print(f"{name:>14} | {r['mu_droite']:8.2f} | {r['mu_dr']:6.2f} | "
              f"{r['tau_derive']:8.2f} | {r['cov50']:5.2f} | {r['cov90']:5.2f} | "
              f"{r['logscore']:8.3f}")
    print("\nLecture : biais stables entre configs => dominés par les données. "
          "cov90 qui monte avec des priors de variance plus larges => "
          "la sous-couverture était (en partie) un effet de priors trop serrés.")


if __name__ == "__main__":
    main()
