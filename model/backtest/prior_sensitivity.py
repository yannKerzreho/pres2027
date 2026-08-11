"""Sensibilité aux priors (mode prod, pas recherche).

Pour un jeu de configurations de priors (`house_effects_model.priors_cfg`), on
refait le backtest hors-échantillon (fit sur 2017, prédiction sur 2022) et on
regarde bouger :
  - les biais têtes d'affiche  bloc_bias_mean[droite], bloc_bias_mean[droite_radicale] ;
  - campaign_drift_sd (amplitude de dérive, pilote la largeur des IC lointains) ;
  - la couverture 90 % hors-échantillon + le log-score.

Objectif : (a) vérifier que les biais sont robustes (dominés par les données,
pas par le prior) ; (b) tester si la sous-couverture du backtest vient de priors
de variance trop serrés — auquel cas les desserrer doit remonter la couverture.

Exécution : .venv/bin/python model/backtest/prior_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.historical import load_calibration_frame
from model.models.bayesian_nowcast import HouseEffectsCalibration
from backtest_loo import predict, coverage_report  # même dossier

import numpy as np

# Configurations : nom -> surcharge de DEFAULT_PRIORS_CFG (le reste = référence).
CONFIGS = {
    "référence": {},
    "biais_large": {"bloc_bias_mean_sd": 10.0, "bloc_bias_sd_sd": 6.0},
    "biais_serré": {"bloc_bias_mean_sd": 2.0, "bloc_bias_sd_sd": 1.0},
    "dérive_large": {"campaign_drift_sd_sd": 2.5},
    "dérive_serrée": {"campaign_drift_sd_sd": 0.4},
    "excès_large": {"excess_log_scale_sd": 2.0, "excess_horizon_slope_sd": 2.0},
}


def evaluate(train, test, cfg: dict) -> dict:
    cal = HouseEffectsCalibration()
    cal.draws, cal.tune, cal.chains = 500, 500, 2
    cal.priors_cfg = cfg
    bank = cal.calibrate(train)

    known = set(bank.bloc_bias_mean.mean.coords["bloc"].values.tolist())
    tk = test[test["bloc"].isin(known) & test["intention"].notna()
             & test["echantillon"].notna()].reset_index(drop=True)
    pred = predict(bank, tk)
    z = (tk["ecart"].to_numpy(float) - pred["mean"]) / pred["sd"]
    cov = coverage_report(z)
    logscore = float(np.mean(-0.5 * np.log(2 * np.pi * pred["sd"] ** 2) - 0.5 * z ** 2))
    return {
        "mu_droite": bank.bloc_bias_mean.at(bloc="droite")[0],
        "mu_dr": bank.bloc_bias_mean.at(bloc="droite_radicale")[0],
        "campaign_drift_sd": bank.campaign_drift_sd.item()[0],
        "cov90": cov["90%"], "cov50": cov["50%"], "logscore": logscore,
    }


def main():
    df = load_calibration_frame()
    train = df[df.election == 2017].reset_index(drop=True)
    test = df[df.election == 2022].reset_index(drop=True)

    print(f"{'config':>14} | {'μ_droite':>8} | {'μ_dr':>6} | {'drift_sd':>8} | "
          f"{'cov50':>5} | {'cov90':>5} | {'logscore':>8}")
    print("-" * 78)
    for name, cfg in CONFIGS.items():
        r = evaluate(train, test, cfg)
        print(f"{name:>14} | {r['mu_droite']:8.2f} | {r['mu_dr']:6.2f} | "
              f"{r['campaign_drift_sd']:8.2f} | {r['cov50']:5.2f} | {r['cov90']:5.2f} | "
              f"{r['logscore']:8.3f}")
    print("\nLecture : biais stables entre configs => dominés par les données. "
          "cov90 qui monte avec des priors de variance plus larges => "
          "la sous-couverture était (en partie) un effet de priors trop serrés.")


if __name__ == "__main__":
    main()
