"""Backtest hors-échantillon : calibrer sur une élection, prédire l'autre.

On calibre le modèle sur **2017 seulement**, puis on prédit l'écart
`intention − résultat` de chaque sondage **2022** (jamais vu). Pour chaque
observation test on calcule :
  - la moyenne prédite  = biais[institut, bloc] appris sur 2017 (shrinkage vers
    la moyenne du bloc si l'institut est nouveau) ;
  - l'écart-type prédit = sqrt( échantillonnage + excès(horizon)² + dérive² ),
    où la dérive 2022 est inconnue et tirée de Normal(0, campaign_drift_sd) appris.
Le résidu standardisé z = (écart_observé − moyenne) / sd doit alors être ~N(0,1)
si le modèle est bien calibré : on vérifie la **couverture** des IC (50/80/90 %).

Réutilise directement les fonctions dérivées de model/models/bayesian_nowcast.py
(excess_sigma, forecast_drift_sigma, institut_bias_prior) — même transformation
d'horizon que celle calibrée, une seule source de vérité.

Exécution : .venv/bin/python model/backtest/backtest_loo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.historical import load_calibration_frame
from model.core.bank import Bank
from model.models.bayesian_nowcast import (
    HouseEffectsCalibration, excess_sigma, forecast_drift_sigma, institut_bias_prior,
)

FIG_DIR = ROOT / "notebooks" / "figures"
FIG_DIR.mkdir(exist_ok=True)


def predict(bank: Bank, test) -> dict:
    """Moyenne et écart-type prédits pour chaque ligne du jeu test."""
    means, sds = [], []
    for row in test.itertuples(index=False):
        mean, _ = institut_bias_prior(bank, row.institut, row.bloc)
        excess = excess_sigma(bank, row.institut, row.horizon)
        samp = row.intention * (100.0 - row.intention) / max(row.echantillon, 1)
        drift_sd = forecast_drift_sigma(bank, row.horizon)
        means.append(mean)
        sds.append(np.sqrt(samp + excess ** 2 + drift_sd ** 2))
    return {"mean": np.array(means), "sd": np.array(sds)}


def coverage_report(z: np.ndarray) -> dict:
    levels = {"50%": 0.674, "80%": 1.282, "90%": 1.645, "95%": 1.960}
    return {k: float(np.mean(np.abs(z) <= q)) for k, q in levels.items()}


def run_backtest(train, test, seed: int = 27) -> dict:
    cal = HouseEffectsCalibration()
    cal.draws, cal.tune, cal.chains, cal.seed = 600, 600, 2, seed
    bank = cal.calibrate(train)

    known = set(bank.bloc_bias_mean.mean.coords["bloc"].values.tolist())
    test_k = test[test["bloc"].isin(known) & test["intention"].notna()
                 & test["echantillon"].notna()].reset_index(drop=True)
    pred = predict(bank, test_k)
    z = (test_k["ecart"].to_numpy(float) - pred["mean"]) / pred["sd"]
    logscore = float(np.mean(-0.5 * np.log(2 * np.pi * pred["sd"] ** 2) - 0.5 * z ** 2))
    return {
        "bank": bank, "test": test_k, "pred": pred, "z": z,
        "coverage": coverage_report(z),
        "rmse": float(np.sqrt(np.mean((test_k["ecart"].to_numpy(float) - pred["mean"]) ** 2))),
        "logscore": logscore, "n_excluded": int(len(test) - len(test_k)),
    }


def main():
    df = load_calibration_frame()
    train = df[df.election == 2017].reset_index(drop=True)
    test = df[df.election == 2022].reset_index(drop=True)
    print(f"Train 2017 : {len(train)} obs (horizon ≤ {train.horizon.max()} j)")
    print(f"Test  2022 : {len(test)} obs (horizon ≤ {test.horizon.max()} j)\n")

    r = run_backtest(train, test)
    c = r["coverage"]
    print(f"RMSE {r['rmse']:.2f} | logscore {r['logscore']:.3f} | "
          f"couverture 50/80/90/95 : {c['50%']:.2f} {c['80%']:.2f} {c['90%']:.2f} {c['95%']:.2f}  "
          f"(nominal 0.50/0.80/0.90/0.95)")
    print(f"(obs test exclues, bloc absent de 2017 : {r['n_excluded']})")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.scatter(r["test"]["horizon"], r["z"], s=8, alpha=0.35, color="#2b6cb0")
    for q in (1.645, -1.645):
        ax.axhline(q, color="#e53e3e", ls="--", lw=1)
    ax.axhline(0, color="#718096", lw=0.8)
    ax.text(r["test"]["horizon"].max(), 1.70, "IC 90 %", color="#e53e3e", ha="right", fontsize=8)
    ax.set_xlabel("Horizon 2022 (jours avant le 1er tour)")
    ax.set_ylabel("Résidu standardisé  (écart − prédit) / sd")
    ax.set_title("Backtest 2017 → 2022 : résidus standardisés hors-échantillon\n"
                 "(bien calibré ⇒ nuage centré, ~90 % dans la bande)")
    ax.set_ylim(-6, 6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "backtest_residus.png", dpi=110)
    plt.close(fig)
    print(f"\nFigure écrite dans {FIG_DIR}/")


if __name__ == "__main__":
    main()
