"""Backtest hors-échantillon : calibrer sur une élection, prédire l'autre.

On calibre le modèle sur **2017 seulement**, puis on prédit l'écart
`intention − résultat` de chaque sondage **2022** (jamais vu). Pour chaque
observation test on calcule :
  - la moyenne prédite  = biais[institut, bloc] appris sur 2017 (shrinkage vers
    la moyenne du bloc si l'institut est nouveau) ;
  - l'écart-type prédit = sqrt( échantillonnage + excès(horizon)² + dérive² ),
    où la dérive 2022 est inconnue et tirée de Normal(0, tau_derive) appris.
Le résidu standardisé z = (écart_observé − moyenne) / sd doit alors être ~N(0,1)
si le modèle est bien calibré : on vérifie la **couverture** des IC (50/80/90 %).

On compare deux formes d'horizon (`log(1+h)` vs `sqrt(h)`) sur leur couverture et
leur log-score hors-échantillon — pour trancher empiriquement laquelle
extrapole le mieux des horizons courts de 2017 (≤ 38 j) aux horizons longs de
2022 (≤ 660 j).

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
from calibration.fit_house_effects import fit, export_priors, horizon_base

FIG_DIR = ROOT / "notebooks" / "figures"
FIG_DIR.mkdir(exist_ok=True)


def predict(priors: dict, meta: dict, test) -> dict:
    """Moyenne et écart-type prédits pour chaque ligne du jeu test."""
    hf = meta["horizon_form"]
    base = horizon_base(test["horizon"].to_numpy(float), hf)
    z = (base - meta["log_h_mean"]) / meta["log_h_std"]

    vg = priors["variance_globale"]
    le0 = vg["log_excess0"]["mean"]
    b_h = vg["b_horizon"]["mean"]
    tau_d = vg["tau_derive"]["mean"]
    deff = vg["design_effect_quotas"]["mean"]

    means, sds = [], []
    for i, row in enumerate(test.itertuples(index=False)):
        inst = priors["institut"].get(row.institut)
        bloc = row.bloc
        # moyenne : biais institut×bloc, sinon repli sur la population du bloc
        if inst is not None and bloc in inst["biais_par_bloc"]:
            mean = inst["biais_par_bloc"][bloc]["mean"]
            s_inst = inst["precision_generale_log"]["mean"]
        else:
            mean = priors["bloc"][bloc]["biais_moyen_population"]["mean"]
            s_inst = 0.0
        excess = np.exp(le0 + s_inst + b_h * z[i])
        samp = deff * row.intention * (100.0 - row.intention) / max(row.echantillon, 1)
        drift_var = (tau_d * base[i]) ** 2
        means.append(mean)
        sds.append(np.sqrt(samp + excess**2 + drift_var))
    return {"mean": np.array(means), "sd": np.array(sds)}


def coverage_report(z: np.ndarray) -> dict:
    from math import erf, sqrt
    levels = {"50%": 0.674, "80%": 1.282, "90%": 1.645, "95%": 1.960}
    return {k: float(np.mean(np.abs(z) <= q)) for k, q in levels.items()}


def run_form(train, test, horizon_form: str, seed: int = 27):
    idata, meta = fit(train, horizon_form=horizon_form, draws=600, tune=600,
                      chains=2, seed=seed, progressbar=False)
    priors = export_priors(idata, meta)
    # blocs absents du train (ex. ecologistes en 2017) : non prédictibles -> exclus
    known = set(priors["bloc"])
    mask = test["bloc"].isin(known).to_numpy()
    test_k = test[mask].reset_index(drop=True)
    pred = predict(priors, meta, test_k)
    z = (test_k["ecart"].to_numpy(float) - pred["mean"]) / pred["sd"]
    # log-score gaussien moyen (plus haut = mieux)
    logscore = float(np.mean(-0.5 * np.log(2 * np.pi * pred["sd"] ** 2)
                             - 0.5 * z ** 2))
    return {
        "form": horizon_form, "priors": priors, "meta": meta,
        "test": test_k, "pred": pred, "z": z,
        "coverage": coverage_report(z), "rmse": float(np.sqrt(np.mean(
            (test_k["ecart"].to_numpy(float) - pred["mean"]) ** 2))),
        "logscore": logscore, "n_excluded": int((~mask).sum()),
    }


def main():
    df = load_calibration_frame()
    train = df[df.election == 2017].reset_index(drop=True)
    test = df[df.election == 2022].reset_index(drop=True)
    print(f"Train 2017 : {len(train)} obs (horizon ≤ {train.horizon.max()} j)")
    print(f"Test  2022 : {len(test)} obs (horizon ≤ {test.horizon.max()} j)\n")

    results = {hf: run_form(train, test, hf) for hf in ("log", "sqrt")}

    print(f"{'forme':>6} | {'RMSE':>5} | {'logscore':>8} | couverture 50/80/90/95")
    for hf, r in results.items():
        c = r["coverage"]
        print(f"{hf:>6} | {r['rmse']:5.2f} | {r['logscore']:8.3f} | "
              f"{c['50%']:.2f} {c['80%']:.2f} {c['90%']:.2f} {c['95%']:.2f}  "
              f"(nominal 0.50/0.80/0.90/0.95)")
    print(f"\n(blocs test exclus car absents de 2017 : "
          f"{results['log']['n_excluded']} obs — ex. écologistes)")

    # --- Figure 1 : résidus standardisés vs horizon (forme log) ---
    r = results["log"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.scatter(r["test"]["horizon"], r["z"], s=8, alpha=0.35, color="#2b6cb0")
    for q, lab in [(1.645, "90 %"), (-1.645, None)]:
        ax.axhline(q, color="#e53e3e", ls="--", lw=1)
    ax.axhline(0, color="#718096", lw=0.8)
    ax.text(test.horizon.max(), 1.70, "IC 90 %", color="#e53e3e", ha="right", fontsize=8)
    ax.set_xlabel("Horizon 2022 (jours avant le 1er tour)")
    ax.set_ylabel("Résidu standardisé  (écart − prédit) / sd")
    ax.set_title("Backtest 2017 → 2022 : résidus standardisés hors-échantillon\n"
                 "(bien calibré ⇒ nuage centré, ~90 % dans la bande)")
    ax.set_ylim(-6, 6)
    fig.tight_layout(); fig.savefig(FIG_DIR / "backtest_residus.png", dpi=110)
    plt.close(fig)

    # --- Figure 2 : écart observé 2022 vs bande prédite, bloc droite_radicale ---
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    sub = r["test"]["bloc"] == "droite_radicale"
    hh = r["test"]["horizon"].to_numpy()[sub.to_numpy()]
    yy = r["test"]["ecart"].to_numpy(float)[sub.to_numpy()]
    ax.scatter(hh, yy, s=12, alpha=0.5, color="#805ad5", label="écart observé 2022")
    grid = np.arange(0, test.horizon.max() + 1)
    from calibration.fit_house_effects import horizon_base as hb
    base = hb(grid, "log"); z = (base - r["meta"]["log_h_mean"]) / r["meta"]["log_h_std"]
    vg = r["priors"]["variance_globale"]
    mean_dr = r["priors"]["bloc"]["droite_radicale"]["biais_moyen_population"]["mean"]
    excess = np.exp(vg["log_excess0"]["mean"] + vg["b_horizon"]["mean"] * z)
    samp = 1.0 * 12 * 88 / 1000  # p~12 % typique RN, n~1000
    sd = np.sqrt(samp + excess**2 + (vg["tau_derive"]["mean"] * base) ** 2)
    ax.plot(grid, np.full_like(grid, mean_dr, dtype=float), color="#805ad5", lw=1.5,
            label="biais prédit (appris 2017)")
    ax.fill_between(grid, mean_dr - 1.645 * sd, mean_dr + 1.645 * sd,
                    color="#805ad5", alpha=0.15, label="bande 90 % prédite")
    ax.invert_xaxis()
    ax.set_xlabel("Horizon 2022 (jours avant le 1er tour)")
    ax.set_ylabel("Écart intention − résultat (points)")
    ax.set_title("Backtest 2017 → 2022, bloc droite radicale :\n"
                 "écarts 2022 observés vs bande de prévision apprise sur 2017")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "backtest_droite_radicale.png", dpi=110)
    plt.close(fig)

    print(f"\nFigures écrites dans {FIG_DIR}/")


if __name__ == "__main__":
    main()
