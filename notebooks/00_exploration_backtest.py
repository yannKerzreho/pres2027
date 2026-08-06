"""Phase 0 — notebook exploratoire : backtest visuel sondage vs résultat.

Objectif (roadmap §7, Phase 0) : visualiser l'écart entre intentions publiées et
résultat final (2017 + 2022), par institut et par bloc politique, et regarder comment
la dispersion de cet écart évolue avec l'horizon (jours avant le scrutin). C'est
le signal brut que la Phase 1 va modéliser proprement (hiérarchique bayésien).

Exécution :
    .venv/bin/python notebooks/00_exploration_backtest.py

Sorties : figures PNG dans notebooks/figures/ + résumés imprimés.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.historical import load_calibration_frame

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def main() -> None:
    df = load_calibration_frame()
    print(f"Observations de calibration : {len(df)}")
    print(f"Instituts : {df['institut'].nunique()} | Blocs : {sorted(df['bloc'].unique())}")
    print(f"Horizon (jours avant T1) : min={df['horizon'].min()} max={df['horizon'].max()}\n")

    # ------------------------------------------------------------------
    # 1) Biais moyen par institut (tous blocs confondus) + IC ~95 %
    # ------------------------------------------------------------------
    g = df.groupby("institut")["ecart"]
    stats = g.agg(["mean", "std", "count"]).sort_values("mean")
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])
    print("=== Biais moyen par institut (points, intention − résultat) ===")
    print(stats.round(2).to_string(), "\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        stats["mean"], range(len(stats)), xerr=1.96 * stats["sem"],
        fmt="o", color="#2b6cb0", ecolor="#a0aec0", capsize=3,
    )
    ax.axvline(0, color="#e53e3e", lw=1, ls="--")
    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels(stats.index)
    ax.set_xlabel("Écart moyen intention − résultat (points)")
    ax.set_title("Biais moyen par institut — présidentielle 2017 + 2022 (1er tour)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "biais_par_institut.png", dpi=110)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2) Biais moyen par bloc politique (le RN est le cas d'école)
    # ------------------------------------------------------------------
    gb = df.groupby("bloc")["ecart"]
    sb = gb.agg(["mean", "std", "count"]).sort_values("mean")
    sb["sem"] = sb["std"] / np.sqrt(sb["count"])
    print("=== Biais moyen par bloc ===")
    print(sb.round(2).to_string(), "\n")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(sb.index, sb["mean"], yerr=1.96 * sb["sem"], color="#2b6cb0", capsize=3)
    ax.axhline(0, color="#e53e3e", lw=1, ls="--")
    ax.set_ylabel("Écart moyen (points)")
    ax.set_title("Biais moyen par bloc politique — 2017 + 2022 (1er tour)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "biais_par_bloc.png", dpi=110)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3) Dispersion de l'écart en fonction de l'horizon
    #    (justifie la pondération temporelle mesurée, cf. §4.3)
    # ------------------------------------------------------------------
    bins = [0, 7, 14, 30, 60, 120, 240, 10_000]
    labels = ["0-7j", "8-14j", "15-30j", "31-60j", "61-120j", "121-240j", ">240j"]
    df["horizon_bin"] = pd.cut(df["horizon"], bins=bins, labels=labels, right=True)
    disp = df.groupby("horizon_bin", observed=True)["ecart"].agg(
        rmse=lambda s: np.sqrt(np.mean(s**2)), std="std", n="count"
    )
    print("=== Dispersion de l'écart par tranche d'horizon ===")
    print(disp.round(2).to_string(), "\n")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(disp)), disp["rmse"], "o-", color="#2b6cb0", label="RMSE de l'écart")
    ax.plot(range(len(disp)), disp["std"], "s--", color="#38a169", label="Écart-type")
    ax.set_xticks(range(len(disp)))
    ax.set_xticklabels(disp.index, rotation=30, ha="right")
    ax.set_ylabel("Points")
    ax.set_title("Dispersion de l'erreur selon l'horizon — 2017 + 2022")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dispersion_par_horizon.png", dpi=110)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4) Biais institut × bloc pour le bloc droite_radicale (RN + Reconquête)
    #    -> visualise l'histoire du redressement du vote RN
    # ------------------------------------------------------------------
    dr = df[df["bloc"] == "droite_radicale"]
    sr = dr.groupby("institut")["ecart"].agg(["mean", "std", "count"]).sort_values("mean")
    sr["sem"] = sr["std"] / np.sqrt(sr["count"])
    print("=== Biais par institut sur le bloc droite_radicale ===")
    print(sr.round(2).to_string(), "\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        sr["mean"], range(len(sr)), xerr=1.96 * sr["sem"],
        fmt="o", color="#805ad5", ecolor="#c3b3e6", capsize=3,
    )
    ax.axvline(0, color="#e53e3e", lw=1, ls="--")
    ax.set_yticks(range(len(sr)))
    ax.set_yticklabels(sr.index)
    ax.set_xlabel("Écart moyen intention − résultat (points)")
    ax.set_title("Biais par institut — bloc droite radicale, 2017 + 2022")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "biais_droite_radicale.png", dpi=110)
    plt.close(fig)

    print(f"Figures écrites dans {FIG_DIR}/")


if __name__ == "__main__":
    main()
