"""Visualisation des résultats de calibration (Phase 1).

Lit model/models/bayesian_nowcast/bank.json et trace ce que le modèle a appris :
  1. biais_calibres.png       — house effect (biais à J-0) par bloc, avec IC 90 %
  2. ic_horizon.png            — « fan chart » : bande de prévision 90 % autour d'un
                                candidat, en fonction de l'horizon (jours avant le vote)
  3. variance_decomp.png        — décomposition de l'incertitude : échantillonnage /
                                excès house-effect / dérive future
  4. biais_institut_<bloc>.png    — zoom par institut, pour les blocs où la
                                dispersion inter-instituts (bloc_bias_sd) est
                                statistiquement distinguable de 0 : le modèle fait
                                déjà du partial pooling institut×bloc (pas un choix
                                binaire bloc vs institut), ce graphe montre juste où
                                les données soutiennent réellement d'aller plus loin
                                que la moyenne du bloc.

Imprime aussi un diagnostic « biais extrapolé vs écart brut à très court
horizon » : le biais à J-0 du modèle (`bloc_bias_mean`) est une EXTRAPOLATION
de la tendance de dérive jusqu'à horizon 0, pas une mesure directe (peu de
sondages sont faits la veille du scrutin). Si le biais n'était que de la
dérive de dernière minute mal capturée par la forme lisse
`campaign_drift · base(horizon)`, il devrait DIMINUER quand on regarde les
sondages réellement les plus proches du vote (J<=3, J<=7) — puisqu'un sondage
pris à J-1 a eu le temps d'enregistrer tout mouvement d'opinion. S'il ne
diminue pas, l'écart est déjà là dans des sondages pris essentiellement à
l'isoloir : pas un problème de délai de mesure, donc plutôt un vrai écart
intention-vote (comportement stratégique, vote utile de dernière minute, etc.)
qu'un biais d'institut au sens propre.

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
from pipeline.historical import load_calibration_frame
from model.core.bank import Bank
from model.models.bayesian_nowcast import (
    BANK_PATH, excess_sigma, forecast_drift_sigma, sampling_sigma,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
Z90 = 1.645  # quantile 95 % de la normale -> demi-bande à 90 %

# Candidat « type » pour les prévisions illustratives.
P_REF = 22.0       # intention (%)
N_REF = 1000       # taille d'échantillon
INST_REF = "Ifop"


def forecast_sigma(bank: Bank, institut, horizon, p=P_REF, n=N_REF):
    """Écart-type de prévision 2027 : échantillonnage ⊕ excès ⊕ dérive future."""
    return np.sqrt(
        sampling_sigma(p, n) ** 2
        + excess_sigma(bank, institut, horizon) ** 2
        + forecast_drift_sigma(bank, horizon) ** 2
    )


def fig_biais(bank: Bank):
    blocs = bank.bloc_bias_mean.mean.coords["bloc"].values.tolist()
    m = np.array([bank.bloc_bias_mean.at(bloc=b)[0] for b in blocs])
    s = np.array([bank.bloc_bias_mean.at(bloc=b)[1] for b in blocs])
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


def print_biais_vs_court_horizon(bank: Bank, h_max=(3, 7)):
    """Compare le biais à J-0 EXTRAPOLÉ par le modèle (bloc_bias_mean) à
    l'écart brut mesuré sur les sondages réellement pris à très court horizon.
    Voir le docstring du module pour l'interprétation."""
    df = load_calibration_frame()
    blocs = bank.bloc_bias_mean.mean.coords["bloc"].values.tolist()
    print("\n=== Biais extrapolé (modèle) vs écart brut à court horizon ===")
    header = "  ".join(f"J<={h:<2}" for h in h_max)
    print(f"{'bloc':18s}  bloc_bias_mean(J-0)  {header}")
    for bloc in blocs:
        mu, _ = bank.bloc_bias_mean.at(bloc=bloc)
        cells = []
        for h in h_max:
            sub = df[(df["bloc"] == bloc) & (df["horizon"] <= h)]
            if len(sub):
                se = sub["ecart"].std(ddof=1) / np.sqrt(len(sub)) if len(sub) > 1 else float("nan")
                cells.append(f"{sub['ecart'].mean():+5.2f} (n={len(sub):>2}, se={se:.2f})")
            else:
                cells.append("     aucune obs")
        print(f"{bloc:18s}  {mu:+9.2f}     " + "   ".join(cells))


def fig_biais_institut(bank: Bank, bloc: str):
    """Zoom institut x bloc, réservé aux blocs où bloc_bias_sd (dispersion
    inter-instituts) est nettement > 0 : ailleurs le partial pooling a déjà
    ramené les instituts près de la moyenne du bloc, un zoom n'y montrerait
    que du bruit de shrinkage, pas un vrai signal institut."""
    df = load_calibration_frame()
    n_obs = df[df["bloc"] == bloc].groupby("institut").size()

    instituts = bank.institut_bias.mean.coords["institut"].values.tolist()
    rows = [(inst, *bank.institut_bias.at(institut=inst, bloc=bloc), int(n_obs.get(inst, 0)))
            for inst in instituts]
    rows.sort(key=lambda r: r[1])
    insts, m, s, n = zip(*rows)
    m, s = np.array(m), np.array(s)
    mu_pop, _ = bank.bloc_bias_mean.at(bloc=bloc)
    tau, _ = bank.bloc_bias_sd.at(bloc=bloc)

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(insts) + 1.6))
    ax.errorbar(m, range(len(insts)), xerr=Z90 * s, fmt="o", color="#2b6cb0",
                ecolor="#a0aec0", capsize=3)
    ax.axvline(mu_pop, color="#805ad5", lw=1.2, ls=":",
               label=f"moyenne population ({mu_pop:+.2f})")
    ax.axvline(0, color="#e53e3e", lw=1, ls="--")
    ax.set_yticks(range(len(insts)))
    ax.set_yticklabels([f"{i}  (n={k})" for i, k in zip(insts, n)])
    ax.set_xlabel("Biais à J-0 (points) — intention − résultat")
    ax.set_title(f"House effect par institut, bloc « {bloc} » (IC 90 %)\n"
                 f"bloc_bias_sd = {tau:.2f}", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"biais_institut_{bloc}.png", dpi=110)
    plt.close(fig)


def fig_fan_chart(bank: Bank):
    h = np.arange(0, 400)
    sig = np.array([forecast_sigma(bank, INST_REF, x) for x in h])
    band = Z90 * sig

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.fill_between(h, P_REF - band, P_REF + band, color="#2b6cb0", alpha=0.20,
                    label="bande de prévision 90 %")
    ax.plot(h, np.full_like(h, P_REF, dtype=float), color="#2b6cb0", lw=1.5,
            label=f"intention « vraie » supposée ({P_REF:.0f} %)")
    ax.invert_xaxis()
    ax.set_xlabel("Horizon (jours avant le 1er tour)")
    ax.set_ylabel("Intention (%)")
    ax.set_title(f"Courbe d'IC apprise — bande de prévision 90 % vs horizon\n"
                 f"(candidat à {P_REF:.0f} %, sondage {INST_REF} n={N_REF})")
    ax.legend(loc="upper right")
    for x in (7, 30, 90, 180, 365):
        b = Z90 * forecast_sigma(bank, INST_REF, x)
        ax.annotate(f"±{b:.1f}", (x, P_REF + b), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8, color="#2b6cb0")
    fig.tight_layout(); fig.savefig(FIG_DIR / "ic_horizon.png", dpi=110)
    plt.close(fig)


def fig_variance_decomp(bank: Bank):
    h = np.arange(0, 400)
    samp = np.full_like(h, sampling_sigma(P_REF, N_REF), dtype=float)
    exc = np.array([excess_sigma(bank, INST_REF, x) for x in h])
    drift = np.array([forecast_drift_sigma(bank, x) for x in h])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(h, samp, label="échantillonnage  √(p(100−p)/n)", color="#718096", lw=2)
    ax.plot(h, exc, label="excès house-effect  (croît avec l'horizon)", color="#38a169", lw=2)
    ax.plot(h, drift, label="dérive future 2027  campaign_drift_sd·√h", color="#d69e2e", lw=2)
    tot = np.sqrt(samp ** 2 + exc ** 2 + drift ** 2)
    ax.plot(h, tot, label="total (⊕ en quadrature)", color="#2b6cb0", lw=2.5, ls="--")
    ax.invert_xaxis()
    ax.set_xlabel("Horizon (jours avant le 1er tour)")
    ax.set_ylabel("Écart-type (points)")
    ax.set_title("Décomposition de l'incertitude de prévision par source")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(FIG_DIR / "variance_decomp.png", dpi=110)
    plt.close(fig)


def main():
    bank = Bank.load(BANK_PATH)
    if bank is None:
        raise SystemExit(f"Bank absente ({BANK_PATH}) — lancer d'abord "
                         ".venv/bin/python -m model.models.bayesian_nowcast")

    print("Demi-bande 90 % (candidat 22 %, Ifop n=1000) :")
    for x in (0, 7, 30, 90, 180, 365):
        print(f"  J-{x:<4} : ±{Z90 * forecast_sigma(bank, INST_REF, x):.2f} pts")

    blocs = bank.bloc_bias_mean.mean.coords["bloc"].values.tolist()
    print("\nDispersion inter-instituts (bloc_bias_sd) par bloc, triée :")
    for b in sorted(blocs, key=lambda b: -bank.bloc_bias_sd.at(bloc=b)[0]):
        print(f"  {b:18s} {bank.bloc_bias_sd.at(bloc=b)[0]:.3f}")

    print_biais_vs_court_horizon(bank)

    fig_biais(bank); fig_fan_chart(bank); fig_variance_decomp(bank)
    # zoom institut réservé aux blocs où bloc_bias_sd est nettement > 0 (cf.
    # tri ci-dessus) — pour le moment, seul "droite" qualifie.
    fig_biais_institut(bank, "droite")
    print(f"\nFigures écrites dans {FIG_DIR}/")


if __name__ == "__main__":
    main()
