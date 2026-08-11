"""Notebook exploratoire — distribution du drift RÉEL d'opinion (2017 + 2022).

On regarde la DISTRIBUTION de `drift / √h` (échelle marche aléatoire) : est-elle
gaussienne, ou à queues épaisses ? C'est ce qui a motivé le choix d'un saut
terminal paramétrique sinh-arcsinh plutôt qu'une dérive gaussienne simple
(`movement_pool` + `TerminalJumpCalibration`, dans
`model/models/bayesian_nowcast/calibration.py`, consommés en production par
le nowcast bayésien) — ce fichier-ci n'est que le diagnostic qui justifie ce
choix, il n'est pas exécuté en production.

Exécution :
    .venv/bin/python notebooks/02_drift_analysis.py

Sorties : figure PNG dans notebooks/figures/ + résumé imprimé.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.historical import load_calibration_frame
from model.core.bank import Bank
from model.models.bayesian_nowcast import BANK_PATH, institut_bias_prior

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def debiased_drift(bank: Bank) -> pd.DataFrame:
    """Débiaise chaque sondage historique et moyenne par (élection, candidat,
    fenêtre d'horizon) pour isoler le drift réel (bruit d'échantillon annulé)."""
    df = load_calibration_frame()
    df["bias"] = [institut_bias_prior(bank, r.institut, r.bloc)[0] for r in df.itertuples()]
    df["ecart_deb"] = df["ecart"] - df["bias"]           # ≈ drift + bruit

    bins = [0, 15, 45, 90, 180, 400]
    df["hbin"] = pd.cut(df["horizon"], bins=bins)
    g = (df.groupby(["election", "candidat", "bloc", "hbin"], observed=True)
           .agg(drift=("ecart_deb", "mean"),
                horizon=("horizon", "mean"),
                n=("ecart_deb", "size"))
           .reset_index())
    g = g[g["n"] >= 2]                                    # au moins 2 sondages -> bruit réduit
    g["drift_norm"] = g["drift"] / np.sqrt(g["horizon"])  # échelle √h
    return g


def empirical_drift_norm(bank: Bank | None = None) -> np.ndarray:
    """Pool des drifts réels mesurés (drift / √horizon) sur 2017+2022."""
    return debiased_drift(bank or Bank.load(BANK_PATH))["drift_norm"].to_numpy()


def main():
    bank = Bank.load(BANK_PATH)
    if bank is None:
        raise SystemExit(f"Bank absente ({BANK_PATH}) — lancer d'abord "
                         ".venv/bin/python -m model.models.bayesian_nowcast")
    g = debiased_drift(bank)
    d = g["drift_norm"].to_numpy()
    tau, _ = bank.campaign_drift_sd.item()

    sd = d.std(ddof=1)
    from scipy.stats import kurtosis, skew, t as student_t
    exk = kurtosis(d)            # excès de kurtosis (0 = gaussien)
    sk = skew(d)
    print(f"Drift réel mesuré (2017+2022), {len(d)} points (candidat×fenêtre) :")
    print(f"  sd(drift/√h)      = {sd:.3f}   (campaign_drift_sd calibré = {tau:.3f})")
    print(f"  skew              = {sk:+.2f}")
    print(f"  excès de kurtosis = {exk:+.2f}   (>0 ⇒ queues plus épaisses qu'une gaussienne)")
    g2 = g.reindex(g["drift"].abs().sort_values(ascending=False).index)
    print("\n  Plus gros drifts réels observés (points) :")
    for r in g2.head(6).itertuples():
        print(f"    {r.election} {r.candidat:22s} {r.drift:+5.1f} pts à ~J-{r.horizon:.0f}")

    df_t, loc_t, scale_t = student_t.fit(d)
    print(f"\n  Student-t ajusté sur drift/√h : df={df_t:.1f}, scale={scale_t:.3f}")

    fig, (axq, axh) = plt.subplots(1, 2, figsize=(11, 4.6))
    from scipy import stats
    stats.probplot(d, dist="norm", plot=axq)
    axq.set_title("Q-Q plot du drift/√h vs gaussienne\n(points hors ligne = queues épaisses)")
    axq.get_lines()[0].set_markersize(4)

    xs = np.linspace(d.min() - 0.5, d.max() + 0.5, 200)
    axh.hist(d, bins=15, density=True, color="#a0aec0", alpha=0.7, label="drift réel mesuré")
    axh.plot(xs, stats.norm.pdf(xs, 0, sd), "b-", lw=2, label=f"gaussienne (sd={sd:.2f})")
    axh.plot(xs, student_t.pdf(xs, df_t, loc_t, scale_t), "r--", lw=2,
             label=f"Student-t (df={df_t:.1f})")
    axh.set_title("Distribution du drift / √horizon"); axh.legend(fontsize=8)
    axh.set_xlabel("drift / √h (points par √jour)")
    fig.tight_layout()
    out = FIG_DIR / "drift_reel_2017_2022.png"
    fig.savefig(out, dpi=110)
    print(f"\nFigure : {out}")
    return dict(sd=float(sd), excess_kurtosis=float(exk), df_t=float(df_t), scale_t=float(scale_t))


if __name__ == "__main__":
    main()
