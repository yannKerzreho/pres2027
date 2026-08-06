"""Mesure du drift RÉEL d'opinion sur 2017 & 2022 (ground truth).

Idée (cf. critique : la dérive gaussienne sous-estime les queues) : au lieu de
supposer `drift ~ N(0, tau·√h)`, on le **mesure**. Pour chaque sondage historique
on retire ce qu'on sait déjà :

    ecart = intention - resultat_final
          = biais[institut,bloc] (calibré)  +  drift(candidat, horizon)  +  bruit_echantillon

On **débiaise** (soustraction du biais calibré). Le résidu ≈ drift + bruit
d'échantillonnage. En moyennant les sondages proches (même candidat, fenêtre
d'horizon) le bruit s'annule et il reste le **drift réel** du candidat : de
combien l'opinion a bougé entre l'horizon h et le scrutin.

On regarde ensuite la DISTRIBUTION de `drift / √h` (échelle marche aléatoire) :
est-elle gaussienne, ou à queues épaisses ? C'est ce qui décide la loi à utiliser
dans la simulation 2027.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.historical import load_calibration_frame, load_results
from calibration.priors_utils import load_priors, expected_bias

FIG_DIR = ROOT / "notebooks" / "figures"


def _clr(v: np.ndarray) -> np.ndarray:
    """Centered log-ratio d'un vecteur de parts (ramené au simplexe)."""
    v = np.clip(np.asarray(v, dtype=float), 1e-4, None)
    v = v / v.sum()
    lg = np.log(v)
    return lg - lg.mean()


def clr_movement_pool(priors: dict | None = None, hmin: float = 45.0) -> np.ndarray:
    """Mouvements RÉELS d'opinion, en espace log-ratio, à ré-échantillonner.

    Pour chaque élection et fenêtre d'horizon (assez lointaine, ≥ hmin jours), on
    débiaise les sondages, on forme le vecteur de parts par candidat, on le passe
    en CLR, et on calcule le mouvement `clr(résultat) − clr(sondage_débiaisé)` :
    de combien, en log-ratio, chaque candidat a bougé jusqu'au scrutin. Ce pool
    (queues épaisses, asymétrie, saturation incluses) alimente la dérive 2027 dans
    le bon espace — softmax garantissant π ∈ (0,1)."""
    priors = priors or load_priors()
    df = load_calibration_frame()
    df["bias"] = [expected_bias(priors, r.institut, r.bloc)["mean"] for r in df.itertuples()]
    df["deb"] = df["intention"] - df["bias"]
    res = load_results()
    res = res[res["tour"] == "Premier tour"]

    df["hbin"] = pd.cut(df["horizon"], [0, 45, 90, 180, 400])
    moves: list[float] = []
    for elec in df["election"].unique():
        de = df[df["election"] == elec]
        cand = sorted(de["candidat"].unique())
        rmap = dict(zip(res.loc[res.election == elec, "candidat"],
                        res.loc[res.election == elec, "resultat"]))
        r = np.array([rmap.get(c, np.nan) for c in cand])
        for _, grp in de.groupby("hbin", observed=True):
            if grp["horizon"].mean() < hmin:
                continue
            smap = grp.groupby("candidat")["deb"].mean()
            s = np.array([smap.get(c, np.nan) for c in cand])
            mask = ~np.isnan(s) & ~np.isnan(r) & (s > 0)
            if mask.sum() < 4:
                continue
            moves.extend((_clr(r[mask]) - _clr(s[mask])).tolist())
    return np.array(moves)


def debiased_drift(priors: dict) -> pd.DataFrame:
    """Débiaise chaque sondage historique et moyenne par (élection, candidat,
    fenêtre d'horizon) pour isoler le drift réel (bruit d'échantillon annulé)."""
    df = load_calibration_frame()
    df["bias"] = [expected_bias(priors, r.institut, r.bloc)["mean"] for r in df.itertuples()]
    df["ecart_deb"] = df["ecart"] - df["bias"]           # ≈ drift + bruit

    # fenêtres d'horizon (jours avant le scrutin)
    bins = [0, 15, 45, 90, 180, 400]
    df["hbin"] = pd.cut(df["horizon"], bins=bins)
    g = (df.groupby(["election", "candidat", "bloc", "hbin"], observed=True)
           .agg(drift=("ecart_deb", "mean"),
                horizon=("horizon", "mean"),
                n=("ecart_deb", "size"))
           .reset_index())
    g = g[g["n"] >= 2]                                    # au moins 2 sondages -> bruit réduit
    g["drift_norm"] = g["drift"] / np.sqrt(g["horizon"]) # échelle √h
    return g


def empirical_drift_norm(priors: dict | None = None) -> np.ndarray:
    """Pool des drifts réels mesurés (drift / √horizon) sur 2017+2022, à
    ré-échantillonner par bootstrap dans la simulation 2027."""
    return debiased_drift(priors or load_priors())["drift_norm"].to_numpy()


def main():
    priors = load_priors()
    g = debiased_drift(priors)
    d = g["drift_norm"].to_numpy()
    tau = priors["variance_globale"]["tau_derive"]["mean"]

    # moments empiriques
    sd = d.std(ddof=1)
    from scipy.stats import kurtosis, skew, t as student_t
    exk = kurtosis(d)            # excès de kurtosis (0 = gaussien)
    sk = skew(d)
    print(f"Drift réel mesuré (2017+2022), {len(d)} points (candidat×fenêtre) :")
    print(f"  sd(drift/√h)      = {sd:.3f}   (tau_derive calibré = {tau:.3f})")
    print(f"  skew              = {sk:+.2f}")
    print(f"  excès de kurtosis = {exk:+.2f}   (>0 ⇒ queues plus épaisses qu'une gaussienne)")
    # plus gros drifts observés (en points, ramenés à un horizon de référence)
    g2 = g.reindex(g["drift"].abs().sort_values(ascending=False).index)
    print("\n  Plus gros drifts réels observés (points) :")
    for r in g2.head(6).itertuples():
        print(f"    {r.election} {r.candidat:22s} {r.drift:+5.1f} pts à ~J-{r.horizon:.0f}")

    # ajustement d'un Student-t sur drift/√h
    df_t, loc_t, scale_t = student_t.fit(d)
    print(f"\n  Student-t ajusté sur drift/√h : df={df_t:.1f}, scale={scale_t:.3f}")

    # --- figure : Q-Q + histogramme vs gaussienne/Student-t ---
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
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "drift_reel_2017_2022.png"
    fig.savefig(out, dpi=110)
    print(f"\nFigure : {out}")
    return dict(sd=float(sd), excess_kurtosis=float(exk), df_t=float(df_t), scale_t=float(scale_t))


if __name__ == "__main__":
    main()
