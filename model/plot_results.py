"""Phase 3 — visualisation des résultats du modèle live (resultats_2027.json)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "notebooks" / "figures"


def main():
    d = json.loads((ROOT / "model" / "resultats_2027.json").read_text())
    fc = d["forecast_scrutin"]
    slots = sorted(fc, key=lambda s: fc[s]["part_moyenne"])
    means = np.array([fc[s]["part_moyenne"] * 100 for s in slots])
    lo = np.array([fc[s]["ic90"][0] * 100 for s in slots])
    hi = np.array([fc[s]["ic90"][1] * 100 for s in slots])
    ptop2 = [fc[s]["p_qualifie_top2"] for s in slots]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [2, 1]})

    # --- parts + IC 90 %, colorées par P(top2) ---
    cmap = plt.cm.viridis
    colors = [cmap(p) for p in ptop2]
    y = np.arange(len(slots))
    ax1.barh(y, means, color=colors, height=0.6)
    ax1.errorbar(means, y, xerr=[means - lo, hi - means], fmt="none",
                 ecolor="#4a5568", capsize=3, lw=1)
    for i, s in enumerate(slots):
        ax1.text(hi[i] + 0.6, i, f"{ptop2[i]:.0%}", va="center", fontsize=8, color="#2d3748")
    ax1.set_yticks(y); ax1.set_yticklabels(slots)
    ax1.set_xlabel("Part au 1er tour (%) — IC 90 %, couleur = P(qualifié top 2)")
    ax1.set_title(f"Prévision 1er tour 2027 (à J-{d['meta']['horizon_prevision_jours']}, "
                  f"{d['meta']['n_sondages']} sondages)")

    # --- duels de 2nd tour ---
    duels = d["duels_probables"][:6]
    labels = [" vs\n".join(x["candidats"]) for x in duels][::-1]
    probs = [x["probabilite"] for x in duels][::-1]
    ax2.barh(range(len(duels)), probs, color="#805ad5", height=0.6)
    for i, p in enumerate(probs):
        ax2.text(p + 0.01, i, f"{p:.0%}", va="center", fontsize=8)
    ax2.set_yticks(range(len(duels))); ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Probabilité du duel")
    ax2.set_title("Duels de 2nd tour probables")
    ax2.set_xlim(0, max(probs) * 1.25)

    fig.suptitle("Modèle live — présidentielle 2027 (probabilités, pas des prédictions)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "forecast_2027.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Figure écrite : {out}")


if __name__ == "__main__":
    main()
