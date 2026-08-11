"""Backtest de COUVERTURE : les IC 90 % publiés tiennent-ils leur promesse ?

Un intervalle de crédibilité à 90 % qui ne contient le résultat réel que 60 %
du temps est un mensonge chiffré — c'est le défaut le plus grave possible pour
ce projet, et il est invisible sans ce test : rien dans les sorties du modèle
ne le signale.

Méthode : on rejoue `linear-pooling` **à l'identique** sur 2017 et 2022, à
plusieurs horizons, et on compte combien de fois le résultat officiel tombe
dans l'intervalle annoncé. Même mélange pondéré par demi-vie, même bruit Beta
d'échantillonnage, même saut d'opinion en deux jambes (sondage → `as_of`, puis
`as_of` → scrutin) — cf. `model/models/linear_pooling/model.py`.

Une couverture nettement SOUS 90 % = intervalles trop étroits (surconfiance) ;
nettement AU-DESSUS = intervalles trop larges (modèle inutilement vague).

Exécution : .venv/bin/python -m model.backtest.coverage
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from model.core.bank import Bank
from model.core.terminal_jump import jump_moves
from model.models.linear_pooling.model import BANK_JUMP_PATH, half_life_weight
from pipeline.historical import load_calibration_frame, load_results

HORIZONS = [30, 60, 90, 120, 180, 250, 300]
HALF_LIFE = 14.0
N_DRAWS = 4000
SEED = 27


def _polls_at(df: pd.DataFrame, h_as_of: float) -> pd.DataFrame:
    """Sondages disponibles à `h_as_of` jours du scrutin : ceux terminés AVANT,
    donc d'horizon supérieur. Une moyenne par (sondage, candidat) évite qu'un
    sondage à plusieurs hypothèses pèse plusieurs fois."""
    d = df[df["horizon"] >= h_as_of]
    return (d.groupby(["id", "candidat", "bloc"], as_index=False)
             .agg(horizon=("horizon", "mean"), intention=("intention", "mean"),
                  echantillon=("echantillon", "median")))


def forecast_at(df: pd.DataFrame, h_as_of: float, jump_bank: Bank,
                rng: np.random.Generator, seed: int) -> tuple[list[str], np.ndarray]:
    """Tirages (S, K) des parts au scrutin, vues depuis `h_as_of`. Reproduit la
    chaîne du modèle live ; renvoie (candidats, tirages)."""
    sub = _polls_at(df, h_as_of)
    cands = sorted(sub["candidat"].unique())
    if not cands:
        return [], np.empty((0, 0))

    theta = np.empty((N_DRAWS, len(cands)))
    ages = np.zeros((N_DRAWS, len(cands)))
    blocs = []
    for k, c in enumerate(cands):
        s = sub[sub["candidat"] == c]
        blocs.append(s["bloc"].iloc[0])
        n = s["echantillon"].to_numpy(dtype=float)
        n = np.where(np.isfinite(n) & (n > 0), n, 1000.0)
        y = np.clip(s["intention"].to_numpy(dtype=float) / 100.0, 1e-4, 1 - 1e-4)
        age = s["horizon"].to_numpy(dtype=float) - h_as_of
        w = half_life_weight(age, n, HALF_LIFE)
        if w.sum() <= 0:                      # tous les sondages hors de portée
            w = np.ones_like(w)
        p = w / w.sum()
        pick = rng.choice(len(s), size=N_DRAWS, p=p)
        theta[:, k] = rng.beta(np.clip(y[pick] * n[pick], 1e-3, None),
                               np.clip((1 - y[pick]) * n[pick], 1e-3, None))
        ages[:, k] = age[pick]

    def _softmax(a):
        e = np.exp(a - a.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    # Jambe 1 : du sondage sélectionné jusqu'à `as_of` (âge propre à chaque tirage).
    alpha = np.log(np.clip(theta, 1e-6, None))
    alpha = alpha + jump_moves(jump_bank, blocs, h_from=h_as_of + ages, h_to=h_as_of,
                               size=alpha.shape, seed=seed + 1)
    # Jambe 2 : de `as_of` jusqu'au scrutin.
    alpha = np.log(np.clip(_softmax(alpha), 1e-6, None))
    alpha = alpha + jump_moves(jump_bank, blocs, h_from=h_as_of, h_to=0.0,
                               size=alpha.shape, seed=seed + 2)
    return cands, _softmax(alpha)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--niveau", type=float, default=0.90,
                    help="niveau nominal de l'intervalle testé (défaut 0.90)")
    args = ap.parse_args()
    lo_q, hi_q = 50 * (1 - args.niveau), 100 - 50 * (1 - args.niveau)

    jump_bank = Bank.load(BANK_JUMP_PATH)
    df = load_calibration_frame()
    res = load_results()
    res = res[res["tour"] == "Premier tour"]
    rng = np.random.default_rng(SEED)

    lignes = []
    for elec in sorted(df["election"].unique()):
        de = df[df["election"] == elec]
        verite = dict(zip(res.loc[res.election == elec, "candidat"],
                          res.loc[res.election == elec, "resultat"]))
        for h in HORIZONS:
            cands, pi = forecast_at(de, float(h), jump_bank, rng, SEED)
            if not cands:
                continue
            for k, c in enumerate(cands):
                if c not in verite:
                    continue
                lo, hi = np.percentile(pi[:, k], [lo_q, hi_q]) * 100
                lignes.append({"election": elec, "horizon": h, "candidat": c,
                               "reel": verite[c], "lo": lo, "hi": hi,
                               "dans": lo <= verite[c] <= hi,
                               "largeur": hi - lo})
    t = pd.DataFrame(lignes)
    if t.empty:
        raise SystemExit("Aucune observation testable.")

    n = int(args.niveau * 100)
    print(f"Couverture des IC {n} % — {len(t)} observations "
          f"(candidat × horizon × élection)\n")
    print(f"{'horizon':>8s} {'n':>4s} {'couverture':>11s} {'largeur med.':>13s}")
    for h, g in t.groupby("horizon"):
        print(f"{h:8d} {len(g):4d} {g['dans'].mean():11.2f} {g['largeur'].median():12.1f} pt")
    print(f"\n{'par élection':>12s}")
    for e, g in t.groupby("election"):
        print(f"{e:12} {len(g):4d} {g['dans'].mean():11.2f}")
    glob = t["dans"].mean()
    print(f"\nCOUVERTURE GLOBALE : {glob:.2f}  (nominal {args.niveau:.2f})")
    if glob < args.niveau - 0.05:
        print("  -> SOUS-COUVERTURE : intervalles trop étroits, modèle surconfiant.")
    elif glob > args.niveau + 0.05:
        print("  -> SUR-COUVERTURE : intervalles trop larges, modèle peu informatif.")
    else:
        print("  -> calibration acceptable.")

    rates = t[~t["dans"]]
    if not rates.empty:
        print(f"\n{len(rates)} observations hors intervalle, les plus marquées :")
        rates = rates.assign(ecart=np.where(rates["reel"] < rates["lo"],
                                            rates["lo"] - rates["reel"],
                                            rates["reel"] - rates["hi"]))
        for r in rates.nlargest(8, "ecart").itertuples():
            cote = "sous" if r.reel < r.lo else "au-dessus"
            print(f"  {r.election} J-{r.horizon:<3d} {r.candidat:22s} réel {r.reel:5.1f} "
                  f"vs [{r.lo:5.1f};{r.hi:5.1f}]  ({cote}, écart {r.ecart:.1f} pt)")


if __name__ == "__main__":
    main()
