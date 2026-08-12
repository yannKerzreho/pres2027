"""Mouvements d'opinion réellement observés en 2017 et 2022.

Jeu de données empirique partagé : pour chaque candidat, l'écart en log-ratio
centré (CLR) entre un sondage et le résultat du scrutin, avec l'horizon auquel
il a été mesuré. Mesuré, pas supposé.

C'est la matière première de toute loi de projection jusqu'au scrutin. Ce module
n'en ajuste aucune : la mesure et le modèle qui l'exploite sont séparés, parce
que la même quantité sert à estimer deux choses distinctes — la diffusion
d'opinion (qui grandit avec l'horizon) et l'écart entre intention mesurée et
bulletin déposé (qui n'en dépend pas). Cf.
`model/models/gp_pooling/terminal.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.core.bank import Bank
from model.core.utils import clr
from pipeline.historical import load_calibration_frame, load_results


def institut_bias_prior(bank: Bank | None, institut: str, bloc: str) -> tuple[float, float]:
    """(mean, sd) du biais institut×bloc calibré sur l'historique. Repli sur la
    moyenne de population du bloc si l'institut est inconnu (shrinkage) ; prior
    faible et centré si `bank is None` — un modèle qui ne modélise pas de house
    effects passe `None` et travaille alors sur des sondages non débiaisés,
    ce qui est le comportement voulu, pas une dégradation silencieuse."""
    if bank is None:
        return 0.0, 5.0
    try:
        return bank.institut_bias.at(institut=institut, bloc=bloc)
    except KeyError:
        return bank.bloc_bias_mean.at(bloc=bloc)


def movement_pool(bank: Bank | None, hmin: float = 45.0) -> pd.DataFrame:
    """Mouvements RÉELS d'opinion 2017/2022, en espace log-ratio (CLR), par
    candidat × élection × fenêtre d'horizon — le jeu de données du fit
    paramétrique du saut terminal.

    Débiaise chaque sondage historique du biais calibré (si une `bank` est
    fournie), moyenne les sondages proches (même candidat, même fenêtre
    d'horizon) pour annuler le bruit d'échantillonnage, puis calcule le
    mouvement `clr(résultat) − clr(sondage_débiaisé)`.

    Bloc ET horizon (moyen de la fenêtre) sont conservés par mouvement : `loc`
    est fitté par bloc, et `TerminalJumpCalibration` normalise par horizon
    avant de fitter.
    """
    df = load_calibration_frame()
    df["bias"] = [institut_bias_prior(bank, r.institut, r.bloc)[0] for r in df.itertuples()]
    df["deb"] = df["intention"] - df["bias"]
    res = load_results()
    res = res[res["tour"] == "Premier tour"]

    df["hbin"] = pd.cut(df["horizon"], [0, 45, 90, 180, 400])
    moves: list[float] = []
    blocs: list[str] = []
    horizons: list[float] = []
    for elec in df["election"].unique():
        de = df[df["election"] == elec]
        cand = sorted(de["candidat"].unique())
        bloc_of = de.drop_duplicates("candidat").set_index("candidat")["bloc"]
        rmap = dict(zip(res.loc[res.election == elec, "candidat"],
                        res.loc[res.election == elec, "resultat"]))
        r = np.array([rmap.get(c, np.nan) for c in cand])
        for _, grp in de.groupby("hbin", observed=True):
            h = grp["horizon"].mean()
            if h < hmin:
                continue
            smap = grp.groupby("candidat")["deb"].mean()
            s = np.array([smap.get(c, np.nan) for c in cand])
            mask = ~np.isnan(s) & ~np.isnan(r) & (s > 0)
            if mask.sum() < 4:
                continue
            cand_kept = [c for c, keep in zip(cand, mask) if keep]
            moves.extend((clr(r[mask]) - clr(s[mask])).tolist())
            blocs.extend(bloc_of.loc[cand_kept].tolist())
            horizons.extend([float(h)] * int(mask.sum()))
    return pd.DataFrame({"bloc": blocs, "move": moves, "horizon": horizons})
