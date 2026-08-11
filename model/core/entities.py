"""Transformer des colonnes catégorielles d'un DataFrame en index entiers
directement utilisables dans NumPyro (indexer un site latent comme
`biais[index]`), sans dictionnaire de correspondance écrit à la main dans
chaque modèle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def index_groups(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, list[tuple]]:
    """Pour chaque ligne de `df`, l'indice entier de sa combinaison de valeurs
    (`columns`) parmi les combinaisons uniques rencontrées, dans l'ordre
    d'apparition.

    Renvoie `(index_par_ligne, groupes)` : `groupes[k]` est le tuple de valeurs
    représenté par l'indice `k` — utile pour retrouver, par exemple, le nom de
    l'institut et du bloc associés à `biais[k]` après l'inférence.
    """
    keys = list(df[columns].itertuples(index=False, name=None))
    groups = list(dict.fromkeys(keys))
    position = {group: i for i, group in enumerate(groups)}
    index = np.array([position[k] for k in keys])
    return index, groups
