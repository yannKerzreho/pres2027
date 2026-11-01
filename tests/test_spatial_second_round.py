from __future__ import annotations

import numpy as np
import pandas as pd

from model.models.spatial_pooling.fit import SpatialPoolingFit, second_round_from_fit


def _fit() -> SpatialPoolingFit:
    # Deux tirages suffisent : le test vérifie le branchement et que le score
    # vient de w/mu/sigma partagés, pas d'un agrégateur T2 séparé.
    return SpatialPoolingFit(
        candidates=["Centre", "Le Pen", "Mélenchon"], anchors=np.array([.5, .8, .2]),
        tested_mask=np.ones((1, 3)),
        mu_draws=np.array([[.5, .8, .2], [.5, .8, .2]]),
        sigma_draws=np.full((2, 3), .15),
        w_draws=np.array([[0., 1., 0.], [1., 0., 0.]]), diagnostics={},
    )


def test_second_round_is_evaluated_from_the_shared_spatial_draws():
    raw = pd.DataFrame([
        {"tour": "Deuxième tour", "notice": "n1", "hypothese": "duel", "candidat": "Centre"},
        {"tour": "Deuxième tour", "notice": "n1", "hypothese": "duel", "candidat": "Le Pen"},
    ])
    out = second_round_from_fit(_fit(), raw)
    assert len(out) == 1
    duel = out[0]
    assert duel["candidats"] == ["Centre", "Le Pen"]
    assert duel["n_sondages_directs"] == 1
    assert duel["modele"] == "posterior spatial conjoint T1 + T2"
    assert duel["intentions"]["Centre"]["moyenne"] + duel["intentions"]["Le Pen"]["moyenne"] == 1.0


def test_second_round_ignores_a_non_duel_or_candidate_outside_roster():
    raw = pd.DataFrame([
        {"tour": "Deuxième tour", "notice": "n1", "hypothese": "x", "candidat": "Centre"},
        {"tour": "Deuxième tour", "notice": "n1", "hypothese": "x", "candidat": "Le Pen"},
        {"tour": "Deuxième tour", "notice": "n1", "hypothese": "x", "candidat": "Inconnu"},
    ])
    assert second_round_from_fit(_fit(), raw) == []
