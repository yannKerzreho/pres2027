"""Neutralité en production (revue du 24/11 ~17:00Z) : run.py ne charge QUE le premier tour.

Charger les T2 à cette frontière changeait listes.json, derniers-sondages.json et les alertes publiés
(commit d7b1575). Le câblage T2 attend PUBLIER_T2 et une PR dédiée."""
import pandas as pd
import pytest

import model.run as R


def test_run_charge_premier_tour_seulement(monkeypatch):
    vu = {}

    def faux(path=None, tour="Premier tour", as_of=None):
        vu["tour"] = tour
        return pd.DataFrame()                    # vide ⇒ run s'arrête aussitôt (SystemExit)
    monkeypatch.setattr(R, "load_raw_polls", faux)
    with pytest.raises(SystemExit):
        R.run("gp-pooling", as_of="2026-11-24")
    assert vu["tour"] == "Premier tour"


def test_publier_t2_reste_faux():
    from model.models.spatial_pooling import forecast_model
    assert forecast_model.PUBLIER_T2 is False
