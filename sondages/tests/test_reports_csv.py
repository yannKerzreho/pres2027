"""Contrat des CSV versionnés : reports T1->T2 et élections 2002-2012."""

from pathlib import Path

import pandas as pd
import pytest

from sondages import schema

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("annee", ["2017", "2022"])
def test_reports_somment_a_100_par_electorat(annee):
    from sondages.reports import COLUMNS
    df = pd.read_csv(ROOT / "data" / "parsed" / f"reports_t2_{annee}.csv")
    assert list(df.columns) == COLUMNS and len(df) > 100
    s = df.groupby(["notice", "electorat_t1"])["part"].sum()
    assert s.between(97, 103).all(), s[~s.between(97, 103)]
    assert df["date_fin"].notna().all() and df["source"].notna().all()
    assert (df["date_fin"] > f"{annee}-04-01").all()


@pytest.mark.parametrize("annee", ["2002", "2007", "2012"])
def test_elections_anciennes(annee):
    it = pd.read_csv(ROOT / "data" / "parsed" / f"intentions_{annee}_wiki.csv")
    schema.validate(it)
    t2 = it[it["tour"] == "Deuxième tour"]
    assert t2["notice"].nunique() >= 25 and t2["date_fin"].notna().all()
    assert (it["date_fin"].dropna() <= f"{annee}-05-31").all()
    res = pd.read_csv(ROOT / "data" / "historical" / f"results_{annee}.csv")
    for tour, g in res.groupby("tour"):
        assert abs(g["resultat"].sum() - 100) < 0.1, (annee, tour)
    assert set(res["candidat"]) <= set(it["candidat"])          # noms alignés sur les sondages
