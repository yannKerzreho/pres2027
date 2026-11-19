"""Tests de `model/derniers_sondages.py` : date du dernier sondage par nom affiché."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.derniers_sondages import FICHIER, dates_par_nom, produire


def _raw():
    return pd.DataFrame([
        {"notice": "a", "date_fin": pd.Timestamp("2026-06-24"), "candidat": "Bardella", "slot": None},
        {"notice": "a", "date_fin": pd.Timestamp("2026-06-24"), "candidat": "Le Pen", "slot": "RN"},
        {"notice": "b", "date_fin": pd.Timestamp("2026-11-17"), "candidat": "Le Pen", "slot": "RN"},
        {"notice": "c", "date_fin": pd.Timestamp("2026-12-01"), "candidat": "Bardella", "slot": None},
    ])


def test_cles_nom_et_slot_sans_alias():
    d = dates_par_nom(_raw()[lambda x: x["date_fin"] <= "2026-11-30"])
    assert d["Le Pen"] == {"dernier": "2026-11-17", "n_sondages": 2}
    assert d["RN"]["dernier"] == "2026-11-17"
    # Bardella n'entre pas dans « RN » : sa date reste la sienne
    assert d["Bardella"] == {"dernier": "2026-06-24", "n_sondages": 1}


def test_produire_respecte_as_of(tmp_path):
    m = produire(_raw(), "2026-11-18", dossier=tmp_path)
    assert m["noms"]["Bardella"]["dernier"] == "2026-06-24"      # le sondage du 1er déc. est futur
    sur_disque = json.loads((tmp_path / FICHIER).read_text())
    assert sur_disque["as_of"] == "2026-11-18" and sur_disque["noms"] == m["noms"]
