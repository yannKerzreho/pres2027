"""Tests du jeu de données de calibration (Phase 0/1)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.historical import load_calibration_frame, load_results, load_blocs


def test_calibration_frame_non_vide():
    df = load_calibration_frame()
    assert len(df) > 1000
    expected = {"election", "id", "institut", "horizon", "ecart",
                "bloc", "intention", "resultat"}
    assert expected.issubset(df.columns)


def test_ecart_est_intention_moins_resultat():
    df = load_calibration_frame()
    recomputed = df["intention"] - df["resultat"]
    assert (df["ecart"] - recomputed).abs().max() < 1e-9


def test_horizons_positifs():
    df = load_calibration_frame()
    assert (df["horizon"] >= 0).all()


def test_tous_les_candidats_ont_un_bloc():
    df = load_calibration_frame()
    assert df["bloc"].notna().all()


def test_candidats_de_calibration_sont_des_candidats_reels():
    df = load_calibration_frame()
    res = load_results()
    real = set(res[res.tour == "Premier tour"].candidat)
    assert set(df["candidat"]).issubset(real)


def test_blocs_connus():
    blocs = set(load_blocs().bloc)
    attendus = {
        "gauche_radicale", "gauche", "ecologistes", "centre",
        "droite", "droite_radicale", "divers",
    }
    assert blocs.issubset(attendus)
