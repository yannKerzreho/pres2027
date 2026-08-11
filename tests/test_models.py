"""Tests du framework multi-modèles (contrat de sortie).

Chaque modèle enregistré doit produire un snapshot valide → un contributeur qui
ajoute son modèle est couvert automatiquement par ces tests.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model.core.registered  # noqa: F401  (peuple le registre)
from model.core.base import all_models, validate_snapshot
from model.core.live_dataset import load_raw_polls

PARSED = ROOT / "data" / "parsed" / "intentions_2027_wiki.csv"
BANK = ROOT / "model" / "models" / "bayesian_nowcast" / "bank.json"
needs = pytest.mark.skipif(not (PARSED.exists() and BANK.exists()),
                           reason="données live / banque de paramètres absentes")

AS_OF = "2026-08-06"


def test_registre_non_vide():
    assert set(all_models()) >= {"bayesian-nowcast", "bayesian-nowcast-no-house-effects"}


def test_raw_polls_riches():
    # Gemini : run() doit recevoir les données les plus riches possibles.
    if not PARSED.exists():
        pytest.skip("intentions_2027_wiki.csv absent")
    raw = load_raw_polls(as_of=AS_OF)
    for col in ("institut", "echantillon", "methode", "hypothese", "candidat", "commanditaire"):
        assert col in raw.columns, f"colonne brute manquante : {col}"


@needs
@pytest.mark.parametrize("model_id", list(all_models()))
def test_modele_respecte_le_contrat(model_id):
    raw = load_raw_polls(as_of=AS_OF)
    m = all_models()[model_id]
    for a in ("draws", "tune"):        # accélère les modèles bayésiens en test
        if hasattr(m, a):
            setattr(m, a, 150)
    if hasattr(m, "chains"):
        m.chains = 2
    snap = m.run(raw, AS_OF)
    validate_snapshot(snap)            # ne lève pas
    assert snap["model"] == model_id
    fc = snap["forecast_scrutin"]
    # exactement 2 qualifiés / 1 premier par tirage
    assert abs(sum(f["p_qualifie_top2"] for f in fc.values()) - 2.0) < 0.05
    assert abs(sum(f["p_arrive_premier"] for f in fc.values()) - 1.0) < 0.03
    # les sondages exposés pour le front sont dans l'espace des slots
    assert snap["polls"] and all("slot" in p for p in snap["polls"])
