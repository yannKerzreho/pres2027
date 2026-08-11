"""Tests du modèle live 2027 (Phase 3) — logique rapide, sans échantillonnage."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.core.live_dataset import map_candidate, load_live_observations, SLOTS

PARSED = ROOT / "data" / "parsed" / "intentions_2027_wiki.csv"
needs_data = pytest.mark.skipif(not PARSED.exists(), reason="intentions_2027_wiki.csv absent")


def test_slots_fusionnent_les_alternatives():
    # Modèle de base = une seule hypothèse par bloc (cf. SLOTS, session du
    # 2026-08-09) : Le Pen est LE candidat RN retenu, Bardella (scénario
    # d'inéligibilité) n'est plus mappé du tout.
    assert map_candidate("Marine Le Pen")[0] == "RN"
    assert map_candidate("Jordan Bardella") == (None, None)
    # variantes d'accent / trait d'union canonicalisées
    assert map_candidate("Eric Zemmour") == map_candidate("Éric Zemmour")
    assert map_candidate("Nicolas Dupont Aignan")[0] == "Dupont-Aignan"
    assert map_candidate("Édouard Philippe")[0] == "Philippe (Horizons)"
    assert map_candidate("Gabriel Attal") == (None, None)


def test_chaque_slot_a_un_bloc_connu():
    blocs = {"gauche_radicale", "gauche", "ecologistes", "centre", "droite", "droite_radicale"}
    for slot, (bloc, names) in SLOTS.items():
        assert bloc in blocs


@needs_data
def test_observations_horizon_positif():
    obs = load_live_observations()
    assert len(obs) > 0
    assert (obs["horizon"] >= 0).all()
    assert obs["slot"].isin(SLOTS).all()


def test_forecast_from_draws_probabilites_bien_formees():
    # tirages π synthétiques : connu et net (RN dominant), pour vérifier la logique.
    from model.core.simulate import forecast_from_draws

    slots = list(SLOTS)
    K = len(slots)
    base = np.full(K, 0.5 / (K - 1))
    base[slots.index("RN")] = 0.5           # RN très haut
    pi = np.random.default_rng(0).dirichlet(base * 200, size=1000)  # (S, K)

    res = forecast_from_draws(pi, slots, forecast_horizon=200)
    fc = res["forecast_scrutin"]
    # probabilités dans [0,1]
    for s in slots:
        assert 0.0 <= fc[s]["p_qualifie_top2"] <= 1.0
        assert 0.0 <= fc[s]["p_arrive_premier"] <= 1.0
    # exactement 2 qualifiés par tirage -> somme des P(top2) ≈ 2
    assert abs(sum(fc[s]["p_qualifie_top2"] for s in slots) - 2.0) < 0.05
    # somme des P(1er) ≈ 1
    assert abs(sum(fc[s]["p_arrive_premier"] for s in slots) - 1.0) < 0.02
    # RN dominant doit être quasi certainement qualifié
    assert fc["RN"]["p_qualifie_top2"] > 0.9
    # la dérive ajoute de l'incertitude (sd log-ratio > 0)
    assert res["drift_sd_logratio"] > 0
    # simplexe respecté : toutes les parts prévues dans [0, 1]
    for s in slots:
        assert 0.0 <= fc[s]["part_moyenne"] <= 1.0
        assert 0.0 <= fc[s]["ic90"][0] <= fc[s]["ic90"][1] <= 1.0
