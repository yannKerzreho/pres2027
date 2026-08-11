"""Tests du modèle « linear-pooling » (lissage par demi-vie, sans variable
latente) — cf. model/models/linear_pooling/spec_linear_pooling.md.

Vérifie les formules du cœur mathématique indépendamment du framework
(test_models.py couvre déjà le contrat de sortie générique) : poids demi-vie,
récupération de la moyenne du mélange (§4.1 de la spec), renormalisation sur
le simplexe, repli sur un slot jamais testé.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.models.linear_pooling.model import (
    half_life_weight, linear_pooling_nowcast, resample_slot,
)


def test_half_life_weight_decroit_de_moitie_par_demi_vie():
    n = np.array([1000.0])
    w0 = half_life_weight(np.array([0.0]), n, half_life_days=14.0)
    w1 = half_life_weight(np.array([14.0]), n, half_life_days=14.0)
    assert w0[0] == pytest.approx(1000.0)
    assert w1[0] == pytest.approx(500.0)


def test_half_life_weight_proportionnel_a_l_echantillon():
    age = np.array([10.0, 10.0])
    n = np.array([500.0, 1000.0])
    w = half_life_weight(age, n, half_life_days=14.0)
    assert w[1] == pytest.approx(2 * w[0])


def _sub(rows):
    return pd.DataFrame(rows)


def test_resample_slot_recupere_l_estimateur_ponctuel_en_moyenne():
    # §4.1 de la spec : E[mélange] = estimateur pondéré, preuve directe.
    as_of = pd.Timestamp("2026-08-10")
    sub = _sub([
        {"date_fin": "2026-07-31", "echantillon": 1000, "n_hypotheses": 1, "part": 34.0},
        {"date_fin": "2026-07-16", "echantillon": 800, "n_hypotheses": 1, "part": 31.0},
        {"date_fin": "2026-08-05", "echantillon": 1200, "n_hypotheses": 1, "part": 35.0},
    ])
    rng = np.random.default_rng(0)
    draws, age, diag = resample_slot(sub, as_of, half_life_days=14.0, n_draws=200_000, rng=rng)
    assert draws.mean() == pytest.approx(diag["point_estimate"], abs=0.002)
    # ordre de grandeur de l'exemple chiffré §4.3 (34.14 %)
    assert diag["point_estimate"] == pytest.approx(0.3414, abs=0.001)


def test_resample_slot_variance_superieure_a_la_variance_de_moyenne_classique():
    # §4/§4.3 : la variance du mélange (désaccord + bruit) doit dépasser la
    # variance classique de la moyenne pondérée (qui, elle, s'annule avec
    # plus de données) — sinon le mélange n'apporte rien face au sondage B,
    # nettement en dehors du groupe A/C.
    as_of = pd.Timestamp("2026-08-10")
    sub = _sub([
        {"date_fin": "2026-07-31", "echantillon": 1000, "n_hypotheses": 1, "part": 34.0},
        {"date_fin": "2026-07-16", "echantillon": 800, "n_hypotheses": 1, "part": 31.0},
        {"date_fin": "2026-08-05", "echantillon": 1200, "n_hypotheses": 1, "part": 35.0},
    ])
    rng = np.random.default_rng(1)
    draws, _, _ = resample_slot(sub, as_of, half_life_days=14.0, n_draws=200_000, rng=rng)
    mixture_sd = draws.std()
    # variance classique de la moyenne pondérée : Σ w_p² σ_p² / Ω²
    age = (as_of - pd.to_datetime(sub["date_fin"])).dt.days.to_numpy().astype(float)
    n_eff = sub["echantillon"].to_numpy().astype(float)
    y = sub["part"].to_numpy() / 100.0
    w = half_life_weight(age, n_eff, 14.0)
    sigma2 = y * (1 - y) / n_eff
    classic_var = (w ** 2 * sigma2).sum() / w.sum() ** 2
    assert mixture_sd > np.sqrt(classic_var)


def test_resample_slot_repli_si_jamais_teste():
    as_of = pd.Timestamp("2026-08-10")
    rng = np.random.default_rng(2)
    draws, age, diag = resample_slot(pd.DataFrame(columns=["date_fin", "echantillon",
                                                            "n_hypotheses", "part"]),
                                     as_of, half_life_days=14.0, n_draws=1000, rng=rng)
    assert diag["n_polls"] == 0
    assert 0.0 < draws.mean() < 0.05   # repli faible (~1%), jamais 0 ni dominant


def _raw(rows):
    """Trame au format `load_raw_polls` : `scenario_id` identifie le champ de
    candidats réellement soumis (institut × date × hypothèse), ce dont dépend le
    filtre de roster exact."""
    d = pd.DataFrame(rows)
    d["notice_url"] = "http://x/" + d["notice"]
    d["hypothese"] = None
    return d


SLOTS3 = ["RN", "Philippe (Horizons)", "Mélenchon (LFI)"]


def test_linear_pooling_nowcast_simplexe_par_tirage():
    as_of = "2026-08-10"
    # Deux scénarios testant EXACTEMENT les 3 slots — seuls admissibles.
    raw = _raw([
        {"scenario_id": "a", "notice": "s1", "institut": "Ifop", "date_fin": "2026-08-01",
         "echantillon": 1000, "candidat": "Marine Le Pen", "intention": 45.0,
         "slot": "RN", "bloc": "droite_radicale"},
        {"scenario_id": "a", "notice": "s1", "institut": "Ifop", "date_fin": "2026-08-01",
         "echantillon": 1000, "candidat": "Édouard Philippe", "intention": 32.0,
         "slot": "Philippe (Horizons)", "bloc": "centre"},
        {"scenario_id": "a", "notice": "s1", "institut": "Ifop", "date_fin": "2026-08-01",
         "echantillon": 1000, "candidat": "Jean-Luc Mélenchon", "intention": 23.0,
         "slot": "Mélenchon (LFI)", "bloc": "gauche_radicale"},
        {"scenario_id": "b", "notice": "s2", "institut": "Elabe", "date_fin": "2026-08-05",
         "echantillon": 900, "candidat": "Marine Le Pen", "intention": 47.0,
         "slot": "RN", "bloc": "droite_radicale"},
        {"scenario_id": "b", "notice": "s2", "institut": "Elabe", "date_fin": "2026-08-05",
         "echantillon": 900, "candidat": "Édouard Philippe", "intention": 30.0,
         "slot": "Philippe (Horizons)", "bloc": "centre"},
        {"scenario_id": "b", "notice": "s2", "institut": "Elabe", "date_fin": "2026-08-05",
         "echantillon": 900, "candidat": "Jean-Luc Mélenchon", "intention": 23.0,
         "slot": "Mélenchon (LFI)", "bloc": "gauche_radicale"},
    ])
    res = linear_pooling_nowcast(raw, as_of, SLOTS3, half_life_days=14.0, n_draws=500, seed=0)
    sums = res.pi.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-9)
    assert res.pi.shape == (500, 3)
    # Le champ étant complet (les parts somment à 100), la renormalisation ne
    # déplace pas l'estimation : RN doit retomber sur ~46 %, sa part mesurée.
    assert 0.40 < res.pi[:, 0].mean() < 0.52


def test_scenario_au_champ_different_est_ecarte():
    """Le cœur du filtre : une hypothèse testant un candidat de PLUS ne mesure
    pas la même chose (le vote se répartit autrement) et doit être écartée,
    même si elle contient bien les trois slots demandés."""
    as_of = "2026-08-10"
    base = [
        {"scenario_id": "a", "notice": "s1", "institut": "Ifop", "date_fin": "2026-08-01",
         "echantillon": 1000, "candidat": "Marine Le Pen", "intention": 45.0,
         "slot": "RN", "bloc": "droite_radicale"},
        {"scenario_id": "a", "notice": "s1", "institut": "Ifop", "date_fin": "2026-08-01",
         "echantillon": 1000, "candidat": "Édouard Philippe", "intention": 32.0,
         "slot": "Philippe (Horizons)", "bloc": "centre"},
        {"scenario_id": "a", "notice": "s1", "institut": "Ifop", "date_fin": "2026-08-01",
         "echantillon": 1000, "candidat": "Jean-Luc Mélenchon", "intention": 23.0,
         "slot": "Mélenchon (LFI)", "bloc": "gauche_radicale"},
    ]
    intrus = dict(base[0], candidat="Gabriel Attal", intention=12.0, slot=None, bloc=None)

    from model.core.live_dataset import filter_scenarios_by_exact_slots
    assert not filter_scenarios_by_exact_slots(_raw(base), set(SLOTS3)).empty
    assert filter_scenarios_by_exact_slots(_raw(base + [intrus]), set(SLOTS3)).empty

    # et le modèle refuse alors de produire un nowcast plutôt que d'en inventer un
    with pytest.raises(ValueError, match="roster"):
        linear_pooling_nowcast(_raw(base + [intrus]), as_of, SLOTS3,
                               half_life_days=14.0, n_draws=100, seed=0)
