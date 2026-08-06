"""Tests des priors calibrés (calibration/priors.json).

Ces tests ne relancent pas l'inférence (coûteuse) ; ils vérifient la structure et
la cohérence du fichier de priors produit par la Phase 1, ainsi que les
utilitaires de consommation. Ils sont ignorés si priors.json n'a pas encore été
généré.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calibration.priors_utils import (
    PRIORS_PATH, load_priors, expected_bias,
    excess_sigma, observation_sigma, sampling_sigma, forecast_drift_sigma,
)

pytestmark = pytest.mark.skipif(
    not PRIORS_PATH.exists(),
    reason="priors.json absent — lancer d'abord calibration/fit_house_effects.py",
)


def test_structure_priors():
    p = load_priors()
    for key in ("meta", "variance_globale", "bloc", "institut"):
        assert key in p
    assert "log_h_mean" in p["meta"] and "log_h_std" in p["meta"]


def test_convergence_rhat():
    p = load_priors()
    assert p["meta"]["rhat_max"] < 1.05


def test_excess_croit_avec_horizon():
    # b_h > 0 mesuré : l'excès de dispersion doit croître avec l'horizon.
    p = load_priors()
    inst = next(iter(p["institut"]))
    assert excess_sigma(p, inst, 5) < excess_sigma(p, inst, 180)


def test_observation_sigma_domine_par_echantillonnage_a_court_horizon():
    # À J-1 sur un gros bloc, la dispersion doit être proche du plancher
    # d'échantillonnage (l'excès house-effect est faible près du scrutin).
    p = load_priors()
    inst = next(iter(p["institut"]))
    samp = sampling_sigma(p, 25.0, 1000)
    obs = observation_sigma(p, inst, 1, 25.0, 1000)
    assert obs >= samp                      # l'obs inclut le plancher
    assert obs < 2.5 * samp                 # près du scrutin, pas dominé par l'excès


def test_forecast_drift_nul_le_jour_du_vote_et_croissant():
    # La dérive future s'annule à J-0 et grandit avec l'horizon (IC temps réel).
    p = load_priors()
    assert forecast_drift_sigma(p, 0) == 0.0
    assert forecast_drift_sigma(p, 10) < forecast_drift_sigma(p, 120)


def test_design_effect_superieur_a_un():
    # Correction quotas : le design effect doit être >= 1.
    p = load_priors()
    assert p["variance_globale"]["design_effect_quotas"]["mean"] >= 1.0


def test_shrinkage_institut_inconnu():
    p = load_priors()
    bloc = next(iter(p["bloc"]))
    res = expected_bias(p, "InstitutInexistant2027", bloc)
    assert res["source"] == "population_bloc"
    assert res["mean"] == p["bloc"][bloc]["biais_moyen_population"]["mean"]


def test_biais_droite_bien_reduit_par_decomposition():
    # Après séparation biais / dérive, le biais à J-0 du bloc droite doit être
    # nettement plus petit que l'écart brut confondu (~+6 pts) : l'essentiel de
    # l'effondrement Pécresse part dans la dérive, pas dans le biais.
    p = load_priors()
    assert p["bloc"]["droite"]["biais_moyen_population"]["mean"] < 4.5


def test_derive_specifique_a_l_election():
    # La dérive du bloc droite doit être de signe opposé entre 2017 (Fillon) et
    # 2022 (Pécresse) : c'est ce qui justifie une dérive par élection, et la
    # dérive 2022 doit être positive (surestimation loin du scrutin).
    p = load_priors()
    d2017 = p["derive_temporelle"]["2017"]["droite"]["mean"]
    d2022 = p["derive_temporelle"]["2022"]["droite"]["mean"]
    assert d2022 > 0 > d2017


def test_deux_composantes_house_effect_exportees():
    # Le house effect a bien deux formes distinctes dans les priors :
    # biais directionnel par bloc ET précision générale (dispersion).
    p = load_priors()
    inst = next(iter(p["institut"].values()))
    assert "biais_par_bloc" in inst          # composante 1 : biais
    assert "precision_generale_log" in inst  # composante 2 : dispersion
