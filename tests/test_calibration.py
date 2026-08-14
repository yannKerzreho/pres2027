"""Tests de la Bank calibrée (model/models/bayesian_nowcast/bank.json).

Ces tests ne relancent pas l'inférence (coûteuse) ; ils vérifient la
structure et la cohérence de la Bank produite par
model/models/bayesian_nowcast.py, ainsi que les fonctions dérivées de
consommation. Ignorés si la Bank n'a pas encore été calibrée.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.core.bank import Bank

from model.models.bayesian_nowcast import (
    BANK_PATH, excess_sigma, forecast_drift_sigma, institut_bias_prior, sampling_sigma,
)

pytestmark = pytest.mark.skipif(
    not BANK_PATH.exists(),
    reason="Bank absente — lancer d'abord .venv/bin/python -m model.models.bayesian_nowcast",
)


@pytest.fixture(scope="module")
def bank() -> Bank:
    return Bank.load(BANK_PATH)


def test_structure_bank(bank):
    for site in ("bloc_bias_mean", "bloc_bias_sd", "institut_bias", "campaign_drift",
                "campaign_drift_sd", "excess_log_scale", "excess_institut_offset",
                "excess_horizon_slope", "horizon_scale_mean", "horizon_scale_std"):
        assert hasattr(bank, site), f"site manquant dans la Bank : {site}"


def test_excess_croit_avec_horizon(bank):
    # l'excès de dispersion doit croître avec l'horizon.
    inst = bank.institut_bias.mean.coords["institut"].values.tolist()[0]
    assert excess_sigma(bank, inst, 5) < excess_sigma(bank, inst, 180)


def test_observation_sigma_domine_par_echantillonnage_a_court_horizon(bank):
    # À J-1 sur un gros échantillon, la dispersion totale doit être proche du
    # plancher d'échantillonnage (l'excès house-effect est faible près du scrutin).
    inst = bank.institut_bias.mean.coords["institut"].values.tolist()[0]
    samp = sampling_sigma(25.0, 1000)
    obs = math.hypot(samp, excess_sigma(bank, inst, 1))
    assert obs >= samp
    assert obs < 2.5 * samp


def test_forecast_drift_nul_le_jour_du_vote_et_croissant(bank):
    # La dérive future s'annule à J-0 et grandit avec l'horizon (IC temps réel).
    assert forecast_drift_sigma(bank, 0) == 0.0
    assert forecast_drift_sigma(bank, 10) < forecast_drift_sigma(bank, 120)


def test_shrinkage_institut_inconnu(bank):
    bloc = bank.bloc_bias_mean.mean.coords["bloc"].values.tolist()[0]
    res = institut_bias_prior(bank, "InstitutInexistant2027", bloc)
    assert res == bank.bloc_bias_mean.at(bloc=bloc)


def test_biais_droite_bien_reduit_par_decomposition(bank):
    # Après séparation biais / dérive, le biais à J-0 du bloc droite doit être
    # nettement plus petit que l'écart brut confondu (~+6 pts) : l'essentiel de
    # l'effondrement Pécresse part dans la dérive, pas dans le biais.
    assert bank.bloc_bias_mean.at(bloc="droite")[0] < 4.5


def test_derive_specifique_a_l_election(bank):
    # La dérive du bloc droite doit être de signe opposé entre 2017 (Fillon) et
    # 2022 (Pécresse) : c'est ce qui justifie une dérive par élection, et la
    # dérive 2022 doit être positive (surestimation loin du scrutin).
    d2017 = bank.campaign_drift.at(election=2017, bloc="droite")[0]
    d2022 = bank.campaign_drift.at(election=2022, bloc="droite")[0]
    assert d2022 > 0 > d2017


def test_deux_composantes_house_effect_exportees(bank):
    # Le house effect a bien deux formes distinctes : biais directionnel
    # (institut_bias) ET précision générale / dispersion (excess_institut_offset).
    assert hasattr(bank, "institut_bias")
    assert hasattr(bank, "excess_institut_offset")
