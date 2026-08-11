"""Tests de la Bank du saut terminal
(model/models/bayesian_nowcast/bank_jump.json, TerminalJumpCalibration).

Ne relance pas l'inférence (coûteuse) ; vérifie la structure produite par
`TerminalJumpCalibration.calibrate` — cf. docs/spec_ssm_nowcast.md §2.2/§7.
Ignorés si la Bank n'a pas encore été calibrée.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.core.bank import Bank
from model.models.bayesian_nowcast import JUMP_BANK_PATH

pytestmark = pytest.mark.skipif(
    not JUMP_BANK_PATH.exists(),
    reason="Bank (saut terminal) absente — lancer d'abord "
          ".venv/bin/python -m model.models.bayesian_nowcast",
)


@pytest.fixture(scope="module")
def jump_bank() -> Bank:
    return Bank.load(JUMP_BANK_PATH)


def test_structure_bank_jump(jump_bank):
    for site in ("jump_skew", "jump_tail", "jump_loc", "jump_scale", "jump_horizon_ref"):
        assert hasattr(jump_bank, site), f"site manquant dans la Bank (saut terminal) : {site}"


def test_horizon_ref_positif(jump_bank):
    href, _ = jump_bank.jump_horizon_ref.item()
    assert href > 0.0


def test_skew_tail_sont_scalaires_globaux(jump_bank):
    # skew/tail GLOBAUX (spec §7 : pas assez d'élections historiques pour un
    # ajustement par bloc de ces paramètres de forme) — pas de dimension `bloc`.
    assert jump_bank.jump_skew.mean.dims == ()
    assert jump_bank.jump_tail.mean.dims == ()


def test_tail_positif(jump_bank):
    tail, _ = jump_bank.jump_tail.item()
    assert tail > 0.0


def test_scale_positif_par_bloc(jump_bank):
    for bloc in jump_bank.jump_scale.mean.coords["bloc"].values.tolist():
        scale, _ = jump_bank.jump_scale.at(bloc=bloc)
        assert scale > 0.0


def test_loc_scale_ont_la_meme_dimension_bloc(jump_bank):
    locs = set(jump_bank.jump_loc.mean.coords["bloc"].values.tolist())
    scales = set(jump_bank.jump_scale.mean.coords["bloc"].values.tolist())
    assert locs == scales
