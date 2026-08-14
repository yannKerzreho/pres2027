"""Tests de la Bank du saut terminal (`model/core/terminal_jump.py`).

Ne relance pas l'inférence (coûteuse) ; vérifie la structure produite par
`TerminalJumpCalibration.calibrate` telle que la consomme
`model/core/simulate.py` (`_sinh_arcsinh_moves`) — si un de ces sites ou une
de ces dimensions disparaît, la prévision casse silencieusement.

La Bank testée est celle de `bayesian-nowcast`, seul modèle de cette branche
à utiliser encore le saut terminal paramétrique. Ignorés si elle n'a pas
encore été calibrée.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.core.bank import Bank

from model.models.bayesian_nowcast import JUMP_BANK_PATH as BANK_JUMP_PATH

pytestmark = pytest.mark.skipif(
    not BANK_JUMP_PATH.exists(),
    reason="Bank (saut terminal) absente — lancer d'abord "
           ".venv/bin/python -m model.models.bayesian_nowcast",
)


@pytest.fixture(scope="module")
def jump_bank() -> Bank:
    return Bank.load(BANK_JUMP_PATH)


def test_structure_bank_jump(jump_bank):
    for site in ("jump_skew", "jump_tail", "jump_loc", "jump_scale", "jump_tau"):
        assert hasattr(jump_bank, site), f"site manquant dans la Bank (saut terminal) : {site}"


def test_tau_positif(jump_bank):
    tau, _ = jump_bank.jump_tau.item()
    assert tau > 0.0


def test_saturation_croissante_et_bornee(jump_bank):
    # `sat(h) = 1 − e^(−h/τ)` : nulle à J-0, croissante, jamais > 1. C'est ce
    # qui garantit que la variance d'un segment `jump_moves` reste positive et
    # que deux jambes enchaînées s'additionnent exactement.
    import numpy as np
    from model.core.terminal_jump import saturation
    tau, _ = jump_bank.jump_tau.item()
    h = np.array([0.0, 30.0, 100.0, 300.0, 1000.0])
    s = saturation(h, tau)
    assert s[0] == 0.0
    assert np.all(np.diff(s) > 0)
    assert np.all(s <= 1.0)


def test_variance_des_deux_jambes_est_exactement_additive(jump_bank):
    """L'invariant qui autorise à appliquer le saut DEUX fois (sondage →
    `as_of`, puis `as_of` → scrutin) sans double comptage ni trou.

    Var(sondage→as_of) + Var(as_of→scrutin) doit valoir Var(sondage→scrutin).
    Si cette égalité casse, les intervalles publiés sont soit trop larges
    (double comptage) soit trop étroits (segment perdu) — sans que rien ne le
    signale ailleurs.
    """
    import numpy as np
    from model.core.terminal_jump import saturation
    tau, _ = jump_bank.jump_tau.item()
    h_sondage, h_as_of = 290.0, 250.0     # sondage vieux de 40 j, scrutin à 250 j
    jambe_courte = saturation(h_sondage, tau) - saturation(h_as_of, tau)
    jambe_longue = saturation(h_as_of, tau) - saturation(0.0, tau)
    total = saturation(h_sondage, tau) - saturation(0.0, tau)
    assert jambe_courte > 0.0
    assert np.isclose(jambe_courte + jambe_longue, total, rtol=1e-12)


def test_loc_est_global(jump_bank):
    # `loc` par bloc rejouerait 2017/2022 (effondrement LR) au lieu de
    # diffuser : il doit rester identique pour tous les blocs, et proche de 0
    # puisque les mouvements CLR somment à zéro par construction.
    blocs = jump_bank.jump_loc.mean.coords["bloc"].values.tolist()
    locs = [jump_bank.jump_loc.at(bloc=b)[0] for b in blocs]
    assert max(locs) - min(locs) < 1e-9
    assert abs(locs[0]) < 1.0


def test_skew_tail_sont_scalaires_globaux(jump_bank):
    # skew/tail GLOBAUX (seulement 2 élections historiques : pas de quoi
    # ajuster des paramètres de forme par bloc) — pas de dimension `bloc`.
    assert jump_bank.jump_skew.mean.dims == ()
    assert jump_bank.jump_tail.mean.dims == ()


def test_tail_positif(jump_bank):
    tail, _ = jump_bank.jump_tail.item()
    assert tail > 0.0


def test_scale_positif_par_bloc(jump_bank):
    for bloc in jump_bank.jump_scale.mean.coords["bloc"].values.tolist():
        scale, _ = jump_bank.jump_scale.at(bloc=bloc)
        assert scale > 0.0


def test_scale_identique_pour_tous_les_blocs(jump_bank):
    # `scale` est global et broadcasté sur la dimension bloc (cf.
    # terminal_jump_model) : garder la dimension pour le schéma Bank, mais une
    # divergence entre blocs signalerait un retour au pooling hiérarchique
    # abandonné (inversions de p_qualifie_top2 non justifiées à n=2 élections).
    blocs = jump_bank.jump_scale.mean.coords["bloc"].values.tolist()
    scales = [jump_bank.jump_scale.at(bloc=b)[0] for b in blocs]
    assert max(scales) - min(scales) < 1e-9


def test_loc_scale_ont_la_meme_dimension_bloc(jump_bank):
    locs = set(jump_bank.jump_loc.mean.coords["bloc"].values.tolist())
    scales = set(jump_bank.jump_scale.mean.coords["bloc"].values.tolist())
    assert locs == scales


def test_blocs_couvrent_ceux_des_slots_du_modele(jump_bank):
    # `forecast_from_draws` fait un lookup `.at(bloc=...)` pour CHAQUE slot :
    # un bloc présent dans SLOTS mais absent de la Bank ferait planter la
    # prévision en production, pas en test unitaire du modèle.
    from model.core.live_dataset import SLOTS
    connus = set(jump_bank.jump_loc.mean.coords["bloc"].values.tolist())
    attendus = {bloc for bloc, _names in SLOTS.values()}
    assert attendus <= connus, f"blocs sans saut terminal calibré : {attendus - connus}"
