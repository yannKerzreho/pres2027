from __future__ import annotations

from model.backtest.second_round import validation_report


def _row(rhat: float, covered: bool) -> dict:
    return {"option_exterieure": "omega=0", "rhat_max": rhat,
            "couvre_ic90": covered}


def test_validation_refuse_un_rhat_non_fini_ou_une_couverture_insuffisante():
    report = validation_report([_row(float("nan"), False), _row(1.005, False)])
    assert report["decision"] == "reject"
    assert report["summary"]["rhat_max"] is None


def test_validation_accepte_seulement_des_chainres_convergees_et_couvertes():
    report = validation_report([_row(1.005, True), _row(1.009, True)])
    assert report["decision"] == "accept"
