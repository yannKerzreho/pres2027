"""§8.42 : a posteriori de τ_T2 sur grille."""
import numpy as np

from model.backtest.tau_t2_historique import GRILLE, posterieur, resume


def test_recupere_tau_simule():
    rng = np.random.default_rng(0)
    obs = [{"d_logit": float(x), "v": 1e-4} for x in rng.normal(0, 0.2, 200)]
    r = resume(posterieur(obs, "normale"))
    assert 0.17 < r["mediane"] < 0.23
    assert r["ic90"][0] < r["mediane"] < r["ic90"][1]


def test_sans_erreur_tau_petit_et_student_plus_large_en_queue():
    w = posterieur([{"d_logit": 0.0, "v": 1e-4}] * 5, "normale")
    assert abs(w.sum() - 1) < 1e-9 and resume(w)["mediane"] < 0.15
    obs = [{"d_logit": d, "v": 1e-4} for d in (0.0, 0.01, -0.02, 0.6)]
    assert resume(posterieur(obs, "student4"))["mediane"] < resume(posterieur(obs, "normale"))["mediane"]
    assert len(GRILLE) == 1501
