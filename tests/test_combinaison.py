import numpy as np

from model.backtest.combinaison import _quantiles_melange, scores


def test_quantiles_melange_degenere_en_normale():
    q = _quantiles_melange(1.0, np.array([0.3]), np.array([0.02]), np.array([0.5]), np.array([0.1]), 0.95)
    assert abs(q[0] - (0.3 + 1.6449 * 0.02)) < 1e-4


def test_score_intervalle_penalise_hors_ic():
    s = scores(np.array([0.30, 0.40]), np.array([0.3, 0.3]), np.array([0.28, 0.28]), np.array([0.32, 0.32]))
    assert np.allclose(s["is"], [4.0, 4.0 + 20 * 8.0])
    assert s["cov"].tolist() == [1.0, 0.0]


def test_prospectif_scorer_sans_duel_nouveau(tmp_path, monkeypatch):
    import json
    import pandas as pd
    from model.backtest import prospectif_2027 as pr
    monkeypatch.setattr(pr, "DOSSIER", tmp_path)
    (tmp_path / "prediction_hier_2099-01-01.json").write_text(json.dumps(
        {"variante": "hier", "as_of": "2099-01-01", "sigma_t2": 0.01, "paires_sondees": [], "paires": {}}))
    monkeypatch.setattr(pr, "load_raw_polls", lambda tour=None: pd.DataFrame())
    monkeypatch.setattr(pr.v, "_duels", lambda raw: pd.DataFrame(columns=["a", "b", "y", "date", "n_ech"]))
    res = pr.scorer()
    assert res["prediction_hier_2099-01-01"]["n_duels_nouveaux"] == 0
