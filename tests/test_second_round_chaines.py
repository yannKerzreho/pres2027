"""Découpage d'un backtest T2 en processus d'une chaîne puis fusion
(`model.backtest.second_round --chaine k / --fusionner`)."""
import numpy as np
import numpyro
import numpyro.distributions as dist

from model.backtest.second_round import _runner_chaine, _runner_fusion
from model.core.inference import run_numpyro_mcmc


def _jouet():
    numpyro.sample("x", dist.Normal(0.0, 1.0).expand([3]))


def test_chaines_separees_puis_fusion_equivalent_a_un_run(tmp_path):
    kw = dict(draws=30, tune=30, chains=2, seed=5, extra_fields=("diverging",))
    paths = [tmp_path / f"t_c{k}.npz" for k in range(2)]
    par_chaine = [_runner_chaine(p, k)(_jouet, {}, **kw)[0]["x"] for k, p in enumerate(paths)]
    samples, extra = _runner_fusion(paths)(_jouet, {}, **kw)
    assert samples["x"].shape == (2, 30, 3)
    assert extra["diverging"].shape == (60,)
    np.testing.assert_allclose(samples["x"][1], np.asarray(par_chaine[1])[0])
    # Graines distinctes : les deux chaînes ne sont pas des copies.
    assert not np.allclose(samples["x"][0], samples["x"][1])
    # Une chaîne seule = le run standard à 1 chaîne, graine seed + k.
    ref, _ = run_numpyro_mcmc(_jouet, {}, **{**kw, "chains": 1, "seed": 6})
    np.testing.assert_allclose(samples["x"][1], np.asarray(ref["x"])[0])


def test_fusion_refuse_une_chaine_manquante(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        _runner_fusion([tmp_path / "absent.npz"])


def test_cadre_historique_garde_toutes_les_hypotheses():
    """Spec §8.3 : la variation des champs identifie les positions. L'ancien
    cadre (une hypothèse par notice) ne laissait que 4 champs à 2022 T1−30."""
    import pandas as pd
    import model.backtest.second_round as sr
    from model.models.spatial_pooling.roster import (ORDER_GROUPS_HISTORIQUES, build_poll_arrays,
                                                     build_roster, split_tours)
    as_of = sr.T1_DATES[2022] - pd.Timedelta(days=30)
    champs = {}
    for toutes in (True, False):
        t1, _ = split_tours(sr.polls_historiques(2022, as_of, toutes_hypotheses=toutes))
        cands, _, _ = build_roster(t1, order_groups=ORDER_GROUPS_HISTORIQUES[2022])
        a = build_poll_arrays(t1, cands)
        champs[toutes] = len({tuple(np.flatnonzero(r)) for r in a["tested_mask"]})
        if toutes:
            assert {"Bertrand", "Taubira", "Montebourg"} <= set(cands)
    assert champs[True] >= 50 and champs[False] <= 5


def test_cible_par_chaine_detecte_une_chaine_dans_un_autre_mode(monkeypatch):
    import model.backtest.second_round as sr
    monkeypatch.setattr(sr, "CHAINES_RETENUES", [])
    monkeypatch.setattr(sr, "LP_CHAINES", [])
    rng = np.random.default_rng(0)
    d = 1 / (1 + np.exp(-rng.normal(0.3, 0.1, 1600)))
    ok = sr._cible_par_chaine(d, 4)
    assert ok["rhat_cible"] < 1.05 and ok["ecart_chaines_pt"] < 1 and ok["concluant"]
    d[:400] = 1 / (1 + np.exp(-rng.normal(0.6, 0.1, 400)))
    ko = sr._cible_par_chaine(d, 4)
    assert ko["rhat_cible"] > 1.1 and ko["ecart_chaines_pt"] > 5 and not ko["concluant"]
    assert not sr._cible_par_chaine(d[:400], 1)["concluant"]      # une chaîne : non concluant


def test_chaines_retenues_tient_compte_de_l_erreur_type():
    from model.backtest.second_round import _chaines_retenues
    # écart 10,8 nats mais SE 3 nats par chaîne : pas de rejet (cas Pécresse, revue 2026-11-15)
    assert _chaines_retenues([10547.4, 10549.0, 10552.4, 10541.6], [3, 3, 3, 3]).tolist() == [0, 1, 2, 3]
    # SE faibles : le piège à 100 nats est rejeté, les voisins à < 10 nats gardés
    assert _chaines_retenues([100.0, 0.0, 95.0, 101.0], [0.5] * 4).tolist() == [0, 2, 3]


def test_log_densite_par_chaine_classe_les_chaines():
    import numpyro
    import numpyro.distributions as dist
    from model.backtest.second_round import _log_densite_par_chaine

    def m():
        numpyro.sample("x", dist.Normal(0.0, 1.0))
    lp, se = _log_densite_par_chaine(m, {}, {"x": np.array([[0.0] * 20, [3.0] * 20])})
    assert lp[0] - lp[1] > 4 and se[0] == 0.0
