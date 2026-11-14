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
