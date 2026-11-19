"""Modèle joint multi-élections (spec_second_tour §8.47) : construction et partage."""
from functools import partial

import jax
import numpy as np
from numpyro import handlers
from numpyro.infer import init_to_median
from numpyro.infer.util import initialize_model

import model.backtest.multi_election as M


def test_personnes_recurrentes():
    blocs = [{"candidats": ["Jean-Luc Mélenchon", "Marine Le Pen", "François Fillon"]},
             {"candidats": ["Jean-Luc Mélenchon", "Marine Le Pen", "Valérie Pécresse"]}]
    pers = M.personnes(blocs)
    assert pers == ["melenchon", "pen"]
    M.indices_personnes(blocs, pers)
    assert blocs[0]["pidx"].tolist() == [0, 1, -1]
    assert blocs[1]["pidx"].tolist() == [0, 1, -1]


def test_modele_joint_partage_et_densite_finie():
    blocs, pers, cercle = M.construire("A_cercle")
    assert {"melenchon", "pen", "macron"} <= set(pers)
    assert "rep_y" in blocs[0]["kwargs"] and "rep_y" not in blocs[1]["kwargs"]   # reports 2017 seulement
    fn = partial(M.modele_multi, blocs=blocs, pers=pers, cercle=cercle)
    info = initialize_model(jax.random.PRNGKey(0), fn, model_kwargs={}, init_strategy=init_to_median)
    assert np.isfinite(float(info.potential_fn(info.param_info.z)))
    # s_mu → 0 : une personne récurrente a le MÊME écart d'ancre dans les deux élections.
    tr = handlers.trace(handlers.substitute(handlers.seed(fn, 1), {"s_mu": 1e-9, "s_sigma": 1e-9,
                                                                   "s_kappa": 1e-9})).get_trace()
    z17, z22 = np.asarray(tr["e2017__z_pos"]["value"]), np.asarray(tr["e2022__z_pos"]["value"])
    i17 = blocs[0]["candidats"].index("Jean-Luc Mélenchon")
    i22 = blocs[1]["candidats"].index("Jean-Luc Mélenchon")
    assert abs(z17[i17] - z22[i22]) < 1e-6
    assert abs(float(tr["e2017__kappa"]["value"]) - float(tr["e2022__kappa"]["value"])) < 1e-6
    assert float(tr["e2017__arc"]["value"]) == float(tr["e2022__arc"]["value"])
