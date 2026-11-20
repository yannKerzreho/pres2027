"""Abstention située sur le cercle (spec_second_tour §8.48 bis)."""
import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.infer import init_to_median
from numpyro.infer.util import initialize_model

from model.models.spatial_pooling.geometry import THETA, duel_expression


def test_gamma_nul_emboite_le_cercle():
    rng = np.random.default_rng(0)
    A_a, A_b, lse = (jnp.asarray(rng.normal(size=(3, THETA.shape[0]))) for _ in range(3))
    r0 = duel_expression(A_a, A_b, 0.7, lse, -2.5)
    r1 = duel_expression(A_a, A_b, 0.7, lse, -2.5, omega_g=0.0 * jnp.cos(THETA))
    assert np.allclose(r0, r1)


def test_abstention_plus_forte_a_l_arriere():
    A = jnp.zeros((1, THETA.shape[0]))
    r = np.asarray(duel_expression(A, A, 1.0, A, -1.0, omega_g=-2.0 * jnp.cos(THETA)))[0]
    centre, arriere = int(np.argmin(np.abs(np.asarray(THETA)))), int(np.argmax(np.abs(np.asarray(THETA))))
    assert r[centre] > r[arriere]          # on s'exprime plus au centre qu'à l'arrière


def test_modele_initialise_avec_gamma():
    import model.backtest.multi_election as M
    M.VARIANTES["hiercercle_ra_abs"] = {"eta_hier": True, "carte_cercle": True, "_reports_avant": True,
                                        "abst_situee": True}
    b = M.preparer(2022, "T1-30", "hiercercle_ra_abs")
    from model.models.spatial_pooling.joint_model import spatial_pooling_model_ou
    assert b["kwargs"].get("abst_situee") is True
    info = initialize_model(jax.random.PRNGKey(0), spatial_pooling_model_ou, model_kwargs=b["kwargs"],
                            init_strategy=init_to_median)
    assert np.isfinite(float(info.potential_fn(info.param_info.z)))
    tr = handlers.trace(handlers.seed(spatial_pooling_model_ou, 0)).get_trace(**b["kwargs"])
    assert "gamma_abs" in tr
