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


def test_reports_groupes_par_hypothese():
    """§8.50 : une notice qui teste deux duels donne deux lignes distinctes, non mélangées."""
    import pandas as pd
    from model.models.spatial_pooling.fit import arrays_reports
    cands = ["Jean-Luc Mélenchon", "Emmanuel Macron", "Marine Le Pen", "Xavier Bertrand"]
    lignes = []
    for hyp, b, vals in (("Macron – Le Pen", "Le Pen", (60.0, 15.0, 25.0)),
                         ("Macron – Bertrand", "Bertrand", (20.0, 30.0, 50.0))):
        for c, v in zip(("Macron", b, "Aucun"), vals):
            lignes.append({"notice": "N1", "date_fin": "2022-03-01", "hypothese": hyp,
                           "electorat_t1": "Jean-Luc Mélenchon", "candidat_t2": c, "part": v})
    rep = arrays_reports(pd.DataFrame(lignes), cands, "2022-03-10",
                         ["Jean-Luc Mélenchon", "Emmanuel Macron", "Marine Le Pen"])
    assert rep["y"].shape == (2, 3)
    i_lp = [r[3] for r in rep["rows"]].index("Macron – Le Pen")
    assert np.allclose(rep["y"][i_lp], [0.60, 0.15, 0.25])
    # champ par ligne : Bertrand seulement dans sa propre ligne
    assert rep["champ"][i_lp][3] == 0.0 and rep["champ"][1 - i_lp][3] == 1.0


def test_fidelite_revelee():
    """§8.56 : électorat d'un finaliste → (σ(−ω), ε, σ(ω)) ; autres lignes inchangées."""
    from model.models.spatial_pooling.geometry import W, reports_cellules
    rng = np.random.default_rng(1)
    R, N, G = 3, 4, W.shape[0]
    A = jnp.asarray(rng.normal(size=(R, N, G)))
    lse = jax.nn.logsumexp(A, axis=1)
    ia, ib, k = jnp.array([0, 0, 0]), jnp.array([1, 1, 1]), jnp.array([0, 1, 2])
    args = (A, jnp.ones(N), ia, ib, k, 0.8, jnp.zeros(R), lse, -3.0, jnp.full(R, 0.5), W)
    c0, _ = reports_cellules(*args)
    c1, _ = reports_cellules(*args, fidelite=True)
    u = (np.arange(32) + 0.5) / 32
    g = -np.log(-np.log(u))
    rho = float((1 / (1 + np.exp(-(0.8 * g + 3.0)))).mean())
    assert np.allclose(c1[0], [rho * 0.98, rho * 0.02, 1 - rho], atol=1e-6)
    assert np.allclose(c1[1], [rho * 0.02, rho * 0.98, 1 - rho], atol=1e-6)
    assert np.allclose(c1.sum(-1), 1.0, atol=1e-6)
    assert np.allclose(c1[2], c0[2])
