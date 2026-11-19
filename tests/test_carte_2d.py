"""Carte 2D et reports en vraisemblance (spec_second_tour §8.44, piste R2)."""
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from model.models.spatial_pooling.geometry import (W, W2, duel_shares, duel_shares_poids, log_attractivite,
                                                   log_attractivite_2d, reports_cellules, spatial_shares,
                                                   spatial_shares_2d)


def _geo(seed=0, N=5):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(np.sort(rng.uniform(.1, .9, N))), jnp.full(N, .12),
            jnp.asarray(rng.normal(0, .5, N)), jnp.asarray(rng.uniform(.2, .8, N)))


def test_lambda_grand_redonne_le_1d():
    mu, sg, w, nu = _geo()
    m = jnp.asarray([1, 1, 0, 1, 1.0])
    assert float(jnp.abs(spatial_shares(mu, sg, w, m) - spatial_shares_2d(mu, sg, nu, 1e3, w, m)).max()) < 1e-6
    A1, A2 = log_attractivite(mu, sg, w), log_attractivite_2d(mu, sg, nu, 1e3, w)
    l1, l2 = jnp.asarray(logsumexp(A1, 0)), jnp.asarray(logsumexp(A2, 0))
    s1 = duel_shares(A1[0], A1[4], 0.8, 0.1, -0.2, l1, -3.0)
    s2 = duel_shares_poids(A2[0], A2[4], 0.8, 0.1, -0.2, l2, -3.0, W2)
    assert abs(float(s1) - float(s2)) < 1e-5
    assert abs(float(duel_shares_poids(A1[0], A1[4], 0.8, 0.1, -0.2, l1, -3.0, W)) - float(s1)) < 1e-7


def test_newton_egale_bissection():
    from model.backtest.reports_t2 import composantes, melange
    mu, sg, w, _ = _geo(seed=1)
    A = np.asarray(log_attractivite(mu, sg, w))[None]
    lse = logsumexp(A, axis=1)
    champ = np.arange(5)
    E, R, G = composantes(A, champ, 0, 4, np.array([0.9]), np.array([-2.5]), lse)
    s = np.array([0.57])
    ref = melange(E, R, G, s)
    for k in range(5):
        th, d = reports_cellules(jnp.asarray(A), jnp.ones(5), jnp.array([0]), jnp.array([4]), jnp.array([k]),
                                 0.9, jnp.array([0.3]), jnp.asarray(lse), -2.5, jnp.asarray(s), W, n_newton=6)
        assert abs(float(d[0]) - float(ref["delta"][0])) < 1e-4
        assert np.allclose(np.asarray(th[0]), [ref["a"][0, k], ref["b"][0, k], ref["aucun"][0, k]], atol=1e-4)


def test_arrays_reports_2022():
    from model.models.spatial_pooling.fit import arrays_reports
    rep = pd.read_csv("data/parsed/reports_t2_2022.csv")
    cands = ["Emmanuel Macron", "Marine Le Pen", "Jean-Luc Mélenchon", "Eric Zemmour", "Valérie Pécresse",
             "Yannick Jadot", "Fabien Roussel"]
    r = arrays_reports(rep, cands, "2022-04-17", cands)
    assert r is not None and len(r["y"]) > 0
    assert np.allclose(r["y"].sum(1), 1.0)
    assert set(r["a"]) == {0} and set(r["b"]) == {1}
    assert set(r["k"]) <= {2, 3, 4, 5}
    assert (r["dates"] <= pd.Timestamp("2022-04-17").toordinal()).all()


def test_ancres_u():
    from model.models.spatial_pooling.fit import ancres_u
    c = ["Jean-Luc Mélenchon", "Emmanuel Macron", "Marine Le Pen"]
    assert ancres_u(c, {"a": np.array([1]), "b": np.array([2])}) == {"nu_ancre": 1, "nu_ordre": 2}


def test_modele_2d_densite_finie():
    """Trace du modèle joint avec carte 2D et reports : log-densité finie, ν_RN ≥ ν_C."""
    import numpyro
    from numpyro.infer.util import log_density
    from model.models.spatial_pooling.joint_model import spatial_pooling_model_ou
    rng = np.random.default_rng(0)
    N, P, T = 4, 6, 3
    tm = np.ones((P, N)); Y = rng.dirichlet(np.ones(N), P)
    kl = np.eye(2, T)
    kw = dict(tested_mask=jnp.asarray(tm), Y=jnp.asarray(Y), Np=jnp.full(P, 1000.0),
              date_idx=jnp.asarray(rng.integers(0, T, P)), kl_basis=jnp.asarray(kl),
              pos_anchor=jnp.asarray([.2, .4, .6, .8]), tau_ou=200.0,
              duel_a=jnp.array([1]), duel_b=jnp.array([3]), duel_y=jnp.array([.55]), duel_n=jnp.array([1000.0]),
              duel_date_idx=jnp.array([2]), carte_2d=True, nu_ancre=1, nu_ordre=3,
              rep_y=jnp.asarray([[.5, .2, .3]]), rep_date_idx=jnp.array([2]), rep_k=jnp.array([0]),
              rep_a=jnp.array([1]), rep_b=jnp.array([3]), rep_champ=jnp.ones(N))
    with numpyro.handlers.seed(rng_seed=0):
        tr = numpyro.handlers.trace(spatial_pooling_model_ou).get_trace(**kw)
    params = {k: v["value"] for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")}
    ld, _ = log_density(spatial_pooling_model_ou, (), kw, params)
    assert np.isfinite(float(ld))
    nu = np.asarray(tr["nu"]["value"])
    assert abs(nu[1] - 0.3) < 1e-6 and nu[3] >= nu[1]
    assert "ll_reports" in tr


def test_newton_cas_sature():
    """Revue 19/11, point 8 : σ ≈ 0,02, κ = 0,75 (logits saturés) — 4 pas suffisent."""
    from model.backtest.reports_t2 import composantes, melange
    rng = np.random.default_rng(7)
    mu = jnp.asarray(np.sort(rng.uniform(.1, .9, 6)))
    sg = jnp.full(6, 0.02)
    w = jnp.asarray(rng.normal(0, .5, 6))
    A = np.asarray(log_attractivite(mu, sg, w))[None]
    lse = logsumexp(A, axis=1)
    E, R, G = composantes(A, np.arange(6), 1, 4, np.array([0.75]), np.array([-3.0]), lse)
    s = np.array([0.55])
    ref = melange(E, R, G, s)
    th, d = reports_cellules(jnp.asarray(A), jnp.ones(6), jnp.array([1]), jnp.array([4]), jnp.array([2]),
                             0.75, jnp.array([0.0]), jnp.asarray(lse), -3.0, jnp.asarray(s), W, n_newton=4)
    assert np.allclose(np.asarray(th[0]), [ref["a"][0, 2], ref["b"][0, 2], ref["aucun"][0, 2]], atol=2e-3)
