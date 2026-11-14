"""Second tour de spatial-pooling (spec_second_tour.md) : maths du duel,
mise en forme des duels, et lecture depuis un fit -- sans NUTS."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpyro import handlers

from model.models.spatial_pooling.fit import (SpatialPoolingFit, duel_draws, p_elu,
                                              projeter_w)
from model.models.spatial_pooling.geometry import (duel_shares, log_attractivite,
                                                   spatial_shares)
from model.models.spatial_pooling.joint_model import spatial_pooling_model_ou
from model.models.spatial_pooling.roster import build_duel_arrays, split_tours

MU = jnp.asarray([0.2, 0.5, 0.8])
SG = jnp.asarray([0.15, 0.12, 0.15])


def _duel(w, kappa=1.0, eta=(0.0, 0.0, 0.0), a=0, b=2):
    A = log_attractivite(MU, SG, jnp.asarray(w))
    eta = jnp.asarray(eta)
    return float(duel_shares(A[a], A[b], kappa, eta[a], eta[b]))


# --- maths du duel (spec §2.2) -------------------------------------------------

def test_duel_invariant_par_translation_de_w_et_eta():
    """Le défaut de l'option extérieure Codex : une translation de `w`
    changeait le sens de omega. Ici seules les différences comptent."""
    base = _duel([0.0, 0.3, 0.1], 0.7, (0.1, 0.0, -0.2))
    assert _duel([5.0, 5.3, 5.1], 0.7, (1.1, 1.0, 0.8)) == pytest.approx(base, abs=1e-6)


def test_duel_ne_depend_pas_d_un_candidat_elimine():
    """Le `w` du centre éliminé faisait bouger le duel gauche/droite (0,489 ->
    0,477) dans la version Codex. Il ne doit avoir AUCUN effet."""
    vals = [_duel([0.0, wc, 0.3]) for wc in (-3.0, 0.0, 3.0)]
    assert np.ptp(vals) < 1e-7


def test_duel_kappa_1_eta_0_est_le_premier_tour_restreint_aux_deux_noms():
    """Lien positions -> reports : sans paramètre T2, le duel EST le softmax
    spatial du premier tour sur le champ {a, b}."""
    w = jnp.asarray([0.2, 1.0, -0.1])
    t1 = np.asarray(spatial_shares(MU, SG, w, jnp.asarray([1.0, 0.0, 1.0])))
    assert _duel(w) == pytest.approx(float(t1[0]), abs=1e-6)


def test_duel_valence_favorise_son_candidat_et_symetrie():
    s0 = _duel([0.0, 0.0, 0.0])
    s1 = _duel([0.0, 0.0, 0.0], eta=(0.4, 0.0, 0.0))
    assert s1 > s0
    A = log_attractivite(MU, SG, jnp.zeros(3))
    ab = float(duel_shares(A[0], A[2], 0.8, 0.3, -0.1))
    ba = float(duel_shares(A[2], A[0], 0.8, -0.1, 0.3))
    assert ab + ba == pytest.approx(1.0, abs=1e-6)


# --- mise en forme -------------------------------------------------------------

def _t2(lignes):
    base = dict(date_fin=pd.Timestamp("2026-10-01"), echantillon=1000.0, institut="Ifop",
                tour="Deuxième tour")
    return pd.DataFrame([{**base, **l} for l in lignes])


def test_build_duel_arrays_une_observation_par_duel_sans_deflation():
    df = _t2([dict(notice="n1", hypothese="h1", candidat="Z", intention=45.0),
              dict(notice="n1", hypothese="h1", candidat="X", intention=55.0),
              dict(notice="n1", hypothese="h2", candidat="Y", intention=60.0),
              dict(notice="n1", hypothese="h2", candidat="Z", intention=40.0)])
    d = build_duel_arrays(df, ["X", "Y", "Z"])
    assert list(d["a"]) == [0, 1] and list(d["b"]) == [2, 2]     # ordre du roster
    assert d["y"] == pytest.approx([0.55, 0.60])
    assert d["n"] == pytest.approx([1000.0, 1000.0])            # pas de n/n_hyp (spec §4)


def test_build_duel_arrays_ignore_un_duel_incomplet_ou_hors_roster():
    df = _t2([dict(notice="n1", hypothese="h", candidat="X", intention=50.0),
              dict(notice="n1", hypothese="h", candidat="Inconnu", intention=50.0),
              dict(notice="n2", hypothese="h", candidat="X", intention=100.0)])
    assert len(build_duel_arrays(df, ["X", "Y"])["y"]) == 0


def test_split_tours_sans_colonne_tour_tout_est_premier_tour():
    df = pd.DataFrame({"candidat": ["X"]})
    t1, t2 = split_tours(df)
    assert len(t1) == 1 and len(t2) == 0


# --- vraisemblance --------------------------------------------------------------

def _kwargs(with_duels: bool):
    tm = jnp.asarray([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    kw = dict(tested_mask=tm, Y=jnp.asarray([[0.3, 0.3, 0.4], [0.45, 0.0, 0.55]]),
              Np=jnp.asarray([1000.0, 1000.0]), date_idx=jnp.asarray([0, 1]),
              kl_basis=jnp.asarray(np.eye(3)[:, :2].T), pos_anchor=jnp.asarray([0.25, 0.5, 0.75]),
              tau_ou=200.0)
    if with_duels:
        kw.update(duel_a=jnp.asarray([0]), duel_b=jnp.asarray([2]), duel_y=jnp.asarray([0.48]),
                  duel_n=jnp.asarray([900.0]), duel_date_idx=jnp.asarray([1]))
    return kw


def test_modele_sans_duel_n_a_aucun_site_t2():
    tr = handlers.trace(handlers.seed(spatial_pooling_model_ou, 0)).get_trace(**_kwargs(False))
    assert not {"kappa", "z_eta", "sigma_t2", "ll_t2"} & set(tr)


def test_modele_avec_duel_ajoute_un_seul_terme_par_duel():
    tr = handlers.trace(handlers.seed(spatial_pooling_model_ou, 0)).get_trace(**_kwargs(True))
    assert {"kappa", "z_eta", "sigma_t2", "ll_t2"} <= set(tr)
    assert np.shape(tr["ll_t2"]["fn"].log_factor) == ()
    assert np.isfinite(float(tr["ll_t2"]["fn"].log_factor))


# --- lecture depuis un fit ------------------------------------------------------

def _fit(S=200, seed=0) -> SpatialPoolingFit:
    rng = np.random.default_rng(seed)
    N = 3
    return SpatialPoolingFit(
        candidates=["G", "C", "D"], anchors=np.array([.25, .5, .75]),
        tested_mask=np.ones((1, N)),
        mu_draws=np.tile([.2, .5, .8], (S, 1)) + rng.normal(0, .01, (S, N)),
        sigma_draws=np.full((S, N), .15), w_draws=np.tile([0.0, 0.5, 0.2], (S, 1)),
        diagnostics={}, kappa_draws=np.full(S, 0.8), eta_draws=np.tile([0., .2, -.2], (S, 1)),
        level_draws=np.zeros((S, N)), sigma_w_draws=np.full(S, 0.3),
        dynamic=np.array([True, True, False]), tau_ou=200.0, as_of="2026-11-14",
        duel_pairs=[("C", "D")])


def test_projection_ou_sans_horizon_est_l_identite_et_statiques_fixes():
    f = _fit()
    assert np.array_equal(projeter_w(f, 0), f.w_draws)
    w = projeter_w(f, 150)
    assert np.array_equal(w[:, 2], f.w_draws[:, 2])
    assert w[:, 1].std() > 0.05


def test_duel_draws_incertitude_croit_avec_l_horizon_et_le_scrutin():
    f = _fit()
    a = duel_draws(f, "C", "D", 0, scrutin=False)
    b = duel_draws(f, "C", "D", 0, scrutin=True)
    c = duel_draws(f, "C", "D", 150, scrutin=True)
    assert a.std() < b.std() < c.std()
    assert ((0 < c) & (c < 1)).all()


def test_p_elu_somme_a_un_sur_le_champ():
    out = p_elu(_fit(), np.ones(3), horizon_t1=100, horizon_t2=114)
    assert sum(out["p_elu"].values()) == pytest.approx(1.0, abs=1e-6)
    assert sum(d["p"] for d in out["duels_qualifies"]) <= 1.0 + 1e-9


def test_valence_agregee_a_un_effet_independant_des_rayons():
    """eta s'applique à l'agrégat : son effet en points ne dépend pas de sigma
    (au niveau du votant il variait ×7 entre sigma 0,08 et 0,30)."""
    effets = []
    for s in (0.08, 0.15, 0.30):
        A = log_attractivite(jnp.asarray([0.3, 0.7]), jnp.asarray([s, s]), jnp.zeros(2))
        effets.append(float(duel_shares(A[0], A[1], 1.0, 0.5, 0.0))
                      - float(duel_shares(A[0], A[1], 1.0, 0.0, 0.0)))
    assert np.ptp(effets) < 1e-4 and effets[0] == pytest.approx(0.1225, abs=1e-3)


def _duel_abst(w, omega=-3.0, kappa=0.8, a=0, b=2):
    import jax
    A = log_attractivite(MU, SG, jnp.asarray(w))
    lse = jax.nn.logsumexp(A, axis=0)
    return float(duel_shares(A[a], A[b], kappa, 0.0, 0.0, lse, omega))


def test_abstention_invariante_par_translation_de_w():
    """Le défaut central de l'option extérieure Codex : ici omega garde un sens."""
    assert _duel_abst([5.0, 5.3, 5.1]) == pytest.approx(_duel_abst([0.0, 0.3, 0.1]), abs=1e-6)


def test_abstention_retire_l_electorat_du_centre_elimine():
    """Duel gauche/droite, centre éliminé : plus le centre est attractif, plus
    son électorat s'abstient -- l'effet dépend de l'électorat, pas d'un candidat.
    Sans abstention le centre éliminé n'a AUCUN effet ; avec, il en a un."""
    from model.models.spatial_pooling.geometry import duel_expression
    import jax
    for wc in (-3.0, 3.0):
        A = log_attractivite(MU, SG, jnp.asarray([0.0, wc, 0.0]))
        rho = duel_expression(A[0], A[2], 0.8, jax.nn.logsumexp(A, axis=0), -3.0)
        assert ((0 < rho) & (rho < 1)).all()
    A_lo = log_attractivite(MU, SG, jnp.asarray([0.0, -3.0, 0.0]))
    A_hi = log_attractivite(MU, SG, jnp.asarray([0.0, 3.0, 0.0]))
    e_lo = float(jnp.sum(duel_expression(A_lo[0], A_lo[2], 0.8, jax.nn.logsumexp(A_lo, 0), -3.0) * __import__("model.models.spatial_pooling.geometry", fromlist=["W"]).W))
    e_hi = float(jnp.sum(duel_expression(A_hi[0], A_hi[2], 0.8, jax.nn.logsumexp(A_hi, 0), -3.0) * __import__("model.models.spatial_pooling.geometry", fromlist=["W"]).W))
    assert e_hi < e_lo
