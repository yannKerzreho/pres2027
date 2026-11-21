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


def test_omega_par_electorat_emboite():
    """§8.58 (R8) : ω_k tous égaux à ω ⇒ mêmes duels et mêmes cellules que sans R8."""
    from model.models.spatial_pooling.geometry import W, duel_shares_poids, reports_cellules, rho_electorats
    rng = np.random.default_rng(2)
    R, N, G = 3, 4, W.shape[0]
    A = jnp.asarray(rng.normal(size=(R, N, G)))
    lse = jax.nn.logsumexp(A, axis=1)
    ia, ib, k = jnp.array([0, 0, 0]), jnp.array([1, 1, 1]), jnp.array([2, 3, 0])
    Aa, Ab = A[jnp.arange(R), ia], A[jnp.arange(R), ib]
    om = -2.5
    s0 = duel_shares_poids(Aa, Ab, 0.8, 0.0, 0.0, lse, om, W)
    rn, pn = rho_electorats(A, Aa, Ab, 0.8, lse, jnp.full(N, om))
    s1 = duel_shares_poids(Aa, Ab, 0.8, 0.0, 0.0, lse, om, W, rho=(pn * rn).sum(-2))
    assert np.allclose(s0, s1, atol=1e-6)
    args = (A, jnp.ones(N), ia, ib, k, 0.8, jnp.zeros(R), lse, om, s0, W)
    c0, _ = reports_cellules(*args, fidelite=True)
    c1, _ = reports_cellules(*args, fidelite=True, omega_k=jnp.full(N, om))
    assert np.allclose(c0, c1, atol=1e-6)
    # un ω plus haut pour l'électorat 2 ⇒ plus d'abstention dans sa cellule
    c2, _ = reports_cellules(*args, omega_k=jnp.array([om, om, om + 1.5, om]))
    assert float(c2[0, 2]) > float(c0[0, 2])


def test_modele_initialise_avec_omega_k():
    import model.backtest.multi_election as M
    M.VARIANTES["hier_ra_fid_omk"] = {"eta_hier": True, "_reports_avant": True, "fidelite": True,
                                      "omega_electorat": True}
    b = M.preparer(2022, "T1-30", "hier_ra_fid_omk")
    from model.models.spatial_pooling.joint_model import spatial_pooling_model_ou
    info = initialize_model(jax.random.PRNGKey(0), spatial_pooling_model_ou, model_kwargs=b["kwargs"],
                            init_strategy=init_to_median)
    assert np.isfinite(float(info.potential_fn(info.param_info.z)))
    tr = handlers.trace(handlers.seed(spatial_pooling_model_ou, 0)).get_trace(**b["kwargs"])
    assert tr["omega_k"]["value"].shape == (len(b["candidats"]),)


def test_melange_lambda_emboite():
    """§8.61 (R9) : λ = 0 ⇒ R8c inchangé ; λ = 1 ⇒ q ≡ ½ hors lignes propres."""
    from model.models.spatial_pooling.geometry import W, duel_shares_poids, reports_cellules
    rng = np.random.default_rng(3)
    R, N, G = 3, 4, W.shape[0]
    A = jnp.asarray(rng.normal(size=(R, N, G)))
    lse = jax.nn.logsumexp(A, axis=1)
    ia, ib, k = jnp.array([0, 0, 0]), jnp.array([1, 1, 1]), jnp.array([2, 3, 0])
    Aa, Ab = A[jnp.arange(R), ia], A[jnp.arange(R), ib]
    s0 = duel_shares_poids(Aa, Ab, 0.8, 0.0, 0.0, lse, -2.5, W)
    assert np.allclose(s0, duel_shares_poids(Aa, Ab, 0.8, 0.0, 0.0, lse, -2.5, W, lam=0.0), atol=1e-7)
    assert np.allclose(duel_shares_poids(Aa, Ab, 0.8, 0.0, 0.0, lse, -2.5, W, lam=1.0), 0.5, atol=1e-6)
    args = (A, jnp.ones(N), ia, ib, k, 0.8, jnp.zeros(R), lse, -2.5, s0, W)
    c0, _ = reports_cellules(*args, fidelite=True, omega_k=jnp.full(N, -2.5))
    c1, _ = reports_cellules(*args, fidelite=True, omega_k=jnp.full(N, -2.5), lam=jnp.zeros(R))
    assert np.allclose(c0, c1, atol=1e-6)
    c2, _ = reports_cellules(*args, fidelite=True, omega_k=jnp.full(N, -2.5), lam=0.4)
    assert np.allclose(c2[2], c0[2], atol=1e-6)                 # ligne propre (k = a) inchangée


def test_modele_initialise_avec_lambda():
    import model.backtest.multi_election as M
    M.VARIANTES["hiercercle_ra_fid_omk_lam"] = {"eta_hier": True, "carte_cercle": True, "_reports_avant": True,
                                                "fidelite": True, "omega_electorat": True, "melange_lambda": True}
    b = M.preparer(2022, "T1-30", "hiercercle_ra_fid_omk_lam")
    from model.models.spatial_pooling.joint_model import spatial_pooling_model_ou
    info = initialize_model(jax.random.PRNGKey(0), spatial_pooling_model_ou, model_kwargs=b["kwargs"],
                            init_strategy=init_to_median)
    assert np.isfinite(float(info.potential_fn(info.param_info.z)))
    tr = handlers.trace(handlers.seed(spatial_pooling_model_ou, 0)).get_trace(**b["kwargs"])
    assert 0.0 < float(tr["lam_mix"]["value"]) < 1.0


def test_temoin_U_tau_nul():
    """§8.63 : τ = 0 ⇒ tous les électorats non finalistes au même partage ; τ = 1 ⇒ inchangé."""
    from model.models.spatial_pooling.geometry import W, reports_cellules
    rng = np.random.default_rng(4)
    R, N, G = 3, 5, W.shape[0]
    A = jnp.tile(jnp.asarray(rng.normal(size=(1, N, G))), (R, 1, 1))   # même situation, électorats 2-4
    lse = jax.nn.logsumexp(A, axis=1)
    ia, ib, k = jnp.zeros(R, int), jnp.ones(R, int), jnp.array([2, 3, 4])
    args = (A, jnp.ones(N), ia, ib, k, 0.8, jnp.zeros(R), lse, -2.5, jnp.full(R, 0.6), W)
    c0, _ = reports_cellules(*args)
    assert np.allclose(c0, reports_cellules(*args, tau=1.0)[0], atol=1e-6)
    cu, _ = reports_cellules(*args, tau=0.0)
    cu = np.asarray(cu)
    part = cu[:, 0] / (cu[:, 0] + cu[:, 1])
    assert np.allclose(part, part[0], atol=1e-5)


def test_ufid_finalistes_fixes():
    """§8.64 bis (c) : U_fid redonne l'agrégat, les non-finalistes sont uniformes, fid_newton=False inchangé."""
    from model.models.spatial_pooling.geometry import W, reports_cellules
    rng = np.random.default_rng(5)
    R, N, G = 3, 5, W.shape[0]
    A = jnp.tile(jnp.asarray(rng.normal(size=(1, N, G))), (R, 1, 1))
    lse = jax.nn.logsumexp(A, axis=1)
    ia, ib, k = jnp.zeros(R, int), jnp.ones(R, int), jnp.array([2, 3, 4])
    args = (A, jnp.ones(N), ia, ib, k, 0.8, jnp.zeros(R), lse, -2.5, jnp.full(R, 0.6), W)
    assert np.allclose(reports_cellules(*args, tau=0.0)[0], reports_cellules(*args, tau=0.0, fid_newton=False)[0])
    cf = np.asarray(reports_cellules(*args, tau=0.0, fid_newton=True, n_newton=12)[0])
    part = cf[:, 0] / (cf[:, 0] + cf[:, 1])
    assert np.allclose(part, part[0], atol=1e-5)
    c0 = np.asarray(reports_cellules(*args, tau=0.0)[0])
    assert not np.allclose(part[0], (c0[0, 0] / (c0[0, 0] + c0[0, 1])), atol=1e-3)


def test_v2_sans_paire_retire_duels_et_reports(monkeypatch):
    """§8.69 : PRES2027_SANS_PAIRE retire les duels de la paire ET les reports de l'hypothèse."""
    import pandas as pd
    import model.backtest.second_round as sr
    as_of = sr.T1_DATES[2022] - pd.Timedelta(days=30)
    raw = sr.polls_historiques(2022, as_of)
    r2 = sr._sans_paire(raw, "Emmanuel Macron", "Jean-Luc Mélenchon")
    d = sr._duels_moyens(r2)
    assert d[(d.a == "Emmanuel Macron") & (d.b == "Jean-Luc Mélenchon")].empty
    assert not sr._duels_moyens(raw).query("a == 'Emmanuel Macron' and b == 'Jean-Luc Mélenchon'").empty
    cap = {}
    def faux_fit(raw, as_of, **kw):
        cap.update(kw); raise RuntimeError("stop")
    monkeypatch.setattr(sr, "fit_spatial_pooling", faux_fit)
    monkeypatch.setenv("PRES2027_SANS_PAIRE", "Emmanuel Macron|Jean-Luc Mélenchon")
    sr.VARIANTE_KW.clear(); sr.VARIANTE_KW.update({"_reports_avant": True})
    try:
        sr._fit(raw, as_of, 2022, {})
    except RuntimeError:
        pass
    hyp = cap["reports"]["hypothese"].str.replace("—", "–")
    assert not hyp.eq("Mélenchon – Macron").any() and hyp.eq("Macron – Zemmour").any()
