import pytest
"""§8.92 : le retrait V2 ôte duels ET reports de la paire ; V1 n'ôte que les reports."""
import model.backtest.essai_2027 as X


def test_retraits():
    raw, rep, champ = X.donnees()
    assert len(champ) >= 5 and "Le Pen" in champ
    d = X.duels(raw)
    r2, p2 = X.retirer(raw, rep, "v2:Le Pen|Philippe")
    e = X.duels(r2)
    assert e[(e.a == "Le Pen") & (e.b == "Philippe")].empty
    assert len(d) - len(e) == len(d[(d.a == "Le Pen") & (d.b == "Philippe")]) > 0
    assert not p2.hypothese.eq("Philippe – Le Pen").any() and p2.hypothese.eq("Attal – Le Pen").any()
    r1, p1 = X.retirer(raw, rep, "v1:Attal – Le Pen")
    assert len(r1) == len(raw) and not p1.hypothese.eq("Attal – Le Pen").any()


def test_verdict_regle_figee():
    from model.backtest.lire_essai_2027 import verdict

    def S(lfi_s, lfi_c, v2_s, v2_c, v1_s, v1_c, suf=""):
        return {f"LFI_RN_segment{suf}": lfi_s, f"LFI_RN_cercle{suf}": lfi_c, f"V2_segment{suf}": [v2_s, 8],
                f"V2_cercle{suf}": [v2_c, 8], f"V1_segment{suf}": [v1_s, 9], f"V1_cercle{suf}": [v1_c, 9]}
    assert verdict(S(9.41, 10.53, 2.18, 1.95, 7.54, 8.88), "") == "non séparés"     # §8.97
    assert verdict(S(10, 8, 2, 2.5, 7, 7.5), "") == "cercle"                         # bords inclus
    assert verdict(S(10, 8, 2, 2.6, 7, 7), "") == "non séparés"
    assert verdict(S(6, 9, 2.4, 2, 7.4, 7, "_date"), "_date") == "segment"


def test_delta_ks_racine():
    import numpy as np
    from model.backtest.lire_essai_2027 import delta_ks
    p, n, l = [30, 20, 50], [0.5, 0.1, 0.02], [0.3, -1.0, 2.0]
    d = delta_ks(p, n, l, 0.55)
    w = np.array(p) * (1 - np.array(n))
    assert abs((w / (1 + np.exp(-(np.array(l) + d)))).sum() / w.sum() - 0.55) < 1e-9
    assert delta_ks(p, n, [0, 0, 0], 0.5) == pytest.approx(0, abs=1e-9)


def test_kappa_rho_defaut_inchange():
    import jax.numpy as jnp
    import numpy as np
    from model.models.spatial_pooling.geometry import W, reports_cellules
    rng = np.random.default_rng(0)
    R, N, G = 3, 4, len(np.asarray(W))
    A = jnp.asarray(rng.normal(size=(R, N, G)))
    lse = jnp.asarray(rng.normal(size=(R, G)))
    arg = (A, jnp.ones(N), jnp.array([0, 0, 1]), jnp.array([1, 1, 2]), jnp.array([2, 3, 3]), 0.9,
           jnp.zeros(R), lse, -2.0, jnp.array([.5, .45, .6]), W)
    a, _ = reports_cellules(*arg, fidelite=True)
    b, _ = reports_cellules(*arg, fidelite=True, kappa_rho=0.9)
    c, _ = reports_cellules(*arg, fidelite=True, kappa_rho=0.3)
    assert np.allclose(a, b, atol=0, rtol=0)
    assert not np.allclose(a, c)


def test_masque_lse_finalistes():
    import numpy as np
    from types import SimpleNamespace
    from model.models.spatial_pooling.fit import _masque_lse
    fit = SimpleNamespace(masque_lse=np.array([1.0, 0.0, 1.0, 0.0]))
    assert np.array_equal(_masque_lse(fit, np.array([1, 0]), np.array([2, 3]), False)[0], fit.masque_lse)
    m = _masque_lse(fit, np.array([1, 0]), np.array([2, 3]), True)
    assert m.tolist() == [[1, 1, 1, 0], [1, 0, 1, 1]]
    assert _masque_lse(SimpleNamespace(masque_lse=None), np.array([0]), np.array([1]), True) is None


def _faux_fit(cercle: bool, masque):
    import numpy as np
    from types import SimpleNamespace
    rng = np.random.default_rng(1)
    S, N = 3, 4
    return SimpleNamespace(mu_draws=np.tile(np.linspace(0.15, 0.85, N), (S, 1)),
                           sigma_draws=np.full((S, N), 0.12), kappa_draws=np.full(S, 0.8),
                           eta_draws=rng.normal(0, 0.1, (S, N)), omega_draws=np.full(S, -2.5),
                           masque_lse=np.asarray(masque, float), nu_draws=None,
                           arc_draws=np.full(S, 4.0) if cercle else None), rng.normal(0, 0.5, (S, N))


def test_lse_finalistes_non_regression():
    """§8.115 : drapeau éteint inchangé ; allumé = éteint quand les finalistes sont déjà dans le masque ;
    allumé ≠ éteint quand un finaliste en sort ; deux géométries, `ia` variable."""
    import numpy as np
    from model.models.spatial_pooling.fit import _duel_par_tirage
    ia, ib = np.array([0, 1, 0]), np.array([2, 3, 3])
    for cercle in (False, True):
        fit, w = _faux_fit(cercle, [1, 1, 1, 1])
        a = _duel_par_tirage(fit, w, ia, ib)
        assert np.array_equal(a, _duel_par_tirage(fit, w, ia, ib, lse_finalistes=True))
        fit, w = _faux_fit(cercle, [1, 1, 1, 0])
        off = _duel_par_tirage(fit, w, ia, ib)
        on = _duel_par_tirage(fit, w, ia, ib, lse_finalistes=True)
        assert np.array_equal(off[0], on[0]) and not np.allclose(off[1:], on[1:])
