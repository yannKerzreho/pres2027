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
