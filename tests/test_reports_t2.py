"""(A)(1) reports prédits (spec_second_tour §8.38) : cohérence avec le duel."""
import numpy as np

from model.backtest.reports_t2 import critere_D, reports_predits
from model.models.spatial_pooling.geometry import W, log_attractivite


def _cas(seed=0, S=5, N=5, sigma_egal=False):
    rng = np.random.default_rng(seed)
    mu = np.sort(rng.uniform(0.1, 0.9, (S, N)), axis=1)
    sigma = np.full((S, N), 0.12) if sigma_egal else rng.uniform(0.08, 0.2, (S, N))
    w = rng.normal(0, 0.5, (S, N))
    A = np.asarray(log_attractivite(mu, sigma, w))
    from scipy.special import logsumexp
    return A, logsumexp(A, axis=1), rng.uniform(0.5, 1.5, S), rng.normal(-2, 0.5, S)


def test_somme_et_agregat():
    A, lse, kappa, omega = _cas()
    s_ab = np.full(len(kappa), 0.58)
    champ = np.arange(A.shape[1])
    r = reports_predits(A, champ, 1, 3, kappa, omega, lse, s_ab)
    tot = r["a"] + r["b"] + r["aucun"]
    assert np.allclose(tot, 1.0)
    # agrégat des électorats parmi exprimés = duel publié
    agg = (r["E"] * r["a"]).sum(1) / (r["E"] * (r["a"] + r["b"])).sum(1)
    assert np.allclose(agg, s_ab, atol=1e-6)
    assert np.allclose(r["E"].sum(1), 1.0, atol=1e-6)


def test_monotone_en_position_sans_abstention():
    A, _, kappa, _ = _cas(seed=1, sigma_egal=True)
    champ = np.arange(A.shape[1])
    r = reports_predits(A, champ, 0, 4, kappa, np.zeros(len(kappa)), None, np.full(len(kappa), 0.5))
    assert np.allclose(r["aucun"], 0.0, atol=1e-6)
    part = r["a"] / (r["a"] + r["b"])
    # électorats rangés par position : plus à gauche ⇒ plus pour le finaliste de gauche (§2.3)
    assert (np.diff(part, axis=1) <= 1e-9).all()   # σ égaux : rapport de vraisemblance monotone


def test_critere_D():
    l = [{"variante": "hier", "cellule": "partage", "electorat": "Jean-Luc Mélenchon",
          "election": e, "erreur": 12.0, "z": 2.5} for e in (2017, 2022)]
    assert critere_D(l, "hier")["ouvrir_D"]
    l[0]["z"] = 1.0
    assert not critere_D(l, "hier")["ouvrir_D"]


def test_fer_a_cheval_extremes_vers_b():
    """§8.40 : Δν < 0 fait fuir vers b les électorats excentrés, à agrégat fixé."""
    A, _, kappa, _ = _cas(seed=2, sigma_egal=True)
    kappa = np.full(len(kappa), 0.1)              # pente géométrique faible : le terme ν se voit
    champ = np.arange(A.shape[1])
    s = np.full(len(kappa), 0.55)
    args = (A, champ, 2, 3, kappa, np.zeros(len(kappa)), None, s)
    r0, r1 = reports_predits(*args), reports_predits(*args, dnu=-3.0)
    p0, p1 = (r["a"] / (r["a"] + r["b"]) for r in (r0, r1))
    assert (p1[:, 0] < p0[:, 0]).all()           # l'électorat le plus à gauche fuit vers b
    agg = (r1["E"] * r1["a"]).sum(1) / (r1["E"] * (r1["a"] + r1["b"])).sum(1)
    assert np.allclose(agg, s, atol=1e-6)


def test_plancher_rapproche_de_la_moitie():
    A, lse, kappa, omega = _cas(seed=3)
    champ = np.arange(A.shape[1])
    s = np.full(len(kappa), 0.5)
    r = reports_predits(A, champ, 0, 4, kappa, omega, lse, s, plancher=0.4)
    p = r["a"] / (r["a"] + r["b"])
    assert (p > 0.15).all() and (p < 0.85).all()


def test_2d_point_fixe_garde_le_t1():
    """§8.41 : après re-résolution de w, parts T1 du champ = cible ; λ grand ≈ 1D."""
    from model.backtest.reports_2d import attractivite_2d, parts_t1, resoudre_w
    rng = np.random.default_rng(4)
    S, N = 3, 5
    mu = np.sort(rng.uniform(0.1, 0.9, (S, N)), axis=1)
    sigma = np.full((S, N), 0.12)
    w = rng.normal(0, 0.5, (S, N))
    champ = np.arange(4)
    A1 = np.asarray(log_attractivite(mu, sigma, w))
    cible = parts_t1(A1, champ, np.asarray(W))
    nu = rng.uniform(0.2, 0.8, N)
    _, _, ecart = resoudre_w(mu, sigma, w, nu, 0.15, champ, cible)
    assert ecart < 1e-4
    A, poids = attractivite_2d(mu, sigma, w, nu, 50.0)        # u sans effet
    assert np.allclose(parts_t1(A, champ, poids), cible, atol=1e-6)


def test_melange_anti_systeme():
    """§8.43 : π_as pousse vers b ; δ recalé garde le duel ; cellules sommant à 1."""
    from model.backtest.reports_t2 import composantes, melange
    A, lse, kappa, omega = _cas(seed=5)
    champ = np.arange(A.shape[1])
    E, R, G = composantes(A, champ, 1, 3, kappa, omega, lse)
    s = np.full(len(kappa), 0.6)
    r0 = melange(E, R, G, s)
    pa = np.array([0.3, 0, 0, 0, 0.1])
    r1 = melange(E, R, G, s, pa, np.array([0, 0, 0, 0, 0.05]))
    assert np.allclose(r1["a"] + r1["b"] + r1["aucun"], 1.0)
    assert (r1["b"][:, 0] > r0["b"][:, 0]).all()
    agg = (E * r1["a"]).sum(1) / (E * (r1["a"] + r1["b"])).sum(1)
    assert np.allclose(agg, s, atol=1e-6)
    assert (r1["decalage_duel"] < 0).all()            # sans recalage, le mélange fait perdre a


def test_ecarts_stabilite():
    from model.backtest.stabilite_2027 import ecarts
    e = ecarts({"A|B": [0.50, 0.53, 0.51, 0.40]}, [0, 1, 2])
    assert e["A|B"]["ecart_toutes_pt"] == 13.0 and e["A|B"]["ecart_retenues_pt"] == 3.0
