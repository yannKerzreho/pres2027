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
