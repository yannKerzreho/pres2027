"""Tests du modèle « gp-pooling » — cf. model/models/gp_pooling/spec_gp_pooling.md.

Deux familles :
  - les identités mathématiques du postérieur (vérifiables sans données) ;
  - la propriété de fond qui justifie ce modèle : ajouter un sondage doit
    RÉDUIRE l'incertitude, là où `linear-pooling` l'augmente.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.core.gp_math import (
    clr_rows, draws_from_posterior, gp_posterior, ou_kernel, sampling_cov_clr_diag,
)

PARAMS = {"tau": 260.0, "sigma2": 0.116, "sigma_h": 0.126, "sigma_p": 0.039}


def _jeu(n_polls, meme_institut=False, h=None):
    """Petit jeu synthétique : `n_polls` sondages de 3 candidats."""
    rng = np.random.default_rng(0)
    Y = np.clip(np.array([[0.35, 0.40, 0.25]]) + rng.normal(0, 0.01, (n_polls, 3)), 0.01, None)
    Y = Y / Y.sum(axis=1, keepdims=True)
    n = np.full(n_polls, 1000.0)
    h = np.full(n_polls, 250.0) if h is None else np.asarray(h, dtype=float)
    inst = np.zeros(n_polls, int) if meme_institut else np.arange(n_polls)
    return clr_rows(Y), sampling_cov_clr_diag(Y, n) + PARAMS["sigma_p"] ** 2, h, inst


def test_noyau_ou_decroit_et_vaut_sigma2_a_distance_nulle():
    k = ou_kernel(np.array([0.0]), np.array([0.0, 50.0, 500.0]), tau=260.0, sigma2=0.116)[0]
    assert k[0] == pytest.approx(0.116)
    assert k[0] > k[1] > k[2] > 0.0


def test_clr_somme_a_zero():
    Z = clr_rows(np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]))
    assert np.allclose(Z.sum(axis=1), 0.0, atol=1e-12)


def test_softmax_du_clr_redonne_la_composition():
    # `draws_from_posterior` applique un softmax : ce n'est pas une déformation
    # ajoutée, c'est l'inverse exact de la transformation d'entrée.
    Y = np.array([[0.2, 0.3, 0.5]])
    Z = clr_rows(Y)
    e = np.exp(Z); assert np.allclose(e / e.sum(), Y, atol=1e-12)


def test_variance_decroit_quand_on_ajoute_un_sondage():
    """LA propriété qui justifie ce modèle (spec §1).

    `linear-pooling` fait l'inverse : son terme de désaccord grandit avec le
    pool, donc deux sondages du même jour élargissent son intervalle.
    """
    var = []
    for k in (1, 2, 4, 8):
        Z, V, h, inst = _jeu(k)
        post = gp_posterior(Z, V, h, inst, h_star=250.0, **{p: PARAMS[p]
                            for p in ("tau", "sigma2", "sigma_h")})
        var.append(post.var_latent[0])
    assert all(a > b for a, b in zip(var, var[1:])), f"variance non décroissante : {var}"


def test_sondages_du_meme_institut_reduisent_moins_que_d_instituts_differents():
    """Le house effect est partagé au sein d'un institut : empiler ses sondages
    ne doit pas faire tomber l'incertitude comme le feraient des maisons
    indépendantes."""
    kw = {p: PARAMS[p] for p in ("tau", "sigma2", "sigma_h")}
    Z, V, h, _ = _jeu(6, meme_institut=True)
    meme = gp_posterior(Z, V, h, np.zeros(6, int), h_star=250.0, **kw).var_latent[0]
    Z, V, h, _ = _jeu(6, meme_institut=False)
    diff = gp_posterior(Z, V, h, np.arange(6), h_star=250.0, **kw).var_latent[0]
    assert meme > diff


def test_variance_croit_avec_l_eloignement_temporel():
    """Extrapoler loin des sondages doit coûter — c'est le noyau qui l'impose."""
    kw = {p: PARAMS[p] for p in ("tau", "sigma2", "sigma_h")}
    Z, V, h, inst = _jeu(4, h=[300.0, 305.0, 310.0, 315.0])
    proche = gp_posterior(Z, V, h, inst, h_star=300.0, **kw).var_latent[0]
    loin = gp_posterior(Z, V, h, inst, h_star=100.0, **kw).var_latent[0]
    assert loin > proche


def test_variance_bornee_par_la_variance_a_priori():
    """Mean-reversion : très loin de toute observation, on retombe sur le prior,
    jamais au-delà. C'est ce qu'une marche aléatoire ne garantirait pas."""
    kw = {p: PARAMS[p] for p in ("tau", "sigma2", "sigma_h")}
    Z, V, h, inst = _jeu(4)
    tres_loin = gp_posterior(Z, V, h, inst, h_star=-100000.0, **kw).var_latent[0]
    # borne : sigma2 (prior) + l'incertitude sur mu, qui reste finie
    assert np.isfinite(tres_loin) and tres_loin < 10 * PARAMS["sigma2"]


def test_tirages_sur_le_simplexe():
    mean = np.array([0.4, -0.1, -0.3]); var = np.array([0.02, 0.03, 0.01])
    pi = draws_from_posterior(mean, var, 500, seed=0)
    assert pi.shape == (500, 3)
    assert np.allclose(pi.sum(axis=1), 1.0)
    assert (pi > 0).all()


def test_variance_d_observation_depasse_la_variance_latente():
    """`var_obs` ajoute le house effect et le bruit d'échantillonnage du sondage
    cible : prédire une OBSERVATION est nécessairement plus incertain que
    prédire l'opinion latente."""
    kw = {p: PARAMS[p] for p in ("tau", "sigma2", "sigma_h")}
    Z, V, h, inst = _jeu(5)
    post = gp_posterior(Z, V, h, inst, h_star=250.0, target_inst=0, target_v=V[0], **kw)
    assert (post.var_obs > post.var_latent).all()


def test_contrat_du_modele_sur_donnees_reelles():
    parsed = ROOT / "data" / "parsed" / "intentions_2027_wiki.csv"
    if not parsed.exists():
        pytest.skip("intentions_2027_wiki.csv absent")
    from model.core.base import validate_snapshot
    from model.core.live_dataset import load_raw_polls
    from model.models.gp_pooling.model import GPPooling

    m = GPPooling()
    raw = load_raw_polls(as_of="2026-08-11")
    snap = m.run(raw, "2026-08-11")
    validate_snapshot(snap)
    assert snap["model"] == "gp-pooling"
    # le snapshot ne doit décrire que les sondages réellement consommés
    assert snap["meta"]["n_sondages"] <= snap["meta"]["n_sondages_disponibles"]
