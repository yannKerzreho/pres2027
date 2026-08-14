"""Tests de la géométrie ILR (model/models/bayesian_nowcast/latent.py) —
logique pure, sans échantillonnage. Vérifie les propriétés mathématiques
établies dans docs/spec_ssm_espace_latent.md avant de les utiliser dans le
filtre de Kalman (nowcast.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# `bayesian_nowcast` importe dynamax, absent des requirements de `main` où ce
# modèle monte sans être enregistré (cf. model/core/registered.py). Skip plutôt
# qu'échec de collecte : la même suite doit passer sur les deux branches.
pytest.importorskip("dynamax")

from model.models.bayesian_nowcast.latent import (
    PADDING_VAR, clr, helmert_basis, ilr_decode, ilr_encode, poll_observation,
)

RNG = np.random.default_rng(27)


def random_composition(K: int, concentration: float = 5.0) -> np.ndarray:
    return RNG.dirichlet(np.full(K, concentration)) * 100.0


def test_helmert_basis_orthonormee():
    for K in (3, 6, 12):
        V = helmert_basis(K)
        assert V.shape == (K, K - 1)
        np.testing.assert_allclose(V.T @ V, np.eye(K - 1), atol=1e-10)


def test_ilr_round_trip():
    K = 12
    V = helmert_basis(K)
    for _ in range(20):
        pi = random_composition(K)
        alpha = ilr_encode(pi, V)
        pi_hat = np.asarray(ilr_decode(alpha, V)) * 100.0
        np.testing.assert_allclose(pi_hat / pi_hat.sum(), pi / pi.sum(), atol=1e-6)


def test_ilr_isometrie_distance_aitchison():
    # ||alpha_1 - alpha_2||_2 == distance CLR (= Aitchison) entre les deux
    # compositions — c'est LA propriété qui justifie ILR (spec_ssm_espace_latent §3).
    K = 6
    V = helmert_basis(K)
    for _ in range(20):
        pi1, pi2 = random_composition(K), random_composition(K)
        d_alpha = np.linalg.norm(ilr_encode(pi1, V) - ilr_encode(pi2, V))
        d_clr = np.linalg.norm(clr(pi1) - clr(pi2))
        assert abs(d_alpha - d_clr) < 1e-8


def test_poll_observation_sondage_complet():
    K = 5
    V = helmert_basis(K)
    pi_true = random_composition(K)
    alpha_true = ilr_encode(pi_true, V)
    y, H, R = poll_observation(pi_true.copy(), n=2000, V=V)
    assert y.shape == (K - 1,) and H.shape == (K - 1, K - 1) and R.shape == (K - 1, K - 1)
    # sondage complet (m=K) : aucune ligne de padding
    assert not np.any(np.diag(R) == PADDING_VAR)
    # H . alpha_true doit reconstruire y (à peu de bruit près, ici nul car pas
    # d'échantillonnage simulé — pi_true EST les parts observées). Tolérance
    # float32 (poll_observation passe par JAX, précision par défaut).
    np.testing.assert_allclose(H @ alpha_true, y, atol=1e-6)


def test_poll_observation_sondage_partiel_padding_neutre():
    K = 8
    V = helmert_basis(K)
    shares = random_composition(K)
    shares_partial = shares.copy()
    tested_mask = np.array([True, True, True, False, True, False, True, True])  # 6/8 testés
    shares_partial[~tested_mask] = np.nan

    y, H, R = poll_observation(shares_partial, n=1000, V=V)
    m = tested_mask.sum()
    n_pad = (K - 1) - (m - 1)
    assert n_pad > 0
    # lignes de padding : H nulle, y nul, R = PADDING_VAR (contribution
    # constante à la vraisemblance, indépendante de alpha)
    pad_rows = slice(m - 1, K - 1)
    assert np.all(H[pad_rows, :] == 0.0)
    assert np.all(y[pad_rows] == 0.0)
    assert np.all(np.diag(R)[pad_rows] == PADDING_VAR)
    assert np.all(R[pad_rows, :m - 1] == 0.0) and np.all(R[:m - 1, pad_rows] == 0.0)


def test_poll_observation_covariance_symetrique_definie_positive():
    K = 7
    V = helmert_basis(K)
    shares = random_composition(K)
    shares[[2, 5]] = np.nan
    _, _, R = poll_observation(shares, n=800, V=V)
    np.testing.assert_allclose(R, R.T)
    eigvals = np.linalg.eigvalsh(R)
    assert np.all(eigvals > 0)


def test_R_est_bien_la_variance_multinomiale_sur_les_parts():
    """LE test de la covariance d'observation (session du 2026-08-11, travail
    sur la couverture) : sur un sondage COMPLET, `H` est inversible, donc `R`
    induit une covariance explicite sur `alpha`, qu'on peut pousser jusqu'aux
    PARTS par la jacobienne de `softmax`. Elle doit valoir EXACTEMENT la
    covariance multinomiale `(diag(p) - p p^T)/n` — donc en particulier
    `Var(p_i) = p_i(1-p_i)/n`, la formule dont tout le reste dérive.

    Vérifie du même coup que le facteur `(1-p_i)`, absent de la forme
    log-ratio `1/(n p_i)`, est bien restitué par la projection CLR."""
    K = 8
    V = helmert_basis(K)
    p = np.array([34.0, 19.0, 13.0, 10.0, 8.0, 5.0, 3.0, 8.0])
    p = p / p.sum()
    n = 1000.0
    _, H, R = poll_observation(p * 100.0, n=n, V=V)

    Hinv = np.linalg.inv(H)
    cov_alpha = Hinv @ R @ Hinv.T
    jac = (np.diag(p) - np.outer(p, p)) @ V        # d(pi)/d(alpha), softmax
    cov_pi = jac @ cov_alpha @ jac.T

    np.testing.assert_allclose(np.diag(cov_pi), p * (1 - p) / n, rtol=1e-6)
    np.testing.assert_allclose(cov_pi, (np.diag(p) - np.outer(p, p)) / n, atol=1e-9)


def test_deff_deflate_l_echantillon():
    K = 6
    V = helmert_basis(K)
    shares = random_composition(K)
    _, _, R1 = poll_observation(shares.copy(), n=1000, V=V)
    _, _, R2 = poll_observation(shares.copy(), n=1000, V=V, deff=2.0)
    _, _, R3 = poll_observation(shares.copy(), n=500, V=V)
    np.testing.assert_allclose(R2, 2.0 * R1, rtol=1e-6)
    np.testing.assert_allclose(R2, R3, rtol=1e-6)   # deff=2 <=> n/2


def test_ilr_local_et_alr_local_donnent_le_meme_posterieur():
    """Théorème d'invariance (spec_ssm.md §5.2) : sous la covariance
    delta-method, ILR local et ALR local ne sont que deux systèmes de
    coordonnées du MÊME modèle de bruit — la contribution à la vraisemblance
    (forme précision : `HᵀR⁻¹H` et `HᵀR⁻¹y`) doit être identique. Depuis que
    `full_covariance` ne bascule plus la FORME de `R` (2026-08-11), c'est un
    vrai test de non-régression de l'encodage."""
    K = 12
    V = helmert_basis(K)
    shares = np.full(K, np.nan)
    tested = [0, 1, 2, 3, 5, 6, 7, 8]
    shares[tested] = [1.0, 13.0, 5.0, 3.5, 34.0, 19.0, 10.0, 4.0]
    m = len(tested)

    prec = []
    for full_cov in (False, True):
        y, H, R = poll_observation(shares.copy(), n=1000, V=V, full_covariance=full_cov)
        s = slice(0, m - 1)
        Rinv = np.linalg.inv(R[s, s])
        prec.append((H[s].T @ Rinv @ H[s], H[s].T @ Rinv @ y[s]))
    np.testing.assert_allclose(prec[0][0], prec[1][0], atol=1e-3)   # float32 JAX
    np.testing.assert_allclose(prec[0][1], prec[1][1], atol=1e-3)


def test_bruit_isotrope_sous_estime_les_grands_candidats():
    """`isotropic=True` (variante de comparaison) revient à supposer `p_i=1` :
    sa variance est plus petite que la multinomiale d'un facteur de l'ordre
    de `1/p`. Test de la RAISON pour laquelle ce mode n'est plus le défaut."""
    K = 6
    V = helmert_basis(K)
    shares = np.array([34.0, 25.0, 20.0, 12.0, 6.0, 3.0])
    _, _, R_multi = poll_observation(shares.copy(), n=1000, V=V)
    _, _, R_iso = poll_observation(shares.copy(), n=1000, V=V, isotropic=True)
    assert np.all(np.diag(R_multi) > np.diag(R_iso))
    assert np.trace(R_multi) / np.trace(R_iso) > 3.0


def test_poll_observation_moins_de_deux_slots_leve():
    K = 5
    V = helmert_basis(K)
    shares = np.full(K, np.nan)
    shares[0] = 40.0
    with pytest.raises(ValueError):
        poll_observation(shares, n=500, V=V)
