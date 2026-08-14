"""Tests du filtre de Kalman maison (`model/core/lgssm.py`).

La référence n'est pas une autre implémentation — c'est la DÉFINITION. Un modèle
d'état linéaire-gaussien rend le couple (états, observations) conjointement
gaussien ; on construit donc à la main la loi jointe de y_0..y_{T-1}, et on
compare :

  - `marginal_loglik` à la densité de cette gaussienne multivariée complète ;
  - `filtered_means[t]` / `filtered_covariances[t]` au conditionnement exact de
    z_t sur y_0..y_t.

C'est ce qui rend la suppression de dynamax vérifiable sans dynamax : ces tests
tomberaient sur n'importe quelle erreur de convention (décalage d'un pas entre
`F[t]` et l'observation, `init_cov` interprété comme état filtré plutôt que
prédit, oubli du terme de propagation), pas seulement sur une faute de calcul.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.core.lgssm import kalman_filter

# jax tourne en float32 par défaut : les écarts attendus sont ceux de la simple
# précision accumulée sur T pas, pas ceux d'une divergence d'algorithme.
RTOL, ATOL = 2e-4, 2e-4

RNG = np.random.default_rng(2027)


def _psd(n: int, scale: float = 1.0) -> np.ndarray:
    A = RNG.normal(size=(n, n))
    return scale * (A @ A.T + n * np.eye(n))


def _model(T: int, D: int, K: int, *, F_statique: bool):
    """Tire un modèle d'état aléatoire mais bien conditionné."""
    init_mean = RNG.normal(size=D)
    init_cov = _psd(D)
    Q = np.stack([_psd(D, 0.3) for _ in range(T)])
    H = RNG.normal(size=(T, K, D))
    R = np.stack([_psd(K, 0.5) for _ in range(T)])
    y = RNG.normal(size=(T, K))
    if F_statique:
        F = np.eye(D)
        F_par_pas = np.stack([np.eye(D)] * T)
    else:
        F_par_pas = np.stack([0.2 * RNG.normal(size=(D, D)) + np.eye(D) for _ in range(T)])
        F = F_par_pas
    return init_mean, init_cov, F, F_par_pas, Q, H, R, y


def _loi_jointe(init_mean, init_cov, F_par_pas, Q, H, R, T, D, K):
    """Moyenne et covariance de (y_0, ..., y_{T-1}), empilées en (T*K,).

    Convention testée : `init_*` décrit z_0 AVANT toute observation, et `F[t]`
    est la transition t -> t+1 (le dernier n'intervient donc jamais)."""
    mu_z = np.zeros((T, D))
    P_z = np.zeros((T, D, D))
    mu_z[0], P_z[0] = init_mean, init_cov
    for t in range(T - 1):
        mu_z[t + 1] = F_par_pas[t] @ mu_z[t]
        P_z[t + 1] = F_par_pas[t] @ P_z[t] @ F_par_pas[t].T + Q[t]

    # Cov(z_t, z_s) = P_t (F_{s-1} ... F_t)^T pour s >= t
    cov_z = np.zeros((T, T, D, D))
    for t in range(T):
        cov_z[t, t] = P_z[t]
        prop = np.eye(D)
        for s in range(t + 1, T):
            prop = F_par_pas[s - 1] @ prop
            cov_z[t, s] = P_z[t] @ prop.T
            cov_z[s, t] = cov_z[t, s].T

    mu_y = np.concatenate([H[t] @ mu_z[t] for t in range(T)])
    cov_y = np.zeros((T * K, T * K))
    for t in range(T):
        for s in range(T):
            bloc = H[t] @ cov_z[t, s] @ H[s].T
            if t == s:
                bloc = bloc + R[t]
            cov_y[t * K:(t + 1) * K, s * K:(s + 1) * K] = bloc
    return mu_y, cov_y, mu_z, P_z, cov_z


def _logpdf(x, mean, cov):
    d = x - mean
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, d)
    return -0.5 * (z @ z + len(x) * np.log(2 * np.pi)) - np.sum(np.log(np.diag(L)))


CAS = [
    pytest.param(5, 3, 3, True, id="F_statique_T5"),
    pytest.param(1, 4, 2, True, id="un_seul_pas"),
    pytest.param(6, 4, 2, False, id="F_datee_T6"),
    pytest.param(9, 6, 3, False, id="F_datee_T9"),
]


@pytest.mark.parametrize("T,D,K,F_statique", CAS)
def test_loglik_marginale_egale_la_gaussienne_jointe(T, D, K, F_statique):
    init_mean, init_cov, F, F_par_pas, Q, H, R, y = _model(T, D, K, F_statique=F_statique)
    mu_y, cov_y, *_ = _loi_jointe(init_mean, init_cov, F_par_pas, Q, H, R, T, D, K)

    obtenu = float(kalman_filter(init_mean, init_cov, F, Q, H, R, y).marginal_loglik)
    attendu = _logpdf(y.reshape(-1), mu_y, cov_y)
    assert obtenu == pytest.approx(attendu, rel=RTOL, abs=ATOL)


@pytest.mark.parametrize("T,D,K,F_statique", CAS)
def test_trajectoire_filtree_egale_le_conditionnement_exact(T, D, K, F_statique):
    init_mean, init_cov, F, F_par_pas, Q, H, R, y = _model(T, D, K, F_statique=F_statique)
    _, cov_y, mu_z, P_z, cov_z = _loi_jointe(init_mean, init_cov, F_par_pas, Q, H, R, T, D, K)

    res = kalman_filter(init_mean, init_cov, F, Q, H, R, y)

    for t in range(T):
        # Conditionnement gaussien de z_t sur y_0..y_t (et RIEN au-delà : c'est
        # un filtre, pas un lisseur — si l'implémentation utilisait y_{t+1}, ce
        # test le verrait).
        n_obs = (t + 1) * K
        cov_zy = np.concatenate([cov_z[t, s] @ H[s].T for s in range(t + 1)], axis=1)
        cov_yy = cov_y[:n_obs, :n_obs]
        mu_y_vu = np.concatenate([H[s] @ mu_z[s] for s in range(t + 1)])
        gain = cov_zy @ np.linalg.inv(cov_yy)

        mean_attendu = mu_z[t] + gain @ (y[:t + 1].reshape(-1) - mu_y_vu)
        cov_attendu = P_z[t] - gain @ cov_zy.T

        np.testing.assert_allclose(np.asarray(res.filtered_means[t]), mean_attendu,
                                   rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(np.asarray(res.filtered_covariances[t]), cov_attendu,
                                   rtol=RTOL, atol=ATOL)


def test_covariances_filtrees_symetriques_et_psd():
    """Le filtre alimente `dist.MultivariateNormal` en aval : une covariance qui
    dérive vers l'asymétrie ou une valeur propre négative y casse
    l'échantillonnage, et le message d'erreur n'aiderait pas à remonter ici."""
    init_mean, init_cov, F, _, Q, H, R, y = _model(12, 5, 3, F_statique=False)
    covs = np.asarray(kalman_filter(init_mean, init_cov, F, Q, H, R, y).filtered_covariances)
    for t, cov in enumerate(covs):
        np.testing.assert_allclose(cov, cov.T, rtol=0, atol=1e-6, err_msg=f"pas {t}")
        assert np.linalg.eigvalsh(cov).min() > -1e-6, f"valeur propre negative au pas {t}"


def test_parametres_statiques_et_dates_donnent_le_meme_resultat():
    """Un paramètre constant passé en (D, D) ou empilé en (T, D, D) doit donner
    exactement la même chose — c'est ce qui autorise le chemin « niveau » à
    passer `eye` sans le dupliquer T fois."""
    T, D, K = 8, 4, 4
    init_mean, init_cov, _, _, Q, H, R, y = _model(T, D, K, F_statique=True)
    R_statique = R[0]
    R_empile = np.stack([R[0]] * T)

    a = kalman_filter(init_mean, init_cov, np.eye(D), Q, H, R_statique, y)
    b = kalman_filter(init_mean, init_cov, np.stack([np.eye(D)] * T), Q, H, R_empile, y)

    assert float(a.marginal_loglik) == pytest.approx(float(b.marginal_loglik), rel=1e-6)
    np.testing.assert_allclose(np.asarray(a.filtered_means), np.asarray(b.filtered_means),
                               rtol=1e-5, atol=1e-6)
