"""Invariants de `tendance.py`."""
from __future__ import annotations

import numpy as np
import pytest

from model.models.spatial_pooling import tendance as td

H = 242.0          # horizon type : as_of -> 18 avril 2027


@pytest.fixture
def tirages():
    rng = np.random.default_rng(0)
    base = np.array([0.05, 0.10, 0.15, 0.22, 0.30, 0.18])
    pi = base * np.exp(rng.normal(0, 0.25, (400, len(base))))
    return pi / pi.sum(1, keepdims=True)


def _cible(K, k, v):
    c = np.zeros(K); c[k] = v
    return c


def test_pente_nulle_ne_fait_rien(tirages):
    """Le chemin par défaut du site : aucune tendance posée, aucun effet."""
    rho = td.calibre_pente(tirages, np.zeros(tirages.shape[1]))
    assert np.array_equal(td.applique(tirages, rho, H), tirages)


def test_masse_conservee(tirages):
    """`(1-Rt) + Rt = 1` : exact, pas renormalisé après coup."""
    rho = td.calibre_pente(tirages, _cible(tirages.shape[1], 4, 1.0))
    assert np.abs(td.applique(tirages, rho, H).sum(1) - 1.0).max() < 1e-12


@pytest.mark.parametrize("k,cible", [(4, 1.0), (0, 1.0), (2, -0.3), (4, 0.5)])
def test_le_curseur_dit_ce_quil_fait(tirages, k, cible):
    """« +1 pp/mois » doit réaliser 1 pp/mois, pour un gros comme pour un petit.

    C'est la propriété qui a fait retenir le prélèvement UNIFORME : en
    prélèvement local le facteur de conversion tombait à 0,3-0,5 et variait du
    simple au double selon le candidat, si bien que le curseur ne voulait pas
    dire la même chose d'une ligne à l'autre.
    """
    rho = td.calibre_pente(tirages, _cible(tirages.shape[1], k, cible))
    pit = td.applique(tirages, rho, H)
    realise = (np.median(pit[:, k]) - np.median(tirages[:, k])) * 100 / (H / td.JOURS_PAR_MOIS)
    assert abs(realise - cible) < 1e-3


def test_linearite(tirages):
    """La trajectoire est exactement linéaire : c'est ce qui permet la forme
    fermée dans le navigateur, sans solveur."""
    rho = td.calibre_pente(tirages, _cible(tirages.shape[1], 4, 1.0))
    a = td.applique(tirages, rho, H)
    b = td.applique(tirages, rho, H / 2)
    assert np.abs((tirages + a) / 2 - b).max() < 1e-12


def test_la_perte_est_proportionnelle_a_la_part(tirages):
    """Prélèvement UNIFORME : chacun perd au prorata de sa part, pas de sa
    position sur l'axe. C'est l'hypothèse assumée (spec §2) et il faut qu'un
    test la fige — la remplacer par une captation locale doit casser ici."""
    K = tirages.shape[1]
    rho = td.calibre_pente(tirages, _cible(K, 4, 1.0))
    pit = td.applique(tirages, rho, H)
    m0, m1 = np.median(tirages, 0), np.median(pit, 0)
    perte = (m0 - m1)[np.arange(K) != 4] / m0[np.arange(K) != 4]
    assert perte.std() < 1e-9          # même perte RELATIVE pour tous


def test_bornes_asymetriques(tirages):
    """Une trajectoire linéaire ne sature pas : elle sort du simplexe. La borne
    basse mord vite (on ne perd pas plus qu'on n'a, sur AUCUN tirage), la haute
    est lâche. L'interface doit refléter cette asymétrie."""
    lo, hi = td.bornes_pp_mois(tirages, H)
    assert np.all(lo < 0) and np.all(hi > 0)
    assert np.all(hi > -lo * 5)                 # franchement asymétrique
    m = np.median(tirages, 0)
    assert lo[np.argmin(m)] > lo[np.argmax(m)]  # le petit est le plus contraint


@pytest.mark.parametrize("cote", ["bas", "haut"])
def test_les_bornes_tiennent(tirages, cote):
    """Juste en deçà de la borne les parts restent positives ; au-delà, non."""
    K = tirages.shape[1]
    lo, hi = td.bornes_pp_mois(tirages, H)
    k = int(np.argmin(np.median(tirages, 0))) if cote == "bas" else int(np.argmax(np.median(tirages, 0)))
    b = lo[k] if cote == "bas" else hi[k]
    dedans = td.applique(tirages, td.calibre_pente(tirages, _cible(K, k, b * 0.97)), H)
    dehors = td.applique(tirages, td.calibre_pente(tirages, _cible(K, k, b * 1.10)), H)
    assert dedans.min() > 0
    assert dehors.min() < 0
