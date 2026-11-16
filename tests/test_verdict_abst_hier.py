import numpy as np

from model.backtest.verdict_abst_hier import bootstrap, log_score_kde


def test_kde_proche_de_la_gaussienne_unimodale():
    rng = np.random.default_rng(0)
    z = rng.normal(0.2, 0.1, 4000)
    t = 1 / (1 + np.exp(-z))
    y = 0.56
    zy = np.log(y / (1 - y))
    exact = -0.5 * np.log(2 * np.pi * 0.01) - (zy - 0.2) ** 2 / 0.02 - np.log(y * (1 - y))
    assert abs(log_score_kde(t, y) - exact) < 0.1


def test_kde_bimodale_bat_la_gaussienne_au_mode():
    rng = np.random.default_rng(1)
    z = np.r_[rng.normal(-0.4, 0.03, 800), rng.normal(0.4, 0.03, 800)]
    t = 1 / (1 + np.exp(-z))
    y = float(1 / (1 + np.exp(-0.4)))
    m, s = z.mean(), z.std()
    gauss = -0.5 * np.log(2 * np.pi * s ** 2) - (0.4 - m) ** 2 / (2 * s ** 2) - np.log(y * (1 - y))
    assert log_score_kde(t, y) > gauss + 1.0


def test_bootstrap_constant():
    assert bootstrap(np.full(5, 0.3)) == [0.3, 0.3]
