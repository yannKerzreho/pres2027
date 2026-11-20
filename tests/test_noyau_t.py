"""Noyau de Student-t sur le segment (spec_second_tour §8.51)."""
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import init_to_median
from numpyro.infer.util import initialize_model

from model.models.spatial_pooling.geometry import log_attractivite, spatial_shares


def _cas():
    mu = jnp.array([0.15, 0.45, 0.9]); sg = jnp.array([0.08, 0.08, 0.08]); w = jnp.zeros(3)
    return mu, sg, w


def test_nu_grand_redonne_le_gaussien():
    mu, sg, w = _cas()
    assert np.allclose(log_attractivite(mu, sg, w), log_attractivite(mu, sg, w, 1e7), atol=1e-3)
    m = jnp.ones(3)
    assert np.allclose(spatial_shares(mu, sg, w, m), spatial_shares(mu, sg, w, m, nu=1e7), atol=1e-4)


def test_queues_lourdes_adoucissent_le_votant_lointain():
    """Votant à l'extrême gauche (v ≈ 0,05) : écart d'attractivité centre / droite plus petit en t."""
    mu, sg, w = _cas()
    g, t = np.asarray(log_attractivite(mu, sg, w)), np.asarray(log_attractivite(mu, sg, w, 3.0))
    b = 2                                    # nœud bas de la grille (v petit)
    assert abs(t[1, b] - t[2, b]) < abs(g[1, b] - g[2, b])


def test_nu_par_tirage():
    mu, sg, w = _cas()
    S = 4
    A = log_attractivite(jnp.tile(mu, (S, 1)), jnp.tile(sg, (S, 1)), jnp.tile(w, (S, 1)), jnp.full(S, 4.0))
    assert A.shape[:2] == (S, 3)


def test_modele_initialise_avec_noyau_t():
    import model.backtest.multi_election as M
    M.VARIANTES["hier_ra_t"] = {"eta_hier": True, "_reports_avant": True, "noyau_t": True}
    b = M.preparer(2022, "T1-30", "hier_ra_t")
    from model.models.spatial_pooling.joint_model import spatial_pooling_model_ou
    assert b["kwargs"].get("noyau_t") is True
    info = initialize_model(jax.random.PRNGKey(0), spatial_pooling_model_ou, model_kwargs=b["kwargs"],
                            init_strategy=init_to_median)
    assert np.isfinite(float(info.potential_fn(info.param_info.z)))


def test_partition_reproduit_les_modes_d_A2_ancien():
    """Contrôle figé avant lecture (revue locale du 20/11 PM, piste 1) : sur les valeurs des 8
    anciennes chaînes d'A′2 (results/diag_modes_hier_ra_2022_T1-30.json), le sélecteur doit
    rendre H = {1, 4, 6, 7}, B = {2, 3, 5}, {0}."""
    import json
    from pathlib import Path
    from model.backtest.modes import partition
    d = json.loads((Path(__file__).resolve().parents[1] / "model/backtest/results/diag_modes_hier_ra_2022_T1-30.json").read_text())
    lignes = [{"chaine": x["chaine"], "lp": x["lp"], "sigma_melenchon": x["sigma_jlm"]} for x in d]
    p = partition(lignes)
    assert sorted(p["dominant"]) == [1, 4, 6, 7]
    assert sorted(map(sorted, p["autres"])) == [[0, 2, 3, 5]] or sorted(map(sorted, p["autres"])) == [[0], [2, 3, 5]]
