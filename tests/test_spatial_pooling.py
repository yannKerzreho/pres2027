"""Invariants de `spatial_pooling`, sans lancer de MCMC.

Le paquet a été très largement réécrit et n'avait aucun test, alors que les
autres modèles en ont. On couvre ici ce qui se vérifie en quelques
millisecondes : les fonctions pures de `geometry.py`, les invariants de mise en
forme des sondages, et le fait que le modèle TRACE avec les sites attendus. Un
fit complet coûte 90 s, il n'a pas sa place dans la suite.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpyro import handlers

from model.models.spatial_pooling import (
    ORDER_GROUPS, ancres_ecartees, build_poll_arrays, ou_kl_basis, position_anchors,
    spatial_pooling_model_ou, spatial_shares,
)

logit = lambda p: np.log(p / (1.0 - p))  # noqa: E731


# --- geometry : ancres --------------------------------------------------------

def test_ancres_ecartees_impose_le_plancher_et_garde_l_ordre():
    a = np.array([0.10, 0.11, 0.115, 0.60, 0.90])   # trois quasi confondues
    out = ancres_ecartees(a, min_gap=0.25)
    assert np.all(np.diff(logit(np.sort(out))) >= 0.25 - 1e-9)
    assert np.array_equal(np.argsort(a), np.argsort(out)), "l'ordre doit être préservé"
    assert 0.0 < out.min() and out.max() < 1.0


def test_ancres_ecartees_ne_touche_pas_ce_qui_est_deja_conforme():
    """Propriété qui manquait à la première version : une cascade de poussées
    déplaçait TOUT l'aval dès qu'un couple touchait le plancher."""
    a = 1.0 / (1.0 + np.exp(-np.array([-2.0, -1.0, 0.0, 1.0, 2.0])))   # écarts de 1,0
    out = ancres_ecartees(a, min_gap=0.25)
    assert np.allclose(out, a), "aucun couple ne viole la contrainte, rien ne doit bouger"


def test_ancres_ecartees_conserve_la_moyenne_en_logit():
    a = np.array([0.20, 0.21, 0.22, 0.80])
    out = ancres_ecartees(a, min_gap=0.30)
    assert logit(out).mean() == pytest.approx(logit(a).mean(), abs=1e-9)


def test_ancres_ecartees_est_idempotente():
    a = np.array([0.05, 0.06, 0.4, 0.95])
    une = ancres_ecartees(a, 0.25)
    assert np.allclose(ancres_ecartees(une, 0.25), une)


# --- geometry : parts et base KL ---------------------------------------------

def test_spatial_shares_somme_a_un_sur_le_champ_masque():
    rng = np.random.default_rng(0)
    N = 6
    mu = jnp.asarray(np.sort(rng.uniform(0.1, 0.9, N)))
    sigma = jnp.asarray(np.full(N, 0.15))
    w = jnp.asarray(rng.normal(size=N))
    mask = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    pi = np.asarray(spatial_shares(mu, sigma, w, jnp.asarray(mask)))
    assert pi[mask == 0] == pytest.approx(0.0, abs=1e-12), "un candidat hors champ pèse 0"
    assert pi.sum() == pytest.approx(1.0, abs=1e-9)


def test_spatial_shares_retirer_un_candidat_profite_a_son_VOISIN():
    """La promesse du modèle : le report va aux positions voisines, pas au
    prorata. On retire le candidat du milieu et on vérifie que son voisin
    immédiat gagne plus, en relatif, que le candidat le plus éloigné."""
    mu = jnp.asarray([0.20, 0.50, 0.55, 0.90])
    sigma = jnp.asarray(np.full(4, 0.12))
    w = jnp.asarray(np.zeros(4))
    plein = np.array([1.0, 1.0, 1.0, 1.0])
    sans = np.array([1.0, 0.0, 1.0, 1.0])
    a = np.asarray(spatial_shares(mu, sigma, w, jnp.asarray(plein)))
    b = np.asarray(spatial_shares(mu, sigma, w, jnp.asarray(sans)))
    gain_voisin = b[2] / a[2]        # à 0,55, collé au retiré (0,50)
    gain_lointain = b[3] / a[3]      # à 0,90
    assert gain_voisin > gain_lointain


def test_ou_kl_basis_centree_est_orthogonale_a_la_constante():
    dates = np.arange(10, dtype=float) * 7.0
    base, k = ou_kl_basis(dates, dates.max() + 1.0, tau_ou=262.0, center=True)
    assert base.shape == (k, len(dates) + 1), "la dernière colonne est `as_of`"
    assert np.abs(base.sum(axis=1)).max() < 1e-9, "le niveau est porté ailleurs"


def test_ou_kl_basis_non_centree_capture_la_variance_visee():
    dates = np.arange(20, dtype=float) * 3.0
    base, k = ou_kl_basis(dates, dates.max() + 1.0, tau_ou=262.0, var_cible=0.99)
    t = np.concatenate([dates, [dates.max() + 1.0]])
    C = np.exp(-np.abs(t[:, None] - t[None, :]) / 262.0)
    assert np.trace(base.T @ base) / np.trace(C) > 0.98


# --- mise en forme des sondages ----------------------------------------------

def _sondages_jouets() -> pd.DataFrame:
    lignes = []
    for notice, (hyp, champ) in enumerate(
            [("A", {"X": 30.0, "Y": 25.0, "Z": 20.0}), ("B", {"X": 40.0, "Y": 30.0})]):
        for cand, val in champ.items():
            lignes.append(dict(notice=f"n{notice}", hypothese=hyp, candidat=cand,
                               intention=val, date_fin=pd.Timestamp("2026-03-01"),
                               echantillon=1000.0, institut="Ifop"))
    return pd.DataFrame(lignes)


def test_build_poll_arrays_renormalise_et_deflate():
    a = build_poll_arrays(_sondages_jouets(), ["X", "Y", "Z"])
    tm, Y = a["tested_mask"], a["Y"]
    assert tm.shape[0] == 2
    for p in range(2):
        assert Y[p][tm[p] > 0].sum() == pytest.approx(1.0), "Y est renormalisé sur le champ"
        assert Y[p][tm[p] == 0].sum() == pytest.approx(0.0)
    # `Np` = échantillon x masse conservée ; ici le roster couvre tout le bulletin
    assert (a["Np"] <= 1000.0 + 1e-9).all()


def test_build_poll_arrays_jette_les_champs_a_moins_de_deux_candidats():
    df = _sondages_jouets()
    df = pd.concat([df, pd.DataFrame([dict(notice="n9", hypothese="A", candidat="X",
                                           intention=100.0, date_fin=pd.Timestamp("2026-03-01"),
                                           echantillon=1000.0, institut="Ifop")])])
    a = build_poll_arrays(df, ["X", "Y", "Z"])
    assert a["tested_mask"].shape[0] == 2, "le nœud à un seul candidat est écarté"


# --- le modèle trace ----------------------------------------------------------

def test_le_modele_trace_avec_les_sites_attendus():
    rng = np.random.default_rng(0)
    N, P, M = 5, 12, 8
    dates = np.arange(M, dtype=float) * 5.0
    kl, _ = ou_kl_basis(dates, dates.max() + 1.0, 262.0, center=True)
    mask = (rng.random((P, N)) > 0.3).astype(float)
    mask[mask.sum(axis=1) < 2] = 1.0
    kw = dict(tested_mask=jnp.asarray(mask),
              Y=jnp.asarray(rng.random((P, N)) * mask / N),
              Np=jnp.asarray(np.full(P, 900.0)),
              date_idx=jnp.asarray(rng.integers(0, M, P)),
              kl_basis=jnp.asarray(kl),
              pos_anchor=jnp.asarray((np.arange(N) + 1.0) / (N + 1.0)),
              dynamic_mask=np.array([1, 1, 1, 0, 0], bool), tau_ou=262.0)
    tr = handlers.trace(handlers.seed(spatial_pooling_model_ou,
                                      jax.random.PRNGKey(0))).get_trace(**kw)
    sites = {k for k, v in tr.items() if v["type"] == "sample"}
    assert {"sigma_global", "z_sigma", "z_pos", "sigma_w", "z_m", "z_w"} <= sites
    assert not sites & {"sigma_slot", "sd_delta_mu", "sd_delta_sigma", "mu_base", "mu_gaps"}, \
        "les paramètres de blocs ont été retirés (§12.26)"
    mu = np.asarray(tr["mu"]["value"])
    assert ((mu > 0) & (mu < 1)).all()
    # les candidats statiques n'ont pas de chemin : une seule ligne de `z_w` par dynamique
    assert tr["z_w"]["value"].shape[0] == 3


def test_les_candidats_statiques_ont_un_w_constant():
    rng = np.random.default_rng(1)
    N, P, M = 4, 10, 6
    dates = np.arange(M, dtype=float) * 5.0
    kl, _ = ou_kl_basis(dates, dates.max() + 1.0, 262.0, center=True)
    mask = np.ones((P, N))
    kw = dict(tested_mask=jnp.asarray(mask), Y=jnp.asarray(rng.random((P, N)) / N),
              Np=jnp.asarray(np.full(P, 900.0)),
              date_idx=jnp.asarray(rng.integers(0, M, P)), kl_basis=jnp.asarray(kl),
              pos_anchor=jnp.asarray(np.array([0.2, 0.4, 0.6, 0.8])),
              dynamic_mask=np.array([1, 1, 0, 0], bool), tau_ou=262.0)
    tr = handlers.trace(handlers.seed(spatial_pooling_model_ou,
                                      jax.random.PRNGKey(2))).get_trace(**kw)
    niveau = np.asarray(tr["w_level"]["value"])
    assert np.asarray(tr["w_now"]["value"])[2:] == pytest.approx(niveau[2:], abs=1e-9)


# --- cohérence des tables éditoriales ----------------------------------------

def test_position_anchors_suit_l_ordre_du_catalogue():
    cands = ["Le Pen", "Mélenchon", "Attal"]          # volontairement désordonnés
    a = position_anchors(cands)
    assert a[1] < a[2] < a[0], "gauche -> droite selon ORDER_GROUPS, pas selon l'entrée"
    assert np.all((a > 0) & (a < 1))


def test_position_anchors_leve_sur_un_candidat_inconnu():
    with pytest.raises(ValueError, match="ORDER_GROUPS"):
        position_anchors(["Mélenchon", "Personne Inconnue"])


def test_les_ancres_de_rang_sont_deja_bien_separees():
    """Propriété qui justifie de s'en contenter : l'équirépartition en position
    donne des écarts qui croissent vers les extrêmes une fois en logit, donc
    `ancres_ecartees` n'a rien à corriger même sur un gros roster."""
    cands = [c for _, membres in ORDER_GROUPS for c in membres]
    a = position_anchors(cands)
    assert np.diff(np.sort(logit(a))).min() > 0.15
    assert np.allclose(ancres_ecartees(a, 0.15), a)
