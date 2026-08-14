"""Le modèle NumPyro : inférence JOINTE de la géométrie et du chemin de `w`.

Trois paramètres par candidat (position, rayon, niveau), deux échelles globales,
aucun bloc. Voir `spec_spatial_pooling.md` pour la dérivation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from .geometry import spatial_shares

# --- Rayon d'action : prior d'ORIGINE, resserrement ANNULÉ (spec §12.13) --------
# Un resserrement à 0,06 avait été introduit au motif qu'un `sigma` large
# ferait « recruter Zemmour à l'extrême gauche ». MESURÉ : faux. Le noyau n'est
# pas le profil de recrutement -- la compétition du softmax écrase tout, et
# même au sigma le plus large 0,04 % seulement de l'électorat de Zemmour vient
# de la gauche (0,12 % en médiane sur 4000 tirages de `w`).
#
# `sigma` n'étant identifié par AUCUNE donnée (§12.12 : contraction ≈ 0, le
# postérieur suit le prior), il ne reste que deux critères mesurables. Les deux
# désignent la valeur d'origine :
#
#   prior sigma | redistribution | div. 2027 | div. 2022
#         0,06  |     0,893 pt   |     3     |  69-275
#         0,10  |     0,845 pt   |     0     |   14-18
#         0,15  |     0,845 pt   |     0     |    5-6
#
# Le prior serré dégrade la REDISTRIBUTION -- exactement ce que le modèle
# promet -- en plus de la stabilité. Annulé.
# --- Réglages retenus (spec §12.26-12.31) ----------------------------------------
SIGMA_PRIOR_MEDIAN = 0.15     # médiane du prior sur le rayon global
POS_SCALE = 0.4               # liberté de position autour de l'ancre, en logit
SIGMA_JITTER = 0.30           # largeur FIXE du bruit de rayon par candidat
MIN_GAP_ANCRES = 0.25         # écart minimal entre ancres voisines (identifiabilité)
LEVEL_SCALE = 1.5             # échelle du niveau par candidat
SEUIL_DYNAMIQUE = 0.05        # part moyenne au-dessus de laquelle `w` a un CHEMIN

def spatial_pooling_model_ou(tested_mask, Y, Np, date_idx, kl_basis, pos_anchor,
                             excess_var=None, tau_ou=None, sigma_w_prior=0.5,
                             level_scale=1.5, pos_scale=0.4,
                             sigma_log_mu=None, sigma_log_scale=0.6, sigma_jitter=0.30,
                             dynamic_mask=None):
    """Le modèle. Inférence JOINTE de la géométrie et du chemin de `w`.

    Trois paramètres par candidat, deux échelles globales, aucun bloc :

        mu_i    = sigmoid( logit(ancre_i) + z^mu_i · pos_scale )
        sigma_i = sigma_global · exp( z^sigma_i · sigma_jitter )
        w_i(t)  = m_i + sigma_w · (dérive OU tronquée en base de Karhunen-Loève)

    **Positions.** L'ancre vient de l'ordre politique connu (`POSITION_ANCHORS`,
    passée par l'appelant après `ancres_ecartees`). Ce qui a longtemps cassé
    l'échantillonnage n'était ni les blocs comme découpage ni la chaîne
    ordonnée, mais **plusieurs candidats partageant une position latente avec un
    écart individuel libre de signe** : le postérieur admettait alors leurs
    permutations, chaque chaîne en choisissait une, et R-hat mesurait ce
    désaccord (2022 J-210 : R-hat 2,019, ESS 2,7, 45 divergences ; avec une
    ancre par candidat, 1,011 / 117 / 1). Spec §12.25-12.26.

    L'ordre n'est PAS imposé : `pos_scale` laisse de quoi croiser deux voisins,
    et le postérieur retrouve pourtant l'ordre attendu. C'est voulu — le
    placement devient une prédiction vérifiable plutôt qu'un postulat, et c'est
    ainsi qu'on a corrigé l'ordre interne du groupe `MLP` (§12.27).

    **Rayon.** Un `sigma_global` échantillonné, plus un bruit blanc par candidat
    de largeur FIXE : le delta dit à quel point le candidat est un parti
    attrape-tout. La largeur est figée parce que la version d'origine
    l'échantillonnait (`sd_delta_sigma`) — une échelle libre multipliant un
    vecteur de dimension N, c'est-à-dire un entonnoir. Ne pas confondre avec
    FIXER sigma lui-même, mesuré désastreux (ESS 4,0, §12.26) : sigma n'est
    identifié par presque aucune donnée mais il ABSORBE le désajustement du
    modèle unidimensionnel, et le figer renvoie ce désajustement sur les
    positions et sur `w`, qui eux sont contraints.

    **Niveau et dérive séparés.** Sans `m_i`, `sigma_w` devait porter à la fois
    l'écart de niveau ENTRE candidats (écart-type 1,14 en log-odds sur 2027) et
    la dérive temporelle de chacun — il arbitrait à ~0,70, sous-dispersant les
    niveaux et sur-dispersant la dérive d'un facteur 2. Séparés, `sigma_w` tombe
    à ~0,31 et ne signifie plus qu'une chose (§12.23). C'est aussi ce qui
    restaure la structure de krigeage universel de §11, perdue au passage au
    modèle joint.

    **Noyau OU** plutôt que marche aléatoire : la variance de dérive sature à
    `sigma_w²(1-e^{-2h/tau})` au lieu de croître sans borne (§11.1 ; la marche
    sur-disperse d'un facteur 3 à 7 jours, d'où 93,7 % d'IC90). `tau_ou` est
    FIXÉ, repris de la banque commune : sur une fenêtre de campagne seul le
    rapport `sigma_w²/tau` est contraint (§11.4), les laisser libres tous deux
    revient à choisir un point d'une crête plate.

    `dynamic_mask` : seuls les candidats au-dessus du seuil reçoivent un CHEMIN ;
    les autres gardent un `w` statique, leur trajectoire n'étant pas
    identifiable sous ±1 pt de bruit (§12.19).
    """
    N = tested_mask.shape[1]
    if sigma_log_mu is None:
        sigma_log_mu = jnp.log(SIGMA_PRIOR_MEDIAN)
    if tau_ou is None:
        from model.core.opinion import load_law
        tau_ou = float(load_law()["tau"])

    sigma_global = numpyro.sample("sigma_global", dist.LogNormal(sigma_log_mu, sigma_log_scale))
    z_sigma = numpyro.sample("z_sigma", dist.Normal(0.0, 1.0).expand([N]))
    sigma = numpyro.deterministic("sigma", sigma_global * jnp.exp(z_sigma * sigma_jitter))

    raw_anchor = jnp.log(pos_anchor / (1.0 - pos_anchor))
    z_pos = numpyro.sample("z_pos", dist.Normal(0.0, 1.0).expand([N]))
    mu = numpyro.deterministic("mu", jax.nn.sigmoid(raw_anchor + z_pos * pos_scale))

    sigma_w = numpyro.sample("sigma_w", dist.HalfNormal(sigma_w_prior))
    z_m = numpyro.sample("z_m", dist.Normal(0.0, 1.0).expand([N]))
    m = numpyro.deterministic("w_level", z_m * level_scale)

    # Chemin dans la base de Karhunen-Loève tronquée du complément orthogonal de
    # la constante (`ou_kl_basis(center=True)`) : K composantes au lieu de M
    # valeurs par candidat. La troncature ne réduit PAS la profondeur d'arbre
    # (mesuré, §12.22 — c'était l'hypothèse et elle est fausse) ; elle réduit la
    # mémoire, et elle est la condition technique du niveau séparé, qu'on ne
    # peut retirer qu'en manipulant explicitement les composantes.
    dyn = np.ones(N, dtype=bool) if dynamic_mask is None else np.asarray(dynamic_mask, dtype=bool)
    idx = np.flatnonzero(dyn)
    z_w = numpyro.sample("z_w", dist.Normal(0.0, 1.0).expand([len(idx), kl_basis.shape[0]]))
    w_full = jnp.zeros((N, kl_basis.shape[1])) + m[:, None]
    w_full = w_full.at[idx].add(sigma_w * (z_w @ kl_basis))
    numpyro.deterministic("w_now", w_full[:, -1])           # dernière colonne = `as_of`

    pi = spatial_shares(mu, sigma, w_full[:, date_idx].T, tested_mask)
    var = pi * (1.0 - pi) / jnp.clip(Np[:, None], 1.0, None)
    if excess_var is not None:
        var = var + excess_var[:, None]
    numpyro.deterministic("pi", pi)
    numpyro.factor("ll", jnp.sum(dist.Normal(pi, jnp.sqrt(var + 1e-8)).log_prob(Y) * tested_mask))

