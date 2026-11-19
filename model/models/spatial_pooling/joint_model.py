"""Le modèle NumPyro : inférence JOINTE de la géométrie et du chemin de `w`.

Trois paramètres par candidat (position, rayon, niveau), deux échelles globales,
aucun bloc. Voir `spec_spatial_pooling.md` pour la dérivation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from .geometry import (W, W2, duel_shares, duel_shares_poids, log_attractivite, log_attractivite_2d,
                       reports_cellules, spatial_shares, spatial_shares_2d)

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
# Largeur FIXE du bruit de rayon par candidat. 0,30 jusqu'au 2026-09-29 :
# ramené à 0,20 pour lever une BIMODALITÉ. Depuis les sondages de septembre, un
# second mode -- Mélenchon étroit (sigma 0,06) décalé à droite de Roussel (mu
# 0,09 -> 0,22), Glucksmann élargi (sigma 0,10 -> 0,23) -- avait la MÊME
# énergie que le mode principal : +4,9 nats de vraisemblance, gain diffus sur
# des dizaines de nœuds, contre -5 nats de prior. Le runner Linux y envoyait une
# chaîne sur quatre (R-hat 1,7 à 2,5, publication refusée), le Mac jamais. Le
# mode B demande des rayons à ±2,2 écarts-types du jitter ; à 0,20 il en coûte
# ~6 nats de plus. Mesuré : 0 graine sur 3 dans le mode B (3 sur 4 à 0,30), et
# un coût invisible au backtest 2026 hors échantillon (`notebooks/04l`) --
#   jitter | couverture 50/80/90 | redistribution | moitié difficile
#    0,30  |  55,1 / 81,5 / 90,4 |    0,612 pt    |    0,652 pt
#    0,25  |  54,7 / 81,1 / 90,0 |    0,620 pt    |    0,661 pt
#    0,20  |  54,7 / 80,7 / 89,2 |    0,631 pt    |    0,678 pt
# (proportionnel 1,268 pt, 80 % des 51 paires gagnées dans les trois cas). À ne
# pas confondre avec le resserrement de SIGMA_PRIOR_MEDIAN annulé plus haut :
# celui-là déplaçait TOUS les rayons, celui-ci borne leur dispersion.
SIGMA_JITTER = 0.20
MIN_GAP_ANCRES = 0.25         # écart minimal entre ancres voisines (identifiabilité)
LEVEL_SCALE = 1.5             # échelle du niveau par candidat
SEUIL_DYNAMIQUE = 0.05        # part moyenne au-dessus de laquelle `w` a un CHEMIN
# --- Second tour (spec_second_tour.md §2.1) ---------------------------------------
# Valence de 2nd tour par candidat, échelle FIXE (pas d'échelle échantillonnée :
# avec ~8 candidats en duel, une échelle libre ferait un entonnoir, cf. le même
# choix pour SIGMA_JITTER). Appliquée à l'agrégat (`duel_shares`) : 0,5 en logit
# ~ 12 pt de duel autour de 50/50, indépendamment des rayons.
ETA_SCALE = 0.5
KAPPA_LOG_SCALE = 0.5         # log kappa ~ N(0, 0,5) : centré sur « même agrégation qu'au T1 »
SIGMA_T2_PRIOR = 0.02         # écart-type d'excès des duels (part), demi-normale
# Abstention/blanc/nul au 2nd tour (spec §3.1, idée utilisateur 2026-11-14).
# omega fixe le niveau de non-expression : -3,7 ~ 20 % sur la géométrie 2027.
OMEGA_PRIOR = (-3.0, 1.5)

def spatial_pooling_model_ou(tested_mask, Y, Np, date_idx, kl_basis, pos_anchor,
                             excess_var=None, tau_ou=None, sigma_w_prior=0.5,
                             level_scale=1.5, pos_scale=0.4,
                             sigma_log_mu=None, sigma_log_scale=0.6, sigma_jitter=SIGMA_JITTER,
                             dynamic_mask=None, duel_a=None, duel_b=None, duel_y=None,
                             duel_n=None, duel_date_idx=None, duel_maison=None, n_maisons=None,
                             duel_blocs=None, duel_K_blocs=None, eta_scale=ETA_SCALE,
                             kappa_log_scale=KAPPA_LOG_SCALE, sigma_t2_prior=SIGMA_T2_PRIOR,
                             abstention=True, omega_prior=OMEGA_PRIOR, eta_hier=False,
                             masque_lse=None, sigma_w_par_candidat=False,
                             carte_2d=False, nu_ancre=None, nu_ordre=None, nu_ancre_val=0.3,
                             rep_y=None, rep_date_idx=None, rep_k=None, rep_a=None, rep_b=None,
                             rep_champ=None):
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

    if carte_2d:
        # Carte 2D (spec_second_tour §8.44) : nu_i = logistique(zeta_i) ; ancre
        # dure nu_C = nu_ancre_val (translation), ordre nu_RN ≥ nu_C (réflexion).
        zeta = jnp.asarray(numpyro.sample("zeta", dist.Normal(0.0, 1.0).expand([N])))   # init_to_value : numpy
        z_c = jnp.log(nu_ancre_val / (1.0 - nu_ancre_val))
        if nu_ancre is not None:
            zeta = zeta.at[nu_ancre].set(z_c)
        if nu_ordre is not None:
            xi = numpyro.sample("xi", dist.Normal(0.0, 1.0))
            zeta = zeta.at[nu_ordre].set(z_c + jax.nn.softplus(xi))
        nu = numpyro.deterministic("nu", jax.nn.sigmoid(zeta))
        lam = numpyro.deterministic("lambda_u", jnp.exp(numpyro.sample(
            "log_lambda", dist.Normal(jnp.log(0.2), 0.5))))
        _A = lambda m_, s_, w_: log_attractivite_2d(m_, s_, nu, lam, w_)   # noqa: E731
        poids = W2
    else:
        _A = log_attractivite
        poids = None

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
    if sigma_w_par_candidat:
        # σ_w hiérarchique par candidat (spec_spatial_pooling §10, piste C) :
        # σ_w,i = σ_w · exp(s_σ z_i), z_i ~ N(0,1), s_σ ~ HalfNormal(0,5).
        s_sig = numpyro.sample("s_sigma_w", dist.HalfNormal(0.5))
        z_sig = numpyro.sample("z_sigma_w", dist.Normal(0.0, 1.0).expand([len(idx)]))
        sw_i = sigma_w * jnp.exp(s_sig * z_sig)
        numpyro.deterministic("sigma_w_i", jnp.full(N, sigma_w).at[idx].set(sw_i))
        w_full = w_full.at[idx].add(sw_i[:, None] * (z_w @ kl_basis))
    else:
        w_full = w_full.at[idx].add(sigma_w * (z_w @ kl_basis))
    numpyro.deterministic("w_now", w_full[:, -1])           # dernière colonne = `as_of`
    numpyro.deterministic("w_path", w_full)                 # tracker (spec_spatial_pooling §10)

    pi = (spatial_shares_2d(mu, sigma, nu, lam, w_full[:, date_idx].T, tested_mask) if carte_2d
          else spatial_shares(mu, sigma, w_full[:, date_idx].T, tested_mask))
    var = pi * (1.0 - pi) / jnp.clip(Np[:, None], 1.0, None)
    if excess_var is not None:
        var = var + excess_var[:, None]
    numpyro.deterministic("pi", pi)
    numpyro.factor("ll", jnp.sum(dist.Normal(pi, jnp.sqrt(var + 1e-8)).log_prob(Y) * tested_mask))

    # --- Duels de second tour (spec_second_tour.md §2 et §4) ---------------------
    # UNE observation par duel (la part du premier nommé) : les deux parts sont
    # parfaitement anticorrélées, les mettre toutes deux dans la gaussienne
    # compterait chaque duel deux fois. Bloc absent => modèle T1 inchangé.
    if duel_y is not None:
        kappa = numpyro.sample("kappa", dist.LogNormal(0.0, kappa_log_scale))
        z_eta = numpyro.sample("z_eta", dist.Normal(0.0, 1.0).expand([N]))
        if eta_hier:
            # Échelle des valences APPRISE (spec_second_tour.md §8.10).
            s_eta = numpyro.sample("s_eta", dist.HalfNormal(eta_scale))
            eta = numpyro.deterministic("eta", z_eta * s_eta)
        else:
            eta = numpyro.deterministic("eta", z_eta * eta_scale)
        sigma_t2 = numpyro.sample("sigma_t2", dist.HalfNormal(sigma_t2_prior))
        w_d = w_full[:, duel_date_idx].T                         # (D, N)
        A_all = _A(mu, sigma, w_d)                               # (D, N, B) ou (D, N, B·B_U)
        rows = jnp.arange(w_d.shape[0])
        A_a, A_b = A_all[rows, duel_a], A_all[rows, duel_b]
        if abstention:
            omega = numpyro.sample("omega", dist.Normal(*omega_prior))
            # valeur inclusive (D, B) ; `masque_lse` : sur le CHAMP seulement (§8.27)
            lse = jax.nn.logsumexp(A_all if masque_lse is None
                                   else A_all + jnp.log(masque_lse)[None, :, None], axis=1)
            s = (duel_shares_poids(A_a, A_b, kappa, eta[duel_a], eta[duel_b], lse, omega, poids) if carte_2d
                 else duel_shares(A_a, A_b, kappa, eta[duel_a], eta[duel_b], lse, omega))
        else:
            omega = None
            s = (duel_shares_poids(A_a, A_b, kappa, eta[duel_a], eta[duel_b], None, None, poids) if carte_2d
                 else duel_shares(A_a, A_b, kappa, eta[duel_a], eta[duel_b]))
        # Part prédite par duel observé, à SA date : sert aux résidus (diagnostic
        # 2027, résidus standardisés) ; n'entre dans aucun calcul d'inférence.
        numpyro.deterministic("duel_s", s)
        var_d = s * (1.0 - s) / jnp.clip(duel_n, 1.0, None) + sigma_t2 ** 2
        if duel_maison is not None:
            # Effets maison et notice par candidat (spec §8.23 bis, forme par blocs) :
            # maison explicite (non centrée, institut × candidat) ; notice
            # marginalisée bloc par bloc (les duels d'une même notice).
            s_n = numpyro.sample("sigma_notice_t2", dist.HalfNormal(0.01))
            r = duel_y - s
            if n_maisons:                         # 0 = effet notice seul (§8.23 bis, suite)
                s_h = numpyro.sample("sigma_maison_t2", dist.HalfNormal(0.01))
                z_h = numpyro.sample("z_maison_t2", dist.Normal(0.0, 1.0).expand([n_maisons, N]))
                r = r - s_h * (z_h[duel_maison, duel_a] - z_h[duel_maison, duel_b])
            m = duel_blocs >= 0
            ix = jnp.where(m, duel_blocs, 0)
            R = jnp.where(m, r[ix], 0.0)
            V = jnp.where(m, var_d[ix] + 1e-8, 1.0)
            cov = jax.vmap(jnp.diag)(V) + s_n ** 2 * duel_K_blocs
            numpyro.factor("ll_t2", jnp.sum(dist.MultivariateNormal(
                jnp.zeros(R.shape[1]), covariance_matrix=cov).log_prob(R)))
        else:
            numpyro.factor("ll_t2", jnp.sum(dist.Normal(s, jnp.sqrt(var_d + 1e-8)).log_prob(duel_y)))
        if rep_y is not None:
            # Reports T1→T2 comme observations (spec_second_tour §8.44.4) :
            # Dirichlet(c·θ_k(t)) par (notice, électorat), c appris.
            w_r = w_full[:, rep_date_idx].T                       # (R, N)
            A_r = _A(mu, sigma, w_r)
            rr = jnp.arange(w_r.shape[0])
            pq = poids if carte_2d else W
            lse_r = jax.nn.logsumexp(A_r, axis=1) if abstention else None
            s_r = duel_shares_poids(A_r[rr, rep_a], A_r[rr, rep_b], kappa, eta[rep_a], eta[rep_b],
                                    lse_r, omega, pq)
            theta, _ = reports_cellules(A_r, rep_champ, rep_a, rep_b, rep_k, kappa,
                                        eta[rep_a] - eta[rep_b], lse_r, omega, s_r, pq)
            c = numpyro.deterministic("c_reports", jnp.exp(numpyro.sample(
                "log_c", dist.Normal(jnp.log(100.0), 1.0))))
            numpyro.deterministic("reports_theta", theta)
            numpyro.factor("ll_reports", jnp.sum(dist.Dirichlet(c * jnp.clip(theta, 1e-4, None)).log_prob(rep_y)))
