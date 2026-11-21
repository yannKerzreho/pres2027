"""Géométrie du modèle : la grille de quadrature, le placement des ancres, le
softmax masqué qui produit les parts, et la base de Karhunen-Loève du chemin
temporel. Aucune dépendance aux données ni à NumPyro — que des fonctions pures,
donc testables sans fit."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# --- Grille spatiale (spec §1, révisée §12.8) ------------------------------------
# `pi_i = ∫ softmax(A(v))_i f(v) dv` est une INTÉGRALE ; la grille en est la
# quadrature. Elle était échantillonnée uniformément (`linspace` + poids 1/B),
# ce qui converge en O(1/B) seulement : l'intégrande ne s'annule pas aux bords
# de [0,01 ; 0,99], donc les extrémités dominent l'erreur. Mesuré sur la
# géométrie réellement fittée, erreur maximale sur `pi` :
#
#     B        uniforme     Gauss-Legendre
#     25        1,204 pt        0,0015 pt
#     50        0,599 pt        0,0015 pt
#   1000        0,028 pt        0,0015 pt
#
# 0,6 pt à B=50 n'est pas négligeable : c'est PLUS que l'écart de modèle estimé
# (0,32 pt, §12.7) et une fraction notable du bruit d'échantillonnage (~1 pt).
# L'erreur est en outre maximale là où `sigma` est petit (3,9 nœuds par sigma
# pour le plus étroit), donc elle biaise justement les `sigma` les plus serrés.
#
# Gauss-Legendre intègre exactement les polynômes de degré 2B-1 : même
# intégrale, même électorat UNIFORME (les poids somment à 1), simplement des
# nœuds et poids choisis au lieu d'un pas constant.
#
# B=25 : mesuré sur un roster de 24 candidats avec des sigma jusqu'à 0,35,
# l'erreur maximale sur `pi` vaut 0,00000 pt dès B=20 (0,00008 pt à B=15).
# Le tenseur `P×N×B` domine le coût du gradient, donc passer de 50 à 25 divise
# par deux la vraisemblance sans aucune perte -- c'est la propriété de
# convergence spectrale de Gauss-Legendre sur un intégrande lisse.
import os as _os
# Surchargeable pour un backtest de sensibilité (spec_second_tour §8.48) ; production : 25.
B = int(_os.environ.get("PRES2027_B_QUAD", "25"))
_gl_x, _gl_w = np.polynomial.legendre.leggauss(B)
V = jnp.asarray(0.01 + (0.99 - 0.01) * (_gl_x + 1.0) / 2.0)
W = jnp.asarray(_gl_w / 2.0)

def ancres_ecartees(anchors: np.ndarray, min_gap: float = 0.25) -> np.ndarray:
    """Impose un écart MINIMAL entre ancres voisines, en logit (spec §12.31).

    Le placement expert traduit la proximité politique réelle, et certaines
    paires y sont presque confondues — Poisson/Ciotti à 0,061 en logit,
    Barnier/Bertrand à 0,090. C'est honnête, et c'est ingérable : avec
    `pos_scale = 0,4` chaque candidat bouge de ±0,4, soit 4 à 6 fois cet écart.
    Les deux redeviennent alors librement permutables et la multimodalité de
    §12.26 revient — mesuré, J-150 (qui cumule 11 couples sous 0,25) repasse de
    R-hat 1,055 à 1,371, ESS 37,7 à 6,4.

    L'écart minimal est donc une contrainte d'IDENTIFIABILITÉ, pas une
    affirmation politique : 0,25 en logit vaut ~1 point de part au milieu de
    l'axe, très en deçà de ce qu'un sondage distingue. On ne prétend pas savoir
    que Bertrand est à cette distance de Pécresse ; on refuse seulement de dire
    qu'ils sont au même endroit, ce que les données ne peuvent pas trancher.

    L'ordre est préservé (on ne fait que pousser vers la droite dans l'ordre
    croissant) et la moyenne est conservée, donc l'axe n'est pas décalé.
    """
    x = np.log(anchors / (1.0 - anchors))
    o = np.argsort(x)
    xs = x[o]
    # PROJECTION L2 sous la contrainte `x_{i+1} - x_i >= g`, pas une cascade.
    # Une simple poussée vers la droite (`x_i = max(x_i, x_{i-1} + g)`) propage :
    # dès qu'un couple touche le plancher, tout l'aval s'aligne dessus. Mesuré
    # sur le roster 2022 J-150, elle collait 17 couples sur 20 à la borne et
    # remplaçait le placement expert par une grille régulière -- en resserrant
    # justement les extrêmes, que le logit espaçait naturellement.
    #
    # La projection correcte s'obtient en substituant `y_i = x_i - i·g` : la
    # contrainte devient « y non décroissant », et le minimum de `sum (y-y0)²`
    # sous cette contrainte est l'isotonie (algorithme PAVA). Elle écarte les
    # voisins SYMÉTRIQUEMENT et ne touche pas aux couples déjà conformes.
    y = xs - np.arange(len(xs)) * min_gap
    val, poids = [], []
    for v in y:
        val.append(v); poids.append(1.0)
        while len(val) > 1 and val[-2] > val[-1]:
            v2, p2 = val.pop(), poids.pop()
            v1, p1 = val.pop(), poids.pop()
            val.append((v1 * p1 + v2 * p2) / (p1 + p2)); poids.append(p1 + p2)
    y_iso = np.repeat(val, [int(p) for p in poids])
    xs = y_iso + np.arange(len(xs)) * min_gap
    xs = xs - (xs.mean() - x[o].mean())
    out = np.empty_like(x)
    out[o] = xs
    return 1.0 / (1.0 + np.exp(-out))


def ancres_par_famille(candidates, order_groups, min_gap: float = 0.25,
                       etendue: float | None = None) -> np.ndarray:
    """Ancres CARDINALES par famille (spec_spatial_pooling §10, piste B révisée).

    Les familles d'`order_groups` sont des centres équidistants en logit sur
    l'étendue `etendue` (par défaut celle des ancres par rang, pour ne pas
    changer l'échelle de l'axe) ; l'écart minimal `min_gap` ne s'applique
    qu'ENTRE MEMBRES d'une même famille (alias), centrés sur leur famille.
    Deux familles voisines peuvent donc être plus proches ou plus loin que
    deux alias, ce que l'écart minimal global interdisait (22/24 écarts au
    plancher en 2022)."""
    fam = {c: (f, [m for m in ms if m in candidates].index(c))
           for f, (_, ms) in enumerate(order_groups) for c in ms if c in candidates}
    familles = sorted({f for f, _ in fam.values()})
    if etendue is None:
        from .roster import position_anchors
        r = ancres_ecartees(position_anchors(list(candidates), order_groups), min_gap)
        x = np.log(r / (1 - r)); etendue = float(x.max() - x.min())
    centres = {f: -etendue / 2 + etendue * k / max(len(familles) - 1, 1) for k, f in enumerate(familles)}
    taille = {f: sum(1 for g, _ in fam.values() if g == f) for f in familles}
    x = np.array([centres[fam[c][0]] + min_gap * (fam[c][1] - (taille[fam[c][0]] - 1) / 2) for c in candidates])
    return 1.0 / (1.0 + np.exp(-x))


def _noyau(mu, sigma, nu=None, grille=None) -> jnp.ndarray:
    """Log-noyau NON normalisé (...,N,B). `nu` None : gaussien ; sinon Student-t de
    degré `nu` (scalaire ou (S,), spec §8.51), qui tend vers le gaussien quand nu → ∞."""
    g = V if grille is None else grille
    z2 = (g - mu[..., None]) ** 2 / sigma[..., None] ** 2
    if nu is None:
        return -z2 / 2
    n = jnp.asarray(nu)
    n = n.reshape(n.shape + (1, 1)) if n.ndim else n
    return -(n + 1) / 2 * jnp.log1p(z2 / n)


def spatial_shares(mu, sigma, w, tested_mask, nu=None) -> jnp.ndarray:
    """Coeur du modèle d'observation (spec §4) : softmax masqué sur la grille +
    agrégation par les poids démographiques W -> pi (P,N) ou (N,) selon la
    forme de tested_mask. `mu`/`sigma`/`w` : (N,) OU (S,N) pour S tirages
    postérieurs poussés en parallèle (cf. `pi_draws_for_mask`, spec §"Incertitude") ;
    dans ce dernier cas `tested_mask` doit être (N,) (un seul scénario) et la
    sortie est (S,N)."""
    D = _noyau(mu, sigma, nu)                                              # (...,N,B)
    # Noyau NORMALISÉ sur la grille (spec §12.10) : on retranche
    # `log(Σ_b W_b e^{D_b})` pour que le profil d'attractivité de chaque
    # candidat intègre à 1, quels que soient `mu` et `sigma`.
    #
    # Deux effets. (a) `w` devient comparable entre candidats : sans ça, un
    # candidat proche d'un bord perd la part de son noyau tombant hors de
    # [0,1] et doit compenser par un `w` plus grand, que le prior N(0,1.5)
    # pénalise -- il repoussait donc artificiellement les candidats loin des
    # bords. (b) Surtout, ça SUPPRIME la crête `w ↔ log σ` de §12.9 : la part
    # d'un candidat non dominant valait `∝ exp(w)·σ·κ(mu,σ)`, donc seul
    # `w + log σ` était identifié ; elle vaut désormais `∝ exp(w)` seul, et
    # `σ` n'est plus contraint que par la FORME, c'est-à-dire par la
    # redistribution observée sur les hypothèses imbriquées.
    #
    # C'est une reparamétrisation exacte (une constante par candidat, avant un
    # softmax) : la famille de vraisemblance est inchangée, et toute la
    # mécanique de §11 reste valable (`Φ` garde le même hessien).
    A = w[..., None] + D - jax.nn.logsumexp(D, axis=-1, b=W, keepdims=True)
    A = A - jnp.max(A, axis=-2, keepdims=True)
    expA = jnp.exp(A)
    numerator = expA * tested_mask[..., :, None]
    denom = jnp.clip(numerator.sum(axis=-2, keepdims=True), 1e-12, None)
    return jnp.sum((numerator / denom) * W, axis=-1)


def ou_kl_basis(unique_dates, as_of, tau_ou, var_cible=0.99, k_max=15, center=False):
    """Base de Karhunen-Loève du chemin OU, calculée UNE FOIS hors échantillonnage.

    `tau_ou` étant fixé (banque commune, §11.4), la covariance
    `exp(-|t-t'|/tau)` sur la grille des dates est connue avant tout tirage :
    on peut donc la diagonaliser et ne garder que les composantes qui portent
    la variance.

    Pourquoi c'est nécessaire et pas seulement économique. Avec `tau ≈ 262 j` et
    des sondages espacés de 2-3 jours, `rho = exp(-dt/tau) ≈ 0,99` : le
    processus est presque une constante sur une fenêtre de campagne. Mesuré, la
    première composante porte 76 à 89 % de la variance, et 3 à 4 composantes en
    portent 95 %. Échantillonner `M` valeurs indépendantes par candidat (44 sur
    2022 J-150) revient donc à créer des dizaines de directions que la
    vraisemblance ne contraint pas — directions plates que NUTS doit parcourir,
    d'où une profondeur d'arbre bloquée à 255 pas et des chaînes qui errent.

    `center=True` retire d'abord la direction CONSTANTE (projection orthogonale
    sur le complément de `1`) : la base ne porte alors plus que la DÉRIVE autour
    de la moyenne temporelle, le niveau étant confié à un paramètre séparé
    (`level_scale` de `spatial_pooling_model_ou`, §12.23). Sans ça, la première
    composante EST le niveau — mesuré `|<v_0, 1/sqrt(M)>| = 0,999` sur la grille
    2027 — et l'échelle `sigma_w` se retrouve à porter deux quantités sans
    rapport : l'écart de niveau ENTRE candidats (écart-type 1,14 en log-odds sur
    2027) et la dérive temporelle de chacun.

    Renvoie `(base (K, M+1), k)` : la dernière colonne est `as_of`, incluse dans
    la grille pour que l'extrapolation sorte de la même base plutôt que d'un
    terme ajouté à part.
    """
    t = np.concatenate([np.asarray(unique_dates, dtype=float), [float(as_of)]])
    C = np.exp(-np.abs(t[:, None] - t[None, :]) / tau_ou)
    if center:
        Pp = np.eye(len(t)) - np.ones((len(t), len(t))) / len(t)
        C = Pp @ C @ Pp
    val, vec = np.linalg.eigh(C)
    val, vec = val[::-1], vec[:, ::-1]
    val = np.clip(val, 0.0, None)
    k = int(np.searchsorted(np.cumsum(val) / val.sum(), var_cible)) + 1
    k = max(3, min(k, k_max, len(t)))
    return (vec[:, :k] * np.sqrt(val[:k])).T, k



# --- Second tour (spec_second_tour.md §2) ----------------------------------------
def log_attractivite(mu, sigma, w, nu=None) -> jnp.ndarray:
    """`A_i(v) = w_i + log phi_i(v)`, noyau normalisé sur la grille (même
    convention que `spatial_shares`). `mu`/`sigma`/`w` : (...,N) -> (...,N,B).
    `nu` : degré du noyau de Student-t (§8.51), None = gaussien."""
    D = _noyau(mu, sigma, nu)
    return w[..., None] + D - jax.nn.logsumexp(D, axis=-1, b=W, keepdims=True)


def duel_shares(A_a, A_b, kappa, eta_a, eta_b, lse=None, omega=None) -> jnp.ndarray:
    """Part de `a` parmi les exprimés du duel (a, b), spec §2.1 :

        g_ab = sum_b W_b · logistique( kappa·[A_a(V_b) - A_b(V_b)] )   (géométrie)
        s_ab = logistique( logit(g_ab) + eta_a - eta_b )                (valence)

    La valence s'applique à l'AGRÉGAT, pas au votant : au niveau du votant, `q`
    est une quasi-marche en v (pente kappa·(mu_a-mu_b)/sigma²) et eta ne fait
    que déplacer le point de bascule -- son effet en points de duel dépendait
    alors de sigma (×7 entre sigma 0,08 et 0,30), qui est mal identifié. Ici
    eta = 0,5 vaut ~12 pt autour de 50/50, quel que soit sigma.

    `A_a`, `A_b` : (...,B) ; `kappa`, `eta_a`, `eta_b` : diffusables sur `...`.
    Invariant par translation commune de `w` et de `eta`.

    Abstention/blanc/nul (spec §3.1), si `lse` et `omega` sont donnés : un
    « candidat » de non-expression d'utilité `omega + kappa·lse(v)`, où
    `lse(v) = log sum_{j in roster} exp A_j(v)` est la valeur inclusive du
    votant (son attachement à TOUTE l'offre). Le votant dont le favori est
    finaliste exprime ; celui dont tout l'entourage est éliminé s'abstient
    davantage. Parts parmi exprimés : g = ∫ rho·q / ∫ rho. Invariant par
    translation de `w` (contrairement au `max` non posé de la version Codex)."""
    k = jnp.asarray(kappa)[..., None]
    q = jax.nn.sigmoid(k * (A_a - A_b))
    if lse is None:
        g = jnp.sum(q * W, axis=-1)
    else:
        rho = duel_expression(A_a, A_b, kappa, lse, omega)
        g = jnp.sum(rho * q * W, axis=-1) / jnp.clip(jnp.sum(rho * W, axis=-1), 1e-12, None)
    g = jnp.clip(g, 1e-6, 1.0 - 1e-6)
    return jax.nn.sigmoid(jnp.log(g) - jnp.log1p(-g) + jnp.asarray(eta_a) - jnp.asarray(eta_b))


def duel_expression(A_a, A_b, kappa, lse, omega, omega_g=None) -> jnp.ndarray:
    """rho(v) : probabilité d'exprimer un vote pour a ou b (...,B).
    `1 - ∫rho` = non-expression parmi les exprimés du premier tour.
    `omega_g` (..., B) : terme situé de l'option extérieure (abstention située, §8.48 bis)."""
    k = jnp.asarray(kappa)[..., None]
    ins = jnp.logaddexp(k * A_a, k * A_b)
    om = jnp.asarray(omega)[..., None]
    if omega_g is not None:
        om = om + omega_g
    return jax.nn.sigmoid(ins - (om + k * lse))


# --- Carte 2D (spec_second_tour §8.44, piste R2 ; drapeau `carte_2d`) ------------
# Second axe u (système / anti-système), quadrature produit Gauss-Legendre
# B × B_U, noyau SÉPARABLE N(v; mu, sigma)·N(u; nu, lambda) normalisé sur la
# grille. Sur une grille aplatie de longueur B·B_U ; avec lambda → ∞ on retrouve
# exactement le 1D (le facteur en u devient uniforme et se normalise).
B_U = 7
_glu_x, _glu_w = np.polynomial.legendre.leggauss(B_U)
U_1D = 0.01 + (0.99 - 0.01) * (_glu_x + 1.0) / 2.0
WU_1D = _glu_w / 2.0
V2 = jnp.asarray(np.repeat(np.asarray(V), B_U))          # (B·B_U,), v lent, u rapide
U2 = jnp.asarray(np.tile(U_1D, B))
W2 = jnp.asarray(np.outer(np.asarray(W), WU_1D).ravel())


def log_attractivite_2d(mu, sigma, nu, lam, w) -> jnp.ndarray:
    """`A_i(v,u) = w_i + log phi_i(v,u)` sur la grille 2D aplatie (...,N,B·B_U).
    `lam` : scalaire ou (...,) diffusable (λ commun aux candidats)."""
    lam = jnp.asarray(lam)
    lam = lam[..., None, None] if lam.ndim else lam
    D = (-((V2 - mu[..., None]) ** 2) / (2 * sigma[..., None] ** 2)
         - ((U2 - nu[..., None]) ** 2) / (2 * lam ** 2))
    return w[..., None] + D - jax.nn.logsumexp(D, axis=-1, b=W2, keepdims=True)


def spatial_shares_2d(mu, sigma, nu, lam, w, tested_mask) -> jnp.ndarray:
    """Analogue 2D de `spatial_shares` (mêmes formes d'entrée et de sortie)."""
    A = log_attractivite_2d(mu, sigma, nu, lam, w)
    A = A - jnp.max(A, axis=-2, keepdims=True)
    expA = jnp.exp(A)
    numerator = expA * tested_mask[..., :, None]
    denom = jnp.clip(numerator.sum(axis=-2, keepdims=True), 1e-12, None)
    return jnp.sum((numerator / denom) * W2, axis=-1)


def duel_shares_poids(A_a, A_b, kappa, eta_a, eta_b, lse, omega, poids, omega_g=None) -> jnp.ndarray:
    """`duel_shares` avec des poids de quadrature quelconques (grille 2D)."""
    k = jnp.asarray(kappa)[..., None]
    q = jax.nn.sigmoid(k * (A_a - A_b))
    if lse is None:
        g = jnp.sum(q * poids, axis=-1)
    else:
        rho = duel_expression(A_a, A_b, kappa, lse, omega, omega_g)
        g = jnp.sum(rho * q * poids, axis=-1) / jnp.clip(jnp.sum(rho * poids, axis=-1), 1e-12, None)
    g = jnp.clip(g, 1e-6, 1.0 - 1e-6)
    return jax.nn.sigmoid(jnp.log(g) - jnp.log1p(-g) + jnp.asarray(eta_a) - jnp.asarray(eta_b))


def reports_cellules(A, champ_mask, ia, ib, k_idx, kappa, eta_diff, lse, omega, s_ab, poids, n_newton=4,
                     omega_g=None, fidelite=False):
    """Cellules (a, b, aucun) de l'électorat T1 `k_idx` (spec §8.38, §8.44.4),
    différentiables : δ par `n_newton` pas de Newton depuis `eta_diff` pour que
    l'agrégat des électorats du champ redonne `s_ab`.

    A : (R, N, G) ; champ_mask : (N,) ou (R, N) 0/1 (champ par ligne, §8.50) ; ia, ib, k_idx : (R,) entiers ;
    kappa, omega : scalaires ; eta_diff, s_ab : (R,) ; lse : (R, G) ou None."""
    rows = jnp.arange(A.shape[0])
    cm = jnp.asarray(champ_mask)
    cm = cm if cm.ndim == 2 else cm[None, :]
    Am = A + jnp.log(jnp.clip(cm, 1e-30, None))[:, :, None]
    p = jnp.exp(Am - jax.nn.logsumexp(Am, axis=1, keepdims=True))         # (R,N,G)
    A_a, A_b = A[rows, ia], A[rows, ib]
    kq = jnp.asarray(kappa)
    kq = kq[:, None] if kq.ndim else kq                     # scalaire (modèle) ou (R,) (tirages)
    q = jax.nn.sigmoid(kq * (A_a - A_b))
    rho = jnp.ones_like(q) if lse is None else duel_expression(A_a, A_b, kappa, lse, omega, omega_g)
    E = jnp.einsum("rng,g->rn", p, poids)
    pr = jnp.einsum("rng,rg,g->rn", p, rho, poids)
    Rk = pr / jnp.clip(E, 1e-12, None)
    G = jnp.clip(jnp.einsum("rng,rg,rg,g->rn", p, rho, q, poids) / jnp.clip(pr, 1e-12, None), 1e-6, 1 - 1e-6)
    lG = jnp.log(G) - jnp.log1p(-G)
    wgt = E * Rk * cm
    wgt = wgt / jnp.clip(wgt.sum(1, keepdims=True), 1e-12, None)
    d = jnp.asarray(eta_diff)
    for _ in range(n_newton):
        sg = jax.nn.sigmoid(lG + d[:, None])
        f = (wgt * sg).sum(1) - s_ab
        fp = (wgt * sg * (1 - sg)).sum(1)
        d = d - f / jnp.clip(fp, 1e-6, None)
    Gt = jax.nn.sigmoid(lG[rows, k_idx] + d)
    R = Rk[rows, k_idx]
    cel = jnp.stack([R * Gt, R * (1 - Gt), 1 - R], axis=-1)
    if fidelite and lse is not None:
        # §8.56 (R7, révisé 21/11) : électorat d'un finaliste. Choc d'utilité T1 persistant
        # (Gumbel) : U_k = lse + G, G ~ Gumbel(0,1), et k domine aussi sur {a, b} ; l'électeur
        # s'exprime avec ρ = E_G σ(κG − ω) (quadrature à 32 quantiles fixes), et vote k sauf une
        # défection fixe ε = 2 % (défection publiée typique ; figé avant le fit).
        om = jnp.asarray(omega)
        om = om if om.ndim else jnp.full(cel.shape[0], om)
        kq_ = jnp.asarray(kappa)
        kq_ = kq_ if kq_.ndim else jnp.full(cel.shape[0], kq_)
        u = (jnp.arange(32) + 0.5) / 32
        g = -jnp.log(-jnp.log(u))                                         # quantiles Gumbel(0,1)
        rho_f = jax.nn.sigmoid(kq_[:, None] * g[None, :] - om[:, None]).mean(axis=1)
        e = 0.02
        s_ = rho_f * (1 - e)
        e = rho_f * e
        fid_a = jnp.stack([s_, e, 1 - rho_f], axis=-1)
        fid_b = jnp.stack([e, s_, 1 - rho_f], axis=-1)
        ka, kb = (jnp.asarray(k_idx) == jnp.asarray(ia))[:, None], (jnp.asarray(k_idx) == jnp.asarray(ib))[:, None]
        cel = jnp.where(ka, fid_a, jnp.where(kb, fid_b, cel))
        cel = cel / cel.sum(-1, keepdims=True)
    return cel, d


# --- Votants sur un cercle (spec_second_tour §8.46, piste R3 ; drapeau `carte_cercle`) -----
# Les positions 1D mu ∈ (0,1) sont posées sur un ARC AVANT de longueur `arc` (rad), centré
# en θ = 0 ; le reste du cercle (arc arrière, 2π − arc) porte aussi des votants, voisins
# des deux extrêmes. Noyau de von Mises, κ_i = 1/(arc·σ_i)², normalisé sur la quadrature
# (trapèzes, exacte pour les fonctions périodiques lisses).
N_THETA = 128                  # §8.46.4 révisé : 64 nœuds donnent 0,5-0,9 pt d'erreur à σ = 0,02
THETA = jnp.asarray(2 * np.pi * np.arange(N_THETA) / N_THETA - np.pi)
W_THETA = jnp.full(N_THETA, 1.0 / N_THETA)


def log_attractivite_cercle(mu, sigma, arc, w, theta=None, poids=None) -> jnp.ndarray:
    """`A_i(θ) = w_i + κ_i cos(θ − θ_i) − log Σ W κ…`, θ_i = arc·(mu_i − ½). (...,N,N_θ)."""
    th = THETA if theta is None else theta
    pw = W_THETA if poids is None else poids
    arc = jnp.asarray(arc)
    arc_ = arc[..., None] if arc.ndim else arc
    t_i = arc_ * (mu - 0.5)
    k_i = 1.0 / (arc_ * sigma) ** 2
    D = k_i[..., None] * (jnp.cos(th - t_i[..., None]) - 1.0)       # −1 : stabilité (κ grand)
    return w[..., None] + D - jax.nn.logsumexp(D, axis=-1, b=pw, keepdims=True)


def spatial_shares_cercle(mu, sigma, arc, w, tested_mask, theta=None, poids=None) -> jnp.ndarray:
    """Analogue circulaire de `spatial_shares`."""
    pw = W_THETA if poids is None else poids
    A = log_attractivite_cercle(mu, sigma, arc, w, theta, poids)
    A = A - jnp.max(A, axis=-2, keepdims=True)
    expA = jnp.exp(A)
    numerator = expA * tested_mask[..., :, None]
    denom = jnp.clip(numerator.sum(axis=-2, keepdims=True), 1e-12, None)
    return jnp.sum((numerator / denom) * pw, axis=-1)
