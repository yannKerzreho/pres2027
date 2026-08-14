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
B = 25
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


def spatial_shares(mu, sigma, w, tested_mask) -> jnp.ndarray:
    """Coeur du modèle d'observation (spec §4) : softmax masqué sur la grille +
    agrégation par les poids démographiques W -> pi (P,N) ou (N,) selon la
    forme de tested_mask. `mu`/`sigma`/`w` : (N,) OU (S,N) pour S tirages
    postérieurs poussés en parallèle (cf. `pi_draws_for_mask`, spec §"Incertitude") ;
    dans ce dernier cas `tested_mask` doit être (N,) (un seul scénario) et la
    sortie est (S,N)."""
    D = -((V[None, :] - mu[..., None]) ** 2) / (2 * sigma[..., None] ** 2)   # (...,N,B)
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

