"""Modèle « spatial-pooling » (Hotelling-Downs discrétisé) — cf. spec_spatial_pooling.md
pour la dérivation complète et les résultats de validation (synthétique,
backtest half_life 2017/2022, sanity check 2027).

Portage des prototypes `notebooks/03_spatial_prototype.py` à
`04c_spatial_sanity_check.py` — même maths, structuré en module réutilisable.
PAS ENCORE un `ForecastModel` enregistré (`@register`) : décision explicite de
l'utilisateur (le contrat de sortie du site est encore en réflexion, cf.
spec §7/9). Ce module expose les fonctions nécessaires pour qu'un futur
`ForecastModel` (ou un endpoint de scénario interactif) les compose, sans rien
imposer sur la forme du contrat.

Différence clé avec `bayesian_nowcast`/`linear_pooling` : pas de vocabulaire
de "slots" figé (`SLOTS`, `model/core/live_dataset.py`) — chaque candidat réel
(Le Pen ET Bardella, Attal ET Philippe) est un paramètre séparé, tous deux
exploitables simultanément (c'est le point du modèle, cf. spec §0).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.diagnostics import summary as numpyro_summary

from model.core.bank import Bank
from model.core.inference import run_numpyro_mcmc
from model.core.live_dataset import load_raw_polls
from model.core.simulate import forecast_from_draws

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

# --- Roster / fenêtre de campagne (spec §6) --------------------------------------
MIN_POLL_DATE = pd.Timestamp("2026-01-01")   # même repli que bayesian_nowcast.nowcast
MIN_POLLS = 5                                 # sondages RÉELS distincts (`notice`), pas hypothèses
# Un candidat cesse d'être modélisé s'il n'a plus été testé depuis assez
# longtemps (retrait, hypothèse abandonnée par les instituts). NON CALIBRÉ :
# aucun candidat éligible n'en est proche aujourd'hui (le plus ancien est
# Bardella à 16 jours), c'est un garde-fou pour la suite, pas un réglage.
MAX_LAST_POLL_AGE_DAYS = 90
# Pas de la ligne du temps du chemin de `w` (modèle joint). Testé à 7 jours
# pour réduire la dimension : DÉGRADE la convergence (J-150 passe de R-hat
# 1,021 à 1,330). Regrouper force plusieurs sondages à se réconcilier sur une
# même valeur de `w`, ce qui durcit la géométrie -- l'inverse du but recherché.
# Laissé à 1 jour (spec §12.18).
TIME_BIN_DAYS = 1.0

# Ordre gauche->droite fourni à la main (session du 2026-08-10, corrigé le
# 2026-08-10 -- retour utilisateur) : un groupe = un point de la séquence,
# plusieurs candidats peuvent le partager (alternates, ou familles politiques
# proches) SANS ordre imposé entre eux -- cf. spec §2.1.
#
# Corrections apportées : (1) le candidat 2027 du groupe le plus à gauche est
# Arthaud, qui représente LO (Lutte Ouvrière), PAS le NPA (Poutou, NPA, ne
# passe pas le filtre >=5 sondages -- listé ici en repli inerte, tant qu'il
# n'atteint pas le seuil ça ne change rien) ; (2) "ecolo_ps_coco" scindé en
# deux points ordonnés, Coco/Écolo (PCF/EELV) < PS (Glucksmann/Hollande/Faure) ;
# (3) Bardella RÉINTÉGRÉ au groupe MLP (session du 2026-08-12, cf. spec §12).
# Il en avait été retiré au motif que Le Pen est la candidate OFFICIELLE du RN
# -- mais c'est un choix de SCÉNARIO, et ce modèle choisit ses scénarios à la
# LECTURE (`pi_draws_for_mask`, §8), pas au fit. L'appliquer au roster créait
# un trou de normalisation : Bardella est testé dans 46 nœuds sur 78, donc
# ~35% du bulletin disparaissait du modèle sans disparaître des données, et la
# somme des intentions par nœud tombait à 0,66 en médiane face à un `pi` qui
# somme à 1 par construction. Mesuré : 0,660 -> 1,000 en le réintégrant.
# Pour n'afficher que Le Pen, masquer Bardella À LA LECTURE.
#
# ORDER_GROUPS est un CATALOGUE, pas le roster : il doit couvrir tout candidat
# susceptible de franchir le seuil, y compris ceux qui ne l'ont pas encore
# franchi (inertes tant qu'ils ne l'atteignent pas). Les groupes réellement
# vides sont retirés à l'exécution par `build_roster`, donc en pré-classer
# davantage ne change pas la géométrie du modèle.
ORDER_GROUPS: list[tuple[str, list[str]]] = [
    ("LO",         ["Arthaud", "Poutou"]),
    ("LFI",        ["Mélenchon", "Ruffin"]),
    ("coco_ecolo", ["Roussel", "Tondelier"]),
    ("PS",         ["Glucksmann", "Hollande", "Faure"]),
    ("Villepin",   ["Villepin"]),
    ("Attal",      ["Attal", "Lecornu"]),
    ("Philippe",   ["Philippe"]),
    ("LR",         ["Retailleau", "Wauquiez", "Lisnard", "Darmanin"]),
    ("MLP",        ["Le Pen", "Bardella", "Dupont-Aignan"]),
    ("Zemmour",    ["Zemmour", "Knafo"]),
]

# Affectations PAR DÉFAUT, à confirmer (même statut que les réserves de §2.1
# sur Dupont-Aignan et Hollande) -- aucune n'est active aujourd'hui, tous ces
# candidats étant sous le seuil :
#   - Villepin : groupe PROPRE, placé entre PS et Attal. Placement réellement
#     ambigu (ex-Premier ministre gaulliste, donc à droite par trajectoire,
#     mais qui capte aujourd'hui un électorat de gauche/centre-gauche). C'est
#     le plus proche du seuil (4 sondages, testé le 2026-07-10 à 3,8 %) : à
#     trancher en priorité.
#   - Darmanin, Lisnard -> LR ; Lecornu -> Attal (macronie) ; Knafo -> Zemmour
#     (Reconquête).

# half_life (jours) : calibré par backtest hors-échantillon 2017/2022
# (notebooks/04b_spatial_halflife_backtest.py, cf. spec §5.2) -- 15j nettement
# meilleur que 45j/infini sur les deux élections, mais balayage grossier
# ({15,45,∞}), PAS affiné. Pas échantillonnable par NUTS (vraisemblance
# tempérée, cf. spec §3.1) -- constante calibrée séparément, comme
# `half_life_days` sur linear_pooling (model/models/linear_pooling/model.py).
DEFAULT_HALF_LIFE_DAYS = 15.0

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
NORMALIZE_KERNEL = True      # bascule de diagnostic (spec §12.10)
SIGMA_PRIOR_MEDIAN = 0.15
SIGMA_PRIOR_LOG_SCALE = 0.4
SD_DELTA_SIGMA_SCALE = 0.15

# --- Prior de position ancré sur 2022 -- DISPONIBLE, PAS ACTIVÉ (spec §12.5) -----
# Une fois le centrage de `sample_ordered_slots` corrigé (§12.4), la seule
# CONTRAINTE D'ORDRE suffit : R-hat 1,002, ESS 1970, 0 divergence sur le roster
# 2027. Ce prior informatif n'est donc pas nécessaire, et `use_position_prior`
# vaut False par défaut -- on ne paie pas une hypothèse dont on n'a pas besoin,
# et un prior d'analogie historique est précisément ce que §2 avait écarté pour
# de bonnes raisons (Attal et Philippe n'ont pas de prédécesseur direct).
# Le mécanisme reste implémenté et testé, prêt à servir si un roster futur
# (plus de groupes, moins de sondages) redevenait mal identifié.
#
# À n'utiliser QUE sur le roster live. Employer les positions 2022 comme prior
# d'un fit SUR 2022 (le backtest de §11.5) serait circulaire : `fit_geometry`
# (notebooks/04g) ne le passe donc pas.
BLOC_ANALOGUE_2022: dict[str, str] = {
    "LO": "gauche_radicale",
    "LFI": "gauche_radicale",
    "coco_ecolo": "ecologistes",
    "PS": "gauche",
    "Villepin": "centre",
    "Attal": "centre",
    "Philippe": "centre",
    "LR": "droite",
    "MLP": "droite_radicale",
    "Zemmour": "droite_radicale",
}
BANK_POS_PATH = __file__.replace("model.py", "bank_positions_2022.json")


def slot_prior_positions(slot_names: list[str]) -> np.ndarray | None:
    """Positions a priori des groupes, reprises des blocs 2022 analogues.

    Plusieurs groupes 2027 partagent un bloc 2022 (LO et LFI sont tous deux
    `gauche_radicale`) : leur prior de position est alors le MÊME, et c'est aux
    données de les séparer. L'ordre strict reste garanti par construction
    (§2.2), donc deux priors égaux n'autorisent pas une inversion.

    `None` si la banque n'a pas été produite -- repli sur le prior faible.
    """
    import json
    from pathlib import Path
    p = Path(BANK_POS_PATH)
    if not p.exists():
        return None
    pos = json.loads(p.read_text())["bloc_pos"]
    out = []
    for g in slot_names:
        bloc = BLOC_ANALOGUE_2022.get(g)
        if bloc is None or bloc not in pos:
            return None
        out.append(pos[bloc])
    out = np.asarray(out, dtype=float)
    # Les groupes partageant un bloc reçoivent la même valeur : on les écarte
    # d'un epsilon croissant pour que les `gap` a priori restent > 0.
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + 1e-3)
    return np.clip(out, 1e-3, 1 - 1e-3)


def sample_ordered_slots(name: str, n_slots: int, base_loc=0.0, base_scale=1.2,
                         gap_log_mu=-0.5, gap_log_scale=0.7, target_pos=None):
    """n_slots positions STRICTEMENT croissantes dans (0,1) : base + sauts positifs
    (LogNormal) cumulés, écrasés par sigmoid (monotone -> préserve l'ordre sans
    transformation `Ordered` explicite). Cf. spec §2.2.

    La somme cumulée est **CENTRÉE** avant d'ajouter `base` (spec §12.4). Sans
    ce centrage, `raw` part de 0 et ne fait que croître : à la médiane du prior
    (donc au point de départ d'`init_to_median`) les positions valent
    `sigmoid([0, 0.6, ..., 4.8])` = `[0.50, 0.65, ..., 0.99]`, c'est-à-dire
    toute la configuration écrasée dans la MOITIÉ DROITE de la grille. Le prior
    affirmait ainsi que le champ politique occupe la droite de l'axe -- ce qui
    n'a aucun sens, l'axe étant latent -- et surtout les chaînes démarraient
    dans un mauvais bassin qu'elles ne quittaient pas toutes : mesuré sur le
    roster 2027, 2 chaînes sur 4 restaient bloquées à 932 unités de
    log-vraisemblance du bon mode (R-hat 1,59). Centré, `base = 0` donne
    `sigmoid([-2.4, ..., +2.4])` = `[0.08, ..., 0.92]`, qui couvre la grille.
    `base` reste libre : c'est une re-paramétrisation de la LOCALISATION du
    prior, pas une contrainte ajoutée."""
    if target_pos is not None:
        # Prior INFORMATIF : on centre chaque écart sur celui qu'on observe
        # entre les blocs analogues d'une élection passée (spec §12.5). On agit
        # sur les GAPS, pas sur les positions : c'est la seule façon de garder
        # l'ordre strict garanti par construction (§2.2) tout en informant la
        # localisation. `gap_log_scale` reste inchangé, donc le prior est
        # informatif sur le CENTRE et pas plus serré qu'avant.
        raw_t = jnp.log(target_pos / (1.0 - target_pos))
        d = jnp.clip(jnp.diff(raw_t), 1e-3, None)
        gap_log_mu = jnp.log(d)
        base_loc = jnp.mean(raw_t)   # pas de float() : `target_pos` peut être tracé

    base = numpyro.sample(f"{name}_base", dist.Normal(base_loc, base_scale))
    if n_slots > 1:
        gaps = numpyro.sample(f"{name}_gaps", dist.LogNormal(gap_log_mu, gap_log_scale).expand([n_slots - 1]))
        cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(gaps)])
        raw = base + cum - jnp.mean(cum)
    else:
        raw = jnp.reshape(base, (1,))
    return numpyro.deterministic(f"{name}_slot_pos", jax.nn.sigmoid(raw))


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
    if NORMALIZE_KERNEL:
        A = w[..., None] + D - jax.nn.logsumexp(D, axis=-1, b=W, keepdims=True)
    else:
        A = w[..., None] + D
    A = A - jnp.max(A, axis=-2, keepdims=True)
    expA = jnp.exp(A)
    numerator = expA * tested_mask[..., :, None]
    denom = jnp.clip(numerator.sum(axis=-2, keepdims=True), 1e-12, None)
    return jnp.sum((numerator / denom) * W, axis=-1)


def weighted_loglik(mu, sigma, w, tested_mask, Y, Np, kappa, excess_var=None) -> tuple[jnp.ndarray, jnp.ndarray]:
    """`tested_mask`/`Y`/`Np`/`kappa` : (P,N)/(P,N)/(P,)/(P,) -- vraisemblance
    gaussienne (approx. du bruit multinomial, spec §4) pondérée par récence
    (spec §3.1 -- PAS une densité normalisée en `kappa`, cf. mise en garde
    sur pourquoi `half_life` n'est pas échantillonnable par NUTS).

    `excess_var` (P,) optionnel : variance d'excès (house effects) PAR NOEUD,
    en fraction² -- ajoutée à la variance d'échantillonnage pure
    `pi(1-pi)/N_p`, cf. spec section "Variance d'excès" / `excess_var_for_nodes`.
    `None` (défaut) = comportement d'origine (bruit d'échantillonnage seul)."""
    pi = spatial_shares(mu, sigma, w, tested_mask)                          # (P,N)
    var = pi * (1.0 - pi) / jnp.clip(Np[:, None], 1.0, None)
    if excess_var is not None:
        var = var + excess_var[:, None]
    var = var + 1e-8
    ll = dist.Normal(pi, jnp.sqrt(var)).log_prob(Y) * tested_mask
    return pi, jnp.sum(kappa[:, None] * ll)


def weighted_loglik_blocked(mu, sigma, w, tested_mask, Y, Np_full, kappa, notice_idx, n_notices,
                            rho, sigma_model=0.0, excess_var=None):
    """Vraisemblance à erreurs CORRÉLÉES entre hypothèses d'un même sondage
    (spec §12.7). Remplace `weighted_loglik`, qui les traite comme des nœuds
    indépendants à échantillon déflaté.

    Les `h` hypothèses d'un sondage sont posées aux **mêmes répondants** (ici
    jusqu'à 11 pour un seul sondage). Deux conséquences que le traitement
    indépendant rend incompatibles :

    - le NIVEAU ne doit pas gagner d'information en ajoutant des hypothèses —
      ce que la déflation `n/h` obtenait déjà, correctement ;
    - la DIFFÉRENCE entre deux hypothèses est mesurée bien plus PRÉCISÉMENT
      qu'entre deux sondages distincts, puisque le bruit d'échantillonnage se
      compense. Or c'est exactement ce différentiel qui identifie la géométrie
      (`mu`, `sigma`) : une hypothèse isolée s'explique par `w` seul, c'est le
      retrait d'un candidat qui révèle OÙ vont ses électeurs. La déflation
      gonfle la variance de cette différence d'un facteur `h` et dilue donc
      précisément le signal le plus informatif du jeu de données.

    Modèle : `Y_{i,p} = pi_{i,p} + u_{i,notice} + eps_{i,p}`, soit une
    corrélation `rho` entre hypothèses d'un même sondage, à variance TOTALE
    `v = pi(1-pi)/n + excès` bâtie sur l'échantillon PLEIN (`Np_full`) — plus
    de déflation, la corrélation joue son rôle. `rho -> 1` redonne « h
    hypothèses = un sondage » ; `rho -> 0` redonne l'indépendance à `n` plein.

    Forme close d'une gaussienne équicorrélée (bloc de taille `m`) :
    `Sigma = v[(1-rho)I + rho·11ᵀ]`, d'où
    `quad = [S2 - rho/(1+(m-1)rho)·S1²]/(1-rho)` et
    `logdet = Σ log v + (m-1)log(1-rho) + log(1+(m-1)rho)`,
    avec `S1 = Σ z`, `S2 = Σ z²`, `z` le résidu standardisé.

    `kappa` est constant par sondage (toutes les hypothèses partagent la date),
    donc la pondération de récence s'applique proprement au BLOC entier — elle
    n'aurait pas de sens appliquée à une composante d'un vecteur corrélé.
    """
    pi = spatial_shares(mu, sigma, w, tested_mask)                       # (P,N)
    var = pi * (1.0 - pi) / jnp.clip(Np_full[:, None], 1.0, None)
    if excess_var is not None:
        var = var + excess_var[:, None]
    var = var + 1e-8

    r = jnp.where(tested_mask > 0, Y - pi, 0.0)                          # résidus BRUTS
    seg = lambda x: jax.ops.segment_sum(x, notice_idx, num_segments=n_notices)
    S1, S2 = seg(r), seg(r ** 2)                                         # (n_notices, N)
    m = seg(tested_mask)
    v_bar = seg(jnp.where(tested_mask > 0, var, 0.0)) / jnp.clip(m, 1.0, None)

    # Sigma = a·I + b·11ᵀ  avec  a = (1-rho)·v + delta²  et  b = rho·v.
    # `delta` est l'ÉCART DE MODÈLE, indépendant entre hypothèses : chaque
    # hypothèse est un champ différent, donc l'erreur du modèle y diffère. Sans
    # lui, retirer la déflation exige du modèle qu'il reproduise un sondage à la
    # précision d'échantillonnage -- ce qu'un spatial à N candidats sur une
    # grille ne peut pas faire, et il se contorsionne (mesuré : Arthaud et
    # Mélenchon fusionnant à la même position, spec §12.7).
    a = (1.0 - rho) * v_bar + sigma_model ** 2
    b = rho * v_bar
    quad = (S2 - b / (a + m * b) * S1 ** 2) / a
    logdet = m * jnp.log(a) + jnp.log1p(m * b / a)
    ll_bc = -0.5 * (quad + logdet + m * jnp.log(2.0 * jnp.pi))
    ll_bc = jnp.where(m > 0, ll_bc, 0.0)                                 # candidat absent du sondage

    kappa_notice = jax.ops.segment_sum(kappa, notice_idx, num_segments=n_notices) / jnp.clip(
        jax.ops.segment_sum(jnp.ones_like(kappa), notice_idx, num_segments=n_notices), 1.0, None)
    return pi, jnp.sum(kappa_notice[:, None] * ll_bc)


def make_kappa(dates: np.ndarray, as_of: float, half_life: float) -> np.ndarray:
    if half_life >= 9999:
        return np.ones_like(dates, dtype=float)
    return 2.0 ** (-(as_of - dates) / half_life)


def spatial_pooling_model(slot_of, n_slots, tested_mask, Y, Np, kappa, excess_var=None,
                          sd_delta_mu_scale=0.06, sd_delta_sigma_scale=SD_DELTA_SIGMA_SCALE,
                          w_scale=1.5, sigma_slot_log_mu=None,
                          sigma_slot_log_scale=SIGMA_PRIOR_LOG_SCALE,
                          slot_prior_pos=None, notice_idx=None, n_notices=None,
                          Np_full=None, rho_hyp=None, sigma_model_fixed=None):
    """`slot_of` : (N,) indice de groupe d'ordre par candidat (spec §2.1).
    Positions ordonnées au niveau des GROUPES ; chaque candidat a son propre
    delta (mu, spec §2.3), son propre sigma (pooling par groupe), et son
    propre w (spec §3 -- pas de hiérarchie partagée sur w, vérifié sans effet
    sur pi dans notebooks/03b_spatial_debug.py).

    `excess_var` (P,) optionnel -- cf. `weighted_loglik`/`excess_var_for_nodes` :
    variance d'excès par institut, réutilisée depuis la Bank déjà calibrée de
    `bayesian_nowcast` (spec section "Variance d'excès"). Corrige la
    sous-couverture mesurée empiriquement (spec §6.4/§6.5) -- SANS ce terme,
    le modèle traite `pi(1-pi)/N` comme le seul bruit d'observation, ce que
    `notebooks/06b_same_day_poll_coherence_matched.py` a démontré insuffisant
    sur 2017/2022 (RMS(z)=1,58 entre sondages de MÊME champ à écart nul)."""
    N = slot_of.shape[0]
    if sigma_slot_log_mu is None:
        sigma_slot_log_mu = jnp.log(SIGMA_PRIOR_MEDIAN)

    slot_pos = sample_ordered_slots("mu", n_slots, target_pos=slot_prior_pos)
    sigma_slot = numpyro.sample("sigma_slot",
                                dist.LogNormal(sigma_slot_log_mu, sigma_slot_log_scale).expand([n_slots]))

    sd_delta_mu = numpyro.sample("sd_delta_mu", dist.HalfNormal(sd_delta_mu_scale))
    sd_delta_sigma = numpyro.sample("sd_delta_sigma", dist.HalfNormal(sd_delta_sigma_scale))
    z_mu = numpyro.sample("z_mu", dist.Normal(0.0, 1.0).expand([N]))
    z_sigma = numpyro.sample("z_sigma", dist.Normal(0.0, 1.0).expand([N]))
    mu = numpyro.deterministic("mu", slot_pos[slot_of] + z_mu * sd_delta_mu)
    sigma = numpyro.deterministic("sigma", sigma_slot[slot_of] * jnp.exp(z_sigma * sd_delta_sigma))

    w = numpyro.sample("w_now", dist.Normal(0.0, w_scale).expand([N]))

    if rho_hyp is not None and notice_idx is not None:
        # `rho_hyp="sample"` : la corrélation entre hypothèses d'un même sondage
        # est ESTIMÉE. Contrairement à `half_life` (§3.1), c'est un paramètre de
        # vraisemblance légitime -- il apparaît dans une densité normalisée (le
        # terme `logdet` est bien inclus), donc rien ne pousse à le dégénérer.
        # `rho` et `delta` décrivent le PROTOCOLE de sondage (hypothèses posées
        # au même panel, capacité du modèle à reproduire un champ), pas la
        # campagne : les échantillonner rend NUTS fragile là où le signal est
        # ténu -- sur 2022, dont les sondages testent 10 candidats sur 11, les
        # hypothèses d'un même sondage diffèrent trop peu pour identifier `rho`,
        # qui dérive (R-hat 2,52, 103 divergences à J-120). Les FIXER retire les
        # deux directions difficiles, comme la variance d'excès est fixée depuis
        # sa propre banque (spec §12.7).
        rho = (numpyro.sample("rho_hyp", dist.Beta(2.0, 2.0))
               if isinstance(rho_hyp, str) else rho_hyp)
        sig_m = (numpyro.sample("sigma_model", dist.HalfNormal(0.01))
                 if sigma_model_fixed is None else sigma_model_fixed)
        pi, ll = weighted_loglik_blocked(mu, sigma, w, tested_mask, Y, Np_full, kappa,
                                         notice_idx, n_notices, rho, sigma_model=sig_m,
                                         excess_var=excess_var)
    else:
        pi, ll = weighted_loglik(mu, sigma, w, tested_mask, Y, Np, kappa, excess_var=excess_var)
    numpyro.deterministic("pi", pi)
    numpyro.factor("weighted_ll", ll)


# Petit plancher numérique ajouté à CHAQUE Δt de la marche dense (spec §3bis)
# -- même rôle que MIN_DT_EPS dans bayesian_nowcast/nowcast.py : évite une
# variance de transition nulle si deux dates uniques tombent le même jour
# (ne devrait pas arriver après dédoublonnage, mais coûte rien de le garder)
# ou si le dernier sondage tombe le jour même de `as_of`.
MIN_DT_EPS = 1e-6


def spatial_pooling_model_tau(slot_of, n_slots, tested_mask, Y, Np, date_idx, dt_gaps, as_of_dt,
                              sd_delta_mu_scale=0.06, sd_delta_sigma_scale=0.15,
                              tau_prior_scale=0.05, sigma0_prior_scale=0.5,
                              sigma_slot_log_mu=None, sigma_slot_log_scale=0.4):
    """PARKÉE (session du 2026-08-10, cf. spec §3.bis) : converge proprement
    (R-hat 1.005, ESS 1168, 7/4000 divergences) sur le roster 2027 réel mais
    vers un MODE DÉGÉNÉRÉ -- les mu se sont effondrés entre 0,70 et 0,99
    (quasi plus d'écart gauche-droite), le modèle distinguant les candidats
    via `w` plutôt que la position. R-hat/ESS sains n'excluent PAS un mode
    dégénéré (les chaînes peuvent converger ENSEMBLE vers le même mauvais
    mode). Aussi ~7x plus lent que `spatial_pooling_model` (2066s vs ~300s,
    taille comparable). La stratification par âge de la couverture (spec
    §6.4) suggère par ailleurs que la diffusion n'est probablement pas la
    cause dominante de la sous-couverture -- priorité donnée à un terme de
    variance d'excès (§10) à la place. Code gardé pour référence/reprise
    éventuelle, PAS la voie recommandée actuellement.

    Variante avec VRAIE diffusion temporelle (spec §3.bis) -- remplace la
    vraisemblance tempérée de `spatial_pooling_model` (§3.1) : `w_i(t)` suit
    une marche aléatoire gaussienne sur une ligne du temps GLOBALE dédupliquée
    (M dates uniques, partagée par tous les candidats -- DENSE, pas creuse,
    cf. `build_poll_arrays`), avec `tau` échantillonné normalement par NUTS
    (prior faible ici, PAS encore calibré sur 2017/2022 -- cf. spec §3bis
    pour ce qui reste à faire). Contrairement à `spatial_pooling_model`, la
    vraisemblance n'est PAS pondérée (`kappa` disparaît) : le poids de
    récence est maintenant une conséquence de la diffusion elle-même
    (un sondage vieux de 60 jours pèse moins parce que `tau²·60` a eu le
    temps de rendre l'état d'alors moins informatif sur l'état actuel, pas
    parce qu'on a artificiellement réduit sa vraisemblance).

    Dimension ajoutée par rapport à `spatial_pooling_model` : `N * M` (ex.
    13 candidats x 13 dates uniques = 169 sur le roster 2027 réel au
    2026-08-10 -- petit en pratique, pas besoin d'EKF/UKF pour ce volume de
    données)."""
    N = slot_of.shape[0]
    M = dt_gaps.shape[0] + 1
    if sigma_slot_log_mu is None:
        sigma_slot_log_mu = jnp.log(0.15)

    slot_pos = sample_ordered_slots("mu", n_slots)
    sigma_slot = numpyro.sample("sigma_slot",
                                dist.LogNormal(sigma_slot_log_mu, sigma_slot_log_scale).expand([n_slots]))

    sd_delta_mu = numpyro.sample("sd_delta_mu", dist.HalfNormal(sd_delta_mu_scale))
    sd_delta_sigma = numpyro.sample("sd_delta_sigma", dist.HalfNormal(sd_delta_sigma_scale))
    z_mu = numpyro.sample("z_mu", dist.Normal(0.0, 1.0).expand([N]))
    z_sigma = numpyro.sample("z_sigma", dist.Normal(0.0, 1.0).expand([N]))
    mu = numpyro.deterministic("mu", slot_pos[slot_of] + z_mu * sd_delta_mu)
    sigma = numpyro.deterministic("sigma", sigma_slot[slot_of] * jnp.exp(z_sigma * sd_delta_sigma))

    # tau/sigma0 PARTAGÉS (pas par candidat) pour commencer -- même prudence
    # que le SSM historique sur `tau_candidat` (essayé, pas adopté par
    # défaut faute de gain net, cf. docs/biais_rn_investigation.md point 9).
    tau = numpyro.sample("tau", dist.HalfNormal(tau_prior_scale))
    sigma0 = numpyro.sample("sigma0_w", dist.HalfNormal(sigma0_prior_scale))

    step_sd = jnp.concatenate([
        jnp.full((N, 1), sigma0),                                    # (N,1) état initial
        jnp.broadcast_to(tau * jnp.sqrt(dt_gaps + MIN_DT_EPS), (N, M - 1)),  # (N,M-1) pas de diffusion
    ], axis=1)
    z_w = numpyro.sample("z_w_path", dist.Normal(0.0, 1.0).expand([N, M]))
    w_path = jnp.cumsum(z_w * step_sd, axis=1)                        # (N,M)

    # Extrapolation d'un pas au-delà de la dernière date unique, jusqu'à
    # `as_of` -- même logique que l'extrapolation finale du SSM
    # (bayesian_nowcast_ssm_model, `extrap_cov`), pas de nouvelle observation.
    z_extra = numpyro.sample("z_w_extrap", dist.Normal(0.0, 1.0).expand([N]))
    w_now = numpyro.deterministic("w_now", w_path[:, -1] + tau * jnp.sqrt(as_of_dt + MIN_DT_EPS) * z_extra)

    w_at_node = w_path[:, date_idx].T                                 # (P,N) -- w_i au jour de chaque noeud
    D = -((V[None, :] - mu[:, None]) ** 2) / (2 * sigma[:, None] ** 2)  # (N,B)
    A = w_at_node[:, :, None] + D[None, :, :]                          # (P,N,B)
    A = A - jnp.max(A, axis=1, keepdims=True)
    expA = jnp.exp(A) * tested_mask[:, :, None]
    denom = jnp.clip(expA.sum(axis=1, keepdims=True), 1e-12, None)
    pi = jnp.sum((expA / denom) * W[None, None, :], axis=2)           # (P,N)

    var = pi * (1.0 - pi) / jnp.clip(Np[:, None], 1.0, None) + 1e-8
    ll = dist.Normal(pi, jnp.sqrt(var)).log_prob(Y) * tested_mask
    numpyro.deterministic("pi", pi)
    numpyro.factor("ll", jnp.sum(ll))


def ou_kl_basis(unique_dates, as_of, tau_ou, var_cible=0.99, k_max=15):
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

    Renvoie `(base (K, M+1), k)` : la dernière colonne est `as_of`, incluse dans
    la grille pour que l'extrapolation sorte de la même base plutôt que d'un
    terme ajouté à part.
    """
    t = np.concatenate([np.asarray(unique_dates, dtype=float), [float(as_of)]])
    C = np.exp(-np.abs(t[:, None] - t[None, :]) / tau_ou)
    val, vec = np.linalg.eigh(C)
    val, vec = val[::-1], vec[:, ::-1]
    val = np.clip(val, 0.0, None)
    k = int(np.searchsorted(np.cumsum(val) / val.sum(), var_cible)) + 1
    k = max(3, min(k, k_max, len(t)))
    return (vec[:, :k] * np.sqrt(val[:k])).T, k


def spatial_pooling_model_ou(slot_of, n_slots, tested_mask, Y, Np, date_idx, kl_basis,
                             excess_var=None, sd_delta_mu_scale=0.06,
                             sd_delta_sigma_scale=SD_DELTA_SIGMA_SCALE,
                             tau_ou=None, sigma_w_prior=0.5,
                             sigma_slot_log_mu=None, sigma_slot_log_scale=SIGMA_PRIOR_LOG_SCALE,
                             slot_prior_pos=None, dynamic_mask=None):
    """Modèle JOINT à noyau Ornstein-Uhlenbeck sur `w` (spec §12.17).

    Combine les deux propriétés qu'aucune version précédente n'avait ensemble :

    - **inférence jointe** de la géométrie et du chemin de `w`, donc propagation
      exacte de l'incertitude. La lecture en deux temps de §11 ne peut pas le
      faire : les mêmes données informent les deux étapes, et aucune répartition
      des rôles ne l'évite (mesuré sur données simulées, §12.16 — 99,0 % d'IC90
      en géométrie variable, 79,4 % en géométrie figée, le vrai étant encadré).
    - **noyau OU** plutôt que marche aléatoire : la variance de dérive sature à
      `sigma_w²(1-e^{-2h/tau})` au lieu de croître sans borne. C'est l'argument
      du §11.1, et c'est ce qui manquait à `spatial_pooling_model_tau` (mesuré :
      sur-dispersion d'un facteur 3 à 7 jours d'horizon, d'où 93,7 % d'IC90).

    Transition exacte, sans approximation d'Euler :
    `w(t_k) | w(t_{k-1}) ~ N(rho_k·w(t_{k-1}), sigma_w²(1-rho_k²))`,
    `rho_k = exp(-dt_k/tau_ou)`. Écrit sous forme matricielle `w = L·z` (L
    triangulaire des produits cumulés de `rho`) plutôt qu'en `scan` : `M` est
    petit (13 dates uniques sur le roster 2027) et NUTS différentie mieux un
    produit matriciel qu'une boucle.

    `tau_ou` est FIXÉ, repris de la banque commune (`model/core/opinion.py`) :
    sur une fenêtre de campagne, seul le rapport `sigma_w²/tau` est contraint
    (§11.4), les laisser libres tous deux revient à choisir un point d'une
    crête plate.
    """
    N = slot_of.shape[0]
    if sigma_slot_log_mu is None:
        sigma_slot_log_mu = jnp.log(SIGMA_PRIOR_MEDIAN)
    if tau_ou is None:
        from model.core.opinion import load_law
        tau_ou = float(load_law()["tau"])

    slot_pos = sample_ordered_slots("mu", n_slots, target_pos=slot_prior_pos)
    sigma_slot = numpyro.sample("sigma_slot",
                                dist.LogNormal(sigma_slot_log_mu, sigma_slot_log_scale).expand([n_slots]))
    sd_delta_mu = numpyro.sample("sd_delta_mu", dist.HalfNormal(sd_delta_mu_scale))
    sd_delta_sigma = numpyro.sample("sd_delta_sigma", dist.HalfNormal(sd_delta_sigma_scale))
    z_mu = numpyro.sample("z_mu", dist.Normal(0.0, 1.0).expand([N]))
    z_sigma = numpyro.sample("z_sigma", dist.Normal(0.0, 1.0).expand([N]))
    mu = numpyro.deterministic("mu", slot_pos[slot_of] + z_mu * sd_delta_mu)
    sigma = numpyro.deterministic("sigma", sigma_slot[slot_of] * jnp.exp(z_sigma * sd_delta_sigma))

    sigma_w = numpyro.sample("sigma_w", dist.HalfNormal(sigma_w_prior))

    # Chemin dans la base de Karhunen-Loève tronquée (`ou_kl_basis`) : K
    # composantes au lieu de M valeurs par candidat -- K vaut 3 à 13 quand M va
    # de 12 à 44. Les composantes écartées sont exactement celles dont la
    # vraisemblance ne dit rien : avec tau ≈ 262 j et des sondages espacés de
    # 2-3 jours, la première composante porte déjà 76 à 89 % de la variance.
    # Les échantillonner créait des dizaines de directions PLATES que NUTS
    # devait parcourir sans contrainte -- d'où une profondeur d'arbre bloquée à
    # 255 pas de leapfrog par itération, et des chaînes qui errent (§12.22).
    #
    # La dernière colonne de `kl_basis` est `as_of` : l'extrapolation sort de la
    # même base au lieu d'être un terme ajouté après coup.
    if dynamic_mask is None:
        dyn = jnp.ones(N, dtype=bool)
    else:
        dyn = jnp.asarray(dynamic_mask, dtype=bool)
    z_w = numpyro.sample("z_w", dist.Normal(0.0, 1.0).expand([N, kl_basis.shape[0]]))
    w_full = sigma_w * (z_w @ kl_basis)                                 # (N, M+1)
    # Candidats sous le seuil : `w` STATIQUE (§12.19), un seul paramètre.
    w_static = numpyro.sample("w_static", dist.Normal(0.0, 1.0).expand([N])) * sigma_w
    w_full = jnp.where(dyn[:, None], w_full, w_static[:, None])
    numpyro.deterministic("w_now", w_full[:, -1])
    w_path = w_full[:, :-1]

    w_at_node = w_path[:, date_idx].T                                   # (P,N)
    pi = spatial_shares(mu, sigma, w_at_node, tested_mask)
    var = pi * (1.0 - pi) / jnp.clip(Np[:, None], 1.0, None)
    if excess_var is not None:
        var = var + excess_var[:, None]
    ll = dist.Normal(pi, jnp.sqrt(var + 1e-8)).log_prob(Y) * tested_mask
    numpyro.deterministic("pi", pi)
    numpyro.factor("ll", jnp.sum(ll))


# --- EKF + Ornstein-Uhlenbeck pour w (spec §3.ter, session du 2026-08-12) --------
# Remplace la vraisemblance tempérée (§3.1) ET la marche brownienne parquée
# (§3.bis, mode dégénéré) par un vrai filtre séquentiel (extended_kalman_filter,
# dynamax -- déjà une dépendance du projet, cf. model/core/gp_math.py qui
# utilise le MÊME noyau OU pour une observation LINÉAIRE en CLR, avec forme
# close). Ici l'observation (softmax masqué sur la grille) est NON-LINÉAIRE en
# w, pas de forme close possible (contrairement à gp_pooling) -- EKF nécessaire,
# comme anticipé en session. `tau_w`/`sigma_w` échantillonnés par NUTS, prior
# faible, PAS calibrés sur 2017/2022 (contrairement à `sigma2`/`tau` de
# model/core/opinion.py, qui vivent dans un autre espace -- CLR agrégé, pas
# w candidat par candidat -- et ne sont donc pas directement transférables).
PADDING_VAR_EKF = 1e6   # même esprit que PADDING_VAR (bayesian_nowcast/latent.py) : neutralise
                        # la contribution des candidats non testés à un nœud, sans les exclure
                        # du vecteur d'émission à taille FIXE qu'exige l'EKF.


def sort_by_date(arrays: dict) -> dict:
    """Trie les nœuds de `build_poll_arrays` par date croissante et ajoute
    `dt_forward[p]` = écart (jours) vers le nœud SUIVANT (dernier = 0,
    inutilisé) -- nécessaire pour un filtre SÉQUENTIEL (EKF), contrairement
    au modèle tempéré/à la marche dense qui n'ont pas besoin d'ordre
    temporel explicite. Même convention que `dt_forward` dans
    `bayesian_nowcast/nowcast.py::NowcastData`."""
    order = np.argsort(arrays["dates"])
    out = dict(arrays)
    out["tested_mask"] = arrays["tested_mask"][order]
    out["Y"] = arrays["Y"][order]
    out["Np"] = arrays["Np"][order]
    out["dates"] = arrays["dates"][order]
    out["instituts"] = [arrays["instituts"][i] for i in order]
    dt_forward = np.diff(out["dates"])
    out["dt_forward"] = np.append(dt_forward, 0.0)
    return out


def spatial_pooling_model_ekf(slot_of, n_slots, tested_mask, Y, Np, dt_forward, as_of_dt, excess_var=None,
                              sd_delta_mu_scale=0.06, sd_delta_sigma_scale=0.15,
                              tau_w_log_mu=None, tau_w_log_sd=0.8, sigma_w_prior_scale=1.0,
                              sigma_slot_log_mu=None, sigma_slot_log_scale=0.4):
    """`w_i(t)` suit un Ornstein-Uhlenbeck (mean-reversion vers 0 -- inoffensif,
    la jauge de `w` n'est pas identifiée en niveau absolu, §3.2), filtré par
    EKF : NUTS n'échantillonne que les hyperparamètres GLOBAUX (`mu`,`sigma`,
    `tau_w`,`sigma_w`), pas un `w` par nœud (évite la dimension `N x P`
    qu'un NUTS-sur-chemin-complet aurait demandée). `tested_mask`/`Y`/`Np`
    DOIVENT être triés par date croissante (`sort_by_date`) : le filtre est
    séquentiel, contrairement aux autres variantes.

    Transition OU exacte (pas une approximation Euler) sur un pas `dt` :
    `w(t+dt) | w(t) ~ N(w(t)·e^{-dt/τ}, σ_w²·(1-e^{-2dt/τ}))`. Observation
    `h(w) = spatial_shares(mu, sigma, w, mask)` -- non-linéaire, d'où l'EKF
    (linéarise `h` à chaque pas via le Jacobien, `jax.jacfwd` dans dynamax).
    `R` construit à partir des valeurs OBSERVÉES `Y` (pas de `pi` prédit par
    l'état) -- même convention que `poll_observation_values`
    (bayesian_nowcast/latent.py) : évite toute circularité (R ne doit pas
    dépendre de l'état filtré)."""
    from dynamax.nonlinear_gaussian_ssm.inference_ekf import extended_kalman_filter
    from dynamax.nonlinear_gaussian_ssm.models import ParamsNLGSSM

    N = slot_of.shape[0]
    P = tested_mask.shape[0]
    if sigma_slot_log_mu is None:
        sigma_slot_log_mu = jnp.log(0.15)
    if tau_w_log_mu is None:
        tau_w_log_mu = jnp.log(45.0)   # ordre de grandeur du `tau` calibré pour la
                                       # diffusion CLR (model/core/opinion.py, médiane ~60j)
                                       # -- repli raisonnable, PAS calibré pour w lui-même

    slot_pos = sample_ordered_slots("mu", n_slots)
    sigma_slot = numpyro.sample("sigma_slot",
                                dist.LogNormal(sigma_slot_log_mu, sigma_slot_log_scale).expand([n_slots]))
    sd_delta_mu = numpyro.sample("sd_delta_mu", dist.HalfNormal(sd_delta_mu_scale))
    sd_delta_sigma = numpyro.sample("sd_delta_sigma", dist.HalfNormal(sd_delta_sigma_scale))
    z_mu = numpyro.sample("z_mu", dist.Normal(0.0, 1.0).expand([N]))
    z_sigma = numpyro.sample("z_sigma", dist.Normal(0.0, 1.0).expand([N]))
    mu = numpyro.deterministic("mu", slot_pos[slot_of] + z_mu * sd_delta_mu)
    sigma = numpyro.deterministic("sigma", sigma_slot[slot_of] * jnp.exp(z_sigma * sd_delta_sigma))

    tau_w = numpyro.sample("tau_w", dist.LogNormal(tau_w_log_mu, tau_w_log_sd))
    sigma_w = numpyro.sample("sigma_w", dist.HalfNormal(sigma_w_prior_scale))
    sigma_w2 = sigma_w ** 2

    def f(w, u):
        dt = u[-1]
        return w * jnp.exp(-dt / tau_w)

    def h(w, u):
        mask = u[:-1]
        return spatial_shares(mu, sigma, w, mask)

    var_obs = Y * (1.0 - Y) / jnp.clip(Np[:, None], 1.0, None)
    if excess_var is not None:
        var_obs = var_obs + excess_var[:, None]
    var_obs = jnp.where(tested_mask > 0, var_obs, PADDING_VAR_EKF)
    R = jax.vmap(jnp.diag)(var_obs)                                   # (P,N,N)

    Q_scale = sigma_w2 * (1.0 - jnp.exp(-2.0 * dt_forward / tau_w))    # (P,) -- vers le nœud SUIVANT
    Q = jax.vmap(lambda q: q * jnp.eye(N))(Q_scale)                    # (P,N,N)

    inputs = jnp.concatenate([tested_mask, dt_forward[:, None]], axis=1)   # (P, N+1)

    params = ParamsNLGSSM(
        initial_mean=jnp.zeros(N), initial_covariance=sigma_w2 * jnp.eye(N),
        dynamics_function=f, dynamics_covariance=Q,
        emission_function=h, emission_covariance=R,
    )
    post = extended_kalman_filter(params, Y, inputs=inputs,
                                  output_fields=["filtered_means", "filtered_covariances", "marginal_loglik"])
    # `post.marginal_loglik` est la trace CUMULATIVE par pas de temps (dynamax
    # accumule `ll` dans le carry du scan et la restitue à chaque pas, cf.
    # source lue en session) -- seul le DERNIER élément est le vrai
    # log-vraisemblance marginal total ; sommer tout le tableau compterait
    # les termes en double (bug trouvé par contrôle direct, valeurs non
    # monotones sinon).
    ll_total = post.marginal_loglik[-1] if P > 0 else 0.0
    numpyro.factor("ekf_ll", ll_total)

    # Extrapolation d'UN pas au-delà du dernier nœud, jusqu'à `as_of` -- même
    # logique que l'extrapolation finale du SSM historique (`extrap_cov`),
    # PAS une nouvelle observation.
    if P > 0:
        last_mean, last_cov = post.filtered_means[-1], post.filtered_covariances[-1]
    else:
        last_mean, last_cov = jnp.zeros(N), sigma_w2 * jnp.eye(N)
    decay = jnp.exp(-as_of_dt / tau_w)
    extrap_mean = last_mean * decay
    extrap_cov = (decay ** 2) * last_cov + sigma_w2 * (1.0 - decay ** 2) * jnp.eye(N)
    L = jnp.linalg.cholesky(extrap_cov + 1e-9 * jnp.eye(N))
    z_extra = numpyro.sample("z_w_extrap", dist.Normal(0.0, 1.0).expand([N]))
    numpyro.deterministic("w_now", extrap_mean + L @ z_extra)


# --- Roster & mise en forme des sondages -----------------------------------------
def build_roster(raw: pd.DataFrame, as_of=None) -> tuple[list[str], np.ndarray, list[str]]:
    """Roster modélisé : tout candidat testé dans >= MIN_POLLS sondages RÉELS
    distincts (`notice`, pas hypothèses) ET encore testé récemment
    (MAX_LAST_POLL_AGE_DAYS), assigné à son groupe d'ordre.

    Un candidat éligible absent de ORDER_GROUPS **lève une erreur**. Il était
    auparavant écarté avec un simple `warning`, et c'est ce qui a produit le
    défaut le plus grave du modèle (spec §12) : Bardella, testé dans 46 nœuds
    sur 78, disparaissait du roster sans disparaître des données, faisant
    tomber la somme des intentions à 0,66 par nœud face à un `pi` qui somme à
    1. Un candidat non classé n'est pas une nuisance cosmétique : c'est un trou
    dans le bulletin que la vraisemblance ne peut pas absorber. Mieux vaut un
    job qui échoue avec la liste à classer qu'un modèle publié faux.

    Les groupes SANS aucun candidat éligible sont retirés : `ORDER_GROUPS` peut
    donc pré-classer des candidats encore sous le seuil sans que leur groupe
    vide n'occupe une position dans la séquence ordonnée (ce qui resserrerait
    inutilement les autres).
    """
    counts = raw.groupby("candidat")["notice"].nunique()
    eligible = set(counts[counts >= MIN_POLLS].index)

    dates = pd.to_datetime(raw["date_fin"])
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else dates.max()
    last_seen = raw.assign(_d=dates).groupby("candidat")["_d"].max()
    stale = {c for c in eligible
             if (as_of_ts - last_seen[c]).days > MAX_LAST_POLL_AGE_DAYS}
    eligible -= stale

    unclassified = eligible - {m for _, members in ORDER_GROUPS for m in members}
    if unclassified:
        raise ValueError(
            f"candidats éligibles absents de ORDER_GROUPS : {sorted(unclassified)}. "
            "Les classer dans model.py (cf. spec §12) — les ignorer creuserait un "
            "trou de normalisation dans la vraisemblance."
        )

    candidates, slot_of, slot_names = [], [], []
    for gname, members in ORDER_GROUPS:
        members_in = [m for m in members if m in eligible]
        if not members_in:
            continue
        for m in members_in:
            candidates.append(m)
            slot_of.append(len(slot_names))
        slot_names.append(gname)
    return candidates, np.array(slot_of), slot_names


def build_poll_arrays(df: pd.DataFrame, candidates: list[str], notice_col="notice", hyp_col="hypothese",
                      candidat_col="candidat", date_col="date_fin", intention_col="intention",
                      echantillon_col="echantillon", institut_col="institut") -> dict:
    """DataFrame long (une ligne = un candidat testé dans une hypothèse d'un
    sondage) -> (tested_mask, Y, Np, dates, instituts) restreint à
    `candidates`. Même déflation `echantillon / n_hypotheses` que
    `aggregate_to_slots` (model/core/live_dataset.py) : les hypothèses d'un
    même sondage partagent le terrain, pas des mesures indépendantes.
    `instituts` (P,) -- un nom d'institut par nœud, nécessaire pour
    `excess_var_for_nodes` (variance d'excès par institut)."""
    idx = {c: i for i, c in enumerate(candidates)}
    df = df[df[candidat_col].isin(candidates)].copy()
    df[hyp_col] = df[hyp_col].fillna("__unique__")
    g = (df.groupby([notice_col, hyp_col, date_col, echantillon_col, institut_col, candidat_col])[intention_col]
          .sum().reset_index())
    g["n_hyp"] = g.groupby(notice_col)[hyp_col].transform("nunique")

    N = len(candidates)
    tested_mask, Y, Np, dates, instituts = [], [], [], [], []
    Np_full, notices = [], []          # cf. `notice_idx` plus bas (spec §12.7)
    for (notice, hyp), grp in g.groupby([notice_col, hyp_col], sort=False):
        mask = np.zeros(N)
        y = np.zeros(N)
        for r in grp.itertuples(index=False):
            cand = getattr(r, candidat_col)
            if cand in idx:
                mask[idx[cand]] = 1.0
                y[idx[cand]] = getattr(r, intention_col) / 100.0
        if mask.sum() < 2:
            continue
        # SOUS-COMPOSITION (session du 2026-08-12, spec §12) : `spatial_shares`
        # renvoie un `pi` qui somme à 1 sur le champ masqué, alors que `Y`
        # restreint au roster ne somme à 1 que si le roster couvre TOUT le
        # bulletin -- 0,94 en médiane sur 2017/2022 (petits candidats hors
        # blocs), et jusqu'à 0,66 sur 2027 avant la réintégration de Bardella.
        # Comparer les deux tels quels rend la vraisemblance structurellement
        # insatisfiable. La restriction d'un multinomial à un sous-ensemble EST
        # un multinomial : on renormalise `Y` et on déflate `N_p` d'autant
        # (nombre de répondants exprimant une préférence pour un candidat du
        # roster) -- pas d'ajustement ad hoc, c'est la loi exacte.
        tot = float(y[mask > 0].sum())
        if tot <= 0:
            continue
        y = y / tot
        tested_mask.append(mask)
        Y.append(y)
        n_raw = float(grp[echantillon_col].iloc[0])
        Np.append(n_raw / float(grp["n_hyp"].iloc[0]) * tot)
        Np_full.append(n_raw * tot)     # SANS déflation par n_hyp
        notices.append(notice)
        dates.append(pd.Timestamp(grp[date_col].iloc[0]))
        instituts.append(str(grp[institut_col].iloc[0]))

    dates_num = np.array([d.toordinal() for d in dates], dtype=float)

    # Ligne du temps GLOBALE dédupliquée (jours calendaires distincts, pas
    # nœuds) -- nécessaire pour la marche aléatoire dense de
    # `spatial_pooling_model_tau` (spec §3bis) : `date_idx[p]` pointe vers la
    # position du nœud p sur cette ligne, `dt_gaps[m]` l'écart (jours) entre
    # deux dates uniques consécutives. Calculé systématiquement (coût nul
    # pour le modèle tempéré, qui ignore juste ces clés).
    # Ligne du temps regroupée par PAS DE `TIME_BIN_DAYS` jours (spec §12.18).
    # Le chemin de `w` du modèle joint est échantillonné à chaque pas : à la
    # journée, 2022 J-30 en compte 97 (dimension N·M = 1067) alors que le
    # roster 2027 n'en a que 13 (169). On demandait donc au modèle d'estimer
    # l'opinion jour par jour, à une résolution où elle ne bouge pas de façon
    # mesurable -- sur-paramétrisation qui casse NUTS (R-hat 1,59 à 2,60 sur 3
    # coupures 2022 sur 5) sans rien apporter.
    binned = np.floor(dates_num / TIME_BIN_DAYS) * TIME_BIN_DAYS
    unique_dates = np.sort(np.unique(binned))
    date_idx = np.searchsorted(unique_dates, binned)
    dt_gaps = np.diff(unique_dates)

    # `notice_idx` : à quel SONDAGE appartient chaque nœud. Deux hypothèses d'un
    # même sondage sont posées aux MÊMES répondants, leurs erreurs sont donc
    # corrélées -- `weighted_loglik_blocked` s'en sert (spec §12.7). `Np_full`
    # est la taille d'échantillon NON déflatée par `n_hyp` : la déflation est
    # le procédé de repli qui suppose l'indépendance, l'autre vraisemblance
    # modélise la corrélation explicitement et n'en a pas besoin.
    uniq_notices = {n: i for i, n in enumerate(dict.fromkeys(notices))}
    notice_idx = np.array([uniq_notices[n] for n in notices])

    return dict(tested_mask=np.array(tested_mask), Y=np.array(Y), Np=np.array(Np), dates=dates_num,
               unique_dates=unique_dates, date_idx=date_idx, dt_gaps=dt_gaps, instituts=instituts,
               Np_full=np.array(Np_full), notice_idx=notice_idx, n_notices=len(uniq_notices))


# --- Variance d'excès (house effects) -- calibration PROPRE à spatial_pooling ------
def excess_var_for_nodes(instituts: list[str], bank) -> np.ndarray:
    """Variance d'excès PAR NOEUD (P,), en fraction² -- ajoutée à
    `pi(1-pi)/N` dans `weighted_loglik`.

    Historique (spec §6.5) : une première version réutilisait TELLE QUELLE la
    Bank `bayesian_nowcast` (`excess_sigma`, calibrée en espace ILR contre
    l'écart au résultat final) -- mesuré en SUR-correction sévère (IC90
    empirique à 99,6% au lieu de 90% sur 2022) : cette Bank mélange bruit
    d'institut et dérive de campagne dans sa propre décomposition, pas
    directement transférable à la géométrie de `spatial_pooling`.

    Remplacée par une calibration DÉDIÉE (`calibration.py::SpatialExcessCalibration`,
    `excess_sigma_spatial`) -- fit sur le même diagnostic model-free qui a
    révélé le problème (paires de sondages de champ EXACTEMENT identique,
    `notebooks/06b_same_day_poll_coherence_matched.py`), pas emprunté à un
    autre modèle. Vérifié par contrôle postérieur (RMS(z) sur les mêmes
    paires : 1,57 sans excès -> 1,11 avec, cf. spec §6.5)."""
    from model.models.spatial_pooling.calibration import excess_sigma_spatial
    if bank is None:
        return np.zeros(len(instituts))
    return np.array([(excess_sigma_spatial(bank, inst) / 100.0) ** 2 for inst in instituts])


# --- Fit + lecture AVEC incertitude (pas un point estimate) ----------------------
@dataclass
class SpatialPoolingFit:
    """Résultat d'un fit : tirages POSTÉRIEURS complets (mu, sigma, w), pas
    des moyennes -- nécessaire pour propager l'incertitude (cf. spec section
    "Incertitude", et `pi_draws_for_mask` ci-dessous).

    `w_draws` ne vient PAS forcément du même postérieur que `mu`/`sigma` : par
    défaut (`w_dynamics=True`) il est ré-estimé après coup par
    `w_dynamics.w_draws_ou` (inversion locale exacte + Ornstein-Uhlenbeck en
    forme close, spec §11), la vraisemblance tempérée n'ayant aucun plancher de
    variance temporel. `w_source` dit laquelle des deux lectures a été
    employée."""
    candidates: list[str]
    slot_of: np.ndarray
    slot_names: list[str]
    mu_draws: np.ndarray      # (S, N)
    sigma_draws: np.ndarray   # (S, N)
    w_draws: np.ndarray       # (S, N)
    diagnostics: dict
    w_source: str = "tempered"


def fit_spatial_pooling(raw_polls: pd.DataFrame, as_of: str, half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
                        draws: int = 1000, tune: int = 1000, chains: int = 4, seed: int = 27,
                        target_accept: float = 0.95, excess_bank=None,
                        w_dynamics: bool = True, n_geom: int = 200,
                        use_position_prior: bool = False, blocked: bool = False) -> SpatialPoolingFit:
    """Fit complet : roster -> arrays -> NUTS. Retourne les tirages bruts
    (mu/sigma/w), PAS un pi déjà agrégé -- `pi_draws_for_mask` construit pi
    APRÈS coup pour un scénario donné (le point de cette architecture, cf.
    spec §7 : un scénario personnalisé est un calcul direct sur ces tirages,
    pas une ré-inférence).

    `excess_bank` : Bank de variance d'excès à réutiliser (`excess_var_for_nodes`)
    -- `None` (défaut) charge celle calibrée pour CE modèle
    (`calibration.BANK_EXCESS_PATH`) si présente, repli sans excès sinon (pas
    d'erreur si jamais calibrée). PAS la Bank `bayesian_nowcast` -- mesurée en
    sur-correction sévère, cf. `excess_var_for_nodes`.

    `w_dynamics` (défaut True) : ré-estime `w_now` par
    `w_dynamics.w_draws_ou` au lieu de garder le `w_now` tempéré du fit NUTS.
    Le fit NUTS lui-même est INCHANGÉ -- il ne sert plus qu'à la géométrie
    (`mu`, `sigma`), la seule chose qu'il estime bien. Coût de l'ajout :
    quelques secondes, aucun MCMC (spec §11). `False` restaure la lecture
    d'origine (spec §3.1), utile pour comparer.
    `n_geom` : tirages de géométrie propagés dans la dynamique de `w` (la
    résolution linéaire est en `O((P·N)³)` par tirage -- 200 suffit largement
    à résumer un postérieur, cf. la stabilité mesurée en §11)."""
    raw = raw_polls[pd.to_datetime(raw_polls["date_fin"]) >= MIN_POLL_DATE].copy()
    candidates, slot_of, slot_names = build_roster(raw, as_of=as_of)
    arrays = build_poll_arrays(raw, candidates)
    as_of_ord = pd.Timestamp(as_of).toordinal()
    kappa = make_kappa(arrays["dates"], as_of=as_of_ord, half_life=half_life_days)

    if excess_bank is None:
        from model.models.spatial_pooling.calibration import BANK_EXCESS_PATH
        excess_bank = Bank.load(BANK_EXCESS_PATH)
    excess_var = excess_var_for_nodes(arrays["instituts"], excess_bank)

    prior_pos = slot_prior_positions(slot_names) if use_position_prior else None
    kwargs = dict(
        slot_of=jnp.asarray(slot_of), n_slots=len(slot_names),
        tested_mask=jnp.asarray(arrays["tested_mask"]), Y=jnp.asarray(arrays["Y"]),
        Np=jnp.asarray(arrays["Np"]), kappa=jnp.asarray(kappa), excess_var=jnp.asarray(excess_var),
        slot_prior_pos=None if prior_pos is None else jnp.asarray(prior_pos),
    )
    if blocked:
        # NON RECOMMANDÉ en production (spec §12.11) : converge sur le roster
        # 2027 (78 nœuds) mais PAS sur les fits historiques 2022 (234 nœuds,
        # R-hat 2,35 même à `rho`/`delta` fixés). Retirer la déflation multiplie
        # l'information par 7,6 et rend la vraisemblance trop piquée pour un
        # modèle qui ne reproduit pas les sondages à cette précision. Gardée
        # pour reprise éventuelle, désactivée par défaut.
        # Vraisemblance à erreurs corrélées entre hypothèses d'un même sondage
        # + écart de modèle (spec §12.7) : les hypothèses partagent les mêmes
        # répondants, donc leur DIFFÉRENCE -- le signal qui identifie la
        # géométrie -- est bien plus précise que ne le suppose un traitement
        # indépendant à échantillon déflaté.
        kwargs.update(notice_idx=jnp.asarray(arrays["notice_idx"]),
                      n_notices=int(arrays["n_notices"]),
                      Np_full=jnp.asarray(arrays["Np_full"]), rho_hyp="sample")
    t0 = time.time()
    samples, extra = run_numpyro_mcmc(spatial_pooling_model, kwargs, draws=draws, tune=tune,
                                      chains=chains, seed=seed, target_accept=target_accept,
                                      extra_fields=("diverging",))
    elapsed = time.time() - t0

    N = len(candidates)
    diag = numpyro_summary(samples, prob=0.9)
    diagnostics = {
        "half_life_days": half_life_days,
        "n_noeuds": int(arrays["tested_mask"].shape[0]),
        "n_candidats": N,
        "temps_secondes": round(elapsed, 1),
        "rhat_max": round(max(float(np.max(d["r_hat"])) for d in diag.values()), 4),
        "ess_min": round(min(float(np.min(d["n_eff"])) for d in diag.values()), 1),
        "n_divergences": int(np.sum(extra["diverging"])),
        "excess_var_moyenne_pt": round(float(np.mean(excess_var)) * 10000, 3),   # fraction² -> pt²
        "excess_bank_chargee": excess_bank is not None,
        "vraisemblance": "bloquee+delta" if blocked else "independante",
    }
    for site in ("rho_hyp", "sigma_model"):
        if site in samples:
            diagnostics[site] = round(float(np.mean(np.asarray(samples[site]))), 4)
    mu_draws = np.asarray(samples["mu"]).reshape(-1, N)
    sigma_draws = np.asarray(samples["sigma"]).reshape(-1, N)
    w_draws = np.asarray(samples["w_now"]).reshape(-1, N)
    w_source = "tempered"

    if w_dynamics:
        from model.models.spatial_pooling.w_dynamics import (build_pseudo_observations, load_w_law,
                                                             w_draws_ou)
        law = load_w_law()
        sel = np.linspace(0, len(mu_draws) - 1, min(n_geom, len(mu_draws))).astype(int)
        t0 = time.time()
        pobs = build_pseudo_observations(mu_draws[sel], sigma_draws[sel], arrays["tested_mask"],
                                         arrays["Y"], arrays["Np"], arrays["dates"],
                                         arrays["instituts"], excess_var)
        w_draws = w_draws_ou(pobs, as_of=float(as_of_ord), sigma_w2=law["sigma_w2"],
                             tau_w=law["tau_w"], sigma_house=law.get("sigma_house", 0.0),
                             seed=seed, n_draw_per_geom=5)
        mu_draws, sigma_draws = np.repeat(mu_draws[sel], 5, axis=0), np.repeat(sigma_draws[sel], 5, axis=0)
        w_source = "ou"
        # `gap_jours` : âge du dernier sondage AYANT TESTÉ chaque candidat --
        # c'est lui, et non l'âge du sondage le plus récent tous candidats
        # confondus, qui pilote le plancher de variance de ce candidat.
        tested_any = arrays["tested_mask"] > 0
        gaps = [float(as_of_ord - arrays["dates"][tested_any[:, i]].max()) if tested_any[:, i].any()
                else float("nan") for i in range(N)]
        diagnostics.update({
            "w_source": "ou",
            "w_sigma_w2": round(float(law["sigma_w2"]), 4),
            "w_tau_w": round(float(law["tau_w"]), 1),
            "w_temps_secondes": round(time.time() - t0, 1),
            "w_gap_jours": {c: g for c, g in zip(candidates, gaps)},
        })

    return SpatialPoolingFit(
        candidates=candidates, slot_of=slot_of, slot_names=slot_names,
        mu_draws=mu_draws, sigma_draws=sigma_draws, w_draws=w_draws,
        diagnostics=diagnostics, w_source=w_source,
    )


def pi_draws_for_mask(fit: SpatialPoolingFit, mask: np.ndarray) -> np.ndarray:
    """LE point de cette architecture (spec §7) : pi pour un scénario
    ARBITRAIRE (ex. coché par un utilisateur sur le site), calculé en poussant
    CHAQUE tirage postérieur déjà en mémoire à travers le softmax masqué --
    pas de ré-inférence, O(tirages x B). `mask` : (N,) 0/1, quels candidats
    du scénario sont "en lice". Retourne (S, N) -- une composition (sur les
    candidats du masque) par tirage, comme `Nowcast.draws` (model/core/base.py).

    C'est aussi ce qui porte l'incertitude correctement : chaque tirage vient
    du postérieur JOINT (mu, sigma, w), qui s'élargit déjà tout seul quand les
    sondages récents d'un candidat se contredisent (pas besoin d'un mélange
    explicite comme linear_pooling -- cf. spec section "Incertitude" pour la
    comparaison et ses limites connues)."""
    return np.asarray(spatial_shares(jnp.asarray(fit.mu_draws), jnp.asarray(fit.sigma_draws),
                                     jnp.asarray(fit.w_draws), jnp.asarray(mask)))


def summarize_pi(pi_draws: np.ndarray, labels: list[str]) -> dict:
    """(S,N) -> {label: {mean, ic90}} -- même résumé que `Nowcast.summary()`
    (model/core/base.py), réutilisable indépendamment du contrat final."""
    out = {}
    for i, label in enumerate(labels):
        vals = pi_draws[:, i]
        out[label] = {
            "mean": round(float(vals.mean()), 4),
            "ic90": [round(float(np.percentile(vals, 5)), 4), round(float(np.percentile(vals, 95)), 4)],
        }
    return out


# --- Saut terminal (nowcast -> scrutin) : machinerie PARTAGÉE, plus rien à faire ici --
# `model/core/projection.py::projeter_au_scrutin` + `model/core/opinion.py::load_law()`
# -- diffusion d'opinion (Ornstein-Uhlenbeck) puis écart sondages-urne δ,
# calibrés une seule fois pour tout le dépôt (`bank_opinion.json`), pas par
# modèle. `spatial_pooling` n'a plus de saut/Bank qui lui soit propre (l'ancien
# `TerminalJumpCalibration`/`BANK_JUMP_PATH` par modèle a disparu avec la
# restructuration -- cf. `bayesian_nowcast`/`gp_pooling`, même pattern partout).
def forecast_spatial_pooling(fit: SpatialPoolingFit, mask: np.ndarray, horizon_days: int) -> dict:
    """Projette un scénario (`mask`, ex. coché par un utilisateur) jusqu'au
    scrutin. `pi_draws_for_mask` donne l'incertitude au nowcast (spec section
    "Incertitude") ; `projeter_au_scrutin` ajoute la dérive résiduelle d'ici
    au scrutin, EXACTEMENT comme les autres modèles -- rien de spécifique à
    spatial_pooling au-delà de fournir `pi` dans le bon format."""
    from model.core.opinion import load_law
    from model.core.projection import projeter_au_scrutin

    mask_bool = mask.astype(bool)
    labels = [c for c, keep in zip(fit.candidates, mask_bool) if keep]
    pi = pi_draws_for_mask(fit, mask)[:, mask_bool]
    pi = projeter_au_scrutin(pi, horizon_days, load_law())
    return forecast_from_draws(pi, labels, horizon_days)
