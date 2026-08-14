# Le SSM du nowcast bayésien — spec consolidée

Fusion de `spec_ssm_nowcast.md`, `spec_ssm_espace_latent.md`,
`spec_ssm_encodage_partiel.md`, `spec_ssm_implementation.md` et
`spec_ssm_tendance.md` en un seul document. Statuts conservés tels quels
(**[TRANCHÉ]**, **[OUVERT]**, proposition) : ce document ne re-tranche rien,
il regroupe. Décrit à la fois la conception (pourquoi) et l'implémentation
telle qu'elle existe dans
`model/models/bayesian_nowcast/{latent,calibration,nowcast}.py`.

## 1. Contexte et objectif

Le nowcast traitait auparavant l'opinion comme une **quantité statique
unique** `pi` : tous les sondages disponibles jusqu'à `as_of`, qu'ils datent
d'hier ou de 4 ans, étaient des observations bruitées de la MÊME valeur —
aucune pondération temporelle. Constaté concrètement sur le RN mi-2026 : des
sondages 2022/2023 à 25-32 % tiraient encore le nowcast vers le bas malgré
des sondages récents à 34-37 %. La dérive jusqu'au scrutin était un second
étage disjoint, un saut ponctuel (bootstrap sur `clr_movement_pool`), pas une
trajectoire.

Remplacé par un **State Space Model (SSM)** — état latent indexé par le
temps, lissé entre deux sondages, extrapolé un peu au-delà du dernier, puis
un saut terminal (distinct) jusqu'au jour du scrutin.

**Statut global : implémenté.** 34/34 tests passent (`tests/test_latent.py`,
`tests/test_terminal_jump.py`, `tests/test_calibration.py`,
`tests/test_models.py`, ...), calibration de production relancée avec succès
(R-hat max 1.005, ESS min 913).

## 2. Architecture : deux étages

### 2.1 Le chemin (nowcast + lissage) — linéaire-gaussien **[TRANCHÉ]**

- État `alpha_t ∈ R^(K-1)` (paramétrisation ILR, §3), à des **nœuds
  temporels** (un nœud par date de sondage distincte), pas un état par jour
  calendaire — chaque nœud additionnel est quasi gratuit avec le
  filtre/smoother de Kalman (O(nœuds)), contrairement à un état-par-jour
  naïf en NUTS.
- Transition : marche aléatoire gaussienne, `alpha_t = alpha_{t-1} + eps_t`,
  `eps_t ~ Normal(0, τ² · Δt · I)`.
- Observation : chaque sondage observe l'état au nœud de sa propre date,
  `y_i ~ Normal(f(alpha_{t_i}) · 100 + biais[institut,bloc], Σ_i)`.
- **Reste gaussien du début à la fin** — ce qui permet la marginalisation
  fermée (§4) et évite d'exploser en nombre de paramètres latents à
  échantillonner en NUTS.

### 2.2 Le saut terminal (nowcast → jour du scrutin) — non-gaussien, distinct **[TRANCHÉ]**

Remplace `clr_movement_pool` (bootstrap empirique, 88 points seulement) par
un ajustement paramétrique **sinh-arcsinh d'une normale** (Jones & Pewsey
2009) sur les mêmes mouvements historiques :

```
z ~ Normal(0, 1)
x = sinh( (asinh(z) + skew) / tail )
move = loc + scale · x
```

Appliqué **une seule fois par tirage postérieur**, pas à chaque pas de la
marche — point clé pour la tractabilité (§4) et parce que l'évidence
empirique le justifie (le saut de fin de campagne est la vraie source de
dispersion, pas l'accumulation jour après jour).

`skew`/`tail` **globaux** (partagés entre blocs), `loc`/`scale` **par bloc**
**[TRANCHÉ]** : avec seulement 2 élections historiques (2017/2022), un
ajustement par bloc de 4 paramètres de forme risquerait de capturer les
particularités statistiques de CES deux campagnes plutôt que des
fondamentaux généralisables. `loc`/`scale` sont des moments d'ordre plus
bas, moins gourmands en données (déjà par bloc via
`campaign_drift`/`campaign_drift_sd`).

Pourquoi séparé du chemin plutôt que fusionné dans le même SSM
linéaire-gaussien : le saut terminal est délibérément non-gaussien, ce qui
casserait la marginalisation fermée par filtre de Kalman qui fait tout
l'intérêt calculatoire de l'architecture.

## 3. Espace latent : CLR / ILR / ALR — pourquoi ILR

`K` = nombre de catégories. `pi` (le nowcast) vit à la granularité **slot**
(`K=12`, `model/core/live_dataset.py`, `SLOTS`) ; `institut_bias` et
`campaign_drift` vivent à la granularité **bloc** (`K_bloc=6`,
`calibration.py`). Le SSM garde `alpha_t` au niveau slot avec le biais bloc
rajouté par-dessus.

### 3.1 Le simplexe et pourquoi on ne travaille pas dessus directement

Une composition vit dans le simplexe `S^K = {x ∈ R^K : x_i > 0, Σ x_i = 1}`.
Ce n'est pas un espace vectoriel, donc y mettre directement une marche
aléatoire gaussienne n'a pas de sens — d'où le détour par un espace
log-ratio, où les opérations vectorielles standard (somme, marche gaussienne,
filtre de Kalman) redeviennent légitimes.

### 3.2 CLR et la géométrie d'Aitchison

`CLR(x)_i = log(x_i) − mean_j(log x_j)`. Deux faits structurent tout le
reste :

**(a) CLR est une isométrie.** Le simplexe muni de la métrique d'Aitchison
(Aitchison 1986) — invariante par perturbation/mise à l'échelle — est
isométrique à l'hyperplan `H = {v ∈ R^K : Σ v_i = 0}` muni de la métrique
euclidienne usuelle, via CLR : `dist_Aitchison(x,y) = ‖CLR(x) − CLR(y)‖₂`.
C'est ce qui justifie la géométrie euclidienne standard (marche gaussienne
isotrope, filtre de Kalman) *après* le passage par CLR — à condition de
rester dans `H`.

**(b) CLR brut est non identifié.** `CLR(x)` vit toujours dans `H`, de
dimension `K−1`, pas `K`. Modéliser directement `K` coordonnées CLR avec une
marche gaussienne pleine donne un degré de liberté à une direction plate :
`softmax(a + c·1) = softmax(a)` pour tout `c`. La vraisemblance est donc
strictement plate le long de `1 = (1,...,1)` : matrice d'information de
Fisher singulière → covariance singulière, mauvais mélange NUTS, divergences.
D'où la nécessité d'une paramétrisation à `K−1` coordonnées *libres* : ILR
ou ALR.

### 3.3 ILR : base orthonormée de `H`

Soit `V` une base orthonormée de `H` (`K×(K−1)`, `VᵀV = I_{K−1}`).
`alpha = Vᵀ · CLR(x) ∈ R^{K−1}` ; reconstruction `x = softmax(V · alpha)`.

Base de Helmert (choix « par défaut » sans hiérarchie naturelle entre
catégories) :

```
v_k[i] = 1/sqrt(k(k+1))   pour i ≤ k
v_k[i] = -k/sqrt(k(k+1))  pour i = k+1
v_k[i] = 0                sinon
```

**Propriété clé** : `V` étant orthonormée, `‖alpha_1 − alpha_2‖₂ =`
distance d'Aitchison`(x_1, x_2)`. Donc **une marche gaussienne isotrope sur
`alpha` correspond exactement à une marche isotrope au sens d'Aitchison sur
le simplexe** — aucune direction n'est structurellement privilégiée ou
pénalisée.

```python
def helmert_basis(K: int) -> np.ndarray:
    V = np.zeros((K, K - 1))
    for k in range(1, K):
        V[:k, k - 1] = 1.0 / np.sqrt(k * (k + 1))
        V[k, k - 1] = -k / np.sqrt(k * (k + 1))
    return V
```

`V` est une constante calculée une fois à l'initialisation — coût nul en
boucle NUTS/Kalman.

### 3.4 ALR : coordonnée de référence, et pourquoi ce n'est *pas* une isométrie

Fixer une référence `r`. `ALR(x)_i = log(x_i/x_r)` pour `i ≠ r`
(`K−1` coordonnées), `alr = A · clr(x)`, `A` de taille `(K−1)×K`.

`A` restreinte à `H` est une bijection linéaire (donc un système de
coordonnées valide), mais **pas une isométrie** dès que `K ≥ 3`. Diagonaliser
`AᵀA` sur `H` donne deux valeurs propres : `K` (multiplicité 1, direction
« `r` contre la moyenne symétrique du reste ») et `1` (multiplicité `K−2`,
contrastes entre non-référence).

**Conséquence : un bruit isotrope `Normal(0,σ²I)` sur les coordonnées ALR
induit, dans la géométrie d'Aitchison réelle, un bruit anisotrope avec un
facteur `1/K` sur l'axe « référence contre reste »** — cette direction est
artificiellement `K` fois plus « raide » que les autres.

Exemple `K=3` (RN, LR, Centre, référence = Centre) : facteur d'anisotropie
`3`. Avec `K=12` (slots) le facteur serait `12` ; avec `K=6` (blocs), `6`.
Concrètement : si "Autres"/écologistes est choisi comme référence, les
mouvements « écologistes contre la moyenne symétrique des autres » seraient
artificiellement `K` fois plus contraints — un effet de bord du choix de
coordonnées, pas une hypothèse de modélisation voulue.

### 3.5 Pourquoi ça touche `τ`

`campaign_drift_sd` et `clr_movement_pool` (`calibration.py`) sont **déjà
calculés en CLR centré** — donc dans la même famille géométrique qu'ILR (ILR
n'en est qu'une reparamétrisation orthonormée, qui préserve les distances).
Si le chemin tournait en ALR pendant que le saut terminal et sa calibration
historique restent en CLR, `τ` (chemin) et l'écart-type du saut terminal ne
mesureraient plus la même chose : `τ` en ALR serait contaminé par le facteur
`1/K` selon la référence choisie. Risque concret de bug de cohérence, pas
juste une question d'élégance.

### 3.6 Ce qui ne dépend pas du choix ALR/ILR

Le floor numérique sur les parts de sondage avant `log()` (`np.clip(v,
1e-4, None)`, évite l'explosion quand une part est à 0%) est nécessaire dans
les deux cas — pas un argument pour choisir l'un ou l'autre.

### 3.7 Synthèse des compromis

| | ALR | ILR |
|---|---|---|
| Code | Plus simple (pas de matrice de base) | Construction de `V` une fois (formule fermée) |
| Isotropie | Non — facteur `1/K` sur l'axe référence/reste | Oui, exacte |
| Cohérence avec `campaign_drift_sd`/`clr_movement_pool` | Décalage géométrique à corriger explicitement | Directe (même famille que CLR) |
| Interprétation de `τ` | Dépend du bloc de référence | Basis-free, comparable entre runs |
| Risque si mauvais choix de référence | Contamine spécifiquement le bloc référence (éviter RN) | N/A |

### 3.8 Décision **[TRANCHÉ]**

**ILR retenu**, sur base mathématique (isométrie exacte, cohérence
géométrique avec `campaign_drift_sd`/`clr_movement_pool` existants), pas sur
une préférence empirique — pas d'interface ALR/ILR swappable construite.

## 4. Formulation mathématique complète (implémentation)

### 4.1 Notation

- `K = 12` slots (`model/core/live_dataset.py:SLOTS`).
- `V ∈ R^{K×(K-1)}` : base de Helmert (`latent.py:helmert_basis`), `VᵀV = I_{K-1}`.
- `alpha_t ∈ R^{K-1}` : état latent (coordonnées ILR) au nœud `t`.
- `clr(x) = V·alpha` reconstruit les coordonnées CLR ; `pi = softmax(clr(x))`
  reconstruit la composition.
- Un **nœud** = un sondage, trié par date (`nowcast.py:NowcastData`), pas un
  pas de temps calendaire.

### 4.2 Le chemin

```
alpha_t = alpha_{t-1} + eps_t,   eps_t ~ N(0, Q_t)
Q_t = tau² · Δt_t · I_{K-1}          (pas de plafond, cf. §6.2)
```

`Δt_t` = écart en jours vers le nœud suivant, sur les sondages **depuis
`MIN_POLL_DATE = 2026-01-01` seulement** (§6.2). `tau ~
HalfNormal(tau_prior)`, `tau_prior` = `campaign_drift_sd` calibré sur
2017/2022 si une Bank existe, sinon un repli faible
(`nowcast.py:DEFAULT_SSM_PRIORS_CFG`). État initial :

```
alpha_0 ~ N(historical_prior_mean, C · I_{K-1})
```

`historical_prior_mean = ilr_encode(historical_prior_pi(slots), V)` — **pas**
`N(0, ...)` (§6.3 : pourquoi une moyenne nulle est elle-même un mauvais
prior). `C = 1.0` (§6.1 pour pourquoi pas 25).

### 4.3 Observation d'un sondage — encodage partiel

Un sondage 2027 ne teste **jamais** les 12 slots simultanément (7 à 11 sur
12 dans les données actuelles). Pour un sondage testant `S ⊆ {1..K}`
(`|S| = m`), avec référence locale `r = argmax_{i∈S}(part_i)` :

```
y_j = log(p_j / p_r)   pour j ∈ S \ {r}     (m-1 valeurs réelles)
H_{j,:} = V[j,:] − V[r,:]                    (m-1)×(K-1), linéaire en alpha
Σ_{j,j} = 1/(n·p_j) + 1/(n·p_r)
Σ_{j,k} = 1/(n·p_r)   (j≠k)                  delta-method multinomial-log-ratio
```

où `p` = parts renormalisées sur `S` (débiaisées du `biais[institut,bloc]`
échantillonné, **avant** la transformation). Paddé à taille fixe `K-1` :
`H=0`, `y=0`, `Σ_{diag}=10⁶` sur les dimensions non testées — contribution
EXACTEMENT constante à la vraisemblance, donc neutre, pas une approximation
(`latent.py:poll_observation_values`).

Voir §5 pour la justification du choix de référence `argmax` (invariance
prouvée) et pour la variante `full_covariance=False` (`R=Id`, ILR local)
devenue le nouveau défaut.

### 4.4 Marginalisation par filtre de Kalman

Le chemin est un SSM linéaire-gaussien standard : NumPyro échantillonne
`tau` et `biais` par NUTS ; pour un tirage donné,
`model.core.lgssm.kalman_filter` calcule la vraisemblance marginale des
sondages avec le chemin latent intégré analytiquement, injectée via
`numpyro.factor("chemin_marginal", ...)`. Aucun `alpha_t` par nœud n'est
échantillonné individuellement — coût O(nœuds), pas de paramètre latent
supplémentaire pour NUTS.

**[TRANCHÉ]** : la marginalisation analytique par filtre de Kalman est choisie
pour cette raison précise (pur JAX, différentiable), pas pour remplacer
NumPyro/NUTS qui garde le rôle d'orchestrateur des hyperparamètres.

**[RÉVISÉ le 2026-08-14]** : ce filtre venait de `dynamax`, dont c'était le seul
usage du dépôt. dynamax tire tensorflow-probability, dont la dernière version
(0.25) casse à partir de jax 0.7 alors que numpyro >= 0.20 exige jax >= 0.7 :
deux bornes inconciliables qui enfermaient ce modèle dans un environnement figé.
Le filtre est réimplémenté dans `model/core/lgssm.py` (50 lignes de `lax.scan`,
mêmes conventions), vérifié identique à dynamax au bit près sur la
log-vraisemblance et les covariances, et testé contre la marginale gaussienne
exacte dans `tests/test_lgssm.py`. Le smoother n'a jamais été utilisé : si
l'affichage d'une courbe lissée le redemande un jour, il s'ajoutera au même
endroit (RTS backward, ~20 lignes de plus).

### 4.5 Extrapolation jusqu'à `as_of`

Un pas de plus au-delà du dernier sondage, même loi, pas de nouvelle
observation :

```
alpha_now ~ N(alpha_filtré_dernier, Σ_filtrée_dernière + tau²·Δt_as_of·I)
pi_now = softmax(V · alpha_now)             # numpyro.deterministic("pi", ...)
```

`pi_now` (un tirage par échantillon postérieur de `(tau, biais)`) est ce que
`BayesianNowcast.nowcast()` renvoie comme tirages du nowcast.

### 4.6 Le saut terminal — dépendance à l'horizon

En espace CLR par-candidat (pas dans l'espace ILR abstrait, cf. §2.2) :

```
move_href ~ SinhArcsinh(loc[bloc], scale[bloc], skew, tail)   (TFP, par candidat)
move = move_href · sqrt(horizon_forecast / horizon_ref)
theta = softmax(log(pi_now) + move)
```

`horizon_ref` = horizon moyen du movement pool 2017/2022 (~150 jours),
stocké dans la Bank (`jump_horizon_ref`) ; le fit est réalisé sur les
mouvements renormalisés à `horizon_ref` (`TerminalJumpCalibration.calibrate`
divise par `sqrt(horizon/horizon_ref)` avant de fitter) — un mouvement
mesuré à J-400 et un mouvement à J-50 ne sont pas la même quantité, la
dérive continue jusqu'au scrutin.

## 5. Encodage des sondages partiels : le choix de référence locale ne biaise personne

Hypothèse soulevée en session : le choix de la référence locale
(`r_local = argmax` des parts testées) privilégierait structurellement le
RN, presque toujours le candidat le plus haut testé, en ne l'observant
jamais « directement » — seulement via les ratios des autres candidats.

**Conclusion : hypothèse réfutée mathématiquement et empiriquement.** Le
vrai mécanisme de la dérive observée sur RN est ailleurs (§5.4/§6) : la
covariance jointe entre candidats, pas le choix de référence.

### 5.1 Ce que fait le code, et pourquoi ce n'est PAS une approximation

`poll_plan` choisit `r = argmax_{i∈T} shares_i` et encode `y_o = log(p_o /
p_r)` pour chaque autre candidat testé `o`. Comme `p_i = π_i / S_T` avec
`S_T` la même constante pour tout `i ∈ T`, `S_T` s'annule exactement dans le
ratio :

```
log(p_o/p_r) = log(π_o) - log(π_r) = CLR(π)_o - CLR(π)_r = (V[o,:] - V[r,:]) · alpha
```

`H_o · alpha` reconstruit **exactement** un vrai log-ratio de la composition
complète à 12 candidats — pas une approximation, pas un ratio « local » qui
dépendrait des candidats non testés.

### 5.2 Théorème d'invariance

Soit `y_ALR = A_r · CLR_T(p)` (ALR, référence `r`) et `y_ILR = W^T ·
CLR_T(p)` (ILR local, base orthonormée `W`). Les deux sont des coordonnées
linéaires **inversibles** du même vecteur `CLR_T(p)` (espace à `m-1`
dimensions) : `y_ALR = M · y_ILR` pour une matrice inversible `M`. Si la
covariance est le pushforward exact du même modèle de bruit (`R_ALR = M ·
R_ILR · M^T` — vérifié pour le code actuel, `Sigma_real ∝ 1/p_ref`, dérivée
standard delta-method), alors la vraisemblance gaussienne en fonction de
`alpha` est identique à une constante additive près (indépendante de
`alpha`) :

```
N(y_ALR | H_ALR·alpha, R_ALR) ∝_alpha N(y_ILR | H_ILR·alpha, R_ILR)
```

**Le postérieur bayésien sur `alpha` (moyenne ET covariance) est EXACTEMENT
le même**, quel que soit `r` (ALR) ou `W` (ILR) choisi — identité algébrique,
pas une approximation qui se trouve être petite. Seule la constante additive
à `marginal_loglik` change (n'affecte pas l'inférence, seule une comparaison
brute du `marginal_loglik` entre deux ENCODAGES différents serait invalide —
cas qui ne se présente pas dans ce modèle).

### 5.3 Vérification numérique (deux checks indépendants)

**Synthétique** (8/12 slots testés, prior arbitraire, les 8 références
possibles) : écarts de moyenne/covariance/loglik tous `< 1.3e-7` (bruit
float64). RN décodé identique à `9.9887%` sur les 8 références.

**Cas réel de production** (`as_of=2026-03-27`, nœud Elabe 25-27 mars,
référence actuelle = RN, `tau` fixé à 0.03 hors NUTS pour isoler la
question) : RN passe de 31,01 % à 29,80 % à ce nœud (le -1,2 pt exact
rapporté en session). **Test décisif** : forcer la référence locale à
Mélenchon au lieu de RN pour ce nœud précis, puis rejouer toute la
trajectoire séquentielle : écart max sur RN sur tous les nœuds = **0,000e+00
point** ; écart max sur tous les slots = 1,9e-06 (bruit float32 JAX).
**Changer la référence locale, y compris pour LE nœud où RN était déjà la
référence, ne change RIEN.**

### 5.4 D'où vient alors la baisse de 1,2 point sur RN ? Le vrai mécanisme

Décomposition du même nœud (moyenne prédite juste avant le sondage vs parts
rapportées, renormalisées sur le même sous-ensemble testé) :

| candidat | prédit (%) | rapporté (%) | écart |
|---|---|---|---|
| RN | 45,19 | 43,31 | **-1,87** |
| Retailleau | 11,28 | 12,74 | **+1,46** |
| Glucksmann | 14,79 | 15,92 | **+1,13** |
| Dupont-Aignan | 2,80 | 3,18 | +0,38 |
| Mélenchon | 15,34 | 15,29 | -0,05 |
| Roussel | 3,30 | 3,18 | -0,11 |
| Arthaud | 1,45 | 1,27 | -0,18 |
| Tondelier | 5,86 | 5,10 | -0,76 |

Le sondage dit, essentiellement : **Glucksmann et Retailleau sont
sensiblement plus hauts, RELATIVEMENT à RN et aux autres, que ce que le
modèle croyait déjà** (résidus les plus grands du vecteur d'innovation).
Le filtre de Kalman doit satisfaire *simultanément* tous les ratios
rapportés, pondérés par la précision relative de chaque composante de
l'état — et RN, dénominateur commun de tous ces ratios, absorbe une partie
de l'ajustement nécessaire pour réconcilier « Mélenchon/RN quasi stable »
avec « Glucksmann/RN et Retailleau/RN en forte hausse », **via la
covariance JOINTE entre les composantes de `alpha`** (héritée des nœuds
précédents, en particulier de la fusion non-diagonale du 1er jour,
`fuse_first_date_observations` — source d'un effet de corrélation d'environ
-2,8 points sur les nœuds qui ne testent pas RN du tout). Ce n'est PAS un
artefact du choix de référence (réfuté au §5.3) : **c'est le comportement
correct d'une mise à jour bayésienne multivariée**, étant donné la
covariance actuelle du modèle. Que cette covariance soit elle-même bien
calibrée est une question différente et plus large (§5.6, [OUVERT]).

### 5.5 Pourquoi l'argmax est numériquement optimal (pas seulement arbitraire)

Condition number de `R` sur le nœud Elabe, pour les 8 choix de référence
possibles :

```
ref=Arthaud     (1,00%)  cond(R)=1.71e+02
ref=Roussel     (2,50%)  cond(R)=7.42e+01
ref=Dupont-A.   (2,50%)  cond(R)=7.42e+01
ref=Tondelier   (4,00%)  cond(R)=5.04e+01
ref=Retailleau (10,00%)  cond(R)=2.89e+01
ref=Glucksmann (12,50%)  cond(R)=2.64e+01
ref=Mélenchon  (12,00%)  cond(R)=2.68e+01
ref=RN         (34,00%)  cond(R)=1.27e+01   <- argmax, choix actuel
```

Cohérent avec `Sigma_real ∝ 1/p_ref` : le plus grand dénominateur minimise
le conditionnement. L'argmax est **le choix qui minimise le conditionnement
parmi toutes les alternatives**, et se trouve meilleur que l'ILR local
générique (Helmert) sur ce même exemple (12,7 vs 21,5).

**[TRANCHÉ]** (choix de référence locale) : `argmax` n'introduit AUCUN
biais et ne désavantage AUCUN candidat, y compris celui choisi comme
référence — prouvé algébriquement et vérifié numériquement à deux reprises,
y compris sur le cas réel exact qui a motivé l'investigation. L'ILR local
esquissé en alternative, bien que mathématiquement correct, ne changerait
aucun résultat et serait légèrement moins bien conditionné — non
implémenté sous cette forme (mais cf. §5.7, adopté pour une raison
différente).

### 5.6 [OUVERT] La covariance jointe est-elle bien calibrée ?

La baisse de 1,2 point observée sur RN vient de la covariance jointe du
modèle, pas de l'encodage — reste à déterminer si cette covariance
elle-même est bien calibrée. Pistes à investiguer séparément :
- l'ampleur de l'effet de corrélation non-diagonale hérité de
  `fuse_first_date_observations` (~-2,8 pts déjà mesuré séparément sur les
  nœuds sans RN testé) ;
- si la précision accumulée par candidat (Glucksmann/Retailleau très
  testés vs RN comparativement moins souvent avec des hypothèses
  complètes) crée une asymétrie légitime mais mal calibrée dans `tau` ou
  dans le prior initial ;
- observé en passant, pas creusé : le prior décodé pour Philippe (Horizons)
  au nœud Elabe est à 20 % — inhabituellement élevé, potentiellement un
  symptôme du même mécanisme sur un autre candidat.

Vérification proposée si repris : tracer, nœud par nœud, `Var(alpha_RN)` et
`Corr(alpha_RN, alpha_Glucksmann)`/`Corr(alpha_RN, alpha_Retailleau)` sur
tout le backfill 2026, avant/après une éventuelle correction de
`fuse_first_date_observations` ou de la structure de `tau`.

### 5.7 Addendum : simplification à `R=Id` + ILR local — implémenté

Décision séparée de l'utilisateur : pour un premier « modèle de base » (SSM
simple, sondage traité comme valide + incertitude d'échantillonnage lissée
par la marche aléatoire), **remplacer** le bruit d'observation delta-method
complet par `R = (deff/n)·I` — élimine la structure de corrélation `1/p_ref`
entre candidats testés (un des deux mécanismes de la dérive du RN, §5.4 ;
l'autre étant la covariance jointe héritée des nœuds précédents).

**Ce n'est pas la même décision que §5.5** (choix de référence LOCALE sous
covariance complète, resté neutre). Ici la covariance elle-même change de
forme, et dans ce cas précis **l'encodage local redevient pertinent** : un
`R=Id` n'est neutre entre candidats que sous une base ORTHONORMÉE (ILR
local), pas sous un changement de référence ALR (transformation non
orthogonale). D'où le passage à l'ILR local (`poll_plan`) **conjointement**
à la simplification de `R`.

**Implémenté** dans `latent.py`/`nowcast.py`, bascule explicite
`full_covariance` (`False` = nouveau défaut, ILR local + `R=Id` ; `True` =
ancien comportement, ALR local argmax + delta-method complet), exposée
comme paramètre (`NowcastData.from_dataframe`, `BayesianNowcast.full_covariance`)
et comme modèle de comparaison enregistré
(`bayesian-nowcast-covariance-complete`, `public=False`), même pattern que
`use_biais`/`use_tendance`.

**Vérification empirique** (nœud Elabe, `tau` fixé à 0,03, hors NUTS,
`as_of=2026-03-27`) : la simplification réduit la baisse du RN à ce nœud
mais ne l'élimine PAS — de -1,21 pt (`full_covariance=True`) à -1,03 pt
(`False`). Cohérent avec §5.4 : la corrélation `1/p_ref` intra-sondage
n'était qu'UNE partie du mécanisme ; la covariance JOINTE entre candidats
(héritée des nœuds précédents, PAS touchée par ce changement) reste active
et domine l'effet. Sur l'ensemble de la trajectoire (10 nœuds), les deux
modes convergent vers une estimation RN finale très proche (29,64 % vs
29,71 % à `as_of`) malgré des chemins nœud-par-nœud différents (jusqu'à
±0,77 pt d'écart en cours de route).

#### 5.7.1 Cause racine identifiée : ce n'est PAS un bug — c'est `SLOTS` qui écarte Bardella

Investigation complémentaire suite à une remarque utilisateur (« le tau
doit être une info par jour, la fréquence ne devrait pas trop jouer » +
« je veux que seuls les sondages pertinents soient utilisés ») : la
covariance jointe (§5.4/5.6) N'EST ELLE-MÊME QU'UN SYMPTÔME. La vraie cause
est en amont, dans `model/core/live_dataset.py` (`SLOTS`) — décision
délibérée, **confirmée et conservée** par l'utilisateur : un seul candidat
par bloc pour RN (Marine Le Pen), Jordan Bardella explicitement écarté
(« scénario d'inéligibilité »). Toute hypothèse de sondage testant Bardella
au lieu de Le Pen tombe sur `slot=None` et est silencieusement jetée par
`aggregate_to_slots`.

**Mesuré sur la fenêtre janvier-mars 2026** : Bardella est testé dans **16
hypothèses** (moyenne 36,6%), Le Pen dans seulement **4** — dont 2 le jour 1
(fusionnées dans `init_mean`/`init_cov`) et 2 au nœud Elabe. RN n'a donc,
dans TOUT le chemin séquentiel de ce backfill, qu'un seul moment de test
direct (le nœud Elabe) — d'où l'écart-type énorme (~17-18 points, mesuré dès
`init_mean`, avant le premier nœud séquentiel) et la forte corrélation à
tous les autres candidats.

**Vérifications faites (diagnostic seul, aucun changement de code)** :
- **Contrefactuel** : ré-encoder les lignes Bardella comme RN (localement,
  dans un script, PAS dans `SLOTS`) réduit la baisse au nœud Elabe de -2,37
  pt à -0,68 pt (`TAU_FIXED=0.10`), et l'écart final sur `as_of` de ~1,75 pt
  (29,7% réel vs 31,4% avec Bardella) — confirme que la majorité de l'effet
  vient de la donnée manquante, pas d'un défaut du filtre.
- **Santé numérique de la trajectoire RÉELLE (Le Pen seule)** : covariance
  SPD à chaque nœud, aucun NaN/Inf, et la variance de RN diminue
  STRICTEMENT à chaque fois qu'elle est réellement testée (sd 18,5 → 1,78 pt
  au nœud Elabe) — le lissage lui-même est correct, pas de bug.

**[TRANCHÉ]** : la baisse de RN observée au nœud Elabe et sa variance
élevée dans ce backfill sont la conséquence mathématique attendue d'avoir
très peu d'observations directes de Marine Le Pen (choix `SLOTS` confirmé,
pas de fusion/slot séparé pour Bardella) — **PAS un bug** du SSM, ni de la
covariance d'observation, ni de l'encodage ALR/ILR, ni de l'inférence NUTS
(R-hat/divergences sains dans les deux modes de covariance).

## 6. Diagnostics et corrections de l'implémentation

Historique : la variance de transition avait d'abord reçu un plafond ad hoc
(`min(tau²·Δt, C)`), **abandonné** — un patch sur le symptôme, pas la cause
(§6.2-6.3 : ce qui l'a réellement remplacé). §6.1 reste utile : la
démonstration de *pourquoi* une grande variance ILR casse le nowcast est
correcte et sert de base au diagnostic final, seule la conclusion
opérationnelle ("donc plafonner") a changé.

### 6.1 Pourquoi une grande variance ILR casse le nowcast

**Le mauvais argument (à corriger)** : « l'incertitude sur une composition
reste bornée, un simplexe n'a pas de variance infinie ». Faux comme
justification — `pi = softmax(clr)` est TOUJOURS un point valide du
simplexe, pour n'importe quel `clr ∈ R^K` fini. Le vrai problème est
ailleurs.

**Le vrai problème : dégénérescence de la loi de `pi` vers un sommet
aléatoire.** Soit `alpha ~ N(mu, σ²·I_{K-1})`, `clr = V·alpha`, `pi =
softmax(clr)`. Écrivons `clr = mu_clr + σ·Z`, `Z = V·ε`, `ε ~ N(0,
I_{K-1})`. `Cov(Z) = V·Vᵀ = I_K − (1/K)·𝟙𝟙ᵀ` (matrice de centrage,
indépendante du choix de base) — `Z` a la même loi que `η − mean(η)·𝟙` pour
`η ~ N(0, I_K)` iid, donc `Z` est **échangeable** : `argmax_i Z_i` est
uniformément distribué sur `{1,...,K}`, indépendamment de `mu`.

Pour `i ≠ argmax_j Z_j` (noté `i*`) : `clr_i − clr_{i*} = (mu_clr_i −
mu_clr_{i*}) + σ·(Z_i − Z_{i*})`. Le second terme est linéaire en `σ` et
strictement négatif ; le premier terme est borné, indépendant de `σ`. Donc
quand `σ → ∞`, `clr_i − clr_{i*} → −∞` pour tout `i ≠ i*`, et :

```
pi = softmax(clr) → e_{i*}      (le sommet i* du simplexe)
```

**`pi` converge en loi vers `Uniform{e_1, ..., e_K}`** — un sommet tiré
uniformément au hasard, indépendant de `mu`. `E[pi_i] → 1/K` (8,3 % pour
`K=12`) quand `σ → ∞`, quel que soit `i` : un petit candidat est tiré vers
le haut vers 1/K, un grand candidat vers le bas. C'est exactement le
symptôme observé : Arthaud (~1 % dans tous les sondages) à 15-17 % dans
plusieurs snapshots du backfill, RN (~32 % attendu) à 4-24 % selon la date.
Mécanisme précis : le postérieur de `tau` a une queue lourde sur certaines
dates (§6.2), donc une fraction non négligeable des tirages `(tau, biais)`
post-NUTS implique une `σ` assez grande pour approcher ce régime dégénéré ;
la moyenne arithmétique sur les tirages mélange alors des tirages sains et
des tirages quasi-uniformes-sur-un-sommet.

**Note honnête** : `C=25` (variance ILR, `σ=5` — utilisée comme prior du
premier nœud et ancien plafond de `Q_t`) n'était pas déjà dans un régime
parfaitement « sûr » — à `K=12`, `σ=5` en log-ratio commence à s'approcher
du régime où l'écart-type de `Z_{(1)} − Z_{(2)}` (premier gap entre
statistiques d'ordre) est du même ordre que `σ`, poussant déjà `pi` à être
plus concentré que ce que `mu` seul suggérerait. `C=25` reste un choix
pragmatique, pas une valeur dérivée d'un critère formel.

### 6.2 La vraie cause du grand écart : des sondages hors-scope, pas un défaut du modèle

Sans plafond, `Q_t = tau² · Δt_t` croît sans borne avec `Δt_t`. Le jeu de
données complet a un écart de **861 jours** entre le dernier sondage
« prémonitoire » (2023-10-20, hypothèse de champ 2027 très spéculative) et
la reprise des sondages de campagne réelle (2026-02-27). Sur ce pas précis,
un grand `tau` rend la transition presque non-informative sans coûter cher
en vraisemblance localement — alors que les pas courts et denses de 2026
(`Δt` souvent < 20 jours) préfèrent un `tau` petit. Un seul `tau` global
devant satisfaire les deux régimes, le postérieur explorait une région où
`tau` est anormalement grand (`tau` moyen observé 0.545, écart-type 0.527,
contre un optimum de vraisemblance à `tau≈0.21` à biais fixé — 0 divergence
NUTS, donc pas un problème de mixing : la géométrie du problème admettait
cette région).

**Correction retenue, à la racine plutôt qu'en aval** (retour utilisateur :
un plafond de variance est ad hoc, *"le modèle doit être fonctionnel"*) :
les sondages antérieurs à `MIN_POLL_DATE = 2026-01-01` sont **écartés**
(`nowcast.py:NowcastData.from_dataframe`). Les sondages 2022/2023 mesurent
un régime différent (champ hypothétique, contexte d'avant-campagne) — les
mélanger avec 2026 sous un seul `tau` mélange deux processus génératifs
distincts. Une fois ce filtre appliqué, le plus grand écart résiduel entre
deux sondages consécutifs tombe à ~34 jours (`tests/test_nowcast_ssm.py`) :
`tau` redevient serré (0.030 ± 0.007, mesuré sur 2026-07-10) sans aucun
plafond.

### 6.3 Le prior initial : la moyenne était fausse aussi, pas seulement la variance

`alpha_0 ~ N(0, C·I)` centre le premier nœud sur des parts égales entre les
12 slots (`mu=0` en ILR ⟺ composition uniforme après `softmax`). Faux comme
prior, pas seulement large — on connaît déjà, avant tout sondage 2027,
l'ordre de grandeur des rapports de force via le résultat 2022. Symptôme
concret : **Poutou (NPA) n'est testé dans AUCUN sondage 2026** — son
estimation ne dépend donc QUE du prior initial. Avec `mu=0`, sa part se
reconstruit à partir de bruit essentiellement non informatif (§6.1) : 17 %
en moyenne postérieure sur 2026-07-10, contre ~1 % réel en 2023.

**Correction** : `historical_prior_pi` construit `mu` à partir des résultats
2022, par candidat mappé à son analogue 2027 (`HISTORICAL_2022_ANALOG` —
mapping explicite nom-à-nom, RN←Le Pen, Centre←Macron, etc.) ; un slot sans
analogue 2022 (Ruffin, nouvel entrant) reçoit un plancher
(`NEW_ENTRANT_FLOOR = 1.0` point, avant renormalisation à 100). Pas un
prior basé sur le premier sondage 2027 lui-même (écarté — circulaire).

Avec cette moyenne corrigée mais une covariance encore isotrope, Poutou
retombait à ~4 % — mieux que 17 %, mais encore ~4× son niveau réel (§6.4
corrige la covariance).

### 6.4 La covariance du prior devait être proportionnelle, pas isotrope

Une variance ILR isotrope traite RN et Poutou de façon identique en
coordonnées log-ratio, mais pour un vrai sondage de taille `n`, la
delta-method donne `Var(log p) ≈ 1/(n·p)` — plus grande pour un petit `p`.
Le prior isotrope (`C` constant) n'imite ni ce régime ni le régime en `p`
correctement.

**Correction** : traiter le prior 2022 comme un **sondage virtuel** de
taille effective `n` (`HISTORICAL_PRIOR_N = 300`, `historical_prior_cov`),
réutilisant la même formule delta-method qu'un vrai sondage. Dérivation :
`Cov(log p)_{ij} ≈ δ_ij/(n·p_i) − 1/n` ; en notant `P = V·Vᵀ = I_K −
(1/K)·𝟙𝟙ᵀ` le projecteur sur l'hyperplan à somme nulle, `CLR(p) =
P·log(p)`, donc `Cov(CLR) = P·Cov(log p)·P`. Comme `P·𝟙 = 0`, le terme
`−1/n` s'annule sous projection, et `Cov(CLR) = P·diag(1/(n·p))·P`. Avec
`alpha = Vᵀ·CLR(p)` et `P·V = V` :

```
Cov(alpha_ILR) = Vᵀ · diag(1/(n·p)) · V
```

— formule fermée, sans paramètre libre au-delà de `n`. Résultat mesuré sur
2026-07-10 : Poutou passe de 4,08 % à **0,85 %** (médiane 0,69 %) — cohérent
avec ~1 % réel, moyenne et médiane redeviennent quasiment identiques (l'effet
Jensen du §6.5 s'efface quand la variance résiduelle est correctement
dimensionnée).

### 6.5 Moyenne vs médiane : pourquoi la moyenne reste la bonne statistique

Tentative écartée : remplacer la moyenne postérieure par la médiane dans
`Nowcast.summary()`/`forecast_from_draws` pour absorber l'asymétrie de
`softmax` (effet Jensen : `E[softmax(X)] > softmax(E[X])` pour un petit
candidat). Casse le contrat de sortie : `Σ_c pi_c = 1` exactement à chaque
tirage, donc `Σ_c E[pi_c] = 1` (linéarité de l'espérance) — mais `Σ_c
médiane(pi_c) ≠ médiane(Σ_c pi_c)` en général. Mesuré : `validate_snapshot`
exige que les parts sur `forecast_scrutin` somment à 1 à ±10 % près ; avec
la médiane, la somme tombait à 0.88 sur un cas réel — détecté par
`test_modele_respecte_le_contrat`. La moyenne arithmétique est la SEULE
statistique de résumé par-candidat qui préserve cette propriété. La bonne
réponse à l'asymétrie est de réduire la variance quand elle est injustifiée
(§6.4), pas de changer la statistique de résumé.

### 6.6 `biais[institut,bloc]` désactivé par défaut

Régression trouvée sur `2026-04-30` (5 sondages seulement, Δt max 34 jours —
donc pas le problème du grand écart de §6.2) : Arthaud ressortait encore à
14,9 %, `tau` à 0,62 ± 0,60. Cause : `biais` a un paramètre par groupe
(institut,bloc) rencontré (jusqu'à ~44 sur l'historique complet, ~10-20
selon la date en backfill) — avec 5 sondages disponibles, le postérieur
joint `(tau, biais)` est structurellement sous-déterminé. Mesuré : sans
`biais`, le même run passe de `tau=0,62±0,60` à `tau=0,042±0,014`, 0
divergence, tous les candidats plausibles, et le temps de calcul chute de
~35-90s à **~12s** (le nombre de paramètres NUTS domine largement le temps,
pas la taille du chemin).

**Décision** (retour utilisateur : les biais institut ne ressortent
généralement pas significatifs) : `use_biais = False` par défaut sur
`BayesianNowcast` — `biais` n'est même plus échantillonné par NUTS
(`debiased = raw_values` directement). Variante enregistrée séparément,
`bayesian-nowcast-avec-biais-institut` (`use_biais = True`), disponible pour
exploration ponctuelle, documentée comme non fiable à faible volume de
sondages.

### 6.7 `init_to_median` : une chaîne NUTS peut rester figée à une initialisation aberrante

Régression trouvée après §6.6 (`use_biais=False` déjà en place) : le
snapshot `2026-07-08` restait cassé (RN 20,6 %, Poutou 21,3 %) alors que
`2026-07-09` et `2026-07-10` (presque les mêmes données) étaient sains.
Diagnostic par chaîne : sur 4 chaînes, **2 étaient figées EXACTEMENT** à
leur valeur d'initialisation tout le run (`tau` = 5,09 et 3,15, écart-type
**0,0000** sur 500 tirages — un blocage total), pendant que les 2 autres
exploraient correctement la région raisonnable (`tau≈0,028`). La
vraisemblance marginale décroît strictement et fortement avec `tau` (-239 à
`tau=0,02` contre -428 à `tau=1,0`, testé indépendamment) — donc pas un
problème de forme de la vraisemblance, un pur problème d'initialisation :
`init_to_uniform` (défaut NumPyro) tire un point de départ uniforme dans
l'espace non contraint, ce qui peut placer `tau` très loin dans la queue
d'un `HalfNormal` ; à cette échelle, le gradient de la vraisemblance à
travers le filtre de Kalman peut s'annuler numériquement — NUTS n'a alors
aucun signal pour s'échapper de son point de départ.

**Correction** : `run_numpyro_mcmc` (`model/core/inference.py`, partagé par
TOUS les modèles) utilise `init_strategy=init_to_median` par défaut au lieu
du défaut NumPyro. Effet mesuré : `2026-07-08` retrouve ses 4 chaînes
cohérentes (`tau≈0,028` partout), et la suite de tests complète passe de
2min13 à **53s** (les chaînes figées ne gaspillaient pas que la qualité du
résultat, aussi du temps de calcul).

### 6.8 Décalage RN vs sondages bruts — lissage correct, pas un bug identifié

Observation utilisateur, non résolue en bug mais creusée : sur certaines
fenêtres (ex. autour de `2026-06-24`), l'estimation filtrée de RN reste
systématiquement 1 à 3 points sous le dernier sondage brut. Deux hypothèses
testées et **écartées** : (a) le choix de la référence locale (testé en
forçant Centre au lieu de RN comme référence — résultat rigoureusement
identique, cohérent avec l'invariance du §5.2) ; (b) la force du prior 2022
(`HISTORICAL_PRIOR_N`, testé 300 vs 10 — quasi aucun effet). Explication la
plus probable, non confirmée comme définitive : sur cette fenêtre, les
sondages RN bruts font un creux réel (36→32 %) puis remontent (→35 %) ; avec
un `tau` petit — préféré nettement par la vraisemblance elle-même, pas un
artefact de prior — un lisseur reste prudent sur une remontée récente plutôt
que de suivre chaque rebond, comportement standard et correct d'un filtre
de Kalman à faible variance de transition. **[OUVERT]** : vérifier si la
bande IC90 affichée reste statistiquement cohérente avec le nuage de
sondages bruts (pas juste le point central).

### 6.9 [OUVERT] `tau` unique et isotrope + absence de débiaisage par défaut

Relance de l'investigation avec toutes les données disponibles
(`as_of=2026-08-10`, 33 hypothèses testant Le Pen sur 19 sondages, toujours
sans Bardella). Le SSM lui-même n'est pas cassé — NUTS converge proprement
sur les 68 nœuds réels (R-hat max 1,0036, ESS min 1375, **0 divergence**),
covariance SPD, aucun NaN. Mais le nowcast final affiche un sd de **14,6
points** pour RN, anormalement large vu 30 tests directs tous entre 31 et
36 %.

**Localisé précisément** : au dernier sondage réel (10 juillet), RN a un sd
de 0,7 point — raisonnable. Le dernier sondage date de 31 jours avant
`as_of` (pas de sondage en août). Extrapoler ces 31 jours avec
`tau≈0,123/jour` (estimé par NUTS) fait exploser le sd à 16,5 points. Pas
spécifique à RN — même mécanisme sur tous les candidats, proportionnellement
à leur taille (Philippe : 0,42→10,4 ; Mélenchon : 0,35→9,2 ; Glucksmann :
0,28→6,9).

**Cause à deux facteurs, tous deux vérifiés empiriquement** :
1. **`tau` est un seul paramètre partagé (isotrope) sur les 12 candidats**,
   alors que leur volatilité réelle diffère fortement (points/jour mesurés
   sur les rapports bruts) : Philippe 1,40, Glucksmann 0,98, Zemmour 0,66,
   **RN 0,50** (RN est en fait un candidat plutôt STABLE, bande étroite
   31-35 % sur toute la fenêtre) — `tau` doit satisfaire le candidat le
   plus volatil (Philippe/centre) et applique ce même rythme à RN qui n'en
   a pas besoin. Exactement le point resté ouvert au §2.1 (« offset par
   bloc à vérifier statistiquement »), jamais implémenté jusqu'ici.
2. **Écarts systématiques entre instituts non modélisés par défaut**
   (`use_biais=False` en prod, §6.6) : absorbés dans `tau` faute d'un autre
   canal. Vérifié directement : `use_biais=True` fait chuter `tau` de 0,123
   à **0,066** (quasi /2) et le sd final de RN de 14,6 à **7,8** points —
   inférence toujours saine (R-hat 1,003, 0 divergence). Même avec le
   débiaisage, 7,8 pt reste large pour un candidat empiriquement stable,
   donc le facteur (1) compte aussi, pas seulement (2).

**[TRANCHÉ]** : le mécanisme de filtrage/lissage est correct et sain
(vérifié deux fois, `as_of=2026-03-27` et `as_of=2026-08-10`) — le sujet
RN/Bardella (§5.7.1) est clos. **[OUVERT]** (distinct de la question
ALR/ILR) : `tau` unique et isotrope + absence de débiaisage institut par
défaut produisent une incertitude extrapolée disproportionnée dès qu'un
écart de plusieurs semaines survient sans nouveau sondage — deux pistes
concrètes, non tranchées : activer `use_biais=True` par défaut, et/ou
différencier `tau` par bloc/candidat (anticipé au §2.1, jamais fait).

### 6.10 Synthèse : toutes les corrections combinées, sans plafond ad hoc

`MIN_POLL_DATE` (§6.2) + `historical_prior_pi` (§6.3) + `historical_prior_cov`
(§6.4) + `use_biais=False` (§6.6) + `init_to_median` (§6.7) suffisent,
moyenne arithmétique inchangée (§6.5) : testé sur l'ensemble du backfill
2026 (15 dates, 3 variantes de modèle), `tau` serré et cohérent entre
chaînes à chaque date (< 0.05, jamais de chaîne figée), tous les candidats
dans un ordre de grandeur plausible et stable d'une date à l'autre (RN
29-34 %, Centre 15-19 %, Arthaud/Poutou 0,7-1,6 %). Aucun code ne plafonne
artificiellement une variance ni ne substitue une statistique de résumé non
principielle — les causes structurelles identifiées (sondages hors-scope,
prior mal centré, prior isotrope, sur-paramétrisation du biais à faible
volume de données, initialisation NUTS aberrante, **bruit d'observation qui
n'était pas celui d'un multinomial**, §6.11) sont corrigées directement, pas
leurs symptômes. Point ouvert non résolu : §6.8 (décalage RN vs sondages
bruts) — §6.9 est largement refermé par §6.11.

### 6.11 La couverture : `R` n'était pas la variance d'un multinomial **[TRANCHÉ]** (2026-08-11)

Point de départ : avec le roster cohérent (10 slots), le SSM ne couvrait que
**35 %** des observations à un IC90, avec des intervalles très étroits (1,83
pt). Gonfler `tau` (plancher 0,05 / 0,08) ou l'extrapolation finale n'y
changeait rien — normal, `as_of` tombe souvent un jour de sondage
(`as_of_dt = 0`). Gonfler `R` uniformément (×1,5, ×2, ×3) ne donnait que
35 → 40 %. La sous-couverture n'était donc ni un problème de `tau`, ni un
problème d'ÉCHELLE de `R` : c'était sa **FORME**.

#### 6.11.1 Ce que `R = (deff/n)·I` supposait réellement

Pour un multinomial, `Var(p̂_i) = p_i(1−p_i)/n`. En passant en log-ratio
(delta-method, `J = diag(1/p)`) puis en projetant sur l'hyperplan à somme
nulle (`P = I − 𝟙𝟙ᵀ/m`, qui annule le terme `−𝟙𝟙ᵀ`), et enfin dans la base
ILR locale orthonormée `W` (`W Wᵀ = P`, `Wᵀ W = I`, donc `Wᵀ P = Wᵀ`) :

```
Cov(p̂)      = (diag(p) − p pᵀ)/n
Cov(log p̂)  = (diag(1/p) − 𝟙𝟙ᵀ)/n
Cov(clr)    = P · diag(1/(n p)) · P
R = Cov(y)  = Wᵀ · diag(1/(n p)) · W
```

Le facteur `(1−p_i)` n'est pas perdu : il est **exactement** porté par la
projection. Vérifié en repoussant `R` jusqu'aux parts sur un sondage complet
(`H` inversible, jacobienne de `softmax`) : on retombe sur
`(diag(p) − p pᵀ)/n` à `2e−11` près — `tests/test_latent.py:
test_R_est_bien_la_variance_multinomiale_sur_les_parts`.

`R = (deff/n)·I` revient donc à poser **`p_i = 1` pour tous les candidats** :
la variance d'observation était sous-estimée d'un facteur ~`1/p_i`, soit
**×2,9 pour un candidat à 34 %** et **×100 pour un candidat à 1 %**. C'est
une erreur de forme, pas de réglage — d'où l'inefficacité d'un facteur
multiplicatif uniforme.

Cette simplification (§5.7) avait été adoptée pour atténuer la dérive du RN.
Son motif est tombé : la cause racine s'est révélée être le mélange de
rosters incohérents + la renormalisation implicite après suppression des
non-modélisés (`biais_rn_investigation.md`, addendum du 2026-08-11).

#### 6.11.2 `deff ≈ 2` — mesuré, pas postulé

Un sondage français n'est pas un tirage aléatoire simple. Mesure directe et
sans modèle : on apparie les scénarios de **deux instituts différents testant
exactement le même champ de candidats** à faible écart de date (2017+2022 :
8138 paires candidat×scénario à ≤3 j, 1478 le **même jour**). À champ et date
identiques, l'écart ne peut venir que du bruit d'observation.

```
E[(p_a − p_b)²] / E[Var_multinomiale] = 2,16 (≤3 j)   1,99 (même jour)
```

La **forme** de cet excès a été départagée entre trois modèles (RMS(z) par
tranche de `p` en points, 1,00 = calibration parfaite) :

| modèle | p≈2 | p≈6 | p≈10 | p≈15 | p≈22 | p≈27 |
|---|---|---|---|---|---|---|
| multinomial brut | 1,55 | 1,73 | 1,79 | 1,58 | 1,37 | 1,40 |
| **`deff` × multinomial (retenu)** | **1,06** | **1,17** | **1,21** | **1,07** | **0,93** | **0,95** |
| multinomial + `c²` constant en points | 0,67 | 1,05 | 1,24 | 1,20 | 1,10 | 1,17 |
| multinomial + `(κ·p)²` (log-ratio) | 1,45 | 1,41 | 1,34 | 1,05 | 0,83 | 0,77 |

Seul le facteur **multiplicatif** est plat sur toute la plage de `p`. Un
excès additif en points sur-couvre massivement les petits candidats ; un
excès isotrope en log-ratio — qui aurait pourtant été le plus naturel à
ajouter dans l'espace ILR (`+ σ²·I`, la piste « sur-dispersion
d'observation ») — a la pente inverse. **`deff = 2,0`** retenu (l'estimation
à écart nul, 1,99, est la seule qui isole le bruit pur ; les paires à ≤3 j
contiennent aussi un vrai mouvement d'opinion). Reproduction :
`notebooks/06b_same_day_poll_coherence_matched.py` pour l'appariement.

#### 6.11.3 Effet mesuré sur la couverture

Diagnostic d'innovation un-pas-en-avant (`notebooks/07_ssm_obs_coverage.py`,
roster cohérent 10, `as_of = 2026-07-10`, 72 innovations) : à chaque nœud,
`z = chol(S)⁻¹·(y − H·μ_pred)` avec `S = H P Hᵀ + R` doit être `N(0, I)`.
C'est la bonne quantité — un IC d'ÉTAT comparé à un sondage observé
sous-couvre par construction, puisqu'il oublie le bruit du sondage tenu (même
protocole que `notebooks/04e_spatial_coverage_excess.py` côté
`spatial_pooling`). `tau` est ré-estimé par maximum de vraisemblance **pour
chaque `R`** : sinon la comparaison est biaisée, élargir `R` poussant
mécaniquement `tau` vers le bas.

| `R` | `tau` (ML) | RMS(z) | IC50 | IC80 | IC90 |
|---|---|---|---|---|---|
| `(1/n)·I` (ancien défaut) | 0,1077 | **2,94** | 40,3 % | 55,6 % | 62,5 % |
| `(2/n)·I` | 0,0957 | 2,21 | 37,5 % | 59,7 % | 68,1 % |
| multinomial, `deff=1` | 0,0294 | 1,10 | 51,4 % | 75,0 % | 86,1 % |
| **multinomial, `deff=2`** | **0,0163** | **0,89** | **55,6 %** | **83,3 %** | **91,7 %** |
| multinomial, `deff=3` | 0,0114 | 0,78 | 58,3 % | 90,3 % | 97,2 % |

Le `deff = 2` mesuré indépendamment sur les paires appariées (§6.11.2) tombe
sur la configuration la mieux calibrée — deux routes indépendantes, même
réponse.

**Conséquence sur `tau`, et sur le point ouvert §6.9** : `tau` passe de
**0,108 à 0,016** (÷6,7). C'était mécanique — avec un `R` trop petit, le
filtre n'a d'autre canal que `tau` pour expliquer la variation d'un sondage à
l'autre. Le symptôme central du §6.9 (sd de 14,6 pts sur RN après 31 jours
sans sondage, dû à `tau ≈ 0,123`) se réduit d'un facteur ~7,5 sans toucher ni
à `use_biais` ni à un `tau` par candidat : **`tau` isotrope n'était pas le
problème, c'était un symptôme de `R`.** (À re-vérifier sous NUTS complet : le
tableau ci-dessus estime `tau` par maximum de vraisemblance, pas par MCMC.)

**Réserves, explicitement** :
- 72 innovations seulement (le filtre de roster est restrictif). L'ORDRE des
  configurations est sans ambiguïté (RMS(z) 2,94 → 1,10 → 0,89), mais un
  `deff` entre ~1,5 et ~2,5 n'y est pas départageable — c'est la mesure sur
  paires appariées (n=1478 le même jour) qui ancre la valeur.
- IC50 sur-couvre légèrement (55,6 % pour 50 %) : les sondages sont arrondis
  au demi-point, ce qui crée un atome d'innovations exactement nulles. Même
  effet visible sur les paires appariées.
- Sans filtre de roster (68 nœuds), le verdict sur la FORME tient (isotrope :
  RMS(z) 2,10) mais le NIVEAU de `deff` n'y est pas lisible — même `deff=1`
  sur-couvre (IC90 97,5 %), l'incohérence de roster gonflant la covariance
  d'état bien au-delà de ce que `R` explique. Ce n'est pas une cible de
  calibration valide (cf. `biais_rn_investigation.md`).
- Le biais résiduel par candidat reste visible (`z` moyen : RN +0,78,
  Mélenchon +0,66) — c'est le retard de poursuite d'une marche sans dérive
  face à une tendance soutenue (§7.2), un sujet distinct de la couverture.

#### 6.11.4 Ce que ça rend caduc

`full_covariance` ne bascule plus la FORME de `R` : les deux modes portent
désormais la même covariance delta-method et ne diffèrent que par le système
de coordonnées locales (ILR orthonormé vs ALR). Par le théorème d'invariance
(§5.2) ils donnent le **même** postérieur — c'est devenu un test de
non-régression (`test_ilr_local_et_alr_local_donnent_le_meme_posterieur`)
plutôt qu'une variante de modèle. L'ILR local reste le défaut : il est
orthonormé, donc cohérent avec un `tau` isotrope.

`isotropic_obs_noise=True` (`bayesian-nowcast-obs-isotrope`) conserve
l'ancien comportement, pour mesurer l'écart — pas comme candidat.

### 6.12 Le filtre de roster cohérent devient le défaut **[TRANCHÉ]** (2026-08-11)

`aggregate_to_slots` retire silencieusement les candidats non modélisés d'une
hypothèse, puis `poll_observation_values` RENORMALISE la masse restante. La
part supprimée n'est donc pas traitée comme manquante — elle est
**redistribuée** entre les candidats restants. Mesuré sur la campagne 2026 :
**28,75 pts** de masse jetée en moyenne par hypothèse (médiane 34), et
**43,04 pts quand RN est absente** du champ testé contre 9,41 quand elle est
présente. Le SSM ne voyait donc pas « le même électorat avec RN manquante »
mais une composition renormalisée sur un support plus petit, traitée comme
comparable aux hypothèses contenant RN.

Deux conséquences, mesurées séparément :

| | sans filtre | roster cohérent 10 |
|---|---|---|
| erreur signée moyenne sur RN | −2,82 pt | **−0,26 pt** |
| couverture prédictive IC90 (hors éch.) | 99,3 % | cf. §6.11.3 |
| couverture prédictive IC50 | 78,0 % | — |
| largeur IC90 sur RN | 17,5 pt | — |

La seconde ligne est le point neuf de cette session : le mélange de rosters ne
biaise pas seulement la moyenne, il gonfle la **covariance d'état** bien
au-delà de ce que le bruit d'observation explique — un candidat rarement
observé dans un champ cohérent garde une variance très large, et les
intervalles deviennent inutilisables. C'est visible aussi dans le diagnostic
d'innovation (§6.11.3, dernière réserve) : sans filtre, même `deff=1`
sur-couvre à 97,5 %.

**[TRANCHÉ]** : `BayesianNowcast.roster_filter_slots = COHERENT_ROSTER_SLOTS`
par défaut. `bayesian-nowcast-roster-mixte` conserve l'ancien comportement
pour la mesure. Les autres variantes enregistrées héritent du filtre, donc
chacune ne fait plus varier qu'un seul facteur par rapport au défaut.

**Limite assumée** : le filtre écarte des sondages réels. Les hypothèses
testant Bardella (16 sur la fenêtre janvier-mars 2026, contre 4 pour Le Pen)
ne sont plus utilisées du tout, alors qu'elles portent une information sur le
bloc RN. C'est le prix du choix `SLOTS` (un seul candidat par bloc, Bardella
écarté — décision utilisateur confirmée, §5.7.1). La bonne façon de récupérer
cette information serait un remappage explicite à l'ingestion ou un slot
« bloc RN » agrégé, pas un relâchement du filtre.

## 7. Proposition non tranchée : état de tendance (local linear trend)

**Statut : proposition, en attente de validation avant implémentation.**
Rien ici n'est [TRANCHÉ] tant que le plan de vérification (§7.6) n'a pas
tourné sur des données réelles. Extension du SSM actuel, pas un
remplacement.

### 7.1 Constat empirique

Une fois les bugs d'agrégation des sondages corrigés (une observation SSM
par hypothèse de sondage plutôt qu'une moyenne/sélection inter-hypothèses),
le nowcast RN reste systématiquement **en dessous** du dernier sondage,
d'un écart **quasi constant** une fois la période d'amorçage passée :

| as_of | nowcast RN | dernier sondage RN | écart |
|---|---|---|---|
| 2026-03-27 | 0.292 | 0.328 | -3.5 pt |
| 2026-04-30 | 0.290 | 0.325 | -3.5 pt |
| 2026-05-27 | 0.288 | 0.315 | -2.7 pt |
| 2026-06-24 | 0.294 | 0.320 | -2.6 pt |
| 2026-07-08 | 0.322 | 0.352 | -3.0 pt |
| 2026-08-09 | 0.331 | 0.359 | -2.8 pt |

RN progresse d'environ +4 points sur cette fenêtre (`v ≈ 0.000324` en
fraction de part par jour, soit **≈ 0,97 point de %/mois**). L'écart ne
croît pas, ne décroît pas : il suit RN au même rythme avec un décalage de
phase fixe — signature d'un mécanisme précis (§7.2), pas d'un problème de
données à nettoyer.

### 7.2 Pourquoi une marche aléatoire sans dérive produit un décalage constant

Le modèle actuel traite `alpha_t` comme une marche aléatoire pure
(`alpha_t = alpha_{t-1} + eps_t`, `E[alpha_t − alpha_{t-1}] = 0`). Si
l'opinion réelle suit une tendance directionnelle soutenue (le cas RN), ce
modèle ne peut structurellement pas la représenter comme un déplacement
attendu — seulement comme une accumulation de chocs sans direction
privilégiée.

Approximation en temps continu (`x(t) = x₀ + v·t`, diffusion pure crue par
le filtre `dx = √q·dW`, `q=τ²`, flux d'observations de fréquence `1/Δ` et
variance `R`, densité spectrale `r ≈ R·Δ`). Riccati continue :
`dP/dt = q − P²/r`. Régime stationnaire : `P* = √(qr)`, gain `K* = √(q/r)`.
Erreur de poursuite en régime stationnaire :

```
e* = v / K* = v · √(r/q) = v · √(R·Δ) / τ
```

**`e*` est une CONSTANTE** — exactement le comportement observé. Le
décalage croît avec la vitesse de la tendance et le bruit de mesure,
décroît avec `τ`, mais ne s'annule jamais à `τ` fini : un **biais structurel
du modèle**, pas un problème de réglage (augmenter `τ` réduit `e*` mais
dégrade le lissage du bruit poll-à-poll — vrai compromis biais/variance).
Cette dérivation donne l'ordre de grandeur et la forme du phénomène, pas une
prédiction numérique exacte (le filtre réel est discret, `R`/`Δ` varient,
`τ` non exposé aujourd'hui) — vérification numérique proposée en §7.6.

### 7.3 Modèle proposé

État étendu `z_t = (alpha_t, beta_t) ∈ R^{2(K-1)}`, `beta_t` = tendance
(vitesse) de `alpha_t`, non observée directement par les sondages.

Transition (cinématique connue, pas estimée) :

```
alpha_{i+1} = alpha_i + Δt_i · beta_i + eps_alpha,i     eps_alpha,i ~ N(0, τ_niveau² · Δt_i · I)
beta_{i+1}  = beta_i + eps_beta,i                       eps_beta,i  ~ N(0, τ_tendance² · Δt_i · I)
```

`F_i = [[I, Δt_i·I],[0, I]]`, entièrement déterminée par `Δt_i` — aucun
coefficient appris par NUTS, identité cinématique. `Q_i` bloc-diagonale
(innovations niveau/tendance indépendantes — hypothèse standard du local
linear trend, cohérente avec l'esprit « modèle robuste d'abord » déjà
appliqué à `jump_scale`).

Émission : padding mécanique, `H_i étendue = [H_i^alpha | 0]` — chaque
sondage observe `alpha`, jamais `beta` directement. Même mécanisme que le
padding pour les slots non testés, appliqué deux fois.

État initial : `alpha_0` inchangé. `beta_0 ~ Normal(0, sigma_beta0²·I)`,
`sigma_beta0` fixé (constante de config, pas un hyperparamètre NUTS de
plus) — valeur exacte à traiter comme un réglage à tester par sensibilité
(§7.6), pas une inconnue à résoudre analytiquement avant de coder.

### 7.4 Paramètres ajoutés

**Un seul nouveau paramètre NUTS : `τ_tendance`** (scalaire global, même
parcimonie que `τ` aujourd'hui). `τ_niveau` remplace `τ` (même rôle,
nouveau nom).

Sur le prior de `τ_tendance` : `bank.campaign_drift_sd` est calibré en
points de % avec dépendance en `√horizon`, alors que `τ_tendance` gouverne
un `beta` en unité ILR-par-jour avec sa propre marche en `√Δt` — pas
directement convertible sans delta-method dédié (non fait). **Pas
d'ancrage historique propre disponible pour `τ_tendance`** — proposition :
prior faible générique (`HalfNormal`, échelle par défaut de l'ordre de
celle déjà utilisée pour `τ` sans Bank), laissé aux données du cycle 2027
pour se positionner, vérification a posteriori (§7.6) plutôt qu'une valeur
revendiquée précise.

Décompte total : 1 → 2 hyperparamètres de diffusion estimés par NUTS. Rien
à voir avec l'échec de `use_biais` (§6.6, 20-45 groupes) : ici un scalaire
de plus, sur un nombre de nœuds substantiellement augmenté par le fix
d'agrégation (§7.5).

### 7.5 Identifiabilité — nombre de nœuds disponibles

| as_of | sondages bruts | nœuds SSM (hypothèses) |
|---|---|---|
| 2026-02-27 | 1 | 10 |
| 2026-03-26 | 3 | 14 |
| 2026-05-28 | 9 | 49 |
| 2026-08-09 | 15 | 75 |

Le point faible reste le tout début de campagne (10-14 nœuds) — précisément
là où `use_biais` s'était mal comporté par le passé (`tau` explorant des
valeurs aberrantes). `τ_tendance` est UN SEUL scalaire de plus (pas 20-45
groupes), mais la vérification empirique sur ces dates précises (§7.6)
reste nécessaire avant de considérer l'ajout acquis.

### 7.6 Plan de vérification (avant [TRANCHÉ])

1. **R-hat / ESS sur les dates creuses** (2026-02-27, 2026-03-22,
   2026-03-26) — `τ_tendance` doit converger proprement (R-hat < 1.01, pas
   de chaînes figées comme au §6.7).
2. **Le décalage constant du §7.1 doit se réduire nettement** une fois le
   modèle à tendance en place, sur les mêmes dates de comparaison.
3. **Sensibilité au prior de `τ_tendance`** (et à `sigma_beta0`, §7.3) :
   2-3 échelles de prior différentes, vérifier que la conclusion n'est pas
   un artefact d'un prior mal choisi.
4. **Exposer `τ_niveau`/`τ_tendance` en diagnostic** (actuellement aucun
   des deux n'est persisté ni affiché) — vérifier numériquement la formule
   du §7.2 (`e* ≈ v·√(R·Δ)/τ_niveau`) a posteriori.
5. **Backtesting 2022** (`model/backtest/backtest_loo.py`) — phase
   suivante : vérifier la couverture des IC sur la vraie campagne 2022, où
   des tendances soutenues sont documentées dans les deux sens (montée Le
   Pen fin de campagne, effondrement Pécresse) — test plus dur que la
   comparaison au dernier sondage seul, à faire une fois les points 1-4
   validés sur 2027.

### 7.7 Ce qui ne change pas

- Le saut terminal (`terminal_jump_model`, sinh-arcsinh) reste séparé,
  inchangé (§2.2 pour pourquoi une fusion complète n'est pas praticable).
- `institut_bias`/`use_biais` : aucun changement, reste désactivé par
  défaut (§6.6).
- `jump_scale` global : inchangé, sans lien avec cette proposition.

## 8. Ce qui se réutilise de l'architecture actuelle

- `campaign_drift_sd` (calibration existante) comme point de départ pour la
  valeur a priori de `τ`.
- `clr_movement_pool` (2017/2022) comme jeu de données pour fitter
  `loc`/`scale`/`skew`/`tail` du saut terminal — remplace le
  ré-échantillonnage bootstrap par un ajustement paramétrique, sans changer
  la source de données.
- `institut_bias` (biais institut×bloc) inchangé dans l'observation.
- Architecture générale : `BayesianNowcast(ForecastModel)` dans
  `model/models/bayesian_nowcast/`, calibration + nowcast colocalisés,
  composition (pas d'ABC).

## 9. Hors scope

- Matrice de reports de voix / candidats entrants-sortants (discuté
  séparément — architecture déjà compatible).
- Vote blanc / corps électoral plus fin.
- Modification du contrat de sortie `ForecastModel`/`Nowcast` pour exposer
  la trajectoire lissée sur le site — point d'intégration à traiter avec
  l'agent qui gère `site/`, pas ici.

## 10. Index des points ouverts

- **§5.6** — covariance jointe entre candidats : bien calibrée ou pas ?
- **§6.8** — décalage RN vs sondages bruts sur certaines fenêtres : lissage
  correct, non confirmé comme définitif.
- **§6.9** — `tau` unique/isotrope + absence de débiaisage institut par
  défaut : incertitude extrapolée disproportionnée après un long silence
  sondagier. **Largement refermé par §6.11** : `tau` passe de 0,108 à 0,016
  une fois `R` corrigé, donc la piste « `tau` par bloc/candidat » n'est plus
  prioritaire. Reste à confirmer sous NUTS complet.
- **§6.11** — [TRANCHÉ] `R` est maintenant la covariance delta-method du
  multinomial, déflatée par `deff = 2` (mesuré). Reste ouvert : la valeur de
  `deff` n'est ancrée que sur 2017/2022, à re-mesurer quand la campagne 2027
  aura assez de sondages appariés. `deff` absorbe aussi les house effects
  faute de `use_biais` — à réviser à la baisse si `use_biais` est réactivé,
  sinon on les compte deux fois.
- **§6.12** — [TRANCHÉ] filtre de roster cohérent par défaut. Reste ouvert :
  récupérer l'information des hypothèses écartées (Bardella) par un remappage
  à l'ingestion plutôt que de les jeter.
- **§7** — état de tendance (local linear trend) : proposition complète,
  non implémentée, plan de vérification défini mais pas exécuté.
