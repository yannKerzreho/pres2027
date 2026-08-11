# Spec — modèle spatial (Hotelling-Downs discrétisé)

Statut : **validé sur prototype** (synthétique + données réelles 2017/2022/2027)
**et peuplé dans `model.py`** (ce dossier) — maths, fit, lecture avec
incertitude et saut terminal. **Pas encore branché** sur `ForecastModel` / le
contrat de sortie du site (décision explicite de l'utilisateur, à traiter
séparément). Prototypes de référence (historique des découvertes, pas le code
de production) : `notebooks/_spatial_core.py`, `03_spatial_prototype.py`,
`03b_spatial_debug.py`, `04_spatial_real_data.py`, `04b_spatial_halflife_backtest.py`,
`04c_spatial_sanity_check.py`.

## 0. Motivation — pourquoi ce modèle, en plus du SSM en production

Le SSM compositionnel (`model/models/bayesian_nowcast/`) fusionne les alternates
d'un même bloc en un seul "slot" (`SLOTS`, `model/core/live_dataset.py`) — ex.
Bardella est **exclu**, seule Le Pen est retenue pour le slot RN. Une session de
debug dédiée (biais RN, ~-1,8pt de sous-estimation persistant, cause isolée à
trois niveaux : maths de l'encodage, santé NUTS, pipeline de données — tous
innocentés) a confirmé la cause racine : Bardella teste 46× contre 40× pour Le
Pen sur la campagne, et ces 46 sondages sont **jetés**, ce qui rend le RN
structurellement sous-testé et sa variance disproportionnée au sein d'une
hypothèse partagée.

Deux objectifs qui ne rentrent pas dans le cadre du SSM :

1. **Reports de voix cohérents** entre scénarios de liste (backlog "a priori
   d'expert" — différé faute d'un mécanisme mieux que « designer une matrice de
   reports à la main »). Un modèle spatial en donne un pour presque rien : si un
   candidat est retiré, ses électeurs se redistribuent vers les positions
   voisines par construction du softmax, pas par une matrice ad hoc.
2. **Scénarios interactifs** (checklist candidats sur le site) : leverage TOUTES
   les hypothèses de sondage (chaque institut teste un sous-ensemble différent
   de candidats) au lieu d'en retenir une seule par bloc — et permettre à
   l'utilisateur de recomposer son propre scénario sans relancer d'inférence
   (cf. §7).

## 1. Dimensions et grille spatiale

- $N$ : candidats retenus (roster filtré, cf. §6).
- $P$ : nœuds sondage × hypothèse (un sondage à $h$ hypothèses de champ produit
  $h$ nœuds, même jour, même logique que `poll_plan` dans le SSM).
- $B = 50$ : bins de discrétisation de l'espace politique $[0,1]$.

$$V = \mathrm{linspace}(0.01, 0.99, B), \qquad W_b = \frac{1}{B} \ \forall b$$

($W$ uniforme — l'extension "densité électorale en cloche" de la spec d'origine
n'a pas été implémentée, pas nécessaire pour valider le mécanisme.)

## 2. Positions statiques — $\mu_i$, $\sigma_i$

**Écart avec la spec d'origine** (discussion de session) : pas d'ancrage sur un
"analogue historique" ponctuel (2022 seul) — remplacé par un **ordre de groupes
fourni à la main**, plus robuste aux nouveaux entrants (Attal/Philippe n'ont pas
de prédécesseur direct) et évite d'avoir à fitter une position historique par
candidat avant de pouvoir tourner sur 2027.

### 2.1 Groupes ordonnés

Un groupe = un point de la séquence gauche→droite ; plusieurs candidats
peuvent partager un groupe (alternates, ou familles politiques proches) SANS
ordre imposé entre eux — seul l'ordre ENTRE groupes est une contrainte dure.

Ordre retenu (session du 2026-08-10, corrigé le même jour par retour
utilisateur — roster 2027 filtré à ≥5 sondages) :

```
LO < LFI < Coco/Écolo < PS < Attal < Philippe < LR < MLP < Zemmour
```

| groupe | candidats (roster 2027 filtré) |
|---|---|
| LO | Arthaud |
| LFI | Mélenchon |
| Coco/Écolo | Roussel, Tondelier |
| PS | Glucksmann, Hollande |
| Attal | Attal |
| Philippe | Philippe |
| LR | Retailleau |
| MLP | Le Pen, Dupont-Aignan |
| Zemmour | Zemmour |

Corrections apportées au premier jet (§0 d'origine avait un ordre différent,
révisé sur retour utilisateur) :
- **Arthaud = LO, pas NPA** — le candidat 2027 de ce point de la séquence
  représente Lutte Ouvrière, pas le Nouveau Parti Anticapitaliste (Poutou,
  qui ne passe de toute façon pas le filtre ≥5 sondages).
- **Coco/Écolo scindé de PS**, avec l'ordre Coco/Écolo < PS (au lieu d'un
  seul groupe "écolo/PS/coco" non ordonné).
- **Bardella retiré du groupe MLP** — Marine Le Pen est la candidate
  OFFICIELLE du RN pour 2027, il n'y a plus de scénario alternatif à
  modéliser sur ce point (contrairement à Attal/Philippe, où le choix reste
  ouvert). Ce retrait ne remet PAS en cause l'argument de fond du modèle
  (§0, utiliser tous les alternates plutôt qu'en exclure un) — c'est un
  ajustement factuel (l'ambiguïté a été levée dans la réalité), pas un abandon
  du principe, qui reste actif pour Attal/Philippe.

**Hypothèses non tranchées, à valider** : Dupont-Aignan et Bardella rattachés
au groupe MLP (même bloc `droite_radicale` que Le Pen dans
`data/historical/candidat_blocs.csv`) ; Hollande rattaché à écolo/PS/coco (PS).
Aucune de ces trois affectations n'est demandée explicitement par l'utilisateur
— choix par défaut documentés ici pour être corrigés facilement.

### 2.2 Construction (garantit l'ordre par construction, pas par pénalité)

$$\text{base} \sim \mathcal{N}(0, 1.2), \qquad \text{gap}_k \sim \text{LogNormal}(-0.5, 0.7) \ (k = 2..K)$$

$$\text{raw}_k = \text{base} + \sum_{j \le k} \text{gap}_j, \qquad \text{slot\_pos}_k = \mathrm{sigmoid}(\text{raw}_k) \in (0,1)$$

$\mathrm{sigmoid}$ est monotone → l'ordre est préservé automatiquement, aucune
transformation `Ordered` explicite nécessaire.

### 2.3 Delta individuel (non centré)

$$\mu_i = \text{slot\_pos}_{g(i)} + z^\mu_i \cdot \text{sd}_{\Delta\mu}, \qquad z^\mu_i \sim \mathcal{N}(0,1)$$

$$\sigma_i = \sigma_{\text{slot}, g(i)} \cdot \exp(z^\sigma_i \cdot \text{sd}_{\Delta\sigma}), \qquad \sigma_{\text{slot}} \sim \text{LogNormal}(\log 0.15,\ 0.4)$$

$g(i)$ = groupe du candidat $i$. **`sd_delta_mu`/`sd_delta_sigma` sont
échantillonnés par NUTS** (`HalfNormal`), pas fixés à la main — décision
explicite de session : rien ne garantit qu'un successeur soit "proche" de son
prédécesseur (Attal/Philippe ne sont pas juste "Macron + petit bruit"), donc le
dosage doit être appris des données plutôt que supposé.

### 2.4 Nouveaux candidats (Attal, Philippe...)

Un candidat sans aucun test se replie intégralement sur `slot_pos`/`sigma_slot`
de son groupe ($z_i$ non contraint par la vraisemblance → reste proche de son
prior $\mathcal{N}(0,1)$). Un candidat testé quelques fois commence à se
décoller du repli. Même mécanisme de partial pooling non centré que
`institut_bias` dans `model/models/bayesian_nowcast/calibration.py`.

## 3. Dynamique — $w_i(\text{now})$, SANS state-space séquentiel

Décision de session : pas de marche aléatoire explicite ($w_{i,t}$ pour tout
$t$) — on n'a besoin que du NOWCAST (l'état "maintenant"), pas de retracer tout
l'historique. `w_i` est un paramètre libre par candidat (pas de hiérarchie
partagée — `notebooks/03b_spatial_debug.py` a vérifié que le pooling ne change
rien à la qualité de la prévision, cf. §5) :

$$w_i \sim \mathcal{N}(0, w_{\text{scale}}), \qquad w_{\text{scale}} = 1.5 \text{ (fixe)}$$

### 3.1 Vraisemblance tempérée (pondération de récence)

Chaque nœud $p$ (date $t_p$) contribue à la vraisemblance de $w$ avec un poids
qui décroît avec son âge :

$$\kappa_p = 2^{-(\text{as\_of} - t_p)/\text{half\_life}}$$

$$\mathcal{L}(w, \mu, \sigma) = \sum_p \kappa_p \cdot \sum_{i \in S_p} \log \mathcal{N}\big(Y_{i,p} \mid \pi_{i,p}(w,\mu,\sigma),\ \mathrm{Var}_{i,p}\big)$$

C'est une **vraisemblance de puissance** ("power/tempered likelihood") — PAS une
densité normalisée. Conséquence directe : **`half_life` NE PEUT PAS être
échantillonné par NUTS** (pas de pénalité de complexité/Occam pour un
`half_life` dégénérément court → collapse vers "ne garder que le dernier
sondage"). Calibré séparément par backtest hors-échantillon (§4), comme un
hyperparamètre de lissage classique (cf. choix d'une force de régularisation
par validation croisée).

**Alternative "propre" écartée pour l'instant** : un vrai state-space (marche
aléatoire + EKF/UKF de `dynamax.nonlinear_gaussian_ssm`, déjà une dépendance du
projet) donnerait un `half_life` implicite comme rapport bruit de
processus/bruit d'observation, avec `tau` échantillonné normalement par NUTS —
reporté explicitement ("on commence simple, EKF/UKF ensuite") pour ne pas
cumuler la complexité de l'ordre/hiérarchie ET du filtrage séquentiel dans la
même itération.

### 3.2 Jauge non identifiée de $w$ (découverte du prototype, `03b_spatial_debug.py`)

$\mathrm{softmax}(w + D)$ est invariant à un décalage global : $w_i \to w_i + c$
pour TOUS les candidats laisse `pi` exactement inchangé. `w` brut n'a donc
**pas de niveau absolu identifié** — même phénomène que le CLR
(`clr = log(p) - mean(log p)`) déjà utilisé dans le SSM
(`model/models/bayesian_nowcast/latent.py`). Vérifié empiriquement : comparer
`w` brut à une vérité terrain synthétique donne une erreur énorme, alors que
`pi` (la quantité utile) est recouvré avec une erreur de ~0,003-0,004 (cf. §5).

**Si `w` doit être affiché comme diagnostic interprétable** (site : "momentum
actuel"), fixer la jauge avant affichage — ex. $\sum_i w_i = 0$, ou un candidat
de référence à $w=0$. Non fait dans le prototype actuel (`pi` seul est
consommé), mais sans coût : ça ne change rien à `pi`.

### 3.bis Tentative de diffusion réelle (`spatial_pooling_model_tau`) — ESSAYÉ, PARKÉ

Suite à la sous-couverture mesurée (§6.4), hypothèse de session : `w_now`
statique sans diffusion temporelle explique la sous-couverture. Implémenté
(`spatial_pooling_model_tau`, `model.py`) : marche aléatoire gaussienne dense
sur une ligne du temps globale dédupliquée (M dates uniques),
$w_i(t_m) \sim \mathcal N(w_i(t_{m-1}),\ \tau^2 \cdot \Delta t_m)$, `tau`
échantillonné par NUTS avec un prior faible (PAS encore calibré sur
2017/2022). Contrairement à `half_life`, `tau` est un vrai paramètre
générative-Bayésien (prior/vraisemblance propres, pas de vraisemblance
tempérée) — le bon réflexe en théorie.

**Résultat sur le roster 2027 réel (12 candidats après retrait de Bardella,
13 dates uniques, 78 nœuds, dimension ajoutée N·M=156)** : R-hat 1,005, ESS
1168, 7/4000 divergences (mineur) — **convergence propre en apparence**, MAIS
**mode dégénéré** : toutes les positions $\mu$ se sont effondrées entre 0,70
et 0,99 (Arthaud/LO, censé être le plus à gauche, à 0,701 ; Zemmour, le plus
à droite, à seulement 0,975 — quasiment plus d'écart entre les deux
extrêmes). Le modèle a distingué les candidats presque entièrement via `w`
plutôt que via la position — une solution alternative valide au sens de la
vraisemblance (la jauge/dégénérescence évoquée en §3.2 laisse ce genre de
compromis possible), mais qui vide le modèle spatial de son intérêt (plus de
géométrie interprétable). **R-hat/ESS sains ne garantissent PAS un mode
correct** : les 4 chaînes peuvent converger ensemble vers le MÊME mauvais
mode. **Temps : 2066s (~34 min)**, contre ~5 min pour `spatial_pooling_model`
(tempéré) sur une taille comparable — 7× plus lent, cohérent avec une
géométrie difficile pour NUTS malgré la paramétrisation non centrée.

**Décision (session du 2026-08-10)** : combiné à la stratification par âge
(§6.4, coverage plate → diffusion probablement pas la cause dominante), ce
modèle est **parké** — pas supprimé (le code reste dans `model.py`,
documenté ici), mais pas la voie prioritaire. La correction prioritaire est
un terme de variance d'excès (§10), plus directement ciblé sur la cause
diagnostiquée, plus léger, et sans le risque de mode dégénéré observé ici.

## 4. Modèle d'observation — masquage des sondages partiels

Un sondage 2027 ne teste jamais tous les candidats. $S_p$ = sous-ensemble testé
par le nœud $p$, encodé en masque multiplicatif $M_{i,p} \in \{0,1\}$ (PAS
`-inf` dans l'exponentielle — plus stable numériquement, entièrement vectorisé
sur $(P, N, B)$, aucun `scan`) :

$$D_{i,b} = -\frac{(V_b - \mu_i)^2}{2\sigma_i^2}, \qquad A_{i,b} = w_i + D_{i,b}$$

Stabilisation softmax (indépendante du sous-ensemble testé, valide pour tout $p$) :

$$\tilde{A}_{i,b} = A_{i,b} - \max_j A_{j,b}$$

$$P_{i,b,p} = \frac{M_{i,p}\cdot e^{\tilde{A}_{i,b}}}{\sum_{j} M_{j,p} \cdot e^{\tilde{A}_{j,b}}}, \qquad \pi_{i,p} = \sum_b P_{i,b,p} \cdot W_b$$

### Vraisemblance (approximation gaussienne, comme la spec d'origine §5E)

$$Y_{i,p} \sim \mathcal{N}\left(\pi_{i,p},\ \frac{\pi_{i,p}(1-\pi_{i,p})}{N_p} + \epsilon\right), \qquad i \in S_p$$

($\epsilon = 10^{-8}$, plancher numérique — évite une variance nulle exacte
quand $\pi_{i,p}\to 0$ pour un candidat masqué, sans biaiser le résultat car ce
terme est multiplié par $M_{i,p}=0$ dans la somme pondérée.)

## 5. Validation empirique

### 5.1 Synthétique (`03_spatial_prototype.py`, `03b_spatial_debug.py`)

N=8 candidats/4 blocs, 90 hypothèses, candidat en "surge" tardif simulé. NUTS
sain (R-hat 1,003-1,017 sur 4 configurations de half_life, ESS 273-960 ;
`half_life=15j` seul montre 42 divergences à surveiller). `mu`/`sigma` recouvrés
à ~0,03/0,01 d'erreur absolue moyenne. `w` brut mal recouvré (jauge non
identifiée, §3.2) mais `pi` recouvré à **+0,004 pt** d'écart sur le candidat en
surge — identique avec ou sans pooling hiérarchique sur `w`.

### 5.2 Backtest half_life sur 2017/2022 réels (`04b_spatial_halflife_backtest.py`)

Protocole : fit sur les sondages avant une date de coupure, logscore des
sondages réellement publiés dans les 21 jours suivants (jamais vus du fit).

| half_life | logscore 2017 (J-30) | logscore 2022 (J-60) |
|---|---|---|
| **15j** | **-2,60** | **+1,47** |
| 45j | -6,11 | -27,03 |
| ∞ (pooling uniforme) | -7,53 | -72,57 |

Écart massif et cohérent sur les deux élections en faveur d'un half_life court.
Tendance monotone (plus court = mieux sur la grille testée {15,45,∞}) — un
balayage plus fin (5-25j) n'a pas été fait, `half_life=15j` n'est donc
probablement pas l'optimum exact, juste le meilleur des 3 valeurs testées.
**`half_life=15j` retenu comme valeur de calibration actuelle.**

*Limite connue de ce backtest* : la fenêtre 2017 est peu couverte par la source
Wikipedia (19 sondages sur toute la campagne contre 198 pour 2022) — la
coupure a dû être reculée à J-30 (train=5 sondages, au minimum du seuil
`MIN_POLLS`) faute de données à J-60/90/120/150. Le résultat 2017 doit donc
être lu comme confirmatoire, le résultat 2022 (101 sondages train à J-60) comme
le plus fiable des deux.

### 5.3 Sanity check sur données réelles 2027 (`04c_spatial_sanity_check.py`)

Roster filtré (§6), `half_life=15j`, fit sur l'ensemble des 78 nœuds
disponibles. Comparaison, pour les 10 hypothèses les plus récentes, entre le
score RAPPORTÉ par l'institut et le score PRÉDIT par le modèle restreint au
même sous-ensemble testé (pas de fuite : `pi` ne dépend pas de $Y_p$ plus que
des autres nœuds, postérieur joint habituel) :

- **95 observations candidat×hypothèse, erreur absolue moyenne 0,77 pt
  (médiane 0,50 pt, max 3,17 pt), 98% des écarts ≤ 3 points, biais +0,21 pt**
  (négligeable). Ordre de grandeur comparable au bruit d'échantillonnage d'un
  sondage réel — pas un signe de sous-ajustement.
- R-hat 1,002, **0 divergence sur 4000 tirages**.
- Point notable, **expliqué, pas un bug** : le modèle donne Le Pen légèrement
  au-dessus de Bardella en moyenne prédite (36,5% vs 36,1%) alors que la
  moyenne BRUTE rapportée sur toute la campagne va dans l'autre sens (35,2%
  Bardella vs 33,9% Le Pen). Cause : Bardella n'a plus été testé depuis le
  2026-06-24 (46 jours avant `as_of`), alors que Le Pen a été testée
  quasi-quotidiennement jusqu'au 2026-07-10, à des niveaux (34-37%) égaux ou
  supérieurs au dernier niveau connu de Bardella. Le mécanisme de récence
  fonctionne comme voulu : il ne moyenne pas aveuglément tout l'historique, il
  reflète ce que dit l'évidence la plus fraîche disponible POUR CHAQUE
  candidat séparément — exactement le comportement recherché en abandonnant le
  pooling uniforme (cf. §5.2).

## 6. Incertitude au nowcast — argument théorique solide, MAIS invalidé empiriquement (§6.4)

Question de session : la variance classique d'une moyenne pondérée
$\mathrm{Var}[\hat Y] = \sum_p W_p^2\sigma_p^2/\Omega^2$ **tend vers 0** quand
le nombre de sondages grossit, même s'ils se contredisent — piège identifié et
évité par `linear_pooling` via un **mélange explicite** (tirer un sondage puis
son bruit Beta, cf. `spec_linear_pooling.md` §4). `spatial_pooling` n'a pas ce
problème, mais pas de la même façon.

### 6.1 Le postérieur NUTS absorbe DÉJÀ le désaccord entre sondages

$w_i$/$\mu_i$/$\sigma_i$ ne sont pas des moyennes pondérées calculées à la
main : ce sont des **tirages postérieurs bayésiens** issus de NUTS sur la
vraisemblance pondérée (§3.1). Si les sondages récents d'un candidat se
contredisent, AUCUNE valeur de $w_i$ n'explique bien tous les nœuds pondérés à
la fois — le postérieur s'élargit alors mécaniquement pour rester cohérent
avec chacun, exactement l'effet que `linear_pooling` obtient explicitement par
son mélange (le second terme de sa décomposition de variance, §4.2 de sa
spec — désaccord entre sondages). Ici, c'est une propriété AUTOMATIQUE de
l'inférence bayésienne jointe, pas un mécanisme à construire à part.

**Interprétation propre de $\kappa_p$** : pondérer le log de vraisemblance
gaussien par $\kappa_p$ équivaut exactement à utiliser une variance
$\mathrm{Var}_{i,p}/\kappa_p$ — c'est-à-dire à traiter un sondage ancien comme
s'il avait un échantillon effectif $\kappa_p \cdot N_p$ plus petit. C'est
littéralement la MÊME mécanique qu'un `deff` (design effect), déjà utilisé
ailleurs dans le projet (`latent.py`/`calibration.py` du SSM) pour déflater un
échantillon — pas un artifice ad hoc : un sondage vieux de 30 jours à
`half_life=15j` compte, en évidence, comme s'il avait un quart de son
échantillon réel. Ce recadrage répond directement à la question "l'incertitude
est-elle bien quantifiée" : oui, dans un sens précis et déjà familier au
projet, pas juste "probablement".

### 6.2 Ce qu'il faut faire pour lire cette incertitude — corrigé dans `model.py`

Les scripts `04_spatial_real_data.py`/`04c_spatial_sanity_check.py`
utilisaient les MOYENNES postérieures (`mu_hat`, `sigma_hat`, `w_hat`) pour
lire `pi` — un point estimate, qui jette l'incertitude. `model.py` corrige ça :
`fit_spatial_pooling` garde les tirages BRUTS (`SpatialPoolingFit.mu_draws`
etc., shape `(S,N)`), et `pi_draws_for_mask` pousse CHAQUE tirage à travers le
même softmax masqué (`spatial_shares`, vectorisé sur la dimension `S` en plus
de `(N,B)`) pour obtenir une vraie distribution de `pi` — pas un scalaire.
`summarize_pi` en tire moyenne + IC90, comme `Nowcast.summary()`
(`model/core/base.py`). Simple (aucune nouvelle machinerie, juste ne pas
collapser les tirages trop tôt) et robuste (hérite directement des garanties
du postérieur NUTS, §6.1).

### 6.3 Limites connues, non résolues

- **Vraisemblance gaussienne, pas Beta** (§4) : pour un petit candidat
  (Arthaud ~1%), $\mathcal N(\pi, \pi(1-\pi)/N)$ peut mettre de la masse sous 0
  — `linear_pooling` a délibérément choisi Beta pour cette raison précise
  (`spec_linear_pooling.md` §4, "Beta plutôt que Normal"). Pas corrigé ici,
  amélioration identifiée mais pas critique tant que les candidats concernés
  restent petits en probabilité déplacée.
- **Petit échantillon effectif à `half_life` court** : un candidat avec 1-2
  sondages seulement dans la fenêtre de poids significatif peut avoir un
  postérieur mal approximé par les asymptotiques NUTS habituelles — pas
  vérifié spécifiquement.

### 6.4 Vérification empirique — sous-couverture SÉVÈRE (`04d_spatial_coverage_check.py`)

L'argument théorique de §6.1 est incomplet : vérifié par couverture
hors-échantillon (protocole train/test de §5.2, distribution PRÉDICTIVE
POSTÉRIEURE COMPLÈTE — tous les tirages `(mu,sigma,w)` poussés à travers le
softmax masqué + bruit d'échantillonnage du sondage tenu, pas une
approximation gaussienne séparée) sur 2017 (J-30) et 2022 (J-60),
`half_life=15j` :

| élection | IC50 (nominal 50%) | IC80 (nominal 80%) | IC90 (nominal 90%) | n obs |
|---|---|---|---|---|
| 2017 (J-30) | 23,8% | 38,8% | 56,3% | 80 |
| 2022 (J-60) | 31,6% | 47,2% | 54,1% | 231 |

**Le modèle est nettement surconfiant** — les IC90 affichés ne couvrent que
~54-56% des observations réelles (au lieu de 90%). Le fait que 2022 (101
sondages train, 20× plus que 2017) montre une sous-couverture presque
identique à 2017 (5 sondages train) **élimine l'hypothèse d'un artefact de
petit échantillon** — c'est un problème structurel de spécification du
modèle, pas un manque de données.

**Stratification par âge du sondage tenu (`AGE_BUCKETS`,
`04d_spatial_coverage_check.py`) — hypothèse diffusion INFIRMÉE comme cause
dominante.** Deux hypothèses concurrentes envisagées en session : (a) `w_now`
statique sans diffusion temporelle -- l'incertitude ne grandirait pas avec
l'écart entre `as_of` et la date du sondage prédit (un vrai state-space, §3.1,
réglerait ça nativement) ; (b) absence de house effects/variance d'excès
(§10) -- un sondage a une variance réelle supérieure à $\pi(1-\pi)/N$
(méthodologie, `deff`), indépendamment du temps. Ces deux hypothèses
prédisent des signatures DIFFÉRENTES : (a) prédit une couverture qui se
dégrade avec l'âge, (b) prédit une couverture uniformément mauvaise à tout
âge. Résultat sur 2022 (J-60, n=231, largement assez puissant) :

| IC | 0-7j (n=33) | 7-14j (n=66) | 14-21j (n=110) | nominal |
|---|---|---|---|---|
| 50 | 33,3% | 31,8% | 31,8% | 50% |
| 80 | 45,5% | 50,0% | 44,5% | 80% |
| 90 | 57,6% | 59,1% | 50,0% | 90% |

**Couverture PLATE quel que soit l'âge** — même sévèrement sous-couvrante
dès 0-7 jours (57,6% de couverture IC90 sur les prédictions les plus
FRAÎCHES). Ça pointe vers **(b), pas (a)** comme cause dominante : un
mécanisme purement temporel (diffusion manquante) prédirait une bonne
couverture à court horizon, ce qu'on n'observe pas. La cause principale est
donc plus probablement une variance d'observation sous-estimée
(indépendante du temps) qu'une dynamique manquante.

**Implémenté quand même, en parallèle** (session du 2026-08-10, sur
suggestion utilisateur) : `spatial_pooling_model_tau` (`model.py`) --
marche aléatoire dense réelle sur `w` (ligne du temps globale dédupliquée,
`tau` échantillonné par NUTS, prior faible pas encore calibré sur 2017/2022)
remplace la vraisemblance tempérée. Dimension ajoutée modeste sur le roster
réel (N×M ≈ 13×13 = 169). Le diagnostic ci-dessus suggère que ce correctif
seul ne suffira PROBABLEMENT PAS à résoudre la sous-couverture (cause
dominante ailleurs), mais reste une amélioration de spécification légitime
(vraie vraisemblance générative, `tau` proprement identifiable par NUTS
contrairement à `half_life` — cf. §3.1) et à valider par le même protocole de
couverture avant d'être adopté ou écarté.

### 6.5 Confirmation model-free, puis correctif implémenté — sur-correction mesurée

Avant d'implémenter quoi que ce soit, vérification indépendante SANS aucun
modèle (`notebooks/06b_same_day_poll_coherence_matched.py`, suggestion
utilisateur) : comparer des sondages de DEUX instituts différents testant
EXACTEMENT le même champ de candidats (contrôle strict de la composition du
champ), à un écart temporel ≤ 3 jours. $z = (Y_a-Y_b)/\sqrt{\mathrm{Var}_a+\mathrm{Var}_b}$,
$\mathrm{Var} = Y(100-Y)/N$ (bruit d'échantillonnage SEUL, aucun modèle).
Résultat, 2017+2022 (8135 paires, robuste) : **RMS(z)=1,58** (attendu 1,0 si
$\pi(1-\pi)/N$ suffisait), couverture IC90=72,6% (nominal 90%) — **même à
écart nul**, où il n'y a quasiment pas eu le temps pour l'opinion de bouger.
Confirme (b) de façon complètement indépendante du modèle spatial : le bruit
de méthodologie entre instituts, pas la dynamique temporelle, est la cause
dominante (2027 seul, n=84, était trop peu puissant pour trancher : 91,7% de
couverture IC90, proche du nominal, mais pas contradictoire avec le résultat
2017/2022 vu la taille d'échantillon).

**Correctif implémenté** (`excess_var_for_nodes`, `model.py`) : variance
d'excès ajoutée à `pi(1-pi)/N` dans `weighted_loglik`/`spatial_pooling_model`
(nouveau paramètre `excess_var`, (P,) en fraction², optionnel). Réutilise
**telle quelle** la Bank déjà calibrée de `bayesian_nowcast`
(`excess_sigma(bank, institut, horizon=0)`, `BANK_PATH`) plutôt que de
recalibrer un nouveau modèle — le phénomène a été confirmé indépendamment sur
les mêmes deux élections (§ ci-dessus), pas de raison de refitter.

**Résultat (même protocole que §6.4, `04e_spatial_coverage_excess.py`)** :

| élection | IC50 (nominal 50%) | IC80 (nominal 80%) | IC90 (nominal 90%) | avant (§6.4) |
|---|---|---|---|---|
| 2017 (J-30, n=80) | 73,8% | 90,0% | 97,5% | 23,8% / 38,8% / 56,3% |
| 2022 (J-60, n=231) | 84,4% | 97,4% | 99,6% | 31,6% / 47,2% / 54,1% |

**Le sens de l'erreur s'est inversé : le modèle est passé de nettement
surconfiant à nettement SOUS-confiant** (sur-couverture, surtout visible sur
2022 qui est l'échantillon le plus robuste). Bonne nouvelle : c'est un mode
de défaillance beaucoup plus sûr pour une prévision publique (trop prudent
plutôt que trop confiant) et ça confirme sans ambiguïté que la variance
d'excès est le bon LEVIER (le sens de la correction est net et massif dans
les deux cas). Mauvaise nouvelle : l'AMPLITUDE empruntée telle quelle à
`bayesian_nowcast` est mal calibrée pour la géométrie de ce modèle-ci —
attendu, dans une certaine mesure : `excess_sigma` a été calibré pour la
propre vraisemblance du SSM (espace ILR, sa propre décomposition biais +
dérive), pas pour `spatial_pooling`, qui absorbe déjà une partie du
désaccord entre sondages dans l'élargissement naturel du postérieur NUTS
(§6.1) — additionner l'excès BRUT du SSM par-dessus double-compte
probablement une partie de cet effet.

**Statut : correctif dans le bon sens, PAS ENCORE calibré à la bonne
amplitude.** Prochaine étape : soit chercher empiriquement un facteur
d'échelle sur `excess_var` (recherche rapide sur le même protocole de
backtest), soit calibrer un `excess_sigma` propre à `spatial_pooling`
(même esprit que `HouseEffectsCalibration`, mais fit sous la vraisemblance
de CE modèle plutôt qu'emprunté).

### 6.6 Calibration dédiée (`SpatialExcessCalibration`) — bilan honnête

Implémenté (`model/models/spatial_pooling/calibration.py`) : `excess_i =
exp(excess_log_scale + offset_i)` par institut, partial pooling non centré,
fit sur `Y_a - Y_b ~ Normal(0, sampling_a + sampling_b + excess_a² +
excess_b²)` -- paires de SONDAGES de deux instituts différents (pas
d'estimation de niveau nécessaire, la vraie composition du champ étant
identique par construction). Bank propre (`bank_excess.json`), pas empruntée.

**Bug trouvé et corrigé en cours de route (retour utilisateur)** : la
première version appariait au niveau de l'HYPOTHÈSE (`notice`, `hypothese`),
pas du sondage -- or les hypothèses d'un même sondage sont posées aux MÊMES
répondants. Conséquence mesurée : jusqu'à 17 paires d'hypothèses pour un
même couple de sondages, 40% des 702 paires notice×notice provenant de
seulement 50 couples dupliqués -- traitait un même couple de panels comme
autant de mesures indépendantes qu'il y avait d'hypothèses. Corrigé
(`build_notices_core`) : une ligne par SONDAGE (pas par hypothèse), valeur
moyennée sur les candidats CORE (présents dans toutes les hypothèses de ce
sondage -- variation <1pt déjà établie dans ce projet, donc moyenner est
légitime), comparaison sur l'intersection des ensembles core des deux
sondages plutôt que l'égalité stricte. Paires candidat-niveau : 8222 -> 5857
après dédoublonnage.

**Contrôle postérieur (RMS(z) sur les mêmes paires, doit être proche de 1,0
si bien calibré)** :

| version | n paires | échelle globale | RMS(z) |
|---|---|---|---|
| sans excès | -- | -- | 1,47-1,58 |
| Bank empruntée (bayesian_nowcast) | -- | ~2,4-2,8 pt | (sur-corrige, non testé en RMS(z) direct) |
| dédiée, AVANT dédoublonnage | 8222 | 0,368 pt | 1,11 |
| **dédiée, APRÈS dédoublonnage** | 5857 | 0,317 pt | **1,08** |

**Résultat final sur le backtest de couverture (même protocole que §6.4/§6.5)**,
2022 J-60 (n=231, le plus fiable) :

| version | IC50 | IC80 | IC90 | nominal |
|---|---|---|---|---|
| sans excès (§6.4) | 31,6% | 47,2% | 54,1% | 50/80/90% |
| Bank empruntée (§6.5) | 84,4% | 97,4% | 99,6% | 50/80/90% |
| dédiée, avant dédoublonnage | 36,4% | 61,0% | 72,7% | 50/80/90% |
| **dédiée, après dédoublonnage** | **36,4%** | **58,4%** | **69,7%** | 50/80/90% |

**Bilan honnête** : nette amélioration par rapport à l'absence totale de
correction (IC90 passé de 54% à 70%, plus de la moitié du chemin vers le
nominal), sans la sur-correction sévère de la Bank empruntée. Le
dédoublonnage était méthodologiquement nécessaire et correct à faire
(corrige un vrai biais de calibration), mais n'a que MARGINALEMENT changé le
résultat final (l'échelle globale bouge peu, 0,368 -> 0,317 pt) -- il
affectait surtout la confiance interne de la calibration (RMS(z) plus
stable), pas son point central. **La couverture reste, malgré tout,
en-dessous du nominal** (sous-confiant, mode d'échec plus sûr mais pas
résolu) : soit l'échelle de la variance d'excès reste encore un peu petite,
soit une composante temporelle résiduelle (dérive, écartée en §6.4 sur la
base d'un diagnostic qui utilisait alors la mauvaise variance de référence --
à revérifier) contribue aussi. Non réglé plus avant dans cette session
(rendements décroissants, temps déjà conséquent investi) -- documenté comme
limite connue plutôt que présenté comme résolu.

## 7. Roster — filtre et fenêtre temporelle

- Fenêtre : `date_fin >= 2026-01-01` (même `MIN_POLL_DATE` que le SSM en
  production — les sondages "prémonitoires" 2022/2023 sont un régime
  différent, cf. `model/models/bayesian_nowcast/nowcast.py`).
- Filtre : candidat retenu si testé dans **≥ 5 sondages RÉELS distincts**
  (`notice`, pas hypothèses — plusieurs hypothèses d'un même sondage ne sont
  pas des observations indépendantes).
- Sur le roster 2027 actuel (10 août 2026) : 13 candidats retenus sur 26 testés
  au total depuis 2026-01-01 (cf. §2.1 pour la liste).

## 8. Scénarios personnalisés — implémenté (`pi_draws_for_mask`)

Une fois `mu_i`, `sigma_i`, `w_i` fittés (`fit_spatial_pooling`, une inférence
par jour comme le nowcast SSM aujourd'hui), évaluer un **scénario personnalisé**
(sous-ensemble de candidats coché par un visiteur du site) est un simple calcul
direct sur les tirages déjà en mémoire (`pi_draws_for_mask`) — recalcul du
softmax masqué + produit scalaire avec $W$, $O(\text{tirages} \times B)$,
aucune ré-inférence. C'est le point qui justifie toute cette architecture par
rapport au SSM (qui, lui, fige un vocabulaire de slots à l'inférence).
L'intégration à un endpoint du site (quel scénario par défaut, quelle API)
reste à faire — décision explicitement différée par l'utilisateur.

## 9. Saut terminal (nowcast → scrutin) — réutilisé tel quel, implémenté

`forecast_spatial_pooling` (`model.py`) appelle `forecast_from_draws`
(`model/core/simulate.py`) sur les tirages de `pi_draws_for_mask`, exactement
comme `linear_pooling`/`bayesian_nowcast` : même saut sinh-arcsinh calibré sur
2017/2022 (`TerminalJumpCalibration`). Aucune nouvelle spec ici — seule
particularité : `candidate_blocs_2027` résout les noms COURTS de
`load_raw_polls` ("Le Pen") vers les noms COMPLETS de `candidat_blocs.csv`
("Marine Le Pen", via `pipeline.historical._resolve_candidate`) pour donner à
`forecast_from_draws` le bloc politique de chaque candidat (loc/scale du saut
par bloc). `Bank` propre à ce modèle (`bank_jump.json`, `bank=None` au fit —
même choix que `linear_pooling`, cf. §9 de sa spec) : `spatial_pooling` ne
modélise pas de house effects côté nowcast, débiaiser son movement pool
historique avec la Bank house-effects de `bayesian_nowcast` serait incohérent.

## 10. Limitations connues / travaux non faits

- **Variance d'excès (house effects) implémentée mais pas parfaitement
  calibrée** (§6.6) — réduit la sous-couverture sans l'éliminer (IC90
  empirique ~70% au lieu de 90% sur 2022, contre 54% sans elle) ; reste
  sous-confiant, mode d'échec plus sûr que la sur-confiance mais pas résolu.
  Pas de composante par bloc/candidat, ni dépendante de l'horizon (§6.4 avait
  écarté la piste diffusion sur la base d'un diagnostic utilisant alors une
  mauvaise variance de référence — à revérifier maintenant que la variance
  de base est corrigée).
- **`half_life` non affiné** : balayage grossier {15,45,∞}, pas de recherche
  autour de 15j (ex. 5/10/15/20/25j).
- **Jauge de `w` non fixée** (§3.2) — inoffensif tant que seul `pi` est
  consommé, à corriger avant d'afficher `w` comme diagnostic.
- **Rattachements de groupe non tranchés avec l'utilisateur** : Dupont-Aignan
  (groupe MLP), Hollande (groupe PS) — choix par défaut, cf. §2.1 (Bardella
  tranché : retiré, Le Pen candidate officielle).
- **EKF/UKF explicitement reporté** : la vraisemblance tempérée est une
  approximation pragmatique, pas la version "proprement bayésienne" — cf. §3.1
  pour ce que donnerait un vrai state-space (dynamax en a déjà l'implémentation).
- **Pas branché sur `ForecastModel`** (`model/core/base.py`) ni sur le contrat
  de sortie du site (`assemble_snapshot`/`validate_snapshot`) — décision
  explicite de l'utilisateur, à traiter dans une étape séparée.
- **Grille $W$ uniforme** — l'extension "densité électorale en cloche" de la
  spec d'origine n'a pas été testée.
