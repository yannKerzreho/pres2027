# Spec — modèle spatial (Hotelling-Downs discrétisé)

Statut : **validé sur prototype** (synthétique + données réelles 2017/2022/2027)
**et peuplé dans `model.py`** (ce dossier) — maths, fit, lecture avec
incertitude et saut terminal. **Pas encore branché** sur `ForecastModel` / le
contrat de sortie du site (décision explicite de l'utilisateur, à traiter
séparément). Prototypes de référence (historique des découvertes, pas le code
de production) : `notebooks/_spatial_core.py`, `03_spatial_prototype.py`,
`03b_spatial_debug.py`, `04_spatial_real_data.py`, `04b_spatial_halflife_backtest.py`,
`04c_spatial_sanity_check.py`.

**Ordre de lecture** : les sections sont dans l'ordre où les découvertes ont eu
lieu, pas dans l'ordre de l'état actuel. Deux conclusions y ont été RÉVISÉES
depuis (encadrés en place) : la lecture de `w_now` par vraisemblance tempérée
(§3.1) est remplacée par §11, et le diagnostic « la diffusion n'est pas la
cause dominante » (§6.4) est erroné pour une raison identifiée en §11.1.

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

> **DÉPASSÉ (§11).** Cette section décrit toujours le FIT, qui est inchangé,
> mais `w_now` n'est plus LU depuis ce postérieur : la vraisemblance tempérée
> n'a aucun plancher de variance temporel (démonstration en §11.1), ce qui la
> rend structurellement incapable de porter l'incertitude du nowcast. Elle ne
> sert plus qu'à rendre tenable l'hypothèse d'un `w` statique pendant
> l'estimation de la géométrie.

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

> **CONCLUSION RÉVISÉE (§11.1).** La stratification par âge ci-dessous conclut
> que la diffusion n'est pas la cause dominante. **Ce raisonnement est faux** :
> la signature testée est l'âge du sondage TENU (0-21 j après la coupure),
> alors que le plancher de variance manquant est piloté par l'âge des sondages
> d'ENTRAÎNEMENT (15-60 j avant), identique pour tous les points du test. Une
> couverture plate est donc exactement ce qu'un plancher manquant prédit — ce
> test ne pouvait pas départager les deux hypothèses. La cause dominante EST
> temporelle, cf. §11.

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

## 9. Saut terminal (nowcast → scrutin) — machinerie PARTAGÉE (session du 2026-08-12)

Révisé : le projet a remplacé, pour TOUS les modèles, l'ancien saut sinh-arcsinh
par-modèle par une machinerie commune à noyau Ornstein-Uhlenbeck --
`model/core/opinion.py` (banque commune `bank_opinion.json`, REML sur
2017/2022 : diffusion d'opinion `sigma2`/`tau`, écarts d'institut `sigma_h`/
`sigma_p`, écart sondages→urne `delta_*`) + `model/core/projection.py::projeter_au_scrutin`
(applique la diffusion OU puis δ, en espace log-ratio). Plus aucun modèle
n'a de Bank de saut qui lui soit propre -- `spatial_pooling` n'échappe pas à
la règle, `bank_jump.json`/`TerminalJumpCalibration`/`candidate_blocs_2027`
ont disparu de ce dossier.

`forecast_spatial_pooling` (`model.py`) : `pi_draws_for_mask` -> `projeter_au_scrutin(pi,
horizon_days, load_law())` -> `forecast_from_draws(pi, labels, horizon_days)`
(dont la signature a aussi changé -- `jump_bank`/`candidate_blocs` ont disparu,
la dérive est désormais appliquée par l'appelant AVANT, `model/core/simulate.py`
ne fait plus que le comptage top2/duels). Aucune spec propre à ce modèle ici,
`model/core/opinion.py`/`projection.py` documentent la justification (mean-
reversion contredisant l'ancienne loi en `√h`, dispersion PLATE de 63 à 267
jours mesurée sur le pool).

**Point ouvert, pas encore traité** : ce mécanisme partagé est calibré en
espace CLR/log-ratio, sur des données historiques où le roster de candidats
est FIXE par élection (`filter_scenarios_by_exact_slots` pour `gp_pooling`).
`spatial_pooling` a un roster qui change selon le scénario coché (§8) -- la
diffusion `2σ²(1−e^{-h/τ})` s'applique bien au niveau `pi` (indépendante du
roster), donc `projeter_au_scrutin` reste correct tel quel, mais ça n'a pas
été vérifié empiriquement sur ce modèle spécifiquement (contrairement à
`gp_pooling`, dont les critères d'acceptation §8 de sa spec sont mesurés).

## 10. Limitations connues / travaux non faits

- **Variance d'excès (house effects)** (§6.6) — la sous-couverture qu'elle
  laissait subsister (IC90 ~70 % au lieu de 90 %) est **traitée en §11** : sa
  cause n'était pas l'amplitude de l'excès mais l'absence de plancher de
  variance temporel. La Bank d'excès elle-même reste employée telle quelle,
  sans composante par bloc/candidat ni dépendante de l'horizon.
- **`sigma_w2` ne vaut pas la même chose sur 2017 et sur 2022** (§11.5) :
  facteur ~4 entre les deux optimums de couverture. Une loi de diffusion
  stationnaire ne peut pas être juste sur les deux ; le réglage retenu suit
  2022 (échantillon fiable) et sous-couvre sur 2017.
- **`half_life` non affiné** : balayage grossier {15,45,∞}, pas de recherche
  autour de 15j (ex. 5/10/15/20/25j). Depuis §11 il ne pilote plus QUE la
  géométrie (`mu`, `sigma`), plus l'incertitude de `w` — son réglage est donc
  devenu beaucoup moins critique. Il reste nécessaire : `spatial_pooling_model`
  n'a qu'un `w` statique, et c'est la températion qui rend cette hypothèse
  tenable sur toute une campagne.
- ~~Fit NUTS de géométrie non convergé sur le roster 2027~~ — **RÉSOLU**
  (§12.4) : prior de position mal centré. R-hat 1,62 → 1,002, ESS 8,6 → 1970,
  67 divergences → 0.
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
- ~~Grille $W$ uniforme, extension "densité électorale en cloche" non testée~~
  — **CLOS (§12.6) : la densité électorale n'est pas identifiable séparément
  des positions.** Ce n'est pas une extension à faire, c'est un degré de
  liberté redondant.

## 11. Dynamique de `w` — inversion locale exacte + OU en forme close (session du 2026-08-12)

Statut : **implémenté et calibré** (`w_dynamics.py`, `bank_w_ou.json`), branché
par défaut sur `fit_spatial_pooling(w_dynamics=True)`. Remplace la LECTURE de
`w_now` (§3.1) ; le fit NUTS lui-même est **inchangé** et ne sert plus qu'à la
géométrie (`mu`, `sigma`). Résultat en une ligne : sur 2022 (l'échantillon
fiable, n=231) la couverture IC90 passe de **66,2 % à 93,9 %** (nominal 90),
dont 66,2 → 78,8 par le seul plancher de variance à convention de lecture
identique. Sur 2017 (n=80) elle passe de 66,2 % à 72,5 %, nettement en dessous
du nominal — cf. §11.5 pour ce désaccord entre les deux élections, qui n'est
PAS résolu.

### 11.1 Diagnostic — pourquoi la vraisemblance tempérée ne pouvait pas y arriver

§6.6 laissait deux hypothèses ouvertes pour la sous-couverture résiduelle
(IC90 ≈ 70 % au lieu de 90 %) : échelle d'excès encore trop petite, ou
composante temporelle résiduelle. C'est la seconde, et pour une raison
**structurelle**, pas d'amplitude.

La variance de Laplace du modèle tempéré vaut
$\big(\text{prior}^{-1} + \sum_p \kappa_p I_p\big)^{-1}$ : elle décroît en
$1/n$ **sans borne inférieure**, quel que soit l'âge des sondages. Un
Ornstein-Uhlenbeck impose au contraire un plancher
$\sigma_w^2(1 - e^{-2g/\tau_w})$ qui ne dépend QUE de l'écart $g$ au dernier
sondage et qu'aucune accumulation ne franchit. Mesuré (grille réaliste,
$N_p{=}1000$, excès de §6.6, candidat médian à 7,3 %) :

| âge des sondages | n=1 | n=5 | n=20 | n=100 | plancher OU seul |
|---|---|---|---|---|---|
| 15 j | 1,176 | 0,530 | 0,266 | 0,119 | **0,805** |
| 30 j | 1,646 | 0,748 | 0,375 | 0,168 | **1,107** |

(écart-type de `pi` en points). À 30 jours et 20 sondages, le modèle actuel
annonce 0,375 pt là où le seul plancher en vaut 1,107 — et il descend encore à
0,017 pt à n=10 000. **Aucun choix de $\kappa_p$ ne corrige ça** : la famille
tempérée est incapable de produire un plancher, puisque sa variance tend vers 0
avec $n$ pour tout $\kappa > 0$ fixé.

**Ceci corrige aussi le raisonnement de §6.4.** La stratification par âge y
était PLATE, et on en avait conclu « la diffusion n'est pas la cause
dominante ». La signature testée était la mauvaise : ce qui pilote le plancher
n'est pas l'âge du sondage TENU (0-21 j après la coupure) mais l'âge des
sondages d'ENTRAÎNEMENT (15-60 j avant), qui est le même pour tous les points
du test. Une couverture plate est donc exactement ce qu'un plancher manquant
prédit — §6.4 ne pouvait pas trancher.

### 11.2 Le mécanisme

Trois temps, aucun MCMC (cf. l'en-tête de `w_dynamics.py` pour le détail).

1. **Inversion locale EXACTE.** Pour chaque nœud on résout $\pi(w) = \tilde Y_p$
   sur le champ testé. Ce n'est pas une linéarisation : c'est bien posé, parce
   que $\pi = \nabla\Phi$ avec
   $\Phi(w) = \sum_b W_b\,\mathrm{logsumexp}_{j\in S_p}(w_j + D_{jb})$, dont le
   hessien est
   $J = \sum_b W_b\,(\mathrm{diag}(P_b) - P_b P_b^\top)$ — **SDP, de noyau
   exactement $\mathrm{span}(1)$** (la jauge de §3.2 ; vérifié contre
   `jax.jacfwd` à $10^{-17}$ près). $\Phi$ est donc strictement convexe sur
   $\{\sum w = 0\}$ et l'inversion y est un difféomorphisme (dualité de
   Legendre). Newton amorti converge en **6 itérations médianes** (p95 : 16),
   exactement, y compris sur une composition contenant un candidat à 0,5 %.

   C'est la différence de fond avec l'EKF parqué (§3.ter) : l'EKF linéarise
   autour de la moyenne PRÉDITE, à chaque nœud, et re-différentie le tout sous
   NUTS (44 min sans converger sur le roster réel). Ici on linéarise autour de
   **la donnée**, une fois, hors de tout MCMC.

2. **Méthode delta.** $\hat w_p \approx w(t_p) + \text{bruit}$, de covariance
   $C_p = J_p^{+}\Sigma_p J_p^{+}$. On a donc une pseudo-observation
   **linéaire-gaussienne** de $w(t_p)$ — la situation exacte où `gp_pooling`
   sait travailler en forme close, et que l'agrégation sur la grille
   interdisait jusqu'ici.

3. **Krigeage universel à noyau OU.** $w_i(t) = m_i + u_i(t)$,
   $u \sim \mathrm{OU}(0,\sigma_w^2,\tau_w)$ indépendant par candidat, $m$
   marginalisé par prior plat — même choix que `gp_math.gp_posterior`, et pour
   la même raison : aux grands écarts la prévision revient vers le niveau MOYEN
   estimé du candidat, pas vers 0 (qui n'aurait aucun sens), avec une variance
   qui sature proprement. Une seule factorisation de Cholesky
   $(P{\cdot}N)\times(P{\cdot}N)$ par tirage de géométrie.

**Jauge et masques.** Chaque nœud teste un champ différent, donc n'informe que
les CONTRASTES internes à $S_p$. C'est encodé exactement, sans bricolage, par
le projecteur orthogonal $C_{S_p} = \mathrm{diag}(m_p) - m_p m_p^\top/K_p$ pris
comme matrice de design du nœud ; les directions non identifiées reçoivent
`PADDING_VAR`, même procédé que dans `bayesian_nowcast/latent.py`.

**`tau_w` n'est pas réestimé** : repris de la banque COMMUNE
(`model/core/opinion.py`, $\tau \approx 262$ j). Deux raisons — (a)
*identifiabilité* : sur une fenêtre de campagne tous les écarts sont petits
devant $\tau_w$ et $\sigma_w^2(1-e^{-\Delta/\tau_w}) \approx \sigma_w^2\Delta/\tau_w$,
seul le RAPPORT est contraint (mesuré sur 2017 : `(0,2 ; 262 j)` et
`(0,4 ; 500 j)` sont à 0,0 de nll l'un de l'autre — crête plate) ; (b)
*cohérence* : $\tau$ mesure la vitesse de retour de l'opinion, la même pour
tous les modèles du dépôt, seule l'AMPLITUDE dépend de l'espace de
paramétrage. C'est exactement l'argument de `opinion.py` sur ce qui se reprend
et ce qui ne se reprend pas du saut terminal.

### 11.3 Validation synthétique (vérité terrain connue, aucun NUTS)

`scratchpad t3_synth.py` — 10 candidats, chemin OU simulé sur 150 jours, 80
nœuds de champ et de taille aléatoires, bruit d'échantillonnage + excès.
Couverture de `pi(as_of)` sur 150 réplications indépendantes :

| lecture de `w` | IC50 | IC80 | IC90 |
|---|---|---|---|
| **OU + inversion locale** | **50,8 %** | **80,7 %** | **89,8 %** |
| tempérée (half_life=15 j), MÊME jeu | 15,7 % | 30,4 % | 38,3 % |

Les deux propriétés qualitatives demandées sont vérifiées séparément :

- **des sondages qui se confirment resserrent** : à écart fixe de 10 j,
  sd(`pi`) passe de 1,109 pt (n=1) à 0,609 pt (n=32) — et **sature** au
  plancher OU au lieu de tendre vers 0 ;
- **le temps qui passe élargit** : pour 16 sondages groupés, sd(`pi`) va de
  0,173 pt (écart 0) à 1,854 pt (écart 120 j), en suivant de près le plancher
  (0,472 vs 0,429 à 5 j ; 1,042 vs 0,988 à 30 j).

### 11.4 Identifiabilité mesurée sur données réelles — la crête est bien plate

Profil REML des pseudo-observations (`notebooks/04g_spatial_w_ou.py eval`),
2022 J-60 (276 nœuds, 11 candidats) :

| σ_w² \ τ_w | 60 j | 120 j | 262 j | 500 j |
|---|---|---|---|---|
| 0,1 | **2189,7** | 2195,6 | 2222,0 | 2263,0 |
| 0,2 | 2202,9 | 2190,7 | 2196,2 | 2217,9 |
| 0,4 | 2243,6 | 2208,1 | **2191,6** | 2195,2 |
| 0,8 | 2316,7 | 2251,2 | 2207,8 | 2192,9 |

`(0,1 ; 60 j)`, `(0,2 ; 120 j)`, `(0,4 ; 262 j)` et `(0,8 ; 500 j)` tiennent
dans **3,2 unités de nll** — sur un intervalle de 8× en `σ_w²`. La crête
annoncée en §11.2 n'est donc pas un artefact du synthétique : laisser les deux
paramètres libres reviendrait à choisir un point arbitraire dessus.
`tau_w = 262 j` (banque commune) est fixé, `σ_w²` seul est calibré.

### 11.5 Backtest de couverture 2017/2022 — protocole de §6.4 inchangé

`notebooks/04g_spatial_w_ou.py`. Le fit NUTS de géométrie est celui de 04f
(`half_life=15 j`, variance d'excès de §6.6) ; seule la lecture de `w` change.
2017 J-30 : 21 nœuds train / n=80 test. 2022 J-60 : 276 nœuds train / n=231
test (l'échantillon fiable, cf. §5.2).

**Deux conventions de lecture, à ne pas mélanger.** La lecture tempérée est
figée à `as_of` *par construction* (elle n'a aucun mécanisme temporel), alors
que l'OU peut prédire `w` **à la date du sondage tenu**. C'est légitime — la
date d'un sondage est une métadonnée connue, pas sa valeur — et c'est ce que
fait une vraie prévision ; le protocole de §6.4/§6.5/§6.6 comparait en fait un
nowcast à `as_of` à un sondage publié jusqu'à 21 jours plus tard, confondant
erreur d'estimation et dérive réelle. Les deux sont donc rapportées :

| lecture de `w` | 2017 IC50/80/90 | 2022 IC50/80/90 | pooled (n=311) |
|---|---|---|---|
| tempérée (§3.1 + §6.6), figée | 28,7 / 51,2 / 66,2 | 35,1 / 58,4 / 66,2 | 33,4 / 56,6 / 66,2 |
| OU σ_w²=0,2, **figée à as_of** | 26,2 / 52,5 / 65,0 | 42,4 / 70,1 / 78,8 | 38,2 / 65,6 / 75,2 |
| OU σ_w²=0,1, à la date du sondage | 40,0 / 62,5 / 72,5 | 46,3 / 77,5 / 88,3 | 44,7 / 73,6 / 84,2 |
| OU σ_w²=0,13, à la date du sondage | 41,2 / 65,0 / 72,5 | 48,9 / 79,2 / 89,6 | 46,9 / 75,5 / 85,2 |
| OU σ_w²=0,16, à la date du sondage | 43,8 / 65,0 / 71,2 | 52,8 / 82,3 / 92,2 | 50,5 / 77,9 / 86,8 |
| **OU σ_w²=0,2, à la date du sondage** | 43,8 / 66,2 / 72,5 | 55,8 / 84,8 / 93,9 | **52,7 / 80,0 / 88,4** |
| OU σ_w²=0,4, à la date du sondage | 52,5 / 68,8 / 80,0 | 62,8 / 89,6 / 96,5 | 60,2 / 84,2 / 92,3 |
| OU σ_w²=0,8, à la date du sondage | 60,0 / 83,8 / 91,2 | 74,9 / 95,2 / 99,1 | 71,1 / 92,3 / 97,1 |

Les deux gains se décomposent proprement sur 2022 (IC90) : **66,2 → 78,8** par
le seul plancher de variance, à convention de lecture IDENTIQUE à la tempérée ;
puis **78,8 → 93,9** en prédisant à la date du sondage.

**`σ_w² = 0,2` retenu** (`bank_w_ou.json`) : c'est le réglage dont l'écart
total au nominal sur l'échantillon *pooled* est le plus faible (4,3 points
cumulés sur les trois niveaux, contre 5,8 à 0,16 et 12,4 à 0,13), et il tombe
dans la zone plate du REML (Δnll = 3,2 vs l'argmin à τ=262 j). `σ_w² = 0,16`
lui est statistiquement indiscernable.

**Ce qui reste franchement imparfait, et qu'il ne faut pas maquiller** :

1. *Le réglage optimal n'est pas le même sur les deux élections.* 2022 est
   quasi-nominal à `σ_w² ≈ 0,13` (48,9 / 79,2 / 89,6) ; 2017 n'atteint le
   nominal qu'à `σ_w² ≈ 0,8`, où 2022 sur-couvre massivement (IC90 99,1 %).
   Facteur ~5. Interprétation la plus plausible, NON vérifiée : 2017 J-30 n'a
   que **5 sondages d'entraînement** (limite de la source Wikipedia, §5.2) et
   couvre le dernier mois de campagne, exceptionnellement volatil — une loi de
   diffusion STATIONNAIRE ne peut pas être juste sur les deux à la fois. Le
   choix retenu privilégie 2022 (3× plus d'observations de test, 13× plus de
   nœuds d'entraînement).
2. *À `σ_w²=0,2`, 2022 sur-couvre légèrement et 2017 sous-couvre* — le pooled
   n'est au nominal que parce que les deux erreurs se compensent. C'est un
   compromis assumé, pas une réussite sur chaque élection prise à part.

### 11.6 Effet d'institut persistant (`sigma_house`) — testé, PAS adopté

Le solveur accepte un terme `sigma_house` (effet partagé par tous les sondages
d'un même institut, comme `sigma_h` dans `gp_math.gp_posterior`). Testé à la
valeur de la banque commune (0,126) :

| réglage | 2017 | 2022 | pooled |
|---|---|---|---|
| σ_w²=0,10, σ_house=0,126 | 46,2 / 67,5 / 77,5 | 50,6 / 81,8 / 89,2 | 49,5 / 78,1 / 86,2 |
| σ_w²=0,13, σ_house=0,126 | 50,0 / 68,8 / 80,0 | 52,8 / 82,7 / 90,5 | 52,1 / 79,1 / 87,8 |

Il fait **mieux** que le réglage retenu : il rapproche 2017 du nominal sans
dégrader 2022. Il n'est pourtant pas adopté, pour une raison de méthode : ce
serait un **second paramètre libre réglé sur le même backtest** que le premier,
et son articulation avec la variance d'excès de §6.6 n'est pas tranchée. Les
deux termes n'ont pas la même FORME selon la taille du candidat — l'excès de
§6.6 est une constante en points de `pi` (il corrige donc surtout les petits
candidats), `sigma_house` vit dans l'espace `w` et son effet suit la courbure
multinomiale (mesuré : 0,14 pt sur un candidat à 1 %, 1,64 pt sur un candidat
à 18 %, donc il élargit surtout les GROS). Ils sont peut-être complémentaires
plutôt que redondants — mais c'est aux 8135 paires model-free de §6.5 de
l'arbitrer (l'excès y a été ajusté en supposant une forme constante en points,
hypothèse jamais testée contre l'autre), pas au backtest de couverture. À
traiter dans une session dédiée.

### 11.7 Contrôle sur le roster 2027 (`notebooks/04h_spatial_w_ou_2027.py`)

Géométrie fittée une fois au 2026-07-10 (dernier sondage), puis `w` relu à
plusieurs `as_of`. Largeur de l'IC90 sur `pi`, scénario « tous les candidats » :

| candidat | moy. | as_of+0 | +15 j | +30 j | +60 j | tempérée |
|---|---|---|---|---|---|---|
| Le Pen | 30,8 % | 3,32 | 9,14 | 12,33 | 16,58 | 3,59 |
| Philippe | 15,7 % | 2,65 | 5,50 | 7,12 | 9,92 | 2,06 |
| Mélenchon | 12,8 % | 2,55 | 4,71 | 6,02 | 8,08 | 2,62 |
| Glucksmann | 8,6 % | 1,57 | 4,34 | 5,42 | 7,13 | 1,39 |
| Zemmour | 3,0 % | 0,92 | 1,83 | 2,33 | 3,04 | 0,85 |
| Arthaud | 1,0 % | 0,46 | 0,68 | 0,82 | 1,13 | 0,46 |

Deux lectures. **(a)** À `as_of+0`, les deux méthodes donnent quasiment la même
largeur (3,32 vs 3,59 pour Le Pen) — le mécanisme ne gonfle rien gratuitement
quand il n'y a pas de temps écoulé. **(b)** La colonne `tempérée` est la MÊME
quel que soit `as_of` : c'est exactement le défaut de §11.1, rendu visible sur
données réelles. Le job quotidien tourne à J+33 du dernier sondage au moment
de l'écriture — la lecture tempérée y annoncerait toujours 3,59 pt sur Le Pen.

**Contrôle de magnitude indépendant du backtest** : la loi de diffusion
COMMUNE du dépôt (`bank_opinion.json`, σ²=0,1158, τ=262 j, calibrée par REML
sur 2017/2022 en espace CLR) implique à elle seule, pour un candidat à 30,8 %,
un IC90 de 11,58 / 16,23 / 22,55 pt à +15 / +30 / +60 j. Le mécanisme en
produit 9,14 / 12,33 / 16,58 — **du même ordre, systématiquement ~25 % plus
étroit** (normal : le krigeage conserve l'information de l'historique de
sondages, alors que le calcul de diffusion pure part d'un point). `σ_w² = 0,2`
n'est donc pas un réglage arbitraire ajusté sur un backtest : il correspond à
une diffusion un peu plus lente que celle que le dépôt mesure par ailleurs.

*Limite de ce contrôle* : au 2026-07-10 les 12 candidats du roster ont TOUS été
testés le jour même (`écart j-0` = 0 partout), donc la différenciation
PAR CANDIDAT de l'écart au dernier test — le cas Bardella de §5.3 — n'est pas
exercée par ce jeu de données. Le mécanisme l'implémente (chaque candidat n'est
informé que par les nœuds qui le testent) et la différenciation par quantité
d'information est bien visible (Hollande, 6,4 %, a 2,06 pt là où Retailleau,
7,2 %, en a 1,34), mais l'effet de l'écart lui-même reste à vérifier sur un
roster où il varie.

### 11.8 Coût

Sur le roster 2027 réel (12 candidats, 78 nœuds, `D = P·N = 936`) :
pseudo-observations **~40 s**, postérieur OU + tirages **~6 s**, pour 200
tirages de géométrie — soit **~1 min** ajoutée à un fit qui en prend ~6.
La factorisation reste praticable bien au-delà : sur 2022 J-60 (276 nœuds,
`D = 3036`) un Cholesky complet prend 0,8 s.

Le risque de budget CI (§ contrainte `timeout-minutes: 30`) n'est donc PAS ce
mécanisme mais le fit NUTS de géométrie qui le précède (353 s en local sur
2027, davantage sur un runner GitHub). Optimisations évidentes si besoin :
réduire `n_geom` (la résolution est en `O((P·N)³)` par tirage), ou exploiter
le fait que le noyau OU est markovien pour passer à un filtre de Kalman en
`O(P·N³)` — non fait, inutile aux tailles actuelles.

## 12. Trou de normalisation — la vraisemblance était structurellement insatisfiable (session du 2026-08-12)

Trouvé en cherchant pourquoi le fit NUTS de géométrie ne convergeait pas sur le
roster 2027 (R-hat 1,62, ESS 8,6, cf. §10) alors qu'il converge sans problème
sur 2022. **Ce n'était pas un problème d'échantillonnage.**

### 12.1 Le défaut

`spatial_shares` renvoie un `pi` qui **somme à 1** sur le champ masqué : c'est
un softmax normalisé sur les candidats testés, agrégé sur la grille. La
vraisemblance (§4) compare ce `pi` à `Y`, les intentions rapportées restreintes
au roster du modèle. Or `Y` ne somme à 1 que si le roster couvre TOUT le
bulletin. Mesuré, somme des intentions par nœud :

| jeu | médiane | min | nœuds sous 0,80 |
|---|---|---|---|
| 2017 | 0,955 | 0,470 | 12 / 35 |
| 2022 | 0,940 | 0,750 | 19 / 356 |
| **2027 (roster d'alors)** | **0,660** | **0,425** | **46 / 78** |

Sur 2027, la vraisemblance demandait donc à `(mu, sigma, w)` de produire un
`pi` sommant à 0,66 — ce qu'**aucune** valeur ne peut faire. NUTS ne diverge
pas au hasard : il est poussé dans des directions arbitraires pour absorber un
écart irréductible, ce qui aplatit la géométrie et fait converger les chaînes
vers des compromis différents. Sur 2017/2022 l'écart n'était que de ~5 %, absorbé
comme un biais quasi uniforme — d'où le contraste entre les deux.

**Cause immédiate** : Bardella avait été retiré de `ORDER_GROUPS` (§2.1) au motif
que Le Pen est la candidate officielle du RN. Mais c'est un choix de **scénario**,
et ce modèle choisit ses scénarios **à la lecture** (`pi_draws_for_mask`, §8),
pas au fit. Appliqué au roster, il retirait ~35 % du bulletin du modèle sans le
retirer des données : Bardella est testé dans **46 nœuds sur 78**.

Ce défaut est probablement aussi à la racine du **mode dégénéré de §3.bis**
(les `mu` s'effondrant entre 0,70 et 0,99) : même cause, le modèle déformant sa
géométrie pour approcher une contrainte impossible. Non revérifié.

### 12.2 Les correctifs

1. **Bardella réintégré au roster de fit** (groupe MLP). Somme médiane par
   nœud : **0,660 → 1,000**, plus aucun nœud sous 0,80. Pour n'afficher que Le
   Pen, on masque Bardella à la lecture — ce pour quoi §8 existe.
2. **Sous-composition explicite** (`build_poll_arrays`) : `Y` est renormalisé
   sur le champ modélisé et `N_p` déflaté d'autant. Ce n'est pas un rustinage :
   la restriction d'un multinomial à un sous-ensemble EST un multinomial, de
   taille `N_p·ΣY` (les répondants exprimant une préférence pour un candidat du
   roster). Corrige aussi le résidu de ~5 % sur 2017/2022. Après correctif, les
   trois jeux ont une somme de 1,0000 exactement.
3. **Roster élargi et garde-fou** (`build_roster`) : tout candidat à
   `>= MIN_POLLS` sondages distincts **et** testé depuis moins de
   `MAX_LAST_POLL_AGE_DAYS` (90 j, non calibré — garde-fou, aucun éligible n'en
   approche aujourd'hui) est modélisé. Un candidat éligible absent de
   `ORDER_GROUPS` **lève désormais une erreur** au lieu d'un `warning` : c'est
   précisément le `warning` silencieux qui a laissé passer ce défaut. Les
   groupes vides sont élagués, ce qui permet à `ORDER_GROUPS` de pré-classer
   des candidats encore sous le seuil sans déformer la séquence ordonnée.

Roster 2027 après correctifs : **13 candidats, 9 groupes**, Hollande et
Bardella inclus.

**Le plus proche du seuil, à trancher** : **Villepin** (4 sondages, encore
testé le 2026-07-10, 3,8 %). Pré-classé dans un groupe propre entre PS et
Attal, mais ce placement est réellement ambigu (ex-Premier ministre gaulliste,
donc à droite par trajectoire, qui capte aujourd'hui un électorat de
gauche/centre-gauche). Il basculera dans le roster au prochain sondage qui le
teste. Même statut, moins urgent : Darmanin/Lisnard → LR, Lecornu → Attal,
Knafo → Zemmour.

### 12.3 Ce que ça implique pour §11

La calibration de `sigma_w2` de §11.5 a été faite **avant** ces correctifs.
Les fits de géométrie 2017/2022 employés sont donc entachés du résidu de ~5 %
(pas du trou de 34 %, propre à 2027). L'ordre de grandeur des conclusions de
§11 n'est pas en cause — le diagnostic du plancher de variance manquant (§11.1)
est analytique et indépendant des données — mais **`sigma_w2 = 0,2` doit être
réétalonné** sur des fits corrigés. En cours.

### 12.4 Le prior de position était mal centré — vraie cause du R-hat 1,62

Le trou de normalisation (§12.1) corrigé, le fit 2027 ne convergeait **toujours
pas**. Le diagnostic par site montrait `mu`/`mu_slot_pos`/`mu_base` à R-hat
~1,6 mais `mu_gaps` à 1,06 : les ÉCARTS entre groupes étaient identifiés, pas
la position absolue. Les moyennes par chaîne montraient la configuration
entière glissant le long de l'axe (Arthaud à 0,055 sur deux chaînes, 0,267 et
0,411 sur les deux autres).

**Première hypothèse, DÉMENTIE** : une invariance par translation de la
vraisemblance (`W` uniforme sur la grille ⇒ translater tous les `mu` laisserait
`pi` inchangé). Testé directement : la vraisemblance perd **62 unités** pour
δ = ±0,05 et 1072 pour δ = +0,20. Elle est au contraire très piquée. Ce n'était
donc pas une direction plate.

**Cause réelle** : les chaînes étaient dans des modes de qualité TRÈS
différente. Log-vraisemblance à la moyenne postérieure de chaque chaîne : 699,1
et 698,6 (chaînes 1-2) contre −79,5 et −233,4 (chaînes 0 et 3), soit **932
unités d'écart**. Deux chaînes sur quatre n'avaient simplement jamais trouvé le
bon mode.

Pourquoi ? `raw_k = base + Σ_{j≤k} gap_j` ne fait que **croître** depuis 0. À la
médiane du prior — c'est-à-dire au point de départ d'`init_to_median` — les
positions valent `sigmoid([0 ; 0,6 ; ... ; 4,8])` :

| | positions initiales |
|---|---|
| 6 groupes (2017/2022) | 0,50 · 0,65 · 0,77 · 0,86 · 0,92 · 0,95 |
| **9 groupes (2027)** | 0,50 · 0,65 · 0,77 · 0,86 · 0,92 · 0,95 · **0,974 · 0,986 · 0,992** |

Toute la configuration démarre dans la moitié DROITE de la grille, et avec 9
groupes les quatre derniers démarrent **empilés entre 0,974 et 0,992**, là où
la dérivée du sigmoid vaut ~10⁻³ : gradient quasi nul, bassin dont on ne sort
pas. **C'est le nombre de groupes qui explique le contraste 2022/2027** : avec
6 blocs le défaut est gênant mais franchissable, avec 9 il ne l'est plus. 2022
convergeait par chance de dimension, pas par robustesse — le défaut y était
déjà présent.

**Correctif** (`sample_ordered_slots`) : centrer la somme cumulée,
`raw = base + cum − mean(cum)`. À `base = 0` la configuration devient
`sigmoid([−2,4 ; ... ; +2,4])` = `[0,08 ; ... ; 0,92]`, qui couvre la grille.
`base` reste libre : c'est une re-paramétrisation de la LOCALISATION du prior,
pas une contrainte. Accessoirement le prior d'origine affirmait que le champ
politique occupe la droite de l'axe — ce qui n'a aucun sens, l'axe étant latent.

**Résultat sur le roster 2027** (13 candidats, 78 nœuds, 9 groupes) :

| | avant | après |
|---|---|---|
| R-hat max | 1,616 | **1,002** |
| ESS min | 8,6 | **1970** |
| divergences | 67 | **0** |

Les 4 chaînes s'accordent désormais à la 3ᵉ décimale sur chaque candidat, et la
géométrie est interprétable : Arthaud 0,021 < Mélenchon 0,042 < Roussel/Tondelier
0,09 < Glucksmann/Hollande 0,15 < Attal 0,34 < Philippe 0,41 < Retailleau 0,63 <
Le Pen/Bardella/Dupont-Aignan 0,82 < Zemmour 0,89.

**À revérifier** : le mode dégénéré de §3.bis (`spatial_pooling_model_tau`, les
`mu` effondrés entre 0,70 et 0,99) ressemble beaucoup à ce mauvais bassin —
mêmes positions écrasées à droite. Il a probablement été condamné par ce défaut
d'initialisation plutôt que par sa propre spécification. Non revérifié.

### 12.5 Prior de position ancré sur 2022 — implémenté, PAS activé

Piste proposée en session : informer les positions des groupes 2027 par celles
des blocs politiques mesurées sur 2022, plutôt que de partir d'un prior faible.
§2 avait écarté l'ancrage historique pour deux raisons — les nouveaux entrants
sans prédécesseur (Attal, Philippe) et le coût d'un fit historique préalable.
La seconde ne tient plus : les fits 2022 existent déjà pour calibrer
`bank_w_ou`, donc les positions sont **gratuites** (`notebooks/04i_positions_2022.py`).

Implémenté (`sample_ordered_slots(target_pos=...)`, `BLOC_ANALOGUE_2022`,
`slot_prior_positions`). Le prior agit sur les **écarts**, pas sur les positions :
c'est la seule façon d'informer la localisation sans casser l'ordre strict
garanti par construction (§2.2). `gap_log_scale` est laissé inchangé — le prior
informe le CENTRE, il n'est pas plus serré qu'avant. Plusieurs groupes 2027
partageant un bloc 2022 (LO et LFI sont tous deux `gauche_radicale`) reçoivent
le même prior, à charge pour les données de les séparer. Vérifié : sur une
cible `[0,05 … 0,92]`, la médiane a priori retombe sur `[0,03 … 0,96]`, ordre
strict respecté.

**Non activé** (`use_position_prior=False` par défaut). Une fois le centrage de
§12.4 corrigé, la seule **contrainte d'ordre** suffit : R-hat 1,002, ESS 1970,
0 divergence. On ne paie pas une hypothèse dont on n'a pas besoin — et
l'objection de fond de §2 reste valable (Attal et Philippe n'ont pas de
prédécesseur direct, les ancrer sur « le centre de 2022 » n'est pas neutre).
Le mécanisme est gardé, testé, prêt à servir si un roster futur — plus de
groupes, moins de sondages — redevenait mal identifié.

**Circularité à ne pas commettre** : ce prior ne doit jamais servir à un fit
SUR 2022 (le backtest de §11.5), ce qui reviendrait à donner au modèle la
réponse qu'on lui demande de prédire. `fit_geometry` (04g) ne le passe pas ;
pour 2022 il faudrait des priors faits à la main, indépendants des données.

### 12.6 Densité électorale non uniforme — REDONDANTE, question close

La spec d'origine prévoyait une densité électorale « en cloche » à la place de
`W` uniforme, restée non implémentée et listée en §10 comme travail à faire.
**Elle ne doit pas être faite** : ce paramètre n'est pas identifiable
séparément des positions.

*Argument exact.* Soit `f` la densité électorale et `F` sa CDF. Alors

$$\int g(v)\,f(v)\,dv = \int g(F^{-1}(u))\,du$$

Un électorat non uniforme sur l'axe `v` est donc **exactement** équivalent à un
électorat uniforme sur l'axe `u = F(v)`, les positions étant transportées en
`F(μ)`. Intuitivement (formulation utilisateur) : si l'électorat est de droite,
le PS se retrouve vers 0,3, Attal/Philippe vers 0,4 et LR vers 0,5 — le modèle
dit la même chose avec d'autres coordonnées. L'axe étant **latent**, ces
coordonnées n'ont aucun sens absolu à préserver.

Seule la FORME du noyau n'est pas absorbée (une gaussienne en `v` n'est pas une
gaussienne en `u` si `F` n'est pas affine). Cet effet résiduel est-il
exploitable ? Mesuré : on génère `pi` sous une densité Beta pour 10 champs
différents, puis on ajuste positions/σ/w sous `W` uniforme.

| densité génératrice | résidu max sur `pi` |
|---|---|
| Beta(2,3) — électorat de gauche | 0,034 pt |
| Beta(3,2) — électorat de droite | 0,023 pt |
| Beta(4,4) — en cloche centrée | 0,018 pt |

Soit **30 à 50 fois moins que le bruit d'échantillonnage d'un seul sondage**
(~1 pt). Ajouter une densité libre reviendrait donc à ajouter un paramètre
mou, invisible aux données, sur un modèle dont §12.4 vient de montrer qu'il
souffre déjà des directions mal identifiées. `W` reste uniforme.

**Corollaire sur le noyau.** La même logique répond à la question « faut-il une
Beta plutôt qu'une gaussienne pour le rayon d'action ». Non : `A = w − d²/2σ²`
est une perte quadratique d'utilité (vote spatial classique), pas une densité,
et Beta y couple localisation et forme via (α, β) — sa variance est bornée par
μ(1−μ), donc elle IMPOSERAIT qu'un candidat extrême soit étroit au lieu de le
laisser estimer. Le vrai défaut visé est l'asymétrie de bord (avec μ=0,021 et
σ=0,136, l'essentiel du noyau d'Arthaud tombe hors de [0,1]) ; l'extension
minimale et interprétable serait un σ ASYMÉTRIQUE (σ_gauche, σ_droite), qui
garde la perte quadratique. Non fait, non prioritaire.

### 12.7 Hypothèses d'un même sondage : erreurs corrélées, pas nœuds indépendants

Remarque utilisateur, et c'est là que se trouve l'information de géométrie.
Une hypothèse ISOLÉE s'explique par `w` seul (la force de chaque candidat) ;
c'est le RETRAIT d'un candidat qui révèle où vont ses électeurs, donc qui
identifie `mu` et `sigma`. Or les `h` hypothèses d'un sondage sont posées aux
**mêmes répondants** — jusqu'à 11 pour un seul sondage du roster 2027.

`build_poll_arrays` les traitait comme des nœuds indépendants à échantillon
déflaté (`Np = n/h`). Ce procédé est correct pour le NIVEAU (l'information
totale vaut `h × 1/(h·v) = 1/v`, soit un seul sondage — l'incertitude ne
diminue pas en ajoutant des hypothèses, ce qui est la propriété voulue), mais
il attribue à la DIFFÉRENCE entre deux hypothèses une variance `2·h·v` alors
que les mêmes répondants la rendent bien plus précise. **Il amortit le niveau
correctement et détruit le signal différentiel** — celui qui identifie la
géométrie.

Modèle retenu (`weighted_loglik_blocked`) :
`Y_{i,p} = pi_{i,p} + u_{i,notice} + d_{i,p} + eps_{i,p}`, soit une gaussienne
équicorrélée par (sondage, candidat), `Sigma = a·I + b·11ᵀ` avec
`a = (1-rho)·v + delta²` et `b = rho·v`, sur l'échantillon PLEIN. `kappa` étant
constant par sondage (même date), la pondération de récence s'applique au bloc.

- `rho` : corrélation entre hypothèses. **Estimée à 0,630** [0,499 ; 0,777].
- `delta` (`sigma_model`) : **écart de MODÈLE**, indépendant entre hypothèses
  puisque chaque hypothèse est un champ différent. **Estimé à 0,34 pt.**

`delta` s'est révélé indispensable. Sans lui, retirer la déflation multiplie
l'information par 7,6 et exige du modèle qu'il reproduise un sondage à la
précision d'échantillonnage — ce qu'un spatial à 13 candidats ne peut pas
faire ; il se contorsionne et le fit se dégrade (R-hat 1,277, ESS 11). Avec
`delta`, `rho` redevient identifiable et la convergence revient.

| variante | R-hat | ESS | temps |
|---|---|---|---|
| indépendante (d'origine) | 1,002 | 1970 | 275 s |
| bloquée, `rho` estimé, sans `delta` | 1,277 | 11 | 1214 s |
| bloquée, `rho` figé, sans `delta` | 1,009 | 450 | 314 s |
| **bloquée + `delta`, `rho` estimé** | 1,006 | 431 | 301 s |

Gain sur la redistribution (45 paires 2027) : 0,843 -> 0,816 pt de MAE, soit
-3 %. Modeste mais dans le sens attendu, et cohérent avec la théorie.

**Réserve** : l'ESS chute d'un facteur ~4,5. Suffisant, mais les tirages de
géométrie qui alimentent `w_dynamics` sont plus autocorrélés. À surveiller.
**À vérifier aussi** : `delta` (0,34 pt) et la variance d'excès de §6.6
(0,317 pt) sont du même ordre — ils pourraient se recouvrir partiellement.

### 12.8 Quadrature — la grille uniforme était une source d'erreur

`pi_i = ∫ softmax(A(v))_i f(v) dv` est une intégrale ; la grille en est la
quadrature. Elle était uniforme (`linspace` + poids `1/B`), ce qui converge en
**O(1/B)** seulement — l'intégrande ne s'annule pas aux bords, donc les
extrémités dominent l'erreur. Erreur maximale sur `pi`, géométrie réellement
fittée :

| B | uniforme | Gauss-Legendre |
|---|---|---|
| 25 | 1,204 pt | **0,0015 pt** |
| 50 | 0,599 pt | 0,0015 pt |
| 1000 | 0,028 pt | 0,0015 pt |

**0,6 pt à B=50** — plus que l'écart de modèle estimé (0,32 pt) et une fraction
notable du bruit d'échantillonnage. Gauss-Legendre atteint 0,0015 pt dès B=25,
soit 400× mieux qu'une grille uniforme à B=1000, pour moitié moins de coût.
Adopté à B=50 (marge gratuite). **Ce n'était donc pas la finesse de la grille
qui était en cause mais la règle de quadrature** : passer à B=1000 uniforme
n'aurait rien réglé d'utile.

*Mesure honnête du gain* : le refit à géométrie identique donne exactement les
mêmes `sigma`, le même `sigma_model` (0,322 pt) et le même `rho` (0,630). La
quadrature n'expliquait donc AUCUN des symptômes observés — et `sigma_model`
est bien de l'écart de modèle, pas du bruit numérique.

### 12.9 `w` et `sigma` n'étaient pas identifiés séparément

Pour un candidat **non dominant**, la part vaut
`∫ exp(w − d²/2σ²)/D(v) dv ≈ exp(w)·σ·√(2π)·κ(μ,σ)/D(μ)` où `κ` est la part du
noyau tombant dans [0,1]. **Seule la combinaison `w + log σ` est identifiée.**

Vérifié directement — on multiplie `σ` d'un candidat par `f` en compensant
`w -= log f`, la part devrait bouger si les deux étaient estimés :

| candidat | part | ×0,5 | ×1,0 | ×2,0 |
|---|---|---|---|---|
| **Zemmour** (2,0 %) | | 1,96 | 1,96 | 1,91 |
| Mélenchon (12,9 %) | | 9,25 | 12,94 | 16,61 |

Zemmour est **parfaitement plat** : `σ` multiplié par 4 ne change pas sa part de
0,05 pt. Mélenchon, lui, est dominant, donc son `σ` est bien identifié. C'est
une non-identifiabilité structurelle, cousine de la jauge de `w` (§3.2), jamais
relevée jusqu'ici — et elle invalidait toute lecture de `sigma` chez les petits
candidats.

Conséquence de fond : la valeur de ce modèle est d'**extrapoler à des scénarios
non sondés** (§0, §8), et la redistribution y dépend de la FORME des profils,
donc de `sigma`. Un `sigma` non identifié rend cette extrapolation
prior-dépendante sans que ça se voie. Une bonne couverture sur les champs
observés ne suffit donc pas comme critère d'acceptation.

### 12.10 Deux correctifs : prior de `sigma` resserré + noyau normalisé

**(a) Prior resserré.** `sigma_slot` était `LogNormal(log 0,15 ; 0,4)` :
médiane 0,15, et `P(sigma > 0,10) = 0,85`. Or avec ~9 groupes sur [0,1], deux
groupes voisins sont distants de ~0,1 : un rayon d'action supérieur rend les
positions indiscernables. Un `sigma` large signifie que le NOYAU d'un candidat
s'étend loin — à 0,193 centré sur 0,873, celui de Zemmour couvre encore
l'extrême gauche.

> **PRÉMISSE DÉMENTIE (§12.13).** On en a conclu, à tort, que le modèle faisait
> recruter Zemmour à gauche. C'est faux : le noyau n'est pas le profil de
> recrutement, la compétition du softmax écrase tout. Mesuré, même au `sigma`
> le plus large : 0,04 % de son électorat vient de la gauche. Un candidat de niche
recrutant sur tout le spectre pour une idée transversale existe, mais un modèle
à UNE dimension ne peut pas le représenter : il ne peut l'exprimer que par un
`sigma` large, donc symétrique, ce qui est faux. Nouveau prior : médiane 0,06,
`P(sigma > 0,10) = 0,07`.

**(b) Noyau normalisé sur la grille** (suggestion utilisateur) :
`A_{i,b} = w_i + D_{i,b} − log(Σ_b W_b e^{D_{i,b}})`. Chaque profil intègre
alors à 1 quels que soient `μ` et `σ`. Deux effets :

- `w` devient **comparable entre candidats** : sans ça, un candidat proche d'un
  bord perd la part de son noyau hors de [0,1] et doit compenser par un `w`
  plus grand, que le prior `N(0 ; 1,5)` pénalise — il repoussait donc
  artificiellement les candidats loin des bords ;
- surtout, la part d'un candidat non dominant devient `∝ exp(w)` **seul**, ce
  qui **supprime la crête** de §12.9. `sigma` n'est plus contraint que par la
  forme, c'est-à-dire par la redistribution observée sur les hypothèses
  imbriquées — la bonne source.

C'est une reparamétrisation exacte (constante par candidat avant un softmax) :
famille de vraisemblance inchangée, mécanique du §11 intacte.

**Résultats** (roster 2027, vraisemblance bloquée + `delta`) :

| | avant | après |
|---|---|---|
| `sigma` max | 0,210 | **0,129** |
| `sigma` de Zemmour | 0,193 | **0,057** |
| candidats à `sigma > 0,10` | 11 / 13 | 5 / 13 |
| amplitude le long de la crête (Zemmour) | 0,05 pt | **0,58 pt** |
| MAE redistribution | 0,816 pt | 0,823 pt |
| R-hat / ESS | 1,006 / 431 | 1,007 / 387 |

**Le point décisif est la dernière ligne du milieu** : contraindre `sigma` et
normaliser le noyau ne coûte rien sur la redistribution (0,823 contre 0,816,
dans le bruit) — sur le roster 2027.

> **CORRIGÉ (§12.12).** La phrase « rendant les paramètres identifiés » était
> fausse. La normalisation casse bien la CRÊTE (corrélation `w`/`log σ` de
> −0,67 à −0,1), mais `sigma` reste **non identifié** : son postérieur suit son
> prior sur les deux élections. Et le prior resserré à 0,06, validé ici sur
> 2027 seul, provoque 69 à 275 divergences sur 2022 — régression introduite
> puis mesurée après coup.

Ordre des positions obtenu : Arthaud 0,062 < Mélenchon 0,069 < Roussel 0,148 <
Tondelier 0,180 < Hollande 0,225 < Glucksmann 0,233 < Attal 0,385 <
Philippe 0,413 < Retailleau 0,540 < Dupont-Aignan 0,651 < Bardella 0,677 <
Le Pen 0,689 < Zemmour 0,730.

**À refaire suite à ces changements** : `sigma_w2` (§11.5) et toute la
couverture ont été calibrés avant §12.8-12.10. À reprendre.

### 12.11 La vraisemblance bloquée ne survit pas à 2022 — et 2022 ne peut pas l'arbitrer

Portée en production, la vraisemblance bloquée à `rho`/`delta` ÉCHANTILLONNÉS
échoue sur les fits historiques :

| jeu | R-hat | ESS | divergences | `rho` estimé | `delta` |
|---|---|---|---|---|---|
| 2027 (78 nœuds) | 1,007 | 387 | 11 | 0,64 | 0,34 pt |
| 2022 J-120 (234 nœuds) | **2,52** | **2,4** | **103** | 0,85 | 0,55 pt |
| 2022 J-150 (195 nœuds) | **1,94** | **2,7** | **63** | 0,68 | 0,13 pt |

`rho` varie de 0,64 à 0,85 et `delta` d'un facteur 4 selon la coupure : ces
paramètres ne sont pas identifiés sur 2022. **Inacceptable en production** — un
job quotidien publierait des intervalles faux sans qu'aucun signal ne le dise.

**Cause, et elle est instructive.** Les sondages 2022 testent **10 candidats
sur 11** : deux hypothèses d'un même sondage y sont presque identiques, donc le
signal qui identifie `rho` (la façon dont les hypothèses d'un panel se
ressemblent) est quasi nul. 2027 teste 10 sur 13, avec une médiane de **5
hypothèses par sondage** (jusqu'à 11) et des champs réellement différents.

| | 2022 J-150 | 2027 |
|---|---|---|
| sondages | 51 | 15 |
| hypothèses / sondage (médiane, max) | 3 ; 10 | 5 ; 11 |
| champ testé (médiane) | 10 / 11 | 10 / 13 |
| sondages à 1 seule hypothèse | 3 / 51 | 2 / 15 |

**Conséquence méthodologique, qui dépasse ce paramètre.** Le bénéfice de la
vraisemblance bloquée est la précision du DIFFÉRENTIEL entre hypothèses ; il
n'existe que si les champs varient. Sur 2022 ils ne varient presque pas : le
bénéfice y est nul et le coût (paramètre non identifié) maximal. **2022 teste
donc ce mécanisme dans le régime où il ne peut pas servir.** Le jeu de
validation historique n'est pas représentatif du régime live sur ce point
précis — ce qui ne vaut PAS pour la dynamique de `w` (§11), indépendante du
régime, ni pour la couverture.

**Décision.** `rho` et `delta` décrivent le protocole de sondage (hypothèses
posées au même panel, capacité du modèle à reproduire un champ), pas la
campagne : ils sont **fixés** et non échantillonnés, comme la variance d'excès
l'est depuis sa propre banque. Ça retire les deux directions difficiles. Si
même fixés les fits historiques ne convergent pas, la vraisemblance
INDÉPENDANTE reste le choix de production : son gain mesuré (3 % de MAE sur la
redistribution) ne justifie aucun risque de non-convergence silencieuse en CI.

**À ne pas confondre avec le reste de §12** : la grille Gauss-Legendre (§12.8),
le noyau normalisé et le prior de `sigma` resserré (§12.10), le roster et le
centrage (§12.2/§12.4) sont indépendants de ce choix et restent acquis. C'est
notamment le noyau normalisé, et non la vraisemblance bloquée, qui rend `sigma`
identifiable.

### 12.12 `sigma` n'est pas identifié — et 2022 ne peut pas arbitrer la géométrie

**Constat 1 : la donnée ne détermine pas `sigma`.** Contraction
prior→postérieur (`1 - sd(post)/sd(prior)`) sur le roster 2027 :

| paramètre | contraction |
|---|---|
| `w` | 0,39 à 0,62 — informé |
| `sigma` | −0,96 à +0,36, la plupart ≈ 0 — **non informé** |

Confirmé par un prior délibérément large (`LogNormal(log 0,12 ; 0,8)`,
IC90 [0,032 ; 0,446]) :

| jeu | variation de champ | `sigma` postérieur médian | IC90 |
|---|---|---|---|
| 2027 | forte (10/13 testés, 5 hyp./sondage) | 0,111 | [0,042 ; 0,372] |
| 2022 | faible (10/11 testés, 3 hyp./sondage) | 0,156 | [0,046 ; 0,448] |

L'IC90 postérieur est celui du prior. **Choisir `sigma` est donc une décision de
modélisation, pas une estimation** — ce qui légitime de le fixer par jugement,
mais interdit de présenter la valeur obtenue comme un résultat.

**Constat 2 : pourquoi 2022 tire `sigma` vers le haut.** Avec un noyau
normalisé et `sigma` grand, tous les profils s'aplatissent et
`pi -> softmax(w)` : le modèle dégénère vers un modèle **non spatial**, où
chaque candidat n'a qu'une force. Or un modèle non spatial explique
parfaitement n'importe quel champ UNIQUE — seule la VARIATION de champ
distingue les deux. Les sondages 2022 testant presque toujours les mêmes 10
candidats sur 11, ils contiennent très peu d'information spatiale, et la
vraisemblance y glisse vers la limite non spatiale.

**Conséquence, et c'est la deuxième fois qu'on la rencontre** (cf. §12.11 pour
la vraisemblance bloquée) : **2022 peut valider la machinerie temporelle et la
couverture, mais ni calibrer ni valider la GÉOMÉTRIE spatiale.** Les deux
mécanismes concernés vivent du différentiel entre hypothèses, absent de ce jeu.
Ce n'est pas une coïncidence mais une propriété du jeu de validation.

**Constat 3 : le prix d'un `sigma` serré.** Sur 2022 J-150 (250 tirages,
400 warmup, noyau normalisé) :

| prior `sigma` médian | divergences | R-hat | `sigma` fitté |
|---|---|---|---|
| 0,15 | 5-6 | 1,06-1,13 | 0,173 |
| 0,12 | 5 | 1,04 | 0,156 |
| 0,10 | 14-18 | 1,12-1,16 | 0,103 |
| **0,06** | **69-275** | 1,04-2,38 | 0,069 |

Le conflit est réel : la vraisemblance 2022 tire vers 0,16, un prior à 0,06 la
combat. Sur 2027 le même prior ne pose aucun problème (11 divergences,
R-hat 1,007) — il n'y a pas de conflit là où le champ varie.

`sd_delta_sigma` (0,10 vs 0,15) ne change rien (14 vs 18 divergences) : le
soupçon d'entonnoir de ce côté était infondé.

**Piste écartée** : l'erreur de quadrature aux petits `sigma`. Mesurée nulle —
Gauss-Legendre reste exact à 0,0000 pt jusqu'à `sigma`=0,06 et 0,005 pt à 0,03
(l'intégrande étant lisse, GL converge spectralement ; le raisonnement « nœuds
par sigma » ne vaut que pour une grille uniforme).

**Métastabilité observée.** La configuration `sigma`=0,06 + normalisation donne
R-hat 1,039 à 250 tirages/400 warmup et **R-hat 2,60 avec 925 divergences** à
500/750 — même modèle, même graine. Le compte de divergences est ici un
indicateur bien plus fiable que R-hat, et c'est lui qu'il faut surveiller en
production.

### 12.13 Le noyau n'est pas le profil de recrutement — la prémisse était fausse

Le resserrement du prior de `sigma` (§12.10) reposait sur un argument
d'interprétation : un `sigma` large signifierait « attire des électeurs à
toutes les positions », donc un Zemmour à `sigma`=0,19 recruterait à l'extrême
gauche, ce qui est absurde. **Cet argument est faux**, et il fallait le
mesurer avant d'agir dessus.

`sigma` décrit le NOYAU du candidat, avant compétition. Le profil de
recrutement, lui, sort du softmax : à gauche, Mélenchon et Arthaud dominent
Zemmour de plusieurs ordres de grandeur, quelle que soit la largeur de son
noyau. D'où viennent réellement ses électeurs, par zone de l'axe :

| prior `sigma` | `sigma` Zemmour | gauche (<0,35) | centre | droite (>0,6) |
|---|---|---|---|---|
| 0,15 | 0,177 | **0,04 %** | 3,4 % | 96,6 % |
| 0,10 | 0,115 | 0,00 % | 0,6 % | 99,4 % |
| 0,06 | 0,057 | 0,00 % | 0,3 % | 99,7 % |

Robustesse au choix de `w` (4000 tirages du prior complet, au `sigma` le plus
large) : part gauche médiane **0,12 %**, p95 0,55 %, max 4,8 %. Même en forçant
`w_Zemmour = +3` et tous les autres à −3, on plafonne à 5,2 %.

**Le modèle n'a donc jamais fait recruter Zemmour à gauche**, à aucun `sigma`
testé et pour aucun `w` plausible. Le problème d'interprétation qui a motivé
§12.10 n'existait pas.

**Ce qu'il faut en retenir pour la suite.** `sigma` n'étant identifié par
aucune donnée (§12.12) et son interprétation directe étant trompeuse, il ne
doit être choisi ni sur une intuition politique ni sur une lecture littérale du
paramètre, mais sur les deux seuls critères mesurables : la **qualité
d'échantillonnage** et la **qualité de la redistribution**, qui est ce que le
modèle promet.

| prior `sigma` | divergences 2027 | divergences 2022 |
|---|---|---|
| 0,15 | 4 | 5-6 |
| **0,10** | **0** | 14-18 |
| 0,06 | 11 | 69-275 |

### 12.14 `sigma_w2` dépend de la géométrie — la banque n'est pas transférable

Mesuré deux fois dans la même session : **l'optimum de `sigma_w2` se déplace
d'un ordre de grandeur quand la géométrie change**, à protocole de backtest
identique.

| configuration de géométrie | `sigma_w2` optimal | couverture à 0,05 |
|---|---|---|
| avant §12.8-12.10 | ~0,20 | — |
| noyau non normalisé, prior sigma 0,06 | ~0,05-0,08 | 50,0 / 78,4 / 86,7 |
| **noyau normalisé, prior sigma 0,15** | **< 0,05** | 66,6 / 92,0 / 96,6 |

Raison : `w_dynamics` inverse `pi(w) = Y` et propage le bruit par
`C_p = J⁺ Σ J⁺`. Le jacobien `J` dépend entièrement de la géométrie — noyau
normalisé ou non, valeur de `sigma`, règle de quadrature. À `sigma_w2` égal, la
dynamique de `w` hérite donc d'une incertitude différente.

**Conséquence opérationnelle** : `bank_w_ou.json` est valide POUR UNE GÉOMÉTRIE
DONNÉE. Tout changement de noyau, de prior sur `sigma`, de grille ou de roster
l'invalide et impose de refaire les 5 fits et le balayage. Ce n'est pas une
banque « commune » au sens de `model/core/opinion.py` — celle-là vit dans un
espace (CLR des parts) indépendant du modèle qui la consomme.

**Contrôle de non-régression utile** : la référence tempérée n'utilise PAS
`w_dynamics`. Elle est restée à 45,8 / 68,3 / 76,8 contre 45,7 / 67,7 / 75,9
avant les correctifs, alors que l'OU se décalait fortement. Si un jour la
référence tempérée bouge alors qu'on n'a touché qu'à la dynamique de `w`, c'est
qu'il existe une fuite entre les deux chemins de lecture.

**Piège de protocole** : la grille de balayage était figée à [0,05 ; 0,30].
L'optimum étant passé sous 0,05, le balayage l'a manqué en entier et a désigné
sa borne inférieure. Une grille de calibration doit être vérifiée non bornante
avant d'être lue — `S2_GRID` est désormais surchargeable.

### 12.15 Le protocole de couverture surestimait le bruit d'observation

**Défaut du PROTOCOLE, pas du modèle**, présent depuis `04d` et donc dans tous
les chiffres de couverture de cette spec, §6.4 à §6.6 comprises.

`build_poll_arrays` déflate `Np` par le nombre d'hypothèses du sondage, pour
que le FIT ne compte pas `h` fois l'information d'un même panel. `04d`/`04e`/
`04f` réutilisaient ce `Np` déflaté comme **bruit d'observation du sondage
tenu**. Or un sondage de `n` personnes posant `h` hypothèses interroge les
MÊMES `n` personnes sur chaque scénario : la variance marginale d'une hypothèse
vaut `pi(1-pi)/n`, pas `pi(1-pi)/(n/h)`. La déflation décrit la vraisemblance
JOINTE, pas la loi marginale d'une observation.

Ampleur, facteur `Np_plein / Np_déflaté` sur les nœuds de test 2022 :

| coupure | facteur médian | nœuds test |
|---|---|---|
| J-30, J-60, J-120 | 1,0 | 31, 21, 8 |
| J-90 | 2,0 | 17 |
| **J-150** | **5,0** (max 6) | 31 |

Le biais touche ~44 % de l'échantillon poolé et n'est PAS uniforme : il déforme
la comparaison entre coupures, il ne se contente pas de décaler le niveau.

**Contrôle** : J-120, dont les sondages de test n'ont qu'une hypothèse, est
rigoureusement inchangé par la correction (37,5 / 84,1 / 89,8 avant comme
après), alors que J-150 passe de 66,6 / 95,5 / 99,0 à 55,1 / 87,6 / 95,5.
L'effet apparaît exactement là où il doit apparaître, et nulle part ailleurs.

**Effet sur les conclusions** (2022, 5 coupures, n=1161) :

| lecture | avant (`Np` déflaté) | après (`Np` plein) |
|---|---|---|
| tempérée | 45,8 / 68,3 / 76,8 | **39,8 / 60,7 / 71,5** |
| OU, `sigma_w2`=0,01 | 61,2 / 88,7 / 95,3 | **58,3 / 85,8 / 94,2** |

La lecture tempérée était donc **plus** sur-confiante que ne le disait la spec
(IC90 réel 71,5 %, pas 76,8 %). Le diagnostic du §11 en sort renforcé, mais
**les chiffres de §6.4-§6.6 ne sont pas comparables aux nouveaux** : ils sont
optimistes, d'un facteur qui dépend de la coupure.

`04d`/`04e`/`04f` n'ont pas été corrigés (ce sont des prototypes historiques) ;
`04g` l'est, via `Np_full_te` dans le cache.

### 12.16 BLOCAGE OUVERT — la couverture est erratique entre coupures

Sur le protocole corrigé (§12.15), `sigma_w2` = 0,01, 2022 :

| coupure | train | n | IC90 | écart au nominal |
|---|---|---|---|---|
| J-30 | 308 | 341 | 98,2 ± 3,2 | **+5,0 sd** |
| J-60 | 276 | 231 | 91,3 ± 3,9 | +0,7 sd |
| J-90 | 250 | 187 | 83,4 ± 4,3 | **−3,0 sd** |
| J-120 | 234 | 88 | 92,0 ± 6,3 | +0,6 sd |
| J-150 | 195 | 314 | 96,2 ± 3,3 | **+3,7 sd** |
| **poolé** | | 1161 | **93,4** | |

Trois coupures sur cinq sont significativement hors nominal, **dans des
directions opposées**. Le poolé à 93,4 % est une moyenne sans signification.

**Ce n'est pas un problème de réglage.** Aucune valeur de `sigma_w2` ne peut
élargir J-90 (trop étroit) et resserrer J-30 (trop large) simultanément. Le
balayage le confirme : optimum toujours sur la borne basse, et `sd(w)`
insensible à un facteur 10 sur `sigma_w2`.

**Explications testées et ÉCARTÉES** (corrélation avec l'IC90 par coupure) :

| hypothèse | mesure |
|---|---|
| volume d'entraînement | corr. **+0,11** |
| position dans la campagne | corr. **−0,09** |
| biais de déflation du protocole | corrigé (§12.15), le motif subsiste |

Pour mémoire, les autres pistes déjà démenties sur cette même question :
invariance par translation (§12.4), erreur de quadrature (§12.8, §12.12),
conditionnement du jacobien (§12.12), largeur du prior de `sigma` (levier réel
mais de −21 % seulement, insuffisant).

**Ce qui EST établi sur la cause** : la largeur du prédictif vient de la
variation de `w` **entre tirages de géométrie** (mesuré : `sd(w)` = 0,64 entre
géométries contre 0,036-0,082 à l'intérieur d'un tirage), et non de la
dynamique OU. La lecture jointe (postérieur NUTS complet) donne `sd(pi)` =
0,21 pt là où la lecture OU en donne 1,71 : ré-inférer `w` par tirage de
géométrie **casse l'anti-corrélation `w` ↔ géométrie** que le postérieur joint
maintient pour garder `pi` stable.

**Hypothèse restante, non testée** : le défaut est architectural, dans le
découpage en deux temps (§11), et non dans un paramètre. Prochain diagnostic
proposé : calibration sur données SIMULÉES depuis le modèle lui-même — si la
chaîne fit → lecture OU → prédictif ne retrouve pas le nominal sur ses propres
données, le défaut est dans la chaîne. À noter que le test synthétique de
§11.3 (50,8 / 80,7 / 89,8) utilisait la VRAIE géométrie sans fit NUTS : il ne
couvrait donc pas le chemin par lequel l'incertitude de géométrie se propage,
qui est précisément le suspect.

**Statut : la mise en production est suspendue à cette question.** La
géométrie et la redistribution (0,470 vs 0,790 pt hors échantillon) sont
acquises et utilisables ; c'est l'incertitude AFFICHÉE qui ne l'est pas.

### 12.17 RÉSOLUTION — inférence jointe + noyau OU

Le blocage de §12.16 est levé. Le défaut n'était ni un réglage ni une
spécification du modèle spatial : c'était le **découpage en deux temps** de
§11, et un noyau de diffusion inadapté.

**Diagnostic, sur données SIMULÉES depuis le modèle** (chaîne complète, fit
NUTS inclus — ce que §11.3 ne faisait pas, d'où son verdict trompeur) :

| architecture | noyau de `w` | IC50 | IC80 | IC90 |
|---|---|---|---|---|
| deux temps, géométrie variable *(§11)* | OU | 66,0 | 95,4 | **99,0** |
| deux temps, mixte | OU | 58,8 | 88,1 | 96,4 |
| deux temps, géométrie figée | OU | 33,5 | 64,4 | 79,4 |
| joint (`spatial_pooling_model_tau`) | marche aléatoire | 60,2 | 85,3 | 93,7 |
| **joint (`spatial_pooling_model_ou`)** | **OU** | **52,4** | **82,2** | **89,5** |
| *nominal* | | *50* | *80* | *90* |

Chaque ligne isole une cause :

1. *Le découpage en deux temps ne peut pas propager l'incertitude de
   géométrie.* Faire varier la géométrie et ré-inverser `Y` à chaque tirage
   compte le bruit d'échantillonnage DEUX FOIS (le postérieur `p(mu,sigma|Y)`
   le contient déjà, `C_p = J⁺ Σ J⁺` le rajoute) ; figer la géométrie perd une
   incertitude réelle. Le vrai est encadré, et aucune répartition intermédiaire
   ne le corrige — le mode « mixte » est même incohérent (`w` inféré sous une
   géométrie, utilisé sous une autre), ce qui ajoute sa propre variance.
2. *La marche aléatoire sur-disperse.* À 7 jours d'horizon elle ajoute
   `tau·sqrt(7)` = 0,070 d'écart-type là où la dérive OU en vaut 0,023 — un
   facteur 3.

**Le modèle retenu** combine les deux propriétés qu'aucune version précédente
n'avait ensemble : inférence **jointe** de la géométrie et du chemin de `w`
(propagation exacte par construction) et noyau **OU** (variance de dérive
saturante, l'argument du §11.1). Transition exacte
`w(t_k) | w(t_{k-1}) ~ N(rho_k w(t_{k-1}), sigma_w²(1-rho_k²))` écrite sous
forme matricielle ; covariance implicite vérifiée contre
`sigma_w² exp(-|dt|/tau)` à **1,6e-9** près.

**Coût** : 271 s, contre 275 s de fit + ~50 s de lecture pour la chaîne en deux
temps. L'architecture est donc plus simple ET plus rapide.

**Ce que ça rend obsolète** : `w_dynamics.py` en entier, `bank_w_ou.json`, et
avec eux le couplage géométrie/calibration de §12.14 (l'optimum de `sigma_w2`
qui se déplaçait d'un ordre de grandeur à chaque changement de géométrie).
`sigma_w` est désormais un paramètre du modèle, estimé par NUTS, sans banque à
maintenir.

**Pourquoi `spatial_pooling_model_tau` avait été parqué à tort** (§3.bis) : son
« mode dégénéré » (mu effondrés entre 0,70 et 0,99) et sa lenteur (2066 s)
étaient des symptômes du prior de position mal centré (§12.4). Après
correction, le même modèle donne des mu de 0,023 à 0,869, 0 divergence, en
310 s. Une piste abandonnée sur un artefact d'initialisation.

**À faire** : confirmer sur données réelles (backtest 2022, 5 coupures) — le
simulé valide la CHAÎNE, pas l'adéquation aux vraies données.

### 12.20 DOMAINE DE VALIDITÉ — un modèle d'exploration d'hypothèses

Cadrage utilisateur, et il réorganise rétrospectivement une bonne partie de
§12 : **ce modèle ne sert que tant que la liste de candidats bouge**. Une fois
le champ figé, il n'apporte rien que `gp_pooling` ne fasse mieux — et il
devient MAL POSÉ.

*Pourquoi c'est un théorème et pas une préférence.* L'inversion `pi(w) = Y` est
un difféomorphisme sur l'intérieur du simplexe (§11.2) : pour **n'importe
quelle** géométrie `(mu, sigma)`, il existe un `w` qui reproduit exactement les
parts observées. Un sondage à champ unique ne contraint donc RIEN sur la
géométrie — `w` absorbe tout. L'information géométrique ne peut venir que de la
**variation de champ**. Si le champ ne varie pas, le postérieur en `(mu,sigma)`
a des directions plates, et NUTS ne peut pas les explorer.

*Mesuré sur 2022* (élection le 10 avril) :

| période | nœuds | champs distincts | nouveaux champs |
|---|---|---|---|
| 2021-04 → 2021-11 | 225 | 4 à 11 par mois | 19 |
| 2021-12 | 16 | 1 | 0 |
| 2022-01 | 25 | 2 | 1 |
| **2022-02 → 04** | **89** | **1** | **0** |

La liste se fige en décembre 2021. Et le contraste avec le régime live est net :

| | nœuds | champs distincts | champ dominant |
|---|---|---|---|
| 2022 J-30 (figé) | 308 | 21 | 36,7 % |
| **2027 (actif)** | **78** | **30** | **15,4 %** |

2027 a PLUS de champs distincts avec quatre fois moins de nœuds.

**Ce que ça explique d'un coup**, et que j'avais diagnostiqué comme trois
problèmes séparés :

- 2022 ne peut pas valider la vraisemblance bloquée (§12.11) ;
- 2022 ne peut pas calibrer `sigma` (§12.12) ;
- le modèle joint ne converge pas sur 2022 alors qu'il converge sur 2027
  (R-hat 1,59 à 2,60 sur 3 coupures / 5) et sur données simulées (1,012).

**Une seule cause** : les coupures J-30 à J-120 entraînent majoritairement sur
la période FIGÉE. Passer de J-150 à J-30 ajoute 113 nœuds pour **un seul champ
distinct de plus**. Ces échecs de convergence n'étaient pas un défaut du
modèle : c'était le symptôme correct d'un modèle appliqué hors de son domaine.
Un modèle qui convergerait franchement là-dessus devrait inquiéter — il
prétendrait estimer ce que les données ne contiennent pas.

**Protocole corrigé** : coupures J-150 / J-180 / J-210 / J-240 / J-300, toutes
dans la période active (mai à novembre 2021). J-270 écarté, aucun sondage dans
sa fenêtre de test. La dimension latente tombe de 484 à 132 sans rien forcer :
moins de sondages, mais autant de champs distincts (14 à 20 partout).

**Conséquence en production** : `spatial_pooling` est un outil de **2026**, pas
de 2027. Il se retire quand la liste se fige et `gp_pooling` prend le relais.
Ça détend aussi la contrainte de budget CI : il tourne précisément quand les
champs sont nombreux et les sondages encore peu nombreux, soit le régime le
moins coûteux.

### 12.18/12.19 — deux pistes de réduction de dimension, une seule retenue

**Regroupement temporel hebdomadaire : ÉCARTÉ.** Testé pour réduire la
dimension du chemin de `w` (2022 J-30 : 1067 -> 440). **Dégrade** la
convergence : J-150 passe de R-hat 1,021 à 1,330. Regrouper force plusieurs
sondages à se réconcilier sur une même valeur de `w`, ce qui durcit la
géométrie au lieu de l'assouplir (`sigma_w` chute de 0,78 à 0,53). Réduire la
dimension ne suffit pas — c'est QUELLES contraintes on impose qui compte.
`TIME_BIN_DAYS` laissé à 1 jour.

**Chemin temporel réservé aux gros candidats : retenu** (suggestion
utilisateur). Un candidat à 1 % a une trajectoire non identifiable — le bruit
d'un sondage (±1 pt) dépasse ce qu'on prétendrait y lire. Les candidats sous
5 % de part moyenne gardent un `w` STATIQUE. Sur 2027 : Arthaud, Roussel,
Tondelier, Dupont-Aignan et Zemmour, soit une dimension de 169 -> 69.
Contrairement au regroupement temporel, ceci retire des paramètres **non
contraints** sans forcer aucune réconciliation. Seuil de 5 % non calibré ; à
terme le critère devrait porter sur la VARIABILITÉ observée de la part, pas sur
son niveau — un petit candidat en dynamique mériterait un chemin.

### 12.21 Roster historique reconstruit depuis les SONDAGES, pas depuis les candidats

`notebooks/04b::load_historical_long` ne gardait que les candidats présents dans
`candidat_blocs.csv`, c'est-à-dire ceux qui se sont **réellement présentés** —
11 sur 2022. Or les sondages de 2021 testaient massivement des hypothèses qui ne
se sont jamais concrétisées : **24 candidats à ≥ 5 sondages distincts**.

| candidat testé, jamais parti | sondages | part moyenne |
|---|---|---|
| Bertrand | 59 | **15,0 %** |
| Wauquiez | | 10,8 % |
| Barnier | 9 | 9,2 % |
| Ciotti | | 5,3 % |

Les écarter creusait un **trou de normalisation de 10 à 20 %** des intentions —
et concentré sur la droite, c'est-à-dire précisément la zone que le modèle doit
apprendre à découper. C'est le même défaut qu'en §12.1, mais du côté du roster
plutôt que du côté de la renormalisation : le trou est ramené à ~1 %.

Ce n'est pas un correctif cosmétique de backtest. Ce que ce modèle prétend
faire, c'est **redistribuer entre hypothèses** (§12.20) : lui cacher les
hypothèses effectivement testées, c'est le valider sur autre chose que sa
promesse. La règle est désormais la même qu'en live (`build_roster`) — tout
candidat réellement testé compte, sans savoir de l'avenir.

Deux conséquences de mise en œuvre :

- **Nom canonique.** Bertrand, Barnier, Wauquiez n'existent pas dans la table
  des blocs : le nom brut du sondage sert de clé, `_resolve_candidate` ne
  s'appliquant qu'aux candidats réels.
- **`BLOCS_HYPOTHETIQUES` prime sur la table.** Elle sert aussi à replacer sur
  l'axe des candidats que `candidat_blocs.csv` classe `divers` — Lassalle,
  169 sondages à 1,3 %, n'appartenait à aucun bloc ordonné et retombait donc
  dans le trou qu'on ferme. Un candidat éligible sans bloc lève désormais une
  `ValueError` au lieu d'être silencieusement écarté.

**Affectations à confirmer.** Le placement gauche/centre/droite de Bertrand,
Barnier, Wauquiez, Ciotti, Juvin, Montebourg, Taubira ne fait pas débat. Les
quatre suivantes sont discutables et personne ne les a arbitrées :
**Asselineau** et **Poisson** (rangés `droite_radicale` et `droite` — souverainisme
et conservatisme religieux ne sont pas la même chose que le RN), **Thouy**
(`ecologistes` : le parti animaliste recrute transversalement, ce que l'axe
unique ne peut pas représenter, cf. notice §7), **Lassalle** (`centre` par
défaut). Ces quatre pèsent peu, mais ils sont placés par convention, pas par
mesure.

### 12.22 Troncature de Karhunen-Loève du chemin de `w` — la dimension tombe, la profondeur d'arbre NON

**Le constat de départ.** Le temps de fit vaut exactement
`(tirages + warmup) × pas_de_leapfrog × chaînes`, vérifié. Or NUTS faisait
**exactement 255 pas par itération** (profondeur 8 pleine, jamais le plafond de
10) : la trajectoire est intégrale, à chaque itération, sans jamais faire de
demi-tour. Améliorer le conditionnement vaut donc plus que toute optimisation de
code.

**L'hypothèse.** `tau_ou ≈ 262 j` avec des sondages espacés de 2-3 jours donne
`rho ≈ 0,99` : sur une fenêtre de campagne, l'OU est presque une constante.
Mesuré, la première composante de `exp(-|t-t'|/tau)` porte **76 à 89 %** de la
variance. Échantillonner les `M` valeurs du chemin par candidat créait donc des
dizaines de directions que la vraisemblance ne contraint pas — supposées
responsables des 255 pas.

**La mise en œuvre** (`ou_kl_basis`) diagonalise cette covariance **une fois,
hors échantillonnage** (`tau_ou` est fixé, §11.4) et ne garde que les
composantes portant 99 % de la variance. `as_of` est dans la grille, donc
l'extrapolation sort de la même base au lieu d'être ajoutée après coup.

**Résultat sur 2027** (78 nœuds, 13 candidats, 4 chaînes, 400+600) :

| | avant | après |
|---|---|---|
| dimension du chemin | 169 | 91 |
| dimension latente totale | | 138 |
| **pas de leapfrog (médiane)** | **255** | **255** |
| R-hat max | 1,032 | 1,022 |
| ESS min | | 245 |
| divergences | | 0 |

**Et sur 2022 J-150**, la coupure la plus difficile du protocole corrigé
(§12.20), avec le roster reconstruit de §12.21 — 24 candidats, 195 nœuds,
`M = 44`, `K = 13` :

| | |
|---|---|
| **pas de leapfrog** | **médiane 511, max 1023 — profondeur 10 SATURÉE** |
| R-hat max | **1,387** |
| ESS min | **10,2** |
| divergences | 0 |
| temps | 1444 s |
| `sigma_w` | 0,788 |

**L'hypothèse est fausse.** Sur 2027 la dimension tombe de 46 % et la profondeur
d'arbre ne bouge pas d'un pas — 93,5 % des itérations restent à 255. Sur 2022
J-150 elle est carrément pire, et l'échantillonneur touche le plafond de
profondeur. Les directions plates n'étaient donc pas la cause : NUTS traverse un
espace mal *conditionné*, pas seulement grand. C'est cohérent avec la théorie
(une direction de prior pur est parfaitement conditionnée : elle coûte de la
mémoire, pas de la profondeur), et ça réoriente le travail vers la géométrie du
postérieur — cf. §12.23.

> **Ne pas lire ce 1,387 comme une régression du KL.** §12.18 relevait
> R-hat 1,021 à J-150, mais sur l'ANCIEN roster de 11 candidats. §12.21 en met
> 24. Les deux chiffres ne portent pas sur le même problème et la comparaison
> n'a pas de sens ; ce qui est établi ici, c'est seulement que la troncature ne
> rachète pas la convergence à J-150, et que la profondeur y sature.

La troncature est **conservée** : elle ne coûte rien, elle réduit la mémoire, et
elle est la condition technique de §12.23 (on ne peut séparer niveau et dérive
qu'en manipulant explicitement les composantes). Mais elle ne doit pas être
créditée d'un gain de vitesse.

**D'où vient alors le gain de temps** (377 s → 160 s) : entièrement du coût par
gradient, puisque le nombre de pas est inchangé. C'est `B = 50 → 25` (§12.8) qui
le produit — le tenseur `P×N×B` domine le gradient, et Gauss-Legendre donne
0,00000 pt d'erreur sur `pi` dès `B = 20`. La troncature KL n'y contribue pas.

**`chain_method` (`model/core/inference.py`) — et ce que ça invalide.** Le
paramètre est devenu explicite (défaut `"sequential"` inchangé, pour ne pas
toucher aux autres modèles) ; `04k` passe `"vectorized"`. Deux mesures :

- **Aucun gain de temps** : 377 s contre 378 s sur 2027. La docstring qui
  annonce `"vectorized"` « nettement plus rapide sur CPU » est démentie ; seul
  le pic mémoire change, et vers le haut.
- **Les verdicts de convergence changent** : sur 2027, mêmes données et même
  graine, **R-hat 2,417 en `sequential` contre 1,032 en `vectorized`**.

Le second point est le sérieux : une partie des échecs de convergence attribués
au modèle plus haut dans §12 ont été établis en `sequential` et sont
peut-être des artefacts d'échantillonneur. Le tableau « 3 échecs sur 5 » de
§12.20 est à rejuger dans ce cadre avant d'en tirer quoi que ce soit — la
lecture « le modèle échoue hors de son domaine » reste plausible, mais elle
n'est plus étayée par ces chiffres-là.

### 12.24 Audit du traitement des données — deux défauts, dont un qui renverse §12.20

Cadrage utilisateur : *l'essentiel des problèmes de 2027 s'est réglé en
choisissant soigneusement quels candidats modéliser et lesquels écarter.* Audit
complet de la chaîne de données (2027 live + les 5 coupures 2022), sans aucun
fit.

**Ce qui est correct** et n'a pas besoin d'être retouché : la sous-composition
(renormalisation de `Y` + déflation de `N_p`) est la loi exacte et elle est
appliquée ; la déflation par `n_hyp` est en place, `Np_full` conservé pour la
vraisemblance bloquée ; les nœuds à moins de 2 candidats sont jetés (3 par
coupure) ; 2027 est propre — aucun candidat du roster non testé, aucun `Y = 0`,
un seul nœud sous 0,5 %.

**Défaut 1 — le roster de backtest connaissait l'avenir.** `MIN_NOTICES` était
appliqué sur les 400 jours entiers, puis les sondages seulement étaient tranchés
à la coupure. Conséquence : des candidats entraient dans le roster avec **zéro
nœud**, parce qu'ils ne sont testés qu'APRÈS la coupure — Taubira et Thouy à
J-150, cinq candidats à J-240 et J-300. Ce sont des paramètres de prior pur, et
c'est une fuite. Corrigé par `load_historical_long(election, as_of=...)`, qui
compte les sondages à la coupure comme le fait `build_roster` en live :

| coupure | roster avant | roster as-of | retirés |
|---|---|---|---|
| J-150 | 24 | **21** | Juvin, Taubira, Thouy |
| J-180 | 24 | 20 | + Ciotti |
| J-210 | 24 | 19 | + Philippot |
| J-240 | 24 | 16 | + Barnier, Zemmour, Montebourg |
| J-300 | 24 | **15** | + Wauquiez |

**Le trou de normalisation, mesuré pour de bon.** Il a fallu trois tentatives et
les deux premières étaient fausses — à noter, c'est un piège qui se represente :

1. mesurée sur le DataFrame déjà filtré par le roster → 1,000 tautologique ;
2. mesurée sur le fichier brut sans rejouer `_resolve_candidate` → 0,000,
   l'appariement échouait silencieusement (le brut porte les noms de sondage,
   le roster les noms canoniques).

La bonne mesure rejoue la résolution PUIS compare. Masse du bulletin conservée :

| coupure | ancien roster (11) | roster as-of |
|---|---|---|
| J-150 | méd **0,865**, min 0,750 | méd **1,000**, min 0,900, p05 0,980 |
| J-300 | méd 0,846, min 0,750 | méd 1,000, min 0,900, p05 0,920 |

Le trou de 13 à 25 % annoncé en §12.21 est confirmé et refermé. Le résidu (min
0,900) vient de `MIN_NOTICES = 5`, qui laisse dehors Baroin (1 sondage, 10 %),
Retailleau (3, 6,7 %) et Juvin (4, 4,0 %) — arbitrage assumé : un candidat à 1
sondage ne serait pas identifiable, et la renormalisation traite le trou
exactement.

**Défaut 2 — le « champ figé » de §12.20 est un ARTEFACT du roster tronqué, et
c'est le point important.** §12.20 concluait que 2022 sort du domaine de
validité du modèle parce que la liste se fige en décembre 2021, et en tirait que
les échecs de convergence n'étaient pas un défaut mais « le symptôme correct
d'un modèle appliqué hors de son domaine ». Cette conclusion reposait sur un
comptage de champs distincts fait avec l'ancien roster de 11 candidats. Or
retirer Bertrand, Barnier, Wauquiez et Ciotti **fait collapser sur le même
masque des hypothèses réellement différentes**. Recompté :

| coupure | \| ancien roster : nœuds / champs / dominant | \| roster complet : nœuds / champs / dominant |
|---|---|---|
| J-30 | 308 / **21** / 36,7 % | 308 / **112** / **5,8 %** |
| J-120 | 234 / 20 / 32,1 % | 234 / 99 / 5,1 % |
| J-150 | 195 / 20 / 29,2 % | 195 / **92** / 5,6 % |
| J-210 | 86 / 17 / 41,9 % | 86 / 51 / 12,8 % |
| J-300 | 36 / 14 / 47,2 % | 36 / 23 / 19,4 % |

La première colonne reproduit exactement les chiffres de §12.20 (21 champs,
36,7 % à J-30) — c'est bien la même mesure, sur le mauvais roster. Avec le bon,
**2022 est PLUS riche en champs que 2027** (30 champs sur 78 nœuds, dominant
15,4 %) : 92 champs sur 195 nœuds à J-150, champ dominant 5,6 %.

Conséquences, à prendre au sérieux :

- **L'explication des échecs de convergence de 2022 tombe.** Le théorème de
  §12.20 (sans variation de champ, la géométrie n'est pas identifiée, `w`
  absorbe tout) reste vrai ; c'est sa prémisse empirique qui était fausse. 2022
  J-150 est largement dans le domaine du modèle, et il faut donc chercher la
  cause ailleurs — la thèse rassurante « un modèle qui convergerait là-dessus
  devrait inquiéter » ne tient plus.
- **La conclusion de production est fragilisée, pas annulée.** « `spatial_pooling`
  se retire quand la liste se fige » reste un énoncé raisonnable, mais il n'est
  plus étayé par 2022 : sur les données réelles correctement traitées, la liste
  ne se fige pas au sens du modèle.
- §12.11 et §12.12 (« 2022 ne peut pas arbitrer ») sont à rejuger pour la même
  raison.

**Défaut mineur, non corrigé.** 2022 porte 15 nœuds où un candidat TESTÉ est à
`Y = 0` exactement, et ~170 couples (nœud, candidat) sous 0,5 %. C'est la limite
de vraisemblance gaussienne déjà listée en notice §7, mais elle mord sur 2022
bien plus que sur 2027 (0 et 1 respectivement) : à 0,3 % de part, l'écart-type
d'échantillonnage vaut ~0,004 et la gaussienne met une masse non négligeable
sous zéro. À traiter le jour où la vraisemblance passera en Beta/Dirichlet.
