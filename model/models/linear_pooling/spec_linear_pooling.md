# Spec — Lissage linéaire par demi-vie (modèle « linear-pooling »)

Statut : **implémenté** (`model.py`), mais deux points restent **[OUVERT]** —
pas des erreurs, des réglages à valider empiriquement avant de les considérer
définitifs : la demi-vie `H` (§3) et le biais de renormalisation (§6). Tout le
reste est **[TRANCHÉ]** (session de conception du 2026-08-10).

## 1. Objectif

Une alternative simple et transparente à `bayesian-nowcast` (SSM à état
latent, `docs/spec_ssm_nowcast.md`) : pas de marche aléatoire, pas
d'inférence NUTS, pas de house effects — juste une moyenne pondérée directe
des sondages par candidat, où le poids décroît avec l'ancienneté. Sert de
comparaison lisible (« qu'est-ce que les sondages bruts disent, sans
modélisation de l'opinion sous-jacente ? ») et de garde-fou : un écart marqué
avec `bayesian-nowcast` signale soit un effet de la marche aléatoire/du
prior, soit un problème dans l'un des deux modèles.

## 2. Espace des candidats

Ce modèle **n'introduit pas** de notion de configuration/scénario exacte
$c \subset \mathcal K$ (contrairement à une première version de cette spec) :
il réutilise directement `SLOTS`/`aggregate_to_slots`
(`model/core/live_dataset.py`), le même vocabulaire que `bayesian-nowcast`.
Le choix « Philippe pas Attal », « Le Pen pas Bardella », « Ruffin inclus »
est donc déjà tranché ailleurs, partagé entre les deux modèles, et modifiable
en un seul endroit (`SLOTS`) au fil de la campagne — la décroissance demi-vie
(§3) fait que changer d'hypothèse en cours de route se répercute vite, sans
discontinuité gérée à la main.

Raison du changement par rapport à la configuration exacte : aucun sondage
2027 ne teste les 12 slots simultanément (7 à 11 sur 12, cf.
`docs/spec_ssm_nowcast.md`, note d'implémentation) — indexer par $c$ exact
aurait fragmenté $\mathcal P$ en dizaines de sous-ensembles quasi disjoints,
chacun avec très peu de sondages. `aggregate_to_slots` résout déjà ce
problème (fusion des hypothèses mutuellement exclusives, cf. sa docstring) ;
pas de raison de le résoudre une seconde fois ici.

## 3. Pondération demi-vie

Pour chaque observation $(p,h)$ (un couple sondage × hypothèse, sortie
d'`aggregate_to_slots`) testant le slot $s$, à la date cible $T$ (`as_of`) :

$$W_{p,h}(T) = n_{p,h} \cdot 0.5^{\frac{T - t_p}{H}}$$

où $n_{p,h}$ est l'échantillon **déjà déflaté** par le nombre d'hypothèses du
sondage (`echantillon / n_hypotheses`, même convention que
`NowcastData.from_dataframe` de `bayesian-nowcast` — plusieurs hypothèses du
même sondage partagent le même terrain, pas des mesures indépendantes).

Estimateur ponctuel et indice de confiance, **par slot** (pas par
configuration) :

$$\Omega_s(T) = \sum_{(p,h)\in\mathcal P_s} W_{p,h}(T), \qquad
\hat Y_s(T) = \frac{1}{\Omega_s(T)}\sum_{(p,h)\in\mathcal P_s} W_{p,h}(T)\, Y_{s,p,h}$$

**[OUVERT]** `H` (jours) : fixé à 14 dans `LinearPooling.half_life_days`
comme point de départ raisonnable (poids divisé par 2 toutes les deux
semaines), **pas calibré**. `model/backtest/backtest_loo.py` permettrait un
balayage empirique (minimiser l'erreur LOO 2017/2022) comme cela a été fait
pour `tau`/`historical_prior_n` sur `bayesian-nowcast` — non fait à ce stade.

## 4. Incertitude au jour `as_of` — mélange pondéré, pas variance de la moyenne

**Rejeté** : la variance classique d'une moyenne pondérée,
$\mathrm{Var}[\hat Y_s(T)] = \sum_p W_p^2 \sigma_p^2 / \Omega_s^2$, tend vers 0
quand $\mathcal P_s$ grossit — même si les sondages se contredisent. Mauvais
pour un `ic90` affiché publiquement (cf. exemple chiffré en §4.3).

**Retenu** : un **mélange pondéré** (linear pooling — d'où le nom du modèle).
Pour chaque tirage Monte Carlo $j = 1,\dots,S$, indépendamment par slot $s$ :

1. Tirer $(p,h) \sim \mathrm{Categorical}\big(W_{p,h}(T)/\Omega_s(T)\big)_{(p,h)\in\mathcal P_s}$
   — le sondage choisi, avec probabilité proportionnelle à son poids demi-vie.
2. Tirer $\theta_{s,j} \sim \mathrm{Beta}(\alpha,\beta)$ avec
   $\alpha = n_{p,h}\cdot Y_{s,p,h}$, $\beta = n_{p,h}\cdot(1-Y_{s,p,h})$
   ($Y$ en fraction $[0,1]$) — le bruit d'échantillonnage du sondage
   **choisi**, pas dilué par le poids demi-vie qui n'a servi qu'à la
   sélection. Beta plutôt que Normal : reste dans $[0,1]$ pour les petits
   candidats (Poutou, Arthaud), variance $\approx Y(1-Y)/n$ cohérente avec
   `sampling_sigma` (`model/models/bayesian_nowcast/calibration.py`) —
   approximation `/n` plutôt que `/(n+1)` (Beta exacte) gardée pour rester
   cohérente avec cette convention existante, écart négligeable aux $n$ en jeu.

### 4.1 Preuve — la moyenne du mélange retombe sur l'estimateur ponctuel

$$E[\theta_{s}] = \sum_{(p,h)} \frac{W_{p,h}(T)}{\Omega_s(T)}\cdot E[\mathrm{Beta}(\alpha,\beta)]
= \sum_{(p,h)} \frac{W_{p,h}(T)}{\Omega_s(T)}\cdot Y_{s,p,h} = \hat Y_s(T)$$

(la moyenne d'une $\mathrm{Beta}(\alpha,\beta)$ construite ainsi est
exactement $\alpha/(\alpha+\beta) = Y_{s,p,h}$). Le mélange est donc un
raffinement de l'estimateur ponctuel du §3, pas une quantité différente.

### 4.2 Variance du mélange — décomposition (loi de la variance totale)

$$\mathrm{Var}[\theta_s] = \underbrace{\sum_{(p,h)} \frac{W_{p,h}}{\Omega_s}\cdot \frac{Y_{p,h}(1-Y_{p,h})}{n_{p,h}}}_{\text{bruit d'échantillonnage moyen}}
\;+\; \underbrace{\sum_{(p,h)} \frac{W_{p,h}}{\Omega_s}\cdot \big(Y_{p,h} - \hat Y_s(T)\big)^2}_{\text{désaccord entre sondages}}$$

Le second terme est la raison d'être du mélange : il ne s'annule PAS avec
plus de sondages tant qu'ils continuent de se contredire — contrairement à
$\mathrm{Var}[\hat Y_s(T)]$ (variance de l'estimateur, qui elle décroît).

### 4.3 Exemple chiffré — slot RN, $H=14$j, 3 sondages

| Sondage | âge (j) | $n$ | $Y$ |
|---|---|---|---|
| A | 10 | 1000 | 34 % |
| B | 25 | 800  | 31 % |
| C | 5  | 1200 | 35 % |

Poids : $W_A = 1000\cdot0.5^{10/14}=609{,}8$ ; $W_B=800\cdot0.5^{25/14}=231{,}9$ ;
$W_C=1200\cdot0.5^{5/14}=937{,}1$ ; $\Omega=1778{,}8$.

Probabilités du mélange : $p_A=0{,}343$, $p_B=0{,}130$, $p_C=0{,}527$.

Estimateur ponctuel : $\hat Y = 0{,}343\times34 + 0{,}130\times31 + 0{,}527\times35 = 34{,}14\%$
— tiré vers C (le plus récent ET le plus gros), légèrement retenu par B.

- Bruit d'échantillonnage moyen (terme 1) : $\approx 0{,}000212$ (fraction²)
- Désaccord entre sondages (terme 2) : $\approx 0{,}000168$ (fraction²) —
  porté presque entièrement par B, l'outlier à 31 %
- **Variance totale du mélange : $0{,}00038$ → écart-type $\approx 1{,}95$ pt**

Comparaison avec la variance classique de la moyenne pondérée (rejetée,
§4) : $\mathrm{Var}[\hat Y] = \sum W_p^2\sigma_p^2/\Omega^2 \approx 0{,}0000835$
→ écart-type $\approx 0{,}91$ pt, **environ deux fois plus étroit** — parce
qu'elle mesure la précision de la MOYENNE, pas la dispersion plausible de la
vraie valeur compte tenu du désaccord réel des 3 sondages. C'est cet écart
concret qui motive le choix du mélange plutôt que la formule classique.

### 4.4 Sondage multi-hypothèses : où utiliser l'échantillon déflaté

`aggregate_to_slots` déflate l'échantillon par `n_hypotheses` (un sondage à 6
variantes teste 6 fois « la même » liste de candidats avec un mot près, pas 6
terrains indépendants). Cette déflation sert **UNIQUEMENT à la probabilité de
sélection** $w_{p,h} = n_{p,h}\cdot 0.5^{\text{âge}/H}$ (§3) — pas au bruit
Beta du §4 : une fois un sondage/hypothèse choisi par le mélange, le bruit
d'échantillonnage doit refléter le terrain RÉEL (ex. 1503 répondants), pas un
sous-échantillon fictif de 250 (1503/6).

Trouvé en session (2026-08-10, question utilisateur sur la largeur du RN) :
une première version utilisait par erreur l'échantillon déflaté aussi pour le
bruit — sur le cas réel RN au 10/08/2026 (91 % du poids du mélange concentré
sur les sondages de juillet, 34–37 %, cf. §4.3 pour la mécanique), ça gonflait
l'écart-type de **1,88 à 2,96 points** (IC90 [32,0 %, 38,3 %] →
[30,0 %, 39,7 %]) — une pénalité DOUBLE du même sondage (une fois à la
sélection, une fois au bruit). Différence avec `bayesian-nowcast` : ce
dernier fusionne TOUTES les hypothèses d'un sondage comme observations
SIMULTANÉES du même nœud SSM (§`NowcastData.from_dataframe`), où déflater
évite de compter le même terrain plusieurs fois dans la fusion ; ici, le
mélange n'en tire qu'UNE SEULE par tirage — la correction a déjà eu lieu à la
sélection, la répéter au bruit revient à corriger deux fois le même biais.

## 5. Absence de house effects

Ce modèle ne débiaise PAS par institut (pas de `biais[institut,bloc]`,
contrairement à `bayesian-nowcast`) : pooling brut des sondages, poids
demi-vie + échantillon uniquement. Cohérent avec l'esprit « le plus simple
possible, sans modélisation de l'opinion » — un institut structurellement
optimiste/pessimiste n'est pas corrigé, il pèse comme n'importe quel autre
sondage de même taille et de même fraîcheur.

## 6. Renormalisation sur le simplexe

Chaque tirage $\theta_{s,j}$ est indépendant par slot (pas de corrélation
inter-slots au sein d'un même sondage — même simplification que le mode
`full_covariance=False` de `bayesian-nowcast`, cf.
`docs/spec_ssm_encodage_partiel.md`). La somme $\sum_s \theta_{s,j}$ n'est
donc pas exactement 1 (ni même $\sum_s \hat Y_s(T)$ au niveau des points,
pour les mêmes raisons qu'en §5B de la spec initiale : arrondis, catégorie
« Autres »). On renormalise **par tirage** :

$$\pi_{s,j} = \frac{\theta_{s,j}}{\sum_{s'} \theta_{s',j}}$$

nécessaire pour respecter le contrat `Nowcast.draws` (une ligne = une
composition valide), consommé tel quel par `forecast_from_draws`.

**[OUVERT]** Cette renormalisation est un ratio de variables aléatoires :
$E[\pi_s] \neq E[\theta_s]/E[\sum\theta]$ en général (biais de second ordre,
analogue à l'effet Jensen déjà documenté pour `softmax` dans
`historical_prior_cov`, `model/models/bayesian_nowcast/nowcast.py`).

Vérifié empiriquement (cas réel du 10/08/2026, session de conception) : la
somme des $\hat Y_s(T)$ bruts sur les 12 slots vaut 106,2 % (couverture
partielle différente par slot, pas d'anomalie), et le ratio
`nowcast_mean / point_estimate` est quasi constant (~0,943 ≈ 1/1,062) sur les
12 slots — la renormalisation se comporte comme une simple mise à l'échelle
proportionnelle, pas comme une distorsion Jensen mesurable à cette échelle
d'incertitude. Rien ne garantit que ça reste vrai à un stade de campagne où
la couverture partielle des slots serait plus hétérogène (ex. un candidat
retiré en cours de route, cf. §7) — à revérifier si ce cas se présente.

## 7. Slot jamais testé (repli)

Un slot sans observation à `as_of` (ex. Ruffin en tout début de campagne)
reçoit un repli $\mathrm{Beta}(1, 99)$ (moyenne 1 %, confiance faible,
équivalent $n=100$) — même esprit que `NEW_ENTRANT_FLOOR` de
`bayesian-nowcast`, mais avec une incertitude explicite plutôt qu'une part
gelée. Constantes arbitraires (`FALLBACK_MEAN_PCT`, `FALLBACK_N`), pas
calibrées.

## 8. Saut terminal (T → scrutin) — réutilisé tel quel

Aucune spécification propre : `forecast()` appelle directement
`forecast_from_draws` (`model/core/simulate.py`) avec les tirages
renormalisés du §6, exactement comme `bayesian-nowcast`. Le saut sinh-arcsinh
(`TerminalJumpCalibration`, `docs/spec_ssm_nowcast.md` §2.2) est calibré
séparément pour ce modèle (`bank_jump.json` propre à ce dossier, `bank=None`
lors du fit — cf. §5, pas de débiaisage house-effects même dans le movement
pool historique), mais c'est la MÊME mécanique, les mêmes données
2017/2022. Pas de nouveau code, pas de nouvelle spec.

## 9. Limites assumées

- Pas de corrélation inter-slots au sein d'un même tirage (§6).
- Pas de house effects (§5) — un institut biaisé pèse comme les autres.
- `H` non calibré (§3) — point de départ raisonnable, pas une valeur validée.
- Biais de renormalisation non quantifié (§6).
- Repli d'un slot non testé arbitraire, non calibré (§7).

Ces limites sont assumées par construction (c'est le prix de la simplicité
du modèle) plutôt que des bugs — mais elles doivent rester visibles pour
quiconque compare `linear-pooling` à `bayesian-nowcast` sur le site.
