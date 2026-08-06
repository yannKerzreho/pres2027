# Les house effects ont deux composantes : biais et dispersion

Un « house effect » (effet d'institut) n'est pas une seule quantité. Un institut
peut se tromper de deux façons **orthogonales** :

1. **Biais directionnel** — il penche systématiquement dans un sens sur une
   famille politique. Ex. surestimer structurellement le bloc de droite
   radicale, ou au contraire le sous-estimer à cause d'un redressement. C'est un
   **décalage de moyenne**, signé, spécifique au bloc.
2. **Dispersion / fiabilité** — il est simplement plus *bruité* que les autres,
   dans les deux sens, indépendamment de la direction. Deux instituts peuvent
   avoir un biais moyen nul et pourtant l'un être régulièrement à ±1 pt, l'autre
   à ±4 pts. C'est une **inflation de variance**, non signée.

Ces deux composantes sont indépendantes : un institut peut être biaisé mais
précis, non biaisé mais bruité, les deux, ou aucun. Les confondre (par ex. ne
retenir qu'une « note » globale) revient à jeter de l'information.

## Comment on l'écrit dans notre modèle

Pour chaque observation `i` (un sondage × un candidat réel), l'écart
`intention − résultat` (en points), avec **moyenne** et **variance** décomposées :

```
ecart_i ~ Normal( mu_i , sqrt(var_i) )

mu_i   = biais[institut_i, bloc_i]  +  derive[election_i, bloc_i] · log(1+horizon_i)
var_i  = deff · p_i(100−p_i)/n_i    +  excess_i²
excess_i = exp( log_excess0 + s_inst[institut_i] + b_h · z(horizon_i) )
```

**Moyenne :**
- `biais[institut, bloc]` → **composante 1 (biais directionnel)**, par institut ×
  bloc, mutualisé entre instituts (`biais[k,b] ~ Normal(mu_bloc[b], tau_bloc[b])`)
  et **partagé entre élections** (composante structurelle transférable).
- `derive[election, bloc]` → **nuisance** : mouvement réel de l'opinion pendant la
  campagne (Fillon 2017, Pécresse 2022), nul à J-0. Il est **hiérarchique et de
  moyenne nulle** : `derive[e,b] ~ Normal(0, tau_derive)`, et `tau_derive` (ampleur
  typique de la dérive par unité de log-horizon) est **estimé**. C'est ce qui
  permet, pour 2027, de propager l'incertitude de dérive en temps réel (§ IC 2027).

**Variance — deux morceaux :**
- `deff · p(100−p)/n` → **plancher d'échantillonnage connu** (pas estimé). On ne
  peut pas prétendre à un biais de bloc juste parce qu'un sondage s'écarte du
  résultat : cet écart peut n'être que du bruit d'échantillon. Le biais n'est
  crédible que s'il **dépasse ce plancher, se répète, et distingue l'institut**
  des autres dans la durée — ce dont le partial pooling se charge. `deff` ≥ 1 est
  le *design effect* des sondages par quotas (correction du §4.4, ici estimée).
- `excess_i` → **composante 2 (dispersion / fiabilité house-effect)**. `s_inst`
  est le niveau de bruit *général* de l'institut (mutualisé, transférable, §4) ;
  `b_h · z(horizon)` fait croître l'excès avec l'horizon (volatilité d'opinion,
  pas erreur d'échantillon).

Autrement dit, le house effect a bien **deux formes** : `biais[·,bloc]` (décalage
signé) et `s_inst[·]` (bruit). Exportées séparément dans `priors.json`.

## Intervalles de crédibilité en temps réel pour 2027

Pour un candidat 2027 à l'horizon `h`, l'incertitude de prévision empile trois
sources, toutes calibrées :

```
Var_prevision(h) = deff·p(100−p)/n    (échantillonnage du sondage courant)
                 + excess(institut,h)² (bruit house-effect, croît avec h)
                 + tau_derive²·log(1+h)²  (dérive future INCONNUE de la campagne)
```

Le dernier terme est nul le jour du vote (`log(1+0)=0`) et grandit avec `h` : les
IC sont **larges à un an, resserrés à quelques jours**, sans réglage arbitraire —
c'est `tau_derive`, appris sur 2017/2022, qui fixe l'échelle. (Voir
`calibration/priors_utils.py : forecast_drift_sigma`.)

## Comment font les autres modélisations

| Modèle | Biais (composante 1) | Dispersion (composante 2) |
|--------|----------------------|----------------------------|
| **Nate Silver / 538** | « house effect » = lean statistique par institut, *mean-reverted* vers 0 | « pollster rating » d'erreur/précision, sert à **pondérer** (∝ 1/erreur²) |
| **The Economist** (Gelman, Morris, Heidemanns 2020) | terme de biais par institut **et par mode** (tel/online), mutualisé, contraint à somme nulle | variance d'observation (taille d'échantillon) + inflation par institut/mode |
| **Linzer 2013** (Votamatic) | deltas additifs de house effect par institut, sur une marche aléatoire latente | variance surtout pilotée par la taille d'échantillon |
| **pollsposition** (A. Andorra, PyMC — référence française) | house effect (biais) par institut sur une popularité latente (GP / marche aléatoire) | bruit d'observation par sondage |
| **depuis1958.fr** | Dirichlet-multinomial ; house effects plus légers | taille d'échantillon *effective* (réduite selon méthode/quotas) |

Points communs :
- **Tout le monde sépare biais et précision.** 538 est le plus explicite : la note
  d'erreur sert à pondérer, le house effect sert à décaler. C'est exactement la
  décomposition ci-dessus.
- **Identifiabilité du biais.** Dans un modèle *live* avec une vérité latente
  (on ne connaît pas encore le résultat), le biais n'est identifiable qu'à une
  constante près : il faut une contrainte **somme-nulle** sur les instituts ou un
  institut de référence, sinon le biais commun et la vérité latente se
  confondent. → Notre calibration hors-ligne **contourne ce problème** : on cale
  l'écart contre le **résultat réel** des scrutins passés, donc le biais est
  identifié en absolu, pas seulement en relatif. C'est un vrai avantage de
  calibrer sur l'historique avant d'entrer en live.

## Pourquoi séparer biais et dérive

Sans séparation, l'écart brut `intention(t) − résultat_final` attribue au biais
tout l'effet du temps. Exemple mesuré sur 2022 : le bloc droite ressortait à
**+6,1 pts de biais** — absurde, car c'était surtout l'effondrement de Pécresse
(12 % en décembre → 4,8 % en avril), pas une erreur des instituts. En modélisant
`derive[election,bloc]·log(1+horizon)` (nul le jour du vote), le biais à J-0 du
bloc droite retombe à **~+0,7 pt** (intervalle incluant 0), et l'effondrement est
correctement rangé dans la dérive. Le biais qu'on exporte est donc l'erreur
*réelle des instituts le jour du vote*, pas un artefact temporel.

## Les autres calibrent-ils le biais sur l'historique ?

Nuance importante : **non, pas vraiment pour le biais directionnel** — et pour
une bonne raison.

- **Les modèles live (538, The Economist) estiment les house effects *dans le
  cycle courant***, comme l'écart de chaque institut à la *moyenne des sondages*
  du moment, avec une contrainte **somme-nulle** (ou un institut de référence).
  Ils ne transfèrent pas un biais partisan d'une élection à l'autre : les têtes
  d'affiche changent, un biais « pro-candidat X » de 2022 n'a pas de sens en 2027.
- **L'historique sert surtout à la variance temporelle** : de combien les
  sondages se trompent-ils à l'horizon `h` ? de combien l'opinion peut-elle
  encore bouger ? → c'est exactement nos `tau_derive` et `b_h`. Les *pollster
  ratings* de Silver (précision) relèvent aussi de ça, pas d'un biais partisan
  figé.

Notre choix : calibrer le biais sur le **résultat réel** des scrutins passés
l'identifie en **absolu** (et non à une constante près comme en live) — un vrai
atout. Le risque de transférabilité entre candidats est réel mais **acceptable
ici car les biais mesurés sont petits** : à J-0, tous les blocs sont sous ~0,7 pt
en valeur absolue, souvent compatibles avec 0. On les utilise donc comme
**priors informatifs faibles**, pas comme des vérités. Si un biais ressortait un
jour grand et douteux, on basculerait vers une contrainte somme-nulle (au moins
par mode) comme les modèles live.

## Et l'effet tél / internet ?

Il *faudrait* un biais et une variance par mode de recueil, comme The Economist.
Mais dans nos données, **2017 et 2022 sont à 100 % en ligne** (réalité du marché
français depuis ~2012) : un terme de mode serait **non identifiable** (aucun
contraste téléphone), donc on ne l'ajoute pas. Il faudrait 2007/2012, dont on n'a
pas encore les sondages, pour le calibrer. Limite documentée plutôt que paramètre
fantôme.

## Forme de la croissance temporelle : √h, pas log

De combien l'incertitude croît-elle avec l'horizon ? Deux choix ont été testés
pour le régresseur d'horizon (tous deux nuls à J-0, donc sans effet sur le biais) :

- `log(1 + h)` — forme réduite pragmatique, concave ;
- `√h` — cohérente avec une **marche aléatoire / mouvement brownien** de l'opinion
  (variance ∝ temps ⇒ écart-type ∝ √temps), qui est l'hypothèse implicite des
  modèles dynamiques de la littérature (Linzer 2013, The Economist / Gelman,
  pollsposition). Le log **n'est pas** une forme canonique de la littérature.

**Le backtest hors-échantillon tranche** (`model/backtest/backtest_loo.py`, fit
sur 2017 → prédiction 2022) : `√h` couvre mieux (couverture 90 % ≈ 0,87 vs 0,82
pour le log). On retient donc **√h par défaut** (`horizon_form="sqrt"`). Le log
reste disponible en option.

Nuance : `√h` sur-élargit aux horizons extrêmes (log-score un peu moins bon), ce
qui suggère qu'à terme une **marche aléatoire avec réversion à la moyenne**
(l'opinion est bornée) — le modèle dynamique de la Phase 6 — serait plus juste
qu'une forme fixe.

## Sensibilité aux priors

Vérifié en refaisant le backtest sous plusieurs jeux de priors
(`model/backtest/prior_sensitivity.py`) :

- **Biais robustes** : `mu_bloc[droite]` et `mu_bloc[droite_radicale]` bougent de
  < 0,05 pt quand on fait varier l'écart-type du prior de biais d'un facteur 5
  (2 → 10). Les house effects sont **dominés par les données**, pas par le prior.
- **La sous-couverture n'est pas un effet de priors trop serrés** : desserrer le
  prior de `tau_derive` (×2,5) ne change ni son posterior (≈ 0,5) ni la couverture
  (0,87). La donnée pin `tau_derive`, le prior n'est pas contraignant. La
  sous-couverture est **structurelle** (2017 ne couvre que 5 semaines calmes, ne
  peut pas enseigner la volatilité de début de campagne 2022) — on la corrige par
  plus d'élections / un modèle dynamique, pas en élargissant les priors.

## Autres extensions possibles (non retenues au MVP)

- **Dispersion par bloc** : autoriser certains instituts à être spécifiquement
  bruités sur un bloc (ex. RN). Le §4 considère la variance *générale* comme la
  partie transférable ; une variance par institut × bloc demande plus d'élections
  de recul pour ne pas surajuster. À garder mutualisée si ajoutée.
- **Réintégrer 2012 / 2007** : surtout utile pour (a) élargir la calibration de
  `tau_derive` et (b) rendre l'effet de mode identifiable.
