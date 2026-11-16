# Définition des listes gp-pooling : le regroupement paie-t-il ?

Demande utilisateur (via l'orchestrateur, 2026-11-15) : « le coût en précision
d'un regroupement est-il compensé par plus de données ? ». Le modèle
gp-pooling (`GPListe`) reste **inchangé** ; seule change la définition des
sondages qu'on lui donne. Code : `model/backtest/listes_definition.py`.
`model/listes.py` et le site ne sont pas modifiés.

## Protocole (fixé avant tout résultat)

**Définitions comparées.**
- **D0, champ exact** (référence actuelle) : pour la liste $L$, les sondages
  dont l'hypothèse teste exactement $L$.
- **D1(s), champ exact sans les petits candidats**, s ∈ {1, 2, 3, 5} %. Est
  « petit » un candidat dont la part moyenne sur les sondages **antérieurs à la
  coupure** est < s (aucune fuite : la mesure du sondage lui-même n'entre pas
  dans la décision). On retire les petits de chaque hypothèse, et l'on regroupe
  toutes les hypothèses dont l'ensemble réduit vaut $L' = L \setminus$ petits ;
  leurs parts sont renormalisées sur $L'$ (`scenarios_matrix` le fait déjà).
- **D2, slots gp-pooling** (fusion d'alternatives type Bardella/Le Pen) :
  repose sur la table candidat → slot de 2027 (`map_candidate`), inexistante
  pour 2017/2022 ⇒ **évalué au diagnostic 2027 seulement**.

**Coupures.** 2022 : $h$ = 180, 150, 120, 90, 60, 30, 7 jours avant le T1
(mensuelles + T1−30 / T1−7). 2017 : mêmes $h$, mais les sondages 2017 sont
presque tous sans date (la plupart des coupures n'ont aucune donnée).
Listes : les 3 de tête à la coupure, par la règle du site
(`model.listes.classement`, champ exact). Dates décalées de
T1 2027 − T1 historique (le modèle compte les horizons depuis le T1 2027 ;
noyau OU stationnaire ⇒ horizons exacts ; **σ_h est l'effet institut** — correction du 16/11).

**Cibles et métriques**, toujours sur la **même base** pour que D0 et D1(s)
soient comparables : parts renormalisées sur $L'$ (pour D0, ses tirages sur
$L$ sont renormalisés sur $L'$).
- (a) moyenne des sondages futurs de la liste **exacte** $L$ dans
  $(c, c+7]$ et $(c, c+30]$ : erreur absolue moyenne (pt), couverture des
  IC50/80/90 (bruit binomial de la moyenne de $n$ sondages ajouté, $N = 1000$) ;
- (b) résultat du T1 (projection au scrutin du modèle : diffusion + écart de
  traduction δ, comme `forecast`) pour les listes dont tous les candidats de
  $L'$ se sont présentés, résultats renormalisés sur $L'$ ;
- nombre de sondages par courbe (le gain attendu du regroupement).

**Critère de recommandation** (fixé d'avance) : une définition D1(s) est
recommandée si son erreur (a) à 7 et 30 jours n'est pas pire que D0 de plus
de 0,1 pt ET si sa couverture IC90 reste dans [0,85 ; 0,95], avec un gain
de sondages par courbe ; sinon D0 reste.

## Résultats (2026-11-15)

**Comparaison appariée** : pour chaque seuil s, D0 et D1(s) sont évalués sur
les MÊMES (coupure, liste, candidat, cible), sur la même base $L'$ (D0
renormalisé sur $L'$). Les tableaux non appariés (D1 privé des petits
candidats, dont les petites erreurs flattent D0) sont trompeurs : écartés.

### 2022 (7 coupures, 3 listes de tête chacune)

| s | cible | n | MAE D0 | MAE D1 | IC50 D0 / D1 | IC90 D0 / D1 | sondages par courbe D0 → D1 |
|---|---|---|---|---|---|---|---|
| 1 % | 7 j | 93 | 0,649 | 0,655 | 0,52 / 0,42 | 0,87 / 0,91 | 12,5 → 17,6 |
| 1 % | 30 j | 129 | 0,834 | **0,776** | 0,52 / 0,45 | 0,84 / 0,82 | 10,5 → 15,2 |
| 2 % | 7 j | 78 | 0,740 | 0,776 | 0,55 / 0,45 | 0,90 / 0,88 | 12,5 → 19,0 |
| 2 % | 30 j | 108 | 0,952 | **0,895** | 0,54 / 0,44 | 0,83 / 0,83 | 10,5 → 17,1 |
| 3 % | 7 j | 62 | 0,843 | 0,895 | 0,56 / 0,47 | 0,94 / 0,89 | 12,4 → 19,4 |
| 3 % | 30 j | 86 | 1,109 | **1,057** | 0,56 / 0,45 | 0,84 / 0,84 | 10,4 → 18,1 |
| 5 % | 7 j | 54 | 0,895 | 0,938 | 0,57 / 0,48 | 0,96 / 0,91 | 12,3 → 19,7 |
| 5 % | 30 j | 75 | 1,190 | **1,118** | 0,59 / 0,48 | 0,87 / 0,88 | 10,4 → 18,4 |

Résultat du T1 (projection au scrutin) : identique à ≤ 0,05 pt pour tous les
s (ex. s = 5 % : 6,43 / 6,39 pt, IC90 0,79 / 0,77) — le regroupement ne
change rien à la prévision du scrutin.

### 2017
5 à 7 couples seulement (dates manquantes) : D1 un peu meilleur (MAE 0,52-0,67
contre 0,59-0,85), aucune conclusion possible.

### Diagnostic 2027 (3 listes de tête au 13/11, D0 / D1(5 %) / D2)

| liste | déf. | sondages | résidu moyen (pt) | variation jour à jour (pt/j, 90 j) | rejeu sans le dernier sondage (pt) |
|---|---|---|---|---|---|
| 39fd3710 | D0 | 26 | 0,88 | 0,18 | 0,35 |
| | D1(5 %) | 28 | 0,82 | 0,29 | 0,57 |
| | D2 | 26 | 0,88 | 0,18 | 0,35 |
| 1e7ab50b | D0 | 26 | 1,05 | 0,17 | 0,43 |
| | D1(5 %) | 28 | 0,99 | 0,24 | 0,47 |
| | D2 | 26 | 1,05 | 0,17 | 0,42 |
| 26fa8eae | D0 | 23 | 0,83 | 0,16 | 0,42 |
| | D1(5 %) | 27 | 0,80 | 0,22 | 0,58 |
| | D2 | 23 | 0,83 | 0,16 | 0,41 |

Résidus : dans l'échantillon, sur les sondages de la liste EXACTE, parts
renormalisées sur les candidats communs. Rejeu : écart moyen du nowcast quand
on retire le dernier sondage de la courbe.

- **D2 = D0 aujourd'hui** : aucune des listes de tête ne contient deux
  alternatives d'un même slot (Bardella n'est plus testé à côté de Le Pen) ;
  le regroupement par slots n'ajoute aucun sondage. Il ne servirait qu'en
  présence d'alternatives — à réévaluer si elles reviennent.
- **D1(5 %) en 2027** : seulement +2 à +4 sondages par courbe (+8 à +17 %),
  résidus −0,03 à −0,06 pt, mais une trajectoire **40 à 60 % plus agitée**
  (0,22-0,29 contre 0,16-0,18 pt/jour) et **moins stable au rejeu**
  (0,47-0,58 contre 0,35-0,43 pt) : les hypothèses ajoutées ne mesurent pas
  la même chose et font sauter la courbe.

## Lecture

- Le regroupement apporte bien **+40 à +80 % de sondages par courbe**.
- À **7 jours**, il coûte un peu de précision (+0,01 à +0,05 pt) ; à
  **30 jours**, il en gagne (−0,06 à −0,07 pt) : plus de données stabilise la
  courbe à moyen terme, au prix d'un léger retard à court terme (les
  hypothèses regroupées ne mesurent pas exactement la même chose).
- **Les IC de D1 sont trop étroits au centre** : couverture IC50 de 0,42 à
  0,48 contre 0,52 à 0,59 pour D0 — plus de sondages resserrent le postérieur
  plus que ne le justifie l'hétérogénéité qu'on a introduite. IC90 comparables.
- Au critère fixé d'avance (MAE pas pire de plus de 0,1 pt à 7 ET 30 j, IC90
  dans [0,85 ; 0,95]) : **seul D1(5 %) passe** (7 j : +0,04 pt, IC90 0,91 ;
  30 j : −0,07 pt, IC90 0,88). D1(1-3 %) échouent sur l'IC90 à 30 j
  (0,82-0,84) — comme D0 lui-même à ces seuils de base (0,83-0,84).

## Recommandation

Gain réel mais modeste, et un défaut de calibration (IC50 trop étroit) à
corriger avant tout passage. **Recommandation : garder D0 pour l'instant** ;
D1(5 %) est le seul candidat recevable, à condition de gonfler l'incertitude
des courbes regroupées (p. ex. un σ_p d'hétérogénéité propre aux hypothèses
regroupées, recalibré sur ce backtest) puis de re-mesurer la couverture IC50.
Le diagnostic 2027 le confirme : le gain de sondages y est faible (+8-17 %)
et se paie en régularité et en stabilité ; D2 n'apporte rien tant qu'aucune
alternative de slot n'est testée.

RÉSULTAT POUR L'ORCHESTRATEUR : sur 2022 (apparié, 54 à 129 couples par
seuil), regrouper les listes en retirant les candidats < s donne +40-80 % de
sondages par courbe, −0,06 pt d'erreur à 30 j mais +0,01-0,05 pt à 7 j, et des
IC50 trop étroits (0,42-0,48) ; seul s = 5 % passe le critère fixé d'avance ;
recommandation : garder le champ exact (D0), D1(5 %) seulement après
recalibrage de son incertitude ; prévision du scrutin inchangée. Sur 2027,
D1(5 %) n'ajoute que +8-17 % de sondages et rend les courbes 40-60 % plus
agitées et moins stables au rejeu ; D2 (slots) = D0 tant qu'aucune
alternative de slot n'est testée.

## Révision après la revue Opus du 2026-11-16

- « Seul D1(5 %) passe » est **dans le bruit** : 8 à 11 unités (coupure,
  liste), fenêtres de 30 j qui se chevauchent ; bootstrap par unité de la
  couverture IC90 de D1(5 %) à 30 j : [0,76 ; 0,96] — la fenêtre [0,85 ; 0,95]
  ne sépare pas les seuils. Les seuils n'ont pas les mêmes effectifs (93 contre
  54 couples) : non comparables entre eux. **Seul le déficit d'IC50 de D1 est
  net** ([−0,18 ; −0,04]).
- Les paramètres de gp-pooling ont été calés sur 2017/2022 (REML et
  `FACTEUR_DIFFUSION`) : le backtest est dans l'échantillon. À refaire avec un
  gp recalibré sans 2022.
- Diagnostic 2027 : « dernier sondage » retiré n'est pas le même pour D0 et D1,
  et la variation jour à jour n'est pas une règle de score.
- **La recommandation « garder D0 » tient** (elle ne repose pas sur le
  classement des seuils : D1 n'apporte pas de gain établi, et son IC50 est
  trop étroit) ; le « D1(5 %) recevable » est retiré.
- Calibration (complément du 16/11) : D0 et D1 partagent les MÊMES paramètres
  gp, donc le biais dans l'échantillon pèse sur les deux ; la comparaison
  RELATIVE D0/D1 en est moins affectée que la comparaison gp/spatial. Avec gp
  recalibré sans 2022 (voir spec spatial_pooling §10), gp sur-couvre : les
  chiffres absolus de couverture de ce document sont à relire à cette aune.

