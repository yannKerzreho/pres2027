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
noyau OU stationnaire, $\sigma_h$ fonction de l'horizon ⇒ horizons exacts).

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

### D2 (slots gp-pooling)
Non évalué : la table candidat → slot n'existe que pour 2027 ; à faire au
diagnostic 2027 (régularité, stabilité au rejeu, résidus).

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
Une seule élection exploitable (2022) : à confirmer sur 2027 (diagnostic D2 et
rejeu) avant décision.

RÉSULTAT POUR L'ORCHESTRATEUR : sur 2022 (apparié, 54 à 129 couples par
seuil), regrouper les listes en retirant les candidats < s donne +40-80 % de
sondages par courbe, −0,06 pt d'erreur à 30 j mais +0,01-0,05 pt à 7 j, et des
IC50 trop étroits (0,42-0,48) ; seul s = 5 % passe le critère fixé d'avance ;
recommandation : garder le champ exact (D0), D1(5 %) seulement après
recalibrage de son incertitude ; prévision du scrutin inchangée.
