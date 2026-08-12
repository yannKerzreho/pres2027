# pres2027 — Agrégateur bayésien de sondages, présidentielle française 2027

Disclaimer : Le text ci-dessous a été généré par IA, une relecture à été faite.

Agrégation des sondages du 1er tour (**18 avril 2027**) : parts par candidature,
intervalles de crédibilité, probabilités de qualification et de duels de 2nd tour.

Une nouvelle estimation est publiée chaque jour sur GitHub Pages.

## État d'avancement

| Phase | État |
|------|------|
| Récupération des données historiques (2017, 2022) | ✅ |
| Ingestion live des sondages 2027 (Wikipedia + parsing PDF) | ✅ |
| Modèle live (opinion latente + projection au scrutin) | 🟡 MVP — 1er tour et duels ; le *vainqueur* du 2nd tour n'est pas modélisé |
| CI, cron quotidien, GitHub Pages | ✅ |

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Utilisation

```bash
# 1. Ingestion des sondages -> data/parsed/intentions_2027_wiki.csv (non versionné)
.venv/bin/python -m sondages.wiki

# 2. Estimation du jour -> site/data/<modèle>/AAAA-MM-JJ.json
.venv/bin/python -m model.run

# 3. (optionnel) Rejouer l'historique pour peupler la courbe
.venv/bin/python -m model.backfill --since 2026-01-01
```

Les paramètres appris sur 2017/2022 sont versionnés : le job quotidien les
recharge, il ne réapprend rien. À relancer seulement si le modèle ou les données
historiques changent :

```bash
.venv/bin/python -m model.models.gp_pooling.calibration   # diffusion de l'opinion
.venv/bin/python -m model.models.gp_pooling.terminal      # écart sondages-urne
```

## Le modèle

Un modèle est publié : **`gp-pooling`**. L'opinion y est une quantité latente
θ(t) dont chaque sondage est une mesure bruitée (échantillonnage, effet
d'institut, effet propre au sondage). θ diffuse selon un processus
d'Ornstein-Uhlenbeck ; le postérieur est en forme close, sans MCMC.

Conséquence directe : deux sondages proches **resserrent** l'estimation, au lieu
de l'élargir comme le ferait un simple mélange pondéré. Mesuré sur six instituts
ramenés au même jour, l'IC 90 % du RN passe de 10,9 à 6,4 pt à mesure qu'on les
ajoute.

Spécification complète :
[`spec_gp_pooling.md`](model/models/gp_pooling/spec_gp_pooling.md).

- **Slots de candidature** — un seul candidat par bloc là où plusieurs
  personnalités sont testées en alternative (RN, centre, LR, PS-PP).
- **Roster exact** — seules sont retenues les hypothèses testant *exactement* ces
  candidats, ni moins ni plus. La part d'un candidat dépend du champ face auquel
  il est testé : Philippe mesuré face à Attal n'est pas Philippe mesuré seul.
  C'est coûteux — 6 sondages retenus sur 22 aujourd'hui — mais la cohérence des
  mesures prime sur leur nombre, et le volume croîtra avec la campagne.
- **Tout se passe en log-ratio.** Une dérive additive sur les pourcentages sort
  du simplexe (parts négatives, renormalisations qui déforment les queues). On
  travaille sur `α = log(π)` puis `π = softmax(α)` : les parts restent dans (0,1)
  et somment à 1.
- **Trois sources de bruit séparées** dans un sondage : l'échantillonnage (connu,
  décroît en 1/√n), l'effet d'institut (partagé par tous les sondages d'une même
  maison, donc non réductible en les empilant) et un effet propre au sondage.
- **La dérive d'opinion est mesurée, pas supposée**, sur les mouvements réels de
  2017 et 2022. Elle **sature** : la dispersion observée est plate de 63 à
  267 jours du scrutin (0,508 / 0,610 / 0,590), là où une loi en `√horizon`
  prédirait 0,286 / 0,402 / 0,590. D'où le processus d'Ornstein-Uhlenbeck, qui
  revient vers un niveau moyen au lieu de diffuser sans borne.
- **Aucune dérive systématique par famille politique.** Les mouvements étant en
  log-ratio centré, leur moyenne vaut exactement 0 : un terme de dérive par bloc
  ne mesurerait rien, il répartirait ce zéro sur 3 à 18 observations issues de
  2 campagnes. Il valait −0,50 sur la droite — Fillon puis Pécresse — et imposait
  à tout candidat LR de perdre la moitié de sa part quels que soient ses
  sondages. On échantillonne la *distribution* des dérives, on ne rejoue pas le
  passé.
- **L'écart entre sondages et urne est modélisé à part.** À deux semaines du
  scrutin, l'écart `sondage → résultat` vaut déjà 0,111 quand `sondage → sondage`
  sur la même durée ne vaut que 0,018 : ce n'est pas de la dérive, c'est une
  constante (participation, indécis, vote utile). La confondre avec la diffusion
  faussait la loi d'horizon.

Sortie (`site/data/<modèle>/<date>.json`) : parts et IC 90 %, P(qualifié top 2),
P(arrive 1er), duels de 2nd tour probables.

## Les intervalles tiennent-ils leur promesse ?

Un IC 90 % qui ne contient la vérité que 60 % du temps est un mensonge chiffré,
et rien dans les sorties du modèle ne le signalerait. Deux backtests sur 2017 et
2022, qui testent des choses différentes — un modèle doit passer les deux.

```bash
.venv/bin/python -m model.backtest.predictive_coverage   # les intentions du jour
.venv/bin/python -m model.backtest.coverage              # la prévision au scrutin
```

**Le nowcast** (ce que le site affiche sous « intentions ») se teste en retirant
un sondage et en le prédisant à partir des seuls sondages antérieurs. La
couverture est ventilée par nombre de sondages disponibles, parce que c'est là
que tout se joue :

| sondages disponibles | couverture | largeur médiane |
|---|---|---|
| 1-2 | 0,870 | 6,0 pt |
| 3-4 | 0,900 | 4,2 pt |
| 5-9 | 0,906 | 3,9 pt |
| 10 et plus | 0,900 | 2,3 pt |
| global | **0,898** | |

Le nominal est tenu presque partout. À titre de comparaison, une moyenne pondérée
classique (testée puis écartée) ne couvrait que 0,663 avec un ou deux sondages,
et 0,981 au-delà de dix — c'est-à-dire surconfiante quand les données manquent,
et inutilement vague quand elles abondent.

**La prévision au scrutin** couvre 0,834 pour un nominal de 0,90. Cette
sous-couverture est un point ouvert assumé : elle vient de la projection jusqu'au
vote, pas de l'estimation du moment. Séparer explicitement la dérive d'opinion de
l'écart sondages-urne (fait, cf. la spec §5) ne l'a pas résorbée.

## Ajouter un modèle

Créer `model/models/mon_modele/`, sous-classer `ForecastModel`
([`model/core/base.py`](model/core/base.py)), décorer `@register`, l'importer dans
[`model/core/registered.py`](model/core/registered.py). Le runner, le backfill, le
sélecteur du site et les tests de contrat le prennent automatiquement.

`calibrate()` est **optionnelle** : c'est ainsi qu'on gère « certains modèles
apprennent des paramètres sur le passé, d'autres non ». `run()` reçoit les
sondages bruts au niveau candidat (institut, échantillon, méthode, hypothèse…) ;
l'agrégation en slots est un helper, pas une obligation.

**Publié ≠ enregistré** : `ForecastModel.public` décide de la présence dans le
sélecteur du site, et `python -m model.run` ne lance que les modèles publics —
une variante de diagnostic est un run d'inférence complet dont personne ne lit la
courbe. `python -m model.run --all` les inclut.

## Structure

```
sondages/       Module autonome : Wikipedia / notices PDF -> intentions structurées
pipeline/       Données de calibration historique (2017, 2022)
data/           historical/ (versionné) ; parsed/ et raw/ (régénérables, ignorés)
model/          run.py, backfill.py (entrées) ; core/ (moteur) ; models/ (un dossier par modèle)
model/core/     base (contrat + registre), movements (mouvements observés 2017/2022),
                bank, inference, simulate, live_dataset, utils, registered
model/models/   gp_pooling/ — le modèle, sa spec et ses paramètres calibrés
model/backtest/ coverage (au scrutin) et predictive_coverage (le nowcast)
site/           Front statique + site/data/ (snapshots datés, index, manifeste)
```

Le nowcast bayésien SSM, le modèle spatial et les notebooks d'exploration vivent
sur la branche `dev`.

## Limites

- **Deux élections seulement** (2017, 2022) pour tout calibrer, et 2017 ne
  couvre que les ~5 dernières semaines de campagne (voir
  [`data/historical/README.md`](data/historical/README.md)).
- **Sous-couverture au jour du scrutin** (0,834 contre 0,90 nominal), sur
  338 observations. Ce n'est pas l'estimation du moment qui est en cause — elle
  est calibrée — mais la projection jusqu'au vote. Séparer la dérive d'opinion de
  l'écart sondages-urne n'a pas suffi.
- **Peu de sondages exploitables** : le roster exact n'en retient que 6 sur 22
  aujourd'hui. Ce n'est pas un défaut du modèle mais de la couverture des
  sondages, et le volume croîtra avec la campagne.
- Le **vainqueur** du 2nd tour n'est pas modélisé : il faudrait une matrice de
  reports de voix. On s'arrête aux duels.
- Ce que le modèle produit, ce sont **des probabilités**, pas des prédictions.

## Sources & crédits

- **Sondages 2027 (source primaire)** :
  [Wikipédia — Liste de sondages sur l'élection présidentielle française de 2027](https://fr.wikipedia.org/wiki/Liste_de_sondages_sur_l%27élection_présidentielle_française_de_2027),
  sous licence [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.fr).
- **Sondages historiques** : [`nsppolls/nsppolls`](https://github.com/nsppolls/nsppolls) (MIT, cycle 2022) ;
  [`pollsposition/data`](https://github.com/pollsposition/data) (cycle 2017 + résultats).
- **Notices de sondages** : [`nsppolls/sondages-commission-index`](https://codeberg.org/nsppolls/sondages-commission-index)
  — interagir sur **Codeberg**, pas sur le mirroir GitHub (demande des mainteneurs).
- **Résultats officiels** : Conseil constitutionnel, Ministère de l'Intérieur.
- Les notices de sondages sont publiques (Commission des sondages) ; créditer les
  instituts et leurs notices.

### Remerciements

Ce projet ne serait pas possible sans le travail bénévole des **contributrices et
contributeurs de Wikipédia**, qui recensent, vérifient et tiennent à jour les
sondages au fil de la campagne, avec leurs sources et leurs notices. C'est un
travail d'archivage patient et rarement crédité : il constitue ici l'intégralité
des données d'entrée du modèle.

Merci également aux mainteneurs de **NSPPolls** et de **PollsPosition**, dont les
jeux de données historiques rendent la calibration possible, et aux **instituts
de sondage** qui publient leurs notices détaillées auprès de la Commission des
sondages — sans lesquelles la taille d'échantillon et les hypothèses testées
resteraient inaccessibles.

## Licence

[MIT](LICENSE), cohérent avec les dépôts NSPPolls.
