# pres2027 — Agrégateur bayésien de sondages, présidentielle française 2027

Agrégation des sondages du 1er tour (**18 avril 2027**) : parts par candidature,
intervalles de crédibilité, probabilités de qualification et de duels de 2nd tour.

Une nouvelle estimation est publiée chaque jour sur GitHub Pages.

## État d'avancement

| Phase | État |
|------|------|
| Récupération des données historiques (2017, 2022) | ✅ |
| Ingestion live des sondages 2027 (Wikipedia + parsing PDF) | ✅ |
| Modèle live + saut terminal calibré | 🟡 MVP — 1er tour et duels ; le *vainqueur* du 2nd tour n'est pas modélisé |
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

La calibration du saut terminal est versionnée
(`model/models/linear_pooling/bank_jump.json`) : le job quotidien la recharge, il
ne réapprend rien. À relancer seulement si le modèle de saut ou les données
historiques changent :

```bash
.venv/bin/python -m model.models.linear_pooling
```

## Le modèle

Un modèle est publié : **`linear-pooling`**. Chaque candidature est une moyenne
pondérée directe de ses sondages (poids = taille d'échantillon × décroissance
demi-vie de 14 jours), sans état latent. Puis on projette jusqu'au scrutin.

- **Slots de candidature** — un seul candidat par bloc là où plusieurs
  personnalités sont testées en alternative (RN, centre, LR, PS-PP).
- **Roster exact** — seules sont agrégées les hypothèses testant *exactement*
  ces candidats, ni moins ni plus. La part d'un candidat dépend du champ face
  auquel il est testé : Philippe mesuré face à Attal n'est pas Philippe mesuré
  seul. Mélanger les deux fausserait tous les candidats à la renormalisation.
  C'est coûteux — 6 sondages retenus sur 22 aujourd'hui — mais la cohérence des
  mesures prime sur leur nombre, et le volume disponible croîtra avec la campagne.
- **Tout se passe en log-ratio.** Une dérive additive sur les pourcentages sort du
  simplexe (parts négatives, renormalisations qui déforment les queues). On fait
  évoluer `α = log(π)` puis `π = softmax(α)` : les parts restent dans (0,1) et
  somment à 1.
- **La dérive est mesurée, pas supposée.** On isole les mouvements réels entre un
  sondage et le résultat, en 2017 et 2022 (59 mouvements), et on y ajuste une loi
  sinh-arcsinh — asymétrique et à queues ajustables, parce que le pool l'est.
- **Elle sature.** La dispersion observée est plate de 63 à 267 jours du scrutin
  (0,508 / 0,610 / 0,590) : l'essentiel du mouvement se joue près du vote. Une loi
  en `√horizon` prédirait 0,286 / 0,402 / 0,590 — elle est donc contredite par les
  données, et remplacée par `σ(h) = scale·√(1−e^(−h/τ))` (variance d'un processus
  d'Ornstein-Uhlenbeck, `τ ≈ 73 j`).
- **La même loi sert deux fois** : du dernier sondage jusqu'à aujourd'hui, puis
  d'aujourd'hui jusqu'au scrutin. Les variances s'additionnent exactement, donc
  ni double comptage ni segment oublié. C'est ce qui fait que l'incertitude
  s'élargit quand les sondages vieillissent.
- **Aucune dérive systématique par famille politique.** Les mouvements étant en
  log-ratio centré, leur moyenne vaut exactement 0 : un terme de dérive par bloc
  ne mesurerait rien, il répartirait ce zéro sur 3 à 18 observations issues de
  2 campagnes. Il valait −0,50 sur la droite — Fillon puis Pécresse — et imposait
  à tout candidat LR de perdre la moitié de sa part quels que soient ses sondages.
  On échantillonne la *distribution* des dérives de campagne, on ne rejoue pas le
  passé : la loi est la même pour tous les candidats.

Sortie (`site/data/<modèle>/<date>.json`) : parts et IC 90 %, P(qualifié top 2),
P(arrive 1er), duels de 2nd tour probables.

## Les intervalles tiennent-ils leur promesse ?

Un IC 90 % qui ne contient le résultat que 60 % du temps est un mensonge chiffré,
et rien dans les sorties du modèle ne le signalerait. On le vérifie en rejouant
le modèle à l'identique sur 2017 et 2022, à plusieurs horizons :

```bash
.venv/bin/python -m model.backtest.coverage
```

**Couverture mesurée : 0,87** pour un nominal de 0,90, sur 145 observations
(candidat × horizon × élection) — l'écart n'est pas significatif. Plus faible sur
2022 (0,83) que sur 2017 (0,92) : les ratés sont l'effondrement de Pécresse et la
poussée tardive de Mélenchon, manqués de 0,5 à 1,5 pt.

Pas de biais directionnel détectable : la position moyenne du résultat réel dans
la distribution prédite est de 0,52 (0,50 attendu), et de 0,54 pour Le Pen. En
revanche le bloc `droite` est surestimé (0,29) et `divers` sous-estimé (0,68) —
c'est le prix assumé du refus d'une dérive par famille politique.

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
model/core/     base (contrat + registre), terminal_jump (loi de dérive partagée),
                bank, inference, simulate, live_dataset, utils, registered
site/           Front statique + site/data/ (snapshots datés, index, manifeste)
```

Le nowcast bayésien SSM, le modèle spatial, leurs backtests et les notebooks
d'exploration vivent sur la branche `dev`.

## Limites

- **Deux élections seulement** (2017, 2022) pour calibrer la dérive, et 2017 ne
  couvre que les ~5 dernières semaines de campagne (voir
  [`data/historical/README.md`](data/historical/README.md)). `τ` est estimé à
  73 j mais avec une incertitude large — aucun mouvement du pool n'est mesuré à
  moins de 59 jours du scrutin.
- Le **vainqueur** du 2nd tour n'est pas modélisé : il faudrait une matrice de
  reports de voix. On s'arrête aux duels.
- Ce que le modèle produit, ce sont **des probabilités**, pas des prédictions.

## Sources & crédits

- **Sondages historiques** : [`nsppolls/nsppolls`](https://github.com/nsppolls/nsppolls) (MIT, cycle 2022) ;
  [`pollsposition/data`](https://github.com/pollsposition/data) (cycle 2017 + résultats).
- **Index live 2027** : [`nsppolls/sondages-commission-index`](https://codeberg.org/nsppolls/sondages-commission-index)
  — interagir sur **Codeberg**, pas sur le mirroir GitHub (demande des mainteneurs).
- **Résultats officiels** : Conseil constitutionnel, Ministère de l'Intérieur.
- Les notices de sondages sont publiques (Commission des sondages) ; créditer les
  instituts et leurs notices.

## Licence

[MIT](LICENSE), cohérent avec les dépôts NSPPolls.
