# pres2027 — Agrégateur bayésien de sondages, présidentielle française 2027

Agrégation bayésienne des sondages de la présidentielle française 2027
(1er tour le **18 avril 2027**, 2nd tour le **2 mai 2027**), avec calibration
des *house effects* par institut sur données historiques, et simulation Monte
Carlo du second tour.


## État d'avancement

| Phase | Description | État |
|------|-------------|------|
| **0 — Setup** | Repo, licence, récupération données historiques, backtest exploratoire | ✅ fait |
| **1 — Calibration historique** | Modèle hiérarchique biais/variance par institut × bloc, sur 2017 + 2022 | ✅ fait |
| **2 — Parsing PDF + ingestion live** | `sondages-commission-index` → intentions structurées | ✅ 4 instituts (Ifop, Odoxa, Ipsos, Harris) = 86 % des sondages d'intentions |
| **3 — MVP modèle live** | Nowcast bayésien + house effects + Monte Carlo qualification/duels | 🟡 MVP (1er tour + duels ; vainqueur 2nd tour = reports à venir) |
| 4 — CI/CD | Tests sur PR + cron quotidien + GitHub Pages | à venir |
| 5 — Contenu data science | Articles `docs/` expliquant le modèle | à venir |

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

> Note Python : le projet a été mis en place sous Python 3.9. `scipy` est épinglé
> à `1.12` car `arviz` 0.17 importe encore `scipy.signal.gaussian`, alias retiré
> dans scipy 1.13. Sous Python ≥ 3.11 ces contraintes peuvent être relâchées.

## Reproduire les phases 0 et 1

```bash
# Phase 0 — backtest exploratoire (figures dans notebooks/figures/)
.venv/bin/python notebooks/00_exploration_backtest.py

# Phase 1 — fit hiérarchique des house effects (écrit model/models/bayesian_nowcast/bank.json)
#           défaut prod : sqrt(horizon), NumPyro (HMC/JAX), 1000/1000 × 4
.venv/bin/python -m model.models.bayesian_nowcast

# Visualisation des priors appris (biais, fan chart d'IC, décomposition variance)
.venv/bin/python notebooks/01_calibration_results.py

# Validation hors-échantillon (fit 2017 -> prédiction 2022) + sensibilité priors
.venv/bin/python model/backtest/backtest_loo.py
.venv/bin/python model/backtest/prior_sensitivity.py
```

## Phase 2 — ingestion live + parsing PDF (MVP)

```bash
# Pipeline live de bout en bout : index NSPPolls -> PDF -> intentions structurées
.venv/bin/python -m sondages.build --since 2025 --limit 40
# -> data/parsed/intentions_2027.csv + rapport de couverture par institut/statut
```

## Phase 3 — modèle live (MVP)

```bash
# Tous les modèles enregistrés -> site/data/<modèle>/AAAA-MM-JJ.json (consommé par le front)
.venv/bin/python -m model.run
```

Chaîne : `data/parsed/intentions_2027.csv` (sondages) + `model/models/bayesian_nowcast/bank.json`
(house effects) → parts du 1er tour → probabilités de qualification et de duels.

- **Slots de candidature** : les alternatives mutuellement exclusives sont
  fusionnées (RN = Le Pen *ou* Bardella ; Centre = Attal/Philippe/Lecornu ;
  LR = Retailleau/Wauquiez ; PS-PP = Glucksmann/Faure/Hollande), ce qui rend tous
  les sondages comparables quelle que soit l'hypothèse testée et colle aux blocs
  calibrés.
- **Débiaisage** : chaque part sondée est corrigée du biais `biais[institut,bloc]`
  calibré en Phase 1.
- **Deux incertitudes séparées** : bruit de sondage (échantillonnage + excès,
  *réductible* en accumulant les sondages) et **dérive d'opinion d'ici au scrutin**
  (*irréductible*, commune à tous les sondages).
- **Dérive dans l'espace log-ratio (softmax), pas en pourcentages bruts.**
  Appliquer une dérive additive sur les % viole le simplexe (parts < 0, clip,
  renormalisation qui déforme les queues). On fait évoluer la dynamique sur
  `α = log(π) ∈ ℝ` puis `π = softmax(α)` : les parts restent toujours dans (0,1)
  et somment à 1 (approche « The Economist » / Gelman-Morris).
- **Dérive mesurée, pas supposée** (`model/core/drift_analysis.py`) : on débiaise les
  sondages 2017/2022 (biais + échantillonnage connus) pour isoler le mouvement
  réel de chaque candidat, mesuré **en log-ratio**. Il est **non gaussien** —
  queues épaisses (excès de kurtosis +1,6) et asymétrique (surges d'outsiders),
  jusqu'à **±11 pts** (Mélenchon 2022) ; il **sature** avec l'horizon (mean-
  reversion, pas marche aléatoire pure : Mélenchon dérive ~−11 pts à J-68 *comme*
  à J-258). La simulation **ré-échantillonne ce mouvement réel par bootstrap**.
  Effet : P(RN qualifié) passe de **97 % surconfiant à ~82 %**, et des scénarios
  où le RN rate le 2nd tour apparaissent enfin — réaliste.
- **Sortie** (`site/data/<modèle>/<date>.json`) : parts + IC 90 %, P(qualifié top 2),
  P(arrive 1er), duels de 2nd tour probables. Le **vainqueur** du 2nd tour n'est
  pas encore modélisé (nécessite une matrice de reports de voix) — on s'arrête aux
  duels, honnêtement (point de vigilance §8 : communiquer des probabilités).

## Architecture multi-modèles (back-end)

Le back-end est conçu pour comparer des approches et accueillir des contributions.

- **`ForecastModel`** ([`model/core/base.py`](model/core/base.py)) : contrat commun. Phase
  d'apprentissage `calibrate()` **optionnelle** (c'est ainsi qu'on gère « certains
  estiment des params sur le passé, d'autres non »), puis `nowcast()` + `forecast()`.
  L'orchestration `run()` fige le **schéma du snapshot** ; `validate_snapshot()`
  protège des contributions cassées.
- **`BayesianModel`** ([`model/core/bayesian_base.py`](model/core/bayesian_base.py)) : couche
  intermédiaire où l'on **écrit seulement la likelihood NumPyro** (un site `pi` sur
  le simplexe) + la préparation des données. Inférence NUTS, extraction des tirages
  et dérive sont hérités.
- **Données brutes riches** : `run()` reçoit les sondages au **niveau candidat**
  avec `institut`, `echantillon`, `methode`, `hypothese`… L'agrégation en slots est
  un helper *optionnel* — un modèle peut utiliser l'info brute (le baseline pondère
  par la taille d'échantillon). Seule l'agrégation d'affichage est fixée.
- **Sorties namespacées** : `site/data/<model_id>/AAAA-MM-JJ.json` + `index.json`,
  et `site/data/models.json` **généré depuis le registre**.

**Ajouter un modèle** : créer `model/models/mon_modele.py`, sous-classer `ForecastModel`
(ou `BayesianModel`), l'importer dans `model/core/registered.py`, `@register`. Le
runner, le backfill, le front (dropdown) et les tests de contrat le prennent
automatiquement. Deux modèles en place : `bayesian-nowcast` (avec house effects,
débiaisage par institut) et `bayesian-nowcast-no-house-effects` (même modèle,
sans le débiaisage) — comparer les deux MESURE l'apport de la calibration au
lieu de le supposer acquis.

```bash
python -m model.run                 # tous les modèles, aujourd'hui (job quotidien)
python -m model.backfill            # rejoue chaque modèle sur les dates passées
python -m model.run --model bayesian-nowcast-no-house-effects --as-of 2026-03-01
```

## Source ingestion Phase 2

Source : [`nsppolls/sondages-commission-index`](https://codeberg.org/nsppolls/sondages-commission-index)
(index des notices + URL des PDF ; **interagir sur Codeberg**, pas le mirroir GitHub).
On ne re-scrape pas la Commission — on consomme l'index et on parse les notices.

État du parsing (risque technique n°1 du projet, §8 : notices hétérogènes) :

- **Instituts outillés** : **Ifop, Odoxa, Ipsos, Harris/Toluna** — métadonnées +
  toutes les hypothèses de 1er tour, mises en page différentes gérées (Ifop : une
  hypothèse par page ; Odoxa/Ipsos : lignes `Nom 12%` avec/ sans marges d'erreur ;
  Harris : récapitulatif à N colonnes côte à côte).
- **Garde-fou qualité** : on ne conserve un groupe (hypothèse) que si les
  intentions **somment à ~100 %** — élimine récapitulatifs de duels, croisements
  et pages de rappel captés par erreur.
- **Pièges gérés** : tables de *rappel* de vote passé (résultats/redressement
  2022, législatives/européennes 2024) ignorées ; baromètres d'image / d'ambition
  sans intentions → `no_intentions` (pas d'erreur) ; croisements
  démographiques/régionaux filtrés (règles de nom propre).
- **Couverture mesurée** (sur les 106 notices présidentielles 2025+, dont **22**
  contiennent de vraies intentions de vote 2027) : les 4 instituts outillés
  couvrent **19/22 = 86 %** des sondages d'intentions. Décision assumée de s'en
  tenir aux instituts principaux tant qu'on dépasse ~80 %.
- **Volontairement non outillés** :
  - **Elabe, Cluster17** → **0** sondage d'intentions dans l'échantillon (ce sont
    des baromètres d'image/personnalités) : rien à parser, aucun coût à les laisser ;
  - **Opinion Way** (+2 sondages, ~95 % si ajouté) → résultats à 9 colonnes de
    redressement (Brut / Socio-Démo / +Présid2022 / +Légis2024) ; extraire la
    bonne colonne « Publié » est fragile pour 2 sondages → différé.
  Ces instituts restent en `unsupported_institut` (fallback, pas d'erreur).

## Structure

```
sondages/           MODULE AUTONOME parsing notices -> intentions structurées (réutilisable)
  ingest, parse_pdf, build, audit, schema (contrat), tests/ + fixtures, README, requirements
data/historical/    Sondages 2017 (pollsposition) + 2022 (nsppolls) + résultats + blocs
data/raw/ parsed/   PDF de notices (cache) + intentions live extraites (sortie de sondages/)
pipeline/           historical.py, ingest_2017.py — données de CALIBRATION historique
model/              run.py, backfill.py (entrées) ; core/ (moteur+utils) ; models/
model/core/         base (contrat ForecastModel + registre), inference (run_numpyro_mcmc),
                    bank (Bank/Param génériques), simulate, live_dataset, registered
model/models/       un dossier par modèle (ex. bayesian_nowcast/ : calibration.py + nowcast.py +
                    bank.json, ré-exportés par __init__.py) — la calibration est OPTIONNELLE et
                    vit à côté du nowcast qui la consomme, pas dans un framework séparé
model/backtest/     Validation hors-échantillon + sensibilité aux priors
notebooks/          Exploration (00) + visualisation des priors appris (01)
site/               Front statique (nowcast + évolution + points cliquables -> notices)
docs/               Articles data science (méthodo house effects)
```

## Le modèle de calibration (Phase 1)

Le house effect d'un institut a **deux composantes orthogonales**, toutes deux
estimées et exportées (voir [`docs/methodo_house_effects.md`](docs/methodo_house_effects.md)) :

1. **Biais directionnel** `biais[institut, bloc]` — décalage signé sur une famille
   politique, partagé entre élections (composante structurelle transférable).
2. **Dispersion / fiabilité** `s_inst[institut]` — niveau de bruit général de
   l'institut (inflation de variance non signée).

La **variance** est elle aussi décomposée : plancher d'échantillonnage connu
`p(100−p)/n` + excès house-effect + incertitude de **dérive future**
`τ_derive · √horizon`. La croissance temporelle suit **√horizon** (marche
aléatoire, cohérent avec la littérature — Linzer, The Economist, pollsposition ;
choix validé par le backtest, cf. plus bas), pas une forme ad hoc.

Point clé de modélisation : l'écart brut `intention(t) − résultat` est décomposé
en **biais** (house effect à J-0) + **dérive temporelle** `derive[élection, bloc]`
(mouvement réel de l'opinion pendant la campagne, spécifique à chaque scrutin,
nul le jour du vote). Sans cette séparation, le bloc droite ressortait à un biais
absurde de **+6,1 pts** en 2022 ; après séparation il tombe à un niveau modéré
(~+2,7 pt à J-0, surestimation persistante de la droite classique en 2017 *et*
2022), l'essentiel de l'effondrement Pécresse partant dans la dérive. La dérive
2017 du même bloc est de signe opposé
(Fillon), ce qui valide de la rendre spécifique à l'élection.

## Ce qui a été trouvé (backtest 2017 + 2022)

- Le **biais moyen d'un institut, tous candidats confondus, est ~nul** : les
  sur/sous-estimations se compensent entre familles politiques → justifie de
  mesurer le biais **par bloc**.
- Après décomposition, les **biais à J-0 sont modérés** (la plupart sous ±1 pt),
  sauf la droite classique (~+2,7 pt, surestimée près du vote en 2017 et 2022) :
  l'essentiel de l'« erreur » apparente vient de la dérive d'opinion et du bruit,
  pas d'un biais d'institut.
- La **dispersion de l'erreur croît en √horizon** (marche aléatoire) : mesurée,
  pas décrétée.

## Validation hors-échantillon & robustesse

- **Backtest** (`model/backtest/backtest_loo.py`) : calibré sur **2017 seul**,
  appliqué à **2022** (jamais vu). Le biais se transfère bien (résidus centrés) ;
  la couverture des IC 90 % est ≈ **0,87** (un peu sous 0,90 : 5 semaines calmes
  de 2017 ne peuvent pas anticiper la volatilité de début de campagne 2022).
- **Forme d'horizon** : le backtest départage `√h` (couverture 0,87) contre
  `log(1+h)` (0,82) → **√h retenu par défaut**, en accord avec la littérature.
- **Sensibilité aux priors** (`model/backtest/prior_sensitivity.py`) : les biais
  bougent de < 0,05 pt quand on fait varier les priors d'un facteur 5 (dominés par
  les données) ; la sous-couverture n'est **pas** un effet de priors trop serrés
  (desserrer `τ_derive` ne la change pas) mais une limite structurelle de données.
- **Qualité d'échantillonnage** (config prod, √h, NumPyro 1000/1000 × 4) :
  R-hat max **1,005**, ESS bulk min **~900**, 0 divergence.

## Limites (à assumer explicitement)

- **Deux élections disponibles** (2017 + 2022) pour la calibration, c'est peu :
  les intervalles de crédibilité sur les biais par institut restent larges — le
  *partial pooling* rend cette incertitude visible au lieu de la masquer. 2017 ne
  couvre que les ~5 dernières semaines de campagne (voir
  [`data/historical/README.md`](data/historical/README.md)) : il informe surtout
  le biais près du scrutin, peu la dérive longue.
- Les probabilités produites par le modèle sont **des probabilités**, pas des
  certitudes (point de vigilance §8 du plan).

## Sources & crédits

- **Données sondages historiques** : [`nsppolls/nsppolls`](https://github.com/nsppolls/nsppolls) (MIT) — cycle 2022 ; [`pollsposition/data`](https://github.com/pollsposition/data) — cycle 2017 + résultats officiels (Conseil constitutionnel).
- **Index live 2027** (Phase 2) : [`nsppolls/sondages-commission-index`](https://codeberg.org/nsppolls/sondages-commission-index)
  — interagir sur **Codeberg**, pas sur le mirroir GitHub (demande explicite des mainteneurs).
- **Résultats officiels** : Ministère de l'Intérieur.
- Les notices de sondages sont publiques (Commission des sondages) ; créditer les
  instituts et leurs notices.

## Licence

[MIT](LICENSE), cohérent avec les dépôts NSPPolls.
