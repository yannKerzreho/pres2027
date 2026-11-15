# Journal — recherche second tour (`spatial-pooling`)

Mémoire de la routine cloud horaire « pres2027 — recherche T2 ». Chaque réveil
repart à froid : TOUT l'état utile vit ici. Lire en entier avant d'agir, mettre
à jour avant de finir. Spec : `spec_second_tour.md`. Ne pas réécrire
l'historique, ajouter en bas de « Historique ».

## État (mis à jour à chaque réveil)

- Dernière revue Opus : 2026-11-14 (routine cloud, cf. « Revue 2026-11-14 »).
- OBJECTIF (utilisateur, via l'orchestrateur, 2026-11-14) : « un modèle qui est
  backé par le passé mais qui représente bien les sondages et tendances
  actuels ». 2017/2022 servent à CALIBRER et VALIDER ; le critère final est la
  crédibilité face aux sondages 2027. Le CSV `data/parsed/intentions_2027_wiki.csv`
  va être versionné sur main (mis à jour par le job quotidien) : le récupérer
  par `git merge origin/main` dès qu'il y est.
- Production : `PUBLIER_T2 = False` (forecast_model.py). Ne PAS passer à True
  sans accord explicite de l'utilisateur.
- Modèle courant : géométrie T1 + κ + valence η agrégée + abstention par valeur
  inclusive (ω). Variante de comparaison : `--variante eta` (sans abstention).
- Versions épinglées (requirements.txt) : jax 0.10.2 / numpyro 0.22.0.
- Lot en attente de publication : `claude/pin-requirements` (204fe6c), vérifié
  (voir Faits établis) ; fusion/push = orchestrateur.

## File de tâches (prendre la première non cochée)

- [x] PRIORITAIRE dès que `data/parsed/intentions_2027_wiki.csv` est sur
      origin/main (sinon passer à la suite) — DIAGNOSTIC 2027 : fit joint T1+T2
      sur 2027 (variante abst), puis (a) résidus par duel sondé et par institut
      (moyenne, écart-type, signe), (b) tendance récente : le modèle suit-il ou
      lisse-t-il les duels des dernières semaines ? (résidus contre la date),
      (c) comparaison avec ce qu'afficherait le site (duels sondés, P(élu) champ
      modal via `p_elu`) et avec la moyenne brute récente des sondages de chaque
      paire. Consigner dans la spec §8 « Diagnostic 2027 ».
      — FAIT 2026-11-14 (spec §8.1) : 5 critères passent (abst et eta) ; limite :
      Mélenchon/Le Pen lissé (r̄30j −1,3 pt, modèle−brut 1,8 pt) ; Harris −0,9 /
      YouGov +0,9 pt. Relancer `python -m model.backtest.diagnostic_2027` (≈30 min,
      1 cœur par variante) après tout changement de modèle.
- [x] CRITÈRE DE PUBLICATION (à écrire dans la spec §8, règle permanente) : aucune
      proposition de `PUBLIER_T2 = True` sans LES DEUX : couverture historique
      2017/2022 ET diagnostic 2027 satisfaisant (résidus par duel petits devant
      l'incertitude affichée, pas de biais systématique par institut ou par
      candidat, tendance récente captée). — FAIT : spec §8 « Règle permanente »,
      seuils chiffrés.
- [x] ARBITRAGES À REMONTER À L'UTILISATEUR (ne pas trancher seul) : tout choix où
      la fidélité au passé et la fidélité aux sondages actuels divergent — p. ex.
      un τ_T2 ou un biais directionnel anti-RN calibrés sur 2017/2022 qui
      écraseraient ou décaleraient les duels 2027 ; un prior qui lisse une
      tendance 2027 réelle. Les lister dans une section « Arbitrages ouverts » du
      journal avec chiffres (effet sur les duels 2027 publiés) et la ligne
      « ARBITRAGE À TRANCHER : … » en fin de réponse du réveil. — FAIT : section
      « Arbitrages ouverts » (à tenir à jour).

Backtests unitaires (un fit NUTS complet chacun, ~30-40 min sur un cœur ;
lancer 4 tâches en parallèle par réveil). Liste : `python -m model.backtest.second_round --lister`.
Pour chaque ligne, les deux variantes `--variante abst` et `--variante eta`.
Résultats : `model/backtest/results/*.json` (committés).

- [x] **PRIORITAIRE (bloque toute la file)** : une tâche complète (4 chaînes
      séquentielles × 600+400) NE TIENT PAS dans un réveil : 4 tâches en
      parallèle tuées à 55 min sans JSON (2026-11-14). Ajouter à
      `second_round.py` `--chaine k` (1 chaîne, graine k, JSON des tirages
      bruts) + `--fusionner` (concatène 4 chaînes, calcule R-hat/ESS, écrit le
      JSON final) ; 1 réveil = 4 chaînes d'UNE tâche sur 4 cœurs (~12-20 min
      estimé, à mesurer). Tester sur une coupure avant de dérouler la file.
      — FAIT (2026-11-14 17:40Z) : `--chaine k` (graine 27+k, tirages dans
      `results/chaines/`, hors git) puis `--fusionner` (aucun NUTS, R-hat/ESS
      recalculés par `fit_spatial_pooling` via `mcmc_runner` injectable) ;
      test `tests/test_second_round_chaines.py`. Essai écarté : 4 chaînes en
      `chain_method="parallel"` dans un processus = fuite mémoire JAX/CPU
      (+100 Mo/s, OOM 15 Go en 2 min ; séquentiel plat à 1,4 Go).
      Recette d'une tâche : `for k in 0 1 2 3; do python -m model.backtest.second_round
      <tâche> --variante abst --chaine $k & done; wait; python -m
      model.backtest.second_round <tâche> --variante abst --fusionner`.
- [x] **PRIORITAIRE** : géométrie T1 non identifiée dans le cadre historique
      (spec §8.2 : 100/108 notices au même champ, R-hat 35,8 à 2022 T1−30).
      Écrire dans la spec les maths du nouveau cadre : `polls_historiques`
      garde TOUTES les hypothèses T1 de la fenêtre (candidats non qualifiés
      inclus, roster par `build_roster` comme en 2027, ORDER_GROUPS_HISTORIQUES
      à compléter : Bertrand, Barnier, Taubira, Montebourg, etc. ; 2017 :
      Juppé, Sarkozy, Bayrou, Valls...) ; puis coder, puis mesurer le nombre
      de champs distincts et la convergence sur 2022 T1−30. Les backtests
      ci-dessous ne valent rien avant cela (verdicts non transposables à 2027).
      — CODÉ 2026-11-14 18:10Z (spec §8.3) : toutes les hypothèses T1, roster
      `build_roster`, putatifs classés ; 2022 T1−30 : 66 champs / 222 nœuds /
      23 candidats (contre 4 / 108 / 12). Test ajouté.
- [x] Fit de contrôle abst v2 2022 T1−30, nouveau cadre (spec §8.3) : R-hat
      15,3 (contre 35,8), ordre gauche-droite stable entre chaînes ; reste
      Lassalle bimodal (0,15 / 0,80), une chaîne à gauche comprimée, σ Zemmour.
- [x] **PRIORITAIRE** : même coupure en T1 SEUL (`second_tour=False`, à
      ajouter en option `--sans-t2` à second_round) : si le T1 seul converge,
      c'est le joint qui crée les modes (repli « coupé » de la spec §8) ; sinon
      c'est la géométrie historique (pistes : ancre de Lassalle plus serrée ou
      groupe propre, `pos_scale` du backtest, init commune des chaînes à
      l'interdit — masquerait les modes). Maths dans la spec avant tout code.
      — EN COURS (début 2026-11-14T18:50Z, session 01JZeMw2) : option `--sans-t2`
      ajoutée (V2 seulement, sortie = diagnostics T1 + μ moyens),
      `--only v2 --election 2022 --coupure T1-30 --sans-t2`, 4 chaînes.
      — FAIT 18:58Z : T1 seul R-hat 10,5 ⇒ modes = géométrie T1 (permutations
      de voisins), pas le joint. Duel cible stable entre chaînes (0,6 pt) ⇒
      critère de convergence DU BACKTEST sur la quantité prédite (spec §8.4 :
      R̂(logit q) ≤ 1,1 et écart de chaînes ≤ 1 pt ; garde de prod inchangée).
      2022 T1−30 abst : CONCLUANT, couvert, log-score = moyenne brute (2,64).
- [x] abst : v2 2022 T1-30, T1-1, T2-7 (nouveau cadre) — FAIT (spec §8.5) : cible convergée ; modèle = moyenne des sondages ±0,4 pt ; non couvert à T1−1 et T2−7 avec τ = 0,10, couvert à τ = 0,19 (analytique, dans l'échantillon). Ancien cadre :
      T1−30 committé sous `second_round_abst_v2_2022_T1-30_ancien_cadre.json` (non convergé, R-hat 35,8, témoin) ; T1−1 perdu
      (VM récupérée vers 18:00Z), non relancé : sans objet.
- [x] abst : v1 2022 Macron|Zemmour, Mélenchon, Le Pen, Pécresse — FAIT (spec §8.6) : NON CONCLUANT, écarts de chaînes 4,7-9,3 pt (géométrie T1 multimodale transmise aux duels non observés) ; MAE modèle 6,5 / géom. 9,1 / BT 12,2 (indicatif).
- [ ] **PRIORITAIRE** : lever la multimodalité de la géométrie T1 du cadre
      historique 2022 (permutations Mélenchon/LO, Zemmour/souverainistes,
      Lassalle). Pistes à chiffrer, maths dans la spec AVANT code : (a) fenêtre
      plus courte (90 j au lieu de 180 : moins de candidats putatifs, moins de
      chemin w) en comptant les champs restants ; (b) les coupures du backtest
      T1 publié (J-300…J-150, R-hat ≤ 1,06, spec_spatial_pooling §8) : quelle
      config différait ? (c) en dernier recours, ancre de Lassalle (groupe
      propre). Ne pas « régler » sur le résultat 2022 : critère = écart de
      chaînes ≤ 1 pt sur les V1 (§8.4), pas l'erreur.
      Mesures 21:30Z : (a) fenêtre à T1−1 : 180 j = 45 champs/22 cand. ;
      120 j = 17/17 ; 90 j = 14/17 ; 60 j = 5/15 — raccourcir coûte de
      l'identification, piste peu prometteuse. Hypothèse « ancres trop
      serrées en 2022 » RÉFUTÉE : écart logit min/médian 0,25/0,25 en 2027
      (17 cand.) comme en 2022 (22). Expérience EN COURS (début
      2026-11-14T21:35Z, session 01JZeMw2) : V1 Macron|Le Pen avec
      `--pos-scale 0.25` (diagnostic : les modes sont-ils des permutations
      permises par la liberté autour de l'ancre ?).
      — FAIT 21:47Z (spec §8.7) : écart 7,6 → 5,8 pt, pas suffisant. Mode
      minoritaire = Roussel/Jadot recentrés à σ large, COMPENSÉ par η
      (Macron +0,29 / Le Pen −0,30). Suite : η qui ne puisse pas absorber la
      géométrie (hiérarchique à échelle apprise, ou retrait vers la
      prédiction géométrique) — maths dans la spec d'abord, mesurer l'écart de
      chaînes V1 Macron|Le Pen (référence 7,6 pt).
      — RÉORIENTÉ 22:20Z (spec §8.8) : les modes ont des log-densités très
      inégales (10-130 nats) ⇒ pièges d'échantillonneur, pas ambiguïté ; η
      n'est pas à contraindre. Règle de sélection des chaînes (Δ = 10 nats)
      codée dans la fusion (`lp_par_chaine`, `chaines_retenues`, `concluant`),
      tous les JSON 2022 re-fusionnés : V2 concluant ×3, V1 non concluant ×4.
- [ ] Phase 2 « relance au mode dominant » (`--init-depuis`, spec §8.8) :
      V1 Macron|Le Pen — FAIT 22:35Z : puits stable (4 chaînes retenues,
      rhat_max 2,3, écart 1,4 pt) mais prédiction 64,0 % pour 55,3 observé.
- [ ] Variantes sans valence (spec §8.9, question utilisateur : « le T2 ne
      peut-il pas estimer μ, σ, ω ? κ η ω sont-ils nécessaires ? ») : `geo`
      (κ+ω, η ≡ 0) et `geo1` (ω seul, κ ≡ 1). V1 Macron|Le Pen `geo` phase 1 —
      EN COURS (relancé 2026-11-14T23:10Z, session 01JZeMw2 : le run de 22:37Z a été perdu, VM récupérée dès la fin du tour) ; puis phase 2 ;
      puis `geo1` ; puis V2 2022 et diagnostic 2027 des variantes gagnantes.
      — FAIT 23:40Z (spec §8.9) : geo 56,2 % (erreur 0,9, un seul puits sans
      phase 2) ; geo1 50,0 % (erreur 5,3) ; abst 64,0 % (8,7). κ nécessaire,
      η nuisible sur cette paire.
- [x] geo sur les 3 autres V1 2022 — FAIT 00:29Z (spec §8.9) : convergé
      (3/4 concluants) mais Zemmour 51,1 pour 66,7 et Pécresse 53,1 pour 65,6
      (repoussoirs non représentables en 1D sans valence). MAE geo 7,3, abst 6,8.
- [x] Variante `hier` (échelle de valence apprise, spec §8.10) sur les 4 V1
      2022 — FAIT 01:50Z : MAE 4,9 pt (abst 6,8, geo 7,3), couverts 3/4,
      concluants 2/4 ; s_η bimodal (0,16 / 0,35) sur Zemmour et Pécresse. Tests : 150 passés (hors réseau).
- [x] Pécresse ratée par toutes les variantes (~53 contre 65,6) : vérifier la
      cible V1 (fenêtre 30 j pendant l'effondrement) et la lecture de `w` à
      `as_of` dans `v1_paire_retiree` AVANT toute conclusion de modélisation.
      — FAIT 02:00Z (spec §8.10) : cible et lecture correctes ; cause = plus
      que 2 duels de Pécresse une fois la paire retirée ⇒ extrapolation
      géométrique vers 50/50 (limite structurelle). Pistes : abstention
      croissante avec la distance (à spécifier) ; mention produit pour les
      duels de candidats sans aucun duel.
- [ ] `hier` : traiter la bimodalité de s_η (phase 2 §8.8, ou prior
      informatif), puis V2 2022 ×3 et diagnostic 2027 en `hier`.
      Phase 2 Zemmour puis Pécresse — EN COURS (début 2026-11-15T02:02Z,
      session 01JZeMw2).
- [ ] abst : v1 2017 Macron|Le Pen, Fillon|Le Pen ; v2 2017 T1-30 (attendu : sans nœud T1), T1-1, T2-7
- [ ] eta : mêmes tâches (comparaison)
- [ ] Revue 2026-11-14, sans fit : corriger la base V2 (écart-type τ_T2 en plus
      de « largeur du modèle », rapporter les deux) ; post-fit : corrélation
      intra-notice des résidus (`duel_s` + `duels["notices"]` désormais dans le fit).
      Déjà mesuré sur 2027 (§8.1) : |corr(log κ, η_i)| ≤ 0,22, z̄² = 0,75 —
      à refaire sur les fits 2017/2022.
- [ ] Effets maison au T2 (Harris −0,9 / YouGov +0,9 pt sur 2027) : mesurer
      sur 2017/2022 avant toute modélisation ; maths dans la spec d'abord.
- [ ] Décision τ_T2 (maths dans la spec §6 avant code) : la mesure du 14/11
      donne RMS ≈ 4,3 pt ⇒ τ ≈ 0,18-0,20 logit ; centrage sur 0 = choix
      éditorial, à faire trancher par l'utilisateur (ne pas coder de biais).
- [ ] Synthèse dans spec §8 : V1 MAE modèle / géométrie seule / BT ; V2 couverture,
      log-score vs sondages ; R-hat, rhat_t2, κ, ω, η ; corrélation κ–η.
- [ ] Puis pistes (choisir selon les résultats, maths dans la spec AVANT le code) :
  - entre-deux-tours conditionné sur le résultat T1 (nœud quasi exact) ;
  - P(élu) à aléa unique (w projeté au T1 -> π(T1) -> prolongé au T2) ;
  - η hiérarchique partagé Le Pen/Bardella (bloc), V1 sur paires sans autre duel ;
  - τ_T2 en leave-one-election-out ;
  - valeur inclusive sur le CHAMP (masque) au lieu du roster (alias LR 2027) ;
  - contrôle κ = 1 fixé vs κ libre (si crête κ–η confirmée) ;
  - effet aléatoire par notice sur les duels (si corr. intra-notice > 0,3) ;
  - tableaux croisés T1×T2 des instituts (si la session données les fournit).

## Arbitrages ouverts (pour l'utilisateur — ne pas trancher seul)

1. **Largeur τ_T2 de l'aléa de scrutin** (actuel 0,10 logit ; mesure
   2017/2022 ⇒ 0,18-0,20). Backtest V2 2022 (spec §8.5) : à 0,10 le
   résultat sort de l'IC90 à T1−1 et T2−7 (z ≈ 2), à 0,19 il y rentre.
   Effet sur 2027 (estimation analytique depuis
   §8.1, abst) : P(Philippe bat Le Pen) 0,14 → ≈ 0,21 ; IC90 du duel
   ≈ ±7,6 → ±10 pt. Fidélité au passé = élargir ; ne change pas les centres.
2. **Biais directionnel des sondages de T2 (anti-RN en 2017/2022, ordre
   2-3,5 pt)** : NON codé. Le corriger décalerait tous les duels 2027 de
   ~+2,5 pt contre le RN : P(Philippe bat Le Pen) 0,14 → ≈ 0,30, P(élu)
   Le Pen nettement en baisse — le modèle ne « représenterait » plus les
   sondages actuels. Deux élections = pas d'estimation fiable.
3. **Lissage des tendances récentes** (Mélenchon/Le Pen : modèle 33,3 contre
   31,5 en brut sur 30 j) : suivre plus vite (τ_OU T2 plus court ou chemin
   propre à η) = plus de bruit ; garder = retard de ~1-2 pt sur un mouvement réel.

## Faits établis (ne pas re-mesurer)

- Géométrie T1 seule : MAE 6,3 pt sur les 11 paires 2027 ; abstention (κ, ω)
  3,2 ; η seule 2,9 ; les deux 2,4 (une paire retirée, géométrie figée).
- Codex (d7b1575) : bug option extérieure, duel compté deux fois, gate bloquant —
  corrigés (cf. spec §9).
- Données 2017 : la plupart des sondages T1 sans date (note à la session données).
- VM cloud : 4 cœurs, 15 Go ; fit rapide 75-89 s ; fit complet 4 chaînes
  séquentielles > 55 min quand 4 tournent en parallèle ; installer avec
  `python -m pip install --ignore-installed packaging -r requirements.txt`.
- Réseau de la VM cloud : `fr.wikipedia.org` BLOQUÉ (proxy 403). Pas
  d'ingestion live ni de `model.run` 2027 dans la routine ; le CSV 2027 est
  désormais versionné sur main (job quotidien) : `git merge origin/main` le
  rafraîchit. Diagnostic 2027 possible hors réseau. Le test
  `sondages/tests/test_wiki.py::test_build_dataset_live_page_matches_schema_contract`
  y échoue pour cette seule raison : ne pas le « corriger ».
- Vérif versions épinglées (2026-11-14, VM cloud) : install OK, 152 tests passés,
  1 échec réseau ci-dessus. Versions identiques au job quotidien vert du 13/11.
- VM cloud récupérée dès qu'un tour se termine (uptime 0 au réveil suivant,
  3 fois le 14/11) : un fit lancé en arrière-plan ne survit PAS à la fin du
  tour. Rester dans le tour jusqu'à la fin des fits (attente par boucles
  `until`), et committer les JSON aussitôt produits.
- Colab gratuit : session `t2bt` perdue en cours de run (VM récupérée) le
  2026-11-14 — ne plus en dépendre.

## Historique

- 2026-11-14 (session locale) : audit Codex, spec, implémentation, 2 relectures,
  abstention ajoutée sur idée utilisateur, épinglage, création de la routine.
- 2026-11-14 (routine cloud, 1er réveil) : revue Opus (11 points, intégrés
  ci-dessous) ; mesure sans fit de l'erreur des duels selon l'horizon (τ_T2
  sous-dimensionné, spec §6) ; spec §3 corrigée. 4 backtests `abst` lancés,
  tués à 55 min sans résultat ⇒ tâche prioritaire « une chaîne par cœur ».
- 2026-11-14 16:10-16:50Z (routine) : main fusionnée (CSV 2027) ; `duel_s`
  déterministe + `duels` dans le fit (sans effet sur l'inférence) ; script
  `model/backtest/diagnostic_2027.py` ; diagnostic 2027 abst+eta (§8.1),
  règle de publication (§8), arbitrages ouverts ; résultat de fumée périmé
  retiré. Tests : 144 passés (hors réseau). Mesure utile pour le découpage :
  un fit 2027 complet (4 chaînes séquentielles, 600+400, 92 duels) = 27-30 min
  avec 2 fits en parallèle sur la VM 4 cœurs.
- 2026-11-14 17:07-17:50Z (routine) : backtests découpés en processus d'une
  chaîne + fusion (mode parallèle JAX écarté : fuite mémoire) ; 1re tâche
  abst v2 2022 T1−30 en 14 min sur 4 cœurs ; R-hat 35,8 ⇒ diagnostic :
  géométrie T1 non identifiée dans le cadre historique (spec §8.2), tâche
  prioritaire ajoutée. Tests : suites spatial_pooling + T2 + chaînes vertes (35) ; suite complète non relancée (fits en cours).
- 2026-11-14 18:07-18:40Z (routine + question utilisateur sur les hypothèses) :
  VM récupérée vers 18:00Z (T1−1 ancien cadre perdu, non relancé). Cadre
  historique à toutes les hypothèses T1 (spec §8.3) : 66 champs au lieu de 4 ;
  contrôle 2022 T1−30 : R-hat 35,8 → 15,3, Lassalle bimodal. Tests : 147
  passés (hors réseau).
- 2026-11-14 18:43-19:55Z (« Continue » utilisateur) : T1 seul aussi
  multimodal ⇒ critère de convergence sur la quantité prédite (spec §8.4) ;
  V2 2022 abst complet (spec §8.5). Tests : 4 de chaînes/cadre verts.
- 2026-11-14 19:48-21:25Z : V1 2022 abst (4 paires) non concluant (§8.6).
- 2026-11-14 21:27-21:50Z (réveils 19:07/20:07/21:07 reçus en retard, traités
  en un) : fenêtre, ancres, pos_scale 0,25 (§8.7) ; cause des modes V1 =
  compensation géométrie/η.
- 2026-11-15 01:52-02:00Z (réveils 00:07/01:07 reçus en retard) :
  vérification Pécresse (limite structurelle, §8.10).
- 2026-11-14 22:35-2026-11-15 00:35Z (question utilisateur : « le T2 ne
  peut-il pas estimer μ, σ, ω ? κ η ω nécessaires ? ») : phase 2 (mode
  dominant stable, V1 Macron–Le Pen 64 %) ; variantes geo/geo1 (§8.9) :
  η libre = ce qui laissait coexister les géométries ; geo converge mais rate
  les repoussoirs ; κ nécessaire ; `hier` (§8.10) : meilleure MAE V1 (4,9).
- 2026-11-14 22:07-22:25Z (réveil) : log-densité par chaîne ⇒ modes inégaux
  (spec §8.8), sélection Δ = 10 nats, re-fusion 2022, phase 2 lancée.
  Tests : 6 chaînes/cadre verts. Tests : 148 passés (hors réseau).

## Revue Opus 2026-11-14 (routine cloud) — intégration point par point

1. Résultat `second_round_fast_v2_2022.json` périmé (pré-abstention, 1 chaîne,
   R-hat 1,2-3,1) : ACCEPTÉ, aucun chiffre n'en est tiré ; il reste committé
   (interdiction de supprimer), remplacé par les runs `abst` 4 chaînes.
2. τ_T2 trop étroit / pas fonction de l'horizon : ACCEPTÉ en partie, MESURÉ
   (spec §6) : erreur 3-6 pt à tous les horizons, pas de décroissance nette
   (T2−7/T2−1 = 0,9 en 2017, 1,7 en 2022) ⇒ pas de τ(h) prouvé, mais τ global
   trop petit. Tâche ajoutée.
3. Valeur inclusive sur le roster : ACCEPTÉ pour 2027 (alias exclusifs), sans
   effet sur les backtests 2017/2022 (roster = champ). Piste ajoutée ; note §3.1.
4. ρ avec η dans la spec, §3.2 contradictoire : ACCEPTÉ, spec corrigée.
5. Crête κ/η (κ 0,71→1,77 sur les fits 2022) : PLAUSIBLE mais les chiffres
   viennent du run périmé à 1 chaîne/ESS 3, non probants ; à mesurer sur les
   fits complets (tâche ajoutée) avant de fixer κ.
6. σ_T2 au plancher, n total vs exprimés, notices : ACCEPTÉ comme test
   post-fit (tâche ajoutée). Remarque : n surestimé de ~25 % rend la
   vraisemblance trop précise, ce qui devrait POUSSER σ_T2 vers le haut ; un
   σ_T2 au plancher suggère plutôt que les duels sont sous-dispersés autour de
   s(t) (variance d'échantillon déjà > dispersion réelle, cf. §4 : 0,9-1,4 pt).
7. Base V2 = largeur du modèle, pas ± τ_T2 : ACCEPTÉ en partie. Le choix
   « même largeur » est délibéré (comparer les centres) mais ne permet pas
   d'évaluer le critère §8 ; on rapportera les deux bases (tâche).
8. Fuites (champ final connu, hypothèse la plus proche, τ in-sample) : ACCEPTÉ
   comme limites documentées ; elles avantagent modèle ET base pareillement,
   sauf τ in-sample (à traiter en leave-one-election-out).
9. Cible BT (180 j) ≠ cible V1 (moyenne 30 j) : ACCEPTÉ, à aligner lors de la
   synthèse V1 (BT sur ≤ 30 j en plus).
10. P(élu) à deux aléas indépendants : déjà connu (spec §7), piste « aléa
    unique » dans la file. Gate sur rhat_t2 seul : correct tant que
    PUBLIER_T2 = False ; les critères V1/V2 seront à ajouter avant publication.
11. Pas de garde « sans nœud T1 » en V1 2017 : MINEUR, les paires V1 se
    fittent à T1−1 où les nœuds existent ; à vérifier sur le run 2017.
Idées A-F : A faite (point 2) ; B, D, E, F ajoutées à la file ; C (nœud
résultat T1) = piste « entre-deux-tours conditionné » déjà en file.
