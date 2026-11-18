# Journal — recherche second tour (`spatial-pooling`)

Mémoire de la routine cloud horaire « pres2027 — recherche T2 ». Chaque réveil
repart à froid : TOUT l'état utile vit ici. Lire en entier avant d'agir, mettre
à jour avant de finir. Spec : `spec_second_tour.md`. Ne pas réécrire
l'historique, ajouter en bas de « Historique ».

## État (mis à jour à chaque réveil)

- Dernière revue Opus : 2026-11-18 ~11:20Z (routine, cf. « Revue Opus 2026-11-18 »).
- OBJECTIF (utilisateur, via l'orchestrateur, 2026-11-14) : « un modèle qui est
  backé par le passé mais qui représente bien les sondages et tendances
  actuels ». 2017/2022 servent à CALIBRER et VALIDER ; le critère final est la
  crédibilité face aux sondages 2027. Le CSV `data/parsed/intentions_2027_wiki.csv`
  va être versionné sur main (mis à jour par le job quotidien) : le récupérer
  par `git merge origin/main` dès qu'il y est.
- Production : `PUBLIER_T2 = False` (forecast_model.py). Ne PAS passer à True
  sans accord explicite de l'utilisateur.
- Modèle courant : géométrie T1 + κ + valence η agrégée + abstention par valeur
  inclusive (ω). Variantes de backtest : `eta`, `geo` (η ≡ 0), `geo1` (κ ≡ 1),
  `hier` (échelle de η apprise) — verdict §8.20 retiré (revue 17/11) puis
  REFAIT selon le protocole robuste §8.29 (10 tâches, CRPS + LS plafonné,
  inversion des signes) : **hier préférée** (p = 0,04 / 0,008, même signe sur
  les 3 élections ; grappes p = 1/8) ⇒ `ETA_HIER_T2 = True`, τ_paire = 0,08.
- Backtests : cadre historique à toutes les hypothèses T1 (§8.3) ; 4 processus
  d'une chaîne + fusion ; sélection des chaînes par log-densité (§8.8) ;
  phase 2 `--init-depuis` ; convergence jugée sur la quantité prédite (§8.4).
- Versions épinglées (requirements.txt) : jax 0.10.2 / numpyro 0.22.0.
- Lot en attente de publication : `claude/pin-requirements` (204fe6c), vérifié
  (voir Faits établis) ; fusion/push = orchestrateur.

## File de tâches (prendre la première non cochée)

- [ ] **NOUVELLES PISTES (utilisateur, 2026-11-17 ~19:30Z)** — dans cet ordre, maths
      dans la spec AVANT tout code, rien réglé sur les résultats qu'on valide :
  - [x] [FAIT 18/11 13:50Z, spec_spatial_pooling §10 : ancres par famille ⇒ MAE V1 2022 4,66 → 6,75 pt, multimodalité non levée ⇒ non adopté ; CHES non prioritaire] [RÉVISÉ 18/11 (revue) : le blocage vient de l'écart minimal d'ancres (2022 : 22/24 écarts au plancher), pas de la source ; À FAIRE : écart minimal seulement dans une famille, puis 4 V1 2022] [ancien : BLOQUÉ 17/11 22:55Z : CHES inaccessible (chesdata.eu hors domaines autorisés) ; repli ORDER_GROUPS sans effet (ancres ≤ 0,011 d'écart, contrainte d'écart minimal) — spec_spatial_pooling §10] (B) **Positions a priori externes pour la géométrie historique 2017/2022**
        (multimodalité §8.2-8.7) : prior centré par candidat sur une source
        indépendante (famille politique / placements d'experts type CHES ou
        équivalent sourçable hors réseau ; à défaut l'ordre ORDER_GROUPS déjà
        codé transformé en positions) + fenêtre historique 90 j si elle retire
        des putatifs sans nœud utile. Critère : écart de chaînes V1 2022
        ≤ 1 pt et MAE V1 2017/2022/2027 non dégradée (protocole §8.29).
        Lien mémoire : « matrice de reports à la main » différée — ne pas la coder ici.
  - [x] [FAIT 18/11, spec_spatial_pooling §10 : MAE +0,01, score d'intervalle +0,17 pt, couverture +0,005 ⇒ non adopté, rôles spatial/gp assumés] (C) **σ_w par candidat** (hiérarchique, au lieu d'un σ_w commun) pour
        les tendances T1 amorties (spec_spatial_pooling §10) : rejouer
        `model/backtest/suivi.py` 2022 (8 coupures) ; adopter seulement si MAE
        et couverture battent le σ_w commun. Sinon consigner « rôles
        spatial/gp assumés ».
  - [ ] (A) **Tableaux de reports T1→T2 des instituts** — BLOQUÉ DONNÉES
        (demandés à la session pipeline-donnees le 17/11) : dès leur arrivée sur
        main, (1) validation sans fit : reports prédits par le modèle (tirages
        par électorat T1) contre reports mesurés, MAE par électorat ; (2) puis
        comme observations (multinomiale par électorat, maths d'abord).
  - [ ] (D) Espace 2D (axe système/anti-système) — SEULEMENT si (A)(1) montre
        une erreur systématique par électorat que l'axe unique n'absorbe pas.
  - Toujours BLOQUÉ DONNÉES (même demande) : τ_T2 2002-2012, dates 2017.

- [x] [(a), (b), (c) FAITS 2026-11-16 ; (c) : IC90 duels +2,5-3 pt de chaque côté, P(élu) Le Pen 84,8 → 80,3 %, spec §8.22 : `TAU_T2_SCRUTIN = 0.19`, spec §6.2 ; τ_paire spec §6.3 + §8.22, `TAU_PAIRE = 0.08` (V1 2027, hier), code duel_draws/p_elu + test ; reste (c) diagnostic_2027, après le placebo] **ARBITRAGES TRANCHÉS (2026-11-15 ~21:45Z, voir section Arbitrages)** — à faire
      AVANT la relance ci-dessous : (a) spec §6 : τ_T2 = 0,19, justification
      (RMS 4,3 pt 2017/2022, couverture 5/5) ; `TAU_T2_SCRUTIN = 0.19` dans fit.py,
      tests adaptés ; (b) spec : maths de τ_paire (bruit N(0, τ_paire²) en logit
      sur les seuls duels sans aucun sondage, appliqué en prédiction, pas dans la
      vraisemblance), calibrage leave-one-pair-out sur les résultats V1 2027
      committés (0 fit), puis code + test ; (c) relancer `diagnostic_2027` et
      consigner l'effet sur P(élu) et les IC des duels 2027. Biais anti-RN et
      lissage : rien à coder (décision = statu quo), noter dans la spec.

- [x] [(1)-(5) FAITS le 16/11] **RELANCE 2026-11-15 21:30Z (session locale)** : le réveil de 21:07Z a
      conclu « rien à faire » alors que des tâches SANS dépendance restent
      ouvertes. Ordre à suivre, une par réveil, sans attendre les arbitrages :
      (1) [FAIT 16/11, §8.24] « Synthèse dans spec §8 » (plus bas) : tableau unique V1/V2 2017/2022/2027,
      abst = défaut, chiffres déjà committés, 0 fit ;
      (2) [FAIT 16/11, spec §7.1 : écarts ≤ 2 pt de P(élu) (Le Pen 80,3 → 78,9), drapeau éteint par défaut — dérive T1 de l'OU non calibrée sur les parts] « P(élu) à aléa unique » : maths spec §7 puis code + test, puis relancer
      `diagnostic_2027` ;
      (3) [FAIT 16/11, spec §8.25 : nœud T1 déplace le duel de +0,3 / −0,9 pt, erreur inchangée (4,5 / 5,9 pt) ⇒ non retenu] « entre-deux-tours conditionné sur le résultat T1 » : maths puis V2 T2−7
      2017/2022 ;
      (4) [FAIT 16/11, §8.26 : aucun gain] « (revue, a/d) stacking des chaînes » ;
      (5) [FAIT 16/11, spec_spatial_pooling §10 « Synthèse pour l'utilisateur »] synthèse tracker pour l'utilisateur (rôles spatial / gp-pooling).
      Les arbitrages ouverts (τ_T2, biais anti-RN, lissage, τ_paire) attendent
      l'utilisateur : coder chaque option DERRIÈRE un drapeau éteint par défaut
      est permis (chiffrer l'effet 2027), la choisir ne l'est pas. Les tâches
      2017 dépendant des dates et τ_T2 2002-2012 attendent la session données.
      « Rien à faire » n'est légitime que si TOUT ce qui précède est coché ou bloqué.

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
- [x] **PRIORITAIRE** : lever la multimodalité de la géométrie T1 du cadre
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
- [x] Phase 2 « relance au mode dominant » (`--init-depuis`, spec §8.8) :
      V1 Macron|Le Pen — FAIT 22:35Z : puits stable (4 chaînes retenues,
      rhat_max 2,3, écart 1,4 pt) mais prédiction 64,0 % pour 55,3 observé.
- [x] Variantes sans valence (spec §8.9, question utilisateur : « le T2 ne
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
- [x] `hier` : traiter la bimodalité de s_η (phase 2 §8.8, ou prior
      informatif), puis V2 2022 ×3 et diagnostic 2027 en `hier`.
      Phase 2 Zemmour puis Pécresse — FAIT 02:38Z : concluants tous deux
      (Zemmour 69,2 couvert ; Pécresse 52,6, limite structurelle).
      Phase 2 Mélenchon puis V2 2022 T1−1 `hier` — FAIT 03:15Z : MAE V1
      hier 5,0 pt ; V2 inchangé (52,3 contre 52,4).
- [x] Diagnostic 2027 `hier` (`python -m model.backtest.diagnostic_2027
      --variante hier`) — EN COURS (début 2026-11-15T03:20Z, session 01JZeMw2).
      Comparer à abst (§8.1) : critères (ii) et duels affichés.
      — FAIT 03:45Z : règle (ii) passe, convergence meilleure (R-hat 1,0095),
      duels affichés = abst à ≤ 0,5 pt. `hier` proposée en référence (§8.10).
- [x] 2017 en `hier` (V1 Macron|Le Pen, Fillon|Le Pen ; V2 T1−1, T2−7) —
      FAIT 03:53Z (spec §8.10) : 4 tâches convergées ; V2 = sondages, T2−7
      non couvert à τ 0,10 ; V1 centres bons (0,7 / 4,9 pt), IC très larges ;
      η ≈ 0 (l'échelle se rétracte).
- [x] (revue 15/11, b) [module `model/backtest/v1_2027.py`, spec §8.12 ; 4
      grosses paires × abst/hier FAITES 10:11Z ; complément FAIT 10:40Z
      (§8.13) : 6 tâches exécutables, hier MAE 2,19 / LS 11,61 / 6 concluants
      contre abst 3,05 / 7,95 / 5 ; 4 tâches Bardella INEXÉCUTABLES (t_k = 27/05,
      7 notices, Le Pen hors roster) ; couverture 4/6 : ajouter σ_T2 et le bruit
      de la cible à l'IC V1 (tâche ci-dessous)] V1 2027 « une paire retirée » (8 paires à n ≥ 4) et
      « un candidat retiré » (Attal, Bardella), variantes abst et hier, protocole
      §8.11 : ~20 fits, 4 par réveil ; d'abord ajouter le mode à
      `diagnostic_2027.py` (ou un module `v1_2027`) avec `--chaine/--fusionner`.
- [x] [FAIT 2026-11-15 16:50Z : `model/backtest/listes_definition.py`, livrable
      `model/backtest/results/listes_definition.md` ; 2022 apparié : D1 +40-80 %
      de sondages, −0,06 pt à 30 j, +0,01-0,05 pt à 7 j, IC50 trop étroits ;
      seul s = 5 % passe ; reco : garder D0. Diagnostic 2027 + D2 FAITS 17:20Z :
      D1(5 %) +8-17 % de sondages, courbes 40-60 % plus agitées ; D2 = D0]
      EN PARALLÈLE des fits NUTS (gp-pooling est en forme close, ~0,1 s/date :
      ça tient sur un cœur pendant les fits) — DÉFINITION DES LISTES gp-pooling
      (demande utilisateur via l'orchestrateur, 2026-11-15) : « le coût en
      précision d'un regroupement est-il compensé par plus de données ? ».
      Contexte : `model/listes.py` (main a3c9815), définition isolée dans
      `cle_champ` ; aujourd'hui champ EXACT (81 champs 2027, listes de tête à
      26/26/21 sondages). ATTENTION : un seuil sur la part MESURÉE dans chaque
      sondage est faux (la même liste se scinde à 2,9/3,1 % — c'est ce qui avait
      donné 106 champs et 15/15/14, chiffres invalides de la session locale).
      Définitions à comparer, le modèle gp-pooling restant inchangé :
        D0 champ exact (référence) ;
        D1(s) champ exact SANS les « petits » candidats, petit = part moyenne
          < s sur les sondages ANTÉRIEURS à la coupure (pas de fuite), s ∈
          {1, 2, 3, 5} % ; leurs voix retirées et les parts renormalisées ;
        D2 slots de gp-pooling (fusion d'alternatives type Bardella/Le Pen).
      Backtest 2017/2022, coupures ~mensuelles + T1−30/T1−7 : pour les 3 listes
      de tête à la coupure (règle de `model/listes.py`), erreur et couverture
      IC50/80/90 (a) sur les sondages suivants de la liste à 7 et 30 jours,
      (b) sur le résultat T1 (projection au scrutin du modèle) ; nombre de
      sondages par courbe. Diagnostic 2027 : régularité (variation jour à jour),
      stabilité au rejeu (ajout d'un sondage), résidus aux sondages. Livrable :
      tableau chiffré par définition × horizon + recommandation, dans
      `model/backtest/results/listes_definition.md`, et ligne
      « RÉSULTAT POUR L'ORCHESTRATEUR : … » en fin de réveil. Ne modifie PAS
      `model/listes.py` ni le site (session site) : backtest dans
      `model/backtest/listes_definition.py`.
- [x] [CLOS 16/11 : synthèse faite (spec_spatial_pooling §10) ; seul reste 2017, BLOQUÉ données] [spec_spatial_pooling §10 écrite ; `w_path` + `trajectoire()` codés et
      testés ; run 2027 FAIT 11:17Z (spec_spatial_pooling §10) : niveaux
      justes (≤ 0,4 pt), tendances amorties d'1/4 à 1/2 (Philippe −1,0 pt sur
      30 j). RESTE : backtest de suivi 2017/2022 par coupures + comparaison
      gp-pooling sur les listes du site : règle = `model.listes.classement`,
      reprise telle quelle] Étape 1 (spatial, 2022, 8 coupures) `model/backtest/
      suivi.py` FAIT 13:25Z, + étape 2 gp-pooling (`--gp`, dates décalées
      de T1 2027 − T1 2022) : sur listes sondées, gp gagne (MAE 0,62 contre
      0,78 ; couverture 0,90 contre 0,73 ; p = 0,0001) ⇒ le spatial ne doit
      pas remplacer gp-pooling comme tracker d'une liste sondée (spec
      spatial_pooling §10). IC prolongé à h = 7 j FAIT 13:40Z : couverture
      0,73 → 0,82 (≥ 5 % : 0,90), MAE inchangée. Étape 3 (détection Zemmour
      2021, spec spatial_pooling §10) FAIT 14:45Z : spatial détecte le
      05/10 comme le brut (retard nul), gp-pooling indisponible avant le 12/10
      (liste trop peu sondée) ⇒ avantage net du spatial pour un candidat
      nouveau. 2017 (Mélenchon mars-avril) BLOQUÉ PAR LES DONNÉES (15/11
      15:10Z) : avant le 05/04/2017, 2-3 notices datées et roster vide, or la
      montée est en mars — dépend de la session données (dates 2017). Reste : synthèse tracker pour
      l'utilisateur (rôle : spatial pour listes non sondées et candidats
      nouveaux, gp pour une liste sondée établie). APRÈS le diagnostic 2027 (demande utilisateur 2026-11-15) — TRACKER PAR
      CANDIDAT tiré du spatial : exporter la trajectoire w_i(t) du fit (chemin
      OU, déjà calculé ; seuls les candidats « dynamiques » > 5 % en ont une),
      présentée comme la PART de chaque candidat dans un champ de référence au
      fil du temps, avec bande IC90 (w brut est relatif, illisible). Puis
      BACKTEST DE SUIVI DE TENDANCE 2017/2022 : un fit par coupure (sondages
      connus à la coupure seulement — un fit unique en fin de campagne lisse avec
      le recul et ne vaut pas validation), ~10 coupures/élection, d'abord en fits
      allégés (2 chaînes) puis complets sur les coupures informatives. Comparer
      à gp-pooling sur les mêmes coupures (erreur, couverture, retard de
      détection des montées : Mélenchon 2017, Zemmour 2021). Maths/choix de
      présentation dans la spec avant le code. Ne touche PAS au site.
      COMPARAISON : sur les MÊMES listes que celles affichées par les gp-pooling
      multi-listes (session site : listes les plus testées parmi les ~10
      derniers sondages, chacune ajustée sur tout son historique) — reprendre
      leur règle de sélection à chaque coupure.
- [x] IC V1 : y ajouter σ_T2 et le bruit d'échantillonnage de la cible
      (moyenne de n sondages) avant de lire la couverture — 0 fit (re-fusion).
      FAIT 10:55Z (spec §8.14) : élargissement faible, couverture 2027 4/6 ;
      ratés = biais de structure 3-4 pt ⇒ piste τ_paire (ci-dessous, arbitrage 4).
- [x] (revue, c) abst en phase 2 sur Zemmour, Mélenchon, Pécresse 2022 — FAIT 16:17Z ; + abst 2017 ; VERDICT protocole §8.11 (spec §8.15) : hier gagne le log-score total (10,5 contre 7,7) mais perd la MAE 2017 ⇒ PAS préférée, abst reste le défaut
      (3 tâches) pour un traitement égal à hier.
- [x] [FAIT 10:30Z, spec spatial_pooling §10 : hors échantillon, MAE gp 0,60 (centres meilleurs, robuste) mais score d'intervalle gp 5,4-5,8 contre 4,3 spatial (IC gp trop larges) ⇒ « gp meilleur » vaut pour les centres seulement ; listes : comparaison relative peu affectée ; σ_h/σ_p dans le prédictif : non fait] (revue 16/11, A) **PRIORITAIRE, 0 fit** : recalibrer gp-pooling SANS 2022
      (REML et FACTEUR_DIFFUSION sur 2017 seul), puis relancer `suivi --gp` et
      `listes_definition` ; rapporter score d'intervalle α = 0,1 / CRPS par
      unité (coupure, liste) avec bootstrap, les deux modèles lus à c + 7 avec
      σ_h²/n_inst + σ_p²/n dans le prédictif. Tranche : si MAE gp > 0,70 ou
      score d'intervalle gp > 4,32, « gp bat le spatial » tombe.
- [x] (revue 16/11, 5) verdict abst/hier refait (spec §8.20, 2026-11-16) :
      6 tâches concluantes dédoublonnées, LS KDE ; Δ LS hier − abst = +0,53,
      IC90 [+0,14 ; +0,94], Δ erreur −0,79 pt [−1,62 ; −0,02] ⇒ **hier
      préférée par la règle**, robuste au retrait d'une tâche (de justesse).
      Réserve : Macron–Pécresse (écartée car abst non concluante) où hier est
      sûre d'elle à tort (LS −22). `ETA_HIER_T2 = True` (sans effet, T2 non
      publié). À faire relire à la revue du 17/11.
- [x] (revue 16/11, 4) détection Zemmour refaite dès h = 243 (spec
      spatial_pooling §10, 16/11) : brut détecte le 28/09, spatial le 05/10 —
      retard dû à l'entrée au roster (Zemmour hors roster jusqu'au 31/08, 1
      sondage en août), pas au lissage ; à base égale (liste L_c), brut et
      spatial même semaine ; gp sur L_c 3 semaines plus tard (26/10).
      Piste : abaisser MIN_POLLS pour un candidat nouveau ? (non chiffré).
- [x] (revue 16/11, 6 et C) placebo même-notice (spec §8.21, 16/11) : bras
      notice pire que le placebo sur 0/4 tâches ⇒ fuite non établie (cohérent
      avec l'effet notice §8.23 : duels d'une notice redondants). τ_paire
      calibré : 0,08 (V1 2027, hier ; §8.22).
- [x] (revue 16/11, 7) σ_maison par REML (spec §8.23, 16/11) : σ_h 0,5-1 pt
      par institut, établi en 2017/2022 (p ≤ 0,008), non établi en 2027
      (p ≈ 0,1) ; effet notice 0,8-1,1 pt. §8.17 retiré.
- [x] [FAIT 16/11, spec §8.23 bis : LS meilleur 3/4 mais erreur moyenne 4,97 → 6,40 pt (Zemmour +7,8, non convergé) ⇒ NON RETENU ; σ_T2 1,3 → 0,2-0,3 pt, absorbé par maison/notice] (§8.23) effets maison + notice dans la vraisemblance des duels : maths
      complètes dans la spec, code derrière un drapeau, V1 2022 (4 paires,
      16 fits) contre le modèle actuel, log-score KDE.
- [x] [FAIT 17/11, spec §8.23 bis : LS meilleur 2/4, erreur 4,97 → 5,35 pt ⇒ non retenu ; aucun changement de modèle] (§8.23 bis, suite) effet NOTICE seul (marginalisé par blocs, 0 latent) en V1 2022, 16 fits.
- [x] [FAIT 17/11, spec_spatial_pooling §10 « Résultat B » : C1 (centre gp, largeur spatial) IS 3,84 contre 4,30 / 5,83, mais Δ contre spatial [−1,00 ; +0,14] ⇒ non recommandée selon la règle ; à revoir sur 2017] (revue 16/11, B) combiner spatial et gp (stacking par unité, poids selon
      le nombre de sondages de la liste et l'horizon ; ou gp sur résidus du
      spatial), évaluation leave-one-coupure-out (0 fit).
- [x] [FAIT 16/11, spec §8.26 : 18 fusions 2022, Δ MAE −0,04 pt [−0,39 ; +0,28] ⇒ sélection gardée] (revue, a/d) stacking des chaînes (log-vraisemblance par nœud sur les
      tirages sauvés) ou max local + Laplace par chaîne — maths dans la spec.
- [x] (revue, e) [FAIT 19:05Z (spec §8.16) : fuite comptée (74/86), mais retirer les notices entières ne dégrade pas l'erreur des centres ; seuls les IC s'élargissent ⇒ V1 non gonflée par la fuite] V1 en retirant la NOTICE entière (fuite même-répondants) :
      comptage sans fit, puis 4 fits.
- [x] [FAIT 17/11, spec §8.29 : hier préférée, CRPS p = 0,04, LS plafonné p = 0,008, même signe 3/3 élections] (revue 17/11, 1/B) verdict abst/hier sur les 10 tâches : CRPS ou LS
      plafonné (−5) sur `tirages_predictifs`, inversion des signes par grappe
      d'élection, croisé avec τ_paire par variante (0 fit).
- [x] [FAIT 17/11, spec §8.30 : recentrage ≤ 1 pt ; 1 gain / 1 perte sur les concluantes, Zemmour déraille ⇒ non adopté] (revue 17/11, 3/A) hier_maison : recentrer la cible V1 avec les z_maison
      a posteriori des instituts de la cible, re-scorer les 4 tâches 2022 avec
      la règle d'exclusion de §8.20 (0 fit).
- [x] [FAIT 17/11, spec §8.31 : hier 36,2 contre 36,5 (erreur 0,3 pt), τ_paire ne fait que coûter] (revue 17/11, 2/C) V1 « X contre Y » : Mélenchon–Philippe (2 sondages),
      abst, 4 chaînes (~15 min) — seul type de paire qui reçoit ν en production.
- [x] [FAIT 17/11, spec §8.32 : erreurs 3,5 / 4,0 / 6,4 pt selon la graine, sd ≈ 1,6 pt ; gain Attal non robuste] (revue 17/11, 6) variance de graine du sous-échantillon : bras notice
      partielle Attal–Le Pen avec 2 graines de plus (8 fits).
- [x] [FAIT 17/11, spec §8.33 : τ_paire sans effet sur P(élu) ; d = −3 pt ⇒ Le Pen 80,3 → 74,7 %] (revue 17/11, C) sensibilité de P(élu) sur le champ modal : décalage
      anti-RN 0/−1/−2/−3 pt × τ_paire 0,11/0,19 (éclaire l'arbitrage 2, ne le
      tranche pas ; 0-1 fit).
- [x] [FAIT 18/11 01:40Z] diagnostic 2027 hier rejoué sur le jeu à 118 notices : duels affichés inchangés à ≤ 0,2 pt, P(élu) Le Pen 80,3 → 80,8 %, Philippe 13,9 → 13,6 % ; R̂ 1,017 ; résultats 2027 antérieurs toujours valables (`diagnostic_2027_hier_2026-11-13_118notices.json`).
- [x] [FAIT 18/11, spec §8.35 : λ hors élection 1,3-2,2, couverture 6→8 (abst), 7→9 (hier) ; τ additif équivalent ≈ 0,17-0,19] (revue 18/11, idée 2) facteur d'élargissement λ des IC V1 appris hors
      élection (2 élections → la 3e), cible RMS z ≈ 1 (actuel 1,9) ; 0 fit.
- [x] [FAIT 18/11 19:40Z, spec §8.36 : scénario cohérent (biais −3,5 pt + τ 0,10) ⇒ P(élu) Le Pen 82,1 → 76,0 %, Philippe 13,4 → 18,7 %] (revue 18/11, 5) `p_elu(tau_t2=…)` ; sensibilité cohérente d = −3,5 pt
      avec τ résiduel ≈ 0,10 (1 fit).
- [ ] [GEL FAIT 18/11 20:40Z : prédictions abst et hier au 17/11 (78 paires, R̂ 1,015) dans results/prospectif/ ; À CHAQUE RÉVEIL où main apporte des duels datés après le 17/11 : `python -m model.backtest.prospectif_2027 --scorer`, committer scores.json ; lecture après ≥ 20 duels (§8.37)] (revue 18/11, idée 1) test prospectif 2027 : figer les prédictions abst
      et hier de toutes les paires (2 fits), puis scorer chaque nouveau duel
      publié (CRPS, z) sans fit.
- [ ] (revue, 9) τ_T2 : a posteriori sur 2002-2022 (normale vs Student-t),
      données à sourcer (session données), effet sur P(Philippe bat Le Pen).
- [x] [PÉRIMÉ, nettoyé 2026-11-15 21:30Z par la session locale : revue Opus faite
      08:20Z, `hier` jugée non préférée §8.15] Ensuite : revue Opus avant de changer le défaut.
- [x] [FAIT : JSON abst v1/v2 2017 présents dans results/, intégrés §8.15] abst : v1 2017 Macron|Le Pen, Fillon|Le Pen ; v2 2017 T1-1, T2-7
- [x] [ABANDONNÉ : eta n'est plus candidate au défaut (§8.15), pas de valeur ajoutée] eta : mêmes tâches (comparaison)
- [x] [FAIT 20:12Z, spec §8.18 : base brut ± τ : 2/5 couverts à 0,10, 5/5 à 0,19 ; corr. intra-notice : sans objet après §8.16] Revue 2026-11-14, sans fit : corriger la base V2 (écart-type τ_T2 en plus
      de « largeur du modèle », rapporter les deux) ; post-fit : corrélation
      intra-notice des résidus (`duel_s` + `duels["notices"]` désormais dans le fit).
      Déjà mesuré sur 2027 (§8.1) : |corr(log κ, η_i)| ≤ 0,22, z̄² = 0,75 —
      à refaire sur les fits 2017/2022.
- [x] [FAIT 19:15Z, spec §8.17 : aucun effet stable (Harris +1,6 / −0,2 / −0,9 selon l'élection ; 2022 tout à ±0,4 pt) ⇒ pas de paramètre] Effets maison au T2 (Harris −0,9 / YouGov +0,9 pt sur 2027) : mesurer
      sur 2017/2022 avant toute modélisation ; maths dans la spec d'abord.
- [x] [FAIT 16/11 : 0,19 appliqué, spec §6.2] Décision τ_T2 (maths dans la spec §6 avant code) : la mesure du 14/11
      donne RMS ≈ 4,3 pt ⇒ τ ≈ 0,18-0,20 logit ; centrage sur 0 = choix
      éditorial, à faire trancher par l'utilisateur (ne pas coder de biais).
- [x] [FAIT 16/11, spec §8.24] Synthèse dans spec §8 : V1 MAE modèle / géométrie seule / BT ; V2 couverture,
      log-score vs sondages ; R-hat, rhat_t2, κ, ω, η ; corrélation κ–η.
- [x] [TRI 17/11 02:15Z : entre-deux-tours FAIT §8.25 ; aléa unique FAIT §7.1 ;
      effet notice FAIT §8.23 bis ; κ = 1 SANS OBJET (corr(log κ, η) ≤ 0,17 en 2027,
      pas de crête) ; η partagé Le Pen/Bardella NON MESURABLE (les V1 Bardella sont
      inexécutables, §8.13) ; τ_T2 leave-one-election-out et tableaux croisés :
      BLOQUÉS données ; valeur inclusive sur le champ FAIT §8.27 : identique à ≤ 0,2 pt, sans objet]
      Puis pistes (choisir selon les résultats, maths dans la spec AVANT le code) :
  - entre-deux-tours conditionné sur le résultat T1 (nœud quasi exact) ;
  - P(élu) à aléa unique (w projeté au T1 -> π(T1) -> prolongé au T2) ;
  - η hiérarchique partagé Le Pen/Bardella (bloc), V1 sur paires sans autre duel ;
  - τ_T2 en leave-one-election-out ;
  - valeur inclusive sur le CHAMP (masque) au lieu du roster (alias LR 2027) ;
  - contrôle κ = 1 fixé vs κ libre (si crête κ–η confirmée) ;
  - effet aléatoire par notice sur les duels (si corr. intra-notice > 0,3) ;
  - tableaux croisés T1×T2 des instituts (si la session données les fournit).

## Arbitrages ouverts (pour l'utilisateur — ne pas trancher seul)

**TRANCHÉS par l'utilisateur le 2026-11-15 ~21:45Z** (« je valide tes points de
vue », sur les propositions de la session locale) :
1. τ_T2 → **0,19** (élargir ; centres inchangés).
2. Biais anti-RN → **NON corrigé** (deux élections, estimation fragile).
3. Lissage → **inchangé** (pas de τ_OU T2 plus court).
- ARBITRAGE / ALERTE (18/11 14:15Z) : Polony et Riner (5 notices chacun, 1,3 % et
  3,7 %) sont apparus dans les sondages 2027 et ne sont pas dans ORDER_GROUPS :
  tout fit spatial 2027 échoue (« candidats éligibles absents de ORDER_GROUPS »).
  La maj quotidienne de main du 18/11 n'a produit AUCUN fichier
  docs/data/spatial-pooling/2026-11-18.json (gp-pooling, lui, a tourné). Leur
  placement est éditorial : à faire sur main (roster.py, ORDER_GROUPS).
- ARBITRAGE À TRANCHER (18/11, §8.34) : variante T2 — hier (meilleure en moyenne
  sur 10 tâches, mais le verdict ne tient qu'au plafond du log-score ; sûre
  d'elle à tort sur Pécresse 2022) ou abst (plus prudente) ? hier en place, sans
  effet publié.
- (Ajout 18/11 §8.35 : hors élection, les IC V1 demandent un élargissement
  équivalent à τ ≈ 0,17-0,19 sur les 10 tâches 2017-2027.)
- ARBITRAGE À TRANCHER (mis à jour 2026-11-18 : τ_paire mis à **0,10**, borne basse
  de la fourchette validée ; ancien texte du 17/11 11:20Z ci-après, spec §8.22/§8.29) :
  variante hier (défaut, §8.29) ⇒ calibrage V1 2027 **0,08** (appliqué, SOUS la
  fourchette 0,10-0,15 validée) ; abst donnerait 0,11 ; avec 2017-2022, hier
  demande **0,19** (Pécresse). Courbe plate, 4 tâches toutes « RN contre X » :
  les données ne tranchent pas. 0,08, 0,10-0,15 ou 0,19 ? (Ajout 17/11 §8.33 : τ_paire ne change pas P(élu), seulement les duels
  jamais sondés de l'onglet Scénarios ; enjeu faible.) (Ajout 17/11 §8.31 :
  la seule paire « X contre Y » testable, Mélenchon–Philippe, est prédite à
  0,3 pt près ; τ_paire n'y fait que coûter.)
- À PRÉSENTER À L'UTILISATEUR (revue 17/11, point 9) : « gp pour le niveau,
  spatial pour les IC » est un choix de présentation du site.
4. τ_paire → **ADOPTÉ** pour les seuls duels jamais sondés, calibré en
   leave-one-pair-out (≈ 0,10-0,15 logit).
Ces décisions NE valent PAS accord de publication : `PUBLIER_T2` reste False
(critère de publication §8 + accord explicite requis). Le texte ci-dessous est
conservé pour l'historique.


1. **Largeur τ_T2 de l'aléa de scrutin** (actuel 0,10 logit ; mesure
   2017/2022 ⇒ 0,18-0,20). Backtest V2 2022 (spec §8.5) : à 0,10 le
   résultat sort de l'IC90 à T1−1 et T2−7 (z ≈ 2), à 0,19 il y rentre.
   Base « sondages bruts ± τ » (§8.18) : 2/5 couverts à 0,10, 5/5 à 0,19.
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

4. **Incertitude d'extrapolation des duels jamais sondés** (spec §8.14) : la
   géométrie se trompe de 3-4 pt sur 2 des 6 paires retirées 2027 sans que l'IC
   le prévoie. Ajouter $\nu_{ab} \sim \mathcal N(0, \tau_{paire}^2)$ (≈ 0,10-0,15
   logit, calibré en leave-one-pair-out) aux duels sans sondage élargirait leur
   IC90 d'environ ±4-6 pt ; ne touche pas les duels sondés. Choix de
   présentation (honnêteté vs précision affichée) : à trancher.

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

- 2026-11-18 20:08-20:40Z (routine) : test prospectif 2027 (§8.37) : gel des prédictions abst et hier au 17/11 (2 fits).
- 2026-11-18 19:08-19:40Z (routine) : main fusionnée (Riner, Polony placés) ; sensibilité P(élu) cohérente (§8.36, 1 fit).
- 2026-11-18 14:08Z (routine) : sensibilité P(élu) cohérente bloquée — Polony/Riner absents de ORDER_GROUPS (fit refusé) ; alerte remontée.
- 2026-11-18 13:50Z (routine) : main fusionnée ; élargissement λ hors élection (§8.35, 0 fit) : τ équivalent 0,17-0,19.
- 2026-11-18 12:08-13:50Z (routine) : piste B révisée (ancres par famille, 16 fits) : non adoptée.
- 2026-11-18 11:08-11:30Z (routine) : revue Opus (8 points intégrés) ; τ_paire 0,10 ; verdict hier requalifié « préférence faible », arbitrage remonté ; (B) réouverte.
- 2026-11-18 00:30Z (routine) : main fusionnée — nouveau parseur wiki 2027 : 92 → 118 notices (dernier terrain toujours 13/11). Les résultats 2027 (V1, diagnostic) datent du jeu à 92 notices ; à rejouer si un écart matériel est suspecté. Pas de tableaux de reports encore (A bloquée).
- 2026-11-17 22:48 → 18/11 00:30Z (routine) : pistes utilisateur — (B) bloquée (CHES hors domaines réseau ; repli ORDER_GROUPS sans effet) ; (C) σ_w par candidat (32 fits) : non adopté.
- 2026-11-17 16:20Z (routine) : main fusionnée (maj quotidienne : 92 notices, dernier terrain 13/11) ; tests verts hors test réseau. File : seule reste τ_T2 2002-2012 (bloquée données).
- 2026-11-17 15:08-16:20Z (routine) : sensibilité de P(élu) (§8.33, 2 fits) : τ_paire sans effet ; anti-RN −3 pt ⇒ Le Pen −5,6 pt de P(élu).
- 2026-11-17 14:08-14:25Z (routine) : variance de graine du sous-échantillon (§8.32, 8 fits) : ≈ 1,6 pt.
- 2026-11-17 13:08-13:25Z (routine) : V1 Mélenchon–Philippe (§8.31, 8 fits) : erreur 0,3 pt.
- 2026-11-17 12:08Z (routine) : hier_maison recentré par les effets maison a posteriori (§8.30, 0 fit) : aucun gain établi.
- 2026-11-17 11:08Z (routine) : verdict robuste abst/hier (§8.29, 0 fit) : hier préférée ; ETA_HIER_T2 = True, τ_paire = 0,08.
- 2026-11-17 10:07-10:30Z (routine) : revue Opus quotidienne (10 points intégrés) ; verdict hier retiré, abst rétabli, τ_paire 0,11.
- 2026-11-17 02:08-02:45Z (routine) : tri des pistes ; valeur inclusive sur le champ (§8.27, 16 fits) : sans effet. File : ne reste que des tâches bloquées par les données.
- 2026-11-17 01:10Z (routine) : idée B (combinaison spatial/gp, 0 fit) : C1 meilleure, non établie.
- 2026-11-16 22:30 → 17/11 01:10Z (routine ; VM redémarrée en route, reprise) : effet notice seul en V1 2022 (20 fits) : non retenu.
- 2026-11-16 20:39-22:30Z (routine) : effets maison/notice sous forme par blocs, V1 2022 (20 fits) : non retenus (Zemmour déraille) ; piste effet notice seul.
- 2026-11-16 18:10-20:35Z (routine) : synthèse des rôles spatial/gp ; effets maison T2 marginalisés (§8.23 bis) codés, mais V1 2022 arrêtée (aucune chaîne en 2 h 15 : MVN pleine trop chère) ⇒ reformulation par blocs.
- 2026-11-16 18:08Z (routine) : synthèse des rôles spatial / gp-pooling (spec_spatial_pooling §10) ; RELANCE du 15/11 entièrement traitée.
- 2026-11-16 17:08-17:45Z (routine) : stacking des chaînes (§8.26), 18 re-fusions : aucun gain, sélection gardée.
- 2026-11-16 16:08-16:50Z (routine) : entre-deux-tours conditionné sur le résultat T1 (§8.25, 12 fits) : non retenu.
- 2026-11-16 15:20-15:40Z (routine) : P(élu) à aléa unique (§7.1), drapeau éteint ; diagnostic 2027 rejoué.
- 2026-11-16 15:08Z (routine) : main fusionnée ; synthèse V1/V2 (spec §8.24, 0 fit).
- 2026-11-16 13:08-14:15Z (routine) : arbitrages appliqués (τ_T2 = 0,19,
  τ_paire = 0,08 calibré, §6, §8.22 ; valeur 0,08/0,19 remontée) ; placebo
  même-notice (28 fits) : fuite non établie (§8.21) ; effets maison T2 par REML
  (§8.23) : 0,5-1 pt établis en 2017/2022, §8.17 retiré.
- 2026-11-16 12:08-12:30Z (routine) : détection Zemmour refaite dès h = 243
  (16 fits) + séries à base égale L_c ; spatial = brut à base égale, gp en
  retard de 3 semaines ; retard du spatial tous champs = entrée au roster.
- 2026-11-16 11:10-12:00Z (routine) : verdict abst/hier refait (§8.20) —
  hier 2027 ×5 + abst Hollande rejoués sur les données à jour, 16 re-fusions à
  l'identique avec tirages prédictifs, log-score KDE, bootstrap : hier
  préférée par la règle ; `ETA_HIER_T2 = True` (sans effet publié).
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
- 2026-11-16 10:07-10:35Z (réveil) : tâche A (gp recalibré sur 2017 seul) :
  centres gp meilleurs (robuste), intervalles spatial meilleurs hors
  échantillon.
- 2026-11-16 09:07-09:35Z (réveil) : revue Opus quotidienne, 9 points
  acceptés ; rétractations (gp contre spatial, décalage de dates, détection
  Zemmour, fuite, effets maison, τ 5/5, D1(5 %)) ; tâches A (prioritaire),
  4, 5, 6/C, 7, B en file.
- 2026-11-15 20:08-20:15Z (réveil) : base V2 brut ± τ (§8.18).
- 2026-11-15 19:07-19:20Z (réveil) : effets maison T2 mesurés (§8.17),
  aucun stable.
- 2026-11-15 18:07-19:10Z (réveil) : fuite même-notice (§8.16) : comptée,
  mesurée (8 fits), sans effet sur les centres.
- 2026-11-15 17:07-17:25Z (réveil) : diagnostic 2027 des définitions de
  listes (D0/D1/D2), livrable complété.
- 2026-11-15 16:27-16:55Z (réveil 16:07) : backtest de définition des listes
  gp-pooling (demande orchestrateur), livrable listes_definition.md.
- 2026-11-15 15:07-16:30Z (réveil) : suivi 2017 bloqué par les données ;
  abst phase 2 (2022) et abst 2017 ; verdict du protocole : abst reste le
  défaut (spec §8.15).
- 2026-11-15 14:07-14:50Z (réveil) : étape 3 du suivi (détection Zemmour
  2021) : spatial sans retard, gp indisponible.
- 2026-11-15 13:28-13:45Z (réveil 13:07) : IC du suivi spatial prolongé
  (couverture 0,82) ; merge main.
- 2026-11-15 12:07-13:30Z (réveil) : backtest de suivi 2022, étapes 1-2
  (spatial vs gp-pooling sur les listes du site) ; Wauquiez classé.
- 2026-11-15 11:08-11:25Z (réveil) : merge main ; tracker (spec
  spatial_pooling §10, `w_path`, `trajectoire`, `tracker.py`), run 2027 ;
  test de main `test_listes` adapté (mock de load_raw_polls sans `tour`,
  interaction avec le chargement des deux tours de la branche). Tests : 165
  passés (hors réseau).
- 2026-11-15 10:38-10:58Z (réveil) : IC prédictif V1 (§8.14), re-fusion,
  arbitrage 4 (τ_paire) ouvert.
- 2026-11-15 09:13-10:45Z (réveil) : module V1 2027 (§8.12) ; 12 fits ;
  hier meilleur sur 2027 (§8.13) ; tâches Bardella inexécutables ; journal
  nettoyé (tâches faites cochées), tracker utilisateur remonté en file.
- 2026-11-15 ~08:15-08:45Z (demande utilisateur) : revue Opus ; règle de
  sélection corrigée (L sur tous les tirages, SE, aval restreint), s_η dans
  rhat_t2 ; re-fusion de tous les backtests ; « hier = référence » retirée ;
  protocole fixé (spec §8.11). Tests : 150 passés (hors réseau).
- 2026-11-15 04:07-04:12Z (réveil) : piste d'abstention écartée par le
  calcul (§8.10), sans fit.
- 2026-11-15 03:44-03:55Z (réveils 02:07/03:07 reçus en retard) : 2017 hier
  (4 tâches, §8.10).
- 2026-11-15 02:00-03:45Z : hier phase 2 (Zemmour, Pécresse, Mélenchon),
  V2 T1−1 hier, diagnostic 2027 hier ⇒ hier proposée en référence.
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

## Revue Opus 2026-11-18 — intégration point par point

1. §8.29 fragile (plafond) : **INTÉGRÉ** (vérifié : −5/−8/−10/aucun → p 0,008 /
   0,053 / 0,22 / 0,50) ; requalifié « préférence faible », arbitrage remonté.
2. (B) mal diagnostiquée : **INTÉGRÉ** (vérifié : 22/24 écarts au plancher en
   2022) ; tâche réouverte (écart minimal intra-famille).
3. IC V1 trop étroits : **INTÉGRÉ** (RMS z 1,86 / 1,95, couverture 6/10, 7/10) ;
   λ en file.
4. τ_paire hors fourchette validée : **INTÉGRÉ**, mis à 0,10.
5. §8.33 double compte + bruit MC : **INTÉGRÉ** ; rejeu cohérent en file.
6. §8.30 tableau mélangé, « concluant » ignorant R̂ T2 : **INTÉGRÉ** (texte).
7. §8.31/§8.32 : **INTÉGRÉ** (texte).
8. 118 notices / (C) : **INTÉGRÉ** (« duels inchangés » mécanique ; (C) non
   adopté maintenu).
Idées 1 (test prospectif 2027) et 2 (λ hors élection) : en file.

## Revue Opus 2026-11-17 — intégration point par point

1. §8.20 hier préférée : **INTÉGRÉ, verdict retiré** (vérifié : t90 [−0,01 ;
   +1,06], inversion des signes p = 0,0625) ; abst rétabli ; refaire sur 10
   tâches (file).
2. τ_paire : **INTÉGRÉ** — 0,11 (calibrage abst, gain hors éch. +0,17) au lieu
   de 0,08 ; non-représentativité (« RN contre X ») notée, V1 Mélenchon–Philippe
   en file ; arbitrage mis à jour.
3. §8.23 bis : **INTÉGRÉ** — « non évaluable », pas « non retenu » ; recentrage
   par z_maison en file (0 fit).
4. REML : **INTÉGRÉ** (réserves 2017 K_n = I, Harris 2022, cellules seules 2027
   écrites §8.28).
5. Détection Zemmour : **INTÉGRÉ, rétractation** (dates de première détection
   possibles imposées ; seul l'écart de niveau tient), spec spatial_pooling §10.
6. Placebo : **INTÉGRÉ** (raisonnement corrigé ; variance de graine en file).
7. §8.27 : **INTÉGRÉ** (masque contaminé par les alias ; non refait, effet
   attendu nul).
8. §8.24/§8.25 : **INTÉGRÉ** (couverture dans l'échantillon ; usage non testé).
9. Combinaison C1 : **INTÉGRÉ** (choix a posteriori, IC optimiste ; choix de
   présentation remonté).
10. Stacking / §7.1 : **INTÉGRÉ** (poids LOO peu fiables, décision inchangée).
Idées A, B, C : en file.

## Revue Opus 2026-11-16 — intégration point par point

1. gp contre spatial, couverture : ACCEPTÉ, VÉRIFIÉ en partie (13 unités, pas
   171 couples) ; seul l'écart de MAE tient ; score d'intervalle à égalité
   (4,32). Rétractation écrite (spec spatial_pooling §10).
2. Décalage de dates « exactement valide » : ACCEPTÉ, VÉRIFIÉ (bank_opinion.json
   par REML sur blocs historiques ; FACTEUR_DIFFUSION = 3 calé sur 2017/2022 ;
   σ_h = effet institut, `gp_math.py` : `base = Kou + sigma_h**2 * same`).
   Rétractation écrite ; tâche A prioritaire.
3. Lecture à dates différentes et IC sans σ_h/σ_p : ACCEPTÉ ; dans la tâche A.
4. Détection « sans retard » imposée par le protocole (première date possible
   = 05/10) : ACCEPTÉ ; conclusion retirée ; tâche (4) avec coupures dès août.
5. Verdict §8.15 doublonné et veto non concluant : ACCEPTÉ ; à refaire (0 fit) ;
   le défaut abst reste en place, sans nouvelle décision.
6. Fuite même-notice non établie : ACCEPTÉ ; placebo en file.
7. Effets maison : ACCEPTÉ (mauvaise hypothèse testée) ; REML par campagne en
   file ; la « décision pas de paramètre » est suspendue.
8. τ_T2 5/5 non probant (2 erreurs indépendantes, même signe, IC90 de τ
   [0,12 ; 0,92]) : ACCEPTÉ ; réserve écrite (spec §8.19) ; données 1995-2012
   nécessaires ; l'arbitrage 1 reste à l'utilisateur.
9. Listes « seul D1(5 %) passe » dans le bruit : ACCEPTÉ ; retiré ; la
   recommandation « garder D0 » tient pour une autre raison (pas de gain
   établi, IC50 trop étroit).
Idées A, B, C : toutes en file, A prioritaire (0 fit).

## Revue Opus 2026-11-15 — intégration point par point

1. Règle Δ = 10 nats au niveau du bruit (20 tirages, SE ~4 nats sur un écart ;
   faux rejet Pécresse phase 2) : ACCEPTÉ, VÉRIFIÉ (400 tirages : 10549,2 /
   10547,4 / 10547,5 / 10547,2, SE 1,6-2,0) ; CORRIGÉ : L̄ sur tous les
   tirages, seuil max(10, 4·SE), re-fusion de tous les JSON.
2. « Masse ∝ e^L » faux à l'échelle de Δ (entropie 4-23 nats) : ACCEPTÉ ;
   documenté (§8.11) ; la règle reste un garde-fou contre les pièges ≫ 10
   nats ; stacking mis en file.
3. Phase 2 ne teste pas l'existence d'autres modes : ACCEPTÉ — elle teste la
   stabilité d'un puits, pas son unicité ; libellé corrigé, cartographie
   (8 chaînes dispersées ou max local + Laplace) mise en file.
4. Sélection appliquée à la cible seule : ACCEPTÉ, CORRIGÉ (les chaînes
   rejetées sont retirées avant tout l'aval).
5. `rhat_t2` sans s_η ; V2 T1−1 hier à 1 chaîne cité comme concluant ; NaN :
   ACCEPTÉ, CORRIGÉ (s_η ajouté, nanmax/nanmin) ; après re-fusion V2 T1−1 hier
   garde 2 chaînes (concluant) ; phrase « converge mieux » restreinte à 2027.
6. « hier = référence » prématuré : ACCEPTÉ. Après re-fusion cohérente,
   log-score Σ hier −5,0 / abst −5,5 / geo −82,9 (l'inversion relevée venait
   en partie du mélange poolé/retenu), mais égalité de fait, traitement
   inégal et jardin aux sentiers : proposition retirée, protocole fixé (§8.11).
7. Le backtest historique ne teste pas le régime de production : ACCEPTÉ ;
   V1 2027 (paire retirée, candidat retiré) devient le jeu de test principal.
8. Fuite même-notice en V1 ; Le Pen–Pécresse annoncé mais non produit :
   ACCEPTÉ (mis en file) ; Le Pen–Pécresse n'a que 1 sondage < MIN_SONDAGES_PAIRE
   = 3, d'où son absence de `--lister` — spec §8 à corriger en ce sens.
9. τ_T2 décide de la couverture et de P(élu) plus que la variante : ACCEPTÉ ;
   reste un arbitrage utilisateur ; a posteriori 2002-2022 mis en file.
Idées (a)-(e) : toutes en file, (a) partiellement faite (L sur tous les
tirages), (b) prioritaire.

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
