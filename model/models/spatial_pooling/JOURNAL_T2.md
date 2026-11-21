# Journal — recherche second tour (`spatial-pooling`)

Mémoire de la routine cloud horaire « pres2027 — recherche T2 ». Chaque réveil
repart à froid : TOUT l'état utile vit ici. Lire en entier avant d'agir, mettre
à jour avant de finir. Spec : `spec_second_tour.md`. Ne pas réécrire
l'historique, ajouter en bas de « Historique ».

## État (mis à jour à chaque réveil)

- Dernière revue Opus : 2026-11-20 ~11:20Z (routine, couvre R3, R3b, R3c, §8.48 ; cf. « Revue Opus 2026-11-20 »).
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

- [x] [FAIT 21/11 08:05Z, §8.55 « Résultat de R6 » : ÉCHEC (1 : 8,2 ; 2 : Mélenchon 50,5 ; 4 : mode dominant à 1 chaîne ; « Aucun » 9,1), mais σ_LP/σ_M 0,97 ✓ ; parcimonie : R6 ≥ A′1 sur 1-3, mais instable ; prédiction de la revue contredite (JLM 50,5 > 48,6 ; Zemmour→JLM 23 < 35) ; Zemmour |z_pos| 3,9 ⇒ ancres forcées ; suite logique : R7] [EN COURS VM (début 2026-11-20 23:02Z) : R6 `hiercercle_ra_ferme --lse-officiel`, 4 processus ; durée attendue proche d'A′1 corrigé (environ 10 h par chaîne), donc des résultats vers 09:00Z] **DEMANDE VM n° 3 (session locale, 2026-11-20 ~23:10Z)** — R6, cercle FERMÉ (A = 2π, un
      paramètre de moins), spec §8.55 (critères figés avant tout fit), code sur la branche. Même
      plateforme et même nombre de chaînes qu'A′1 corrigé, auquel il est comparé (VM, 4 chaînes) :
      `python -m model.backtest.second_round --only v2 --election 2022 --coupure T1-30 --variante hiercercle_ra_ferme --lse-officiel --chaine k`
      pour k = 0..3, puis `--fusionner --lse-officiel` et
      `python -m model.backtest.modes --variante hiercercle_ra_ferme --stem second_round_hiercercle_ra_ferme_v2_2022_T1-30_lseo --n 4 --lse-officiel`
      (`PRES2027_N_CHAINES=4`). Verdict selon §8.55, règle de parcimonie comprise.

- **ALERTE (session locale, 2026-11-20 12:30Z, revue Opus locale point 1.1, VÉRIFIÉ) — reports
  d'avant T1 BROUILLÉS par `arrays_reports`** (fit.py, groupement par (notice, électorat) : les
  hypothèses de duel d'une même notice sont mélangées, la dernière lue gagne). 2022 : 179 groupes
  (notice, électorat) sur 196 de la fenêtre T1−30 mêlent plusieurs hypothèses (jusqu'à 6). Les
  tableaux d'entre-deux-tours (une hypothèse par notice) ne sont PAS touchés. Conséquences :
  · A′1 (VM, en cours), A′2, B′2, les diagnostics de modes et la config `A2_cercle` sont INVALIDES ;
  · la routine peut ARRÊTER A′1 (ou le finir, mais ne pas en tirer de verdict) ;
  · R3c (DEMANDE VM n° 2) est SUSPENDU jusqu'au correctif et à la reprise d'A′ ;
  · le rerun Mac d'A′2 avec `--lse-officiel` est arrêté.
  Correctif en cours par la session locale (groupement par hypothèse + champ par ligne, point 1.2).

- ORIENTATION UTILISATEUR (2026-11-20 ~11:00Z), à respecter dans la suite de R3b et des revues :
  « passer le plus d'info possible via la géométrie pour révéler la structure politique plutôt
  que de la contraindre » ; le cercle reste une option viable. Avis d'expert : avec le front
  républicain, un σ très large pour Le Pen n'est pas crédible. Or c'est exactement le mécanisme
  par lequel le SEGMENT reproduit les reports en A′2/B′2 : σ_Le Pen 0,19-0,26 contre σ_Macron
  0,07, si bien que la gaussienne large de Le Pen l'emporte aux deux extrémités (« fer à cheval »
  par la largeur). Conséquences :
  (i) LECTURE DE PLAUSIBILITÉ ajoutée à toute évaluation A′/B′ (sans changer les verdicts
      figés) : σ_Le Pen / σ_Macron et σ_Le Pen / médiane des σ du champ ; un rapport ≫ 1 = le
      mécanisme jugé non crédible par l'utilisateur ;
  (ii) le CERCLE garde la priorité de modélisation : il fait croiser les extrêmes par l'arrière
      sans élargir Le Pen. A′1 (VM) et A′3 (GPU) à lire avec (i) ;
  (iii) un témoin « segment, σ_Le Pen contraint étroit » serait le test du mécanisme ; à
      spécifier (maths et critères) seulement si A′1 ne tranche pas.
  La parcimonie (consigne du même jour) se lit ensuite : à plausibilité égale, le plus simple.
- RÉSULTAT B′2 (session locale, Mac, 2026-11-20 10:45Z ; segment `hier_ra`, 2017 à T1−30 avec
  reports d'AVANT T1, 4 chaînes de 8-18 min ; JSON `second_round_hier_ra_v2_2017_T1-30.json`,
  `eval_r2_hier_ra_2017_T1-30{,_par_chaine}.json`) — critères §8.47.6 :
  1. partage Mélenchon 83,0 (mesuré 76,4 ; +6,6) ✓ sur la chaîne retenue (n° 2, seule) ; les 3 autres
     chaînes donnent 92 (+15,6) ;
  2. MAE des 4 partages 14,8 pt ✗ (Hamon +1,5, Fillon +2,6, Mélenchon +6,6, Dupont-Aignan +48,5).
     Dupont-Aignan a rallié Le Pen ENTRE les deux tours (accord du 28/04/2017) : aucun report
     d'avant T1 ne pouvait le prévoir. Hors Dupont-Aignan, la MAE vaut 3,6. Le critère, figé, reste
     un échec ; l'explication est consignée, pas utilisée pour le réécrire ;
  3. stabilité ✗ : lp 2887 / 2889 / 3005 / 2888 (118 nats ; une seule chaîne dans le mode dominant),
     duels avec Hamon jusqu'à 31 pt d'écart ;
  4. T1 contre le résultat 1,96 pt ✓ (≤ 3,1).
  LECTURE : même sens qu'A′2. Les reports d'avant T1 rapprochent le partage Mélenchon de la mesure
  (+6,6 au lieu de la saturation), mais la multimodalité reste l'obstacle. Le mode dominant n'est
  visité que par 1 chaîne sur 4.
- DIAGNOSTIC 8 CHAÎNES DE B′2 (session locale, 2026-11-20 11:50Z, `results/diag_modes_hier_ra_2017_T1-30.json`).
  Deux modes, séparés de 117 nats :
  · mode dominant (lp ≈ 3004-3005, chaînes 2, 4, 7) : partage Mélenchon 83 (mesuré 76,4), Fillon 65
    (62,8), σ_LP/σ_Macron 2,4 (Le Pen LARGE), T1 1,95-1,96 ;
  · mode secondaire (lp ≈ 2887-2889, 5 chaînes) : Mélenchon 92, Fillon 51, σ_LP/σ_Macron 1,1
    (Le Pen ÉTROITE), T1 1,73.
  La lecture de B′2 (chaîne retenue = mode dominant) tient. Mais 2017 confirme le mécanisme vu en
  2022 : dans le SEGMENT, ce qui fait coller les reports est l'élargissement de Le Pen, que
  l'utilisateur juge non crédible. Le mode « Le Pen étroite » prédit MIEUX le T1 (1,73 contre 1,96)
  et PIRE les reports. Le T1 veut une Le Pen étroite, les reports la veulent large : le segment
  ne peut pas satisfaire les deux, c'est la tension que le cercle (A′1) et l'abstention située
  (R3c) doivent lever.
- CORRECTION DE LA LECTURE D'A′2 (session locale, 2026-11-20 11:35Z ; diagnostic à 8 chaînes, graines
  27-34, `results/diag_modes_hier_ra_2022_T1-30.json`). Deux modes nets :
  · mode H (densité HAUTE, lp 11236-11241, chaînes 1, 4, 6, 7) : σ_Mélenchon 0,021-0,032,
    partage Mélenchon 85-91 % (mesuré 63), T1 2,87-2,92 pt, σ_LP/σ_Macron 2,3-2,5 ;
  · mode B (lp 11229-11231, chaînes 2, 3, 5) : σ_Mélenchon 0,12-0,13, partage 49 %, T1 2,98,
    σ_LP/σ_Macron 3,7-3,8 ; la chaîne 0 (lp 11208, 62,8 %) est en dessous des deux.
  Le « 67,4 % » d'A′2 venait du MÉLANGE des chaînes 1 (mode H) et 3 (mode B), retenues ensemble
  parce que leur écart (8 nats) passe sous le seuil de sélection de §8.8 (10 nats). Le mode le
  plus dense, qui porte ≈ e^8-e^10 fois plus de masse, donne 85-91 % : dans le postérieur, les
  reports d'avant T1 NE corrigent PAS Mélenchon dans le segment. Le critère 1 d'A′2, lu sur le
  mode dominant, ÉCHOUE (+22 à +28 pt). Mêmes réserves à appliquer à toute fusion qui mêle des
  chaînes à moins de 10 nats mais dans des géométries différentes : faille du protocole §8.8, à
  soumettre à la revue Opus (sélection par mode plutôt que par seuil de lp ?). B′2 (2017) n'est
  pas concerné : une seule chaîne y est retenue, 117 nats au-dessus des autres.
  Plausibilité : σ_LP/σ_Macron vaut 2,3 à 3,8 dans tous les modes du segment.
- RÉSULTAT R4 (session locale, Mac, 2026-11-21 05:00Z ; noyau de Student-t, segment, reports d'avant T1
  corrigés, `--lse-officiel`, B = 100, 8 chaînes) — §8.51/§8.52/§8.54 :
  · ν est appris et lourd : médiane 1,7 à 4,5 selon la chaîne ; sd(log ν) 0,08 ; q95 5,1 (6′ ✓).
  · PATHOLOGIE : σ_Mélenchon s'effondre (0,0005 à 0,01) dans les 8 chaînes ; la log-densité s'étale sur
    160 nats (12260 à 12419). Mode dominant = la seule chaîne 1 (ν 4,5, σ_JLM 0,010).
  · critères sur le mode dominant :
    1. MAE des partages 10,4 ✗ ;
    2. Mélenchon 85,1 (63) ✗ ;
    3. σ_LP/σ_M 0,64 ✓ ; 3b. monotonie 100 % ✓ ;
    4. stabilité ✗ (1 chaîne) ;
    5. T1 2,90 ✓ ;
    Aucun 7,2 ✗ ; structure : LP→JLM 15,2 ✗ (≥ 45), Zemmour→JLM 14,9 ✓.
  · corr(log ν, log σ_LP) 0,15 ; résidus « Aucun » du fit : propres +3,15, autres −1,46 ; c 17,9.
  LECTURE : les queues lourdes rendent Le Pen étroite et la monotonie juste, mais Mélenchon se replie
  sur un pic quasi ponctuel dont seules les queues recrutent. Le fer à cheval disparaît
  (LP→JLM 15 contre 57) et Mélenchon repart à 85 %. Règle §8.52-7′ : R4 n'est PAS retenu (échec à 1
  et 2, comme le gaussien). Le noyau t ne lève pas la tension : il la déplace.
- RÉSULTAT A′2 CORRIGÉ (session locale, Mac, 2026-11-20 17:25Z ; segment gaussien `hier_ra`, reports d'avant
  T1 corrigés §8.50, `--lse-officiel`, B = 100, 8 chaînes ; lecture `model/backtest/modes.py`) :
  · UN SEUL MODE : 8 chaînes sur 8 à moins de 3 nats (lp 12028-12031), duels clés à ≤ 0,37 pt
    entre chaînes. La multimodalité d'hier venait des données brouillées et des fantômes.
  · critères §8.50 / §8.52, mode dominant = 8 chaînes :
    1. MAE des 4 partages 7,27 > 3,61 ✗ — Mélenchon 66,0 (63,0), Pécresse 51,5 (65,0),
       Jadot 78,5 (87,5), Zemmour 13,1 (9,4) ;
    2. Mélenchon +2,9 ✓ ;
    3. σ_LP/σ_Macron 3,71 ✗ ; 3b. monotonie à gauche 0 % des tirages ✗ (Le Pen l'emporte sur
       Macron chez les votants à gauche de Macron : le mécanisme jugé non crédible) ;
    4. stabilité ✓ (8/8, 0,37 pt ; VERDICT FAVORABLE SUR MAC = non probant, cf. règle) ;
    5. T1 3,03 ✓ (couverture IC90 2/12) ; Aucun : MAE 5,47 > 3,75 ✗ (Pécresse 15,2 contre 29,9).
  · σ_Mélenchon ≈ 0,27 : Mélenchon devient lui aussi très large.
  LECTURE : avec les bonnes données, le segment gaussien est stable et place bien Mélenchon,
  mais par le mécanisme « Le Pen très large » (3b = 0 %), et il rate Pécresse et Jadot. Selon
  §8.52-7′, le gaussien échoue à 1 et 3b : R4 (Student-t) a donc sa chance. Il tourne ensuite
  sur le Mac.
- RÉSULTAT A′2 (session locale, Mac, 2026-11-20 10:20Z ; segment `hier_ra`, 2022 à T1−30 avec
  reports d'AVANT T1, 4 chaînes de 25-47 min ; JSON `second_round_hier_ra_v2_2022_T1-30.json`,
  `eval_r2_hier_ra_2022_T1-30{,_par_chaine}.json`, commit 989f304) — critères §8.47.4 :
  1. partage Mélenchon 67,4 (mesuré 63,0 ; +4,4) ✓ — sans ces reports : 100 (§8.44) ;
  2. MAE des 4 partages 3,2 pt (Mélenchon +4,4, Pécresse +1,8, Jadot +2,1, Zemmour +4,4) ✓
     (seuil 14,7 ; persistance naïve après T1 : 3,6) ;
  3. stabilité ✗ : partage Mélenchon 62,8 / 86,0 / 49,0 / 48,8 selon la chaîne (écart 37 pt),
     duels Le Pen–Mélenchon 9,6, Mélenchon–Zemmour 9,3, Mélenchon–Pécresse 9,3 ; lp
     11208 / 11239 / 11229 / 11231 (31 nats) ; R̂ max 7,4. ÉCHEC probant même sur le Mac
     (le Mac sous-explore les modes, il ne les invente pas) ;
  4. T1 contre le résultat 2,91-2,99 pt ✓ (1D sans reports 2,98).
  LECTURE : le SEGMENT, avec les reports d'avant T1, prédit en moyenne les reports de Mélenchon
  avant le T1 : le cercle n'est pas nécessaire pour cela (parcimonie, consigne utilisateur du
  20/11). Reste la multimodalité de Mélenchon entre deux modes (≈ 49 et ≈ 86 %). À comparer
  avec A′1 (cercle, VM).
- [x] [CLOS 20/11 22:55Z : refits faits avec le drapeau, A′1 (VM, cercle) et A′2 (Mac, segment), lus en §8.54 ; le défaut reste à mesurer sur le fit 2027 avant toute publication du T2 (non publié)] [CODE FAIT 20/11 11:39Z (`--lse-officiel` dans second_round et eval_r2, `masque_lse_officiel`, masque aussi dans le bloc reports) ; mesure 1 à 0 fit (§8.49) : sans les fantômes, la case « Aucun » de Pécresse 26,3 → 11,1 (mesuré 29,9) ⇒ le fit utilisait Bertrand comme réservoir d'abstention ; refits avec le drapeau NÉCESSAIRES, après A′1] **(NOUVEAU 20/11, revue point 1, PRÉALABLE à R3c et aux reruns A′) Valeur inclusive sans candidats fantômes** : spec §8.49 ; drapeau `lse_officiel` (backtests : champ officiel du T1 ∪ finalistes), tests ; mesure à 0 fit sur les chaînes existantes, puis reruns A′1/A′2 avec le drapeau, en 8 chaînes, à B = 100 pour le segment.
- [x] [FAIT Mac 2026-11-21 04:47Z, `results/modes_second_round_hier_ra_t_v2_2022_T1-30_lseo.json` ; CODE 477d0eb, et sur T4 si le GPU revient (jeu `_lseo_t4` séparé, avec un A′2 corrigé T4 jumeau)] **(R4) Noyau de Student-t** (revue locale 3.3, spec §8.51, critères §8.50 + ν < 30 figés) : `hier_ra_t --lse-officiel`, B = 100, 8 chaînes ; comparé à A′2 corrigé sur la MÊME plateforme.
- [ ] [CODE FAIT 603f493 ; EN FILE Mac juste après R4] **(R7) Fidélité révélée** (spec §8.56, critères figés) : `hier_ra_fid --lse-officiel`, B = 100, 8 chaînes.
- [ ] [CODE FAIT f1bfcbc ; EN FILE Mac après R7 ; prédiction figée : échec au critère 2] **(R5) σ_Le Pen ≡ σ_Macron** (revue locale PM, point 4 ; spec §8.53, critères figés) : `hier_ra_lie --lse-officiel`, B = 100, 8 chaînes ; s'il passe 1, 2 et 3b, il prime sur R4.
- [x] [FAIT Mac 2026-11-20 17:20Z, `results/modes_second_round_hier_ra_v2_2022_T1-30_lseo.json`] **A′2 relancé selon §8.49 (revue, points 1-2)** :
      `hier_ra --lse-officiel`, `PRES2027_B_QUAD=100`, 8 chaînes (graines 27-34), 2 à la fois sur le
      Mac ; sortie `second_round_hier_ra_v2_2022_T1-30_lseo_c*` ; lecture chaîne par chaîne. La
      routine n'a PAS à le lancer.
- NOTE (session locale, 12:00Z) : la DEMANDE VM n° 2 (R3c) ci-dessous se lance AVEC `--lse-officiel`
  (§8.49 ; `--chaine k --lse-officiel`, puis `--fusionner --lse-officiel`, `eval_r2 … --lse-officiel`),
  et se compare à A′1 RELANCÉ avec le même drapeau, pas à A′1 actuel.
- [ ] [EN ATTENTE (routine, 20/11 11:40Z) : la revue (points 1 et 5) montre que la valeur inclusive inclut des candidats fantômes (Bertrand w = 3,3 > Macron dans le cercle 2022) ; or R3c modifie justement le terme d'abstention ⇒ R3c sera lancé APRÈS le correctif §8.49, avec le drapeau, pour ne pas tester une abstention située sur une abstention biaisée. Si la session locale ou l'utilisateur préfère le lancer tel quel, l'écrire ici.] **DEMANDE VM n° 2 (session locale, utilisateur, 2026-11-20 ~11:30Z) — À LANCER APRÈS A′1, sur
      la VM (même plateforme qu'A′1, auquel il est comparé)** : R3c, abstention située sur le
      cercle (idée utilisateur ; spec §8.48 bis, critères figés ; code et tests sur la branche).
      `python -m model.backtest.second_round --only v2 --election 2022 --coupure T1-30 --variante hiercercle_ra_abs --chaine k`
      pour k = 0..3, puis `--fusionner`, puis
      `python -m model.backtest.eval_r2 --variante hiercercle_ra_abs --coupure T1-30` et `--par-chaine`.
      Lire aussi γ (quantile 5 %) et σ_Le Pen/σ_Macron contre A′1. Verdict selon §8.48 bis.5.
- [x] [FAIT 20/11 22:50Z, §8.54 « Résultat d'A′1 corrigé » : ÉCHEC (1 : MAE 7,9 contre 2,6 pour la persistance ; 2 : Mélenchon 48,6 contre 63 ; 3 : σ_LP/σ_M 1,64 ; « Aucun » 9,0), mais STABLE (duels à 0,09 pt dans le mode dominant 0/2/3), monotonie circulaire 0 %, contrôles de structure OK ; κ 0,35, arc 3,97 (37 % à l'arrière) ; environ 10 h par chaîne sur la VM] [21:00Z : 8 h 27 sans qu'aucune chaîne ait fini, processus sains (1 cœur chacun) ; plus lent que le cercle à T2−7 (environ 3 h, 511 pas) ⇒ probablement au plafond de 1023 pas. On le laisse tourner, la VM n'ayant rien d'autre à faire, jusqu'à 03:00Z au plus ; au-delà, coût jugé prohibitif sur CPU (fermer le cercle, A = 2π, ou passer par le T4)] [EN COURS VM (début 2026-11-20 12:35Z) : A′1 RELANCÉ avec les deux correctifs (groupement par hypothèse §8.50, 46740e6 ; `--lse-officiel` §8.49), 4 processus, sortie `second_round_hiercercle_ra_v2_2022_T1-30_lseo` ; le premier A′1 (09:32Z) a été ARRÊTÉ à 12:33Z, invalide (reports brouillés et fantômes), sans verdict] **DEMANDE VM (session locale, au nom de l'utilisateur, 2026-11-20 ~09:35Z)** — le GPU Colab
      est indisponible depuis 23:50Z ; l'utilisateur demande d'utiliser la VM et le Mac.
      Étape **A′1 de R3b** (spec §8.47.5, critères 1-4 de §8.47.4 figés) sur la VM, 4 processus
      en parallèle (marquer EN COURS, watcher) :
      `python -m model.backtest.second_round --only v2 --election 2022 --coupure T1-30 --variante hiercercle_ra --chaine k`
      pour k = 0..3, puis `--fusionner`, puis
      `python -m model.backtest.eval_r2 --variante hiercercle_ra --coupure T1-30` et `--par-chaine`.
      Commiter les JSON, consigner les chiffres en spec §8.47 et le verdict selon les critères.
      Pour le critère 3, l'écart de lp est celui de la fusion (une seule élection). Code : cd9f341
      (`_reports_avant`), déjà sur la branche. La session locale fait A′2 (`hier_ra`) sur le Mac
      et garde A et A′3 (joints) pour le GPU : ne PAS lancer ceux-là.

- **COORDINATION GPU (utilisateur, 2026-11-19 ~19:50Z) — règle permanente.** Le GPU (T4
  Google Colab) est piloté UNIQUEMENT par la session locale, depuis le Mac de l'utilisateur.
  La routine n'utilise pas Colab et ne cherche aucun identifiant. Pour un fit lourd :
  1. la routine pousse le code et les maths, puis ajoute ici une ligne
     « - [ ] DEMANDE GPU (<date UTC>) : `<commande python -m … --chaine k>` pour k = … ;
     sortie attendue : <fichiers .npz> » et la marque EN COURS côté demande ;
  2. la session locale exécute sur le T4, toutes les chaînes d'une même étape sur la même
     plateforme, et commite les `.npz` dans `model/backtest/results/chaines/` sous les noms
     standard, puis coche la ligne « FAIT (T4, <date>) » ;
  3. la routine fusionne (`--fusionner`) et rend le verdict selon les critères figés.
  Ne jamais mélanger dans une même étape des chaînes VM et T4. Si la session locale ne
  répond pas (Mac fermé), la routine peut lancer la demande sur la VM, mais seulement si
  aucune ligne FAIT ou EN COURS (T4) n'existe pour cette étape : dans ce cas elle écrit
  « REPRISE VM » sur la ligne et l'étape entière passe sur la VM.
  Étape 1 de R3 (V2 T2−7 2022, avec reports) : sur la VM, en cours (lancée 18:31Z). AUSSI
  lancée par la session locale sur T4 (20:29Z) comme JEU COMPLET SÉPARÉ, sortie
  `second_round_hiercercle_v2_2022_T2-7_t4` (4 chaînes T4), pour débloquer R3b. Pas de
  mélange : les chaînes VM et T4 restent deux fits distincts ; le premier fini fait foi pour
  avancer, l'autre sert de contrôle de reproductibilité entre plateformes.
  RÉSULTAT T4 (session locale, 21:30Z ; 10-15 min par chaîne ; fusion et `eval_r2` sur le Mac,
  JSON `second_round_hiercercle_v2_2022_T2-7_t4.json`, `eval_r2_hiercercle_2022_t4.json`,
  `eval_r2_hiercercle_2022_T2-7_par_chaine_t4.json`) — lecture brute, verdict formel à la
  routine (§8.46.8, étape 1) :
  - partages (toutes notices) : Mélenchon 71,1 (mesuré 63,0 ; +8,1), Pécresse 61,5 (65,0),
    Jadot 87,5 (87,5), Zemmour 9,0 (9,4) ⇒ MAE des 4 partages 3,0 pt (2D 2,6 ; 1D avec
    reports 8,2) ; NON calculé : la version hors fit (11 notices après `as_of`) ;
  - T1 contre le résultat 1,09 pt (1D 1,23 ; 2D 1,17) ; σ_v 0,049 ; κ 0,73 ; arc 6,04 rad ⇒
    f_arrière 0,038 seulement ;
  - stabilité : lp par chaîne 5024 / 5695 / 5526 / 5692 (écart 671 nats), R̂ max 9,3 (mu),
    chaînes retenues 1 et 3 ; écarts entre chaînes : partage Pécresse 8,7 pt, duel
    Le Pen–Pécresse 7,8, Zemmour–Pécresse 5,7 ⇒ critère 4 révisé ÉCHOUE, comme pour la 2D.
  ⇒ Le cercle REPRODUIT les reports quand il les a (prérequis de R3b rempli), mais sa
  géométrie est multimodale.
  Étape 2 de R3 (V2 T1−30 2022, sans reports) : FAIT (T4, 19:41-20:14Z ; 6-10 min par chaîne
  sur le T4). Les `.npz` sont gitignorés : la session locale a fait elle-même la fusion
  (`--fusionner`) et `eval_r2 --variante hiercercle_sr --coupure T1-30 [--par-chaine]` sur
  le Mac (sans NUTS), et commité les 3 JSON (`second_round_hiercercle_sr_v2_2022_T1-30.json`,
  `eval_r2_hiercercle_sr_2022_T1-30{,_par_chaine}.json`). Lecture brute, à consigner en
  spec §8.46 par la routine (verdict selon §8.46.8, étape 2) :
  - partage Mélenchon prédit 99,9 % dans les 4 chaînes (mesuré 63) : le T1 seul ne place PAS
    Mélenchon à l'arrière ; Jadot 52,7 % (mesuré 87,5 ; 38,9 à 84,8 selon la chaîne) ;
  - écart de lp entre chaînes 299 nats (9124,9 / 8835,3 / 8853,1 / 9134,4 ; 1D : 4,0 ;
    2D : 28,7) ; chaînes retenues 0 et 3, R̂ max 1,92 même entre elles ;
  - duels hypothétiques entre chaînes : Le Pen–Zemmour 13,6 pt, Mélenchon–Pécresse 13,1,
    Le Pen–Mélenchon 5,5 (seuil 3) ; Macron–Le Pen 0,5 ;
  - MAE des parts T1 contre le résultat 2,84 à 2,94 pt par chaîne (1D 2,98-2,99) ;
  - arc 5,54 rad ⇒ f_arrière 0,12 ; σ_v moyen 0,055 ; κ 0,80 ; Macron–Le Pen 59,1
    (résultat 58,55).
  ⇒ Selon la règle figée : étape 2 ÉCHOUE (Mélenchon, stabilité, multimodalité). Reste à
  lire l'étape 1 : si elle passe, conclusion « outil d'entre-deux-tours seulement »,
  comparée à la 2D en coût et en stabilité.

- [x] [FAIT 19/11 23:20Z, §8.48 : refit B = 100 ⇒ σ_Mélenchon 0,020 → 0,031 (artefact partiel), Macron–Le Pen 53,5 → 53,6, reports 1D toujours saturés, T1 inchangé ⇒ aucune conclusion ne change ; règle : backtests historiques à B = 100 si `quadrature_fragile` ; production non touchée] [ex-EN COURS (début 2026-11-19 22:14Z) : §8.48 mesure faite (duels Mélenchon jusqu'à 1,1 pt, Macron–Le Pen 0,5 à T1−30 ⇒ sensible) ; garde-fou `sigma_min` / `quadrature_fragile` ajouté aux diagnostics ; refit témoin 1D T2−7 2022 avec B = 100 lancé (4 chaînes, `r2_temoin_hier_B100`)] **(NOUVEAU 19/11, constat de §8.46.4) Précision de la quadrature 1D aux petits σ** : GL B = 25 donne jusqu'à 1,6 pt d'erreur sur π dans les fits 2022 (σ min 0,018) ; sans effet en 2027 (σ min 0,051). Tâche : garde-fou (alerte si σ_min < 0,035, ou B adaptatif) et mesure de l'effet sur les backtests 2022 publiés (V1/V2) ; priorité moyenne, pas de changement de production sans chiffrage.
- NOTE (session locale, 2026-11-19 ~19:45Z, REMPLACE l'aperçu Mac) : l'ÉTAPE 2 OFFICIELLE de R3
  (`--only v2 --election 2022 --coupure T1-30 --variante hiercercle_sr`, chaînes 0-3) tourne
  sur un GPU T4 Google Colab (Linux x86 + CUDA, jax 0.10.2 / numpyro 0.22.0, float32 comme
  la production), pilotée par la session locale (le CLI colab et ses identifiants sont sur le
  Mac de l'utilisateur, pas sur la VM). NE PAS lancer l'étape 2 sur la VM. La session locale
  commitera les 4 `.npz` dans `results/chaines/` sous les noms standard ; la routine fera la
  fusion et le verdict selon §8.46.8. Les 4 chaînes sont sur la MÊME plateforme (T4).
  Banc d'essai T4 (modèle cercle, V2 T1−30 2022) : gradient 0,63 ms (1 chaîne, float32),
  0,29 ms/chaîne en vmap 16-256 (gain ×2 seulement : le GPU est déjà saturé par une chaîne),
  float64 5 fois plus lent ; NUTS 1 chaîne 0,43 s/itération (≈ 508 pas/itération), soit
  ≈ 7 min pour 600+400. Chaînes Mac `_mac` arrêtées, non utilisées.
- [ ] [EN COURS session locale 2026-11-19 ~21:00Z : maths §8.47 (4e49fa7), code `model/backtest/multi_election.py` + tests (0459aa9, 249 verts) ; T4 : étape A (configs A_cercle puis A_segment, 4 chaînes chacune) en file derrière l'étape 1 _t4 ; 21:35Z session Colab perdue, GPU retenu côté serveur (1 GPU max sur le compte gratuit) : la session locale retente toutes les 15 min ; la routine NE reprend PAS R3b sur la VM sans note contraire] [20/11 ~00:00Z : main fusionnée ici (3c931e6, conflit .gitignore résolu : les deux lignes) — la routine n'a plus à la refaire. Reports d'AVANT T1 (b6c29d3) branchés : drapeau `_reports_avant`, variantes `hiercercle_ra`/`hier_ra`, config `A2_cercle` (cd9f341), protocole A′ figé §8.47.5 ; file T4 : A (en cours) puis A′1 cercle seul, A′2 segment seul, A′3 joint avec reports d'avant T1] [20/11 01:40Z : session T4 perdue vers 23:50Z ; Colab répond « Service Unavailable » à toute demande de GPU (quota gratuit du jour probablement épuisé, ~4 h de GPU utilisées). Rapatriées : A_cercle c0, c1. Le pilote local retente toutes les 10 min jusqu'à ~07:00Z. La routine NE reprend PAS sur la VM : on ne mélange pas les plateformes dans une étape, et l'étape A complète sur VM coûterait plus que l'attente] [PRIS PAR LA SESSION LOCALE (utilisateur, 2026-11-19 ~21:40Z) : maths, code et fits T4 faits par la session locale ; la routine NE le fait PAS, elle relit et critique (revue quotidienne) et termine l'étape 1 de R3] **(R3b) CERCLE NOURRI PAR TOUTES LES DONNÉES (utilisateur, 2026-11-19 ~21:20Z)** —
      suite de l'échec de l'étape 2 de R3 (cercle sans reports à T1−30 : Mélenchon 99,9 %,
      299 nats entre chaînes). Consigne utilisateur : « utilise les reports si nécessaire ;
      toutes les données doivent nourrir le même modèle ». Maths dans la spec AVANT le code,
      critères figés AVANT lecture, drapeau éteint, production inchangée, `PUBLIER_T2` False.
      Fits lourds : DEMANDE GPU (règle de coordination ci-dessus).
      **Diagnostic.** Avant le T1 d'une élection, il n'existe aucun tableau de reports
      (vérifié : `reports_t2_2017/2022.csv` sont tous datés après le T1). Le T1 seul ne
      dit pas qui est « à l'arrière ». Mais les reports des élections PASSÉES le disent, et
      la structure (LFI/Mélenchon, RN/Le Pen) persiste d'une élection à l'autre. La bonne
      forme est donc un modèle JOINT MULTI-ÉLECTIONS, où les reports passés informent la
      géométrie de l'élection en cours.
      **Modèle à écrire (proposition, à critiquer avant code).**
      - Fit conjoint 2017 + 2022 (puis + 2027) : T1, duels, reports de chaque élection,
        chacun avec sa vraisemblance actuelle.
      - Positions angulaires hiérarchiques par FAMILLE politique (table §8.43) :
        θ_{candidat,e} = θ_{famille} + δ_{famille,e} + ε_{candidat,e}, avec δ (dérive
        d'élection) et ε (candidat) d'échelles apprises, priors centrés 0. Les candidats
        récurrents (Mélenchon, Le Pen, Macron) peuvent porter un terme propre partagé.
      - Arc A : global, ou hiérarchique par élection (A_e = A + a_e).
      - σ_i : idem, hiérarchique par famille.
      - Ce qui reste propre à chaque élection : w (niveaux et chemins), κ, η, ω, τ.
      - Multimodalité (299 nats) : la hiérarchie par famille ancre les positions ; chiffrer
        aussi l'effet de priors plus serrés sur A et sur θ_famille. Ne PAS initialiser les
        chaînes depuis un postérieur avec reports (ça fausserait le test de stabilité) :
        inits dispersées comme d'habitude.
      **Protocole (critères proposés, à figer dans la spec avant lecture).**
      0. Pré-requis : étape 1 de R3 (V2 T2−7 2022 avec reports, VM) lue. Si le cercle ne
         reproduit même pas les reports AVEC eux, R3b est sans objet : consigner, arrêter.
      1. Backtest prospectif honnête : 2017 complet (T1, duels, reports) + 2022 jusqu'à
         T1−30 (SANS reports 2022). On prédit les reports 2022 mesurés, les duels et le T1.
         - partage Mélenchon 2022 à < 10 pt du mesuré (63) ;
         - MAE des 4 partages 2022 < celle du 1D (14,7 sans reports) et, si possible,
           < persistance naïve (3,6, à recalculer pour ce cadre) ;
         - duels hypothétiques stables entre 4 chaînes (≤ 3 pt), écart de lp ≤ 10 nats ;
         - T1 2022 non dégradé contre le 1D (MAE des parts contre le résultat).
      2. Symétrique : 2022 complet + 2017 jusqu'à T1−30. Même critères pour 2017 (76,4).
      3. Si 1 et 2 passent : fit 2017 + 2022 + 2027, effet sur 2027 (P(élu), duels,
         stabilité §8.45), puis « ARBITRAGE À TRANCHER » sur l'adoption (touche le T1
         publié).
      Témoin obligatoire : le même modèle joint en 1D (segment), pour séparer l'apport du
      cercle de celui des reports passés.
      **Données en plus, demandées le 19/11 à la session pipeline-données :** tableaux
      croisés « intention de vote T2 selon le vote T1 » publiés AVANT le T1 (2017, 2022,
      2027). S'ils existent, ce sont des reports d'avant T1 directement utilisables dans
      l'élection en cours ; les intégrer comme observations dès leur arrivée sur main.
- [x] [PROTOCOLE 1 RÉPLIQUÉ SUR LA VM 19/11 22:10Z, §8.46 « Réplication VM » : 4 chaînes CPU d'environ 3 h, toutes retenues (lp à 6 nats près, contre 671 sur T4) ; Mélenchon hors fit +6,8 (11 notices après as_of), MAE hors fit 3,3 ; T1 1,08 ; échec STRICT sur la stabilité entre chaînes (partages Mélenchon 5,6, Pécresse 4,5 ; duel Zemmour–Pécresse 6,1, dû à une chaîne). Même verdict que T4 (d1c98e5) : le cercle reproduit les reports, la stabilité échoue ⇒ arrêt de R3 ; la suite est R3b (session locale)] [ex-EN COURS (début 2026-11-19 18:31Z) : code fait (drapeau `carte_cercle`, `--variante hiercercle|hiercercle_sr`, tests), protocole 1 lancé : V2 T2−7 2022 hiercercle, 4 chaînes] [MATHS FAITES 19/11 18:26Z, spec §8.46 (positions 1D posées sur un arc avant A appris, von Mises κ = 1/(Aσ)², trou derrière les extrêmes, N_θ = 64, critères figés) ; À FAIRE : code + tests + protocole 1] **(R3) VOTANTS SUR UN CERCLE (utilisateur, 2026-11-19 ~18:30Z)** — suite
      de §8.43-8.44. Maths dans la spec (nouveau §) AVANT le code, critères
      fixés AVANT lecture, code derrière un drapeau ÉTEINT, production
      inchangée, `PUBLIER_T2` False. Fits sur la VM cloud uniquement.
      **Idée.** Remplacer le segment $v\in[0{,}01;0{,}99]$ par un cercle
      $\theta\in[0,2\pi)$ : électorat uniforme, noyau de von Mises
      $\varphi_i(\theta)\propto\exp(\kappa_i\cos(\theta-\mu_i))$, normalisé
      sur la quadrature comme en 1D. Différence avec §8.40 : ce n'est pas un
      terme « fer à cheval » ajouté à un axe linéaire (qui restait saturé), mais
      une géométrie où les extrêmes peuvent devenir voisins par l'arrière. Sur
      un cercle, la médiatrice du duel Macron–Le Pen coupe le cercle en DEUX
      points : l'un au centre, l'autre « à l'arrière ». Un candidat placé près
      du point arrière (Mélenchon ?) voit son électorat se partager entre les
      deux finalistes sans désaturer le logit et sans élargir σ. C'est
      exactement le mécanisme qui manquait en 1D.
      **Points forts à vérifier.**
      - Coût ≈ 1D : une seule coordonnée, pas de produit de quadratures (R2
        coûtait 7 fois plus). Vérifier le nombre de nœuds nécessaire sur le
        cercle (quadrature trapèze, exacte pour les fonctions périodiques
        lisses).
      - Le segment est un cas limite : si l'arc entre les deux extrêmes est
        grand devant les κ_i, on retrouve le 1D. Les données tranchent, ce qui
        fait du 1D un témoin naturel.
      - Moins de symétries que R2 : rotation (fixer un angle, par exemple le
        centre) et réflexion (contrainte d'ordre gauche/droite sur les
        voisins du centre).
      **Points durs à écrire dans la spec.**
      - ORDER_GROUPS devient un ordre CYCLIQUE avec un « trou » libre : où le
        cercle se ferme-t-il ? Prior sur la longueur de l'arc arrière, ou les
        deux extrêmes libres de se rejoindre.
      - Correspondance des priors de position et de σ (von Mises κ_i ↔ σ_i).
      - La quantité η/κ/ω du T2 et la valeur inclusive ne changent pas, seule
        l'intégrale porte sur θ.
      - Le site affiche une position gauche-droite : définir sa projection
        (angle relatif au centre), ou dire qu'elle n'a plus de sens.
      **Protocole, dans cet ordre (le test décisif est le 2, celui que R2 a
      raté).**
      1. V2 T2−7 2022 AVEC reports (comme hier2d) : partage Mélenchon hors
         fit (11 notices après `as_of`) à < 10 pt du mesuré ; MAE hors fit des
         partages ≤ celle de hier2d (2,6) ou au moins < 1D avec reports (8,2) ;
         T1 non dégradé ; stabilité entre chaînes (critère 4 révisé de §8.44).
      2. V2 T1−30 2022 SANS reports, 4 chaînes indépendantes : est-ce que le
         T1 seul place Mélenchon près du point arrière ? Critères de l'idée 2
         de §8.44 : partage Mélenchon prédit, stabilité des duels
         hypothétiques entre chaînes (≤ 3 pt), multimodalité (écart de lp) pas
         pire que le 1D, T1 non dégradé.
      3. Si 2 passe : 2017 (V2 T1−30), puis V1 selon §8.29, puis l'effet sur
         2027 (P(élu), duels, stabilité §8.45), puis « ARBITRAGE À TRANCHER »
         sur l'adoption, qui toucherait aussi le T1 publié.
      Si 1 échoue, consigner et s'arrêter. Si 1 passe mais 2 échoue : même
      conclusion que R2 (outil d'entre-deux-tours seulement), à dire
      explicitement en comparant cercle et 2D sur coût et stabilité.

- [ ] **NOUVELLES PISTES REPORTS (utilisateur, 2026-11-19 ~08:00Z)** — suite de
      §8.41. Le diagnostic de la 2D (« le noyau séparable garde l'écart en v, σ
      étroit fixé par le T1 sature le logit ») vaut pour une 2D GREFFÉE sur un T1
      1D figé, pas pour une carte 2D estimée dès le T1. Les deux pistes ci-dessous
      sont dans cet ordre, maths dans la spec AVANT tout code, critères fixés
      AVANT lecture. Elles passent AVANT la revue Opus de (A), sauf si la revue
      quotidienne est due : dans ce cas la revue d'abord, et elle couvre aussi
      ces deux pistes. Le scoring prospectif 2027 reste à chaque réveil.
  - [x] [FAIT 19/11 ~08:00Z, spec §8.43, `results/reports_t2_r1_{hier,abst}.json` : ÉCHEC (hier R1a) — Mélenchon −7,2 / +13,3 pt ; π_GR 0,09 (2017) contre 0,27 (2022), instable ; année retirée −17 / +25 pt ; seul π_GR nettement non nul (forme juste, paramètre d'élection et non de famille) ; duel déplacé de 4-11 pt sans recalage ⇒ R2 suit] (R1) **Part anti-système par électorat (0 fit)**, sur les tirages V2
        T2−7 de §8.38. C'est un mélange orienté, pas le plancher symétrique de
        §8.39 : $q_k = (1-\pi_k)\,q_k^{spatial} + \pi_k\,q^{AS}$, où $q^{AS}$
        vote pour le finaliste « anti-système » (Le Pen contre Macron en
        2017/2022), et symétriquement pour « Aucun » si les données le
        demandent. $\pi_k$ est par FAMILLE (gauche radicale, gauche/écolo,
        droite, droite radicale, centre), pas par candidat, et commune aux deux
        années. On minimise les cellules (Macron, Le Pen, Aucun) 2017+2022.
        Critères proposés (à figer dans la spec avant lecture, ajustables avec
        justification) :
        - partage Mélenchon < 10 pt les deux années ;
        - MAE des autres électorats non dégradée de plus de 1 pt ;
        - $\pi$ stables entre 2017 et 2022 (ajustement séparé par année :
          écart < 0,1) ;
        - duels inchangés à δ recalé près (le mélange déplace $s_{ab}$ : le
          mesurer et le dire).
        Si R1 passe : spécifier son entrée dans le modèle (π par famille estimé
        par les reports, (A)(2) devient possible sans tordre la géométrie) et
        le backtest selon §8.29. Si R1 échoue : le consigner, passer à R2.
        R2 se fait dans les deux cas, en mode réduit (2022 seul) si R1 passe.
  - [x] [FAIT 19/11 14:35Z, §8.44 « Avant le T1 » : sans reports, 2D = 1D sur les reports (Mélenchon 100 %, 0 % dans une chaîne), T1 égal (MAE 2,8-3,0 contre 3,0), lp multimodal (29 contre 4 nats) ⇒ R2 ABANDONNÉ POUR LE SITE ; piste (A) close] [ex-EN COURS (début 2026-11-19 12:20Z) : V2 T1−30 2022 hier2d, 4 chaînes indépendantes (aucun report n'est ≤ as_of) ; témoin 1D hier T1−30 ensuite, mêmes données (aucun hier T1−30 committé)] (revue 19/11, idée 2) **R2 V1 2022 à T1−30 sans reports**, 4 chaînes indépendantes (pas de phase 2), avec δ par notice (point 7) et le plancher de 1 % (point 8) ; critère 4 révisé (§8.44, révision 6) et écarts des duels hypothétiques entre chaînes. Si ν reste au prior et que les duels varient de plus de 2 pt ⇒ R2 abandonné pour le site. Après le témoin.
  - [x] [FAIT 19/11 15:40Z, §8.45 : STABLE — 4 chaînes retenues, écart max 0,90 pt sur les 78 paires (demi-IC90 9-17 pt) ⇒ aucun arbitrage] **(NOUVEAU 19/11, constat de §8.44) Stabilité entre chaînes des duels publiés 2027** : en 1D, Le Pen–Zemmour varie de 7,9 pt entre chaînes à T1−30 2022. Appliquer le critère (écart ≤ 2 pt entre ≥ 3 chaînes indépendantes) au fit de production 2027 (hier) sur toutes les paires publiées, en particulier celles jamais sondées : 0 fit si les chaînes du gel prospectif sont gardées, sinon 1 fit (4 chaînes, environ 30 min). Si des duels publiés dépassent 2 pt : ARBITRAGE À TRANCHER (élargir leur IC, ou ne pas les afficher).
  - [x] [FAIT 19/11 16:15Z, §8.42 : fenêtre T2−13..T2−7 ⇒ médiane 0,185, moyenne 0,206 ; 0,19 confirmé] (revue 19/11, point 9) sensibilité τ_T2 avec la moyenne à T2−7, 0 fit, priorité basse.
  - [x] [FAIT 19/11 12:20Z, §8.44 « Témoin équitable » : 1D avec reports, Mélenchon +17,2, Jadot −8,3, hors fit MAE 8,2 contre 2,6 en 2D ⇒ 2D justifiée en entre-deux-tours] [ex-EN COURS (début 2026-11-19 11:32Z) : témoin équitable `hier_rep` (1D + reports), V2 T2−7 2022, 4 chaînes ; phase 2 ARRÊTÉE à 11:30Z (revue 19/11, point 5 : R̂ bon par construction, ne répond à rien)] [V2 2022 FAIT 19/11 10:35Z, spec §8.44 « Résultats » : critères 1-3 PASSENT (Mélenchon 62,3 contre 63,0 ; hors fit −3,5 pt contre +34,2 en 1D ; σ_v ×1,33 ; T1 non dégradé), 4 ÉCHOUE (R̂ ν 7,5), 5 de peu (s_ab +0,7 pt) ; κ 0,75 → 0,22. À FAIRE : (a) phase 2 §8.8 depuis la chaîne 0 (`--init-depuis second_round_hier2d_v2_2022_T2-7.json`), 4 chaînes ~1 h 30 ; (b) si s_ab > 0,5 pt, retirer le duel M–LP des notices à reports ; puis critères à nouveau, puis V1 2022 si 1-5 passent] [ex-EN COURS (début 2026-11-19 08:15Z) : code fait (drapeaux `carte_2d`, `reports` ; `--variante hier2d` ; tests test_carte_2d), fit V2 T2−7 2022 hier2d lancé (4 processus d'une chaîne, ~4-5 h, données de main au 19/11)] [MATHS FAITES 19/11 07:55Z, spec §8.44 (noyau séparable, ancres ν_C = 0,3 + ordre RN ≥ C, Dirichlet des reports avec c appris, Newton pour δ, critères figés) ; À FAIRE : code derrière drapeau (`carte_2d`, `reports`) + tests, puis fit V2 T2−7 2022 hier2d (4 chaînes × ~4 h)] (R2) **Carte 2D estimée dès le T1** (fit, VM cloud ; jamais sur le
        MacBook). Hypothèse à tester : avec un second axe u (système /
        anti-système), u porte une partie de la séparation entre candidats
        proches en v (Mélenchon / Hamon / Jadot / Roussel), σ_v s'élargit et
        le logit T2 se désature. Maths d'abord, dans la spec (nouveau §) :
        - noyau (séparable ou non, σ_v et λ libres) ;
        - levée des symétries (réflexion de u, ancres : contrainte d'ordre RN ≥
          centre sur u, et une seconde ancre à justifier) ;
        - priors sur ν et λ ;
        - identification : faible par le T1, forte par les reports ⇒ fit
          CONJOINT T1 + duels + reports (vraisemblance des reports de (A)(2) :
          multinomiale ou Dirichlet-multinomiale par électorat avec n effectif,
          dépendance avec les duels de la même notice) ;
        - coût : B_u = 7 ⇒ ≈ 4 h/chaîne, 4 chaînes en parallèle sur 4 cœurs ;
          marquer EN COURS, sous-agent Sonnet + watcher.
        Code derrière un drapeau ÉTEINT (production inchangée, `PUBLIER_T2`
        False). D'abord 2022 seul, V2 T2−7. Critères fixés avant lecture :
        - report Mélenchon→Le Pen à < 10 pt du mesuré ;
        - σ_v a posteriori plus large qu'en 1D (le mécanisme attendu, sinon
          le résultat ne s'explique pas) ;
        - T1 non dégradé (V2 T1 2022 : MAE et couverture contre le 1D) ;
        - duels V1 2022 non dégradés (protocole §8.29) ;
        - ν interprétables et stables entre chaînes (R̂, sélection §8.8).
        Si 2022 passe : 2017, puis le chiffrage de l'effet sur 2027 (P(élu),
        duels), puis « ARBITRAGE À TRANCHER » sur l'adoption, qui toucherait
        aussi le T1 publié. Positions d'experts (CHES, dimension anti-élite) :
        inaccessibles depuis la VM ; si une ancre externe est nécessaire,
        l'écrire en « ARBITRAGE À TRANCHER » (l'utilisateur peut fournir le
        fichier).

- [ ] **PRIORITÉ (orchestrateur, au nom de l'utilisateur, 2026-11-18 ~23:05Z)** —
      LES DONNÉES SONT SUR MAIN depuis 844495b (déjà fusionnée ici via ab2178c) ;
      (A) et τ_T2 2002-2012 sont DÉBLOQUÉS. Roster (fb69009) et palette
      (9b95194) sur main ⇒ fits 2027 débloqués. Ordre :
  - [x] [FAIT 2026-11-19 05:05Z, spec §8.38, `results/reports_t2_A1.json` : CRITÈRE (D) REMPLI (abst et hier, 2017 et 2022) — Mélenchon→Le Pen prédit 0-1 % contre 15 / 21 % mesurés, partage parmi exprimés +23 / +37 pt (z 3,4-7,4) ; reports trop nets partout (cellules 0/100 % contre 5-10 % de transfert croisé) ; « Aucun » par électorat non tenu (erreur 10-60 pt). Fits V2 T2−7 reconstruits depuis les chaînes sauvées sur les données de ee6365b (les données 2017 de main ajoutent Hollande/Pinel ⇒ chaînes incompatibles)] (A)(1) **Validation sans fit des tableaux de reports** :
        `data/parsed/reports_t2_2017.csv`, `reports_t2_2022.csv` (colonnes
        notice, institut, dates, hypothese « X – Y », electorat_t1 = nom complet
        du candidat T1, candidat_t2 ∈ {X, Y, Aucun}, part en %, base_electorat
        souvent vide, rolling ; parseur `sondages/reports.py`). Reports prédits
        par le modèle (tirages par électorat T1, fits existants ou refits Colab
        si les tirages ne sont pas sauvés) contre reports mesurés : MAE par
        électorat et par candidat_t2 (y compris « Aucun » ↔ abstention ω),
        z par cellule. Maths dans la spec (§ nouveau) AVANT le code. Si erreur
        systématique par électorat que l'axe unique n'absorbe pas ⇒ ouvrir (D).
  - [x] [FAIT 19/11 07:00Z, spec §8.41 : (D-b) 2D séparable ÉCHOUE au test d'existence (Mélenchon 2022 reste à 100 % pour λ 0,1/0,2/0,3, même ν libres ; ν non interprétables) ⇒ (D) ABANDONNÉ par la règle fixée avant lecture ; on ne publie que s_ab (inchangé)] [(ii) et test (D-a) FAITS 19/11 05:55Z, spec §8.39-8.40 : plancher ε = 0,2 ⇒ MAE des cellules −3 à −10 pt mais Mélenchon 2022 encore +24-29 pt ; fer à cheval (D-a) ÉCHOUE au test d'identification (Mélenchon 22 → 22, 37 → 37 pt), logit par votant saturé par le noyau gaussien étroit ⇒ À FAIRE : maths complètes de (D-b) 2D séparable (coût, quadrature, priors ν, identification T1) puis test sans fit si possible] **(D) OUVERT par (A)(1)** — maths AVANT (A)(2), qui ne doit pas entrer
        dans le 1D (il tordrait la géométrie, §8.38 point 4) : (i) écrire dans la
        spec la seconde coordonnée (axe système/anti-système, 1 paramètre par
        candidat, prior centré 0, ou « fer à cheval » §2.3) et son identification
        par les reports ; (ii) chiffrer d'abord, sans fit, l'alternative minimale
        « plancher de report » (bruit logistique par votant) sur les mêmes
        tirages : elle couvre la netteté (point 2) mais pas Mélenchon (point 1).
  - [x] [NON ENTREPRIS 19/11 (§8.41) : aucun mécanisme (1D, plancher, fer à cheval, 2D) ne représente le report Mélenchon→Le Pen ; les reports en vraisemblance tordraient la géométrie] (A)(2) [à faire DANS le modèle retenu en (D), pas en 1D] **Reports comme observations** (multinomiale par électorat, ou
        Dirichlet-multinomiale avec n effectif si base_electorat manque ; maths
        d'abord : vraisemblance, dépendance avec les duels de la même notice,
        poids). Backtest V1/V2 2017 et 2022 selon le protocole §8.29
        (10 tâches, CRPS + LS plafonné) ; adopter seulement si non dégradé.
  - [x] [FAIT 19/11 11:20Z avec la revue quotidienne (section « Revue Opus 2026-11-19 »)] **Revue Opus dédiée à (A)** dès (A)(2) fini (sous-agent model opus :
        spec + code + résultats de (A)) ; intégrer ou réfuter point par point.
  - [x] [FAIT 19/11 07:40Z, spec §8.42, `results/tau_t2_historique.json` (0 fit) : médiane a posteriori 0,164 (normale), moyenne 0,184, IC90 [0,10 ; 0,34] ; sans 2002 0,165 ; Student-t4 0,151, moins bon en LOO (1,37 contre 1,70) ; écart à 0,19 < 0,05 ⇒ 0,19 CONFIRMÉ, pas d'arbitrage ; effet interpolé ≈ 1 pt sur P(élu) Le Pen] **τ_T2 calibré sur 2002-2022** (5 élections ; tâche « revue, 9 »
        ci-dessous) : `data/historical/results_{2002,2007,2012}.csv`,
        `data/parsed/intentions_{2002,2007,2012}_wiki.csv` (tour = « Second tour »).
        Erreur de scrutin sondages finaux → résultat ; normale vs Student-t ;
        sensibilité à 2002 (duel atypique) ; effet sur P(élu) 2027 et
        P(Philippe bat Le Pen). Dates 2017 corrigées aussi sur main.
  - [ ] **Crédibilité face aux duels 2027** : test prospectif (tâche gelée
        ci-dessous) à scorer à chaque réveil ; diagnostic 2027 sur le roster
        à 19 (Riner, Polony) si le roster a changé depuis le gel.
  - Tout choix éditorial ⇒ ligne « ARBITRAGE À TRANCHER : … » dans la section
    Arbitrages ET en fin de réponse du réveil (la session locale la relaie à
    l'orchestrateur). Chaque lot finit par « PRÊT À PUBLIER : … ».

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
  - [x] [CLOS 19/11 : (A)(1) fait §8.38, (A)(2) réalisé dans R2 §8.44, abandonné pour le site] [DÉBLOQUÉ 18/11 23:05Z : données sur main (844495b), voir bloc PRIORITÉ en tête] (A) **Tableaux de reports T1→T2 des instituts** — ex-BLOQUÉ DONNÉES
        (demandés à la session pipeline-donnees le 17/11) : dès leur arrivée sur
        main, (1) validation sans fit : reports prédits par le modèle (tirages
        par électorat T1) contre reports mesurés, MAE par électorat ; (2) puis
        comme observations (multinomiale par électorat, maths d'abord).
  - [x] [CLOS 19/11 : D-a et D-b greffés §8.40-8.41 (faux négatifs, revue), R2 §8.44 : abandonné pour le site] [OUVERT 19/11 par (A)(1), voir bloc PRIORITÉ] (D) Espace 2D (axe système/anti-système) — SEULEMENT si (A)(1) montre
        une erreur systématique par électorat que l'axe unique n'absorbe pas.
  - DÉBLOQUÉ 18/11 (844495b) : τ_T2 2002-2012, dates 2017.

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
- [x] [FAIT 19/11, §8.42 : 0,19 confirmé] (revue, 9) τ_T2 : a posteriori sur 2002-2022 (normale vs Student-t),
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

- 2026-11-19 00:08Z (routine) : main fusionnée (bruit de scénarios par candidat) ; test_alertes_prod adapté (mock acceptant `tour=`, comme test_listes) ; aucun duel postérieur au 17/11 (score prospectif vide).
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

- 2026-11-19 04:30-05:10Z (réveil) : CORRECTION — les réveils 00:00-04:30Z
  ont conclu « rien à faire » d'après une mémoire périmée alors que le bloc
  PRIORITÉ (23:05Z) était actionnable. (A)(1) fait (spec §8.38) : critère (D)
  rempli (Mélenchon→Le Pen 0-1 % prédit contre 15/21 % mesuré) ⇒ (D) ouvert,
  (A)(2) déplacé dans le modèle de (D). Tests : 231 passés (hors réseau Wikipédia).

- 2026-11-19 05:10-05:55Z (réveil) : (D)(ii) plancher de report (§8.39) et
  test sans fit du fer à cheval (§8.40) : plancher utile pour la netteté,
  fer à cheval sans effet (logit saturé) ⇒ (D-b) 2D séparable à spécifier.
  Résultats `reports_t2_plancher.json`, `reports_t2_nu.json`. Tests : 5 dans
  test_reports_t2.

- 2026-11-19 06:10-07:05Z (réveil) : (D-b) 2D séparable, maths §8.41 et
  test d'existence sans fit (4 essais parallèles, 40 tirages, w re-résolu) :
  échec (Mélenchon 2022 figé à 100 %), (D) abandonné, (A)(2) non entrepris.
  Revue Opus de (A) groupée avec la revue quotidienne. Suivant : τ_T2 2002-2022.

- 2026-11-19 07:10-07:45Z (réveil) : τ_T2 a posteriori 2002-2022 (§8.42,
  0 fit) : médiane 0,164, IC90 [0,10 ; 0,34], normale meilleure que t4 en LOO ;
  0,19 confirmé. Tests : 2 (test_tau_t2_historique).

- 2026-11-19 07:48-08:00Z (réveil) : piste utilisateur R1 (mélange
  anti-système par famille, §8.43, 0 fit) : échoue (π_GR instable 0,09 →
  0,27, Mélenchon 2022 +13 pt) ; R2 (carte 2D dès le T1, fit) est la
  prochaine tâche : maths d'abord. Tests : test_reports_t2 (7).

- 2026-11-19 07:45-07:55Z (réveil) : maths de R2 (§8.44) ; limite notée : sans
  reports avant le T1, ν n'est appris que par le T1 (critère V1 sans reports).

- 2026-11-19 07:55-08:15Z (réveil, suite) : code R2 derrière drapeaux
  (`geometry` 2D, `reports_cellules` Newton, `arrays_reports`, `ancres_u`,
  `--variante hier2d`), fit de fumée OK (37 lignes de reports ≤ as_of) ;
  fit V2 T2−7 2022 lancé. Production inchangée (drapeaux éteints).

- 2026-11-19 08:15-10:35Z (même tour) : fit V2 T2−7 2022 hier2d (4 × 1 h 30)
  et témoin 1D hier refitté sur les mêmes données (4 × 25 min) ; évaluation
  `eval_r2.py` : reports reproduits (hors fit inclus), T1 non dégradé, mais ν
  instables (R̂ 7,5) et s_ab +0,7 pt ⇒ phase 2 à faire.



## Revue Opus 2026-11-20 — intégration point par point

Revue lancée à 11:20Z, en lecture seule, sur les commits depuis le 19/11 11:20Z.

1. **Candidats fantômes dans la valeur inclusive (haute).** VÉRIFIÉ : à T2−7 2022 sur le cercle,
   `w_now` vaut 3,34 pour Bertrand contre 2,01 pour Macron (chaîne 1), alors qu'aucun sondage ne
   le teste depuis des mois. μ_Montebourg vaut 0,76 dans la chaîne 0 et 0,08 dans la chaîne 1 :
   c'est la vraie source du R̂ de 23. La lecture « un extrême qui passe par l'arrière » de la
   réplication VM est donc RETIRÉE. INTÉGRÉ : correctif §8.49 (`lse_officiel`), préalable à R3c
   et aux reruns. Le même défaut touche potentiellement le fit 2027 (lse sur le roster entier,
   `lse_champ` non adopté, §8.27) : à mesurer après le correctif, avant toute publication du T2.
2. **Critère 1 d'A′2 passé par moyenne de deux modes (86,0 et 48,8).** ACCEPTÉ : désormais, les
   critères de reports se lisent chaîne par chaîne et ne passent que si toutes les chaînes
   retenues passent. Le verdict d'A′2 est à relire par la session locale.
3. **Base trop faible.** ACCEPTÉ : la persistance des tableaux d'avant T1 (MAE 1,8 pt en 2022,
   4,0 en 2017) devient un témoin obligatoire des étapes A′/B′. Un modèle qui ne la bat pas
   n'apporte rien sur les électorats mesurés. Sa valeur ajoutée est ailleurs : électorats et
   paires non mesurés, d'où l'idée 2.
4. **B′2 ne colle pas à ses propres données** (Dupont-Aignan 86 contre 45 ; κ 2017 à 0,06-0,13).
   NOTÉ pour la session locale, auteur de B′2.
5. **R3c, problème de signe** : cos θ vaut −0,6 pour Zemmour et 0,1 pour Mélenchon, et le ω
   uniforme reproduit déjà les « Aucun ». ACCEPTÉ comme pronostic : γ ≈ 0 est probable. Raison de
   plus pour faire passer le correctif §8.49 avant (cf. file).
6. **Sélection des chaînes du joint sur la log-densité d'une seule élection.** NOTÉ pour la
   session locale (code `multi_election`) : choisir sur `potential_energy`.
7. **A′2 et B′2 à B = 25**, malgré la règle §8.48. NOTÉ ; les reruns A′ se feront à B = 100 pour le
   segment.
8. **Protocole.** Ordre des commits correct. Réserves acceptées : B′2 figé une minute après A′2 ;
   la « lecture de plausibilité » affaiblit la règle de parcimonie de §8.47.5, ce qu'il faut dire.
9. **Plateformes et nombre de chaînes.** ACCEPTÉ : comparer A′1 (VM) et A′2 (Mac) mêle deux
   plateformes. Les reruns se feront avec 8 chaînes courtes par variante, sur une même
   plateforme.
10. **Cercle.** ACCEPTÉ : f = 3,5 % ne mesure pas l'arrière. Entre Roussel et Le Pen, l'écart fait
    0,76-0,90 rad (12-14 % de l'électorat). Mélenchon est sur le flanc (θ ≈ −1,47), et le noyau
    le plus large est celui de Macron, pas celui de Le Pen. Le cercle n'utilise pas le mécanisme
    « Le Pen large ».

Idées :

- (1) correctif `lse` puis reruns A′1/A′2 : EN FILE (tâche §8.49) ;
- (2) retirer de la fenêtre les tableaux de l'électorat Mélenchon et le prédire : seule mesure
  d'une valeur ajoutée au-delà des tableaux, à proposer à la session locale (segment, Mac) ;
- (3) fermer le cercle (A = 2π) : seulement si (1) donne l'avantage au cercle.

- 2026-11-20 09:30-22:50Z (routine) : A′1 (invalide, arrêté), revue Opus (candidats fantômes,
  correctif §8.49), A′1 corrigé (environ 10 h par chaîne) : échec aux critères, mais stable et
  plausible en structure. R3c reste suspendu (arbitrage de la session locale).

## Revue Opus 2026-11-19 — intégration point par point

Revue lancée à 11:20Z, en lecture seule. Elle couvre (A)(1), le plancher, le fer à cheval, la 2D
greffée, τ_T2, R1 et R2.

1. **« Désaturation surtout par κ » faux.** INTÉGRÉ et VÉRIFIÉ : contrefactuels rejoués, 1D à
   κ 0,1 donne 98,7 ; 2D sans u, 82,4 ; 2D complet, 62,3. Interaction v-κ-u ; σ_Mélenchon
   0,020 → 0,053 / 0,080 ; Jadot plus à gauche que Mélenchon en 1D. Lecture 2 de §8.44
   remplacée.
2. **§8.40 et §8.41 faux négatifs (κ, σ figés).** INTÉGRÉ : conclusion de §8.41 retirée, règle
   « une porte à 0 fit qui gèle des paramètres couplés ne peut pas exclure ».
3. **Témoin 1D inéquitable.** INTÉGRÉ : `hier_rep` (1D avec reports) lancé à 11:32Z.
4. **Hors échantillon faible.** INTÉGRÉ : persistance naïve à 3,6 pt contre 2,6 pt pour le
   modèle, requalifié en « prolongation à quelques jours ».
5. **μ multimodal aussi ; phase 2 sans valeur.** INTÉGRÉ : phase 2 arrêtée à 11:30Z. Le
   précédent du 14/11 (puits stable mais faux) est rappelé.
6. **ν non identifié, grave pour le site.** INTÉGRÉ : nouveau critère 4 (≥ 3 chaînes
   indépendantes ; écarts ≤ 3 pt sur les reports, ≤ 0,5 pt sur le T1, ≤ 2 pt sur les duels, y
   compris hypothétiques). Le run actuel y échoue : Mélenchon–Le Pen de 43,4 à 48,9.
7. **δ par notice au lieu du repli.** INTÉGRÉ pour le prochain run R2. Le témoin en cours garde
   $s_{ab}$ modélisé, comme le 2D, pour rester comparable.
8. **Code.** Test de Newton en cas saturé AJOUTÉ (vert). Plancher de 1 % : au prochain run.
   Couverture T1 3/12 : notée, non expliquée.
9. **τ_T2 : fenêtre postérieure à `as_of`.** PRÉCISÉ : la dérive après `as_of` est portée par la
   projection OU, pas par τ. Sensibilité « moyenne à T2−7 » mise en file, priorité basse. LOO 5/5
   non informatif : accepté.
10. **Protocole.** ACCEPTÉ : désormais, les critères sont commités dans un commit séparé, avant de
    lancer le script. R2 §8.44 l'a été (31a68b3, avant le code) ; §8.38 et R1 non, à 3 et 13 min
    d'écart.

Idées :

- (1) le témoin équitable, en cours ;
- (2) V1 2022 à T1−30 en R2, sans reports, avec 4 chaînes indépendantes : c'est le seul test de
  la valeur de R2 pour le site, en file juste après (1) ;
- (3) δ par notice et tableau de contrefactuels permanent dans `eval_r2.py` : en file, intégré
  au prochain run.

- 2026-11-19 12:16-14:35Z (même tour) : témoin équitable 1D + reports (2D justifiée
  en entre-deux-tours) ; critère de stabilité révisé en échec à T2−7 ; V2 T1−30 2D
  et 1D : sans reports, R2 n'apporte rien ⇒ abandonné pour le site. Constat
  incident : un duel hypothétique est instable en 1D, nouvelle tâche.

- 2026-11-19 15:08-15:40Z (réveil) : stabilité entre chaînes des duels 2027
  (§8.45, 1 fit en 4 processus) : écart max 0,90 pt, rien à arbitrer.

- 2026-11-19 16:08-16:15Z (réveil) : sensibilité τ_T2 à T2−7 (0 fit), 0,19
  confirmé ; (A) et (D) cochés (clos).

- 2026-11-19 18:24-18:31Z (réveil) : piste utilisateur R3 (cercle) : maths §8.46
  commitées seules, quadrature mesurée (N_θ = 128, coût ~5× le 1D) ; constat
  incident : erreur de quadrature 1D jusqu'à 1,6 pt dans les fits 2022 (σ min
  0,018), nulle en 2027 ; code, tests (247), protocole 1 lancé.

- 2026-11-19 18:31-22:10Z (même tour) : protocole 1 de R3 (4 chaînes d'environ 3 h) ;
  échec strict sur la stabilité, le reste passe ; même verdict que T4 ⇒ arrêt de R3, suite R3b (session locale).

- 2026-11-19 22:08-23:20Z (réveil) : quadrature 1D (§8.48) : sensibilité mesurée,
  garde-fou `sigma_min`, refit B = 100 (σ_Mélenchon 0,02 → 0,03, conclusions
  inchangées) ; R3 coché (clos, suite R3b en session locale).

- 2026-11-19 23:17Z (réveil) : FUSION DE MAIN ABANDONNÉE — conflit sur `.gitignore` en
  fusionnant b6c29d3 (données : reports T1→T2 publiés AVANT le T1, 2017/2022/2027,
  `data/parsed/reports_avant_t1_*.csv`). `git merge --abort` fait, branche inchangée. À
  résoudre par la session locale ou l'orchestrateur. Ces données concernent R3b (session
  locale). Aucun nouveau duel 2027 après le 17/11 : pas de score prospectif.

## Revue Opus 2026-11-21 ~04:00Z (session locale) — code R7 avant le fit

- **1 Dérivation de σ(−ω) fausse** (lse ≫ A_k pour les extrêmes) ; la forme tient avec un choc Gumbel
  persistant. ACCEPTÉ : on passe à ρ = E_G σ(κG − ω), sur 32 quantiles fixes, toujours sans
  paramètre. « ρ_k moyen avec q = 1 » redonnerait A′2 : rejeté, comme le propose la revue.
- **2 ε = 1e-4 coûte environ 850 nats et pousse c.** ACCEPTÉ : ε = 0,02 figé (option (a), sans
  paramètre). La défection apprise (b) n'est pas retenue (parcimonie).
- **3 Détection des lignes propres correcte** (164, aucun homonyme). Pris acte.
- **4 Critères §8.56 illisibles.** CORRIGÉ : `modes.lecture_reports_fit` (c médian, résidus « Aucun »
  propres/autres), contrôle sur A′2 = 14,9 / +5,55 / −1,30. « c ≥ 22 » déclaré non probant.
  `omega_g` ignoré par la branche fidélité : la combinaison avec `abst_situee` est à interdire si
  R3c revient.
- **5 Verdict « corriger puis lancer ».** FAIT : R7 suspendu le temps du correctif (1bf4100), puis
  relancé sur le Mac dès la fin de R4.

## Revue Opus 2026-11-21 ~00:30Z (session locale) — A′1 corrigé, R6, case « Aucun »

- **1 Deux causes distinctes.** ACCEPTÉ.
  · Mélenchon 48,6 et son « Aucun » à 24,6 (contre 43,6) : l'arrière du cercle (37 %) rapproche Le
    Pen. Les abstentionnistes de Mélenchon votent alors Le Pen. Le segment, à valeur inclusive et
    vraisemblance égales, a ce « Aucun » juste.
  · κ = 0,35 n'est pas une cause : c'est la crête κ–σ (netteté effective κ/σ̄² de 67 contre 58).
  · Pécresse : même échec dans les deux géométries, donc pas une question de géométrie.
- **2 R6 : code correct, mais « un paramètre de moins » seulement en apparence.** L'arc peut passer
  dans le prior via une compression des μ. Pathologie aux ancres : Macron et Le Pen sont
  antipodaux, et Mélenchon tombe du côté Le Pen. ACCEPTÉ : prédiction CONTRAIRE à §8.55 figée en
  §8.56 (Mélenchon ≤ 48,6, Zemmour → Mélenchon > 35) ; θ_i et |z_pos| seront rapportés. R6 continue
  sur la VM : la prédiction doit être confrontée.
- **3 « Aucun » et écart à la persistance : d'abord la vraisemblance.** ACCEPTÉ, c'est le point
  majeur. 28 % des lignes sont « propres » (l'électorat k est un finaliste), et le modèle ne peut pas
  y représenter la fidélité. c tombe à 15 et ω est tiré dans deux sens. Correctif à 0 paramètre, la
  FIDÉLITÉ RÉVÉLÉE : spec §8.56 (0495c01), code et test 603f493. Il passe AVANT R5 sur le Mac,
  juste après R4.
- **3 (a) c_r = φ·n_eff : contre-productif (28 % → 44 % de lignes propres).** ACCEPTÉ, abandonné.
- **3 (b) R3c : l'ordre des données est contraire** (Zemmour s'abstient à 13 % près de l'arrière,
  Pécresse à 30 % près du centre). ACCEPTÉ comme argument. R3c reste suspendu ; la décision
  d'abandon revient à l'utilisateur (ARBITRAGE ci-dessous, mis à jour).
- **3 (c) κ_abs distinct.** Gardé en réserve (piste 3), seulement si R7 laisse le « Aucun » de
  Pécresse sous 22.
- **3 (d) « Aucun » = 100 − a − b.** Négligeable (8 groupes) ; reporté.
- **4 Pistes.** Ordre retenu : R7 segment (Mac), puis cercle + R7 (VM ou T4), puis κ_abs si besoin.
  Le témoin « sans lignes propres » n'est pas lancé : R7 garde l'information d'abstention de ces
  lignes.

ARBITRAGE À TRANCHER (mise à jour) : deux revues jugent R3c (abstention située) inutile, et l'une
montre un ordre des données contraire. Proposition : l'abandonner. Décision : utilisateur.

## Revue Opus 2026-11-20 ~20:00Z (session locale) — lecture d'A′2 corrigé, R4/R5, diagnostic de fond

- **1.1-1.2 Lecture d'A′2 juste ; η ne change pas la monotonie (0 % même avec le décalage δ).**
  Pris acte. Nuance consignée : au niveau des électorats, la gauche vote majoritairement Macron
  (Mélenchon 66, Jadot 78) ; c'est le MÉCANISME qui est rejeté (votants v < 0,08 à 100 % Le Pen),
  pas l'agrégat. Monotonie PONDÉRÉE par la masse ajoutée au rapport (§8.54) : 0,296 sur A′2
  corrigé, 30 % des votants à gauche de Macron. Le critère 3b d'origine reste le critère du segment.
- **1.1 `monotonie_gauche` n'a pas de sens sur le cercle.** ACCEPTÉ, CORRIGÉ AVANT LECTURE d'A′1 :
  monotonie circulaire (arc entre Mélenchon et Macron, part mal classée ≤ 10 %), §8.54, `modes.py`.
- **1.3 « Un seul mode » quasi probant** (jumeau T4 c0 identique). Pris acte ; le jumeau T4 complet
  tourne (`_lseo_t4`).
- **1.4 Pécresse ratée EN ÉCHANTILLON** (données d'apprentissage : 62,8 ; modèle : 51,5). Aucun
  artefact. ACCEPTÉ : c'est un désajustement structurel, pas un défaut de généralisation.
- **2.1-2.2 R5 correct mais tautologique sur 3/3b ; prédiction analytique : échec au critère 2
  (Mélenchon ≥ 80).** ACCEPTÉ, prédiction FIGÉE (§8.54). R5 tourne quand même sur le Mac, après R4,
  comme contrôle bon marché de la prédiction.
- **2.3 R4 : le levier « σ_LP large » survit à la limite lointaine ; confusion ν/σ.** ACCEPTÉ :
  corr(log ν, log σ_LP) rapportée, et R4 déclaré « équivalent au gaussien » si q95(ν) ≥ 30 (§8.54).
- **3.1-3.3 Le segment imite un fer à cheval par DEUX noyaux larges (Le Pen ET Mélenchon) ;
  Pécresse et Jadot en sont les dégâts collatéraux.** ACCEPTÉ comme diagnostic principal. Contrôles de
  structure figés : σ_Mélenchon ≤ 0,15 ; électorat LP → Mélenchon ≥ 45 % ; électorat Zemmour →
  Mélenchon ≤ 40 % dans Mélenchon–Macron (§8.54). Mesuré sur A′2 corrigé : 64,1 / 64,2, σ 0,27. Le
  segment ne distingue pas Le Pen de Zemmour.
- **3.4 κ de choix séparé, η par électorat, Le Pen plus centrale : rejetés.** Accord (le deuxième est
  non géométrique et contraire à la parcimonie).
- **3.5 Forme minimale = le cercle (1 paramètre).** ACCEPTÉ, et cohérent avec l'orientation de
  l'utilisateur. Le cercle avec reports d'avant T1 corrigés est AJOUTÉ à la file T4 (8 chaînes,
  `_lseo_t4`, jumeau Linux de l'A′1 de la VM), derrière A′2/R4 T4.
- **4.2 Diagnostic en échantillon sur toutes les cellules d'avant T1 (0 fit).** RETENU, à coder
  (critère : aucune cellule de plus de 50 lignes avec une erreur supérieure à 8 pt).

## Revue Opus 2026-11-20 ~16:15Z (session locale) — correctif §8.50 et code R4

- **1.1 Groupement correct** (2022 : 671 lignes retenues, aucun doublon ni inversion). Pris acte.
- **1.2 Lignes écartées quand « Aucun » est omis (≈ 0 ; 12 lignes en 2022, 6 en 2017, des cases
  de fidélité).** ACCEPTÉ. Correctif (Aucun = 100 − a − b) REPORTÉ au tour suivant pour garder
  les mêmes données entre A′2 corrigé (déjà lancé) et R4. §8.52.
- **1.3 Repli 2027 : Le Pen et Bardella dans le même champ.** ACCEPTÉ, à corriger avant tout fit
  2027 avec reports (champ par ligne tiré de l'hypothèse T1 de la notice).
- **1.4 Autres consommateurs sains.** Pris acte.
- **1.5 Fusion limitée à 4 chaînes ; mauvaise souche par défaut ; B à l'évaluation.** ACCEPTÉ,
  CORRIGÉ : `PRES2027_N_CHAINES`, `model/backtest/modes.py` avec `--stem` explicite, B = 100 à
  l'évaluation.
- **2.1-2.2 Code R4 exact sur le chemin des critères.** Pris acte.
- **2.3 ν oublié dans suivi, reports_t2, reports_r1, reports_2d ; six répartiteurs dupliqués.**
  ACCEPTÉ. Sans effet sur A′/R4. Une méthode unique `fit.A(w)` / `fit.parts(w, mask)` est à faire
  si R4 est adopté.
- **2.4 ν de forme (1,).** Pris acte (cas absent).
- **2.5 ν non exporté.** CORRIGÉ dans `modes.py`.
- **3.1 Mode dominant non codé ; règle ambiguë.** CORRIGÉ : règle précisée (Δσ par rapport à la
  meilleure chaîne), codée, testée sur l'ancienne partition (H = {1, 4, 6, 7} retrouvé ; la
  chaîne 0 tombe avec B par σ).
- **3.2 Volume ½·log det Σ de ±5 nats.** ACCEPTÉ : la masse relative ne se lit plus comme e^ΔL̄
  (§8.52). La phrase « e^8-e^10 fois plus de masse » de la correction d'A′2 est donc trop forte ;
  le mode H reste le plus dense, sans facteur chiffré.
- **3.3 Critères 6-7 mal calibrés.** ACCEPTÉ, remplacés par 6′ et 7′ (parcimonie), §8.52.
- **3.4 Critère direct de monotonie à gauche.** ACCEPTÉ (3b, §8.52).
- **3.5 Témoin « Aucun » (1,75 / 4,8).** ACCEPTÉ (critère « Aucun », §8.52).
- **4 Parcimonie : témoin « σ_LP lié à σ_Macron », un paramètre de MOINS.** RETENU : spécifié et
  codé (maths d'abord) après R4 sur le Mac ; règle de §8.52-7′ étendue : si le gaussien contraint
  passe 1-3b, il prime sur R4.
- **5.4 ν fixé à 4 en repli.** Noté, seulement si 6′ échoue par non-identification.

## Revue Opus 2026-11-20 ~12:15Z (session locale) — intégration point par point

Relecteur Opus en lecture seule ; rapport complet dans la transcription de la session locale.

- **1.1 Reports d'avant T1 brouillés (hypothèses d'une même notice mélangées).** INTÉGRÉ,
  VÉRIFIÉ à la main : 179 groupes sur 196 en 2022 à T1−30. Correctif §8.50 (46740e6). A′1, A′2,
  B′2, les diagnostics de modes et `A2_cercle` sont invalides ; A′2 corrigé est relancé sur le Mac
  (8 chaînes, B = 100, `--lse-officiel`).
- **1.2 Champ des reports commun à toutes les lignes (fantômes).** INTÉGRÉ : champ par ligne,
  §8.50.
- **1.3 Fuites.** Accord : aucune au sens strict. La fuite méthodologique est réelle : la cible
  « Mélenchon 2022 = 63 » a été regardée une dizaine de fois. Réponse : les critères révisés
  (§8.50) sont relatifs à un témoin de persistance calculé avant tout fit, et la seule cible
  vierge restante, les reports d'avant T1 2027 à paraître, est réservée au test final (à figer
  le moment venu).
- **1.4 Critères trop faciles ; couverture du T1 ignorée.** INTÉGRÉ : seuils relatifs à la
  persistance (2,6 / 4,1 pt), partage Mélenchon à moins de 5 pt, couverture IC90 du T1 rapportée.
- **1.5 « Le T1 veut une Le Pen étroite » sur-interprété.** ACCEPTÉ : la tension n'est pas
  démontrée (écart de MAE de 0,1 à 0,2 pt, données de reports fausses). La phrase du diagnostic
  B′2 est retirée comme conclusion ; la décomposition de la log-densité par facteur se fera sur
  les fits corrigés.
- **1.6 Défauts de `multi_election.py`** (variance 1 + s², appariement des putatifs par nom,
  sélection sur une seule élection). ACCEPTÉ, sans urgence : le joint passe en sommeil (2.2).
- **2.1 Arc A = crête, fermer le cercle (A = 2π).** RETENU comme piste 4.5 : un paramètre de moins.
  Le cercle reste prioritaire selon l'orientation utilisateur, mais sous sa forme fermée,
  seulement après le segment corrigé et la Student-t.
- **2.2 Joint en sommeil.** ACCEPTÉ : les reports d'avant T1 apportent l'information pré-T1 que le
  joint devait transporter. Les chaînes A restantes ne tournent que si le GPU revient ; aucun
  nouveau travail sur le joint.
- **2.3 Abstention située γ (R3c) à abandonner.** PARTIELLEMENT ACCEPTÉ : l'argument tient (la
  valeur inclusive fait déjà s'abstenir les électeurs loin des deux finalistes ; un terme en cos θ
  est symétrique). R3c reste SUSPENDU, pas supprimé. C'est une idée de l'utilisateur : la décision
  de l'abandonner lui revient (ARBITRAGE À TRANCHER ci-dessous).
- **2.4 Pondération des reports par l'effectif** (c_r = φ·n_eff). RETENU comme piste, à figer
  avant le prochain fit avec reports.
- **2.5 B = 100.** On garde B = 100 (règle §8.48). Le plancher sur σ serait une contrainte
  supplémentaire.
- **3.1-3.2 La saturation et la multimodalité tiennent à la forme du noyau (modèle mal spécifié).**
  ACCEPTÉ comme diagnostic, à confirmer sur les fits corrigés.
- **3.3 / 4.3 Noyau Student-t (log ν ~ N(log 5 ; 1), un paramètre, emboîte le gaussien).**
  RETENU, PRIORITÉ 1 après A′2 corrigé : c'est la forme minimale qui croise les extrêmes sans
  élargir Le Pen. Maths et critères (§8.50 + « ν < 30 ») à écrire avant code.
- **4.1 Diagnostic à 0 fit** (log-densité par facteur). Reporté sur les fits corrigés, faute de
  sens sur des chaînes ajustées à de fausses données.
- **4.4 Témoin « σ_LP contraint ».** RETENU, après 4.3.

ARBITRAGE À TRANCHER (utilisateur) : R3c (abstention située en cos θ) est suspendu. La revue
pronostique γ ≈ 0, parce que la valeur inclusive produit déjà l'écart Mélenchon/Zemmour. Faut-il
l'abandonner, ou le tester quand même, sur les données corrigées et après la Student-t ?

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
