# Journal — recherche second tour (`spatial-pooling`)

Mémoire de la routine cloud horaire « pres2027 — recherche T2 ». Chaque réveil
repart à froid : TOUT l'état utile vit ici. Lire en entier avant d'agir, mettre
à jour avant de finir. Spec : `spec_second_tour.md`. Ne pas réécrire
l'historique, ajouter en bas de « Historique ».

## État (mis à jour à chaque réveil)

- Dernière revue Opus : 2026-11-14 (routine cloud, cf. « Revue 2026-11-14 »).
- Production : `PUBLIER_T2 = False` (forecast_model.py). Ne PAS passer à True
  sans accord explicite de l'utilisateur.
- Modèle courant : géométrie T1 + κ + valence η agrégée + abstention par valeur
  inclusive (ω). Variante de comparaison : `--variante eta` (sans abstention).
- Versions épinglées (requirements.txt) : jax 0.10.2 / numpyro 0.22.0.
- Lot en attente de publication : `claude/pin-requirements` (204fe6c), vérifié
  (voir Faits établis) ; fusion/push = orchestrateur.

## File de tâches (prendre la première non cochée)

Backtests unitaires (un fit NUTS complet chacun, ~30-40 min sur un cœur ;
lancer 4 tâches en parallèle par réveil). Liste : `python -m model.backtest.second_round --lister`.
Pour chaque ligne, les deux variantes `--variante abst` et `--variante eta`.
Résultats : `model/backtest/results/*.json` (committés).

- [ ] abst : v2 2022 T1-30, T1-1, T2-7
- [ ] abst : v1 2022 Macron|Zemmour, Macron|Mélenchon, Macron|Le Pen, Macron|Pécresse
- [ ] abst : v1 2017 Macron|Le Pen, Fillon|Le Pen ; v2 2017 T1-30 (attendu : sans nœud T1), T1-1, T2-7
- [ ] eta : mêmes tâches (comparaison)
- [ ] Revue 2026-11-14, sans fit : corriger la base V2 (écart-type τ_T2 en plus
      de « largeur du modèle », rapporter les deux) ; post-fit : corr(log κ, η_i),
      résidus standardisés des duels (variance ≈ 1 ?) et corrélation intra-notice.
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

## Faits établis (ne pas re-mesurer)

- Géométrie T1 seule : MAE 6,3 pt sur les 11 paires 2027 ; abstention (κ, ω)
  3,2 ; η seule 2,9 ; les deux 2,4 (une paire retirée, géométrie figée).
- Codex (d7b1575) : bug option extérieure, duel compté deux fois, gate bloquant —
  corrigés (cf. spec §9).
- Données 2017 : la plupart des sondages T1 sans date (note à la session données).
- VM cloud : 4 cœurs, 15 Go ; fit rapide 75-89 s ; installer avec
  `python -m pip install --ignore-installed packaging -r requirements.txt`.
- Réseau de la VM cloud : `fr.wikipedia.org` BLOQUÉ (proxy 403). Pas
  d'ingestion live ni de `model.run` 2027 dans la routine ; le CSV 2027 est
  hors dépôt (.gitignore). Travailler sur 2017/2022 (CSV versionnés). Le test
  `sondages/tests/test_wiki.py::test_build_dataset_live_page_matches_schema_contract`
  y échoue pour cette seule raison : ne pas le « corriger ».
- Vérif versions épinglées (2026-11-14, VM cloud) : install OK, 152 tests passés,
  1 échec réseau ci-dessus. Versions identiques au job quotidien vert du 13/11.
- Colab gratuit : session `t2bt` perdue en cours de run (VM récupérée) le
  2026-11-14 — ne plus en dépendre.

## Historique

- 2026-11-14 (session locale) : audit Codex, spec, implémentation, 2 relectures,
  abstention ajoutée sur idée utilisateur, épinglage, création de la routine.

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
