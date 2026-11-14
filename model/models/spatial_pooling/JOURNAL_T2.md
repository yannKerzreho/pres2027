# Journal — recherche second tour (`spatial-pooling`)

Mémoire de la routine cloud horaire « pres2027 — recherche T2 ». Chaque réveil
repart à froid : TOUT l'état utile vit ici. Lire en entier avant d'agir, mettre
à jour avant de finir. Spec : `spec_second_tour.md`. Ne pas réécrire
l'historique, ajouter en bas de « Historique ».

## État (mis à jour à chaque réveil)

- Dernière revue Opus : jamais (relectures Sonnet/Opus du 2026-11-14 faites en
  session locale, intégrées dans la spec).
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
- [ ] Synthèse dans spec §8 : V1 MAE modèle / géométrie seule / BT ; V2 couverture,
      log-score vs sondages ; R-hat, rhat_t2, κ, ω, η ; corrélation κ–η.
- [ ] Puis pistes (choisir selon les résultats, maths dans la spec AVANT le code) :
  - entre-deux-tours conditionné sur le résultat T1 (nœud quasi exact) ;
  - P(élu) à aléa unique (w projeté au T1 -> π(T1) -> prolongé au T2) ;
  - η hiérarchique partagé Le Pen/Bardella (bloc), V1 sur paires sans autre duel ;
  - τ_T2 en leave-one-election-out ;
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
