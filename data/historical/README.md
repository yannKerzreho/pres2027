# Données historiques

Données servant à la calibration hors ligne des *house effects* (Phase 1).

## Fichiers

| Fichier | Contenu | Source |
|---------|---------|--------|
| `presidentielle_2022_nsppolls.csv` | Sondages 1er/2nd tour du cycle 2022 | [`nsppolls/nsppolls`](https://github.com/nsppolls/nsppolls) (`presidentielle.csv`), MIT |
| `presidentielle_2017_pollsposition.csv` | Sondages 1er/2nd tour 2017 (converti au même schéma par `pipeline/ingest_2017.py`) | [`pollsposition/data`](https://github.com/pollsposition/data) |
| `results_2022.csv` | Résultats officiels finaux 2022 (% exprimés) | Ministère de l'Intérieur |
| `results_2017.csv` | Résultats officiels finaux 2017 (% exprimés) | Conseil constitutionnel (via pollsposition) |
| `results_2002.csv`, `results_2007.csv`, `results_2012.csv` | Résultats officiels T1/T2 (% exprimés), générés par `python -m pipeline.results_anciens` | Conseil constitutionnel (voix, via `sources/resultats_pollsposition.json`) |
| `candidat_blocs.csv` | Mapping `candidat → bloc politique`, par élection | Construit ici (à maintenir élection par élection, §4 du plan) |
| `sources/` | Fichiers bruts pollsposition (JSON) avant conversion | pollsposition/data |

## Blocs politiques

Taxonomie stable réutilisée d'une élection à l'autre (§4 du plan) :

`gauche_radicale`, `gauche`, `ecologistes`, `centre`, `droite`,
`droite_radicale`, `divers`.

Le biais des instituts est estimé **par bloc** (et non par candidat) car les
têtes d'affiche changent d'une élection à l'autre, alors que le biais
méthodologique d'un institut envers une *famille* politique est plus structurel
(ex. redressement historique du vote RN).

## Couverture des élections

- **2022** (`nsppolls`) : cycle complet, ~350 sondages, dates 2020→2022 (horizons
  jusqu'à ~660 jours) → informe biais **et** dérive longue.
- **2017** (`pollsposition`) : 59 sondages, mais uniquement les **~5 dernières
  semaines** avant le 1er tour (16 mars → 5 mai 2017, horizons ≤ 38 jours) →
  informe surtout le **biais près du scrutin**, peu la dérive longue. Le champ
  `date_fin` du JSON source est corrompu (année 2021) ; `ingest_2017.py` utilise
  la date encodée dans la clé du sondage, fiable.

Deux élections restent peu pour un modèle hiérarchique — d'où le *partial
pooling*, qui absorbe cette faiblesse au lieu de la masquer.

**À faire** pour renforcer : retrouver les sondages **2012** (pollsposition n'a
que les résultats 2012, pas les sondages). Ajouter alors le fichier ici, une
entrée dans `pipeline/historical.py` (`ELECTION_DATES`, `POLL_FILES`,
`RESULT_FILES`), le mapping des blocs 2012 et les résultats officiels. Le code de
calibration empile déjà plusieurs élections sans modification.

## Résultats officiels 2022 (rappel)

1er tour (% exprimés) : Macron 27,85 · Le Pen 23,15 · Mélenchon 21,95 ·
Zemmour 7,07 · Pécresse 4,78 · Jadot 4,63 · Lassalle 3,13 · Roussel 2,28 ·
Dupont-Aignan 2,06 · Hidalgo 1,75 · Poutou 0,77 · Arthaud 0,56.

2nd tour : Macron 58,55 · Le Pen 41,45.

## Ajouts du 2026-11-18 (session données, pour la recherche T2)

Dans `data/parsed/` :

- `intentions_2002_wiki.csv`, `intentions_2007_wiki.csv`, `intentions_2012_wiki.csv` :
  pages Wikipedia de ces élections, même schéma (`sondages.schema`), T1 et T2.
  `python -m sondages.wiki --page "Liste de sondages sur l'élection présidentielle française de 2012" --url-page-repli --out …`.
  Limites : pas de lien de notice (`notice_url` = URL de la page Wikipedia),
  pas d'échantillon, une seule date par sondage (copiée en `date_debut` et
  `date_fin` ; souvent la date de publication). Seconds tours : 29 / 121 / 133
  sondages, dont 6 / 18 / 11 réalisés à partir du soir du 1er tour (y compris
  les enquêtes du soir même). Non branchés dans
  `pipeline/historical.py` (`ELECTION_DATES`, `POLL_FILES`, `RESULT_FILES`) :
  c'est un choix de calibration.
- `reports_t2_2017.csv`, `reports_t2_2022.csv` : reports T1 -> T2 publiés par
  les instituts (section « Reports de voix » de Wikipedia, `python -m sondages.reports --annee 2017`).
  Une ligne par (sondage, électorat T1, option T2), `part` en % de l'électorat
  T1, options Macron / Aucun (abstention, blanc, nul) / Le Pen. Wikipedia ne
  relève que les 4 principaux électorats éliminés (2017 : Mélenchon, Hamon,
  Fillon, Dupont-Aignan ; 2022 : Mélenchon, Jadot, Pécresse, Zemmour). Tous
  datés ENTRE les deux tours (vote T1 déclaré) ; pour la période avant le T1,
  voir `reports_avant_t1_*`.
- `reports_avant_t1_2017.csv`, `_2022`, `_2027` : duels de 2nd tour croisés par
  INTENTION de vote au 1er tour, publiés avant le 1er tour, lus dans les
  rapports PDF des instituts cités par les lignes de 2nd tour de
  `intentions_{annee}_wiki.csv` (`python -m sondages.reports_avant_t1 --annee 2022 --telecharge` ;
  cache PDF dans `data/raw/pdf_reports/`, hors dépôt, repli web.archive.org).
  Schéma de `reports_t2_*` sans `rolling`, plus `nature_t1` (« intention » ;
  `reports_t2_*` relève le « vote » T1 déclaré) et `page` (page du PDF).
  `electorat_t1 = "Aucun"` : abstentionnistes / blancs du T1, quand l'institut
  les croise ; pas de ligne « Ensemble ». Base des parts : quand un duel n'a
  pas d’option « Aucun » (BVA 2022, Ifop 2019, la moitié des duels Opinion Way 2017), les parts
  sont en % des exprimés de l'électorat ; sinon en % de l'électorat entier.
  Couverture (sources de duels / PDF obtenus / sources extraites) : 2017
  84 / 50 / 20 (Ifop 12, Opinion Way 6, Harris 2), 2022 145 / 100 / 40 (Harris
  21, Elabe 9, BVA 5, Ipsos 3, Ifop 2), 2027 34 / 28 / 8 (Harris seul).
  Absents par nature : pages HTML (Odoxa, BVA, Ipsos, TNS Sofres), notices
  sans tableau croisé (Opinion Way 2022, Ifop 2027, Elabe 2027…), tris croisés
  Ifop 2017-2021 qui croisent le VOTE de l'élection précédente (écartés
  explicitement). Non lu : panneaux doubles d'Elabe 2020-2021 (deux
  hypothèses de T1 côte à côte, duel hors du titre).
  Contrôles : chaque électorat somme dans [97, 103] (sinon écarté) ; un
  finaliste garde >= 75 % de son propre électorat (test) ; le T2 reconstruit
  (parts T1 du sondage x reports) retrouve le T2 publié à 0,6 point près en
  médiane (max 4,4 en 2022, 4,1 en 2027, sur les électorats couverts).
- `intentions_2017_wiki.csv` : dates de terrain complétées
  (`python -m sondages.complete_dates`, lignes existantes inchangées hors
  dates) et sondages absents ajoutés en fin de fichier (`--ajoute-absentes`).

- `intentions_2022_wiki.csv` : deux sondages de fin décembre 2021 étaient datés
  2022-12-31 (horizon négatif en calibration), corrigés en 2021-12-31
  (`python -m sondages.complete_dates --corrige-apres-scrutin`) ; OpinionWay
  « 28 février - mars 2022 » daté 28 février - 1er mars.
