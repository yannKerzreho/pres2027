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
  Fillon, Dupont-Aignan ; 2022 : Mélenchon, Jadot, Pécresse, Zemmour). Pas de
  2027 : pas de reports observés avant un 1er tour.
- `intentions_2017_wiki.csv` : dates de terrain complétées
  (`python -m sondages.complete_dates`, lignes existantes inchangées hors
  dates) et sondages absents ajoutés en fin de fichier (`--ajoute-absentes`).

