# Données historiques

Données servant à la calibration hors ligne des *house effects* (Phase 1).

## Fichiers

| Fichier | Contenu | Source |
|---------|---------|--------|
| `presidentielle_2022_nsppolls.csv` | Sondages 1er/2nd tour du cycle 2022 | [`nsppolls/nsppolls`](https://github.com/nsppolls/nsppolls) (`presidentielle.csv`), MIT |
| `presidentielle_2017_pollsposition.csv` | Sondages 1er/2nd tour 2017 (converti au même schéma par `pipeline/ingest_2017.py`) | [`pollsposition/data`](https://github.com/pollsposition/data) |
| `results_2022.csv` | Résultats officiels finaux 2022 (% exprimés) | Ministère de l'Intérieur |
| `results_2017.csv` | Résultats officiels finaux 2017 (% exprimés) | Conseil constitutionnel (via pollsposition) |
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
