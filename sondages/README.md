# sondages — parseur de notices → intentions structurées

Module **autonome et réutilisable** : transforme les sondages de la présidentielle
française en un jeu **structuré au niveau candidat**, avec traçabilité vers la
notice source (Commission des sondages). C'est le chaînon public manquant pour la
présidentielle 2027.

Conçu pour être extrait un jour en repo/paquet indépendant : dépendances
minimales, aucune dépendance de modélisation, schéma de sortie stable
(`schema.py`), tests + fixtures embarqués.

## Deux sources, un seul schéma

**`wiki.py` — source primaire**, en production pour 2027/2022/2017 : parse les
pages Wikipedia « Liste de sondages sur l'élection présidentielle française de
AAAA », qui font déjà, notice par notice, le travail de lecture pour tous les
instituts (pas seulement les 4 outillés par le parseur PDF). Chaque cellule
« Sondeur » lie vers le PDF de la notice Commission des sondages, donc la
traçabilité est préservée. Voir le docstring de `wiki.py` pour le détail des
pièges de parsing gérés.

```bash
python -m sondages.wiki --page "Liste de sondages sur l'élection présidentielle française de 2027" \
                         --out data/parsed/intentions_2027_wiki.csv
```

**`build.py`/`parse_pdf.py` — pipeline PDF, en spot-check seulement** : n'a plus
besoin de couvrir tous les instituts en production ; utile pour vérifier
ponctuellement une notice contre le rendu Wikipedia (91/108 valeurs identiques à
±0,6 pt sur un spot-check 2022, le reste à ≤1,5 pt d'arrondi).

```python
from sondages import build_dataset, validate

intentions, statuts = build_dataset(since_year=2025)   # 1 ligne = candidat × hypothèse × sondage
validate(intentions)                                    # garde-fou du contrat
```

```bash
python -m sondages.build --since 2025      # -> data/parsed/intentions_2027.csv (spot-check) + couverture
python -m sondages.audit --since 2025      # audit du parsing PDF : recall + notices à corriger
```

2017/2022 sont des exports statiques (élections closes, pas de re-scraping
nécessaire) ; seul 2027 est ré-ingéré par le job quotidien.

**Dates manquantes (`date_recovery.py`)** : le texte de cellule Wikipedia omet
souvent l'année (donnée par le contexte de page, cf. `wiki.py`) ; sur 2017,
une partie reste irrécupérable par le texte seul (section « avant publication
de la liste officielle des candidats », ~18 mois sans année nulle part dans
le wikitext). Pour ce reliquat, `date_recovery.py` cherche une correspondance
**vérifiée** dans l'index NSPPolls : une notice PDF candidate n'est acceptée
que si elle reproduit les mêmes intentions candidat par candidat que la ligne
Wikipedia (±1 pt) — jamais une date devinée par proximité. Utilise le fallback
LLM (`extract_llm.extract_full_with_llm`) pour les instituts sans parseur par
règles. Résultat sur 2017 : 24 % → 37 % de dates résolues (plafond structurel :
l'archive NSPPolls ne remonte pas avant 2016 et sa couverture y est clairsemée
pour plusieurs instituts — pas un bug de parsing).

```bash
python -m sondages.date_recovery --csv data/parsed/intentions_2017_wiki.csv --year 2017
```

## Schéma de sortie (contrat stable)

Une ligne = un candidat × une hypothèse × un sondage. Colonnes (cf.
[`schema.py`](schema.py)) : `notice_id, notice, notice_url, institut,
commanditaire, date_debut, date_fin, date_notice, echantillon, methode, tour,
hypothese, candidat, intention`.

**Hors périmètre** (volontairement) : l'agrégation en « blocs » ou « slots » de
candidature est une décision de *modélisation*, pas de parsing — elle vit côté
modèle. Ce module fournit les intentions **brutes et riches**.

## Robustesse & vérifiabilité (pipeline PDF, spot-check)

Les pièges propres au parsing Wikipedia (colonnes qui bougent, templates de
style, cellules fusionnées, lignes de rappel de résultat...) et leurs
garde-fous sont documentés dans le docstring de `wiki.py`, pas ici.

- **Parsers par institut** (Ifop, Odoxa, Ipsos, Harris/Toluna), avec un fallback
  qui signale `unsupported_institut` au lieu de casser.
- **Garde-fous** : on ne retient une hypothèse que si les intentions **somment à
  ~100 %** ; les tables de rappel de vote passé (2022/2024), baromètres d'opinion
  et croisements démographiques sont écartés.
- **Filtre représentativité** : les sondages sur un **sous-électorat** (personnes
  LGBT, sympathisant·es d'un parti, primaire, « bloc central »…) somment aussi à
  100 % mais ne sont pas représentatifs de l'ensemble des inscrits — ils sont
  **écartés** (statut `biased_subsample`), jamais publiés dans le dataset. Le
  garde-fou est un filtre de mots-clés sur l'intitulé (cf. `_is_biased_subsample`) ;
  le fallback LLM le double d'un contrôle sémantique (`is_representative`).
- **Audit** (`python -m sondages.audit`) : compare, pour chaque notice, un
  détecteur indépendant (« cette notice contient-elle des intentions 2027 ? ») au
  résultat du parser, et chiffre le **recall** par institut + liste les ratés.
- **Tests + fixtures** (`sondages/tests/`) : notices réelles de référence.

## Fallback LLM (optionnel)

Pour les notices que les règles ne savent pas parser (layouts exotiques, nouveaux
instituts), un **fallback par LLM** (`extract_llm.py`) peut prendre le relais : un
simple appel structuré (Haiku par défaut, cheap) qui renvoie le même schéma, **puis
repasse par le garde-fou somme ~100 %** — une hallucination est rejetée, pas publiée.
Sert deux usages : le spot-check PDF (`extract_with_llm`) et `date_recovery.py`
(`extract_full_with_llm`, qui renvoie aussi date/échantillon pour vérifier une
correspondance Wikipedia ↔ notice candidate).

Désactivé par défaut. Pour l'activer :

```bash
pip install -r sondages/requirements-llm.txt   # dépendance optionnelle : anthropic
export ANTHROPIC_API_KEY=...                     # jamais dans le repo (cf. plus bas)
SONDAGES_USE_LLM=1 python -m sondages.build --since 2025
```

Le cœur du module reste **sans dépendance IA** : le rule-parser marche seul, le LLM
n'est appelé que quand les règles ne sortent rien **et** que la notice ressemble à
une notice d'intentions (pré-check bon marché `_looks_like_intentions_pdf` :
« intention » + « suffrages exprimés » + un marqueur de tour) — pour ne pas dépenser
un appel payant sur un baromètre d'opinion.

En plus des intentions, l'appel LLM renvoie `is_representative` : il **vérifie que
le sondage porte sur l'ensemble des inscrits** (et non un sous-électorat), et écarte
le cas échéant. C'est la version « sémantique » du filtre de mots-clés ci-dessus —
on pourrait, à terme, faire passer *toute* notice par cette vérification (au prix
d'un appel LLM par notice) pour valider représentativité + échantillon + méthode ;
aujourd'hui c'est un **fallback** (règles d'abord, LLM en secours) pour rester cheap.

**Clé API — gestion sûre :**
- **Local** : `export ANTHROPIC_API_KEY=...` dans le shell, ou un fichier `.env`
  (gitignoré). Jamais en dur dans le code, jamais commité.
- **GitHub Actions** : Settings → Secrets and variables → Actions → New secret
  `ANTHROPIC_API_KEY`, puis dans le workflow :
  `env: { ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }} }`.
- Le code lit `os.environ["ANTHROPIC_API_KEY"]` — la clé n'apparaît jamais en clair.

## Sources & crédits

- **Primaire** : pages Wikipedia « Liste de sondages sur l'élection
  présidentielle française de AAAA » (licence CC BY-SA) — les contributeurs y
  font déjà, sondage par sondage, le travail de lecture des notices ; chaque
  cellule institut lie vers le PDF de la notice Commission des sondages
  d'origine, ce qui préserve la traçabilité du contrat `schema.py`.
- **Spot-check / repli** : notices publiques de la Commission des sondages,
  indexées par [NSPPolls](https://codeberg.org/nsppolls/sondages-commission-index)
  (MIT — interagir sur Codeberg, pas sur le mirroir GitHub). Utilisé par
  `build.py`/`parse_pdf.py` (spot-check PDF) et `date_recovery.py`
  (récupération vérifiée des dates manquantes).
