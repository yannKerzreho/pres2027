# pres2027 — Agrégateur bayésien de sondages, présidentielle française 2027

Plan de projet. Premier tour le 18 avril 2027, second tour le 2 mai 2027.

## 1. Inspiration

Le "scientifique américain" est très probablement **Nate Silver** (FiveThirtyEight, puis *Silver Bulletin* depuis son départ en 2023 après la fermeture de FiveThirtyEight par ABC News en 2024). Son approche : agréger les sondages avec des poids par institut (house effects, calculés désormais de façon bayésienne chez lui), pondérer par ancienneté et taille d'échantillon, ajouter des "fondamentaux" (économie, notoriété), puis faire tourner des simulations Monte Carlo pour obtenir des probabilités de victoire plutôt qu'un simple pourcentage.

Deux références plus directement utiles pour la partie statistique :

- **Drew Linzer** (Votamatic, 2012) — *Dynamic Bayesian Forecasting of Presidential Elections in the States*, JASA 2013. Modèle bayésien dynamique combinant un a priori "fondamentaux" et les sondages d'état, avec lissage des tendances communes entre États. Sa prédiction 2012 (332-206) était exacte.
- **G. Elliott Morris** (ex-538, *Strength In Numbers*) — a un simulateur électoral open source sur GitHub (`elliottmorris/toy-us-election-simulator`), bon exemple pédagogique de pipeline sondages → agrégation → simulation.

Et surtout, un précédent **français quasi identique à ton idée existe déjà** : **depuis1958.fr**. Modèle Dirichlet-multinomial explicitement bayésien, adapté au scrutin à deux tours français :

- Chaque sondage est un tirage multinomial `X | θ ~ Multinomial(θ)`.
- La loi a priori conjuguée est un Dirichlet ; la mise à jour après un sondage donne `θ | X ~ Dir(α + x)`.
- Le poids d'un sondage (sa taille effective N) est ajusté selon le temps restant avant le scrutin et la méthode d'échantillonnage (quotas).
- Le second tour est traité par **calcul de probabilité totale** : `P(élu A) = P(θ_A > 0.5 au 1er tour) + Σ_B P(second tour A-B) × P(A gagne A vs B)`, résolu par simulation Monte Carlo (plusieurs millions de tirages) car la loi de Dirichlet en dimension M n'a pas de solution analytique pour ces événements.

C'est presque exactement l'architecture statistique que tu décris. Tu n'as pas besoin de la redécouvrir : le papier de Linzer + la page méthodo de depuis1958.fr te donnent la base théorique complète. Ta valeur ajoutée peut être : rendre ça open source et documenté (contrairement à depuis1958.fr et Silver Bulletin qui ne publient pas leur code), ajouter une vraie modélisation des house effects par institut avec historique, et écrire des articles "data scientist" pour expliquer chaque choix de modélisation.

## 2. Source de données : deux dépôts NSPPolls, deux usages différents

Correction par rapport à la v1 de ce plan : **`github.com/nsppolls/nsppolls` n'est plus maintenu depuis ~2022** (c'était l'agrégateur de la présidentielle 2022). Il reste utile mais uniquement comme **jeu de données historique déjà parsé** (voir §4).

Pour du **live sur 2027**, le dépôt actif est **`nsppolls/sondages-commission-index`** :

- Vit principalement sur **Codeberg** (`codeberg.org/nsppolls/sondages-commission-index`), mirroré automatiquement sur GitHub. Le README demande explicitement d'interagir sur Codeberg, pas sur le mirroir GitHub — à respecter (issues, contact, etc.).
- Un workflow (Forgejo Actions) tourne tous les matins à 06:00 UTC : il scrape l'index du site de la Commission des sondages et télécharge les PDF de notices manquants.
- Fournit `base.csv` (liste des sondages référencés) et `files.csv` (métadonnées + PDF téléchargés), plus un flux RSS.
- **Important : ce dépôt n'extrait pas les intentions de vote.** C'est un index + un miroir de PDF, pas des chiffres structurés. Contrairement à l'ancien `nsppolls/nsppolls`, il n'y a pas de JSON avec les pourcentages par candidat.

Décision révisée :

- **Ne pas re-scraper la Commission des sondages nous-mêmes** — `sondages-commission-index` fait déjà ce travail (index + récupération des PDF), et le dupliquer serait redondant.
- **Construire notre propre étape de parsing PDF → intentions de vote structurées.** C'est un vrai morceau de pipeline à écrire (extraction de tableaux depuis des PDF hétérogènes selon l'institut), qui a une vraie valeur : c'est le chaînon manquant que personne ne maintient publiquement pour 2027 pour l'instant. Bon exercice de dev, pas du travail dupliqué.
- Utiliser `nsppolls/nsppolls` (2017/2022, déjà parsé) comme **données d'entraînement pour la calibration historique** des house effects (§4).

Point d'attention légal/éthique inchangé : notices publiques par la loi, dépôts NSPPolls en MIT avec demande de citation — créditer les deux clairement dans le README, et interagir sur Codeberg pour `sondages-commission-index`.

## 3. Stack retenue

- **Modèle** : Python, PyMC (ou NumPyro si tu veux du JAX/plus rapide) pour l'inférence bayésienne et l'échantillonnage Monte Carlo.
- **Pipeline données** : Python (pandas/polars) — ingestion NSPPolls, nettoyage, normalisation des hypothèses (candidats déclarés vs putatifs), historisation.
- **Sortie** : JSON statique versionné (résultats du modèle : moyennes a posteriori, intervalles de crédibilité, probabilités de victoire, probabilités de duel au second tour).
- **Front** : site statique HTML/JS, graphes avec Plotly.js ou Observable Plot (pas besoin de framework front lourd pour un premier projet).
- **Déploiement** : **GitHub Pages**, via GitHub Actions — un job cron quotidien qui relance le pipeline (ingestion → modèle → export JSON) et republie le site. Zéro coût, zéro serveur à opérer, et c'est un excellent exercice de CI/CD (build, tests, déploiement automatisé) sans la complexité d'une vraie infra applicative.
- Évolution possible plus tard si tu veux te rapprocher d'un poste "data scientist" plus classique : passer à FastAPI + base de données + déploiement continu vers un service comme Render/Fly.io. Pas nécessaire pour le MVP.

## 4. Modèle bayésien — composantes à prévoir

Principe directeur (à retenir, corrige une erreur de la v1 de ce plan) : **ce qu'on veut mesurer, c'est l'écart historique de chaque institut par rapport au résultat réel de l'élection**, pas une confiance "a priori" arbitraire. On a le résultat final des scrutins passés (2007, 2012, 2017, 2022) : c'est un signal directement exploitable pour calibrer le modèle *avant* que la campagne 2027 ne commence, pas quelque chose que le modèle doit "apprendre au fil de l'eau" sans base.

1. **Vraisemblance** : chaque sondage = tirage multinomial (Dirichlet comme conjugué), comme depuis1958.fr.

2. **Calibration historique en deux temps (le vrai cœur du projet) :**
   - **Étape A — hors ligne, une fois (puis ré-exécutée à chaque nouvelle élection) :** à partir des sondages historiques déjà structurés (`nsppolls/nsppolls`, élections 2017 et 2022 ; éventuellement 2012 si on retrouve les données), on construit pour chaque sondage un triplet `(institut, horizon = nb de jours avant le scrutin, écart = intention_publiée − résultat_final_du_candidat)`. On en tire, par institut : un **biais systématique** et une **variance d'erreur en fonction de l'horizon**. C'est la logique des "pollster ratings" de Nate Silver, appliquée aux instituts français.
   - **Le biais n'est pas mesuré par candidat, mais par bloc politique.** Un biais candidat par candidat ne se transfère pas d'une élection à l'autre (les têtes d'affiche changent — Macron ne se représente pas en 2027). Ce qui est structurel et stable, c'est le biais d'un institut envers une *famille politique* (méthodologie d'échantillonnage, redressements appliqués — ex. le redressement historique de certains instituts sur le vote RN pour compenser une sous-déclaration supposée). Il faut donc : (a) une table de mapping `candidat → bloc` (ex. gauche radicale / gauche / centre / droite classique / droite radicale-RN / écologistes) réutilisée à chaque élection et mise à jour pour 2027 dès que les candidatures se précisent, et (b) estimer le biais et la variance de chaque institut *par bloc*, pas par nom. La composante de variance "générale" d'un institut (précision/bruit indépendamment du bloc) reste elle directement transférable telle quelle.
   - **Étape B — modèle hiérarchique, pas un prior neutre :** avec seulement 2 à 4 élections de recul et ~10 instituts actifs, certains instituts auront très peu d'observations historiques. Un prior "faible/neutre" jeté au hasard serait effectivement peu défendable (à raison). La bonne réponse est un **modèle bayésien hiérarchique** : un biais et une variance "population" commune à tous les instituts, et le biais/variance de chaque institut est tiré autour de cette moyenne commune (*partial pooling*). Un institut avec beaucoup d'historique garde son biais propre estimé finement ; un institut avec peu ou pas d'historique (nouveau, ou peu actif) est automatiquement "ramené" (shrinkage) vers la moyenne du groupe plutôt que de recevoir un prior arbitraire. C'est la bonne façon bayésienne de traiter le manque de données, pas un aveu d'impuissance.
   - Ces paramètres (biais + décroissance de variance par horizon) deviennent les **priors informatifs** du modèle live 2027, mis à jour séquentiellement à mesure que les sondages 2027 arrivent.

3. **Pondération temporelle** : découle directement de l'étape A ci-dessus — au lieu d'une décote arbitraire "un sondage vieux de X semaines pèse moins", on utilise la fonction `variance(horizon)` réellement mesurée sur l'historique (qui peut très bien ne pas être monotone : un institut peut être historiquement moins fiable à J-60 qu'à J-10, ou l'inverse).

4. **Correction méthode des quotas** : réduire N effectif pour les sondages par quotas (comme le fait déjà depuis1958.fr) — en pratique cet effet est en grande partie déjà capturé par la variance mesurée à l'étape A si on a assez d'historique par institut/méthode ; sinon décote manuelle en attendant plus de données.

5. **Second tour** : calcul de probabilité totale + simulation Monte Carlo, exactement comme décrit en §1, avec les mêmes biais/variances calibrés par institut appliqués aux hypothèses de duel.

Limite à documenter honnêtement : 2 à 4 élections de recul, c'est peu pour un modèle hiérarchique — les intervalles de crédibilité sur les biais par institut seront larges au démarrage. C'est justement pourquoi le hiérarchique (plutôt qu'un fit indépendant par institut) est le bon choix : il ne prétend pas à une précision que les données ne permettent pas.

## 5. Pipeline de données — étapes

1. `ingest/` : récupération quotidienne de `base.csv` / `files.csv` depuis `sondages-commission-index` (Codeberg, mirroré GitHub) — liste des notices + PDF déjà téléchargés par leur pipeline.
2. `parse/` : **étape à écrire nous-mêmes** — extraction des intentions de vote depuis les PDF de notices (tableaux hétérogènes selon l'institut ; probablement `pdfplumber`/`camelot` + règles par institut, avec fallback manuel pour les cas non parsables). C'est le chaînon qui manque publiquement pour 2027.
3. `normalize/` : parsing des hypothèses (chaque sondage teste souvent plusieurs hypothèses de candidatures), mapping candidat → identifiant stable **et → bloc politique** (table `candidat_blocs.csv`, tenue à jour élection par élection), gestion des non-réponses/indécis.
4. `calibrate/` : (offline, sur `nsppolls/nsppolls` 2017+2022, via la table de blocs) fit du modèle hiérarchique de biais/variance par institut **× bloc** et par horizon — voir §4. Produit les priors informatifs consommés par le modèle live.
5. `store/` : stockage versionné (simple : fichiers Parquet/CSV commités ou artefacts CI ; pas besoin de vraie base de données pour un site statique).
6. `model/` : ingestion → mise à jour bayésienne (avec priors calibrés) → simulation Monte Carlo → export des résultats (moyennes, intervalles de crédibilité à 90 %, probabilités de victoire, probabilité de chaque duel au 2nd tour).
7. `publish/` : génération du JSON consommé par le front, déploiement GitHub Pages.

Chaque étape = un job séparé dans le workflow GitHub Actions, avec tests unitaires (parsing PDF sur un jeu de fixtures, cohérence des probabilités qui somment à 1, non-régression du modèle sur les données 2022 comme "backtest").

## 6. Squelette de repo proposé

```
pres2027/
  .github/workflows/
    ci.yml           # lint + tests sur chaque PR
    daily-update.yml # cron quotidien : ingestion -> parse -> modèle -> déploiement Pages
  data/
    historical/      # 2017/2022 depuis nsppolls/nsppolls (déjà parsé) - pour calibration
    raw/              # notices + PDF depuis sondages-commission-index
    parsed/           # sorties du parsing PDF, normalisées
  calibration/
    fit_house_effects.py  # modèle hiérarchique biais/variance par institut, sur historique
    priors.json            # sortie : priors informatifs par institut, versionnés
  model/
    bayesian_model.py
    simulate.py
    backtest/        # validation sur 2017 et 2022
  pipeline/
    ingest.py
    parse_pdf.py      # extraction intentions de vote depuis les notices PDF
    normalize.py
    export.py
  site/              # front statique (HTML/JS/Plotly)
  tests/
    fixtures/pdf/     # échantillons de notices pour tester le parsing
  docs/              # articles "data scientist" expliquant le modèle
  README.md
  LICENSE            # MIT, cohérent avec NSPPolls
```

## 7. Roadmap suggérée

**Phase 0 — Setup (1er sprint)**
Repo GitHub, licence, README, récupération des données historiques 2017/2022 (`nsppolls/nsppolls`), premier notebook exploratoire (backtest visuel : écart sondage/résultat par institut).

**Phase 1 — Calibration historique (avant le modèle live, pas après)**
Fit du modèle hiérarchique de biais/variance par institut et par horizon sur 2017+2022 (§4). C'est la fondation : le modèle live en Phase 2 en dépend. Documenter les limites (peu d'élections de recul) dès cette phase.

**Phase 2 — Parsing PDF + ingestion live**
Connexion à `sondages-commission-index`, écriture du parseur PDF → intentions de vote structurées, tests sur fixtures.

**Phase 3 — MVP modèle live**
Dirichlet-multinomial + priors calibrés en Phase 1 + simulation Monte Carlo pour le second tour. Export JSON, front minimal (courbes de tendance + probabilités).

**Phase 4 — CI/CD**
Workflow de tests sur PR, workflow cron quotidien (ingestion → parse → modèle → déploiement Pages). C'est le cœur de l'objectif "première expérience de dev" — à ne pas bâcler.

**Phase 5 — Contenu "data scientist"**
Articles dans `docs/` expliquant chaque choix (façon Silver Bulletin ou depuis1958.fr) : comment lire les probabilités, comment la calibration historique a été faite, limites du modèle.

**Phase 6 (optionnelle)** — réintégration de 2012 si les données se retrouvent, modèle dynamique façon Linzer (marche aléatoire temporelle plutôt que décote), passage à une vraie API si tu veux aller vers une stack applicative complète.

## 8. Points de vigilance

- **Communiquer les probabilités comme des probabilités**, pas des certitudes — c'est la principale critique récurrente faite aux modèles à la Nate Silver (mauvaise lecture du public d'un "70 % de chances" comme une prédiction ferme).
- **Créditer les deux dépôts NSPPolls et la Commission des sondages** clairement (licence MIT, obligation de citer les instituts et leurs notices). Pour `sondages-commission-index`, interagir sur Codeberg (pas sur le mirroir GitHub) si besoin d'ouvrir une issue ou de contacter les mainteneurs.
- **Peu de données historiques françaises** (2 à 4 scrutins présidentiels selon jusqu'où on remonte) pour calibrer un modèle hiérarchique — c'est justement pourquoi le hiérarchique (partial pooling) est retenu plutôt qu'un fit indépendant par institut : il absorbe cette faiblesse au lieu de la masquer. Rester transparent sur la largeur des intervalles de crédibilité qui en résulte.
- Le parsing PDF des notices est le risque technique principal du projet (formats hétérogènes par institut, changements possibles dans le temps) — prévoir du temps ici, et un chemin de secours (saisie manuelle ponctuelle) pour les cas non parsables plutôt que de bloquer tout le pipeline.
- Le calendrier est fixé : 18 avril / 2 mai 2027, donc la fenêtre de calibration/backtest est claire dès maintenant.
