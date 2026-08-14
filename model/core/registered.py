"""Importer ce module peuple le registre avec tous les modèles disponibles.

Un contributeur ajoute son modèle en créant `model/models/mon_modele/`
(sous-classe `ForecastModel` décorée `@register`) et en l'important ici. Le
runner, le backfill, le manifeste `models.json` et les tests le prennent alors
automatiquement.

Être enregistré ≠ être exposé : `ForecastModel.surface` décide de la RUBRIQUE du
site où le modèle apparaît — « suivi » (le sélecteur de l'onglet Suivi des
intentions, la voie codifiée et la valeur par défaut), « scenarios » (onglet
Scénarios, qui demande en plus un artefact sur mesure), ou `None` (hors du site).
`python -m model.run` ne lance par défaut que les modèles rattachés à une
rubrique.

Branche `main` : `gp-pooling` (rubrique « suivi ») et `spatial-pooling`
(rubrique « scénarios ») sont enregistrés et tournent chaque jour.
`model/models/bayesian_nowcast/` est présent dans l'arbre mais VOLONTAIREMENT
absent de ce fichier. C'est un CHOIX DE PUBLICATION, plus une contrainte
technique : jusqu'au 2026-08-14 ce modèle dépendait de dynamax, absent des
requirements, et l'importer ici aurait fait tomber le registre entier — donc la
CI et le job quotidien. Depuis que le filtre de Kalman vit dans
`model/core/lgssm.py`, l'import ne casserait plus rien ; l'enregistrer ici
suffirait à le faire tourner chaque jour. Il reste dehors parce que le site ne
l'affiche pas et qu'un run NUTS quotidien que personne ne lit allonge le cron
pour rien.

⚠️ Divergence assumée entre les branches : ce fichier est le seul qui diffère
structurellement entre `main` et `dev`. Chaque fusion `dev` -> `main` le remettra
en conflit ; la résolution est toujours la même, garder la version de la branche
d'arrivée.
"""

from model.models import gp_pooling  # noqa: F401  (registre gp-pooling)
from model.models import spatial_pooling  # noqa: F401  (registre spatial-pooling)

from model.core.base import all_models  # noqa: F401
