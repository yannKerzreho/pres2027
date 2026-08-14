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
absent de ce fichier : il dépend de dynamax, qui n'est pas dans les requirements
de cette branche. L'importer ici planterait le registre entier — donc la CI et
le job quotidien — pour un modèle que le site n'affiche pas. Il reste lançable
sur `dev`, où dynamax est installé.

⚠️ Divergence assumée entre les branches : ce fichier est le seul qui diffère
structurellement entre `main` et `dev` (avec `requirements.txt`). Chaque fusion
`dev` -> `main` le remettra en conflit ; la résolution est toujours la même,
garder la version de la branche d'arrivée.
"""

from model.models import gp_pooling  # noqa: F401  (registre gp-pooling)
from model.models import spatial_pooling  # noqa: F401  (registre spatial-pooling)

from model.core.base import all_models  # noqa: F401
