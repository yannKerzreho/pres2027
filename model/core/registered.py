"""Importer ce module peuple le registre avec tous les modèles disponibles.

Un contributeur ajoute son modèle en créant `model/models/mon_modele/`
(sous-classe `ForecastModel` décorée `@register`) et en l'important ici. Le
runner, le backfill, le manifeste `models.json` et les tests le prennent alors
automatiquement.

Être enregistré ≠ être exposé : `ForecastModel.public` décide de la présence
dans le sélecteur du site, et `python -m model.run` ne lance par défaut que les
modèles publics.

Branche `dev` : en plus du modèle publié, elle porte les modèles expérimentaux.
`spatial_pooling` n'est volontairement PAS importé ici — il n'est pas encore un
`ForecastModel` enregistré (décision documentée dans son `__init__.py`).
"""

from model.models import bayesian_nowcast  # noqa: F401  (registre les variantes SSM)
from model.models import gp_pooling  # noqa: F401  (registre gp-pooling)

from model.core.base import all_models  # noqa: F401
