"""Importer ce module peuple le registre avec tous les modèles disponibles.

Un contributeur ajoute son modèle en créant `model/models/mon_modele/`
(sous-classe `ForecastModel` décorée `@register`) et en l'important ici. Le
runner, le backfill, le manifeste `models.json` et les tests le prennent alors
automatiquement.

Être enregistré ≠ être exposé : `ForecastModel.public` décide de la présence
dans le sélecteur du site, et `python -m model.run` ne lance par défaut que les
modèles publics (les variantes de comparaison restent lançables à la main via
`--model <id>`).
"""

from model.models import bayesian_nowcast  # noqa: F401  (registre bayesian-nowcast + variantes)
from model.models import linear_pooling  # noqa: F401  (registre linear-pooling)

from model.core.base import all_models  # noqa: F401
