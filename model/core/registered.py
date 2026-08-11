"""Importer ce module peuple le registre avec tous les modèles disponibles.

Un contributeur ajoute son modèle en créant `model/model_xxx.py` (sous-classe
`ForecastModel` décorée `@register`) et en l'important ici. Le runner, le
backfill, le manifeste `models.json` et les tests le prennent alors automatiquement.
"""

from model.models import bayesian_nowcast  # noqa: F401  (registre bayesian-nowcast + -no-house-effects)
from model.models import linear_pooling  # noqa: F401  (registre linear-pooling)

from model.core.base import all_models  # noqa: F401
