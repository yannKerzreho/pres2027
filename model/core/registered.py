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

Branche `dev` : en plus du modèle publié, elle porte les modèles expérimentaux.
"""

from model.models import bayesian_nowcast  # noqa: F401  (registre les variantes SSM)
from model.models import gp_pooling  # noqa: F401  (registre gp-pooling)
from model.models import spatial_pooling  # noqa: F401  (registre spatial-pooling)

from model.core.base import all_models  # noqa: F401
