"""Modèle spatial (Hotelling-Downs discrétisé) — reports de voix cohérents et
scénarios de listes de candidats personnalisables.

Chaque candidat occupe une position sur un axe latent gauche-droite et recrute
par un softmax masqué sur cet axe. Retirer un candidat d'un champ redistribue
donc ses électeurs vers ses VOISINS, par construction et non par une matrice de
reports posée à la main. C'est la promesse du modèle, et la seule chose qu'on
mesure pour l'accepter : hors échantillon sur 2022, sur les paires où le report
n'est PAS proportionnel, 0,73 pt d'erreur contre 1,23 pt pour le proportionnel,
et 96 % des paires gagnées.

Domaine d'emploi : l'exploration d'HYPOTHÈSES. L'information géométrique ne
vient que de la variation du champ testé d'un sondage à l'autre — un champ
unique ne contraint rien, `w` absorbant tout. Le modèle sert donc tant que les
instituts testent des listes différentes.

Voir spec_spatial_pooling.md pour les maths, et `historic/` à la racine pour
le journal de développement.
"""

from .calibration import (  # noqa: F401
    BANK_EXCESS_PATH, SpatialExcessCalibration, excess_sigma_spatial, main, matched_pairs,
)
from .forecast_model import SpatialPooling  # noqa: F401
from .fit import (  # noqa: F401
    SpatialPoolingFit, fit_spatial_pooling, forecast_spatial_pooling, pi_draws_for_mask,
    summarize_pi,
)
from .geometry import B, V, W, ancres_ecartees, ou_kl_basis, spatial_shares  # noqa: F401
from .joint_model import (  # noqa: F401
    LEVEL_SCALE, MIN_GAP_ANCRES, POS_SCALE, SEUIL_DYNAMIQUE, SIGMA_JITTER,
    spatial_pooling_model_ou,
)
from .roster import (  # noqa: F401
    MIN_POLL_DATE, MIN_POLLS, ORDER_GROUPS, build_poll_arrays, build_roster,
    excess_var_for_nodes, position_anchors,
)
