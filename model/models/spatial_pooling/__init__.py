"""Modèle spatial (Hotelling-Downs discrétisé) — reports de voix cohérents et
scénarios de listes candidats personnalisables.

STATUT (2026-08-10) : maths + fit + lecture avec incertitude implémentés
(`model.py`, portage de notebooks/03_spatial_prototype.py à
04c_spatial_sanity_check.py), validés (synthétique, backtest half_life
2017/2022, sanity check 2027 -- cf. spec_spatial_pooling.md). PAS ENCORE un
`ForecastModel` enregistré (`@register`, model/core/base.py) : le contrat de
sortie du site est encore en réflexion (décision explicite de l'utilisateur).

Voir spec_spatial_pooling.md dans ce dossier pour la spécification complète.
"""

from .calibration import (  # noqa: F401
    BANK_EXCESS_PATH, SpatialExcessCalibration, excess_sigma_spatial, matched_pairs,
)
from .model import (  # noqa: F401
    B, BANK_JUMP_PATH, DEFAULT_HALF_LIFE_DAYS, MIN_POLL_DATE, MIN_POLLS, ORDER_GROUPS, V, W,
    SpatialPoolingFit, build_poll_arrays, build_roster, calibrate_jump, candidate_blocs_2027,
    excess_var_for_nodes, fit_spatial_pooling, forecast_spatial_pooling, load_jump_bank, make_kappa,
    pi_draws_for_mask, spatial_pooling_model, spatial_shares, summarize_pi, weighted_loglik,
)
