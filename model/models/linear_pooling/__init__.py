"""Modèle « linear-pooling » : lissage par demi-vie sans variable latente (cf.
spec_linear_pooling.md). Package minimal — un seul module, pas de séparation
calibration/nowcast comme bayesian_nowcast : le saut terminal réutilise tel
quel `TerminalJumpCalibration`, seule sa propre Bank (`bank_jump.json`, sans
débiaisage house-effects) est spécifique à ce modèle."""

from .model import (  # noqa: F401
    BANK_JUMP_PATH, LinearPooling, half_life_weight, linear_pooling_nowcast, resample_slot,
)
