"""Correction de diffusion de `gp-pooling` (cf. `FACTEUR_DIFFUSION`).

Ce que ces tests protègent : la propriété qui rend la correction sûre, à savoir
qu'elle se déplace le long de la crête σ²/τ constante. C'est elle qui garantit
que le nowcast — validé à 0,899 par `predictive_coverage` — ne bouge pas, et que
seule la projection au scrutin s'élargit. Un futur réglage qui toucherait σ² ou
τ séparément casserait cette garantie sans qu'aucun autre test ne le dise.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model.core.registered  # noqa: F401
from model.core.base import get_model
from model.core.opinion import load_law
from model.models.gp_pooling.model import FACTEUR_DIFFUSION

PARSED = ROOT / "data" / "parsed" / "intentions_2027_wiki.csv"
needs = pytest.mark.skipif(not PARSED.exists(), reason="données live absentes")


def test_taux_court_terme_inchange():
    """2σ²/τ — la diffusion à court terme — doit être conservé exactement."""
    law = load_law()
    mdl = get_model("gp-pooling")
    mdl.load_artifacts()
    avant = 2 * law["sigma2"] / law["tau"]
    apres = 2 * mdl.params["sigma2"] / mdl.params["tau"]
    assert apres == pytest.approx(avant, rel=1e-12)


def test_plateau_releve():
    """Le plateau 2σ², lui, doit monter — c'est tout l'objet de la correction."""
    law = load_law()
    mdl = get_model("gp-pooling")
    mdl.load_artifacts()
    assert mdl.params["sigma2"] == pytest.approx(law["sigma2"] * FACTEUR_DIFFUSION)
    assert mdl.params["tau"] == pytest.approx(law["tau"] * FACTEUR_DIFFUSION)
    assert FACTEUR_DIFFUSION > 1.0


def test_banque_intacte():
    """La banque est PARTAGÉE avec `spatial-pooling`, qui n'a pas de backtest de
    couverture au scrutin : la correction ne doit pas y fuir."""
    mdl = get_model("gp-pooling")
    mdl.load_artifacts()
    law = load_law()          # relu APRÈS load_artifacts
    assert law["sigma2"] != mdl.params["sigma2"]
    assert law["tau"] != mdl.params["tau"]


@needs
def test_nowcast_stable_et_scrutin_elargi():
    """Sur les données réelles : le nowcast bouge à peine, le scrutin s'élargit.

    C'est la conséquence attendue d'un déplacement sur la crête, et la seule
    raison pour laquelle la correction est publiable sans retoucher la courbe
    d'intentions déjà en ligne."""
    from model.core.live_dataset import load_raw_polls

    raw = load_raw_polls()
    if raw.empty:
        pytest.skip("aucun sondage exploitable")
    as_of = str(pd.to_datetime(raw["date_fin"]).max().date())
    mdl = get_model("gp-pooling")

    snap_corrige = mdl.run(raw, as_of)
    law = load_law()
    mdl.params = law                       # court-circuite la correction
    original = mdl.load_artifacts
    mdl.load_artifacts = lambda: setattr(mdl, "params", law)
    try:
        snap_banque = mdl.run(raw, as_of)
    finally:
        mdl.load_artifacts = original

    def largeur(snap, bloc):
        return np.mean([v["ic90"][1] - v["ic90"][0] for v in snap[bloc].values()])

    now_b, now_c = largeur(snap_banque, "nowcast"), largeur(snap_corrige, "nowcast")
    scr_b, scr_c = largeur(snap_banque, "forecast_scrutin"), largeur(snap_corrige, "forecast_scrutin")
    assert now_c == pytest.approx(now_b, rel=0.05), "le nowcast ne doit quasiment pas bouger"
    assert scr_c > scr_b, "les intervalles au scrutin doivent s'élargir"
