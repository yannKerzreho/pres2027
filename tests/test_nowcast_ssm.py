"""Tests du chemin SSM (model/models/bayesian_nowcast/nowcast.py) — logique
rapide, sans NUTS. Couvre la régression du 2026-08-09 : le nowcast donnait
des résultats délirants (ex. Arthaud, ~1% dans tous les sondages, ressortait
à 15-17%) à cause de PLUSIEURS défauts distincts corrigés à la racine (pas
par un plafond ad hoc, cf. docs/spec_ssm_implementation.md §7-10) :
  1. les sondages "prémonitoires" 2022/2023 créaient un silence de 861 jours
     avant la reprise de la campagne 2026 (MIN_POLL_DATE les écarte) ;
  2. le prior du premier nœud était centré sur des parts égales entre les 12
     slots (`alpha_0 ~ N(0, ...)`) au lieu des rapports de force connus
     (`historical_prior_pi`, informé par les résultats 2022) ;
  3. sa covariance était isotrope (même variance ABSOLUE pour un grand et un
     petit candidat) au lieu de proportionnelle par delta-method
     (`historical_prior_cov`, cf. §9-10 pour la dérivation)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.core.live_dataset import SLOTS
from model.core.utils import SinhArcsinh

# `bayesian_nowcast` importe dynamax, absent des requirements de `main` où ce
# modèle monte sans être enregistré (cf. model/core/registered.py). Skip plutôt
# qu'échec de collecte : la même suite doit passer sur les deux branches.
pytest.importorskip("dynamax")

from model.models.bayesian_nowcast.latent import helmert_basis, ilr_decode
from model.models.bayesian_nowcast.nowcast import (
    MIN_POLL_DATE, filter_scenarios_by_exact_slots, filter_scenarios_by_required_slots,
    historical_prior_cov, historical_prior_pi,
)

PARSED = ROOT / "data" / "parsed" / "intentions_2027_wiki.csv"
BANK = ROOT / "model" / "models" / "bayesian_nowcast" / "bank.json"


def test_historical_prior_reflete_les_rapports_de_force():
    # RN/Centre (grands blocs 2022) doivent dominer très largement les
    # candidats marginaux — pas un prior uniforme.
    slots = list(SLOTS)
    p = historical_prior_pi(slots)
    by_slot = dict(zip(slots, p))
    assert by_slot["RN"] > 15.0
    assert by_slot["Philippe (Horizons)"] > 15.0
    assert by_slot["Arthaud (LO)"] < 3.0
    assert by_slot["Dupont-Aignan"] < 5.0
    np.testing.assert_allclose(p.sum(), 100.0)


def test_historical_prior_plancher_nouvel_entrant():
    # Un slot sans équivalent en 2022 doit recevoir le plancher, pas zéro :
    # sinon le prior le condamnerait avant tout sondage. Testé sur un nom
    # arbitraire, pour ne pas dépendre de la composition du jour de `SLOTS`.
    slots = list(SLOTS) + ["Nouvel Entrant (inconnu 2022)"]
    p = historical_prior_pi(slots)
    assert dict(zip(slots, p))["Nouvel Entrant (inconnu 2022)"] > 0.0
    np.testing.assert_allclose(p.sum(), 100.0)


def test_min_poll_date_exclut_les_sondages_premonitoires():
    assert str(MIN_POLL_DATE.date()) == "2026-01-01"


def test_historical_prior_cov_pas_degenere_softmax():
    """Régression directe : sous la covariance delta-method du prior, tiré
    autour de la moyenne informée (pas 0), softmax(alpha) ne doit PAS
    dégénérer vers un sommet quasi aléatoire du simplexe (c'était le cas à
    var=25 isotrope, ~88 % du temps, cf. spec_ssm_implementation.md §7)."""
    slots = list(SLOTS)
    K = len(slots)
    V = helmert_basis(K)
    from model.models.bayesian_nowcast.latent import ilr_encode
    hist_pi = historical_prior_pi(slots)
    mean = ilr_encode(hist_pi, V)
    cov = historical_prior_cov(hist_pi, V)
    rng = np.random.default_rng(0)
    alphas = rng.multivariate_normal(mean, cov, size=2000)
    pis = np.array([np.asarray(ilr_decode(a, V)) for a in alphas])
    p_degenere = (pis.max(axis=1) > 0.7).mean()
    assert p_degenere < 0.20, f"prior trop dégénéré : P(un slot >70%) = {p_degenere:.2f}"


def test_historical_prior_cov_plus_serree_pour_petit_candidat():
    """La delta-method donne une variance ABSOLUE (en part reconstruite, pas
    en unités ILR brutes) plus resserrée pour un petit candidat que pour un
    grand — pas une covariance isotrope arbitraire entre eux (cf. §9-10)."""
    slots = list(SLOTS)
    V = helmert_basis(len(slots))
    from model.models.bayesian_nowcast.latent import ilr_encode
    hist_pi = historical_prior_pi(slots)
    mean = ilr_encode(hist_pi, V)
    cov = historical_prior_cov(hist_pi, V)
    rng = np.random.default_rng(0)
    alphas = rng.multivariate_normal(mean, cov, size=3000)
    pis = np.array([np.asarray(ilr_decode(a, V)) for a in alphas])
    sd_petit = pis[:, slots.index("Arthaud (LO)")].std()
    sd_rn = pis[:, slots.index("RN")].std()
    assert sd_petit < sd_rn, f"écart-type Arthaud ({sd_petit}) >= RN ({sd_rn})"


def test_filter_scenarios_by_required_slots_garde_les_scenarios_compatibles():
    df = pd.DataFrame({
        "scenario_id": ["a", "a", "b", "b", "c", "c"],
        "slot": ["RN", "Philippe (Horizons)",
                 "RN", "Mélenchon (LFI)",
                 "RN", "Philippe (Horizons)"],
        "notice": ["n1", "n1", "n2", "n2", "n3", "n3"],
    })
    kept = filter_scenarios_by_required_slots(df, ["RN", "Philippe (Horizons)"])
    assert set(kept["scenario_id"]) == {"a", "c"}


def test_filter_scenarios_by_required_slots_noop_si_vide():
    df = pd.DataFrame({
        "scenario_id": ["a", "b"],
        "slot": ["RN", "Philippe (Horizons)"],
    })
    kept = filter_scenarios_by_required_slots(df, [])
    pd.testing.assert_frame_equal(kept.reset_index(drop=True), df.reset_index(drop=True))


def test_filter_scenarios_by_exact_slots_garde_seulement_les_rosters_identiques():
    df = pd.DataFrame({
        "scenario_id": ["a", "a", "b", "b", "c", "c", "c"],
        "slot": ["RN", "Philippe (Horizons)",
                 "RN", "Mélenchon (LFI)",
                 "RN", "Philippe (Horizons)", "Mélenchon (LFI)"],
    })
    kept = filter_scenarios_by_exact_slots(df, ["RN", "Philippe (Horizons)"])
    assert set(kept["scenario_id"]) == {"a"}


def test_sinh_arcsinh_redevient_normale_si_skew_zero_tail_un():
    d = SinhArcsinh(loc=1.5, scale=2.0, skewness=0.0, tailweight=1.0)
    x = np.array([-1.0, 0.0, 2.5])
    got = np.asarray(d.log_prob(x))
    expected = np.asarray(
        -0.5 * np.log(2.0 * np.pi) - np.log(2.0) - 0.5 * ((x - 1.5) / 2.0) ** 2
    )
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not (PARSED.exists() and BANK.exists()),
                    reason="données live / banque de paramètres absentes")
def test_aucun_ecart_calendaire_pathologique_apres_filtre():
    """Régression directe du gap de 861 jours : une fois MIN_POLL_DATE
    appliqué, le plus grand Δt entre deux sondages consécutifs (2026, dense)
    doit rester raisonnable — pas plusieurs centaines de jours."""
    from model.core.bank import Bank
    from model.core.live_dataset import load_raw_polls
    from model.models.bayesian_nowcast.nowcast import NowcastData

    bank = Bank.load(BANK)
    raw = load_raw_polls()
    as_of = str(raw["date_fin"].max())[:10]
    data = NowcastData.from_dataframe(raw, as_of, bank)
    assert data.dt_forward[:-1].max() < 100, (
        f"écart calendaire suspect après filtre MIN_POLL_DATE : {data.dt_forward}")
