"""Modèle « linear-pooling » : lissage par demi-vie, sans variable latente.

Alternative simple et transparente à bayesian-nowcast (SSM) : au lieu d'un
état latent qui diffuse dans le temps (marche aléatoire + NUTS), chaque
candidat/slot est une moyenne pondérée DIRECTE de ses sondages — poids =
taille d'échantillon × décroissance demi-vie. L'incertitude au jour `as_of`
est un MÉLANGE pondéré (linear pooling), pas la variance d'une moyenne
pondérée classique — cf. spec_linear_pooling.md §4 pour la justification et
les dérivations complètes.

Réutilise SLOTS/aggregate_to_slots (vocabulaire des candidats — Philippe pas
Attal, Le Pen pas Bardella, cf. model/core/live_dataset.py) et le saut
terminal sinh-arcsinh déjà utilisé par bayesian-nowcast (même mécanisme,
`TerminalJumpCalibration` + `forecast_from_draws`, cf. model/core/simulate.py)
: ce modèle ne réinvente que le NOWCAST, pas la projection jusqu'au scrutin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from model.core.bank import Bank
from model.core.base import ForecastModel, Nowcast, register
from model.core.live_dataset import ELECTION_T1, SLOTS, aggregate_to_slots
from model.core.simulate import forecast_from_draws
from model.core.terminal_jump import TerminalJumpCalibration, jump_moves

BANK_JUMP_PATH = Path(__file__).parent / "bank_jump.json"

# Plancher pour un slot jamais testé à `as_of` (ex. Ruffin en tout début de
# campagne) — même esprit que NEW_ENTRANT_FLOOR de bayesian-nowcast
# (model/models/bayesian_nowcast/nowcast.py), mais exprimé directement en
# paramètres Beta plutôt qu'en part fixe : une moyenne faible (1%) ET une
# confiance faible (n=100 équivalent), pas une part gelée sans incertitude.
FALLBACK_MEAN_PCT = 1.0
FALLBACK_N = 100.0


def half_life_weight(age_days: np.ndarray, n_effectif: np.ndarray, half_life_days: float) -> np.ndarray:
    """W_p(T) = n_p · 0.5^(age/H) — cf. spec §3."""
    return n_effectif * 0.5 ** (age_days / half_life_days)


def resample_slot(sub: pd.DataFrame, as_of: pd.Timestamp, half_life_days: float,
                  n_draws: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """Tirages (n_draws,) pour UN slot : mélange pondéré par demi-vie (spec
    §4) — on choisit un sondage par tirage (probabilité ∝ poids demi-vie),
    puis on tire dans SA loi Beta(n_p·Y_p, n_p·(1-Y_p)) : le poids demi-vie ne
    sert qu'à la sélection, le bruit d'échantillonnage reste celui, réel, du
    sondage choisi (pas dilué/gonflé par la décroissance temporelle).

    Renvoie aussi l'ÂGE (jours) du sondage sélectionné à chaque tirage : le
    bruit ci-dessous est celui du terrain du sondage, valable le jour où il a
    été mesuré, pas à `as_of`. La dérive d'opinion depuis cette date est
    ajoutée en amont par `linear_pooling_nowcast` — sans quoi un sondage
    vieux de trois ans serait aussi « précis » qu'un sondage d'hier.
    """
    if sub.empty:
        draws = rng.beta(FALLBACK_MEAN_PCT / 100.0 * FALLBACK_N,
                         (1.0 - FALLBACK_MEAN_PCT / 100.0) * FALLBACK_N, size=n_draws)
        # Âge 0 : le repli est déjà une loi large (n=100 équivalent), il ne
        # décrit aucun sondage daté qu'il faudrait projeter.
        return draws, np.zeros(n_draws), {"omega": 0.0, "n_polls": 0,
                                          "point_estimate": FALLBACK_MEAN_PCT / 100.0}

    age = (as_of - pd.to_datetime(sub["date_fin"])).dt.days.to_numpy().astype(float)
    # n_eff (déflaté par n_hypotheses) sert à la SÉLECTION — évite qu'un
    # sondage à 6 variantes pèse 6x plus qu'un sondage à 1 seule dans le
    # mélange (même logique que bayesian-nowcast, cf. n_echant dans
    # NowcastData.from_dataframe). n_brut (échantillon RÉEL) sert au bruit
    # UNE FOIS le sondage choisi : contrairement à bayesian-nowcast (qui
    # fusionne TOUTES les hypothèses d'un sondage comme observations
    # simultanées, où déflater évite de compter le même terrain plusieurs
    # fois), ici le mélange n'en choisit qu'UNE SEULE par tirage — la
    # déflation a déjà fait son travail à l'étape de sélection, la répéter
    # sur le bruit revient à pénaliser deux fois le même sondage (mesuré :
    # gonfle l'écart-type du RN de ~1,9 à ~3,0 points sur le cas réel de la
    # session du 2026-08-10, cf. spec §4.4).
    n_eff = (sub["echantillon"] / sub["n_hypotheses"]).to_numpy()
    n_brut = sub["echantillon"].to_numpy()
    y = (sub["part"] / 100.0).to_numpy()
    w = half_life_weight(age, n_eff, half_life_days)
    omega = float(w.sum())
    probs = w / omega
    point_estimate = float((probs * y).sum())

    picks = rng.choice(len(sub), size=n_draws, p=probs)
    n_pick, y_pick = n_brut[picks], y[picks]
    alpha = np.clip(y_pick * n_pick, 1e-3, None)
    beta = np.clip((1.0 - y_pick) * n_pick, 1e-3, None)
    draws = rng.beta(alpha, beta)
    return draws, age[picks], {"omega": round(omega, 1), "n_polls": int(len(sub)),
                               "point_estimate": round(point_estimate, 4),
                               "age_moyen_j": round(float(age[picks].mean()), 1)}


@dataclass
class LinearPoolingResult:
    pi: np.ndarray             # (S, K) simplexe par tirage
    slots: list[str]
    diagnostics: dict


def linear_pooling_nowcast(raw_polls: pd.DataFrame, as_of: str, slots: list[str],
                           half_life_days: float, n_draws: int, seed: int,
                           jump_bank=None, candidate_blocs: dict[str, str] | None = None,
                           drift_loc: bool = True) -> LinearPoolingResult:
    """Cœur du nowcast : mélange pondéré indépendant par slot (§4), **projeté à
    `as_of`**, puis renormalisation de CHAQUE tirage sur le simplexe — nécessaire
    pour respecter le contrat `Nowcast.draws` (une ligne = une composition), pas
    seulement la moyenne agrégée.

    Deux étapes distinctes, à ne pas confondre :

    1. *Mesure* — le mélange donne la part telle que le sondage sélectionné l'a
       mesurée, avec le bruit d'échantillonnage de SON terrain, valable au jour
       de ce terrain.
    2. *Dérive jusqu'à `as_of`* — l'opinion a bougé depuis. On applique le saut
       calibré (`model/core/terminal_jump.py`) sur l'âge PROPRE À CHAQUE TIRAGE,
       en espace log-ratio. Sans cette étape, l'incertitude du nowcast ne
       grandissait pas d'un pouce quand les sondages vieillissaient (mesuré :
       0,000 pt d'écart entre un calcul le jour du dernier sondage et 32 jours
       plus tard), et un sondage de 2023 restait aussi « précis » qu'un sondage
       d'hier.

    `forecast()` applique ensuite la MÊME loi sur la jambe `as_of → scrutin`.
    Les deux tirages sont indépendants : la variance se cumule exactement
    (`scale²·âge + scale²·horizon = scale²·(âge+horizon)`, propriété d'une
    diffusion à variance ∝ t). `drift_loc=False` retire la composante
    systématique de CETTE jambe-ci — cf. `jump_moves` pour pourquoi `loc`, lui,
    ne se découpe pas proprement.
    """
    obs = aggregate_to_slots(raw_polls)
    as_of_ts = pd.Timestamp(as_of)
    rng = np.random.default_rng(seed)

    theta = np.empty((n_draws, len(slots)))
    ages = np.zeros((n_draws, len(slots)))
    diag_per_slot = {}
    for k, s in enumerate(slots):
        sub = obs[obs["slot"] == s]
        draws, age, diag = resample_slot(sub, as_of_ts, half_life_days, n_draws, rng)
        theta[:, k] = draws
        ages[:, k] = age
        diag_per_slot[s] = diag

    diagnostics = {"half_life_days": half_life_days, "par_slot": diag_per_slot}
    alpha = np.log(np.clip(theta, 1e-6, None))
    if jump_bank is not None and candidate_blocs is not None:
        blocs = [candidate_blocs[s] for s in slots]
        # Horizons comptés en jours AVANT LE SCRUTIN : le sondage sélectionné
        # est à `h_as_of + âge`, `as_of` est à `h_as_of`. La jambe couvre donc
        # exactement l'intervalle sondage -> as_of, et `forecast()` enchaîne
        # sur as_of -> scrutin sans recouvrement (cf. `jump_moves`).
        h_as_of = float((ELECTION_T1 - as_of_ts).days)
        alpha = alpha + jump_moves(jump_bank, blocs, h_from=h_as_of + ages, h_to=h_as_of,
                                   size=alpha.shape, seed=seed + 1, with_loc=drift_loc)
        tau, _ = jump_bank.jump_tau.item()
        diagnostics["derive_vers_as_of"] = {
            "applique": True, "avec_loc": drift_loc, "tau_j": round(float(tau), 1),
            "age_moyen_pondere_j": round(float(ages.mean()), 1),
            "horizon_as_of_j": round(h_as_of),
        }
    else:
        diagnostics["derive_vers_as_of"] = {"applique": False}
    e = np.exp(alpha - alpha.max(axis=1, keepdims=True))
    pi = e / e.sum(axis=1, keepdims=True)
    return LinearPoolingResult(pi=pi, slots=slots, diagnostics=diagnostics)


@register
class LinearPooling(ForecastModel):
    id = "linear-pooling"
    label = "Lissage linéaire (demi-vie)"
    description = ("Moyenne pondérée directe des sondages par candidat/slot — poids = "
                   "taille d'échantillon × décroissance demi-vie, sans état latent ni "
                   "marche aléatoire. Incertitude par mélange pondéré (rééchantillonnage "
                   "+ bruit Beta d'échantillonnage), saut terminal sinh-arcsinh calibré "
                   "sur 2017/2022 pour la projection jusqu'au scrutin (même mécanisme que "
                   "bayesian-nowcast).")
    trains_on_history = True
    public = True

    # Demi-vie (jours) : cf. spec_linear_pooling.md §3 — valeur de départ
    # NON encore backtestée (model/backtest/backtest_loo.py permettrait un
    # balayage empirique comme pour tau/historical_prior_n sur
    # bayesian-nowcast), pas une constante définitive.
    half_life_days = 14.0
    n_draws = 4000
    seed = 27

    def calibrate(self, history=None) -> None:
        # bank=None : pas de débiaisage house-effects lors du fit du saut
        # (cohérent avec ce modèle, qui ne modélise pas de biais institut
        # dans son propre nowcast — cf. spec §5).
        TerminalJumpCalibration().calibrate(None).save(BANK_JUMP_PATH)

    def load_artifacts(self) -> None:
        self.jump_bank = Bank.load(BANK_JUMP_PATH)

    def nowcast(self, raw_polls, as_of) -> Nowcast:
        slots = list(SLOTS)
        res = linear_pooling_nowcast(raw_polls, as_of, slots, self.half_life_days,
                                     self.n_draws, self.seed,
                                     jump_bank=self.jump_bank,
                                     candidate_blocs=self._candidate_blocs())
        draws = xr.DataArray(res.pi, dims=("draw", "candidat"), coords={"candidat": slots})
        return Nowcast(draws=draws, diagnostics=res.diagnostics)

    def forecast(self, nc: Nowcast, horizon_days: int) -> dict:
        return forecast_from_draws(nc.draws.values, nc.draws.candidat.values.tolist(),
                                   horizon_days, jump_bank=self.jump_bank,
                                   candidate_blocs=self._candidate_blocs())

    def _candidate_blocs(self) -> dict[str, str]:
        return {slot: bloc for slot, (bloc, _names) in SLOTS.items()}


def main() -> None:
    """Calibre le saut terminal et exporte la Bank.
    `.venv/bin/python -m model.models.linear_pooling`"""
    print("Calibration du saut terminal (linear-pooling, sans débiaisage house-effects)...")
    LinearPooling().calibrate()
    print(f"Bank (saut terminal) écrite dans {BANK_JUMP_PATH}")


if __name__ == "__main__":
    main()
