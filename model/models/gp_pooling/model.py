"""Modèle « gp-pooling » : état latent gaussien à noyau Ornstein-Uhlenbeck.

Corrige un défaut MESURÉ de `linear-pooling` : son incertitude est un mélange
pondéré, donc tout désaccord entre sondages y est traité comme irréductible, et
deux sondages du même jour qui diffèrent ÉLARGISSENT l'intervalle. Ici il existe
une quantité θ(t) dont les sondages sont des mesures bruitées — deux mesures de
la même quantité la RESSERRENT. Mesuré : IC 90 % du RN sur six instituts ramenés
au même jour, 11,0 → 5,4 pt quand on les ajoute, là où `linear-pooling` passe de
4,5 à 5,9 pt.

Tout est gaussien, donc le postérieur est en forme close (`gp.py`) : pas de
MCMC, pas de divergences, pas de R-hat à surveiller.

Ce que ce modèle NE change pas
------------------------------
La projection `as_of → scrutin` reste le saut terminal sinh-arcsinh
(`model/core/movements.py`), via `forecast_from_draws`. Ce n'est pas de la
paresse : le saut est calibré sur `clr(résultat) − clr(sondage)`, qui contient
un écart systématique **sondages → urne** en plus du mouvement d'opinion —
mesuré à δ ≈ 0,35 en CLR (facteur 1,42 sur les parts), constant avec l'horizon
(à J-14, `sondage → résultat` vaut déjà 0,111 quand `sondage → sondage` sur la
même durée vaut 0,018). Le noyau du GP, lui, est calibré sur le mouvement
sondage à sondage. Deux quantités distinctes, chacune employée là où elle a été
mesurée — d'où le rejet de l'identité `σ² = scale²/2` que proposait la première
version de la spec (elle dégrade la vraisemblance de 465 unités).

Roster exact
------------
On ne consomme que les hypothèses testant EXACTEMENT les slots du modèle
(`filter_scenarios_by_exact_slots`). Le CLR suppose un champ complet, et deux
champs différents ne mesurent pas la même quantité. C'est ce qui rend ce modèle
nettement plus simple que sa première version : aucune imputation des slots
manquants n'est nécessaire.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from model.core.bank import Bank
from model.core.base import ForecastModel, Nowcast, register
from model.core.live_dataset import (
    ELECTION_T1, SLOTS, aggregate_to_slots, filter_scenarios_by_exact_slots,
)
from model.core.simulate import forecast_from_draws
from model.core.opinion import load_law
from model.core.projection import delta_draws
from model.core.gp_math import (
    clr_rows, draws_from_posterior, gp_posterior, sampling_cov_clr_diag,
)

SHARE_FLOOR = 1e-4


def _softmax(a):
    e = np.exp(a - a.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def scenarios_matrix(raw_polls: pd.DataFrame, slots: list[str]
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Met les sondages en forme pour le GP : `(Z, n, h, inst_idx, instituts)`.

    UNE ligne par SONDAGE (pas par hypothèse) : les hypothèses d'un même sondage
    qui passent le filtre de roster décrivent le même terrain, les moyenner
    évite de compter deux fois la même mesure. C'est aussi ce qui rend `σ_p`
    (effet propre au sondage) purement diagonal dans le postérieur.
    """
    obs = aggregate_to_slots(raw_polls)
    if obs.empty:
        raise ValueError("aucune hypothèse exploitable après filtrage du roster")

    # moyenne des hypothèses retenues d'un même sondage, par slot
    piv = (obs.groupby(["notice", "institut", "date_fin", "echantillon", "slot"],
                       dropna=False, as_index=False)["part"].mean())
    wide = piv.pivot_table(index=["notice", "institut", "date_fin", "echantillon"],
                           columns="slot", values="part", dropna=False)
    wide = wide.reindex(columns=slots)
    wide = wide.dropna()                       # sécurité : le filtre garantit déjà le champ complet
    if wide.empty:
        raise ValueError("aucun sondage au roster complet")

    meta = wide.index.to_frame(index=False)
    parts = wide.to_numpy(dtype=float) / 100.0
    parts = np.clip(parts, SHARE_FLOOR, None)
    parts = parts / parts.sum(axis=1, keepdims=True)

    h = (ELECTION_T1 - pd.to_datetime(meta["date_fin"])).dt.days.to_numpy(dtype=float)
    n = meta["echantillon"].to_numpy(dtype=float)
    n = np.where(np.isfinite(n) & (n > 0), n, np.nanmedian(n[n > 0]) if (n > 0).any() else 1000.0)
    instituts = sorted(meta["institut"].unique().tolist())
    inst_idx = meta["institut"].map({x: i for i, x in enumerate(instituts)}).to_numpy()
    return parts, n, h, inst_idx, instituts


@register
class GPPooling(ForecastModel):
    id = "gp-pooling"
    label = "Estimation de l'opinion"
    # Texte affiché sur le site : il s'adresse à un lecteur non technique. Les
    # détails (noyau, postérieur, calibration) sont dans spec_gp_pooling.md.
    description = ("Chaque sondage est une mesure imparfaite de l'opinion du moment : il "
                   "comporte une marge d'erreur liée au nombre de personnes interrogées, et "
                   "chaque institut a ses habitudes de mesure. Plutôt que de faire une simple "
                   "moyenne, le modèle reconstitue l'opinion qui explique le mieux l'ensemble "
                   "des sondages — deux sondages proches dans le temps se confirment donc l'un "
                   "l'autre et resserrent l'estimation. Il ajoute ensuite deux incertitudes : "
                   "l'opinion peut encore bouger d'ici au vote, et le résultat des urnes "
                   "s'écarte toujours un peu des intentions déclarées. Ces deux effets sont "
                   "mesurés sur les campagnes de 2017 et 2022.")
    trains_on_history = True
    # Rubrique « Suivi des intentions » : les quatre critères d'acceptation de la
    # spec §8 sont atteints (couverture prédictive 0,870 / 0,900 / 0,906 / 0,900
    # selon la taille du pool, contre 0,663 / 0,900 / 0,900 / 0,981 pour
    # `linear-pooling` ; même couverture au scrutin avec des intervalles 15 %
    # plus étroits). Publier un IC 90 % mesuré à ~81 % quand une alternative
    # calibrée existe irait contre la ligne du projet.
    surface = "suivi"

    n_draws = 4000
    seed = 27

    def load_artifacts(self) -> None:
        self.params = load_law()

    def used_polls(self, raw_polls):
        return filter_scenarios_by_exact_slots(raw_polls, set(SLOTS))

    def nowcast(self, raw_polls, as_of) -> Nowcast:
        slots = list(SLOTS)
        retenu = self.used_polls(raw_polls)
        if retenu.empty:
            raise ValueError(f"aucune hypothèse au roster exact à la date {as_of}")
        parts, n, h, inst_idx, instituts = scenarios_matrix(retenu, slots)

        Z = clr_rows(parts)
        # σ_p : effet propre au sondage (au-delà de l'échantillonnage et de
        # l'institut), calibré par REML. Purement diagonal ici puisqu'une ligne
        # = un sondage.
        V = sampling_cov_clr_diag(parts, n) + self.params["sigma_p"] ** 2

        h_as_of = float((ELECTION_T1 - pd.Timestamp(as_of)).days)
        post = gp_posterior(Z, V, h, inst_idx, h_star=h_as_of,
                            tau=self.params["tau"], sigma2=self.params["sigma2"],
                            sigma_h=self.params["sigma_h"])
        pi = draws_from_posterior(post.mean, post.var_latent, self.n_draws, self.seed)

        # Le MÊME postérieur prolongé jusqu'au jour du scrutin (h = 0) : la
        # diffusion d'ici au vote sort du noyau, pas d'un mécanisme séparé.
        post0 = gp_posterior(Z, V, h, inst_idx, h_star=0.0,
                             tau=self.params["tau"], sigma2=self.params["sigma2"],
                             sigma_h=self.params["sigma_h"])
        rng = np.random.default_rng(self.seed + 2)
        self._theta_scrutin = rng.normal(post0.mean, np.sqrt(np.maximum(post0.var_latent, 0.0)),
                                         size=(self.n_draws, len(slots)))

        draws = xr.DataArray(pi, dims=("draw", "candidat"), coords={"candidat": slots})
        diagnostics = {
            "noyau": "Ornstein-Uhlenbeck",
            "tau_j": round(self.params["tau"], 1),
            "sigma": round(float(np.sqrt(self.params["sigma2"])), 4),
            "sigma_h": round(self.params["sigma_h"], 4),
            "sigma_p": round(self.params["sigma_p"], 4),
            "n_sondages": int(len(h)),
            "instituts": instituts,
            "age_sondage_le_plus_recent_j": round(float(h.min() - h_as_of), 1),
            "sd_latente_par_slot": {s: round(float(np.sqrt(v)), 4)
                                    for s, v in zip(slots, post.var_latent)},
        }
        return Nowcast(draws=draws, diagnostics=diagnostics)

    def forecast(self, nc: Nowcast, horizon_days: int) -> dict:
        """Deux mécanismes distincts, chacun estimé sur ce qui le mesure :

        1. la **diffusion** de l'opinion d'ici au scrutin, prolongée par le même
           postérieur GP jusqu'à `h = 0` (noyau OU, calibré sondage à sondage) ;
        2. l'**écart de traduction** δ entre intention mesurée et bulletin
           déposé, qui ne dépend pas de l'horizon (`terminal.py`).

        L'ancien saut terminal ajustait une seule loi sur la somme des deux, avec
        une forme nulle en `h = 0` qui interdisait à δ d'exister.
        """
        slots = nc.draws.candidat.values.tolist()
        theta0 = self._theta_scrutin                      # (S, K) posé par nowcast()
        pi = _softmax(theta0 + delta_draws(self.params, theta0.shape, self.seed + 3))
        out = forecast_from_draws(pi, slots, horizon_days, drift_sd=0.0)
        # Texte affiché sur le site : sans jargon (les détails sont dans la spec).
        out["drift_modele"] = (f"l'opinion a encore {horizon_days} jours pour bouger "
                               f"d'ici au vote, et le résultat des urnes s'écarte "
                               f"toujours un peu des intentions déclarées")
        return out
