"""Calibration de la variance d'excès (house effects) PROPRE à `spatial_pooling`
-- cf. spec_spatial_pooling.md §6.5. Remplace l'emprunt brut de la Bank
`bayesian_nowcast` (mesuré en sur-correction : IC90 empirique à 99,6% au lieu
de 90% sur 2022, cf. §6.5) par un fit dédié sur le MÊME diagnostic model-free
qui a révélé le problème (`notebooks/06b_same_day_poll_coherence_matched.py`) :
paires de sondages de DEUX instituts différents testant EXACTEMENT le même
champ de candidats, à <= MAX_GAP_DAYS jours d'écart -- pas de fuite de
dérive temporelle (contrôle strict de la composition du champ), pas
d'emprunt d'une géométrie de modèle différente (bayesian_nowcast travaille en
CLR/ILR contre un résultat final, ce qui mélange bruit d'institut et dérive
de campagne dans son propre `excess_sigma` -- cf. §6.5 pour la discussion).

Modèle : `Y_a - Y_b ~ Normal(0, sampling_a + sampling_b + excess_a^2 + excess_b^2)`
-- la vraie composition du champ (donc pi) est IDENTIQUE par construction
(même scénario exact), la différence est donc un bruit PUR, pas besoin
d'estimer de niveau. Partial pooling non centré sur `excess_institut_offset`,
même pattern que `institut_bias`/`bloc_bias_mean` dans
`bayesian_nowcast/calibration.py::house_effects_model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

from model.core.bank import Bank
from model.core.inference import run_numpyro_mcmc

BANK_EXCESS_PATH = Path(__file__).parent / "bank_excess.json"
MAX_GAP_DAYS = 3   # écart temporel maximal entre deux sondages appariés -- cf. spec §6.5


def build_notices_core(df: pd.DataFrame, candidat_col="candidat", hyp_col="hypothese",
                       institut_col="institut", date_col="date_fin", notice_col="notice",
                       intention_col="intention", ech_col="echantillon") -> list[dict]:
    """Une entrée par SONDAGE (notice), PAS par hypothèse -- correction d'un
    bug de calibration (session du 2026-08-11, retour utilisateur) : les
    hypothèses d'un même sondage sont posées aux MÊMES répondants, donc
    comparer (hyp_a d'un institut) à (hyp_b d'un autre) puis RÉPÉTER pour
    chaque paire d'hypothèses matchées traite le MÊME couple de panels comme
    autant de mesures indépendantes qu'il y a d'hypothèses -- mesuré :
    jusqu'à 17 paires d'hypothèses pour un même couple de sondages sur
    2017/2022, 40% des 702 paires notice x notice provenant de seulement 50
    couples dupliqués.

    `candidate_set` ici = candidats CORE d'un sondage -- présents dans TOUTES
    ses hypothèses (donc leur valeur ne dépend pas de quelle hypothèse a été
    choisie), valeur = MOYENNE à travers ces hypothèses (légitime : la
    variation entre hypothèses d'un même sondage pour un candidat partagé est
    <1pt, déjà vérifié dans ce projet -- même panel, quasi la même mesure).
    Chaque SONDAGE devient une seule ligne (pas une par hypothèse)."""
    df = df.copy()
    df[hyp_col] = df[hyp_col].fillna("__unique__")
    notices = []
    for notice, g in df.groupby(notice_col):
        hyps = g[hyp_col].unique()
        sets_per_hyp = [set(g.loc[g[hyp_col] == h, candidat_col]) for h in hyps]
        core = set.intersection(*sets_per_hyp) if sets_per_hyp else set()
        if not core:
            continue
        vals = g[g[candidat_col].isin(core)].groupby(candidat_col)[intention_col].mean()
        notices.append(dict(
            notice=notice, institut=g[institut_col].iloc[0],
            date=pd.Timestamp(g[date_col].iloc[0]), echantillon=float(g[ech_col].iloc[0]),
            candidate_set=frozenset(core), values=vals.to_dict(),
        ))
    return notices


def matched_pairs(notices: list[dict], max_gap_days: int = MAX_GAP_DAYS) -> pd.DataFrame:
    """Paires de SONDAGES (pas d'hypothèses, cf. `build_notices_core`) de
    DEUX instituts différents, écart <= max_gap_days -- une ligne par
    (paire de sondages, candidat CORE des deux). Comparer sur l'INTERSECTION
    des candidats core (pas l'égalité stricte des ensembles) : un candidat
    core d'un sondage a une valeur robuste au choix d'hypothèse DE CE
    sondage, la comparaison reste valide même si les deux instituts ne
    testent pas exactement le même roster complet."""
    out = []
    for i, j in combinations(range(len(notices)), 2):
        a, b = notices[i], notices[j]
        if a["institut"] == b["institut"]:
            continue
        common = a["candidate_set"] & b["candidate_set"]
        if not common:
            continue
        if abs((a["date"] - b["date"]).days) > max_gap_days:
            continue
        if not (np.isfinite(a["echantillon"]) and np.isfinite(b["echantillon"])):
            continue
        for cand in common:
            pa, pb = a["values"][cand], b["values"][cand]
            if not (np.isfinite(pa) and np.isfinite(pb)):
                continue
            out.append(dict(institut_a=a["institut"], institut_b=b["institut"],
                            na=a["echantillon"], nb=b["echantillon"], pa=pa, pb=pb))
    return pd.DataFrame(out)


@dataclass
class PairwiseExcessData:
    n_institut: int
    institut_a_idx: jnp.ndarray
    institut_b_idx: jnp.ndarray
    na: jnp.ndarray
    nb: jnp.ndarray
    pa: jnp.ndarray
    pb: jnp.ndarray
    institut_names: list[str]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "PairwiseExcessData":
        instituts = sorted(set(df["institut_a"]) | set(df["institut_b"]))
        idx = {inst: i for i, inst in enumerate(instituts)}
        return cls(
            n_institut=len(instituts),
            institut_a_idx=jnp.array([idx[i] for i in df["institut_a"]]),
            institut_b_idx=jnp.array([idx[i] for i in df["institut_b"]]),
            na=jnp.array(df["na"].to_numpy(dtype=float)), nb=jnp.array(df["nb"].to_numpy(dtype=float)),
            pa=jnp.array(df["pa"].to_numpy(dtype=float)), pb=jnp.array(df["pb"].to_numpy(dtype=float)),
            institut_names=instituts,
        )


DEFAULT_PRIORS_CFG = {
    "excess_log_scale_mu": float(np.log(1.0)), "excess_log_scale_sd": 1.0,
    "excess_institut_sd_sd": 0.5,
}


def pairwise_excess_model(data: PairwiseExcessData, priors_cfg: dict | None = None):
    """`pa - pb ~ Normal(0, sampling_a + sampling_b + excess_a^2 + excess_b^2)`
    -- même champ exact (cf. `matched_pairs`) donc pas de niveau à estimer,
    uniquement la variance. `excess_i = exp(excess_log_scale + offset_i)`,
    partial pooling non centré (comme `institut_bias` dans
    bayesian_nowcast/calibration.py)."""
    cfg = {**DEFAULT_PRIORS_CFG, **(priors_cfg or {})}
    excess_log_scale = numpyro.sample("excess_log_scale",
                                      dist.Normal(cfg["excess_log_scale_mu"], cfg["excess_log_scale_sd"]))
    excess_institut_sd = numpyro.sample("excess_institut_sd", dist.HalfNormal(cfg["excess_institut_sd_sd"]))
    z_off = numpyro.sample("excess_institut_offset_z", dist.Normal(0.0, 1.0).expand([data.n_institut]))
    excess_institut_offset = numpyro.deterministic("excess_institut_offset", z_off * excess_institut_sd)

    excess_a = jnp.exp(excess_log_scale + excess_institut_offset[data.institut_a_idx])
    excess_b = jnp.exp(excess_log_scale + excess_institut_offset[data.institut_b_idx])
    sampling_var = (data.pa * (100.0 - data.pa) / jnp.clip(data.na, 1.0, None)
                    + data.pb * (100.0 - data.pb) / jnp.clip(data.nb, 1.0, None))
    total_var = sampling_var + excess_a ** 2 + excess_b ** 2
    numpyro.sample("obs", dist.Normal(0.0, jnp.sqrt(total_var)), obs=data.pa - data.pb)


class SpatialExcessCalibration:
    """Pas d'ABC : compose run_numpyro_mcmc + Bank, même pattern que
    `HouseEffectsCalibration` (bayesian_nowcast/calibration.py)."""
    draws = 1000
    tune = 1000
    chains = 4
    target_accept = 0.9
    seed = 27
    priors_cfg: dict | None = None

    def calibrate(self, pairs_df: pd.DataFrame) -> Bank:
        data = PairwiseExcessData.from_dataframe(pairs_df)
        samples, _ = run_numpyro_mcmc(
            pairwise_excess_model, {"data": data, "priors_cfg": self.priors_cfg},
            draws=self.draws, tune=self.tune, chains=self.chains,
            seed=self.seed, target_accept=self.target_accept)
        return Bank(samples, dims={"excess_institut_offset": ["institut"]},
                   coords={"institut": data.institut_names})


def excess_sigma_spatial(bank: Bank | None, institut: str) -> float:
    """Écart-type d'excès (points de %) pour UN institut -- accesseur utilisé
    par `excess_var_for_nodes` (model.py). Repli sur la moyenne de
    population si l'institut est inconnu de la Bank (shrinkage, même logique
    que `institut_bias_prior`), 0.0 si `bank is None`."""
    if bank is None:
        return 0.0
    log_scale, _ = bank.excess_log_scale.item()
    try:
        offset, _ = bank.excess_institut_offset.at(institut=institut)
    except KeyError:
        offset = 0.0
    return float(np.exp(log_scale + offset))


def main() -> None:
    """Calibre la variance d'excès sur 2017+2022+2027 (tous les sondages
    appariables disponibles) et exporte la Bank.
    `.venv/bin/python -m model.models.spatial_pooling.calibration`"""
    from model.core.live_dataset import load_raw_polls
    from pipeline.historical import ELECTION_DATES, PARSED_DIR, POLL_FILES, _resolve_candidate, load_blocs

    all_pairs = []
    for elec in (2017, 2022):
        polls = pd.read_csv(PARSED_DIR / POLL_FILES[elec])
        t1 = polls[polls["tour"] == "Premier tour"].copy()
        blocs = load_blocs()
        real = set(blocs[blocs["election"] == elec]["candidat"])
        t1["candidat"] = t1["candidat"].apply(lambda raw: _resolve_candidate(raw, real))
        t1 = t1[t1["candidat"].isin(real)]
        date_fin = pd.to_datetime(t1["date_fin"], errors="coerce")
        date_notice = pd.to_datetime(t1.get("date_notice"), errors="coerce")
        t1["date_fin"] = date_fin.fillna(date_notice)
        t1 = t1[t1["date_fin"].notna()]
        elec_date = pd.Timestamp(ELECTION_DATES[elec])
        t1 = t1[(t1["date_fin"] > elec_date - pd.Timedelta(days=400)) & (t1["date_fin"] <= elec_date)]
        all_pairs.append(matched_pairs(build_notices_core(t1)))

    raw2027 = load_raw_polls()
    raw2027 = raw2027[pd.to_datetime(raw2027["date_fin"]) >= pd.Timestamp("2026-01-01")]
    all_pairs.append(matched_pairs(build_notices_core(raw2027)))

    pairs_df = pd.concat(all_pairs, ignore_index=True)
    print(f"Calibration sur {len(pairs_df)} paires ({pairs_df['institut_a'].nunique() + pairs_df['institut_b'].nunique()} instituts, bruts)")

    cal = SpatialExcessCalibration()
    bank = cal.calibrate(pairs_df)
    bank.save(BANK_EXCESS_PATH)
    print(f"Bank (variance d'excès) écrite dans {BANK_EXCESS_PATH}")
    print(f"excess_log_scale -> échelle globale = {np.exp(bank.excess_log_scale.item()[0]):.3f} pt")


if __name__ == "__main__":
    main()
