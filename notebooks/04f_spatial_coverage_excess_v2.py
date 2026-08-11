"""Comme 04e_spatial_coverage_excess.py, mais avec la Bank de variance
d'excès CALIBRÉE POUR spatial_pooling (`calibration.SpatialExcessCalibration`,
`BANK_EXCESS_PATH`) au lieu de la Bank empruntée à `bayesian_nowcast` --
04e avait mesuré une sur-correction sévère (IC90 empirique 99,6% au lieu de
90% sur 2022, cf. spec §6.5). Vérifie si la Bank dédiée retombe proche du
nominal.

Exécution (un seul run, dispatché en parallèle) :
    .venv/bin/python notebooks/04f_spatial_coverage_excess_v2.py <election> <cutoff_days> <half_life>
Agrégation :
    .venv/bin/python notebooks/04f_spatial_coverage_excess_v2.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_spec = importlib.util.spec_from_file_location("bt04b", "notebooks/04b_spatial_halflife_backtest.py")
bt04b = importlib.util.module_from_spec(_spec)
sys.modules["bt04b"] = bt04b
_spec.loader.exec_module(bt04b)

_spec2 = importlib.util.spec_from_file_location("cov04e", "notebooks/04e_spatial_coverage_excess.py")
cov04e = importlib.util.module_from_spec(_spec2)
sys.modules["cov04e"] = cov04e
_spec2.loader.exec_module(cov04e)

from model.core.bank import Bank
from model.core.inference import run_numpyro_mcmc
from model.models.spatial_pooling.calibration import BANK_EXCESS_PATH
from model.models.spatial_pooling.model import build_poll_arrays, excess_var_for_nodes, make_kappa, spatial_pooling_model
from pipeline.historical import ELECTION_DATES

EVAL_WINDOW_DAYS = bt04b.EVAL_WINDOW_DAYS
RESULTS_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
posterior_predictive_coverage_excess = cov04e.posterior_predictive_coverage_excess


def run_one(election: int, cutoff_days: int, half_life: float, draws=1000, tune=1000, chains=4) -> dict | None:
    df, candidates, slot_of = bt04b.load_historical_long(election)
    elec_date = pd.Timestamp(ELECTION_DATES[election])
    cutoff_date = elec_date - pd.Timedelta(days=cutoff_days)
    eval_end = cutoff_date + pd.Timedelta(days=EVAL_WINDOW_DAYS)
    train_df = df[df["date_fin"] <= cutoff_date]
    test_df = df[(df["date_fin"] > cutoff_date) & (df["date_fin"] <= eval_end)]
    if train_df["notice"].nunique() < 5 or test_df.empty:
        return None

    arrays_tr = build_poll_arrays(train_df, candidates)
    if arrays_tr["tested_mask"].shape[0] < 5:
        return None
    as_of = arrays_tr["dates"].max()
    kappa = make_kappa(arrays_tr["dates"], as_of=as_of, half_life=half_life)

    bank = Bank.load(BANK_EXCESS_PATH)
    excess_var_tr = excess_var_for_nodes(arrays_tr["instituts"], bank)

    kwargs = dict(
        slot_of=jnp.asarray(slot_of), n_slots=len(bt04b.BLOC_ORDER),
        tested_mask=jnp.asarray(arrays_tr["tested_mask"]), Y=jnp.asarray(arrays_tr["Y"]),
        Np=jnp.asarray(arrays_tr["Np"]), kappa=jnp.asarray(kappa), excess_var=jnp.asarray(excess_var_tr),
    )
    samples, _ = run_numpyro_mcmc(spatial_pooling_model, kwargs, draws=draws, tune=tune,
                                  chains=chains, seed=27, target_accept=0.9)
    N = len(candidates)
    mu_draws = np.asarray(samples["mu"]).reshape(-1, N)
    sigma_draws = np.asarray(samples["sigma"]).reshape(-1, N)
    w_draws = np.asarray(samples["w_now"]).reshape(-1, N)

    arrays_te = build_poll_arrays(test_df, candidates)
    if arrays_te["tested_mask"].shape[0] == 0:
        return None
    excess_var_te = excess_var_for_nodes(arrays_te["instituts"], bank)
    ages = arrays_te["dates"] - cutoff_date.toordinal()
    cov = posterior_predictive_coverage_excess(mu_draws, sigma_draws, w_draws, arrays_te["tested_mask"],
                                               arrays_te["Y"], arrays_te["Np"], excess_var_te, ages=ages)
    return dict(election=election, cutoff_days=cutoff_days, half_life=half_life, **cov)


def main():
    if len(sys.argv) == 4:
        election, cutoff_days, half_life = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
        t0 = time.time()
        res = run_one(election, cutoff_days, half_life)
        elapsed = time.time() - t0
        out_path = RESULTS_DIR / f"covexcess2_{election}_{cutoff_days}_{half_life:.0f}.json"
        out_path.write_text(json.dumps(res))
        print(f"election={election} cutoff={cutoff_days} half_life={half_life:.0f} "
              f"-> {res} ({elapsed:.0f}s) écrit dans {out_path}")
        return

    results = []
    for election in (2017, 2022):
        for cutoff_days in bt04b.CUTOFFS_BY_ELECTION[election]:
            p = RESULTS_DIR / f"covexcess2_{election}_{cutoff_days}_15.json"
            if not p.exists():
                print(f"(manquant : {p.name})")
                continue
            r = json.loads(p.read_text())
            if r is not None:
                results.append(r)

    print("\n=== Couverture AVEC variance d'excès DÉDIÉE spatial_pooling (half_life=15j) ===")
    for r in results:
        print(f"\n{r['election']} (coupure J-{r['cutoff_days']}) :")
        for lvl in (50, 80, 90):
            key = str(lvl)
            n = r["n_obs"].get(key, r["n_obs"].get(lvl))
            cov = r["coverage"].get(key, r["coverage"].get(lvl))
            width = r["width_moyenne_pt"].get(key, r["width_moyenne_pt"].get(lvl))
            print(f"  IC{lvl} : couverture empirique = {cov*100:5.1f}% (nominal {lvl}%), "
                  f"largeur moyenne = {width:.1f} pt, n={n}")


if __name__ == "__main__":
    main()
