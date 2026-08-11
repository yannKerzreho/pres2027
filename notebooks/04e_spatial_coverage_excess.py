"""Même protocole que 04d_spatial_coverage_check.py, mais avec la variance
d'excès (house effects, `excess_var_for_nodes`) ajoutée à la vraisemblance du
fit ET à la distribution prédictive postérieure -- vérifie si ça corrige la
sous-couverture mesurée en 04d (spec §6.4/§6.5). Réutilise TELLE QUELLE la
Bank house-effects de `bayesian_nowcast` (calibrée sur 2017/2022), cf.
`excess_var_for_nodes` (model/models/spatial_pooling/model.py) pour la
justification de ce choix.

Exécution (un seul run, dispatché en parallèle) :
    .venv/bin/python notebooks/04e_spatial_coverage_excess.py <election> <cutoff_days> <half_life>
Agrégation :
    .venv/bin/python notebooks/04e_spatial_coverage_excess.py
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

_spec2 = importlib.util.spec_from_file_location("cov04d", "notebooks/04d_spatial_coverage_check.py")
cov04d = importlib.util.module_from_spec(_spec2)
sys.modules["cov04d"] = cov04d
_spec2.loader.exec_module(cov04d)

from model.core.bank import Bank
from model.core.inference import run_numpyro_mcmc
from model.models.bayesian_nowcast.calibration import BANK_PATH
from model.models.spatial_pooling.model import build_poll_arrays, excess_var_for_nodes, make_kappa, spatial_pooling_model, spatial_shares
from pipeline.historical import ELECTION_DATES

EVAL_WINDOW_DAYS = bt04b.EVAL_WINDOW_DAYS
RESULTS_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
LEVELS = cov04d.LEVELS
AGE_BUCKETS = cov04d.AGE_BUCKETS


def posterior_predictive_coverage_excess(mu_draws, sigma_draws, w_draws, tested_mask, Y, Np, excess_var_te,
                                         ages=None, seed=27) -> dict:
    """Comme `cov04d.posterior_predictive_coverage`, mais le bruit prédictif
    ajouté à chaque tirage inclut la variance d'excès du sondage TENU
    (`excess_var_te[p]`, PAS celle du fit -- l'institut qui a publié le
    sondage tenu peut différer de ceux du train)."""
    rng = np.random.default_rng(seed)
    P, N = tested_mask.shape
    covered = {50: [], 80: [], 90: []}
    widths = {50: [], 80: [], 90: []}
    by_age = {50: {}, 80: {}, 90: {}} if ages is not None else None
    for p in range(P):
        mask_p = tested_mask[p]
        if mask_p.sum() < 1:
            continue
        pi_draws = np.asarray(spatial_shares(jnp.asarray(mu_draws), jnp.asarray(sigma_draws),
                                             jnp.asarray(w_draws), jnp.asarray(mask_p)))
        var = np.clip(pi_draws * (1 - pi_draws), 1e-8, None) / max(Np[p], 1.0) + excess_var_te[p]
        y_pred = pi_draws + rng.normal(size=pi_draws.shape) * np.sqrt(var)
        bucket = None
        if ages is not None:
            for lo_b, hi_b in AGE_BUCKETS:
                if lo_b <= ages[p] < hi_b:
                    bucket = (lo_b, hi_b)
                    break
        for i in np.where(mask_p > 0)[0]:
            obs = Y[p, i]
            for level, (lo_q, hi_q) in LEVELS.items():
                lo, hi = np.percentile(y_pred[:, i], [lo_q, hi_q])
                ok = bool(lo <= obs <= hi)
                covered[level].append(ok)
                widths[level].append(float(hi - lo))
                if bucket is not None:
                    by_age[level].setdefault(bucket, []).append(ok)
    out = {
        "coverage": {lvl: float(np.mean(v)) for lvl, v in covered.items()},
        "width_moyenne_pt": {lvl: round(float(np.mean(w)) * 100, 2) for lvl, w in widths.items()},
        "n_obs": {lvl: len(v) for lvl, v in covered.items()},
    }
    if by_age is not None:
        out["coverage_par_age"] = {
            lvl: {f"{lo}-{hi}j": {"coverage": float(np.mean(v)), "n": len(v)} for (lo, hi), v in sorted(b.items())}
            for lvl, b in by_age.items()
        }
    return out


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

    bank = Bank.load(BANK_PATH)
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
        out_path = RESULTS_DIR / f"covexcess_{election}_{cutoff_days}_{half_life:.0f}.json"
        out_path.write_text(json.dumps(res))
        print(f"election={election} cutoff={cutoff_days} half_life={half_life:.0f} "
              f"-> {res} ({elapsed:.0f}s) écrit dans {out_path}")
        return

    results = []
    for election in (2017, 2022):
        for cutoff_days in bt04b.CUTOFFS_BY_ELECTION[election]:
            p = RESULTS_DIR / f"covexcess_{election}_{cutoff_days}_15.json"
            if not p.exists():
                print(f"(manquant : {p.name})")
                continue
            r = json.loads(p.read_text())
            if r is not None:
                results.append(r)

    print("\n=== Couverture AVEC variance d'excès (half_life=15j) ===")
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
