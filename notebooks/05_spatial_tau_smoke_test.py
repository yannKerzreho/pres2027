"""Test de faisabilité du modèle à diffusion réelle (`spatial_pooling_model_tau`,
model/models/spatial_pooling/model.py) sur les vraies données 2027 -- avant de
lancer le backtest de couverture complet. Vérifie : ça tourne, R-hat/ESS/
divergences, temps, et que mu/sigma restent cohérents avec le modèle tempéré
déjà validé (04c_spatial_sanity_check.py).

Exécution : .venv/bin/python notebooks/05_spatial_tau_smoke_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro.diagnostics import summary as numpyro_summary

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.core.inference import run_numpyro_mcmc
from model.models.spatial_pooling.model import MIN_POLL_DATE, build_poll_arrays, build_roster, spatial_pooling_model_tau
from model.core.live_dataset import load_raw_polls


def main():
    raw = load_raw_polls()
    raw = raw[pd.to_datetime(raw["date_fin"]) >= MIN_POLL_DATE].copy()
    candidates, slot_of, slot_names = build_roster(raw)
    N = len(candidates)
    arrays = build_poll_arrays(raw, candidates)
    P = arrays["tested_mask"].shape[0]
    M = len(arrays["unique_dates"])
    as_of = arrays["dates"].max()   # "as_of" = dernier jour observé, pas d'extrapolation ici
    as_of_dt = as_of - arrays["unique_dates"][-1]

    print(f"{N} candidats, {P} noeuds, {M} dates uniques -> dimension ajoutée N*M={N*M}")

    kwargs = dict(
        slot_of=jnp.asarray(slot_of), n_slots=len(slot_names),
        tested_mask=jnp.asarray(arrays["tested_mask"]), Y=jnp.asarray(arrays["Y"]),
        Np=jnp.asarray(arrays["Np"]), date_idx=jnp.asarray(arrays["date_idx"]),
        dt_gaps=jnp.asarray(arrays["dt_gaps"]), as_of_dt=float(as_of_dt),
    )
    t0 = time.time()
    samples, extra = run_numpyro_mcmc(spatial_pooling_model_tau, kwargs, draws=1000, tune=1000,
                                      chains=4, seed=27, target_accept=0.95,
                                      extra_fields=("diverging",))
    elapsed = time.time() - t0

    diag = numpyro_summary(samples, prob=0.9)
    rhat_max = max(float(np.max(d["r_hat"])) for d in diag.values())
    ess_min = min(float(np.min(d["n_eff"])) for d in diag.values())
    n_div = int(np.sum(extra["diverging"]))
    print(f"temps={elapsed:.0f}s | R-hat max={rhat_max:.3f} | ESS min={ess_min:.0f} | "
          f"divergences={n_div}/{samples['mu'].shape[0]*samples['mu'].shape[1]}")

    tau_mean = float(np.asarray(samples["tau"]).mean())
    tau_sd = float(np.asarray(samples["tau"]).std())
    sigma0_mean = float(np.asarray(samples["sigma0_w"]).mean())
    print(f"tau (diffusion, unites w/jour^0.5) : {tau_mean:.4f} +/- {tau_sd:.4f}")
    print(f"sigma0_w (etat initial) : {sigma0_mean:.4f}")

    mu_hat = np.asarray(samples["mu"]).reshape(-1, N).mean(axis=0)
    sigma_hat = np.asarray(samples["sigma"]).reshape(-1, N).mean(axis=0)
    w_hat = np.asarray(samples["w_now"]).reshape(-1, N).mean(axis=0)

    order = np.argsort(mu_hat)
    print(f"\n{'candidat':16s} {'groupe':12s} {'mu':>6s} {'sigma':>6s} {'w_now':>7s}")
    for i in order:
        print(f"{candidates[i]:16s} {slot_names[slot_of[i]]:12s} {mu_hat[i]:6.3f} "
              f"{sigma_hat[i]:6.3f} {w_hat[i]:7.3f}")


if __name__ == "__main__":
    main()
