"""Prototype spatial sur données RÉELLES (2027) — filtre les candidats trop
peu sondés, impose l'ordre gauche-droite fourni à la main (par GROUPE, pas
par candidat individuel : les alternates d'un même groupe -- Le Pen/Bardella,
Attal seul, Roussel/Tondelier/Glucksmann/Hollande... -- partagent un point
d'ancrage avec un delta libre, cf. notebooks/_spatial_core.py).

Filtre : >= 5 sondages RÉELS distincts (`notice`), sur la fenêtre de
campagne 2026+ (même `MIN_POLL_DATE` que le SSM en production, cf.
model/models/bayesian_nowcast/nowcast.py -- les sondages "prémonitoires"
2022/2023 sont un régime différent).

Half-life : passé en argument, valeur par défaut = résultat du backtest
2017/2022 (notebooks/04b_spatial_halflife_backtest.py) une fois disponible ;
en attendant, valeur de repli raisonnable ci-dessous.

Exécution : .venv/bin/python notebooks/04_spatial_real_data.py [half_life_jours]
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
from notebooks._spatial_core import build_poll_arrays, make_kappa, spatial_model_ordered
from model.core.inference import run_numpyro_mcmc
from model.core.live_dataset import load_raw_polls

MIN_POLL_DATE = pd.Timestamp("2026-01-01")
MIN_POLLS = 5

# Ordre fourni : NPA < LFI < {écolo/PS/coco} < Attal < Philippe < LR < MLP < Zemmour.
# Candidats non explicitement listés, rattachés par défaut (À VALIDER) :
#   - Dupont-Aignan -> groupe MLP (même bloc "droite_radicale" que Le Pen/RN
#     dans data/historical/candidat_blocs.csv) ;
#   - Bardella -> groupe MLP (alternate RN de Le Pen) ;
#   - Hollande -> groupe {écolo/PS/coco} (PS).
ORDER_GROUPS: list[tuple[str, list[str]]] = [
    ("NPA",              ["Arthaud", "Poutou"]),
    ("LFI",              ["Mélenchon", "Ruffin"]),
    ("ecolo_ps_coco",    ["Roussel", "Tondelier", "Glucksmann", "Hollande", "Faure"]),
    ("Attal",            ["Attal"]),
    ("Philippe",         ["Philippe"]),
    ("LR",                ["Retailleau", "Wauquiez"]),
    ("MLP",              ["Le Pen", "Bardella", "Dupont-Aignan"]),
    ("Zemmour",          ["Zemmour"]),
]


def build_roster(raw: pd.DataFrame) -> tuple[list[str], np.ndarray, list[str]]:
    counts = raw.groupby("candidat")["notice"].nunique()
    eligible = set(counts[counts >= MIN_POLLS].index)

    candidates, slot_of, slot_names = [], [], []
    for gi, (gname, members) in enumerate(ORDER_GROUPS):
        for m in members:
            if m in eligible:
                candidates.append(m)
                slot_of.append(gi)
        slot_names.append(gname)

    dropped = eligible - set(candidates)
    if dropped:
        print(f"(candidats éligibles mais absents de ORDER_GROUPS, à classer : {sorted(dropped)})")
    return candidates, np.array(slot_of), slot_names


def main():
    half_life = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0

    raw = load_raw_polls()
    raw = raw[pd.to_datetime(raw["date_fin"]) >= MIN_POLL_DATE].copy()
    candidates, slot_of, slot_names = build_roster(raw)
    print(f"{len(candidates)} candidats retenus (>= {MIN_POLLS} sondages, depuis {MIN_POLL_DATE.date()}) :")
    for gi, gname in enumerate(slot_names):
        members = [c for c, s in zip(candidates, slot_of) if s == gi]
        if members:
            print(f"  [{gi}] {gname:16s} : {', '.join(members)}")

    arrays = build_poll_arrays(raw, candidates)
    P, N = arrays["tested_mask"].shape
    as_of = arrays["dates"].max()
    print(f"\n{P} noeuds sondage x hypothèse, {N} candidats, half_life={half_life:.0f}j")

    kappa = make_kappa(arrays["dates"], as_of=as_of, half_life=half_life)
    kwargs = dict(
        slot_of=jnp.asarray(slot_of), n_slots=len(slot_names),
        tested_mask=jnp.asarray(arrays["tested_mask"]), Y=jnp.asarray(arrays["Y"]),
        Np=jnp.asarray(arrays["Np"]), kappa=jnp.asarray(kappa),
    )
    t0 = time.time()
    samples, extra = run_numpyro_mcmc(spatial_model_ordered, kwargs, draws=1000, tune=1000,
                                      chains=4, seed=27, target_accept=0.95,
                                      extra_fields=("diverging",))
    elapsed = time.time() - t0

    diag = numpyro_summary(samples, prob=0.9)
    rhat_max = max(float(np.max(d["r_hat"])) for d in diag.values())
    ess_min = min(float(np.min(d["n_eff"])) for d in diag.values())
    n_div = int(np.sum(extra["diverging"]))
    print(f"\ntemps={elapsed:.0f}s | R-hat max={rhat_max:.3f} | ESS min={ess_min:.0f} | "
          f"divergences={n_div}/{samples['mu'].shape[0] * samples['mu'].shape[1]}")

    mu_hat = np.asarray(samples["mu"]).reshape(-1, N).mean(axis=0)
    sigma_hat = np.asarray(samples["sigma"]).reshape(-1, N).mean(axis=0)
    w_hat = np.asarray(samples["w_now"]).reshape(-1, N).mean(axis=0)
    pi_full = np.asarray(samples["pi"]).reshape(-1, P, N)

    order = np.argsort(mu_hat)
    print("\nPositions estimées (triées gauche -> droite) :")
    print(f"{'candidat':16s} {'groupe':16s} {'mu':>6s} {'sigma':>6s} {'w_now':>7s}")
    for i in order:
        print(f"{candidates[i]:16s} {slot_names[slot_of[i]]:16s} {mu_hat[i]:6.3f} "
              f"{sigma_hat[i]:6.3f} {w_hat[i]:7.3f}")

    # nowcast "tous testés ensemble" -- juste pour lire une part de vote
    # comparable par candidat (pas un vrai scénario de sondage).
    all_mask = np.ones(N)
    from notebooks._spatial_core import weighted_loglik
    pi_all, _ = weighted_loglik(jnp.asarray(mu_hat), jnp.asarray(sigma_hat), jnp.asarray(w_hat),
                                jnp.asarray(all_mask[None, :]), jnp.zeros((1, N)),
                                jnp.array([1000.0]), jnp.array([1.0]))
    print("\nPart de vote implicite si tous testés ensemble (lecture seule, PAS un vrai scénario) :")
    for i in np.argsort(-np.asarray(pi_all)[0]):
        print(f"  {candidates[i]:16s} {float(pi_all[0, i]) * 100:5.1f}%")


if __name__ == "__main__":
    main()
