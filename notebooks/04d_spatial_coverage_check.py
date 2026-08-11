"""Vérification de la COUVERTURE des IC50/80/90 du modèle spatial (spec
spatial_pooling.md §6.3, "à faire") -- complète le backtest logscore
(04b_spatial_halflife_backtest.py) : logscore dit "en moyenne, la vraisemblance
prédictive est bonne", couverture dit "les intervalles qu'on afficherait
vraiment sont-ils fiables" (un intervalle trop étroit peut avoir un bon
logscore moyen tout en ratant sa couverture nominale).

Protocole (même train/test que 04b, réutilise `load_historical_long`) : pour
chaque observation test (candidat x hypothèse tenue hors du fit), construit
la distribution PRÉDICTIVE POSTÉRIEURE complète -- pas juste la moyenne --
en poussant TOUS les tirages postérieurs (mu,sigma,w) à travers le softmax
masqué (`spatial_shares`, model/models/spatial_pooling/model.py) PUIS en
ajoutant le bruit d'échantillonnage du sondage tenu (même variance que la
vraisemblance du fit, `pi(1-pi)/N_p`) -- pas une approximation gaussienne
séparée, littéralement ce que le modèle prédit pour "un nouveau sondage".
Vérifie si le score RÉELLEMENT rapporté tombe dans les IC50/80/90 empiriques
de cette distribution.

Exécution (un seul run, dispatché en parallèle comme 04b) :
    .venv/bin/python notebooks/04d_spatial_coverage_check.py <election> <cutoff_days> <half_life>
Agrégation :
    .venv/bin/python notebooks/04d_spatial_coverage_check.py
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

from model.core.inference import run_numpyro_mcmc
from model.models.spatial_pooling.model import build_poll_arrays, make_kappa, spatial_pooling_model, spatial_shares
from pipeline.historical import ELECTION_DATES

EVAL_WINDOW_DAYS = bt04b.EVAL_WINDOW_DAYS
RESULTS_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)

LEVELS = {50: (25, 75), 80: (10, 90), 90: (5, 95)}


AGE_BUCKETS = [(0, 7), (7, 14), (14, 21)]   # jours depuis la coupure (as_of du fit)


def posterior_predictive_coverage(mu_draws, sigma_draws, w_draws, tested_mask, Y, Np, ages=None,
                                  seed=27) -> dict:
    """Pour chaque (noeud test, candidat testé) : pousse TOUS les tirages
    (mu,sigma,w) à travers spatial_shares -> pi_draws (S,), ajoute le bruit
    d'échantillonnage du sondage tenu (Y ~ N(pi, pi(1-pi)/N_p), la vraie
    hypothèse de vraisemblance du modèle) -> distribution prédictive
    complète. Vérifie si l'observation réelle tombe dans les IC empiriques.

    `ages` (P,) optionnel : âge (jours depuis `as_of` du fit) de chaque noeud
    test -- si fourni, la couverture est aussi cassée par tranche d'âge
    (`AGE_BUCKETS`) pour tester si l'incertitude prédictive devrait grandir
    avec le temps écoulé depuis le fit (hypothèse "diffusion de w manquante",
    cf. session -- pas de terme de marche aléatoire explicite sur w_now, donc
    l'incertitude prédictive ne grandit PAS avec l'horizon de prévision, ce
    qui est le comportement attendu si cette hypothèse est la cause dominante
    de la sous-couverture)."""
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
                                             jnp.asarray(w_draws), jnp.asarray(mask_p)))   # (S,N)
        var = np.clip(pi_draws * (1 - pi_draws), 1e-8, None) / max(Np[p], 1.0)
        y_pred = pi_draws + rng.normal(size=pi_draws.shape) * np.sqrt(var)                 # (S,N) prédictif
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
            lvl: {f"{lo}-{hi}j": {"coverage": float(np.mean(v)), "n": len(v)}
                 for (lo, hi), v in sorted(buckets.items())}
            for lvl, buckets in by_age.items()
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
    kwargs = dict(
        slot_of=jnp.asarray(slot_of), n_slots=len(bt04b.BLOC_ORDER),
        tested_mask=jnp.asarray(arrays_tr["tested_mask"]), Y=jnp.asarray(arrays_tr["Y"]),
        Np=jnp.asarray(arrays_tr["Np"]), kappa=jnp.asarray(kappa),
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
    ages = arrays_te["dates"] - cutoff_date.toordinal()
    cov = posterior_predictive_coverage(mu_draws, sigma_draws, w_draws,
                                        arrays_te["tested_mask"], arrays_te["Y"], arrays_te["Np"], ages=ages)
    return dict(election=election, cutoff_days=cutoff_days, half_life=half_life, **cov)


def main():
    if len(sys.argv) == 4:
        election, cutoff_days, half_life = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
        t0 = time.time()
        res = run_one(election, cutoff_days, half_life)
        elapsed = time.time() - t0
        out_path = RESULTS_DIR / f"covage_{election}_{cutoff_days}_{half_life:.0f}.json"
        out_path.write_text(json.dumps(res))
        print(f"election={election} cutoff={cutoff_days} half_life={half_life:.0f} "
              f"-> {res} ({elapsed:.0f}s) écrit dans {out_path}")
        return

    results = []
    for election in (2017, 2022):
        for cutoff_days in bt04b.CUTOFFS_BY_ELECTION[election]:
            p = RESULTS_DIR / f"cov_{election}_{cutoff_days}_15.json"
            if not p.exists():
                print(f"(manquant : {p.name})")
                continue
            r = json.loads(p.read_text())
            if r is not None:
                results.append(r)

    print("\n=== Couverture empirique des IC50/80/90 (half_life=15j, nominal en regard) ===")
    for r in results:
        print(f"\n{r['election']} (coupure J-{r['cutoff_days']}) :")
        for lvl in (50, 80, 90):
            n = r["n_obs"][str(lvl)] if str(lvl) in r["n_obs"] else r["n_obs"].get(lvl)
            cov = r["coverage"].get(str(lvl), r["coverage"].get(lvl))
            width = r["width_moyenne_pt"].get(str(lvl), r["width_moyenne_pt"].get(lvl))
            print(f"  IC{lvl} : couverture empirique = {cov*100:5.1f}% (nominal {lvl}%), "
                  f"largeur moyenne = {width:.1f} pt, n={n}")

    if results:
        print("\n=== Moyenne pondérée sur les deux élections ===")
        for lvl in (50, 80, 90):
            tot_n = sum(r["n_obs"].get(str(lvl), r["n_obs"].get(lvl)) for r in results)
            tot_cov = sum(r["coverage"].get(str(lvl), r["coverage"].get(lvl)) * r["n_obs"].get(str(lvl), r["n_obs"].get(lvl)) for r in results)
            print(f"  IC{lvl} : {tot_cov/tot_n*100:5.1f}% (nominal {lvl}%, n total={tot_n})")


if __name__ == "__main__":
    main()
