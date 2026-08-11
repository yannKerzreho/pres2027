"""Sanity check du prototype spatial sur données RÉELLES 2027 (half_life=15j,
recommandé par le backtest, cf. notebooks/04b_spatial_halflife_backtest.py).

Question : les scores prédits par le modèle (mu/sigma/w_now fittés une fois)
sont-ils cohérents avec ce que les instituts rapportent RÉELLEMENT, sondage
par sondage, hypothèse par hypothèse (chacune teste un sous-ensemble
DIFFÉRENT de candidats -- Le Pen ou Bardella, avec ou sans Attal...) ? On
recalcule pi UNIQUEMENT à partir du sous-ensemble réellement testé par
chaque hypothèse (comme le ferait le modèle en prévision), puis on compare
au chiffre publié -- pas de fuite d'information (pi ne dépend pas de Y ici,
seulement de mu/sigma/w_now déjà fittés sur l'ensemble des données).

Exécution : .venv/bin/python notebooks/04c_spatial_sanity_check.py
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro.diagnostics import summary as numpyro_summary

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

spec = importlib.util.spec_from_file_location("realdata", "notebooks/04_spatial_real_data.py")
realdata = importlib.util.module_from_spec(spec)
sys.modules["realdata"] = realdata
spec.loader.exec_module(realdata)

from notebooks._spatial_core import build_poll_arrays, make_kappa, spatial_model_ordered, weighted_loglik
from model.core.inference import run_numpyro_mcmc
from model.core.live_dataset import load_raw_polls

HALF_LIFE = 15.0
N_CHECK_HYP = 10   # nombre d'hypothèses réelles vérifiées (les plus récentes)


def main():
    raw = load_raw_polls()
    raw = raw[pd.to_datetime(raw["date_fin"]) >= realdata.MIN_POLL_DATE].copy()
    candidates, slot_of, slot_names = realdata.build_roster(raw)
    N = len(candidates)
    arrays = build_poll_arrays(raw, candidates)
    P = arrays["tested_mask"].shape[0]
    as_of = arrays["dates"].max()

    kappa = make_kappa(arrays["dates"], as_of=as_of, half_life=HALF_LIFE)
    kwargs = dict(
        slot_of=jnp.asarray(slot_of), n_slots=len(slot_names),
        tested_mask=jnp.asarray(arrays["tested_mask"]), Y=jnp.asarray(arrays["Y"]),
        Np=jnp.asarray(arrays["Np"]), kappa=jnp.asarray(kappa),
    )
    t0 = time.time()
    samples, extra = run_numpyro_mcmc(spatial_model_ordered, kwargs, draws=1000, tune=1000,
                                      chains=4, seed=27, target_accept=0.95,
                                      extra_fields=("diverging",))
    print(f"fit : {time.time()-t0:.0f}s, {P} noeuds, {N} candidats, half_life={HALF_LIFE:.0f}j")

    diag = numpyro_summary(samples, prob=0.9)
    rhat_max = max(float(np.max(d["r_hat"])) for d in diag.values())
    n_div = int(np.sum(extra["diverging"]))
    print(f"R-hat max={rhat_max:.3f} | divergences={n_div}/{samples['mu'].shape[0]*samples['mu'].shape[1]}")

    mu_hat = np.asarray(samples["mu"]).reshape(-1, N).mean(axis=0)
    sigma_hat = np.asarray(samples["sigma"]).reshape(-1, N).mean(axis=0)
    w_hat = np.asarray(samples["w_now"]).reshape(-1, N).mean(axis=0)

    # pi prédit pour CHAQUE noeud réel (son propre sous-ensemble testé), à
    # partir des paramètres fittés sur l'ENSEMBLE des données -- pas de fuite
    # (mu/sigma/w_now ne dépendent pas de Y d'un noeud particulier plus que
    # des autres, c'est le postérieur joint habituel).
    pi_pred, _ = weighted_loglik(jnp.asarray(mu_hat), jnp.asarray(sigma_hat), jnp.asarray(w_hat),
                                 jnp.asarray(arrays["tested_mask"]), jnp.zeros_like(jnp.asarray(arrays["Y"])),
                                 jnp.asarray(arrays["Np"]), jnp.ones(P))
    pi_pred = np.asarray(pi_pred)

    order = np.argsort(-arrays["dates"])[:N_CHECK_HYP]   # les N_CHECK_HYP hypothèses les plus RÉCENTES
    print(f"\n=== Comparaison sondage vs modèle sur les {N_CHECK_HYP} hypothèses les plus récentes ===")
    errs = []
    for p in order:
        mask = arrays["tested_mask"][p].astype(bool)
        d = pd.Timestamp.fromordinal(int(arrays["dates"][p]))
        tested = [candidates[i] for i in range(N) if mask[i]]
        print(f"\n-- {d.date()} ({len(tested)} candidats testés) --")
        for i in range(N):
            if not mask[i]:
                continue
            rep, pred = arrays["Y"][p, i] * 100, pi_pred[p, i] * 100
            err = pred - rep
            errs.append(err)
            print(f"  {candidates[i]:16s} rapporté={rep:5.1f}%  modèle={pred:5.1f}%  écart={err:+5.1f}")

    errs = np.array(errs)
    print(f"\n=== Résumé ({len(errs)} observations candidat x hypothèse) ===")
    print(f"erreur absolue moyenne = {np.abs(errs).mean():.2f} pt, "
          f"médiane = {np.median(np.abs(errs)):.2f} pt, max = {np.abs(errs).max():.2f} pt")
    print(f"biais (moyenne signée) = {errs.mean():+.2f} pt")
    within_3 = (np.abs(errs) <= 3.0).mean() * 100
    print(f"part des écarts <= 3 points : {within_3:.0f}%")

    # Cohérence Bardella vs Le Pen (même groupe MLP, alternates jamais co-testés) :
    # le modèle doit refléter la tendance réelle observée (Bardella teste
    # systématiquement plus haut que Le Pen, cf. session de conception initiale).
    i_bardella, i_lepen = candidates.index("Bardella"), candidates.index("Le Pen")
    mask_b = arrays["tested_mask"][:, i_bardella].astype(bool)
    mask_l = arrays["tested_mask"][:, i_lepen].astype(bool)
    print(f"\nBardella : rapporté moyen={arrays['Y'][mask_b, i_bardella].mean()*100:.1f}%, "
          f"modèle moyen={pi_pred[mask_b, i_bardella].mean()*100:.1f}% "
          f"(mu={mu_hat[i_bardella]:.3f}, w={w_hat[i_bardella]:+.2f})")
    print(f"Le Pen   : rapporté moyen={arrays['Y'][mask_l, i_lepen].mean()*100:.1f}%, "
          f"modèle moyen={pi_pred[mask_l, i_lepen].mean()*100:.1f}% "
          f"(mu={mu_hat[i_lepen]:.3f}, w={w_hat[i_lepen]:+.2f})")


if __name__ == "__main__":
    main()
