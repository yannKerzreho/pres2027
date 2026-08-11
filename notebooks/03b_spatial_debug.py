"""Diagnostic ponctuel : l'écart sur w_now (03_spatial_prototype.py) vient-il
d'un vrai problème d'identification, ou d'un artefact de pooling hiérarchique
(sd_w PARTAGÉ entre les 8 candidats, qui écrase un outlier) ? Et surtout :
est-ce que ça se répercute sur pi (la quantité utile pour la prévision) ?

Exécution : .venv/bin/python notebooks/03b_spatial_debug.py
"""
import importlib.util
import sys
import time

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

spec = importlib.util.spec_from_file_location("proto", "notebooks/03_spatial_prototype.py")
proto = importlib.util.module_from_spec(spec)
sys.modules["proto"] = proto
spec.loader.exec_module(proto)

from model.core.inference import run_numpyro_mcmc

N, T, V, W = proto.N, proto.T, proto.V, proto.W
SURGE = proto.SURGE_CANDIDATE


def model_indep_w(mu_analogue, sigma_analogue, tested_mask, Y, Np, kappa,
                  sd_delta_mu_scale=0.1, sd_delta_sigma_scale=0.15, w_scale=1.5):
    """Variante : w_now SANS hiérarchie partagée -- prior fixe large par
    candidat (pas de sd_w commun qui pourrait écraser un outlier)."""
    N_ = mu_analogue.shape[0]
    sd_delta_mu = numpyro.sample("sd_delta_mu", dist.HalfNormal(sd_delta_mu_scale))
    sd_delta_sigma = numpyro.sample("sd_delta_sigma", dist.HalfNormal(sd_delta_sigma_scale))
    z_mu = numpyro.sample("z_mu", dist.Normal(0.0, 1.0).expand([N_]))
    z_sigma = numpyro.sample("z_sigma", dist.Normal(0.0, 1.0).expand([N_]))
    mu = numpyro.deterministic("mu", mu_analogue + z_mu * sd_delta_mu)
    sigma = numpyro.deterministic("sigma", sigma_analogue * jnp.exp(z_sigma * sd_delta_sigma))

    w_now = numpyro.sample("w_now", dist.Normal(0.0, w_scale).expand([N_]))

    D = -((V[None, :] - mu[:, None]) ** 2) / (2 * sigma[:, None] ** 2)
    A = w_now[:, None] + D
    A = A - jnp.max(A, axis=0, keepdims=True)
    expA = jnp.exp(A)
    numerator = expA[None, :, :] * tested_mask[:, :, None]
    denom = jnp.clip(numerator.sum(axis=1, keepdims=True), 1e-12, None)
    Pib = numerator / denom
    pi = jnp.sum(Pib * W[None, None, :], axis=2)
    var = pi * (1.0 - pi) / jnp.clip(Np[:, None], 1.0, None) + 1e-8
    ll = dist.Normal(pi, jnp.sqrt(var)).log_prob(Y) * tested_mask
    numpyro.factor("weighted_ll", jnp.sum(kappa[:, None] * ll))


def pi_from_params(mu, sigma, w, tested_mask):
    return proto.true_pi(np.asarray(mu), np.asarray(sigma), np.asarray(w), tested_mask)


def main():
    truth = proto.simulate_truth()
    polls = proto.simulate_polls(truth)
    kappa = proto.make_kappa(polls["dates"], as_of=T, half_life=15.0)
    kwargs = dict(
        mu_analogue=jnp.asarray(truth["mu_analogue"]), sigma_analogue=jnp.asarray(truth["sigma_analogue"]),
        tested_mask=jnp.asarray(polls["tested_mask"]), Y=jnp.asarray(polls["Y"]),
        Np=jnp.asarray(polls["Np"]), kappa=jnp.asarray(kappa),
    )

    # "now" = un sous-ensemble représentatif testant le candidat en surge, pour comparer pi
    now_mask = polls["tested_mask"][polls["tested_mask"][:, SURGE] == 1][-1]
    true_pi_now = pi_from_params(truth["true_mu"], truth["true_sigma"], truth["w_path"][-1], now_mask)

    for label, model_fn, extra in (
        ("baseline (sd_w partagé, hiérarchique)", proto.spatial_model, {}),
        ("w_now indépendant (prior large, pas de pooling)", model_indep_w, {}),
    ):
        t0 = time.time()
        samples, _ = run_numpyro_mcmc(model_fn, {**kwargs, **extra}, draws=500, tune=500,
                                      chains=4, seed=27, target_accept=0.9)
        elapsed = time.time() - t0
        mu_hat = np.asarray(samples["mu"]).reshape(-1, N).mean(axis=0)
        sigma_hat = np.asarray(samples["sigma"]).reshape(-1, N).mean(axis=0)
        w_hat = np.asarray(samples["w_now"]).reshape(-1, N).mean(axis=0)
        pi_hat_now = pi_from_params(mu_hat, sigma_hat, w_hat, now_mask)

        print(f"\n--- {label} ({elapsed:.0f}s) ---")
        print(f"w_now candidat surge (C{SURGE}) : vrai={truth['w_path'][-1, SURGE]:.3f}, "
              f"estimé={w_hat[SURGE]:.3f}")
        print(f"pi (part de vote) candidat surge, 'now' : vrai={true_pi_now[SURGE]:.3f}, "
              f"estimé={pi_hat_now[SURGE]:.3f}  (écart={pi_hat_now[SURGE]-true_pi_now[SURGE]:+.3f})")
        err_pi = np.abs(pi_hat_now - true_pi_now)[now_mask.astype(bool)]
        print(f"pi : erreur absolue moyenne (candidats testés) = {err_pi.mean():.4f}, max={err_pi.max():.4f}")


if __name__ == "__main__":
    main()
