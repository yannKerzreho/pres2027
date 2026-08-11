"""Prototype exploratoire — modèle spatial (Hotelling-Downs), sur données SYNTHÉTIQUES.

Valide, AVANT de toucher au pipeline réel, les trois choix de design actés en
session de conception (backlog "a priori d'expert" / reports de voix) :

  1. mu_i/sigma_i hiérarchiques : ancrés sur un "analogue historique" FIXE
     (mu_analogue/sigma_analogue, jouant le rôle d'une Bank 2017/2022 déjà
     calibrée) + delta gaussien non centré, `sd_delta_mu`/`sd_delta_sigma`
     ESTIMÉS par NUTS (pas fixés à la main — cf. discussion : rien ne garantit
     qu'un successeur soit "proche" de son prédécesseur). Comme l'ancrage est
     une CONSTANTE (pas un site NumPyro), aucune contrainte d'ordre (`Ordered`,
     spec §6) n'est nécessaire dans ce modèle live : l'ordre gauche-droite est
     hérité tel quel de la calibration historique.

  2. w_i("now") : PAS de state-space séquentiel (pas de filtre, pas de
     `lax.scan`) — une vraisemblance TEMPÉRÉE (poids de récence
     `2^(-Δt/half_life)`) sur toutes les hypothèses de sondage disponibles,
     entièrement vectorisée. `half_life` est un scalaire FIXÉ ici (pas
     échantillonné — une vraisemblance tempérée n'a pas d'interprétation
     générative propre, cf. discussion ; à calibrer hors-NUTS par backtest
     dans une itération suivante). On balaie plusieurs valeurs pour observer
     l'effet.

  3. Sondages partiels : masque MULTIPLICATIF (0/1) plutôt que `-inf` dans le
     softmax — plus stable numériquement, et permet de tout vectoriser
     (P, N, B) sans boucle/scan.

Ce script simule un petit monde (N=8 candidats, 4 "blocs" de 2 alternates
chacun — pour tester le cas Attal/Philippe : jamais co-testés, mais reliés
indirectement via les candidats communs des autres blocs) où la vérité
terrain est connue, y compris un candidat en "surge" tardif (dernier bloc,
2e candidat) pour vérifier que la pondération par récence le rattrape mieux
qu'un pooling uniforme (half_life infini = équivalent à tout peser pareil).

Vérifie : (a) recouvrement de mu/sigma/w_now, (b) santé NUTS (R-hat/ESS/
divergences), (c) temps d'exécution, (d) sensibilité au choix de half_life.

Exécution : .venv/bin/python notebooks/03_spatial_prototype.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary as numpyro_summary

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.core.inference import run_numpyro_mcmc

rng = np.random.default_rng(27)

# --- Config -------------------------------------------------------------------
N = 8                      # candidats (4 blocs x 2 alternates)
B = 50                     # bins de la grille politique
T = 220                    # jours de "campagne" simulée
P = 90                     # hypothèses de sondage (sondages partiels)
BLOCS = np.array([0, 0, 1, 1, 2, 2, 3, 3])   # paire = alternates du même bloc
NAMES = [f"C{i}" for i in range(N)]
SURGE_CANDIDATE = 7        # dernier candidat : montée tardive (30 derniers jours)

V = jnp.linspace(0.01, 0.99, B)
W = jnp.ones(B) / B


# --- 1. Vérité terrain ----------------------------------------------------------
def simulate_truth():
    bloc_centers = np.array([0.15, 0.42, 0.68, 0.88])
    true_mu = bloc_centers[BLOCS] + rng.normal(0, 0.05, N)
    true_sigma = np.clip(0.15 + rng.normal(0, 0.03, N), 0.06, None)

    # w_true(t) : marche aléatoire journalière de faible volatilité, PLUS un
    # "surge" tardif déterministe pour un candidat -- ce que le vrai monde
    # produirait (ex. RN qui accélère en fin de campagne) et que le modèle
    # (qui suppose w_now ~ constant, cf. docstring du module) doit quand même
    # rattraper via la pondération de récence, pas via une dynamique explicite.
    steps = rng.normal(0, 0.012, size=(T, N))
    w_path = np.cumsum(steps, axis=0) + rng.normal(0, 0.25, N)[None, :]
    ramp = np.clip((np.arange(T) - (T - 30)) / 30.0, 0, 1)
    w_path[:, SURGE_CANDIDATE] += 1.4 * ramp

    # analogue historique : proche de la vérité MAIS PAS identique (cf. la
    # mise en garde de la session : Attal/Philippe != Macron + bruit tout
    # petit par hypothèse, ici on le simule volontairement avec un écart non
    # négligeable pour vérifier que sd_delta l'absorbe sans être fixé à la main).
    true_delta_mu_sd, true_delta_sigma_sd = 0.045, 0.08
    mu_analogue = true_mu + rng.normal(0, true_delta_mu_sd, N)
    sigma_analogue = true_sigma * np.exp(rng.normal(0, true_delta_sigma_sd, N))

    return dict(true_mu=true_mu, true_sigma=true_sigma, w_path=w_path,
                mu_analogue=mu_analogue, sigma_analogue=sigma_analogue)


def sample_tested_subset() -> np.ndarray:
    """Un sondage 2027 ne teste jamais deux alternates du même bloc ensemble
    (Attal OU Philippe, jamais les deux) -- cf. discussion sur SLOTS."""
    mask = np.zeros(N, dtype=bool)
    for b in range(4):
        i, j = 2 * b, 2 * b + 1
        r = rng.random()
        if r < 0.45:
            mask[i] = True
        elif r < 0.90:
            mask[j] = True
    if mask.sum() < 2:
        return sample_tested_subset()
    return mask


def true_pi(mu, sigma, w, tested_mask) -> np.ndarray:
    D = -((np.asarray(V)[None, :] - mu[:, None]) ** 2) / (2 * sigma[:, None] ** 2)
    A = w[:, None] + D
    A = A - A.max(axis=0, keepdims=True)
    expA = np.exp(A) * tested_mask[:, None]
    Pib = expA / expA.sum(axis=0, keepdims=True)
    return (Pib * np.asarray(W)[None, :]).sum(axis=1)


def simulate_polls(truth: dict):
    dates = np.concatenate([
        rng.uniform(0, T - 40, int(P * 0.4)),
        rng.uniform(T - 40, T, int(P * 0.6)),
    ])
    dates = np.sort(dates)
    tested_mask = np.zeros((P, N))
    Y = np.zeros((P, N))
    Np = rng.integers(700, 1500, P).astype(float)

    for p in range(P):
        mask = sample_tested_subset()
        tested_mask[p] = mask
        t = int(np.clip(dates[p], 0, T - 1))
        w_t = truth["w_path"][t]
        pi = true_pi(truth["true_mu"], truth["true_sigma"], w_t, mask)
        noisy = pi + rng.normal(0, np.sqrt(np.clip(pi * (1 - pi), 1e-6, None) / Np[p]))
        Y[p] = np.clip(noisy, 1e-4, None) * mask

    return dict(dates=dates, tested_mask=tested_mask, Y=Y, Np=Np)


# --- 2. Modèle NumPyro ----------------------------------------------------------
def spatial_model(mu_analogue, sigma_analogue, tested_mask, Y, Np, kappa,
                  sd_delta_mu_scale=0.1, sd_delta_sigma_scale=0.15, sd_w_scale=0.6):
    N_ = mu_analogue.shape[0]

    sd_delta_mu = numpyro.sample("sd_delta_mu", dist.HalfNormal(sd_delta_mu_scale))
    sd_delta_sigma = numpyro.sample("sd_delta_sigma", dist.HalfNormal(sd_delta_sigma_scale))
    z_mu = numpyro.sample("z_mu", dist.Normal(0.0, 1.0).expand([N_]))
    z_sigma = numpyro.sample("z_sigma", dist.Normal(0.0, 1.0).expand([N_]))
    mu = numpyro.deterministic("mu", mu_analogue + z_mu * sd_delta_mu)
    sigma = numpyro.deterministic("sigma", sigma_analogue * jnp.exp(z_sigma * sd_delta_sigma))

    sd_w = numpyro.sample("sd_w", dist.HalfNormal(sd_w_scale))
    z_w = numpyro.sample("z_w", dist.Normal(0.0, 1.0).expand([N_]))
    w_now = numpyro.deterministic("w_now", z_w * sd_w)

    D = -((V[None, :] - mu[:, None]) ** 2) / (2 * sigma[:, None] ** 2)   # (N,B)
    A = w_now[:, None] + D
    A = A - jnp.max(A, axis=0, keepdims=True)                            # softmax stable, poll-agnostique
    expA = jnp.exp(A)                                                    # (N,B)

    numerator = expA[None, :, :] * tested_mask[:, :, None]               # (P,N,B)
    denom = jnp.clip(numerator.sum(axis=1, keepdims=True), 1e-12, None)  # (P,1,B)
    Pib = numerator / denom
    pi = jnp.sum(Pib * W[None, None, :], axis=2)                         # (P,N)

    var = pi * (1.0 - pi) / jnp.clip(Np[:, None], 1.0, None) + 1e-8
    ll = dist.Normal(pi, jnp.sqrt(var)).log_prob(Y) * tested_mask        # (P,N)
    numpyro.factor("weighted_ll", jnp.sum(kappa[:, None] * ll))


def make_kappa(dates: np.ndarray, as_of: float, half_life: float) -> np.ndarray:
    if half_life >= 9999:
        return np.ones_like(dates)
    return 2.0 ** (-(as_of - dates) / half_life)


# --- 3. Fit + diagnostics --------------------------------------------------------
def run_once(truth, polls, half_life: float, draws=800, tune=800, chains=4):
    kappa = make_kappa(polls["dates"], as_of=T, half_life=half_life)
    kwargs = dict(
        mu_analogue=jnp.asarray(truth["mu_analogue"]),
        sigma_analogue=jnp.asarray(truth["sigma_analogue"]),
        tested_mask=jnp.asarray(polls["tested_mask"]),
        Y=jnp.asarray(polls["Y"]),
        Np=jnp.asarray(polls["Np"]),
        kappa=jnp.asarray(kappa),
    )
    t0 = time.time()
    samples, extra = run_numpyro_mcmc(spatial_model, kwargs, draws=draws, tune=tune,
                                      chains=chains, seed=27, target_accept=0.9,
                                      extra_fields=("diverging",))
    elapsed = time.time() - t0

    diag = numpyro_summary(samples, prob=0.9)
    rhat_max = max(float(np.max(d["r_hat"])) for d in diag.values())
    ess_min = min(float(np.min(d["n_eff"])) for d in diag.values())
    n_div = int(np.sum(extra["diverging"]))

    mu_hat = np.asarray(samples["mu"]).reshape(-1, N).mean(axis=0)
    sigma_hat = np.asarray(samples["sigma"]).reshape(-1, N).mean(axis=0)
    w_hat = np.asarray(samples["w_now"]).reshape(-1, N).mean(axis=0)
    w_sd = np.asarray(samples["w_now"]).reshape(-1, N).std(axis=0)

    w_true_now = truth["w_path"][T - 1]
    return dict(elapsed=elapsed, rhat_max=rhat_max, ess_min=ess_min, n_div=n_div,
                mu_err=mu_hat - truth["true_mu"], sigma_err=sigma_hat - truth["true_sigma"],
                w_err=w_hat - w_true_now, w_hat=w_hat, w_sd=w_sd, w_true=w_true_now)


def report(label: str, r: dict):
    print(f"\n--- {label} ---")
    print(f"temps={r['elapsed']:.1f}s | R-hat max={r['rhat_max']:.3f} | "
          f"ESS min={r['ess_min']:.0f} | divergences={r['n_div']}")
    print(f"mu   : erreur abs moyenne = {np.abs(r['mu_err']).mean():.4f} "
          f"(max {np.abs(r['mu_err']).max():.4f})")
    print(f"sigma: erreur abs moyenne = {np.abs(r['sigma_err']).mean():.4f} "
          f"(max {np.abs(r['sigma_err']).max():.4f})")
    print(f"w_now: erreur abs moyenne = {np.abs(r['w_err']).mean():.4f} "
          f"(candidat en surge, C{SURGE_CANDIDATE} : "
          f"vrai={r['w_true'][SURGE_CANDIDATE]:.3f}, "
          f"estimé={r['w_hat'][SURGE_CANDIDATE]:.3f}±{r['w_sd'][SURGE_CANDIDATE]:.3f})")


def main():
    truth = simulate_truth()
    polls = simulate_polls(truth)
    print(f"{P} hypothèses simulées, {int(polls['tested_mask'].sum())} observations "
          f"candidat×hypothèse ({polls['tested_mask'].sum(axis=1).mean():.1f} candidats/hyp. en moyenne)")
    print(f"Candidat surge (C{SURGE_CANDIDATE}) : w_true(t=0)={truth['w_path'][0, SURGE_CANDIDATE]:.3f} "
          f"-> w_true(t={T})={truth['w_path'][-1, SURGE_CANDIDATE]:.3f}")

    for hl in (15.0, 30.0, 60.0, 9999.0):
        label = f"half_life={hl:.0f}j" if hl < 9999 else "half_life=∞ (pooling uniforme)"
        r = run_once(truth, polls, half_life=hl)
        report(label, r)


if __name__ == "__main__":
    main()
