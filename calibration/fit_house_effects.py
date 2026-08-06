"""Phase 1 — calibration hiérarchique des house effects (§4, étape B).

On modélise l'écart historique `intention − résultat` (en points) par un modèle
bayésien hiérarchique à *partial pooling*, qui produit les priors informatifs
consommés par le modèle live 2027.

Structure du modèle
-------------------
Pour chaque observation i (un sondage × un candidat réel) :

    ecart_i ~ Normal( mu_i , sigma_i )

**Décomposition biais vs dérive temporelle (point clé).** L'écart brut
`intention(t) − résultat_final` mélange deux choses très différentes : le vrai
biais d'un institut (son erreur *le jour du vote*) et la dérive des préférences
entre la date du sondage et le scrutin (ex. Pécresse à 12 % en décembre 2021 puis
4,8 % en avril : ce n'est pas un biais d'institut, l'électorat a bougé). On les
sépare explicitement :

    mu_i     = biais[institut_i, bloc_i]  +  derive[election_i, bloc_i] * log(1 + horizon_i)
    biais[k, b]  ~ Normal( mu_bloc[b], tau_bloc[b] )         (non centré)
    mu_bloc[b]   ~ Normal(0, 5)
    tau_bloc[b]  ~ HalfNormal(3)
    derive[e, b] ~ Normal(0, 3)

Le biais `biais[institut, bloc]` est **partagé entre élections** (composante
structurelle transférable, hypothèse du §4), tandis que la dérive
`derive[election, bloc]` est **spécifique à chaque campagne** (Fillon 2017,
Pécresse 2022 : dynamiques d'effondrement différentes).

Comme `log(1 + horizon)` vaut **0 le jour du vote**, l'intercept
`biais[institut, bloc]` est le house effect *extrapolé à J-0* — la seule
quantité qu'on veut exporter comme prior. Le terme `derive[bloc]·log(1+h)`
absorbe le mouvement d'opinion systématique par bloc ; c'est un paramètre de
nuisance (spécifique à l'élection, non transférable), pas un biais. Un institut
qui voyait Pécresse juste à J-1 obtient un `biais` ≈ 0 même avec un gros écart à
J-200.

C'est aussi le cœur du §4 : un institut avec beaucoup d'historique sur un bloc
garde son biais propre ; un institut peu observé est ramené (*shrinkage*) vers la
moyenne du bloc `mu_bloc[b]` plutôt que de recevoir un prior arbitraire.

Variance décomposée en un **plancher d'échantillonnage connu** et un **excès** :

    Var[ecart_i]  = deff * p_i(100 - p_i)/n_i   +   excess_i^2
    excess_i      = exp( log_excess0 + s_inst[institut_i] + b_h * h_std_i )
    s_inst[k]     ~ Normal(0, tau_s)                          (non centré)

- `p_i(100-p_i)/n_i` est la variance d'échantillonnage irréductible du sondage
  (connue, pas estimée) ; le modèle ne peut donc pas confondre le simple bruit
  d'échantillon avec un house effect. Un biais de bloc n'est crédible que s'il
  dépasse ce plancher, se répète, et est propre à l'institut (partial pooling).
- `deff` ≥ 1 est le *design effect* des sondages par quotas (§4.4), estimé.
- `excess_i` est le bruit en excès (propre à l'institut via `s_inst`, croissant
  avec l'horizon via `b_h` — volatilité d'opinion, pas erreur d'échantillon).

Prévision 2027 en temps réel : la dérive d'une campagne future est inconnue mais
tirée de `Normal(0, tau_derive)`, donc à l'horizon h elle ajoute une variance
`tau_derive^2 * log(1+h)^2` aux intervalles — larges loin du scrutin, resserrés
près. `tau_derive` est calibré sur l'historique (c'est le vrai rôle de
l'historique côté variance temporelle).

Limite assumée : deux élections (2017, 2022) seulement, toutes deux 100 % en
ligne — un effet de mode (tél/internet) serait non identifiable et n'est donc pas
modélisé. Les intervalles sur les biais par institut × bloc restent larges, ce
que le hiérarchique rend visible au lieu de le masquer.

Sortie : calibration/priors.json (moyennes/écarts a posteriori) + trace .nc.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.historical import load_calibration_frame

OUT_DIR = Path(__file__).resolve().parent
TRACE_DIR = OUT_DIR / "traces"


def horizon_base(horizon, horizon_form: str = "log"):
    """Transforme l'horizon en régresseur, nul le jour du vote (J-0).

    - "log"  : log(1 + horizon)          — forme réduite pragmatique.
    - "sqrt" : sqrt(horizon)             — cohérent avec une marche aléatoire /
               mouvement brownien de l'opinion (variance ∝ temps), qui est
               l'hypothèse implicite des modèles dynamiques de la littérature
               (Linzer, The Economist, pollsposition).
    Les deux valent 0 à horizon 0, donc l'intercept reste le biais à J-0.
    """
    h = np.asarray(horizon, dtype=float)
    return np.sqrt(h) if horizon_form == "sqrt" else np.log1p(h)


# Hyperparamètres des priors (échelles). Valeurs par défaut = configuration de
# référence. Surchargées par l'analyse de sensibilité (model/backtest).
DEFAULT_PRIORS = {
    "mu_bloc_sd": 5.0,       # a priori sur le biais population par bloc
    "tau_bloc_sd": 3.0,      # dispersion inter-instituts du biais
    "tau_derive_sd": 1.0,    # amplitude de la dérive (pilote la largeur des IC lointains)
    "log_excess0_mu": np.log(2.0),
    "log_excess0_sd": 1.0,   # niveau de l'excès de variance
    "b_h_sd": 1.0,           # pente de l'excès vs horizon
    "tau_s_sd": 0.5,         # dispersion inter-instituts de la précision générale
}


def build_model(df: pd.DataFrame, deff: float = 1.0, horizon_form: str = "sqrt",
                priors_cfg: dict | None = None):
    """Construit le modèle.

    deff : design effect des sondages par quotas, appliqué au plancher
    d'échantillonnage (`Var = deff·p(100-p)/n + excess²`). **Constante connue**,
    pas estimée : deff et l'excès de variance se confondent (tous deux scalent la
    variance), donc estimer deff crée une crête non identifiable qui casse
    l'échantillonnage. On le fixe (1.0 = plancher binomial pur ; ~1.2–1.5 pour
    matérialiser l'effet quotas du §4.4) et on laisse l'excès absorber le reste.

    horizon_form : "sqrt" (défaut, cohérent marche aléatoire / littérature) ou "log".
    priors_cfg   : surcharge des échelles de priors (cf. DEFAULT_PRIORS).
    """
    cfg = {**DEFAULT_PRIORS, **(priors_cfg or {})}
    institut_codes, institut_index = pd.factorize(df["institut"], sort=True)
    bloc_codes, bloc_index = pd.factorize(df["bloc"], sort=True)
    elec_codes, elec_index = pd.factorize(df["election"], sort=True)

    # Régresseur d'horizon : sert à la fois de pente de dérive dans la moyenne
    # (brut, car nul à J-0 -> intercept = biais à J-0) et, une fois standardisé,
    # d'effet horizon sur la variance. On mémorise les paramètres de
    # standardisation pour reconstruire sigma(horizon) en 2027.
    log_h = horizon_base(df["horizon"].to_numpy(dtype=float), horizon_form)
    log_h_mean = float(log_h.mean())
    log_h_std = float(log_h.std())
    h_std = (log_h - log_h_mean) / log_h_std

    ecart = df["ecart"].to_numpy(dtype=float)

    # Variance d'échantillonnage connue par observation : p(100-p)/n (en points²).
    # C'est le plancher irréductible de bruit d'un sondage ; le modèle ne doit pas
    # l'attribuer à un house effect. n manquant -> médiane (imputation neutre).
    p = df["intention"].to_numpy(dtype=float)
    n = df["echantillon"].to_numpy(dtype=float)
    n_med = np.nanmedian(n[n > 0])
    n = np.where((~np.isfinite(n)) | (n <= 0), n_med, n)
    samp_var = p * (100.0 - p) / n

    coords = {
        "institut": list(institut_index),
        "bloc": list(bloc_index),
        "election": [int(e) for e in elec_index],
        "obs": np.arange(len(df)),
    }

    with pm.Model(coords=coords) as model:
        inst_i = pm.Data("inst_i", institut_codes, dims="obs")
        bloc_i = pm.Data("bloc_i", bloc_codes, dims="obs")
        elec_i = pm.Data("elec_i", elec_codes, dims="obs")
        h_i = pm.Data("h_i", h_std, dims="obs")            # horizon standardisé (excès de variance)
        logh_i = pm.Data("logh_i", log_h, dims="obs")      # base(horizon) brut, nul à J-0 (dérive)
        sv_i = pm.Data("samp_var_i", samp_var, dims="obs")  # variance d'échantillonnage connue

        # --- Biais (house effect à J-0) : partial pooling des instituts, par bloc.
        #     Partagé entre élections : composante structurelle transférable (§4). ---
        mu_bloc = pm.Normal("mu_bloc", mu=0.0, sigma=cfg["mu_bloc_sd"], dims="bloc")
        tau_bloc = pm.HalfNormal("tau_bloc", sigma=cfg["tau_bloc_sd"], dims="bloc")
        z_bias = pm.Normal("z_bias", mu=0.0, sigma=1.0, dims=("institut", "bloc"))
        biais = pm.Deterministic(
            "biais", mu_bloc[None, :] + z_bias * tau_bloc[None, :],
            dims=("institut", "bloc"),
        )

        # --- Dérive temporelle par (élection × bloc), HIÉRARCHIQUE et de moyenne
        #     nulle : chaque campagne tire sa dérive dans Normal(0, tau_derive).
        #     tau_derive (amplitude typique du mouvement d'opinion par unité de
        #     log-horizon) est ESTIMÉ -> pour 2027 la dérive inconnue apporte une
        #     variance tau_derive^2 * log(1+h)^2, d'où des IC qui s'élargissent
        #     loin du scrutin, calculables en temps réel. ---
        tau_derive = pm.HalfNormal("tau_derive", sigma=cfg["tau_derive_sd"])
        z_der = pm.Normal("z_der", mu=0.0, sigma=1.0, dims=("election", "bloc"))
        derive = pm.Deterministic("derive", z_der * tau_derive, dims=("election", "bloc"))

        # --- Variance : plancher d'échantillonnage connu (deff fixé) + excès
        #     (bruit propre à l'institut + volatilité croissante avec l'horizon). ---
        log_excess0 = pm.Normal("log_excess0", mu=cfg["log_excess0_mu"], sigma=cfg["log_excess0_sd"])
        tau_s = pm.HalfNormal("tau_s", sigma=cfg["tau_s_sd"])
        z_s = pm.Normal("z_s", mu=0.0, sigma=1.0, dims="institut")
        s_inst = pm.Deterministic("s_inst", z_s * tau_s, dims="institut")
        b_h = pm.Normal("b_h", mu=0.0, sigma=cfg["b_h_sd"])

        mu_i = biais[inst_i, bloc_i] + derive[elec_i, bloc_i] * logh_i
        excess_i = pm.math.exp(log_excess0 + s_inst[inst_i] + b_h * h_i)
        var_i = deff * sv_i + excess_i**2

        pm.Normal("y", mu=mu_i, sigma=pm.math.sqrt(var_i), observed=ecart, dims="obs")

    meta = {
        "institut_index": list(institut_index),
        "bloc_index": list(bloc_index),
        "election_index": [int(e) for e in elec_index],
        "log_h_mean": log_h_mean,
        "log_h_std": log_h_std,
        "n_obs": int(len(df)),
        "median_n": float(n_med),
        "deff": float(deff),
        "horizon_form": horizon_form,
    }
    return model, meta


def export_priors(idata, meta: dict) -> dict:
    """Résume la trace en priors informatifs pour le modèle live 2027."""
    post = idata.posterior

    def ms(name):
        v = post[name]
        return v.mean(dim=["chain", "draw"]), v.std(dim=["chain", "draw"])

    biais_m, biais_s = ms("biais")
    mub_m, mub_s = ms("mu_bloc")
    taub_m, _ = ms("tau_bloc")
    der_m, der_s = ms("derive")
    taud_m, taud_s = ms("tau_derive")
    sinst_m, sinst_s = ms("s_inst")
    le0_m, le0_s = ms("log_excess0")
    bh_m, bh_s = ms("b_h")

    instituts = meta["institut_index"]
    blocs = meta["bloc_index"]
    elections = meta["election_index"]

    priors = {
        "meta": {
            "description": "Priors informatifs calibrés (biais à J-0 + variance) pour le modèle live 2027.",
            "elections_utilisees": elections,
            "n_obs": meta["n_obs"],
            "median_n": meta["median_n"],
            "mean_model": "E[ecart] = biais[institut,bloc] + derive[election,bloc] * log1p(horizon)  (biais = house effect a J-0, partage entre elections)",
            "variance_model": "Var[ecart] = deff * p(100-p)/n + excess^2 ; excess = exp(log_excess0 + s_inst[institut] + b_h * z(horizon))",
            "forecast_2027_drift_var": "variance de derive inconnue a l'horizon h : tau_derive^2 * base(h)^2",
            "horizon_form": meta["horizon_form"],
            "horizon_transform": "base(h) = log1p(h) [log] ou sqrt(h) [sqrt] ; z = (base - log_h_mean) / log_h_std",
            "log_h_mean": meta["log_h_mean"],
            "log_h_std": meta["log_h_std"],
            "unite": "points de pourcentage",
        },
        "variance_globale": {
            "log_excess0": {"mean": float(le0_m), "sd": float(le0_s)},
            "b_horizon": {"mean": float(bh_m), "sd": float(bh_s)},
            "design_effect_quotas": {"mean": meta["deff"], "sd": 0.0, "note": "constante fixée, non estimée"},
            "tau_derive": {"mean": float(taud_m), "sd": float(taud_s)},
        },
        "derive_temporelle": {
            # dérive réalisée par (élection × bloc) — audit rétrospectif seulement.
            # Pour 2027, utiliser tau_derive (variance_globale) comme incertitude
            # a priori de la dérive, PAS ces valeurs (spécifiques aux campagnes passées).
            str(e): {
                b: {"mean": float(der_m.sel(election=e, bloc=b)),
                    "sd": float(der_s.sel(election=e, bloc=b))}
                for b in blocs
            }
            for e in elections
        },
        "bloc": {
            b: {
                "biais_moyen_population": {"mean": float(mub_m.sel(bloc=b)),
                                           "sd": float(mub_s.sel(bloc=b))},
                "dispersion_inter_instituts": float(taub_m.sel(bloc=b)),
            }
            for b in blocs
        },
        "institut": {
            inst: {
                "precision_generale_log": {"mean": float(sinst_m.sel(institut=inst)),
                                           "sd": float(sinst_s.sel(institut=inst))},
                "biais_par_bloc": {
                    b: {"mean": float(biais_m.sel(institut=inst, bloc=b)),
                        "sd": float(biais_s.sel(institut=inst, bloc=b))}
                    for b in blocs
                },
            }
            for inst in instituts
        },
    }
    return priors


def fit(df, draws: int = 500, tune: int = 500, chains: int = 2, seed: int = 27,
        target_accept: float = 0.9, sampler: str = "numpyro", deff: float = 1.0,
        horizon_form: str = "sqrt", priors_cfg: dict | None = None,
        progressbar: bool = True):
    """Ajuste le modèle sur `df` et renvoie (idata, meta). Réutilisé par le
    backtest (fit sur une élection, prédiction sur l'autre)."""
    sample_kwargs = dict(
        draws=draws, tune=tune, chains=chains,
        target_accept=target_accept, random_seed=seed, progressbar=progressbar,
    )
    if sampler == "numpyro":
        sample_kwargs["nuts_sampler"] = "numpyro"
    else:
        sample_kwargs["cores"] = chains

    model, meta = build_model(df, deff=deff, horizon_form=horizon_form,
                              priors_cfg=priors_cfg)
    with model:
        idata = pm.sample(**sample_kwargs)
    return idata, meta


def main(draws: int = 1000, tune: int = 1000, chains: int = 4, seed: int = 27,
         target_accept: float = 0.95, sampler: str = "numpyro",
         horizon_form: str = "sqrt") -> None:
    """Calibre le modèle et exporte les priors.

    Défaut = config prod : √h (cohérent marche aléatoire / littérature),
    1000/1000 × 4 via NumPyro (HMC/JAX). Viser R-hat < 1.01 + ESS élevé.

    sampler : "numpyro" (JAX, rapide) ou "pymc" (repli sans JAX).
    """
    df = load_calibration_frame()
    print(f"Calibration sur {len(df)} observations "
          f"({df['institut'].nunique()} instituts × {df['bloc'].nunique()} blocs) "
          f"— sampler={sampler}, horizon={horizon_form}, {chains}×{draws} (tune {tune}).")

    idata, meta = fit(df, draws=draws, tune=tune, chains=chains, seed=seed,
                      target_accept=target_accept, sampler=sampler,
                      horizon_form=horizon_form)

    # diagnostics compacts
    import arviz as az

    summary = az.summary(
        idata,
        var_names=["mu_bloc", "derive", "tau_derive", "tau_bloc",
                   "log_excess0", "b_h", "tau_s"],
        round_to=3,
    )
    print("\n=== Diagnostics (paramètres de population) ===")
    print(summary.to_string())
    max_rhat = float(az.rhat(idata).max().to_array().max())
    ess_bulk_min = float(az.ess(idata).min().to_array().min())
    n_div = int(idata.sample_stats["diverging"].sum())
    print(f"\nR-hat max : {max_rhat:.4f} | ESS bulk min : {ess_bulk_min:.0f} "
          f"| divergences : {n_div}")

    # Déliverable principal : les priors informatifs. Écrit en premier.
    priors = export_priors(idata, meta)
    priors["meta"]["rhat_max"] = max_rhat
    priors["meta"]["ess_bulk_min"] = ess_bulk_min
    priors["meta"]["n_divergences"] = n_div
    with open(OUT_DIR / "priors.json", "w", encoding="utf-8") as f:
        json.dump(priors, f, ensure_ascii=False, indent=2)
    print(f"\nPriors écrits dans {OUT_DIR / 'priors.json'}")

    # Trace complète (optionnelle) : utile pour ré-analyse, mais nécessite un
    # backend netcdf (h5py). On ne bloque pas la calibration si absent.
    try:
        TRACE_DIR.mkdir(exist_ok=True)
        idata.to_netcdf(TRACE_DIR / "house_effects.nc")
        print(f"Trace écrite dans {TRACE_DIR / 'house_effects.nc'}")
    except Exception as exc:  # pragma: no cover
        print(f"(Trace .nc non sauvegardée : {exc})")


if __name__ == "__main__":
    main()
