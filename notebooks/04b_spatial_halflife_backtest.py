"""Backtest du half-life (vraisemblance tempérée de w_now, cf.
notebooks/_spatial_core.py) sur les VRAIES campagnes 2017 et 2022.

Protocole (nowcast hors-échantillon, même esprit que
model/backtest/backtest_loo.py mais sur la récence plutôt que le biais
institut) : à plusieurs dates de coupure `as_of` dans chaque campagne, on fit
le modèle spatial sur les sondages disponibles AVANT la coupure pour
plusieurs valeurs de half_life, puis on note le log-score des sondages
RÉELLEMENT publiés dans les 21 jours suivants (jamais vus du fit) sous la
part de vote prédite. Le half_life qui prédit le mieux hors-échantillon
gagne -- pas celui qui fit le mieux en-échantillon.

Blocs (pas les candidats individuels, plus léger pour ce balayage) :
gauche_radicale < gauche < ecologistes < centre < droite < droite_radicale
(bloc "divers" exclu -- ne s'inscrit pas franchement sur l'axe gauche-droite).

Exécution : .venv/bin/python notebooks/04b_spatial_halflife_backtest.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from notebooks._spatial_core import build_poll_arrays, make_kappa, spatial_model_ordered, weighted_loglik
from model.core.inference import run_numpyro_mcmc
from pipeline.historical import DATA_DIR, ELECTION_DATES, PARSED_DIR, POLL_FILES, _resolve_candidate, load_blocs

BLOC_ORDER = ["gauche_radicale", "gauche", "ecologistes", "centre", "droite", "droite_radicale"]
# Coupure PAR ÉLECTION : la source wiki couvre très inégalement 2017 (19
# sondages sur toute la fenêtre de campagne) et 2022 (198) -- à J-60, 2017
# n'a que 3 sondages en train (< MIN_POLLS), retenu à J-30 seulement (5 train,
# 10 test). Réduit aussi par rapport au plan initial (coût NUTS réel mesuré
# ~5-6min/fit -- un balayage plus large serait trop long).
CUTOFFS_BY_ELECTION = {2017: [30], 2022: [60]}
EVAL_WINDOW_DAYS = 21
HALF_LIVES = [15.0, 45.0, 9999.0]
RESULTS_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)


# Blocs des candidats TESTÉS mais jamais candidats (spec §12.21). `candidat_blocs.csv`
# ne contient que ceux qui se sont réellement présentés — or les sondages 2021
# testaient massivement des hypothèses qui ne se sont pas concrétisées : Bertrand
# pesait 15 % sur 59 sondages, Wauquiez 10,8 %, Barnier 9,2 %, Ciotti 5,3 %. Les
# écarter creusait un trou de 10 à 20 % des intentions, concentré sur la droite,
# c'est-à-dire précisément la zone que le modèle doit apprendre à distinguer.
# Affectations à confirmer ; les quatre dernières sont discutables (cf. §12.21).
BLOCS_HYPOTHETIQUES = {
    "Montebourg": "gauche", "Taubira": "gauche",
    "Lagarde": "centre", "Jean Lassalle": "centre",   # nom résolu : il s'est présenté
    "Juvin": "droite", "Barnier": "droite", "Bertrand": "droite",
    "Wauquiez": "droite", "Ciotti": "droite", "Poisson": "droite",
    "Philippot": "droite_radicale", "Asselineau": "droite_radicale",
    "Thouy": "ecologistes",
}
MIN_NOTICES = 5


def load_historical_long(election: int, as_of=None) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Roster construit depuis les SONDAGES (>= MIN_NOTICES sondages distincts),
    pas depuis la liste des candidats effectivement partis — même règle que
    `model.build_roster` sur 2027, et pour la même raison : ce modèle sert à
    explorer des hypothèses, donc tout candidat réellement testé compte.

    `as_of` : date de coupure pour le COMPTAGE des sondages (spec §12.24). Sans
    elle, `MIN_NOTICES` est appliqué sur les 400 jours entiers et le roster d'un
    backtest connaît l'avenir : à J-150, Taubira et Thouy y entrent avec **zéro**
    nœud (ils ne sont testés qu'après la coupure), et jusqu'à 5 candidats sur
    zéro nœud à J-240/J-300. Ce sont des paramètres de prior pur, et c'est une
    fuite. Les lignes renvoyées ne sont PAS tronquées : l'appelant découpe
    train/test comme il l'entend, seul le roster est arrêté à `as_of`."""
    polls = pd.read_csv(PARSED_DIR / POLL_FILES[election])
    t1 = polls[polls["tour"] == "Premier tour"].copy()

    blocs = load_blocs()
    blocs_e = blocs[blocs["election"] == election]
    real = set(blocs_e["candidat"])
    bloc_of_real = dict(zip(blocs_e["candidat"], blocs_e["bloc"]))

    date_fin = pd.to_datetime(t1["date_fin"], errors="coerce")
    t1["date_fin"] = date_fin.fillna(pd.to_datetime(t1.get("date_notice"), errors="coerce"))
    t1 = t1[t1["date_fin"].notna()]
    elec_date = pd.Timestamp(ELECTION_DATES[election])
    t1 = t1[(t1["date_fin"] > elec_date - pd.Timedelta(days=400)) & (t1["date_fin"] <= elec_date)]

    # nom canonique : celui de la table quand le candidat s'est présenté, sinon
    # le nom brut du sondage (Bertrand, Barnier... n'existent pas dans la table)
    t1["candidat"] = [
        r if (_resolve_candidate(r, real) not in real) else _resolve_candidate(r, real)
        for r in t1["candidat"]
    ]
    vus = t1 if as_of is None else t1[t1["date_fin"] <= pd.Timestamp(as_of)]
    counts = vus.groupby("candidat")["notice"].nunique()
    eligible = [c for c in counts[counts >= MIN_NOTICES].index]

    def bloc_of(c):
        # la table locale PRIME : elle sert justement à replacer sur l'axe des
        # candidats que `candidat_blocs.csv` classe `divers` (Lassalle, 169
        # sondages à 1,3 %), lequel n'appartient à aucun bloc ordonné et les
        # ferait retomber dans le trou de normalisation qu'on cherche à fermer.
        return BLOCS_HYPOTHETIQUES.get(c) or bloc_of_real.get(c)

    non_classes = [c for c in eligible if bloc_of(c) is None]
    if non_classes:
        raise ValueError(f"candidats éligibles sans bloc : {sorted(non_classes)} — "
                         "les ajouter à BLOCS_HYPOTHETIQUES")
    candidates = sorted(c for c in eligible if bloc_of(c) in BLOC_ORDER)
    t1 = t1[t1["candidat"].isin(candidates)]
    slot_of = np.array([BLOC_ORDER.index(bloc_of(c)) for c in candidates])
    return t1, candidates, slot_of


def fit_and_score(train_df, test_df, candidates, slot_of, half_life, election_date, draws=300, tune=300, chains=2):
    arrays_tr = build_poll_arrays(train_df, candidates)
    if arrays_tr["tested_mask"].shape[0] < 5:
        return None
    as_of = arrays_tr["dates"].max()
    kappa = make_kappa(arrays_tr["dates"], as_of=as_of, half_life=half_life)
    kwargs = dict(
        slot_of=jnp.asarray(slot_of), n_slots=len(BLOC_ORDER),
        tested_mask=jnp.asarray(arrays_tr["tested_mask"]), Y=jnp.asarray(arrays_tr["Y"]),
        Np=jnp.asarray(arrays_tr["Np"]), kappa=jnp.asarray(kappa),
    )
    samples, _ = run_numpyro_mcmc(spatial_model_ordered, kwargs, draws=draws, tune=tune,
                                  chains=chains, seed=27, target_accept=0.9)
    N = len(candidates)
    mu_hat = np.asarray(samples["mu"]).reshape(-1, N).mean(axis=0)
    sigma_hat = np.asarray(samples["sigma"]).reshape(-1, N).mean(axis=0)
    w_hat = np.asarray(samples["w_now"]).reshape(-1, N).mean(axis=0)

    arrays_te = build_poll_arrays(test_df, candidates)
    if arrays_te["tested_mask"].shape[0] == 0:
        return None
    pi_pred, _ = weighted_loglik(jnp.asarray(mu_hat), jnp.asarray(sigma_hat), jnp.asarray(w_hat),
                                 jnp.asarray(arrays_te["tested_mask"]),
                                 jnp.zeros_like(jnp.asarray(arrays_te["Y"])),
                                 jnp.asarray(arrays_te["Np"]), jnp.ones(arrays_te["tested_mask"].shape[0]))
    pi_pred = np.asarray(pi_pred)
    mask = arrays_te["tested_mask"].astype(bool)
    var = np.clip(pi_pred * (1 - pi_pred), 1e-6, None) / np.clip(arrays_te["Np"][:, None], 1.0, None)
    z = (arrays_te["Y"] - pi_pred) / np.sqrt(var)
    logscore = float(np.mean(-0.5 * np.log(2 * np.pi * var[mask]) - 0.5 * z[mask] ** 2))
    return logscore, int(mask.sum())


def run_one(election: int, cutoff_days: int, half_life: float) -> dict | None:
    """Un seul (élection, coupure, half_life) -- pour dispatcher en parallèle
    (chaque fit prend ~5-6 min, un balayage séquentiel serait trop long)."""
    df, candidates, slot_of = load_historical_long(election)
    elec_date = pd.Timestamp(ELECTION_DATES[election])
    cutoff_date = elec_date - pd.Timedelta(days=cutoff_days)
    eval_end = cutoff_date + pd.Timedelta(days=EVAL_WINDOW_DAYS)
    train_df = df[df["date_fin"] <= cutoff_date]
    test_df = df[(df["date_fin"] > cutoff_date) & (df["date_fin"] <= eval_end)]
    if train_df["notice"].nunique() < 5 or test_df.empty:
        return None
    out = fit_and_score(train_df, test_df, candidates, slot_of, half_life, elec_date)
    if out is None:
        return None
    logscore, n_obs = out
    return dict(election=election, cutoff_days=cutoff_days, half_life=half_life,
               logscore=logscore, n_obs=n_obs)


def main():
    if len(sys.argv) == 4:
        # mode "un seul fit", dispatché par le sweep parallèle
        election, cutoff_days, half_life = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
        t0 = time.time()
        res = run_one(election, cutoff_days, half_life)
        elapsed = time.time() - t0
        out_path = RESULTS_DIR / f"hl_{election}_{cutoff_days}_{half_life:.0f}.json"
        out_path.write_text(json.dumps(res))
        print(f"election={election} cutoff={cutoff_days} half_life={half_life:.0f} "
              f"-> {res} ({elapsed:.0f}s) écrit dans {out_path}")
        return

    # mode agrégation : relit les JSON déjà produits par les runs individuels
    results = []
    for election in (2017, 2022):
        for cutoff_days in CUTOFFS_BY_ELECTION[election]:
            for hl in HALF_LIVES:
                p = RESULTS_DIR / f"hl_{election}_{cutoff_days}_{hl:.0f}.json"
                if not p.exists():
                    print(f"(manquant : {p.name})")
                    continue
                r = json.loads(p.read_text())
                if r is not None:
                    results.append(r)

    res = pd.DataFrame(results)
    print(res.to_string(index=False))
    print("\n=== Moyenne du logscore par half_life (poids égal par (élection, coupure)) ===")
    agg = res.groupby("half_life")["logscore"].mean().sort_values(ascending=False)
    print(agg.to_string())
    print(f"\n-> half_life recommandé : {agg.index[0]:.0f}j")


if __name__ == "__main__":
    main()
