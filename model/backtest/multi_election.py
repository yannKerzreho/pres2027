"""Modèle spatial JOINT multi-élections (spec_second_tour §8.47, piste R3b).

Un seul modèle pour plusieurs élections : chacune garde sa vraisemblance (T1,
duels, reports, nœud du résultat), et partage avec les autres l'arc du cercle,
les écarts d'ancre des personnes récurrentes et κ (hiérarchiques).

    python -m model.backtest.multi_election --config A_cercle --chaine k   # NUTS (GPU)
    python -m model.backtest.multi_election --config A_cercle --evaluer    # sans NUTS

Les chaînes sont sauvées PAR ÉLECTION sous les noms de sites habituels
(`results/chaines/multi_<config>_e<année>_c<k>.npz`) : la reconstruction et
l'évaluation d'une élection passent par `second_round._fit` et
`_runner_fusion` tels quels.
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro import handlers

import model.backtest.second_round as sr
from model.core.inference import run_numpyro_mcmc
from model.models.spatial_pooling.fit import _nom_court
from model.models.spatial_pooling.joint_model import spatial_pooling_model_ou
from model.models.spatial_pooling.roster import ORDER_GROUPS_HISTORIQUES, build_roster, split_tours

VARIANTES = {
    "hier": {"eta_hier": True},
    "hier_rep": {"eta_hier": True, "_reports": True},
    "hiercercle": {"eta_hier": True, "carte_cercle": True, "_reports": True},
    "hiercercle_sr": {"eta_hier": True, "carte_cercle": True},
    # §8.47.5 : + reports publiés avant le T1
    "hiercercle_full": {"eta_hier": True, "carte_cercle": True, "_reports": True, "_reports_avant": True},
    "hiercercle_ra": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True},
}

# (élection, coupure, variante) ; « complet » = T2−1 + nœud du résultat T1.
CONFIGS = {
    "A_cercle": {"cercle": True, "blocs": [(2017, "complet", "hiercercle"), (2022, "T1-30", "hiercercle_sr")]},
    "A_segment": {"cercle": False, "blocs": [(2017, "complet", "hier_rep"), (2022, "T1-30", "hier")]},
    "B_cercle": {"cercle": True, "blocs": [(2022, "complet", "hiercercle"), (2017, "T1-30", "hiercercle_sr")]},
    "A2_cercle": {"cercle": True, "blocs": [(2017, "complet", "hiercercle_full"), (2022, "T1-30", "hiercercle_ra")]},
    "B_segment": {"cercle": False, "blocs": [(2022, "complet", "hier_rep"), (2017, "T1-30", "hier")]},
}


class _Capture(Exception):
    pass


def as_of_de(election: int, coupure: str) -> pd.Timestamp:
    if coupure == "complet":
        return sr.T2_DATES[election] - pd.Timedelta(days=1)
    ref = sr.T1_DATES[election] if coupure.startswith("T1") else sr.T2_DATES[election]
    return ref - pd.Timedelta(days=int(coupure.split("-")[1]))


def donnees(election: int, coupure: str) -> tuple[pd.DataFrame, pd.Timestamp]:
    as_of = as_of_de(election, coupure)
    raw = sr.polls_historiques(election, as_of)
    if coupure == "complet":
        raw = sr.avec_noeud_resultat(raw, election)
    return raw, as_of


def regler_variante(variante: str) -> None:
    sr.ABSTENTION = True
    sr.SECOND_TOUR = True
    sr.VARIANTE_KW.clear()
    sr.VARIANTE_KW.update(VARIANTES[variante])


def preparer(election: int, coupure: str, variante: str) -> dict:
    """Arguments du modèle d'UNE élection, exactement ceux de `second_round._fit`."""
    regler_variante(variante)
    raw, as_of = donnees(election, coupure)
    cap = {}

    def capturer(model_fn, model_kwargs, **kw):
        cap["kwargs"] = model_kwargs
        raise _Capture

    try:
        sr._fit(raw, as_of, election, dict(draws=400, tune=600, chains=1, mcmc_runner=capturer))
    except _Capture:
        pass
    raw1, _ = split_tours(raw)
    candidats, _, _ = build_roster(raw1, as_of=str(as_of.date()),
                                   order_groups=ORDER_GROUPS_HISTORIQUES[election])
    assert len(candidats) == len(cap["kwargs"]["pos_anchor"]), "roster incohérent avec le fit"
    return {"election": election, "coupure": coupure, "variante": variante, "tag": f"e{election}",
            "kwargs": cap["kwargs"], "candidats": list(candidats)}


def personnes(blocs: list[dict]) -> list[str]:
    """Noms de famille normalisés présents dans au moins deux élections."""
    from collections import Counter
    n = Counter(nom for b in blocs for nom in {_nom_court(c) for c in b["candidats"]})
    return sorted(k for k, v in n.items() if v >= 2)


def indices_personnes(blocs: list[dict], pers: list[str]) -> None:
    for b in blocs:
        b["pidx"] = np.array([pers.index(_nom_court(c)) if _nom_court(c) in pers else -1
                              for c in b["candidats"]])


def modele_multi(blocs, pers, cercle):
    """§8.47.1 : partage de l'arc, des écarts d'ancre des personnes récurrentes, de κ."""
    P = len(pers)
    zb_mu = numpyro.sample("zbar_mu", dist.Normal(0.0, 1.0).expand([P]))
    zb_sg = numpyro.sample("zbar_sigma", dist.Normal(0.0, 1.0).expand([P]))
    s_mu = numpyro.sample("s_mu", dist.HalfNormal(0.5))
    s_sg = numpyro.sample("s_sigma", dist.HalfNormal(0.5))
    log_kb = numpyro.sample("log_kappa_bar", dist.Normal(0.0, 0.5))
    s_k = numpyro.sample("s_kappa", dist.HalfNormal(0.3))
    arc = None
    if cercle:
        a = numpyro.sample("a_arc_commun", dist.Normal(jnp.log(0.75 / 0.25), 0.75))
        arc = numpyro.deterministic("arc_commun", 2 * jnp.pi * jax.nn.sigmoid(a))
    for b in blocs:
        e, n = b["tag"], len(b["candidats"])
        has = jnp.asarray(b["pidx"] >= 0)
        ix = jnp.asarray(np.where(b["pidx"] >= 0, b["pidx"], 0))
        eps_mu = numpyro.sample(f"{e}_eps_mu", dist.Normal(0.0, 1.0).expand([n]))
        eps_sg = numpyro.sample(f"{e}_eps_sigma", dist.Normal(0.0, 1.0).expand([n]))
        eps_k = numpyro.sample(f"{e}_eps_kappa", dist.Normal(0.0, 1.0))
        z_mu = jnp.where(has, zb_mu[ix] + s_mu * eps_mu, eps_mu)
        z_sg = jnp.where(has, zb_sg[ix] + s_sg * eps_sg, eps_sg)
        kappa = jnp.exp(log_kb + s_k * eps_k)
        with handlers.scope(prefix=e, divider="__"):
            spatial_pooling_model_ou(**b["kwargs"], z_pos_fixe=z_mu, z_sigma_fixe=z_sg,
                                     arc_fixe=arc, kappa_fixe=kappa)


def construire(config: str) -> tuple[list[dict], list[str], bool]:
    c = CONFIGS[config]
    blocs = [preparer(*t) for t in c["blocs"]]
    pers = personnes(blocs)
    indices_personnes(blocs, pers)
    return blocs, pers, c["cercle"]


def stem(config: str, election: int) -> str:
    return f"multi_{config}_e{election}"


def lancer_chaine(config: str, k: int, draws: int = 400, tune: int = 600) -> None:
    blocs, pers, cercle = construire(config)
    fn = partial(modele_multi, blocs=blocs, pers=pers, cercle=cercle)
    samples, extra = run_numpyro_mcmc(fn, {}, draws=draws, tune=tune, chains=1, seed=27 + k,
                                      target_accept=0.9,
                                      extra_fields=("diverging", "num_steps", "potential_energy"))
    sr.CHAINES_DIR.mkdir(parents=True, exist_ok=True)
    ex = {f"e__{n}": np.asarray(v).reshape(-1) for n, v in extra.items() if n != "potential_energy"}
    for b in blocs:
        pre = b["tag"] + "__"
        s = {f"s__{n[len(pre):]}": np.asarray(v) for n, v in samples.items() if n.startswith(pre)}
        np.savez_compressed(sr.CHAINES_DIR / f"{stem(config, b['election'])}_c{k}.npz", **s, **ex)
    partages = {n: np.asarray(v) for n, v in samples.items() if "__" not in n}
    np.savez_compressed(sr.CHAINES_DIR / f"multi_{config}_partage_c{k}.npz",
                        **{f"s__{n}": v for n, v in partages.items()},
                        potential_energy=np.asarray(extra["potential_energy"]).reshape(-1),
                        personnes=np.array(pers))
    print(json.dumps({"config": config, "chaine": k, "personnes": pers,
                      "lp_joint": float(-np.asarray(extra["potential_energy"]).mean()),
                      "divergences": int(np.sum(extra["diverging"])),
                      "pas_median": int(np.median(np.asarray(extra["num_steps"])))}, ensure_ascii=False))


def evaluer(config: str) -> dict:
    """Sans NUTS. Élection prédite (la dernière du bloc, coupure T1−30) : critères §8.47.4."""
    from model.backtest import eval_r2
    c = CONFIGS[config]
    e, coupure, variante = c["blocs"][-1]
    eval_r2.ELECTION = e
    st = stem(config, e)
    out = {"spec": "spec_second_tour §8.47.4", "config": config, "election_predite": e,
           "coupure": coupure, "variante": variante, "stem": st}
    zs = [np.load(sr.CHAINES_DIR / f"multi_{config}_partage_c{k}.npz") for k in range(sr.N_CHAINES)]
    out["lp_joint_par_chaine"] = [round(float(-z["potential_energy"].mean()), 1) for z in zs]
    out["ecart_lp_joint"] = round(max(out["lp_joint_par_chaine"]) - min(out["lp_joint_par_chaine"]), 1)
    out["partages_hyper"] = {n[3:]: [round(float(np.asarray(z[n]).mean()), 3) for z in zs]
                             for n in zs[0].files if n.startswith("s__") and np.asarray(zs[0][n]).ndim <= 2}
    fit, raw, as_of = eval_r2.reconstruire(variante, st, coupure)
    ev = eval_r2.evaluer(fit, raw, as_of)
    out["global"] = ev
    out["par_chaine"] = eval_r2.par_chaine(variante, st, coupure)
    mes = {"Jean-Luc Mélenchon": 63.0 if e == 2022 else 76.4}
    part = {r["electorat"]: (r["modele"], r["mesure"]) for r in ev["reports"] if r["cellule"] == "partage"}
    jlm = part.get("Jean-Luc Mélenchon")
    ecarts_duels = out["par_chaine"]["ecart_duels"]
    out["criteres"] = {
        "1_melenchon_lt10": None if jlm is None else bool(abs(jlm[0] - jlm[1]) < 10),
        "1_melenchon_modele_mesure": jlm,
        "2_mae_partages": round(float(np.mean([abs(m - x) for m, x in part.values()])), 2) if part else None,
        "2_mae_lt_14_7": bool(np.mean([abs(m - x) for m, x in part.values()]) < 14.7) if part else None,
        "3_ecart_duels_max": max(ecarts_duels.values()),
        "3_duels_le_3": bool(max(ecarts_duels.values()) <= 3.0),
        "3_lp_joint_le_10": bool(out["ecart_lp_joint"] <= 10.0),
        "4_t1_mae": ev["t1_resultat"]["mae_pt"],
        "4_t1_le_3_1": bool(ev["t1_resultat"]["mae_pt"] <= 3.1),
    }
    del mes
    dest = sr.OUT_DIR / f"multi_{config}_eval.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + "\n")
    print(json.dumps(out["criteres"], ensure_ascii=False, indent=1))
    print("lp joint par chaîne", out["lp_joint_par_chaine"])
    print(f"-> {dest}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=sorted(CONFIGS), required=True)
    ap.add_argument("--chaine", type=int, default=None)
    ap.add_argument("--evaluer", action="store_true")
    ap.add_argument("--fumee", action="store_true", help="20 + 20 itérations, pour vérifier que ça tourne")
    args = ap.parse_args()
    if args.evaluer:
        evaluer(args.config)
    elif args.chaine is not None:
        lancer_chaine(args.config, args.chaine, *((20, 20) if args.fumee else ()))
    else:
        ap.error("--chaine k ou --evaluer")


if __name__ == "__main__":
    main()
