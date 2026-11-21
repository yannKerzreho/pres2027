"""Dépistage MAP avant NUTS (spec_second_tour §8.57, revue locale du 21/11 08:00Z, piste B).

    PRES2027_B_QUAD=100 python -m model.backtest.map_screen --variante hier_ra_fid_omk \\
        [--election 2022] [--coupure T1-30] [--inits 6] [--pas 400] [--lse-officiel]

Pour chaque initialisation (init_to_median + bruit N(0, 0,5) dans l'espace non contraint,
graines 0..n−1) : Adam sur −log p(θ, y) (Jacobiens compris, comme NUTS), puis le point est
enregistré comme une PSEUDO-CHAÎNE `results/chaines/map_<variante>_<élection>_<coupure>_c<k>.npz`
(20 copies du même point), lisible par `model.backtest.modes --stem map_… --n <inits>`. Ce n'est
PAS un postérieur : un dépistage des modes et des critères 2 / 3b avant de payer NUTS.
"""
from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.infer import init_to_median
from numpyro.infer.util import initialize_model

import model.backtest.second_round as sr
from model.backtest import eval_r2


class _Capture(Exception):
    pass


def capturer(variante: str, election: int, coupure: str, lse_officiel: bool):
    eval_r2.ELECTION = election
    sr.ABSTENTION = True
    sr.VARIANTE_KW.clear()
    dico = {
        "hier_ra": {"eta_hier": True, "_reports_avant": True},
        "hier_ra_fid": {"eta_hier": True, "_reports_avant": True, "fidelite": True},
        "hier_ra_fid_omk": {"eta_hier": True, "_reports_avant": True, "fidelite": True, "omega_electorat": True},
        "hier_ra_lie": {"eta_hier": True, "_reports_avant": True, "sigma_lie": True},
        "hiercercle_ra": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True},
        "hiercercle_ra_ferme": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True, "arc_ferme": True},
        "hiercercle_ra_fid": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True, "fidelite": True},
        "hiercercle_ra_fid_omk": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True, "fidelite": True,
                                  "omega_electorat": True},
        "hier": {"eta_hier": True},
        "hiercercle_ra_fid_omk_lam": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True,
                                      "fidelite": True, "omega_electorat": True, "melange_lambda": True},
    }
    sr.VARIANTE_KW.update(dico[variante])
    if lse_officiel:
        sr.VARIANTE_KW["_lse_officiel"] = True
    ref = sr.T1_DATES[election] if coupure.startswith("T1") else sr.T2_DATES[election]
    import pandas as pd
    as_of = ref - pd.Timedelta(days=int(coupure.split("-")[1]))
    raw = sr.polls_historiques(election, as_of)
    import os
    if os.environ.get("PRES2027_SANS_PAIRE"):                    # V2 au MAP (§8.69)
        pa, pb = os.environ["PRES2027_SANS_PAIRE"].split("|")
        raw = sr._sans_paire(raw, pa, pb)
    cap = {}

    def run(model_fn, model_kwargs, **kw):
        cap.update(fn=model_fn, kw=model_kwargs)
        raise _Capture
    try:
        sr._fit(raw, as_of, election, dict(draws=4, tune=4, chains=1, mcmc_runner=run))
    except _Capture:
        pass
    return cap["fn"], cap["kw"]


def map_point(fn, kw, seed: int, pas: int, lr: float = 0.01, decroissant: bool = False):
    info = initialize_model(jax.random.PRNGKey(seed), fn, model_kwargs=kw, init_strategy=init_to_median)
    z0 = info.param_info.z
    rng = np.random.default_rng(seed)
    if seed > 0:
        z0 = {k: v + 0.5 * jnp.asarray(rng.standard_normal(np.shape(v)), dtype=jnp.asarray(v).dtype)
              for k, v in z0.items()}
    pot = jax.jit(jax.value_and_grad(info.potential_fn))
    m = {k: jnp.zeros_like(v) for k, v in z0.items()}
    v2 = {k: jnp.zeros_like(v) for k, v in z0.items()}
    z = dict(z0)
    val = None
    for it in range(1, pas + 1):
        val, g = pot(z)
        for k in z:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            v2[k] = 0.999 * v2[k] + 0.001 * g[k] ** 2
            lr_t = lr * (0.05 ** (it / pas)) if decroissant else lr        # §8.67 (c) : 0,01 → 5e-4
            z[k] = z[k] - lr_t * (m[k] / (1 - 0.9 ** it)) / (jnp.sqrt(v2[k] / (1 - 0.999 ** it)) + 1e-8)
    gnorm = float(jnp.sqrt(sum(jnp.sum(gg ** 2) for gg in pot(z)[1].values())))
    cons = info.postprocess_fn(z)
    tr = handlers.trace(handlers.substitute(handlers.seed(fn, 0), data=cons)).get_trace(**kw)
    vals = {n: np.asarray(s["value"]) for n, s in tr.items()
            if s["type"] in ("sample", "deterministic") and not s.get("is_observed", False)}
    vals["_gnorm"] = np.asarray(gnorm)
    return float(val), vals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variante", required=True)
    ap.add_argument("--election", type=int, default=2022)
    ap.add_argument("--coupure", default="T1-30")
    ap.add_argument("--inits", type=int, default=6)
    ap.add_argument("--pas", type=int, default=5000)
    ap.add_argument("--lse-officiel", action="store_true")
    ap.add_argument("--decroissant", action="store_true", help="§8.67 (c) : lr 0,01 → 5e-4 (géométrique)")
    args = ap.parse_args()
    fn, kw = capturer(args.variante, args.election, args.coupure, args.lse_officiel)
    stem = f"map_{args.variante}_{args.election}_{args.coupure}"
    if args.pas != 5000 or args.decroissant:
        stem += f"_p{args.pas}" + ("d" if args.decroissant else "")
    import os
    if os.environ.get("PRES2027_SANS_HYP"):                       # V1 (§8.59)
        stem += "_sans_" + "".join(ch for ch in os.environ["PRES2027_SANS_HYP"] if ch.isalnum())
    if os.environ.get("PRES2027_SANS_PAIRE"):
        stem += "_v2sans_" + "".join(ch for ch in os.environ["PRES2027_SANS_PAIRE"] if ch.isalnum())
    sr.CHAINES_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for k in range(args.inits):
        t = time.time()
        U, vals = map_point(fn, kw, k, args.pas, decroissant=args.decroissant)
        gn = float(vals.pop("_gnorm"))
        S = max(20, sr.N_LOTS_SE * 2)      # au moins N_LOTS_SE tirages (erreur-type par lots)
        np.savez_compressed(sr.CHAINES_DIR / f"{stem}_c{k}.npz",
                            **{f"s__{n}": np.repeat(v[None, None], S, axis=1) for n, v in vals.items()},
                            e__diverging=np.zeros(S, dtype=bool), e__num_steps=np.ones(S, dtype=int))
        out.append({"init": k, "potentiel": round(U, 1), "norme_gradient": round(gn, 3),
                    "secondes": round(time.time() - t, 1)})
        print(json.dumps(out[-1]), flush=True)
    print(f"-> pseudo-chaînes {stem}_c0..c{args.inits - 1} ; lire : python -m model.backtest.modes "
          f"--variante {args.variante} --stem {stem} --n {args.inits}"
          + (" --lse-officiel" if args.lse_officiel else ""))


if __name__ == "__main__":
    main()
