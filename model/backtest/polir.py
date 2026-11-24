"""Polissage L-BFGS de pseudo-chaînes MAP existantes (spec_second_tour §8.79 ; revue de la routine
du 23/11, point 3 ; code proposé par la revue du 24/11, relu).

    JAX_ENABLE_X64=1 PRES2027_SANS_PAIRE="A|B" [PRES2027_ARC_FIXE=…] PRES2027_B_QUAD=100 \\
        python -m model.backtest.polir --variante hiercercle_ra_fid --stem <stem> --n 6 --lse-officiel

Pour chaque init k : valeurs contraintes du npz → espace non contraint → L-BFGS-B sur le potentiel U
(jacobiens compris, l'objectif d'Adam) → pseudo-chaîne `<stem>_poli_c<k>.npz`. Rapport JSON : U avant /
après, ‖∇‖, convergé si |ΔU| < 1 et ‖∇‖ < 1e-2. Classement par U.
"""
from __future__ import annotations

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from numpyro import handlers
from numpyro.infer import init_to_median
from numpyro.infer.util import initialize_model, unconstrain_fn
from scipy.optimize import minimize

import model.backtest.second_round as sr
from model.backtest.map_screen import capturer


def polir(potential_fn, z, maxiter=5000, gtol=1e-3):
    flat0, unravel = ravel_pytree(z)
    fg = jax.jit(jax.value_and_grad(lambda x: potential_fn(unravel(x))))

    def fun(x):
        v, g = fg(jnp.asarray(x, dtype=flat0.dtype))
        return float(v), np.asarray(g, dtype=np.float64)

    r = minimize(fun, np.asarray(flat0, np.float64), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=maxiter, gtol=gtol, ftol=1e-15, maxcor=50))
    x1 = jnp.asarray(r.x, dtype=flat0.dtype)
    v1, g1 = fg(x1)
    print("lbfgs:", r.message, "dtype", flat0.dtype, flush=True)
    return unravel(x1), float(v1), float(jnp.linalg.norm(g1)), int(r.nit)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variante", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--election", type=int, default=2022)
    ap.add_argument("--coupure", default="T1-30")
    ap.add_argument("--lse-officiel", action="store_true")
    ap.add_argument("--maxiter", type=int, default=5000)
    a = ap.parse_args()
    fn, kw = capturer(a.variante, a.election, a.coupure, a.lse_officiel)
    info = initialize_model(jax.random.PRNGKey(0), fn, model_kwargs=kw, init_strategy=init_to_median)
    sites = list(info.param_info.z.keys())
    out = []
    for k in range(a.n):
        z0 = np.load(sr.CHAINES_DIR / f"{a.stem}_c{k}.npz")
        params = {s: jnp.asarray(np.asarray(z0[f"s__{s}"][0, 0], dtype=np.float64)) for s in sites}
        zu = unconstrain_fn(fn, (), kw, params)
        U0 = float(info.potential_fn(zu))
        z1, U1, gn, nit = polir(info.potential_fn, zu, maxiter=a.maxiter)
        cons = info.postprocess_fn(z1)
        tr = handlers.trace(handlers.substitute(handlers.seed(fn, 0), data=cons)).get_trace(**kw)
        vals = {n: np.asarray(s["value"]) for n, s in tr.items()
                if s["type"] in ("sample", "deterministic") and not s.get("is_observed", False)}
        S = max(20, sr.N_LOTS_SE * 2)
        np.savez_compressed(sr.CHAINES_DIR / f"{a.stem}_poli_c{k}.npz",
                            **{f"s__{n}": np.repeat(v[None, None], S, axis=1) for n, v in vals.items()},
                            e__diverging=np.zeros(S, dtype=bool), e__num_steps=np.ones(S, dtype=int))
        out.append({"init": k, "U_adam": round(U0, 2), "U_poli": round(U1, 2), "delta_U": round(U0 - U1, 2),
                    "norme_gradient": gn, "iterations": nit, "converge": abs(U0 - U1) < 1.0 and gn < 1e-2})
        print(json.dumps(out[-1]), flush=True)
    rang = sorted(out, key=lambda o: o["U_poli"])
    rep = {"spec": "§8.79", "variante": a.variante, "stem": a.stem, "x64": bool(jax.config.jax_enable_x64),
           "env": {e: os.environ.get(e) for e in ("PRES2027_SANS_PAIRE", "PRES2027_SANS_HYP", "PRES2027_ARC_FIXE")},
           "inits": out, "meilleur": rang[0]["init"],
           "a_1_nat": [o["init"] for o in rang if o["U_poli"] - rang[0]["U_poli"] < 1.0]}
    (sr.OUT_DIR / f"polir_{a.stem}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("meilleur", "a_1_nat")}))


if __name__ == "__main__":
    main()
