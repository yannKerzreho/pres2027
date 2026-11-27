"""Log-vraisemblance prédictive des observations cachées, cercle contre segment (spec_second_tour §8.96 f,
§8.110 f1), 0 fit.

Pour chacune des 11 épreuves (8 V2, 3 V1) et chaque géométrie, le fit sauvegardé (meilleur U) est rejoué, puis
on évalue, avec la vraisemblance du modèle, les observations que l'épreuve a retirées :
- duels cachés (V2) : Normal(y ; s, √(s(1−s)/n + σ_t2²)), s à la date du duel (chemin de w, nœud le plus proche
  par `searchsorted` comme partout dans l'essai) ;
- reports cachés (V2 : hypothèse de la paire, s'il y en a ; V1 : hypothèse cachée) : Dirichlet(c·max(θ, 1e-4)),
  θ = cellules du modèle à la date de la notice, mêmes lignes que `arrays_reports` (cellules complètes,
  plancher 0,5 %, renormalisées).
Critère figé : cercle retenu si ΣΔ(cercle − segment) ≥ 5 nats ET cercle meilleur sur au moins 8 épreuves sur
11 ; sinon segment (parcimonie).

Usage : JAX_ENABLE_X64=1 PRES2027_B_QUAD=100 python -m model.backtest.vraisemblance_essai_2027
"""
from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro import distributions as dist
from scipy.stats import norm

from model.backtest import essai_2027 as X
from model.backtest import second_round as sr
from model.backtest.beta_essai_2027 import reconstruire
from model.models.spatial_pooling.fit import MIN_POLL_DATE, _duel_par_tirage, arrays_reports
from model.models.spatial_pooling.roster import build_duel_arrays


def caches(raw0, rep0, ep, candidates):
    """(duels cachés, reports cachés) au format des tableaux du modèle, indices dans `candidates`."""
    t2 = raw0[(raw0.tour == "Deuxième tour") & (pd.to_datetime(raw0.date_fin) >= pd.Timestamp(MIN_POLL_DATE))
              & (pd.to_datetime(raw0.date_fin) <= pd.Timestamp(X.AS_OF))]
    D = build_duel_arrays(t2, candidates)
    court = [X.court(c) for c in candidates]
    if ep.startswith("v2:"):
        a, b = ep[3:].split("|")
        m = np.array([{court[i], court[j]} == {a, b} for i, j in zip(D["a"], D["b"])], dtype=bool)
        duels = {k: np.asarray(D[k])[m] for k in ("a", "b", "y", "n", "dates")}
        hyps = [h for h in rep0.hypothese.unique() if {x.strip() for x in h.split("–")} == {a, b}]
    else:
        duels = None
        hyps = [ep[3:]]
    R = arrays_reports(rep0[rep0.hypothese.isin(hyps)], candidates, X.AS_OF, None) if hyps else None
    return duels, R


def ll_epreuve(geo, ep, raw0, rep0):
    nom = "".join(ch for ch in ep if ch.isalnum())
    r = min(json.loads((sr.OUT_DIR / f"essai2027_{geo}_{nom}.json").read_text())["inits"], key=lambda i: i["U"])
    fit4, _, _, _, _ = reconstruire(geo, ep, r, 4)
    duels, R = caches(raw0, rep0, ep, list(fit4.candidates))
    n_d = 0 if duels is None else len(duels["y"])
    n_r = 0 if R is None else len(R["y"])
    S = max(n_d, n_r, 4)
    fit, _, vals, champ, _ = reconstruire(geo, ep, r, S)
    assert list(fit.candidates) == list(fit4.candidates)

    def idx_date(t):
        return int(np.clip(np.searchsorted(fit.w_dates, float(t)), 0, len(fit.w_dates) - 1))
    out = {"n_duels": n_d, "n_reports": n_r, "ll_duels": 0.0, "ll_reports": 0.0}
    if n_d:
        pad = S - n_d
        jj = [idx_date(t) for t in duels["dates"]] + [idx_date(duels["dates"][0])] * pad
        ia = np.concatenate([duels["a"], np.full(pad, duels["a"][0])])
        ib = np.concatenate([duels["b"], np.full(pad, duels["b"][0])])
        w = fit.w_path_draws[np.arange(S), :, np.array(jj)]
        s = np.asarray(_duel_par_tirage(fit, w, ia, ib))[:n_d]
        sd = np.sqrt(s * (1 - s) / np.clip(duels["n"], 1.0, None) + float(vals["sigma_t2"]) ** 2 + 1e-8)
        out["ll_duels"] = float(norm.logpdf(duels["y"], s, sd).sum())
        out["mae_duels_pt"] = float(100 * np.abs(s - duels["y"]).mean())
    if n_r:
        pad = S - n_r
        jj = [idx_date(t) for t in R["dates"]] + [idx_date(R["dates"][0])] * pad
        A = np.concatenate([R["a"], np.full(pad, R["a"][0])])
        B = np.concatenate([R["b"], np.full(pad, R["b"][0])])
        K = np.concatenate([R["k"], np.full(pad, R["k"][0])])
        w = fit.w_path_draws[np.arange(S), :, np.array(jj)]
        th = X.cellules(fit, champ, A, B, K, w=w) / 100
        th = np.atleast_2d(th)[:n_r]
        c = float(np.exp(vals["log_c"]))
        out["ll_reports"] = float(dist.Dirichlet(c * jnp.clip(jnp.asarray(th), 1e-4, None)).log_prob(
            jnp.asarray(R["y"])).sum())
    out["ll"] = out["ll_duels"] + out["ll_reports"]
    return out


def main() -> dict:
    raw0, rep0, _ = X.donnees()
    eps = [e for e in X.epreuves() if e != "complet"]
    res = {"spec": "§8.96 f, §8.110 f1", "epreuves": {}}
    for ep in eps:
        L = {geo: ll_epreuve(geo, ep, raw0, rep0) for geo in ("segment", "cercle")}
        L["delta_cercle_moins_segment"] = round(L["cercle"]["ll"] - L["segment"]["ll"], 2)
        res["epreuves"][ep] = L
        print(json.dumps({ep: L}, ensure_ascii=False), flush=True)
    d = np.array([v["delta_cercle_moins_segment"] for v in res["epreuves"].values()])
    S = {"n_epreuves": len(d), "somme_delta": round(float(d.sum()), 2), "cercle_meilleur": int((d > 0).sum()),
         "delta_par_epreuve": {k: v["delta_cercle_moins_segment"] for k, v in res["epreuves"].items()}}
    S["verdict"] = "cercle" if (S["somme_delta"] >= 5 and S["cercle_meilleur"] >= 8) else "segment"
    res["synthese"] = S
    (sr.OUT_DIR / "essai2027_vraisemblance.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(S, ensure_ascii=False, indent=1))
    return res


if __name__ == "__main__":
    main()
