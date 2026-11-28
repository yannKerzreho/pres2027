"""Log-vraisemblance prédictive des observations cachées, cercle contre segment (spec_second_tour §8.96 f,
§8.110 f1), 0 fit.

Pour chacune des 11 épreuves (8 V2, 3 V1) et chaque géométrie, le fit sauvegardé (meilleur U) est rejoué, puis
on évalue, avec la vraisemblance du modèle, les observations que l'épreuve a retirées :
- duels cachés (V2) : Normal(y ; s, √(s(1−s)/n + σ_t2²)), s à la date du duel (chemin de w, nœud SUIVANT par
  `searchsorted` comme partout dans l'essai ; écart nul pour les duels cachés, §8.113 a) ;
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


def ll_epreuve(geo, ep, raw0, rep0, lse_finalistes=False, fidelite=True, hors_finalistes=False):
    nom = "".join(ch for ch in ep if ch.isalnum())
    r = min(json.loads((sr.OUT_DIR / f"essai2027_{geo}_{nom}.json").read_text())["inits"], key=lambda i: i["U"])
    fit4, _, _, _, _ = reconstruire(geo, ep, r, 4)
    duels, R = caches(raw0, rep0, ep, list(fit4.candidates))
    if hors_finalistes and R is not None:               # §8.113 e4 : électorats non finalistes seulement
        m = (R["k"] != R["a"]) & (R["k"] != R["b"])
        R = {k: (np.asarray(v)[m] if k in ("y", "k", "a", "b", "dates") else v) for k, v in R.items()}
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
        s = np.asarray(_duel_par_tirage(fit, w, ia, ib, lse_finalistes=lse_finalistes))[:n_d]
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
        th = X.cellules(fit, champ, A, B, K, w=w, fidelite=fidelite) / 100
        th = np.atleast_2d(th)[:n_r]
        c = float(np.exp(vals["log_c"]))
        out["ll_reports"] = float(dist.Dirichlet(c * jnp.clip(jnp.asarray(th), 1e-4, None)).log_prob(
            jnp.asarray(R["y"])).sum())
    out["ll"] = out["ll_duels"] + out["ll_reports"]
    return out


def main(lse_finalistes: bool = False) -> dict:
    raw0, rep0, _ = X.donnees()
    eps = [e for e in X.epreuves() if e != "complet"]
    res = {"spec": "§8.96 f, §8.110 f1", "epreuves": {}}
    for ep in eps:
        L = {geo: ll_epreuve(geo, ep, raw0, rep0, lse_finalistes) for geo in ("segment", "cercle")}
        L["delta_cercle_moins_segment"] = round(L["cercle"]["ll"] - L["segment"]["ll"], 2)
        res["epreuves"][ep] = L
        print(json.dumps({ep: L}, ensure_ascii=False), flush=True)
    d = np.array([v["delta_cercle_moins_segment"] for v in res["epreuves"].values()])
    S = {"n_epreuves": len(d), "somme_delta": round(float(d.sum()), 2), "cercle_meilleur": int((d > 0).sum()),
         "delta_par_epreuve": {k: v["delta_cercle_moins_segment"] for k, v in res["epreuves"].items()}}
    S["verdict"] = "cercle" if (S["somme_delta"] >= 5 and S["cercle_meilleur"] >= 8) else "segment"
    res["synthese"] = S
    res["lse_finalistes"] = lse_finalistes                 # §8.113 e1
    nomf = "essai2027_vraisemblance_lsefin.json" if lse_finalistes else "essai2027_vraisemblance.json"
    (sr.OUT_DIR / nomf).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(S, ensure_ascii=False, indent=1))
    return res




def variante(geo: str = "cercle2pi") -> dict:
    """§8.113 e5 / e2 : variante du cercle (arc fixé à 2π, ou η ≡ 0) contre le cercle de référence, mêmes observations cachées,
    finalistes dans le lse (§8.114). Critère figé : fixer l'arc si ΣΔ(fixé − libre) ≥ −2,3 nats."""
    raw0, rep0, _ = X.donnees()
    libre = json.loads((sr.OUT_DIR / "essai2027_vraisemblance_lsefin.json").read_text())["epreuves"]
    res = {"spec": "§8.113 e5", "epreuves": {}}
    for ep in [e for e in X.epreuves() if e != "complet"]:
        L = ll_epreuve(geo, ep, raw0, rep0, lse_finalistes=True)
        d = round(L["ll"] - libre[ep]["cercle"]["ll"], 2)
        res["epreuves"][ep] = {geo: L, "cercle_ll": libre[ep]["cercle"]["ll"],
                               "segment_ll": libre[ep]["segment"]["ll"], "delta_fixe_moins_libre": d}
        print(json.dumps({ep: d}, ensure_ascii=False), flush=True)
    d = [v["delta_fixe_moins_libre"] for v in res["epreuves"].values()]
    ds = [v[geo]["ll"] - v["segment_ll"] for v in res["epreuves"].values()]
    res["geo"] = geo
    res["synthese"] = {"somme_delta_fixe_moins_libre": round(float(sum(d)), 2),
                       "epreuves_degradees": int(sum(x < 0 for x in d)),
                       "somme_delta_variante_moins_segment": round(float(sum(ds)), 2),
                       "variante_meilleure_que_segment": int(sum(x > 0 for x in ds))}
    if geo == "cercle2pi":                                 # §8.113 e5
        res["synthese"]["arc_fixe_retenu"] = bool(sum(d) >= -2.3)
    if geo == "cercle_eta0":                               # §8.113 e2
        res["synthese"]["eta0_retenu"] = bool(sum(d) >= -5 and sum(x < 0 for x in d) <= 6)
    (sr.OUT_DIR / ("essai2027_vraisemblance_arc2pi.json" if geo == "cercle2pi"
                   else f"essai2027_vraisemblance_{geo}.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(res["synthese"], ensure_ascii=False))
    return res


def r7() -> dict:
    """§8.113 e4 : R7 en refit. Reports cachés des 3 V1, électorats NON finalistes ; référence cercle η ≡ 0 lue avec
    R7, variante sans R7 lue sans R7. R7 gardée si la retirer coûte plus de 2 nats."""
    raw0, rep0, _ = X.donnees()
    res = {"spec": "§8.113 e4", "epreuves": {}}
    for ep in [e for e in X.epreuves() if e.startswith("v1:")]:
        a = ll_epreuve("cercle_eta0", ep, raw0, rep0, True, fidelite=True, hors_finalistes=True)
        b = ll_epreuve("cercle_eta0_sansR7", ep, raw0, rep0, True, fidelite=False, hors_finalistes=True)
        res["epreuves"][ep] = {"avec_R7": a, "sans_R7": b, "delta_sans_moins_avec": round(b["ll"] - a["ll"], 2)}
        print(json.dumps({ep: res["epreuves"][ep]["delta_sans_moins_avec"]}, ensure_ascii=False), flush=True)
    d = sum(v["delta_sans_moins_avec"] for v in res["epreuves"].values())
    res["synthese"] = {"somme_delta_sans_moins_avec": round(d, 2), "R7_gardee": bool(d < -2)}
    (sr.OUT_DIR / "essai2027_vraisemblance_r7.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(res["synthese"]))
    return res


if __name__ == "__main__":
    import sys
    if "--arc-fixe" in sys.argv:
        variante("cercle2pi")
    elif "--eta0" in sys.argv:
        variante("cercle_eta0")
    elif "--r7" in sys.argv:
        r7()
    elif "--variante" in sys.argv:
        variante(sys.argv[sys.argv.index("--variante") + 1])
    else:
        main(lse_finalistes="--lse-finalistes" in sys.argv)
