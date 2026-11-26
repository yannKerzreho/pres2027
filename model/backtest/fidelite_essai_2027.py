"""Fidélité R7 en lecture (spec_second_tour §8.105 e3), 0 fit.

Sur les 6 fits V1 rejoués, cellules des électorats des FINALISTES, avec R7 (`fidelite=True`, comme au fit) et
sans (`fidelite=False` : la géométrie seule), comparées aux mesures :
- cellules d'entraînement (lignes de reports du fit dont l'électorat est un finaliste) ;
- cellules cachées (hypothèse V1 cachée, électorats a et b, à la date de chaque notice complète).
Critère figé : R7 se justifie si sa suppression dégrade la MAE de ces cellules de plus de 1 pt.

Usage : JAX_ENABLE_X64=1 PRES2027_B_QUAD=100 python -m model.backtest.fidelite_essai_2027
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from model.backtest import essai_2027 as X
from model.backtest import second_round as sr
from model.backtest.beta_essai_2027 import reconstruire


def main() -> dict:
    _, rep, _ = X.donnees()
    res = {"spec": "§8.105 e3", "fits": {}}
    tout = {"avec": {"entrainement": [], "cachees": []}, "sans": {"entrainement": [], "cachees": []}}
    par_geo = {}
    for geo in ("segment", "cercle"):
        G = {"avec": {"entrainement": [], "cachees": []}, "sans": {"entrainement": [], "cachees": []}}
        for h in X.CACHES_V1:
            a, b = [x.strip() for x in h.split("–")]
            ep = f"v1:{h}"
            nom = "".join(ch for ch in ep if ch.isalnum())
            r = min(json.loads((sr.OUT_DIR / f"essai2027_{geo}_{nom}.json").read_text())["inits"], key=lambda i: i["U"])
            # entraînement
            fit4, kw, _, champ, _ = reconstruire(geo, ep, r, 4)
            ka, kb, kk = (np.asarray(kw[x]) for x in ("rep_a", "rep_b", "rep_k"))
            fin = np.where((kk == ka) | (kk == kb))[0]
            # cachées
            cach = []
            for e, g in rep[rep.hypothese == h].groupby("electorat_t1"):
                if X.court(e) not in (a, b):
                    continue
                t = g.pivot_table(index="notice", columns="candidat_t2", values="part")
                if not {a, b, "Aucun"} <= set(t.columns):
                    continue
                for nt, v in t[[a, b, "Aucun"]].dropna().iterrows():
                    cach.append((X.court(e), pd.Timestamp(g[g.notice == nt].date_fin.iloc[0]), v.to_numpy() / v.sum()))
            n_e, n_c = len(fin), len(cach)
            S = max(n_e + n_c, 4)
            fit, _, _, _, _ = reconstruire(geo, ep, r, S)
            idx = {X.court(c): i for i, c in enumerate(fit.candidates)}
            jd = np.asarray(kw["rep_date_idx"])
            dates_c = [int(np.clip(np.searchsorted(fit.w_dates, float(t.toordinal())), 0, len(fit.w_dates) - 1))
                       for _, t, _ in cach]
            J = list(jd[fin]) + dates_c
            IA = list(ka[fin]) + [idx[a]] * n_c
            IB = list(kb[fin]) + [idx[b]] * n_c
            K = list(kk[fin]) + [idx[e] for e, _, _ in cach]
            pad = S - len(J)
            J, IA, IB, K = J + [J[0]] * pad, IA + [IA[0]] * pad, IB + [IB[0]] * pad, K + [IA[0]] * pad
            w = fit.w_path_draws[np.arange(S), :, np.array(J)]
            y = list(np.asarray(kw["rep_y"])[fin]) + [m for _, _, m in cach]
            for mode, fid in (("avec", True), ("sans", False)):
                th = X.cellules(fit, champ, np.array(IA), np.array(IB), np.array(K), w=w, fidelite=fid)[:n_e + n_c]
                err = np.abs(th - 100 * np.array(y)).mean(1)
                G[mode]["entrainement"] += list(err[:n_e])
                G[mode]["cachees"] += list(err[n_e:])
                res["fits"].setdefault(f"{geo}|{h}", {})[mode] = {
                    "entrainement": round(float(err[:n_e].mean()), 2) if n_e else None,
                    "cachees": round(float(err[n_e:].mean()), 2) if n_c else None,
                    "exemple_cache": [round(float(x), 1) for x in th[n_e]] if n_c else None}
            res["fits"][f"{geo}|{h}"]["n"] = {"entrainement": n_e, "cachees": n_c}
            print(json.dumps({f"{geo}|{h}": res["fits"][f"{geo}|{h}"]}, ensure_ascii=False), flush=True)
        par_geo[geo] = {m: {k: round(float(np.mean(v)), 2) for k, v in G[m].items()} for m in G}
        tous = {m: float(np.mean(G[m]["entrainement"] + G[m]["cachees"])) for m in G}
        par_geo[geo]["degradation_sans_R7"] = round(tous["sans"] - tous["avec"], 2)
        par_geo[geo]["R7_justifiee"] = bool(tous["sans"] - tous["avec"] > 1.0)
    res["synthese"] = par_geo
    (sr.OUT_DIR / "essai2027_fidelite.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(par_geo, ensure_ascii=False, indent=1))
    return res


if __name__ == "__main__":
    main()
