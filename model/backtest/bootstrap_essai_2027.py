"""Bootstrap apparié par notice de V1 (spec_second_tour §8.105 e2), 0 fit.

Unité rééchantillonnée : la notice, tirée avec remise À L'INTÉRIEUR de chaque hypothèse cachée (les 3 hypothèses
restent présentes ; Glucksmann – Le Pen n'a qu'une notice). Pour chaque tirage, chaque cellule (hypothèse,
électorat non finaliste) est la moyenne sur les notices tirées, pour la mesure ET pour chaque prédicteur
(segment, cercle à la date de la notice ; K constant ; K_s de la notice) ; V1 = MAE sur les 8 cellules hors
cellule polluée (9 rapportées). 2 000 tirages, graine 0.

Critères figés (§8.105 e2) :
- la géométrie apporte seulement si l'IC à 90 % de V1(géo) − V1(meilleur de K et K_s, par tirage) est
  entièrement sous −0,5 ;
- « segment préférable » si P(V1 segment ≤ V1 cercle) ≥ 0,8 (rapporté pour l'arbitrage, sans le trancher).

Usage : JAX_ENABLE_X64=1 PRES2027_B_QUAD=100 python -m model.backtest.bootstrap_essai_2027
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from model.backtest import essai_2027 as X
from model.backtest import second_round as sr
from model.backtest.beta_essai_2027 import reconstruire
from model.backtest.lire_essai_2027 import k_s, mesure

N_BOOT, GRAINE = 2000, 0


def table(raw, rep):
    """Lignes (h, e, notice, mesure, segment, cercle, K, K_s), en points."""
    rows = []
    for h in X.CACHES_V1:
        a, b = [x.strip() for x in h.split("–")]
        hs = [x for x in rep.hypothese.unique() if x != h and x.endswith("– Le Pen")]
        autres = {h2: mesure(rep, h2) for h2 in hs}
        KS, _ = k_s(raw, rep, h, autres, par_notice=True)
        cel = []
        for e, g in rep[rep.hypothese == h].groupby("electorat_t1"):
            ec = X.court(e)
            if ec in (a, b):
                continue
            k = [m[ec] for h2, m in autres.items() if ec in m and ec not in {y.strip() for y in h2.split("–")}]
            if not k:
                continue
            t = g.pivot_table(index="notice", columns="candidat_t2", values="part")
            if not {a, b, "Aucun"} <= set(t.columns):
                continue
            t = t[[a, b, "Aucun"]].dropna()
            for nt, v in t.iterrows():
                m = 100 * v.to_numpy() / v.sum()
                cel.append({"h": h, "e": ec, "notice": nt, "date": pd.Timestamp(g[g.notice == nt].date_fin.iloc[0]),
                            "mesure": m, "K": np.mean(k, axis=0), "K_s": KS.get((ec, nt))})
        for geo in ("segment", "cercle"):
            ep = f"v1:{h}"
            nom = "".join(ch for ch in ep if ch.isalnum())
            J = json.loads((sr.OUT_DIR / f"essai2027_{geo}_{nom}.json").read_text())
            r = min(J["inits"], key=lambda i: i["U"])
            n = len(cel)
            fit, _, _, champ, _ = reconstruire(geo, ep, r, max(n, 4))   # ≥ 4 tirages pour le résumé numpyro
            idx = {X.court(c): i for i, c in enumerate(fit.candidates)}
            S = fit.w_path_draws.shape[0]
            jj = [int(np.clip(np.searchsorted(fit.w_dates, float(c["date"].toordinal())), 0, len(fit.w_dates) - 1))
                  for c in cel]
            jj += [jj[0]] * (S - n)
            ks = [idx[c["e"]] for c in cel] + [idx[a]] * (S - n)
            w = fit.w_path_draws[np.arange(S), :, np.array(jj)]
            th = X.cellules(fit, champ, np.full(S, idx[a]), np.full(S, idx[b]), np.array(ks), w=w)[:n]
            for c, v in zip(cel, th):
                c[geo] = v
        rows += cel
    return rows


def v1(rows, choix, cle, exclure_pollue=True):
    """MAE sur les cellules, chaque cellule moyennée sur les notices tirées (`choix` : {h: [notices]})."""
    err = []
    for (h, e) in sorted({(r["h"], r["e"]) for r in rows}):
        if exclure_pollue and h.startswith("Glucksmann") and e == "Attal":
            continue
        sel = [r for nt in choix[h] for r in rows if r["h"] == h and r["e"] == e and r["notice"] == nt]
        if not sel or any(r[cle] is None for r in sel):
            continue
        err.append(np.abs(np.mean([r[cle] for r in sel], 0) - np.mean([r["mesure"] for r in sel], 0)).mean())
    return float(np.mean(err))


def main() -> dict:
    raw, rep, _ = X.donnees()
    rows = table(raw, rep)
    nts = {h: sorted({r["notice"] for r in rows if r["h"] == h}) for h in X.CACHES_V1}
    obs = {k: round(v1(rows, nts, k), 2) for k in ("segment", "cercle", "K", "K_s")}
    obs9 = {k: round(v1(rows, nts, k, False), 2) for k in ("segment", "cercle", "K", "K_s")}
    rng = np.random.default_rng(GRAINE)
    B = {k: [] for k in ("segment", "cercle", "K", "K_s")}
    for _ in range(N_BOOT):
        choix = {h: list(rng.choice(v, size=len(v), replace=True)) for h, v in nts.items()}
        for k in B:
            B[k].append(v1(rows, choix, k))
    B = {k: np.array(v) for k, v in B.items()}
    tem = np.minimum(B["K"], B["K_s"])
    S = {"n_notices": {h: len(v) for h, v in nts.items()}, "V1_8_observe": obs, "V1_9_observe": obs9}
    for g in ("segment", "cercle"):
        d = B[g] - tem
        lo, hi = np.quantile(d, [0.05, 0.95])
        S[g] = {"diff_temoin_IC90": [round(float(lo), 2), round(float(hi), 2)],
                "diff_temoin_moyenne": round(float(d.mean()), 2), "apporte": bool(hi < -0.5)}
    S["P_segment_le_cercle"] = round(float((B["segment"] <= B["cercle"]).mean()), 3)
    S["segment_preferable"] = bool(S["P_segment_le_cercle"] >= 0.8)
    res = {"spec": "§8.105 e2", "n_boot": N_BOOT, "graine": GRAINE, "synthese": S,
           "lignes": [{k: (v.tolist() if isinstance(v, np.ndarray) else (str(v) if k == "date" else v))
                       for k, v in r.items()} for r in rows]}
    (sr.OUT_DIR / "essai2027_bootstrap.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(S, ensure_ascii=False, indent=1))
    return res


if __name__ == "__main__":
    main()
