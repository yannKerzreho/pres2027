"""Lecture de l'essai 2027 selon les règles figées (spec_second_tour §8.92). 0 fit.

    python -m model.backtest.lire_essai_2027
"""
from __future__ import annotations

import json

import numpy as np

import model.backtest.essai_2027 as X
import model.backtest.second_round as sr


def charger(geo, ep):
    p = sr.OUT_DIR / f"essai2027_{geo}_{''.join(c for c in ep if c.isalnum())}.json"
    return json.loads(p.read_text()) if p.exists() else None


def modes(res):
    I = sorted(res["inits"], key=lambda r: r["U"])
    return I[0], [r for r in I if r["U"] - I[0]["U"] < 10.0]


def mesure(rep, hyp):
    """{électorat: [a, b, Aucun]} moyenne des notices, en points, cellules complètes."""
    a, b = [x.strip() for x in hyp.split("–")]
    out = {}
    for e, g in rep[rep.hypothese == hyp].groupby("electorat_t1"):
        t = g.pivot_table(index="notice", columns="candidat_t2", values="part")
        if {a, b, "Aucun"} <= set(t.columns):
            v = t[[a, b, "Aucun"]].dropna()
            out[X.court(e)] = (100 * v.div(v.sum(axis=1), axis=0)).mean().to_numpy()
    return out


def verdict(S: dict, suf: str) -> str:
    """Règle figée de §8.92 : cercle (ou segment) si LFI meilleur d'au moins 2 pt ET V2, V1 pas pires de plus de 0,5."""
    e = S[f"LFI_RN_segment{suf}"] - S[f"LFI_RN_cercle{suf}"]
    dv2, dv1 = (S[f"{t}_cercle{suf}"][0] - S[f"{t}_segment{suf}"][0] for t in ("V2", "V1"))
    return ("cercle" if e >= 2 and dv2 <= .5 and dv1 <= .5 else
            "segment" if -e >= 2 and -dv2 <= .5 and -dv1 <= .5 else "non séparés")


def main() -> dict:
    raw, rep, champ = X.donnees()
    d = X.duels(raw)
    G = ("segment", "cercle")
    out = {"spec": "§8.92", "v2": [], "v1": [], "lfi_rn": []}
    # --- V2 : orientation « X » = le camp opposé à Le Pen quand elle est dans la paire ---
    moy = {(a, b): X.cible_v2(d, a, b)[0] for (a, b) in {(r.a, r.b) for r in d.itertuples()}}
    contre = lambda a, b, y: y if b == "Le Pen" else 100 - y                          # noqa: E731
    for ep in X.epreuves():
        if not ep.startswith("v2:"):
            continue
        a, b = ep[3:].split("|")
        y, n = X.cible_v2(d, a, b)
        L = {"paire": ep[3:], "n_total": int(((d.a == a) & (d.b == b)).sum()), "n_cible": n, "observe": round(y, 1)}
        for g in G:
            r = charger(g, ep)
            if r:
                best, proches = modes(r)
                L[g] = round(best["duels"][f"{a}|{b}"] - y, 2)
                L[g + "_modes"] = [round(m["duels"][f"{a}|{b}"] - y, 1) for m in proches]
                if best.get("duels_date"):                     # revue du 26/11 : lu aux dates de la cible
                    L[g + "_date"] = round(best["duels_date"][f"{a}|{b}"] - y, 2)
        autres = d[~((d.a == a) & (d.b == b))]
        L["BT"] = round(100 * float(sr._bradley_terry(autres, a, b)) - y, 2)
        if "Le Pen" in (a, b):
            t = [contre(p, q, v) for (p, q), v in moy.items() if "Le Pen" in (p, q) and (p, q) != (a, b)]
            L["T0"] = round(float(np.mean(t)) - contre(a, b, y), 2)
        out["v2"].append(L)
    # --- V1 et LFI -> RN ---
    for h in X.CACHES_V1:
        cible = mesure(rep, h)
        autres = [mesure(rep, h2) for h2 in rep.hypothese.unique() if h2 != h and h2.endswith("– Le Pen")]
        fin = {x.strip() for x in h.split("–")}
        for e, c in cible.items():
            if e in fin:
                continue
            k = [m[e] for h2, m in zip([x for x in rep.hypothese.unique() if x != h and x.endswith("– Le Pen")], autres)
                 if e in m and e not in {x.strip() for x in h2.split("–")}]
            L = {"hypothese": h, "electorat": e, "mesure": [round(float(x), 1) for x in c],
                 "K": [round(float(x), 1) for x in np.mean(k, axis=0)] if k else None}
            for g in G:
                r = charger(g, f"v1:{h}")
                if r:
                    best, proches = modes(r)
                    f = lambda m: next(z["pred"] for z in m["cellules"] if z["hypothese"] == h and X.court(z["electorat"]) == e)  # noqa: E731,E501
                    L[g] = f(best)
                    L[g + "_modes"] = [f(m) for m in proches]
                    fd = lambda m: next(z.get("pred_date") for z in m["cellules"] if z["hypothese"] == h and X.court(z["electorat"]) == e)  # noqa: E731,E501
                    if fd(best):
                        L[g + "_date"] = fd(best)
            out["v1"].append(L)
    mae = lambda key, K: float(np.mean([np.abs(np.array(x[key]) - x["mesure"]).mean() for x in K if x.get(key)]))  # noqa: E731,E501
    lfi = [x for x in out["v1"] if x["electorat"] == "Mélenchon"]
    S = out["synthese"] = {}
    for k in G + tuple(g + "_date" for g in G) + ("BT", "T0"):
        v = [abs(x[k]) for x in out["v2"] if x.get(k) is not None]
        S[f"V2_{k}"] = [round(float(np.mean(v)), 2), len(v)] if v else None
    lp = [x for x in out["v2"] if x.get("T0") is not None]
    for k in G + ("BT", "T0"):
        S[f"V2_avecT0_{k}"] = round(float(np.mean([abs(x[k]) for x in lp])), 2) if all(k in x for x in lp) else None
    avecK = [x for x in out["v1"] if x["K"]]
    for k in G + tuple(g + "_date" for g in G) + ("K",):
        if all(k in x for x in avecK):
            S[f"V1_{k}"] = [round(mae(k, avecK), 2), len(avecK)]
            S[f"LFI_RN_{k}"] = round(float(np.mean([np.abs(np.array(x[k])[1:] - np.array(x["mesure"])[1:]).mean()
                                                   for x in lfi if x.get(k)])), 2)
    # §8.98 b, g1 : K sans la cellule polluée (Glucksmann – Le Pen, électorat Attal : K = seule hypothèse Mélenchon – Le Pen)
    propre = [x for x in avecK if not (x["hypothese"].startswith("Glucksmann") and x["electorat"] == "Attal")]
    for k in G + tuple(g + "_date" for g in G) + ("K",):
        if all(k in x for x in propre):
            S[f"V1_sans_pollue_{k}"] = [round(mae(k, propre), 2), len(propre)]
    # §8.98 a : part RN parmi les EXPRIMÉS de l'électorat Mélenchon (lecture rapportée, sans seuil)
    for k in G + tuple(g + "_date" for g in G) + ("K",):
        if all(x.get(k) for x in lfi):
            S[f"LFI_RN_exprimes_{k}"] = round(float(np.mean([abs(100 * x[k][1] / (x[k][0] + x[k][1])
                                                              - 100 * x["mesure"][1] / (x["mesure"][0] + x["mesure"][1]))
                                                          for x in lfi])), 2)
    if all(S.get(f"V2_{g}") and S.get(f"V1_{g}") for g in G):
        S["verdict_geometrie"] = verdict(S, "")
    # revue du 26/11 ~14:40Z : la même règle figée, appliquée par le code à la lecture DATÉE
    if all(S.get(f"V2_{g}_date") and S.get(f"V1_{g}_date") for g in G):
        S["verdict_geometrie_date"] = verdict(S, "_date")
    for x in ("complet",):
        for g in G:
            r = charger(g, x)
            if r:
                b, p = modes(r)
                S[f"complet_{g}"] = {"U": b["U"], "arc": b["arc"], "kappa": round(b["kappa"], 2), "omega": round(b["omega"], 2),
                                     "inits_a_1nat": sum(i["U"] - b["U"] < 1 for i in r["inits"]), "R": b.get("R"),
                                     "z_pos_max": round(float(np.abs(b["z_pos"]).max()), 2),
                                     "z_sigma_max": round(float(np.abs(b["z_sigma"]).max()), 2)}
    (sr.OUT_DIR / "essai2027_lecture.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    return out


if __name__ == "__main__":
    o = main()
    for x in o["v2"]:
        print(x)
    for x in o["v1"]:
        print({k: v for k, v in x.items() if not k.endswith("_modes")})
    print(json.dumps(o["synthese"], ensure_ascii=False, indent=1))
