"""Verdict abst contre hier refait (spec_second_tour §8.20), sans fit.

Lit les JSON V1 re-fusionnés (tirages prédictifs sauvés), écarte les tâches
dédoublonnées et non concluantes, calcule le log-score par KDE en logit et les
écarts appariés avec IC90 bootstrap par tâche.

    python -m model.backtest.verdict_abst_hier
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"

# (élection, libellé, fichier abst, fichier hier) — §8.20.
H = "second_round_{v}_v1_{e}_{p}{s}.json"
TACHES = [
    (2017, "Macron–Le Pen", H.format(v="{v}", e=2017, p="Emmanuel_Macron-Marine_Le_Pen", s="")),
    (2017, "Fillon–Le Pen", H.format(v="{v}", e=2017, p="François_Fillon-Marine_Le_Pen", s="")),
    (2022, "Macron–Zemmour", H.format(v="{v}", e=2022, p="Emmanuel_Macron-Eric_Zemmour", s="_init")),
    (2022, "Macron–Mélenchon", H.format(v="{v}", e=2022, p="Emmanuel_Macron-Jean-Luc_Mélenchon", s="_init")),
    (2022, "Macron–Pécresse", H.format(v="{v}", e=2022, p="Emmanuel_Macron-Valérie_Pécresse", s="_init")),
    (2022, "Macron–Le Pen", {"abst": "second_round_abst_v1_2022_Emmanuel_Macron-Marine_Le_Pen_init.json",
                             "hier": "second_round_hier_v1_2022_Emmanuel_Macron-Marine_Le_Pen.json"}),
    (2027, "Le Pen–Philippe", "v1_2027_{v}_Le_Pen-Philippe.json"),
    (2027, "Le Pen–Mélenchon", "v1_2027_{v}_Le_Pen-Mélenchon.json"),
    (2027, "Glucksmann–Le Pen", "v1_2027_{v}_Glucksmann-Le_Pen.json"),
    (2027, "Attal–Le Pen", "v1_2027_{v}_Attal-Le_Pen.json"),
]
SENSIBILITE = [(2027, "Hollande–Le Pen", "v1_2027_{v}_Hollande-Le_Pen.json")]


def log_score_kde(tirages: np.ndarray, y: float) -> float:
    """log densité prédictive en y, KDE gaussienne en logit (Silverman) + jacobien."""
    z = np.log(tirages / (1 - tirages))
    s = z.std(ddof=1)
    iqr = np.subtract(*np.percentile(z, [75, 25]))
    h = 0.9 * min(s, iqr / 1.34 if iqr > 0 else s) * len(z) ** -0.2
    zy = np.log(y / (1 - y))
    k = -0.5 * ((zy - z) / h) ** 2
    m = k.max()
    lf = m + np.log(np.exp(k - m).sum()) - np.log(len(z) * h * np.sqrt(2 * np.pi))
    return float(lf - np.log(y * (1 - y)))


def _ligne(fichier: str) -> dict:
    d = json.loads((RESULTS / fichier).read_text())
    lignes = d.get("lignes") or d.get("v1")
    assert len(lignes) == 1, fichier
    return lignes[0]


def evalue(taches) -> list[dict]:
    out = []
    for e, nom, f in taches:
        r = {"election": e, "tache": nom}
        for v in ("abst", "hier"):
            li = _ligne(f[v] if isinstance(f, dict) else f.format(v=v))
            y = li["observe_30j"]
            t = np.asarray(li["tirages_predictifs"])
            r[v] = {"erreur_pt": li.get("erreur_pt", round(100 * abs(li["modele"] - y), 2)),
                    "ls_kde": round(log_score_kde(t, y), 3),
                    "ls_gauss": li["log_score_predictif"],
                    "couvre": li["couvre_predictif"], "concluant": li.get("concluant", False)}
        r["concluant"] = r["abst"]["concluant"] and r["hier"]["concluant"]
        r["delta_ls"] = round(r["hier"]["ls_kde"] - r["abst"]["ls_kde"], 3)
        r["delta_erreur_pt"] = round(r["hier"]["erreur_pt"] - r["abst"]["erreur_pt"], 2)
        out.append(r)
    return out


def bootstrap(x: np.ndarray, n: int = 10_000, seed: int = 20) -> list[float]:
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [round(float(v), 3) for v in np.percentile(m, [5, 95])]


def resume(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["concluant"]]
    dl = np.array([r["delta_ls"] for r in ok]); de = np.array([r["delta_erreur_pt"] for r in ok])
    ic_l, ic_e = bootstrap(dl), bootstrap(de)
    if ic_l[0] > 0 and not ic_e[0] > 0:
        verdict = "hier préférée"
    elif ic_l[1] < 0 and not ic_e[1] < 0:
        verdict = "abst préférée"
    else:
        verdict = "pas de différence établie : abst reste le défaut"
    par_e = {}
    for e in sorted({r["election"] for r in ok}):
        g = [r for r in ok if r["election"] == e]
        par_e[e] = {"n": len(g),
                    "ls_kde_abst": round(sum(r["abst"]["ls_kde"] for r in g), 2),
                    "ls_kde_hier": round(sum(r["hier"]["ls_kde"] for r in g), 2),
                    "mae_abst": round(float(np.mean([r["abst"]["erreur_pt"] for r in g])), 2),
                    "mae_hier": round(float(np.mean([r["hier"]["erreur_pt"] for r in g])), 2)}
    return {"n_taches": len(ok), "ecartees_non_concluantes": [r["tache"] for r in rows if not r["concluant"]],
            "delta_ls_moyen": round(float(dl.mean()), 3), "ic90_delta_ls": ic_l,
            "delta_erreur_moyen_pt": round(float(de.mean()), 2), "ic90_delta_erreur_pt": ic_e,
            "par_election": par_e, "verdict": verdict}


def main() -> None:
    rows = evalue(TACHES)
    res = {"taches": rows, "principal": resume(rows)}
    try:
        sens = evalue(SENSIBILITE)
        res["sensibilite_hollande"] = {"taches": sens, "resume": resume(rows + sens)}
    except (FileNotFoundError, KeyError) as exc:
        res["sensibilite_hollande"] = f"indisponible ({exc})"
    out = RESULTS / "verdict_abst_hier.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    for r in rows:
        print(f"{r['election']} {r['tache']:20s} concl={r['concluant']!s:5s} "
              f"LS abst {r['abst']['ls_kde']:7.2f} hier {r['hier']['ls_kde']:7.2f} "
              f"(gauss {r['abst']['ls_gauss']:6.2f}/{r['hier']['ls_gauss']:6.2f})  "
              f"err {r['abst']['erreur_pt']:.2f}/{r['hier']['erreur_pt']:.2f}")
    print(json.dumps(res["principal"], ensure_ascii=False, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()


# --- §8.29 : verdict robuste sur les 10 tâches -----------------------------------

def crps_tirages(t: np.ndarray, y: float) -> float:
    """CRPS empirique (pt) : E|X − y| − ½ E|X − X'| (forme triée, O(T log T))."""
    x = np.sort(np.asarray(t, float))
    n = len(x)
    e1 = np.abs(x - y).mean()
    e2 = 2.0 * np.sum(x * (2 * np.arange(1, n + 1) - n - 1)) / n ** 2
    return float(100 * (e1 - 0.5 * e2))


def inversion_signes(d: np.ndarray) -> float:
    """p unilatéral exact (moyenne > 0) par inversion des signes."""
    import itertools
    d = np.asarray(d, float)
    m = d.mean()
    tot = [np.mean(np.array(s) * np.abs(d)) >= m - 1e-12 for s in itertools.product([1, -1], repeat=len(d))]
    return float(np.mean(tot))


def verdict_robuste() -> dict:
    rows = []
    for e, nom, f in TACHES:
        r = {"election": e, "tache": nom}
        for v in ("abst", "hier"):
            li = _ligne(f[v] if isinstance(f, dict) else f.format(v=v))
            t = np.asarray(li["tirages_predictifs"]); y = li["observe_30j"]
            r[v] = {"crps": round(crps_tirages(t, y), 3), "ls_plafonne": round(max(log_score_kde(t, y), -5.0), 3),
                    "concluant": bool(li.get("concluant", False))}
        r["d_crps"] = round(r["abst"]["crps"] - r["hier"]["crps"], 3)          # > 0 : hier meilleur
        r["d_ls"] = round(r["hier"]["ls_plafonne"] - r["abst"]["ls_plafonne"], 3)
        rows.append(r)
    out = {"taches": rows}
    for k in ("d_crps", "d_ls"):
        d = np.array([r[k] for r in rows])
        par_e = {e: float(np.mean([r[k] for r in rows if r["election"] == e])) for e in (2017, 2022, 2027)}
        out[k] = {"moyenne": round(float(d.mean()), 3), "hier_meilleur": int((d > 0).sum()),
                  "p_taches": round(inversion_signes(d), 4),
                  "par_election": {e: round(v, 3) for e, v in par_e.items()},
                  "p_grappes": round(inversion_signes(np.array(list(par_e.values()))), 4)}
    ok = all(out[k]["p_taches"] <= 0.10 and all(v > 0 for v in out[k]["par_election"].values())
             for k in ("d_crps", "d_ls"))
    out["verdict"] = "hier préférée" if ok else "pas de différence établie : abst reste le défaut"
    return out
