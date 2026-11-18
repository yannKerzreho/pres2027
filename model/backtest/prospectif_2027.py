"""Test prospectif 2027 (revue du 18/11, idée 1 ; spec_second_tour §8.37).

1) `--figer --variante abst|hier` : un fit sur les sondages à date, puis tirages
   de la part du premier nommé pour TOUTES les paires du champ modal ∪ candidats
   en duel, aux horizons 0, 14 et 30 jours (sans aléa de scrutin), figés dans
   `results/prospectif/prediction_<variante>_<as_of>.json`. Aucune retouche ensuite.
2) `--scorer` : sans fit, chaque duel publié APRÈS la date de gel est scoré
   contre la prédiction figée la plus proche en horizon (prédictif d'UN sondage :
   + bruit binomial et σ_T2) : erreur, CRPS, z, couverture IC90.

    python -m model.backtest.prospectif_2027 --figer --variante hier
    python -m model.backtest.prospectif_2027 --scorer
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from model.backtest import v1_2027 as v
from model.backtest.verdict_abst_hier import crps_tirages
from model.core.live_dataset import load_raw_polls
from model.models.spatial_pooling.fit import duel_draws, fit_spatial_pooling

DOSSIER = Path(__file__).resolve().parent / "results" / "prospectif"
HORIZONS = (0, 14, 30)
N_TIRAGES = 400


def figer(variante: str) -> Path:
    raw = load_raw_polls(tour=None)
    as_of = str(pd.to_datetime(raw["date_fin"]).max().date())
    fit = fit_spatial_pooling(raw, as_of=as_of, second_tour=True, abstention=True,
                              **v.VARIANTES[variante], draws=400, tune=600)
    tm = fit.tested_mask > 0
    modal = Counter(tuple(np.flatnonzero(r)) for r in tm).most_common(1)[0][0]
    cands = sorted({fit.candidates[i] for i in modal}
                   | set(fit.diagnostics.get("candidats_en_duel", [])))
    rng = np.random.default_rng(0)
    paires = {}
    for a, b in combinations(cands, 2):
        a, b = sorted((a, b))
        par_h = {}
        for h in HORIZONS:
            d = duel_draws(fit, a, b, h, scrutin=False)
            idx = rng.choice(len(d), min(N_TIRAGES, len(d)), replace=False)
            par_h[str(h)] = [round(float(x), 4) for x in d[idx]]
        paires[f"{a}|{b}"] = par_h
    out = DOSSIER / f"prediction_{variante}_{as_of}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"variante": variante, "as_of": as_of, "sigma_t2": fit.diagnostics.get("sigma_t2"),
                               "diagnostics": {k: fit.diagnostics.get(k) for k in ("rhat_max", "rhat_t2", "n_duels")},
                               "paires_sondees": sorted({f"{x}|{y}" for x, y in (tuple(sorted(p)) for p in fit.duel_pairs)}),
                               "paires": paires}, ensure_ascii=False) + "\n")
    return out


def scorer() -> dict:
    raw = load_raw_polls(tour=None)
    duels = v._duels(raw)
    res = {}
    for f in sorted(glob.glob(str(DOSSIER / "prediction_*.json"))):
        p = json.loads(Path(f).read_text())
        t0 = pd.Timestamp(p["as_of"])
        nouv = duels[duels.date > t0]
        rows = []
        for r in nouv.itertuples():
            cle = f"{r.a}|{r.b}"
            if cle not in p["paires"]:
                continue
            h = int((r.date - t0).days)
            hh = min(HORIZONS, key=lambda x: abs(x - h))
            d = np.asarray(p["paires"][cle][str(hh)])
            sd = np.sqrt(d * (1 - d) / max(r.n_ech, 1) + (p["sigma_t2"] or 0) ** 2)
            dp = np.clip(d + sd * np.random.default_rng(1).standard_normal(len(d)), 1e-4, 1 - 1e-4)
            z = (np.log(r.y / (1 - r.y)) - np.log(dp / (1 - dp)).mean()) / np.log(dp / (1 - dp)).std()
            lo, hi = np.percentile(dp, [5, 95])
            rows.append({"paire": cle, "date": str(r.date.date()), "horizon_j": h, "observe": round(r.y, 4),
                         "predit": round(float(dp.mean()), 4), "erreur_pt": round(100 * abs(dp.mean() - r.y), 2),
                         "crps_pt": round(crps_tirages(dp, r.y), 3), "z": round(float(z), 2),
                         "couvre_ic90": bool(lo <= r.y <= hi), "paire_sondee_au_gel": cle in p["paires_sondees"]})
        df = pd.DataFrame(rows)
        res[Path(f).stem] = {"as_of": p["as_of"], "n_duels_nouveaux": len(rows),
                             "resume": ({"mae_pt": round(df.erreur_pt.mean(), 2), "crps_pt": round(df.crps_pt.mean(), 3),
                                         "rms_z": round(float(np.sqrt((df.z ** 2).mean())), 2),
                                         "couverture_ic90": round(df.couvre_ic90.mean(), 3)} if len(df) else None),
                             "duels": rows}
    out = DOSSIER / "scores.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figer", action="store_true")
    ap.add_argument("--variante", choices=list(v.VARIANTES), default="hier")
    ap.add_argument("--scorer", action="store_true")
    a = ap.parse_args()
    if a.figer:
        print(figer(a.variante))
    if a.scorer:
        print(json.dumps({k: (x["n_duels_nouveaux"], x["resume"]) for k, x in scorer().items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
