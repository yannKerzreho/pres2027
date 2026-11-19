"""Stabilité entre chaînes des duels 2027 (spec_second_tour §8.45).

    python -m model.backtest.stabilite_2027 --chaine k      # k = 0..3, un processus chacun
    python -m model.backtest.stabilite_2027 --analyser
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

import model.backtest.second_round as sr
from model.backtest import v1_2027 as v
from model.core.live_dataset import load_raw_polls
from model.models.spatial_pooling.fit import duel_draws, fit_spatial_pooling, paire_sondee

VARIANTE = "hier"
STEM = "stabilite_2027_hier"
SEUIL_PT = 2.0


def _fit(runner):
    raw = load_raw_polls(tour=None)
    as_of = str(pd.to_datetime(raw["date_fin"]).max().date())
    fit = fit_spatial_pooling(raw, as_of=as_of, second_tour=True, abstention=True,
                              **v.VARIANTES[VARIANTE], draws=400, tune=600, chains=1, mcmc_runner=runner)
    return fit, as_of


def _paires(fit):
    tm = fit.tested_mask > 0
    modal = Counter(tuple(np.flatnonzero(r)) for r in tm).most_common(1)[0][0]
    cands = sorted({fit.candidates[i] for i in modal} | set(fit.diagnostics.get("candidats_en_duel", [])))
    return [tuple(sorted(p)) for p in combinations(cands, 2)]


def ecarts(moyennes: dict, retenues: list[int]) -> dict:
    """moyennes : {paire: [s̄ par chaîne]} -> écarts max − min (pt) sur toutes / retenues."""
    out = {}
    for p, m in moyennes.items():
        m = np.asarray(m)
        out[p] = {"ecart_toutes_pt": round(100 * float(m.max() - m.min()), 2),
                  "ecart_retenues_pt": round(100 * float(m[retenues].max() - m[retenues].min()), 2)
                  if len(retenues) > 1 else None}
    return out


def analyser() -> dict:
    paths = [sr.CHAINES_DIR / f"{STEM}_c{k}.npz" for k in range(sr.N_CHAINES)]
    fit_all, as_of = _fit(sr._runner_fusion(paths))           # sélection §8.8 sur les 4 chaînes
    retenues = list(sr.CHAINES_RETENUES)
    lp = list(sr.LP_CHAINES)
    paires = _paires(fit_all)
    moy = {f"{a}|{b}": [] for a, b in paires}
    ic = {}
    for k in range(sr.N_CHAINES):
        fit_k, _ = _fit(sr._runner_fusion([paths[k]]))
        for a, b in paires:
            moy[f"{a}|{b}"].append(float(duel_draws(fit_k, a, b, 0.0, scrutin=False).mean()))
    for a, b in paires:
        d = duel_draws(fit_all, a, b, 0.0, scrutin=True)
        lo, hi = np.percentile(d, [5, 95])
        ic[f"{a}|{b}"] = {"demi_ic90_publie_pt": round(50 * float(hi - lo), 2),
                          "sondee": bool(paire_sondee(fit_all, a, b))}
    e = ecarts(moy, retenues)
    lignes = [{"paire": p, **e[p], **ic[p], "moyennes_par_chaine": [round(100 * x, 2) for x in moy[p]]} for p in moy]
    t = pd.DataFrame(lignes)
    col = "ecart_retenues_pt" if len(retenues) > 1 else "ecart_toutes_pt"
    instables = t[t[col] > SEUIL_PT]
    return {"spec": "spec_second_tour §8.45", "variante": VARIANTE, "as_of": as_of, "lp_par_chaine": lp,
            "chaines_retenues": retenues, "critere_colonne": col, "seuil_pt": SEUIL_PT,
            "n_paires": len(t), "n_instables": int(len(instables)),
            "n_instables_au_dela_demi_ic90": int((instables[col] > instables.demi_ic90_publie_pt).sum()),
            "paires": lignes}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chaine", type=int, default=None)
    ap.add_argument("--analyser", action="store_true")
    a = ap.parse_args()
    if a.chaine is not None:
        _fit(sr._runner_chaine(sr.CHAINES_DIR / f"{STEM}_c{a.chaine}.npz", a.chaine))
        return
    if not a.analyser:
        ap.error("--chaine k ou --analyser")
    out = analyser()
    dest = sr.OUT_DIR / f"stabilite_2027_{VARIANTE}_{out['as_of']}.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + "\n")
    t = pd.DataFrame(out["paires"]).sort_values(out["critere_colonne"], ascending=False)
    print({k: out[k] for k in ("as_of", "lp_par_chaine", "chaines_retenues", "n_paires", "n_instables",
                               "n_instables_au_dela_demi_ic90")})
    print(t.head(25).to_string(index=False))
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
