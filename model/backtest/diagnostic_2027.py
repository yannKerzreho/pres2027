"""Diagnostic 2027 du bloc second tour (JOURNAL_T2, tâche prioritaire).

Fit joint T1+T2 sur les sondages 2027 (variante abst par défaut), puis :
  (a) résidus des duels sondés, par paire et par institut (moyenne, écart-type,
      résidu standardisé z et sa variance — ≈ 1 si l'incertitude est juste) ;
  (b) tendance : corrélation résidu/date, résidu moyen des 30 derniers jours,
      et duel du modèle AUJOURD'HUI contre la moyenne brute des 30 derniers jours ;
  (c) ce qu'afficherait le site (`resume_duel` à l'horizon du T2, `p_elu` du
      champ modal) ;
  + diagnostics de convergence et corr(log κ, η_i).

Le résidu est pris contre la part prédite À LA DATE du sondage (`duel_s`), pas
contre le nowcast : sinon une vraie tendance passerait pour un biais.

    python -m model.backtest.diagnostic_2027 [--variante abst|eta] [--draws 400 --tune 600]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from model.core.base import ELECTION_T1
from model.core.live_dataset import ELECTION_T2, load_raw_polls
from model.models.spatial_pooling.fit import fit_spatial_pooling, p_elu, resume_duel

RESULTS = Path(__file__).resolve().parent / "results"
FENETRE_RECENTE = 30


def _stats(g: pd.DataFrame) -> dict:
    return {"n": int(len(g)),
            "resid_moy_pt": round(100 * g.resid.mean(), 2),
            "resid_sd_pt": round(100 * g.resid.std(ddof=0), 2),
            "z_moy": round(g.z.mean(), 2),
            "z_var": round(float((g.z ** 2).mean()), 2),
            "sd_pred_moy_pt": round(100 * g.sd_pred.mean(), 2)}


def diagnostic(variante: str = "abst", draws: int = 400, tune: int = 600, seed: int = 27) -> dict:
    raw = load_raw_polls(tour=None)
    as_of = str(pd.to_datetime(raw["date_fin"]).max().date())
    fit = fit_spatial_pooling(raw, as_of=as_of, draws=draws, tune=tune, seed=seed,
                              second_tour=True, abstention=(variante in ("abst", "hier")),
                              eta_hier=(variante == "hier"))
    d = fit.duels
    S = fit.duel_s_draws                                            # (S, D)
    sigma_t2 = fit.diagnostics["sigma_t2"]
    m = S.mean(0)
    sd_pred = np.sqrt(S.var(0) + m * (1 - m) / np.clip(d["n"], 1, None) + sigma_t2 ** 2)
    inst = (raw.drop_duplicates("notice").set_index("notice")["institut"])
    df = pd.DataFrame({
        "paire": [f"{a} / {b}" for a, b in d["pairs"]],
        "a": [a for a, _ in d["pairs"]],
        "institut": [inst.get(nt, "?") for nt in d["notices"]],
        "date": [pd.Timestamp.fromordinal(int(x)) for x in d["dates"]],
        "y": d["y"], "pred": m, "sd_pred": sd_pred,
    })
    df["resid"] = df.y - df.pred
    df["z"] = df.resid / df.sd_pred
    t_now = pd.Timestamp(as_of)
    df["age_j"] = (t_now - df.date).dt.days

    par_paire = {}
    h2 = max(int((ELECTION_T2 - t_now).days), 0)
    for p, g in df.groupby("paire"):
        a, b = p.split(" / ")
        rec = g[g.age_j <= FENETRE_RECENTE]
        now = resume_duel(fit, a, b, 0)           # aujourd'hui (+ aléa de scrutin)
        site = resume_duel(fit, a, b, h2, len(g))
        e = {**_stats(g),
             "corr_resid_date": (round(float(np.corrcoef(g.resid, g.date.map(pd.Timestamp.toordinal))[0, 1]), 2)
                                 if len(g) > 2 and g.resid.std() > 0 else None),
             "n_recents": int(len(rec)),
             "resid_moy_recents_pt": round(100 * rec.resid.mean(), 2) if len(rec) else None,
             "brut_recent_a": round(float(rec.y.mean()), 4) if len(rec) else None,
             "modele_recent_a_aux_dates": round(float(rec.pred.mean()), 4) if len(rec) else None,
             "modele_aujourdhui_a": now["intentions"][a]["moyenne"],
             "site_a": site["intentions"][a]["moyenne"], "site_ic90_a": site["intentions"][a]["ic90"],
             "site_p_gagne_a": site["intentions"][a]["p_gagne"],
             "dernier_sondage": str(g.date.max().date())}
        par_paire[p] = e
    par_institut = {k: _stats(g) for k, g in df.groupby("institut")}
    # Biais par candidat : résidu signé du point de vue du candidat.
    rows = []
    for _, r in df.iterrows():
        a, b = r.paire.split(" / ")
        rows += [(a, r.resid, r.z), (b, -r.resid, -r.z)]
    pc = pd.DataFrame(rows, columns=["candidat", "resid", "z"])
    par_candidat = {c: {"n": int(len(g)), "resid_moy_pt": round(100 * g.resid.mean(), 2),
                        "z_moy": round(g.z.mean(), 2)} for c, g in pc.groupby("candidat")}

    # corr(log κ, η_i) : crête κ–η (revue 2026-11-14).
    idx = {c: i for i, c in enumerate(fit.candidates)}
    lk = np.log(fit.kappa_draws)
    corr_k_eta = {c: round(float(np.corrcoef(lk, fit.eta_draws[:, idx[c]])[0, 1]), 2)
                  for c in fit.diagnostics["candidats_en_duel"]}

    tm = fit.tested_mask > 0
    modal = Counter(tuple(np.flatnonzero(r)) for r in tm).most_common(1)[0][0]
    mask = np.zeros(len(fit.candidates)); mask[list(modal)] = 1
    h1 = max(int((ELECTION_T1 - t_now).days), 0)
    pe = p_elu(fit, mask, h1, h2)

    return {"as_of": as_of, "variante": variante, "draws": draws, "tune": tune, "seed": seed,
            "diagnostics": fit.diagnostics, "corr_logkappa_eta": corr_k_eta,
            "global": _stats(df), "par_paire": par_paire, "par_institut": par_institut,
            "par_candidat": par_candidat, "p_elu_champ_modal": pe,
            "champ_modal": [fit.candidates[i] for i in modal],
            "residus": [{k: (str(v.date()) if k == "date" else (round(float(v), 4) if isinstance(v, (float, np.floating)) else v))
                         for k, v in r.items()} for r in df.drop(columns=["a"]).to_dict("records")]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variante", default="abst", choices=["abst", "eta", "hier"])
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--tune", type=int, default=600)
    ap.add_argument("--seed", type=int, default=27)
    a = ap.parse_args()
    res = diagnostic(a.variante, a.draws, a.tune, a.seed)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"diagnostic_2027_{a.variante}_{res['as_of']}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(out)
    print(json.dumps({k: res[k] for k in ("global", "par_paire", "corr_logkappa_eta")},
                     ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
