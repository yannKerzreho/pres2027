"""V1 2027 : duel retiré dans le régime de production (spec_second_tour.md §8.12).

    python -m model.backtest.v1_2027 --lister
    python -m model.backtest.v1_2027 --tache "Le Pen|Philippe" --variante hier --chaine k
    python -m model.backtest.v1_2027 --tache "Le Pen|Philippe" --variante hier --fusionner
    python -m model.backtest.v1_2027 --tache "candidat:Attal" ...

Même découpage par chaîne, même sélection par log-densité et même critère de
convergence que `model.backtest.second_round` (spec §8.4, §8.8, §8.11).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model.backtest import second_round as sr
from model.core.live_dataset import load_raw_polls
from model.models.spatial_pooling.fit import duel_draws, fit_spatial_pooling
from model.models.spatial_pooling.roster import split_tours

PAIRES = ["Le Pen|Philippe", "Le Pen|Mélenchon", "Attal|Le Pen", "Glucksmann|Le Pen",
          "Bardella|Philippe", "Bardella|Mélenchon", "Attal|Bardella", "Hollande|Le Pen"]
CANDIDATS = ["candidat:Attal", "candidat:Bardella"]
VARIANTES = {"abst": {}, "hier": {"eta_hier": True}}
FENETRE = 30


def _duels(raw: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par duel sondé (notice, hypothèse) : paire triée, part du 1er, date."""
    t2 = split_tours(raw)[1].copy()
    t2["_h"] = t2["hypothese"].fillna("__u__")
    rows = []
    for (n, h), g in t2.groupby(["notice", "_h"], sort=False):
        if len(g) != 2 or g.candidat.nunique() != 2:
            continue
        g = g.sort_values("candidat")
        v = g.intention.to_numpy(float)
        rows.append((n, h, g.candidat.iloc[0], g.candidat.iloc[1], v[0] / v.sum(),
                     pd.Timestamp(g.date_fin.iloc[0])))
    return pd.DataFrame(rows, columns=["notice", "_h", "a", "b", "y", "date"])


def paires_cibles(tache: str, duels: pd.DataFrame) -> list[tuple[str, str]]:
    if tache.startswith("candidat:"):
        c = tache.split(":", 1)[1]
        p = duels[(duels.a == c) | (duels.b == c)][["a", "b"]].drop_duplicates()
        return [tuple(r) for r in p.itertuples(index=False)]
    return [tuple(sorted(tache.split("|")))]


def jeu_de_donnees(raw: pd.DataFrame, tache: str):
    """(sondages d'apprentissage, t_k, cibles) — spec §8.12."""
    duels = _duels(raw)
    cibles = paires_cibles(tache, duels)
    mask = duels[["a", "b"]].apply(tuple, axis=1).isin(cibles)
    t_k = duels.loc[mask, "date"].max()
    retirer = set(map(tuple, duels.loc[mask, ["notice", "_h"]].to_numpy()))
    cle = list(zip(raw["notice"], raw["hypothese"].fillna("__u__")))
    t2 = raw["tour"].ne("Premier tour").to_numpy()
    garde = ~(t2 & np.array([k in retirer for k in cle]))
    train = raw[garde & (pd.to_datetime(raw["date_fin"]) <= t_k).to_numpy()]
    fen = duels[mask & duels.date.ge(t_k - pd.Timedelta(days=FENETRE))]
    obs = {p: (float(g.y.mean()), int(len(g))) for p, g in fen.groupby(["a", "b"])}
    return train, t_k, obs


def _log_score_logit(d: np.ndarray, y: float) -> float:
    z = np.log(d / (1 - d))
    m, s = z.mean(), z.std()
    zy = np.log(y / (1 - y))
    return float(-0.5 * np.log(2 * np.pi * s ** 2) - (zy - m) ** 2 / (2 * s ** 2) - np.log(y * (1 - y)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lister", action="store_true")
    ap.add_argument("--tache")
    ap.add_argument("--variante", choices=list(VARIANTES), default="hier")
    ap.add_argument("--chaine", type=int, default=None)
    ap.add_argument("--fusionner", action="store_true")
    ap.add_argument("--init-depuis", default=None)
    a = ap.parse_args()
    if a.lister:
        for t in PAIRES + CANDIDATS:
            print(f'--tache "{t}"')
        return
    raw = load_raw_polls(tour=None)
    train, t_k, obs = jeu_de_donnees(raw, a.tache)
    nom = a.tache.replace("candidat:", "cand-").replace("|", "-").replace(" ", "_")
    out = sr.OUT_DIR / (f"v1_2027_{a.variante}_{nom}" + ("_init" if a.init_depuis else "") + ".json")
    base = sr.CHAINES_DIR / out.stem
    nuts = dict(draws=400, tune=600, chains=4)
    if a.chaine is not None:
        init = sr._init_mode_dominant(a.init_depuis) if a.init_depuis else None
        nuts["mcmc_runner"] = sr._runner_chaine(base.with_name(f"{out.stem}_c{a.chaine}.npz"), a.chaine, init)
        out = base.with_name(f"{out.stem}_c{a.chaine}.json")
    elif a.fusionner:
        nuts["mcmc_runner"] = sr._runner_fusion([base.with_name(f"{out.stem}_c{k}.npz")
                                                 for k in range(sr.N_CHAINES)])
    fit = fit_spatial_pooling(train, as_of=str(t_k.date()), second_tour=True, abstention=True,
                              **VARIANTES[a.variante], **nuts)
    lignes = []
    for (x, y), (yo, n) in obs.items():
        d = duel_draws(fit, x, y, 0, scrutin=False)
        lignes.append({"paire": [x, y], "observe_30j": round(yo, 4), "n_cible": n,
                       "modele": round(float(d.mean()), 4),
                       "modele_ic90": [round(float(v), 4) for v in np.percentile(d, [5, 95])],
                       "erreur_pt": round(100 * abs(float(d.mean()) - yo), 2),
                       "log_score": round(_log_score_logit(d, yo), 3),
                       **sr._cible_par_chaine(d, 1 if a.chaine is not None else 4)})
    res = {"tache": a.tache, "variante": a.variante, "t_k": str(t_k.date()),
           "diagnostics": {k: fit.diagnostics.get(k) for k in
                           ("rhat_max", "rhat_t2", "ess_min", "n_divergences", "kappa", "omega",
                            "sigma_t2", "n_duels")},
           "lignes": lignes}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:2000])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
