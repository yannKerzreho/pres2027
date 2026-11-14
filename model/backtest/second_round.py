"""Backtest du second tour de spatial-pooling (spec_second_tour.md §8).

    python -m model.backtest.second_round                 # tout (≈ 13 fits NUTS : Colab)
    python -m model.backtest.second_round --fast          # écran de fumée (1 chaîne)
    python -m model.backtest.second_round --only v2 --election 2022

V1 « paire retirée » : à T1−1 j, retirer TOUS les sondages d'une paire, ajuster,
prédire la moyenne de ces sondages sur les 30 derniers jours. C'est le test des
reports : la paire n'est vue qu'à travers la géométrie et les autres duels.
Comparé, sur les MÊMES tirages, à la géométrie seule (κ = 1, η = 0) et à un
Bradley-Terry pur (η seul, ajusté sur les autres paires).

V2 « résultat » : à T1−30 j, T1−1 j et T2−7 j, prédire le résultat réel du duel
final (dérive OU jusqu'au T2 + erreur de scrutin). Couverture IC90, erreur,
log-score, contre la moyenne brute des 7 derniers jours de sondages de la paire
± la même erreur de scrutin.

Limites connues (relecture du 2026-11-14) : TAU_T2_SCRUTIN est calibré sur
les erreurs 2017/2022 elles-mêmes (couverture V2 DANS l'échantillon) ; le
champ T1 réel est connu a posteriori (`names_t1`) ; les priors κ/η ont été
choisis en regardant 2027 ; 2017 est presque sans dates de terrain.

Aucun verdict automatique : deux élections ne valident rien statistiquement.
Le rapport est écrit dans `model/backtest/results/` pour être lu.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pipeline.historical import (PARSED_DIR, POLL_FILES, _resolve_candidate,
                                 _select_best_hypothesis, load_results)
from model.models.spatial_pooling.fit import (TAU_T2_SCRUTIN, duel_draws,
                                              fit_spatial_pooling)
from model.models.spatial_pooling.geometry import duel_shares, log_attractivite
from model.models.spatial_pooling.roster import ORDER_GROUPS_HISTORIQUES

T1_DATES = {2017: pd.Timestamp("2017-04-23"), 2022: pd.Timestamp("2022-04-10")}
T2_DATES = {2017: pd.Timestamp("2017-05-07"), 2022: pd.Timestamp("2022-04-24")}
FINALE = {2017: ("Emmanuel Macron", "Marine Le Pen"), 2022: ("Emmanuel Macron", "Marine Le Pen")}
MIN_SONDAGES_PAIRE = 3
OUT_DIR = Path(__file__).resolve().parent / "results"


def polls_historiques(election: int, as_of: pd.Timestamp, history_days: int = 180) -> pd.DataFrame:
    """Sondages wiki ≤ as_of dans le vocabulaire des résultats officiels.
    T1 : une hypothèse par notice (la plus proche du champ final), candidats du
    scrutin. T2 : TOUTES les paires de deux candidats du scrutin (les duels
    hypothétiques sont ceux qui renseignent les reports)."""
    raw = pd.read_csv(Path(PARSED_DIR) / POLL_FILES[election])
    res = load_results()
    res_e = res[res.election.eq(election)]
    names = set(res_e["candidat"])
    names_t1 = set(res_e.loc[res_e.tour.eq("Premier tour"), "candidat"])
    raw["candidat"] = raw["candidat"].apply(lambda x: _resolve_candidate(x, names))
    dates = pd.to_datetime(raw["date_fin"], errors="coerce")
    raw["date_fin"] = dates.fillna(pd.to_datetime(raw.get("date_notice"), errors="coerce"))
    raw = raw[raw["date_fin"].notna() & raw["date_fin"].le(as_of)
              & raw["date_fin"].ge(as_of - pd.Timedelta(days=history_days))].copy()
    t1 = _select_best_hypothesis(raw[raw.tour.eq("Premier tour") & raw.candidat.isin(names_t1)],
                                 names_t1)
    t2 = raw[raw.tour.eq("Deuxième tour")].copy()
    t2["_h"] = t2["hypothese"].fillna("__u__")
    ok = t2.groupby(["notice", "_h"])["candidat"].transform(
        lambda x: len(x) == 2 and x.isin(names_t1).all() and x.nunique() == 2)
    t2 = t2[ok.astype(bool)].drop(columns="_h")
    return pd.concat([t1, t2], ignore_index=True)


def _duels_moyens(raw: pd.DataFrame, depuis: pd.Timestamp | None = None) -> pd.DataFrame:
    """Une ligne par duel sondé : paire triée, part du premier nom, date."""
    t2 = raw[raw.tour.eq("Deuxième tour")].copy()
    if depuis is not None:
        t2 = t2[t2.date_fin.ge(depuis)]
    rows = []
    for (_, _), g in t2.groupby(["notice", t2["hypothese"].fillna("__u__")]):
        g = g.sort_values("candidat")
        v = g.intention.to_numpy(float)
        rows.append((g.candidat.iloc[0], g.candidat.iloc[1], v[0] / v.sum(), g.date_fin.iloc[0]))
    return pd.DataFrame(rows, columns=["a", "b", "y", "date"])


def _sans_paire(raw: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    t2 = raw.tour.eq("Deuxième tour")
    key = raw["notice"].astype(str) + "|" + raw["hypothese"].fillna("__u__").astype(str)
    paires = raw[t2].groupby(key[t2])["candidat"].agg(lambda x: frozenset(x))
    drop = set(paires[paires.eq(frozenset((a, b)))].index)
    return raw[~(t2 & key.isin(drop))]


def _geometrie_seule(fit, a: str, b: str) -> np.ndarray:
    i = {c: k for k, c in enumerate(fit.candidates)}
    A = np.asarray(log_attractivite(np.asarray(fit.mu_draws), np.asarray(fit.sigma_draws),
                                    np.asarray(fit.w_draws)))
    return np.asarray(duel_shares(A[:, i[a]], A[:, i[b]], 1.0, 0.0, 0.0))


def _bradley_terry(train: pd.DataFrame, a: str, b: str, sd_eta: float = 0.5) -> float:
    """η seul, ridge (même prior que le modèle), sur la moyenne logit par paire."""
    m = train.assign(z=np.log(train.y / (1 - train.y))).groupby(["a", "b"]).z.agg(["mean", "count"])
    cands = sorted(set(m.index.get_level_values(0)) | set(m.index.get_level_values(1)) | {a, b})
    ci = {c: k for k, c in enumerate(cands)}

    def loss(e):
        r = sum(n * (e[ci[x]] - e[ci[y]] - z) ** 2 for (x, y), (z, n) in m.iterrows())
        return r / (2 * 0.1 ** 2) + (e ** 2).sum() / (2 * sd_eta ** 2)
    e = minimize(loss, np.zeros(len(cands)), method="L-BFGS-B").x
    return float(1 / (1 + np.exp(-(e[ci[a]] - e[ci[b]]))))


ABSTENTION = True   # surchargé par --variante


def _fit(raw, as_of, election, nuts):
    t = time.time()
    fit = fit_spatial_pooling(raw, str(as_of.date()), min_poll_date="1900-01-01",
                              order_groups=ORDER_GROUPS_HISTORIQUES[election],
                              couverture_min=None, abstention=ABSTENTION, **nuts)
    return fit, round(time.time() - t, 1)


def _diag(fit, dt):
    d = fit.diagnostics
    return {"rhat_max": d["rhat_max"], "ess_min": d["ess_min"], "divergences": d["n_divergences"],
            "kappa": d.get("kappa"), "omega": d.get("omega"), "rhat_t2": d.get("rhat_t2"), "sigma_t2": d.get("sigma_t2"), "secondes": dt,
            "eta": {c: round(float(x), 3) for c, x in zip(fit.candidates, fit.eta_draws.mean(0))}
            if fit.eta_draws is not None else None}


def v1_paire_retiree(election: int, nuts: dict) -> list[dict]:
    as_of = T1_DATES[election] - pd.Timedelta(days=1)
    raw = polls_historiques(election, as_of)
    duels = _duels_moyens(raw, as_of - pd.Timedelta(days=30))
    tous = _duels_moyens(raw)
    paires = tous.groupby(["a", "b"]).size()
    out = []
    for (a, b), n in paires.items():
        if n < MIN_SONDAGES_PAIRE:
            continue
        cible = duels[(duels.a == a) & (duels.b == b)]
        if cible.empty:
            continue
        print(f"[V1 {election}] paire retirée {a} / {b} ({n} sondages)", flush=True)
        fit, dt = _fit(_sans_paire(raw, a, b), as_of, election, nuts)
        obs = float(cible.y.mean())
        mod = duel_draws(fit, a, b, 0, scrutin=False)
        geo = _geometrie_seule(fit, a, b)
        bt = _bradley_terry(tous[~((tous.a == a) & (tous.b == b))], a, b)
        autres = tous[((tous.a.isin([a, b])) | (tous.b.isin([a, b]))) & ~((tous.a == a) & (tous.b == b))]
        out.append({"election": election, "paire": [a, b], "n_sondages_retires": int(n),
                    "observe_30j": round(obs, 4), "modele": round(float(mod.mean()), 4),
                    "modele_ic90": [round(float(x), 4) for x in np.percentile(mod, [5, 95])],
                    "geometrie_seule": round(float(geo.mean()), 4), "bradley_terry": round(bt, 4),
                    "autres_duels_des_deux": int(len(autres)), **_diag(fit, dt)})
    return out


def _log_score(draws_logit: np.ndarray, y: float) -> float:
    m, s = draws_logit.mean(), draws_logit.std()
    z = np.log(y / (1 - y))
    return float(-0.5 * np.log(2 * np.pi * s ** 2) - (z - m) ** 2 / (2 * s ** 2) - np.log(y * (1 - y)))


def v2_resultat(election: int, nuts: dict, coupures=("T1-30", "T1-1", "T2-7")) -> list[dict]:
    a, b = sorted(FINALE[election])
    res = load_results()
    vrai = res[(res.election == election) & (res.tour == "Deuxième tour")].set_index("candidat")
    y = float(vrai.loc[a, "resultat"]) / (vrai.loc[a, "resultat"] + vrai.loc[b, "resultat"])
    out = []
    for c in coupures:
        ref = T1_DATES[election] if c.startswith("T1") else T2_DATES[election]
        as_of = ref - pd.Timedelta(days=int(c.split("-")[1]))
        print(f"[V2 {election}] {c} ({as_of.date()})", flush=True)
        raw = polls_historiques(election, as_of)
        n_t1 = int(raw.loc[raw.tour.eq("Premier tour"), "notice"].nunique())
        try:
            fit, dt = _fit(raw, as_of, election, nuts)
        except (IndexError, ValueError) as exc:
            # 2017 : la plupart des sondages T1 n'ont aucune date (cf. note
            # données) -- une coupure peut n'avoir aucun nœud T1.
            out.append({"election": election, "coupure": c, "as_of": str(as_of.date()),
                        "erreur": f"{type(exc).__name__}: {exc}", "n_notices_t1": n_t1})
            continue
        h = int((T2_DATES[election] - as_of).days)
        d = duel_draws(fit, a, b, h, scrutin=True)
        recents = _duels_moyens(raw, as_of - pd.Timedelta(days=7))
        recents = recents[(recents.a == a) & (recents.b == b)]
        if recents.empty:
            recents = _duels_moyens(raw).query("a == @a and b == @b").tail(5)
        p0 = float(recents.y.mean())
        rng = np.random.default_rng(0)
        # MÊME variance totale que le modèle (logit) : on ne compare que les centres.
        zd = np.log(d / (1 - d))
        base = np.log(p0 / (1 - p0)) + rng.normal(0, zd.std(), 4000)
        out.append({"election": election, "coupure": c, "as_of": str(as_of.date()),
                    "candidat": a, "resultat": round(y, 4), "n_notices_t1": n_t1,
                    "modele": round(float(d.mean()), 4),
                    "modele_ic90": [round(float(x), 4) for x in np.percentile(d, [5, 95])],
                    "couvre_ic90": bool(np.percentile(d, 5) <= y <= np.percentile(d, 95)),
                    "log_score": round(_log_score(np.log(d / (1 - d)), y), 3),
                    "sondages_7j": round(p0, 4), "n_sondages_7j": int(len(recents)),
                    "log_score_sondages": round(_log_score(base, y), 3),
                    "couvre_ic90_sondages": bool(np.percentile(1 / (1 + np.exp(-base)), 5) <= y
                                                 <= np.percentile(1 / (1 + np.exp(-base)), 95)),
                    **_diag(fit, dt)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--only", choices=["v1", "v2"], default=None)
    ap.add_argument("--election", type=int, choices=[2017, 2022], default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--variante", choices=["abst", "eta"], default="abst",
                    help="abst = abstention (valeur inclusive) + valence ; eta = valence seule")
    args = ap.parse_args()
    global ABSTENTION
    ABSTENTION = args.variante == "abst"
    nuts = dict(draws=60, tune=60, chains=1) if args.fast else dict(draws=400, tune=600, chains=4)
    rows = {"v1": [], "v2": []}
    for e in ([args.election] if args.election else [2017, 2022]):
        if args.only in (None, "v1"):
            rows["v1"] += v1_paire_retiree(e, nuts)
        if args.only in (None, "v2"):
            rows["v2"] += v2_resultat(e, nuts)
    OUT_DIR.mkdir(exist_ok=True)
    out = args.out or OUT_DIR / (f"second_round_{args.variante}_{'fast_' if args.fast else ''}"
                                 f"{args.only or 'all'}_{args.election or 'both'}.json")
    out.write_text(json.dumps({"variante": args.variante, "nuts": nuts, "tau_t2_scrutin": TAU_T2_SCRUTIN, **rows},
                              ensure_ascii=False, indent=1) + "\n")
    for k, r in rows.items():
        if r:
            cols = [c for c in pd.DataFrame(r).columns if c != "eta"]
            print(f"\n== {k} ==\n" + pd.DataFrame(r)[cols].to_string(index=False))
    if rows["v1"]:
        t = pd.DataFrame(rows["v1"])
        for c in ("modele", "geometrie_seule", "bradley_terry"):
            print(f"V1 MAE {c}: {100 * (t[c] - t.observe_30j).abs().mean():.2f} pt")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
