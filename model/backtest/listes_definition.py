"""Définition des listes gp-pooling : D0 champ exact contre D1(s) sans les petits
candidats (protocole : `model/backtest/results/listes_definition.md`).

    python -m model.backtest.listes_definition --election 2022

gp-pooling est en forme close : aucune MCMC, quelques secondes par coupure.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from model.backtest import second_round as sr
from model.backtest import suivi as su
from model.core.base import ELECTION_T1
from model.listes import GPListe, classement, cle_champ
from model.models.gp_pooling.model import _softmax
from model.core.projection import delta_draws
from pipeline.historical import load_results

HS = [180, 150, 120, 90, 60, 30, 7]
SEUILS = [0.01, 0.02, 0.03, 0.05]
NIVEAUX = (0.5, 0.8, 0.9)


class GPReduit(GPListe):
    """gp-pooling inchangé, alimenté par les hypothèses dont l'ensemble RÉDUIT
    (petits candidats retirés) vaut `candidats` ; parts renormalisées en aval."""

    def __init__(self, candidats: list[str], petits: set[str]):
        super().__init__(cle_champ(candidats), candidats)
        self.petits = set(petits)

    def used_polls(self, raw_polls):
        d = raw_polls[~raw_polls["candidat"].isin(self.petits)]
        red = d.groupby("scenario_id")["candidat"].agg(lambda x: cle_champ(set(x)))
        sids = set(red[red == self.cle].index)
        sub = d[d["scenario_id"].isin(sids)].copy()
        sub["slot"] = sub["candidat"].map(self.labels)
        return sub.reset_index(drop=True)


class GPSlots(GPListe):
    """D2 : l'unité est le SLOT gp-pooling quand le candidat en a un (Bardella et
    Le Pen -> « RN »), le candidat lui-même sinon ; on regroupe les hypothèses
    dont l'ensemble d'unités vaut celui de la liste (somme par slot en aval)."""

    def __init__(self, unites: list[str]):
        super().__init__(cle_champ(unites), unites)

    def used_polls(self, raw_polls):
        d = raw_polls.assign(_u=raw_polls["slot"].fillna(raw_polls["candidat"]).astype(str))
        cles = d.groupby("scenario_id")["_u"].agg(lambda x: cle_champ(set(x)))
        sub = d[d["scenario_id"].isin(set(cles[cles == self.cle].index))].copy()
        sub["slot"] = sub["_u"].map(self.labels)
        return sub.drop(columns="_u").reset_index(drop=True)


def _prep(raw: pd.DataFrame, delta) -> pd.DataFrame:
    t1 = raw[raw.tour.eq("Premier tour")].copy()
    t1["date_fin"] = pd.to_datetime(t1["date_fin"]) + delta
    t1["scenario_id"] = t1["notice"].astype(str) + "|" + t1["hypothese"].fillna("__u__").astype(str)
    t1["bloc"] = "?"
    return t1


def _futurs(election, c, cle, jours) -> tuple[pd.DataFrame, int]:
    fut = sr.polls_historiques(election, c + pd.Timedelta(days=jours), history_days=jours)
    fut = fut[fut.tour.eq("Premier tour") & fut.date_fin.gt(c)]
    if fut.empty:
        return fut, 0
    ch = su._champs_t1(fut)
    sids = set(ch.loc[ch.cle.eq(cle), "scenario_id"])
    fut = fut.assign(scenario_id=fut["notice"].astype(str) + "|" + fut["hypothese"].fillna("__u__").astype(str))
    sel = fut[fut.scenario_id.isin(sids)]
    return sel, int(sel.scenario_id.nunique())


def _scores(draws: np.ndarray, noms: list[str], cible: dict, n: int, rng) -> list[dict]:
    """draws (S, K) sur `noms` (déjà renormalisés) ; cible {nom: part}."""
    out = []
    for j, x in enumerate(noms):
        if x not in cible:
            continue
        d = draws[:, j]
        dp = d + rng.standard_normal(d.shape) * np.sqrt(d * (1 - d) / 1000 / max(n, 1))
        e = {"candidat": x, "observe": round(cible[x], 4), "modele": round(float(d.mean()), 4),
             "erreur_pt": round(100 * abs(float(d.mean()) - cible[x]), 3)}
        for q in NIVEAUX:
            lo, hi = np.percentile(dp, [50 * (1 - q), 100 - 50 * (1 - q)])
            e[f"couvre_{int(100 * q)}"] = bool(lo <= cible[x] <= hi)
        out.append(e)
    return out


def evaluer(election: int, h: int) -> list[dict]:
    c = sr.T1_DATES[election] - pd.Timedelta(days=h)
    raw = sr.polls_historiques(election, c)
    if raw[raw.tour.eq("Premier tour")].empty:
        return []
    delta = ELECTION_T1 - sr.T1_DATES[election]
    t1 = _prep(raw, delta)
    as_of = str((c + delta).date())
    parts = t1.groupby("candidat").intention.mean() / 100        # antérieurs à c
    res = load_results()
    vrai = res[(res.election == election) & (res.tour == "Premier tour")].set_index("candidat")["resultat"]
    rng = np.random.default_rng(5)
    lignes = []
    for L in classement(su._champs_t1(raw), c):
        cand = list(L["candidats"])
        # D0 une fois : tirages sur L (nowcast et scrutin).
        try:
            m0 = GPListe(L["cle"], cand)
            m0.load_artifacts()
            n0 = int(m0.used_polls(t1)["notice"].nunique())
            nc0 = m0.nowcast(t1, as_of)
        except Exception as exc:
            lignes.append({"h": h, "liste": L["id"], "definition": "D0", "erreur": str(exc)[:120]})
            continue
        d0 = np.stack([np.asarray(nc0.draws.sel(candidat=m0.labels[x])) for x in cand], axis=1)
        th0 = m0._theta_scrutin
        s0 = _softmax(th0 + delta_draws(m0.params, th0.shape, m0.seed + 3))
        for s in [None] + SEUILS:
            petits = set() if s is None else {x for x in cand if parts.get(x, 0) < s}
            red = [x for x in cand if x not in petits]
            if len(red) < 2 or (s is not None and not petits):
                continue                     # D1(s) identique à D0 : rien à comparer
            idx = [cand.index(x) for x in red]
            paires = [("D0", n0, d0[:, idx], s0[:, idx])]
            if s is not None:
                try:
                    m1 = GPReduit(red, petits)
                    m1.load_artifacts()
                    n1 = int(m1.used_polls(t1)["notice"].nunique())
                    nc1 = m1.nowcast(t1, as_of)
                except Exception as exc:
                    lignes.append({"h": h, "liste": L["id"], "definition": f"D1_{int(100 * s)}",
                                   "erreur": str(exc)[:120]})
                    continue
                d1 = np.stack([np.asarray(nc1.draws.sel(candidat=m1.labels[x])) for x in red], axis=1)
                th1 = m1._theta_scrutin
                paires.append((f"D1_{int(100 * s)}", n1, d1,
                               _softmax(th1 + delta_draws(m1.params, th1.shape, m1.seed + 3))))
            for nom, n_courbe, dr, sc in paires:
                dr = dr / dr.sum(1, keepdims=True)          # base commune : L' = red
                sc = sc / sc.sum(1, keepdims=True)
                base = {"h": h, "coupure": str(c.date()), "liste": L["id"], "definition": nom,
                        "base_seuil": s, "n_petits": len(petits), "n_sondages_courbe": n_courbe}
                for jours in (7, 30):
                    fut, n = _futurs(election, c, L["cle"], jours)
                    if n == 0:
                        continue
                    m = fut[fut.candidat.isin(red)].groupby("candidat").intention.mean()
                    for e in _scores(dr, red, (m / m.sum()).to_dict(), n, rng):
                        lignes.append({**base, "cible": f"{jours}j", "n_cible": n, **e})
                if set(red) <= set(vrai.index):
                    v = vrai[red] / vrai[red].sum()
                    for e in _scores(sc, red, v.to_dict(), 10 ** 6, rng):
                        lignes.append({**base, "cible": "resultat_T1", "n_cible": 0, **e})
    return lignes


def diagnostic_2027(as_of: str = "2026-11-13") -> dict:
    """D0 / D1(5 %) / D2 sur les 3 listes de tête 2027 : sondages par courbe,
    résidus aux sondages de la liste EXACTE (dans l'échantillon), régularité
    (|Δ| jour à jour de la trajectoire sur 90 j) et stabilité au rejeu (nowcast
    sans le dernier sondage). Unités comparées : celles communes aux trois."""
    from model.core.live_dataset import load_raw_polls
    from model.listes import champs
    raw = load_raw_polls(as_of=as_of)
    ch = champs(raw)
    parts = raw.groupby("candidat").intention.mean() / 100
    slot_de = raw.dropna(subset=["slot"]).drop_duplicates("candidat").set_index("candidat")["slot"].to_dict()
    out = []
    for L in classement(ch, as_of):
        cand = list(L["candidats"])
        petits = {x for x in cand if parts.get(x, 0) < 0.05}
        defs = {"D0": (GPListe(L["cle"], cand), cand),
                "D1_5": (GPReduit([x for x in cand if x not in petits], petits),
                         [x for x in cand if x not in petits]),
                "D2": (GPSlots(sorted({slot_de.get(x, x) for x in cand})), None)}
        sids = set(ch.loc[ch.cle.eq(L["cle"]), "scenario_id"])
        exact = raw[raw.scenario_id.isin(sids)]
        for nom, (mdl, unites) in defs.items():
            mdl.load_artifacts()
            used = mdl.used_polls(raw)
            nc = mdl.nowcast(raw, as_of)
            tr = nc.trajectoire
            mdl2 = type(mdl).__new__(type(mdl)); mdl2.__dict__ = dict(mdl.__dict__)
            u = used.assign(_d=pd.to_datetime(used["date_fin"]))
            dernier = u.sort_values(["_d", "notice"])["notice"].iloc[-1]   # dernier sondage DE la courbe
            nc2 = mdl2.nowcast(raw[raw.notice != dernier], as_of)
            # unités : candidat -> label du modèle
            unite = (lambda x: x) if nom != "D2" else (lambda x: slot_de.get(x, x))
            communs = [x for x in cand if x not in petits]
            res, reg, rej = [], [], []
            for x in communs:
                lab = mdl.labels.get(unite(x))
                if lab is None or lab not in tr["slots"]:
                    continue
                serie = pd.Series(tr["slots"][lab]["mean"], index=pd.to_datetime(tr["dates"]))
                serie = serie[~serie.index.duplicated(keep="last")]
                # sondages exacts, renormalisés sur les unités communes
                e = exact[exact.candidat.isin(communs)]
                tot = e.groupby("scenario_id").intention.transform("sum")
                ex = e.assign(p=e.intention / tot)[e.candidat.eq(x)]
                base_tr = sum(pd.Series(tr["slots"][mdl.labels[unite(y)]]["mean"],
                                        index=pd.to_datetime(tr["dates"])).groupby(level=0).last()
                              for y in communs if mdl.labels.get(unite(y)) in tr["slots"])
                m = (serie / base_tr).reindex(pd.to_datetime(ex.date_fin), method="nearest")
                res.extend(np.abs(ex.p.to_numpy() - m.to_numpy()) * 100)
                d90 = serie[serie.index >= serie.index.max() - pd.Timedelta(days=90)]
                reg.append(float(np.abs(np.diff(d90.to_numpy())).mean() * 100) if len(d90) > 1 else np.nan)
                a1 = float(np.asarray(nc.draws.sel(candidat=lab)).mean())
                a2 = float(np.asarray(nc2.draws.sel(candidat=lab)).mean())
                rej.append(abs(a1 - a2) * 100)
            out.append({"liste": L["id"], "definition": nom, "n_sondages": int(used["notice"].nunique()),
                        "residu_moyen_pt": round(float(np.mean(res)), 3) if res else None,
                        "variation_jour_pt": round(float(np.nanmean(reg)), 4) if reg else None,
                        "rejeu_sans_dernier_pt": round(float(np.mean(rej)), 3) if rej else None})
            print(out[-1], flush=True)
    return {"as_of": as_of, "lignes": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--election", type=int, choices=[2017, 2022], default=2022)
    ap.add_argument("--diagnostic-2027", action="store_true")
    a = ap.parse_args()
    if a.diagnostic_2027:
        r = diagnostic_2027()
        out = sr.OUT_DIR / "listes_definition_diag2027.json"
        out.write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n")
        print(f"-> {out}")
        return
    lignes = [l for h in HS for l in evaluer(a.election, h)]
    out = sr.OUT_DIR / f"listes_definition_{a.election}.json"
    out.write_text(json.dumps(lignes, ensure_ascii=False, indent=1) + "\n")
    print(f"-> {out} ({len(lignes)} lignes)")


if __name__ == "__main__":
    main()
