"""τ_T2 a posteriori sur 2002-2022 (spec_second_tour §8.42). 0 fit.

    python -m model.backtest.tau_t2_historique
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import halfnorm, norm, t as student

RACINE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results" / "tau_t2_historique.json"
T1 = {2002: "2002-04-21", 2007: "2007-04-22", 2012: "2012-04-22", 2017: "2017-04-23", 2022: "2022-04-10"}
T2 = {2002: "2002-05-05", 2007: "2007-05-06", 2012: "2012-05-06", 2017: "2017-05-07", 2022: "2022-04-24"}
GRILLE = np.arange(0.0, 1.5001, 0.001)
NU_T = 4


def _nom(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()
    return s.split()[-1] if s else s


def finale(election: int) -> tuple[str, str, float]:
    r = pd.read_csv(RACINE / "data/historical" / f"results_{election}.csv")
    r = r[r.tour.str.startswith("Deux")].sort_values("resultat", ascending=False)
    a, b = r.candidat.tolist()[:2]
    ya, yb = r.resultat.tolist()[:2]
    return a, b, ya / (ya + yb)


def sondages_finaux(election: int) -> tuple[pd.DataFrame, str]:
    """Une ligne par notice : part du vainqueur parmi les deux finalistes."""
    a, b, _ = finale(election)
    d = pd.read_csv(RACINE / "data/parsed" / f"intentions_{election}_wiki.csv")
    d = d[d.tour.eq("Deuxième tour")].copy()
    d["date_fin"] = pd.to_datetime(d.date_fin, errors="coerce")
    d["nom"] = d.candidat.map(_nom)
    cle = d.notice_id.where(d.notice_id.notna(), d.notice).astype(str) + "|" + d.hypothese.astype(str)
    d["cle"] = cle
    na, nb = _nom(a), _nom(b)
    rows = []
    for k, g in d.groupby("cle"):
        noms = set(g.nom)
        if noms != {na, nb}:
            continue
        ia = float(g.loc[g.nom == na, "intention"].iloc[0])
        ib = float(g.loc[g.nom == nb, "intention"].iloc[0])
        rows.append({"cle": k, "institut": g.institut.iloc[0], "date_fin": g.date_fin.iloc[0],
                     "echantillon": g.echantillon.iloc[0], "p": ia / (ia + ib)})
    s = pd.DataFrame(rows)
    t1, t2 = pd.Timestamp(T1[election]), pd.Timestamp(T2[election])
    fen = s[(s.date_fin >= t2 - pd.Timedelta(days=7)) & (s.date_fin <= t2 - pd.Timedelta(days=1))]
    if len(fen):
        return fen, "T2-7..T2-1"
    return s[(s.date_fin > t1) & (s.date_fin < t2)], "repli entre-deux-tours"


def observation(election: int) -> dict:
    _, _, y = finale(election)
    s, fenetre = sondages_finaux(election)
    p = s.p.to_numpy()
    k = len(p)
    pbar = float(p.mean())
    n = pd.to_numeric(s.echantillon, errors="coerce").fillna(1000.0).mean()
    v_bin = 1.0 / (k * n * pbar * (1 - pbar))
    v_emp = float(np.var(logit(p), ddof=1)) / k if k > 1 else 0.0
    return {"election": election, "fenetre": fenetre, "k": k, "p_bar": round(pbar, 4), "y": round(y, 4),
            "d_logit": float(logit(y) - logit(pbar)), "erreur_pt": round(100 * (y - pbar), 2),
            "v": max(v_bin, v_emp)}


def posterieur(obs: list[dict], loi: str) -> np.ndarray:
    lp = halfnorm.logpdf(GRILLE, scale=0.5)
    for o in obs:
        sd = np.sqrt(GRILLE ** 2 + o["v"])
        lp = lp + (norm.logpdf(o["d_logit"], 0, sd) if loi == "normale"
                   else student.logpdf(o["d_logit"], NU_T, 0, sd))
    w = np.exp(lp - lp.max())
    return w / w.sum()


def resume(w: np.ndarray) -> dict:
    c = np.cumsum(w)
    q = lambda x: float(GRILLE[np.searchsorted(c, x)])  # noqa: E731
    return {"mediane": round(q(0.5), 3), "moyenne": round(float((w * GRILLE).sum()), 3),
            "ic90": [round(q(0.05), 3), round(q(0.95), 3)]}


def loo(obs: list[dict], loi: str) -> list[dict]:
    out = []
    for i, o in enumerate(obs):
        w = posterieur(obs[:i] + obs[i + 1:], loi)
        sd = np.sqrt(GRILLE ** 2 + o["v"])
        dens = norm.pdf(o["d_logit"], 0, sd) if loi == "normale" else student.pdf(o["d_logit"], NU_T, 0, sd)
        # IC90 prédictif par mélange sur la grille
        xs = np.linspace(-1.5, 1.5, 3001)
        cdf = np.array([(w * (norm.cdf(x, 0, sd) if loi == "normale" else student.cdf(x, NU_T, 0, sd))).sum()
                        for x in xs])
        lo, hi = xs[np.searchsorted(cdf, 0.05)], xs[np.searchsorted(cdf, 0.95)]
        out.append({"election": o["election"], "lpd": round(float(np.log((w * dens).sum())), 3),
                    "ic90_logit": [round(float(lo), 3), round(float(hi), 3)],
                    "couvre": bool(lo <= o["d_logit"] <= hi)})
    return out


def main() -> None:
    obs = [observation(e) for e in T1]
    res = {"spec": "spec_second_tour §8.42", "observations": obs, "lois": {}}
    sous = {"2002-2022": obs, "sans_2002": [o for o in obs if o["election"] != 2002],
            "2017-2022": [o for o in obs if o["election"] >= 2017]}
    for loi in ("normale", "student4"):
        res["lois"][loi] = {nom: resume(posterieur(o, loi)) for nom, o in sous.items()}
        l = loo(obs, loi)
        res["lois"][loi]["loo"] = l
        res["lois"][loi]["lpd_loo_total"] = round(sum(x["lpd"] for x in l), 3)
        res["lois"][loi]["couverture_loo"] = f"{sum(x['couvre'] for x in l)}/{len(l)}"
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n")
    print(pd.DataFrame(obs).to_string(index=False))
    print(json.dumps(res["lois"], ensure_ascii=False, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
