"""(A)(1) Reports T1→T2 prédits par le modèle contre tableaux des instituts
(spec_second_tour §8.38). 0 fit : les V2 T2−7 committés sont reconstruits par
fusion des chaînes sauvées.

    python -m model.backtest.reports_t2            # 2017 et 2022, abst et hier
"""
from __future__ import annotations

import argparse
import json
import unicodedata

import numpy as np
import pandas as pd
from scipy.special import expit, logit, logsumexp

import model.backtest.second_round as sr
from model.models.spatial_pooling.fit import duel_draws
from model.models.spatial_pooling.geometry import V, W, log_attractivite

OUT = sr.OUT_DIR / "reports_t2_A1.json"


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def reports_predits(A, champ, ia, ib, kappa, omega, lse, s_ab, n_iter=60, plancher=0.0, dnu=0.0, poids=None):
    """Parts (en fraction de l'électorat T1) par tirage, spec §8.38.

    A : (S, N, B) log-attractivités ; champ : indices du T1 réel ; ia, ib :
    finalistes ; kappa, omega, s_ab : (S,) ; lse : (S, B) ou None (sans
    abstention : rho = 1). Retour : dict de (S, K) pour K = len(champ),
    clés 'a', 'b', 'aucun', plus 'delta' (S,) et 'E' (S, K)."""
    Wq = np.asarray(W) if poids is None else np.asarray(poids)   # poids de quadrature (2D : §8.41)
    if poids is not None and dnu:
        raise ValueError("dnu (fer à cheval 1D) incompatible avec une quadrature 2D")
    Ac = A[:, champ, :]
    p = np.exp(Ac - logsumexp(Ac, axis=1, keepdims=True))                 # (S,K,B)
    k = kappa[:, None]
    q = expit(k * (A[:, ia, :] - A[:, ib, :]) + (dnu * np.abs(2 * np.asarray(V) - 1) if dnu else 0.0))   # (S,B) ; fer à cheval §8.40
    q = (1 - plancher) * q + plancher / 2          # plancher de report (§8.39) : ε des votants tirent au hasard
    if lse is None:
        rho = np.ones_like(q)
    else:
        ins = np.logaddexp(k * A[:, ia, :], k * A[:, ib, :])
        rho = expit(ins - (omega[:, None] + k * lse))
    E = np.einsum("skb,b->sk", p, Wq)
    pr = np.einsum("skb,sb,b->sk", p, rho, Wq)
    R = pr / E
    G = np.clip(np.einsum("skb,sb,sb,b->sk", p, rho, q, Wq) / pr, 1e-9, 1 - 1e-9)
    return melange(E, R, G, s_ab, n_iter=n_iter)


def _bissection(f, cible, n_iter=60):
    lo, hi = np.full(len(cible), -20.0), np.full(len(cible), 20.0)
    for _ in range(n_iter):                                               # f croissante en d
        d = (lo + hi) / 2
        m = f(d)
        hi = np.where(m > cible, d, hi)
        lo = np.where(m > cible, lo, d)
    return (lo + hi) / 2


def melange(E, R, G, s_ab, pi_as=None, pi_aucun=None, n_iter=60):
    """Cellules par électorat (§8.38) avec mélange orienté anti-système (§8.43).

    pi_as, pi_aucun : (K,) ou None. Une fraction pi_as[k] de l'électorat k vote
    pour b (finaliste anti-système), pi_aucun[k] ne s'exprime pas, le reste suit
    le modèle spatial. δ est recalé pour que le duel agrégé reste s_ab (lecture
    (i)) ; 'decalage_duel' donne le déplacement du duel si δ garde sa valeur
    sans mélange (lecture (ii))."""
    K = E.shape[1]
    pa = np.zeros(K) if pi_as is None else np.asarray(pi_as, float)
    pn = np.zeros(K) if pi_aucun is None else np.asarray(pi_aucun, float)
    sp = 1.0 - pa - pn                                                    # part spatiale
    cible = np.clip(s_ab, 1e-9, 1 - 1e-9)
    lG = logit(G)

    def agg(d):
        Gt = expit(lG + d[:, None])
        num = (E * sp * R * Gt).sum(1)
        return num / (E * (sp * R + pa)).sum(1)

    d0 = _bissection(lambda d: (E * R * expit(lG + d[:, None])).sum(1) / (E * R).sum(1), cible, n_iter)
    d = d0 if (pi_as is None and pi_aucun is None) else _bissection(agg, cible, n_iter)
    Gt = expit(lG + d[:, None])
    out = {"a": sp * R * Gt, "b": sp * R * (1 - Gt) + pa, "aucun": sp * (1 - R) + pn, "delta": d, "E": E}
    out["decalage_duel"] = agg(d0) - cible
    return out


def composantes(A, champ, ia, ib, kappa, omega, lse):
    """(E, R, G) de §8.38 sans recalage, pour réutiliser le mélange (§8.43)."""
    Ac = A[:, champ, :]
    p = np.exp(Ac - logsumexp(Ac, axis=1, keepdims=True))
    k = kappa[:, None]
    q = expit(k * (A[:, ia, :] - A[:, ib, :]))
    if lse is None:
        rho = np.ones_like(q)
    else:
        rho = expit(np.logaddexp(k * A[:, ia, :], k * A[:, ib, :]) - (omega[:, None] + k * lse))
    Wq = np.asarray(W)
    E = np.einsum("skb,b->sk", p, Wq)
    pr = np.einsum("skb,sb,b->sk", p, rho, Wq)
    return E, pr / E, np.clip(np.einsum("skb,sb,sb,b->sk", p, rho, q, Wq) / pr, 1e-9, 1 - 1e-9)


def _fit_v2(election: int, variante: str):
    sr.ABSTENTION = True
    sr.VARIANTE_KW.clear()
    sr.VARIANTE_KW.update({"hier": {"eta_hier": True}}.get(variante, {}))
    stem = f"second_round_{variante}_v2_{election}_T2-7"
    paths = [sr.CHAINES_DIR / f"{stem}_c{k}.npz" for k in range(sr.N_CHAINES)]
    as_of = sr.T2_DATES[election] - pd.Timedelta(days=7)
    raw = sr.polls_historiques(election, as_of)
    nuts = dict(draws=400, tune=600, chains=4, mcmc_runner=sr._runner_fusion(paths))
    fit, _ = sr._fit(raw, as_of, election, nuts)
    return fit


def mesures(rep: pd.DataFrame) -> pd.DataFrame:
    """Moyenne et écart-type entre notices par (électorat, candidat_t2), en %,
    plus le partage parmi exprimés M/(M+LP) par notice."""
    rep = rep.copy()
    piv = rep.pivot_table(index=["notice", "electorat_t1"], columns="candidat_t2",
                          values="part", aggfunc="mean").reset_index()
    piv["partage"] = 100 * piv["Macron"] / (piv["Macron"] + piv["Le Pen"])
    long = piv.melt(id_vars=["notice", "electorat_t1"], value_vars=["Macron", "Le Pen", "Aucun", "partage"],
                    var_name="cellule", value_name="y").dropna()
    return long.groupby(["electorat_t1", "cellule"]).y.agg(["mean", "std", "count"]).reset_index()


def analyse(election: int, variante: str, plancher: float = 0.0, fit=None, dnu: float = 0.0) -> list[dict]:
    fit = fit or _fit_v2(election, variante)
    a, b = sr.FINALE[election]
    idx = {_norm(c): i for i, c in enumerate(fit.candidates)}
    res = sr.load_results()
    t1 = res[(res.election == election) & (res.tour == "Premier tour")].candidat
    champ = [idx[_norm(c)] for c in t1 if _norm(c) in idx]
    absents = [c for c in t1 if _norm(c) not in idx]
    A = np.asarray(log_attractivite(fit.mu_draws, fit.sigma_draws, fit.w_draws))
    Am = A if fit.masque_lse is None else A + np.log(fit.masque_lse)[None, :, None]
    lse = logsumexp(Am, axis=1) if fit.omega_draws is not None else None
    s_ab = duel_draws(fit, a, b, 0.0, scrutin=False)
    r = reports_predits(A, np.array(champ), idx[_norm(a)], idx[_norm(b)], fit.kappa_draws,
                        fit.omega_draws if fit.omega_draws is not None else np.zeros(len(s_ab)), lse, s_ab,
                        plancher=plancher, dnu=dnu)
    rep = pd.read_csv(sr.PARSED_DIR / f"reports_t2_{election}.csv")
    m = mesures(rep)
    out = []
    for _, row in m.iterrows():
        nk = _norm(row.electorat_t1)
        if nk not in idx or idx[nk] not in champ:
            continue
        j = champ.index(idx[nk])
        tir = {"Macron": r["a"][:, j], "Le Pen": r["b"][:, j], "Aucun": r["aucun"][:, j],
               "partage": r["a"][:, j] / (r["a"][:, j] + r["b"][:, j])}[row.cellule] * 100
        e = float(tir.mean() - row["mean"])
        sd = float(np.sqrt(tir.var() + row["std"] ** 2))
        out.append({"election": election, "variante": variante, "plancher": plancher, "dnu": dnu,
                    "electorat": row.electorat_t1,
                    "cellule": row.cellule, "mesure": round(row["mean"], 1), "sd_notices": round(row["std"], 1),
                    "n_notices": int(row["count"]), "modele": round(float(tir.mean()), 1),
                    "modele_ic90": [round(float(x), 1) for x in np.percentile(tir, [5, 95])],
                    "erreur": round(e, 1), "z": round(e / sd, 2)})
    meta = {"election": election, "variante": variante, "absents_du_fit": absents,
            "delta_moyen": round(float(r["delta"].mean()), 3),
            "s_ab_modele": round(float(s_ab.mean()), 4),
            "part_t1_modele": {fit.candidates[c]: round(float(r["E"][:, i].mean()), 4)
                               for i, c in enumerate(champ)}}
    return out, meta


def critere_D(lignes: list[dict], variante: str) -> dict:
    """§8.38 : Mélenchon, partage parmi exprimés, |e| > 10 et |z| > 2 les deux
    années, sens « report Le Pen sous-prédit » (partage Macron surprédit, e > 0)."""
    sel = [l for l in lignes if l["variante"] == variante and l["cellule"] == "partage"
           and _norm(l["electorat"]) == _norm("Jean-Luc Mélenchon")]
    ok = {l["election"]: (l["erreur"] > 10 and abs(l["z"]) > 2) for l in sel}
    return {"variante": variante, "par_annee": ok, "ouvrir_D": len(ok) == 2 and all(ok.values())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variantes", nargs="+", default=["abst", "hier"])
    ap.add_argument("--planchers", nargs="+", type=float, default=None,
                    help="§8.39 : grille de ε (sortie reports_t2_plancher.json)")
    ap.add_argument("--nu-grille", nargs="+", type=float, default=None,
                    help="§8.40 : profil de Δν = ν_Macron − ν_Le Pen (sortie reports_t2_nu.json)")
    args = ap.parse_args()
    if args.nu_grille:
        lignes = []
        for v in args.variantes:
            for e in (2017, 2022):
                fit = _fit_v2(e, v)
                for eps in (args.planchers or [0.0]):
                    for dn in args.nu_grille:
                        lignes += analyse(e, v, eps, fit, dn)[0]
        t = pd.DataFrame(lignes)
        t = t[t.cellule == "partage"].assign(abs_err=lambda d: d.erreur.abs(),
                                            mel=lambda d: d.electorat == "Jean-Luc Mélenchon")
        prof = (t.groupby(["variante", "election", "plancher", "dnu", "mel"]).abs_err.mean()
                .unstack("mel").rename(columns={True: "err_melenchon", False: "mae_autres"}).round(1).reset_index())
        out = OUT.with_name("reports_t2_nu.json")
        out.write_text(json.dumps({"spec": "spec_second_tour §8.40", "cellules": lignes,
                                   "profil": prof.to_dict("records")}, ensure_ascii=False, indent=1) + "\n")
        print(prof.to_string(index=False)); print(f"-> {out}")
        return
    if args.planchers:
        lignes = []
        for v in args.variantes:
            for e in (2017, 2022):
                fit = _fit_v2(e, v)
                for eps in args.planchers:
                    lignes += analyse(e, v, eps, fit)[0]
        t = pd.DataFrame(lignes).assign(abs_err=lambda d: d.erreur.abs())
        mae = (t[t.cellule != "Aucun"].groupby(["variante", "election", "plancher"]).abs_err.mean()
               .round(1).reset_index().rename(columns={"abs_err": "mae_M_LP_partage"}))
        mel = t[(t.cellule == "partage") & (t.electorat == "Jean-Luc Mélenchon")][
            ["variante", "election", "plancher", "mesure", "modele", "erreur", "z"]]
        out = OUT.with_name("reports_t2_plancher.json")
        out.write_text(json.dumps({"spec": "spec_second_tour §8.39", "cellules": lignes,
                                   "mae": mae.to_dict("records")}, ensure_ascii=False, indent=1) + "\n")
        print(mae.to_string(index=False)); print(mel.to_string(index=False)); print(f"-> {out}")
        return
    lignes, metas = [], []
    for v in args.variantes:
        for e in (2017, 2022):
            l, mt = analyse(e, v)
            lignes += l
            metas.append(mt)
    t = pd.DataFrame(lignes)
    resume = (t.assign(abs_err=t.erreur.abs())
              .groupby(["variante", "election", "cellule"]).abs_err.mean().round(1).reset_index()
              .rename(columns={"abs_err": "mae"}).to_dict("records"))
    crit = [critere_D(lignes, v) for v in args.variantes]
    OUT.write_text(json.dumps({"spec": "spec_second_tour §8.38", "cellules": lignes, "meta": metas,
                               "mae": resume, "critere_D": crit}, ensure_ascii=False, indent=1) + "\n")
    print(t.drop(columns=["modele_ic90"]).to_string(index=False))
    print(pd.DataFrame(resume).to_string(index=False))
    print(json.dumps(metas, ensure_ascii=False, indent=1))
    print(crit)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
