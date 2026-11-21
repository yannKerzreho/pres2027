"""Lecture par MODE d'un run à n chaînes (spec_second_tour §8.50, §8.51 ; revue locale du 20/11).

    PRES2027_B_QUAD=100 python -m model.backtest.modes --variante hier_ra --stem <souche> \\
        [--election 2022] [--coupure T1-30] [--n 8] [--lse-officiel]

Pour chaque chaîne (sans NUTS, reconstruction par `eval_r2.reconstruire`) : L̄, partages mesurés,
case « Aucun », σ_Mélenchon, σ_Le Pen/σ_Macron, ν (noyau t), MAE du T1, et monotonie à gauche
(q_Macron(v) ≥ 0,5 pour tout v < μ_Macron, duel Macron–Le Pen, part des tirages).

Mode dominant (règle figée §8.50, précisée le 20/11) : chaînes avec L̄ ≥ L̄_max − 10 nats ET
|σ_Mélenchon − σ_Mélenchon(meilleure chaîne)| ≤ 0,02. Les autres modes sont regroupés par
σ_Mélenchon (écart > 0,02 entre groupes) et rapportés. Les critères sont lus sur la fusion des
chaînes du mode dominant.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations

import jax.numpy as jnp
import numpy as np

import model.backtest.eval_r2 as E
import model.backtest.second_round as sr
from model.models.spatial_pooling.fit import duel_draws

JLM, MLP, EM = "Jean-Luc Mélenchon", "Marine Le Pen", "Emmanuel Macron"


def _sig(fit, nom):
    return float(fit.sigma_draws[:, fit.candidates.index(nom)].mean())


def _q_macron_lp(fit):
    A, _ = E._A(fit, fit.w_draws)                                   # (S, N, G)
    im, il = fit.candidates.index(EM), fit.candidates.index(MLP)
    k = np.asarray(fit.kappa_draws)[:, None]
    return 1 / (1 + np.exp(-k * (A[:, im] - A[:, il])))             # (S, G)


def monotonie_ponderee(fit) -> float | None:
    """§8.54 : part (pondérée par la densité uniforme des votants) de l'arc/segment « gauche »
    (entre Mélenchon et Macron sur le cercle ; v < μ_Macron sur le segment) où q_Macron < 0,5,
    moyenne a posteriori. Critère cercle : ≤ 0,10 ; rapportée seulement pour le segment."""
    q = _q_macron_lp(fit)
    im, ij = fit.candidates.index(EM), fit.candidates.index(JLM)
    if fit.arc_draws is not None:
        from model.models.spatial_pooling.geometry import THETA
        th = np.asarray(THETA)[None, :]
        t = fit.arc_draws[:, None] * (fit.mu_draws - 0.5)           # (S, N)
        lo, hi = np.minimum(t[:, ij], t[:, im])[:, None], np.maximum(t[:, ij], t[:, im])[:, None]
        zone = (th >= lo) & (th <= hi)
        w = np.ones(th.shape[1])                                     # trapèzes : poids égaux
    elif fit.nu_draws is None:
        from model.models.spatial_pooling.geometry import V, W
        zone = np.asarray(V)[None, :] < fit.mu_draws[:, im][:, None]
        w = np.asarray(W)                                           # Gauss-Legendre : poids de masse
    else:
        return None
    mal = ((q < 0.5) & zone) * w[None, :]
    return float((mal.sum(1) / np.clip((zone * w[None, :]).sum(1), 1e-12, None)).mean())


def structure(fit) -> dict:
    """§8.54 : électorats Le Pen et Zemmour dans Mélenchon–Macron (fer à cheval mesuré)."""
    from scipy.special import logsumexp
    from model.models.spatial_pooling.geometry import reports_cellules
    c = fit.candidates
    if JLM not in c or EM not in c:
        return {}
    A, poids = E._A(fit, fit.w_draws)
    S = A.shape[0]
    lse = logsumexp(A if fit.masque_lse is None else A + np.log(fit.masque_lse)[None, :, None], axis=1)
    ia, ib = c.index(JLM), c.index(EM)
    res = E.sr.load_results()
    t1 = res[(res.election == E.ELECTION) & (res.tour == "Premier tour")]
    idx = {E._norm(x): i for i, x in enumerate(c)}
    mask = np.zeros(len(c)); mask[[idx[E._norm(x)] for x in t1.candidat if E._norm(x) in idx]] = 1.0
    s_ab = duel_draws(fit, JLM, EM, 0.0, scrutin=False)
    out = {}
    for nom, cle in ((MLP, "electorat_LP_vers_JLM"), ("Eric Zemmour", "electorat_Zemmour_vers_JLM")):
        if nom not in c:
            continue
        th, _ = reports_cellules(jnp.asarray(A), jnp.asarray(mask), jnp.full(S, ia), jnp.full(S, ib),
                                 jnp.full(S, c.index(nom)), jnp.asarray(fit.kappa_draws),
                                 jnp.asarray(fit.eta_draws[:, ia] - fit.eta_draws[:, ib]), jnp.asarray(lse),
                                 jnp.asarray(fit.omega_draws), jnp.asarray(s_ab), poids, n_newton=8,
                                 omega_g=E._omega_g(fit),
                                 omega_k=(None if getattr(fit, "omega_k_draws", None) is None
                                          else jnp.asarray(fit.omega_k_draws)),
                                 lam=(None if getattr(fit, "lam_mix_draws", None) is None
                                      else jnp.asarray(fit.lam_mix_draws)))
        th = np.asarray(th)
        out[cle] = round(float(100 * (th[:, 0] / (th[:, 0] + th[:, 1])).mean()), 1)
    out["sigma_melenchon"] = round(_sig(fit, JLM), 3)
    out["ok_sigma_jlm"] = out["sigma_melenchon"] <= 0.15
    if "electorat_LP_vers_JLM" in out:
        out["ok_LP"] = out["electorat_LP_vers_JLM"] >= 45.0
    if "electorat_Zemmour_vers_JLM" in out:
        out["ok_Zemmour"] = out["electorat_Zemmour_vers_JLM"] <= 40.0
    return out


def monotonie_gauche(fit) -> float:
    """Part des tirages où q_Macron(v) ≥ 0,5 pour tout nœud v < μ_Macron (duel Macron–Le Pen).
    Segment seulement (§8.54) ; sur le cercle, voir `monotonie_ponderee`."""
    if fit.arc_draws is not None:
        return float("nan")
    A, _ = E._A(fit, fit.w_draws)                                   # (S, N, G)
    im, il = fit.candidates.index(EM), fit.candidates.index(MLP)
    from model.models.spatial_pooling.geometry import V
    v = np.asarray(V)
    k = np.asarray(fit.kappa_draws)[:, None]
    q = 1 / (1 + np.exp(-k * (A[:, im] - A[:, il])))               # (S, G)
    gauche = v[None, :] < fit.mu_draws[:, im][:, None]
    ok = np.where(gauche, q >= 0.5, True).all(axis=1)
    return float(ok.mean())


def par_chaine(variante, stem, coupure, n) -> list[dict]:
    out = []
    for k in range(n):
        fit, raw, as_of = E.reconstruire(variante, stem, coupure, chaines=[k])
        ev = E.evaluer(fit, raw, as_of)
        part = {r["electorat"]: (r["modele"], r["mesure"]) for r in ev["reports"] if r["cellule"] == "partage"}
        aucun = {r["electorat"]: (r["modele"], r["mesure"]) for r in ev["reports"] if r["cellule"] == "Aucun"}
        out.append({"chaine": k, "lp": round(float(sr.LP_CHAINES[0]), 1),
                    "partages": part, "aucun": aucun,
                    "sigma_melenchon": round(_sig(fit, JLM), 4),
                    "sigma_lp_sur_macron": round(_sig(fit, MLP) / _sig(fit, EM), 3),
                    "nu": (round(float(np.median(fit.nu_t_draws)), 2) if getattr(fit, "nu_t_draws", None) is not None
                           else None),
                    "log_nu_sd": (round(float(np.log(fit.nu_t_draws).std()), 3)
                                  if getattr(fit, "nu_t_draws", None) is not None else None),
                    "nu_q95": (round(float(np.percentile(fit.nu_t_draws, 95)), 1)
                               if getattr(fit, "nu_t_draws", None) is not None else None),
                    "t1_mae": ev["t1_resultat"]["mae_pt"], "t1_couv": ev["t1_resultat"]["couverture_ic90"],
                    "monotonie_gauche": round(monotonie_gauche(fit), 3)})
        print(json.dumps({x: out[-1][x] for x in ("chaine", "lp", "sigma_melenchon", "sigma_lp_sur_macron", "nu")},
                         ensure_ascii=False), flush=True)
    return out


def partition(lignes: list[dict]) -> dict:
    best = max(lignes, key=lambda l: l["lp"])
    dom = [l["chaine"] for l in lignes if l["lp"] >= best["lp"] - 10.0
           and abs(l["sigma_melenchon"] - best["sigma_melenchon"]) <= 0.02]
    reste = sorted((l for l in lignes if l["chaine"] not in dom), key=lambda l: l["sigma_melenchon"])
    groupes, cur = [], []
    for l in reste:
        if cur and l["sigma_melenchon"] - cur[-1]["sigma_melenchon"] > 0.02:
            groupes.append(cur); cur = []
        cur.append(l)
    if cur:
        groupes.append(cur)
    return {"dominant": dom, "autres": [[l["chaine"] for l in g] for g in groupes]}


def _reconstruire_avec_kwargs(variante, stem, coupure, chaines):
    """`eval_r2.reconstruire`, en capturant les arguments du modèle (lignes de reports)."""
    cap, orig = {}, sr._runner_fusion

    def enveloppe(paths):
        r = orig(paths)

        def run(model_fn, model_kwargs, **kw):
            cap["kw"] = model_kwargs
            return r(model_fn, model_kwargs, **kw)
        return run
    sr._runner_fusion = enveloppe
    try:
        fit, raw, as_of = E.reconstruire(variante, stem, coupure, chaines=chaines)
    finally:
        sr._runner_fusion = orig
    return fit, raw, as_of, cap.get("kw", {})


def lecture_reports_fit(fit, kw, stem, dom) -> dict:
    """§8.56 : c médian et résidu moyen de la case « Aucun » sur les lignes DU FIT,
    séparément pour les lignes propres (k ∈ {a, b}) et les autres."""
    out = {}
    cs = [np.asarray(np.load(sr.CHAINES_DIR / f"{stem}_c{k}.npz")["s__c_reports"]).ravel()
          for k in dom if "s__c_reports" in np.load(sr.CHAINES_DIR / f"{stem}_c{k}.npz").files]
    if cs:
        out["c_median"] = round(float(np.median(np.concatenate(cs))), 1)
    if fit.reports_theta_draws is not None and "rep_y" in kw:
        th = np.asarray(fit.reports_theta_draws).mean(0)            # (R, 3)
        y = np.asarray(kw["rep_y"])
        k, a, b = (np.asarray(kw[x]) for x in ("rep_k", "rep_a", "rep_b"))
        propre = (k == a) | (k == b)
        r = 100 * (th[:, 2] - y[:, 2])
        out["residu_aucun_propres"] = round(float(r[propre].mean()), 2)
        out["residu_aucun_autres"] = round(float(r[~propre].mean()), 2)
        out["n_lignes_propres"] = int(propre.sum())
    return out


def v1(fit, election=2022, as_of="2022-03-11", hyp=None) -> dict:
    """§8.59 : cellules de Mélenchon–Macron prédites contre la moyenne des notices de mars (30 j)."""
    import os
    hyp = hyp or os.environ.get("PRES2027_SANS_HYP") or "Mélenchon – Macron"     # §8.63 : V1-LOHO
    import pandas as pd
    from scipy.special import logsumexp
    from model.models.spatial_pooling.geometry import reports_cellules
    r = pd.read_csv(sr.PARSED_DIR / f"reports_avant_t1_{election}.csv")
    r = r[(r.hypothese.str.replace("—", "–") == hyp)
          & (pd.to_datetime(r.date_fin) <= pd.Timestamp(as_of))
          & (pd.to_datetime(r.date_fin) >= pd.Timestamp(as_of) - pd.Timedelta(days=30))]
    c = fit.candidates
    idx = {E._norm(x): i for i, x in enumerate(c)}
    sa, sb = [x.strip() for x in hyp.split("–")]
    fa = next(x for x in c if E._norm(x).endswith(E._norm(sa)))
    fb = next(x for x in c if E._norm(x).endswith(E._norm(sb)))
    ia, ib = c.index(fa), c.index(fb)
    A, poids = E._A(fit, fit.w_draws)
    S = A.shape[0]
    lse = logsumexp(A if fit.masque_lse is None else A + np.log(fit.masque_lse)[None, :, None], axis=1)
    res = E.sr.load_results()
    t1 = res[(res.election == election) & (res.tour == "Premier tour")]
    mask = np.zeros(len(c)); mask[[idx[E._norm(x)] for x in t1.candidat if E._norm(x) in idx]] = 1.0
    s_ab = duel_draws(fit, fa, fb, 0.0, scrutin=False)
    out = {}
    for e, g in r.groupby("electorat_t1"):
        if E._norm(e) not in idx or e in (fa, fb):
            continue
        p = g.pivot_table(index="notice", columns="candidat_t2", values="part")
        if not {sa, sb} <= set(p.columns):
            continue
        obs_part = float((100 * p[sa] / (p[sa] + p[sb])).mean())   # part de a = 1er nom de l'hypothèse
        obs_aucun = float(p["Aucun"].mean()) if "Aucun" in p.columns else None
        th, _ = reports_cellules(jnp.asarray(A), jnp.asarray(mask), jnp.full(S, ia), jnp.full(S, ib),
                                 jnp.full(S, idx[E._norm(e)]), jnp.asarray(fit.kappa_draws),
                                 jnp.asarray(fit.eta_draws[:, ia] - fit.eta_draws[:, ib]), jnp.asarray(lse),
                                 jnp.asarray(fit.omega_draws), jnp.asarray(s_ab), poids, n_newton=8,
                                 omega_g=E._omega_g(fit),
                                 omega_k=(None if getattr(fit, "omega_k_draws", None) is None
                                          else jnp.asarray(fit.omega_k_draws)),
                                 lam=(None if getattr(fit, "lam_mix_draws", None) is None
                                      else jnp.asarray(fit.lam_mix_draws)))
        th = np.asarray(th)
        thu, _ = reports_cellules(jnp.asarray(A), jnp.asarray(mask), jnp.full(S, ia), jnp.full(S, ib),
                                  jnp.full(S, idx[E._norm(e)]), jnp.asarray(fit.kappa_draws),
                                  jnp.asarray(fit.eta_draws[:, ia] - fit.eta_draws[:, ib]), jnp.asarray(lse),
                                  jnp.asarray(fit.omega_draws), jnp.asarray(s_ab), poids, n_newton=8,
                                  omega_g=E._omega_g(fit),
                                  omega_k=(None if getattr(fit, "omega_k_draws", None) is None
                                           else jnp.asarray(fit.omega_k_draws)), tau=0.0)   # §8.63 témoin U
        thu = np.asarray(thu)
        out[e] = {"partage_U": round(float((100 * thu[:, 0] / (thu[:, 0] + thu[:, 1])).mean()), 1),
                  "partage_modele": round(float((100 * th[:, 0] / (th[:, 0] + th[:, 1])).mean()), 1),
                  "partage_mesure": round(obs_part, 1),
                  "aucun_modele": round(float(100 * th[:, 2].mean()), 1),
                  "aucun_mesure": None if obs_aucun is None else round(obs_aucun, 1)}
    if out:
        mae = float(np.mean([abs(v["partage_modele"] - v["partage_mesure"]) for v in out.values()]))
        lpd = out.get(MLP, {})
        zmd = out.get("Eric Zemmour", out.get("Éric Zemmour", {}))
        ecart = lambda d: (None if not d else round(d["partage_modele"] - d["partage_mesure"], 1))  # noqa: E731
        # §8.61 : témoin trivial 50/50 et critère révisé (≤ 12 ET ≤ 50/50 − 1,5)
        mae_5050 = float(np.mean([abs(50.0 - v["partage_mesure"]) for v in out.values()]))
        mae_U = float(np.mean([abs(v["partage_U"] - v["partage_mesure"]) for v in out.values()]))
        out["_criteres"] = {"mae_partages": round(mae, 2), "mae_5050": round(mae_5050, 2), "mae_U": round(mae_U, 2), "loho_ok_U": mae <= mae_U - 2.0,
                            "temoin_structurel": 25.9,
                            "mae_ok": mae <= 12.0 and mae <= mae_5050 - 1.5,
                            "LP_ecart": ecart(lpd), "LP_ok": lpd != {} and abs(ecart(lpd)) <= 10,
                            "Zemmour_ecart": ecart(zmd), "Zemmour_ok": zmd != {} and abs(ecart(zmd)) <= 10}
    return out


def criteres(variante, stem, coupure, lignes, dom, election) -> dict:
    fit, raw, as_of, kw = _reconstruire_avec_kwargs(variante, stem, coupure, dom)
    ev = E.evaluer(fit, raw, as_of)
    part = {r["electorat"]: (r["modele"], r["mesure"]) for r in ev["reports"] if r["cellule"] == "partage"}
    aucun = {r["electorat"]: (r["modele"], r["mesure"]) for r in ev["reports"] if r["cellule"] == "Aucun"}
    pers = {2022: 2.61, 2017: 4.11}[election]
    pers_aucun = {2022: 1.75, 2017: 4.8}[election]
    mae = float(np.mean([abs(m - x) for m, x in part.values()]))
    mae_a = float(np.mean([abs(m - x) for m, x in aucun.values()]))
    jlm = part.get(JLM)
    duels = {}
    for k in dom:
        f, _, _ = E.reconstruire(variante, stem, coupure, chaines=[k])
        for a, b in combinations(E.DUELS_CLES_PAR_ELECTION[election], 2):
            duels.setdefault(f"{a} / {b}", []).append(100 * float(duel_draws(f, a, b, 0.0, scrutin=False).mean()))
    ec = {d: round(max(v) - min(v), 2) for d, v in duels.items()}
    ratio = _sig(fit, MLP) / _sig(fit, EM)
    res = {
        "1_mae_partages": round(mae, 2), "1_ok": mae <= pers + 1.0, "persistance": pers,
        "2_melenchon": jlm, "2_ok": jlm is not None and abs(jlm[0] - jlm[1]) < 5.0,
        "3_sigma_lp_sur_macron": round(ratio, 3), "3_ok": ratio < 1.5,
        "3b_monotonie_gauche": round(monotonie_gauche(fit), 3), "3b_ok": monotonie_gauche(fit) >= 0.9,
        "4_n_dominant": len(dom), "4_ecart_duels_max": max(ec.values()) if len(dom) > 1 else None,
        "4_ok": len(dom) >= 3 and max(ec.values()) <= 3.0,
        "5_t1_mae": ev["t1_resultat"]["mae_pt"], "5_ok": ev["t1_resultat"]["mae_pt"] <= 3.1,
        "5_t1_couverture": ev["t1_resultat"]["couverture_ic90"],
        "aucun_mae": round(mae_a, 2), "aucun_ok": mae_a <= pers_aucun + 2.0,
    }
    mp = monotonie_ponderee(fit)
    res["3b_pondere_part_mal"] = None if mp is None else round(mp, 3)
    if fit.arc_draws is not None:                                   # §8.54 : 3b circulaire
        res["3b_ok"] = mp is not None and mp <= 0.10
        res["3b_monotonie_gauche"] = None
    res["structure"] = structure(fit)
    res["reports_fit"] = lecture_reports_fit(fit, kw, stem, dom)          # §8.56
    if election == 2022:
        res["v1_melenchon_macron"] = v1(fit)                                 # §8.59
    if getattr(fit, "nu_t_draws", None) is not None:
        res["corr_lognu_logsigmaLP"] = round(float(np.corrcoef(np.log(fit.nu_t_draws),
                                                               np.log(fit.sigma_draws[:, fit.candidates.index(MLP)]))[0, 1]), 3)
    if getattr(fit, "nu_t_draws", None) is not None:
        lnu = np.log(fit.nu_t_draws)
        res.update({"6_log_nu_sd": round(float(lnu.std()), 3), "6_nu_q95": round(float(np.percentile(fit.nu_t_draws, 95)), 1),
                    "6_ok": float(lnu.std()) <= 0.5 and float(np.percentile(fit.nu_t_draws, 95)) < 30})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variante", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--election", type=int, default=2022)
    ap.add_argument("--coupure", default="T1-30")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--lse-officiel", action="store_true")
    ap.add_argument("--sans-criteres", action="store_true", help="partition seule (contrôle sur d'anciennes chaînes)")
    ap.add_argument("--v1-modes", action="store_true", help="§8.61 : V1 lu sur TOUS les modes (médiane, pire)")
    args = ap.parse_args()
    E.ELECTION = args.election
    E.LSE_OFFICIEL = args.lse_officiel
    lignes = par_chaine(args.variante, args.stem, args.coupure, args.n)
    part = partition(lignes)
    out = {"spec": "spec_second_tour §8.50-8.51", "variante": args.variante, "stem": args.stem,
           "election": args.election, "coupure": args.coupure, "B_quad": int(__import__("os").environ.get("PRES2027_B_QUAD", "25")),
           "chaines": lignes, "partition": part}
    if not args.sans_criteres:
        out["criteres_mode_dominant"] = criteres(args.variante, args.stem, args.coupure, lignes, part["dominant"],
                                                 args.election)
    if args.v1_modes and args.election == 2022:                       # §8.61
        par_mode = []
        for g in [part["dominant"]] + part["autres"]:
            f, _, _ = E.reconstruire(args.variante, args.stem, args.coupure, chaines=g)
            c = v1(f).get("_criteres", {})
            par_mode.append({"chaines": g, **c})
        # §8.63 : modes retenus par log-densité (Δlp ≤ 10 du meilleur ; pseudo-chaînes MAP : SE nul)
        lpmax = max(l["lp"] for l in lignes)
        for m in par_mode:
            lp_m = max(l["lp"] for l in lignes if l["chaine"] in m["chaines"])
            zp = max(float(np.abs(np.load(sr.CHAINES_DIR / f"{args.stem}_c{k}.npz")["s__z_pos"]).mean(axis=(0, 1)).max())
                     for k in m["chaines"])
            m.update({"lp_max": lp_m, "retenu_lp": lp_m >= lpmax - 10.0, "max_abs_z_pos": round(zp, 2),
                      "z_pos_force": zp > 3.0})
        ret = [m for m in par_mode if m["retenu_lp"]]
        out["v1_modes_retenus_lp"] = [{k: m.get(k) for k in ("chaines", "mae_partages", "mae_U", "loho_ok_U",
                                                            "max_abs_z_pos")} for m in ret]
        maes = [m["mae_partages"] for m in par_mode if "mae_partages" in m]
        out["v1_tous_modes"] = {"par_mode": par_mode, "mediane": round(float(np.median(maes)), 2),
                                "pire": round(float(max(maes)), 2)}
        print(json.dumps(out["v1_tous_modes"], ensure_ascii=False, indent=1, default=float))
    pref = "v1loho" if (args.v1_modes and args.sans_criteres) else "modes"    # ne pas écraser une lecture complète
    dest = sr.OUT_DIR / f"{pref}_{args.stem}.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + "\n")
    print(json.dumps({"partition": part, **({"criteres": out["criteres_mode_dominant"]} if "criteres_mode_dominant" in out else {})},
                     ensure_ascii=False, indent=1, default=float))
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
