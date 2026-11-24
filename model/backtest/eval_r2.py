"""Critères de R2 (spec_second_tour §8.44.8) sur un fit V2 T2−7 fusionné.

    python -m model.backtest.eval_r2 --variante hier2d [--stem second_round_hier2d_v2_2022_T2-7]
    python -m model.backtest.eval_r2 --variante hier --stem r2_temoin_hier_v2_2022_T2-7

Reconstruit le fit par fusion des chaînes sauvées (sélection §8.8), puis :
reports prédits (toutes les notices 2022, y compris après `as_of`), rayon moyen
en v sur le champ T1 réel, résidus des nœuds T1, parts au T1 contre le résultat,
R̂ des ν, duel et σ_T2.
"""
from __future__ import annotations

import argparse
import json

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.special import logsumexp

import model.backtest.second_round as sr
from model.backtest.reports_t2 import _norm, mesures
from model.models.spatial_pooling.fit import duel_draws
from model.models.spatial_pooling.geometry import (W, W2, log_attractivite, log_attractivite_2d,
                                                   reports_cellules, spatial_shares, spatial_shares_2d)
from model.models.spatial_pooling.roster import build_poll_arrays, split_tours

ELECTION = 2022
LSE_OFFICIEL = False   # §8.49 : --lse-officiel (réévaluation ou fit refait avec le drapeau)


def reconstruire(variante: str, stem: str, coupure: str = "T2-7", chaines=None, lse_officiel: bool = False):
    sr.ABSTENTION = True
    sr.VARIANTE_KW.clear()
    sr.VARIANTE_KW.update({"hier": {"eta_hier": True},
                           "hier2d": {"eta_hier": True, "carte_2d": True, "_reports": True},
                           "hier_rep": {"eta_hier": True, "_reports": True},
                           "hiercercle": {"eta_hier": True, "carte_cercle": True, "_reports": True},
                           "hiercercle_sr": {"eta_hier": True, "carte_cercle": True},
                           "hiercercle_ra": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True},
                           "hier_ra": {"eta_hier": True, "_reports_avant": True},
                           "hiercercle_ra_abs": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True,
                                                 "abst_situee": True},
                           "hier_ra_t": {"eta_hier": True, "_reports_avant": True, "noyau_t": True},
                        "hier_ra_lie": {"eta_hier": True, "_reports_avant": True, "sigma_lie": True},
                        "hiercercle_ra_ferme": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True, "arc_ferme": True},
                        "hier_ra_fid": {"eta_hier": True, "_reports_avant": True, "fidelite": True},
                        "hiercercle_ra_fid": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True, "fidelite": True},
                        "hier_ra_fid_omk": {"eta_hier": True, "_reports_avant": True, "fidelite": True, "omega_electorat": True},
                        "hiercercle_ra_fid_omk": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True, "fidelite": True, "omega_electorat": True},
                        "hiercercle_ra_fid_omk_lam": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True, "fidelite": True, "omega_electorat": True, "melange_lambda": True},
                        "hiercercle_ra_fid_sigeq": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True, "fidelite": True, "sigma_egaux": True},
                        "hiercercle_ra_fid_sigdyn": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True, "fidelite": True, "sigma_dynamiques": True},
                        "hiercercle_ra_fid_sigdyn30": {"eta_hier": True, "carte_cercle": True, "_reports_avant": True, "fidelite": True, "sigma_dynamiques_30j": True},
                        "hiercercle_ra_fid_sigdyn_ext": {"eta_extremite": True, "carte_cercle": True, "_reports_avant": True, "fidelite": True, "sigma_dynamiques": True}}[variante])
    if lse_officiel or LSE_OFFICIEL:
        sr.VARIANTE_KW["_lse_officiel"] = True
    ref = sr.T1_DATES[ELECTION] if coupure.startswith("T1") else sr.T2_DATES[ELECTION]
    as_of = ref - pd.Timedelta(days=int(coupure.split("-")[1]))
    raw = sr.polls_historiques(ELECTION, as_of)
    import os
    if os.environ.get("PRES2027_SANS_PAIRE"):                    # V2 au MAP (§8.69)
        raw = sr._sans_paire(raw, *os.environ["PRES2027_SANS_PAIRE"].split("|"))
    paths = [sr.CHAINES_DIR / f"{stem}_c{k}.npz" for k in (chaines if chaines is not None else range(sr.N_CHAINES))]
    fit, _ = sr._fit(raw, as_of, ELECTION, dict(draws=400, tune=600, chains=4, mcmc_runner=sr._runner_fusion(paths)))
    return fit, raw, as_of


def _omega_g(fit):
    """Terme situé de l'abstention (§8.48 bis), (S, N_θ), ou None."""
    if getattr(fit, "gamma_draws", None) is None:
        return None
    from model.models.spatial_pooling.geometry import THETA
    return jnp.asarray(-fit.gamma_draws[:, None] * np.cos(np.asarray(THETA))[None, :])


def _A(fit, w):
    if fit.arc_draws is not None:
        from model.models.spatial_pooling.geometry import W_THETA, log_attractivite_cercle
        return np.asarray(log_attractivite_cercle(jnp.asarray(fit.mu_draws), jnp.asarray(fit.sigma_draws),
                                                  jnp.asarray(fit.arc_draws), jnp.asarray(w))), W_THETA
    if fit.nu_draws is not None:
        return np.asarray(log_attractivite_2d(jnp.asarray(fit.mu_draws), jnp.asarray(fit.sigma_draws),
                                              jnp.asarray(fit.nu_draws), jnp.asarray(fit.lambda_draws),
                                              jnp.asarray(w))), W2
    nu = None if getattr(fit, "nu_t_draws", None) is None else jnp.asarray(fit.nu_t_draws)
    return np.asarray(log_attractivite(jnp.asarray(fit.mu_draws), jnp.asarray(fit.sigma_draws), jnp.asarray(w), nu)), W


def _parts(fit, w, mask):
    if fit.arc_draws is not None:
        from model.models.spatial_pooling.geometry import spatial_shares_cercle
        return np.asarray(spatial_shares_cercle(jnp.asarray(fit.mu_draws), jnp.asarray(fit.sigma_draws),
                                                jnp.asarray(fit.arc_draws), jnp.asarray(w), jnp.asarray(mask)))
    if fit.nu_draws is not None:
        return np.asarray(spatial_shares_2d(jnp.asarray(fit.mu_draws), jnp.asarray(fit.sigma_draws),
                                            jnp.asarray(fit.nu_draws), jnp.asarray(fit.lambda_draws),
                                            jnp.asarray(w), jnp.asarray(mask)))
    nu = None if getattr(fit, "nu_t_draws", None) is None else jnp.asarray(fit.nu_t_draws)
    return np.asarray(spatial_shares(jnp.asarray(fit.mu_draws), jnp.asarray(fit.sigma_draws),
                                     jnp.asarray(w), jnp.asarray(mask), nu=nu))


def evaluer(fit, raw, as_of) -> dict:
    cands = fit.candidates
    idx = {_norm(c): i for i, c in enumerate(cands)}
    res = sr.load_results()
    t1 = res[(res.election == ELECTION) & (res.tour == "Premier tour")]
    champ = [idx[_norm(c)] for c in t1.candidat if _norm(c) in idx]
    mask = np.zeros(len(cands)); mask[champ] = 1.0
    a, b = sr.FINALE[ELECTION]
    ia, ib = idx[_norm(a)], idx[_norm(b)]
    out = {"n_tirages": int(len(fit.mu_draws))}

    # (1) reports prédits à `as_of` (w_now), tous électorats mesurés
    A, poids = _A(fit, fit.w_draws)
    S = A.shape[0]
    lse = logsumexp(A if fit.masque_lse is None else A + np.log(fit.masque_lse)[None, :, None], axis=1)   # §8.49
    s_ab = duel_draws(fit, a, b, 0.0, scrutin=False)
    m = mesures(pd.read_csv(sr.PARSED_DIR / f"reports_t2_{ELECTION}.csv"))
    cell = []
    for e in m.electorat_t1.unique():
        k = idx[_norm(e)]
        th, _ = reports_cellules(jnp.asarray(A), jnp.asarray(mask), jnp.full(S, ia), jnp.full(S, ib), jnp.full(S, k),
                                 jnp.asarray(fit.kappa_draws), jnp.asarray(fit.eta_draws[:, ia] - fit.eta_draws[:, ib]),
                                 jnp.asarray(lse), jnp.asarray(fit.omega_draws), jnp.asarray(s_ab), poids, n_newton=8,
                                 omega_g=_omega_g(fit),
                                 omega_k=(None if getattr(fit, "omega_k_draws", None) is None
                                          else jnp.asarray(fit.omega_k_draws)),
                                 lam=(None if getattr(fit, "lam_mix_draws", None) is None
                                      else jnp.asarray(fit.lam_mix_draws)))
        th = 100 * np.asarray(th)
        pred = {"Macron": th[:, 0], "Le Pen": th[:, 1], "Aucun": th[:, 2], "partage": 100 * th[:, 0] / (th[:, 0] + th[:, 1])}
        for _, r in m[m.electorat_t1 == e].iterrows():
            x = pred[r.cellule]
            cell.append({"electorat": e, "cellule": r.cellule, "mesure": round(float(r["mean"]), 1),
                         "modele": round(float(x.mean()), 1), "ic90": [round(float(v), 1) for v in np.percentile(x, [5, 95])],
                         "erreur": round(float(x.mean() - r["mean"]), 1)})
    out["reports"] = cell

    # (2) rayon moyen en v sur le champ T1 réel
    sv = fit.sigma_draws[:, champ].mean(1)
    out["sigma_v_champ"] = {"mediane": round(float(np.median(sv)), 4),
                            "ic90": [round(float(v), 4) for v in np.percentile(sv, [5, 95])]}

    # (3) T1 : résidus des nœuds (moyenne postérieure) et parts au T1 contre le résultat
    t1raw, _ = split_tours(raw[pd.to_datetime(raw["date_fin"]) >= pd.Timestamp("1900-01-01")])
    arr = build_poll_arrays(t1raw, cands, couverture_min=None)
    col = np.searchsorted(fit.w_dates, np.floor(arr["dates"]))
    col = np.clip(col, 0, fit.w_path_draws.shape[2] - 1)
    resid = []
    for p in range(len(arr["Y"])):
        pi = _parts(fit, fit.w_path_draws[:, :, col[p]], arr["tested_mask"][p]).mean(0)
        t = arr["tested_mask"][p] > 0
        resid += list(np.abs(pi[t] - arr["Y"][p, t]))
    out["t1_mae_noeuds_pt"] = round(100 * float(np.mean(resid)), 3)
    c_t1 = int(np.clip(np.searchsorted(fit.w_dates, float(sr.T1_DATES[ELECTION].toordinal())), 0,
                       fit.w_path_draws.shape[2] - 1))
    pi_t1 = _parts(fit, fit.w_path_draws[:, :, c_t1], mask)
    vrai = t1.assign(i=t1.candidat.map(lambda c: idx.get(_norm(c)))).dropna(subset=["i"])
    y = vrai.resultat.to_numpy() / vrai.resultat.sum()
    pm = pi_t1[:, vrai.i.astype(int)]
    lo, hi = np.percentile(pm, [5, 95], axis=0)
    out["t1_resultat"] = {"mae_pt": round(100 * float(np.abs(pm.mean(0) - y).mean()), 3),
                          "couverture_ic90": f"{int(((lo <= y) & (y <= hi)).sum())}/{len(y)}"}

    # (4) ν et (5) duel, σ_T2
    if fit.nu_draws is not None:
        out["nu"] = {cands[i]: round(float(fit.nu_draws[:, i].mean()), 3) for i in champ}
        out["lambda_u"] = round(float(fit.lambda_draws.mean()), 4)
    if fit.arc_draws is not None:
        out["arc"] = round(float(fit.arc_draws.mean()), 4)
        out["f_arriere"] = round(float(1 - fit.arc_draws.mean() / (2 * np.pi)), 4)
    out["s_ab"] = round(float(s_ab.mean()), 4)
    out["sigma_t2"] = fit.diagnostics.get("sigma_t2")
    out["kappa"] = fit.diagnostics.get("kappa")
    out["rhat_max"] = fit.diagnostics.get("rhat_max")
    out["rhat_pires_sites"] = fit.diagnostics.get("rhat_pires_sites")
    out["chaines_retenues"] = list(sr.CHAINES_RETENUES)
    return out


def rhat_nu(stem: str, champ_noms: list[str], cands: list[str]) -> float | None:
    from numpyro.diagnostics import split_gelman_rubin
    zs = [np.load(sr.CHAINES_DIR / f"{stem}_c{k}.npz") for k in range(sr.N_CHAINES)]
    if "s__nu" not in zs[0].files:
        return None
    keep = sr.CHAINES_RETENUES or list(range(len(zs)))
    x = np.concatenate([zs[k]["s__nu"] for k in keep], axis=0)          # (C, D, N)
    idx = [cands.index(c) for c in champ_noms]
    return round(float(np.nanmax(split_gelman_rubin(x[..., idx]))), 3)


DUELS_CLES_PAR_ELECTION = {
    2022: ["Emmanuel Macron", "Marine Le Pen", "Jean-Luc Mélenchon", "Eric Zemmour", "Valérie Pécresse"],
    2017: ["Emmanuel Macron", "Marine Le Pen", "Jean-Luc Mélenchon", "François Fillon", "Benoît Hamon"],
}
DUELS_CLES = DUELS_CLES_PAR_ELECTION[2022]


def par_chaine(variante: str, stem: str, coupure: str) -> dict:
    """Critère 4 révisé (§8.44, révision 6) : écarts ENTRE chaînes indépendantes
    sur les partages de reports, les parts T1 et 10 duels clés (dont hypothétiques)."""
    from itertools import combinations
    lignes = []
    for k in range(sr.N_CHAINES):
        fit, raw, as_of = reconstruire(variante, stem, coupure, chaines=[k])
        ev = evaluer(fit, raw, as_of)
        duels = {}
        for a, b in combinations(DUELS_CLES_PAR_ELECTION[ELECTION], 2):
            duels[f"{a} / {b}"] = round(100 * float(duel_draws(fit, a, b, 0.0, scrutin=False).mean()), 2)
        part = {r["electorat"]: r["modele"] for r in ev["reports"] if r["cellule"] == "partage"}
        nu_sd = (float(np.mean([fit.nu_draws[:, fit.candidates.index(c)].std() for c in ev["nu"]
                                if c not in ("Emmanuel Macron",)])) if fit.nu_draws is not None else None)
        lignes.append({"chaine": k, "lp": float(sr.LP_CHAINES[0]) if sr.LP_CHAINES else None,
                       "partages": part, "duels": duels, "t1_mae_pt": ev["t1_resultat"]["mae_pt"],
                       "kappa": ev["kappa"], "nu": ev.get("nu"), "nu_sd_post_moyen": nu_sd})
    ecart = lambda vals: round(float(max(vals) - min(vals)), 2)  # noqa: E731
    return {"chaines": lignes,
            "ecart_partages": {e: ecart([l["partages"][e] for l in lignes]) for e in lignes[0]["partages"]},
            "ecart_duels": {d: ecart([l["duels"][d] for l in lignes]) for d in lignes[0]["duels"]},
            "ecart_lp": ecart([l["lp"] for l in lignes if l["lp"] is not None]) if lignes[0]["lp"] is not None else None,
            "nu_sd_prior": 0.208}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variante", choices=["hier", "hier2d", "hier_rep", "hiercercle", "hiercercle_sr", "hiercercle_ra", "hier_ra", "hiercercle_ra_abs", "hier_ra_t", "hier_ra_lie", "hiercercle_ra_ferme", "hier_ra_fid", "hiercercle_ra_fid", "hier_ra_fid_omk", "hiercercle_ra_fid_omk", "hiercercle_ra_fid_sigeq", "hiercercle_ra_fid_sigdyn", "hiercercle_ra_fid_sigdyn30", "hiercercle_ra_fid_sigdyn_ext"], required=True)
    ap.add_argument("--stem", default=None)
    ap.add_argument("--coupure", default="T2-7")
    ap.add_argument("--par-chaine", action="store_true", help="critère 4 révisé (écarts entre chaînes)")
    ap.add_argument("--election", type=int, choices=[2017, 2022], default=2022)
    ap.add_argument("--lse-officiel", action="store_true", help="valeur inclusive sur le champ officiel (§8.49)")
    args = ap.parse_args()
    global ELECTION, LSE_OFFICIEL
    ELECTION = args.election
    LSE_OFFICIEL = args.lse_officiel
    sfx = "_lseo" if args.lse_officiel else ""
    stem = args.stem or f"second_round_{args.variante}_v2_{ELECTION}_{args.coupure}"
    if args.par_chaine:
        out = {"spec": "spec_second_tour §8.44 (révision 6)", "variante": args.variante, "stem": stem,
               "coupure": args.coupure, **par_chaine(args.variante, stem, args.coupure)}
        dest = sr.OUT_DIR / f"eval_r2_{args.variante}_{ELECTION}_{args.coupure}{sfx}_par_chaine.json"
        dest.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + "\n")
        print(json.dumps({k: out[k] for k in ("ecart_partages", "ecart_duels", "ecart_lp")}, ensure_ascii=False, indent=1))
        for l in out["chaines"]:
            print(l["chaine"], l["lp"], l["kappa"], l["t1_mae_pt"], l["partages"], l["nu_sd_post_moyen"])
        print(f"-> {dest}")
        return
    fit, raw, as_of = reconstruire(args.variante, stem, args.coupure)
    out = {"spec": "spec_second_tour §8.44.8", "variante": args.variante, "stem": stem, **evaluer(fit, raw, as_of)}
    if fit.nu_draws is not None:
        out["rhat_nu_champ"] = rhat_nu(stem, list(out["nu"]), fit.candidates)
    dest = sr.OUT_DIR / (f"eval_r2_{args.variante}_{ELECTION}{sfx}.json" if args.coupure == "T2-7"
                         else f"eval_r2_{args.variante}_{ELECTION}_{args.coupure}{sfx}.json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "reports"}, ensure_ascii=False, indent=1))
    print(pd.DataFrame(out["reports"]).to_string(index=False))
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
