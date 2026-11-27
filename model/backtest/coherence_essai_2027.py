"""Cohérence duel / reports dans le modèle (spec_second_tour §8.105 e1), 0 fit.

Pour chaque notice n d'une hypothèse V1 cachée (13 notices Harris) et chaque géométrie :
- données : écart s_n − agrégat des cellules MESURÉES sur E_n, poids p_e = T1 de la même notice ;
- modèle : écart s_n − agrégat des cellules du MODÈLE (fit V1 rejoué, à la date de la notice) sur le même E_n,
  mêmes poids.
Critère figé : la contrainte du Newton transmet le biais si l'écart du modèle a le même signe que celui des données
dans au moins 10 notices sur 13 ET une moyenne d'au moins 1 pt.

Usage : JAX_ENABLE_X64=1 PRES2027_B_QUAD=100 python -m model.backtest.coherence_essai_2027
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from model.backtest import essai_2027 as X
from model.backtest import second_round as sr
from model.backtest.beta_essai_2027 import reconstruire


def notices(raw, rep, h):
    """[(notice, date, E_n, p_e, s_n, cellules mesurées (E,3) en parts)] — même construction que K_s (§8.102)."""
    a, b = [x.strip() for x in h.split("–")]
    out = []
    for nt, g in rep[rep.hypothese == h].groupby("notice"):
        t = g.pivot_table(index="electorat_t1", columns="candidat_t2", values="part")[[a, b, "Aucun"]].dropna()
        r = raw[(raw.institut == g.institut.iloc[0]) & (raw.date_fin.astype(str).str[:10] == str(g.date_fin.iloc[0])[:10])]
        duel = r[(r.tour == "Deuxième tour") & (r.hypothese.str.replace("Hypothèse ", "").str.strip() == h)]
        t1 = r[r.tour == "Premier tour"]
        E = [X.court(x) for x in t.index]
        hyps = [x for x, gg in t1.groupby("hypothese") if {a, b, *E} <= set(gg.candidat)]
        if len(duel) != 2 or not hyps:
            continue
        d = duel.set_index("candidat").intention
        pe = np.mean([t1[t1.hypothese == x].set_index("candidat").intention.reindex(E).to_numpy() for x in hyps], axis=0)
        M = t.to_numpy() / 100
        M = M / M.sum(1, keepdims=True)
        t1c = sorted(set(t1[t1.hypothese == hyps[0]].candidat))      # §8.110 f3 : champ T1 de la notice
        out.append((nt, pd.Timestamp(g.date_fin.iloc[0]), E, pe, float(d[a] / (d[a] + d[b])), M, t1c))
    return out


def agregat(pe, M):
    return float((pe * M[:, 0]).sum() / (pe * (M[:, 0] + M[:, 1])).sum())


def main() -> dict:
    raw, rep, _ = X.donnees()
    res = {"spec": "§8.105 e1", "notices": []}
    for h in X.CACHES_V1:
        a, b = [x.strip() for x in h.split("–")]
        L = notices(raw, rep, h)
        for geo in ("segment", "cercle"):
            ep = f"v1:{h}"
            nom = "".join(ch for ch in ep if ch.isalnum())
            J = json.loads((sr.OUT_DIR / f"essai2027_{geo}_{nom}.json").read_text())
            r = min(J["inits"], key=lambda i: i["U"])
            lignes = [(i, e) for i, (_, _, E, _, _, _, _) in enumerate(L) for e in E]
            fit, _, _, champ, _ = reconstruire(geo, ep, r, len(lignes))
            idx = {X.court(c): i for i, c in enumerate(fit.candidates)}
            S = fit.w_path_draws.shape[0]
            jj = [int(np.clip(np.searchsorted(fit.w_dates, float(L[i][1].toordinal())), 0, len(fit.w_dates) - 1))
                  for i, _ in lignes]
            w = fit.w_path_draws[np.arange(S), :, np.array(jj)]
            k = np.array([idx[e] for _, e in lignes])
            th = X.cellules(fit, champ, np.full(S, idx[a]), np.full(S, idx[b]), k, w=w) / 100
            # §8.110 f3 : part de duel DU MODÈLE (s_mod) et parts T1 DU MODÈLE, à la date de chaque ligne
            from model.models.spatial_pooling.fit import _duel_par_tirage
            from model.backtest.eval_r2 import _parts
            s_mod = np.asarray(_duel_par_tirage(fit, w, np.full(S, idx[a]), np.full(S, idx[b])))
            pos = 0
            for nt, t, E, pe, s_n, M, t1c in L:
                P = th[pos:pos + len(E)]
                r0 = pos
                pos += len(E)
                mask = np.array([float(X.court(c) in t1c) for c in fit.candidates])
                pm = np.asarray(_parts(fit, w[r0:r0 + 1], mask[None, :]))[0]
                pm = 100 * np.array([pm[idx[e]] for e in E])
                sm = float(s_mod[r0])
                res["notices"].append({"hypothese": h, "notice": nt, "geo": geo, "s_n": round(s_n, 4),
                                       "s_mod": round(sm, 4),
                                       "ecart_donnees": round(100 * (s_n - agregat(pe, M)), 2),
                                       "ecart_modele": round(100 * (s_n - agregat(pe, P)), 2),
                                       "ecart_modele_smod": round(100 * (sm - agregat(pe, P)), 2),
                                       "ecart_modele_smod_poids_modele": round(100 * (sm - agregat(pm, P)), 2),
                                       "couverture_T1": round(float(pe.sum()), 1)})
    S = {}
    for geo in ("segment", "cercle"):
        x = [n for n in res["notices"] if n["geo"] == geo]
        dm = np.array([n["ecart_modele"] for n in x])
        dd = np.array([n["ecart_donnees"] for n in x])
        meme = int((np.sign(dm) == np.sign(dd)).sum())
        S[geo] = {"n": len(x), "moyenne_ecart_modele": round(float(dm.mean()), 2),
                  "moyenne_ecart_donnees": round(float(dd.mean()), 2), "meme_signe": meme,
                  "transmet_biais": bool(meme >= 10 and dm.mean() >= 1.0)}
    for cle in ("ecart_modele_smod", "ecart_modele_smod_poids_modele"):   # §8.110 f3
        for geo in ("segment", "cercle"):
            x = [n for n in res["notices"] if n["geo"] == geo]
            dm = np.array([n[cle] for n in x])
            dd = np.array([n["ecart_donnees"] for n in x])
            S[geo][cle] = {"moyenne": round(float(dm.mean()), 2), "meme_signe": int((np.sign(dm) == np.sign(dd)).sum())}
        g1, g2 = S["segment"][cle], S["cercle"][cle]
        S[f"distingue_{cle}"] = bool(abs(g1["moyenne"] - g2["moyenne"]) >= 1 and abs(g1["meme_signe"] - g2["meme_signe"]) >= 3)
    res["synthese"] = S
    (sr.OUT_DIR / "essai2027_coherence.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(S, ensure_ascii=False, indent=1))
    return res


if __name__ == "__main__":
    main()
