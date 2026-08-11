"""Correction de 06_same_day_poll_coherence.py : le test précédent moyennait
un candidat à travers TOUTES les hypothèses d'un institut, sans vérifier que
les deux instituts comparés testaient le MÊME champ de candidats -- un écart
peut venir de la composition du champ (Le Pen face à Attal+Philippe n'est pas
la même mesure que Le Pen face à Attal seul), pas d'un house effect. Ici,
on n'apparie que des scénarios (institut x date x hypothèse) avec
EXACTEMENT le même ensemble de candidats testés.

Exécution : .venv/bin/python notebooks/06b_same_day_poll_coherence_matched.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.core.live_dataset import load_raw_polls

MIN_POLL_DATE = pd.Timestamp("2026-01-01")
MAX_GAP_DAYS = 3


def build_scenarios(df: pd.DataFrame, candidat_col="candidat", hyp_col="hypothese",
                    institut_col="institut", date_col="date_fin", notice_col="notice",
                    intention_col="intention", ech_col="echantillon") -> list[dict]:
    """Un scénario = (notice, hypothese) -- l'ensemble EXACT de candidats testés
    ensemble, avec leurs valeurs. C'est l'unité de comparaison correcte (pas
    l'institut, qui peut tester plusieurs champs différents le même jour)."""
    df = df.copy()
    df[hyp_col] = df[hyp_col].fillna("__unique__")
    scenarios = []
    for (notice, hyp), g in df.groupby([notice_col, hyp_col]):
        scenarios.append(dict(
            notice=notice, hypothese=hyp, institut=g[institut_col].iloc[0],
            date=pd.Timestamp(g[date_col].iloc[0]), echantillon=float(g[ech_col].iloc[0]),
            candidate_set=frozenset(g[candidat_col]),
            values={r[candidat_col]: r[intention_col] for _, r in g.iterrows()},
        ))
    return scenarios


def pairwise_check_matched(scenarios: list[dict], max_gap_days: int) -> pd.DataFrame:
    """N'apparie que des scénarios de DEUX instituts différents avec
    EXACTEMENT le même ensemble de candidats testés, à <= max_gap_days
    d'écart -- contrôle strict de la composition du champ."""
    out = []
    for i, j in combinations(range(len(scenarios)), 2):
        a, b = scenarios[i], scenarios[j]
        if a["institut"] == b["institut"] or a["notice"] == b["notice"]:
            continue
        if a["candidate_set"] != b["candidate_set"]:
            continue
        gap = abs((a["date"] - b["date"]).days)
        if gap > max_gap_days:
            continue
        if not (np.isfinite(a["echantillon"]) and np.isfinite(b["echantillon"])):
            continue   # echantillon manquant dans la donnée brute (ex. Ifop 7-8 juillet) -- paire ignorée, pas comptée comme un échec
        for cand in a["candidate_set"]:
            pa, pb = a["values"][cand], b["values"][cand]
            na, nb = a["echantillon"], b["echantillon"]
            if not (np.isfinite(pa) and np.isfinite(pb)):
                continue
            var = pa * (100 - pa) / max(na, 1) + pb * (100 - pb) / max(nb, 1)
            if var <= 0:
                continue
            z = (pa - pb) / np.sqrt(var)
            out.append(dict(candidat=cand, gap_days=gap, institut_a=a["institut"], institut_b=b["institut"],
                            champ=sorted(a["candidate_set"]), pa=pa, pb=pb, z=z))
    return pd.DataFrame(out)


def coverage_report(z: np.ndarray) -> dict:
    levels = {"50%": 0.674, "80%": 1.282, "90%": 1.645}
    return {k: float(np.mean(np.abs(z) <= q)) for k, q in levels.items()}


def main():
    raw = load_raw_polls()
    raw2027 = raw[pd.to_datetime(raw["date_fin"]) >= MIN_POLL_DATE]
    scenarios_2027 = build_scenarios(raw2027)
    pairs = pairwise_check_matched(scenarios_2027, MAX_GAP_DAYS)
    print(f"2027 : {len(scenarios_2027)} scénarios (institut x date x hypothèse), "
          f"{len(pairs)} paires candidat x scénario avec champ EXACTEMENT identique")
    if len(pairs):
        print(f"  RMS(z) = {np.sqrt(np.mean(pairs['z'].to_numpy() ** 2)):.2f} (attendu ~1.0)")
        cov = coverage_report(pairs["z"].to_numpy())
        for k, v in cov.items():
            print(f"  couverture |z|<=seuil {k} : {v*100:5.1f}% (nominal {k})")
        print("\n  Détail (gap=0, même jour) :")
        same_day = pairs[pairs["gap_days"] == 0]
        print(same_day[["candidat", "champ", "institut_a", "pa", "institut_b", "pb", "z"]].to_string(index=False))
    else:
        print("  Aucune paire à champ EXACTEMENT identique sur 2027 -- champ trop hétérogène "
              "entre instituts pour ce test sur si peu de sondages.")

    print("\n=== Même test sur 2017+2022 (plus de scénarios, matching plus probable) ===")
    from pipeline.historical import ELECTION_DATES, PARSED_DIR, POLL_FILES, _resolve_candidate, load_blocs
    all_pairs = []
    total_scenarios = 0
    for elec in (2017, 2022):
        polls = pd.read_csv(PARSED_DIR / POLL_FILES[elec])
        t1 = polls[polls["tour"] == "Premier tour"].copy()
        blocs = load_blocs()
        real = set(blocs[blocs["election"] == elec]["candidat"])
        t1["candidat"] = t1["candidat"].apply(lambda raw_name: _resolve_candidate(raw_name, real))
        t1 = t1[t1["candidat"].isin(real)]
        date_fin = pd.to_datetime(t1["date_fin"], errors="coerce")
        date_notice = pd.to_datetime(t1.get("date_notice"), errors="coerce")
        t1["date_fin"] = date_fin.fillna(date_notice)
        t1 = t1[t1["date_fin"].notna()]
        elec_date = pd.Timestamp(ELECTION_DATES[elec])
        t1 = t1[(t1["date_fin"] > elec_date - pd.Timedelta(days=400)) & (t1["date_fin"] <= elec_date)]
        scenarios = build_scenarios(t1)
        total_scenarios += len(scenarios)
        hp = pairwise_check_matched(scenarios, MAX_GAP_DAYS)
        hp["election"] = elec
        all_pairs.append(hp)
    hist_pairs = pd.concat(all_pairs, ignore_index=True)
    print(f"{total_scenarios} scénarios (2017+2022), {len(hist_pairs)} paires à champ identique")
    if len(hist_pairs):
        print(f"  RMS(z) = {np.sqrt(np.mean(hist_pairs['z'].to_numpy() ** 2)):.2f} (attendu ~1.0)")
        cov = coverage_report(hist_pairs["z"].to_numpy())
        for k, v in cov.items():
            print(f"  couverture |z|<=seuil {k} : {v*100:5.1f}% (nominal {k})")
        print("\n  Par tranche d'écart (jours) :")
        for gap in range(0, MAX_GAP_DAYS + 1):
            sub = hist_pairs[hist_pairs["gap_days"] == gap]
            if len(sub) == 0:
                continue
            cov_g = coverage_report(sub["z"].to_numpy())
            print(f"    gap={gap}j (n={len(sub)}) : IC50={cov_g['50%']*100:.0f}% "
                  f"IC80={cov_g['80%']*100:.0f}% IC90={cov_g['90%']*100:.0f}%")
        print(f"\n  Taille de champ (nb candidats) la plus fréquente parmi les paires : "
              f"{hist_pairs['champ'].apply(len).mode().iloc[0]}")


if __name__ == "__main__":
    main()
