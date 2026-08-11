"""Vérifie si pi(1-pi)/N (bruit d'échantillonnage SEUL, sans aucun modèle) est
réaliste, en comparant des sondages RÉELLEMENT proches dans le temps (même
jour ou quelques jours d'écart, instituts DIFFÉRENTS -- pas les hypothèses
d'un même sondage, qui partagent le terrain). S'il n'y a quasiment pas eu le
temps pour l'opinion de bouger, tout écart au-delà du bruit d'échantillonnage
mesuré doit venir d'un bruit de mesure sous-estimé (house effects), PAS d'une
dérive non captée -- ce test isole la question, contrairement à la
stratification par âge (04d), qui confond les deux (une dérive non captée
donne une erreur à peu près CONSTANTE dans le temps, pas croissante --
indiscernable d'un excès de variance par ce test-là).

Exécution : .venv/bin/python notebooks/06_same_day_poll_coherence.py
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
MAX_GAP_DAYS = 3   # "proche dans le temps" -- au-delà, la dérive redevient un facteur


def build_one_row_per_poll(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (notice, candidat) -- moyenne sur les hypothèses du même
    sondage (elles partagent le terrain, cf. session initiale : <1pt d'écart
    entre hypothèses d'un même sondage), PAS une nouvelle observation
    indépendante par hypothèse."""
    return (df.groupby(["notice", "institut", "date_fin", "echantillon", "candidat"])["intention"]
             .mean().reset_index())


def pairwise_check(df: pd.DataFrame, max_gap_days: int) -> pd.DataFrame:
    rows = build_one_row_per_poll(df)
    rows["date_fin"] = pd.to_datetime(rows["date_fin"])
    out = []
    for cand, g in rows.groupby("candidat"):
        g = g.sort_values("date_fin").reset_index(drop=True)
        for i, j in combinations(range(len(g)), 2):
            a, b = g.iloc[i], g.iloc[j]
            if a["notice"] == b["notice"] or a["institut"] == b["institut"]:
                continue   # même institut : pas indépendant (méthodo partagée), on veut du croisé
            gap = abs((a["date_fin"] - b["date_fin"]).days)
            if gap > max_gap_days:
                continue
            pa, na = a["intention"], a["echantillon"]
            pb, nb = b["intention"], b["echantillon"]
            var = pa * (100 - pa) / max(na, 1) + pb * (100 - pb) / max(nb, 1)
            if var <= 0:
                continue
            z = (pa - pb) / np.sqrt(var)
            out.append(dict(candidat=cand, gap_days=gap, institut_a=a["institut"], institut_b=b["institut"],
                            date_a=a["date_fin"], date_b=b["date_fin"], pa=pa, pb=pb, z=z))
    return pd.DataFrame(out)


def coverage_report(z: np.ndarray) -> dict:
    levels = {"50%": 0.674, "80%": 1.282, "90%": 1.645}
    return {k: float(np.mean(np.abs(z) <= q)) for k, q in levels.items()}


def main():
    raw = load_raw_polls()
    raw2027 = raw[pd.to_datetime(raw["date_fin"]) >= MIN_POLL_DATE]
    pairs = pairwise_check(raw2027, MAX_GAP_DAYS)
    print(f"2027 (>= {MIN_POLL_DATE.date()}, écart <= {MAX_GAP_DAYS}j, instituts différents) : "
          f"{len(pairs)} paires sondage x sondage x candidat")
    if len(pairs):
        print(f"  RMS(z) = {np.sqrt(np.mean(pairs['z']**2)):.2f} (attendu ~1.0 si pi(1-pi)/N est réaliste)")
        cov = coverage_report(pairs["z"].to_numpy())
        for k, v in cov.items():
            print(f"  couverture |z|<=seuil {k} : {v*100:5.1f}% (nominal {k})")
        print("\n  Détail des paires avec gap=0 (même jour) :")
        same_day = pairs[pairs["gap_days"] == 0]
        print(same_day[["candidat", "institut_a", "pa", "institut_b", "pb", "z"]].to_string(index=False))

    # Même check sur 2017+2022 (bien plus de données, pour un résultat plus robuste)
    from pipeline.historical import ELECTION_DATES, PARSED_DIR, POLL_FILES, _resolve_candidate, load_blocs
    print("\n=== Même test sur 2017+2022 (plus de données) ===")
    all_hist_pairs = []
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
        hp = pairwise_check(t1, MAX_GAP_DAYS)
        hp["election"] = elec
        all_hist_pairs.append(hp)
    hist_pairs = pd.concat(all_hist_pairs, ignore_index=True)
    print(f"{len(hist_pairs)} paires (2017+2022 combinés)")
    if len(hist_pairs):
        print(f"  RMS(z) = {np.sqrt(np.mean(hist_pairs['z']**2)):.2f} (attendu ~1.0)")
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


if __name__ == "__main__":
    main()
