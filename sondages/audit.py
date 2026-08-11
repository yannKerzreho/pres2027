"""Audit du parsing : mesure et rend VÉRIFIABLE la couverture des sondages.

Pour chaque notice présidentielle d'un institut outillé, on compare deux choses
indépendantes :
  - un **détecteur** (heuristique, indépendant du parser) qui dit si la notice
    contient une vraie table d'intentions 2027 ;
  - le **parser** de production (status + nb de candidats + somme ~100 %).

On classe alors chaque notice :
  - `ok`       : détectée ET parsée correctement ;
  - `MISS`     : détectée mais le parser n'a rien sorti  → **faux négatif à corriger** ;
  - `SUSPECT`  : parsée mais somme/candidats douteux      → **à vérifier** ;
  - `hors`     : pas d'intentions (baromètre/opinion), correctement ignorée.

Recall d'un institut = ok / (ok + MISS). Objectif : ~100 % pour Ifop/Odoxa/
Ipsos/Harris. La liste des `MISS` est la todo de robustesse du parsing.

Usage : python -m sondages.audit [--since 2025] [--json audit.json]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import pdfplumber

from sondages.ingest import fetch_index, filter_presidential, download_notice
from sondages.parse_pdf import parse_notice, detect_institut, PARSERS

# Candidats SPÉCIFIQUES 2027 (jamais candidats en 2022) → évite le piège des
# tables de rappel 2022 qui ressemblent à des intentions.
_A2027 = ["Bardella", "Attal", "Faure", "Tondelier", "Glucksmann", "Retailleau",
          "Wauquiez", "Darmanin", "Ruffin", "Bellamy", "Maréchal", "Knafo", "Lecornu"]
_BAD = ("mémoire", "reconstitution", "image", "confiance", "soutenez", "apprécie",
        "ambition", "pronostic", "souhait")


def detector_has_intentions(pdf) -> bool:
    """Vrai si une page contient une table d'intentions 2027 (indépendant du parser)."""
    for pg in pdf.pages:
        t = pg.extract_text() or ""
        tl = t.lower()
        if (sum(a in t for a in _A2027) >= 4 and "exprim" in tl
                and "intention" in tl and not any(b in tl for b in _BAD)):
            return True
    return False


def audit(since_year: int = 2025) -> list[dict]:
    idx = filter_presidential(fetch_index(), since_year)
    idx = idx.copy()
    idx["institut"] = idx["name"].apply(detect_institut)
    idx = idx[idx["institut"].isin(PARSERS)]            # instituts outillés seulement

    rows = []
    for _, r in idx.iterrows():
        p = download_notice(r)
        if not p:
            rows.append({"institut": r["institut"], "notice": r["name"],
                         "classe": "download_failed"})
            continue
        try:
            with pdfplumber.open(p) as pdf:
                detected = detector_has_intentions(pdf)
        except Exception:
            detected = False
        n = parse_notice(p, r["name"])
        n_records = len(n.records)
        n_hyp = len({rec["hypothese"] for rec in n.records}) if n.records else 0

        if n.status == "biased_subsample":
            classe = "hors"                              # sous-électorat : écarté à dessein
        elif n_records:
            # cohérence des sommes déjà garantie par le parser ; on marque SUSPECT
            # si trop peu de candidats (< 8) pour un 1er tour
            per_hyp = defaultdict(int)
            for rec in n.records:
                per_hyp[rec["hypothese"]] += 1
            classe = "ok" if max(per_hyp.values()) >= 8 else "SUSPECT"
        elif detected:
            classe = "MISS"                              # faux négatif : à corriger
        else:
            classe = "hors"                              # pas d'intentions, OK

        rows.append({"institut": r["institut"], "notice": r["name"][:55],
                     "detected": detected, "status": n.status,
                     "n_hyp": n_hyp, "n_records": n_records, "classe": classe})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2025)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = audit(args.since)

    # récap par institut
    by_inst = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_inst[r["institut"]][r["classe"]] += 1
    print(f"{'institut':<20} | {'ok':>3} {'MISS':>4} {'SUSP':>4} {'hors':>4} | recall")
    print("-" * 60)
    tot_ok = tot_miss = 0
    for inst in sorted(by_inst):
        c = by_inst[inst]
        ok, miss = c["ok"], c["MISS"]
        tot_ok += ok; tot_miss += miss
        rec = ok / (ok + miss) if (ok + miss) else 1.0
        print(f"{inst:<20} | {ok:>3} {miss:>4} {c['SUSPECT']:>4} {c['hors']:>4} | {rec:.0%}")
    glob = tot_ok / (tot_ok + tot_miss) if (tot_ok + tot_miss) else 1.0
    print("-" * 60)
    print(f"{'TOTAL':<20} | {tot_ok:>3} {tot_miss:>4} {'':>4} {'':>4} | recall global {glob:.0%}")

    misses = [r for r in rows if r["classe"] in ("MISS", "SUSPECT")]
    if misses:
        print("\n=== À corriger (MISS/SUSPECT) ===")
        for r in misses:
            print(f"  [{r['classe']:<7}] {r['institut']:<18} {r['notice']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"\nDétail écrit dans {args.json}")


if __name__ == "__main__":
    main()
