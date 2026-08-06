"""Phase 2 — pipeline live de bout en bout : index -> PDF -> intentions structurées.

Enchaîne l'ingestion (`ingest.py`) et le parsing (`parse_pdf.py`) pour produire
le jeu de sondages 2027 structuré (`data/parsed/intentions_2027.csv`) + un
rapport de couverture par institut et par statut. Les notices non parsables
(institut non outillé, ou sans table d'intentions) sont listées pour revue,
plutôt que de bloquer le pipeline.

Exécution : .venv/bin/python pipeline/build_live_dataset.py --since 2025 --limit 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline.ingest import fetch_index, filter_presidential, download_notice
from pipeline.parse_pdf import parse_notice

PARSED_DIR = Path(__file__).resolve().parent.parent / "data" / "parsed"


def build(since_year: int = 2025, limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = filter_presidential(fetch_index(), since_year)
    if limit:
        idx = idx.head(limit)

    records, statuses = [], []
    for _, row in idx.iterrows():
        pdf = download_notice(row)
        if not pdf:
            statuses.append({"notice": row["name"], "institut": None, "status": "download_failed"})
            continue
        n = parse_notice(pdf, row["name"])
        statuses.append({"notice": row["name"], "institut": n.institut,
                         "status": n.status, "n_records": len(n.records)})
        # date de la notice (index NSPPolls) : repli fiable si le parseur n'a pas
        # extrait de date (ex. Harris), pour calculer l'horizon côté modèle.
        date_notice = str(row.get("pdf_creation_date", ""))[:10] or None
        for r in n.records:
            records.append({
                "notice": row["name"], "institut": n.institut,
                "commanditaire": n.commanditaire, "date_debut": n.date_debut,
                "date_fin": n.date_fin or date_notice, "date_notice": date_notice,
                "echantillon": n.echantillon,
                "methode": n.methode, "tour": r["tour"], "hypothese": r["hypothese"],
                "candidat": r["candidat"], "intention": r["intention"],
            })

    return pd.DataFrame(records), pd.DataFrame(statuses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2025)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    intentions, status = build(args.since, args.limit)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PARSED_DIR / "intentions_2027.csv"
    intentions.to_csv(out, index=False)

    print(f"\n=== Couverture ({len(status)} notices) ===")
    print(status.groupby("status").size().to_string())
    if not intentions.empty:
        print(f"\nIntentions extraites : {len(intentions)} lignes, "
              f"{intentions['notice'].nunique()} sondages, "
              f"{intentions['candidat'].nunique()} candidats.")
        parsed_ok = status[status.status == "ok"]
        print("\nSondages parsés par institut :")
        print(parsed_ok.groupby("institut").size().to_string())
    print(f"\nÉcrit : {out}")

    # notices à outiller (institut détecté mais pas encore de parseur)
    todo = status[status.status == "unsupported_institut"]
    if not todo.empty:
        print("\nInstituts à outiller (revue/règles à écrire) :")
        print(todo.groupby("institut").size().to_string())


if __name__ == "__main__":
    main()
