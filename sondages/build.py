"""Pipeline de bout en bout : index NSPPolls -> PDF -> intentions structurées.

Enchaîne l'ingestion (`ingest`) et le parsing (`parse_pdf`) pour produire le jeu
de sondages structuré (contrat `schema.COLUMNS`, niveau candidat, avec
`notice_url` pour la traçabilité) + un rapport de couverture. Les notices non
parsables sont listées pour revue plutôt que de bloquer le pipeline.

Exécution : python -m sondages.build --since 2025 [--limit N]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from sondages.ingest import fetch_index, filter_presidential, download_notice
from sondages.parse_pdf import parse_notice
from sondages import schema

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "parsed" / "intentions_2027.csv"


def _notice_id(name: str) -> str | None:
    m = re.match(r"\s*(\d+)", str(name))
    return m.group(1) if m else None


def build_dataset(since_year: int = 2025, limit: int | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (intentions, statuts). `intentions` respecte `schema.COLUMNS`."""
    idx = filter_presidential(fetch_index(), since_year)
    if limit:
        idx = idx.head(limit)

    records, statuses = [], []
    for _, row in idx.iterrows():
        pdf = download_notice(row)
        if not pdf:
            statuses.append({"notice": row["name"], "institut": None,
                             "status": "download_failed", "n_records": 0})
            continue
        n = parse_notice(pdf, row["name"])
        statuses.append({"notice": row["name"], "institut": n.institut,
                         "status": n.status, "n_records": len(n.records)})
        date_notice = str(row.get("pdf_creation_date", ""))[:10] or None
        for r in n.records:
            records.append({
                "notice_id": _notice_id(row["name"]),
                "notice": row["name"],
                "notice_url": row.get("url"),
                "institut": n.institut,
                "commanditaire": n.commanditaire,
                "date_debut": n.date_debut,
                "date_fin": n.date_fin or date_notice,
                "date_notice": date_notice,
                "echantillon": n.echantillon,
                "methode": n.methode,
                "tour": r["tour"],
                "hypothese": r["hypothese"],
                "candidat": r["candidat"],
                "intention": r["intention"],
            })

    intentions = (pd.DataFrame(records)[list(schema.COLUMNS)] if records
                  else schema.empty())
    return schema.validate(intentions), pd.DataFrame(statuses)


# alias de compat
build = build_dataset


def write_dataset(intentions: pd.DataFrame, out: Path = DEFAULT_OUT) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    intentions.to_csv(out, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2025)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    intentions, status = build_dataset(args.since, args.limit)
    out = write_dataset(intentions)

    print(f"\n=== Couverture ({len(status)} notices) ===")
    print(status.groupby("status").size().to_string())
    if not intentions.empty:
        print(f"\nIntentions extraites : {len(intentions)} lignes, "
              f"{intentions['notice'].nunique()} sondages, "
              f"{intentions['candidat'].nunique()} candidats.")
        print("\nSondages parsés par institut :")
        print(status[status.status == "ok"].groupby("institut").size().to_string())
    print(f"\nÉcrit : {out}")

    todo = status[status.status == "unsupported_institut"]
    if not todo.empty:
        print("\nInstituts à outiller :")
        print(todo.groupby("institut").size().to_string())


if __name__ == "__main__":
    main()
