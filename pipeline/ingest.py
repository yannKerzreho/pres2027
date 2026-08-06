"""Phase 2 — ingestion live depuis `sondages-commission-index`.

On ne re-scrape PAS la Commission des sondages : `nsppolls/sondages-commission-index`
(Codeberg, mirroré GitHub) fait déjà l'indexation et fournit `base.csv`
(liste des notices) et `files.csv` (métadonnées + URL des PDF). On consomme cet
index, on filtre les notices présidentielles récentes, et on télécharge les PDF
(avec cache local) pour les passer au parseur (`parse_pdf.py`).

Crédit : données NSPPolls (MIT) — interagir sur Codeberg, pas sur le mirroir
GitHub. Les notices sont des documents publics de la Commission des sondages.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import requests

RAW_BASE = "https://raw.githubusercontent.com/nsppolls/sondages-commission-index/main"
PDF_CACHE = Path(__file__).resolve().parent.parent / "data" / "raw" / "pdf"

# Mots-clés indiquant une notice présidentielle (dans le champ `name`).
PRES_KEYS = ("pres", "présid", "presid")


def fetch_index() -> pd.DataFrame:
    """Récupère et fusionne base.csv + files.csv depuis l'index NSPPolls.

    Renvoie une ligne par PDF : name, url (commission), path_local, filename,
    pdf_creation_date, year (déduite du chemin d'archive).
    """
    files = pd.read_csv(io.StringIO(requests.get(f"{RAW_BASE}/files.csv", timeout=60).text))
    files = files.rename(columns={"pdf creation-date": "pdf_creation_date"})
    # année depuis path_local (archives/AAAA/mois/)
    files["year"] = (
        files["path_local"].str.extract(r"archives/(\d{4})/").astype("Int64")
    )
    return files


def filter_presidential(df: pd.DataFrame, since_year: int = 2025) -> pd.DataFrame:
    """Notices présidentielles à partir d'une année (défaut : cycle 2027)."""
    name_l = df["name"].fillna("").str.lower()
    is_pres = name_l.apply(lambda s: any(k in s for k in PRES_KEYS))
    recent = df["year"].fillna(0) >= since_year
    return df[is_pres & recent].reset_index(drop=True)


def download_notice(row, cache_dir: Path = PDF_CACHE, timeout: int = 60) -> Path | None:
    """Télécharge un PDF de notice (avec cache). Renvoie le chemin local ou None."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / row["filename"]
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        r = requests.get(row["url"], timeout=timeout)
        r.raise_for_status()
        if not r.content[:4] == b"%PDF":
            return None  # pas un PDF (page d'erreur, LFS pointer, ...)
        dest.write_bytes(r.content)
        return dest
    except requests.RequestException:
        return None


def sync(since_year: int = 2025, limit: int | None = None) -> pd.DataFrame:
    """Télécharge les notices présidentielles récentes. Renvoie l'index filtré
    avec une colonne `local_path` (chemin PDF ou vide si échec)."""
    idx = filter_presidential(fetch_index(), since_year)
    if limit:
        idx = idx.head(limit)
    paths = []
    for _, row in idx.iterrows():
        p = download_notice(row)
        paths.append(str(p) if p else "")
    idx = idx.copy()
    idx["local_path"] = paths
    ok = (idx["local_path"] != "").sum()
    print(f"{ok}/{len(idx)} notices téléchargées (depuis {since_year}).")
    return idx


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingestion notices présidentielles (index NSPPolls).")
    ap.add_argument("--since", type=int, default=2025, help="année minimale (défaut 2025)")
    ap.add_argument("--limit", type=int, default=None, help="nombre max de notices")
    args = ap.parse_args()
    df = sync(since_year=args.since, limit=args.limit)
    out = PDF_CACHE.parent / "notices_index.csv"
    df.to_csv(out, index=False)
    print(f"Index écrit dans {out}")
