"""Reports de voix T1 -> T2 publiés par les instituts, relevés par Wikipedia
(section « Reports de voix » des pages 2017 et 2022).

Une ligne = (sondage, électorat du 1er tour, option du 2nd tour) :

    notice, institut, date_debut, date_fin, echantillon, hypothese,
    electorat_t1, candidat_t2, part, base_electorat, rolling, source

- `part` : % de l'électorat T1 qui se reporte sur `candidat_t2`. Les options
  d'un électorat somment à ~100 (contrôlé, tolérance SUM_LO/SUM_HI) ;
- `candidat_t2 = "Aucun"` : abstention, blanc ou nul (libellé Wikipedia, non
  détaillé) ;
- `echantillon` : taille de l'échantillon total, reprise de la ligne
  d'intentions de 2nd tour de la même notice quand elle existe (la base de
  l'électorat n'est pas publiée par Wikipedia : `base_electorat` vide) ;
- `rolling` : vague d'une enquête glissante (non indépendante des voisines).

Pas de 2027 : il n'y a de reports observés qu'après un 1er tour.

    python -m sondages.reports --annee 2017   # -> data/parsed/reports_t2_2017.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from sondages import wiki
from sondages.parse_pdf import detect_institut

ROOT = Path(__file__).resolve().parent.parent
PAGE = "Liste de sondages sur l'élection présidentielle française de {annee}"
SUM_LO, SUM_HI = 97.0, 103.0
COLUMNS = ["notice", "institut", "date_debut", "date_fin", "echantillon", "hypothese",
           "electorat_t1", "candidat_t2", "part", "base_electorat", "rolling", "source"]
_PCT = re.compile(r"(\d{1,3}(?:[.,]\d+)?)\s*%?")


def _nom(cell: dict) -> str:
    return cell["linked_names"][0] if cell["linked_names"] else cell["text"].split("(")[0].strip()


def extract_reports(grid: list[list[dict]], annee: str, warnings: list[str]) -> list[dict]:
    """Grille d'un tableau « Reports de voix » -> enregistrements."""
    entete = grid[0]
    i_opt = next(i for i, row in enumerate(grid) if any(c["text"] == "Aucun" for c in row))
    options = grid[i_opt]
    n_meta = 2  # Sondeur, Date
    colonnes = [(_nom(e), _nom(o)) for e, o in zip(entete[n_meta:], options[n_meta:])]
    finalistes = list(dict.fromkeys(o for _, o in colonnes if o != "Aucun"))
    hypothese = " – ".join(finalistes)

    out = []
    for row in grid[i_opt + 1:]:
        sondeur = row[0]
        institut = detect_institut(sondeur["text"]) if sondeur["text"] else None
        if institut is None:
            continue  # ligne de couleurs
        date_debut, date_fin = wiki._parse_dates(row[1]["text"], annee)
        notice = f"{institut} {row[1]['text']} (WP)"
        if date_fin is None:
            warnings.append(f"date illisible {row[1]['text']!r} ({notice}) -> sans date")
        for (electorat, option), cell in zip(colonnes, row[n_meta:]):
            m = _PCT.search(cell["text"])
            if not m:
                continue
            out.append({
                "notice": notice, "institut": institut,
                "date_debut": date_debut, "date_fin": date_fin, "echantillon": None,
                "hypothese": hypothese, "electorat_t1": electorat, "candidat_t2": option,
                "part": float(m.group(1).replace(",", ".")), "base_electorat": None,
                "rolling": "rolling" in sondeur["text"].lower(), "source": sondeur["url"],
            })
    return out


def build_reports(annee: str, intentions_csv: Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    page = PAGE.format(annee=annee)
    sections = wiki.fetch_sections(page)
    warnings: list[str] = []
    records: list[dict] = []
    for s in sections:
        if "reports de voix" not in s["line"].lower():
            continue
        for grid in wiki.parse_wikitables(wiki.fetch_section_wikitext(page, s["index"])):
            records.extend(extract_reports(grid, annee, warnings))
    df = pd.DataFrame(records, columns=COLUMNS)
    if df.empty:
        return df, warnings

    df = wiki.desambiguise_notices(df.rename(columns={"source": "notice_url"})).rename(
        columns={"notice_url": "source"})

    # échantillon : repris des intentions de 2nd tour de la même notice
    intentions_csv = intentions_csv or ROOT / "data" / "parsed" / f"intentions_{annee}_wiki.csv"
    if intentions_csv.exists():
        it = pd.read_csv(intentions_csv)
        t2 = it[it["tour"] == "Deuxième tour"].dropna(subset=["echantillon"])
        par_url = t2.groupby("notice_url")["echantillon"].first()
        # repli (institut, date_fin) quand l'URL diffère entre les deux
        # tableaux de la page, seulement si l'échantillon y est unique
        uniq = t2.groupby(["institut", "date_fin"])["echantillon"].agg(
            lambda s: s.iloc[0] if s.nunique() == 1 else None)
        par_date = pd.Series([uniq.get((i, d)) for i, d in zip(df["institut"], df["date_fin"])],
                             index=df.index)
        df["echantillon"] = df["source"].map(par_url).fillna(par_date).astype("Int64")

    sommes = df.groupby(["notice", "electorat_t1"])["part"].sum()
    hors = sommes[(sommes < SUM_LO) | (sommes > SUM_HI)]
    for (notice, electorat), total in hors.items():
        warnings.append(f"somme {total:g} % hors [{SUM_LO:g}, {SUM_HI:g}] : {notice} / {electorat} -> écarté")
    if len(hors):
        cle = df.set_index(["notice", "electorat_t1"]).index
        df = df[~cle.isin(hors.index)].reset_index(drop=True)
    return df[COLUMNS], warnings


def main():
    ap = argparse.ArgumentParser(description="Reports de voix T1 -> T2 (Wikipedia)")
    ap.add_argument("--annee", required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    df, warnings = build_reports(a.annee)
    for w in warnings:
        print("ATTENTION :", w)
    out = a.out or ROOT / "data" / "parsed" / f"reports_t2_{a.annee}.csv"
    df.to_csv(out, index=False)
    print(f"{out} : {len(df)} lignes, {df['notice'].nunique()} sondages, "
          f"{df['electorat_t1'].nunique()} électorats T1")


if __name__ == "__main__":
    main()
