"""Récupération VÉRIFIÉE des dates manquantes (typiquement le cycle 2017, où
le wikitext ne donne l'année nulle part pour la période « avant publication
de la liste officielle des candidats ») via l'index NSPPolls.

Principe : pour chaque notice Wikipedia sans date, on télécharge les notices
PDF candidates du MÊME institut sur le cycle électoral (index NSPPolls), on
les extrait (parseur par règles si l'institut est outillé, sinon fallback LLM
Haiku — cf. `extract_llm.py`), et on n'accepte la date que si les intentions
extraites REPRODUISENT celles de Wikipedia (mêmes candidats, mêmes % à
tolérance serrée). Une notice PDF qui ne matche AUCUNE cible, ou une cible
qui matche PLUSIEURS notices PDF, n'est jamais résolue par supposition — la
date reste manquante plutôt que fausse.

Limite connue : l'index NSPPolls ne remonte qu'à 2016 (rien avant) ; les
notices antérieures ne peuvent être recouvrées que si `notice_url` (donné par
Wikipedia) est lui-même un PDF téléchargeable (cf. `wiki.fill_missing_dates_from_pdf`).
"""

from __future__ import annotations

import unicodedata

import pandas as pd
import pdfplumber

from sondages import extract_llm
from sondages.ingest import fetch_index, filter_by_date_range, download_notice
from sondages.parse_pdf import parse_notice, detect_institut, PARSERS, _iso as _pdf_iso  # noqa: F401

MATCH_TOL = 1.0  # points : tolérance de comparaison intention Wikipedia vs PDF


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _norm_name(s: str) -> str:
    return _strip_accents(s.lower()).strip()


def _record_signature(records: list[dict]) -> dict[str, float]:
    return {_norm_name(r["candidat"]): r["intention"] for r in records}


def _wiki_target_signature(group: pd.DataFrame) -> dict[str, float]:
    return {_norm_name(c): v for c, v in zip(group["candidat"], group["intention"])}


def _matches(wiki_sig: dict, pdf_sig: dict, tol: float = MATCH_TOL) -> bool:
    """Vrai si CHAQUE candidat de la cible Wikipedia trouve, dans la notice PDF
    candidate, un candidat de nom apparenté (inclusion, gère « Le Pen » vs
    « Marine Le Pen ») avec une intention à ±tol — sans réutiliser deux fois le
    même candidat PDF (évite qu'un seul gros score fasse illusion sur tout)."""
    if not wiki_sig or not pdf_sig:
        return False
    used = set()
    for wname, wval in wiki_sig.items():
        hit = None
        for pname, pval in pdf_sig.items():
            if pname in used:
                continue
            if (wname in pname or pname in wname) and abs(wval - pval) <= tol:
                hit = pname
                break
        if hit is None:
            return False
        used.add(hit)
    return True


def _extract_candidate_pdf(path, institut: str, year: int, llm_calls: list) -> dict | None:
    """(date_debut, date_fin, records) depuis un PDF candidat NSPPolls."""
    if institut in PARSERS:
        n = parse_notice(path, institut)
        if n.status == "ok" and n.records and n.date_debut:
            return {"date_debut": n.date_debut, "date_fin": n.date_fin, "records": n.records}
        return None
    if not extract_llm.available():
        return None
    try:
        with pdfplumber.open(path) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception:
        return None
    llm_calls.append(1)
    result = extract_llm.extract_full_with_llm(text, institut, year=year)
    if not result or not result.get("date_debut") or not result["records"]:
        return None
    return result


def recover_dates(df: pd.DataFrame, year: int, date_start: str = "2016-01-01",
                  date_end: str | None = None) -> tuple[pd.DataFrame, dict]:
    """`df` = dataset Wikipedia (contrat `schema.COLUMNS`). Renvoie (df corrigé,
    rapport). Ne modifie QUE les lignes `date_debut` manquantes ; le reste est
    inchangé.

    `date_start`/`date_end` bornent le cycle électoral pour la recherche de
    notices candidates -> filtre par PLAGE DE DATES (`pdf_creation_date`), pas
    par mot-clé (`filter_presidential`) : ce dernier est fiable pour 2027 mais
    rate la quasi-totalité des notices 2016/2022 (convention de nommage
    différente, vérifié : 1 seul PDF trouvé pour TNS Sofres avec le mot-clé,
    contre une couverture bien plus large en plage de dates)."""
    date_end = date_end or f"{year}-04-23"
    df = df.copy()
    missing = df[df["date_debut"].isna()]
    report = {"n_resolved_rows": 0, "n_targets": 0, "n_pdf_downloaded": 0,
             "n_llm_calls": 0, "resolved": [], "ambiguous": []}
    if missing.empty:
        return df, report

    targets = (missing.groupby(["notice", "institut"])
              .apply(lambda g: _wiki_target_signature(g)).reset_index(name="signature"))
    report["n_targets"] = len(targets)

    idx = filter_by_date_range(fetch_index(), date_start, date_end)
    idx["institut_detecte"] = idx["name"].apply(detect_institut)

    llm_calls: list = []
    for institut in targets["institut"].unique():
        cand_idx = idx[idx["institut_detecte"] == institut]
        institut_targets = targets[targets["institut"] == institut]
        candidates = []
        for _, row in cand_idx.iterrows():
            path = download_notice(row)
            if not path:
                continue
            report["n_pdf_downloaded"] += 1
            extracted = _extract_candidate_pdf(path, institut, year, llm_calls)
            if extracted:
                candidates.append((extracted["date_debut"], extracted["date_fin"],
                                   _record_signature(extracted["records"])))

        for _, trow in institut_targets.iterrows():
            hits = [c for c in candidates if _matches(trow["signature"], c[2])]
            if len(hits) == 1:
                d1, d2, _ = hits[0]
                mask = (df["notice"] == trow["notice"]) & df["date_debut"].isna()
                n = int(mask.sum())
                df.loc[mask, "date_debut"] = d1
                df.loc[mask, "date_fin"] = d2
                report["n_resolved_rows"] += n
                report["resolved"].append((trow["notice"], d1, n))
            elif len(hits) > 1:
                report["ambiguous"].append((trow["notice"], len(hits)))

    report["n_llm_calls"] = len(llm_calls)
    return df, report


def main():
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="CSV du contrat schema.COLUMNS à corriger (in-place)")
    ap.add_argument("--year", type=int, required=True, help="année de l'élection (ex. 2017)")
    ap.add_argument("--date-start", default="2016-01-01")
    ap.add_argument("--date-end", default=None)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    n_before = df["date_debut"].isna().sum()
    fixed, report = recover_dates(df, args.year, args.date_start, args.date_end)
    fixed.to_csv(csv_path, index=False)

    print(f"Dates manquantes : {n_before} -> {fixed['date_debut'].isna().sum()} (sur {len(fixed)} lignes)")
    print(f"{report['n_targets']} notices ciblées, {report['n_pdf_downloaded']} PDF téléchargés, "
         f"{report['n_llm_calls']} appels LLM.")
    print(f"{len(report['resolved'])} résolues, {len(report['ambiguous'])} ambiguës (écartées) :")
    for notice, date, n in report["resolved"]:
        print(f"  OK   {notice} -> {date} ({n} lignes)")
    for notice, n_hits in report["ambiguous"]:
        print(f"  SKIP {notice} : {n_hits} correspondances possibles, aucune retenue")
    print(f"\nÉcrit : {csv_path}")


if __name__ == "__main__":
    main()
