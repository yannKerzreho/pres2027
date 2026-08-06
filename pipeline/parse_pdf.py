"""Phase 2 — extraction des intentions de vote depuis les notices PDF.

C'est le chaînon manquant public pour 2027 (et le risque technique n°1 : les
notices sont hétérogènes selon l'institut). Architecture :

  - `detect_institut()`  : identifie l'institut depuis le nom de la notice ;
  - un parseur **par institut** (pour l'instant : Ifop, le plus fréquent et le
    plus régulier) qui renvoie des enregistrements normalisés ;
  - un **fallback** qui, pour un institut non encore outillé, ne casse pas le
    pipeline mais renvoie un statut « revue manuelle » (conforme au §5.2 du plan).

Pièges gérés explicitement :
  - les pages de **rappel** de vote passé (« Reconstitution du 1er tour 2022 »,
    européennes/législatives 2024) ressemblent à des intentions mais n'en sont
    pas — on les ignore ;
  - toutes les notices « pres » ne contiennent pas d'intentions (certaines sont
    des sondages d'opinion) — statut `no_intentions` au lieu d'une erreur.

Sortie normalisée par enregistrement : institut, commanditaire, date_debut,
date_fin, echantillon, methode, tour, hypothese, candidat, intention (%).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pdfplumber

# --- Détection de l'institut -------------------------------------------------
INSTITUT_PATTERNS = [
    ("Ifop", ("ifop",)),
    ("Opinion Way", ("opinionway", "opinion way", "opinion-way")),
    ("Harris Interactive", ("harris", "toluna")),
    ("Elabe", ("elabe",)),
    ("Ipsos", ("ipsos",)),
    ("Odoxa", ("odoxa",)),
    ("BVA", ("bva",)),
    ("Cluster17", ("cluster",)),
    ("Kantar Public", ("kantar",)),
]

# Libellés de lignes à exclure des tables d'intentions.
_NON_CANDIDATE = re.compile(
    r"^(total|base|sous-total|ne se prononce|résultats|bruts?|redress|"
    r"vote au|reconstitution|hypoth|question)", re.IGNORECASE
)


def detect_institut(name: str) -> str | None:
    s = (name or "").lower()
    for institut, keys in INSTITUT_PATTERNS:
        if any(k in s for k in keys):
            return institut
    return None


@dataclass
class Notice:
    institut: str | None
    source_name: str
    status: str = "ok"                 # ok | no_intentions | unsupported_institut | error
    commanditaire: str | None = None
    date_debut: str | None = None
    date_fin: str | None = None
    echantillon: int | None = None
    methode: str | None = None
    records: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --- Utilitaires -------------------------------------------------------------
def _clean_name(raw: str) -> str:
    s = raw.replace("•", " ")
    s = re.sub(r"\.{2,}", " ", s)           # points de conduite «......»
    s = re.sub(r"\s+", " ", s).strip(" .\t")
    return s


def _parse_pct(raw: str) -> float | None:
    if raw is None:
        return None
    m = re.search(r"(\d{1,3}(?:[.,]\d+)?)", str(raw))
    if not m:
        return None
    val = float(m.group(1).replace(",", "."))
    return val if 0 <= val <= 100 else None


def _looks_like_candidate(name: str) -> bool:
    if not name or _NON_CANDIDATE.match(name):
        return False
    if re.search(r"\d", name) or "," in name or len(name) > 40:
        return False  # exclut croisements démographiques et listes de régions
    words = [w for w in name.split() if re.match(r"[A-Za-zÀ-ÿ]", w)]
    if any(w.isupper() and len(w) >= 2 for w in words):
        return False  # exclut les acronymes de région (PACA, IDF, ...)
    # un nom de candidat est un nom propre : >= 2 mots dont les deux premiers sont
    # capitalisés (« Nathalie Arthaud », « Marine Le Pen »). Écarte les libellés
    # socio-pro (« Profession intermédiaire », « Autre inactif ») dont le 2e mot
    # est en minuscule.
    return len(words) >= 2 and words[0][:1].isupper() and words[1][:1].isupper()


# --- Parseur Ifop ------------------------------------------------------------
_MONTHS = {m: i for i, m in enumerate(
    ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
     "septembre", "octobre", "novembre", "décembre"], start=1)}


def _iso(day: str, month: str, year: str) -> str | None:
    month = unicodedata.normalize("NFC", month.lower())
    mnum = _MONTHS.get(month)
    return f"{year}-{mnum:02d}-{int(day):02d}" if mnum else None


def _ifop_metadata(full_text: str, notice: Notice) -> None:
    # commanditaire
    m = re.search(r"réalisée par\s+Ifop\s+pour\s+(.+?)(?:\n|Echantillon|$)", full_text, re.I)
    if m:
        notice.commanditaire = m.group(1).strip(" .")
    # échantillon (personnes inscrites de préférence)
    m = re.search(r"échantillon de\s*([\d  ]+?)\s*personnes", full_text, re.I)
    if m:
        notice.echantillon = int(re.sub(r"\D", "", m.group(1)))
    # mode de recueil
    if re.search(r"en ligne|internet|auto-administré", full_text, re.I):
        notice.methode = "internet"
    elif re.search(r"téléphone|téléphonique", full_text, re.I):
        notice.methode = "téléphone"
    # dates : « du 26 au 27 mars 2025 » (ou « le 31 mars 2025 »)
    m = re.search(r"du\s+(\d{1,2})\s+au\s+(\d{1,2})\s+([A-Za-zÀ-ÿ]+)\s+(\d{4})", full_text, re.I)
    if m:
        notice.date_debut = _iso(m.group(1), m.group(3), m.group(4))
        notice.date_fin = _iso(m.group(2), m.group(3), m.group(4))
    else:
        m = re.search(r"\ble\s+(\d{1,2})\s+([A-Za-zÀ-ÿ]+)\s+(\d{4})", full_text, re.I)
        if m:
            notice.date_debut = notice.date_fin = _iso(m.group(1), m.group(2), m.group(3))


def _ifop_hypothese(title: str) -> str | None:
    m = re.search(r"Hypoth[èe]se\s+(.+)", title.replace("\n", " "), re.I)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None


def parse_ifop(pdf, notice: Notice) -> None:
    full_text = "\n".join(pg.extract_text() or "" for pg in pdf.pages)
    _ifop_metadata(full_text, notice)

    for pg in pdf.pages:
        t = pg.extract_text() or ""
        title = t.split("Question")[0]
        low = title.lower()
        # page d'intentions réelle (et pas une page de rappel de vote passé)
        if "intentions de vote" not in low or "reconstitution" in low:
            continue
        tour = "Deuxième tour" if "second tour" in low or "deuxième tour" in low else "Premier tour"
        hypothese = _ifop_hypothese(title)
        for tbl in pg.extract_tables():
            for row in tbl:
                if not row or len(row) < 2:
                    continue
                name = _clean_name(row[0] or "")
                pct = _parse_pct(row[-1])
                if pct is not None and _looks_like_candidate(name):
                    notice.records.append({
                        "tour": tour, "hypothese": hypothese,
                        "candidat": name, "intention": pct,
                    })


# --- Helpers partagés pour parsers basés sur le texte -----------------------
_MARGIN = re.compile(r"\+/-\s*\d+(?:[.,]\d+)?\s*points?", re.I)
_HYP = re.compile(r"Hypoth[èe]se\s*:?\s*(.+)", re.I)
_NAME = r"[A-ZÀ-Ÿ][A-Za-zÀ-ÿ'’.\-]*(?:\s+[A-Za-zÀ-ÿ'’.\-]+){1,3}?"
_PAIR_PCT = re.compile(rf"({_NAME})\s+(\d{{1,3}}(?:,\d+)?)\s*%")
_PAIR_INT = re.compile(rf"({_NAME})\s+(\d{{1,2}}(?:,\d+)?)(?!\d|,\d|%)")


def _percent_pairs(text: str) -> list[tuple[str, float]]:
    """(candidat, %) sur un texte où les scores sont suffixés de « % ».
    Robuste aux marges d'erreur « +/- 1,9 point » (retirées d'abord)."""
    text = _MARGIN.sub(" ", text.replace("\n", " "))
    out = []
    for m in _PAIR_PCT.finditer(text):
        name, pct = _clean_name(m.group(1)), _parse_pct(m.group(2))
        if pct is not None and _looks_like_candidate(name):
            out.append((name, pct))
    return out


def _hypothese(page_text: str) -> str | None:
    m = _HYP.search(page_text.split("Question")[0].split("Si le")[0])
    return re.sub(r"\s+", " ", m.group(1)).strip(" .:") if m else None


def _tour(page_text: str) -> str:
    low = page_text[:400].lower()
    return "Deuxième tour" if ("second tour" in low or "2nd tour" in low
                               or "2 nd tour" in low or "deuxième tour" in low) else "Premier tour"


def _generic_metadata(full_text: str, notice: Notice) -> None:
    m = re.search(r"pour\s+([A-ZÀ-Ÿ][\w'’ .\-]+?)(?:\n|,|\.|Echantillon|»)", full_text)
    if m:
        notice.commanditaire = m.group(1).strip()
    m = re.search(r"(\d[\d  ]{2,5}\d)\s*(?:personnes|répondants|inscrits)", full_text, re.I)
    if m:
        notice.echantillon = int(re.sub(r"\D", "", m.group(1)))
    if re.search(r"en ligne|internet|auto-administré|online", full_text, re.I):
        notice.methode = "internet"
    elif re.search(r"téléphone|téléphonique", full_text, re.I):
        notice.methode = "téléphone"
    m = re.search(r"du\s+(\d{1,2})\s+au\s+(\d{1,2})\s+([A-Za-zÀ-ÿ]+)\s+(\d{4})", full_text, re.I)
    if m:
        notice.date_debut = _iso(m.group(1), m.group(3), m.group(4))
        notice.date_fin = _iso(m.group(2), m.group(3), m.group(4))


def _is_intention_page(t: str) -> bool:
    low = t[:600].lower()
    return ("intentions de vote" in low
            and "hypoth" in low
            and not any(b in low for b in ("mémoire", "reconstitution", "répartition")))


# --- Parseur Odoxa (1 hypothèse/page, lignes « Nom 12% ») -------------------
def parse_odoxa(pdf, notice: Notice) -> None:
    full = "\n".join(pg.extract_text() or "" for pg in pdf.pages)
    _generic_metadata(full, notice)
    for pg in pdf.pages:
        t = pg.extract_text() or ""
        if not _is_intention_page(t):
            continue
        hyp, tour = _hypothese(t), _tour(t)
        for name, pct in _percent_pairs(t):
            notice.records.append({"tour": tour, "hypothese": hyp,
                                   "candidat": name, "intention": pct})


# --- Parseur Ipsos (1 hypothèse/page, « Nom 12% +/- marge ») ----------------
def parse_ipsos(pdf, notice: Notice) -> None:
    full = "\n".join(pg.extract_text() or "" for pg in pdf.pages)
    _generic_metadata(full, notice)
    for pg in pdf.pages:
        t = pg.extract_text() or ""
        if not _is_intention_page(t):
            continue
        hyp, tour = _hypothese(t), _tour(t)
        for name, pct in _percent_pairs(t):
            notice.records.append({"tour": tour, "hypothese": hyp,
                                   "candidat": name, "intention": pct})


# --- Parseur Harris/Toluna (récapitulatif, N hypothèses côte à côte) --------
def parse_harris(pdf, notice: Notice) -> None:
    full = "\n".join(pg.extract_text() or "" for pg in pdf.pages)
    _generic_metadata(full, notice)
    for pg in pdf.pages:
        t = pg.extract_text() or ""
        low = t[:120].lower()
        if "récapitulatif" not in low or "intentions de vote" not in low:
            continue
        tour = _tour(t)
        lines = t.splitlines()
        # étiquettes d'hypothèses : nom(s) propre(s) sur les lignes d'en-tête,
        # avant la 1re ligne de données (une ligne de données finit par un nombre).
        col_labels: list[str] = []
        cols: dict[int, list] = {}
        started = False
        for ln in lines:
            pairs = _PAIR_INT.findall(ln)
            names_only = re.findall(_NAME, ln)
            if not pairs and not started and "hypoth" not in ln.lower() and names_only:
                col_labels = [_clean_name(x) for x in names_only]  # ligne d'en-tête
            if pairs:
                started = True
                for ci, (name, val) in enumerate(pairs):
                    name = _clean_name(name)
                    v = _parse_pct(val)
                    if v is not None and _looks_like_candidate(name):
                        cols.setdefault(ci, []).append((name, v))
        for ci, recs in cols.items():
            hyp = col_labels[ci] if ci < len(col_labels) else f"hypothèse {ci + 1}"
            for name, v in recs:
                notice.records.append({"tour": tour, "hypothese": hyp,
                                       "candidat": name, "intention": v})


PARSERS = {
    "Ifop": parse_ifop,
    "Odoxa": parse_odoxa,
    "Ipsos": parse_ipsos,
    "Harris Interactive": parse_harris,
}


def _validate_records(notice: Notice, lo: float = 90.0, hi: float = 110.0) -> None:
    """Garde-fou qualité : un bulletin somme à ~100 %. On ne conserve que les
    groupes (tour × hypothèse) dont la somme des intentions est plausible ; cela
    élimine les mauvaises captures (récapitulatifs de duels, croisements, pages
    de rappel passées entre les mailles)."""
    groups: dict[tuple, list] = {}
    for r in notice.records:
        groups.setdefault((r["tour"], r["hypothese"]), []).append(r)
    kept = []
    for recs in groups.values():
        total = sum(r["intention"] for r in recs)
        if lo <= total <= hi:
            kept.append(recs)
    notice.records = [r for recs in kept for r in recs]


def parse_notice(pdf_path: str | Path, source_name: str | None = None) -> Notice:
    """Parse une notice PDF -> Notice normalisée (ne lève pas : statut dans .status)."""
    pdf_path = Path(pdf_path)
    name = source_name or pdf_path.stem
    institut = detect_institut(name)
    notice = Notice(institut=institut, source_name=name)

    if institut not in PARSERS:
        notice.status = "unsupported_institut"
        return notice

    try:
        with pdfplumber.open(pdf_path) as pdf:
            PARSERS[institut](pdf, notice)
    except Exception as exc:  # PDF illisible / corrompu
        notice.status = "error"
        notice.commanditaire = notice.commanditaire or f"(erreur: {exc})"
        return notice

    _validate_records(notice)
    if not notice.records:
        notice.status = "no_intentions"
    return notice


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        n = parse_notice(path)
        print(f"\n### {Path(path).name}\n  institut={n.institut} statut={n.status} "
              f"commanditaire={n.commanditaire} dates={n.date_debut}->{n.date_fin} "
              f"n={n.echantillon} methode={n.methode} | {len(n.records)} intentions")
        hyps = {}
        for r in n.records:
            hyps.setdefault(r["hypothese"], []).append(r)
        for h, rs in hyps.items():
            print(f"  · Hypothèse: {h}  ({len(rs)} candidats)")
            for r in rs[:4]:
                print(f"      {r['candidat']:<24} {r['intention']}")
