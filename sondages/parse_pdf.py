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

import os
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
    # AVANT "Kantar Public" : l'institut s'est rebaptisé Kantar Sofres puis
    # Kantar Public en cours de cycle 2017 ("TNS", "TNS Sofres", "Kantar
    # Sofres" désignent la même entité) -> "sofres" doit matcher ici, pas
    # tomber dans le "kantar" générique plus bas.
    ("TNS Sofres", ("tns sofres", "tns-sofres", "sofres")),
    ("Kantar Public", ("kantar",)),
    ("Verian", ("verian",)),
    ("YouGov", ("yougov",)),
    ("CSA", ("csa",)),
    ("Dedicated Research", ("dedicated research",)),
]

# Libellés de lignes à exclure des tables d'intentions.
_NON_CANDIDATE = re.compile(
    r"^(total|base|sous-total|ne se prononce|résultats|bruts?|redress|"
    r"vote au|reconstitution|hypoth|question)", re.IGNORECASE
)

# Sondages sur un SOUS-ÉCHANTILLON (électorat/groupe ciblé) : ils somment à
# 100 % mais ne sont PAS représentatifs de l'ensemble du corps électoral, donc
# inutilisables pour un agrégat national. On les reconnaît au thème de la notice.
# (Un simple garde-fou de mots-clés ; le fallback LLM peut affiner, cf. README.)
_BIASED_SUBSAMPLE = re.compile(
    r"\b(lgbt|sympathisant|primaire|potentiel[\s-]?(?:de[\s-]?)?vote|"
    r"electorat|electeurs?[\s-]+d|bloc[\s-]central)\b|(?<![a-z])rn(?![a-z])",
    re.IGNORECASE,
)


def _is_biased_subsample(name: str) -> bool:
    """Vrai si la notice cible un sous-électorat (LGBT, sympathisants RN,
    primaire, bloc central…) plutôt que l'ensemble des inscrits."""
    return bool(_BIASED_SUBSAMPLE.search(_strip_accents(name or "")))


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
    extraction_method: str = "rules"   # rules | llm | none
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


# Ligne « Nom  [Brut] Redressé » : nom puis 1 ou 2 nombres. On garde le DERNIER
# (colonne redressée / « Résultats publiés »), qui est le score publié.
_IFOP_ROW = re.compile(r"^(.+?)((?:\s+\d{1,3}(?:,\d+)?){1,2})\s*%?$")


def _ifop_text_rows(t: str) -> list[tuple[str, float]]:
    """Repli texte pour les notices dont `extract_tables()` ne capte pas la
    colonne de valeurs (layout Ifop 2026 à une seule colonne « publiés »)."""
    out = []
    for line in t.splitlines():
        line = _MARGIN.sub(" ", line)
        line = re.sub(r"\.{2,}", " ", line)             # points de conduite
        line = re.sub(r"\s+", " ", line).strip()
        m = _IFOP_ROW.match(line)
        if not m:
            continue
        name = _clean_name(m.group(1))
        pct = _parse_pct(re.findall(r"\d{1,3}(?:,\d+)?", m.group(2))[-1])
        if pct is not None and _looks_like_candidate(name):
            out.append((name, pct))
    return out


def parse_ifop(pdf, notice: Notice) -> None:
    full_text = "\n".join(pg.extract_text() or "" for pg in pdf.pages)
    _ifop_metadata(full_text, notice)

    for pg in pdf.pages:
        t = pg.extract_text() or ""
        title = t.split("Question")[0]
        low = title.lower()
        # page d'intentions réelle : « intention(s) de vote … 1er/2nd tour »
        # (accepte le singulier « L'intention de vote », format Ifop 2026) et pas
        # une page de rappel de vote passé ni la page d'intention d'aller voter.
        is_intent = ("intention" in low
                     and any(k in low for k in ("premier tour", "1er tour",
                             "second tour", "2nd tour", "deuxième tour")))
        if not is_intent or "reconstitution" in low:
            continue
        tour = "Deuxième tour" if "second tour" in low or "deuxième tour" in low else "Premier tour"
        hypothese = _ifop_hypothese(title)
        before = len(notice.records)
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
        if len(notice.records) == before:          # tables muettes -> repli texte
            for name, pct in _ifop_text_rows(t):
                notice.records.append({"tour": tour, "hypothese": hypothese,
                                       "candidat": name, "intention": pct})


# --- Helpers partagés pour parsers basés sur le texte -----------------------
_MARGIN = re.compile(r"\+/-\s*\d+(?:[.,]\d+)?\s*points?", re.I)
_HYP = re.compile(r"Hypoth[èe]se\s*:?\s*(.+)", re.I)
_NAME = r"[A-ZÀ-Ÿ][A-Za-zÀ-ÿ'’.\-]*(?:\s+[A-Za-zÀ-ÿ'’.\-]+){1,3}?"
_PAIR_PCT = re.compile(rf"({_NAME})\s+(\d{{1,3}}(?:,\d+)?)\s*%")


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


# --- Parseur Harris/Toluna (récapitulatif positionnel) ----------------------
# Le récapitulatif Harris est une GRILLE : une ligne d'en-tête « Hypothèse »
# par colonne, une colonne de noms, puis N colonnes de valeurs (une par
# hypothèse). Deux layouts existent :
#   A. une seule colonne de noms partagée + N colonnes de valeurs (layout récent) ;
#   B. N blocs indépendants « Nom valeur » côte à côte (layout ancien).
# On parse par POSITION (`extract_words`) pour couvrir les deux, plus les pièges :
# cellules « Non testé/présent(e) » (candidat absent de l'hypothèse), noms coupés
# sur 2 lignes (« Nicolas Dupont-\nAignan ») et noms collés (« EricZemmour »).
_HARRIS_NUM = re.compile(r"^\d{1,3}(?:[.,]\d+)?$")
_HARRIS_SKIP = {"non", "teste", "present", "presente", "presentee", "testee"}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _harris_join(words: list[dict]) -> str:
    """Recolle des mots (triés) en un libellé : gère les noms coupés par un
    trait d'union en fin de ligne (« Dupont- » + « Aignan ») et les noms collés
    en CamelCase (« EricZemmour » -> « Eric Zemmour »)."""
    out = ""
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        tok = w["text"]
        if not out:
            out = tok
        elif out.endswith("-"):
            out += tok                                   # recolle « Dupont-Aignan »
        else:
            out += " " + tok
    # recolle les noms collés (« EricZemmour » -> « Eric Zemmour »). La classe
    # majuscule est explicite (À-Ö, Ø-Þ) : « À-Ÿ » couvrirait par erreur les
    # minuscules accentuées (Ÿ = U+0178) et couperait « Raphaël » en « Rapha ël ».
    out = re.sub(r"(?<=[a-zà-ÿ])(?=[A-ZÀ-ÖØ-Þ])", " ", out)
    return _clean_name(out)


def _harris_blocks(words: list[dict]) -> list[list[dict]]:
    """Regroupe des mots-noms en blocs horizontaux (un « bloc » = un candidat).
    Un grand espace en x sépare deux colonnes de noms (layout ancien à 2 blocs) ;
    un nom sur 2 lignes reste un seul bloc car ses mots se chevauchent en x."""
    blocks: list[list[dict]] = []
    for w in sorted(words, key=lambda w: w["x0"]):
        if blocks and w["x0"] - max(x["x1"] for x in blocks[-1]) <= 60:
            blocks[-1].append(w)
        else:
            blocks.append([w])
    return blocks


def _harris_recap(pg, tour: str, notice: Notice) -> None:
    import bisect
    words = pg.extract_words()
    hyp_words = [w for w in words if w["text"].strip().lower().startswith("hypoth")]
    if len(hyp_words) < 2:
        return                                     # pas la grille attendue
    anchors = sorted((w["x0"] + w["x1"]) / 2 for w in hyp_words)
    n = len(anchors)
    bounds = [(anchors[i] + anchors[i + 1]) / 2 for i in range(n - 1)]
    col_of = lambda cx: bisect.bisect_right(bounds, cx)

    hyp_top = min(w["top"] for w in hyp_words)
    footer = [w["top"] for w in words
              if _strip_accents(w["text"]).strip(":").lower() in ("base", "rappel", "lecture")
              and w["top"] > hyp_top]
    footer_top = min(footer) if footer else pg.height
    region = [w for w in words if hyp_top < w["top"] < footer_top]

    is_num = lambda w: (bool(_HARRIS_NUM.match(w["text"]))
                        and 0 <= float(w["text"].replace(",", ".")) <= 100)
    is_skip = lambda w: _strip_accents(w["text"]).strip(":").lower() in _HARRIS_SKIP
    is_hyp = lambda w: w["text"].strip().lower().startswith("hypoth")
    value_words = [w for w in region if is_num(w)]
    if not value_words:
        return
    first_data_top = min(w["top"] for w in value_words)

    # lignes de valeurs : mots numériques regroupés par `top`.
    value_words.sort(key=lambda w: w["top"])
    rows: list[dict] = []
    for w in value_words:
        if rows and abs(w["top"] - rows[-1]["top"]) <= 6:
            rows[-1]["vals"].append(w)
        else:
            rows.append({"top": w["top"], "vals": [w], "names": []})
    for r in rows:
        r["top"] = min(w["top"] for w in r["vals"])

    # libellés d'hypothèses : tout ce qui est au-dessus de la 1re ligne de données.
    labels: dict[int, str] = {}
    lbl_by_col: dict[int, list] = {}
    for w in region:
        if w["top"] < first_data_top and not is_hyp(w):
            lbl_by_col.setdefault(col_of((w["x0"] + w["x1"]) / 2), []).append(w)
    for k in range(n):
        labels[k] = _harris_join(lbl_by_col.get(k, [])) or f"hypothèse {k + 1}"

    # noms de candidats : chaque mot-nom est rattaché à la ligne de valeurs la plus
    # proche verticalement (évite tout débordement sur les lignes voisines).
    for w in region:
        if (w["top"] >= first_data_top - 1 and not is_num(w) and not is_skip(w)
                and not is_hyp(w) and re.search(r"[A-Za-zÀ-ÿ]", w["text"])):
            r = min(rows, key=lambda r: abs(w["top"] - r["top"]))
            if abs(w["top"] - r["top"]) <= 18:
                r["names"].append(w)

    for r in rows:
        blocks = _harris_blocks(r["names"])
        for v in r["vals"]:
            # candidat = bloc de noms le plus à droite situé à gauche de la valeur.
            cand = next((b for b in reversed(blocks)
                         if max(w["x1"] for w in b) < v["x0"] - 1), None)
            if not cand:
                continue
            candidat = _harris_join(cand)
            if not _looks_like_candidate(candidat):
                continue
            notice.records.append({
                "tour": tour, "hypothese": labels[col_of((v["x0"] + v["x1"]) / 2)],
                "candidat": candidat, "intention": float(v["text"].replace(",", ".")),
            })


def parse_harris(pdf, notice: Notice) -> None:
    full = "\n".join(pg.extract_text() or "" for pg in pdf.pages)
    _generic_metadata(full, notice)
    for pg in pdf.pages:
        t = pg.extract_text() or ""
        low = t[:200].lower()
        if "récapitulatif" not in low or "intentions de vote" not in low:
            continue
        _harris_recap(pg, _tour(t), notice)


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


def _full_text(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(pg.extract_text() or "" for pg in pdf.pages)


def _looks_like_intentions_pdf(full_text: str) -> bool:
    """Pré-check bon marché avant d'appeler le LLM : le texte ressemble-t-il à une
    notice d'intentions de vote 2027 ? Évite un appel LLM (payant) sur les
    baromètres d'opinion / notices sans table d'intentions. Signature retenue :
    « intention » + « suffrages/votes exprimés » + un marqueur de tour."""
    low = full_text.lower()
    tour = any(k in low for k in ("1er tour", "premier tour", "2nd tour",
                                  "second tour", "deuxième tour"))
    return "intention" in low and "exprim" in low and tour


def parse_notice(pdf_path: str | Path, source_name: str | None = None,
                 use_llm: bool | None = None) -> Notice:
    """Parse une notice PDF -> Notice normalisée (ne lève pas : statut dans .status).

    use_llm : active le fallback LLM si les règles ne sortent rien. None (défaut)
    => lit la variable d'environnement SONDAGES_USE_LLM (et nécessite une clé API).
    Le résultat du LLM repasse par le même garde-fou (somme ~100 %) que les règles.
    """
    pdf_path = Path(pdf_path)
    name = source_name or pdf_path.stem
    institut = detect_institut(name)
    notice = Notice(institut=institut, source_name=name)
    if use_llm is None:
        use_llm = os.environ.get("SONDAGES_USE_LLM", "").lower() in ("1", "true", "yes")

    # Sondage sur sous-électorat (LGBT, sympathisants RN, primaire…) : non
    # représentatif de l'ensemble des inscrits -> exclu, jamais publié.
    if _is_biased_subsample(name):
        notice.status = "biased_subsample"
        notice.extraction_method = "none"
        return notice

    if institut in PARSERS:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                PARSERS[institut](pdf, notice)
        except Exception as exc:  # PDF illisible / corrompu
            notice.status = "error"
            notice.commanditaire = notice.commanditaire or f"(erreur: {exc})"
            return notice
        _validate_records(notice)

    # Fallback LLM : institut non outillé, ou règles sans résultat. On ne dépense
    # un appel LLM que si le PDF ressemble à une notice d'intentions (pré-check).
    if not notice.records and use_llm:
        try:
            from sondages.extract_llm import extract_with_llm, available
            full_text = _full_text(pdf_path)
            if available() and _looks_like_intentions_pdf(full_text):
                notice.records = extract_with_llm(full_text, institut)
                _validate_records(notice)          # même garde-fou qu'en règles
                if notice.records:
                    notice.status = "ok"
                    notice.extraction_method = "llm"
        except Exception:
            pass                                   # le LLM ne doit jamais casser le pipeline

    if not notice.records:
        notice.status = "no_intentions" if institut in PARSERS else "unsupported_institut"
        notice.extraction_method = "none"
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
