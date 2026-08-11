"""Ingestion depuis les pages Wikipedia « Liste de sondages sur l'élection
présidentielle française » — source **primaire** pour 2027 (remplace le
pipeline PDF `ingest.py`/`parse_pdf.py` en ingestion courante).

Pourquoi : les contributeurs de la page font déjà, notice par notice, le
travail de lecture et de normalisation dans un tableau wikitext unique et
homogène — pour TOUS les instituts, pas seulement les 4 outillés par
`parse_pdf.py`. Chaque cellule « Sondeur » lie directement vers le PDF de la
notice Commission des sondages (`notice_url`), donc la traçabilité du contrat
`schema.py` est préservée. Les hypothèses multiples d'une même notice sont
conservées comme des sous-lignes `rowspan` distinctes (pas une seule
hypothèse par sondage).

Robustesse — trois pièges vérifiés en pratique sur du wikitext réel :
  1. Les colonnes candidats bougent au fil de la campagne -> mapping PAR NOM
     d'en-tête (`_header_names`), jamais par position.
  2. Les valeurs dans un template de style (`{{blanc|14}}`, `{{formatnum:1415}}`)
     disparaissent silencieusement si on ne les désamorce pas explicitement
     (`strip_code()` jette les templates par défaut). Liste blanche stricte ;
     tout template RESTANT après désamorçage est un signal d'alerte loggé
     (`unsupported_template`), pas une valeur ignorée en silence.
  3. Une cellule fusionnée (`colspan`) répétée sur ses colonnes couvertes
     double/triple-compte si on ne déduplique pas par identité d'objet.

Généralisé et validé sur les pages 2027/2022/2017 (pas seulement 2027) : le
nombre de colonnes méta avant les candidats varie (2017 intercale
« Abstention, blanc ou nul »), le tour 1er/2e n'a pas le même numéro de
section d'une page à l'autre, une section peut empiler plusieurs tableaux
sous un même titre (« hypothèses abandonnées », un par
`{{Boîte déroulante/début|titre=...}}`), des lignes « (rolling) » republient
une moyenne glissante dans le même tableau qu'un sondage brut, et un institut
non reconnu est le signal le plus fiable d'une ligne de RAPPEL (résultat
officiel inséré en comparaison) plutôt qu'un vrai sondage — toujours écarté,
jamais deviné. Comparé au pipeline PDF/NSPPolls sur 2022 (Ifop, hypothèse de
base) : 91/108 valeurs identiques à ±0,6 pt, le reste à ≤1,5 pt (arrondi).

Garde-fou : somme ~100 % par (notice, hypothèse, tour), tolérance resserrée
par rapport au pipeline PDF ([90,110]) — sur du wikitext propre, un écart
franc signale presque toujours un bug de parsing plutôt qu'un vrai sondage
aberrant. PAS de tolérance nulle : l'arrondi à 0,5 point des pourcentages
publiés fait régulièrement dévier la somme réelle d'un point (vérifié sur les
fixtures PDF existantes : Ipsos/Odoxa y somment à 99,0 — jamais 100,0 pile).

Le pipeline PDF reste utile en **spot-check** (cf. `README.md`) : on n'a plus
besoin qu'il couvre tous les instituts pour la production, donc plus besoin
non plus du fallback LLM (`extract_llm.py`) en ingestion primaire.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from itertools import groupby, product
from pathlib import Path

import mwparserfromhell as mwph
import pandas as pd
import pdfplumber
import requests

from sondages import schema
from sondages.parse_pdf import detect_institut, _iso as _pdf_iso

PDF_CACHE = Path(__file__).resolve().parent.parent / "data" / "raw" / "pdf_wiki"

API_URL = "https://fr.wikipedia.org/w/api.php"
USER_AGENT = "pres2027-sondages/0.1 (+https://github.com/yannKerzreho/pres2027)"
DEFAULT_PAGE = "Liste de sondages sur l'élection présidentielle française de 2027"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "parsed" / "intentions_2027_wiki.csv"

# Tolérance de somme par (notice, hypothèse, tour) — cf. docstring module.
SUM_LO, SUM_HI = 97.0, 103.0

# Templates dont on connaît le rôle. VALUE = porte une donnée qu'on réinjecte
# en clair ; STYLE = pure mise en forme, supprimé sans perte d'information.
def _unwrap_date_template(m: "re.Match") -> str:
    # {{date|15 avril 2017}} (1 paramètre) OU {{date|07|07|2026}} (jour|mois|année
    # séparés) -> on rejoint les paramètres par un espace dans les deux cas.
    return m.group(1).replace("|", " ")


_VALUE_TEMPLATES = [
    (re.compile(r"\{\{\s*blanc\s*\|([^{}]*)\}\}", re.I), r"\1"),
    (re.compile(r"\{\{\s*formatnum:([^{}]*)\}\}", re.I), r"\1"),
    (re.compile(r"\{\{\s*date\s*\|([^{}]*)\}\}", re.I), _unwrap_date_template),
]
_STYLE_TEMPLATES = [
    re.compile(r"\{\{\s*Infobox Parti politique fran[çc]ais/couleurs\s*\|[^{}]*\}\}", re.I),
    re.compile(r"\{\{\s*Sondeur\s*\|[^{}]*\}\}", re.I),
    re.compile(r"\{\{\s*unité?/?2?\s*\|[^{}]*\}\}", re.I),
]

_MONTHS = {m: i for i, m in enumerate(
    ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
     "août", "septembre", "octobre", "novembre", "décembre"], start=1)}
_MONTHS.update({"fevrier": 2, "aout": 8, "decembre": 12})  # variantes sans accent
_MONTHS.update({  # abréviations courantes ("avr.", "janv.")
    "janv": 1, "fevr": 2, "févr": 2, "avr": 4, "juil": 7,
    "sept": 9, "oct": 10, "nov": 11, "dec": 12, "déc": 12,
})


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


# --- Client API Wikipedia -----------------------------------------------------
def _api_get(params: dict) -> dict:
    r = requests.get(API_URL, params={**params, "format": "json"},
                     headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_sections(page: str = DEFAULT_PAGE) -> list[dict]:
    return _api_get({"action": "parse", "page": page, "prop": "sections"})["parse"]["sections"]


def fetch_section_wikitext(page: str, section_index: str) -> str:
    return _api_get({"action": "parse", "page": page, "prop": "wikitext",
                     "section": section_index})["parse"]["wikitext"]["*"]


# Sections qui NE sont PAS des tableaux de sondages bruts (même sous une
# rubrique 1er/2e tour) : moyennes glissantes, matrices de reports de voix,
# estimations post-scrutin, sous-échantillons non représentatifs (mêmes
# catégories que `parse_pdf._is_biased_subsample` côté PDF). Les inclure
# corromprait le dataset avec des données dérivées faussement étiquetées
# « sondage brut ».
_EXCLUDE_SECTION = re.compile(
    r"rolling|synth[eè]se graphique|estimation|report.*voix|sous.?[ée]chantillon|"
    r"probabilit|sous-?électorat", re.I)


def leaf_sections(sections: list[dict]) -> list[tuple[dict, str]]:
    """Sections « feuille » (contiennent potentiellement des tableaux) sous les
    rubriques 1er/2e tour, avec le tour associé. On prend les feuilles (pas les
    parents) pour éviter de re-télécharger deux fois le même contenu.

    Le tour est déterminé par le TITRE de la rubrique de plus haut niveau
    (pas par un numéro fixe) : l'ordre 1er/2e tour n'est pas le même d'une
    page à l'autre (2017 a le 2e tour AVANT le 1er tour dans le sommaire)."""
    numbers = {s["number"] for s in sections}
    tour_by_top_number: dict[str, str] = {}
    for s in sections:
        if s["toclevel"] != 1:
            continue
        low = _strip_accents(s["line"]).lower()
        if "premier tour" in low:
            tour_by_top_number[s["number"]] = "Premier tour"
        elif "second tour" in low or "deuxieme tour" in low:
            tour_by_top_number[s["number"]] = "Deuxième tour"

    out = []
    for s in sections:
        num = s["number"]
        top = num.split(".")[0]
        if top not in tour_by_top_number:
            continue
        if _EXCLUDE_SECTION.search(_strip_accents(s["line"])):
            continue
        if any(other != num and other.startswith(num + ".") for other in numbers):
            continue  # a des enfants -> pas une feuille, sera couverte par eux
        out.append((s, tour_by_top_number[top]))
    return out


# --- Parsing bas niveau d'une wikitable (grille avec rowspan/colspan) --------
def _top_level_split(s: str):
    """Coupe au premier '|' de profondeur 0 (hors {{...}}, [[...]], [http...]).
    Renvoie (attrs, contenu) ou (None, s) si aucune profondeur 0 trouvée."""
    depth, i = 0, 0
    while i < len(s):
        two = s[i:i + 2]
        if two in ("{{", "[["):
            depth += 1; i += 2; continue
        if two in ("}}", "]]"):
            depth -= 1; i += 2; continue
        if s[i] == "[":
            depth += 1; i += 1; continue
        if s[i] == "]":
            depth -= 1; i += 1; continue
        if s[i] == "|" and depth == 0:
            return s[:i], s[i + 1:]
        i += 1
    return None, s


def _desamorce(raw: str) -> tuple[str, str | None, list[str], set[str]]:
    """Désamorce les templates connus ; renvoie (texte, URL externe, noms des
    wikiliens internes DANS L'ORDRE, noms des templates INCONNUS restants).
    Un template inconnu est un signal d'alerte : il pourrait porter une
    valeur qu'on jetterait silencieusement sinon.

    Les wikiliens internes servent à identifier le(s) candidat(s) NOMMÉ(S)
    DANS LA CELLULE plutôt que dans l'en-tête : la colonne RN a un en-tête
    générique « Candidat RN » tant que le candidat n'est pas arrêté, mais
    chaque cellule lie vers le candidat effectivement testé pour cette
    hypothèse (ex. `[[Jordan Bardella|Bardella]]`) — sans ça, tous les
    sondages Bardella/Le Pen se retrouvent étiquetés "CandidatRN" et
    disparaissent du mapping candidat en aval.

    PLUSIEURS liens dans la même cellule séparés par « / » (ex.
    `[[Bardella]] / [[Le Pen]]`, observé réellement sur OpinionWay 10-11 juin
    2026 : Wikipedia indique explicitement que le MÊME score vaut pour les
    deux candidats, quand le sondeur ne distingue pas les deux hypothèses
    pour cette question) sont TOUS retournés — `extract_notices` décide quoi
    en faire (garder que le premier casse silencieusement le second, cf. bug
    corrigé le 2026-08-10 : les 3 hypothèses de ce sondage ressortaient 100%
    Bardella, jamais Le Pen). Un cas à NE PAS confondre : un lien vers le
    PARTI entre parenthèses (« [[Hollande]] ([[Parti socialiste|PS]]) ») a
    aussi 2 liens mais n'est PAS une alternative de candidat — d'où le test
    sur « / » explicite dans le texte, pas juste `len(links) > 1`."""
    for pat, repl in _VALUE_TEMPLATES:
        raw = pat.sub(repl, raw)
    for pat in _STYLE_TEMPLATES:
        raw = pat.sub("", raw)
    code = mwph.parse(raw)
    unknown = {str(t.name).strip() for t in code.filter_templates()}
    for ref in code.filter_tags(matches=lambda t: t.tag == "ref"):
        try:
            code.remove(ref)
        except ValueError:
            pass
    ext = code.filter_external_links()
    url = str(ext[0].url) if ext else None
    links = code.filter_wikilinks()
    text = re.sub(r"\s+", " ", code.strip_code().strip())
    if len(links) > 1 and "/" in text:
        linked_names = [str(l.text or l.title).strip() for l in links]
    else:
        linked_names = [str(links[0].text or links[0].title).strip()] if links else []
    return text, url, linked_names, unknown


def _parse_cell(raw: str) -> dict:
    attrs_str, content = _top_level_split(raw)
    attrs = {}
    if attrs_str is not None and "=" in attrs_str:
        for m in re.finditer(r'(\w+)\s*=\s*"?([^"\s]+)"?', attrs_str):
            attrs[m.group(1)] = m.group(2)
    text, url, linked_names, unknown = _desamorce(content)
    return {"text": text, "url": url, "linked_names": linked_names, "unknown": unknown,
            "rowspan": int(attrs.get("rowspan", 1)), "colspan": int(attrs.get("colspan", 1))}


def _split_row_cells(line_block: str) -> list[str]:
    cells, marker = [], None
    for line in line_block.split("\n"):
        line = line.rstrip()
        if line[:1] in ("!", "|") and line[:2] not in ("|-", "|}"):
            marker = marker or line[0]
            if line[0] != marker:
                continue
            sep = "!!" if marker == "!" else "||"
            cells.extend(line[1:].split(sep))
    return cells


def _grid_from_table_text(table_text: str) -> list[list[dict]]:
    row_blocks = table_text.split("\n|-")
    grid: list[list[dict]] = []
    pending: dict[int, dict] = {}
    for rb in row_blocks:
        raw_cells = _split_row_cells(rb)
        if not raw_cells:
            continue
        parsed = [_parse_cell(c) for c in raw_cells]
        row_out, col, pi = [], 0, 0
        while pi < len(parsed) or any(v["remaining"] > 0 for v in pending.values()):
            if pending.get(col, {}).get("remaining", 0) > 0:
                row_out.append(pending[col]["cell"])
                pending[col]["remaining"] -= 1
                col += 1
                continue
            if pi >= len(parsed):
                break
            cell = parsed[pi]; pi += 1
            for k in range(cell["colspan"]):
                row_out.append(cell)
                if cell["rowspan"] > 1 and k == 0:
                    pending[col] = {"remaining": cell["rowspan"] - 1, "cell": cell}
                col += 1
        if row_out:
            grid.append(row_out)
    return grid


def parse_wikitables(section_wikitext: str) -> list[list[list[dict]]]:
    """Renvoie une liste de grilles (une par tableau `{| ... |}` trouvé)."""
    return [g for g, _ in parse_wikitables_with_titles(section_wikitext)]


# Certaines sections (ex. « hypothèses abandonnées ») empilent PLUSIEURS
# tableaux, chacun sous son propre {{Boîte déroulante/début|titre=...}} — le
# vrai libellé d'hypothèse est là, pas dans le titre de la section wiki qui
# les englobe tous. Sans ça, des candidats de duels DIFFÉRENTS partageant
# institut+date se retrouvent groupés sous la même hypothèse -> somme fausse.
_BOITE_TITRE = re.compile(r"\{\{\s*Bo[îi]te\s*d[ée]roulante\s*/\s*d[ée]but\s*\|\s*titre\s*=\s*([^\n|}]+)", re.I)


def parse_wikitables_with_titles(section_wikitext: str) -> list[tuple[list[list[dict]], str | None]]:
    """Renvoie [(grille, titre de la Boîte déroulante englobante ou None), ...]."""
    titres = [(m.start(), m.group(1).strip()) for m in _BOITE_TITRE.finditer(section_wikitext)]
    out = []
    for m in re.finditer(r"\{\|.*?\n\|\}", section_wikitext, re.S):
        grid = _grid_from_table_text(m.group(0))
        if not grid:
            continue
        titre = None
        for pos, t in titres:
            if pos < m.start():
                titre = t
            else:
                break
        out.append((grid, titre))
    return out


# --- Reconstruction des enregistrements structurés ---------------------------
def _parse_dates(raw: str, year_hint: str | None = None) -> tuple[str | None, str | None]:
    """« 22-24 juin 2026 », « 9 - 10 juillet 2026 », « 7-8 juillet » (année
    souvent ABSENTE du texte -> `year_hint`, tiré de l'URL de la notice qui
    contient toujours .../AAAA/mois/...), « 29 juin-2 juillet 2021 » (à cheval
    sur deux mois), « 5 juin 2026 »."""
    raw = re.sub(r"\s+", " ", raw.strip())
    # à cheval sur deux mois : essayé EN PREMIER, sinon le motif "même mois"
    # ci-dessous matche juste le premier jour et ignore silencieusement la
    # fin de plage (et son année, si elle n'est là que sur le 2e mois).
    m = re.match(r"(\d{1,2})\s+([A-Za-zà-ÿ]+)\.?\s*(?:[-–]|au)\s*"
                r"(\d{1,2})\s+([A-Za-zà-ÿ]+)\.?(?:\s+(\d{4}))?", raw)
    if m:
        d1, mois1, d2, mois2, an = m.groups()
        an = an or year_hint
        m1, m2 = _MONTHS.get(mois1.lower()), _MONTHS.get(mois2.lower())
        if m1 and m2 and an:
            return f"{an}-{m1:02d}-{int(d1):02d}", f"{an}-{m2:02d}-{int(d2):02d}"
    m = re.match(r"(\d{1,2})\s*(?:[-–]|au)\s*(\d{1,2})\s+([A-Za-zà-ÿ]+)\.?(?:\s+(\d{4}))?", raw)
    if m:
        d1, d2, mois, an = m.groups()
        an = an or year_hint
        mnum = _MONTHS.get(mois.lower())
        if mnum and an:
            return f"{an}-{mnum:02d}-{int(d1):02d}", f"{an}-{mnum:02d}-{int(d2):02d}"
    m = re.match(r"(\d{1,2})\s+([A-Za-zà-ÿ]+)\.?(?:\s+(\d{4}))?", raw)
    if m:
        d, mois, an = m.groups()
        an = an or year_hint
        mnum = _MONTHS.get(mois.lower())
        if mnum and an:
            iso = f"{an}-{mnum:02d}-{int(d):02d}"
            return iso, iso
    return None, None


_SUBST_NAME = re.compile(r"([A-ZÀ-Ÿ][A-Za-zÀ-ÿ'’\-]+(?:\s+[A-Za-zÀ-ÿ'’\-]+)*)\s*\(([A-ZÉ]{2,6})\)")
# En-tête placeholder ("Candidat RN", "Candidat LR"...) : le nom du sigle de
# parti n'est PAS un nom de candidat, ne devrait jamais apparaître comme tel.
_GENERIC_PLACEHOLDER = re.compile(r"^Candidat[A-ZÉ]", re.I)


def _header_names(names_row: list[dict]) -> list[str]:
    return [c["text"].split(" (")[0].split("(")[0].strip() for c in names_row]


# Sondeur/Date/Échantillon sont toujours les 3 premières colonnes, mais
# certaines pages (2017) intercalent une 4e colonne méta (« Abstention, blanc
# ou nul ») avant les candidats -> le nombre de colonnes méta n'est pas fixe,
# on le détecte par le LIBELLÉ plutôt qu'une position codée en dur.
def _is_meta_label(label: str) -> bool:
    label = _strip_accents(label).strip().lower()
    return label in ("sondeur", "date", "dates", "echantillon") or label.startswith("abstention")


def extract_notices(grid: list[list[dict]], tour: str, hypothese_defaut: str | None,
                    warnings: list[str], year_hint: str | None = None) -> list[dict]:
    # 1. repère la ligne d'en-tête "Sondeur"
    header_idx = next((i for i, row in enumerate(grid)
                       if _strip_accents(row[0]["text"]).strip().lower() == "sondeur"), None)
    if header_idx is None or header_idx + 1 >= len(grid):
        return []
    n_meta = 0
    for c in grid[header_idx]:
        if not _is_meta_label(c["text"]):
            break
        n_meta += 1
    n_meta = max(n_meta, 3)  # Sondeur/Date/Échantillon toujours présents
    names_row = grid[header_idx + 1]
    candidates = _header_names(names_row[n_meta:])

    data_start = header_idx + 2
    if data_start < len(grid):
        # ligne de couleurs (pure style) -> ignorée. Un test "toutes les
        # cellules vides" est fragile si une colonne d'en-tête (ex. "Autres")
        # a aussi rowspan=3 : son texte s'y propage. On tolère ces cellules
        # PROPAGÉES depuis l'en-tête (même objet), pas seulement les vides.
        header_cell_ids = {id(c) for c in grid[header_idx]} | {id(c) for c in names_row}
        row = grid[data_start]
        if all((not c["text"]) or id(c) in header_cell_ids for c in row[n_meta:]):
            data_start += 1
    data_rows = [r for r in grid[data_start:] if not (r and r[0] is r[-1])]  # bannières colspan=N

    records = []
    for _, group_iter in groupby(data_rows, key=lambda r: id(r[0])):
        group = list(group_iter)
        sondeur_cell, date_cell, ech_cell = group[0][0], group[0][1], group[0][2]
        if "rolling" in sondeur_cell["text"].lower():
            # moyenne glissante republiée dans le même tableau que les
            # sondages bruts (annotée « (rolling) ») -> pas une observation
            # indépendante, écartée pour ne pas fausser le calibrage.
            warnings.append(f"rolling écarté ({sondeur_cell['text']!r})")
            continue
        institut = detect_institut(sondeur_cell["text"])
        if institut is None:
            # pas un institut reconnu -> souvent une ligne de RAPPEL (résultat
            # officiel du tour précédent inséré en comparaison dans le tableau),
            # pas un sondage. On écarte plutôt que d'inventer un institut.
            warnings.append(f"institut non reconnu {sondeur_cell['text']!r} -> notice écartée")
            continue
        notice_url = sondeur_cell["url"]
        # Repli année : URL de la notice (le plus précis, propre à CE sondage)
        # -> contexte de section/page (fourni par l'appelant). Le texte de
        # cellule wiki omet souvent l'année (donnée par le contexte).
        m_year = re.search(r"/(\d{4})/", notice_url or "")
        notice_year_hint = m_year.group(1) if m_year else year_hint
        date_debut, date_fin = _parse_dates(date_cell["text"], notice_year_hint)
        m_ech = re.search(r"\d+", ech_cell["text"].replace(" ", ""))
        echantillon = int(m_ech.group(0)) if m_ech else None
        notice_label = f"{institut} {date_cell['text']} (WP)"

        multi = len(group) > 1
        for gi, row in enumerate(group):
            hypothese = hypothese_defaut
            if tour == "Premier tour" and multi:
                hypothese = f"variante_{gi + 1}"
            elif tour == "Premier tour" and not multi:
                hypothese = None

            # Cellule à liens multiples (« [[Bardella]] / [[Le Pen]] » — même
            # score valable pour plusieurs candidats, cf. `_desamorce`) : la
            # 1ère alternative garde le libellé d'hypothèse d'origine
            # (comportement historique, rétro-compatible), chaque alternative
            # SUPPLÉMENTAIRE devient une hypothèse DISTINCTE avec le même
            # score — jamais fusionnées dans la même hypothèse, ça casserait
            # le garde-fou de somme ~100% (schema.py). Dédoublonné par
            # identité de cellule : une cellule à colspan>1 apparaît plusieurs
            # fois dans `row[n_meta:]`, une seule fois logiquement.
            seen_cell_ids, multi_name_cells = set(), []
            for ci, (_, cell) in enumerate(zip(candidates, row[n_meta:])):
                if len(cell["linked_names"]) > 1 and id(cell) not in seen_cell_ids:
                    seen_cell_ids.add(id(cell))
                    multi_name_cells.append((ci, cell))

            if not multi_name_cells:
                combos = [(hypothese, {})]
            else:
                combos = []
                for combo_idx, names in enumerate(product(*(c["linked_names"] for _, c in multi_name_cells))):
                    override = {ci: name for (ci, _), name in zip(multi_name_cells, names)}
                    label = hypothese if combo_idx == 0 else f"{hypothese or 'variante_1'}_alt_{'_'.join(names)}"
                    combos.append((label, override))

            for combo_hypothese, name_override in combos:
                prev_cell, cand_names_used = None, set()
                for ci, (cand, cell) in enumerate(zip(candidates, row[n_meta:])):
                    if cell is prev_cell:
                        continue
                    prev_cell = cell
                    if cell["unknown"]:
                        warnings.append(f"template inconnu {cell['unknown']} dans {notice_label}")
                    txt = cell["text"]
                    if not txt or txt in ("—", "-"):
                        continue
                    m_pct = re.search(r"(\d{1,3}(?:[.,]\d+)?)", txt)
                    if not m_pct:
                        continue
                    pct = float(m_pct.group(1).replace(",", "."))
                    candidat = cand
                    if ci in name_override:
                        candidat = name_override[ci]
                    elif cell["linked_names"]:
                        # La cellule lie explicitement vers un candidat -> prioritaire
                        # sur l'en-tête. Couvre deux cas : la colonne RN a un en-tête
                        # générique « Candidat RN » tant que le candidat n'est pas
                        # arrêté (le vrai nom n'est que dans la cellule, jamais dans
                        # l'en-tête) ; et les cellules de substitution (colspan>1).
                        candidat = cell["linked_names"][0]
                    elif cell["colspan"] > 1:
                        m_name = _SUBST_NAME.search(txt)
                        if not m_name:
                            warnings.append(f"substitution non identifiée ({txt!r}) dans {notice_label}")
                            continue  # on ne devine pas -> on écarte plutôt qu'on mislabel
                        candidat = m_name.group(1).strip()
                    if _GENERIC_PLACEHOLDER.match(candidat):
                        # En-tête générique ("Candidat RN"/"Candidat LR") ET
                        # aucun wikilien dans la cellule pour CETTE ligne -> le
                        # candidat n'était pas encore connu au moment du sondage,
                        # Wikipedia elle-même ne le précise pas ici. On écarte
                        # plutôt que de publier un faux nom de candidat.
                        warnings.append(f"candidat indéterminé ({candidat!r}) dans {notice_label}")
                        continue
                    if candidat in cand_names_used:
                        continue
                    cand_names_used.add(candidat)
                    records.append({
                        "notice_id": None, "notice": notice_label, "notice_url": notice_url,
                        "institut": institut, "commanditaire": None,
                        "date_debut": date_debut, "date_fin": date_fin, "date_notice": date_fin,
                        "echantillon": echantillon, "methode": None,
                        "tour": tour, "hypothese": combo_hypothese,
                        "candidat": candidat, "intention": pct,
                    })
    return records


# --- Repli : dates manquantes récupérées dans la notice elle-même -----------
# Le texte de cellule wiki n'a pas toujours d'année ; la notice PDF, elle,
# l'a presque toujours (« réalisée ... du 26 au 27 mars 2025 »). Un seul
# téléchargement par NOTICE (pas par ligne) : le nombre de notices distinctes
# à dater est petit (quelques dizaines), même quand des milliers de lignes en
# dépendent.
_PDF_DATE_RANGE = re.compile(r"du\s+(\d{1,2})\s+au\s+(\d{1,2})\s+([A-Za-zÀ-ÿ]+)\s+(\d{4})", re.I)
_PDF_DATE_SINGLE = re.compile(r"\ble\s+(\d{1,2})\s+([A-Za-zÀ-ÿ]+)\s+(\d{4})", re.I)


def _dates_from_notice_pdf(url: str) -> tuple[str | None, str | None]:
    PDF_CACHE.mkdir(parents=True, exist_ok=True)
    dest = PDF_CACHE / (re.sub(r"[^\w.-]", "_", url.rsplit("/", 1)[-1]) or "notice.pdf")
    if not (dest.exists() and dest.stat().st_size > 0):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            r.raise_for_status()
            if r.content[:4] != b"%PDF":
                return None, None
            dest.write_bytes(r.content)
        except requests.RequestException:
            return None, None
    try:
        with pdfplumber.open(dest) as pdf:
            text = "\n".join((pg.extract_text() or "") for pg in pdf.pages[:2])
    except Exception:
        return None, None
    m = _PDF_DATE_RANGE.search(text)
    if m:
        d1, d2, mois, an = m.groups()
        return _pdf_iso(d1, mois, an), _pdf_iso(d2, mois, an)
    m = _PDF_DATE_SINGLE.search(text)
    if m:
        d, mois, an = m.groups()
        iso = _pdf_iso(d, mois, an)
        return iso, iso
    return None, None


def fill_missing_dates_from_pdf(df: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque `notice_url` distincte sans date, tente de la récupérer
    dans le PDF de la notice. Ne devine jamais : si la notice n'est pas un
    PDF exploitable, la date reste `None`."""
    df = df.copy()
    missing_urls = df.loc[df["date_debut"].isna(), "notice_url"].dropna().unique()
    for url in missing_urls:
        d1, d2 = _dates_from_notice_pdf(url)
        if d1:
            mask = (df["notice_url"] == url) & df["date_debut"].isna()
            df.loc[mask, "date_debut"] = d1
            df.loc[mask, "date_fin"] = d2
    return df


# --- Orchestration -------------------------------------------------------------
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _year_from_text(*texts: str | None) -> str | None:
    for t in texts:
        if t:
            m = _YEAR_RE.search(t)
            if m:
                return m.group(0)
    return None


def _section_year_hint(sections: list[dict], s: dict) -> tuple[str | None, bool, bool]:
    """Année trouvée dans le titre de la section ou d'un de ses ancêtres (ex.
    « Année 2021 » > « Second semestre 2021 »), + deux signaux de sûreté pour
    le repli sur l'année de la page :
      - `bounded` : la fenêtre est bornée sans ambiguïté par la publication de
        la liste officielle des candidats ou le premier tour (toujours à
        quelques semaines du scrutin, donc dans l'année de l'élection) ;
      - `unsafe` : la section est explicitement « avant » (publication, 1er
        tour, primaires...) -> peut déborder de plusieurs mois voire années
        sur l'AVANT de l'élection (vérifié : un sondage Macron-Le Pen "avant
        le premier tour" daté juin 2021 dans son propre texte se faisait
        écraser en 2022 par un repli page-année trop large côté "2e tour").
    `unsafe` l'emporte toujours sur `bounded`/tour Deuxième tour : mieux vaut
    une date manquante qu'une mauvaise année."""
    num = s["number"]
    ancestors = [_strip_accents(o["line"]).lower() for o in sections
                if o["number"] == num or num.startswith(o["number"] + ".")]
    bounded = any("apres" in a for a in ancestors)
    unsafe = any("avant" in a for a in ancestors)
    raw_ancestors = [o["line"] for o in sections
                     if o["number"] == num or num.startswith(o["number"] + ".")]
    return _year_from_text(*raw_ancestors), bounded, unsafe


def build_dataset(page: str = DEFAULT_PAGE, pdf_date_fallback: bool = True
                  ) -> tuple[pd.DataFrame, dict]:
    """Renvoie (intentions, rapport). `intentions` respecte `schema.COLUMNS`."""
    sections = fetch_sections(page)
    page_year = _year_from_text(page)  # sûr en repli SEULEMENT pour le 2e tour
                                        # (toujours dans l'année de l'élection ;
                                        # le 1er tour, lui, déborde sur l'année
                                        # précédente -> jamais deviné pour lui).
    warnings: list[str] = []
    all_records: list[dict] = []
    n_tables = 0

    for s, tour in leaf_sections(sections):
        wt = fetch_section_wikitext(page, s["index"])
        section_year, bounded, unsafe = _section_year_hint(sections, s)
        for grid, boite_titre in parse_wikitables_with_titles(wt):
            n_tables += 1
            hyp_defaut = (boite_titre or s["line"]) if tour == "Deuxième tour" else None
            year_hint = _year_from_text(boite_titre) or section_year
            if year_hint is None and not unsafe and (tour == "Deuxième tour" or bounded):
                year_hint = page_year
            all_records.extend(extract_notices(grid, tour, hyp_defaut, warnings, year_hint))

    df = pd.DataFrame(all_records)
    if df.empty:
        return schema.empty(), {"n_tables": n_tables, "n_raw": 0, "n_kept": 0,
                                "n_rejected_groups": 0, "warnings": warnings}

    # garde-fou somme ~100% PAR (notice, hypothese, tour) — tolérance resserrée
    g = df.groupby(["notice", "hypothese", "tour"], dropna=False)["intention"].sum()
    bad = set(g[(g < SUM_LO) | (g > SUM_HI)].index)
    keep_mask = ~df.set_index(["notice", "hypothese", "tour"]).index.isin(bad)
    kept = df[keep_mask].reset_index(drop=True)

    if pdf_date_fallback:
        n_before = kept["date_debut"].isna().sum()
        kept = fill_missing_dates_from_pdf(kept)
        n_filled = n_before - kept["date_debut"].isna().sum()
    else:
        n_filled = 0

    intentions = kept[list(schema.COLUMNS)]
    schema.validate(intentions)
    report = {"n_tables": n_tables, "n_raw": len(df), "n_kept": len(intentions),
             "n_rejected_groups": len(bad), "n_dates_filled_from_pdf": n_filled,
             "rejected": sorted(bad, key=lambda t: tuple(str(x) for x in t))[:20],
             "warnings": warnings}
    return intentions, report


def write_dataset(intentions: pd.DataFrame, out: Path = DEFAULT_OUT) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    intentions.to_csv(out, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default=DEFAULT_PAGE)
    ap.add_argument("--out", default=None, help="chemin CSV de sortie (défaut : déduit du nom de page)")
    ap.add_argument("--min-sondages", type=int, default=0,
                    help="échoue (code 1) SANS écrire si moins de N sondages sont "
                         "extraits — garde-fou du job quotidien contre une "
                         "dégradation silencieuse du parsing Wikipedia")
    args = ap.parse_args()

    intentions, report = build_dataset(args.page)

    # Vérifier AVANT d'écrire : la source est une page wiki éditable par
    # n'importe qui, et un changement de gabarit ou un renommage de section
    # ferait tomber l'extraction à quelques sondages sans lever d'erreur. Sans
    # ce contrôle, le CSV serait écrasé puis `model.run` publierait un snapshot
    # calculé sur presque rien — une régression invisible, en ligne.
    n = 0 if intentions.empty else int(intentions["notice"].nunique())
    if args.min_sondages and n < args.min_sondages:
        raise SystemExit(
            f"ABANDON : {n} sondages extraits, minimum attendu {args.min_sondages}. "
            f"Le fichier existant n'a PAS été écrasé. Vérifier la page « {args.page} » "
            f"(gabarit modifié ?) et {len(report['warnings'])} avertissement(s) de parsing.")

    out_path = Path(args.out) if args.out else DEFAULT_OUT
    out = write_dataset(intentions, out_path)

    print(f"{report['n_tables']} tableaux, {report['n_raw']} lignes brutes -> "
          f"{report['n_kept']} retenues ({report['n_rejected_groups']} groupes "
          f"rejetés pour somme hors [{SUM_LO},{SUM_HI}]).")
    if report["rejected"]:
        print("Rejetés (échantillon) :")
        for k in report["rejected"]:
            print(f"  {k}")
    if report["warnings"]:
        print(f"\n{len(report['warnings'])} avertissements (templates inconnus / "
              f"substitutions non identifiées) :")
        for w in report["warnings"][:20]:
            print(f"  {w}")
    if not intentions.empty:
        print(f"\n{intentions['notice'].nunique()} sondages, "
              f"{intentions['candidat'].nunique()} candidats distincts, "
              f"{intentions['institut'].nunique()} instituts.")
        print(intentions.groupby("institut").size().sort_values(ascending=False).to_string())
    print(f"\nÉcrit : {out}")


if __name__ == "__main__":
    main()
