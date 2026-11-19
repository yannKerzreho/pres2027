"""Reports T1 -> T2 publiés AVANT le 1er tour : dans les rapports des instituts,
les duels hypothétiques de 2nd tour sont souvent croisés par INTENTION de vote
au 1er tour (« parmi les électeurs de Mélenchon au T1 : x % Macron, y % Le Pen,
z % abstention »). Wikipedia ne les relève pas (sa section « Reports de voix »
n'existe qu'entre les deux tours, cf. `sondages/reports.py`) : on les lit dans
les PDF cités par les lignes de 2nd tour de `intentions_{annee}_wiki.csv`.

Une ligne = (sondage, électorat T1, option T2), au schéma de
`reports_t2_{annee}.csv` plus :
- `nature_t1` : « intention » (intention T1 déclarée, avant le scrutin) ; les
  fichiers `reports_t2_*` relèvent, eux, le vote T1 déclaré (« vote ») ;
- `page` : page du PDF (contrôle manuel).

`electorat_t1 = "Aucun"` : abstentionnistes / blancs au 1er tour, quand
l'institut les croise aussi. Les lignes « Ensemble » (total) ne sont pas
reprises : c'est l'intention de 2nd tour déjà dans `intentions_*`.

Trois mises en forme reconnues, essayées dans l'ordre :
- barres empilées horizontales (Elabe, Ifop) : les parts sont lues sur les
  étiquettes, et sur la LARGEUR des rectangles quand une petite part n'est pas
  étiquetée (largeur / largeur totale, arrondie) ;
- tableau électorats en lignes (Ipsos, Elabe tableaux) ;
- tableau électorats en colonnes (Harris).
Garde-fous : un électorat n'est gardé que si ses options couvrent les deux
finalistes et somment dans [SUM_LO, SUM_HI] ; tout le reste est écarté et
compté dans le rapport. On n'invente rien : une page illisible est perdue.

    python -m sondages.reports_avant_t1 --annee 2022
"""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "pdf_reports"
SUM_LO, SUM_HI = 97.0, 103.0
DATES_T1 = {"2017": "2017-04-23", "2022": "2022-04-10", "2027": "2027-04-11"}
COLUMNS = ["notice", "institut", "date_debut", "date_fin", "echantillon", "hypothese",
           "electorat_t1", "nature_t1", "candidat_t2", "part", "base_electorat", "source", "page"]
_AUCUN = re.compile(r"abst|blanc|nul|non expr|pas expr|aucun|ne sait|nsp|sans opinion|ne vot|n'ira", re.I)
_NUM = re.compile(r"^(\d{1,3})(?:[.,](\d))?%?$")
_DASH = {"-", "–", "—", "*"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


# candidats testés au T1 mais absents du scrutin (donc de `results_*`), et
# ceux de 2027 (pas encore de `results_2027`)
HORS_SCRUTIN = ["Jean-Luc Mélenchon", "Marine Le Pen", "Xavier Bertrand", "Christiane Taubira", "Benoît Hamon", "Laurent Wauquiez",
                "Arnaud Montebourg", "Michel Barnier", "Éric Ciotti", "Philippe Juvin",
                "Alain Juppé", "Nicolas Sarkozy", "François Bayrou", "Manuel Valls",
                "Yannick Jadot", "Ségolène Royal", "François Hollande",
                "Florian Philippot", "François Asselineau", "Jean Lassalle", "Jacques Cheminade",
                "Sandrine Rousseau", "Eric Piolle", "Rachida Dati", "Bruno Retailleau",
                "Édouard Philippe", "Gabriel Attal", "Jordan Bardella", "Raphaël Glucksmann",
                "François Ruffin", "Laurent Berger", "Gérald Darmanin", "Bruno Le Maire",
                "Marion Maréchal", "Philippe de Villiers", "Olivier Faure", "Clémentine Autain",
                "Fabien Roussel", "Marine Tondelier", "Lucie Castets", "Michel Onfray"]


def nom_complet(annee: str) -> dict[str, str]:
    """Nom de famille (comme dans `intentions_*`) -> nom complet (comme dans
    `results_*` et `reports_t2_*`), pour les candidats du 1er tour réel."""
    res = ROOT / "data" / "historical" / f"results_{annee}.csv"
    noms = list(HORS_SCRUTIN)
    if res.exists():
        r = pd.read_csv(res)
        noms += list(r.loc[r["tour"] == "Premier tour", "candidat"])
    t2 = ROOT / "data" / "parsed" / f"reports_t2_{annee}.csv"
    if t2.exists():  # « Éric Zemmour » (reports_t2) plutôt que « Eric » (results)
        noms += list(pd.read_csv(t2)["electorat_t1"].unique())
    return {_norm(n.split(maxsplit=1)[-1]): n for n in noms if len(n.split()) > 1}


def _cherche(texte_norm: str, noms: dict[str, str]) -> list[str]:
    """Noms (clé = nom de famille normalisé) présents dans un texte normalisé,
    le plus long d'abord (« le pen » avant « pen »)."""
    t = f" {texte_norm} "
    colle = texte_norm.replace(" ", "")   # « Jean-LucMélenchon » (Opinion Way 2017)
    return [v for k, v in sorted(noms.items(), key=lambda kv: -len(kv[0]))
            if f" {k} " in t or (len(k) >= 5 and k.replace(" ", "") in colle)]


def _valeur(txt: str) -> float | None:
    if txt in _DASH:
        return 0.0
    m = _NUM.match(txt.replace(" ", ""))
    if not m:
        return None
    return float(m.group(1) + ("." + m.group(2) if m.group(2) else ""))


def _lignes(words: list[dict], tol: float = 3.0) -> list[list[dict]]:
    out: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if out and abs(out[-1][0]["top"] - w["top"]) <= tol:
            out[-1].append(w)
        else:
            out.append([w])
    return [sorted(l, key=lambda w: w["x0"]) for l in out]


def _centre(w) -> float:
    return (w["x0"] + w["x1"]) / 2


# --------------------------------------------------------------------- options
def _option(label: str, finalistes: dict[str, str]) -> str | None:
    """Libellé d'option de 2nd tour -> nom de famille du finaliste, « Aucun »,
    ou None (inconnu)."""
    trouve = _cherche(_norm(label), finalistes)
    if len(trouve) == 1:
        return trouve[0]
    if not trouve and _AUCUN.search(label):
        return "Aucun"
    return None


def _valide(table: dict[str, dict[str, float]], finalistes: dict[str, str],
            notes: list[str], ou: str) -> dict[str, dict[str, float]]:
    out = {}
    attendus = set(finalistes.values())
    for el, parts in table.items():
        s = sum(parts.values())
        if parts and not attendus <= set(parts) and SUM_LO <= s <= SUM_HI:
            # case vide (« - » non imprimé) alors que le reste fait déjà 100 :
            # l'option manquante vaut 0, rien ne peut s'y cacher
            parts = {**{o: 0.0 for o in attendus}, **parts}
        if not attendus <= set(parts):
            notes.append(f"{ou} / {el} : options {sorted(parts)} incomplètes -> écarté")
        elif not SUM_LO <= s <= SUM_HI:
            notes.append(f"{ou} / {el} : somme {s:g} hors [{SUM_LO:g}, {SUM_HI:g}] -> écarté")
        else:
            out[el] = parts
    return out


# ----------------------------------------------------------------- barres
def _couleur(r) -> tuple | None:
    c = r.get("non_stroking_color")
    if c is None:
        return None
    try:
        c = tuple(round(float(x), 3) for x in (c if isinstance(c, (list, tuple)) else [c]))
    except (TypeError, ValueError):
        return None  # motif (pattern) : pas une couleur de barre
    if c in ((1.0,), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0)):
        return None  # blanc : fond
    return c


def lit_barres(page, finalistes: dict[str, str]) -> dict[str, dict[str, float]] | None:
    rects = {}
    for r in page.rects:
        c = _couleur(r)
        if c is None or r["height"] < 4 or r["width"] <= 0:
            continue
        rects[(round(r["x0"], 1), round(r["x1"], 1), round(r["top"], 1), round(r["bottom"], 1))] = c
    # rangées : rectangles de même hauteur et même bande verticale, contigus
    bandes = defaultdict(list)
    for (x0, x1, t, b), c in rects.items():
        if b - t >= 6:
            bandes[(round(t), round(b))].append((x0, x1, c))
    rangees = []
    for (t, b), segs in bandes.items():
        segs.sort()
        contigu = all(abs(segs[i][1] - segs[i + 1][0]) < 1.5 for i in range(len(segs) - 1))
        if contigu:
            rangees.append((t, b, segs))
    if len(rangees) < 3:
        return None
    # les vraies barres partagent bord gauche et largeur totale (100 %)
    gauche = pd.Series([round(s[0][0]) for _, _, s in rangees]).mode()[0]
    droite = pd.Series([round(s[-1][1]) for _, _, s in rangees]).mode()[0]
    rangees = [r for r in rangees if abs(r[2][0][0] - gauche) < 2 and abs(r[2][-1][1] - droite) < 2]
    if len(rangees) < 3 or droite - gauche < 50:
        return None
    largeur = droite - gauche

    words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
    # légende : petits carrés colorés suivis de leur libellé ; à défaut, ligne
    # de texte sous les barres, découpée aux grands blancs, dans l'ordre des
    # couleurs de la barre la plus complète.
    couleurs_barres = {c for _, _, s in rangees for *_, c in s}
    carres = sorted({(x0, t, c) for (x0, x1, t, b), c in rects.items()
                     if x1 - x0 < 14 and b - t < 14 and c in couleurs_barres})
    libelles: dict[tuple, str] = {}
    if len(carres) >= 2:
        for i, (x0, t, c) in enumerate(carres):
            suivants = [x for x, tt, _ in carres if abs(tt - t) < 4 and x > x0]
            fin = min(suivants) if suivants else 1e9
            mots = [w["text"] for w in words if abs(w["top"] - t) < 8 and x0 < w["x0"] < fin
                    and w["top"] < t + 12]
            libelles[c] = " ".join(mots)
    else:
        bas = max(b for _, b, _ in rangees)
        haut = min(t for t, _, _ in rangees)
        # sous les barres d'abord (Ifop 2021), puis au-dessus (Ifop 2017),
        # la plus proche d'abord
        sous = _lignes([w for w in words if bas < w["top"] < bas + 60])
        dessus = _lignes([w for w in words if haut - 80 < w["bottom"] <= haut])[::-1]
        sous = sous + dessus
        complete = max(rangees, key=lambda r: len(r[2]))[2]
        ordre = list(dict.fromkeys(c for *_, c in complete))
        for l in sous:
            items, cur = [], [l[0]]
            for a, b in zip(l, l[1:]):
                if b["x0"] - a["x1"] > 12:
                    items.append(cur); cur = []
                cur.append(b)
            items.append(cur)
            if len(items) == len(ordre):
                libelles = {c: " ".join(w["text"] for w in it) for c, it in zip(ordre, items)}
                break
    options = {c: _option(lab, finalistes) for c, lab in libelles.items()}
    if not options or None in options.values() or not set(finalistes.values()) <= set(options.values()):
        return None

    table = {}
    for t, b, segs in rangees:
        etiquette = " ".join(w["text"] for w in words
                             if w["x1"] <= gauche + 2 and t - 4 <= (w["top"] + w["bottom"]) / 2 <= b + 4)
        parts: dict[str, float] = defaultdict(float)
        for x0, x1, c in segs:
            if c not in options:
                parts = None
                break
            # caractères (pas mots) dont le centre tombe dans le segment : deux
            # petites étiquettes voisines (« 1 » « 1 ») forment un seul MOT « 11 »
            txt = "".join(ch["text"] for ch in sorted(page.chars, key=lambda ch: ch["x0"])
                          if x0 <= (ch["x0"] + ch["x1"]) / 2 <= x1
                          and t - 2 <= (ch["top"] + ch["bottom"]) / 2 <= b + 2).strip()
            v_largeur = 100 * (x1 - x0) / largeur
            v = _valeur(txt.replace(" ", "")) if txt else None
            if v is None or abs(v - v_largeur) > 3:
                # pas d'étiquette, ou étiquette d'un segment voisin (Ifop
                # superpose « 1% » et « 4% » sur deux petits segments) : la
                # largeur du rectangle fait foi
                v = round(v_largeur)
            parts[options[c]] += v
        if parts is not None and etiquette:
            # barre pleine : une option absente de la barre vaut 0 (« Macron
            # 100 » chez les électeurs de Macron)
            table[etiquette] = {o: parts.get(o, 0.0) for o in set(options.values())}
    return table or None


# ----------------------------------------------------------------- tableaux
def _ancres(lignes, noms: dict[str, str]):
    """(nom, x centre, top) pour chaque nom trouvé dans une ligne de mots."""
    out = []
    for l in lignes:
        toks = [_norm(w["text"]) for w in l]
        for k, v in sorted(noms.items(), key=lambda kv: -len(kv[0])):
            ks = k.split()
            for i in range(len(toks) - len(ks) + 1):
                if toks[i:i + len(ks)] == ks:
                    ws = l[i:i + len(ks)]
                    out.append((v, (ws[0]["x0"] + ws[-1]["x1"]) / 2, ws[0]["top"]))
    return out


def lit_tableau(page, finalistes: dict[str, str], candidats_t1: dict[str, str],
                annee_passee: str | None = None) -> dict[str, dict[str, float]] | None:
    """`annee_passee` : les tris croisés complets (Ifop 2017) empilent des
    sections ; les lignes sous un intertitre « VOTE A LA PRÉSIDENTIELLE 2012 »
    croisent le vote PASSÉ et sont ignorées jusqu'à l'intertitre suivant."""
    words = page.extract_words()
    lignes = _lignes(words)
    nombres = [(l, [w for w in l if _valeur(w["text"]) is not None]) for l in lignes]

    # (a) électorats en COLONNES : une ligne d'en-tête porte >= 3 candidats T1
    for l in lignes:
        ancres = _ancres([l], candidats_t1)
        if len(ancres) < 3:
            continue
        proches = [ll for ll in lignes if abs(ll[0]["top"] - l[0]["top"]) <= 15]
        ancres = _ancres(proches, candidats_t1)
        cols = sorted({(v, round(x)) for v, x, _ in ancres}, key=lambda a: a[1])
        if len({v for v, _ in cols}) != len(cols):
            continue
        top = max(ll[0]["top"] for ll in proches)
        table: dict[str, dict[str, float]] = defaultdict(dict)
        bord = cols[0][1] - 20
        for ll, nums in nombres:
            if ll[0]["top"] <= top + 2:
                continue
            dans = [n for n in nums if n["x0"] > bord - 10]
            if len(dans) < len(cols) - 1:
                continue
            # libellé d'option à gauche, éventuellement sur plusieurs lignes
            # centrées sur celle des chiffres (« Emmanuel / MACRON »)
            y = ll[0]["top"]
            gauche = " ".join(w["text"] for w in words if w["x1"] < bord and abs(w["top"] - y) <= 14)
            opt = _option(gauche, finalistes)
            if opt is None:
                continue
            for n in dans:
                v, x = min(cols, key=lambda c: abs(c[1] - _centre(n)))
                if abs(x - _centre(n)) < 25:
                    table[v][opt] = table[v].get(opt, 0) + _valeur(n["text"])
        if len(table) >= 3:
            return dict(table)

    # (b) électorats en LIGNES : en-tête d'options (finalistes + non exprimés)
    for l in lignes:
        ancres = _ancres([l], finalistes)
        if len({v for v, _, _ in ancres}) != 2:
            continue
        cols = [(v, x) for v, x, _ in ancres]
        # troisième colonne (abstention…) : mot « aucun » à droite/gauche dans
        # les 2 lignes autour de l'en-tête
        voisins = [w for w in words if abs(w["top"] - l[0]["top"]) < 14
                   and (_AUCUN.search(w["text"]) or re.search(r"exprim", w["text"], re.I))
                   and all(abs(_centre(w) - x) > 25 for _, x in cols)]
        if voisins:
            cols.append(("Aucun", _centre(voisins[0])))
        table = {}
        passe = False
        for ll, nums in nombres:
            if ll[0]["top"] <= l[0]["top"] + 2:
                continue
            texte = " ".join(w["text"] for w in ll)
            intertitre = re.search(r"\b(19|20)\d\d\b", texte) or (texte.isupper() and len(texte) > 8)
            if not nums and annee_passee and intertitre:
                passe = annee_passee in texte   # intertitre daté : passé ou pas
                continue
            if passe:
                continue
            lab = " ".join(w["text"] for w in ll if w["x1"] < min(x for _, x in cols) - 25)
            el = _cherche(_norm(lab), candidats_t1)
            if len(el) != 1 and not _AUCUN.search(lab):
                continue
            parts = {}
            for n in nums:
                v, x = min(cols, key=lambda c: abs(c[1] - _centre(n)))
                if abs(x - _centre(n)) < 30:
                    parts[v] = parts.get(v, 0) + _valeur(n["text"])
            if len(parts) >= 2:
                table[lab] = parts
        if len(table) >= 3:
            return table
    return None


# ----------------------------------------------------------------- page / PDF
def _hypothese_de_page(texte: str, hypotheses: list[tuple[str, str]]) -> tuple[str, str] | None:
    """Le duel est dans le TITRE (4 premières lignes) : plus bas, les libellés
    d'électorats citent tous les candidats."""
    tete = _norm(" ".join([l for l in texte.splitlines() if l.strip()][:4]))
    bons = [h for h in hypotheses if all(f" {_norm(c)} " in f" {tete} " for c in h)]
    return bons[0] if len(bons) == 1 else None


def extrait_pdf(chemin: Path, hypotheses: list[tuple[str, str]], candidats_t1: list[str],
                notes: list[str], anciens: list[str] = (), annee_passee: str | None = None
                ) -> list[dict]:
    """`anciens` : candidats de l'élection PRÉCÉDENTE absents de celle-ci. Une
    page dont un électorat en cite un croise le vote PASSÉ (« vote en 2017 »),
    pas l'intention T1 : elle est écartée en entier."""
    import pdfplumber
    cands = {_norm(c): c for c in candidats_t1}
    out = []
    vus = set()
    with pdfplumber.open(chemin) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            texte = page.extract_text() or ""
            if not re.search(r"report|selon (l.intention de |le )?vote au (1er|premier)|intentions? de vote au (1er|premier)"
                             r"|(1er|premier) tour", texte, re.I):
                continue
            if not re.search(r"(2nd|second|2e|deuxi[eè]me) tour", texte, re.I):
                continue
            h = _hypothese_de_page(texte, hypotheses)
            if h is None:
                continue
            finalistes = {_norm(c): c for c in h}
            ou = f"{chemin.name} p{i}"
            table = lit_barres(page, finalistes)
            if table is None:
                table = lit_tableau(page, finalistes, cands, annee_passee)
            if not table:
                continue
            passes = {_norm(a): a for a in anciens}
            # nom de l'élection précédente ET année précédente sur la page :
            # croisement par vote passé. Le nom seul ne suffit pas (Bayrou,
            # testé au T1 2017 par Ifop en février, absent des lignes T1 que
            # Wikipedia relève pour ces vagues).
            rappel = [lab for lab in table if _cherche(_norm(lab), passes)] \
                if annee_passee is None or annee_passee in texte else []
            if rappel:
                notes.append(f"{ou} : croisement par vote passé ({rappel[0]!r}) -> page écartée")
                continue
            for lab, parts in _valide(table, finalistes, notes, ou).items():
                el = _cherche(_norm(lab), cands)
                if len(el) == 1:
                    el = el[0]
                elif _AUCUN.search(lab) and not el:
                    el = "Aucun"
                else:
                    if not re.search(r"ensemble|total|rappel", lab, re.I):
                        notes.append(f"{ou} : électorat {lab!r} non reconnu -> écarté")
                    continue
                cle = (h, el)
                if cle in vus:      # même croisement répété (synthèse + détail)
                    continue
                vus.add(cle)
                for opt, v in parts.items():
                    out.append({"hypothese": " – ".join(h),
                                "electorat_t1": el, "candidat_t2": opt, "part": v, "page": i})
    return out


def _fichier(annee: str, url: str) -> Path:
    return CACHE / f"{annee}_{hashlib.md5(url.encode()).hexdigest()[:10]}.pdf"


def intentions(annee: str) -> pd.DataFrame:
    """`intentions_{annee}_wiki.csv`, candidats ramenés au NOM DE FAMILLE : le
    CSV 2017 porte les noms complets (« Marine Le Pen »), ceux de 2022/2027
    les noms de famille (« Le Pen »). On ne coupe que les noms complets connus
    (`nom_complet`), jamais « Le Pen » en « Pen »."""
    it = pd.read_csv(ROOT / "data" / "parsed" / f"intentions_{annee}_wiki.csv")
    complets = {v: v.split(maxsplit=1)[-1] for v in nom_complet(annee).values()}
    it["candidat"] = it["candidat"].map(lambda c: complets.get(c, c))
    return it


def _duels(annee: str, it: pd.DataFrame | None = None) -> pd.DataFrame:
    it = intentions(annee) if it is None else it
    return it[(it["tour"] == "Deuxième tour") & (it["date_fin"] < DATES_T1[annee])]


def telecharge(annee: str, timeout: int = 60) -> dict[str, int]:
    """Met en cache (data/raw/pdf_reports/, hors dépôt) les PDF cités par les
    duels de 2nd tour. Repli sur web.archive.org quand le site ne répond plus
    ou que sa chaîne TLS est incomplète (ifop.com) ; la vérification TLS n'est
    jamais désactivée. Les pages HTML (BVA, Odoxa, Ipsos…) sont ignorées."""
    import requests
    CACHE.mkdir(parents=True, exist_ok=True)
    bilan = {"cache": 0, "direct": 0, "archive": 0, "echec": 0}
    for url in _duels(annee)["notice_url"].dropna().unique():
        f = _fichier(annee, url)
        if f.exists():
            bilan["cache"] += 1
            continue
        for mode, u in (("direct", url), ("archive", f"https://web.archive.org/web/{annee}id_/{url}")):
            try:
                r = requests.get(u, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
                if r.content[:4] == b"%PDF":
                    f.write_bytes(r.content)
                    bilan[mode] += 1
                    break
            except requests.RequestException:
                pass
        else:
            bilan["echec"] += 1
    return bilan


def build(annee: str) -> tuple[pd.DataFrame, list[str]]:
    it = intentions(annee)
    t1_cands = sorted(it.loc[it["tour"] == "Premier tour", "candidat"].dropna().unique())
    t2 = _duels(annee, it)
    precedent = ROOT / "data" / "historical" / f"results_{int(annee) - 5}.csv"
    anciens = []
    if precedent.exists():
        r = pd.read_csv(precedent)
        anciens = [n.split(maxsplit=1)[-1] for n in r.loc[r["tour"] == "Premier tour", "candidat"]]
    notes: list[str] = []
    rows = []
    for url, g in t2.groupby("notice_url"):
        f = _fichier(annee, url)
        if not f.exists():
            continue
        # ordre du CSV (« Macron – Le Pen », comme `reports_t2_*`), sans doublon
        hyps = {}
        for _, h in g.groupby("hypothese", dropna=False, sort=False):
            h = tuple(h["candidat"].unique())
            hyps.setdefault(frozenset(h), h)
        hyps = [h for h in hyps.values() if len(h) == 2]
        if not hyps:
            continue
        meta = g.iloc[0]
        # électorats possibles : candidats T1 testés par CE sondage (même
        # notice) ; un candidat de l'élection précédente qui n'y figure pas
        # signale un croisement par vote passé.
        propres = it.loc[(it["notice_url"] == url) & (it["tour"] == "Premier tour"), "candidat"].dropna().unique()
        if not len(propres) and pd.notna(meta["date_fin"]):
            # T1 cité sous une autre URL sur Wikipedia : candidats T1 du même
            # institut à ±30 jours (pas ceux de toute la campagne, où Hollande
            # et Sarkozy figurent en 2016 et masqueraient un croisement 2012)
            dt = pd.to_datetime(it["date_fin"], errors="coerce")
            d0 = pd.Timestamp(meta["date_fin"])
            proche = (it["tour"] == "Premier tour") & (it["institut"] == meta["institut"]) \
                & ((dt - d0).abs() <= pd.Timedelta(days=30))
            propres = it.loc[proche, "candidat"].dropna().unique()
        cands = list(propres) if len(propres) else t1_cands
        try:
            recs = extrait_pdf(f, hyps, cands, notes, [n for n in anciens if n not in set(cands)],
                               str(int(annee) - 5))
        except Exception as e:  # PDF corrompu : signalé, pas bloquant
            notes.append(f"{f.name} : illisible ({type(e).__name__})")
            continue
        for r in recs:
            rows.append({"notice": meta["notice"], "institut": meta["institut"],
                         "date_debut": meta["date_debut"], "date_fin": meta["date_fin"],
                         "echantillon": meta["echantillon"], "nature_t1": "intention",
                         "base_electorat": None, "source": url, **r})
    df = pd.DataFrame(rows, columns=COLUMNS)
    if not df.empty:
        df["echantillon"] = df["echantillon"].astype("Int64")
        complets = nom_complet(annee)
        df["electorat_t1"] = df["electorat_t1"].map(lambda n: complets.get(_norm(n), n))
        df = df.sort_values(["date_fin", "notice", "hypothese", "electorat_t1", "candidat_t2"]).reset_index(drop=True)
    return df, notes


def main():
    ap = argparse.ArgumentParser(description="Reports T1 -> T2 publiés avant le 1er tour (PDF des instituts)")
    ap.add_argument("--annee", required=True, choices=sorted(DATES_T1))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--telecharge", action="store_true", help="met d'abord les PDF en cache")
    a = ap.parse_args()
    if a.telecharge:
        print("PDF :", telecharge(a.annee))
    df, notes = build(a.annee)
    for n in notes:
        print("ATTENTION :", n)
    out = a.out or ROOT / "data" / "parsed" / f"reports_avant_t1_{a.annee}.csv"
    df.to_csv(out, index=False)
    print(f"{out} : {len(df)} lignes, {df['notice'].nunique()} sondages, "
          f"{df.groupby(['notice', 'hypothese']).ngroups if len(df) else 0} duels")


if __name__ == "__main__":
    main()
