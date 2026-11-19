"""Reports T1 -> T2 publiés avant le 1er tour (`sondages/reports_avant_t1.py`) :
schéma et invariants des CSV versionnés, et lecteurs sur des pages
synthétiques (le cache PDF n'est pas versionné)."""

from pathlib import Path

import pandas as pd
import pytest

from sondages import reports_avant_t1 as R

ROOT = Path(__file__).resolve().parents[1]
FICHIERS = sorted((ROOT / "data" / "parsed").glob("reports_avant_t1_*.csv"))


def test_les_fichiers_existent():
    assert {f.stem[-4:] for f in FICHIERS} >= {"2017", "2022"}


@pytest.mark.parametrize("f", FICHIERS, ids=lambda f: f.stem)
def test_schema_et_invariants(f):
    d = pd.read_csv(f)
    if d.empty:
        assert list(d.columns) == R.COLUMNS
        return
    assert list(d.columns) == R.COLUMNS
    annee = f.stem[-4:]
    # le schéma de reports_t2_* est inclus, plus nature_t1 et page
    t2 = ROOT / "data" / "parsed" / f"reports_t2_{annee}.csv"
    if t2.exists():
        assert set(pd.read_csv(t2).columns) - {"rolling"} <= set(d.columns)
    assert (d["nature_t1"] == "intention").all()
    assert (d["date_fin"] < R.DATES_T1[annee]).all()          # publié AVANT le T1
    assert d[["notice", "hypothese", "electorat_t1", "candidat_t2", "part", "source"]].notna().all().all()
    assert d["part"].between(0, 100).all()
    assert not d.duplicated(["source", "hypothese", "electorat_t1", "candidat_t2"]).any()
    fin = d["hypothese"].str.split(" – ")
    assert (fin.str.len() == 2).all()
    assert all(c in f_ + ["Aucun"] for c, f_ in zip(d["candidat_t2"], fin))
    # chaque électorat somme à ~100 sur les options du duel
    s = d.groupby(["source", "hypothese", "electorat_t1"])["part"].sum()
    assert s.between(R.SUM_LO, R.SUM_HI).all(), s[~s.between(R.SUM_LO, R.SUM_HI)]
    # un finaliste garde son propre électorat (colonne mal attribuée sinon)
    meme = pd.Series([e.endswith(c) for e, c in zip(d["electorat_t1"], d["candidat_t2"])], index=d.index)
    propre = d[meme & (d["candidat_t2"] != "Aucun")]
    assert (propre["part"] >= 75).all(), propre[propre["part"] < 75]


# ------------------------------------------------------------ pages factices
class _Page:
    def __init__(self, rects=(), chars=(), words=()):
        self.rects, self.chars, self._words = list(rects), list(chars), list(words)

    def extract_words(self, **_):
        return self._words


def _mot(text, x0, top, w=None):
    w = w or 5 * len(text)
    return {"text": text, "x0": x0, "x1": x0 + w, "top": top, "bottom": top + 8}


def _chars(text, x0, top):
    return [{"text": ch, "x0": x0 + 5 * i, "x1": x0 + 5 * i + 5, "top": top, "bottom": top + 8}
            for i, ch in enumerate(text)]


JAUNE, VIOLET, GRIS = (1.0, 0.75, 0.0), (0.44, 0.19, 0.63), (0.61, 0.62, 0.62)
FIN = {"macron": "Macron", "le pen": "Le Pen"}


def test_barres_etiquettes_collees_et_segment_absent():
    """Elabe 7-8/02/2022 : chez les électeurs de Macron, « 1 » et « 1 » voisins
    forment le MOT « 11 » (somme 110, rangée perdue) ; la lecture par
    caractère rend 98/1/1. Une barre pleine sans segment Le Pen -> 0."""
    g, d = 100.0, 500.0           # 400 pt = 100 %
    rangees = {  # étiquette : [(part, couleur, étiquette imprimée)]
        "Jean-Luc MELENCHON": [(30, JAUNE, "30"), (25, VIOLET, "25"), (45, GRIS, "45")],
        "Emmanuel MACRON": [(98, JAUNE, "98"), (1, VIOLET, "1"), (1, GRIS, "1")],
        "Valérie PECRESSE": [(49, JAUNE, "49"), (29, VIOLET, "29"), (22, GRIS, "22")],
        "Marine LE PEN": [(4, JAUNE, ""), (96, VIOLET, "96")],
    }
    rects, chars, words = [], [], []
    for k, (lab, segs) in enumerate(rangees.items()):
        top = 100 + 30 * k
        x = g
        words.append(_mot(lab, 10, top + 4, w=80))
        for part, coul, etiq in segs:
            x1 = x + 4 * part
            rects.append({"x0": x, "x1": x1, "top": top, "bottom": top + 15, "width": x1 - x,
                          "height": 15, "non_stroking_color": coul})
            if etiq:
                c0 = (x + x1) / 2 - 2.5 * len(etiq)
                chars += _chars(etiq, c0, top + 4)
            x = x1
    # deux étiquettes voisines collées en un seul mot (ce que fait pdfplumber)
    words.append(_mot("11", 492, 134, w=10))
    for i, (coul, lab) in enumerate([(JAUNE, "Emmanuel Macron"), (VIOLET, "Marine Le Pen"),
                                     (GRIS, "Abstention, vote blanc")]):
        x = 120 + 130 * i
        rects.append({"x0": x, "x1": x + 6, "top": 250, "bottom": 256, "width": 6, "height": 6,
                      "non_stroking_color": coul})
        words.append(_mot(lab, x + 10, 248))
    t = R.lit_barres(_Page(rects, chars, words), FIN)
    assert t["Emmanuel MACRON"] == {"Macron": 98, "Le Pen": 1, "Aucun": 1}
    assert t["Marine LE PEN"] == {"Macron": 4, "Le Pen": 96, "Aucun": 0}


def test_tableau_electorats_en_colonnes_libelles_sur_deux_lignes():
    """Harris : électorats en colonnes, en-tête « Éric ZEMMOUR » décalé d'une
    ligne, libellé « Emmanuel / MACRON » de part et d'autre des chiffres."""
    cands = {R._norm(c): c for c in ["Mélenchon", "Macron", "Le Pen", "Zemmour"]}
    xs = [200, 260, 320, 380]
    w = [_mot("MELENCHON", 180, 100, 40), _mot("MACRON", 245, 100, 30),
         _mot("LE", 305, 100, 10), _mot("PEN", 318, 100, 15), _mot("ZEMMOUR", 365, 90, 30)]
    w += [_mot("Emmanuel", 20, 125), _mot("MACRON", 20, 141)]
    w += [_mot(v, x - 5, 133, 10) for v, x in zip(["34", "97", "2", "8"], xs)]
    w += [_mot("Marine", 20, 170), _mot("LE", 55, 170), _mot("PEN", 70, 170)]
    w += [_mot(v, x - 5, 170, 10) for v, x in zip(["21", "2", "95", "84"], xs)]
    w += [_mot("Non", 20, 195), _mot("exprimés", 40, 195)]
    w += [_mot(v, x - 5, 205, 10) for v, x in zip(["45", "1", "3", "8"], xs)]
    t = R.lit_tableau(_Page(words=w), FIN, cands)
    assert t["Zemmour"] == {"Macron": 8, "Le Pen": 84, "Aucun": 8}
    assert t["Macron"] == {"Macron": 97, "Le Pen": 2, "Aucun": 1}


def test_validation_complete_une_case_vide_seulement_si_la_somme_est_pleine():
    notes = []
    t = {"Macron": {"Macron": 98, "Aucun": 2}, "Jadot": {"Macron": 60, "Aucun": 20}}
    v = R._valide(t, FIN, notes, "x")
    assert v == {"Macron": {"Macron": 98, "Le Pen": 0.0, "Aucun": 2}}
    assert len(notes) == 1 and "Jadot" in notes[0]


def test_hypothese_lue_dans_le_titre_seulement():
    texte = "Emmanuel Macron / Marine Le Pen\nReports de voix\nx\ny\nÉric ZEMMOUR 10 75 15\nValérie PECRESSE"
    hyps = [("Macron", "Le Pen"), ("Macron", "Zemmour"), ("Macron", "Pécresse")]
    assert R._hypothese_de_page(texte, hyps) == ("Macron", "Le Pen")


def test_section_vote_passe_ignoree_dans_les_tris_croises():
    """Ifop (tris croisés) : la section « VOTE A LA PRÉSIDENTIELLE 2012 »
    croise le vote PASSÉ ; seule la section d'intention T1 2017 est reprise."""
    cands = {R._norm(c): c for c in ["Mélenchon", "Macron", "Le Pen"]}
    w = [_mot("Emmanuel", 300, 50), _mot("Macron", 345, 50), _mot("Marine", 400, 50),
         _mot("Le", 435, 50), _mot("Pen", 448, 50)]
    y = 80
    for lab, vals in [("VOTE A LA PRÉSIDENTIELLE 2012 (1er tour)", None),
                      ("Jean-Luc Mélenchon", ("88", "12")), ("Marine Le Pen", ("6", "94")),
                      ("Emmanuel Macron", ("99", "1")),
                      ("INTENTION DE VOTE AU 1ER TOUR 2017", None),
                      ("Jean-Luc Mélenchon", ("70", "30")), ("Marine Le Pen", ("4", "96")),
                      ("Emmanuel Macron", ("98", "2"))]:
        x = 20
        for mot in lab.split():
            w.append(_mot(mot, x, y)); x += 5 * len(mot) + 4
        if vals:
            w += [_mot(vals[0], 345, y, 10), _mot(vals[1], 435, y, 10)]
        y += 20
    t = R.lit_tableau(_Page(words=w), FIN, cands, annee_passee="2012")
    assert t == {"Jean-Luc Mélenchon": {"Macron": 70, "Le Pen": 30},
                 "Marine Le Pen": {"Macron": 4, "Le Pen": 96},
                 "Emmanuel Macron": {"Macron": 98, "Le Pen": 2}}
