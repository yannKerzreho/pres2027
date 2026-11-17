"""Tests du parseur Wikipedia (sondages/wiki.py) — hors réseau : extraits de
wikitext réels (copiés depuis la page 2027 live), pas d'appel API dans les tests.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # racine du repo
from sondages.wiki import (
    _MONTHS, _parse_dates, _top_level_split, _desamorce, parse_wikitables,
    extract_notices, build_dataset,
)
from sondages import schema


def test_months_index_correct():
    # bug réel rencontré : un dict construit par enumerate() avec des doublons
    # de variantes accentuées décalait tous les mois après février.
    assert _MONTHS["juillet"] == 7
    assert _MONTHS["décembre"] == 12
    assert _MONTHS["mars"] == 3
    assert _MONTHS["aout"] == 8
    assert _MONTHS["février"] == 2


def test_parse_dates_needs_year_hint_when_absent():
    assert _parse_dates("22-24 juin") == (None, None)
    assert _parse_dates("22-24 juin", year_hint="2026") == ("2026-06-22", "2026-06-24")
    assert _parse_dates("7 - 8 juillet 2026") == ("2026-07-07", "2026-07-08")
    assert _parse_dates("5 juin 2026") == ("2026-06-05", "2026-06-05")


def test_top_level_split_ignores_pipes_inside_templates_and_links():
    attrs, content = _top_level_split(
        ' rowspan="8" style="{{Sondeur|Ifop}}" |[https://x.fr/n.pdf Ifop]')
    assert "rowspan" in attrs
    assert "{{Sondeur|Ifop}}" not in content or content.strip().startswith("[")
    assert "commission" not in attrs.lower()  # l'URL n'a pas fui dans les attributs


def test_desamorce_unwraps_value_templates_not_dropped():
    text, url, linked_names, unknown = _desamorce(
        "'''{{blanc|36}}'''<br><small>[[Jordan Bardella|Bardella]]</small>")
    assert "36" in text
    assert "Bardella" in text
    assert linked_names == ["Bardella"]
    assert not unknown


def test_desamorce_flags_unknown_template():
    text, url, linked_names, unknown = _desamorce("8{{note|groupe=alpha|texte}}")
    assert "8" in text
    assert "note" in unknown


def test_desamorce_plusieurs_liens_meme_cellule():
    # Cas réel : OpinionWay 10-11 juin 2026, la cellule RN lie Bardella ET Le
    # Pen (même score valable pour les deux hypothèses) — cf. bug corrigé le
    # 2026-08-10 (seul le 1er lien était gardé, "Le Pen" disparaissait
    # silencieusement).
    text, url, linked_names, unknown = _desamorce(
        "'''{{blanc|34}}'''<br><small>[[Jordan Bardella|Bardella]] {{blanc|/}} "
        "[[Marine Le Pen|Le Pen]]</small>")
    assert linked_names == ["Bardella", "Le Pen"]


# --- Bloc Ifop réel (8 hypothèses via rowspan), copié depuis la page 2027 ----
_IFOP_MULTI_HYP = """
{| class="wikitable"
! rowspan=3 | Sondeur
! rowspan=3 | Date
! rowspan=3 | Échantillon
! | [[Fichier:a.jpg|50x50px]]
! | [[Fichier:b.jpg|50x50px]]
|-
! scope=col | [[Jean-Luc Mélenchon|Mélenchon]]<br><small>([[La France insoumise|LFI]])</small>
! scope=col | [[Gabriel Attal|Attal]]<br><small>([[Renaissance (parti)|RE]])</small>
|- style="line-height:5px;"
| {{Infobox Parti politique français/couleurs|LFI}} |
| {{Infobox Parti politique français/couleurs|RE}} |
|-
| rowspan="2" style="{{Sondeur|Ifop}}" |[https://www.commission-des-sondages.fr/notices/files/notices/2026/juin/1-ifop.pdf Ifop]
| rowspan="2" |22-24 juin
| rowspan="2" |{{formatnum:1415}}
|40
|60
|-
| colspan="2" |45<br><small>'''[[François Hollande|Hollande]] ([[Parti socialiste (France)|PS]])'''</small>
|}
"""


def test_extract_notices_multi_hypothesis_rowspan():
    grids = parse_wikitables(_IFOP_MULTI_HYP)
    assert len(grids) == 1
    warnings = []
    records = extract_notices(grids[0], "Premier tour", None, warnings)
    by_hyp = {}
    for r in records:
        by_hyp.setdefault(r["hypothese"], []).append(r)

    assert len(by_hyp) == 2                              # 2 sous-lignes rowspan -> 2 hypothèses
    base = by_hyp["variante_1"]
    assert {r["candidat"] for r in base} == {"Mélenchon", "Attal"}
    assert sum(r["intention"] for r in base) == 100.0

    subst = by_hyp["variante_2"]
    assert sum(r["intention"] for r in subst) == 45.0     # cellule fusionnée comptée UNE fois
    assert any(r["candidat"] == "Hollande" for r in subst)

    for r in records:
        assert r["notice_url"] == ("https://www.commission-des-sondages.fr/"
                                   "notices/files/notices/2026/juin/1-ifop.pdf")
        assert r["date_debut"] == "2026-06-22" and r["date_fin"] == "2026-06-24"
        assert r["echantillon"] == 1415
        assert r["institut"] == "Ifop"


# --- En-tête fusionné « sauf précision contraire » (page restructurée, oct. 2026)
# La colonne Glucksmann occupe DEUX colonnes dès l'en-tête ; une cellule
# fusionnée sans nom vaut donc Glucksmann. Seule une cellule qui nomme
# quelqu'un d'autre est une substitution.
_ENTETE_FUSIONNE = """
{| class="wikitable"
! rowspan=3 | Sondeur
! rowspan=3 | Date
! rowspan=3 | Échantillon
! | [[Fichier:a.jpg|50x50px]]
! colspan=2 | [[Fichier:b.jpg|50x50px]]
|-
! scope=col | [[Jean-Luc Mélenchon|Mélenchon]]<br><small>([[La France insoumise|LFI]])</small>
! colspan=2 scope=col | [[Raphaël Glucksmann|Glucksmann]]<br><small>([[Place publique (parti politique)|PP]])</small>
|- style="line-height:5px;"
| {{Infobox Parti politique français/couleurs|LFI}} |
| colspan=2 {{Infobox Parti politique français/couleurs|PP}} |
|-
| rowspan="2" style="{{Sondeur|Verian}}" |[https://www.commission-des-sondages.fr/notices/files/notices/2026/juillet/1-verian.pdf Verian]
| rowspan="2" |8-10 juillet
| rowspan="2" |{{formatnum:1047}}
|60
|colspan=2 |40
|-
|55
|colspan=2 |45<br><small>'''[[François Hollande|Hollande]] ([[Parti socialiste (France)|PS]])'''</small>
|}
"""


def test_cellule_fusionnee_sans_nom_sous_entete_fusionne_vaut_l_entete():
    warnings = []
    records = extract_notices(parse_wikitables(_ENTETE_FUSIONNE)[0], "Premier tour", None, warnings)
    by_hyp = {}
    for r in records:
        by_hyp.setdefault(r["hypothese"], {})[r["candidat"]] = r["intention"]
    assert by_hyp["variante_1"] == {"Mélenchon": 60.0, "Glucksmann": 40.0}
    assert by_hyp["variante_2"] == {"Mélenchon": 55.0, "Hollande": 45.0}
    assert not [w for w in warnings if "substitution" in w]


# --- Colonne RN à en-tête générique (bug réel : "Candidat RN" avalait tout) --
# L'en-tête ne nomme PAS le candidat (pas encore arrêté au moment du sondage) ;
# seule la cellule le fait, via un wikilien — sur une colonne NORMALE (colspan=1,
# pas une substitution). Sans le repli par wikilien, `candidat` valait
# "CandidatRN" pour toutes les hypothèses testant Le Pen OU Bardella.
_RN_GENERIC_HEADER = """
{| class="wikitable"
! rowspan=3 | Sondeur
! rowspan=3 | Date
! rowspan=3 | Échantillon
! | [[Fichier:a.jpg|50x50px]]
! | [[Fichier:b.jpg|50x50px]]
|-
! scope=col | [[Gabriel Attal|Attal]]<br><small>([[Renaissance (parti)|RE]])</small>
! scope=col | Candidat<br>[[Rassemblement national|RN]]
|- style="line-height:5px;"
| {{Infobox Parti politique français/couleurs|RE}} |
| {{Infobox Parti politique français/couleurs|RN}} |
|-
| style="{{Sondeur|Ifop}}" |[https://www.commission-des-sondages.fr/notices/files/notices/2026/juin/2-ifop.pdf Ifop]
| 22-24 juin 2026
| {{formatnum:1415}}
|64
| {{Infobox Parti politique français/couleurs|RN}} |'''{{blanc|36}}'''<br><small>'''[[Jordan Bardella|{{blanc|Bardella}}]]'''</small>
|}
"""


def test_extract_notices_rn_generic_header_uses_cell_wikilink():
    grids = parse_wikitables(_RN_GENERIC_HEADER)
    warnings = []
    records = extract_notices(grids[0], "Premier tour", None, warnings)
    by_cand = {r["candidat"]: r["intention"] for r in records}
    assert by_cand == {"Attal": 64.0, "Bardella": 36.0}   # PAS "CandidatRN"
    assert sum(by_cand.values()) == 100.0


# --- Cellule RN à double lien (bug réel : OpinionWay 10-11 juin 2026) --------
# Wikipedia indique explicitement que le même score (34) vaut pour Bardella
# ET Le Pen sur cette hypothèse (le sondeur ne distingue pas les deux) — avant
# le fix, seul "Bardella" (1er lien) survivait, "Le Pen" disparaissait sur les
# 3 hypothèses de ce sondage.
_RN_DOUBLE_LIEN = """
{| class="wikitable"
! rowspan=3 | Sondeur
! rowspan=3 | Date
! rowspan=3 | Échantillon
! | [[Fichier:a.jpg|50x50px]]
! | [[Fichier:b.jpg|50x50px]]
|-
! scope=col | Candidat<br>[[Rassemblement national|RN]]
! scope=col | [[Édouard Philippe|Philippe]]<br><small>([[Horizons (parti politique)|HOR]])</small>
|- style="line-height:5px;"
| {{Infobox Parti politique français/couleurs|RN}} |
| {{Infobox Parti politique français/couleurs|HOR}} |
|-
| style="{{Sondeur|OpinionWay}}" |[https://www.commission-des-sondages.fr/notices/files/notices/2026/juin/3-ow.pdf OpinionWay]
| 10-11 juin 2026
| {{formatnum:963}}
| '''{{blanc|34}}'''<br><small>[[Jordan Bardella|{{blanc|Bardella}}]] {{blanc|/}} [[Marine Le Pen|{{blanc|Le Pen}}]]</small>
|48
|}
"""


def test_extract_notices_cellule_double_lien_devient_deux_hypotheses():
    grids = parse_wikitables(_RN_DOUBLE_LIEN)
    warnings = []
    records = extract_notices(grids[0], "Premier tour", None, warnings)
    by_hyp = {}
    for r in records:
        by_hyp.setdefault(r["hypothese"], {})[r["candidat"]] = r["intention"]

    assert len(by_hyp) == 2, f"attendu 2 hypothèses (Bardella + Le Pen), trouvé {list(by_hyp)}"
    variantes = list(by_hyp.values())
    assert {"Bardella": 34.0, "Philippe": 48.0} in variantes
    assert {"Le Pen": 34.0, "Philippe": 48.0} in variantes
    # aucune hypothèse ne double-compte le score RN (somme ~100%, pas ~134%)
    for cands in variantes:
        assert sum(cands.values()) == 82.0


# --- Ligne de RAPPEL (résultat officiel, pas un sondage) ---------------------
_RESULTS_ROW = """
{| class="wikitable"
! rowspan=3 | Sondeur
! rowspan=3 | Date
! rowspan=3 | Échantillon
! | [[Fichier:a.jpg|50x50px]]
|-
! scope=col | [[Jean-Luc Mélenchon|Mélenchon]]
|- style="line-height:5px;"
| {{Infobox Parti politique français/couleurs|LFI}} |
|-
| [https://www.resultats-elections.interieur.gouv.fr/presidentielle-2022/FE.html Résultats]
| 10 avril 2022
| {{formatnum:1000}}
|22
|}
"""


def test_extract_notices_rejects_unrecognized_institut():
    grids = parse_wikitables(_RESULTS_ROW)
    warnings = []
    records = extract_notices(grids[0], "Premier tour", None, warnings)
    assert records == []                                  # écarté, pas publié sous un faux institut
    assert any("non reconnu" in w for w in warnings)


# --- Table 2e tour simple (1 ligne = 1 sondage, hypothèse = titre section) --
_SECOND_TOUR = """
{| class="wikitable"
! rowspan=3 | Sondeur
! rowspan=3 | Dates
! rowspan=3 | Échantillon
! | x
! | y
|-
! scope=col | [[Gabriel Attal|Attal]]<br><small>([[Renaissance (parti)|RE]])</small>
! scope=col | [[Marine Le Pen|Le Pen]]<br><small>([[Rassemblement national|RN]])</small>
|- style="line-height:5px;"
| {{Infobox Parti politique français/couleurs|RE}} |
| {{Infobox Parti politique français/couleurs|RN}} |
|-
| style="{{Sondeur|Elabe}}" |[https://www.commission-des-sondages.fr/notices/files/notices/2026/juillet/2-elabe.pdf Elabe]
| 9 - 10 juillet 2026
| {{formatnum:1503}}
| 46
| {{Infobox Parti politique français/couleurs|RN}} |'''{{blanc|54}}'''
|-
| colspan="5" style="background-color:#CECECE" | '''Un événement politique.'''
|}
"""


def test_extract_notices_second_tour_hypothese_is_section_title():
    grids = parse_wikitables(_SECOND_TOUR)
    warnings = []
    records = extract_notices(grids[0], "Deuxième tour", "Hypothèse Attal – Le Pen", warnings)
    assert len(records) == 2                              # la ligne bannière (colspan=5) est écartée
    assert all(r["hypothese"] == "Hypothèse Attal – Le Pen" for r in records)
    assert {r["candidat"]: r["intention"] for r in records} == {"Attal": 46.0, "Le Pen": 54.0}
    assert sum(r["intention"] for r in records) == 100.0


def test_build_dataset_live_page_matches_schema_contract():
    intentions, report = build_dataset()
    assert not intentions.empty
    schema.validate(intentions)                            # ne lève pas
    assert report["n_kept"] > 500
    assert report["n_rejected_groups"] < report["n_kept"] * 0.05   # garde-fou : peu de rejets


# --- Fiabilisation nov. 2026 : défauts trouvés sur la page 2027 live ---------
# Colonne « Autres » listant PLUSIEURS petits candidats séparés par <hr>
# (Cluster17 9-10 novembre 2026, variante 4) : seul le 1er couple était lu, la
# somme tombait à 85 et l'hypothèse entière était rejetée. Inclut aussi un
# lien dont le TEXTE est balisé (« [[Xavier Bertrand|<small>Bertrand</small>]] »,
# YouGov 17-21 septembre 2026), qui donnait le candidat « <small>Bertrand</small> ».
_AUTRES_HR = """
{| class="wikitable"
! rowspan=3 | Sondeur
! rowspan=3 | Date
! rowspan=3 | Échantillon
! | a
! | b
|-
! scope=col | [[Marine Le Pen|Le Pen]]
! scope=col | Autres
|- style="line-height:5px;"
| {{Infobox Parti politique français/couleurs|RN}} |
| |
|-
| rowspan="1" style="{{Sondeur|Cluster17}}" | [https://www.commission-des-sondages.fr/notices/files/notices/2026/septembre/10251-pres-cluster-17.pdf Cluster17]
| rowspan="1" | 31 août - {{1er}} septembre
| rowspan="1" | {{Formatnum|1711}}
|90
|1<br><small>'''[[François Asselineau|Asselineau]]'''</small><hr>8<br>'''[[Xavier Bertrand|<small>Bertrand</small>]]'''<hr>1<br><small>'''[[Dominique de Villepin|Villepin]]'''</small>
|}
"""


def test_cellule_autres_multi_candidats_hr():
    warnings = []
    records = extract_notices(parse_wikitables(_AUTRES_HR)[0], "Premier tour", None, warnings)
    assert {r["candidat"]: r["intention"] for r in records} == {
        "Le Pen": 90.0, "Asselineau": 1.0, "Bertrand": 8.0, "Villepin": 1.0}
    r = records[0]
    # {{1er}} : la fin du terrain est le 1er septembre, pas le 31 août
    assert (r["date_debut"], r["date_fin"]) == ("2026-08-31", "2026-09-01")
    assert r["notice"] == "Cluster17 31 août - 1 septembre (WP)"
    assert r["echantillon"] == 1711                     # {{Formatnum|…}} à pipe
    assert not [w for w in warnings if "template inconnu" in w]


def test_desambiguise_notices_meme_libelle_deux_pdf():
    from sondages.wiki import desambiguise_notices
    import pandas as pd
    base = "https://www.commission-des-sondages.fr/notices/files/notices/2026/octobre/"
    df = pd.DataFrame({
        "notice": ["OW 13-14 octobre (WP)"] * 3 + ["Ifop 1-2 octobre (WP)"],
        "notice_url": [base + "10318-presitrack.pdf", base + "10318-presitrack.pdf",
                       base + "10320-autre.pdf", base + "10300-ifop.pdf"],
    })
    out = desambiguise_notices(df)
    assert out["notice"].tolist() == ["OW 13-14 octobre (WP) n°10318"] * 2 + [
        "OW 13-14 octobre (WP) n°10320", "Ifop 1-2 octobre (WP)"]   # l'autre libellé intact


def test_notices_disparues_compare_au_dernier_snapshot(tmp_path):
    import json
    import pandas as pd
    from sondages.wiki import notices_disparues
    mdir = tmp_path / "gp-pooling"
    mdir.mkdir()
    (mdir / "2026-11-01.json").write_text(json.dumps({"polls": [{"notice": "ancien"}]}))
    (mdir / "2026-11-02.json").write_text(json.dumps(
        {"polls": [{"notice": "A"}, {"notice": "B"}, {"notice": "C"}]}))
    (mdir / "index.json").write_text("[]")
    df = pd.DataFrame({"notice": ["A", "B", "D"]})
    disparues, n_ref = notices_disparues(df, tmp_path)
    assert disparues == {"C"} and n_ref == 3            # seul le DERNIER snapshot compte


def test_boite_vote_blanc_exprime_est_hors_contrat():
    # Ifop 9-11 septembre 2026 republié avec « Vote blanc » en option : même PDF
    # que la ligne du tableau principal -> hypothèse à 200 % côté site.
    from sondages.wiki import _EXCLUDE_BOITE
    assert _EXCLUDE_BOITE.search("Sondages comptant le [[vote blanc]] comme un vote exprimé")
    assert not _EXCLUDE_BOITE.search("Sondages réalisés sur la période juillet-septembre 2026")
    assert not _EXCLUDE_BOITE.search("Sondages impliquant Emmanuel Macron, qui ne peut plus se présenter")


def test_doublons_de_notice_signale_un_pdf_sous_deux_libelles():
    import pandas as pd
    from sondages.wiki import doublons_de_notice
    df = pd.DataFrame({
        "tour": ["Premier tour"] * 3 + ["Deuxième tour"],
        "notice": ["Ifop 9-11 septembre (WP)", "Ifop 9-11 septembre 2026 (WP)",
                   "Elabe 1-2 octobre (WP)", "Ifop 9-11 septembre 2026 (WP)"],
        "notice_url": ["u/10292.pdf", "u/10292.pdf", "u/10300.pdf", "u/10292.pdf"],
    })
    w = doublons_de_notice(df)
    assert len(w) == 1 and "10292" not in w[0] and "Premier tour" in w[0]


# Mise en page de la page 2027 au 2026-11-17 (extrait réduit, réel) : photo
# `[[Fichier:...]]` en tête de chaque colonne, noms DANS la ligne « Sondeur »,
# bande de couleurs ensuite, et « Autres » en boîte déroulante multi-lignes.
_WT_2026_11_17 = """{| class="wikitable"
! rowspan=2 | Sondeur
! rowspan=2 | Date
! rowspan=2 | Échantillon
! [[Fichier:Arthaud.jpg|alt=Nathalie Arthaud|50x50px]]<br>[[Nathalie Arthaud|Arthaud]]<br><small>([[Lutte ouvrière|LO]])</small>
! [[Fichier:Le Pen.jpg|50x50px]]<br>[[Marine Le Pen|Le Pen]]<br><small>([[Rassemblement national|RN]])</small>
! Autres
|- style="line-height:5px;"
| {{Infobox Parti politique français/couleurs|LO}} |
| {{Infobox Parti politique français/couleurs|RN}} |
| {{Infobox Parti politique français/couleurs|DIV}} |
|-
| [https://www.commission-des-sondages.fr/notices/files/notices/2026/novembre/10375-pres-iv-elabe-14-novembre.pdf Elabe]
| 12-13 novembre
| 1500
|60
|34
| {{Boîte déroulante|titre=|align=center|contenu=
{{Nowrap|[[Natacha Polony|Polony]] 1,5}}<br/>{{Nowrap|[[Teddy Riner|Riner]] 4,5}} }}
|}"""


def test_mise_en_page_2026_11_17_photos_boite_deroulante():
    grid = parse_wikitables(_WT_2026_11_17)[0]
    warnings = []
    recs = extract_notices(grid, "Premier tour", None, warnings, "2026")
    got = {r["candidat"]: r["intention"] for r in recs}
    assert got == {"Arthaud": 60.0, "Le Pen": 34.0, "Polony": 1.5, "Riner": 4.5}
    assert not [w for w in warnings if "inconnu" in w]


# --- 17 nov. 2026 : page 2027 restructurée, dates 2017 -----------------------
def test_dates_2017_mois_et_annee_dans_la_legende():
    from sondages.wiki import legende_mois
    assert legende_mois("{|\n|+ {{date|Mars 2017}}\n|-\n|}") == ("2017", 3)
    assert legende_mois("{|\n|+ Sondages divers\n|}") is None
    h = ("2017", 3)
    assert _parse_dates("du 15 au 17", month_hint=h) == ("2017-03-15", "2017-03-17")
    assert _parse_dates("le 20", month_hint=h) == ("2017-03-20", "2017-03-20")
    assert _parse_dates("du 28 au 2", month_hint=h) == ("2017-02-28", "2017-03-02")
    assert _parse_dates("du au 2", month_hint=h) == (None, "2017-03-02")
    assert _parse_dates("du 3 au 5", month_hint=("2017", 1)) == ("2017-01-03", "2017-01-05")
    assert _parse_dates("du 30 au 2", month_hint=("2017", 1)) == ("2016-12-30", "2017-01-02")
    assert _parse_dates("27 fév. - 3 mars", "2017") == ("2017-02-27", "2017-03-03")
    assert _parse_dates("28 déc. - 3 janv.", "2017") == ("2016-12-28", "2017-01-03")
    assert _parse_dates("14 et 15 décembre 2015") == ("2015-12-14", "2015-12-15")
    assert _parse_dates("au 5 mars 2017") == (None, "2017-03-05")
    # sans légende, une cellule sans mois reste sans date (jamais devinée)
    assert _parse_dates("du 15 au 17", "2017") == (None, None)


def test_complete_dates_ne_touche_que_les_dates_vides():
    import pandas as pd
    from sondages.complete_dates import complete_dates
    cols = ["notice", "notice_url", "date_debut", "date_fin", "date_notice", "intention"]
    ancien = pd.DataFrame([
        ["BVA du 15 au 17 (WP)", "u1", None, None, None, 20.0],
        ["Ifop 2-3 mars (WP)", "u2", "2017-03-02", "2017-03-03", "2017-03-03", 30.0],
        ["Elabe du 1 au 2 (WP)", "u3", None, None, None, 10.0],   # ambigu dans le nouveau
    ], columns=cols)
    nouveau = pd.DataFrame([
        ["BVA du 15 au 17 (WP) n°2", "u1", "2017-03-15", "2017-03-17", "2017-03-17", 21.0],
        ["Ifop 2-3 mars (WP)", "u2", "2017-03-01", "2017-03-09", "2017-03-09", 30.0],
        ["Elabe du 1 au 2 (WP) n°1", "u3", "2017-02-01", "2017-02-02", None, 10.0],
        ["Elabe du 1 au 2 (WP) n°2", "u3", "2017-03-01", "2017-03-02", None, 10.0],
    ], columns=cols)
    out, _ = complete_dates(ancien, nouveau)
    assert out.loc[0, ["date_debut", "date_fin", "date_notice"]].tolist() == ["2017-03-15", "2017-03-17", "2017-03-17"]
    assert out.loc[0, "intention"] == 20.0                     # valeur d'origine conservée
    assert out.loc[1, "date_fin"] == "2017-03-03"              # date existante jamais écrasée
    assert pd.isna(out.loc[2, "date_fin"])                     # ambigu -> reste vide


# Pages 2002-2012 : pas de colonne Échantillon, pas de lien de notice,
# {{date-|…}}, sous-sections chronologiques sous le duel.
_ANCIEN_T2 = """{| class="wikitable"
|- bgcolor=lightgray
! rowspan="3" | Sondeur
! rowspan="3" | Date
! [[Fichier:Sarkozy.jpg|100x150px]]
! [[Fichier:Royal.jpg|100x140px]]
|-
! [[Nicolas Sarkozy]]
! [[Ségolène Royal]]
|-
| {{Infobox Parti politique français/couleurs|UMP}} |
| {{Infobox Parti politique français/couleurs|PS}} |
|-
| style="{{Sondeur|Sofres}}" |Sofres
| {{date-|4 mai 2007}}
| {{Infobox Parti politique français/couleurs|UMP}} | {{blanc|'''54,5 %'''}}
| 45,5 %
|}"""


def test_page_ancienne_sans_echantillon_ni_lien():
    recs = extract_notices(parse_wikitables(_ANCIEN_T2)[0], "Deuxième tour", "Sarkozy — Royal", [],
                           url_repli="https://fr.wikipedia.org/wiki/X")
    assert [(r["candidat"], r["intention"]) for r in recs] == [("Nicolas Sarkozy", 54.5), ("Ségolène Royal", 45.5)]
    assert recs[0]["echantillon"] is None and recs[0]["date_fin"] == "2007-05-04"
    assert recs[0]["notice_url"] == "https://fr.wikipedia.org/wiki/X"


def test_hypothese_remonte_les_sous_sections_chronologiques():
    from sondages.wiki import _titre_hypothese
    secs = [{"number": "2", "line": "Second tour"},
            {"number": "2.1", "line": "François Hollande - Nicolas Sarkozy"},
            {"number": "2.1.2", "line": "2011"},
            {"number": "2.2.1", "line": "Hypothèse Attal – Le Pen"}]
    assert _titre_hypothese(secs, secs[2]) == "François Hollande - Nicolas Sarkozy"
    assert _titre_hypothese(secs, secs[3]) == "Hypothèse Attal – Le Pen"


def test_annee_d_url_ignore_les_identifiants_de_fiche():
    t = _ANCIEN_T2.replace("|Sofres\n", "|[http://www.bva.fr/data/sondage/sondage_fiche/1977/x.pdf BVA]\n").replace(
        "{{date-|4 mai 2007}}", "5-7 avr.")
    recs = extract_notices(parse_wikitables(t)[0], "Premier tour", None, [], year_hint="2017")
    assert recs[0]["date_fin"] == "2017-04-07"


_REPORTS = """{| class="wikitable"
! rowspan="4" | Sondeur
! rowspan="4" | Date
| rowspan="27" style="background-color:#CECECE;" |
! colspan=3 | [[Jean-Luc Mélenchon]]<br />(1{{er}} tour : 19,58 %)
|- style="line-height:5px;"
| colspan=3 {{Infobox Parti politique français/couleurs|FI}} |
|-
! style="width:70px;" | [[Emmanuel Macron|Macron]] || style="width:70px;" | Aucun || style="width:70px;" | [[Marine Le Pen|Le Pen]]
|- style="line-height:5px;"
| {{Infobox Parti politique français/couleurs|EM}} | || {{Infobox Parti politique français/couleurs|DIV}} | || {{Infobox Parti politique français/couleurs|FN}} |
|-
| style="{{Sondeur|Ipsos}}" | {{nobr|[http://ipsos.fr/x.pdf Ipsos]}}
| {{nobr|30 avr. - 1{{er}} mai}}
| {{Infobox Parti politique français/couleurs|EM}} | 48 % || 38 % || 14 %
|}"""


def test_reports_de_voix():
    from sondages.reports import extract_reports
    recs = extract_reports(parse_wikitables(_REPORTS)[0], "2017", [])
    assert [(r["electorat_t1"], r["candidat_t2"], r["part"]) for r in recs] == [
        ("Jean-Luc Mélenchon", "Macron", 48.0), ("Jean-Luc Mélenchon", "Aucun", 38.0),
        ("Jean-Luc Mélenchon", "Le Pen", 14.0)]
    assert recs[0]["hypothese"] == "Macron – Le Pen"
    assert (recs[0]["date_debut"], recs[0]["date_fin"]) == ("2017-04-30", "2017-05-01")
    assert recs[0]["source"] == "http://ipsos.fr/x.pdf" and recs[0]["rolling"] is False


def test_ajoute_notices_absentes_sans_doublon():
    import pandas as pd
    from sondages.complete_dates import ajoute_notices_absentes
    cols = ["notice", "notice_url", "institut", "date_fin", "tour", "intention"]
    ancien = pd.DataFrame([["Ifop 2-4 mars (WP)", "u1", "Ifop", "2017-03-04", "Premier tour", 20.0]], columns=cols)
    nouveau = pd.DataFrame([
        ["Ifop du 2 au 4 (WP)", "u1", "Ifop", "2017-03-04", "Premier tour", 20.0],    # même URL
        ["Ifop du 1 au 4 (WP)", "u9", "Ifop", "2017-03-04", "Premier tour", 21.0],    # même institut+date
        ["BVA 5-7 avr. (WP)", "u2", "BVA", "2017-04-07", "Premier tour", 1.0],         # nouveau
    ], columns=cols)
    out, notes = ajoute_notices_absentes(ancien, nouveau)
    assert out["notice"].tolist() == ["Ifop 2-4 mars (WP)", "BVA 5-7 avr. (WP)"]


def test_corrige_dates_apres_scrutin():
    import pandas as pd
    from sondages.complete_dates import corrige_dates_apres_scrutin
    cols = ["notice", "notice_url", "date_debut", "date_fin", "date_notice"]
    ancien = pd.DataFrame([["Harris 28-31 décembre (WP)", "u", "2022-12-28", "2022-12-31", "2022-12-31"],
                           ["Ifop 2-3 mars (WP)", "v", "2022-03-02", "2022-03-03", "2022-03-03"]], columns=cols)
    nouveau = pd.DataFrame([["Harris 28-31 décembre (WP)", "u", "2021-12-28", "2021-12-31", "2021-12-31"],
                            ["Ifop 2-3 mars (WP)", "v", "2022-03-01", "2022-03-09", "2022-03-09"]], columns=cols)
    out, _ = corrige_dates_apres_scrutin(ancien, nouveau, "2022")
    assert out.loc[0, ["date_debut", "date_fin", "date_notice"]].tolist() == ["2021-12-28", "2021-12-31", "2021-12-31"]
    assert out.loc[1, "date_fin"] == "2022-03-03"     # date plausible jamais touchée
