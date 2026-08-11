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
