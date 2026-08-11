"""Tests du parseur de notices PDF (Phase 2).

Fixtures : deux vraies notices Ifop (documents publics de la Commission des
sondages, via l'index NSPPolls) :
  - ifop_intentions_ok.pdf          : sondage d'intentions 1er tour (4 hypothèses) ;
  - ifop_barometre_no_intentions.pdf : baromètre d'ambition (SANS intentions de vote).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # racine du repo
from sondages.parse_pdf import parse_notice, detect_institut

FIX = Path(__file__).resolve().parent / "fixtures" / "pdf"
OK = FIX / "ifop_intentions_ok.pdf"
BARO = FIX / "ifop_barometre_no_intentions.pdf"

pytestmark = pytest.mark.skipif(not OK.exists(), reason="fixtures PDF absentes")


def test_detect_institut():
    assert detect_institut("9912 Pres IFOP JDD 30 mars") == "Ifop"
    assert detect_institut("Pres Toluna Harris Interactive RTL") == "Harris Interactive"
    assert detect_institut("pres opinionway les echos") == "Opinion Way"
    assert detect_institut("Pres Odoxa Public Senat") == "Odoxa"
    assert detect_institut("pres ipsos bva le parisien") == "Ipsos"
    assert detect_institut("un truc sans institut connu") is None


# (fixture, institut attendu, nb mini d'hypothèses)
_OK_FIXTURES = [
    ("ifop_intentions_ok.pdf", "Ifop", 3),
    ("odoxa_intentions_ok.pdf", "Odoxa", 2),
    ("ipsos_intentions_ok.pdf", "Ipsos", 6),
    ("harris_intentions_ok.pdf", "Harris Interactive", 2),        # layout ancien (2 blocs)
    ("harris_recap_positionnel_ok.pdf", "Harris Interactive", 5),  # layout récent (grille N cols)
]


@pytest.mark.parametrize("fixture,institut,min_hyp", _OK_FIXTURES)
def test_parsers_intentions_ok(fixture, institut, min_hyp):
    path = FIX / fixture
    if not path.exists():
        pytest.skip(f"fixture {fixture} absente")
    n = parse_notice(path)
    assert n.institut == institut
    assert n.status == "ok"
    # chaque hypothèse retenue somme à ~100 % (garde-fou de validation)
    hyps = {}
    for r in n.records:
        hyps.setdefault((r["tour"], r["hypothese"]), 0.0)
        hyps[(r["tour"], r["hypothese"])] += r["intention"]
    assert len(hyps) >= min_hyp
    assert all(95 <= s <= 105 for s in hyps.values())
    # présence de candidats plausibles du cycle 2027
    cands = {r["candidat"] for r in n.records}
    assert any(c in cands for c in ("Jordan Bardella", "Marine Le Pen", "Jean-Luc Mélenchon"))


def test_ifop_intentions_ok():
    n = parse_notice(OK)
    assert n.institut == "Ifop"
    assert n.status == "ok"
    # métadonnées
    assert n.methode == "internet"
    assert n.echantillon and n.echantillon > 900
    assert n.date_debut and n.date_fin
    # plusieurs hypothèses, chacune sommant ~100 %
    hyps = {}
    for r in n.records:
        hyps.setdefault(r["hypothese"], 0.0)
        hyps[r["hypothese"]] += r["intention"]
    assert len(hyps) >= 3
    assert all(95 <= s <= 105 for s in hyps.values())


def test_ifop_intentions_contient_candidats_attendus():
    n = parse_notice(OK)
    cands = {r["candidat"] for r in n.records}
    for expected in ("Marine Le Pen", "Jean-Luc Mélenchon", "Nathalie Arthaud"):
        assert expected in cands


def test_pas_de_ligne_demographique_ni_region():
    # Régression : les croisements (âge, CSP, région) ne doivent pas passer.
    n = parse_notice(OK)
    for r in n.records:
        assert "ans" not in r["candidat"].lower()
        assert "," not in r["candidat"]
        assert not any(ch.isdigit() for ch in r["candidat"])


def test_barometre_est_no_intentions():
    # Un baromètre d'ambition n'a pas d'intentions de vote 1er tour : le parseur
    # ne doit pas halluciner de candidats (piège des tables de rappel/opinion).
    n = parse_notice(BARO)
    assert n.status == "no_intentions"
    assert n.records == []


def test_institut_non_outille_flag_manuel():
    # Un institut sans parseur dédié -> statut explicite, pas d'erreur.
    n = parse_notice(OK, source_name="9999 pres BVA quelque chose")
    assert n.status == "unsupported_institut"
    assert n.records == []


@pytest.mark.parametrize("nom", [
    "10141 Pres vote LGBT Ifop Tetu",
    "10076 Pres primaire droite Ifop JDD",
    "10067 Pres RN Odoxa",
    "10049 Pres potentiel vote RN Cluster17",
    "10059 Pres bloc central droite Elabe",
])
def test_sous_echantillon_biaise_exclu(nom):
    # Sondages sur un sous-électorat : somment à 100 % mais non représentatifs
    # -> écartés (statut dédié, aucun record publié), même si le PDF est parsable.
    n = parse_notice(OK, source_name=nom)
    assert n.status == "biased_subsample"
    assert n.records == []


def test_sondage_national_non_exclu():
    # Garde-fou inverse : un intitulé national standard ne doit PAS être exclu.
    assert parse_notice(OK).status == "ok"
