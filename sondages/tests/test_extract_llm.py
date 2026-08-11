"""Test OFFLINE du fallback d'extraction par LLM (`extract_llm`).

Aucune vraie clé ni appel réseau : on monkeypatch `anthropic.Anthropic` pour
renvoyer une réponse structurée « canned » (un `types.SimpleNamespace` suffit —
`extract_with_llm` ne fait qu'accéder aux attributs). On vérifie :
  (a) `extract_with_llm` aplatit correctement en records qui somment ~100 ;
  (b) le fallback de `parse_notice(..., use_llm=True)` se déclenche quand les
      règles ne sortent rien et remplit les records avec extraction_method="llm" ;
  (c) une extraction LLM qui somme à 150 est REJETÉE par le garde-fou ~100 % ;
  (d) une extraction marquée non représentative (sous-électorat) est écartée ;
  (e) le pré-check épargne l'appel LLM (payant) sur une notice sans intentions.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sondages import extract_llm
from sondages.parse_pdf import parse_notice

FIX = Path(__file__).resolve().parent / "fixtures" / "pdf"
BARO = FIX / "ifop_barometre_no_intentions.pdf"    # notice SANS table d'intentions
INTENT = FIX / "ipsos_intentions_ok.pdf"           # vraie notice d'intentions
# Nom d'un institut non outillé -> les règles ne tournent pas, on force la voie LLM
# tout en gardant un PDF qui ressemble bien à une notice d'intentions (pré-check OK).
UNSUPPORTED = "9999 pres Kantar Public exemple"


def _canned(hypotheses, has_intentions=True, is_representative=True):
    """Construit un faux `resp` avec `.parsed_output` façon sortie pydantic."""
    hyps = [
        types.SimpleNamespace(
            tour=tour, hypothese=lib,
            candidats=[types.SimpleNamespace(nom=nom, intention=val) for nom, val in cands],
        )
        for tour, lib, cands in hypotheses
    ]
    parsed = types.SimpleNamespace(
        has_intentions=has_intentions, is_representative=is_representative, hypotheses=hyps)
    return types.SimpleNamespace(parsed_output=parsed)


def _install_fake_anthropic(monkeypatch, resp):
    """Rend `available()` vrai (clé factice), fait renvoyer `resp` par l'API, et
    renvoie un compteur d'appels à `messages.parse`."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    import anthropic
    calls = {"n": 0}

    class _FakeClient:
        def __init__(self, *a, **k):
            def _parse(*a, **k):
                calls["n"] += 1
                return resp
            self.messages = types.SimpleNamespace(parse=_parse)

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)
    return calls


_HYP_OK = ("Premier tour", "Hypothèse A", [
    ("Marine Le Pen", 34.0), ("Jean-Luc Mélenchon", 15.0), ("Édouard Philippe", 14.0),
    ("Gabriel Attal", 9.0), ("Raphaël Glucksmann", 9.0), ("Bruno Retailleau", 8.0),
    ("Éric Zemmour", 5.0), ("Marine Tondelier", 4.0), ("Fabien Roussel", 2.0),
])  # somme = 100


def test_extract_with_llm_aplatit_et_somme_100(monkeypatch):
    _install_fake_anthropic(monkeypatch, _canned([_HYP_OK]))
    recs = extract_llm.extract_with_llm("texte de notice", institut="Ipsos")
    assert {r["candidat"] for r in recs} >= {"Marine Le Pen", "Gabriel Attal"}
    assert all(r["tour"] == "Premier tour" for r in recs)
    assert all(r["hypothese"] == "Hypothèse A" for r in recs)
    assert 99 <= sum(r["intention"] for r in recs) <= 101


def test_fallback_parse_notice_use_llm(monkeypatch):
    if not INTENT.exists():
        pytest.skip("fixture intentions absente")
    calls = _install_fake_anthropic(monkeypatch, _canned([_HYP_OK]))
    n = parse_notice(INTENT, source_name=UNSUPPORTED, use_llm=True)  # règles muettes -> LLM
    assert calls["n"] == 1                          # l'appel LLM a bien eu lieu
    assert n.status == "ok"
    assert n.extraction_method == "llm"
    assert 99 <= sum(r["intention"] for r in n.records) <= 101


def test_llm_somme_150_rejetee(monkeypatch):
    if not INTENT.exists():
        pytest.skip("fixture intentions absente")
    hyp_150 = ("Premier tour", "Hypothèse A",
               [("Marine Le Pen", 80.0), ("Gabriel Attal", 70.0)])   # somme = 150
    _install_fake_anthropic(monkeypatch, _canned([hyp_150]))
    n = parse_notice(INTENT, source_name=UNSUPPORTED, use_llm=True)
    assert n.records == []                          # rejeté par le garde-fou ~100 %
    assert n.extraction_method == "none"


def test_llm_non_representatif_ecarte(monkeypatch):
    # Un sous-électorat (is_representative=false) ne doit rien renvoyer.
    _install_fake_anthropic(monkeypatch, _canned([_HYP_OK], is_representative=False))
    assert extract_llm.extract_with_llm("texte", institut="Ifop") == []


def test_precheck_epargne_appel_llm_sur_barometre(monkeypatch):
    # Sur une notice sans intentions (baromètre), le pré-check doit ÉVITER l'appel
    # LLM (payant) : le compteur reste à 0 malgré use_llm=True.
    if not BARO.exists():
        pytest.skip("fixture baromètre absente")
    calls = _install_fake_anthropic(monkeypatch, _canned([_HYP_OK]))
    n = parse_notice(BARO, use_llm=True)
    assert calls["n"] == 0                          # aucun appel LLM
    assert n.status == "no_intentions"
