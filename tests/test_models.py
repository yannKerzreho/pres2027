"""Tests du framework multi-modèles (contrat de sortie).

Chaque modèle enregistré doit produire un snapshot valide → un contributeur qui
ajoute son modèle est couvert automatiquement par ces tests.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model.core.registered  # noqa: F401  (peuple le registre)
from model.core.base import SURFACES, all_models, models_by_surface, validate_snapshot
from model.core.live_dataset import load_raw_polls

PARSED = ROOT / "data" / "parsed" / "intentions_2027_wiki.csv"
needs = pytest.mark.skipif(not PARSED.exists(), reason="données live absentes")

AS_OF = "2026-08-06"


def test_registre_non_vide():
    # Volontairement générique : ce test doit rester vrai quel que soit le
    # sous-ensemble de modèles présent sur la branche (cf. registered.py), pas
    # coder en dur les ids du jour.
    assert all_models(), "aucun modèle enregistré — model/core/registered.py"


def test_raw_polls_riches():
    # Gemini : run() doit recevoir les données les plus riches possibles.
    if not PARSED.exists():
        pytest.skip("intentions_2027_wiki.csv absent")
    raw = load_raw_polls(as_of=AS_OF)
    for col in ("institut", "echantillon", "methode", "hypothese", "candidat", "commanditaire"):
        assert col in raw.columns, f"colonne brute manquante : {col}"


def test_tous_les_modeles_enregistres_sont_importables_et_declares():
    """Garde-fou BON MARCHÉ sur l'ensemble du registre, sans inférence.

    Ce test seul justifie que les modèles non publiés restent couverts :
    `model/core/registered.py` les importe, donc un import cassé dans l'un
    d'eux fait tomber `python -m model.run` en entier — le job quotidien avec.
    C'est exactement la panne qu'on veut ne plus jamais revoir. Vérifier le
    *contrat de sortie* de ces variantes coûterait un run NUTS chacune, pour
    des courbes que le site n'affiche pas : cf. `test_modele_respecte_le_contrat`.
    """
    for mid, m in all_models().items():
        assert m.id == mid
        assert m.label and m.description, f"{mid} : label/description manquants"
        assert m.surface in (None, *SURFACES), f"{mid} : surface inconnue ({m.surface!r})"
        if m.surface == "scenarios":
            # La rubrique Scénarios est servie par un artefact sur mesure : sans
            # son nom, le manifeste ne dit pas à la page quoi charger.
            assert m.scenario_artifact, f"{mid} : surface 'scenarios' sans scenario_artifact"
        for methode in ("nowcast", "forecast", "run"):
            assert callable(getattr(m, methode, None)), f"{mid} : {methode} manquant"


def _modeles_a_tester() -> list[str]:
    """Rubrique « suivi » par défaut : c'est elle qui publie le contrat de
    snapshot, et ses modèles sont assez rapides pour tourner à chaque CI.

    Les modèles de la rubrique « scénarios » en sont exclus par leur COÛT, pas
    par leur statut — chacun est un run NUTS complet de plus d'une minute. Ils
    restent couverts par `test_tous_les_modeles_enregistres_sont_importables…`
    (garde-fou d'import) et par le job quotidien, qui échoue bruyamment si leur
    snapshot ne valide plus. `PRES2027_TEST_ALL_MODELS=1` teste tout le registre.
    """
    if os.environ.get("PRES2027_TEST_ALL_MODELS") == "1":
        return list(all_models())
    return [m.id for m in models_by_surface()["suivi"]]


@needs
@pytest.mark.parametrize("model_id", _modeles_a_tester())
def test_modele_respecte_le_contrat(model_id):
    raw = load_raw_polls(as_of=AS_OF)
    m = all_models()[model_id]
    for a in ("draws", "tune"):        # accélère les modèles bayésiens en test
        if hasattr(m, a):
            setattr(m, a, 150)
    if hasattr(m, "chains"):
        m.chains = 2
    snap = m.run(raw, AS_OF)
    validate_snapshot(snap)            # ne lève pas
    assert snap["model"] == model_id
    fc = snap["forecast_scrutin"]
    # exactement 2 qualifiés / 1 premier par tirage
    assert abs(sum(f["p_qualifie_top2"] for f in fc.values()) - 2.0) < 0.05
    assert abs(sum(f["p_arrive_premier"] for f in fc.values()) - 1.0) < 0.03
    # les sondages exposés pour le front sont dans l'espace des slots
    assert snap["polls"] and all("slot" in p for p in snap["polls"])


@needs
def test_le_manifeste_est_groupe_par_rubrique():
    """`docs/data/models.json` est le SEUL point de contact entre le registre et
    le front : les pages y lisent la rubrique à laquelle elles se rattachent.
    Un manifeste redevenu une liste plate casserait le sélecteur en silence."""
    from model.core.base import SITE_DATA, write_manifest
    import json
    write_manifest()
    man = json.loads((SITE_DATA / "models.json").read_text())
    assert set(man) == set(SURFACES), f"rubriques du manifeste : {sorted(man)}"
    for entrees in man.values():
        for e in entrees:
            assert {"id", "label", "description"} <= set(e)
    # La rubrique Scénarios ne se lit pas sans le chemin de son artefact.
    assert all("artefact" in e for e in man["scenarios"])
