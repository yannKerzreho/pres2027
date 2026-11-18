"""Tests de `model/listes.py` : courbes gp-pooling par liste de candidats.

La règle (validée par l'utilisateur) choisit QUELLES listes afficher d'après les
derniers sondages ; l'ajustement, lui, porte sur TOUT l'historique de la liste.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.core.live_dataset import SLOTS, load_raw_polls
from model.listes import (
    GPListe, champs, classement, cle_champ, historique, id_champ, libelle_liste, produire,
)
from model.models.gp_pooling.model import GPPooling

A = ["Le Pen", "Mélenchon", "Philippe"]
B = ["Le Pen", "Mélenchon", "Attal"]
C = ["Bardella", "Mélenchon", "Philippe"]


def _raw(sondages):
    """`sondages` : [(notice, date, [champ, champ…])] — une hypothèse par champ."""
    rows = []
    for notice, d, chs in sondages:
        for k, ch in enumerate(chs):
            for nom in ch:
                rows.append({"scenario_id": f"{notice}-{k}", "notice": notice,
                             "date_fin": pd.Timestamp(d), "candidat": nom,
                             "intention": 100 / len(ch)})
    return pd.DataFrame(rows)


def test_cle_insensible_a_l_ordre_aux_accents_et_aux_tirets():
    assert cle_champ(["Mélenchon", "Le Pen"]) == cle_champ(["le pen", "Melenchon"])
    assert cle_champ(["Dupont-Aignan"]) == cle_champ(["Dupont Aignan"])
    assert id_champ(cle_champ(A)) == id_champ(cle_champ(list(reversed(A))))


def test_un_candidat_de_plus_fait_un_autre_champ():
    # Aucune équivalence entre champs : c'est la définition de gp-pooling.
    assert cle_champ(A) != cle_champ(A + ["Arthaud"])


def test_classement_compte_les_sondages_recents_pas_les_hypotheses():
    # s1 teste A deux fois (deux hypothèses) : compte pour UN sondage.
    raw = _raw([("s1", "2026-10-01", [A, A]), ("s2", "2026-10-02", [B]),
                ("s3", "2026-10-03", [B]), ("s4", "2026-10-04", [A, B])])
    top = classement(champs(raw), "2026-10-10", min_total=1, min_recents=1)
    assert [x["cle"] for x in top] == [cle_champ(B), cle_champ(A)]
    assert [x["n_recents"] for x in top] == [3, 2]


def test_classement_ne_regarde_que_les_n_derniers_sondages():
    vieux = [(f"v{i}", f"2026-03-{i + 1:02d}", [C]) for i in range(8)]
    recents = [(f"r{i}", f"2026-10-{i + 1:02d}", [A]) for i in range(3)]
    ch = champs(_raw(vieux + recents))
    top = classement(ch, "2026-10-31", n_recents=3, min_recents=1, min_total=1)
    assert [x["cle"] for x in top] == [cle_champ(A)]
    # …mais n_total compte bien tout l'historique
    top = classement(ch, "2026-10-31", n_recents=11, min_recents=1, min_total=1)
    assert {x["cle"]: x["n_total"] for x in top} == {cle_champ(C): 8, cle_champ(A): 3}


def test_classement_plafond_et_planchers():
    raw = _raw([(f"s{i}", f"2026-10-{i + 1:02d}", [A, B, C, A + ["Zemmour"]])
                for i in range(4)] + [("s9", "2026-10-20", [["Seul"]])])
    top = classement(champs(raw), "2026-10-31", n_max=3, min_recents=2, min_total=3)
    assert len(top) == 3
    assert cle_champ(["Seul"]) not in {x["cle"] for x in top}   # 1 seul sondage


def test_classement_respecte_as_of():
    raw = _raw([("s1", "2026-10-01", [A]), ("s2", "2026-10-02", [A]),
                ("s3", "2026-11-01", [B]), ("s4", "2026-11-02", [B])])
    top = classement(champs(raw), "2026-10-15", min_total=1)
    assert [x["cle"] for x in top] == [cle_champ(A)]


def test_historique_note_entrees_et_sorties():
    sondages = [(f"a{i}", f"2026-01-{i + 1:02d}", [A]) for i in range(3)] + \
               [(f"b{i}", f"2026-02-{i + 1:02d}", [B]) for i in range(12)]
    evts, depuis = historique(champs(_raw(sondages)), "2026-03-01", depuis="2026-01-01")
    entrees = [(e["date"], e["id"]) for e in evts if e["type"] == "entree"]
    sorties = [(e["date"], e["id"]) for e in evts if e["type"] == "sortie"]
    assert ("2026-01-03", id_champ(cle_champ(A))) in entrees     # 3e sondage : MIN_TOTAL
    assert any(i == id_champ(cle_champ(A)) for _, i in sorties)   # poussée hors des 10 derniers
    assert depuis == {cle_champ(B): "2026-02-03"}


def test_libelle_dit_la_difference_avec_la_prevision():
    assert libelle_liste(A, A) == "liste de la prévision"
    assert libelle_liste(B, A) == "avec Attal, sans Philippe"
    assert libelle_liste(A + ["Attal"], A) == "avec Attal"


def test_gp_pooling_inchange():
    assert GPPooling().slots_modele() == list(SLOTS)


@pytest.fixture(scope="module")
def raw_reel():
    return load_raw_polls(as_of="2026-11-13")


def test_l_ajustement_utilise_tout_l_historique_de_la_liste(raw_reel):
    ch = champs(raw_reel)
    top = classement(ch, "2026-11-13")
    assert top, "aucune liste retenue sur les données réelles"
    for e in top:
        mdl = GPListe(e["cle"], e["candidats"])
        used = mdl.used_polls(raw_reel)
        assert used["notice"].nunique() == e["n_total"]           # pas seulement les récents
        assert set(used["slot"]) == set(mdl.slots_modele())       # champ complet, rien de perdu


def test_liste_de_reference_est_le_champ_de_gp_pooling(raw_reel):
    ref = GPPooling().used_polls(raw_reel)
    assert champs(ref)["cle"].nunique() == 1


def test_produire_ecrit_le_manifeste_et_nettoie(raw_reel, tmp_path):
    (tmp_path / "perimee.json").write_text("{}")
    m = produire(raw_reel, "2026-11-13", dossier=tmp_path)
    assert not (tmp_path / "perimee.json").exists()
    assert 1 <= len(m["listes"]) <= 3
    for x in m["listes"]:
        if x["reference"]:
            assert x["fichier"] is None            # le front réutilise le snapshot gp-pooling
            continue
        c = json.loads((tmp_path / Path(x["fichier"]).name).read_text())
        assert c["n_sondages"] == x["n_total"]
        assert set(c["trajectoire"]["slots"]) == set(c["candidats"])
        moy = sum(c["nowcast"][s]["mean"] for s in c["candidats"])
        assert moy == pytest.approx(1.0, abs=0.01)
    assert {e["type"] for e in m["evenements"]} <= {"entree", "sortie"}


def test_run_produit_les_listes_et_isole_leur_echec(monkeypatch):
    """Le passage complet appelle `produire` ; s'il échoue, les modèles ont
    quand même écrit leurs snapshots et le job sort en erreur (rouge, visible)."""
    import model.listes
    import model.run as run

    vus = []
    monkeypatch.setattr(run, "_produit", lambda mid, *a: vus.append(mid))
    monkeypatch.setattr(run, "load_raw_polls", lambda as_of=None: pd.DataFrame({"x": [1]}))

    def casse(raw, as_of):
        raise RuntimeError("boum")
    monkeypatch.setattr(model.listes, "produire", casse)
    import model.derniers_sondages
    monkeypatch.setattr(model.derniers_sondages, "produire", lambda raw, as_of: None)
    with pytest.raises(SystemExit, match="listes"):
        run.run("all", as_of="2026-11-13")
    assert vus                                   # les modèles sont passés avant

    vus.clear()
    monkeypatch.setattr(model.listes, "produire",
                        lambda raw, as_of: vus.append("listes") or {"listes": []})
    run.run("gp-pooling", as_of="2026-11-13")    # un --model isolé : pas de listes
    assert "listes" not in vus
