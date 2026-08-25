"""Rejeu des dates touchées par un sondage arrivé en retard (`model/run.py`).

Le cas réel qui a motivé ce code : le sondage Harris des 18-19 août 2026 a été
saisi sur Wikipedia le 24 à 08h06 UTC, 11 minutes APRÈS le passage du job
quotidien. Sans rejeu, la courbe décrochait au 25 (date d'arrivée) alors que le
point de sondage était tracé au 19 (date de terrain) — six snapshots publiés
ignoraient un sondage antérieur à leur propre date.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.run import dates_a_produire


class FauxModele:
    """Modèle minimal : `dates_a_produire` n'a besoin que de `id` + `used_polls`.
    Sans filtre propre, pour que le test porte sur la détection et pas sur le
    roster d'un modèle réel."""
    id = "faux-modele"

    def used_polls(self, raw):
        return raw


def _polls(*paires) -> pd.DataFrame:
    return pd.DataFrame([{"notice": n, "date_fin": pd.Timestamp(d)} for d, n in paires])


def _publie(mdir: Path, jour: str, notices) -> None:
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"{jour}.json").write_text(json.dumps(
        {"as_of": jour, "polls": [{"notice": n, "slot": "RN"} for n in notices]}))
    idx = mdir / "index.json"
    dates = set(json.loads(idx.read_text())) if idx.exists() else set()
    dates.add(jour)
    idx.write_text(json.dumps(sorted(dates)))


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.setattr("model.run.SITE_DATA", tmp_path)
    return tmp_path / FauxModele.id


def test_sondage_arrive_en_retard_rejoue_toute_la_fenetre(site):
    """Un sondage daté du 19 arrivé le 25 périme les snapshots du 19 AU 25.

    Ne rejouer que le 19 produirait deux décrochages : la courbe monterait au
    19 puis retomberait au 20, les jours suivants ignorant toujours le sondage.
    """
    raw = _polls(("2026-08-10", "Ifop"), ("2026-08-19", "Harris"))
    _publie(site, "2026-08-10", ["Ifop"])      # à jour : Harris lui est postérieur
    for jour in ("2026-08-19", "2026-08-20", "2026-08-21"):
        _publie(site, jour, ["Ifop"])          # publiés avant l'arrivée de Harris

    dates, ecartees = dates_a_produire(FauxModele(), raw, "2026-08-22")

    assert dates == ["2026-08-19", "2026-08-20", "2026-08-21", "2026-08-22"]
    assert ecartees == []


def test_date_de_terrain_sans_snapshot_est_creee(site):
    """Le point de courbe doit exister le jour de la MESURE, là où le nuage de
    sondages du site place déjà son point — même si aucun snapshot n'a jamais
    été produit ce jour-là."""
    raw = _polls(("2026-08-19", "Harris"))
    _publie(site, "2026-08-25", ["Harris"])

    dates, _ = dates_a_produire(FauxModele(), raw, "2026-08-25")

    assert dates == ["2026-08-19", "2026-08-25"]


def test_aucun_sondage_exploitable_aucun_point_invente(site):
    """Une date antérieure au premier sondage n'est pas calculable : on
    n'invente pas un point sur la courbe."""
    raw = _polls(("2026-08-19", "Harris"))
    _publie(site, "2026-06-01", [])            # snapshot vide hérité

    dates, _ = dates_a_produire(FauxModele(), raw, "2026-08-25")

    assert "2026-06-01" not in dates


def test_as_of_toujours_produit_meme_a_jour(site):
    """Le graphe des probabilités de qualification bouge chaque jour sans
    nouveau sondage (l'horizon se réduit) : le point du jour a du sens en
    propre, il n'est pas conditionné à l'arrivée d'un sondage."""
    raw = _polls(("2026-08-19", "Harris"))
    _publie(site, "2026-08-19", ["Harris"])
    _publie(site, "2026-08-25", ["Harris"])

    dates, ecartees = dates_a_produire(FauxModele(), raw, "2026-08-25")

    assert dates == ["2026-08-25"]
    assert ecartees == []


def test_plafond_garde_les_plus_recentes_et_reste_chronologique(site):
    """Sous plafond, on rattrape les dates les plus récentes (celles que le
    lecteur regarde). L'ordre RENDU reste chronologique : `spatial-pooling`
    réécrit `scenarios.json` à chaque nowcast, donc la dernière date calculée
    doit être `as_of`, sinon le site publie la géométrie d'une date passée."""
    raw = _polls(("2026-08-01", "Ifop"), ("2026-08-19", "Harris"))
    _publie(site, "2026-08-01", ["Ifop"])      # à jour
    for jour in ("2026-08-19", "2026-08-20", "2026-08-21", "2026-08-22"):
        _publie(site, jour, ["Ifop"])

    dates, ecartees = dates_a_produire(FauxModele(), raw, "2026-08-25", max_rejeu=2)

    assert ecartees == ["2026-08-19", "2026-08-20"]
    assert dates == ["2026-08-21", "2026-08-22", "2026-08-25"]
    assert dates == sorted(dates) and dates[-1] == "2026-08-25"


def test_snapshot_illisible_est_rejoue(site):
    """Un fichier tronqué (job interrompu, disque plein) doit être recalculé,
    pas ignoré silencieusement."""
    raw = _polls(("2026-08-19", "Harris"))
    _publie(site, "2026-08-20", ["Harris"])
    (site / "2026-08-20.json").write_text('{"polls": [{"notice"')

    dates, _ = dates_a_produire(FauxModele(), raw, "2026-08-25")

    assert "2026-08-20" in dates


class FauxModeleSansRejeu(FauxModele):
    """Modèle cher dont personne ne relit les snapshots datés (cf.
    `ForecastModel.rejeu_historique`) — profil de `spatial-pooling`."""
    id = "faux-sans-rejeu"
    rejeu_historique = False


def test_modele_exclu_du_rejeu_ne_produit_que_le_jour(tmp_path, monkeypatch):
    """L'opt-out prime sur la détection : même avec des snapshots périmés et des
    dates de terrain sans fichier, seul `as_of` est calculé. C'est ce qui garde
    le job quotidien à ~3 min quand un modèle coûte 180 s par ajustement."""
    monkeypatch.setattr("model.run.SITE_DATA", tmp_path)
    site = tmp_path / FauxModeleSansRejeu.id
    raw = _polls(("2026-08-10", "Ifop"), ("2026-08-19", "Harris"))
    for jour in ("2026-08-19", "2026-08-20"):
        _publie(site, jour, ["Ifop"])          # périmés : Harris leur manque

    dates, ecartees = dates_a_produire(FauxModeleSansRejeu(), raw, "2026-08-22")

    assert dates == ["2026-08-22"]
    assert ecartees == []


def test_plafond_illimite_par_defaut(site):
    """`MAX_REJEU_DEFAUT = None` : un modèle bon marché rattrape tout son arriéré
    en un passage. Le plafond n'est plus la protection du job — l'opt-out l'est."""
    raw = _polls(("2026-08-01", "Ifop"), ("2026-08-19", "Harris"))
    for jour in ("2026-08-19", "2026-08-20", "2026-08-21", "2026-08-22"):
        _publie(site, jour, ["Ifop"])

    dates, ecartees = dates_a_produire(FauxModele(), raw, "2026-08-25")

    assert ecartees == []
    assert dates == ["2026-08-01", "2026-08-19", "2026-08-20",
                     "2026-08-21", "2026-08-22", "2026-08-25"]
