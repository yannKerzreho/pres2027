"""Tests du modèle live 2027 (Phase 3) — logique rapide, sans échantillonnage."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.core.live_dataset import map_candidate, load_live_observations, SLOTS

PARSED = ROOT / "data" / "parsed" / "intentions_2027_wiki.csv"
needs_data = pytest.mark.skipif(not PARSED.exists(), reason="intentions_2027_wiki.csv absent")


def test_slots_fusionnent_les_alternatives():
    # Modèle de base = une seule hypothèse par bloc (cf. SLOTS, session du
    # 2026-08-09) : Le Pen est LE candidat RN retenu, Bardella (scénario
    # d'inéligibilité) n'est plus mappé du tout.
    assert map_candidate("Marine Le Pen")[0] == "RN"
    assert map_candidate("Jordan Bardella") == (None, None)
    # variantes d'accent / trait d'union canonicalisées
    assert map_candidate("Eric Zemmour") == map_candidate("Éric Zemmour")
    assert map_candidate("Nicolas Dupont Aignan")[0] == "Dupont-Aignan"
    assert map_candidate("Édouard Philippe")[0] == "Philippe (Horizons)"
    assert map_candidate("Gabriel Attal") == (None, None)


def test_chaque_slot_a_un_bloc_connu():
    blocs = {"gauche_radicale", "gauche", "ecologistes", "centre", "droite", "droite_radicale"}
    for slot, (bloc, names) in SLOTS.items():
        assert bloc in blocs


@needs_data
def test_observations_horizon_positif():
    obs = load_live_observations()
    assert len(obs) > 0
    assert (obs["horizon"] >= 0).all()
    assert obs["slot"].isin(SLOTS).all()


def test_forecast_from_draws_probabilites_bien_formees():
    # tirages π synthétiques : connu et net (RN dominant), pour vérifier la logique.
    from model.core.simulate import forecast_from_draws

    slots = list(SLOTS)
    K = len(slots)
    base = np.full(K, 0.5 / (K - 1))
    base[slots.index("RN")] = 0.5           # RN très haut
    pi = np.random.default_rng(0).dirichlet(base * 200, size=1000)  # (S, K)

    res = forecast_from_draws(pi, slots, forecast_horizon=200)
    fc = res["forecast_scrutin"]
    # probabilités dans [0,1]
    for s in slots:
        assert 0.0 <= fc[s]["p_qualifie_top2"] <= 1.0
        assert 0.0 <= fc[s]["p_arrive_premier"] <= 1.0
    # exactement 2 qualifiés par tirage -> somme des P(top2) ≈ 2
    assert abs(sum(fc[s]["p_qualifie_top2"] for s in slots) - 2.0) < 0.05
    # somme des P(1er) ≈ 1
    assert abs(sum(fc[s]["p_arrive_premier"] for s in slots) - 1.0) < 0.02
    # RN dominant doit être quasi certainement qualifié
    assert fc["RN"]["p_qualifie_top2"] > 0.9
    # Par défaut `forecast_from_draws` ne dérive plus : il compte des tirages
    # DÉJÀ projetés au scrutin par le modèle (cf. son docstring). Le repli
    # gaussien reste disponible et doit, lui, élargir la distribution.
    assert res["drift_sd_logratio"] == 0
    large = forecast_from_draws(pi, slots, forecast_horizon=200, drift_sd=0.5)
    for s in slots:
        etroit = res["forecast_scrutin"][s]["ic90"]
        elargi = large["forecast_scrutin"][s]["ic90"]
        assert (elargi[1] - elargi[0]) > (etroit[1] - etroit[0])
    # simplexe respecté : toutes les parts prévues dans [0, 1]
    for s in slots:
        assert 0.0 <= fc[s]["part_moyenne"] <= 1.0
        assert 0.0 <= fc[s]["ic90"][0] <= fc[s]["ic90"][1] <= 1.0


def test_scenario_id_sur_sondage_sans_hypothese_nommee(tmp_path):
    """Non-régression : un sondage à champ unique n'a pas d'hypothèse nommée
    (`hypothese` vide). Avec la dtype `str` de pandas >= 3, `astype(str)`
    propage la valeur manquante au lieu d'écrire "nan", la clé de scénario
    devenait NaN et le hachage levait `AttributeError: 'float' object has no
    attribute 'encode'` — `python -m model.run` tombait, job quotidien cassé.

    Le test force la dtype string quand la version de pandas installée ne
    l'active pas encore par défaut, pour reproduire l'environnement de la CI.
    """
    import pandas as pd
    from model.core.live_dataset import load_raw_polls

    csv = tmp_path / "polls.csv"
    csv.write_text(
        "notice,notice_url,institut,date_debut,date_fin,date_notice,echantillon,"
        "methode,tour,hypothese,candidat,intention\n"
        "s1,,Ifop,2026-08-01,2026-08-01,2026-08-01,1000,,Premier tour,,Le Pen,33\n"
        "s1,,Ifop,2026-08-01,2026-08-01,2026-08-01,1000,,Premier tour,,Philippe,20\n"
        "s2,,Elabe,2026-08-02,2026-08-02,2026-08-02,1000,,Premier tour,variante_1,Le Pen,35\n",
        encoding="utf-8")

    ancien = pd.get_option("future.infer_string")
    try:
        pd.set_option("future.infer_string", True)   # défaut en pandas >= 3
    except (KeyError, ValueError):                   # option retirée : déjà le défaut
        ancien = None
    try:
        raw = load_raw_polls(path=csv)
    finally:
        if ancien is not None:
            pd.set_option("future.infer_string", ancien)

    assert raw["scenario_id"].map(type).eq(str).all(), "clé de scénario non hachée"
    # les deux candidats du sondage sans hypothèse restent le MÊME scénario,
    # distinct de celui d'un autre institut
    ids = raw.groupby("notice")["scenario_id"].nunique()
    assert ids["s1"] == 1
    assert raw["scenario_id"].nunique() == 2


def test_aggregate_to_slots_ne_perd_pas_un_sondage_aux_metadonnees_incompletes():
    """Non-régression : `aggregate_to_slots` groupe sur des colonnes dont
    certaines sont FACULTATIVES (`notice_url` absente de Wikipedia,
    `echantillon` non publié). Par défaut pandas supprime tout groupe dont une
    clé vaut NaN — un sondage réel disparaissait alors de TOUS les modèles sans
    le moindre message, et le nombre affiché sur le site devenait faux.

    Constaté en production : « Ifop 7-8 juillet », l'un des 6 sondages retenus,
    était écarté faute de taille d'échantillon publiée. Un `fillna` existait
    bien, mais APRÈS le groupby qui avait déjà supprimé les lignes.
    """
    import numpy as np
    import pandas as pd
    from model.core.live_dataset import aggregate_to_slots

    base = {"notice": "s1", "notice_url": "http://x/s1", "institut": "Ifop",
            "date_fin": "2026-08-01", "echantillon": 1000.0, "hypothese": None,
            "bloc": "droite_radicale"}
    lignes = [
        {**base, "candidat": "Marine Le Pen", "intention": 33.0, "slot": "RN"},
        {**base, "candidat": "Édouard Philippe", "intention": 20.0,
         "slot": "Philippe (Horizons)", "bloc": "centre"},
        {**base, "notice": "s2", "notice_url": None, "echantillon": np.nan,
         "candidat": "Marine Le Pen", "intention": 35.0, "slot": "RN"},
    ]
    out = aggregate_to_slots(pd.DataFrame(lignes))
    assert set(out["notice"]) == {"s1", "s2"}, "un sondage a disparu de l'agrégation"
    # et sa taille d'échantillon est imputée, sinon les poids seraient NaN
    assert out["echantillon"].notna().all()
