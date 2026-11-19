"""Robustesse de la production (incident du 18/11/2026 : Polony et Riner,
absents de ORDER_GROUPS) : un modèle qui refuse ne bloque pas les autres, et
chaque point de panne humain produit une alerte (annotation + issue)."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _sondages(candidats_par_notice: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for i, (notice, cands) in enumerate(candidats_par_notice.items()):
        for c in cands:
            rows.append({"notice": notice, "candidat": c, "intention": 100 / len(cands),
                         "date_fin": pd.Timestamp("2026-11-01") + pd.Timedelta(days=i),
                         "institut": "Ifop", "hypothese": None})
    return pd.DataFrame(rows)


def _alertes(path: Path) -> dict:
    from pipeline.alertes import lire
    return lire(path)


def test_refus_du_roster_reste_un_refus_mais_porte_l_alerte():
    from model.models.spatial_pooling.roster import CandidatsNonClasses, build_roster
    # noms fictifs : Polony et Riner sont classés depuis fb69009
    raw = _sondages({f"S{i}": ["Mélenchon", "Le Pen", "Zorglub", "Nouvelle"] for i in range(6)})
    with pytest.raises(ValueError) as exc:          # toujours un ValueError
        build_roster(raw)
    e = exc.value
    assert isinstance(e, CandidatsNonClasses) and e.noms == ["Nouvelle", "Zorglub"]
    cle, titre, corps = e.alerte
    assert "ORDER_GROUPS" in titre and "- Nouvelle" in corps and "- Zorglub" in corps


class _Ok:
    surface = "suivi"
    ecrit = []


class _Refuse:
    surface = "suivi"


def test_un_modele_en_echec_n_empeche_pas_les_autres(monkeypatch, tmp_path):
    import model.run as run
    from model.models.spatial_pooling.roster import CandidatsNonClasses
    alertes = tmp_path / "a.jsonl"
    monkeypatch.setenv("ALERTES_JSONL", str(alertes))
    ecrits = []

    def produit(mid, mdl, raw, as_of, rejeu, max_rejeu):
        if mid == "spatial-pooling":
            raise CandidatsNonClasses(["Polony", "Riner"])
        if mid == "autre":
            raise RuntimeError("boum")
        ecrits.append(mid)

    monkeypatch.setattr(run, "load_raw_polls", lambda as_of=None, **_: _sondages({"S": ["A", "B"]}))
    monkeypatch.setattr(run, "_selection", lambda m, p: {"spatial-pooling": _Refuse(),
                                                         "autre": _Refuse(), "gp-pooling": _Ok()})
    monkeypatch.setattr(run, "_produit", produit)
    with pytest.raises(SystemExit, match="spatial-pooling, autre"):
        run.run(model_id="selection", as_of="2026-11-18")
    assert ecrits == ["gp-pooling"]                 # gp-pooling a publié malgré tout
    a = _alertes(alertes)
    assert set(a) == {"spatial-pooling-order-groups", "modele-autre"}
    assert "RuntimeError: boum" in a["modele-autre"]["corps"]


def test_candidats_hors_palette(tmp_path):
    from model.run import candidats_hors_palette
    palette = tmp_path / "palette.js"
    palette.write_text('const CANDIDATS = {\n  "Mélenchon": { parti: "LFI" },\n  "Le Pen": {},\n};\n')
    raw = _sondages({f"S{i}": ["Mélenchon", "Le Pen", "Riner", "Autres"] for i in range(5)})
    vieux = _sondages({f"V{i}": ["Macron"] for i in range(9)}).assign(date_fin=pd.Timestamp("2024-05-01"))
    assert candidats_hors_palette(pd.concat([raw, vieux]), palette) == ["Riner"]


def test_palette_du_site_couvre_les_candidats_actuels():
    """Garde-fou sur la vraie palette et le vrai CSV : un candidat éligible
    sans couleur s'affiche en gris (dégradation voulue) — ce test le rend visible."""
    from model.core.live_dataset import load_raw_polls
    from model.run import candidats_hors_palette
    assert candidats_hors_palette(load_raw_polls()) == []


def test_sondeurs_inconnus_hors_lignes_de_rappel():
    from sondages.wiki import sondeurs_inconnus
    w = ["institut non reconnu 'Résultats' -> notice écartée",
         "institut non reconnu 'Résultats officiels' -> notice écartée",
         "institut non reconnu 'Toluna Nouveau' -> notice écartée",
         "template inconnu {'note'} dans Elabe 25-27 mars (WP)"]
    assert sondeurs_inconnus(w) == ["Toluna Nouveau"]


def test_abandon_d_ingestion_alerte_puis_echoue(monkeypatch, tmp_path):
    from sondages.wiki import _abandon
    monkeypatch.setenv("ALERTES_JSONL", str(tmp_path / "a.jsonl"))
    with pytest.raises(SystemExit, match="ABANDON"):
        _abandon("ABANDON : 52/52 sondages publiés ont disparu")
    assert "ingestion" in _alertes(tmp_path / "a.jsonl")


def test_publier_met_a_jour_l_issue_ouverte_au_lieu_d_en_creer_une(tmp_path):
    from pipeline.alertes import publier, signale
    f = tmp_path / "a.jsonl"
    import os
    os.environ["ALERTES_JSONL"] = str(f)
    try:
        signale("x", "spatial-pooling : candidats à classer dans ORDER_GROUPS", "matin")
        signale("x", "spatial-pooling : candidats à classer dans ORDER_GROUPS", "14:11")
        signale("y", "ingestion : sondeurs non reconnus", "Toluna", niveau="warning")
    finally:
        del os.environ["ALERTES_JSONL"]
    appels = []

    def gh(args, entree=None):
        appels.append((args[:2], args[2] if len(args) > 2 else None, entree))
        if args[:2] == ["issue", "list"]:
            return json.dumps([{"number": 7, "title": "[prod] spatial-pooling : candidats à classer dans ORDER_GROUPS"}])
        return ""

    actions = publier(f, gh=gh)
    assert actions == ["mise à jour #7 : [prod] spatial-pooling : candidats à classer dans ORDER_GROUPS",
                       "créée : [prod] ingestion : sondeurs non reconnus"]
    edit = next(a for a in appels if a[0] == ["issue", "edit"])
    assert edit[1] == "7" and edit[2].startswith("14:11")      # le dernier état gagne
    assert publier(tmp_path / "absent.jsonl", gh=gh) == []


def test_workflow_modeles_tournent_meme_si_l_ingestion_abandonne():
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "daily-update.yml").read_text())
    steps = {s.get("name"): s for s in wf["jobs"]["update"]["steps"]}
    assert "!cancelled()" in steps["Modèles du site -> docs/data/"]["if"]
    assert "!cancelled()" in steps["Alertes -> issues"]["if"]
    assert wf["permissions"]["issues"] == "write"
    assert "ALERTES_JSONL" in wf["jobs"]["update"]["env"]


def _faux_gh(tmp_path: Path, ouvertes: list[dict]) -> tuple[dict, Path]:
    """Exécutable `gh` factice en tête du PATH : enregistre chaque appel
    (arguments + entrée standard) et répond à `issue list`."""
    journal = tmp_path / "appels.jsonl"
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    gh = bin_ / "gh"
    gh.write_text(f"""#!{__import__('sys').executable}
import json, sys
with open({str(journal)!r}, "a") as f:
    f.write(json.dumps({{"args": sys.argv[1:], "stdin": sys.stdin.read()}}) + "\\n")
if sys.argv[1:3] == ["issue", "list"]:
    print(json.dumps({ouvertes!r}))
""")
    gh.chmod(0o755)
    import os
    env = {**os.environ, "PATH": f"{bin_}{os.pathsep}{os.environ['PATH']}",
           "GITHUB_RUN_ID": "123", "GITHUB_REPOSITORY": "proprio/depot",
           "GITHUB_SERVER_URL": "https://github.com"}
    env.pop("ALERTES_JSONL", None)
    env.pop("GITHUB_ACTIONS", None)
    return env, journal


def _cli(env, *args):
    import subprocess, sys
    return subprocess.run([sys.executable, "-m", "pipeline.alertes", "publier", *args],
                          cwd=ROOT, env=env, text=True, capture_output=True, check=True).stdout


def _fichier_alertes(tmp_path: Path) -> Path:
    f = tmp_path / "a.jsonl"
    f.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in [
        {"cle": "x", "titre": "spatial-pooling : candidats à classer dans ORDER_GROUPS",
         "corps": "- Zorglub", "niveau": "error"},
        {"cle": "y", "titre": "ingestion : sondeurs non reconnus", "corps": "Toluna", "niveau": "warning"},
    ]) + "\n")
    return f


def test_chemin_reel_de_la_cli_avec_gh_factice(tmp_path):
    """Le chemin exact de l'étape « Alertes -> issues » (module lancé en
    sous-processus, `gh` appelé par subprocess, URL du run reconstruite), sans
    toucher au dépôt."""
    env, journal = _faux_gh(tmp_path, [{"number": 7, "title":
                                        "[prod] spatial-pooling : candidats à classer dans ORDER_GROUPS"}])
    sortie = _cli(env, str(_fichier_alertes(tmp_path)))
    appels = [json.loads(l) for l in journal.read_text().splitlines()]
    assert [a["args"][:2] for a in appels] == [["issue", "list"], ["issue", "edit"], ["issue", "create"]]
    assert appels[1]["args"][2] == "7" and appels[1]["stdin"].startswith("- Zorglub")
    assert appels[2]["args"][3] == "[prod] ingestion : sondeurs non reconnus"
    assert "https://github.com/proprio/depot/actions/runs/123" in appels[2]["stdin"]
    assert "mise à jour #7" in sortie and "créée : [prod] ingestion" in sortie


def test_a_blanc_ne_fait_que_lire(tmp_path):
    env, journal = _faux_gh(tmp_path, [])
    sortie = _cli(env, str(_fichier_alertes(tmp_path)), "--a-blanc")
    appels = [json.loads(l)["args"][:2] for l in journal.read_text().splitlines()]
    assert appels == [["issue", "list"]]                       # aucune écriture
    assert sortie.count("[à blanc] créée") == 2 and "- Zorglub" in sortie


def test_sans_alerte_gh_n_est_pas_appele(tmp_path):
    env, journal = _faux_gh(tmp_path, [])
    assert _cli(env, str(tmp_path / "absent.jsonl")) == ""
    assert not journal.exists()
