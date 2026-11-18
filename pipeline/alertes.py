"""Alertes de production : annotation GitHub + issue, pour ce qui demande une
décision humaine (candidat à classer, sondeur inconnu, ingestion abandonnée,
modèle en échec).

Deux temps, parce que les étapes qui détectent n'ont pas le jeton GitHub :

1. `signale(cle, titre, corps, niveau)` — appelé là où le problème est vu.
   Imprime `::error::`/`::warning::` (annotation visible sur le run) et, si la
   variable `ALERTES_JSONL` désigne un fichier, y ajoute l'alerte.
2. `python -m pipeline.alertes publier <fichier>` — dernière étape du
   workflow : une issue par `cle`, mise à jour si elle est déjà ouverte (même
   titre), créée sinon. Pas de doublon d'un jour sur l'autre ; l'issue se ferme
   à la main, une fois la décision prise.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PREFIXE_TITRE = "[prod] "


def _echappe(s: str) -> str:
    # format des commandes de workflow : %, CR, LF encodés
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def signale(cle: str, titre: str, corps: str, niveau: str = "error") -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{niveau} title={_echappe(titre)}::{_echappe(corps)}", flush=True)
    else:
        print(f"[{niveau.upper()}] {titre} — {corps}", file=sys.stderr, flush=True)
    chemin = os.environ.get("ALERTES_JSONL")
    if chemin:
        with open(chemin, "a", encoding="utf-8") as f:
            f.write(json.dumps({"cle": cle, "titre": titre, "corps": corps, "niveau": niveau},
                               ensure_ascii=False) + "\n")


def lire(chemin: Path) -> dict[str, dict]:
    """Alertes regroupées par clé (la dernière occurrence d'une clé gagne : le
    créneau de 14:11 réécrit l'état du matin)."""
    out: dict[str, dict] = {}
    if not chemin.exists():
        return out
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        if ligne.strip():
            a = json.loads(ligne)
            out[a["cle"]] = a
    return out


def _gh(args: list[str], entree: str | None = None) -> str:
    return subprocess.run(["gh", *args], input=entree, text=True, check=True,
                          capture_output=True).stdout


def publier(chemin: Path, gh=_gh, url_run: str | None = None) -> list[str]:
    """Ouvre ou met à jour une issue par alerte. Renvoie les actions faites."""
    alertes = lire(chemin)
    if not alertes:
        return []
    ouvertes = {i["title"]: i["number"] for i in json.loads(
        gh(["issue", "list", "--state", "open", "--limit", "200", "--json", "number,title"]))}
    actions = []
    for a in alertes.values():
        titre = PREFIXE_TITRE + a["titre"]
        corps = a["corps"] + (f"\n\nDernier run concerné : {url_run}" if url_run else "") + (
            "\n\n_Issue tenue à jour par le workflow « Daily update » ; à fermer une fois "
            "la décision prise._")
        if titre in ouvertes:
            gh(["issue", "edit", str(ouvertes[titre]), "--body-file", "-"], corps)
            actions.append(f"mise à jour #{ouvertes[titre]} : {titre}")
        else:
            gh(["issue", "create", "--title", titre, "--body-file", "-"], corps)
            actions.append(f"créée : {titre}")
    return actions


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "publier":
        raise SystemExit("usage : python -m pipeline.alertes publier <fichier.jsonl>")
    url = None
    if os.environ.get("GITHUB_RUN_ID"):
        url = (f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
               f"{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{os.environ['GITHUB_RUN_ID']}")
    for action in publier(Path(argv[1]), url_run=url):
        print(action)


if __name__ == "__main__":
    main()
