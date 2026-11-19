"""Date du dernier sondage par candidat, pour l'AFFICHAGE seulement.

    python -m model.derniers_sondages                  # aujourd'hui
    python -m model.derniers_sondages --as-of 2026-10-01

Le site masque les candidats qui ne sont plus testés depuis longtemps (seuil :
`JOURS_SANS_SONDAGE` dans `docs/assets/site.js`). Les modèles n'en savent rien
et leurs snapshots restent intacts : ce fichier dit seulement, pour chaque nom
que le site peut afficher, la date de fin de terrain la plus récente.

Le front ne peut pas lire le CSV des sondages (hors de `docs/`), et le champ
`polls` des snapshots ne suffit pas : `gp-pooling` n'y met que les hypothèses de
SA liste, où Hollande ou Faure ne figurent jamais.

Clés : le nom tel que publié (« Bardella », « Hollande ») ET le libellé de slot
du modèle de base (« RN », « Mélenchon (LFI) »), celui qu'affichent l'accueil et
les courbes par liste. La date d'un slot est celle du dernier sondage d'un de
ses membres ; aucun alias : Bardella n'entre pas dans « RN ».
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from model.core.base import SITE_DATA
from model.core.live_dataset import load_raw_polls

FICHIER = "derniers-sondages.json"


def dates_par_nom(raw: pd.DataFrame) -> dict[str, dict]:
    """{nom ou slot: {"dernier": AAAA-MM-JJ, "n_sondages": n}}."""
    d = raw.dropna(subset=["date_fin"])
    out: dict[str, dict] = {}
    for col in ("candidat", "slot"):
        if col not in d:
            continue
        g = d.dropna(subset=[col]).groupby(col).agg(
            dernier=("date_fin", "max"), n=("notice", "nunique"))
        for nom, r in g.iterrows():
            out[str(nom)] = {"dernier": r["dernier"].date().isoformat(),
                             "n_sondages": int(r["n"])}
    return dict(sorted(out.items()))


def produire(raw: pd.DataFrame, as_of: str, dossier: Path | None = None) -> dict:
    dossier = dossier or SITE_DATA
    raw = raw[raw["date_fin"] <= pd.Timestamp(as_of)]
    m = {"as_of": as_of, "genere_le": date.today().isoformat(),
         "noms": dates_par_nom(raw)}
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / FICHIER).write_text(json.dumps(m, ensure_ascii=False, indent=1))
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--as-of", default=None)
    a = p.parse_args()
    as_of = a.as_of or date.today().isoformat()
    m = produire(load_raw_polls(as_of=as_of), as_of)
    print(f"[derniers-sondages] {len(m['noms'])} noms au {as_of}")


if __name__ == "__main__":
    main()
