"""sondages — parsing des notices de la Commission des sondages -> intentions structurées.

Module **autonome et réutilisable** : notices PDF (via l'index NSPPolls) →
sondages structurés au niveau candidat, avec traçabilité (`notice_url`).
Dépendances minimales (pdfplumber, pandas, requests) — aucune dépendance de
modélisation.

API publique :

    from sondages import build_dataset, parse_notice, fetch_index, validate, COLUMNS

    intentions, statuts = build_dataset(since_year=2025)   # DataFrame conforme au schéma
    validate(intentions)                                   # garde-fou du contrat

Le schéma de sortie est documenté et stable : voir `sondages.schema.COLUMNS`.
L'agrégation en « blocs » / « slots » est **hors périmètre** (décision de
modélisation) : ce module s'arrête aux intentions brutes par candidat.
"""

from sondages.schema import COLUMNS, validate, empty          # noqa: F401
from sondages.ingest import fetch_index, filter_presidential, download_notice  # noqa: F401
from sondages.parse_pdf import parse_notice, detect_institut, PARSERS  # noqa: F401
from sondages.build import build_dataset, write_dataset       # noqa: F401

__all__ = [
    "build_dataset", "write_dataset", "parse_notice", "detect_institut",
    "fetch_index", "filter_presidential", "download_notice",
    "validate", "empty", "COLUMNS", "PARSERS",
]
