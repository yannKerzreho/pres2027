"""Extraction des intentions par LLM (fallback des parsers par règles).

Filet robuste pour les notices que les règles ne savent pas parser (layouts
hétérogènes, nouveaux instituts). Un **simple appel structuré** (pas un agent) :
texte de la notice -> JSON conforme au schéma -> records normalisés. Le résultat
repasse ensuite par le MÊME garde-fou que les règles (somme ~100 %), donc une
hallucination éventuelle est rejetée, pas publiée.

Dépendance **optionnelle** : `anthropic` (+ une clé API). Importé paresseusement
pour ne pas alourdir le cœur du module. Modèle par défaut : Haiku (cheap).

Sécurité : la clé est lue via `os.environ["ANTHROPIC_API_KEY"]` — jamais en dur,
jamais dans le repo.
"""

from __future__ import annotations

import os

MODEL = "claude-haiku-4-5"          # cheap ; surchargables par SONDAGES_LLM_MODEL
MAX_INPUT_CHARS = 60_000            # borne de coût pour les très grosses notices


def _system_prompt(year: int) -> str:
    return (
        f"Tu extrais les intentions de vote d'une notice de sondage de la Commission "
        f"des sondages française, pour la présidentielle {year}. Règles STRICTES :\n"
        f"- Ne retiens QUE les tableaux d'intentions de vote au 1er ou 2nd tour de {year}.\n"
        "- IGNORE les tables de rappel de vote passé (« Reconstitution » d'une élection "
        "précédente, européennes/législatives) — ce ne sont pas des intentions.\n"
        "- IGNORE les baromètres d'image/popularité, les questions d'opinion, les "
        "croisements démographiques (âge, CSP, région).\n"
        "- Une notice peut tester plusieurs hypothèses de candidatures : rends-en une "
        "entrée par hypothèse, avec le libellé de l'hypothèse si présent.\n"
        "- Intentions en % des suffrages exprimés, telles que publiées "
        "(colonne REDRESSÉE / « résultats publiés », pas les scores bruts).\n"
        "- Vérifie la REPRÉSENTATIVITÉ : l'échantillon doit être l'ensemble des "
        "inscrits sur les listes électorales. Si le sondage porte sur un "
        "SOUS-ÉLECTORAT (personnes LGBT, sympathisant·es d'un parti, électeurs d'une "
        "primaire, « bloc central »…), il n'est pas exploitable pour un agrégat "
        f"national : renvoie is_representative=false.\n"
        f"- Si la notice ne contient AUCUNE intention de vote {year}, renvoie "
        "has_intentions=false et une liste vide.\n"
        "- Extrais aussi les métadonnées si présentes : dates de terrain (format "
        "AAAA-MM-JJ), taille d'échantillon (nombre d'inscrits interrogés)."
    )


_SYSTEM = _system_prompt(2027)  # rétrocompatibilité (défaut historique du module)


def _pydantic_models():
    # Annotations explicites via `typing` (et non « str | None ») : pydantic évalue
    # les annotations à la construction du modèle, ce qui casse en Python 3.9 avec
    # la syntaxe d'union récente.
    from typing import List, Optional

    from pydantic import BaseModel  # dépendance de anthropic

    class Candidat(BaseModel):
        nom: str
        intention: float

    class Hypothese(BaseModel):
        tour: str                      # "Premier tour" | "Deuxième tour"
        hypothese: Optional[str]
        candidats: List[Candidat]

    class Extraction(BaseModel):
        has_intentions: bool
        is_representative: bool = True   # faux si sous-électorat (LGBT, RN, primaire…)
        date_debut: Optional[str] = None   # AAAA-MM-JJ, si trouvée dans le texte
        date_fin: Optional[str] = None
        echantillon: Optional[int] = None
        hypotheses: List[Hypothese]

    return Extraction


def available() -> bool:
    """Vrai si l'extraction LLM est utilisable (lib + clé présentes)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        import pydantic  # noqa: F401
    except Exception:
        return False
    return True


def _call(text: str, institut: str | None, year: int, model: str | None):
    if not available():
        raise RuntimeError("Extraction LLM indisponible : installez `anthropic` "
                           "et définissez ANTHROPIC_API_KEY.")
    import anthropic

    Extraction = _pydantic_models()
    client = anthropic.Anthropic()    # lit ANTHROPIC_API_KEY dans l'environnement
    prompt = (f"Institut : {institut or 'inconnu'}\n\n"
              f"Texte de la notice :\n{text[:MAX_INPUT_CHARS]}")

    resp = client.messages.parse(
        model=model or os.environ.get("SONDAGES_LLM_MODEL", MODEL),
        max_tokens=8000,
        system=_system_prompt(year),
        messages=[{"role": "user", "content": prompt}],
        output_format=Extraction,
    )
    return resp.parsed_output


def extract_with_llm(text: str, institut: str | None = None,
                     model: str | None = None, year: int = 2027) -> list[dict]:
    """Texte de notice -> records [{tour, hypothese, candidat, intention}].

    Lève RuntimeError si la dépendance/clé manque (appeler `available()` avant)."""
    data = _call(text, institut, year, model)
    if data is None or not data.has_intentions:
        return []
    if not getattr(data, "is_representative", True):
        return []                          # sous-électorat : non exploitable

    records = []
    for hyp in data.hypotheses:
        tour = "Deuxième tour" if "2" in hyp.tour or "deux" in hyp.tour.lower() else "Premier tour"
        for c in hyp.candidats:
            if 0 <= c.intention <= 100 and c.nom.strip():
                records.append({"tour": tour, "hypothese": hyp.hypothese,
                                "candidat": c.nom.strip(), "intention": float(c.intention)})
    return records


def extract_full_with_llm(text: str, institut: str | None = None,
                          model: str | None = None, year: int = 2027) -> dict | None:
    """Comme `extract_with_llm`, mais renvoie aussi les métadonnées (dates,
    échantillon) — utile pour recouper une notice contre une autre source
    (ex. retrouver la date d'un sondage Wikipedia via la vraie notice PDF).
    Renvoie None si pas d'intentions/non représentatif."""
    data = _call(text, institut, year, model)
    if data is None or not data.has_intentions or not getattr(data, "is_representative", True):
        return None

    records = []
    for hyp in data.hypotheses:
        tour = "Deuxième tour" if "2" in hyp.tour or "deux" in hyp.tour.lower() else "Premier tour"
        for c in hyp.candidats:
            if 0 <= c.intention <= 100 and c.nom.strip():
                records.append({"tour": tour, "hypothese": hyp.hypothese,
                                "candidat": c.nom.strip(), "intention": float(c.intention)})
    return {"date_debut": data.date_debut, "date_fin": data.date_fin,
           "echantillon": data.echantillon, "records": records}
