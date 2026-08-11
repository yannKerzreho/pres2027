"""Framework multi-modèles : classe de base, registre et contrat de sortie.

Tout modèle de prévision sous-classe `ForecastModel` et implémente `nowcast()` +
`forecast()`. La phase d'apprentissage sur l'historique (`calibrate()`) est
**optionnelle** : c'est ainsi qu'on gère « certains estiment des paramètres sur
les données passées, d'autres non ».

L'orchestration `run()` est commune à tous et **fige le schéma du snapshot**
(le contrat), ce qui garde le front agnostique et rend les modèles comparables.
Un `validate_snapshot()` protège des contributions cassées.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
SITE_DATA = ROOT / "site" / "data"
ELECTION_T1 = pd.Timestamp("2027-04-18")


@dataclass
class Nowcast:
    """Sortie de `nowcast()` : tirages des parts courantes (simplexe par
    tirage). `draws` est un xr.DataArray (dims "draw","candidat") — le nom des
    coordonnées "candidat" est libre à chaque modèle (slots aujourd'hui,
    n'importe quel autre vocabulaire demain), rien n'est imposé ici.

    `diagnostics` : dict optionnel, libre au modèle (R-hat, ESS, valeurs
    d'hyperparamètres...) — PAS de schéma imposé ici (contrairement à
    `forecast_scrutin`, cf. `validate_snapshot`) : un diagnostic utile à un
    modèle SSM (ex. `tau_niveau`) n'a pas de sens pour un autre. Simplement
    recopié tel quel dans le snapshot par `assemble_snapshot` s'il est
    présent."""
    draws: xr.DataArray
    diagnostics: dict | None = None

    def summary(self) -> dict:
        d = self.draws
        out = {}
        for c in d.coords["candidat"].values.tolist():
            vals = d.sel(candidat=c).values
            # Moyenne arithmétique — PAS la médiane (essayé puis abandonné) :
            # pour une composition (Σ candidats = 1 à CHAQUE tirage), seule la
            # moyenne préserve Σ_c mean(pi_c) = 1 exactement (linéarité de
            # l'espérance) ; la médiane par candidat n'a aucune raison de
            # sommer à 1 (mesuré : 0.88 sur un cas réel), ce qui casse le
            # contrat `validate_snapshot`. La moyenne reste techniquement
            # biaisée à la hausse pour un petit candidat sous forte incertitude
            # (effet Jensen, E[softmax(X)] > softmax(E[X])) — corrigé à la
            # source en réduisant cette incertitude quand c'est injustifié,
            # pas en changeant la statistique de résumé.
            out[c] = {
                "mean": round(float(vals.mean()), 4),
                "ic90": [round(float(np.percentile(vals, 5)), 4),
                         round(float(np.percentile(vals, 95)), 4)],
            }
        return out


# --- Registre -----------------------------------------------------------------
_REGISTRY: dict[str, "ForecastModel"] = {}


def register(cls):
    """Décorateur : enregistre une instance du modèle sous son id."""
    inst = cls()
    _REGISTRY[inst.id] = inst
    return cls


def all_models() -> dict[str, "ForecastModel"]:
    return dict(_REGISTRY)


def get_model(model_id: str) -> "ForecastModel":
    return _REGISTRY[model_id]


class ForecastModel(ABC):
    id: str = "abstract"
    label: str = ""
    description: str = ""
    trains_on_history: bool = False
    # Visible dans le sélecteur du site public (site/data/models.json) — False
    # pour une variante de comparaison/diagnostic qu'on veut garder exécutable
    # (CLI, tests, backfill) sans l'exposer comme option publique. N'affecte
    # PAS le registre (`all_models()`) : le modèle reste pleinement utilisable
    # partout ailleurs, seul `write_manifest()` filtre dessus.
    public: bool = True

    # --- phase HORS-LIGNE (optionnelle) ---
    def calibrate(self, history=None) -> None:
        """No-op par défaut. À surcharger pour apprendre des paramètres sur les
        élections passées et persister des artefacts (cf. model/core/bank.py)."""
        return

    def load_artifacts(self) -> None:
        """Recharge les artefacts produits par calibrate() (si besoin)."""
        return

    # --- phase EN LIGNE (obligatoire) ---
    @abstractmethod
    def nowcast(self, polls: pd.DataFrame, as_of: str) -> Nowcast:
        ...

    @abstractmethod
    def forecast(self, nowcast: Nowcast, horizon_days: int) -> dict:
        """{forecast_scrutin: {slot: {...}}, duels_probables: [...], drift_modele}."""
        ...

    def used_polls(self, raw_polls: pd.DataFrame) -> pd.DataFrame:
        """Sous-ensemble de `raw_polls` que ce modèle a RÉELLEMENT consommé.

        Par défaut : tout. Un modèle qui filtre en interne (roster exact, date
        plancher, instituts...) **doit** surcharger ceci — c'est ce qui alimente
        `meta.n_sondages`, `meta.instituts` et les points du graphe. Sans cette
        surcharge, le site annonce une base de sondages plus large que celle
        effectivement utilisée : le lecteur croit à une précision qui n'existe
        pas, et les points affichés ne correspondent pas à la courbe tracée.
        """
        return raw_polls

    def display_polls(self, raw_polls: pd.DataFrame) -> list[dict]:
        """Sondages à afficher sur la courbe, dans le vocabulaire de CE
        modèle. Par défaut : slots fusionnés (`aggregate_to_slots`) — un
        modèle dont le nowcast/forecast utilise un autre vocabulaire
        (candidats bruts, scénarios...) surcharge cette méthode plutôt que de
        se faire imposer les slots par le framework.

        `aggregate_to_slots` renvoie UNE ligne par (sondage, hypothèse, slot) —
        plusieurs hypothèses d'un même sondage sont quasi identiques pour un
        candidat qu'elles testent toutes (vérifié : <1pt d'écart), donc on
        moyenne ici par (sondage, slot) pour UN point par sondage sur le
        graphe. Cosmétique uniquement : le nowcast/forecast, eux, consomment
        `aggregate_to_slots` directement (une hypothèse = un nœud SSM, cf.
        `NowcastData.from_dataframe`), pas ce résumé moyenné."""
        from model.core.live_dataset import aggregate_to_slots
        disp = (aggregate_to_slots(raw_polls)
               .groupby(["notice", "notice_url", "institut", "date_fin", "slot"],
                        dropna=False, as_index=False)["part"].mean())
        return [
            {"date_fin": str(r.date_fin)[:10], "institut": r.institut,
             "slot": r.slot, "part": round(float(r.part) / 100.0, 4),
             "notice": r.notice, "notice_url": getattr(r, "notice_url", None)}
            for r in disp.itertuples()
        ]

    # --- orchestration commune (contrat) ---
    def run(self, raw_polls: pd.DataFrame, as_of: str,
            election_date: pd.Timestamp = ELECTION_T1) -> dict:
        """`raw_polls` = sondages BRUTS et riches (niveau candidat, avec institut,
        echantillon, methode, hypothese…). Le modèle décide de son propre nettoyage
        / agrégation. Seule l'agrégation d'affichage (front) est fixée par le contrat."""
        self.load_artifacts()
        nc = self.nowcast(raw_polls, as_of)
        horizon = int((election_date - pd.Timestamp(as_of)).days)
        fc = self.forecast(nc, horizon)
        snap = assemble_snapshot(self, as_of, nc, fc, raw_polls, horizon)
        validate_snapshot(snap)
        return snap


# --- Contrat : assemblage + validation + écriture -----------------------------
def assemble_snapshot(model: ForecastModel, as_of: str, nc: Nowcast, fc: dict,
                      raw_polls: pd.DataFrame, horizon: int) -> dict:
    # Tout ce qui est publié décrit les sondages RÉELLEMENT utilisés, pas ceux
    # reçus en entrée (cf. `ForecastModel.used_polls`) : annoncer 22 sondages
    # quand le modèle en consomme 6 surestimerait la base aux yeux du lecteur.
    used = model.used_polls(raw_polls)
    polls_list = model.display_polls(used)
    n_recus = int(raw_polls["notice"].nunique())
    n_utilises = int(used["notice"].nunique())
    return {
        "model": model.id,
        "as_of": as_of,
        "meta": {
            "genere_le": date.today().isoformat(),
            "as_of": as_of,
            "scrutin_t1": "2027-04-18",
            "n_sondages": n_utilises,
            "n_sondages_disponibles": n_recus,
            "horizon_prevision_jours": horizon,
            "instituts": sorted(used["institut"].unique().tolist()),
            "note": ("Parts au 1er tour par slot (alternatives fusionnées). "
                     "Probabilités, pas des prédictions."),
        },
        "nowcast": nc.summary(),
        "forecast_scrutin": fc["forecast_scrutin"],
        "duels_probables": fc["duels_probables"],
        "drift_modele": fc.get("drift_modele"),
        "diagnostics": nc.diagnostics,
        "polls": polls_list,
    }


def validate_snapshot(s: dict) -> None:
    """Garde-fou du contrat : lève AssertionError si le schéma est invalide."""
    for key in ("model", "as_of", "meta", "nowcast", "forecast_scrutin", "duels_probables"):
        assert key in s, f"clé manquante : {key}"
    fc = s["forecast_scrutin"]
    assert fc, "forecast_scrutin vide"
    tot = 0.0
    for slot, f in fc.items():
        for field in ("part_moyenne", "ic90", "p_qualifie_top2", "p_arrive_premier"):
            assert field in f, f"{slot}: champ {field} manquant"
        assert 0.0 <= f["part_moyenne"] <= 1.0, f"{slot}: part hors [0,1]"
        assert 0.0 <= f["p_qualifie_top2"] <= 1.0, f"{slot}: p_top2 hors [0,1]"
        tot += f["part_moyenne"]
    assert 0.9 <= tot <= 1.1, f"parts ne somment pas à ~1 ({tot:.2f})"


def write_snapshot(snapshot: dict) -> Path:
    """Écrit le snapshot daté + met à jour index.json et models.json."""
    model_id = snapshot["model"]
    mdir = SITE_DATA / model_id
    mdir.mkdir(parents=True, exist_ok=True)
    as_of = snapshot["as_of"]
    (mdir / f"{as_of}.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))

    idx = mdir / "index.json"
    dates = set(json.loads(idx.read_text())) if idx.exists() else set()
    dates.add(as_of)
    idx.write_text(json.dumps(sorted(dates), ensure_ascii=False, indent=2))

    write_manifest()
    return mdir / f"{as_of}.json"


def write_manifest() -> None:
    """(Re)génère site/data/models.json depuis le registre — seuls les
    modèles `public=True` apparaissent (cf. `ForecastModel.public`) ; les
    autres restent exécutables (CLI, tests, backfill) mais hors du
    sélecteur du site."""
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    manifest = [{"id": m.id, "label": m.label, "description": m.description,
                 "trains_on_history": m.trains_on_history}
                for m in sorted(all_models().values(), key=lambda m: m.id) if m.public]
    (SITE_DATA / "models.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
