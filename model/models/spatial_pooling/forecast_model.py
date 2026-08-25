"""`ForecastModel` : branchement de `spatial_pooling` sur le contrat du dépôt.

Le graphe principal du site compare les modèles entre eux, donc il faut qu'ils
parlent du même champ. Ce modèle raisonne par CANDIDAT (Le Pen ET Bardella sont
deux paramètres, c'est tout son intérêt) alors que les autres raisonnent par
SLOT, où le RN est une seule case. Poussé tel quel, le nowcast afficherait le RN
coupé en deux et le lecteur conclurait à un effondrement.

On projette donc le nowcast sur le champ `SLOTS`, qui se trouve être le champ le
plus fréquemment testé par les instituts sur 2026 — ce n'est pas un choix
arbitraire mais l'hypothèse de référence des sondeurs. Tout autre champ reste
accessible sans ré-inférence via `pi_draws_for_mask`, et c'est ce que sert
l'onglet scénarios du site.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from model.core.base import ForecastModel, Nowcast, register
from model.core.live_dataset import SLOTS
from model.core.simulate import forecast_from_draws

from .fit import fit_spatial_pooling, pi_draws_for_mask
from .geometry import V, W
from .roster import MIN_POLL_DATE, build_roster

N_DRAWS_EXPORT = 300


def _slot_de(candidat: str) -> str | None:
    """Nom de slot correspondant à un candidat du roster, s'il y en a un.

    Le roster nomme les candidats par leur patronyme (`Le Pen`, `Dupont-Aignan`)
    là où `SLOTS` porte le nom complet (`Marine Le Pen`) : on teste donc le
    SUFFIXE. Une première version comparait `candidat in nom.split()`, ce qui
    échouait silencieusement sur tous les patronymes composés — Le Pen sortait
    du scénario par défaut, et le nowcast était publié sans le RN.
    """
    for slot, (_, membres) in SLOTS.items():
        if any(candidat == m or m.endswith(" " + candidat) for m in membres):
            return slot
    return None


@register
class SpatialPooling(ForecastModel):
    id = "spatial-pooling"
    label = "Spatial pooling"
    description = (
        "Les candidats occupent une position sur un axe latent gauche-droite et "
        "recrutent par un softmax sur cet axe. Retirer un candidat redistribue ses "
        "électeurs vers ses voisins, par construction — ce qui permet de chiffrer "
        "n'importe quelle liste, y compris jamais sondée."
    )
    trains_on_history = True          # variance d'excès par institut (calibration.py)

    # Validé hors échantillon sur les données 2026 (`notebooks/04l`), pas
    # seulement sur 2021 où il a été réglé : couverture 52,4 / 77,9 / 88,8 pour
    # un nominal 50 / 80 / 90, et redistribution 0,93 pt contre 1,03 pt pour le
    # proportionnel sur 106 paires jamais vues du fit, 75 % des paires gagnées
    # (87 % sur la moitié la plus difficile). Les réglages calés sur 2021
    # transfèrent donc.
    # Rubrique « Scénarios », PAS « suivi » : ce modèle n'est pas fait pour tracer
    # une trajectoire d'opinion, et le présenter dans le sélecteur à côté de
    # `gp-pooling` inviterait à comparer deux choses qui ne répondent pas à la
    # même question. Il alimente l'onglet Scénarios, qui est son usage — et le
    # job quotidien le lance donc à ce titre, sans argument supplémentaire.
    surface = "scenarios"
    # Pas de rejeu des dates passées : un ajustement est une inférence NUTS
    # (~180 s sur le runner GitHub, contre 86 ms pour `gp-pooling`), et RIEN ne
    # relit les snapshots datés de ce modèle — `docs/scenarios.html` ne charge
    # que `models.json` et `scenarios.json`, réécrit à chaque passage. Ce modèle
    # n'a pas à raconter une trajectoire d'opinion : on lui demande les positions
    # et le rapport de force COURANTS. Rejouer six dates lui coûtait 16 min de
    # job pour réécrire des fichiers que personne n'ouvre.
    rejeu_historique = False
    # Artefact sur mesure lu par docs/scenarios.html : ce ne sont pas des
    # résultats pré-calculés mais la GÉOMÉTRIE du postérieur, que le navigateur
    # pousse à travers le softmax masqué pour n'importe quelle liste cochée
    # (cf. `_export_scenarios`). C'est ce qui rend cette rubrique bespoke : le
    # contrat de snapshot ne saurait pas décrire un tel objet.
    scenario_artifact = "scenarios.json"

    def nowcast(self, polls: pd.DataFrame, as_of: str) -> Nowcast:
        fit = fit_spatial_pooling(polls, as_of=as_of)
        _export_scenarios(fit, as_of)

        slots = [(_slot_de(c), i) for i, c in enumerate(fit.candidates)]
        garde = [(s, i) for s, i in slots if s is not None]
        if len(garde) < 2:
            raise RuntimeError(
                f"aucun candidat du roster ne correspond aux SLOTS : {fit.candidates}")
        mask = np.zeros(len(fit.candidates))
        for _, i in garde:
            mask[i] = 1.0
        pi = pi_draws_for_mask(fit, mask)[:, mask.astype(bool)]
        labels = [s for s, _ in garde]

        diag = dict(fit.diagnostics)
        diag["scenario"] = "champ SLOTS (= champ le plus testé sur 2026)"
        diag["candidats_hors_scenario"] = [c for c, (s, _) in
                                           zip(fit.candidates, slots) if s is None]
        return Nowcast(
            draws=xr.DataArray(pi, dims=("draw", "candidat"),
                               coords={"candidat": labels}),
            diagnostics=diag,
        )

    def forecast(self, nowcast: Nowcast, horizon_days: int) -> dict:
        from model.core.opinion import load_law
        from model.core.projection import projeter_au_scrutin
        pi = np.asarray(nowcast.draws.values)
        labels = nowcast.draws.coords["candidat"].values.tolist()
        return forecast_from_draws(projeter_au_scrutin(pi, horizon_days, load_law()),
                                   labels, horizon_days)

    def used_polls(self, raw_polls: pd.DataFrame) -> pd.DataFrame:
        """Le modèle filtre par date plancher ET par roster : sans cette
        surcharge le site annoncerait une base plus large que celle consommée."""
        raw = raw_polls[pd.to_datetime(raw_polls["date_fin"]) >= MIN_POLL_DATE]
        if raw.empty:
            return raw
        cands, _, _ = build_roster(raw)
        return raw[raw["candidat"].isin(cands)]

    def display_polls(self, raw_polls: pd.DataFrame) -> list[dict]:
        """Points du graphe dans le vocabulaire SLOTS, pour rester superposable
        aux autres modèles. On moyenne les hypothèses d'un même sondage qui
        testent le slot — un point par sondage."""
        raw = self.used_polls(raw_polls).copy()
        if raw.empty:
            return []
        raw["slot"] = [_slot_de(c) for c in raw["candidat"]]
        raw = raw[raw["slot"].notna()]
        # part renormalisée sur le champ SLOTS de chaque hypothèse
        raw["hyp"] = raw["hypothese"].fillna("__u__")
        out = []
        for (notice, hyp), g in raw.groupby(["notice", "hyp"], sort=False):
            tot = float(g["intention"].sum())
            if tot <= 0:
                continue
            for r in g.itertuples():
                out.append({"date_fin": str(r.date_fin)[:10], "institut": r.institut,
                            "slot": r.slot, "part": round(float(r.intention) / tot, 4),
                            "notice": r.notice,
                            "notice_url": getattr(r, "notice_url", None)})
        moy: dict[tuple, list] = {}
        for o in out:
            moy.setdefault((o["notice"], o["slot"]), []).append(o)
        return [dict(v[0], part=round(float(np.mean([x["part"] for x in v])), 4))
                for v in moy.values()]


def _export_scenarios(fit, as_of: str) -> None:
    """Écrit les TIRAGES pour l'onglet scénarios du site.

    Le point de l'architecture : pour un champ arbitraire coché par un visiteur,
    `pi` s'obtient en poussant chaque tirage à travers le softmax masqué. C'est
    `O(tirages x B)`, donc faisable dans le navigateur — pas besoin d'un serveur
    ni d'une ré-inférence. On exporte donc la géométrie, pas des résultats
    pré-calculés, et le site peut répondre à n'importe laquelle des 2^N listes.

    300 tirages amincis : l'ESS du fit est de l'ordre de 300, donc au-delà on
    exporterait des tirages corrélés — du poids sans information.
    """
    import json
    from model.core.base import ELECTION_T1, SITE_DATA
    from model.core.opinion import load_law
    law = load_law()

    # Champ le plus fréquemment testé : sert de point de départ à l'onglet, pour
    # que le visiteur parte de l'hypothèse de référence des instituts plutôt que
    # d'une liste arbitraire.
    from collections import Counter
    from .roster import build_poll_arrays
    tm = fit.tested_mask > 0
    modal = Counter(tuple(np.flatnonzero(r)) for r in tm).most_common(1)[0][0]

    S = len(fit.mu_draws)
    sel = np.linspace(0, S - 1, min(N_DRAWS_EXPORT, S)).astype(int)
    r4 = lambda a: [[round(float(x), 4) for x in ligne] for ligne in a]   # noqa: E731
    paquet = {
        "as_of": as_of,
        "candidats": list(fit.candidates),
        "ancres": [round(float(x), 4) for x in fit.anchors],
        "champ_modal": [fit.candidates[i] for i in modal],
        "grille": {"v": [round(float(x), 4) for x in np.asarray(V)],
                   "w": [round(float(x), 6) for x in np.asarray(W)]},
        "tirages": {"mu": r4(fit.mu_draws[sel]), "sigma": r4(fit.sigma_draws[sel]),
                    "w": r4(fit.w_draws[sel])},
        "diagnostics": fit.diagnostics,
        # Loi de projection au scrutin (banque commune) : le navigateur applique
        # la MÊME transformation que `projeter_au_scrutin`, sinon l'onglet
        # afficherait l'incertitude d'aujourd'hui pour une élection dans des mois.
        "projection": {**{k: float(v) for k, v in law.items()
                          if k in ("sigma2", "tau", "delta_scale", "delta_skew", "delta_tail")},
                       "horizon_jours": int((ELECTION_T1 - pd.Timestamp(as_of)).days)},
    }
    d = SITE_DATA / "spatial-pooling"
    d.mkdir(parents=True, exist_ok=True)
    (d / "scenarios.json").write_text(json.dumps(paquet, ensure_ascii=False),
                                      encoding="utf-8")
