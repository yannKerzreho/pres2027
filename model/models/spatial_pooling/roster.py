"""Roster et mise en forme des sondages : quels candidats sont modélisés, où on
les place a priori sur l'axe, et comment un DataFrame de sondages devient les
tableaux `(tested_mask, Y, Np, dates)` que le modèle consomme.

Les tables de placement (`POSITION_ANCHORS`, `ORDER_GROUPS`) sont des choix
ÉDITORIAUX, lâches et corrigeables — pas des mesures. Elles sont ici plutôt que
dans `geometry.py` parce qu'elles portent sur des candidats nommés, pas sur les
maths."""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Roster / fenêtre de campagne (spec §6) --------------------------------------
MIN_POLL_DATE = pd.Timestamp("2026-01-01")   # même repli que bayesian_nowcast.nowcast
MIN_POLLS = 5                                 # sondages RÉELS distincts (`notice`), pas hypothèses
# Un candidat cesse d'être modélisé s'il n'a plus été testé depuis assez
# longtemps (retrait, hypothèse abandonnée par les instituts). NON CALIBRÉ :
# aucun candidat éligible n'en est proche aujourd'hui (le plus ancien est
# Bardella à 16 jours), c'est un garde-fou pour la suite, pas un réglage.
MAX_LAST_POLL_AGE_DAYS = 90
# Pas de la ligne du temps du chemin de `w` (modèle joint). Testé à 7 jours
# pour réduire la dimension : DÉGRADE la convergence (J-150 passe de R-hat
# 1,021 à 1,330). Regrouper force plusieurs sondages à se réconcilier sur une
# même valeur de `w`, ce qui durcit la géométrie -- l'inverse du but recherché.
# Laissé à 1 jour (spec §12.18).
TIME_BIN_DAYS = 1.0

# ANCRES DE POSITION. Elles ne portent QUE l'ordre : la position d'un candidat
# est son rang dans `ORDER_GROUPS`, uniformément réparti sur l'axe. On affirme
# donc qui est à gauche de qui — une connaissance qu'on a — et rien sur les
# distances, qu'on n'a pas.
#
# Un placement expert (grappes serrées à droite, grands écarts au centre) a été
# construit et mesuré. Verdict : les écarts de redistribution entre placements
# sont de l'ordre de 0,01 pt, et sur la coupure la plus difficile la variabilité
# de GRAINE écrase l'effet de configuration — même réglage, ESS de 3 à 112 selon
# la graine. La mesure ne départage donc pas, et le rang est retenu par
# PARCIMONIE, pas parce qu'il a gagné : c'est le choix qui publie le moins
# d'affirmations éditoriales.
#
# Deux propriétés utiles au passage. L'équirépartition en position donne des
# écarts qui CROISSENT vers les extrêmes une fois passés en logit (0,18 au
# centre, 0,74 aux bords sur 21 candidats), ce qui sépare bien les candidats de
# bord. Et elle garantit mécaniquement un écart minimal, là où un placement
# expert peut coller deux candidats au point de les rendre permutables — d'où
# `ancres_ecartees`, qui reste le garde-fou si le placement vient d'ailleurs.
def position_anchors(candidates: list[str]) -> np.ndarray:
    """Ancre de chaque candidat : son rang dans `ORDER_GROUPS`, réparti sur (0,1).

    Lève si un candidat n'est pas au catalogue — un candidat qui franchit le
    seuil doit être placé sur l'axe, et ce placement est un choix éditorial, pas
    un défaut à combler silencieusement.
    """
    rang = {c: i for i, (_, membres) in enumerate(ORDER_GROUPS) for c in membres}
    manquants = [c for c in candidates if c not in rang]
    if manquants:
        raise ValueError(
            f"candidats absents de ORDER_GROUPS : {sorted(manquants)}. Les y placer "
            "— l'ordre gauche-droite est la seule information de position du modèle.")
    ordre = sorted(range(len(candidates)), key=lambda i: rang[candidates[i]])
    place = np.empty(len(candidates), dtype=int)
    place[ordre] = np.arange(len(candidates))
    return (place + 1.0) / (len(candidates) + 1.0)


ORDER_GROUPS: list[tuple[str, list[str]]] = [
    ("LO",         ["Arthaud", "Poutou"]),
    ("LFI",        ["Mélenchon", "Ruffin"]),
    ("coco_ecolo", ["Roussel", "Tondelier"]),
    ("PS",         ["Glucksmann", "Hollande", "Faure"]),
    ("Villepin",   ["Villepin"]),
    ("Attal",      ["Attal", "Lecornu"]),
    ("Philippe",   ["Philippe"]),
    ("LR",         ["Retailleau", "Wauquiez", "Lisnard", "Darmanin"]),
    # Ordre INTERNE gauche->droite : Dupont-Aignan < Bardella < Le Pen. Il était
    # arbitraire tant que les membres d'un groupe partageaient une position (il
    # n'entrait dans aucun calcul) ; il devient une hypothèse ACTIVE dès qu'on
    # passe aux ancres par candidat (§12.26). Cet ordre-ci n'est pas posé a
    # priori : c'est celui que le postérieur 2027 produit spontanément, sans
    # ancre et donc sans qu'on le lui impose — puis validé comme conforme à la
    # réalité politique. Prudence tout de même, les trois partagent un slot dans
    # cette configuration et leur ordre interne y est faiblement identifié.
    ("MLP",        ["Dupont-Aignan", "Bardella", "Le Pen"]),
    ("Zemmour",    ["Zemmour", "Knafo"]),
]

# Affectations PAR DÉFAUT, à confirmer (même statut que les réserves de §2.1
# sur Dupont-Aignan et Hollande) -- aucune n'est active aujourd'hui, tous ces
# candidats étant sous le seuil :
#   - Villepin : groupe PROPRE, placé entre PS et Attal. Placement réellement
#     ambigu (ex-Premier ministre gaulliste, donc à droite par trajectoire,
#     mais qui capte aujourd'hui un électorat de gauche/centre-gauche). C'est
#     le plus proche du seuil (4 sondages, testé le 2026-07-10 à 3,8 %) : à
#     trancher en priorité.
#   - Darmanin, Lisnard -> LR ; Lecornu -> Attal (macronie) ; Knafo -> Zemmour
#     (Reconquête).

# half_life (jours) : calibré par backtest hors-échantillon 2017/2022
# (notebooks/04b_spatial_halflife_backtest.py, cf. spec §5.2) -- 15j nettement
# meilleur que 45j/infini sur les deux élections, mais balayage grossier
# ({15,45,∞}), PAS affiné. Pas échantillonnable par NUTS (vraisemblance
# tempérée, cf. spec §3.1) -- constante calibrée séparément, comme
# `half_life_days` sur linear_pooling (model/models/linear_pooling/model.py).

# --- Roster & mise en forme des sondages -----------------------------------------
def build_roster(raw: pd.DataFrame, as_of=None) -> tuple[list[str], np.ndarray, list[str]]:
    """Roster modélisé : tout candidat testé dans >= MIN_POLLS sondages RÉELS
    distincts (`notice`, pas hypothèses) ET encore testé récemment
    (MAX_LAST_POLL_AGE_DAYS), assigné à son groupe d'ordre.

    Un candidat éligible absent de ORDER_GROUPS **lève une erreur**. Il était
    auparavant écarté avec un simple `warning`, et c'est ce qui a produit le
    défaut le plus grave du modèle (spec §12) : Bardella, testé dans 46 nœuds
    sur 78, disparaissait du roster sans disparaître des données, faisant
    tomber la somme des intentions à 0,66 par nœud face à un `pi` qui somme à
    1. Un candidat non classé n'est pas une nuisance cosmétique : c'est un trou
    dans le bulletin que la vraisemblance ne peut pas absorber. Mieux vaut un
    job qui échoue avec la liste à classer qu'un modèle publié faux.

    Les groupes SANS aucun candidat éligible sont retirés : `ORDER_GROUPS` peut
    donc pré-classer des candidats encore sous le seuil sans que leur groupe
    vide n'occupe une position dans la séquence ordonnée (ce qui resserrerait
    inutilement les autres).
    """
    counts = raw.groupby("candidat")["notice"].nunique()
    eligible = set(counts[counts >= MIN_POLLS].index)

    dates = pd.to_datetime(raw["date_fin"])
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else dates.max()
    last_seen = raw.assign(_d=dates).groupby("candidat")["_d"].max()
    stale = {c for c in eligible
             if (as_of_ts - last_seen[c]).days > MAX_LAST_POLL_AGE_DAYS}
    eligible -= stale

    unclassified = eligible - {m for _, members in ORDER_GROUPS for m in members}
    if unclassified:
        raise ValueError(
            f"candidats éligibles absents de ORDER_GROUPS : {sorted(unclassified)}. "
            "Les classer dans model.py (cf. spec §12) — les ignorer creuserait un "
            "trou de normalisation dans la vraisemblance."
        )

    candidates, slot_of, slot_names = [], [], []
    for gname, members in ORDER_GROUPS:
        members_in = [m for m in members if m in eligible]
        if not members_in:
            continue
        for m in members_in:
            candidates.append(m)
            slot_of.append(len(slot_names))
        slot_names.append(gname)
    return candidates, np.array(slot_of), slot_names


def build_poll_arrays(df: pd.DataFrame, candidates: list[str], notice_col="notice", hyp_col="hypothese",
                      candidat_col="candidat", date_col="date_fin", intention_col="intention",
                      echantillon_col="echantillon", institut_col="institut") -> dict:
    """DataFrame long (une ligne = un candidat testé dans une hypothèse d'un
    sondage) -> (tested_mask, Y, Np, dates, instituts) restreint à
    `candidates`. Même déflation `echantillon / n_hypotheses` que
    `aggregate_to_slots` (model/core/live_dataset.py) : les hypothèses d'un
    même sondage partagent le terrain, pas des mesures indépendantes.
    `instituts` (P,) -- un nom d'institut par nœud, nécessaire pour
    `excess_var_for_nodes` (variance d'excès par institut)."""
    idx = {c: i for i, c in enumerate(candidates)}
    df = df[df[candidat_col].isin(candidates)].copy()
    df[hyp_col] = df[hyp_col].fillna("__unique__")
    g = (df.groupby([notice_col, hyp_col, date_col, echantillon_col, institut_col, candidat_col])[intention_col]
          .sum().reset_index())
    g["n_hyp"] = g.groupby(notice_col)[hyp_col].transform("nunique")

    N = len(candidates)
    tested_mask, Y, Np, dates, instituts = [], [], [], [], []
    Np_full, notices = [], []          # cf. `notice_idx` plus bas (spec §12.7)
    for (notice, hyp), grp in g.groupby([notice_col, hyp_col], sort=False):
        mask = np.zeros(N)
        y = np.zeros(N)
        for r in grp.itertuples(index=False):
            cand = getattr(r, candidat_col)
            if cand in idx:
                mask[idx[cand]] = 1.0
                y[idx[cand]] = getattr(r, intention_col) / 100.0
        if mask.sum() < 2:
            continue
        # SOUS-COMPOSITION (session du 2026-08-12, spec §12) : `spatial_shares`
        # renvoie un `pi` qui somme à 1 sur le champ masqué, alors que `Y`
        # restreint au roster ne somme à 1 que si le roster couvre TOUT le
        # bulletin -- 0,94 en médiane sur 2017/2022 (petits candidats hors
        # blocs), et jusqu'à 0,66 sur 2027 avant la réintégration de Bardella.
        # Comparer les deux tels quels rend la vraisemblance structurellement
        # insatisfiable. La restriction d'un multinomial à un sous-ensemble EST
        # un multinomial : on renormalise `Y` et on déflate `N_p` d'autant
        # (nombre de répondants exprimant une préférence pour un candidat du
        # roster) -- pas d'ajustement ad hoc, c'est la loi exacte.
        tot = float(y[mask > 0].sum())
        if tot <= 0:
            continue
        y = y / tot
        tested_mask.append(mask)
        Y.append(y)
        n_raw = float(grp[echantillon_col].iloc[0])
        Np.append(n_raw / float(grp["n_hyp"].iloc[0]) * tot)
        Np_full.append(n_raw * tot)     # SANS déflation par n_hyp
        notices.append(notice)
        dates.append(pd.Timestamp(grp[date_col].iloc[0]))
        instituts.append(str(grp[institut_col].iloc[0]))

    dates_num = np.array([d.toordinal() for d in dates], dtype=float)

    # Ligne du temps GLOBALE dédupliquée (jours calendaires distincts, pas
    # nœuds) -- nécessaire pour la marche aléatoire dense de
    # `spatial_pooling_model_tau` (spec §3bis) : `date_idx[p]` pointe vers la
    # position du nœud p sur cette ligne, `dt_gaps[m]` l'écart (jours) entre
    # deux dates uniques consécutives. Calculé systématiquement (coût nul
    # pour le modèle tempéré, qui ignore juste ces clés).
    # Ligne du temps regroupée par PAS DE `TIME_BIN_DAYS` jours (spec §12.18).
    # Le chemin de `w` du modèle joint est échantillonné à chaque pas : à la
    # journée, 2022 J-30 en compte 97 (dimension N·M = 1067) alors que le
    # roster 2027 n'en a que 13 (169). On demandait donc au modèle d'estimer
    # l'opinion jour par jour, à une résolution où elle ne bouge pas de façon
    # mesurable -- sur-paramétrisation qui casse NUTS (R-hat 1,59 à 2,60 sur 3
    # coupures 2022 sur 5) sans rien apporter.
    binned = np.floor(dates_num / TIME_BIN_DAYS) * TIME_BIN_DAYS
    unique_dates = np.sort(np.unique(binned))
    date_idx = np.searchsorted(unique_dates, binned)
    dt_gaps = np.diff(unique_dates)

    # `notice_idx` : à quel SONDAGE appartient chaque nœud. Deux hypothèses d'un
    # même sondage sont posées aux MÊMES répondants, leurs erreurs sont donc
    # corrélées -- `weighted_loglik_blocked` s'en sert (spec §12.7). `Np_full`
    # est la taille d'échantillon NON déflatée par `n_hyp` : la déflation est
    # le procédé de repli qui suppose l'indépendance, l'autre vraisemblance
    # modélise la corrélation explicitement et n'en a pas besoin.
    uniq_notices = {n: i for i, n in enumerate(dict.fromkeys(notices))}
    notice_idx = np.array([uniq_notices[n] for n in notices])

    return dict(tested_mask=np.array(tested_mask), Y=np.array(Y), Np=np.array(Np), dates=dates_num,
               unique_dates=unique_dates, date_idx=date_idx, dt_gaps=dt_gaps, instituts=instituts,
               Np_full=np.array(Np_full), notice_idx=notice_idx, n_notices=len(uniq_notices))


# --- Variance d'excès (house effects) -- calibration PROPRE à spatial_pooling ------
def excess_var_for_nodes(instituts: list[str], bank) -> np.ndarray:
    """Variance d'excès PAR NOEUD (P,), en fraction² -- ajoutée à
    `pi(1-pi)/N` dans `weighted_loglik`.

    Historique (spec §6.5) : une première version réutilisait TELLE QUELLE la
    Bank `bayesian_nowcast` (`excess_sigma`, calibrée en espace ILR contre
    l'écart au résultat final) -- mesuré en SUR-correction sévère (IC90
    empirique à 99,6% au lieu de 90% sur 2022) : cette Bank mélange bruit
    d'institut et dérive de campagne dans sa propre décomposition, pas
    directement transférable à la géométrie de `spatial_pooling`.

    Remplacée par une calibration DÉDIÉE (`calibration.py::SpatialExcessCalibration`,
    `excess_sigma_spatial`) -- fit sur le même diagnostic model-free qui a
    révélé le problème (paires de sondages de champ EXACTEMENT identique,
    `notebooks/06b_same_day_poll_coherence_matched.py`), pas emprunté à un
    autre modèle. Vérifié par contrôle postérieur (RMS(z) sur les mêmes
    paires : 1,57 sans excès -> 1,11 avec, cf. spec §6.5)."""
    from model.models.spatial_pooling.calibration import excess_sigma_spatial
    if bank is None:
        return np.zeros(len(instituts))
    return np.array([(excess_sigma_spatial(bank, inst) / 100.0) ** 2 for inst in instituts])

