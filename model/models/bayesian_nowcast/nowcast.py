"""Nowcast du modèle « bayesian-nowcast » : sondages 2027 -> parts par slot.

SSM linéaire-gaussien (docs/spec_ssm_nowcast.md) : l'opinion est un état latent
`alpha_t` (coordonnées ILR, `latent.py`) qui marche aléatoirement entre les
dates de sondage — pas la quantité statique unique d'avant (tous les sondages,
d'hier ou d'il y a 4 ans, comme observations bruitées de LA MÊME valeur). Le
chemin reste gaussien du début à la fin, marginalisé analytiquement par le
filtre de Kalman de `model/core/lgssm.py` (JAX pur, différentiable) ; NumPyro garde le
rôle d'orchestrateur des hyperparamètres (`tau`, `biais`), injectés dans le
chemin marginal via `numpyro.factor` — cf. spec §4 pour la justification
complète de cette architecture (évite d'échantillonner un paramètre latent par
nœud temporel en NUTS).

Chaque sondage est un nœud (pas un état par jour calendaire, cf. spec §5) ; la
Bank de `calibration.py` informe les priors (`institut_bias`, `tau`) si elle
existe (calibration optionnelle — sans elle, priors faibles).

`aggregate_to_slots` reste un CHOIX de ce modèle, pas une étape imposée par le
framework : un futur modèle candidat-par-candidat travaillerait directement
sur `load_raw_polls()` (brut, avec `scenario_id`) sans passer par ici.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.nn
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import xarray as xr
from model.core.bank import Bank
from model.core.base import ForecastModel, Nowcast, register
from model.core.inference import run_numpyro_mcmc
from model.core.lgssm import kalman_filter
from model.core.live_dataset import SLOTS, aggregate_to_slots
from model.core.simulate import forecast_from_draws
from pipeline.historical import load_calibration_frame, load_results

from .calibration import (
    BANK_PATH, JUMP_BANK_PATH, HouseEffectsCalibration, TerminalJumpCalibration,
    institut_bias_prior, measurement_sigma, sampling_sigma,
)
from .latent import PollPlan, helmert_basis, ilr_decode, ilr_encode, poll_observation_values, poll_plan

# Analogue 2022 de chaque slot 2027 — un slot SANS entrée ici (Ruffin,
# nouvel entrant sans candidature 2022) reçoit NEW_ENTRANT_FLOOR (§ci-dessous).
# Noms EXACTS de data/historical/results_2022.csv (pas de fuzzy-matching ici :
# ce mapping sert un but différent de celui, plus permissif, de
# `live_dataset.map_candidate` — associer un slot 2027 à son analogue de
# CAMPAGNE PASSÉE, pas matcher un nom de sondage brut).
HISTORICAL_2022_ANALOG = {
    "RN": "Marine Le Pen",
    "Zemmour (Reconquête)": "Eric Zemmour",
    "Dupont-Aignan": "Nicolas Dupont-Aignan",
    "Mélenchon (LFI)": "Jean-Luc Mélenchon",
    "Poutou (NPA)": "Philippe Poutou",
    "Arthaud (LO)": "Nathalie Arthaud",
    "Roussel (PCF)": "Fabien Roussel",
    "Glucksmann (PP)": "Anne Hidalgo",
    "Philippe (Horizons)": "Emmanuel Macron",
    "Retailleau (LR)": "Valérie Pécresse",
    "Tondelier (EELV)": "Yannick Jadot",
}
NEW_ENTRANT_FLOOR = 1.0   # part (%) plancher pour un slot sans analogue 2022


def historical_prior_pi(slots: list[str]) -> np.ndarray:
    """Prior sur la composition au premier nœud, informé par les résultats
    2022 (par candidat, mappé au slot 2027 analogue) — PAS un prior uniforme
    sur les K slots : on CONNAÎT les rapports de force (RN/Centre/Mélenchon
    grands, Arthaud/Poutou petits), un prior ILR centré à 0 (parts égales)
    serait faux, pas seulement trop large (cf. docs/spec_ssm_implementation.md
    §7 — c'est distinct du problème de VARIANCE : ici c'est la MOYENNE du
    prior qui était mal choisie). Un slot sans analogue 2022 (nouvel entrant,
    ex. Ruffin) reçoit `NEW_ENTRANT_FLOOR`. Renormalisé à 100 après floor —
    léger écart aux vrais totaux 2022 par bloc, sans conséquence : c'est un
    PRIOR, corrigé dès les premiers sondages 2027 réels (pas circulaire —
    contrairement à un prior basé sur le premier sondage 2027 lui-même)."""
    res2022 = load_results()
    res2022 = res2022[(res2022["election"] == 2022) & (res2022["tour"] == "Premier tour")]
    lookup = dict(zip(res2022["candidat"], res2022["resultat"]))
    shares = np.array([lookup.get(HISTORICAL_2022_ANALOG.get(s), NEW_ENTRANT_FLOOR) for s in slots])
    return shares / shares.sum() * 100.0


# Taille d'échantillon EFFECTIVE attribuée au prior 2022 (`historical_prior_pi`),
# comme s'il s'agissait d'un sondage virtuel — plus grand = prior plus
# confiant. 300 ≈ un sondage modeste : assez pour éviter la dégénérescence
# softmax sur un candidat jamais testé en 2026 (cf. §9), pas assez pour
# écraser les tout premiers vrais sondages 2027 s'ils divergent. Utilisé
# comme repli pour les candidats qu'aucune hypothèse du 1er jour ne teste
# (cf. `HISTORICAL_PRIOR_N_IF_TESTED` pour ceux qu'elle teste).
HISTORICAL_PRIOR_N = 300.0

# Poids du repli 2022 pour un candidat déjà testé par le 1er jour de sondage
# réel — nettement plus faible que `HISTORICAL_PRIOR_N` (le vrai sondage doit
# dominer), mais PAS quasi nul (cf. `historical_prior_cov` pour la dérivation
# complète : un N quasi nul fait exploser l'écart-type d'un candidat peu
# testé, ce qui déclenche un effet Jensen qui aggrave la moyenne rapportée
# plutôt que de la corriger). N=5 balayé empiriquement comme meilleur
# compromis (session du 2026-08-10) sur le cas réel RN/27 février 2026 :
# moyenne à 33,6% (sondage réel : 34,5%) avec un écart-type de 2,8 points,
# comparable à l'incertitude d'échantillonnage d'un vrai sondage.
HISTORICAL_PRIOR_N_IF_TESTED = 5.0


def historical_prior_cov(pi_pct: np.ndarray, V: np.ndarray,
                         n: float | np.ndarray = HISTORICAL_PRIOR_N) -> np.ndarray:
    """Covariance ILR du prior initial par delta-method — PAS isotrope
    (`n_prior_var · I`, essayé puis affiné, cf. docs/spec_ssm_implementation.md
    §9-10) : même formule que le bruit d'échantillonnage d'un vrai sondage
    (`latent.py:poll_observation_values`), appliquée au prior 2022 traité
    comme un sondage virtuel de taille `n`. Dérivation (delta-method sur
    log(p), puis projection sur l'hyperplan à somme nulle via `V`, cf.
    spec_ssm_espace_latent.md §2 pour la projection P=VVᵀ) :

        Cov(alpha_ILR) = Vᵀ · diag(1/(n·p)) · V

    Donne une variance ABSOLUE plus resserrée pour un petit candidat (ex.
    Poutou, jamais testé en 2026) que pour un grand — pas une variance
    arbitraire uniforme entre eux : le choc de la marche aléatoire ET
    l'incertitude du prior sont donc, l'un comme l'autre, exprimés dans la
    même géométrie proportionnelle (log-ratio), cohérent avec le chemin
    (§2) plutôt que traités par un cas particulier.

    `n` accepte un vecteur (K,) en plus du scalaire — utilisé par
    `fuse_first_date_observations` (session du 2026-08-10) pour donner un
    poids ALLÉGÉ (PAS quasi nul) au repli 2022 sur les candidats déjà testés
    le 1er jour de sondage réel, plutôt qu'un poids comparable à une vraie
    donnée fraîche (mesuré : -7,5 points sur le RN au 27 février 2026, prior
    2022 ~23,7% à N=300 pesant presque autant que les 2 hypothèses Le Pen
    réelles ce jour-là).

    **Le dosage compte** : un premier essai à N quasi nul (10⁻²) corrige bien
    la moyenne isolée (softmax(mu) -> 34%) mais fait exploser l'écart-type de
    RN à 15 points (plus assez de données pour contraindre grand-chose une
    fois le repli supprimé) — sous cette incertitude, l'effet Jensen
    (`E[softmax(X)] != softmax(E[X])`, cf. `Nowcast.summary()`) fait
    RETOMBER la moyenne RAPPORTÉE (moyenne des tirages, pas mode) à ~23%,
    PIRE qu'avant. Vérifié par calcul direct (200k tirages, hors NUTS) :
    l'écart moyenne/mode grandit avec l'écart-type, donc supprimer TOUTE
    incertitude prior fait plus de mal que de bien. Un dosage MODÉRÉ
    (`HISTORICAL_PRIOR_N_IF_TESTED`, balayé empiriquement : N=5 est le
    meilleur compromis, moyenne à 33,6% avec un écart-type de 2,8 points —
    comparable à l'incertitude d'échantillonnage d'un vrai sondage, pas une
    explosion) donne le meilleur résultat : assez faible pour laisser la
    vraie donnée dominer la moyenne, assez fort pour ne pas déclencher
    l'effet Jensen."""
    p = pi_pct / 100.0
    return V.T @ np.diag(1.0 / (n * p)) @ V

# Design effect appliqué au bruit d'échantillonnage de CHAQUE sondage
# (`latent.poll_observation_values`) : `n_effectif = n / deff`.
#
# Un sondage d'opinion français n'est pas un tirage aléatoire simple (panels
# en ligne, quotas, redressement, filtre « certain d'aller voter ») et son
# erreur réelle ne se réduit pas au `p(1-p)/n` du multinomial. Mesuré
# directement, sans modèle : on apparie les scénarios de DEUX instituts
# différents testant EXACTEMENT le même champ de candidats à ≤ 3 jours
# d'écart (2017+2022, 8138 paires candidat×scénario ; le même jour
# exactement, 1478 paires) — à champ et à date identiques, l'écart entre les
# deux ne peut venir que du bruit d'observation, pas d'un mouvement
# d'opinion. Résultat : `E[(pa-pb)²] / E[Var_multinomiale]` = **2,16**
# (≤3j) et **1,99** (même jour).
#
# La FORME de cet excès a été testée contre deux alternatives (même jeu de
# paires, RMS(z) par tranche de `p`, 1,00 = calibration parfaite) :
#
#   modèle                              p≈2   p≈6   p≈10  p≈15  p≈22  p≈27
#   multinomial brut                    1,55  1,73  1,79  1,58  1,37  1,40
#   deff × multinomial (RETENU)         1,06  1,17  1,21  1,07  0,93  0,95
#   multinomial + c² constant en points 0,67  1,05  1,24  1,20  1,10  1,17
#   multinomial + (kappa·p)² (log-ratio) 1,45  1,41  1,34  1,05  0,83  0,77
#
# Seul le facteur MULTIPLICATIF est plat sur toute la plage de `p` : un
# excès additif constant en points sur-couvre massivement les petits
# candidats, un excès isotrope en log-ratio (qui aurait été le plus naturel
# à ajouter dans l'espace ILR) a la pente inverse. D'où `deff` et pas un
# terme `sigma_obs²·I` ajouté à `R`.
#
# 2,0 retenu plutôt que 2,16 : les paires à ≤3j incluent un vrai mouvement
# d'opinion en plus du bruit d'observation ; l'estimation à gap=0 (1,99) est
# la seule qui isole le bruit pur.
DEFF_DEFAULT = 2.0

# Écart temporel plancher (jours) ajouté à CHAQUE Δt du chemin — évite une
# covariance de transition exactement singulière quand deux sondages tombent
# le même jour (Δt=0) ou quand le dernier sondage tombe le jour même de
# `as_of` ; négligeable partout ailleurs (spec §5 : nœuds quasi gratuits).
MIN_DT_EPS = 1e-6

DEFAULT_SSM_PRIORS_CFG = {
    "tau_prior_scale_default": 1.0,     # repli si pas de Bank (cf. institut_bias_prior)
    # tau_tendance / sigma_beta0 (état de tendance, cf. docs/spec_ssm_tendance.md
    # §4) : pas d'ancrage historique 2017/2022 disponible sans travail dédié de
    # conversion d'unités (campaign_drift_sd est fitté en points de % / √horizon,
    # pas en ILR / jour — même piège que celui déjà identifié sur tau) et,
    # de toute façon, seulement 2 élections pour une quantité de second ordre
    # (volatilité DE LA tendance) — pire ratio paramètres/données que
    # `use_biais`. Priors faibles, ESTIMÉS par NUTS (pas de constante fixée en
    # dur) : c'est aux données 2027 elles-mêmes de les positionner.
    "tau_tendance_prior_scale": 1.0,
    "beta0_prior_scale": 1.0,
}

def fuse_first_date_observations(mu0: np.ndarray, Sigma0: np.ndarray, plans: list[PollPlan],
                                 raw_values: list[np.ndarray], n_echant: list[float],
                                 Km1: int, deff: float = DEFF_DEFAULT,
                                 isotropic_obs_noise: bool = False
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Fusion bayésienne (forme précision, mise à jour de Kalman standard) de
    plusieurs observations survenues au MÊME jour (Δt=0 entre elles, donc
    aucune diffusion à appliquer entre elles — indépendant de `tau`, calculable
    hors du modèle NumPyro) sur un état de départ `(mu0, Sigma0)`.

    Sert à construire le prior du 1er nœud à partir du 1er sondage RÉEL 2027
    lui-même plutôt que du seul résultat 2022 (option A, session du
    2026-08-10) : « un sondage est une estimation de ce qu'on cherche à
    estimer » — on lui applique la même loi d'échantillonnage
    (`poll_observation_values`, delta-method) que n'importe quelle autre
    observation du modèle, pas un traitement spécial. `mu0`/`Sigma0` restent
    un repli (faible) pour les slots que le 1er jour ne teste PAS (ex. Ruffin/
    Poutou au 27 février 2026, cf. docs/spec_ssm_tendance.md) — jamais recopiés
    pour les slots testés, où le vrai sondage domine entièrement.

    Le nœud/les nœuds de ce premier jour sont ensuite RETIRÉS du chemin
    séquentiel (cf. `NowcastData.from_dataframe`) : comptés une seule fois,
    ici, jamais une seconde fois comme observation du filtre — double-compter
    la même information gonflerait artificiellement la précision (division
    par 2 démontrable sur un cas dégénéré à une seule observation)."""
    Lambda = np.linalg.inv(Sigma0)
    eta = Lambda @ mu0
    for plan, values, n in zip(plans, raw_values, n_echant):
        y, R = poll_observation_values(values, plan, Km1, n, deff=deff,
                                       isotropic=isotropic_obs_noise)
        y, R = np.asarray(y), np.asarray(R)
        H = plan.H
        Rinv = np.linalg.inv(R)
        Lambda = Lambda + H.T @ Rinv @ H
        eta = eta + H.T @ Rinv @ y
    Sigma = np.linalg.inv(Lambda)
    mu = Sigma @ eta
    return mu, Sigma


# Les sondages "prémonitoires" 2022/2023 (publiés juste après 2022, sur une
# hypothèse de champ 2027 très différente du champ réel qui se dessine avec
# la campagne 2026) sont un régime différent des sondages de campagne — pas
# une continuation plausible de la MÊME marche aléatoire jour-à-jour. Les
# inclure crée un silence de ~860 jours avant la reprise des sondages de
# campagne : `tau` doit alors expliquer à la fois des écarts serrés (régime
# "campagne") et ce silence énorme avec le MÊME paramètre global, ce qui
# dégrade le mixing NUTS (cf. docs/spec_ssm_implementation.md §8). Restreindre
# aux sondages de la campagne réelle règle la cause plutôt que le symptôme.
MIN_POLL_DATE = pd.Timestamp("2026-01-01")

# Noyau de slots "présidentiels" suffisamment testés pour servir de filtre de
# cohérence de roster sans vider complètement l'échantillon. Idée
# expérimentale (session du 2026-08-11) : le modèle suppose une liste fixe de
# candidats, donc une hypothèse qui remplace l'un de ces slots centraux par
# une alternative (ex. Bardella au lieu de RN/Le Pen, Attal au lieu de
# Philippe) injecte un signal hors-cible ; on peut choisir d'ignorer ces
# scénarios tout en gardant les petits candidats rarement testés (Poutou,
# Ruffin) hors du noyau pour ne pas tomber à zéro donnée.
COHERENT_ROSTER_SLOTS = (
    "Arthaud (LO)",
    "Mélenchon (LFI)",
    "Tondelier (EELV)",
    "Glucksmann (PP)",
    "Roussel (PCF)",
    "RN",
    "Philippe (Horizons)",
    "Retailleau (LR)",
    "Zemmour (Reconquête)",
    "Dupont-Aignan",
)
EXACT_ROSTER_12_SLOTS = tuple(SLOTS)


def filter_scenarios_by_required_slots(raw_polls: pd.DataFrame,
                                       required_slots: tuple[str, ...] | list[str] | None
                                       ) -> pd.DataFrame:
    """Ne garde que les scénarios (`scenario_id`) dont le roster observé
    contient tous les `required_slots`.

    Le modèle de base agrège aujourd'hui TOUTES les hypothèses d'un même
    sondage en observations partielles, même lorsqu'elles testent une liste de
    candidats éloignée de la liste fixe du modèle (ex. RN absent car Bardella
    testé à la place). Ce filtre permet d'expérimenter une version plus
    "cohérente roster" : seules les hypothèses contenant un noyau choisi de
    slots sont considérées comme informatives pour CE modèle.

    `required_slots=None` ou vide : no-op, renvoie `raw_polls` inchangé."""
    if not required_slots:
        return raw_polls
    required = set(required_slots)
    keep_ids: list[str] = []
    for scenario_id, grp in raw_polls.groupby("scenario_id", sort=False):
        slots = set(grp["slot"].dropna())
        if required.issubset(slots):
            keep_ids.append(scenario_id)
    return raw_polls[raw_polls["scenario_id"].isin(keep_ids)].copy()


def filter_scenarios_by_exact_slots(raw_polls: pd.DataFrame,
                                    exact_slots: tuple[str, ...] | list[str] | None
                                    ) -> pd.DataFrame:
    """Ne garde que les scénarios dont le roster observé est EXACTEMENT celui demandé."""
    if not exact_slots:
        return raw_polls
    exact = set(exact_slots)
    keep_ids: list[str] = []
    for scenario_id, grp in raw_polls.groupby("scenario_id", sort=False):
        slots = set(grp["slot"].dropna())
        if slots == exact:
            keep_ids.append(scenario_id)
    return raw_polls[raw_polls["scenario_id"].isin(keep_ids)].copy()


@dataclass
class NowcastData:
    K: int
    slot_names: list[str]
    V: np.ndarray                          # (K, K-1), base ILR (Helmert)
    plans: list[PollPlan]                  # un par sondage, TRIÉS par date
    raw_values: list[np.ndarray]           # parts BRUTES testées par sondage (m_i,)
    n_echant: list[float]
    bias_group_tested: list[np.ndarray]    # (m_i,) indices dans bias_group_names, par slot testé
    dt_forward: np.ndarray                 # (n_polls,) Δt vers le nœud SUIVANT (dernier = inutilisé)
    as_of_dt: float                        # Δt dernier sondage -> as_of (extrapolation)
    init_gap_days: float                   # Δt du 1er jour (fusionné dans init_mean/cov) au 1er nœud restant
    bias_group_names: list[tuple[str, str]]
    bias_prior_mean: np.ndarray
    bias_prior_sd: np.ndarray
    tau_prior_scale: float
    init_mean: np.ndarray                  # (K-1,) état fusionné au 1er jour de sondage, cf. _fuse_observations
    init_cov: np.ndarray                   # (K-1,K-1) covariance ILR associée
    use_biais: bool = False                # cf. bayesian_nowcast_ssm_model
    use_tendance: bool = False             # cf. bayesian_nowcast_ssm_model
    use_tau_candidat: bool = False         # cf. bayesian_nowcast_ssm_model
    roster_filter_slots: tuple[str, ...] = ()
    roster_filter_exact: bool = False
    deff: float = DEFF_DEFAULT             # cf. DEFF_DEFAULT
    isotropic_obs_noise: bool = False      # cf. latent.poll_observation_values

    @classmethod
    def from_dataframe(cls, raw_polls: pd.DataFrame, as_of: str, bank: Bank | None,
                       use_biais: bool = False, use_tendance: bool = False,
                       historical_prior_n: float = HISTORICAL_PRIOR_N,
                       historical_prior_n_if_tested: float = HISTORICAL_PRIOR_N_IF_TESTED,
                       full_covariance: bool = False, use_tau_candidat: bool = False,
                       roster_filter_slots: tuple[str, ...] | list[str] | None = None,
                       roster_filter_exact: bool = False,
                       deff: float = DEFF_DEFAULT, isotropic_obs_noise: bool = False
                       ) -> "NowcastData":
        """Stratégie de CE modèle : fusionne les alternatives mutuellement
        exclusives en slots (aggregate_to_slots), un nœud SSM par **hypothèse**
        de sondage (pas par sondage — un sondage à plusieurs hypothèses en
        produit autant de nœuds, à la même date, `dt=0` entre eux) — un
        sondage 2027 ne teste jamais les K slots simultanément (cf.
        `latent.py`), `poll_plan` gère le padding, y compris pour une
        hypothèse ne testant qu'une minorité de slots. L'échantillon de
        chaque nœud est déflaté par `n_hypotheses` (cf. `aggregate_to_slots`) :
        les hypothèses d'un même sondage partagent le même terrain, ce ne sont
        pas des mesures indépendantes. Sondages antérieurs à `MIN_POLL_DATE`
        écartés (cf. sa docstring). `use_biais`/`use_tendance` : cf.
        `bayesian_nowcast_ssm_model`. `historical_prior_n` : cf.
        `historical_prior_cov` — exposé ici (plutôt que la seule constante
        module `HISTORICAL_PRIOR_N`) pour permettre de comparer des poids de
        prior sans dupliquer `from_dataframe` (session du 2026-08-09 : le
        prior 2022 par défaut, N=300, s'est révélé à 10-11 points du premier
        sondage 2027 réel — RN, Horizons, Mélenchon — un poids largement
        excessif vu l'écart, cf. docs/spec_ssm_tendance.md). `full_covariance` :
        cf. `latent.poll_plan` — choix du SYSTÈME DE COORDONNÉES local (ILR
        orthonormé par défaut, ALR local si `True`), sans effet sur le
        postérieur sous la covariance delta-method (théorème d'invariance,
        spec §5.2). `deff` / `isotropic_obs_noise` : cf. `DEFF_DEFAULT` et
        `latent.poll_observation_values` — c'est là que se joue la largeur
        des intervalles."""
        raw_polls = raw_polls[pd.to_datetime(raw_polls["date_fin"]) >= MIN_POLL_DATE]
        if roster_filter_exact:
            raw_polls = filter_scenarios_by_exact_slots(raw_polls, roster_filter_slots)
        else:
            raw_polls = filter_scenarios_by_required_slots(raw_polls, roster_filter_slots)
        obs = aggregate_to_slots(raw_polls)
        if obs.empty:
            raise ValueError("Aucune hypothèse compatible avec le filtre de roster demandé.")
        slots = list(SLOTS)
        K = len(slots)
        slot_idx = {s: i for i, s in enumerate(slots)}
        V = helmert_basis(K)

        pairs = list(dict.fromkeys(zip(obs["institut"], obs["bloc"])))
        pair_idx = {p: i for i, p in enumerate(pairs)}
        bias_prior = [institut_bias_prior(bank, inst, bloc) for inst, bloc in pairs]
        bias_prior_mean = np.array([m for m, _ in bias_prior])
        bias_prior_sd = np.array([s for _, s in bias_prior])

        plans, raw_values, n_echant, bias_group_tested, dates = [], [], [], [], []
        for _, grp in obs.groupby(["notice", "hypothese"], sort=False):
            shares = np.full(K, np.nan)
            bg_per_slot = np.full(K, -1, dtype=int)
            for r in grp.itertuples():
                si = slot_idx[r.slot]
                shares[si] = r.part
                bg_per_slot[si] = pair_idx[(r.institut, r.bloc)]
            plan = poll_plan(shares, V, full_covariance=full_covariance)
            plans.append(plan)
            raw_values.append(shares[plan.tested])
            bias_group_tested.append(bg_per_slot[plan.tested])
            n_echant.append(float(grp["echantillon"].iloc[0]) / float(grp["n_hypotheses"].iloc[0]))
            dates.append(pd.Timestamp(grp["date_fin"].iloc[0]))

        order = np.argsort(dates)
        dates = [dates[i] for i in order]
        plans = [plans[i] for i in order]
        raw_values = [raw_values[i] for i in order]
        n_echant = [n_echant[i] for i in order]
        bias_group_tested = [bias_group_tested[i] for i in order]

        as_of_dt = max((pd.Timestamp(as_of) - dates[-1]).days, 0)  # sur la liste COMPLETE, cf. ci-dessous

        tau_prior_scale = (float(bank.campaign_drift_sd.item()[0]) if bank is not None
                          else DEFAULT_SSM_PRIORS_CFG["tau_prior_scale_default"])

        # Option A (session du 2026-08-10) : le prior du 1er nœud vient du 1er
        # JOUR de sondage réel 2027 lui-même (fusionné avec un repli 2022
        # faible pour les slots qu'il ne teste pas), pas du seul résultat
        # 2022 — cf. `fuse_first_date_observations`. Ses nœuds sont donc
        # RETIRÉS du chemin séquentiel ci-dessous (comptés une seule fois).
        #
        # `n` VARIE par candidat : poids ALLÉGÉ (`historical_prior_n_if_tested`,
        # PAS quasi nul — cf. `historical_prior_cov` pour pourquoi un poids
        # quasi nul déclenche un effet Jensen qui aggrave la moyenne
        # rapportée) du repli 2022 sur les candidats déjà testés le 1er jour,
        # poids standard (`historical_prior_n`) sur ceux qu'il ne teste pas
        # (ex. Ruffin/Poutou au 27 février 2026).
        Km1 = V.shape[1]
        first_date = dates[0]
        is_first_day = [d == first_date for d in dates]
        tested_day1 = set()
        for keep, plan in zip(is_first_day, plans):
            if keep:
                tested_day1.update(plan.tested.tolist())
        n_par_slot = np.array([historical_prior_n_if_tested if i in tested_day1
                               else historical_prior_n for i in range(K)])

        hist_pi = historical_prior_pi(slots)
        fallback_mean = ilr_encode(hist_pi, V)
        fallback_cov = historical_prior_cov(hist_pi, V, n=n_par_slot)
        init_mean, init_cov = fuse_first_date_observations(
            fallback_mean, fallback_cov,
            [p for p, keep in zip(plans, is_first_day) if keep],
            [v for v, keep in zip(raw_values, is_first_day) if keep],
            [n for n, keep in zip(n_echant, is_first_day) if keep], Km1,
            deff=deff, isotropic_obs_noise=isotropic_obs_noise)

        dates = [d for d, keep in zip(dates, is_first_day) if not keep]
        plans = [p for p, keep in zip(plans, is_first_day) if not keep]
        raw_values = [v for v, keep in zip(raw_values, is_first_day) if not keep]
        n_echant = [n for n, keep in zip(n_echant, is_first_day) if not keep]
        bias_group_tested = [b for b, keep in zip(bias_group_tested, is_first_day) if not keep]

        n_polls = len(dates)
        if n_polls == 0:
            # Le 1er jour est aussi le SEUL jour disponible à cette date
            # `as_of` (ex. tout début de backfill) : rien ne reste après la
            # fusion, l'extrapolation ci-dessous part directement de
            # `init_mean`/`init_cov`.
            init_gap_days = 0.0
            dt_forward = np.array([])
        else:
            init_gap_days = max((dates[0] - first_date).days, 0)
            dt_forward = np.ones(n_polls)  # dernier = inutilisé par le filtre
            for i in range(n_polls - 1):
                dt_forward[i] = max((dates[i + 1] - dates[i]).days, 0)

        return cls(K=K, slot_names=slots, V=V, plans=plans, raw_values=raw_values,
                  n_echant=n_echant, bias_group_tested=bias_group_tested,
                  dt_forward=dt_forward, as_of_dt=as_of_dt, init_gap_days=init_gap_days,
                  bias_group_names=pairs, bias_prior_mean=bias_prior_mean, bias_prior_sd=bias_prior_sd,
                  tau_prior_scale=tau_prior_scale, init_mean=init_mean, init_cov=init_cov,
                  use_biais=use_biais, use_tendance=use_tendance, use_tau_candidat=use_tau_candidat,
                  roster_filter_slots=tuple(roster_filter_slots or ()),
                  roster_filter_exact=roster_filter_exact,
                  deff=deff, isotropic_obs_noise=isotropic_obs_noise)


def bayesian_nowcast_ssm_model(data: NowcastData, priors_cfg: dict | None = None):
    """NumPyro + filtre de Kalman : chemin gaussien marginalisé analytiquement (filtre
    de Kalman), hyperparamètres et `biais` échantillonnés par NUTS. Débiaisage
    en points de % AVANT la transformation ILR (pas un terme additif dans
    l'espace log-ratio) : préserve les unités déjà calibrées de
    `institut_bias` et son raffinement bayésien par les données du cycle en
    cours, comme le modèle Dirichlet d'avant (cf. session de conception,
    docs/spec_ssm_nowcast.md).

    État étendu (niveau, tendance) — local linear trend, cf.
    docs/spec_ssm_tendance.md pour la dérivation complète (§2 : une marche
    sans dérive produit un décalage STATIONNAIRE, pas transitoire, face à une
    tendance d'opinion soutenue ; mesuré concrètement à -2,5/-3,5 points sur
    le RN mi-2026). `alpha` (niveau, ILR) évolue maintenant sous l'effet de
    `beta` (tendance, même dimension) : `alpha_t = alpha_{t-1} + beta_{t-1}·Δt
    + bruit`, `beta_t = beta_{t-1} + bruit`. La matrice de transition F(Δt)
    est une identité CINÉMATIQUE (Δt est connu, la date des sondages) — rien
    de plus n'est appris par NUTS que les DEUX échelles de bruit
    (`tau_niveau`, `tau_tendance`, contre une seule aujourd'hui) plus
    `sigma_beta0` (incertitude sur la tendance au tout premier nœud) : les
    trois estimés par NUTS avec un prior faible (§4 de la spec — pas
    d'ancrage historique fiable disponible, moins mauvais qu'une constante
    fixée en dur sans justification).

    `use_tendance` optionnel, même esprit que `use_biais` juste en dessous —
    mais un VRAI branchement Python (deux chemins de code disjoints), pas un
    epsilon pour mettre `tau_tendance`/`sigma_beta0` à ~0 dans la même algèbre
    2·(K-1) : essayé, et ça casse NUTS (covariance quasi-singulière, rapport
    ~10^13 entre bloc niveau et bloc tendance -> R-hat mesuré entre 180 et
    1950, ESS=1 sur toutes les chaînes, cf. session du 2026-08-09/10). La
    vérification empirique (§6 de la spec) a par ailleurs trouvé un problème
    d'identifiabilité sur les dates creuses côté tendance elle-même (momentum
    RN à -16,7 pts/mois au 27 février 2026, alors que RN montait) — AVANT de
    conclure quoi que ce soit sur la tendance, on isole d'abord l'effet du
    prior initial (cf. `historical_prior_n`, très probablement la cause
    dominante du décalage observé) avec `use_tendance=False`, qui repasse par
    l'ANCIEN modèle (K-1 dimensions, F=I, un seul `tau`) — pas juste "moins
    de tendance", vraiment le même chemin de code qu'avant l'ajout.

    `use_tau_candidat` (session du 2026-08-10, cf. docs/spec_ssm_encodage_partiel.md
    §9) : `tau_niveau` unique et isotrope force TOUS les candidats à
    partager le même rythme jour-à-jour, alors que leur volatilité réelle
    diffère fortement (mesuré : Philippe 1,40 pt/jour, RN 0,50) — et,
    surtout, un candidat peu testé (RN, cf. exclusion Bardella de `SLOTS`)
    voit sa variance gonfler librement entre ses rares tests, ce qui le rend
    « bon marché » à déplacer pour le gain de Kalman dès qu'un AUTRE candidat
    de la MÊME hypothèse a une petite surprise — même sans corrélation a
    priori (vérifié : décorréler la covariance jointe ne change RIEN, la
    cause est la variance individuelle de RN, pas sa corrélation aux
    autres). `tau_candidat` (K,) donne un rythme propre à CHAQUE candidat en
    espace CLR (`Q_CLR = diag(tau_candidat²)`), projeté dans l'espace ILR
    via `V` (`Q_unit = Vᵀ · diag(tau_candidat²) · V`, PSD, cohérent avec la
    géométrie compositionnelle déjà en place) — pas un paramètre par
    direction ILR (qui ne correspond à aucun candidat en particulier), un
    paramètre par candidat RÉEL, projeté ensuite. `K` paramètres au lieu de
    1 : risque de sous-identification pour les candidats jamais testés
    (repli sur le prior, pas un bug) — à vérifier empiriquement (R-hat/ESS)
    avant de l'adopter en défaut de production."""
    cfg = {**DEFAULT_SSM_PRIORS_CFG, **(priors_cfg or {})}
    Km1 = data.V.shape[1]
    K = data.V.shape[0]
    n_polls = len(data.plans)

    if data.use_tau_candidat:
        tau_candidat = numpyro.sample("tau_candidat",
                                      dist.HalfNormal(jnp.full(K, data.tau_prior_scale)))
        numpyro.deterministic("tau_candidat_moyen", jnp.mean(tau_candidat))
        V_jnp = jnp.asarray(data.V)
        Q_unit = V_jnp.T @ jnp.diag(tau_candidat ** 2) @ V_jnp
    else:
        tau_niveau = numpyro.sample("tau_niveau", dist.HalfNormal(data.tau_prior_scale))
        Q_unit = tau_niveau ** 2 * jnp.eye(Km1)
    # `biais` optionnel (use_biais) : avec ~20-45 groupes institut×bloc et
    # parfois aussi peu que 5 sondages disponibles en début de backfill, le
    # postérieur joint (tau, biais) devient mal identifié — `tau` explore des
    # valeurs aberrantes (ex. 0.62±0.60 sur 2026-04-30, contre <0.03 une fois
    # `biais` retiré) : pas assez de données pour contraindre autant de
    # paramètres. Les biais eux-mêmes ressortent rarement significatifs
    # (cf. retour utilisateur, session du 2026-08-09) — `use_biais=False` est
    # le réglage de PRODUCTION par défaut ; `use_biais=True` reste disponible
    # comme variante expérimentale (cf. BayesianNowcastAvecBiaisInstitut).
    if data.use_biais:
        biais = numpyro.sample("biais", dist.Normal(data.bias_prior_mean, data.bias_prior_sd))
    else:
        biais = jnp.zeros(len(data.bias_group_names))

    eye = jnp.eye(Km1)
    zero = jnp.zeros((Km1, Km1))
    dt_arr = data.dt_forward + MIN_DT_EPS
    gap0 = data.init_gap_days + MIN_DT_EPS   # 1er jour (fusionné dans init) -> 1er nœud restant

    if n_polls > 0:
        ys, Rs = [], []
        for i, plan in enumerate(data.plans):
            bg = jnp.asarray(data.bias_group_tested[i])
            debiased = jnp.asarray(data.raw_values[i]) - biais[bg]
            y_i, R_i = poll_observation_values(debiased, plan, Km1, data.n_echant[i],
                                               deff=data.deff,
                                               isotropic=data.isotropic_obs_noise)
            ys.append(y_i)
            Rs.append(R_i)
        emissions = jnp.stack(ys)                                        # (n_polls, Km1)
        H_level = jnp.stack([jnp.asarray(p.H) for p in data.plans])      # (n_polls, Km1, Km1)
        R_arr = jnp.stack(Rs)                                            # (n_polls, Km1, Km1)

    if not data.use_tendance:
        # Chemin ORIGINAL, K-1 dimensions — F=I, un seul tau (cf.
        # spec_ssm_nowcast.md §2.1), inchangé depuis avant l'ajout de la
        # tendance. `init_mean`/`init_cov` représentent déjà l'état AU 1er
        # jour de sondage (option A, `fuse_first_date_observations`) — un pas
        # de diffusion de `gap0` jours les amène à l'état juste avant le 1er
        # nœud restant (ou, si aucun nœud ne reste, directement à l'état
        # utilisé pour l'extrapolation vers `as_of`).
        pred_mean = jnp.asarray(data.init_mean)
        pred_cov = jnp.asarray(data.init_cov) + gap0 * Q_unit

        if n_polls == 0:
            last_mean, last_cov = pred_mean, pred_cov
        else:
            Q_arr = jnp.stack([dt_arr[t] * Q_unit for t in range(n_polls)])
            posterior = kalman_filter(pred_mean, pred_cov, eye, Q_arr, H_level, R_arr, emissions)
            numpyro.factor("chemin_marginal", posterior.marginal_loglik)
            last_mean, last_cov = posterior.filtered_means[-1], posterior.filtered_covariances[-1]

        extrap_cov = last_cov + (data.as_of_dt + MIN_DT_EPS) * Q_unit
        alpha_now = numpyro.sample("alpha_now", dist.MultivariateNormal(last_mean, extrap_cov))
        numpyro.deterministic("pi", ilr_decode(alpha_now, data.V))
        return

    # Chemin ÉTENDU, 2·(K-1) dimensions (niveau, tendance) — cf.
    # docs/spec_ssm_tendance.md §3 pour la dérivation complète.
    D = 2 * Km1
    tau_tendance = numpyro.sample("tau_tendance", dist.HalfNormal(cfg["tau_tendance_prior_scale"]))
    sigma_beta0 = numpyro.sample("sigma_beta0", dist.HalfNormal(cfg["beta0_prior_scale"]))

    def kinematic_step(dt):
        """F(dt)/Q(dt) du pas (niveau, tendance) — F est une identité
        cinématique (fonction connue de dt, RIEN d'appris ici) ; seules les
        échelles de Q (Q_unit -- tau_niveau ou tau_candidat selon le mode,
        cf. plus haut -- et tau_tendance) sont des paramètres NUTS."""
        F = jnp.block([[eye, dt * eye], [zero, eye]])
        Q = jnp.block([[dt * Q_unit, zero],
                       [zero, tau_tendance ** 2 * dt * eye]])
        return F, Q

    init_mean = jnp.concatenate([jnp.asarray(data.init_mean), jnp.zeros(Km1)])
    init_cov = jnp.block([[jnp.asarray(data.init_cov), zero],
                          [zero, sigma_beta0 ** 2 * eye]])
    F0, Q0 = kinematic_step(gap0)
    pred_mean, pred_cov = F0 @ init_mean, F0 @ init_cov @ F0.T + Q0

    if n_polls == 0:
        last_mean, last_cov = pred_mean, pred_cov
    else:
        # colonnes tendance nulles : un sondage observe le niveau, jamais la
        # tendance directement — même mécanisme de padding que les slots non
        # testés (plan.H), appliqué une seconde fois au bloc tendance.
        H_arr = jnp.concatenate([H_level, jnp.zeros_like(H_level)], axis=2)  # (n_polls, Km1, D)
        F_arr, Q_arr = zip(*[kinematic_step(dt_arr[t]) for t in range(n_polls)])
        F_arr, Q_arr = jnp.stack(F_arr), jnp.stack(Q_arr)

        posterior = kalman_filter(pred_mean, pred_cov, F_arr, Q_arr, H_arr, R_arr, emissions)
        numpyro.factor("chemin_marginal", posterior.marginal_loglik)
        last_mean, last_cov = posterior.filtered_means[-1], posterior.filtered_covariances[-1]

    # Extrapolation d'UN pas au-delà du dernier sondage, jusqu'à `as_of` (spec
    # §5) : même transition cinématique que le chemin, PAS de nouvelle
    # observation. La tendance courante fait maintenant avancer la moyenne
    # (elle était figée sous l'ancien modèle sans dérive) — c'est la
    # correction directe du décalage constant documenté en §1-2 de
    # docs/spec_ssm_tendance.md.
    F_extrap, Q_extrap = kinematic_step(data.as_of_dt + MIN_DT_EPS)
    extrap_mean = F_extrap @ last_mean
    extrap_cov = F_extrap @ last_cov @ F_extrap.T + Q_extrap

    z_now = numpyro.sample("z_now", dist.MultivariateNormal(extrap_mean, extrap_cov))
    alpha_now = numpyro.deterministic("alpha_now", z_now[:Km1])
    # tendance courante (points ILR/jour) exposée en diagnostic — utile pour
    # lire directement le "rythme de progression" que le modèle attribue à
    # chaque slot au moment `as_of`, sans avoir à le retro-calculer.
    numpyro.deterministic("beta_now", z_now[Km1:])
    numpyro.deterministic("pi", ilr_decode(alpha_now, data.V))


def _projeter_au_scrutin(pi, slots, candidate_blocs, jump_bank, horizon_days, seed=27):
    """Projection jusqu'au scrutin — délègue à `model/core/projection.py`.

    Ce modèle ne définit plus sa propre loi de projection : diffusion
    d'opinion (Ornstein-Uhlenbeck) puis écart sondages-urne δ, comme tous les
    modèles du dépôt. Les paramètres viennent de la banque COMMUNE
    (`model/core/opinion.py`) — aucun modèle ne lit dans le dossier d'un autre.

    `jump_bank` (ancien saut terminal sinh-arcsinh) n'est plus utilisé ; le
    paramètre reste accepté pour ne pas casser les appelants.
    """
    from model.core.opinion import load_law
    from model.core.projection import projeter_au_scrutin

    return projeter_au_scrutin(pi, horizon_days, load_law(), seed)


@register
class BayesianNowcast(ForecastModel):
    id = "bayesian-nowcast"
    label = "Nowcast bayésien (SSM)"
    description = ("Opinion modélisée comme état latent (State Space Model linéaire-"
                  "gaussien en coordonnées ILR), saut terminal sinh-arcsinh calibré "
                  "sur 2017/2022 — pas une quantité statique unique moyennant tous "
                  "les sondages sans pondération temporelle. Pas de débiaisage par "
                  "institut par défaut (cf. use_biais).")
    trains_on_history = True
    # Hors du site tant que §6.8 de spec_ssm.md (décalage RN vs sondages bruts)
    # reste ouvert : publier une courbe dont on sait qu'elle a un biais non
    # expliqué contredirait la ligne du projet (communiquer des probabilités
    # honnêtes). Le modèle reste pleinement exécutable — CLI, tests de contrat,
    # backfill — et rejoint la rubrique « suivi » en changeant cette ligne.
    surface = None
    use_house_effects = True
    # `biais[institut,bloc]` DÉSACTIVÉ par défaut : avec ~20-45 groupes et
    # parfois aussi peu que 5 sondages en début de campagne, le postérieur
    # joint (tau, biais) devient mal identifié (tau explore des valeurs
    # aberrantes, cf. docs/spec_ssm_implementation.md) — et les biais eux-
    # mêmes ressortent rarement significatifs. Réactivable
    # (`BayesianNowcastAvecBiaisInstitut` ci-dessous) pour une exploration,
    # pas pour la prévision affichée par défaut.
    use_biais = False
    # État de tendance (local linear trend, docs/spec_ssm_tendance.md) :
    # DÉSACTIVÉ par défaut — la vérification empirique a trouvé un problème
    # d'identifiabilité sur les dates creuses avant même d'avoir isolé l'effet
    # du prior initial (cf. `historical_prior_n` ci-dessous). Réactivable pour
    # une exploration une fois `historical_prior_n` réglé.
    use_tendance = False
    # Prior 2022 sur le premier nœud (`historical_prior_pi`/`historical_prior_cov`) :
    # N=300 par défaut historiquement, mais mesuré à 10-11 points du premier
    # sondage 2027 réel (RN, Horizons, Mélenchon) — poids largement excessif
    # vu l'écart (session du 2026-08-09). `historical_prior_n_if_tested`
    # (poids allégé pour un candidat déjà testé le 1er jour, cf.
    # `HISTORICAL_PRIOR_N_IF_TESTED`) fait le plus gros du travail ; `n`
    # reste pertinent pour les candidats jamais testés le 1er jour. Tous deux
    # overridables ici sans dupliquer `from_dataframe`, pour comparer
    # plusieurs valeurs.
    historical_prior_n = HISTORICAL_PRIOR_N
    historical_prior_n_if_tested = HISTORICAL_PRIOR_N_IF_TESTED
    # Encodage des sondages partiels (`latent.poll_plan`) : `False` (défaut,
    # modèle de base, session du 2026-08-10) = ILR local + bruit d'observation
    # isotrope simplifié (R=(deff/n)·I, sans corrélation entre candidats
    # testés) — écarte le mécanisme identifié comme cause d'une partie de la
    # dérive du RN (le RN, dénominateur implicite via sa grande part,
    # absorbait une correction destinée aux autres candidats par corrélation,
    # cf. docs/spec_ssm_encodage_partiel.md §5). `True` = ancien encodage ALR
    # local (référence = plus grande part testée) + covariance delta-method
    # complète du multinomial. Réactivable pour comparer (`BayesianNowcastCovarianceComplete`
    # ci-dessous) — les deux modes NE donnent PAS le même résultat.
    full_covariance = False
    # `tau` par candidat (session du 2026-08-10, cf. bayesian_nowcast_ssm_model
    # et docs/spec_ssm_encodage_partiel.md §9) : DÉSACTIVÉ par défaut — K
    # paramètres au lieu de 1, à vérifier (R-hat/ESS) avant adoption en
    # production. Corrige la cause identifiée du biais RN (~-1.8pt) : sa
    # variance gonflait librement faute de tests fréquents, la rendant "bon
    # marché" à déplacer par le gain de Kalman lors des surprises des AUTRES
    # candidats testés dans la même hypothèse — indépendant de la corrélation
    # (vérifié : décorréler la covariance jointe seule ne change rien).
    use_tau_candidat = False
    # Cohérence de roster : seules les hypothèses testant le noyau
    # `COHERENT_ROSTER_SLOTS` entrent dans le SSM. ACTIVÉ PAR DÉFAUT (session
    # du 2026-08-11) — ce n'est plus une variante expérimentale.
    #
    # Le modèle suppose une liste FIXE de candidats. Une hypothèse qui teste
    # une liste différente (Bardella au lieu de Le Pen, Attal au lieu de
    # Philippe) voit ses candidats non modélisés silencieusement retirés par
    # `aggregate_to_slots`, puis la masse restante RENORMALISÉE par
    # `poll_observation_values` — la part supprimée n'est donc pas traitée
    # comme manquante, elle est REDISTRIBUÉE entre les candidats restants.
    # Mesuré sur la campagne 2026 : 28,75 pts de masse jetée en moyenne
    # (43,04 pts quand RN est absente du roster testé, 9,41 quand elle est
    # présente) — cf. `biais_rn_investigation.md`, addendum du 2026-08-11.
    #
    # Deux effets mesurés, tous deux corrigés par le filtre :
    #   - biais : erreur signée moyenne sur RN -2,82 pt sans filtre contre
    #     -0,26 pt avec (mesure de la session du 2026-08-11) ;
    #   - IC : sans filtre, la couverture prédictive hors échantillon monte à
    #     IC90 99,3 % / IC50 78,0 % avec des bandes de 17,5 pt sur RN — le
    #     mélange de rosters gonfle la covariance d'ÉTAT bien au-delà de ce
    #     que `R` explique (§6.11.3 de la spec).
    #
    # `BayesianNowcastRosterMixte` ci-dessous rétablit l'ancien comportement
    # (aucun filtre) pour pouvoir mesurer l'écart.
    roster_filter_slots: tuple[str, ...] = COHERENT_ROSTER_SLOTS
    roster_filter_exact: bool = False
    # Bruit d'observation (session du 2026-08-11, travail sur la COUVERTURE) :
    # `R` est maintenant la covariance delta-method du multinomial
    # (`Var(p_i) = p_i(1-p_i)/n`, projetée en log-ratio), déflatée par `deff`.
    # `isotropic_obs_noise=True` rétablit la simplification `R=(deff/n)·I` de
    # la session précédente — gardée comme variante de comparaison
    # (`BayesianNowcastObsIsotrope`), pas comme défaut : elle revient à
    # supposer `p_i=1` pour tous les candidats, soit une variance sous-estimée
    # d'un facteur `1/p_i` (couverture IC90 mesurée à 35%).
    deff = DEFF_DEFAULT
    isotropic_obs_noise = False

    draws = 1000
    tune = 1000
    chains = 4
    target_accept = 0.95
    seed = 27

    def calibrate(self, history=None) -> None:
        if self.use_house_effects:
            bank = HouseEffectsCalibration().calibrate(load_calibration_frame())
            bank.save(BANK_PATH)
            TerminalJumpCalibration().calibrate(bank).save(JUMP_BANK_PATH)

    def load_artifacts(self) -> None:
        self.bank = Bank.load(BANK_PATH) if self.use_house_effects else None
        self.jump_bank = Bank.load(JUMP_BANK_PATH) if self.use_house_effects else None

    def nowcast(self, raw_polls, as_of) -> Nowcast:
        data = NowcastData.from_dataframe(raw_polls, as_of, self.bank, use_biais=self.use_biais,
                                          use_tendance=self.use_tendance,
                                          historical_prior_n=self.historical_prior_n,
                                          historical_prior_n_if_tested=self.historical_prior_n_if_tested,
                                          full_covariance=self.full_covariance,
                                          use_tau_candidat=self.use_tau_candidat,
                                          roster_filter_slots=self.roster_filter_slots,
                                          roster_filter_exact=self.roster_filter_exact,
                                          deff=self.deff,
                                          isotropic_obs_noise=self.isotropic_obs_noise)
        samples, extra = run_numpyro_mcmc(
            bayesian_nowcast_ssm_model, {"data": data},
            draws=self.draws, tune=self.tune, chains=self.chains,
            seed=self.seed, target_accept=self.target_accept,
            extra_fields=("diverging",))
        pi = np.asarray(samples["pi"]).reshape(-1, data.K)   # (chains*draws, K)
        draws = xr.DataArray(pi, dims=("draw", "candidat"),
                             coords={"candidat": data.slot_names})
        return Nowcast(draws=draws, diagnostics=self._diagnostics(samples, extra, data))

    def _diagnostics(self, samples: dict, extra: dict, data: "NowcastData") -> dict:
        """« le plus de diagnostic = le mieux » (retour utilisateur, session
        du 2026-08-09) — rien n'était exposé avant (ni `tau`, ni R-hat/ESS,
        ni divergences) : impossible de vérifier a posteriori le mécanisme du
        décalage (docs/spec_ssm_tendance.md §2) ou la santé de l'inférence
        sans relancer un script ad hoc à chaque fois. Tout ce qui est
        raisonnablement bon marché à calculer ici est inclus."""
        from numpyro.diagnostics import summary as numpyro_summary

        diag = numpyro_summary(samples, prob=0.9)

        def hp(name: str) -> dict | None:
            d = diag.get(name)
            if d is None:
                return None
            return {"mean": round(float(np.mean(d["mean"])), 4),
                   "sd": round(float(np.mean(d["std"])), 4),
                   "r_hat": round(float(np.max(d["r_hat"])), 4)}

        hyperparams = {s: hp(s) for s in
                      ("tau_niveau", "tau_candidat_moyen", "tau_tendance", "sigma_beta0", "biais")
                      if s in diag}
        rhat_max = max(float(np.max(d["r_hat"])) for d in diag.values())
        ess_min = min(float(np.min(d["n_eff"])) for d in diag.values())
        n_diverg = int(np.sum(extra["diverging"])) if "diverging" in extra else None

        # `tau` par candidat (use_tau_candidat) : détail par slot en plus de
        # la moyenne ci-dessus — « le plus de diagnostic = le mieux ».
        tau_par_candidat = None
        if "tau_candidat" in diag:
            d = diag["tau_candidat"]
            tau_par_candidat = {slot: {"mean": round(float(m), 4), "r_hat": round(float(r), 4)}
                               for slot, m, r in zip(data.slot_names, d["mean"], d["r_hat"])}

        # tendance courante convertie en points de %/mois par slot — lecture
        # directe et interprétable de `beta_now`, dont les unités brutes
        # (ILR/jour) ne le sont pas. Différence finie (pas une conversion
        # linéaire de `beta`) car pi = softmax(V·alpha) est non-linéaire.
        # Absent si `use_tendance=False` (chemin de code sans état de
        # tendance, cf. `bayesian_nowcast_ssm_model`) — pas d'erreur, juste
        # rien à convertir.
        momentum = None
        if "beta_now" in samples:
            alpha_now = np.asarray(samples["alpha_now"]).reshape(-1, data.K - 1)
            beta_now = np.asarray(samples["beta_now"]).reshape(-1, data.K - 1)
            V = jnp.asarray(data.V)
            pi_now = np.asarray(jax.nn.softmax(jnp.asarray(alpha_now) @ V.T, axis=-1))
            pi_plus1m = np.asarray(jax.nn.softmax(
                (jnp.asarray(alpha_now) + jnp.asarray(beta_now) * 30.0) @ V.T, axis=-1))
            momentum = {slot: round(float(m), 3) for slot, m in
                       zip(data.slot_names, ((pi_plus1m - pi_now) * 100.0).mean(axis=0))}

        return {
            "hyperparametres": hyperparams,
            "rhat_max": round(rhat_max, 4),
            "ess_min": round(ess_min, 1),
            "n_divergences": n_diverg,
            "roster_filter_slots": list(data.roster_filter_slots),
            "roster_filter_exact": data.roster_filter_exact,
            "deff": data.deff,
            "isotropic_obs_noise": data.isotropic_obs_noise,
            "tendance_pts_pourcent_par_mois": momentum,
            "tau_par_candidat": tau_par_candidat,
        }

    def forecast(self, nc: Nowcast, horizon_days: int) -> dict:
        # choix explicite de CE modèle (saut terminal sinh-arcsinh en espace
        # CLR par candidat) — pas hérité silencieusement d'une classe de base.
        slots = nc.draws.candidat.values.tolist()
        pi = _projeter_au_scrutin(nc.draws.values, slots, self._candidate_blocs(),
                                  self.jump_bank, horizon_days)
        return forecast_from_draws(pi, slots, horizon_days)

    def _candidate_blocs(self) -> dict[str, str]:
        return {slot: bloc for slot, (bloc, _names) in SLOTS.items()}


@register
class BayesianNowcastNoHouseEffects(BayesianNowcast):
    """Même modèle que bayesian-nowcast (biais institut déjà désactivé par
    défaut sur les deux) : ne diffère que par les priors calibrés sur
    2017/2022 (tau, saut terminal) — banques ignorées ici, priors faibles.
    Comparaison pour voir si la calibration historique change la prévision."""
    id = "bayesian-nowcast-no-house-effects"
    label = "Nowcast bayésien (SSM, priors faibles)"
    description = ("Même modèle que bayesian-nowcast, sans calibration historique "
                  "(priors faibles sur tau et le saut terminal) — sert de comparaison.")
    trains_on_history = False
    use_house_effects = False
    surface = None   # comparaison interne, hors du site (cf. ForecastModel.surface)


@register
class BayesianNowcastAvecBiaisInstitut(BayesianNowcast):
    """Variante EXPÉRIMENTALE : réactive le débiaisage par institut
    (`biais[institut,bloc]`, désactivé par défaut sur bayesian-nowcast — cf.
    sa docstring `use_biais`). Utile pour explorer si un institut a un biais
    mesurable, PAS pour la prévision affichée par défaut : avec peu de
    sondages disponibles (ex. début de campagne), le postérieur joint
    (tau, biais) est mal identifié."""
    id = "bayesian-nowcast-avec-biais-institut"
    label = "Nowcast bayésien (SSM, avec biais institut — expérimental)"
    description = ("Même modèle que bayesian-nowcast, biais institut×bloc réactivé "
                  "— exploratoire, pas la prévision par défaut (cf. use_biais).")
    trains_on_history = True
    use_house_effects = True
    use_biais = True
    surface = None   # comparaison interne, hors du site (cf. ForecastModel.surface)


@register
class BayesianNowcastCovarianceComplete(BayesianNowcast):
    """Variante de COMPARAISON : encodage ALR local (référence = plus grande
    part testée) au lieu de l'ILR local, à covariance delta-method identique.

    Sert de test de NON-RÉGRESSION du théorème d'invariance (spec §5.2) :
    depuis que `R` est la covariance delta-method dans les DEUX modes, ces
    deux systèmes de coordonnées doivent donner le MÊME postérieur sur
    `alpha` — tout écart au-delà du bruit numérique signale un bug
    d'encodage. (Avant la session du 2026-08-11, ce flag basculait AUSSI la
    forme de `R` : les deux modes différaient alors réellement.)"""
    id = "bayesian-nowcast-covariance-complete"
    label = "Nowcast bayésien (SSM, coordonnées ALR locales — comparaison)"
    description = ("Même modèle que bayesian-nowcast, coordonnées ALR locales au lieu "
                  "d'ILR local — postérieur attendu identique (invariance, spec §5.2).")
    trains_on_history = True
    use_house_effects = True
    full_covariance = True
    surface = None   # comparaison interne, hors du site (cf. ForecastModel.surface)


@register
class BayesianNowcastObsIsotrope(BayesianNowcast):
    """Variante de COMPARAISON : rétablit le bruit d'observation isotrope
    simplifié `R = (deff/n)·I` (défaut de la session du 2026-08-10, remplacé
    le 2026-08-11 par la covariance delta-method du multinomial).

    Conservée pour mesurer l'écart de COUVERTURE entre les deux, pas comme
    candidat sérieux : `R=(deff/n)·I` revient à supposer `p_i=1` pour tous
    les candidats, donc à sous-estimer la variance d'observation d'un facteur
    `1/p_i` (x2,9 à 34 %, x100 à 1 %)."""
    id = "bayesian-nowcast-obs-isotrope"
    label = "Nowcast bayésien (SSM, bruit d'observation isotrope — comparaison)"
    description = ("Même modèle que bayesian-nowcast, bruit d'observation isotrope "
                  "R=(deff/n)·I au lieu de la covariance multinomiale — comparaison "
                  "de couverture uniquement.")
    trains_on_history = True
    use_house_effects = True
    isotropic_obs_noise = True
    surface = None   # comparaison interne, hors du site (cf. ForecastModel.surface)


@register
class BayesianNowcastTauCandidat(BayesianNowcast):
    """Variante de COMPARAISON : `tau` par candidat au lieu d'un seul `tau`
    isotrope partagé (désactivé par défaut sur bayesian-nowcast — cf. sa
    docstring `use_tau_candidat`). Corrige la cause identifiée du biais
    systématique du RN (~-1,8pt vs ~-0,5pt de lag général sur les autres
    candidats, session du 2026-08-10) : sa variance gonflait librement faute
    de tests fréquents, la rendant "bon marché" à déplacer par le gain de
    Kalman lors des surprises des AUTRES candidats testés dans la même
    hypothèse. PAS la prévision par défaut tant que R-hat/ESS ne sont pas
    vérifiés sains avec K=12 paramètres au lieu de 1 — cf.
    docs/spec_ssm_encodage_partiel.md §9."""
    id = "bayesian-nowcast-tau-candidat"
    label = "Nowcast bayésien (SSM, tau par candidat — comparaison)"
    description = ("Même modèle que bayesian-nowcast, tau par candidat au lieu d'un tau "
                  "unique isotrope — corrige le biais RN identifié en session, cf. "
                  "docs/spec_ssm_encodage_partiel.md.")
    trains_on_history = True
    use_house_effects = True
    use_tau_candidat = True
    surface = None   # comparaison interne, hors du site (cf. ForecastModel.surface)


@register
class BayesianNowcastRosterMixte(BayesianNowcast):
    """Variante de COMPARAISON : AUCUN filtre de roster — toutes les
    hypothèses entrent dans le SSM, quel que soit le champ testé.

    C'était le comportement par défaut jusqu'au 2026-08-11 ; conservé pour
    mesurer le coût de l'incohérence de roster, pas comme candidat (cf.
    `BayesianNowcast.roster_filter_slots` pour les chiffres : biais RN
    -2,82 pt contre -0,26 pt, et couverture prédictive IC90 à 99,3 % avec des
    bandes de 17,5 pt sur RN)."""
    id = "bayesian-nowcast-roster-mixte"
    label = "Nowcast bayésien (SSM, sans filtre de roster — comparaison)"
    description = ("Même modèle que bayesian-nowcast, mais sans filtre de cohérence "
                   "de roster — toutes les hypothèses sont agrégées dans le même état "
                   "latent, y compris celles testant un champ différent.")
    trains_on_history = True
    use_house_effects = True
    roster_filter_slots: tuple[str, ...] = ()
    surface = None   # comparaison interne, hors du site (cf. ForecastModel.surface)


@register
class BayesianNowcastNoHouseEffectsRoster12(BayesianNowcastNoHouseEffects):
    """Variante stricte : aucun house effect et uniquement les hypothèses
    testant EXACTEMENT les 12 slots du modèle."""
    id = "bayesian-nowcast-no-house-effects-roster-12"
    label = "Nowcast bayésien (SSM, roster exact 12, sans house effects)"
    description = ("Même modèle que bayesian-nowcast-no-house-effects, mais en ne "
                   "gardant que les hypothèses testant exactement les 12 slots du modèle.")
    roster_filter_slots = EXACT_ROSTER_12_SLOTS
    roster_filter_exact = True
    surface = None   # comparaison interne, hors du site (cf. ForecastModel.surface)
