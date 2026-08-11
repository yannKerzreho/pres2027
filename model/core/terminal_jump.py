"""Saut terminal : loi du mouvement d'opinion entre le nowcast et le scrutin.

Machinerie **partagée**, pas propre à un modèle : elle répond à une question
que tout modèle de suivi doit trancher — « de combien l'opinion peut-elle
encore bouger d'ici au 1er tour ? » — indépendamment de la façon dont il
estime l'opinion *courante*. `model/core/simulate.py` consomme déjà la Bank
produite ici (`forecast_from_draws(..., jump_bank=...)`) ; un modèle branche
son nowcast dessus sans rien réimplémenter.

Deux étages :

1. `movement_pool` construit le **jeu de données empirique** : les mouvements
   RÉELS observés en 2017 et 2022, mesurés en log-ratio centré (CLR) entre un
   sondage débiaisé et le résultat du scrutin. Mesurés, pas supposés.
2. `terminal_jump_model` / `TerminalJumpCalibration` fittent une loi
   **sinh-arcsinh** sur ce pool. Le choix d'une famille asymétrique à queues
   ajustables (plutôt qu'une gaussienne) vient du pool lui-même : excès de
   kurtosis et asymétrie marqués (surges d'outsiders), qu'une gaussienne
   sous-estimerait — ce qui produirait des probabilités de qualification
   surconfiantes.

Normalisation par l'horizon : chaque mouvement historique est mesuré à un
horizon différent (fenêtres 45-90 / 90-180 / 180-400 jours). Un mouvement
observé à J-400 et un mouvement à J-50 ne sont pas la même quantité — la
dérive continue jusqu'au scrutin. On les ramène donc tous à un horizon de
référence commun `horizon_ref` (moyenne du pool) en `sqrt(horizon)` avant de
fitter, et `forecast_from_draws` rescale ensuite le TIRAGE vers l'horizon réel
de la prévision, dans le même sens. `horizon_ref` est stocké dans la Bank pour
que ce rescaling reste cohérent sans dupliquer la valeur en dur ailleurs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

from model.core.bank import Bank
from model.core.inference import run_numpyro_mcmc
from model.core.utils import SinhArcsinh, clr
from pipeline.historical import load_calibration_frame, load_results


def saturation(horizon_days, tau: float):
    """Part de la dérive totale déjà accomplie à `horizon_days` du scrutin :
    `1 − exp(−h/τ)`. Nulle à J-0, tend vers 1 aux grands horizons.

    C'est la variance (normalisée) d'un processus d'Ornstein-Uhlenbeck, donc
    la traduction directe de la mean-reversion : l'opinion ne diffuse pas
    indéfiniment, elle oscille autour d'un niveau. `τ` est le temps
    caractéristique — au-delà de quelques `τ` avant le scrutin, reculer la date
    d'un sondage n'ajoute plus d'incertitude sur le résultat.

    Remplace l'ancienne loi en `√h`, contredite par le pool : la dispersion
    observée des mouvements est PLATE de 63 à 267 jours (0,508 / 0,610 / 0,590),
    là où `√h` prédisait 0,286 / 0,402 / 0,590. `√h` sous-estimait donc les
    horizons courts au fit — et surestimait d'autant la prévision une fois
    renormalisée.
    """
    h = np.clip(np.asarray(horizon_days, dtype=float), 0.0, None)
    return 1.0 - np.exp(-h / tau)


def jump_moves(jump_bank: Bank, blocs: list[str], h_from, h_to, size: tuple[int, int],
               seed: int = 27, with_loc: bool = True) -> np.ndarray:
    """Tirages du mouvement d'opinion entre deux dates, en espace log-ratio,
    forme `size = (S, K)`.

    `h_from`, `h_to` : horizons des deux extrémités, en JOURS AVANT LE SCRUTIN
    (donc `h_from > h_to`). Une seule implémentation pour les deux usages :

      - projeter `as_of` jusqu'au scrutin : `h_from=horizon`, `h_to=0` ;
      - projeter un sondage jusqu'à `as_of` : `h_from=horizon+âge`, `h_to=horizon`.

    Variance du segment = `scale²·[sat(h_from) − sat(h_to)]`, soit
    `scale²·[e^(−h_to/τ) − e^(−h_from/τ)]`. Toujours positive, et **exactement
    additive** : enchaîner sondage→`as_of`→scrutin redonne la variance de
    sondage→scrutin, sans double comptage ni trou. C'est ce qui autorise à
    appliquer le saut deux fois de façon indépendante.

    Les deux extrémités acceptent des tableaux diffusables sur `size`, ce qui
    permet un horizon PROPRE À CHAQUE TIRAGE — chaque tirage du mélange
    sélectionne un sondage d'âge différent (cf. `linear_pooling`).

    `with_loc=False` ne garde que la dispersion. Sans grande conséquence ici :
    `loc` est global (cf. `terminal_jump_model`), donc identique pour tous les
    slots, donc annulé exactement par la renormalisation softmax sur le
    simplexe. L'option reste pour un modèle qui n'aurait pas cette invariance.
    """
    tau, _ = jump_bank.jump_tau.item()
    skew, _ = jump_bank.jump_skew.item()
    tail, _ = jump_bank.jump_tail.item()
    scale = np.array([jump_bank.jump_scale.at(bloc=b)[0] for b in blocs])
    loc = (np.array([jump_bank.jump_loc.at(bloc=b)[0] for b in blocs])
           if with_loc else np.zeros(len(blocs)))

    d = SinhArcsinh(loc=loc, scale=scale, skewness=skew, tailweight=tail)
    unit = np.asarray(d.sample(jax.random.PRNGKey(seed), sample_shape=(size[0],)))
    frac = np.clip(saturation(h_from, tau) - saturation(h_to, tau), 0.0, None)
    return unit * np.sqrt(frac)


def institut_bias_prior(bank: Bank | None, institut: str, bloc: str) -> tuple[float, float]:
    """(mean, sd) du biais institut×bloc calibré sur l'historique. Repli sur la
    moyenne de population du bloc si l'institut est inconnu (shrinkage) ; prior
    faible et centré si `bank is None` — un modèle qui ne modélise pas de house
    effects passe `None` et travaille alors sur des sondages non débiaisés,
    ce qui est le comportement voulu, pas une dégradation silencieuse."""
    if bank is None:
        return 0.0, 5.0
    try:
        return bank.institut_bias.at(institut=institut, bloc=bloc)
    except KeyError:
        return bank.bloc_bias_mean.at(bloc=bloc)


def movement_pool(bank: Bank | None, hmin: float = 45.0) -> pd.DataFrame:
    """Mouvements RÉELS d'opinion 2017/2022, en espace log-ratio (CLR), par
    candidat × élection × fenêtre d'horizon — le jeu de données du fit
    paramétrique du saut terminal.

    Débiaise chaque sondage historique du biais calibré (si une `bank` est
    fournie), moyenne les sondages proches (même candidat, même fenêtre
    d'horizon) pour annuler le bruit d'échantillonnage, puis calcule le
    mouvement `clr(résultat) − clr(sondage_débiaisé)`.

    Bloc ET horizon (moyen de la fenêtre) sont conservés par mouvement : `loc`
    est fitté par bloc, et `TerminalJumpCalibration` normalise par horizon
    avant de fitter.
    """
    df = load_calibration_frame()
    df["bias"] = [institut_bias_prior(bank, r.institut, r.bloc)[0] for r in df.itertuples()]
    df["deb"] = df["intention"] - df["bias"]
    res = load_results()
    res = res[res["tour"] == "Premier tour"]

    df["hbin"] = pd.cut(df["horizon"], [0, 45, 90, 180, 400])
    moves: list[float] = []
    blocs: list[str] = []
    horizons: list[float] = []
    for elec in df["election"].unique():
        de = df[df["election"] == elec]
        cand = sorted(de["candidat"].unique())
        bloc_of = de.drop_duplicates("candidat").set_index("candidat")["bloc"]
        rmap = dict(zip(res.loc[res.election == elec, "candidat"],
                        res.loc[res.election == elec, "resultat"]))
        r = np.array([rmap.get(c, np.nan) for c in cand])
        for _, grp in de.groupby("hbin", observed=True):
            h = grp["horizon"].mean()
            if h < hmin:
                continue
            smap = grp.groupby("candidat")["deb"].mean()
            s = np.array([smap.get(c, np.nan) for c in cand])
            mask = ~np.isnan(s) & ~np.isnan(r) & (s > 0)
            if mask.sum() < 4:
                continue
            cand_kept = [c for c, keep in zip(cand, mask) if keep]
            moves.extend((clr(r[mask]) - clr(s[mask])).tolist())
            blocs.extend(bloc_of.loc[cand_kept].tolist())
            horizons.extend([float(h)] * int(mask.sum()))
    return pd.DataFrame({"bloc": blocs, "move": moves, "horizon": horizons})


# Hyperparamètres des priors du fit sinh-arcsinh (saut terminal).
DEFAULT_JUMP_PRIORS_CFG = {
    "loc_mean_sd": 5.0,
    "log_scale_mu": float(np.log(3.0)), "log_scale_sd": 1.0,
    "skew_sd": 1.0, "tail_log_sd": 0.5,
    # τ (jours) : prior large et peu informatif. Médiane 60 j, un facteur ~2,2
    # d'incertitude à 1 sd — le pool ne contient aucun mouvement en deçà de
    # 59 jours, donc τ n'est identifié que par la COURBURE entre 59 et 290 j.
    "tau_log_mu": float(np.log(60.0)), "tau_log_sd": 0.8,
}


def terminal_jump_model(bloc_idx: jnp.ndarray, move: jnp.ndarray, horizon: jnp.ndarray,
                        n_bloc: int, priors_cfg: dict | None = None):
    """NumPyro pur : `move_i ~ SinhArcsinh(loc·g_i, scale·g_i, skew, tail)`, où
    `g_i = √(1 − exp(−h_i/τ))` est la part de dérive déjà accomplie à l'horizon
    du mouvement `i`.

    Les mouvements entrent **BRUTS**, avec leur horizon : `τ` est ajusté
    conjointement au reste, il ne peut donc pas servir à pré-normaliser les
    données hors du modèle (ce que faisait l'ancienne version avec `√(href/h)`,
    au prix d'une hypothèse `√h` jamais testée — et fausse, cf. `saturation`).

    **TOUS les paramètres sont GLOBAUX** — `loc` compris (décidé le
    2026-08-11). Ce modèle échantillonne la *distribution des dérives de
    campagne*, il ne rejoue pas le passé bloc par bloc.

    Pourquoi `loc` ne peut pas être par bloc. Les mouvements sont mesurés en
    log-ratio CENTRÉ : à chaque (élection × fenêtre d'horizon),
    `clr(résultat) − clr(sondage)` somme à zéro entre candidats, par
    construction. La moyenne du pool vaut donc exactement 0,0000 — et un `loc`
    par bloc ne *découvre* rien, il ne fait que **répartir ce zéro** entre
    familles politiques. Avec 3 à 18 observations par bloc issues de DEUX
    campagnes, cette répartition ne mesure pas une loi de la vie politique :
    elle enregistre qui s'est présenté en 2017 et 2022. Le fit par bloc
    donnait `droite = −0,50`, soit Fillon puis Pécresse, ce qui revenait à
    imposer à tout candidat LR de 2027 un effondrement de moitié de sa part
    relative quels que soient ses sondages — une prédiction déguisée en
    incertitude.

    Ce qu'on garde du pool : sa **forme**. `scale` (dispersion), `skew` et
    `tail` décrivent l'amplitude et l'asymétrie des dérives réellement
    observées (queues épaisses, surges d'outsiders) et s'appliquent
    identiquement à chaque candidat — ce qui est arrivé à la droite en 2017 et
    2022 peut arriver à n'importe qui.

    Conséquence utile : avec `loc ≈ 0`, la dérive devient une diffusion pure,
    donc enchaîner deux jambes indépendantes (sondage → `as_of`, puis `as_of` →
    scrutin) est EXACT — les variances s'additionnent en temps, et il n'y a
    plus de composante systématique comptée deux fois (cf. `jump_moves`).

    `loc` reste un paramètre échantillonné (et non figé à 0) pour que les
    données puissent contredire ce raisonnement : si le postérieur s'écartait
    nettement de 0, ce serait le signal d'un biais global sondages → résultat
    qu'il faudrait expliquer. Il est ensuite broadcasté sur la dimension
    `bloc`, comme `scale`, afin de garder inchangés le schéma de la Bank et les
    lookups `.at(bloc=...)` de `model/core/simulate.py`.
    """
    cfg = {**DEFAULT_JUMP_PRIORS_CFG, **(priors_cfg or {})}

    tau = numpyro.sample("jump_tau", dist.LogNormal(cfg["tau_log_mu"], cfg["tau_log_sd"]))
    g = jnp.sqrt(1.0 - jnp.exp(-horizon / tau))

    loc_global = numpyro.sample("jump_loc_global", dist.Normal(0.0, cfg["loc_mean_sd"]))
    loc = numpyro.deterministic("jump_loc", loc_global * jnp.ones(n_bloc))

    log_scale = numpyro.sample("jump_log_scale",
                               dist.Normal(cfg["log_scale_mu"], cfg["log_scale_sd"]))
    # broadcast en (n_bloc,) : garde le schéma Bank (dims=["bloc"]) et les
    # lookups `.at(bloc=...)` de model/core/simulate.py inchangés, avec la
    # MÊME valeur pour chaque bloc. `scale` est ici la dispersion ASYMPTOTIQUE
    # (celle d'un sondage infiniment loin du scrutin), pas celle à un horizon
    # de référence : c'est `g_i` qui la ramène à l'horizon de chaque mouvement.
    scale = numpyro.deterministic("jump_scale", jnp.exp(log_scale) * jnp.ones(n_bloc))

    skew = numpyro.sample("jump_skew", dist.Normal(0.0, cfg["skew_sd"]))
    tail = numpyro.sample("jump_tail", dist.LogNormal(0.0, cfg["tail_log_sd"]))

    d = SinhArcsinh(loc=loc[bloc_idx] * g, scale=scale[bloc_idx] * g,
                    skewness=skew, tailweight=tail)
    numpyro.sample("move_obs", d, obs=move)


class TerminalJumpCalibration:
    """Fit paramétrique du saut terminal (nowcast → jour du scrutin).

    Pas d'ABC : une classe normale qui compose `run_numpyro_mcmc` + `Bank`.
    Un modèle l'appelle depuis son `calibrate()` et sauvegarde la Bank où il
    veut ; `bank=None` signifie « ne pas débiaiser les sondages historiques
    des house effects » (cohérent pour un modèle qui n'en estime pas).
    """
    draws = 1000
    tune = 1000
    chains = 4
    target_accept = 0.9
    seed = 27
    hmin = 45.0
    priors_cfg: dict | None = None

    def calibrate(self, bank: Bank | None) -> Bank:
        # Les mouvements entrent BRUTS, avec leur horizon : `τ` étant ajusté
        # dans le modèle, il ne peut pas servir à les pré-normaliser ici (ce
        # que faisait l'ancienne version avec `√(href/h)`, en supposant une loi
        # que les données démentent — cf. `saturation`).
        moves = movement_pool(bank, hmin=self.hmin)
        bloc_codes, bloc_names = pd.factorize(moves["bloc"], sort=True)
        data = {"bloc_idx": jnp.array(bloc_codes),
                "move": jnp.array(moves["move"].to_numpy()),
                "horizon": jnp.array(moves["horizon"].to_numpy()),
                "n_bloc": len(bloc_names), "priors_cfg": self.priors_cfg}
        samples, _ = run_numpyro_mcmc(
            terminal_jump_model, data, draws=self.draws, tune=self.tune, chains=self.chains,
            seed=self.seed, target_accept=self.target_accept)
        return Bank(samples, dims={"jump_loc": ["bloc"], "jump_scale": ["bloc"]},
                    coords={"bloc": list(bloc_names)})
