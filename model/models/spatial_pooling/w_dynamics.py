"""Dynamique de `w` — inversion locale exacte + postérieur Ornstein-Uhlenbeck en
forme close (cf. spec_spatial_pooling.md §11).

Ce module REMPLACE la lecture de `w_now` issue de la vraisemblance tempérée
(§3.1). Il ne touche PAS au fit NUTS : la géométrie (`mu`, `sigma`) reste
estimée exactement comme aujourd'hui, c'est la seule chose que ce fit sait bien
faire. Seul `w` est ré-estimé, sans MCMC, par une unique résolution linéaire —
même parti pris que `model/core/gp_math.py` pour `gp_pooling`.

Pourquoi il fallait changer quelque chose
-----------------------------------------
La vraisemblance tempérée n'a **aucun plancher de variance temporel**. Sa
variance de Laplace vaut `(prior^-1 + Σ_p κ_p·I_p)^-1` : elle tend vers 0 en
`1/n` dès qu'on accumule des sondages, **même s'ils sont tous vieux**. Or ce
qu'un Ornstein-Uhlenbeck impose est un plancher `σ_w²(1 − e^{-2g/τ_w})` qui ne
dépend QUE de l'écart `g` au dernier sondage et qu'aucune accumulation ne peut
franchir. Aucun choix de `κ_p` ne peut reproduire ce plancher (mesuré :
à 30 jours et 20 sondages, la version tempérée annonce 0,375 pt d'écart-type
sur `pi` là où le seul plancher OU en vaut 1,107 — et elle descend encore à
0,017 pt à n=10 000). C'est la cause de la sous-couverture résiduelle de §6.6.

Le mécanisme, en trois temps
----------------------------
1. **Inversion locale EXACTE** (`invert_node`). Pour un nœud `p` de champ testé
   `S_p`, on résout `pi(w) = Ỹ_p`. Ce n'est pas une linéarisation : c'est la
   solution exacte, et elle est bien posée. En effet `pi = ∇Φ` avec
   `Φ(w) = Σ_b W_b·logsumexp_{j∈S_p}(w_j + D_{jb})`, convexe ; son hessien
   `J = Σ_b W_b (diag(P_b) − P_b P_bᵀ)` est semi-défini positif de noyau
   EXACTEMENT `span(1)` (la jauge de §3.2). Donc `Φ` est strictement convexe
   sur `{Σw = 0}` et l'inversion y est un difféomorphisme (dualité de
   Legendre) : Newton amorti converge globalement, en 6 itérations médianes.

   C'est ce qui distingue cette approche de l'EKF parqué (§3.ter) : l'EKF
   linéarise `h` autour de la moyenne PRÉDITE, à chaque nœud, et re-différentie
   le tout sous NUTS (44 min sans converger sur le roster réel). Ici on
   linéarise autour de **la donnée elle-même**, une fois, hors de tout MCMC.

2. **Méthode delta** : `ŵ_p ≈ w(t_p) + bruit`, de covariance
   `C_p = J_p⁺ Σ_p J_p⁺` avec `Σ_p` la covariance d'observation du sondage
   (multinomiale + variance d'excès de §6.6). On a donc une pseudo-observation
   LINÉAIRE-GAUSSIENNE de `w(t_p)` — exactement la situation où `gp_pooling`
   sait travailler en forme close.

3. **Krigeage universel à noyau OU** (`w_ou_posterior`). `w_i(t) = m_i + u_i(t)`,
   `u ~ OU(0, σ_w², τ_w)` indépendant par candidat, `m` de prior PLAT et
   marginalisé — même choix que `gp_math.gp_posterior`, et pour la même raison :
   c'est ce qui fait revenir la prévision vers le niveau MOYEN estimé du
   candidat (et non vers 0, qui n'a aucun sens ici) quand l'écart grandit, avec
   une variance qui sature proprement.

Jauge et masques
----------------
`w` n'a pas de niveau absolu (§3.2) et chaque nœud teste un champ différent :
la pseudo-observation d'un nœud n'informe donc que les CONTRASTES au sein de
`S_p`. C'est encodé exactement, sans bricolage, par le projecteur
`C_{S_p} = diag(m_p) − m_p m_pᵀ/K_p` (orthogonal, rang `K_p − 1`) pris comme
matrice de design du nœud. Les directions non identifiées (candidats non testés,
et le mode commun de `S_p`) reçoivent `PADDING_VAR` — même procédé que
`PADDING_VAR` dans `bayesian_nowcast/latent.py` : neutraliser sans exclure.

Coût : une factorisation de Cholesky `(P·N)×(P·N)` par tirage de géométrie.
Aucun MCMC, aucun `scan`, aucun R-hat à surveiller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from model.models.spatial_pooling.model import V, W   # grille spatiale partagée (spec §1)

BANK_W_OU_PATH = Path(__file__).parent / "bank_w_ou.json"

# Neutralise les directions non identifiées (candidats non testés à un nœud, et
# mode commun du champ testé) sans les retirer d'un vecteur à taille fixe.
PADDING_VAR = 1e6
JITTER = 1e-10

# Repli si `bank_w_ou.json` n'a jamais été calibrée (cf. `load_w_law`).
DEFAULT_SIGMA_W2 = 0.2

_V = np.asarray(V, dtype=float)
_W = np.asarray(W, dtype=float)


# --- 1. Inversion locale exacte -------------------------------------------------
def distance_matrix(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """`D[i,b] = −(V_b − mu_i)²/(2σ_i²)` — la partie de l'attractivité qui ne
    dépend pas de `w` (spec §4). `mu`/`sigma` `(N,)` ou `(S,N)`."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return -((_V - mu[..., None]) ** 2) / (2 * sigma[..., None] ** 2)


def _probs(w: np.ndarray, Dsub: np.ndarray) -> np.ndarray:
    """softmax masqué colonne par colonne de la grille. `w` `(...,K)`,
    `Dsub` `(...,K,B)` -> `(...,K,B)`."""
    A = w[..., None] + Dsub
    A = A - A.max(axis=-2, keepdims=True)
    e = np.exp(A)
    return e / e.sum(axis=-2, keepdims=True)


def _pi(w: np.ndarray, Dsub: np.ndarray) -> np.ndarray:
    return _probs(w, Dsub) @ _W


def _jacobian(w: np.ndarray, Dsub: np.ndarray) -> np.ndarray:
    """`J = Σ_b W_b (diag(P_b) − P_b P_bᵀ)` — SDP, noyau exactement `span(1)`
    (vérifié contre `jax.jacfwd` à 1e-17 près). C'est aussi le hessien de `Φ`,
    d'où la convexité stricte sur `{Σw = 0}`."""
    P = _probs(w, Dsub)                                    # (...,K,B)
    PW = P * _W
    diag = PW.sum(axis=-1)                                 # (...,K)
    return np.einsum("...kb,...jb->...kj", PW, P) * -1.0 + _apply_diag(diag)


def _apply_diag(d: np.ndarray) -> np.ndarray:
    out = np.zeros(d.shape + (d.shape[-1],))
    idx = np.arange(d.shape[-1])
    out[..., idx, idx] = d
    return out


def _phi(w: np.ndarray, Dsub: np.ndarray) -> np.ndarray:
    A = w[..., None] + Dsub
    m = A.max(axis=-2)
    return np.einsum("b,...b->...", _W, m + np.log(np.exp(A - m[..., None, :]).sum(axis=-2)))


def _pinv_apply(J: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """`J⁺ · rhs` pour `rhs ⊥ 1`, sans SVD.

    `J` est SDP de noyau EXACTEMENT `span(1)` (T1), donc `J + 11ᵀ/K` vaut
    l'identité sur `span(1)` et `J` sur `1^⊥` : son inverse coïncide avec `J⁺`
    sur `1^⊥`, où vivent tous nos seconds membres (gradient projeté, covariance
    d'observation d'une composition). Un `solve` au lieu d'une pseudo-inverse
    par SVD — même résultat exact, nettement moins cher dans la boucle de
    Newton.
    """
    K = J.shape[-1]
    return np.linalg.solve(J + np.ones((K, K)) / K, rhs)


def invert_node(y: np.ndarray, Dsub: np.ndarray, maxit: int = 40, tol: float = 1e-10):
    """Résout `pi(w) = y` sur `{Σw = 0}`, en batch sur les tirages de géométrie.

    `y` `(K,)` (la composition observée, renormalisée sur le champ testé),
    `Dsub` `(S,K,B)` -> `w` `(S,K)`.

    Newton amorti (Armijo par tirage) sur `min_w Φ(w) − y·w`, strictement
    convexe sur le sous-espace somme-nulle : convergence globale garantie, pas
    de point de départ à choisir avec soin (`w = 0` suffit).
    """
    S, K, _ = Dsub.shape
    w = np.zeros((S, K))
    C = np.eye(K) - 1.0 / K
    for _ in range(maxit):
        g = (_pi(w, Dsub) - y) @ C                          # gradient projeté (S,K)
        if np.abs(g).max() < tol:
            break
        step = _pinv_apply(_jacobian(w, Dsub), g[..., None])[..., 0] @ C
        f0 = _phi(w, Dsub) - w @ y
        gs = np.einsum("sk,sk->s", g, step)
        t = np.ones(S)
        for _ in range(40):
            cand = w - t[:, None] * step
            bad = (_phi(cand, Dsub) - cand @ y) > f0 - 1e-4 * t * gs
            if not bad.any():
                break
            t = np.where(bad, t * 0.5, t)
        w = w - t[:, None] * step
    return w


def _centering(mask: np.ndarray) -> np.ndarray:
    """`C_S = diag(m) − m mᵀ/K` : projecteur orthogonal (C² = C) de rang `K−1`
    sur les contrastes du champ testé. Sert à la fois de matrice de design du
    nœud et de définition de sa jauge."""
    m = np.asarray(mask, dtype=float)
    k = m.sum()
    return np.diag(m) - np.outer(m, m) / max(k, 1.0)


@dataclass
class PseudoObservations:
    """Pseudo-observations linéaires-gaussiennes de `w`, une par nœud.

    `o` `(S,P,N)` : `ŵ_p` complété par des zéros hors du champ testé, de jauge
    `Σ_{i∈S_p} ŵ_i = 0`. `C` `(S,P,N,N)` : leur covariance (méthode delta),
    déjà additionnée de `PADDING_VAR` sur les directions non identifiées.
    `proj` `(P,N,N)` : le projecteur `C_{S_p}`, matrice de design du nœud.
    """
    o: np.ndarray
    C: np.ndarray
    proj: np.ndarray
    dates: np.ndarray
    instituts: list[str]


def build_pseudo_observations(mu_draws: np.ndarray, sigma_draws: np.ndarray, tested_mask: np.ndarray,
                              Y: np.ndarray, Np: np.ndarray, dates: np.ndarray,
                              instituts: list[str], excess_var: np.ndarray | None = None,
                              ) -> PseudoObservations:
    """Étapes 1 et 2 : inversion exacte par nœud puis propagation delta.

    `mu_draws`/`sigma_draws` `(S,N)` : tirages de géométrie du fit NUTS
    (inchangé). `tested_mask`/`Y` `(P,N)`, `Np`/`dates` `(P,)`, `excess_var`
    `(P,)` en fraction² (`model.excess_var_for_nodes`).

    `Σ_p` est bâtie sur les valeurs OBSERVÉES `Ỹ_p`, pas sur un `pi` prédit —
    même convention que `poll_observation_values` (bayesian_nowcast/latent.py) :
    la covariance du bruit ne doit pas dépendre de l'état qu'on estime.
    """
    S, N = mu_draws.shape
    P = tested_mask.shape[0]
    o = np.zeros((S, P, N))
    Cn = np.zeros((S, P, N, N))
    proj = np.zeros((P, N, N))

    Dfull = distance_matrix(mu_draws, sigma_draws)                    # (S,N,B)
    for p in range(P):
        idx = np.flatnonzero(tested_mask[p] > 0)
        K = len(idx)
        proj[p] = _centering(tested_mask[p])
        pad = np.eye(N) - proj[p]
        if K < 2:
            Cn[:, p] = PADDING_VAR * np.eye(N)
            continue

        y = np.asarray(Y[p, idx], dtype=float)
        y = np.clip(y, 1e-5, None)
        y = y / y.sum()                                               # renormalisé sur le champ testé

        Dsub = Dfull[:, idx, :]                                       # (S,K,B)
        w_hat = invert_node(y, Dsub)                                  # (S,K)
        J = _jacobian(w_hat, Dsub)                                    # (S,K,K)

        # Covariance d'observation de Ỹ_p : multinomiale + excès, toutes deux
        # ramenées sur le sous-espace somme-nulle (Ỹ_p somme à 1 par
        # construction, son bruit n'a pas de composante commune) -- c'est ce
        # qui autorise `_pinv_apply` des deux côtés.
        Ck = np.eye(K) - 1.0 / K
        Sig = (np.diag(y) - np.outer(y, y)) / max(float(Np[p]), 1.0)
        if excess_var is not None:
            Sig = Sig + float(excess_var[p]) * Ck
        Sig_b = np.broadcast_to(Sig, J.shape)                         # `solve` veut b.ndim == a.ndim
        Cd = _pinv_apply(J, np.swapaxes(_pinv_apply(J, Sig_b), -1, -2))   # J⁺ Σ J⁺ (symétrique)

        o[:, p][:, idx] = w_hat
        sub = np.ix_(np.arange(S), idx, idx)
        Cn[:, p][sub] = Cd
        Cn[:, p] += PADDING_VAR * pad

    return PseudoObservations(o=o, C=Cn, proj=proj, dates=np.asarray(dates, dtype=float),
                              instituts=list(instituts))


# --- 3. Krigeage universel à noyau OU -------------------------------------------
class OUSolver:
    """Système linéaire du krigeage universel, factorisé UNE fois par tirage de
    géométrie et réutilisé pour tous les instants cibles.

    `w_i(t) = m_i + u_i(t)`, `u ~ OU(0, σ_w², τ_w)` indépendant par candidat,
    `m` marginalisé avec un prior plat (krigeage universel — cf. `gp_math`).
    `sigma_house` : effet d'institut PARTAGÉ par tous les sondages d'une même
    maison (0 = désactivé) ; c'est ce qui empêche deux sondages d'un même
    institut de se confirmer à tort, cf. `gp_math.gp_posterior`.

    Factoriser une seule fois compte : c'est la seule opération non triviale de
    tout le mécanisme (`O((P·N)³)`), et on veut le postérieur à plusieurs dates
    (nowcast, mais aussi date de chaque sondage tenu dans un backtest).
    """

    def __init__(self, pobs: PseudoObservations, sigma_w2: float, tau_w: float,
                 sigma_house: float = 0.0, draw: int = 0):
        o, Cn = pobs.o[draw], pobs.C[draw]
        proj, t = pobs.proj, pobs.dates
        P, N = o.shape
        D = P * N
        self.sigma_w2, self.tau_w, self.t, self.proj = sigma_w2, tau_w, t, proj
        self.P, self.N, self.D = P, N, D

        Kt = sigma_w2 * np.exp(-np.abs(t[:, None] - t[None, :]) / tau_w)
        if sigma_house > 0:
            same = np.array([[a == b for b in pobs.instituts] for a in pobs.instituts], dtype=float)
            Kt = Kt + sigma_house**2 * same

        # G[p,q] = k(t_p,t_q)·C_{S_p}C_{S_q}, + bruit sur la diagonale de blocs
        G = np.einsum("pq,pij,qkj->piqk", Kt, proj, proj)
        for p in range(P):
            G[p, :, p, :] += Cn[p]
        G = G.reshape(D, D) + JITTER * np.eye(D)

        self.o = o.reshape(D)
        self.Bd = proj.reshape(D, N)                      # design du niveau moyen `m`
        self.c = cho_factor(G, lower=True)
        self.Gi_B = cho_solve(self.c, self.Bd)
        self.M = self.Bd.T @ self.Gi_B                    # singulier dans la direction 1 (jauge)
        self.Mp = np.linalg.pinv(self.M, rcond=1e-10)
        self.beta = self.Mp @ (self.Bd.T @ cho_solve(self.c, self.o))
        self.resid = self.o - self.Bd @ self.beta
        self.Gi_resid = cho_solve(self.c, self.resid)
        self.Cgauge = np.eye(N) - np.ones((N, N)) / N     # jauge globale Σ_i w_i = 0

    def _kstar(self, t_star: float) -> np.ndarray:
        k = self.sigma_w2 * np.exp(-np.abs(t_star - self.t) / self.tau_w)
        return np.transpose(k[:, None, None] * self.proj, (1, 0, 2)).reshape(self.N, self.D)

    def posterior_many(self, t_stars) -> list[tuple[np.ndarray, np.ndarray]]:
        """Postérieur à PLUSIEURS instants cibles, en une seule résolution
        triangulaire groupée. Un `cho_solve` à `T·N` seconds membres coûte
        bien moins que `T` résolutions à `N` — et un backtest en demande une
        par date de sondage tenu."""
        N = self.N
        KS = np.concatenate([self._kstar(float(t)) for t in t_stars], axis=0)   # (T·N, D)
        A = cho_solve(self.c, KS.T)                                            # (D, T·N)
        out = []
        for j in range(len(t_stars)):
            ks = KS[j * N:(j + 1) * N]
            mean = self.Cgauge @ self.beta + ks @ self.Gi_resid
            cov = self.sigma_w2 * self.Cgauge - ks @ A[:, j * N:(j + 1) * N]
            R = self.Cgauge - ks @ self.Gi_B              # correction d'estimation de `m`
            cov = cov + R @ self.Mp @ R.T
            out.append((mean, self.Cgauge @ (0.5 * (cov + cov.T)) @ self.Cgauge))
        return out

    def posterior(self, t_star: float) -> tuple[np.ndarray, np.ndarray]:
        """`(mean (N,), cov (N,N))` de `w(t_star)`, exprimés sur les contrastes
        (jauge `Σ_i w_i = 0`) — le niveau global n'est pas identifié et
        n'affecte pas `pi`."""
        return self.posterior_many([t_star])[0]

    def reml_nll(self) -> float:
        """Log-vraisemblance restreinte (niveau `m` marginalisé par prior plat)
        des pseudo-observations — l'objectif de calibration de `σ_w²`/`τ_w`.
        « Restreinte » au même sens que `model/core/opinion.py` : l'estimation
        et l'usage partagent EXACTEMENT la même vraisemblance.

        Les directions non identifiées (padding, mode commun) contribuent une
        constante indépendante des hyperparamètres : elles ne biaisent pas
        l'optimisation.
        """
        quad = float(self.resid @ cho_solve(self.c, self.resid))
        logdet = 2.0 * float(np.sum(np.log(np.diag(self.c[0]))))
        sign, ld_M = np.linalg.slogdet(self.M + 1e-12 * np.eye(self.N))
        return 0.5 * (quad + logdet + (ld_M if sign > 0 else 0.0))


def w_ou_posterior(pobs: PseudoObservations, as_of: float, sigma_w2: float, tau_w: float,
                   sigma_house: float = 0.0, draw: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Raccourci pour un seul instant cible (cf. `OUSolver`)."""
    return OUSolver(pobs, sigma_w2, tau_w, sigma_house, draw).posterior(as_of)


def reml_nll(pobs: PseudoObservations, sigma_w2: float, tau_w: float,
             sigma_house: float = 0.0, draw: int = 0) -> float:
    return OUSolver(pobs, sigma_w2, tau_w, sigma_house, draw).reml_nll()


# --- Loi de dynamique de `w` : `tau_w` emprunté, `sigma_w2` propre ---------------
def load_w_law(path: Path = BANK_W_OU_PATH) -> dict:
    """`{sigma_w2, tau_w, sigma_house}` — la loi de diffusion de `w`.

    **`tau_w` est repris de la banque COMMUNE** (`model/core/opinion.py`,
    `bank_opinion.json`, `tau ≈ 262 j`) et non réestimé. Deux raisons :

    1. *Identifiabilité.* Sur une fenêtre de campagne, tous les écarts sont
       petits devant `tau_w` et `σ_w²(1 − e^{-Δ/τ_w}) ≈ σ_w²Δ/τ_w` : seul le
       RAPPORT est contraint. Mesuré sur le synthétique, `(0.03, 50j)`,
       `(0.06, 100j)` et `(0.12, 200j)` sont à moins de 1.0 de nll les uns des
       autres. Laisser les deux libres, c'est fixer un point arbitraire d'une
       crête — et l'amplitude aux GRANDS écarts (un candidat non testé depuis
       46 jours, cf. spec §5.3) dépend, elle, de `σ_w²` seul.
    2. *Cohérence.* `tau` mesure la vitesse à laquelle l'opinion revient vers
       son niveau moyen. C'est la même opinion pour tous les modèles du dépôt ;
       seule l'AMPLITUDE dépend de l'espace de paramétrage (`w` ici, CLR des
       parts pour `gp_pooling`). Exactement l'argument de `opinion.py` sur
       pourquoi `sigma2` ne se reprend pas du saut terminal alors que la
       structure du noyau, elle, se reprend.

    `sigma_w2` est donc le SEUL paramètre calibré. La valeur livrée dans
    `bank_w_ou.json` a été retenue par **couverture hors-échantillon** sur
    2017+2022 (spec §11.5), pas par le REML de `calibrate_w_law` : sur les
    données réelles le REML est trop plat pour départager (§11.4), et c'est la
    couverture qui est le critère d'acceptation. `calibrate_w_law` reste utile
    pour situer une valeur dans la zone plausible du REML — la valeur retenue
    y est (Δnll = 3,2 vs l'argmin à τ=262 j).
    """
    if path.exists():
        return json.loads(path.read_text())
    from model.core.opinion import load_law
    return {"sigma_w2": DEFAULT_SIGMA_W2, "tau_w": float(load_law()["tau"]),
            "sigma_house": 0.0, "source": "repli (banque non calibrée)"}


def calibrate_w_law(pobs_by_bloc: list[PseudoObservations], tau_w: float,
                    grid: np.ndarray | None = None, sigma_house: float = 0.0,
                    draw: int = 0) -> dict:
    """`σ_w²` par REML sur une ou plusieurs fenêtres (une par élection), en
    vraisemblance composite — même schéma que `opinion.fit` sur ses blocs.
    `tau_w` est fixé (cf. `load_w_law`)."""
    if grid is None:
        grid = np.geomspace(0.01, 2.0, 41)
    nll = [sum(OUSolver(p, float(s2), tau_w, sigma_house, draw=draw).reml_nll()
               for p in pobs_by_bloc) for s2 in grid]
    k = int(np.argmin(nll))
    return {"sigma_w2": float(grid[k]), "tau_w": float(tau_w), "sigma_house": float(sigma_house),
            "reml_nll": float(nll[k]), "n_blocs": len(pobs_by_bloc)}


def w_draws_ou(pobs: PseudoObservations, as_of: float, sigma_w2: float, tau_w: float,
               sigma_house: float = 0.0, seed: int = 27, n_draw_per_geom: int = 1) -> np.ndarray:
    """Un (ou plusieurs) tirage de `w(as_of)` par tirage de géométrie -> `(S',N)`,
    directement consommable par `pi_draws_for_mask`.

    Chaque tirage porte donc les DEUX sources d'incertitude : la géométrie
    (`mu`,`sigma`, du postérieur NUTS) et la dynamique de `w` (postérieur OU).
    """
    rng = np.random.default_rng(seed)
    S, _, N = pobs.o.shape
    out = np.empty((S * n_draw_per_geom, N))
    for s in range(S):
        mean, cov = OUSolver(pobs, sigma_w2, tau_w, sigma_house, draw=s).posterior(as_of)
        vals, vecs = np.linalg.eigh(cov)
        L = vecs * np.sqrt(np.clip(vals, 0.0, None))
        z = rng.standard_normal((n_draw_per_geom, N))
        out[s * n_draw_per_geom:(s + 1) * n_draw_per_geom] = mean + z @ L.T
    return out


def main() -> None:
    """Calibre `sigma_w2` par REML sur les campagnes 2017 et 2022 COMPLÈTES et
    écrit `bank_w_ou.json`. `tau_w` n'est pas réestimé (cf. `load_w_law`).

    Même schéma que `calibration.py::main()` : le module est autonome, il ne
    lit dans le dossier d'aucun autre modèle.
    `.venv/bin/python -m model.models.spatial_pooling.w_dynamics`
    """
    import jax.numpy as jnp
    import pandas as pd

    from model.core.bank import Bank
    from model.core.inference import run_numpyro_mcmc
    from model.core.opinion import load_law
    from model.models.spatial_pooling.calibration import BANK_EXCESS_PATH
    from model.models.spatial_pooling.model import (build_poll_arrays, excess_var_for_nodes,
                                                    make_kappa, spatial_pooling_model)
    from pipeline.historical import (ELECTION_DATES, PARSED_DIR, POLL_FILES, _resolve_candidate,
                                     load_blocs)

    BLOC_ORDER = ["gauche_radicale", "gauche", "ecologistes", "centre", "droite", "droite_radicale"]
    tau_w = float(load_law()["tau"])
    excess_bank = Bank.load(BANK_EXCESS_PATH)
    pobs_all = []

    for election in (2017, 2022):
        polls = pd.read_csv(PARSED_DIR / POLL_FILES[election])
        t1 = polls[polls["tour"] == "Premier tour"].copy()
        blocs = load_blocs()
        be = blocs[(blocs["election"] == election) & (blocs["bloc"].isin(BLOC_ORDER))]
        real = set(be["candidat"])
        t1["candidat"] = t1["candidat"].apply(lambda raw: _resolve_candidate(raw, real))
        t1 = t1[t1["candidat"].isin(real)]
        d = pd.to_datetime(t1["date_fin"], errors="coerce")
        t1["date_fin"] = d.fillna(pd.to_datetime(t1.get("date_notice"), errors="coerce"))
        t1 = t1[t1["date_fin"].notna()]
        ed = pd.Timestamp(ELECTION_DATES[election])
        t1 = t1[(t1["date_fin"] > ed - pd.Timedelta(days=400)) & (t1["date_fin"] <= ed)]

        candidates = sorted(real)
        slot_of = np.array([BLOC_ORDER.index(be.set_index("candidat").loc[c, "bloc"])
                            for c in candidates])
        arrays = build_poll_arrays(t1, candidates)
        as_of = arrays["dates"].max()
        ev = excess_var_for_nodes(arrays["instituts"], excess_bank)
        kwargs = dict(slot_of=jnp.asarray(slot_of), n_slots=len(BLOC_ORDER),
                      tested_mask=jnp.asarray(arrays["tested_mask"]), Y=jnp.asarray(arrays["Y"]),
                      Np=jnp.asarray(arrays["Np"]),
                      kappa=jnp.asarray(make_kappa(arrays["dates"], as_of, 15.0)),
                      excess_var=jnp.asarray(ev))
        samples, _ = run_numpyro_mcmc(spatial_pooling_model, kwargs, draws=500, tune=500,
                                      chains=2, seed=27, target_accept=0.9)
        N = len(candidates)
        mu_d = np.asarray(samples["mu"]).reshape(-1, N)
        sig_d = np.asarray(samples["sigma"]).reshape(-1, N)
        sel = np.linspace(0, len(mu_d) - 1, 30).astype(int)
        pobs_all.append(build_pseudo_observations(mu_d[sel], sig_d[sel], arrays["tested_mask"],
                                                  arrays["Y"], arrays["Np"], arrays["dates"],
                                                  arrays["instituts"], ev))
        print(f"{election} : {N} candidats, {arrays['tested_mask'].shape[0]} nœuds")

    law = calibrate_w_law(pobs_all, tau_w)
    law["methode"] = ("REML sur les pseudo-observations (inversion locale exacte), campagnes "
                      "2017+2022 complètes, vraisemblance composite ; tau_w repris de "
                      "model/core/opinion.py (banque commune), non réestimé")
    BANK_W_OU_PATH.write_text(json.dumps(law, indent=2))
    print(f"\nLoi de w écrite dans {BANK_W_OU_PATH} : {law}")


if __name__ == "__main__":
    main()
