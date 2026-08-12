"""Calibration des hyperparamètres du noyau et du house effect (spec §6).

Estimateur : **maximum de vraisemblance restreint (REML)** dans le modèle exact
du §3.3, sur les sondages 2017/2022. « Restreint » = la moyenne `mu` de chaque
slot est marginalisée avec un prior plat, exactement comme dans
`gp.gp_posterior` — l'estimation et l'usage partagent donc la même définition
de la vraisemblance, pas deux approximations voisines.

Pourquoi ne PAS reprendre `sigma2` du saut terminal
---------------------------------------------------
La spec §2 pariait que le noyau était « déjà calibré » : `sigma2 = scale²/2`
avec `scale` issu de `bank_jump.json`. **Les données démentent ce pari.** Le
`scale` du saut terminal est ajusté sur `clr(résultat) − clr(sondage)`, un écart
qui contient deux choses :

  1. le mouvement d'opinion entre la date du sondage et le scrutin ;
  2. l'écart systématique sondages → résultat le jour J (erreur de sondage,
     indécis, participation différentielle) — qui ne dépend PAS du délai.

Seul (1) est la covariance qu'il faut entre DEUX SONDAGES. Mesuré sur les
paires de sondages d'instituts différents de 2022 (cf. `main()`), l'excès de
variance observé à 14-30 jours d'écart vaut 0,063 là où le noyau de la Bank en
prédit 0,327 — un facteur 5. Utiliser `scale²/2` reviendrait à dire que
l'opinion bouge cinq fois plus vite qu'elle ne bouge, donc à jeter tout sondage
de plus d'une semaine et à produire des intervalles massivement trop larges.

On réestime donc `sigma2` (et `tau`) par REML **sur le mouvement sondage à
sondage**, qui est la quantité dont le modèle a besoin. Le saut terminal, lui,
reste utilisé tel quel pour la jambe `as_of → scrutin` (`model/core/simulate.py`),
où il est calibré sur exactement la bonne quantité.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from model.models.gp_pooling.gp import JITTER, clr_rows, ou_kernel, sampling_cov_clr_diag
from pipeline.historical import load_calibration_frame

PARAMS_PATH = Path(__file__).parent / "params_gp.json"

# Plancher sur les parts avant le log : un candidat crédité de 0,0 % rendrait le
# CLR et sa variance d'échantillonnage infinis. 0,3 % est sous le plus petit
# candidat réellement mesuré (Arthaud ~0,5 %), donc le plancher ne mord jamais
# sur une valeur informative.
SHARE_FLOOR = 0.003


@dataclass
class PollBlock:
    """Un groupe de sondages **comparables** : même élection, même champ exact
    de candidats.

    Le CLR d'une composition dépend de l'ensemble des candidats sur lequel il
    est calculé. Deux instituts qui testent des champs différents ne mesurent
    donc pas la même quantité et leurs CLR ne sont pas superposables (spec §6,
    même motif que `filter_scenarios_by_exact_slots`). D'où le regroupement par
    roster EXACT plutôt qu'un simple empilement.
    """
    election: int
    roster: tuple[str, ...]
    Y: np.ndarray          # (P, K) parts, sommant à 1
    Z: np.ndarray          # (P, K) CLR
    V: np.ndarray          # (P, K) variance d'échantillonnage CLR
    n: np.ndarray          # (P,) taille d'échantillon
    h: np.ndarray          # (P,) horizon (jours avant le scrutin)
    inst: np.ndarray       # (P,) nom d'institut
    inst_idx: np.ndarray   # (P,) code entier d'institut

    @property
    def K(self) -> int:
        return self.Y.shape[1]

    @property
    def P(self) -> int:
        return self.Y.shape[0]


def history_blocks(min_polls: int = 2) -> list[PollBlock]:
    """Sondages 2017/2022 regroupés par (élection × roster exact)."""
    df = load_calibration_frame()
    df = df[df["echantillon"].notna() & df["intention"].notna()]
    blocks: list[PollBlock] = []
    for elec, de in df.groupby("election"):
        roster_of = de.groupby("id")["candidat"].apply(lambda s: tuple(sorted(s)))
        de = de.assign(roster=de["id"].map(roster_of))
        for roster, dg in de.groupby("roster"):
            piv = dg.pivot_table(index="id", columns="candidat", values="intention")
            piv = piv[list(roster)]
            if len(piv) < min_polls or piv.isna().any().any():
                continue
            meta = dg.drop_duplicates("id").set_index("id").loc[piv.index]
            Y = np.clip(piv.to_numpy(float) / 100.0, SHARE_FLOOR, None)
            Y = Y / Y.sum(axis=1, keepdims=True)
            n = meta["echantillon"].to_numpy(float)
            inst = meta["institut"].to_numpy()
            blocks.append(PollBlock(
                election=int(elec), roster=tuple(roster), Y=Y, Z=clr_rows(Y),
                V=sampling_cov_clr_diag(Y, n), n=n,
                h=meta["horizon"].to_numpy(float), inst=inst,
                inst_idx=pd.factorize(inst)[0]))
    return blocks


def reml_nll(blocks: list[PollBlock], tau: float, sigma2: float, sigma_h: float,
             sigma_p: float = 0.0) -> float:
    """Log-vraisemblance restreinte NÉGATIVE, sommée sur les blocs et les slots.

    C'est une vraisemblance **composite** : les `K` composantes CLR d'un même
    sondage ne sont pas indépendantes (elles somment à zéro), mais on somme
    quand même leurs log-densités marginales. C'est cohérent avec le modèle,
    qui traite lui aussi les slots indépendamment (§3.2/§7.2) — l'estimateur
    reste consistant, seules ses barres d'erreur seraient à corriger.

    `sigma_p` : bruit supplémentaire PAR SONDAGE (au-delà de l'échantillonnage
    et du house effect d'institut). Sert à tester le point ouvert §7.4 : si
    l'estimation le renvoie ~0, c'est que l'excès de dispersion est bien
    attaché à l'institut et non au sondage individuel.
    """
    if min(tau, sigma2, sigma_h) <= 0 or sigma_p < 0 or tau > 5000:
        return 1e12
    total = 0.0
    for b in blocks:
        base = (ou_kernel(b.h, b.h, tau, sigma2)
                + sigma_h**2 * (b.inst_idx[:, None] == b.inst_idx[None, :]))
        one = np.ones(b.P)
        for s in range(b.K):
            Sigma = base + np.diag(b.V[:, s] + sigma_p**2) + JITTER * np.eye(b.P)
            try:
                c = cho_factor(Sigma, lower=True)
            except np.linalg.LinAlgError:
                return 1e12
            logdet = 2.0 * np.log(np.diag(c[0])).sum()
            Si1, Siz = cho_solve(c, one), cho_solve(c, b.Z[:, s])
            a = float(one @ Si1)
            quad = float(b.Z[:, s] @ Siz) - float(one @ Siz) ** 2 / a
            total += 0.5 * (logdet + np.log(a) + quad)
    return total


def fit(blocks: list[PollBlock], with_sigma_p: bool = True) -> dict:
    """Ajuste `(sigma_h, sigma2, tau)` — et `sigma_p` si `with_sigma_p` — par REML.

    `sigma_p` (bruit propre à chaque sondage, au-delà de l'échantillonnage et du
    house effect de sa maison) n'était pas dans la spec. Il est ajouté parce que
    les données le réclament : l'ajouter fait gagner 16 unités de
    log-vraisemblance restreinte pour un seul paramètre (test du rapport de
    vraisemblance : 2Δ = 32 à 1 ddl, p < 1e-8). Il reste PETIT devant `sigma_h`
    (0,038 contre 0,125), ce qui tranche le point ouvert §7.4 dans le sens de la
    spec : l'excès de dispersion est bien attaché à l'INSTITUT, `sigma_p` n'en
    est qu'un résidu.
    """
    def obj(x):
        sh, s2, tau = np.exp(x[:3])
        sp = np.exp(x[3]) if with_sigma_p else 0.0
        return reml_nll(blocks, tau, s2, sh, sp)

    best = None
    # Plusieurs départs : la vraisemblance en (sigma2, tau) est plate le long de
    # la direction « amplitude × portée », un seul Nelder-Mead peut s'arrêter
    # sur un plateau.
    for t0 in (20.0, 73.0, 200.0, 600.0):
        for s0 in (0.05, 0.15, 0.6):
            x0 = np.log([0.1, s0, t0] + ([0.03] if with_sigma_p else []))
            r = minimize(obj, x0, method="Nelder-Mead",
                         options={"maxiter": 6000, "maxfev": 6000,
                                  "xatol": 1e-4, "fatol": 1e-4})
            if best is None or r.fun < best.fun:
                best = r
    sh, s2, tau = np.exp(best.x[:3])
    sp = float(np.exp(best.x[3])) if with_sigma_p else 0.0
    return {"sigma_h": float(sh), "sigma2": float(s2), "tau": float(tau),
            "sigma_p": sp, "reml_nll": float(best.fun),
            "n_blocs": len(blocks), "n_sondages": int(sum(b.P for b in blocks)),
            "n_obs": int(sum(b.P * b.K for b in blocks))}


def save_params(params: dict, path: Path = PARAMS_PATH) -> Path:
    path.write_text(json.dumps(params, ensure_ascii=False, indent=2))
    return path


def load_params(path: Path = PARAMS_PATH) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    """`.venv/bin/python -m model.models.gp_pooling.calibration`"""
    blocks = history_blocks()
    print(f"{len(blocks)} blocs (élection × roster), "
          f"{sum(b.P for b in blocks)} sondages, {sum(b.P * b.K for b in blocks)} obs slot×sondage")

    params = fit(blocks)
    print(f"\nREML : sigma_h = {params['sigma_h']:.4f}  sigma_p = {params['sigma_p']:.4f}  "
          f"sigma² = {params['sigma2']:.4f} (sigma = {np.sqrt(params['sigma2']):.3f})  "
          f"tau = {params['tau']:.1f} j")

    # Point ouvert §7.4 : le house effect est-il bien attaché à l'INSTITUT ?
    sans = fit(blocks, with_sigma_p=False)
    print(f"  sans sigma_p : sigma_h = {sans['sigma_h']:.4f}  nll {sans['reml_nll']:.1f} "
          f"(vs {params['reml_nll']:.1f}, soit 2Δ = {2*(sans['reml_nll']-params['reml_nll']):.1f} à 1 ddl)")

    # Comparaison au noyau du saut terminal, que la spec §2 proposait de
    # reprendre tel quel.
    from model.core.bank import Bank
    from model.core.movements import BANK_JUMP_LEGACY
    jb = Bank.load(BANK_JUMP_LEGACY)
    tau_b, s2_b = jb.jump_tau.item()[0], 0.5 * jb.jump_scale.at(bloc="droite")[0] ** 2
    nll_b = reml_nll(blocks, tau_b, s2_b, params["sigma_h"], params["sigma_p"])
    print(f"  noyau du saut terminal (tau={tau_b:.0f}, sigma²={s2_b:.3f}) : "
          f"nll {nll_b:.1f} — soit {nll_b - params['reml_nll']:+.0f} vs le noyau réestimé")

    # Profil de vraisemblance : intervalle de confiance sur sigma_h.
    print("\nProfil sur sigma_h (sigma², tau, sigma_p réoptimisés) :")
    for sh in (0.08, 0.10, 0.12, 0.125, 0.14, 0.16, 0.20):
        rp = minimize(lambda x: reml_nll(blocks, np.exp(x[1]), np.exp(x[0]), sh, np.exp(x[2])),
                      np.log([params["sigma2"], params["tau"], max(params["sigma_p"], 1e-3)]),
                      method="Nelder-Mead")
        print(f"  sigma_h = {sh:.3f} : nll {rp.fun:9.2f}  (Δ = {rp.fun - params['reml_nll']:+7.2f})")

    params["methode"] = ("REML dans le modèle §3.3 (+ sigma_p), blocs "
                         "(élection × roster exact) 2017/2022, mu marginalisé par slot, "
                         "vraisemblance composite sur les slots")
    print(f"\nParamètres écrits dans {save_params(params)}")


if __name__ == "__main__":
    main()
