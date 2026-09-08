"""Tendances — « et si ce candidat gagnait 1 point par mois ? »

La couche « certitudes de l'utilisateur », réduite à ce qui a survécu aux
mesures : le modèle statistique porte l'incertitude, l'utilisateur pose une
trajectoire, et les deux ne se mélangent pas.

Le modèle. Une fraction `rho_i` de l'électorat bascule vers `i` par jour,
prélevée **uniformément** sur le reste du champ :

    pi_i(t) = pi_i(0) * (1 - R*t)  +  rho_i * t,        R = somme_j rho_j

La masse est conservée exactement (`somme pi = (1-Rt) + Rt = 1`), et la
trajectoire est **linéaire en t** — donc calculable en forme fermée, sans
solveur, y compris dans un navigateur.

**Pourquoi uniforme, et pas au voisinage.** Retirer un candidat est une question
de MÉCANISME : c'est là que le modèle spatial gagne sa vie, et c'est mesuré
(`notebooks/04o` : l'erreur sur le score d'un remplaçant passe de 3,89 à 2,21
points contre le réflexe « il hérite du sortant »). Poser une tendance est une
affirmation d'AGRÉGAT : le modèle n'a rien à dire sur qui perd, et prétendre le
contraire ajouterait une affirmation gratuite — même argument de parcimonie que
les ancres de rang (spec §2).

Ça a été confronté aux données (`notebooks/04q`, mouvements sondage à sondage du
même institut, 2022) : les voisins sur l'axe bougent en sens inverse un peu plus
que ne le prédit un null isotrope, mais la pente du résidu sur la distance ne
sort pas du bruit sur 66 couples. Il n'y a donc pas de mandat empirique pour
poser une captation locale, et l'uniformité est retenue **par défaut mesuré**,
pas par commodité.

Conséquence à connaître : la tendance n'a **aucune signature spatiale**. Le
graphique de l'axe (qui montre qui l'emporte à chaque position) est donc
inchangé par une tendance — c'est cohérent, pas un oubli.

**Calibrage du curseur.** Pour un seul candidat en tendance, `pi_k(t) =
pi_k(0) + rho_k * t * (1 - pi_k(0))` : la pente qui réalise une cible de
`c` points par mois vaut donc `c / (1 - pi_k)`, une division. Le facteur
`1 - pi_k` vaut 0,65 à 0,99 selon le candidat, donc « +1 pp/mois » veut dire la
même chose pour tout le monde à quelques pour cent près — c'est ce que le
prélèvement uniforme achète, et c'est la raison décisive de le préférer au
prélèvement local, où le facteur tombait à 0,3-0,5 et variait du simple au double.
"""

from __future__ import annotations

import numpy as np

# Le mois grégorien moyen, pas 30 jours : sur un horizon de sept mois l'écart
# vaut déjà 0,17 point de tendance à 1 pp/mois, du même ordre que ce qu'on
# prétend afficher.
JOURS_PAR_MOIS = 365.2425 / 12.0


def _median_share(pi: np.ndarray) -> np.ndarray:
    pi = np.asarray(pi, dtype=float)
    return np.median(pi, axis=0) if pi.ndim > 1 else pi


def calibre_pente(pi: np.ndarray, cible_pp_mois) -> np.ndarray:
    """`rho` (part par JOUR) réalisant `cible_pp_mois` points par mois.

    `pi` : parts du nowcast, `(K,)` ou `(S, K)` pour des tirages postérieurs.
    `cible_pp_mois` : en POINTS (1.0 = un point de pourcentage), par candidat.

    La pente est calculée **une fois** sur la part médiane, pas par tirage : la
    calculer par tirage ferait atterrir chaque tirage exactement sur la cible et
    supprimerait l'aléa résiduel de la conversion. L'écart est minime, les parts
    variant peu d'un tirage à l'autre.
    """
    m = _median_share(pi)
    cible = np.asarray(cible_pp_mois, dtype=float) / 100.0
    rho = np.zeros_like(m, dtype=float)
    actif = cible != 0.0
    rho[actif] = cible[actif] / np.clip(1.0 - m[actif], 1e-6, None) / JOURS_PAR_MOIS
    return rho


def applique(pi: np.ndarray, rho: np.ndarray, jours: float) -> np.ndarray:
    """`pi(t)` — la trajectoire, en forme fermée. Conserve la masse exactement."""
    pi = np.asarray(pi, dtype=float)
    rho = np.asarray(rho, dtype=float)
    R = float(rho.sum())
    return pi * (1.0 - R * jours) + rho * jours


def bornes_pp_mois(pi: np.ndarray, jours: float) -> tuple[np.ndarray, np.ndarray]:
    """Pente minimale et maximale admissibles, en points par mois, par candidat.

    Une trajectoire linéaire ne sature pas : elle finit par sortir du simplexe.
    Deux bornes, et elles sont **très asymétriques** — c'est le fait à porter
    dans l'interface :

    - vers le BAS, un candidat ne peut pas perdre plus qu'il n'a, sur AUCUN
      tirage : `pi_k + rho*t*(1-pi_k) > 0`. La borne est donc pilotée par le
      tirage le plus faible, et elle mord vite : un candidat à 5 points ne peut
      guère descendre au-delà de -0,2 pp/mois sur sept mois ;
    - vers le HAUT, il faut seulement `R*t < 1` pour que les autres restent
      positifs. C'est très lâche : +8 pp/mois reste admissible pour un gros
      candidat.

    Calculé sur les tirages quand on en fournit, parce que c'est le tirage le
    plus bas qui contraint, pas la médiane.
    """
    pi = np.asarray(pi, dtype=float)
    p = pi if pi.ndim > 1 else pi[None, :]
    m = _median_share(pi)
    # rho*t > -min_s( pi/(1-pi) ), converti en points par mois de CIBLE
    # (la cible vaut rho*t*(1-pi_median) sur l'horizon).
    ratio = np.min(p / np.clip(1.0 - p, 1e-9, None), axis=0)
    lo = -ratio / max(jours, 1e-9) * JOURS_PAR_MOIS * (1.0 - m) * 100.0
    hi = np.full_like(lo, 1.0 / max(jours, 1e-9) * JOURS_PAR_MOIS * 100.0) * (1.0 - m)
    return lo, hi
