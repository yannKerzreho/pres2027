# Biais systématique du RN dans le SSM nowcast — récapitulatif de session (2026-08-10)

Point de départ : le nowcast RN semblait décrocher vers le bas malgré des
sondages récents plus hauts (ex. nœud Elabe 25-27 mars : RN passe de 31,0%
à 29,8% alors que le sondage rapporte 34%). Cette note résume, dans l'ordre,
toutes les hypothèses testées, leur verdict, et l'état actuel — les
dérivations mathématiques complètes de la partie encodage ALR/ILR sont dans
[`spec_ssm_encodage_partiel.md`](spec_ssm_encodage_partiel.md), pas
reproduites ici.

## Ce qui a été testé

| # | Hypothèse | Méthode | Verdict |
|---|---|---|---|
| 1 | Le choix de référence locale (`argmax`) dans `poll_plan` privilégie structurellement le RN | Preuve algébrique (invariance de reparamétrisation) + vérif. numérique synthétique et réelle (forcer une référence différente au nœud Elabe) | **RÉFUTÉ** — écart exactement nul (< 1e-6). L'argmax est même le choix numériquement le plus stable. |
| 2 | Simplifier `R` à `(deff/n)·I` (modèle de base, ILR local) réduit la dérive | Implémenté (`full_covariance` flag), rejoué sur le nœud Elabe | **Amélioration partielle** — baisse réduite de -1,21 à -1,03 pt. La covariance jointe reste active. |
| 3 | RN sous-testé à cause de l'exclusion de Bardella (`SLOTS`, choix du 2026-08-09) | Comptage réel : Bardella testé 16x (moy. 36,6%) vs Le Pen 4x sur la fenêtre mars ; 33 hypothèses Le Pen sur toute la campagne au 10/08 | **Confirmé comme cause de la rareté des tests** — choix `SLOTS` explicitement gardé tel quel (pas de fusion/slot séparé). |
| 4 | Le SSM est cassé (sd=14,6 pts au nowcast final `as_of=2026-08-10`) | NUTS complet (68 nœuds), localisation précise (extrapolation 31j sans sondage), vérifié sur tous les candidats | **RÉFUTÉ** — SSM sain (R-hat 1,004, ESS 1375, 0 divergence). Le sd large touche TOUS les candidats, dû à `tau` isotrope + house effects non modélisés (`use_biais=True` divise `tau` et le sd par ~2). Séparé du sujet principal (biais de la MOYENNE, pas du sd). |
| 5 | Le biais de la moyenne persiste après le 27 mars, sur toute la campagne | Écart moyen (filtré − rapporté) sur les 30 nœuds testant RN, comparé à 4 autres candidats bien testés | **Confirmé** — RN : -1,82 pt (93% sous-estimé) vs ~-0,5/-0,6 pt pour les autres (lag général, normal). RN a un excès de biais ~3x plus grand. |
| 6 | La corrélation croisée dans la covariance jointe (état) cause cet excès | Décorrélation forcée (covariance diagonale à chaque étape) sur les VRAIES données Le Pen seules (pas de Bardella) | **RÉFUTÉ** — biais quasi identique (-1,88 vs -1,82 pt). Le hack casse même la cohérence du filtre (gains hors [0,1]). |
| 7 | Bug dans le pipeline de données (valeur RN mal transmise au modèle) | Comparaison directe `data.raw_values` vs rapports bruts, nœud par nœud | **Écarté** — les valeurs transmises sont exactes. |
| 8 | Asymétrie de VARIANCE (pas de corrélation) au sein d'une observation partagée | Corrélation entre (mouvement de RN à un nœud) et (résidu moyen des AUTRES candidats testés dans la MÊME hypothèse) | **Confirmé** — corrélation -0,75. RN, bien plus incertaine que ses co-testés, absorbe une part disproportionnée de toute correction partagée, même quand SON PROPRE rapport est exact. |
| 9 | `tau` par candidat corrige ça | Implémenté (`use_tau_candidat`, `Q = Vᵀ·diag(tau_candidat²)·V`), NUTS complet | **REJETÉ** — inférence saine (R-hat 1,013, 10 div./3000, mineur) et `tau_RN` bien identifié comme petit (0,021, cohérent avec sa faible volatilité brute) — mais la corrélation ciblée ne bouge presque pas (-0,73 vs -0,75) ET le biais global s'AGGRAVE (-2,01 vs -1,82 pt) : RN est sur une tendance haussière régulière peu bruitée, et un `tau` plus petit la rend plus lente à suivre cette tendance réelle. |
| 10 | Revoir l'encodage local (ALR-référence + bruit diagonal indépendant, au lieu de la base de Helmert ILR-locale dont chaque ligne mélange récursivement plusieurs candidats par construction) | Ré-implémentation ad hoc de l'encodage, testée en filtre déterministe sur les vraies données Le Pen | **Amélioration modeste seulement** — corrélation -0,72 (vs -0,75), biais -1,61 pt (vs -1,82). Dans le même ordre de grandeur que le point 2, pas une correction. |
| 11 | Réactiver le terme de tendance/momentum (`use_tendance=True`, déjà implémenté dans `nowcast.py` mais jamais validé, cf. `spec_ssm_tendance.md`) | NUTS complet (68 nœuds, `use_biais=False`, `full_covariance=False`, sans Bardella) | **INUTILISABLE en l'état** — plus de 160 minutes de calcul (98% CPU) sans produire un seul résultat, contre 2-5 min pour toutes les autres configurations testées cette session. Processus arrêté manuellement. Confirme, sur le jeu de données complet, le problème de géométrie pathologique déjà documenté (`nowcast.py` : covariance niveau/tendance quasi-singulière, R-hat 180-1950 dans une session antérieure). |

## Conclusion actuelle

- **Le SSM n'est pas cassé** — vérifié à trois niveaux indépendants : les
  maths de l'encodage (preuve + numérique), la santé de l'inférence NUTS
  (R-hat/ESS/divergences sur plusieurs configurations), et l'absence de bug
  de données/pipeline.
- **Le biais RN est réel et a une cause précisément identifiée** :
  l'asymétrie de variance entre RN et ses co-testés au sein d'une même
  hypothèse de sondage — pas la corrélation d'état, pas le rythme
  jour-à-jour, pas un bug d'encodage ou de données.
- **La cause racine de cette asymétrie** : RN est testée bien moins souvent
  que les autres grands candidats, à cause de l'exclusion délibérée et
  confirmée de Bardella (`SLOTS`).
- **Quatre corrections tentées, quatre rejetées ou insuffisantes** :
  simplifier `R` (aide un peu, insuffisant), `tau` par candidat (n'aide
  pas, aggrave même le retard sur tendance), changer la base locale
  (ALR-référence + bruit diagonal — amélioration modeste, même ordre de
  grandeur que simplifier `R`), et le terme de tendance/momentum
  (inutilisable en l'état, NUTS ne converge pas en temps raisonnable sur
  le jeu complet).
- Les trois interventions ciblant la corrélation/variance/encodage donnent
  toutes un effet modeste et similaire (~15-20% de réduction du biais,
  jamais une résolution) — ça suggère que le biais résiduel n'est pas
  localisé dans un mécanisme unique et corrigible localement, mais dans la
  contrainte de composition elle-même (softmax + somme=100%) combinée à la
  rareté des tests RN — pas quelque chose qu'un ajustement local de
  l'encodage ou de `tau` peut éliminer sans plus de données RN directes.

## Ce qui reste ouvert

Aucune piste testée à ce stade ne règle le problème de façon satisfaisante.
Options restantes, non essayées ou nécessitant plus de travail :

1. **`use_tendance` avec un état de tendance restreint/simplifié** (ex. un
   seul `tau_tendance` par bloc plutôt que par les 11 dimensions ILR, ou un
   prior beaucoup plus contraint sur `sigma_beta0`) pour contourner le
   problème de géométrie pathologique observé au point 11 — piste plausible
   mais qui demande un travail d'implémentation, pas juste activer le flag
   existant.
2. **Accepter la limite comme structurelle** : sans plus de données RN
   directes (Bardella exclu, décision confirmée), un modèle compositionnel
   joint semble condamné à faire porter à RN une partie du rattrapage des
   autres candidats. Documenter la limite plutôt que continuer à chercher
   un correctif local.

## Addendum 2026-08-11 — cause racine finalement tranchée

Les tests du 2026-08-10 isolaient correctement le **symptôme local** (RN trop
incertaine et "bon marché" à déplacer dans des hypothèses partagées), mais pas
la **cause racine de pipeline**. L'enquête complémentaire du 2026-08-11 montre
que le biais RN/Horizons vient principalement de la combinaison suivante :

1. `aggregate_to_slots()` ne garde que les candidats mappés vers les `SLOTS`
   du modèle (`raw["slot"].notna()`), donc les candidats hors roster fixe sont
   simplement retirés de l'observation.
2. `poll_observation_values()` renormalise ensuite les parts testées par
   `p = values_tested / values_tested.sum()`.
3. Une hypothèse partielle "sans RN" ou "sans Philippe" ne laisse donc PAS
   cette masse "hors modèle" ; elle la **redistribue implicitement** entre les
   candidats restants via la renormalisation.

Ce n'est pas un détail : sur la campagne 2026, la masse moyenne jetée avant
renormalisation est de **28,75 points** (médiane **34 points**). Quand `RN` est
absente d'une hypothèse, la masse jetée monte à **43,04 points** en moyenne ;
quand `RN` est présente, elle tombe à **9,41 points**. Autrement dit, beaucoup
d'hypothèses "sans RN" sont en réalité des observations où presque la moitié du
vote testé est hors modèle, puis **réallouée** aux autres candidats avant
d'entrer dans le SSM.

### Conséquence directe

Le SSM ne voyait pas "le même électorat avec RN manquante" ; il voyait une
**composition renormalisée sur un roster plus petit**, traitée comme si elle
était directement comparable aux hypothèses contenant RN/Philippe. En agrégeant
toutes ces hypothèses dans un même état latent, le modèle poussait
structurellement les candidats présents vers le haut et les absents vers le
bas lors du recollage global — d'où le biais RN/Horizons observé.

### Vérification empirique

Quand on restreint l'analyse à un roster cohérent et réellement observé
(`LO, LFI, PCF, EELV, PP, HOR, LR, DLF, RN, REC`), le biais RN s'effondre :

- SSM "full roster incohérent" : erreur signée moyenne RN ≈ **-2,82 pt**
- SSM "roster exact 10 cohérent" : erreur signée moyenne RN ≈ **-0,26 pt**

Le gros du biais venait donc bien du **mélange d'hypothèses hors-roster +
drop + renormalisation**, pas d'un défaut intrinsèque du filtre de Kalman, ni
du seul prior `tau`.

## Solution retenue à ce stade

### À faire

- **Ne plus mélanger des hypothèses de roster différents dans un même SSM de
  roster fixe.**
- Pour un modèle baseline lisible, choisir un **roster cohérent explicitement
  défini** (par ex. `LO, LFI, PCF, EELV, PP, HOR, LR, DLF, RN, REC`) et ne
  garder que les hypothèses qui testent exactement ce roster.

### À ne pas faire

- Continuer à injecter des hypothèses partielles/hors-roster dans le même
  état latent en espérant que `tau`, `R` ou le choix ILR/ALR corrigeront le
  problème.
- Interpréter une composition renormalisée après suppression des non-modélisés
  comme une observation "partielle mais comparable" du roster cible : ce n'est
  vrai que si la masse supprimée est négligeable, ce qui n'est pas le cas ici.

### Limite restante

Le roster exact 12 slots actuellement codé (`SLOTS`) est **trop strict** pour
les données disponibles au 2026-08-11 : aucune hypothèse ne le teste
exactement. La solution pratique n'est donc pas "forcer 12 slots coûte que
coûte", mais **aligner le roster du modèle sur un noyau réellement observé**
ou revoir l'ingestion pour remapper explicitement certaines alternatives avant
le SSM.
