# Spec — `spatial-pooling`

Notice technique du modèle. Les mesures qui ont motivé ces choix, les pistes
essayées puis abandonnées et l'historique des corrections sont dans
`historic/` à la racine du dépôt ; ici, seulement le modèle.

Même genre que `../gp_pooling/spec_gp_pooling.md`.

---

## 0. Ce que ce modèle fait, et pourquoi il existe

`gp_pooling` estime l'état d'opinion d'un **champ fixe** de candidatures. Il ne
peut rien dire d'un champ qu'il n'a pas vu : retirer un candidat d'une liste
change la composition, et la seule réponse disponible sans modèle est le
**proportionnel** — répartir ses voix au prorata des autres.

Ce modèle-ci place les candidats sur un axe latent et les fait recruter par un
softmax sur cet axe. Retirer un candidat redistribue alors ses électeurs vers
ses **voisins**, par construction et non par une matrice de reports posée à la
main. C'est sa promesse, et le seul critère qui le valide (§8).

Corollaire de portée : l'information géométrique ne vient **que de la variation
du champ testé** d'un sondage à l'autre. Sur un champ unique, l'inversion
$\pi(w)=Y$ admet une solution pour n'importe quelle géométrie (§6), donc rien
n'est identifié. Le modèle sert tant que les instituts testent des listes
différentes.

## 1. Notations

$N$ candidats, indice $i$. $P$ **nœuds** d'observation : un nœud = une
hypothèse d'un sondage (un sondage à $h$ hypothèses produit $h$ nœuds).
$S_p\subseteq\{1..N\}$ est le champ testé au nœud $p$, encodé par le masque
$M_{i,p}\in\{0,1\}$.

L'axe politique est $[0{,}01\,;0{,}99]$, discrétisé en $B=25$ nœuds de
quadrature $V_b$ de poids $W_b$ (**Gauss-Legendre**, $\sum_b W_b=1$). La part
d'un candidat est une intégrale, la grille en est la quadrature : Gauss-Legendre
intègre exactement les polynômes de degré $2B-1$, d'où une erreur de $10^{-5}$
point dès $B=20$ là où une grille uniforme laisse 0,6 point à $B=50$.

> Une densité électorale non uniforme ne serait pas identifiable séparément des
> positions : $\int g(v)f(v)\,dv=\int g(F^{-1}(u))\,du$. Un électorat déformé
> équivaut à un électorat uniforme à positions transportées par $F$, et l'axe
> étant latent, cette liberté n'a pas de contenu.

## 2. Géométrie — position et rayon

Chaque candidat a **sa** position, ancrée sur l'ordre politique connu et libre
autour :

$$\mu_i = \mathrm{sigmoid}\big(\mathrm{logit}(a_i) + z^\mu_i\,s_\mu\big),
\qquad z^\mu_i\sim\mathcal N(0,1),\quad s_\mu = 0{,}4$$

$a_i$ est l'**ancre** : le rang du candidat dans l'ordre politique
(`position_anchors`), uniformément réparti sur l'axe. On affirme donc qui est à
gauche de qui, et **rien sur les distances**. L'ordre lui-même n'est pas imposé
non plus : $s_\mu$ laisse de quoi faire se croiser deux voisins, et le postérieur
retrouve pourtant l'ordre attendu — le placement est une prédiction vérifiable,
pas un postulat. C'est ainsi qu'a été corrigé l'ordre interne du groupe RN, le
postérieur plaçant spontanément Dupont-Aignan à gauche de Bardella, lui-même à
gauche de Le Pen.

> Un placement expert (grappes serrées, écarts variables) a été construit et
> mesuré : les écarts de redistribution entre placements sont de l'ordre de
> 0,01 pt, et sur la coupure la plus difficile la variabilité de graine écrase
> l'effet de configuration. La mesure ne départage pas, et le rang est retenu
> par **parcimonie** — c'est le choix qui publie le moins d'affirmations
> éditoriales — pas parce qu'il aurait gagné.

> **Pourquoi une ancre par candidat, et pas des blocs.** Faire partager une
> position latente à plusieurs candidats, avec un écart individuel libre de
> signe par-dessus, rend le postérieur **multimodal par permutation** : les
> deux ordres expliquent les sondages presque aussi bien, chaque chaîne en
> choisit un, et l'échantillonneur ne peut pas passer de l'un à l'autre. Ce
> n'est pas théorique — c'était le défaut central du modèle, et le corriger fait
> passer 2022 J-210 de R-hat 2,02 / ESS 2,7 à 1,01 / ESS 117.
>
> Corollaire : deux ancres trop proches recréent le problème.
> `ancres_ecartees` impose un écart minimal en logit — contrainte
> d'**identifiabilité**, pas affirmation politique : on refuse de dire que deux
> candidats sont au même endroit, ce que les données ne peuvent pas trancher.
> Avec des ancres de rang elle ne mord jamais (l'équirépartition garantit déjà
> l'écart) ; elle reste le garde-fou si le placement vient d'ailleurs.

Rayon d'action : un rayon global, plus un écart par candidat de largeur **fixe**.

$$\sigma_i = \sigma\,\exp\big(z^\sigma_i\,s_\sigma\big),\qquad
\sigma\sim\mathrm{LogNormal}(\log 0{,}15\,;0{,}6),\quad s_\sigma = 0{,}30$$

L'écart individuel dit à quel point le candidat est un parti **attrape-tout**.
Sa largeur est figée délibérément : l'échantillonner reviendrait à faire
multiplier un vecteur de dimension $N$ par une échelle libre, c'est-à-dire à
créer un entonnoir.

> $\sigma$ n'est identifié par presque aucune donnée — contraction
> prior→postérieur $\approx 0$, sauf pour les candidats en compétition
> rapprochée. Il ne faut pas en conclure qu'on peut le **fixer** : mesuré, cela
> effondre l'ESS à 4. $\sigma$ absorbe le désajustement d'un modèle à une seule
> dimension ; le figer renvoie ce désajustement sur les positions et sur $w$,
> qui eux sont contraints. Un paramètre peut être indéterminé et néanmoins
> indispensable à la souplesse du reste.
>
> Attention aussi à ce que $\sigma$ signifie : c'est le **noyau** avant
> compétition, pas le profil de recrutement. Un $\sigma$ large ne veut pas dire
> qu'un candidat attire loin, le softmax écrasant tout face à un voisin mieux
> placé.

## 3. Attractivité et parts de vote

$$D_{i,b} = -\frac{(V_b-\mu_i)^2}{2\sigma_i^2},\qquad
A_{i,b} = w_i + D_{i,b} - \log\Big(\sum_{b'}W_{b'}e^{D_{i,b'}}\Big)$$

Le troisième terme **normalise le noyau** de chaque candidat à masse 1 sur la
grille. Sans lui, seule la combinaison $w_i+\log\sigma_i$ est identifiée et
$\sigma$ flotte sur une crête. Normalisé, la part vaut $\propto e^{w_i}$ :
$w$ porte le niveau, $\sigma$ ne porte plus que la forme. C'est une
reparamétrisation exacte — une constante par candidat avant un softmax.

Softmax **masqué** sur la grille, puis agrégation :

$$P_{i,b,p} = \frac{M_{i,p}e^{A_{i,b}}}{\sum_j M_{j,p}e^{A_{j,b}}},
\qquad \pi_{i,p} = \sum_b P_{i,b,p}W_b$$

Le masque est multiplicatif plutôt qu'un $-\infty$ dans l'exponentielle : plus
stable, et entièrement vectorisé sur $(P,N,B)$ sans `scan`.

**Jauge.** $A$ est invariant par $w_i\to w_i+c$ : $w$ n'a pas de niveau absolu,
exactement comme le CLR. À fixer avant tout affichage de $w$.

## 4. Dynamique — niveau et dérive

$$w_i(t) = m_i + u_i(t),\qquad m_i = z^m_i\cdot 1{,}5,\qquad
u_i \sim \mathcal{GP}\big(0,\ \sigma_w^2\,e^{-|t-t'|/\tau}\big)$$

$m_i$ est le **niveau moyen** du candidat sur la fenêtre, $u_i$ sa **dérive**
autour. Les séparer est nécessaire, pas cosmétique : sans $m_i$, la même échelle
$\sigma_w$ devrait porter l'écart de niveau entre candidats (écart-type 1,1 en
log-ratio — de 1 % à 36 % de part) et la dérive temporelle de chacun. Le
postérieur arbitre alors à mi-chemin, sous-dispersant les niveaux et
sur-dispersant la dérive d'un facteur 2.

$\tau\approx 262$ jours est **fixé**, repris de la banque commune
(`model/core/opinion.py`) : sur une fenêtre de campagne, seul le rapport
$\sigma_w^2/\tau$ est contraint, les laisser libres tous deux revient à choisir
un point d'une crête plate.

Le noyau OU plutôt qu'une marche aléatoire : la variance de dérive **sature** à
$\sigma_w^2(1-e^{-2h/\tau})$ au lieu de croître sans borne.

**Troncature de Karhunen-Loève.** $\tau$ étant fixé, la covariance sur la grille
des dates est connue avant tout tirage : on la diagonalise une fois, hors
échantillonnage, et on ne garde que les composantes portant 99 % de la variance
(`ou_kl_basis`). Avec $\tau\approx262$ j et des sondages espacés de 2-3 jours,
3 à 4 composantes suffisent souvent. La diagonalisation porte sur le
**complément orthogonal de la constante** — celle-ci est le niveau, déjà porté
par $m_i$.

Seuls les candidats au-dessus de 5 % de part moyenne reçoivent un chemin ; les
autres gardent $w_i = m_i$. La trajectoire d'un candidat à 1 % n'est pas
identifiable sous $\pm1$ point de bruit d'échantillonnage. *(Seuil non calibré,
et le critère devrait porter sur la variabilité observée de la part plutôt que
sur son niveau.)*

## 5. Observations

$\tilde Y_p$ = intentions rapportées, restreintes au roster puis
**renormalisées** sur $S_p$ ; $n_p$ = échantillon $\times$ la masse conservée
par cette restriction. La restriction d'un multinomial à un sous-ensemble étant
un multinomial, c'est la loi exacte et non un ajustement.

$$\tilde Y_{i,p}\sim\mathcal N\big(\pi_{i,p},\ v_{i,p}\big),\qquad
v_{i,p} = \frac{\pi_{i,p}(1-\pi_{i,p})}{n_p} + \tau^2_{\mathrm{inst}(p)}$$

pour $i\in S_p$, les nœuds étant traités comme indépendants.
$\tau_{\mathrm{inst}}$ est la variance d'excès par institut (`bank_excess.json`,
calibrée sur des paires de sondages de champ identique).

$n_p$ est en outre **déflaté par le nombre d'hypothèses** du sondage : les $h$
hypothèses sont posées aux mêmes répondants, donc l'information totale doit
valoir celle d'un seul sondage.

> **Limite assumée.** Cette déflation traite correctement le niveau mais
> attribue à la *différence* entre deux hypothèses une variance $2hv$, alors que
> le partage des répondants la rend bien plus précise — or c'est ce différentiel
> qui identifie la géométrie. Une vraisemblance à erreurs corrélées par sondage
> a été construite et mesurée : elle améliore la redistribution sur le roster
> live mais ne converge pas sur les fits historiques. Écartée.

## 6. Lecture, scénarios, projection

Le fit rend les **tirages** $(\mu,\sigma,w)$, pas des moyennes. Pour un champ
arbitraire coché par un visiteur, $\pi$ s'obtient en poussant chaque tirage à
travers le softmax masqué — $O(\text{tirages}\times B)$, **aucune
ré-inférence**. C'est ce qui justifie l'architecture face à un modèle à slots
figés.

Projection jusqu'au scrutin : machinerie **partagée**
(`model/core/projection.py` + `model/core/opinion.py`), identique à tous les
modèles du dépôt.

L'inversion $\pi(w)=\tilde Y_p$ est bien posée — $\pi=\nabla\Phi$ avec
$\Phi(w)=\sum_b W_b\,\mathrm{logsumexp}_{j\in S_p}A_{j,b}$, de hessien SDP de
noyau exactement $\mathrm{span}(\mathbb 1)$ — ce qui fonde le corollaire de
portée du §0 : à champ unique, un $w$ reproduit exactement les parts observées
quelle que soit la géométrie.

## 7. Coût

Le temps vaut exactement $(\text{tirages}+\text{warmup})\times
\text{pas de leapfrog}\times\text{chaînes}$. Sur le roster live (78 nœuds, 13
candidats, 4 chaînes, 600+400) : **~90 s**, à 127 pas par itération. Améliorer
le conditionnement vaut donc plus que toute optimisation de code — et c'est
mesurable, le nombre de pas est un diagnostic à part entière.

`chain_method="sequential"` est ici à la fois le plus rapide et le plus sobre en
mémoire, contrairement à l'intuition.

## 8. Validation

Le critère est la **redistribution hors échantillon**. Beaucoup de sondages
posent plusieurs hypothèses au même échantillon, dont certaines imbriquées
($B = A$ privé d'un candidat) : même terrain, mêmes répondants, même jour, donc
l'écart entre les deux est de la redistribution quasi pure. On prédit $Y_B$ à
partir de $Y_A$, contre le proportionnel.

Sur 2022, 91 paires jamais vues du fit :

| | erreur moyenne | paires gagnées |
|---|---|---|
| proportionnel | 0,70 pt | — |
| spatial | 0,58 pt | 59 % |

Mais cet agrégat mélange des paires où le modèle est décisif et des paires où il
ne peut rien : le report y est déjà proportionnel, et le modèle n'a alors que du
bruit à ajouter. Ventilé par difficulté du proportionnel :

| quartile | n | proportionnel | spatial | paires gagnées |
|---|---|---|---|---|
| 0,10 – 0,44 | 23 | 0,34 pt | 0,39 pt | 35 % |
| 0,44 – 0,65 | 22 | 0,52 pt | 0,54 pt | 50 % |
| 0,65 – 0,81 | 23 | 0,70 pt | 0,67 pt | 52 % |
| **0,81 – 1,55** | 23 | **1,23 pt** | **0,70 pt** | **100 %** |

Sur les reports réellement non proportionnels, le modèle **gagne les 23 paires
sur 23** et divise l'erreur par 1,8 — gain +0,53 pt, IC95 bootstrap
[+0,40 ; +0,66]. La corrélation entre difficulté du proportionnel et gain du
spatial vaut +0,74 par paire.

**C'est cette lecture ventilée qu'il faut citer**, pas la moyenne : elle est plus
honnête (elle dit où le modèle ne sert à rien) et plus forte. Avec n = 23 sur le
quartile décisif, « 100 % » se lit [85 % ; 100 %] à 95 %.

Diagnostics d'échantillonnage sur les cinq coupures (roster arrêté à la coupure,
4 chaînes, 600 warmup + 400 tirages) :

| coupure | nœuds | R-hat | ESS | divergences |
|---|---|---|---|---|
| J-300 | 36 | 1,044 | 174 | 0 |
| J-240 | 56 | 1,033 | 123 | 0 |
| J-210 | 86 | 1,044 | 94 | 0 |
| J-180 | 151 | 1,027 | 161 | 0 |
| J-150 | 195 | 1,064 | 34 | 0 |

> **Prudence sur ces diagnostics.** Sur la plus grosse coupure, la variabilité
> de GRAINE domine : même configuration, l'ESS va de 3 à 112 selon la graine.
> Une comparaison de variantes sur une seule réalisation n'y mesure rien. La
> redistribution, elle, est reproductible à 0,002 pt entre graines — c'est donc
> elle qui doit arbitrer les choix de modélisation, pas R-hat.

La couverture des intervalles est au nominal en agrégé (48,6 / 79,0 / 88,2 pour
50 / 80 / 90) mais **erratique entre coupures** (82 % à 95 % d'IC90) — défaut
ouvert. À noter qu'elle ne discrimine pas la géométrie : elle est identique
entre un fit convergé et un fit qui ne l'est pas. Elle valide la chaîne
d'observation, pas le mécanisme spatial.

> **Ce qui manque.** Aucune validation hors échantillon sur les données 2026,
> celles que le produit affiche : la mesure de redistribution disponible sur ce
> roster est EN échantillon. Un backtest en sondages retenus (couper avant les
> derniers sondages, prédire les hypothèses suivantes) est faisable sans
> attendre le scrutin et reste à faire.

## 9. Ce que le modèle ne peut pas faire

- **Une seule dimension.** Un candidat recrutant sur tout le spectre pour une
  idée transversale ne peut être représenté que par un $\sigma$ large, donc
  *symétrique* — ce qui est faux. Une seconde dimension serait l'objet correct.
- **$\sigma$ des petits candidats** reste faiblement contraint : ils ne dominent
  nulle part, donc leur forme n'est vue qu'à travers des redistributions de
  faible amplitude. À ne pas afficher sans le dire.
- **Vraisemblance gaussienne, pas Beta** : pour un candidat à ~0,3 %,
  $\mathcal N(\pi,\pi(1-\pi)/n)$ met une masse non négligeable sous zéro. Peu
  visible sur un champ resserré, sensible sur un champ à 20 candidats.
- **Champs jamais sondés** : la validation porte sur des paires d'hypothèses
  réellement observées. Une combinaison inédite repose sur l'extrapolation
  géométrique — c'est le pari du modèle, pas une mesure.
- **L'ordre et les ancres sont des choix éditoriaux.** Ils sont lâches et le
  postérieur peut les contredire, mais ils sont publiés avec le reste.
