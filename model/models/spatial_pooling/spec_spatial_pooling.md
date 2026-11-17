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

## 10. Tracker par candidat (demande utilisateur, 2026-11-15)

**Objet.** Le fit estime déjà un chemin $w_i(t)$ (§4) pour chaque candidat
« dynamique » (> 5 % de part moyenne) sur la grille journalière des dates de
sondage $t_1 < \dots < t_T$ (la dernière colonne étant `as_of`). $w$ brut est
relatif (défini à une translation près, et sans unité lisible) : on publie la
**part** qu'il implique dans un champ de référence $M$ fixé,

$$\pi_i(t \mid M) = \sum_b W_b\,\frac{\mathbb 1_{i\in M}\,e^{A_i(v_b, t)}}
{\sum_{j\in M} e^{A_j(v_b, t)}},\qquad A_i(v,t) = w_i(t) + \log\varphi_i(v),$$

calculée tirage par tirage sur $(\mu^{(s)}, \sigma^{(s)}, w^{(s)}(t))$ — la
même fonction `spatial_shares` que le nowcast, appliquée à chaque date. Résumé
publié : moyenne et IC90 de $\pi_i(t\mid M)$ par date. Les candidats
statiques ($w_i = m_i$) n'ont pas de chemin propre : leur part ne bouge que par
la renormalisation sur les autres — à signaler, ou à ne pas tracer.

**Champ de référence.** Pour comparer avec les `gp-pooling` multi-listes du
site, $M$ = chacune des listes que le site affiche (règle de la session site :
listes les plus testées parmi les ~10 derniers sondages), reprise telle quelle
à chaque coupure du backtest. Le spatial a l'avantage de donner la trajectoire
de N'IMPORTE QUELLE liste sans ré-inférence ; le comparateur, lui, ajuste
chaque liste sur son seul historique.

**Lissage hérité, à mesurer (pas à supposer).** Le chemin OU a
$\tau \approx 262$ j FIXÉ (banque commune) et une base KL tronquée à 99 % de
variance (3-4 composantes) : il est très lisse. Une montée rapide (Zemmour
sept.–nov. 2021, Mélenchon mars–avril 2017) risque d'être suivie avec retard ;
le diagnostic 2027 voit déjà un retard de ~1,5 pt sur Mélenchon–Le Pen en duel
(second_tour §8.1). C'est précisément ce que le backtest doit chiffrer.

**Backtest de suivi de tendance (2017, 2022).** Coupures $c_1..c_K$ (~10 par
élection, toutes les 2-3 semaines sur les 6 derniers mois). À chaque coupure :
fit sur les seuls sondages $\le c_k$ (un fit unique en fin de campagne
lisserait avec le recul : ne vaut pas validation), puis
- **erreur de nowcast** : $\hat\pi_i(c_k\mid M)$ contre la moyenne des
  sondages de la liste $M$ dans $(c_k, c_k + 7\,j]$ (hors échantillon) ;
- **couverture** de l'IC90 (avec le bruit d'échantillonnage de cette moyenne) ;
- **retard de détection** : pour une montée identifiée sur les sondages bruts
  (première date où la moyenne mobile 14 j dépasse de 3 pt sa valeur 30 j
  avant), délai jusqu'à ce que $\hat\pi_i(c_k)$ dépasse le même seuil.
Même grille et mêmes listes pour `gp-pooling`. Fits d'abord allégés
(2 chaînes), puis complets sur les coupures informatives. Ne touche pas au site.

**Implémentation** : déterministe `w_path` (N, T) dans le modèle (aucun effet
sur l'inférence), `w_path_draws` (S, N, T) et `w_dates` dans le fit,
`trajectoire(fit, mask)` → (dates, tirages (S, T, |M|)).

**Premier run 2027 (sondages au 2026-11-13, T1 seul, 4 chaînes retenues,
R-hat 1,027)** — champ le plus testé (Arthaud, Mélenchon, Roussel, Tondelier,
Glucksmann, Philippe, Retailleau, Dupont-Aignan, Le Pen, Zemmour), comparé
aux 26 sondages de ce champ exact (30/04 → 05/11), dans l'échantillon :

| candidat | part au 13/11 [IC90] | résidu moyen | résidu 30 j | pente 90 j brut / tracker (pt/mois) |
|---|---|---|---|---|
| Le Pen | 35,4 [34,5 ; 36,3] | −0,4 | −0,1 | +0,9 / +0,5 |
| Mélenchon | 18,1 [17,4 ; 18,9] | −0,1 | −0,4 | +0,5 / +0,9 |
| Philippe | 16,1 [15,1 ; 17,1] | −0,4 | **−1,0** | −1,6 / −1,2 |
| Glucksmann | 12,3 [11,5 ; 13,1] | 0,0 | +0,3 | +1,0 / +0,8 |
| Retailleau | 7,0 [6,3 ; 7,8] | +0,1 | +0,1 | −0,2 / −0,2 |
| Zemmour | 4,1 [3,7 ; 4,5] | +0,2 | −0,1 | +0,1 / 0,0 |

Lecture : niveau bien représenté (|résidu| ≤ 0,4 pt), tendances de signe
juste mais **amorties d'environ un quart à moitié** sur les mouvements nets
(Le Pen, Philippe : le tracker suit avec retard ; Philippe est à −1,0 pt sur
30 j) — le lissage OU attendu. Dans l'échantillon : le backtest hors
échantillon (coupures 2017/2022) reste à faire pour chiffrer le retard de
détection. Résultat : `model/backtest/results/tracker_2027_2026-11-13.json`.

**Protocole du backtest de suivi, étape 1 (spatial seul, 2022), fixé avant
résultat.** Coupures $c_k = T1 - h$, $h \in \{180, 159, …, 12\}$ j (pas de
21 j, 9 coupures). À chaque coupure : cadre historique à toutes les
hypothèses (second_tour §8.3), fenêtre 180 j, fit T1 seul (comme la
production), 4 processus d'une chaîne + fusion avec sélection par
log-densité (second_tour §8.8/§8.11). Listes : `model.listes.classement`
appliqué aux sondages ≤ $c_k$ (même règle que le site, `scenario_id` =
notice × hypothèse) ; une liste dont un candidat est hors roster est écartée
et comptée. Cible, par liste et candidat : moyenne des sondages de la liste
EXACTE terminés dans $(c_k, c_k + 7\,j]$ ; métriques : erreur absolue
moyenne (pt), couverture de l'IC90 prédictif (bruit binomial de la moyenne
de $n$ sondages ajouté), et biais signé par candidat sur les coupures.
Étape 2 : `gp-pooling` (GPListe) sur les mêmes listes et coupures ; étape 3 :
retard de détection des montées (Zemmour automne 2021).
*Amendement avant tout résultat (disponibilité des cibles, pas erreurs)* : la
fenêtre cible passe à 14 j — à 7 j, 0 à 2 sondages par liste avant mars 2022 ;
h = 180 retiré (aucun sondage cible) : 8 coupures, 13 listes avec cible.

**Résultats étapes 1-2 (2022, 8 coupures de T1−159 à T1−12, listes du site,
cible = sondages de la liste exacte des 14 jours suivants).** 171 couples
(liste, candidat) appariés entre les deux modèles.

| | MAE tous | MAE candidats ≥ 5 % | couverture IC90 |
|---|---|---|---|
| spatial (T1 seul) | 0,78 pt | 1,21 pt | **0,73** |
| gp-pooling (GPListe) | **0,62 pt** | **0,94 pt** | 0,90 |

Écart moyen spatial − gp = +0,16 pt par couple, Wilcoxon p = 0,0001 ; gp
meilleur à 6 coupures sur 7 (spatial meilleur à T1−54 : 0,58 contre 0,92).
Biais signés (candidats ≥ 5 %) de même sens pour les deux modèles
(Le Pen −1,3 / −1,1 ; Zemmour +1,1 / +0,8 ; Macron −0,5 / −0,3) : la cible
bouge après la coupure, et les deux suivent en retard — le spatial un peu
plus. Convergence : R-hat ≤ 1,05 à 7 coupures (T1−12 : 2,5, 3 chaînes
retenues).

**Lecture.** Sur une liste DIRECTEMENT sondée, le modèle propre à la liste
gagne : il ne s'ajuste qu'à elle, quand le spatial doit concilier 50 à 90
champs par une géométrie commune et un chemin OU très lisse. L'intérêt du
spatial reste les listes non sondées (il n'a pas de concurrent là) ; comme
tracker d'une liste sondée, il ne devrait PAS remplacer gp-pooling. Son
IC90 est trop étroit à 14 j (0,73) : le nowcast ne porte aucune dérive
au-delà de `as_of`, alors que la cible s'étale sur 14 jours — à corriger
(prolonger le chemin OU à h = 7 j, `projeter_w`) avant toute publication
d'un tracker spatial. Étape 3 (retard de détection) à faire.
Résultats : `model/backtest/results/suivi_2022_h*.json`, `suivi_gp_2022_h*.json`.

**Correctif d'IC (avant re-mesure).** La cible couvrant $(c, c+14]$, le
nowcast spatial est prolongé de $h = 7$ j par le MÊME processus OU que la
projection du modèle (`projeter_w` : retour vers $m_i$, diffusion
$\sigma_w^2(1-r^2) + (1-r)^2\sigma_w^2$, $r = e^{-h/\tau}$). gp-pooling,
lui, est lu à $c$ (son postérieur intègre déjà l'incertitude latente) ; les
deux restent comparés sur les mêmes cibles. Sorties `suivi_proj_2022_h*.json`
(les résultats sans prolongement restent dans `suivi_2022_h*.json`).

**Re-mesure avec prolongement (re-fusion, 171 couples)** : MAE spatial 0,77
(sans prolongement 0,78 ; gp 0,62) ; couverture IC90 **0,82** (0,73 ; gp
0,90) ; candidats ≥ 5 % : MAE 1,18 (gp 0,94), couverture 0,90 (gp 0,95). Le
prolongement règle l'essentiel du défaut de couverture ; l'écart de précision
avec gp-pooling sur les listes sondées demeure.

**Étape 3, protocole (fixé avant résultat) : détection de la montée Zemmour
2021.** Coupures hebdomadaires $c$ = 07/09 → 26/10/2021 ($h$ = 215 … 166),
fit T1 seul comme aux étapes 1-2. Les listes du site n'incluent Zemmour qu'à
partir de fin septembre, et chaque liste le contenant n'a que ≤ 8 sondages
(dès le 13-29/09) : on fixe donc une liste de référence $R$ = la plus
sondée contenant Zemmour sur sept.-nov. 2021 (Arthaud, Poutou, Mélenchon,
Roussel, Jadot, Montebourg, Hidalgo, Macron, Lassalle, Bertrand,
Dupont-Aignan, Le Pen, Zemmour). Signal brut : moyenne mobile 14 j de
Zemmour sur TOUS les sondages qui le testent. Modèles : part de Zemmour
dans $R$ au jour $c$ (spatial : n'importe quelle liste, sans prolongement ;
gp-pooling : GPListe sur $R$, seulement si ≥ 3 sondages de $R$ avant $c$ —
sinon « indisponible », ce qui EST un résultat). Détection : première
coupure où l'estimation dépasse de 3 pt sa valeur de 28 j plus tôt ; même
règle sur le brut ; retard = différence (résolution 7 j). On rapporte aussi
l'écart estimation − brut à chaque coupure.

**Résultat étape 3 (Zemmour 2021, liste R, coupures hebdomadaires).**

| coupure | brut 14 j (tous sondages) | spatial [IC90] | gp-pooling (n sondages de R) |
|---|---|---|---|
| 07/09 | 7,4 | 7,2 [6,1 ; 8,4] | indisponible (0) |
| 14/09 | 7,8 | 7,9 [6,9 ; 9,0] | indisponible (0) |
| 21/09 | 9,7 | 9,9 [8,9 ; 10,9] | indisponible (0) |
| 28/09 | 11,1 | 12,2 [11,1 ; 13,4] | indisponible (0) |
| 05/10 | 13,4 | 15,8 [14,5 ; 17,2] | indisponible (2) |
| 12/10 | 15,1 | 15,2 [14,5 ; 16,1] | 14,0 (3) |
| 19/10 | 15,8 | 16,7 [15,8 ; 17,6] | 15,9 (6) |
| 26/10 | 16,2 | 16,7 [15,7 ; 17,7] | 15,8 (6) |

Détection (+3 pt en 28 j) : brut le 05/10, **spatial le 05/10 (retard nul à
la résolution hebdomadaire)**, gp-pooling : jamais sur la fenêtre (série
commencée le 12/10, après la montée). Le spatial suit la montée sans
l'amortir — il la devance même la semaine du 28/09-05/10, la moyenne mobile
brute étant elle-même en retard — parce qu'il voit Zemmour dans TOUTES les
listes qui le testent, là où le modèle par liste attend que SA liste
accumule des sondages. C'est l'usage où le spatial a un avantage net : un
candidat nouveau, testé dans des champs dispersés. À rapprocher de
l'amortissement mesuré sur 2027 (mouvements lents de ±1 pt/mois) : le
lissage OU pèse sur les petites pentes, pas sur une montée de 9 pt en 6
semaines portée par de nombreux sondages. Résultat :
`model/backtest/results/detection_zemmour_2022.json`.

**Révision après la revue Opus du 2026-11-16 (rétractations).**
- *« gp-pooling bat le spatial »* : seul l'écart de **MAE** tient (bootstrap
  par unité (coupure, liste) : +0,06 à +0,22 pt, signe 11/13). La
  **couverture** ne départage rien : score d'intervalle α = 0,1 identique
  (4,32 pt pour les deux) ; gp couvre mieux parce que son IC est plus large
  (3,53 contre 2,84 pt). Le Wilcoxon sur 171 couples supposait
  l'indépendance : il n'y a que 13 unités.
- *« Le décalage de dates rend gp-pooling exactement valide sur 2022 »* :
  FAUX. Les horizons sont exacts, mais les paramètres de gp-pooling
  (`bank_opinion.json`, REML sur 2017/2022) et `FACTEUR_DIFFUSION = 3` ont été
  calés sur ces mêmes élections ⇒ avantage dans l'échantillon pour gp, que le
  spatial n'a pas. Et σ_h est l'**effet institut**, pas une fonction de
  l'horizon (erreur de lecture corrigée).
- *Lecture à des dates différentes* : gp lu à c, spatial prolongé à c + 7 ; et
  aucun des deux IC ne porte σ_h ni σ_p des 1-2 sondages cibles. Comparaison
  à refaire à date égale avec ces termes.
- *« Détection Zemmour sans retard »* : NON ÉTABLI. La série commençait le
  07/09 avec un recul de 28 j : le 05/10 était la **première date où une
  détection était possible**, pour toutes les séries. La montée commence fin
  août. « Spatial devance le brut » compare une part dans R à une moyenne sur
  des champs plus larges. « gp indisponible » dépend du choix de R.

**Étape 3 refaite (revue 16/11, point 4 ; protocole fixé avant résultat).**
Coupures hebdomadaires dès $h$ = 243 (10/08/2021) jusqu'à 166 : une détection
est possible dès le 07/09, avant la montée. Même seuil (+3 pt en 28 j). Séries :
(i) brut 14 j sur tous les sondages qui testent Zemmour ; (ii) spatial, part
dans $R$ ; (iii) gp-pooling sur $R$ ; et, pour comparer à base égale,
$L_c$ = liste contenant Zemmour la mieux fournie (notices distinctes) à la
coupure $c$ : (iv) brut 14 j sur la liste EXACTE $L_c$, (v) spatial dans
$L_c$, (vi) gp-pooling sur $L_c$ (si ≥ 3 sondages). $L_c$ peut changer d'une
coupure à l'autre : la règle de détection appliquée aux séries $L_c$ compare
alors deux listes, ce qu'on signale. Résultat attendu : date de détection de
chaque série ; un « retard » n'est lu que s'il dépasse la résolution (7 j).
Sortie : `results/detection_zemmour_2022_v2.json` (l'ancien fichier est
conservé).

**Résultat (2026-11-16, `detection_zemmour_2022_v2.json`).** En août 2021
Zemmour n'est testé que par 1 sondage : il est **hors du roster spatial**
(MIN_POLLS) jusqu'au 31/08 compris ; le brut existe dès le 24/08 (7,0 %).

| coupure | brut 14 j | spatial R [IC90] | brut L_c | spatial L_c | gp L_c (n) |
|---|---|---|---|---|---|
| 24/08, 31/08 | 7,0 | hors roster | — | — | (1) |
| 07/09 | 7,4 | 7,2 [6,1 ; 8,4] | 7,0 | 7,1 | (1) |
| 14/09 | 7,8 | 7,9 | 8,5 | 7,6 | (2) |
| 21/09 | 9,7 | 9,9 | 9,3 | 9,5 | (2) |
| 28/09 | 11,1 | 12,2 | 11,8 | 11,7 | 10,5 (3) |
| 05/10 | 13,4 | 15,8 [14,5 ; 17,2] | 14,7 | 15,3 | 12,6 (4) |
| 12/10 | 15,1 | 15,2 | 15,4 | 14,7 | 14,2 (6) |
| 19/10 | 15,8 | 16,7 | 14,7 | 16,1 | 14,2 (6) |
| 26/10 | 16,2 | 16,7 | — | 16,2 | 14,1 (6) |

Détections (+3 pt en 28 j) : **brut le 28/09** ; spatial R, spatial L_c et
brut L_c le 05/10 ; gp L_c le 26/10 ; gp R jamais. Lecture :
- Le retard du spatial sur le brut (une coupure, 7 j) vient **entièrement de
  l'entrée au roster** : sans valeur au 31/08, la règle ne peut se prononcer
  au 28/09 ; dès qu'il est dans le roster, le spatial suit (12,2 au 28/09,
  au-dessus du brut). C'est un coût du seuil MIN_POLLS = 5, pas du lissage OU.
- À base égale (liste L_c), brut et spatial détectent la même semaine ; le
  spatial dépasse le brut au pic de la montée (05/10 : 15,3 contre 14,7 ;
  R : 15,8 contre 13,4 tous champs).
- gp-pooling sur la liste la mieux fournie détecte **3 semaines après** et
  reste 1,5-2 pt sous le brut de la même liste en octobre : la liste n'a que
  3-6 sondages et le prior OU tire vers l'ancien niveau. C'est l'avantage net
  du spatial sur un candidat nouveau, cette fois à base égale.
- Résolution : 7 j ; une seule montée, une élection ⇒ illustration, pas une
  mesure de retard généralisable.

**Tâche A de la revue du 16/11 : gp-pooling recalibré HORS échantillon.**
REML sur les seuls blocs 2017 (4 blocs, 27 sondages — instable) : σ_h 0,119,
σ² 0,054, τ 38 j, σ_p 0 (banque 2017+2022 : 0,126 / 0,116 / 262 j / 0,039) ;
facteur de diffusion ×3 (comme en production) et ×1 en sensibilité ; gp lu à
c + 7 j comme le spatial prolongé. 13 unités (coupure, liste), bootstrap par
unité, Δ = spatial − gp :

| gp | MAE spatial / gp | Δ MAE [IC90] | score d'intervalle α=0,1 spatial / gp | Δ IS [IC90] | couverture | largeur IC90 gp |
|---|---|---|---|---|---|---|
| banque (dans l'éch.), lu à c | 0,77 / 0,62 | [+0,06 ; +0,23] | 4,32 / 4,32 | [−0,65 ; +0,59] | 0,82 / 0,90 | 3,5 pt |
| banque, c + 7 | 0,77 / 0,62 | [+0,05 ; +0,22] | 4,32 / 4,55 | [−0,89 ; +0,37] | 0,82 / 0,93 | 4,0 pt |
| **REML 2017, ×3, c + 7** | 0,77 / **0,60** | [+0,07 ; +0,25] | **4,32** / 5,83 | **[−2,24 ; −0,77]** | 0,82 / 0,95 | 5,5 pt |
| REML 2017, ×1, c + 7 | 0,77 / 0,61 | [+0,07 ; +0,23] | **4,32** / 5,41 | [−1,73 ; −0,43] | 0,82 / 0,94 | 5,1 pt |

(Sans la coupure T1−12, R-hat 2,5 : mêmes conclusions.)

**Lecture.** (1) **Les centres de gp-pooling sont meilleurs, et ce n'est pas
un artefact de calibration** : la MAE reste ≈ 0,60 hors échantillon (seuil
de rupture fixé : 0,70). (2) **Ses bons intervalles, eux, en étaient un** :
recalibré sans 2022, gp sur-couvre (0,94-0,95) avec des IC de 5-5,5 pt, et le
spatial devient nettement meilleur au score d'intervalle. (3) Le spatial a
donc des IC plus fins mais encore un peu étroits (0,82 pour 0,90), gp des
centres plus justes. C'est exactement le cas d'usage de l'idée B (combiner :
centre de gp, incertitude informée par le spatial, ou stacking par unité).
Réserve : le REML 2017 repose sur 27 sondages (τ = 38 j, σ_p = 0), il est
lui-même fragile. Résultats : `suivi_gp_{bank_c7,r17f3_c7,r17f1_c7}_2022_h*.json`,
paramètres : `bank_opinion_reml_2017_seul.json`.

**Révision après la revue Opus du 2026-11-17.** (1) Détection Zemmour
refaite : brut L_c et spatial L_c n'ont de valeur qu'à partir du 07/09, donc le
05/10 était la première détection possible pour les deux — « à base égale
spatial = brut » est imposé par les données ; de même gp L_c n'existe qu'à
partir du 28/09, donc « gp 3 semaines plus tard » est aussi imposé. Seul
l'**écart de niveau** tient (05/10 : gp 12,6 contre brut L_c 14,7). Avec un recul
de 21 j, spatial L_c et brut L_c détectent tous deux le 28/09. (2) Combinaison
C1 : choisie a posteriori parmi 3, sur 13 unités corrélées (IC trop étroit),
couverture 0,87 ; « gp pour le niveau, spatial pour les IC » reste un choix de
présentation pour l'utilisateur.

**Pistes de l'utilisateur du 2026-11-17 (B, C).**

*(B) Positions a priori externes, 2017/2022.* Source cardinale indépendante
visée : CHES (lrgen 0-10 par parti). **Inaccessible** : `www.chesdata.eu` hors
des domaines réseau autorisés de l'environnement (DNS refusé). Repli prévu
(ORDER_GROUPS transformé en positions) mesuré sans fit : avec 25 candidats et
l'écart minimal d'identifiabilité de 0,25 logit, ancres « par famille » et
« par rang » diffèrent de ≤ 0,011 (2022) — la contrainte d'écart impose déjà un
espacement quasi uniforme ; le repli n'apporte rien. Fenêtre de 90 j : déjà
écartée (§8.7 de spec_second_tour : 14 champs contre 45). ⇒ **(B) bloquée par
la source** : il faut des positions cardinales (CHES ou équivalent fourni dans
le dépôt).

*(C) σ_w par candidat (maths avant code).* Au lieu d'un σ_w commun :
$\sigma_{w,i} = \sigma_w\, e^{s_\sigma z_i}$, $z_i\sim\mathcal N(0,1)$,
$s_\sigma\sim$ HalfNormal(0,5), pour les candidats dynamiques ; le chemin
devient $w_i(t) = m_i + \sigma_{w,i}\cdot(\text{OU KL})_i$ et la projection
(`projeter_w`) utilise $\sigma_{w,i}$. Hypothèse : un σ_w commun amortit les
candidats qui bougent vraiment (Zemmour 2021, Philippe 2027) au profit des
stables. **Mesure** : `suivi.py --sigma-w-candidat`, 2022, les 8 coupures de
l'étape 1 (h = 159 … 12), mêmes listes et cibles ; adopté seulement si MAE ET
couverture IC90 (et score d'intervalle α = 0,1) battent le σ_w commun, IC90
bootstrap par unité (coupure, liste). Sinon : « rôles spatial / gp assumés ».

**Idée B : combiner spatial et gp-pooling (protocole fixé avant résultat,
2026-11-17, 0 fit).** Données : les 13 unités (coupure, liste) de la tâche A,
spatial prolongé à c + 7 (`suivi_proj_2022_h*.json`) et gp recalibré hors
échantillon (REML 2017, ×3, c + 7 : `suivi_gp_r17f3_c7_2022_h*.json`), appariés
par (coupure, liste, candidat). Chaque prédictive est résumée par une normale
$\mathcal N(m, s^2)$, $s$ = largeur IC90 / 3,29 (les tirages ne sont pas
sauvés). Combinaisons :
- **C1 « centre gp, largeur spatial »** : $\mathcal N(m_{gp}, s_{sp}^2)$ ;
- **C2 mélange linéaire** $w\,\mathcal N_{gp} + (1-w)\,\mathcal N_{sp}$, $w$
  choisi en **leave-one-coupure-out** (grille 0-1 par 0,05, log-score de la
  cible sous le mélange), appliqué à la coupure retirée ;
- **C3 moyenne pondérée des centres** $w\,m_{gp} + (1-w)\,m_{sp}$ avec
  $s = \max(s_{sp}, s_{gp})$ ? Non : $s$ ajusté en leave-one-coupure-out
  par un facteur $k$ sur $s_{sp}$ (grille 0,8-2), $w$ comme C2.
Métriques : MAE (pt), score d'intervalle α = 0,1 (pt), couverture IC90 ;
Δ contre gp seul et contre spatial seul, IC90 bootstrap par unité. Une
combinaison est **recommandée** si elle bat les deux modèles seuls au score
d'intervalle avec un IC90 de Δ entièrement < 0, sans MAE pire que gp de plus
de 0,05 pt. Recommandation seulement : aucune modification du site.

**Résultat B (2026-11-17, 171 couples, 13 unités ; `combinaison_spatial_gp_2022.json`)** :

| méthode | MAE (pt) | score d'intervalle α = 0,1 (pt) | couverture IC90 |
|---|---|---|---|
| gp seul | 0,60 | 5,83 | 0,94 |
| spatial seul | 0,77 | 4,30 | 0,82 |
| **C1 centre gp, largeur spatial** | **0,60** | **3,84** | 0,87 |
| C2 mélange (w_gp 0,30-0,45 hors coupure) | 0,67 | 4,68 | 0,95 |
| C3 centres pondérés (w_gp 0,65-0,90, k 1,1-1,4) | 0,62 | 4,36 | 0,91 |

Δ score d'intervalle de C1 : contre gp −1,98 [−2,52 ; −1,42], contre spatial
−0,44 [−1,00 ; +0,14]. **Aucune combinaison ne passe la règle fixée** (IC90
de Δ entièrement < 0 contre les DEUX modèles) : C1 bat nettement gp et prend
ses centres, mais son avantage sur le spatial seul reste dans le bruit (13
unités, une campagne). C1 est la meilleure candidate : à revoir sur 2017
quand les dates 2017 seront disponibles (session données). Recommandation
inchangée (synthèse ci-dessous) ; aucune modification du site.

**Synthèse pour l'utilisateur (2026-11-16) : rôles du spatial et de
gp-pooling dans le suivi du premier tour.** Tout ce qui suit est mesuré sur
2022 (une seule campagne : indications, pas des lois) et 2027.

| question | meilleur | chiffre | référence |
|---|---|---|---|
| niveau d'une liste bien sondée, à 7 j | **gp-pooling** | MAE 0,60 contre 0,77 pt (Δ [+0,07 ; +0,25]), hors échantillon | tâche A |
| incertitude (score d'intervalle α = 0,1) | **spatial** | 4,3 contre 5,4-5,8 pt (Δ [−2,2 ; −0,8]) ; gp sur-couvre (0,95), le spatial sous-couvre un peu (0,82) | tâche A |
| candidat nouveau, testé dans des champs dispersés | **spatial** | Zemmour 2021 : à base égale, spatial = brut (05/10), gp 3 semaines plus tard et 1,5-2 pt sous le brut | étape 3 refaite |
| liste peu ou pas sondée (champ coché sur le site) | **spatial seul** | gp n'existe qu'avec ≥ 3 sondages de la liste exacte | — |
| tendances lentes (±1 pt/mois, 2027) | gp (le spatial amortit) | spatial : pentes amorties d'1/4 à 1/2 (Philippe −1,0 pt sur 30 j) | run 2027 |
| duels de second tour, P(élu) | **spatial seul** | gp ne modélise pas le T2 | spec_second_tour |

**Limites connues du spatial** : seuil d'entrée au roster (5 sondages) qui
retarde un candidat nouveau d'une semaine ou plus ; lissage OU (τ ≈ 262 j)
qui amortit les petites pentes ; IC un peu étroits (0,82 pour 0,90).
**Limites de gp-pooling** : une courbe par liste exacte, donc rien tant que
la liste n'a pas 3 sondages, et retard quand un candidat monte dans des
listes différentes ; IC trop larges une fois recalibré hors échantillon.

**Recommandation** (à confirmer par l'idée B, combinaison évaluée en
leave-one-coupure-out, en file) : afficher gp-pooling pour le **niveau** des
listes bien sondées, le spatial pour les listes peu sondées, les candidats
nouveaux, les **intervalles** et tout le second tour. Aucune modification du
site n'est faite ici ; c'est un choix de présentation à trancher par
l'utilisateur une fois B mesuré.

