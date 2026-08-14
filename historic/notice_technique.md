# Notice technique — `spatial-pooling`

Le modèle, et rien d'autre. Les mesures qui ont motivé ces choix, les pistes
essayées puis abandonnées et l'historique des corrections sont dans
`spec_spatial_pooling.md` ; les backtests sont dans `notebooks/04*`.

Même genre que `../gp_pooling/spec_gp_pooling.md`.

---

## 1. Notations

$N$ candidats, indice $i$. $P$ nœuds d'observation, indice $p$ : un nœud = une
**hypothèse** d'un sondage (un sondage à $h$ hypothèses produit $h$ nœuds).
$S_p \subseteq \{1..N\}$ est le champ testé au nœud $p$, de cardinal $K_p$,
encodé par le masque $M_{i,p}\in\{0,1\}$.

L'espace politique est l'intervalle $[0{,}01\,;0{,}99]$, discrétisé en $B=25$
nœuds de quadrature $V_b$ de poids $W_b$ (**Gauss-Legendre**, $\sum_b W_b = 1$).
Ces poids représentent une densité électorale **uniforme** ; les nœuds ne sont
pas équidistants parce que la grille est une quadrature d'intégrale, pas un
échantillonnage.

> Une densité électorale non uniforme ne serait pas identifiable séparément des
> positions : $\int g(v)f(v)\,dv = \int g(F^{-1}(u))\,du$, donc un électorat
> déformé équivaut à un électorat uniforme à positions transportées par $F$.
> L'axe étant latent, cette liberté n'a pas de contenu.

## 2. Géométrie — $\mu_i$, $\sigma_i$

Les candidats sont répartis en $G$ **groupes** ordonnés de gauche à droite
(`ORDER_GROUPS`). L'ordre entre groupes est une contrainte dure ; à l'intérieur
d'un groupe, rien n'est imposé.

$$\text{base}\sim\mathcal N(0,1{,}2),\qquad \text{gap}_k\sim\text{LogNormal}(-0{,}5\,;0{,}7)$$

$$c_k = \sum_{j\le k}\text{gap}_j,\qquad
\text{raw}_k = \text{base} + c_k - \overline{c},\qquad
\text{slot}_k = \mathrm{sigmoid}(\text{raw}_k)$$

La sigmoïde étant monotone, l'ordre est préservé **par construction** — aucune
transformation `Ordered` n'est nécessaire. Le **centrage** $c_k-\overline c$ est
essentiel : sans lui, $\text{raw}$ ne fait que croître depuis 0 et la
configuration a priori se tasse dans la moitié droite de la grille (avec $G=9$,
les quatre derniers groupes démarrent entre 0,974 et 0,992, où la dérivée de la
sigmoïde vaut $\sim10^{-3}$).

Delta individuel, non centré :

$$\mu_i = \text{slot}_{g(i)} + z^\mu_i\,s_\mu,\qquad
\sigma_i = \sigma_{\text{slot},g(i)}\exp(z^\sigma_i s_\sigma),\qquad z_i\sim\mathcal N(0,1)$$

$$\sigma_{\text{slot}}\sim\text{LogNormal}(\log 0{,}15\,;0{,}4),\qquad
s_\mu\sim\text{HalfNormal}(0{,}06),\quad s_\sigma\sim\text{HalfNormal}(0{,}15)$$

> $\sigma$ **n'est identifié par aucune donnée** : sa contraction
> prior→postérieur est ≈ 0 sur 2017 comme sur 2027, et sous un prior
> délibérément large le postérieur reproduit le prior. C'est donc un choix de
> modélisation. Il a été arrêté sur les deux seuls critères mesurables —
> qualité de redistribution et stabilité d'échantillonnage — qui désignent
> tous deux cette valeur (spec §12.12, §12.13). Attention : $\sigma$ décrit le
> NOYAU avant compétition, pas le profil de recrutement ; un $\sigma$ large ne
> signifie pas qu'un candidat attire loin de sa position, le softmax écrasant
> tout face à un voisin mieux placé.

## 3. Attractivité et parts de vote

$$D_{i,b} = -\frac{(V_b-\mu_i)^2}{2\sigma_i^2},\qquad
A_{i,b} = w_i + D_{i,b} - \log\Big(\sum_{b'} W_{b'}e^{D_{i,b'}}\Big)$$

Le troisième terme **normalise le noyau** de chaque candidat à masse 1 sur la
grille. Sans lui, la part d'un candidat non dominant vaut
$\propto e^{w_i}\sigma_i\kappa(\mu_i,\sigma_i)$ (avec $\kappa$ la fraction du
noyau tombant dans l'intervalle) : seule la combinaison $w_i+\log\sigma_i$ est
alors identifiée, et $\sigma$ flotte sur une crête. Normalisé, la part vaut
$\propto e^{w_i}$ seul : $w$ porte le niveau, $\sigma$ ne porte plus que la
forme — et n'est plus contraint que par la redistribution observée. C'est une
reparamétrisation exacte (une constante par candidat avant un softmax).

Softmax **masqué** sur la grille, puis agrégation :

$$P_{i,b,p} = \frac{M_{i,p}\,e^{A_{i,b}}}{\sum_j M_{j,p}\,e^{A_{j,b}}},\qquad
\pi_{i,p} = \sum_b P_{i,b,p}\,W_b$$

Le masque est **multiplicatif**, pas un $-\infty$ dans l'exponentielle : plus
stable, et entièrement vectorisé sur $(P,N,B)$ sans `scan`.

> $\pi_{\cdot,p}$ somme à 1 sur $S_p$ par construction. Les observations doivent
> donc être **renormalisées sur le champ modélisé** (§4), faute de quoi la
> vraisemblance est structurellement insatisfiable dès que le roster ne couvre
> pas tout le bulletin.

**Jauge.** $A$ est invariant par $w_i \to w_i + c$ : $w$ n'a pas de niveau
absolu, exactement comme le CLR. Sans effet sur $\pi$ ; à fixer avant tout
affichage de $w$.

## 4. Observations

$\tilde Y_p$ = intentions rapportées, restreintes au roster puis **renormalisées**
sur $S_p$ ; $n_p$ = échantillon $\times$ la masse conservée par cette
restriction. La restriction d'un multinomial à un sous-ensemble étant un
multinomial, c'est la loi exacte, pas un ajustement.

$$\tilde Y_{i,p} \sim \mathcal N\big(\pi_{i,p},\ v_{i,p}\big),\qquad
v_{i,p} = \frac{\pi_{i,p}(1-\pi_{i,p})}{n_p} + \tau^2_{\mathrm{inst}(p)}$$

pour $i \in S_p$, les nœuds étant traités comme **indépendants**.
$\tau_{\mathrm{inst}}$ est la variance d'excès par institut (banque
`bank_excess.json`, calibrée sur des paires de sondages de champ identique).

$n_p$ est en outre **déflaté par le nombre d'hypothèses** du sondage : les $h$
hypothèses sont posées aux mêmes répondants, donc l'information totale doit
valoir celle d'un seul sondage, $h\times\tfrac1{hv}=\tfrac1v$, et non $h$ fois
plus.

> **Limite assumée.** Cette déflation traite correctement le NIVEAU mais
> attribue à la *différence* entre deux hypothèses une variance $2hv$, alors que
> le partage des répondants la rend bien plus précise — or c'est ce différentiel
> qui identifie $\mu$ et $\sigma$ (une hypothèse isolée s'explique par $w$
> seul ; c'est le retrait d'un candidat qui révèle où vont ses électeurs). Une
> vraisemblance à erreurs corrélées par sondage a été construite et mesurée
> (spec §12.7, §12.11) : elle améliore la redistribution de 3 % sur 2027 mais
> **ne converge pas** sur les fits historiques (R-hat 2,35 sur 2022 J-120, même
> à corrélation fixée) — retirer la déflation multiplie l'information par 7,6
> sur 3× plus de nœuds. Écartée pour la production ; code disponible
> (`weighted_loglik_blocked`, `blocked=True`).

**Pondération de récence.** Chaque sondage contribue avec un poids
$\kappa_\nu = 2^{-\Delta t_\nu/\text{half\_life}}$, $\text{half\_life}=15$ j.
C'est une vraisemblance **de puissance**, pas une densité normalisée :
`half_life` n'est donc pas échantillonnable et est calibré par backtest.
Cette températion ne sert plus qu'à rendre tenable l'hypothèse d'un $w$
**statique** pendant l'estimation de la géométrie ; l'incertitude temporelle de
$w$ est traitée en §5, par un mécanisme qui, lui, a un plancher.

## 5. Dynamique de $w$ — inversion locale exacte + Ornstein-Uhlenbeck

> **Ce §5 décrit ce que `fit_spatial_pooling` exécute aujourd'hui, pas la cible.**
> La lecture en deux temps ne peut pas propager l'incertitude de géométrie, et
> c'est démontré, pas soupçonné (spec §12.16-12.17). Le remplaçant est
> `spatial_pooling_model_ou` : une inférence **jointe** de la géométrie et du
> chemin de $w$, où $\sigma_w$ est un paramètre du modèle — donc plus de banque
> `bank_w_ou.json` ni de couplage géométrie/calibration. Il tient sur données
> simulées (89,5 % d'IC90 contre 99,0 % ici) et sur le roster 2027, mais **pas
> encore sur les coupures 2022** (R-hat 1,387 à J-150, spec §12.22-12.23). Tant
> que ce n'est pas réglé, la production reste sur la chaîne ci-dessous et
> `w_dynamics.py` n'est pas supprimable.

Le fit ci-dessus n'estime que la **géométrie**. $w$ est relu ensuite, sans MCMC.
Motif : la variance de la vraisemblance tempérée vaut
$(\text{prior}^{-1}+\sum_p\kappa_p I_p)^{-1}$, qui tend vers 0 en $1/n$ **sans
plancher**, même si tous les sondages sont vieux — alors qu'un OU impose un
plancher $\sigma_w^2(1-e^{-2g/\tau_w})$ qu'aucune accumulation ne franchit.

**(a) Inversion exacte par nœud.** On résout $\pi(w)=\tilde Y_p$ sur
$\{\sum w = 0\}$. C'est bien posé : $\pi = \nabla\Phi$ avec
$\Phi(w)=\sum_b W_b\,\mathrm{logsumexp}_{j\in S_p}(A_{j,b})$, dont le hessien

$$J = \sum_b W_b\big(\mathrm{diag}(P_b) - P_bP_b^\top\big)$$

est SDP de noyau **exactement** $\mathrm{span}(\mathbb 1)$. $\Phi$ est donc
strictement convexe sur le sous-espace somme-nulle et l'inversion y est un
difféomorphisme (dualité de Legendre) : Newton amorti converge globalement, en
6 itérations médianes.

**(b) Méthode delta.** $\hat w_p \approx w(t_p) + $ bruit, de covariance
$C_p = J_p^{+}\Sigma_p J_p^{+}$ : pseudo-observation **linéaire-gaussienne** de
$w(t_p)$.

**(c) Krigeage universel à noyau OU.** $w_i(t) = m_i + u_i(t)$,
$u\sim\mathrm{OU}(0,\sigma_w^2,\tau_w)$ indépendant par candidat, $m$ de prior
plat et marginalisé — d'où le retour vers le niveau moyen estimé du candidat, et
non vers 0, aux grands écarts. Chaque nœud n'informant que les contrastes de
$S_p$, la matrice de design est le projecteur orthogonal
$C_{S_p}=\mathrm{diag}(m_p)-m_pm_p^\top/K_p$ ; les directions non identifiées
reçoivent une variance $10^6$. Une seule factorisation de Cholesky
$(P\!\cdot\!N)^2$ par tirage de géométrie.

$\tau_w$ est repris de la banque commune (`model/core/opinion.py`, $\approx262$ j)
et non réestimé : sur une fenêtre de campagne tous les écarts sont petits devant
$\tau_w$ et seul le rapport $\sigma_w^2/\tau_w$ est contraint. $\sigma_w^2$ est
calibré par couverture hors-échantillon (`bank_w_ou.json`).

## 6. Lecture, scénarios, projection

Le fit rend les **tirages** $(\mu,\sigma,w)$, pas des moyennes. Pour un champ
arbitraire coché par un visiteur, $\pi$ s'obtient en poussant chaque tirage à
travers le softmax masqué — $O(\text{tirages}\times B)$, **aucune
ré-inférence**. C'est ce qui justifie l'architecture face à un modèle à slots
figés.

Projection jusqu'au scrutin : machinerie **partagée**
(`model/core/projection.py::projeter_au_scrutin` + `model/core/opinion.py`),
identique à tous les modèles du dépôt — diffusion OU puis écart sondages-urne,
en espace log-ratio.

## 7. Ce que le modèle ne peut pas faire

- **Une seule dimension.** Un candidat recrutant sur tout le spectre pour une
  idée transversale ne peut être représenté que par un $\sigma$ large, donc
  *symétrique* — ce qui est faux. Une seconde dimension serait l'objet correct.
- **$\sigma$ des petits candidats** reste faiblement contraint même après
  normalisation : ils ne dominent nulle part, donc leur forme n'est vue qu'à
  travers des redistributions de faible amplitude.
- **Vraisemblance gaussienne, pas Beta** : pour un candidat à ~1 %,
  $\mathcal N(\pi,\pi(1-\pi)/n)$ met de la masse sous 0.
- **Champs jamais sondés** : la redistribution est validée sur des paires
  d'hypothèses réellement observées. Une combinaison inédite repose sur
  l'extrapolation géométrique, dont c'est le pari — pas une mesure.
