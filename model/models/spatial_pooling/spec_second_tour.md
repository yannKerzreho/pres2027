# Spec — second tour dans `spatial-pooling`

Note technique, session recherche du 2026-11-14. Statut : **proposition**, rien
n'est [TRANCHÉ]. Elle remplace la variante « option extérieure uniforme » du
commit Codex d7b1575, dont l'audit est résumé au §9.

Notations du modèle de premier tour : `spec_spatial_pooling.md` (§1-§4).

---

## 0. Ce qu'on veut produire

1. Pour **toute** paire de finalistes $(a,b)$ du roster, la part de $a$ parmi
   les suffrages exprimés du duel, $s_{ab}$, avec son incertitude au jour du
   scrutin — y compris pour les paires **jamais sondées** (2027 : 11 paires
   sondées sur 136 possibles, 14 notices).
2. $P(\text{élu})$ par candidat : composer le premier tour (quelle paire se
   qualifie) et le duel, tirage par tirage.

Le vainqueur se décide parmi les **exprimés** ; c'est aussi ce que publient les
instituts. L'abstention n'est donc utile que si elle change $s_{ab}$ (§3).

## 1. Rappel : l'attractivité au premier tour

Votant en $v\in[0{,}01;0{,}99]$, électorat uniforme (quadrature $V_b,W_b$).
Attractivité du candidat $i$ à la date $t$ :

$$A_i(v,t) = w_i(t) + \log\varphi_i(v),\qquad
\varphi_i(v)=\frac{e^{-(v-\mu_i)^2/2\sigma_i^2}}{\sum_b W_b\,e^{-(V_b-\mu_i)^2/2\sigma_i^2}}$$

Au premier tour sur le champ $S$ : $p_i(v\mid S)=e^{A_i}/\sum_{j\in S}e^{A_j}$,
$\pi_i(S)=\int p_i(v\mid S)\,dv$. Invariance : $w\mapsto w+c\mathbf 1$ ne change
rien ; le niveau global de $w$ n'est tenu que par le prior de $m_i$.

## 2. Le duel au niveau du votant

### 2.1 Forme

Pour le duel $(a,b)$, le votant en $v$ choisit $a$ parmi les exprimés avec

$$q_{ab}(v,t)=\operatorname{logistique}\Big(\kappa\,[A_a(v,t)-A_b(v,t)] + \eta_a-\eta_b\Big)$$

et la part agrégée parmi les exprimés est

$$s_{ab}(t)=\int q_{ab}(v,t)\,dv=\sum_b W_b\,q_{ab}(V_b,t).$$

**Révision après relecture (implémentée)** : la valence s'applique à
l'**agrégat**, $s_{ab}=\operatorname{logistique}\big(\operatorname{logit}g_{ab}+\eta_a-\eta_b\big)$
avec $g_{ab}=\int\operatorname{logistique}(\kappa[A_a-A_b])$. Au niveau du
votant, $q$ est une quasi-marche en $v$ (pente $\kappa(\mu_a-\mu_b)/\sigma^2$)
et η ne déplace que le point de bascule : l'effet de η = 0,5 variait de
+0,8 pt (σ = 0,08) à +11 pt (σ = 0,30), et σ, mal identifié, aurait réglé la
force du prior T2. À l'agrégat, η = 0,5 ≈ 12 pt autour de 50/50 quel que soit σ.
Les formules ci-dessous qui écrivent η dans $q$ se lisent avec cette révision.

Trois paramètres nouveaux, aucun par paire :

| | rôle | prior proposé |
|---|---|---|
| $\kappa>0$ | « température » du duel relative au T1 | $\log\kappa\sim\mathcal N(0;0{,}5)$ (centré sur κ = 1 : même agrégation qu'au T1) |
| $\eta_i$ | **valence de second tour** du candidat $i$ | $\eta_i\sim\mathcal N(0;0{,}5^2)$, échelle FIXE |
| $\sigma_{T2}$ | écart de modèle / effet institut des duels | $\sigma_{T2}\sim\mathcal{HN}(0{,}02)$ |

### 2.2 Propriétés

- **Invariance.** $q$ ne dépend que de $A_a-A_b$ et de $\eta_a-\eta_b$ : la
  translation de $w$ (§1) reste sans effet, et seule la *différence* des $\eta$
  est identifiée ; leur niveau commun est tenu par le prior. Aucune
  non-identifiabilité nouvelle.
- **Lien positions → reports.** Avec $\kappa=1,\eta=0$, $q_{ab}(v)=p_a(v\mid\{a,b\})$ :
  c'est le premier tour restreint à deux noms, au niveau du **votant** (pas de
  l'agrégat). La matrice de reports en découle sans être posée :
  $$T_{k\to a}=\frac{1}{\pi_k(S_1)}\int p_k(v\mid S_1)\,q_{ab}(v)\,dv,$$
  l'électorat de $k$ est concentré autour de $\mu_k$ et se reporte selon
  l'attractivité relative des finalistes **à cet endroit de l'axe**.
  Rien n'est par paire : $T$ est défini pour toute paire et tout champ $S_1$.
- **Pourquoi $\eta$.** Mesuré sur le postérieur publié du 2026-11-13 (géométrie
  T1 seule, aucun duel dans le fit) contre les 11 paires sondées en 2027 :
  géométrie seule, MAE **6,3 pt**, avec un signe systématique : le RN est
  sous-estimé contre tous ses adversaires (Philippe, Retailleau, Mélenchon,
  Glucksmann) sauf Attal, Mélenchon est surestimé. Validation croisée
  « une paire retirée » (η ajusté sur les 10 autres paires, géométrie figée) :

  | variante | MAE paire retirée |
  |---|---|
  | géométrie seule (κ=1, η=0) | 6,26 pt |
  | géométrie + η, κ = 1 | 2,48 pt |
  | géométrie + η, κ = 0,5 | 1,92 pt |
  | Bradley-Terry pur (η seul, sans géométrie) | 1,93 pt |

  C'est aussi l'explication la plus simple de l'échec Codex (§9) : sans η, des
  dizaines de duels très précis se battent contre les centaines de nœuds T1,
  le fit sous-ajuste les duels (Macron −6 pt en 2017, −5 pt en 2022, *pire* que
  la moyenne brute des sondages) et ne converge pas (R-hat 9 à 14).

  η lit le « front républicain » / « plafond de verre » : un écart d'attraction
  au second tour qui n'existe pas au premier. Il est **par candidat**, pas par
  paire : 8 paramètres en 2027 pour 11 paires, et un candidat jamais testé en
  duel garde $\eta\sim$ prior, donc un duel plus incertain — honnête.
- **Ce que la géométrie apporte encore**, puisque BT pur fait jeu égal sur la
  validation croisée : (i) un a priori pour les candidats **sans aucun duel**
  (Zemmour, Hollande, Villepin… : BT dit 50/50, la géométrie dit quelque chose) ;
  (ii) le co-mouvement duel/premier tour via $w(t)$ partagé ; (iii) des
  reports cohérents avec le champ du premier tour. La validation (§8) doit
  montrer (i) sur des paires dont un candidat n'a aucun autre duel.

### 2.3 Limite connue : reports non monotones en $v$

À $\eta$ fixé, $\kappa(A_a-A_b)$ est **affine en $v$** si $\sigma_a=\sigma_b$
(pente $\kappa(\mu_a-\mu_b)/\sigma^2$) et monotone en pratique : un électorat
plus à gauche se reporte *davantage* sur le finaliste le plus à gauche. Or en
2022, d'après les enquêtes jour du vote (ordre de grandeur, à sourcer),
Mélenchon → ~71/29 Macron/Le Pen parmi exprimés, Jadot → ~92/8. La fuite de la
gauche radicale vers le RN est une **seconde dimension** (anti-système) que le
modèle 1D ne représente pas ; η par *finaliste* ne peut pas dépendre de
l'électorat d'origine. Conséquence : les reports *par électorat* publiés
seraient faux pour la gauche radicale, même si $s_{ab}$ agrégé est juste.
**On ne publie donc que $s_{ab}$ et $P(\text{élu})$, pas la matrice $T$.**

Pistes (non retenues en v1) : axe en « fer à cheval » pour le seul T2
(distance de corde $d_\alpha(u,v)=\tfrac{2}{\alpha}\sin\tfrac{\alpha|u-v|}{2}$,
un paramètre de courbure) ; seconde coordonnée par candidat (bilinéaire).
Les deux demandent des duels contre des adversaires variés pour être
identifiés.

## 3. Abstention, blancs et nuls

> **Révision (2026-11-14, idée utilisateur) — ACTIVÉE par défaut.** La §3.2
> ci-dessous concluait « pas en v1 ». Mesure faite (géométrie du 2026-11-13
> figée, une paire retirée sur les 11 paires 2027) :
>
> | variante | paramètres T2 | MAE |
> |---|---|---|
> | géométrie seule | 0 | 6,3 pt |
> | abstention (κ, ω), sans η | 2 globaux | 3,2 pt |
> | valence η seule | κ + 1/candidat | 2,9 pt |
> | abstention + η | κ, ω + 1/candidat | **2,4 pt** |
>
> Deux paramètres globaux récupèrent l'essentiel de l'écart, avec une
> non-expression plausible (9 à 21 % des exprimés du T1). L'effet est **par
> électorat**, pas par candidat : un centriste perd contre le RN parce que
> l'électorat de gauche, loin des deux, s'abstient et ne lui reporte plus — ce
> que η seul ne sait pas dire. Les deux sont conservés ; prior
> $\omega\sim\mathcal N(-3;1{,}5)$. Backtest : `--variante abst|eta`.

### 3.1 Forme correcte d'une option extérieure

Si l'on ajoute « ne pas exprimer » au choix du votant, son utilité doit être
**relative à l'offre du votant**, sinon $\omega$ et le niveau global de $w$
sont confondus (§1). La forme naturelle est la valeur inclusive du premier
tour (logit emboîté) :

$$U_0(v)=\omega+\kappa\,\log\sum_{j\in R}e^{A_j(v)},\qquad
\rho_{ab}(v)=\frac{e^{\kappa A_a+\eta_a}+e^{\kappa A_b+\eta_b}}{e^{\kappa A_a+\eta_a}+e^{\kappa A_b+\eta_b}+e^{U_0}}$$

($R$ = roster). Invariante par translation de $w$ ; un votant dont le favori
est finaliste exprime presque sûrement, un votant dont tout l'entourage est
éliminé s'abstient davantage. Part exprimée :
$s_{ab}=\int\rho\,q_{ab}\big/\int\rho$.

Le code Codex fait $A\leftarrow A-\max_{j\in R}A_j(v)$ puis ajoute
$e^{\omega}$ **non décalé** : c'est, sans l'avoir posé, $U_0=\omega+\max_jA_j(v)$
avec κ = 1 — la limite « dure » de la forme ci-dessus — mais sans la valence ni
la température, et avec $\rho\le 1/2$ dès que le favori est finaliste
(1/3 ≤ p_out ≤ 1/2). La Beta sur la non-expression publiée est alors
structurellement incompatible avec des taux de 10-25 %.

### 3.2 Pourquoi ne pas l'activer en v1

- **Identification.** $\rho$ ne change $s_{ab}$ qu'en repondérant les
  électorats éloignés des deux finalistes. κ fait presque la même chose (il
  ramène leur $q$ vers 1/2) et η absorbe la moyenne par candidat : ω n'est
  séparé que par la *forme* des écarts entre paires, avec 11 paires.
- **Données.** La non-expression publiée (`non_expression`) est en % des
  **inscrits** : il faudrait un terme de base (abstention structurelle,
  ~22-26 % au T1) pour la comparer à $\int(1-\rho)$, qui est une part des
  exprimés du T1. Disponible en 2017 seulement (72 lignes), 0 en 2022, absente
  du CSV 2027.
- **Enjeu.** Le vainqueur se lit sur les exprimés.

Proposition : **pas d'option extérieure en v1** ; la forme 3.1 est la seule à
réintroduire si un backtest montre un résidu structuré par paire.

## 4. Vraisemblance des duels

Un nœud T2 $p$ = une hypothèse de duel d'un sondage : paire $(a_p,b_p)$, date
$t_p$, échantillon $n_p$, part observée $y_p=Y_{a}/(Y_a+Y_b)$.

$$y_p\sim\mathcal N\Big(s_{a_pb_p}(t_p),\ \frac{s(1-s)}{n_p}+\sigma_{T2}^2\Big)$$

**Une seule** observation par duel. **Pas de déflation** $n/n_{hyp}$
(révision) : elle donnait ~2,9 pt d'écart-type d'échantillon aux duels 2027
contre 0,9-1,4 pt de dispersion mesurée en 2017/2022 ; l'excès va dans $\sigma_{T2}$. Codex mettait les deux parts dans la
vraisemblance gaussienne indépendante du T1 : pour $K=2$ elles sont
parfaitement anticorrélées, c'est exactement **compter chaque duel deux fois**
(précision ×2). $n_p$ est l'échantillon total, alors que les parts sont parmi
les exprimés (~70-80 %) : léger excès de précision absorbé par $\sigma_{T2}$.

**Effets instituts.** La banque `excess_var` est calibrée sur le T1 ; on ne la
réutilise pas. Un $\sigma_{T2}$ global suffit : 2027 = 14 notices T2, trop peu
pour un effet par institut. Mesuré : écart-type des duels Macron–Le Pen autour
de leur moyenne mobile pendant l'entre-deux-tours, 0,9 pt (2022), 1,4 pt (2017).

**Effet notice.** Un même sondage teste souvent plusieurs duels : leurs erreurs
d'échantillon sont corrélées (même panel). Ignoré en v1 (comme au T1, où
`notice_idx` sert au diagnostic) ; à surveiller si un institut domine les duels.

## 5. Dynamique

$w(t)$ est partagé : si Le Pen monte au T1, ses duels montent aussi, sans
paramètre. κ et η sont **constants** sur la fenêtre. Hypothèse testable :
résidu des duels Macron–Le Pen 2022 contre le temps.

## 6. Projection au jour du scrutin

À la date $t_0$ (= `as_of`), pour une paire $(a,b)$ et un tirage postérieur :

1. **Dérive** jusqu'au second tour $T_2$ ($h=T_2-t_0$) : on prolonge le même
   processus OU que le modèle (même $\tau$, même $\sigma_w$) :
   $w_i(T_2)=m_i+(w_i(t_0)-m_i)e^{-h/\tau}+\sigma_w\sqrt{1-e^{-2h/\tau}}\,\xi_i$
   (implémenté avec la variance $\sigma_w^2[1-r^2+(1-r)^2]$, $r=e^{-h/\tau}$ :
   $m_i$ n'est que la moyenne du chemin, pas le centre OU connu),
   pour les candidats dynamiques, $w_i(T_2)=w_i(t_0)$ pour les statiques.
2. **Erreur de scrutin T2** : $\operatorname{logit}s^{\text{scrutin}}=
   \operatorname{logit}s_{ab}(T_2)+\varepsilon$, $\varepsilon\sim\mathcal N(0,\tau_{T2}^2)$.
   Erreur finale des sondages de 2nd tour (dernière semaine → résultat,
   part du premier nommé) : 2017 +3,8 pt (62,3 → 66,1, données du dépôt),
   2022 +2,8 pt (55,8 → 58,55). 2002/2007/2012 de mémoire, **à sourcer** :
   ≈ +2, −0,7, −1,1. RMS ≈ 2,4 pt, soit $\tau_{T2}\approx0{,}10$ en logit autour
   de 60/40. **Centrée sur 0** : les trois duels contre le FN/RN ont tous
   sous-estimé son adversaire, mais n = 3 ; un terme directionnel serait un
   choix éditorial, à trancher par l'utilisateur.

## 7. $P(\text{élu})$

Pour un champ de premier tour $S_1$ (champ modal, ou champ coché sur le site)
et chaque tirage $s$ : parts T1 projetées $\pi^{(s)}$ (`projeter_au_scrutin`) →
paire qualifiée $(a,b)^{(s)}$ (deux premiers) → $s_{ab}^{\text{scrutin},(s)}$
(§6) → élu. $P(\text{élu}=i)$ = fréquence. Les deux étapes partagent le tirage
de $(\mu,\sigma,w,\kappa,\eta)$ : la corrélation « fort au T1 ⇒ fort au T2 »
est conservée.

**Limites connues de la composition (relecture)** : la dérive T1
(`projeter_au_scrutin`, loi calibrée sur les parts + δ) et celle de $w$ (OU du
fit) sont tirées **indépendamment** : un candidat qui bondit au T1 dans un
tirage n'a pas de bonus en duel, et les erreurs de scrutin T1/T2 sont
supposées indépendantes alors qu'elles sont corrélées. Correctif propre : un
seul aléa — projeter $w$ au T1, en déduire π(T1) par `spatial_shares` + δ, puis
prolonger $w$ du T1 au T2 conditionnellement. Dans l'entre-deux-tours, il faut
conditionner sur la paire réelle et le résultat T1 (nœud quasi exact). Non fait.

**Crête κ/η** : pour un candidat statique, $\kappa w_a$ et $\eta_a$ jouent des
rôles voisins ; κ n'est tenu que par la forme en $v$ et le rétrécissement de η.
À vérifier sur les traces (corrélation κ–η).

## 8. Inférence et validation

**Joint, pas coupé.** Les duels entrent dans le même NUTS que le T1. Avec η,
ils n'ont plus besoin de tordre la géométrie pour être ajustés, ce qui était
le risque du joint. Le modèle coupé (T1 seul, puis $(\kappa,\eta,\sigma_{T2})$
conditionnellement) reste le repli si la convergence du T1 se dégrade —
critère : R-hat du fit publié inchangé sur 2027, ±0,02.

**Validation**, 2017 et 2022, au lieu du seul point T2−7 j :

- **V1, paire retirée (le vrai test des reports).** Retirer tous les sondages
  d'une paire, ajuster à T1−1 j, prédire la moyenne de ces sondages. 2022 :
  Macron–Pécresse, –Zemmour, –Mélenchon, Le Pen–Pécresse ; 2017 : Fillon–Le Pen.
  Comparé à : BT pur, géométrie seule.
- **V2, résultat réel.** À T1−30 j, T1−1 j et T2−7 j : couverture IC90 du
  résultat du duel final et log-score, contre la moyenne brute des sondages
  ± même $\tau_{T2}$.
- **Garde T2** : R-hat propre au bloc T2 (κ, σ_T2, η des candidats en duel),
  pas le `rhat_max` du T1.
- **Critères de publication** : R-hat ≤ 1,2 (seuil du modèle publié, pas
  1,01) ; V1 meilleur que la géométrie seule ; V2 couvre au moins 2017 *ou*
  2022 sans élargir au-delà de la moyenne brute ± $\tau_{T2}$. Deux élections
  ne valident rien statistiquement : c'est un **garde-fou**, pas une preuve.
- Le gate ne doit **jamais** bloquer le premier tour : un échec T2 retire les
  duels du snapshot, pas le snapshot.

## 9. Audit Codex (d7b1575, 7dcb84a) — résumé

Détails : note de session `recherche-t2-01-audit-codex.md`.

1. Option extérieure : décalage par $\max_jA_j$ avant d'ajouter $e^\omega$ —
   ω non identifié, duel dépendant du $w$ d'un éliminé, $1/3\le p_{out}\le1/2$.
   Ce bug n'explique **pas** le biais : sans option extérieure, Codex obtient
   déjà 0,598 (2017) et R-hat 9,2 (2022). Le biais vient de l'absence de η (3).
   Le R-hat « NaN » de 2017 est un artefact : `p_non_expression` vaut 0 à
   chaque tirage sur les nœuds T1, et `numpyro_summary` rend NaN.
2. Duel compté deux fois (deux parts anticorrélées, §4).
3. Pas de valence T2 → biais de 5-6 pt et non-convergence (§2.2).
4. Gate `t2_validation.json` = `reject` levant dans `nowcast` : plus aucune
   publication spatial-pooling, y compris le T1. `RHAT_PUBLIABLE` 1,2 → 1,01.
5. Ni dérive jusqu'au scrutin ni erreur de scrutin T2.
6. `run.py` passe T1+T2 à tous les modèles, sans filtre dans
   bayesian-nowcast ni dans `used_polls`/`display_polls` du spatial ; `part`
   (candidats dynamiques) calculé sur nœuds T1+T2. Corrigé : un modèle ne voit
   le T2 que s'il déclare `consomme_second_tour`.
7. Réimport 2017 : 183 lignes T1 ont perdu `date_fin` ; l'entre-deux-tours
   2017 ne compte que 9 duels Macron–Le Pen (beaucoup plus publiés).
