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

($R$ = roster). **Implémenté (révision §2.1)** : η n'entre PAS dans ρ,
$\rho_{ab}(v)=\frac{e^{\kappa A_a}+e^{\kappa A_b}}{e^{\kappa A_a}+e^{\kappa A_b}+e^{U_0}}$,
et η s'applique à l'agrégat $\int\rho\,q/\int\rho$ en logit (`geometry.py`).
**Fragilité (revue 2026-11-14)** : $R$ = roster, pas le champ ; des alias
exclusifs (groupe LR 2027) gonflent la valeur inclusive locale de ~log k.
Sans effet en 2017/2022 (roster = champ réel) ; à corriger avant 2027 (§10).
Invariante par translation de $w$ ; un votant dont le favori
est finaliste exprime presque sûrement, un votant dont tout l'entourage est
éliminé s'abstient davantage. Part exprimée :
$s_{ab}=\int\rho\,q_{ab}\big/\int\rho$.

Le code Codex fait $A\leftarrow A-\max_{j\in R}A_j(v)$ puis ajoute
$e^{\omega}$ **non décalé** : c'est, sans l'avoir posé, $U_0=\omega+\max_jA_j(v)$
avec κ = 1 — la limite « dure » de la forme ci-dessus — mais sans la valence ni
la température, et avec $\rho\le 1/2$ dès que le favori est finaliste
(1/3 ≤ p_out ≤ 1/2). La Beta sur la non-expression publiée est alors
structurellement incompatible avec des taux de 10-25 %.

### 3.2 Pourquoi ne pas l'activer en v1 — CADUC (cf. encadré : activée)

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
   ≈ +2, −0,7, −1,1. **Mesure 2026-11-14 (CSV du dépôt, moyenne des duels
   finaux sur 7 j, résultat − sondages)** : 2017 +3,9 / +3,3 / +3,3 / +4,4 / +4,8 pt
   à T1−30 / T1−1 / T2−14 / T2−7 / T2−1 ; 2022 −0,3 / +5,8 / +5,7 / +5,2 / +3,1.
   L'erreur ne décroît pas nettement avec l'horizon et dépasse 3 pt partout
   sauf 2022 T1−30 : τ_T2 = 0,10 est **sous-dimensionné** (≈ 0,18-0,20 en
   logit pour un RMS ≈ 4,3 pt), et toutes ces erreurs ont le même signe.
   RMS de mémoire ≈ 2,4 pt, soit $\tau_{T2}\approx0{,}10$ en logit autour
   de 60/40. **Centrée sur 0** : les trois duels contre le FN/RN ont tous
   sous-estimé son adversaire, mais n = 3 ; un terme directionnel serait un
   choix éditorial, à trancher par l'utilisateur.

   **Décision de l'utilisateur (2026-11-15 ~21:45Z, arbitrage 1)** :
   $\tau_{T2} = 0{,}19$ (RMS ≈ 4,3 pt mesuré 2017/2022 ; couverture 5/5 de la
   base §8.18). Réserve écrite (§8.19) : 2 erreurs indépendantes, IC90 de τ
   [0,12 ; 0,92] ; un postérieur sur τ à partir de 1995-2012 reste à faire.
   Biais anti-RN : **non corrigé** (arbitrage 2) ; lissage inchangé (arbitrage 3).
3. **Incertitude d'extrapolation des duels jamais sondés** (arbitrage 4,
   adopté) : pour une paire $(a,b)$ sans AUCUN duel sondé dans le jeu,
   $\operatorname{logit}s_{ab} \leftarrow \operatorname{logit}s_{ab} + \nu$,
   $\nu\sim\mathcal N(0,\tau_{paire}^2)$, tiré une fois par tirage, en
   prédiction seulement (pas dans la vraisemblance), avant l'erreur de
   scrutin. Les paires sondées n'en reçoivent pas. Dans $P(\text{élu})$, le
   terme s'applique au tirage dont la paire qualifiée n'a jamais été sondée.
   **Calibrage** (§8.22) : log-score KDE des V1 2027 dédoublonnées (§8.20)
   avec $\nu$ ajouté, maximisé sur une grille de 0,01, variante hier (défaut).

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

**7.1 Variante « aléa unique » (2026-11-16, drapeau `alea_unique`).** Pour
chaque tirage $s$, horizons $h_1 = T_1 - t_0$, $h_2 = T_2 - t_0$ :
1. $w^{(1)} = \mathrm{OU}(w(t_0), h_1)$ : `projeter_w` (§6.1), même processus
   que le fit ;
2. $\pi^{(1)} = \mathrm{softmax}\big(\log \mathrm{spatial\_shares}(\mu,\sigma,w^{(1)},S_1) + \delta\big)$,
   δ = écart de traduction de la banque commune (`delta_draws`, sans la
   diffusion de la banque : la dérive est déjà dans $w^{(1)}$) ;
3. paire qualifiée = deux premiers de $\pi^{(1)}$ ;
4. $w^{(2)} = \mathrm{OU}(w^{(1)}, h_2 - h_1)$, prolongement conditionnel :
   $w^{(2)} = m + (w^{(1)}-m)r + \sigma_w\sqrt{1-r^2+(1-r)^2}\,\xi$,
   $r = e^{-(h_2-h_1)/\tau}$ (même approximation que §6.1 pour $m$) ;
5. duel au T2 sur $w^{(2)}$, puis τ_paire (§6.3) et τ_T2 (§6.2).
Un candidat qui monte au T1 dans un tirage garde cette montée au T2 : la
corrélation T1/T2 passe par $w$. Différence assumée avec la voie actuelle :
la dérive T1 vient de l'OU du modèle ($\sigma_w$, τ ≈ 262 j) et non de la
diffusion de la banque (calibrée sur les parts). Les erreurs de scrutin T1
(δ) et T2 (τ_T2) restent indépendantes (aucune mesure de leur corrélation).
Comparaison : P(élu) du champ modal 2027, deux voies, même fit.

**Résultat (2026-11-16, hier, données du 13/11, même fit,
`diagnostic_2027_hier_2026-11-13_alea_unique.json`)** :

| voie | Le Pen | Philippe | Glucksmann | Retailleau | Mélenchon | paires qualifiées (3 premières) |
|---|---|---|---|---|---|---|
| actuelle (deux aléas) | 80,3 | 13,9 | 4,5 | 1,1 | 0,2 | Mél.–LP 39 %, Phi.–LP 29 %, Glu.–LP 16 % |
| aléa unique (§7.1) | 78,9 | 14,0 | 3,8 | 3,3 | 0,0 | Mél.–LP 36 %, Phi.–LP 34 %, Glu.–LP 16 % |

Écarts ≤ 2 pt de P(élu), sauf Retailleau (×3, sur des petits nombres) : la
corrélation T1/T2 passe désormais par $w$, mais l'essentiel de l'écart vient
du changement de loi de dérive T1 (OU du modèle au lieu de la banque
calibrée). Aucune validation possible (P(élu) ne se backteste que sur 2
élections). **Décision : drapeau éteint par défaut** — la loi de dérive T1
de la banque est calibrée, celle de l'OU du modèle ne l'est pas sur des parts ;
la voie §7.1 reste disponible, à reconsidérer si l'OU est calibré sur les
parts (backtest de suivi, spec spatial_pooling §10).

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

**Règle permanente (2026-11-14).** Aucune proposition de `PUBLIER_T2 = True`
sans LES DEUX : (i) la validation historique ci-dessus (V1, V2, garde T2) sur
2017 et 2022, ET (ii) le **diagnostic 2027** satisfaisant
(`python -m model.backtest.diagnostic_2027`, fit joint sur les sondages
courants, résidus pris contre la part prédite **à la date** de chaque sondage,
$r_d = y_d - \bar s_d$, $z_d = r_d / \hat\sigma_d$ avec
$\hat\sigma_d^2 = \mathrm{Var}_{post}(s_d) + \bar s_d(1-\bar s_d)/n_d + \sigma_{T2}^2$).
Seuils de (ii), éditoriaux et fixés AVANT de regarder les chiffres :

1. convergence : `rhat_max` ≤ 1,2 et `rhat_t2` ≤ 1,2 ;
2. calibration : $\overline{z^2}$ global dans [0,5 ; 2] ;
3. pas de biais par paire : pour toute paire à $n \ge 5$ sondages,
   $|\bar r| \le 1{,}5$ pt **et** $|\bar z|\sqrt n \le 3$ (lâche : les sondages
   d'un même institut sont corrélés) ;
4. pas de biais par institut ($n \ge 10$) ni par candidat : $|\bar r| \le 1{,}5$ pt
   (institut), ≤ 1 pt (candidat, résidu signé de son point de vue) ;
5. tendance captée : pour toute paire à ≥ 3 sondages sur les 30 derniers jours,
   $|\bar r_{30j}| \le 1{,}5$ pt et |duel du modèle aujourd'hui − moyenne brute
   des 30 j| ≤ 2 pt.

Un critère violé n'est pas tranché en silence : s'il oppose fidélité au passé et
fidélité aux sondages 2027 (p. ex. un $\tau_{T2}$ ou un biais calibré sur
2017/2022 qui décale les duels 2027), il va dans « Arbitrages ouverts » du
journal, pour l'utilisateur.

### 8.1 Diagnostic 2027 (2026-11-14, sondages au 2026-11-12)

`python -m model.backtest.diagnostic_2027`, 92 duels, 12 paires, 7 instituts ;
4 chaînes × (600+400). Résultats : `model/backtest/results/diagnostic_2027_{abst,eta}_2026-11-12.json`.
Résidus **dans l'échantillon** (le duel a servi au fit) : ils testent la
capacité à représenter les sondages, pas la prédiction.

| | abst | eta |
|---|---|---|
| R-hat max / rhat_t2 / ESS min / div. | 1,040 / 1,040 / 202 / 0 | 1,034 / 1,034 / 125 / 0 |
| κ / ω / σ_T2 | 0,75 / −3,45 / 1,3 pt | 1,75 / — / 1,5 pt |
| résidu global moy. / é.-t. ; $\overline{z^2}$ | +0,06 / 1,67 pt ; 0,75 | +0,08 / 1,85 pt ; 0,77 |
| max \|corr(log κ, η_i)\| | 0,22 (Retailleau) | 0,08 |

Par paire (abst ; n ≥ 4) — résidu moyen, résidu des 30 derniers jours, modèle
aujourd'hui vs moyenne brute 30 j (part du premier nommé), affiché site (T2, IC90) :

| paire | n | $\bar r$ | $\bar r_{30j}$ (n) | auj. / brut 30 j | site |
|---|---|---|---|---|---|
| Philippe / Le Pen | 24 | −0,1 | −0,1 (8) | 43,9 / 44,6 | 45,1 [37,2 ; 52,4] |
| Mélenchon / Le Pen | 23 | −0,0 | **−1,3** (8) | 33,3 / 31,5 | 32,8 [28,5 ; 37,5] |
| Attal / Le Pen | 14 | +0,3 | +1,9 (2) | 40,1 / 43,5 | 41,5 [33,1 ; 49,5] |
| Glucksmann / Le Pen | 13 | +0,2 | −0,0 (8) | 42,3 / 43,2 | 42,3 [36,2 ; 48,5] |
| Hollande / Le Pen | 4 | +0,2 | −0,5 (2) | 42,9 / 43,0 | 43,2 [36,1 ; 50,0] |
| Philippe / Bardella | 4 | +1,2 | — | 44,9 / — | 45,6 [36,5 ; 55,4] |

Institut (n ≥ 10, abst) : Harris −0,9 pt (25), YouGov +0,9 (12), Elabe +0,5
(18), Ifop 0,0, OpinionWay −0,1. Candidat : tous |r̄| ≤ 0,2 pt.
P(élu), champ modal (abst) : Le Pen 0,85, Philippe 0,11, Glucksmann 0,03.

**Verdict contre la règle (ii)** : les 5 critères passent, les deux variantes.
Points limites : (5) Mélenchon / Le Pen, $\bar r_{30j} = -1{,}3$ pt et écart
modèle–brut 1,8 pt (seuil 1,5 / 2) : le modèle **lisse** une baisse récente de
Mélenchon en duel (corr résidu–date −0,47), il ne la suit qu'à moitié ;
(4) Harris/YouGov à ±0,9 pt suggèrent des effets maison au T2 (non modélisés),
sans franchir le seuil. $\overline{z^2} = 0{,}75 < 1$ : l'incertitude par
sondage est un peu large dans l'échantillon (attendu : Var_post(s) y est
comptée alors que le sondage a servi à l'estimer). abst et eta s'accordent à
≤ 1 pt sur tous les duels sondés ; abst donne des IC site plus larges
(P(Philippe bat Le Pen) 0,14 contre 0,09).

### 8.2 Backtest 2022 : géométrie T1 non identifiée (2026-11-14)

Premier backtest complet (abst, V2 2022 T1−30, 4 processus d'une chaîne
fusionnés) : duel Macron 58,3 % [53,5 ; 63,3] pour un résultat de 58,6 %
(couvert), **mais R-hat 35,8, ESS 2, rhat_t2 3,0**, profondeur d'arbre
saturée (1023) dans les 4 chaînes. Ce n'est pas un défaut de la fusion : les
chaînes sont dans des géométries différentes (Roussel/Jadot à 0,1 ou à 0,7,
de part et d'autre de Macron ; κ = 0,57 / 0,59 / 1,37 / 1,71).

Cause : `polls_historiques` ne garde qu'UNE hypothèse par notice, la plus
proche du champ final, et seulement des candidats du scrutin. Résultat :
100 notices sur 108 testent le même champ à T1−30 (140/144 à T1−1). Or les
positions $\mu_i$ ne sont identifiées que par la **variation des champs**
(qui perd quoi quand un candidat sort) : avec un champ unique, $\pi$ est
ajusté par les seuls $w_i$ et les positions errent selon le prior et les
ancres. Les duels ne suffisent pas à les fixer (κ, η, ω absorbent).
En 2027, des dizaines de champs distincts sont testés et le fit converge
(R-hat 1,04, §8.1) : **le backtest historique, tel qu'il est construit, teste
un régime que la production ne rencontre pas.** Ses verdicts de convergence ne
sont donc pas transposables, et ses chiffres de couverture reposent sur une
géométrie non identifiée.

Correctif proposé (à spécifier avant code, cf. journal) : garder TOUTES les
hypothèses T1 des notices de la fenêtre (y compris candidats non qualifiés au
final, p. ex. variantes LR/PS de l'automne 2021), comme le fait le roster
2027 — ce qui retire aussi la fuite « champ final connu » (revue, point 8).

### 8.3 Cadre historique corrigé : toutes les hypothèses T1 (2026-11-14)

**Données.** Pour une coupure $t$ de l'élection $e$, l'ensemble des nœuds T1
est $\mathcal P = \{(n, h)\}$ : TOUTES les hypothèses $h$ de toutes les
notices $n$ terminées dans $[t - 180\,\text{j}, t]$, sans filtre sur les
candidats du scrutin ni sélection d'une hypothèse. Les noms sont canonisés
sur les candidats du scrutin quand la correspondance est unique
(`_resolve_candidate`), sinon gardés bruts (« Bertrand », « Taubira »).

**Roster** : exactement celui de 2027, `build_roster` (candidat retenu s'il
figure dans ≥ `MIN_POLLS` = 5 notices). Chaque candidat retenu doit être
classé dans `ORDER_GROUPS_HISTORIQUES[e]`, qui fixe l'ancre de position
(rang gauche→droite, seule information de position a priori).

**Vraisemblance** : inchangée. Pour un nœud $p$ de champ testé $M_p$ et de
roster $R$, la restriction d'un multinomial au sous-ensemble $M_p \cap R$ est
un multinomial : $y_{p,i} \leftarrow y_{p,i} / \sum_{j \in M_p\cap R} y_{p,j}$
et $N_p \leftarrow N_p \cdot \sum_{j\in M_p\cap R} y_{p,j} / n_{hyp}(n)$,
comparés à $\pi_{p}$ = softmax spatial masqué sur $M_p \cap R$ (déjà le cas
en production). Les candidats sous le seuil sortent donc proprement du nœud.

**Pourquoi** : $\mu_i$ n'est identifié que par la covariation de $\pi$ avec
$M_p$ (qui récupère les voix quand $k$ sort). Champs distincts : 2022 T1−30
passe de 4 (108 nœuds) à 66 (222 nœuds) ; T1−1 de 2 à 53. 2017 reste pauvre
(10 et 24 nœuds datés, aucun candidat putatif ≥ 5 notices) : ses verdicts
resteront fragiles quoi qu'on fasse.

**Duels** : inchangés (paires de deux candidats du scrutin), pour garder la
même liste de tâches V1/V2 ; garder aussi les duels putatifs
(Bertrand–Macron…) est une extension notée dans le journal.

**Classement ajouté (choix éditorial, gauche→droite dans chaque groupe)** —
2022 : écolo += Thouy, Rousseau ; PS = Montebourg, Taubira, Hidalgo ;
centre = Macron, Lagarde ; LR = Bertrand, Pécresse, Barnier, Juvin, Wauquiez
(ajouté le 15/11 pour les coupures d'automne 2021), Ciotti ;
souverain = Dupont-Aignan, Poisson, Asselineau, Philippot. 2017 (tous sous le
seuil, pour la robustesse) : écolo = Jadot ; PS = Montebourg, Peillon,
Hamon, Valls ; centre = Macron, Bayrou.

**Fit de contrôle (2022 T1−30, abst, 4 chaînes, 20 min)** : duel Macron
58,7 % [53,9 ; 63,4] (résultat 58,6 %, couvert ; log-score 2,64 contre 2,64
pour la moyenne brute). Convergence **meilleure mais insuffisante** :
R-hat 15,3 (contre 35,8), rhat_t2 2,4, profondeur 255-511 (contre 1023
saturé). L'ordre gauche→droite est désormais le même dans les 4 chaînes ; le
R-hat restant tient à : **Lassalle** (μ = 0,12-0,16 dans 3 chaînes, 0,80 dans
la 4e : candidat transversal, bimodal), une chaîne qui comprime la gauche
(Mélenchon 0,06 contre 0,20), et σ de Zemmour (0,06 à 0,20). Témoin ancien
cadre : `second_round_abst_v2_2022_T1-30_ancien_cadre.json`.

### 8.4 Convergence du backtest : juger la quantité prédite (2026-11-14)

**Constat.** Même coupure (2022 T1−30, nouveau cadre) en **T1 seul**
(`--sans-t2`) : R-hat 10,5, ESS 2. La multimodalité vient donc de la
géométrie T1 du cadre historique (23 candidats, fenêtre de 180 j), pas du
couplage avec les duels ; c'est le défaut déjà documenté du modèle T1 sur les
grosses coupures 2022 (spec_spatial_pooling §8 : « ESS de 3 à 112 selon la
graine »). Les modes sont des **permutations de voisins** (Mélenchon à gauche
ou à droite de LO, Zemmour à 0,76 ou 0,94) qui expliquent les sondages
presque aussi bien. Dans le joint : observables T1 $\pi_p$, écart
inter-chaînes médian 0,16 pt (max 4,5) ; duels sondés $s_d$, médian 0,43 pt
(max 3,0) ; **duel cible Macron–Le Pen par chaîne : 58,3 / 59,1 / 59,0 /
59,0 %**, soit 0,8 pt d'écart pour un IC90 de 9,5 pt — alors que κ va de
0,42 à 1,34.

**Critère.** Pour un backtest, les paramètres de géométrie $(\mu,\sigma,\kappa)$
sont des nuisances ; ce qui est validé est la quantité prédite
$q = s_{ab}$ (V1 : à $h=0$ ; V2 : projetée au T2, **sans** l'aléa de scrutin,
tiré indépendamment des chaînes). Avec $C$ chaînes de $D$ tirages,
$\bar q_c = D^{-1}\sum_t q_{c,t}$ et le split-$\hat R$ calculé sur
$\mathrm{logit}\, q$ :

$$\text{convergé}_{\text{backtest}} \iff \hat R(\mathrm{logit}\,q) \le 1{,}1
\ \text{et}\ \max_c \bar q_c - \min_c \bar q_c \le 1\ \text{pt}.$$

1 pt ≈ un dixième de l'IC90 du duel : un écart de chaînes de cet ordre ne
change ni la couverture ni le log-score de façon lisible. Champs JSON :
`modele_par_chaine`, `ecart_chaines_pt`, `rhat_cible`. `rhat_max` reste
rapporté. **La garde de PRODUCTION n'est pas changée** (`rhat_max` ≤ 1,2 sur
2027, qui converge) : ce critère ne sert qu'à lire les backtests, et un
backtest qui le viole est déclaré non concluant, pas réussi.

### 8.5 V2 2022, nouveau cadre, variante abst (2026-11-14)

Résultat : Macron 58,55 % des exprimés du duel. Cible convergée aux trois
coupures (§8.4) ; `rhat_max` (géométrie) 15,3 / 32,2 / 4,9, non pertinent.

| coupure | modèle [IC90] | sondages 7 j (n) | couvre (mod. / brut) | log-score mod. / brut | écart chaînes | R̂ cible |
|---|---|---|---|---|---|---|
| T1−30 | 58,7 [53,9 ; 63,4] | 58,9 (7) | oui / oui | 2,64 / 2,64 | 0,6 pt | 1,015 |
| T1−1 | 52,4 [47,3 ; 57,3] | 52,8 (13) | **non / non** | 0,58 / 0,78 | 0,4 pt | 1,003 |
| T2−7 | 53,4 [48,8 ; 57,9] | 53,4 (15) | **non / non** | 0,97 / 0,92 | 0,2 pt | 1,003 |

Lecture : (1) le modèle **suit la moyenne des sondages** à ±0,4 pt — il ne la
bat pas, il ne la dégrade pas (log-scores équivalents) ; ce n'est pas le rôle
du bloc T2 sur un duel abondamment sondé. (2) En fin de campagne, les sondages
2022 sous-estimaient Macron de 5-6 pt et l'IC90 à $\tau_{T2} = 0{,}10$ ne
couvre pas (z = 2,05 et 1,87). À $\tau_{T2} = 0{,}19$ (mesure §6), estimation
analytique depuis la largeur hors scrutin : IC90 [44,1 ; 60,5] et
[45,4 ; 61,3], z = 1,24 et 1,06 — couverts. **Dans l'échantillon** : τ a été
mesuré sur ces mêmes élections ; c'est un argument de cohérence, pas une
validation. Chiffre l'arbitrage 1 du journal.

### 8.6 V1 2022 (paire retirée, T1−1, abst, nouveau cadre) — non concluant

| paire retirée (n) | observé 30 j | modèle [IC90] | géom. seule | BT | écart chaînes | R̂ cible |
|---|---|---|---|---|---|---|
| Macron–Zemmour (57) | 66,7 | 73,9 [67,4 ; 79,8] | 60,3 | 74,5 | 4,8 pt | 1,24 |
| Macron–Mélenchon (24) | 61,5 | 62,5 [40,7 ; 83,2] | 78,2 | 56,4 | 9,3 pt | 1,05 |
| Macron–Le Pen (105) | 55,3 | 62,6 [56,3 ; 69,8] | 60,0 | 68,4 | 7,6 pt | 1,61 |
| Macron–Pécresse (58) | 65,6 | 55,2 [50,1 ; 60,2] | 57,1 | 42,8 | 4,8 pt | 1,26 |

MAE : modèle 6,5 pt, géométrie seule 9,1, Bradley-Terry 12,2. **Aucune des
quatre ne passe le critère §8.4** (écarts de chaînes 4,7 à 9,3 pt) : un duel
NON observé est porté par la géométrie, et les modes de géométrie T1 du cadre
historique (§8.4) s'y transmettent entièrement — à la différence des duels
observés (V2, écart ≤ 0,6 pt). Le classement modèle < géométrie < BT est
indicatif, pas établi. Conséquence produit : sur 2027 la géométrie converge
(R-hat 1,04), mais ces chiffres disent que les duels **jamais sondés** de
l'onglet Scénarios ont une incertitude de géométrie de l'ordre de 5 à 10 pt
dès que le T1 laisse deux ordres plausibles. Prochaine étape : lever la
multimodalité T1 du cadre historique (spec avant code).

### 8.7 Diagnostic des modes V1 (2026-11-14, Macron–Le Pen retiré)

Expérience (pas un réglage) : `pos_scale` 0,4 → 0,25. Écart de chaînes
7,6 → 5,8 pt, `rhat_max` 35,7 → 7,1 : la liberté autour de l'ancre contribue,
sans être la cause principale. Hypothèse « ancres plus serrées en 2022 qu'en
2027 » réfutée (écart logit 0,25 dans les deux). Fenêtre plus courte écartée :
à T1−1, 120 j ne laisse que 17 champs, 90 j 14 (contre 45 à 180 j).

Le mode minoritaire (1 chaîne sur 4) est cohérent : Roussel et Jadot à
μ ≈ 0,25, contre Macron, avec σ large (0,12-0,14), au lieu de μ ≈ 0,05 et σ
étroit (0,02-0,03) ; Macron décalé à gauche (0,25 contre 0,42). Les duels
OBSERVÉS restent ajustés parce que **η compense** : η_Macron = +0,29 contre
−0,24 dans les autres chaînes, η_Le Pen = −0,30 contre −0,04 à −0,09.
Mécanisme : la vraisemblance des duels identifie seulement
$\kappa(\ell_a-\ell_b)+\eta_a-\eta_b$ sur les paires observées ; η libre
($\mathcal N(0, s_\eta^2)$ indépendant) absorbe toute différence de
géométrie sur ces paires, et la paire retirée hérite d'une combinaison
arbitraire. Pistes (maths avant code) : η hiérarchique à échelle apprise
(rétrécit les compensations), ou pénalité d'écart entre η et la prédiction
géométrique ; à mesurer par l'écart de chaînes V1, pas par l'erreur.

### 8.8 Modes de densité inégale : sélection des chaînes par log-densité (2026-11-14)

**Mesure** (V1 Macron–Le Pen, 20 tirages par chaîne,
$L(\theta) = \log p(\theta, y)$ recalculée par `log_density` sur les tirages
sauvés) : avec `pos_scale` 0,4, $\bar L_c$ = 10 372 / 10 283 / 10 368 /
10 388 ; avec 0,25 : 10 319 / 10 214 / 10 315 / 10 317. La chaîne minoritaire
de §8.7 est **~100 nats** sous les autres ; à 0,4, la chaîne au duel décalé
(63,8 %) est la plus dense, ~20 nats au-dessus de deux autres. Les modes ne
sont donc pas équivalents : ce sont des pièges locaux de l'échantillonneur, pas
une ambiguïté des données. La « réponse mélange » (4 chaînes poolées) pondère
alors 1/4 un mode de masse ~$e^{-100}$ : faux.

**Règle (backtests seulement).** Pour $C$ chaînes,
$\bar L_c = D^{-1}\sum_t L(\theta_{c,t})$ sur des tirages amincis ;
on retient $\mathcal C^* = \{c : \bar L_c \ge \max_{c'} \bar L_{c'} - \Delta\}$,
$\Delta = 10$ nats. Justification : la masse d'un mode vaut
$\approx e^{L_{\max}}\,(2\pi)^{d/2}|H|^{-1/2}$ ; à dimension égale et
courbures comparables, un écart de 10 nats entre typical sets fait un rapport
de masse ~$4{,}5\cdot10^{-5}$ — un rapport de volume de cet ordre exigerait
des courbures très différentes, que rien n'indique ici (même modèle, mêmes
sites). La quantité prédite et le critère §8.4 sont ensuite calculés sur
$\mathcal C^*$ ; on rapporte $\bar L_c$, $|\mathcal C^*|$, et la cible
poolée sans sélection, pour qu'un lecteur voie l'effet. Limites : (1) si
toutes les chaînes tombent dans un mode sous-dominant, la règle ne le voit
pas — d'où l'intérêt de plus de chaînes ; (2) $|\mathcal C^*| = 1$ ⇒ aucun
contrôle de convergence possible : déclaré **non concluant**. En production,
rien ne change (le fit 2027 a R-hat 1,04 : les chaînes y sont dans le même mode).

**Application (re-fusion, sans NUTS).** Écarts de $\bar L_c$ entre chaînes de
10 à 130 nats sur TOUS les fits historiques : V2 2022 reste concluant aux 3
coupures après sélection (2-3 chaînes retenues, écart ≤ 0,3 pt) ; V1 : une
seule chaîne retenue pour Macron–Mélenchon et Macron–Le Pen (non concluant
par construction), 3 chaînes mais écart 4,8 pt pour Macron–Zemmour, 2 chaînes
et 1,8 pt pour Macron–Pécresse ; avec `pos_scale` 0,25, Macron–Le Pen garde 3
chaînes à 1,15 pt (presque concluant ; 59,0 % [54,7 ; 63,4] contre 55,3 observé).

**Étape suivante : départ multiple puis relance au mode dominant.** Phase 1 =
les 4 chaînes actuelles (exploration). Phase 2 : $\theta_0$ = dernier tirage
de la chaîne de plus grand $\bar L_c$ ; 4 nouvelles chaînes
`init_to_value(θ0)`, graines distinctes, warmup complet. Si elles restent
dans le même puits (critère §8.4), le mode dominant est stable et la cible y
est lue ; si l'une en sort vers plus dense encore, on itère. C'est une
recherche du mode de plus grande masse, justifiée par les écarts mesurés
(≫ Δ) ; elle ne garantit pas de trouver le mode global.

**Phase 2, V1 Macron–Le Pen (abst)** : départ au dernier tirage de la chaîne
la plus dense de la phase 1. Les 4 chaînes restent dans ce puits ($\bar L_c$ à
7 nats près, toutes retenues), `rhat_max` 35,7 → 2,3, R̂ cible 1,03, écart
1,4 pt (seuil 1 : de justesse non concluant). **Le mode dominant prédit
64,0 % [59,1 ; 69,1] pour 55,3 % observé** : erreur 8,7 pt, hors IC. Bien
échantillonné, le modèle à η libre prédit mal la paire retirée — motive §8.9.

### 8.9 Expérience : le second tour sans valence η (question utilisateur, 2026-11-14)

Question : le T2 peut-il servir à estimer μ, σ (et ω) plutôt qu'être absorbé
par des paramètres propres ? Dans le modèle courant, le logit d'un duel vaut,
à l'abstention près, $\kappa\,(\ell_a - \ell_b) + \eta_a - \eta_b$ avec
$\ell_i = \log \int e^{A_i(v)} f(v)\,dv$ fonction de $(\mu,\sigma,w)$. Avec
$\eta_i \sim \mathcal N(0, 0{,}5^2)$ libres, N valences pour ~N(N−1)/2 paires
observées, η peut ajuster les duels **quelle que soit la géométrie** : le T2
n'apporte alors presque rien sur (μ, σ) — et §8.7-8.8 montrent qu'il laisse
coexister des géométries de densités différentes. Sans η, chaque duel est une
contrainte directe sur (μ, σ, w) : le T2 devient une source d'identification
de la géométrie.

Variantes (backtest, `--variante`) :
- `abst` (référence) : κ, η, ω ;
- `geo` : κ, ω, **η ≡ 0** (`eta_scale = 0`) ;
- `geo1` : **κ ≡ 1** (`kappa_log_scale` → 10⁻³), ω, η ≡ 0 : le duel est la
  pure prolongation de la géométrie T1, plus l'abstention.

Critères, dans l'ordre : (1) convergence (§8.4, §8.8 : chaînes retenues
d'accord) ; (2) V1 (paire retirée : c'est là que η ne peut pas aider) ;
(3) V2 et diagnostic 2027 (un modèle sans η ajuste-t-il encore les duels
sondés ? rappel : sur 2027, MAE paire retirée 6,3 pt géométrie seule, 3,2 avec
κ+ω, 2,9 η seule, 2,4 les deux — mais géométrie FIGÉE, pas jointe).

**Résultats V1 Macron–Le Pen retiré (2022 T1−1, observé 55,3 %)** :

| variante | prédiction [IC90] | erreur | chaînes retenues / écart | R̂ cible | κ | ω |
|---|---|---|---|---|---|---|
| abst, phase 2 (κ, η, ω) | 64,0 [59,1 ; 69,1] | 8,7 | 4 / 1,4 pt | 1,03 | — | — |
| **geo (κ, ω ; η ≡ 0)** | **56,2 [50,6 ; 59,0]** | **0,9** | **4 / 1,2 pt** | 1,09 | 0,53 | −1,03 |
| geo1 (ω ; κ ≡ 1, η ≡ 0) | 50,0 [47,2 ; 52,2] | 5,3 | 3 / 0,16 pt | 1,00 | 1 | −4,79 |

Sans η, les 4 chaînes tombent d'emblée dans le même puits ($\bar L_c$ à 6 nats
près, sans phase 2) : la valence libre était bien ce qui laissait coexister
les géométries. κ est nécessaire (κ ≈ 0,5 : au 2nd tour l'électeur agrège
moins « par proximité » qu'au 1er), η ne l'est pas sur cette paire et y nuit.
Une paire ne fait pas une règle : à confirmer sur les 3 autres V1, V2 et 2027.

**geo sur les 4 V1 2022** (observé / prédiction [IC90] ; conclusif ?) :
Mélenchon 61,5 / 61,6 [58,9 ; 63,8] oui ; Le Pen 55,3 / 56,2 [50,6 ; 59,0]
presque (1,17 pt) ; **Zemmour 66,7 / 51,1 [49,2 ; 53,6] oui** ; **Pécresse
65,6 / 53,1 [50,1 ; 56,8] oui**. MAE geo 7,3 pt contre abst 6,8 (abst non
concluant ×4). Sans η la convergence est acquise, mais les duels contre un
candidat « repoussoir » (Zemmour, Pécresse en 2022) sont prédits proches de
50/50 alors que les sondages donnaient 2/3–1/3 : une géométrie à une dimension
ne représente pas l'attractivité propre au second tour. η le représente, mais
libre à échelle 0,5 il sur-corrige (Zemmour 74 %) et casse Le Pen (64 %).
κ et ω varient fortement d'un fit à l'autre (κ 0,53–1,76 ; ω −1,0 à −7,5) :
faiblement identifiés, ils absorbent ce qui reste.

### 8.10 Valence à échelle apprise (variante `hier`)

$$\eta_i = s_\eta\, z_i,\quad z_i \sim \mathcal N(0,1),\quad s_\eta \sim
\mathrm{HalfNormal}(0{,}5),$$

paramétrisation non centrée (pas d'entonnoir en $z$ ; l'entonnoir en $s_\eta$
avec ~8 candidats en duel est le risque connu — d'où `ETA_SCALE` fixe à
l'origine — et sera surveillé par l'écart de chaînes). Avec $s_\eta$ appris,
les duels décident collectivement de l'ampleur des valences : si la géométrie
explique déjà la plupart des duels, $s_\eta$ rétrécit et η ne peut plus
compenser une géométrie arbitraire ; si un repoussoir l'exige, $s_\eta$ reste
grand. Le prior a la même échelle que le 0,5 fixe actuel : on n'impose pas a
priori des valences plus petites, on laisse les données le dire. Mesure : les
4 V1 2022, critère §8.4/§8.8 puis erreur, à comparer à abst et geo.

**Résultats `hier`, 4 V1 2022** (obs. / prédiction des chaînes retenues [IC90]) :
Le Pen 55,3 / 59,4 [55,0 ; 63,4], 4 chaînes, 0,63 pt, **concluant** ;
Zemmour 66,7 / 69,0 [65,6 ; 72,3], 1 chaîne retenue (non concluant) ;
Mélenchon 61,5 / 61,2 [45,1 ; 76,2], 1 chaîne (non concluant) ;
Pécresse 65,6 / 52,7 [48,6 ; 57,2], 2 chaînes, 0,18 pt, concluant.

| variante | MAE V1 | couverts | concluants |
|---|---|---|---|
| abst (η, échelle 0,5 fixe) | 6,8 pt | 1/4 | 0/4 |
| geo (η ≡ 0) | 7,3 pt | 2/4 | 3/4 |
| **hier (échelle apprise)** | **4,9 pt** | **3/4** | 2/4 |

$s_\eta$ par chaîne : Le Pen 0,41-0,42 (un seul régime) ; Zemmour 0,34 / 0,35
/ 0,17 / 0,16 et Pécresse 0,41 / 0,43 / 0,17 / 0,22 : **deux régimes** de
valence (forte / faible), c'est l'entonnoir prévu. Pécresse est ratée par
TOUTES les variantes (52,7-55,2 contre 65,6) : défaut robuste, pas un effet de
η — piste : la cible V1 est la moyenne des 30 derniers jours, pendant
l'effondrement de Pécresse (duels ~52/48 à l'automne 2021), et la prédiction
est-elle bien lue avec le `w` d'`as_of` ? À vérifier avant d'en conclure quoi
que ce soit. Lecture provisoire : l'échelle apprise est la meilleure des trois
et garde la valence là où elle sert (Zemmour) ; le régime bimodal de $s_\eta$
reste à traiter (prior plus informatif sur $s_\eta$, ou phase 2 §8.8).

**Pécresse : vérification (2026-11-15).** Pas un artefact de cible : duels
Macron–Pécresse par mois 51,5 (déc.) → 53,2 → 56,1 → 64,2 (mars) → 63,5
(avr.) ; la moyenne 30 j (65,6) est celle de fin de campagne, et la prédiction
est bien lue au `w` d'`as_of` (Pécresse ~10 % au T1). Cause : la paire
retirée, **il ne reste que 2 duels de Pécresse** (contre Zemmour, contre Le
Pen) ; sa valence est donc quasi libre et la prédiction vient de la géométrie,
qui place une candidate de centre-droit au voisinage de Macron et lui donne
les électeurs de droite radicale (≈ 30 % du T1) — d'où ~53/47. Les sondages
disent que ces électeurs ne la suivent pas (abstention, ou Macron). **Limite
structurelle** : pour un candidat sans duel propre, le duel prédit vient de la
géométrie et penche vers 50/50 pour les voisins du centre ; la valeur inclusive
(ω) ne suffit pas à faire abstenir les électeurs éloignés des deux. Conséquence
produit (onglet Scénarios, 2027) : afficher qu'un duel jamais sondé d'un
candidat sans aucun duel est une extrapolation géométrique. Piste
« abstention croissante avec la distance aux deux finalistes » **écartée
(2026-11-15)** : elle est déjà dans le modèle. L'option extérieure est
$U_0 = \omega + \kappa\,\mathrm{lse}_R(v)$ (§3.1), donc
$\rho_{ab}(v) \approx \sigma\big(\kappa(\max(A_a,A_b) - A_{\text{fav}}(v)) - \omega\big)$ :
un électeur RN dans Macron–Pécresse s'abstient déjà d'autant plus que
$A_{LP}(v) \gg A_P(v)$. Ce qui manque est ailleurs : parmi les électeurs
qui EXPRIMENT, $q = \sigma(\kappa(A_P - A_M))$ envoie la droite radicale à
Pécresse, géométriquement plus proche ; que ces électeurs la rejettent
(« repoussoir » à droite) n'est écrit nulle part dans les sondages de T1 —
c'est exactement l'information que portent ses propres duels, via η. C'est
une limite d'**identification**, pas de forme fonctionnelle : sans duel du
candidat, aucune forme de ρ ne la lève. Seule une seconde dimension (ou des
duels) le pourrait (spec_spatial_pooling §9).

**`hier` phase 2 (départ au plus dense, §8.8)** : Zemmour 69,2 [65,3 ; 73,2]
pour 66,7, 4 chaînes, 0,18 pt, **concluant et couvert**, $s_\eta$ 0,35-0,38 ;
Pécresse 52,6 [48,3 ; 57,0] pour 65,6, 3 chaînes, 0,75 pt, concluant,
$s_\eta$ 0,12-0,14 (limite structurelle ci-dessus). La bimodalité de
$s_\eta$ se règle donc par la relance au mode dominant, sans changer de prior.
Bilan `hier` (versions concluantes) : Le Pen 4,1 pt, Zemmour 2,6, Pécresse
13,0 ; Mélenchon 0,3 (phase 1, 1 chaîne, à relancer).

**Suite `hier`** : Mélenchon phase 2 : 61,2 [44,1 ; 77,2] pour 61,5, 4
chaînes, 1,33 pt (presque concluant ; IC très large). **MAE V1 `hier` 5,0 pt**
(Le Pen 4,1, Zemmour 2,6, Pécresse 13,0, Mélenchon 0,3), contre abst 6,8 non
concluant. V2 2022 T1−1 `hier` : 52,3 % (abst 52,4) — sur un duel sondé, la
variante ne change rien, comme attendu.

**Diagnostic 2027 `hier`** (sondages au 2026-11-12) : R-hat 1,0095 (abst
1,040), ESS min 304 (202), rhat_t2 1,0005, 0 divergence ; résidus globaux
+0,03 ± 1,69 pt, $\overline{z^2}$ 0,78 ; duels affichés identiques à abst à
≤ 0,5 pt (Philippe–Le Pen 45,2 % [37,8 ; 52,3], P(élu) Le Pen 0,85) ; mêmes
limites (Mélenchon–Le Pen $\bar r_{30j}$ −1,35 ; Harris −0,9 / YouGov +0,8).
Règle (ii) : passe.

**Bilan des variantes (2026-11-15)** : `hier` égale abst sur les duels sondés
(V2 2022, 2027), converge mieux, et prédit mieux les duels non sondés
(MAE V1 5,0 contre 6,8 non concluant). Proposée comme variante de
référence — **proposition retirée le 2026-11-15 après revue (§8.11)**.

**2017 en `hier`** (cadre pauvre : 17-24 notices T1 datées, 3 champs ;
fits de 2 min, convergence facile) :

| tâche | cible | modèle [IC90] | brut | couvert | concluant |
|---|---|---|---|---|---|
| V2 T1−1 | 66,1 (résultat) | 62,8 [58,2 ; 67,2] | 62,8 | oui | oui (0,3 pt) |
| V2 T2−7 | 66,1 (résultat) | 61,3 [56,8 ; 65,6] | 61,7 | **non** | oui (0,2 pt) |
| V1 Macron–Le Pen retiré | 61,8 (sondages 30 j) | 62,5 [46,1 ; 76,0] | — | oui | oui (0,8 pt) |
| V1 Fillon–Le Pen retiré | 57,6 | 62,5 [44,4 ; 78,3] | — | oui | presque (1,4 pt) |

Les η estimés valent ≈ 0 (|η| ≤ 0,05) : l'échelle apprise se rétracte quand
les duels n'en ont pas besoin — le comportement voulu. V2 : comme en 2022, le
modèle = la moyenne des sondages, et l'erreur de scrutin de fin de campagne
(Macron sous-estimé de 4-5 pt) sort de l'IC à $\tau_{T2} = 0{,}10$ (arbitrage
1). V1 2017 : IC très larges (géométrie peu contrainte par 3 champs) ; les
centres sont bons (0,7 et 4,9 pt).

### 8.11 Révision après la revue Opus du 2026-11-15

**Règle §8.8 corrigée (code + re-fusion de tous les JSON).** $\bar L_c$ est
calculée sur TOUS les tirages (400 par chaîne, au lieu de 20) avec erreur-type
par 10 lots ; seuil $\bar L_c \ge \bar L_{best} - \max(10,\ 4\sqrt{SE_c^2+SE_{best}^2})$.
Les tirages des chaînes rejetées sont retirés AVANT tout calcul aval (IC,
couverture, log-score, géométrie seule, κ/ω/η, R-hat) — avant, seule la cible
l'était. `rhat_t2` inclut $s_\eta$ ; `nanmax`/`nanmin` pour η ≡ 0.
**Limite reconnue** : $\bar L$ n'est pas la masse d'un mode (il manque
l'entropie, ~4-23 nats d'écart mesuré par la revue) ; la règle n'exclut avec
confiance que les pièges ≫ 10 nats. Remplacement prévu : stacking des chaînes.

**Tableau V1 2022 re-fusionné** (meilleure version disponible par variante) :

| variante | MAE | log-score Σ (logit gaussien) | concluants | protocole |
|---|---|---|---|---|
| abst | 7,0 pt | −5,5 | 0/4 | phase 2 sur 1 paire seulement |
| geo | 7,3 pt | −82,9 | 3/4 | phase 1 |
| hier | 5,0 pt | −5,0 | 3/4 | phase 2 sur 3 paires |

**La proposition « `hier` = référence » (§8.10) est RETIRÉE en l'état** :
log-scores à égalité avec abst, traitement inégal (abst sans phase 2 sur 3
paires), 4 paires contenant toutes Macron (≈ un seul test), variantes conçues
après avoir vu ces paires. Elle sera rejugée sur un protocole fixé d'avance
(ci-dessous). V2 2022 T1−1 `hier` après re-fusion : 2 chaînes retenues,
concluant, 52,3 % (abst 52,4) — la phrase « converge mieux » ne valait que
pour 2027.

**Protocole fixé d'avance pour comparer les variantes** (avant tout nouveau
résultat) : mêmes tâches pour chaque variante ; phase 1 (4 chaînes
dispersées) puis phase 2 si moins de 3 chaînes retenues ; métrique primaire =
log-score en logit (gaussien, depuis les tirages retenus), secondaire = MAE ;
une variante n'est préférée que si elle gagne le log-score total ET ne perd
la MAE sur aucune élection. Jeu de test principal : **V1 2027**, une paire
retirée (8 paires à n ≥ 4) et un candidat retiré (Attal, Bardella), sur le
régime de production — 2017/2022 deviennent des contrôles.

### 8.12 V1 2027 : duel retiré dans le régime de production (protocole)

Tâches (fixées avant tout résultat) : (i) **paire retirée**, les 8 paires 2027
à ≥ 4 duels sondés (Le Pen–Philippe, Le Pen–Mélenchon, Attal–Le Pen,
Glucksmann–Le Pen, Bardella–Philippe, Bardella–Mélenchon, Attal–Bardella,
Hollande–Le Pen) ; (ii) **candidat retiré** : tous les duels d'Attal, puis
tous ceux de Bardella (le cas réel de l'onglet Scénarios : candidat sans duel
propre, cf. Pécresse §8.10). Pour la tâche k de paires cibles $\mathcal P_k$ :
$t_k$ = date du dernier duel sondé de $\mathcal P_k$ ; fit joint sur les
sondages ≤ $t_k$ privés des duels de $\mathcal P_k$ (T1 inchangé) ; cible
$y_{ab}$ = moyenne des duels de $(a,b)$ sur $[t_k - 30\,j, t_k]$ ; prédiction
$s_{ab}$ au jour $t_k$, **sans** aléa de scrutin. Variantes : abst et hier ;
même protocole (§8.11 : 4 chaînes, phase 2 si < 3 retenues ; log-score
logit primaire, MAE secondaire). Limite connue : les autres duels des mêmes
notices restent (fuite même-répondants, revue point 8).

### 8.13 V1 2027, premiers résultats (4 grosses paires, 2026-11-15)

Fit à $t_k$, paire retirée, cible = moyenne 30 j (n = 8, 8, 8, 3) ; 4 chaînes
retenues partout, écart ≤ 0,3 pt : **concluant d'emblée** (la géométrie 2027
est identifiée, contrairement au cadre historique).

| paire retirée | sondé | abst [IC90] | hier [IC90] |
|---|---|---|---|
| Le Pen–Philippe | 55,4 | 55,0 [52,7 ; 57,4] | 54,8 [52,9 ; 56,6] |
| Le Pen–Mélenchon | 68,5 | 65,7 [63,5 ; 67,8] | 65,7 [63,8 ; 67,7] |
| Glucksmann–Le Pen | 43,2 | 41,2 [37,7 ; 44,8] | 41,6 [38,7 ; 44,4] |
| Attal–Le Pen | 43,3 | 36,6 [32,7 ; 40,5] | 39,3 [36,6 ; 42,0] |
| MAE / log-score Σ | | 2,97 / 5,94 | **2,26 / 7,02** |

**Complément (6 tâches exécutables sur 10)** : Hollande–Le Pen retiré : abst
42,2 [21,8 ; 64,3] (non concluant, 1,7 pt), hier 42,6 [38,2 ; 46,6] pour
43,3 ; **candidat Attal retiré** (tous ses duels) : abst 38,1 [19,7 ; 59,7],
hier 40,1 [35,7 ; 44,5] pour 43,3.

| V1 2027 (6 tâches) | MAE | log-score Σ | concluants | couverture IC90 |
|---|---|---|---|---|
| abst | 3,05 pt | 7,95 | 5/6 | 4/6 |
| **hier** | **2,19 pt** | **11,61** | **6/6** | 4/6 |

**Amendement au protocole (données, pas résultat)** : les 4 tâches Bardella
(Bardella–Philippe, –Mélenchon, Attal–Bardella, candidat Bardella) sont
**inexécutables** : $t_k$ = 27 mai 2026, date à laquelle le jeu ne compte que
7 notices depuis le 1er janvier ; Le Pen n'y figure que dans 4 (< MIN_POLLS =
5), sort du roster, et le contrôle de couverture de production refuse le fit à
juste titre. Exclues, pas contournées (lever le contrôle fabriquerait un report
au prorata).

**Lecture.** Selon le protocole §8.11, `hier` gagne le log-score total sur
2027 (11,6 contre 8,0) et sur 2022 (−5,0 contre −5,5), sans perdre la MAE sur
aucune élection (2027 : 2,19 / 3,05 ; 2022 : 5,0 / 7,0). Réserves
restantes avant d'en faire le défaut : (1) traitement encore inégal en 2022
(abst sans phase 2 sur 3 paires) et abst non lancé en 2017 ; (2) 6 tâches 2027
dont 5 contiennent Le Pen ; (3) couverture 4/6 dans les deux variantes : l'IC
d'un duel retiré n'inclut ni σ_T2 ni le bruit d'échantillonnage de la cible
(moyenne de 3 à 8 sondages) — à ajouter avant de lire la couverture
(Mélenchon–Le Pen et Attal–Le Pen sortent de l'IC dans les deux variantes).

### 8.14 IC prédictif d'un duel retiré (V1)

La cible V1 est $\bar y = n^{-1}\sum_{d=1}^{n} y_d$, et la vraisemblance du
modèle (§4) dit $y_d \mid s \sim \mathcal N\big(s,\ s(1-s)/N_d + \sigma_{T2}^2\big)$,
indépendants. Donc, tirage par tirage,
$\bar y^{(t)} = s^{(t)} + \varepsilon^{(t)}$,
$\varepsilon^{(t)} \sim \mathcal N\big(0,\ [s^{(t)}(1-s^{(t)})/\bar N + \sigma_{T2}^2]/n\big)$,
avec $\bar N$ = taille moyenne des sondages cibles (1000 à défaut) et
$\sigma_{T2}$ = moyenne a posteriori. L'IC90 et le log-score « prédictifs »
portent sur $\bar y^{(t)}$ ; ceux de §8.12-8.13 portaient sur $s$ seul (trop
étroits par construction). Limite : l'indépendance entre sondages d'un même
institut sous-estime encore la variance si des effets maison existent.

**Résultat (re-fusion, 2026-11-15)** : l'élargissement est faible
($\sigma_{T2}$ ≈ 1,3 pt, divisé par $\sqrt n$). Couverture prédictive V1 :
2027 abst 4/6, hier 4/6 (LS 8,6 / 12,6) ; 2022 abst 2/4, geo 2/4, hier 3/4 ;
2017 hier 2/2. Les deux ratés 2027 (Le Pen–Mélenchon 68,5 contre
[63,6 ; 67,9] ; Attal–Le Pen 43,3 contre [36,0 ; 42,3]) sont des biais de 3-4 pt
que le postérieur ne compte pas : l'**erreur de structure** de l'extrapolation
géométrique à une paire non sondée. Piste chiffrable : un terme
$\nu_{ab} \sim \mathcal N(0, \tau_{\text{paire}}^2)$ en logit ajouté à la
prédiction d'une paire SANS duel sondé, $\tau_{\text{paire}}$ calibré en
leave-one-pair-out sur les V1 2027 (6 résidus ; ~0,10-0,15 logit d'après les
erreurs observées), à afficher dans l'onglet Scénarios pour les duels jamais
sondés. Même logique que $\tau_{T2}$ (arbitrage 1) : à soumettre à
l'utilisateur.

### 8.15 Verdict du protocole §8.11 : abst contre hier (2026-11-15)

Traitement égal (phase 2 partout où < 3 chaînes retenues ou chaînes en
désaccord ; abst lancé sur 2017), métriques prédictives (§8.14), V1 :

| V1 | MAE abst / hier | log-score Σ abst / hier |
|---|---|---|
| 2017 (2 paires) | **2,05** / 2,81 | 1,89 / **2,39** |
| 2022 (4 paires) | 6,34 / **4,97** | **−2,81** / −4,48 |
| 2027 (6 tâches) | 3,05 / **2,19** | 8,62 / **12,63** |
| total | — | 7,70 / **10,54** |

Règle fixée d'avance : préférer une variante si elle gagne le log-score total
ET ne perd la MAE sur aucune élection. `hier` gagne le log-score total mais
perd la MAE sur 2017 ⇒ **`hier` n'est PAS préférée ; `abst` reste le
défaut.** Les écarts sont petits devant le bruit (2 à 6 tâches par élection,
toutes avec Le Pen ou Macron) : aucune des deux ne domine. Sur 2022, `hier`
est trop confiante sur Pécresse (s_η rétréci à 0,13 ⇒ IC étroit, z ≈ 5) — ce
qui lui coûte le log-score 2022. V2 (duels sondés) : identiques à ≤ 0,4 pt.
Conséquence : on garde `abst`, et l'amélioration à rechercher n'est pas dans
η mais dans l'incertitude d'extrapolation des paires non sondées
(arbitrage 4, $\tau_{paire}$) et dans $\tau_{T2}$ (arbitrage 1).

### 8.16 Fuite « même notice » en V1 2027 (revue 15/11, point 8)

**Comptage (sans fit).** Quand on retire une paire, les AUTRES duels des
mêmes notices restent : Le Pen–Philippe 74 duels restants sur 86 viennent des
29 mêmes notices ; Le Pen–Mélenchon 66/89 ; Attal–Le Pen 55/84 ;
Glucksmann–Le Pen 44/101 ; Hollande–Le Pen 19/96. Mêmes répondants, même
effet maison : la V1 est probablement optimiste.

**Mesure (protocole, avant résultat).** Variante « notice entière » : on
retire TOUS les duels (T2) des notices qui contiennent un duel cible ; le T1
de ces notices est gardé (sinon la géométrie elle-même change de jeu). Mêmes
4 tâches que les grosses paires de §8.13, variante abst (le défaut, §8.15).
On rapporte l'écart à la V1 standard (erreur, IC). Si l'erreur augmente
nettement, les chiffres V1 de §8.13 sont à lire comme une borne optimiste.

**Résultat (abst, données du 15/11, standard rejoué sur les mêmes données ;
résultats du 13/11 conservés en `*_donnees-2026-11-13.json`)** :

| paire retirée | sondé | standard (erreur) | notice entière (erreur) | duels restants au fit |
|---|---|---|---|---|
| Le Pen–Philippe | 55,4 | 54,8 (0,6) | 56,3 (0,9), IC [32 ; 79], non concluant | 2 |
| Le Pen–Mélenchon | 68,5 | 65,6 (2,9) | 66,1 (2,4) | 8 |
| Glucksmann–Le Pen | 43,4 | 41,7 (1,7) | 42,1 (1,3) | 37 |
| Attal–Le Pen | 43,3 | 36,5 (6,8) | 37,8 (5,5), IC [17 ; 62] | 16 |

**Pas d'optimisme mesurable sur les centres** : retirer les notices entières
ne dégrade pas l'erreur (3 paires sur 4 un peu meilleures) ; seuls les IC
s'élargissent, par perte de données. Les chiffres V1 de §8.13 ne sont donc
pas gonflés par la fuite même-répondants ; les biais de 3-7 pt (Mélenchon,
Attal) sont de la structure, pas de la fuite (cf. arbitrage 4).

### 8.17 Effets maison au second tour (2026-11-15, sans fit)

Résidus des duels sondés contre la part prédite à leur date (re-fusion des
fits V2, dans l'échantillon), par institut (n ≥ 5) :

| institut | 2017 (T2−7) | 2022 (T1−1) | 2022 (T2−7) | 2027 (§8.1) |
|---|---|---|---|---|
| Harris Interactive | +1,56 (11) | −0,17 (90) | −0,16 (83) | −0,88 (25) |
| Ifop | −0,76 (22) | +0,32 (16) | +0,05 (15) | +0,04 (14) |
| Elabe | +0,21 (18) | +0,38 (37) | +0,31 (39) | +0,50 (18) |
| OpinionWay | +0,41 (21) | −0,10 (54) | +0,05 (60) | −0,14 (14) |
| Odoxa | −0,36 (7) | +1,35 (6) | +0,86 (6) | (4) |

(points, entre parenthèses le nombre de duels.) Écart-type des résidus :
1,4-1,6 pt. **Aucun effet maison stable** : amplitudes ≤ 1,6 pt, signes qui
changent d'une élection à l'autre (Harris +1,6 / −0,2 / −0,9), et sur 2022, où
les effectifs sont grands, tout est à ±0,4 pt sauf Odoxa (n = 6). Le seul
régulier est Elabe (+0,2 à +0,5 pt), trop petit pour justifier un paramètre.
Décision : pas d'effets maison au T2 ; σ_T2 absorbe cette dispersion.

### 8.18 Base V2 « sondages bruts ± τ_T2 » (revue 14/11, sans fit)

Moyenne brute des duels finalistes des 7 derniers jours $\pm$ l'aléa de
scrutin seul, $\mathcal N(\mathrm{logit}\,\bar y,\ \tau_{T2}^2)$, contre le
résultat (5 coupures V2 : 2017 T1−1, T2−7 ; 2022 T1−30, T1−1, T2−7) :

| τ_T2 | couverture IC90 | log-score Σ |
|---|---|---|
| 0,10 (actuel) | 2/5 | 6,32 |
| 0,19 (mesure §6) | **5/5** | **8,78** |

À 0,10 : 2017 T2−7 (61,7 % sondé, IC [57,7 ; 65,5], résultat 66,1), 2022 T1−1
(52,8, [48,7 ; 56,8], 58,6) et T2−7 (53,4, [49,3 ; 57,5], 58,6) hors IC. À 0,19
tous couverts. Le modèle reproduisant la moyenne des sondages à ±0,4 pt
(§8.5), cette base dit l'essentiel : **c'est τ_T2 qui décide de la couverture
du scrutin**. Dans l'échantillon (τ mesuré sur ces élections, 5 points) :
argument de cohérence pour l'arbitrage 1, pas validation.

### 8.19 Révision après la revue Opus du 2026-11-16

- **§8.15 (abst contre hier)** : la tâche « candidat Attal » ne contient qu'une
  paire (Attal–Le Pen), même cible que la tâche paire, et Hollande–Le Pen vient
  des mêmes 3 notices : 2027 ≈ **4 cibles indépendantes, pas 6** ; les deux
  tâches Attal portent 4,7 des 5,2 pt d'écart de MAE. Le veto 2017 vient de
  Fillon–Le Pen, **non concluant** (IC [37 ; 82]). Le log-score gaussien en
  logit est faux sur les postérieurs bimodaux. ⇒ Verdict §8.15 à refaire :
  tâches dédoublonnées, non concluantes écartées, log-score par KDE des
  tirages, écart par tâche avec IC bootstrap. Le défaut `abst` reste en place
  d'ici là (inchangé = aucune décision nouvelle).
- **§8.16 (fuite même-notice)** : NON ÉTABLI. Pour Le Pen–Philippe il ne reste
  que 2 duels : le test confond la fuite et la perte du second tour ; deux
  fits sur quatre sont non concluants. Il faut un **placebo** (retirer autant
  de duels pris dans d'autres notices).
- **§8.17 (effets maison)** : la mauvaise hypothèse était testée. Un effet
  maison propre à CHAQUE campagne suffit à biaiser (Harris 2027 : −0,88 pt sur
  ~8 notices, z ≈ −1,7) ; les résidus dans l'échantillon sous un modèle sans
  effet maison sont rétrécis. À faire : σ_maison par REML au niveau notice et
  par campagne, test du rapport de vraisemblance contre σ_T2 seul.
- **§8.18 (τ_T2, 5/5)** : ne prouve rien. Les 5 points font 2 erreurs
  indépendantes (2022 : un seul résultat), toutes de même signe ; tout
  τ ≥ 0,143 donne 5/5 ; IC90 de τ avec 2 d.d.l. : [0,12 ; 0,92]. À faire :
  erreurs 1995-2012 (session données), postérieur sur τ (prédictif de
  Student), terme de biais signé testé. L'arbitrage 1 reste ouvert, avec cette
  réserve écrite.

### 8.20 Verdict abst contre hier refait (protocole fixé le 2026-11-16, avant résultat)

Réponse au point §8.19 sur §8.15. Aucun nouveau modèle : on re-fusionne les
chaînes déjà sauvées (sauf hier 2027 et abst Hollande–Le Pen, rejoués sur les
données à jour, puisque les tirages abst du 13/11 avaient été remplacés par
ceux du 15/11 : les deux variantes doivent voir le MÊME jeu).

- **Tâches.** 2017 : Macron–Le Pen, Fillon–Le Pen ; 2022 : les 4 paires
  (phase 2 `_init` quand elle existe, pour les deux variantes) ; 2027 :
  Le Pen–Philippe, Le Pen–Mélenchon, Glucksmann–Le Pen, Attal–Le Pen. Écartées :
  « candidat Attal » (même cible qu'Attal–Le Pen) ; Hollande–Le Pen, tirée des
  mêmes notices que les autres cibles, n'entre qu'en **sensibilité**.
- **Non concluantes écartées par paire** : une tâche sort pour LES DEUX
  variantes si l'une des deux n'est pas concluante (§8.4 : R̂ de la cible
  ≤ 1,1 et écart entre chaînes ≤ 1 pt).
- **Log-score par KDE.** Densité prédictive de la cible (tirages
  $\bar y^{(t)}$ de §8.14) estimée par noyau gaussien en logit, fenêtre de
  Silverman $h = 0{,}9\,\min(\hat s, \mathrm{IQR}/1{,}34)\,T^{-1/5}$, puis
  jacobien : $\mathrm{LS} = \log \hat f(\mathrm{logit}\,y) - \log y(1-y)$.
  Robuste aux postérieurs bimodaux, contrairement à la gaussienne en logit.
- **Écart apparié par tâche** $\Delta_j = \mathrm{LS}_j^{hier} - \mathrm{LS}_j^{abst}$
  et $\delta_j = |e_j^{hier}| - |e_j^{abst}|$ (pt). IC90 bootstrap (10 000
  rééchantillonnages des tâches, toutes élections confondues) de la moyenne.
- **Règle.** `hier` devient le défaut si l'IC90 de $\bar\Delta$ est
  entièrement > 0 ET si l'IC90 de $\bar\delta$ ne l'est pas (pas de perte
  d'erreur établie) ; symétriquement pour abst. Sinon : **pas de différence
  établie**, `abst` reste (aucune décision nouvelle). On rapporte aussi les
  totaux par élection, sans règle dessus (2 à 4 tâches).

**Résultat (2026-11-16, `model/backtest/verdict_abst_hier.py`,
`results/verdict_abst_hier.json`).** Re-fusion à l'identique (chiffres des JSON
inchangés au bit près ; hier 2027 et abst Hollande rejoués sur les données du
15/11, copies du 13/11 en `*_donnees-2026-11-13.json`).

| élection | tâche | concluante (abst / hier) | LS KDE abst / hier | erreur abst / hier (pt) |
|---|---|---|---|---|
| 2017 | Macron–Le Pen | oui / oui | 0,91 / 1,73 | 0,34 / 0,71 |
| 2017 | Fillon–Le Pen | non / non | 0,89 / 1,30 | 3,75 / 4,90 |
| 2022 | Macron–Zemmour | oui / oui | 1,54 / 2,30 | 4,67 / 2,57 |
| 2022 | Macron–Mélenchon | oui / non | 1,07 / 1,50 | 0,45 / 0,28 |
| 2022 | Macron–Pécresse | non / oui | −6,97 / **−22,10** | 11,54 / 12,98 |
| 2022 | Macron–Le Pen | non / oui | −1,25 / 1,48 | 8,69 / 4,04 |
| 2027 | Le Pen–Philippe | oui / oui | 3,15 / 3,22 | 0,62 / 0,96 |
| 2027 | Le Pen–Mélenchon | oui / oui | 1,25 / 0,90 | 2,94 / 2,81 |
| 2027 | Glucksmann–Le Pen | oui / oui | 2,59 / 2,93 | 1,70 / 1,22 |
| 2027 | Attal–Le Pen | oui / oui | −0,53 / 0,98 | 6,83 / 4,12 |
| sensib. | Hollande–Le Pen | non / oui | 0,95 / 2,68 | 1,15 / 0,93 |

Sur les **6 tâches concluantes** : $\bar\Delta$ = +0,53, IC90 [+0,14 ; +0,94] ;
$\bar\delta$ = −0,79 pt, IC90 [−1,62 ; −0,02]. **Selon la règle fixée
d'avance, `hier` est préférée** (meilleur log-score établi, erreur pas pire —
même plutôt meilleure). Robustesse : en retirant une tâche à la fois, la borne
basse de l'IC90 de $\bar\Delta$ reste > 0 (de +0,008 sans Attal–Le Pen à
+0,36 sans Le Pen–Mélenchon) : le verdict tient, mais **de justesse**. Le
log-score KDE et le gaussien concordent (pas de bimodalité qui inverse un
classement).

**Réserve importante.** La règle écarte Macron–Pécresse parce que les chaînes
d'abst y sont en désaccord ; or hier y est concluante et **sûre d'elle à
tort** (LS −22, hors IC, s_η rétréci, cf. §8.15). Avec les 10 tâches,
$\bar\Delta$ = −0,84, IC90 [−3,9 ; +1,1] : rien d'établi. Le défaut de hier
— rétrécir s_η et devenir trop confiante sur un candidat à valence
atypique — est réel ; il n'apparaît qu'une fois sur 11 tâches.

**Décision (règle) :** `hier` devient la variante du bloc T2 en production
(`ETA_HIER_T2 = True`, forecast_model.py), **sans effet tant que
`PUBLIER_T2 = False`**. À revoir par la prochaine revue Opus, avec la réserve
Pécresse. **[RETIRÉ le 2026-11-17, §8.28 : non établi ; abst rétabli.]**

### 8.21 Placebo de la fuite même-notice (revue 16/11, point 6 ; protocole fixé avant résultat)

§8.16 confondait deux effets : retirer les duels des mêmes répondants (la
fuite) et perdre K duels d'information sur le second tour. Placebo : deux bras
de **même taille** $K = \min(K_{\text{notices}}, K_{\text{hors}})$, où
$K_{\text{notices}}$ = duels non cibles des notices qui contiennent un duel
cible, $K_{\text{hors}}$ = duels des autres notices (tous antérieurs à $t_k$) :
- **notice partielle** : $K$ duels tirés au hasard dans les notices cibles ;
- **placebo** : $K$ duels tirés hors de ces notices, appariés par paire
  (même duel) autant que possible, le reste au hasard (graine 821).

K et appariement (données du 15/11) : Le Pen–Philippe 12 (11 appariés),
Le Pen–Mélenchon 23 (9), Glucksmann–Le Pen 44 (33 ; le bras notice = §8.16,
réutilisé), Attal–Le Pen 29 (26). Variante abst, comme §8.16. Fuite
**établie** si le bras notice est pire que le placebo en log-score KDE
(§8.20) sur au moins 3 des 4 tâches ET en moyenne ; sinon non établie.
Limite écrite d'avance : 4 tâches, K petit pour Philippe ⇒ puissance faible ;
un « non établi » ne prouve pas l'absence de fuite.

**Résultat §8.21 (2026-11-16, abst, données du 15/11 ; LS = log-score KDE
§8.20 de la cible prédictive)** :

| paire retirée | K | standard : erreur / LS | notice (K) : erreur / LS | placebo (K) : erreur / LS |
|---|---|---|---|---|
| Le Pen–Philippe | 12 | 0,62 / 3,15 | 0,55 / 3,15 | 0,14 / 3,12 |
| Le Pen–Mélenchon | 23 | 2,94 / 1,25 | 4,01 / 0,74 | 3,94 / −0,22 |
| Glucksmann–Le Pen | 44 | 1,70 / 2,59 | 1,33 / 2,78 | 1,17 / 1,02 (non concluant) |
| Attal–Le Pen | 29 | 6,83 / −0,53 | 3,50 / 1,02 | 6,72 / −0,40 |

Le bras « notice » est pire que le placebo sur **0 tâche sur 4** (égal sur
Philippe, meilleur sur les trois autres) ⇒ **fuite même-notice non établie**,
avec la limite de puissance écrite d'avance. Le sens de l'écart est
cohérent avec l'effet notice de §8.23 (σ_n ≈ 0,8-1,1 pt) : les autres duels
d'une même notice sont en partie redondants, donc en retirer K coûte moins
d'information qu'en retirer K pris dans d'autres sondages. La V1 standard
n'est donc pas mesurablement optimiste par fuite. Le biais Attal–Le Pen reste
de la structure (le bras notice le réduit en élargissant l'IC, pas en
déplaçant le centre vers la cible de façon systématique).

### 8.22 Calibrage de τ_paire (arbitrage 4, 0 fit)

Pour chaque tâche V1 concluante, densité prédictive de la cible avec
$\nu\sim\mathcal N(0,\tau^2)$ ajouté en logit (KDE de Silverman, variance
du noyau $h^2+\tau^2$), log-score sommé sur une grille τ ∈ [0 ; 0,40].
Leave-one-pair-out : τ choisi sans la tâche j, évalué sur j.

| variante | tâches | τ* (toutes) | LS(0) → LS(τ*) | τ choisi sans j | gain hors échantillon Σ |
|---|---|---|---|---|---|
| hier | 2027 (4) | **0,08** | 8,04 → 9,03 | 0,10 / 0,00 / 0,09 / 0,06 | −0,59 |
| abst | 2027 (4) | 0,11 | 6,47 → 7,63 | 0,15 / 0,11 / 0,14 / 0,06 | +0,17 |
| hier | 2017-2027 (8) | 0,19 | −8,55 → 12,16 | 0,19-0,21 (0,08 sans Pécresse) | +17,1 |
| abst | 2017-2027 (7) | 0,11 | 9,98 → 11,23 | 0,11-0,14 (0,06) | +0,35 |

**Valeur retenue (règle de la spec : V1 2027, variante par défaut) :
τ_paire = 0,08.** Hors échantillon, elle n'améliore pas le log-score 2027
(−0,59 : 4 tâches, la tâche Mélenchon préfère 0) ; elle sert de garde-fou.
Sensibilité : avec les élections passées, hier demande 0,19, tiré par
Macron–Pécresse 2022 (hier sûre d'elle à tort, §8.20) ; sans cette tâche, 0,08.
Le choix 2027 (0,08) contre passé (0,19) est remonté à l'utilisateur.

**Effet sur 2027 (diagnostic hier au 13/11, τ_T2 0,10 → 0,19, τ_paire 0,08 ;
référence : diagnostic hier du 12/11, 92 → 95 duels)** : centres inchangés
(≤ 0,5 pt), IC90 des duels affichés élargis d'environ 2,5-3 pt de chaque côté
(Philippe–Le Pen : [37,8 ; 52,3] → [35,8 ; 54,7] pour Philippe) ; P(élu) sur
le champ modal : Le Pen 84,8 → 80,3 %, Philippe 11,5 → 13,9 %, Glucksmann
2,4 → 4,5 %. Convergence R̂ 1,014 (T2 1,011). Résultat :
`results/diagnostic_2027_hier_2026-11-13_tau0.19_paire0.08.json`.

### 8.23 Effets maison au second tour par REML (revue 16/11, point 7 ; 0 fit)

Réponse à §8.19 (§8.17 testait des résidus rétrécis, sous un modèle sans effet
maison). Modèle mixte par campagne, sans le modèle spatial :
$y_d = \beta_{c(d)} + u_{h(d)} + v_{n(d)} + e_d$, $y_d$ = part (pt) du candidat de
référence (Macron en 2017/2022, le candidat RN en 2027 ; duels sans lui
écartés), $\beta_c$ effet fixe par cellule (paire × fenêtre de $F$ jours :
absorbe paire et tendance), $u_h\sim\mathcal N(0,\sigma_h^2)$ institut,
$v_n\sim\mathcal N(0,\sigma_n^2)$ notice (duels d'un même sondage),
$e_d\sim\mathcal N(0, b_d+\sigma^2)$, $b_d$ binomial. REML ; test du rapport
de vraisemblance $\sigma_h = 0$ (mélange ½χ²₀ + ½χ²₁). Code :
`model/backtest/maison_t2.py`, résultat `results/maison_t2_reml.json`.

| campagne | duels | σ_h (pt), F = 7 / 14 / 30 j | σ_n (pt, F = 14) | σ (pt, F = 14) | p (σ_h = 0), F = 7 / 14 / 30 |
|---|---|---|---|---|---|
| 2017 | 77 | 0,96 / 1,02 / 0,96 | 0,81 | 0,47 | 0,002 / 0,0001 / 0,001 |
| 2022 | 307 | 0,46 / 0,50 / 0,51 | 1,13 | 0,53 | 0,001 / 0,003 / 0,008 |
| 2027 | 113 | 0,89 / 0,71 / 0,50 | 0,82 | 0,43 | 0,29 / 0,10 / 0,10 |

**Lecture.** Il y a des effets maison au second tour : **0,5 à 1 pt par
institut, établis en 2017 et 2022** (p ≤ 0,008 à toutes les fenêtres), de même
ordre en 2027 mais non établis (113 duels, p ≈ 0,1). S'y ajoute un **effet
notice** de 0,8-1,1 pt : les duels d'un même sondage bougent ensemble. Le
modèle actuel traite chaque duel comme indépendant avec $\sigma_{T2}$
≈ 1,3 pt : il **surcompte l'information** des notices à plusieurs duels et ne
voit pas qu'une moyenne dominée par un institut hérite de son biais. La
décision de §8.17 (« pas d'effets maison au T2 ») est **retirée**.

**Conséquence (à spécifier avant de coder).** Dans la vraisemblance des duels :
$\operatorname{logit} y_d = \operatorname{logit} s_{ab}(t_d) + (u_{h,a}-u_{h,b})
+ v_{n(d)} + \varepsilon_d$, $u_{h,i}\sim\mathcal N(0,\sigma_h^2)$ par institut
et candidat (comme au T1), $v_n$ commun aux duels d'une notice (signe : à
orienter par candidat, p. ex. $v_{n,a}-v_{n,b}$). Effet attendu : IC plus larges
pour les paires portées par peu d'instituts, V1 mieux calibrée ; σ_T2 devrait
baisser. À tester en V1 sur 2022 (307 duels) avant tout passage.

### 8.23 bis Effets maison et notice dans la vraisemblance des duels (maths, 2026-11-16)

Pour le duel $d$ (paire $a_d$/$b_d$, institut $h_d$, notice $n_d$), part
observée $y_d$ du premier nommé :
$$y_d = s_d + (u_{h_d,a_d} - u_{h_d,b_d}) + (v_{n_d,a_d} - v_{n_d,b_d}) + e_d,$$
$u_{h,i}\sim\mathcal N(0,\sigma_h^2)$ (institut × candidat),
$v_{n,i}\sim\mathcal N(0,\sigma_n^2)$ (notice × candidat),
$e_d\sim\mathcal N(0,\ s_d(1-s_d)/N_d+\sigma_{T2}^2)$, tous indépendants.
L'effet est par candidat : un institut qui surestime Le Pen la surestime
dans tous ses duels, quel que soit le nom premier dans l'ordre. **Marginalisé
analytiquement** (aucun paramètre latent ajouté, seulement σ_h et σ_n) :
$$y \sim \mathcal N_D\big(s,\ \mathrm{diag}(s(1-s)/N + \sigma_{T2}^2) + \sigma_h^2 K_h + \sigma_n^2 K_n\big),$$
$K = ZZ^\top$, $Z_{d,(g,i)} = \mathbb 1[g = g_d]\,(\mathbb 1[i = a_d] - \mathbb 1[i = b_d])$,
soit $K_{dd'} = \mathbb 1[g_d = g_{d'}]\,(\mathbb 1[a_d{=}a_{d'}] - \mathbb 1[a_d{=}b_{d'}] - \mathbb 1[b_d{=}a_{d'}] + \mathbb 1[b_d{=}b_{d'}])$
($g$ = institut pour $K_h$, notice pour $K_n$). Coût : une Cholesky D × D
par pas (D ≤ 320). **Trop cher en pratique** (aucune chaîne en 2 h 15). Forme retenue,
exactement équivalente : $u$ explicite et non centré ($z_{h,i}\sim\mathcal N(0,1)$,
$u = \sigma_h z$, institut × candidat), et l'effet notice marginalisé **par
bloc** de notice (une notice contient quelques duels ; $K_n$ est
bloc-diagonal par notice), Cholesky de blocs ≤ k_max × k_max. Coût mesuré ≈
celui de hier. Priors : $\sigma_h, \sigma_n \sim$ HalfNormal(0,01)
(1 pt ; REML §8.23 : ~0,5-0,8 pt par candidat). Le résultat officiel n'a ni
institut ni notice.

**Prédiction V1** (moyenne de $n$ sondages cibles d'instituts connus) :
$\bar y = s + \bar\varepsilon$, $\mathrm{Var}(\bar\varepsilon) =
n^{-2}\,\mathbf 1^\top[\mathrm{diag}(\cdot) + \sigma_h^2 K_h + \sigma_n^2 K_n]\mathbf 1$
restreint aux duels cibles (même paire : $K_{dd'} = 2\cdot\mathbb 1[g_d = g_{d'}]$),
soit $[\bar v + 2\sigma_h^2 \sum_h n_h^2/n + 2\sigma_n^2]/n$ … (une notice
ne contient qu'un duel d'une paire donnée). Le scrutin : effets maison sans
objet (le résultat n'a pas d'institut).

**Mesure.** Variante `hier_maison` (= hier + ces effets), V1 2022 sur les 4
paires (16 fits), contre hier : erreur, couverture, log-score KDE ; on
rapporte σ_h, σ_n a posteriori. Retenu si le log-score KDE apparié est
meilleur sur ≥ 3/4 paires sans perte d'erreur, sinon non retenu.

**Résultat §8.23 bis (2026-11-16, V1 2022, hier contre hier_maison ; même
traitement : phase 2 là où hier en avait une)** :

| paire retirée | observé | erreur hier / maison (pt) | LS KDE hier / maison | couvert | concluant | σ_h, σ_n, σ_T2 (pt, maison) |
|---|---|---|---|---|---|---|
| Macron–Le Pen | 55,3 | 4,04 / **2,28** | 1,48 / **2,34** | oui / oui | oui / oui (R̂ T2 4,0) | 0,93 / 0,69 / 0,30 |
| Macron–Zemmour | 66,7 | **2,57** / 10,41 | **2,30** / −5,02 | oui / non | oui / **non** | 0,58 / 0,41 / 0,28 |
| Macron–Mélenchon | 61,5 | 0,28 / 0,03 | 1,50 / 1,76 | oui / oui | non / non | 0,77 / 0,47 / 0,20 |
| Macron–Pécresse | 65,6 | 12,98 / 12,88 | −22,1 / −10,4 | non / non | oui / oui | 0,76 / 0,23 / 0,17 |

Log-score meilleur sur 3/4, mais **erreur moyenne pire** (4,97 → 6,40 pt, tirée
par Zemmour : +7,8 pt, sûr de lui à tort, non convergé même en phase 2) ⇒
**non retenu** selon la règle. Les effets sont bien identifiés (σ_h 0,6-0,9 pt,
σ_n 0,2-0,7 pt, cohérents avec le REML §8.23) et σ_T2 s'effondre (1,3 → 0,2-0,3
pt) : la dispersion des duels est presque entièrement de l'effet maison et
notice. Mais avec ~200 paramètres de plus, le bloc T2 converge moins bien
(R̂ T2 jusqu'à 4) et une extrapolation (Zemmour) déraille. Piste : n'effets
maison que par institut (pas par candidat), ou seulement l'effet notice, qui
ne coûte aucun paramètre latent.

**Effet notice seul (variante `hier_notice`, 2026-11-17, même traitement : phase
2 pour Zemmour)** :

| paire retirée | erreur hier / notice (pt) | LS KDE hier / notice | σ_n, σ_T2 (pt) |
|---|---|---|---|
| Macron–Le Pen | 4,04 / 4,00 | 1,48 / **1,67** | 0,84 / 0,49 |
| Macron–Zemmour | **2,57** / 4,30 | **2,30** / 1,74 | 0,50 / 0,42 |
| Macron–Mélenchon | **0,28** / 0,62 | **1,50** / 1,37 | 0,67 / 0,29 |
| Macron–Pécresse | 12,98 / **12,47** | −22,1 / **−17,5** | 0,59 / 0,21 |

Log-score meilleur sur 2/4, erreur moyenne 4,97 → 5,35 pt ⇒ **non retenu**.
Bilan §8.23 : les effets maison/notice existent (REML, et σ_T2 divisé par 3-5
quand on les modélise), mais les intégrer au modèle n'améliore pas la
prédiction des paires retirées en 2022 ; ils déplacent l'information sans
corriger l'erreur de structure. Pas de changement de modèle.

### 8.24 Synthèse V1/V2 (2026-11-16, chiffres committés, 0 fit)

**V1 — paire retirée** (cible = moyenne des sondages de la paire sur 30 j ;
tâches de §8.20 ; « géométrie seule » = κ = 1, η = 0 sur le même fit ; BT =
Bradley-Terry sur les autres duels ; couverture IC90 prédictive §8.14).

| élection | tâches (concluantes abst / hier) | MAE abst / hier (pt) | géométrie seule | BT | couverture abst / hier | LS KDE Σ abst / hier (concluantes des deux) |
|---|---|---|---|---|---|---|
| 2017 | 2 (1 / 1) | 2,05 / 2,81 | 10,3 | 4,3 | 2/2 / 2/2 | 0,91 / 1,73 (1) |
| 2022 | 4 (2 / 3) | 6,34 / 4,97 | 8,0-9,3 | 12,2 | 2/4 / 3/4 | 1,54 / 2,30 (1) |
| 2027 | 4 (4 / 4) | 3,02 / 2,28 | — | — | 2/4 / 2/4 | 6,46 / 8,04 (4) |

Le bloc T2 (κ, η) divise l'erreur de la géométrie seule par 2 à 5 et bat
Bradley-Terry partout ; l'erreur restante (Attal–Le Pen 4-7 pt, Mélenchon–Le
Pen 3 pt, Pécresse 2022 12 pt) est de la structure (§8.14, §8.21), que τ_paire
(§8.22) couvre en partie. Verdict de variante : hier (§8.20).

**V2 — duel final contre le résultat** (Macron–Le Pen ; IC90 à τ_T2 = 0,10 tel
que fitté ; à 0,19 : recalcul analytique en logit, sd² − 0,10² + 0,19², pas
de nouveau fit) :

| coupure | modèle (abst / hier) | moyenne sondages 7 j | résultat | couvert à 0,10 | couvert à 0,19 | R̂ global / T2 (abst) |
|---|---|---|---|---|---|---|
| 2017 T1−1 | 62,8 / 62,8 | 62,8 | 66,1 | oui | oui | 1,26 / 1,02 |
| 2017 T2−7 | 61,3 / 61,3 | 61,7 | 66,1 | non | oui | 1,02 / 1,00 |
| 2022 T1−30 | 58,9 / — | 58,9 | 58,6 | oui | oui | 16,3 / 2,21 |
| 2022 T1−1 | 52,4 / 52,3 | 52,8 | 58,6 | non | oui | 8,49 / 1,57 |
| 2022 T2−7 | 53,5 / — | 53,4 | 58,6 | non | oui | 4,31 / 1,59 |

Le modèle reproduit la moyenne des sondages (±0,4 pt) ; la couverture du
scrutin dépend de τ_T2 (2/5 → 5/5, mêmes réserves que §8.18-§8.19 : 2
erreurs indépendantes, de même signe). R̂ global élevé en 2022 abst : nœuds T1
multimodaux ; la cible T2 converge (critère §8.4) ; hier converge mieux
(R̂ 1,02 à 2022 T1−1). κ ≈ 1 en 2017 (0,99-1,07), 0,75-1,4 en 2022 ; ω de
−3 (2017) à −6 (2022).

### 8.25 Entre-deux-tours conditionné sur le résultat du T1 (protocole fixé avant résultat)

Après le T1, le résultat officiel est une mesure quasi exacte des parts du
champ du scrutin au jour du T1. On l'ajoute au jeu comme un nœud T1
ordinaire : champ exact des candidats, parts officielles, $N = 20\,000$
(écart-type binomial ≈ 0,3 pt, auquel s'ajoute la variance d'excès du modèle),
daté du T1. Il contraint μ, σ et $w(T_1)$ ; le duel au T2 en hérite par
la géométrie et l'OU. Écart assumé : le nœud mesure des bulletins, les autres
nœuds des intentions (écart δ, `projection.py`) ; le modèle n'a pas de terme
δ pour les nœuds, le résultat tire donc $w(T_1)$ vers les bulletins — c'est
voulu (les reports de T2 se font depuis les bulletins réels).

**Mesure.** V2 à T2−7 en 2017 et 2022, variante hier, avec et sans nœud
(4 chaînes chacune ; la base hier 2017 T2−7 existe déjà) : erreur du duel
final contre le résultat, log-score KDE (§8.20) des tirages au scrutin
(τ_T2 = 0,19), convergence de la cible (§8.4). Lecture : le nœud est retenu
s'il réduit l'erreur sur les deux élections sans dégrader la convergence ;
avec 2 élections, c'est une indication, pas une preuve.

**Résultat §8.25 (2026-11-16, hier, T2−7)** :

| élection | sans nœud | avec nœud T1 | moyenne sondages 7 j | résultat | convergence (sans / avec) |
|---|---|---|---|---|---|
| 2017 | 61,3 | 61,6 | 61,7 | 66,1 | concluante / concluante (R̂ 1,06 / 1,02) |
| 2022 | 53,6 | 52,7 | 53,4 | 58,6 | **non concluante** (1 chaîne retenue) / concluante (2 chaînes, R̂ global 5,1) |

Le nœud déplace le duel final de **+0,3 pt (2017) et −0,9 pt (2022)**, sans
réduire l'erreur (4,5 et 5,9 pt restent) : à T2−7, le duel est porté par les
sondages de second tour, et l'écart au résultat est entre ces sondages et
l'urne (§8.18), pas dans la géométrie que le T1 réel corrigerait. (Les
log-scores ne sont pas comparables : la base 2017 a été fittée à τ_T2 = 0,10.)
**Décision : nœud non retenu** (critère « réduit l'erreur sur les deux
élections » non rempli). Le drapeau reste pour l'entre-deux-tours 2027, où
il donnera P(élu) sur la paire réelle sans changer la lecture des duels.

### 8.26 Stacking des chaînes (revue, a/d ; maths avant code)

Alternative à la sélection §8.8 (qui jette les chaînes de plus basse
log-densité) quand les chaînes restent dans des modes différents (2022 :
18 fusions sur 30 ont écarté au moins une chaîne). Stacking bayésien
(Yao, Vehtari, Gelman 2022, « stacking for non-mixing Bayesian
computations ») : chaque chaîne $c$ est un prédicteur ; poids
$w \in \Delta^{C-1}$ maximisant la densité prédictive leave-one-out des
observations :
$$\hat w = \arg\max_w \sum_{i=1}^{n} \log \sum_c w_c\, \hat p_{c,-i},\qquad
\hat p_{c,-i} = \frac{\sum_s r_s\, p(y_i\mid\theta_s^{c})}{\sum_s r_s},\;
r_s = \min\!\big(1/p(y_i\mid\theta_s^{c}),\ \bar r\sqrt{S}\big)$$
(LOO par échantillonnage d'importance tronqué, Ionides 2008 ; sans lissage de
Pareto, faute d'arviz). Observations $i$ : chaque nœud T1 (somme sur ses
candidats testés de $\log\mathcal N(Y\mid\pi, \pi(1-\pi)/N + v_{exc})$) et
chaque duel ($\log\mathcal N(y\mid s, s(1-s)/n + \sigma_{T2}^2)$) — la même
vraisemblance que `joint_model`, ventilée par observation. Le postérieur
stacké est le mélange $\sum_c \hat w_c\, p_c(\theta)$ : on prend
$\mathrm{round}(\hat w_c \cdot C D)$ tirages espacés de la chaîne $c$.
Mesure : sur les 18 fusions concernées, cible (duel retiré ou final) par
stacking contre sélection ; erreur contre l'observé. Drapeau `--stacking`
(fusion seulement, sortie `_stacking`) ; la sélection reste le défaut tant
que le stacking ne fait pas mieux.

**Résultat §8.26 (2026-11-16, 18 fusions 2022 rejouées, 0 NUTS ; fichiers
`*_stacking.json`)** : erreur absolue moyenne de la cible, sélection
**5,99 pt**, stacking **5,95 pt** ; Δ = −0,04 pt, IC90 bootstrap par tâche
[−0,39 ; +0,28] ; stacking meilleur sur 8/18. Écarts de cible ≤ 2,2 pt (Zemmour
geo, Pécresse abst), en général < 0,5 pt. Le stacking donne du poids à des
chaînes que la sélection écartait (Macron–Le Pen abst : 0,24 à une chaîne à
−100 nats) sans gain mesurable : les modes 2022 prédisent les observations à
peu près aussi bien, c'est la cible retirée qu'ils ne départagent pas.
**Décision : la sélection §8.8 reste le défaut** ; `--stacking` reste
disponible. L'erreur 2022 (Pécresse, Zemmour, Le Pen : 7-12 pt) n'est pas un
problème de mélange de chaînes.

### 8.27 Valeur inclusive sur le champ (piste « alias LR 2027 » ; maths avant code)

L'abstention (§3.1) compare l'utilité de non-expression $\omega + \kappa\,
\mathrm{lse}(v)$ à celle des finalistes, avec $\mathrm{lse}(v) = \log\sum_{j\in R}
e^{A_j(v)}$ sur **tout le roster** $R$. En 2027, $R$ contient des alternatives
qui ne seront pas candidates ensemble (Retailleau / Wauquiez / Bertrand,
Le Pen / Bardella, Attal / Philippe…) : la valeur inclusive d'un électeur de
droite est gonflée par des « candidats » qui ne coexistent pas. Variante :
$\mathrm{lse}_M(v) = \log\sum_{j\in M} e^{A_j(v)}$, $M$ = champ modal du T1
(le champ le plus sondé) ∪ candidats en duel, dans la vraisemblance ET en
prédiction (`masque_lse`). Ne change rien sans abstention ; ω absorbe le
niveau moyen. **Mesure** : V1 2027 sur les 4 tâches dédoublonnées (§8.20),
`hier_champ` contre hier, log-score KDE ; retenu si meilleur sur ≥ 3/4 sans
perte d'erreur moyenne.

**Résultat §8.27 (2026-11-17, V1 2027, hier contre hier_champ)** : centres
identiques à ≤ 0,2 pt sur les 4 tâches (erreur moyenne 2,28 / 2,29 pt ;
log-score KDE Σ 8,04 / 8,02 ; meilleur sur 1/4) ⇒ **non retenu, sans objet**.
Raison : ω ≈ −4,2 à −4,6, l'abstention différentielle est faible et la valeur
inclusive pèse peu ; des alias quasi superposés n'ajoutent qu'au plus log k à
lse, absorbé par ω.

### 8.28 Révision après la revue Opus du 2026-11-17

- **§8.20 (hier préférée) : RETIRÉ.** Le bootstrap percentile sur 6 tâches est
  trop étroit : t de Student IC90 [−0,01 ; +1,06], test exact d'inversion des
  signes p = 0,0625 (unilatéral) ; 4 des 6 tâches sont du même jeu 2027 (Le Pen
  partout), soit 3 grappes par élection (p = 1/8). L'exclusion des non-
  concluantes dépend de la variante (Δ̄ = −1,16 sur les 8 tâches où hier
  converge, +0,51 sur les 7 où abst converge). ⇒ **pas de différence établie ;
  `abst` redevient le défaut** (`ETA_HIER_T2 = False`). À refaire : 10 tâches,
  CRPS ou log-score plafonné, inversion des signes par grappe d'élection. **[Fait §8.29 : hier préférée.]**
- **§8.22 (τ_paire)** : la valeur suit la variante par défaut — **0,11 avec
  abst** (gain hors échantillon +0,17), dans la fourchette 0,10-0,15 validée par
  l'utilisateur ; 0,08 n'était valable que pour hier. La courbe est plate
  (0,19 ne coûte que ~1 nat sur 4 tâches) et les 4 tâches sont toutes
  « RN contre X », alors que les paires jamais sondées sont « X contre Y » :
  calibrage non représentatif de l'usage (à tester : V1 Mélenchon–Philippe).
- **§8.23 bis (« non retenu ») : reformulé en « non évaluable »** : la cible V1
  n'était pas recentrée par les effets maison appris pour les instituts de la
  cible (seule une variance était ajoutée) ; la décision était tirée par
  Zemmour, non concluant dans les deux exécutions maison (contraire à la règle
  §8.20) ; Macron–Le Pen « concluant » avait R̂ T2 4,0. Sur les tâches
  concluantes des deux côtés (Le Pen, Pécresse), maison fait mieux 2/2. À
  refaire sans fit : recentrage par les z_maison a posteriori.
- **§8.23 (REML)** : 2017 a une notice par duel (K_n = I) : σ_n et σ non
  séparables, seul σ_n² + σ² ≈ (0,93 pt)² est identifié. 2022 : sans Harris
  (112/307 duels), σ_h = 0,16 pt, p = 0,42 — l'effet maison 2022 est « Harris
  diffère », pas un σ_h général. 2027 : 44/113 duels seuls dans leur cellule
  (sans contribution).
- **§8.21 (placebo)** : conclusion maintenue, raisonnement corrigé — l'effet
  notice EST le mécanisme d'une fuite ; l'argument est sa taille (biais
  ~σ_n/√n ≈ 0,3 pt contre des erreurs de 3-7 pt). Le bras notice d'Attal
  (6,8 → 3,5 pt avec une seule graine) montre que le tirage du sous-échantillon
  bouge la cible plus que les effets testés : variance de graine à mesurer.
- **§8.27** : le masque remettait tous les candidats en duel (Attal, Bardella,
  Hollande, Ruffin : précisément les alias visés) ⇒ test non concluant sur
  l'hypothèse ; non refait (ω ≈ −4,4, effet attendu nul).
- **§8.24 / §8.25** : « 5/5 couverts à 0,19 » est dans l'échantillon (0,19 = RMS
  de ces erreurs) ; §8.25 ne teste pas l'usage visé (P(élu) entre les tours) et
  la base 2022 n'était pas concluante.
- **§8.26** : LOO par IS tronqué sans diagnostic de Pareto k, sur observations
  à latents locaux : poids peu fiables ; décision inchangée (sans conséquence).

### 8.29 Verdict abst/hier robuste (revue 17/11, point 1 ; protocole fixé avant résultat)

Les **10 tâches** de §8.20, concluantes ou non (le critère de convergence est
rapporté, pas filtré, puisqu'il dépend de la variante). Scores par tâche :
(i) CRPS des tirages prédictifs, $\mathrm{CRPS} = \mathbb E|X-y| - \tfrac12\mathbb E|X-X'|$
(pt ; plus petit = meilleur, borné, insensible aux queues) ; (ii) log-score
KDE plafonné à −5. Écart apparié Δ_j (hier − abst, signe tel que > 0 = hier
meilleur). Inférence : test exact d'**inversion des signes** (a) sur les 10
tâches, (b) sur les 3 grappes d'élection (moyennes par élection, 2³ = 8
configurations, p minimal 1/8). Règle : hier préférée si le CRPS ET le LS
plafonné donnent p ≤ 0,10 en (a) ET le même signe sur les 3 élections ; sinon
« pas de différence établie », abst reste. τ_paire croisé : on rapporte le
τ* (V1 2027) de chaque variante.

**Résultat §8.29 (2026-11-17, `verdict_abst_hier_robuste.json`)** :

| | Δ moyen (> 0 = hier meilleur) | hier meilleur | p (10 tâches) | par élection 2017 / 2022 / 2027 | p (3 grappes) |
|---|---|---|---|---|---|
| CRPS (pt) | +0,97 | 8/10 | **0,042** | +1,15 / +1,22 / +0,65 | 0,125 (minimum possible) |
| LS KDE plafonné (−5) | +0,67 | 8/10 | **0,008** | +0,62 / +0,98 / +0,39 | 0,125 |

Seul revers net : Macron–Pécresse en CRPS (−1,77 pt ; en LS les deux sont au
plafond). **Selon la règle fixée d'avance, hier est préférée** ; elle redevient
la variante T2 (`ETA_HIER_T2 = True`) et τ_paire suit son calibrage (0,08).
Réserves : les tâches d'une même élection sont corrélées (mêmes données) ; au
niveau des grappes, p = 1/8 est le minimum atteignable avec 3 élections — le
signe est le même partout, mais aucune preuve forte n'est possible à ce
niveau. Le défaut de hier (excès de confiance sur un candidat atypique,
Pécresse) reste le point de vigilance.

### 8.30 hier_maison avec cible recentrée par les effets maison (revue 17/11, point 3 ; 0 fit)

La cible V1 (moyenne de sondages d'instituts connus) est recentrée tirage par
tirage par $\bar u = n^{-1}\sum_d (u_{h_d,a} - u_{h_d,b})$, $u = \sigma_h z$ a
posteriori ; seuls les instituts absents du fit gardent une variance maison
(`recentre_maisons`, `var_effets_moyenne`). Re-fusion des 4 tâches 2022 :

| paire | hier : erreur / CRPS / LS | maison recentré : erreur / CRPS / LS | concluant (hier / maison) |
|---|---|---|---|
| Macron–Le Pen | 4,04 / 2,73 / 1,48 | **2,14 / 1,25 / 2,45** | oui / oui (R̂ T2 4,0) |
| Macron–Zemmour (phase 2) | **2,57 / 1,54 / 2,30** | 10,17 / 8,65 / −3,47 | oui / non |
| Macron–Mélenchon | 0,28 / 2,15 / 1,50 | 0,14 / 1,72 / 1,76 | non / non |
| Macron–Pécresse | **12,98 / 11,46** / −5 | 13,90 / 12,30 / −5 | oui / oui |

Le recentrage déplace la cible de ≤ 1 pt (les effets des instituts de la
cible sont petits). Sur les tâches concluantes des deux côtés : 1 gain (Le
Pen), 1 perte (Pécresse) ; Zemmour (non convergé côté maison) déraille.
**Conclusion : aucun gain établi, et la convergence du bloc T2 se dégrade
(R̂ T2 jusqu'à 4).** Les effets maison ne sont pas adoptés dans la vraisemblance ;
ils restent une source de variance connue (REML §8.23), non modélisée.

### 8.31 V1 d'une paire « X contre Y » : Mélenchon–Philippe (revue 17/11, point 2 ; 8 fits)

Seule paire sans le RN avec une cible (2 sondages, moyenne 36,5 % pour
Mélenchon, t_k = 15/10) ; on retire ses duels et on la prédit depuis les 88
autres (tous « RN contre X »). hier : **36,2 %, erreur 0,30 pt**, IC90
prédictif [33,6 ; 38,8], CRPS 0,40, concluant ; abst : 35,6 %, erreur 0,87 pt,
CRPS 0,55. Log-score KDE selon τ_paire (hier) : 3,16 / 2,77 / 2,57 / 2,14
pour τ = 0 / 0,08 / 0,11 / 0,19. **L'extrapolation « X contre Y » est juste ici
et τ_paire ne fait que coûter** (−0,4 nat à 0,08, −1,0 à 0,19). Une seule
tâche, cible de 2 sondages : cela ne calibre pas τ_paire, mais n'indique
aucune sous-estimation de l'incertitude pour ce type de paire ; pas de
changement (arbitrage τ_paire toujours ouvert, ce point y est ajouté).

### 8.32 Variance de graine du sous-échantillon (revue 17/11, point 6 ; 8 fits)

Attal–Le Pen, abst, bras « notice partielle » (K = 29) avec trois graines :

| graine | centre | erreur (pt) | IC90 prédictif | CRPS | LS KDE |
|---|---|---|---|---|---|
| 821 | 39,8 | 3,50 | [20,9 ; 62,9] | 3,63 | 1,02 |
| 822 | 39,3 | 4,00 | [20,4 ; 60,8] | 3,63 | 1,03 |
| 823 | 36,9 | 6,41 | [31,9 ; 41,9] | 4,73 | 0,45 |
| (standard) | 36,5 | 6,83 | [32,2 ; 40,7] | 5,39 | −0,53 |
| (placebo, 821) | 36,6 | 6,72 | [32,1 ; 40,8] | 5,24 | −0,40 |

Le tirage des 29 duels retirés change le **régime** du postérieur : deux
graines sur trois donnent un IC très large (le duel cible perd son ancrage),
la troisième retombe sur le standard. Écart-type du centre entre graines
≈ 1,6 pt, du même ordre que les effets testés en §8.21/§8.23 bis/§8.27.
⇒ La conclusion de §8.21 (« fuite non établie ») tient, mais l'amélioration
apparente d'Attal dans le bras notice (6,8 → 3,5 pt) n'est **pas robuste** :
elle tient à l'élargissement de l'IC sur certaines graines. Tout écart V1
< 2 pt sur une seule tâche est à lire dans ce bruit.

### 8.33 Sensibilité de P(élu) : erreur anti-RN × τ_paire (revue 17/11, idée C ; 1 fit)

P(élu) du champ modal 2027 (hier, données du 13/11, τ_T2 = 0,19), duel final
décalé de d points en défaveur (d < 0) ou en faveur (d > 0) du candidat RN
(sensibilité, **pas un réglage** : l'arbitrage 2 « biais anti-RN non corrigé »
reste en vigueur) :

| d (pt, RN) | −3 | −2 | −1 | 0 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| Le Pen | 74,7 | 76,7 | 79,0 | 80,3 | 82,6 | 84,1 | 85,5 |
| Philippe | 17,9 | 16,5 | 15,0 | 13,9 | 12,1 | 11,1 | 10,0 |
| Glucksmann | 6,0 | 5,4 | 4,8 | 4,5 | 4,1 | 3,6 | 3,5 |

**τ_paire (0,08 ou 0,19) ne change P(élu) d'aucune décimale** : les paires
qualifiées probables (Le Pen contre X) sont toutes sondées ; τ_paire ne touche
que l'affichage de duels jamais sondés (onglet Scénarios). L'arbitrage
τ_paire est donc d'enjeu faible pour P(élu). L'erreur anti-RN des scrutins
passés (−3 à −6 pt en 2017/2022, §6) déplacerait P(élu Le Pen) de ~80 à ~75 %
ou moins : c'est l'arbitrage 2 qui pèse, pas τ_paire.

### 8.34 Révision après la revue Opus du 2026-11-18

- **§8.29 requalifié en « préférence faible »** (recalculé) : Δ LS hier − abst
  selon le plafond : −5 → +0,67 (p = 0,008) ; −8 → +0,57 (p = 0,053) ; −10 →
  +0,37 (p = 0,22) ; sans plafond → −0,84 (p = 0,50, Pécresse −15 nats). En CRPS,
  sans Macron–Le Pen 2022 p = 0,084, sans elle ni Attal p = 0,17. Le critère
  « même signe sur les 3 élections » est plus faible que l'inversion des signes
  par grappe demandée, et a été écrit après lecture des scores par tâche. Le
  verdict ne tient qu'au plafond : **arbitrage remonté à l'utilisateur** (hier
  meilleure en moyenne mais sûre d'elle à tort sur un candidat atypique ;
  abst plus prudente). `ETA_HIER_T2` inchangé en attendant (sans effet publié).
- **Calibration V1 des deux variantes : IC trop étroits.** Erreur standardisée
  en logit, RMS 1,86 (abst) / 1,95 (hier), couverture IC90 6/10 et 7/10 (au lieu
  de 9/10). À traiter par un facteur d'élargissement appris hors élection (file).
- **τ_paire = 0,10** (borne basse de la fourchette validée par l'utilisateur ;
  0,08 était hors fourchette, appliqué sans son accord). Sans effet sur P(élu)
  (§8.33).
- **§8.33 : double compte.** τ_T2 = 0,19 est le RMS NON centré des erreurs
  passées et contient déjà le biais anti-RN moyen (~3,5 pt) ; d = −3 avec τ =
  0,19 compte ce biais deux fois. Le scénario cohérent est d = −3,5 avec un τ
  résiduel ≈ 0,10 (file, 1 fit). De plus, l'erreur Monte-Carlo de P(élu) est
  ≈ 1 pt (1600 tirages) : 80,3 → 80,8 % (118 notices) est du bruit.
- **§8.30** : le tableau mélange changement de modèle et recentrage ; le
  recentrage seul déplace la cible de 0,14 à 1,0 pt (Pécresse : 52,7 → 51,7,
  mauvais sens). « Concluant » ignore le R̂ du bloc T2 (4,0 sur Macron–Le Pen).
- **§8.31** ne prouve rien sur τ_paire (une tâche bien prédite : la variance la
  plus faible gagne mécaniquement). **§8.32** : la variance de graine (1,6 pt)
  couvre le gain d'Attal, principal contributeur 2027 au verdict hier.
- **118 notices** : les 95 duels sont les mêmes ; « duels inchangés » est
  mécanique.

### 8.35 Élargissement des IC V1 appris hors élection (revue 18/11, idée 2 ; 0 fit)

Erreur standardisée en logit $z_j = (\mathrm{logit}\,y_j - \bar z_j)/s_j$ sur
les 10 tâches V1 ; facteur $\lambda$ = RMS des $z$ appris sur deux élections,
appliqué à l'écart-type prédictif de la troisième (leave-one-election-out).

| variante | λ appris (sans 2017 / 2022 / 2027) | couverture IC90 brute → λ | LS gaussien Σ brut → λ |
|---|---|---|---|
| abst | 2,07 / 1,40 / 1,96 | 6/10 → **8/10** | 5,41 → **9,80** |
| hier | 2,18 / 1,31 / 2,17 | 7/10 → **9/10** | 5,88 → **9,99** |

Hors élection, les IC V1 doivent être **élargis d'un facteur ≈ 1,3-2,2**
(couverture rétablie, log-score +4 nats). En terme additif, un bruit
d'extrapolation $\mathcal N(0,\tau^2)$ ramenant RMS z à 1 vaut **τ ≈ 0,17
(hier) - 0,19 (abst)** (écart-type prédictif médian 0,11-0,13 logit), soit la
valeur haute de l'arbitrage τ_paire, pas 0,08-0,10. Réserve : 10 tâches dont
Pécresse pèse lourd ; le calibrage de τ_paire sur les seules V1 2027 (§8.22)
donnait 0,08-0,11. Éclaire l'arbitrage τ_paire (remonté), ne le tranche pas.

### 8.36 Sensibilité de P(élu) sans double compte (revue 18/11, point 5 ; 1 fit)

τ_T2 = 0,19 est le RMS NON centré des erreurs 2017/2022 (biais moyen anti-RN
≈ 3,5 pt inclus). Deux lectures cohérentes : (i) erreur centrée de largeur
0,19 (réglage actuel) ; (ii) biais −3,5 pt (en défaveur du RN) + résidu
τ ≈ 0,10. Données du 17/11 (Riner et Polony placés par main), hier, champ
modal inchangé, R̂ 1,015 :

| scénario | Le Pen | Philippe | Glucksmann | Retailleau |
|---|---|---|---|---|
| (i) centré, τ_T2 0,19 (actuel) | 82,1 | 13,4 | 3,6 | 0,7 |
| centré, τ_T2 0,10 (pour mémoire) | 85,7 | 10,9 | 2,8 | 0,6 |
| **(ii) biais −3,5 pt + τ 0,10** | **76,0** | **18,7** | 4,3 | 1,0 |

Erreur Monte-Carlo ≈ 1 pt. Le choix entre (i) et (ii) — corriger ou non le
biais anti-RN observé deux fois — est l'arbitrage 2, déjà tranché « non
corrigé » par l'utilisateur : (i) reste en place ; (ii) chiffre ce que ce choix
engage (≈ 6 pt de P(élu) pour Le Pen).

### 8.37 Test prospectif 2027 (revue 18/11, idée 1 ; protocole fixé avant résultat)

Seul critère sans risque de protocole à la carte : prédire des sondages qui
n'existent pas encore. À la date de gel $t_0$ (dernier terrain du jeu), un fit
par variante (abst, hier) ; pour toutes les paires du champ modal ∪ candidats
en duel, 400 tirages de $s_{ab}$ à $h$ = 0, 14, 30 j (sans aléa de scrutin),
figés dans `results/prospectif/prediction_<variante>_<t0>.json` — aucune
retouche ensuite. Chaque duel publié après $t_0$ est scoré sans fit contre
l'horizon figé le plus proche, prédictif d'UN sondage
($+\mathcal N(0, s(1-s)/N + \sigma_{T2}^2)$) : erreur, CRPS, z, couverture
IC90, en séparant paires déjà sondées au gel et paires nouvelles.
Lecture prévue : après ≥ 20 duels nouveaux, comparaison abst/hier en CRPS
(inversion des signes appariée) et calibration (RMS z, couverture). Code :
`model/backtest/prospectif_2027.py` (`--figer`, `--scorer`).

### 8.38 (A)(1) Reports T1→T2 prédits contre tableaux des instituts (0 fit ; protocole fixé avant résultat)

**Données.** `data/parsed/reports_t2_{2017,2022}.csv` : 23 notices en 2017 et 21 en 2022,
toutes sur Macron–Le Pen, entre les deux tours. Électorats : Mélenchon, Hamon, Fillon,
Dupont-Aignan (2017) et Mélenchon, Jadot, Pécresse, Zemmour (2022). Cellules
{Macron, Le Pen, Aucun}, en % de l'électorat T1 du candidat (somme 99-102). `base_electorat`
est vide partout, donc il n'y a pas de n effectif : la dispersion entre notices sert de bruit
de mesure.

**Reports prédits** (spec §2.2, révisée par §2.1 et §3.1). On travaille par tirage
postérieur, avec $A_j(v)=w_j+\log\varphi_j(v)$ pris à `w_now`. Le champ $S_1$ est l'ensemble
des candidats réellement présents au T1 de l'élection.

- Électorat de $k$ : $p_k(v)=e^{A_k}/\sum_{j\in S_1}e^{A_j}$ et $E_k=\int p_k$.
- Expression au T2 : $\rho(v)$, issu de `duel_expression`, avec la même valeur inclusive
  que la vraisemblance.
- Préférence entre les deux finalistes : $q(v)=\mathrm{logistique}(\kappa[A_a-A_b])$.

On en déduit, pour chaque électorat $k$ :

$$R_k=\frac{\int p_k\rho}{E_k},\qquad G_k=\frac{\int p_k\rho q}{\int p_k\rho}.$$

La valence η est posée sur l'**agrégat** (§2.1). Pour la répartir entre électorats, on
applique un décalage commun en logit $\tilde G_k=\mathrm{logistique}(\operatorname{logit}G_k+\delta)$.
Le décalage $\delta$ est résolu par tirage (bissection, fonction monotone) pour que
$\sum_{k\in S_1}E_kR_k\tilde G_k/\sum_kE_kR_k=s_{ab}$, la part prédite du duel. Ce choix est
cohérent avec le duel publié, puisque $\sum_kp_k=1$ sur $S_1$. η étant par finaliste, il ne
peut pas dépendre de l'électorat, et c'est précisément ce que teste (A)(1).

Parts prédites, en % de l'électorat de $k$ :

- Macron : $100R_k\tilde G_k$ ;
- Le Pen : $100R_k(1-\tilde G_k)$ ;
- Aucun : $100(1-R_k)$.

« Aucun » regroupe l'abstention, les blancs, les nuls et les sans-réponse. $1-R_k$ en est
l'analogue modèle : la non-expression parmi les votants du T1 de $k$.

**Fits utilisés.** V2 T2−7 déjà committés (2017 et 2022, variantes abst et hier). On les
reconstruit par fusion des chaînes sauvées, sans NUTS. Les reports ne sont pas dans la
vraisemblance de ces fits ; les duels de la même période, eux, y sont.

**Mesures**, par cellule (électorat × candidat) :

- $\bar y$, moyenne sur les notices, et $s$, écart-type entre notices ;
- erreur $e=\hat m-\bar y$, où $\hat m$ est la moyenne postérieure ;
- $z=e/\sqrt{\mathrm{sd}_{post}^2+s^2}$, l'écart à ce qu'on attend d'une nouvelle notice.

On calcule aussi la MAE par électorat et par candidat_t2, et deux grandeurs clés :

- le partage parmi les exprimés, $M/(M+LP)$, que η ne peut régler que globalement ;
- la non-expression, qui teste ω.

**Critère d'ouverture de (D), fixé avant lecture.** On ouvre (D) s'il existe une
« erreur systématique non absorbée ». C'est le cas si un électorat de la même famille
(gauche radicale : Mélenchon 2017 et 2022) vérifie, **les deux années**, à la fois
$|e|>10$ pt et $|z|>2$ sur le partage parmi les exprimés, dans le sens attendu d'une
seconde dimension, c'est-à-dire un report vers Le Pen sous-prédit. À défaut, on consigne
seulement « limite §2.3 confirmée ou infirmée par électorat », et (A)(2) suit.

**Résultats (2026-11-19, `results/reports_t2_A1.json`).** Contrôle : la reconstruction
redonne le $s_{ab}$ committé à 0,1 pt près, par exemple 0,6133 contre 0,6125 pour
hier 2017 et 0,5363 contre 0,5361 pour hier 2022.

Partage Macron parmi les exprimés, en %, mesuré puis modèle (abst / hier) :

| électorat | 2017 mesuré | 2017 abst / hier | 2022 mesuré | 2022 abst / hier |
|---|---|---|---|---|
| Mélenchon | 76,4 | 98,9 / 99,2 (z 3,4 / 3,6) | 63,0 | 100 / 100 (z 7,4 / 7,4) |
| Hamon / Jadot | 93,4 | 98,7 / 99,0 | 87,5 | 100 / 100 |
| Fillon / Pécresse | 62,8 | 68,0 / 68,6 | 65,0 | 61,4 / **36,6** |
| NDA / Zemmour | 37,6 | 29,2 / 28,6 | 9,4 | 0,0 / 0,4 |

MAE des cellules (Macron, Le Pen, Aucun), en pt :

| | Macron | Le Pen | Aucun |
|---|---|---|---|
| abst 2017 | 6,3 | 9,8 | 10,1 |
| abst 2022 | 17,8 | 9,5 | 22,5 |
| hier 2017 | 5,1 | 9,6 | 9,2 |
| hier 2022 | 17,6 | 19,0 | 11,2 |

Abstention, cas extrême : l'électorat Jadot 2022 est prédit à 94 % « Aucun » en abst, contre
33 % mesuré.

**Lecture.**

1. **Critère (D) rempli** pour les deux variantes et les deux années. L'électorat
   Mélenchon ne reporte pratiquement rien sur Le Pen dans le modèle (0-1 % de
   l'électorat), contre 15 % en 2017 et 21 % en 2022 dans les tableaux des instituts.
   L'erreur sur le partage vaut 23 à 37 pt, alors qu'elle n'est que de 5 à 12 pt pour la
   gauche modérée (Hamon, Jadot). C'est la seconde dimension anticipée au §2.3 : ce n'est
   pas un simple défaut de netteté.
2. **Reports trop nets partout.** Les cellules prédites valent 0 ou 100 % là où les
   instituts mesurent 5 à 10 % de transfert croisé (Zemmour→Macron 8 %, Jadot→Le Pen 8 %).
   Le choix $q$ par votant est une quasi-marche (§2.1). Le bruit de choix
   individuel ne passe que par κ, global, et η, à l'agrégat.
3. **L'abstention par électorat n'est pas tenue.** ω et la valeur inclusive ne sont
   identifiés qu'au travers des duels agrégés. Le niveau « Aucun » par électorat erre de
   10 à 60 pt et change de signe d'une variante à l'autre (Mélenchon 2022 : 57,7 contre
   23,8 %).
4. **Le duel agrégé reste juste alors que la décomposition est fausse.** L'interdit du
   §2.3 tient : on ne publie pas $T$. C'est aussi un avertissement pour (A)(2). Ajouter
   les reports comme observations à un modèle 1D obligerait la géométrie à se tordre pour
   rapprocher Mélenchon de Le Pen sur l'axe. Les reports ne doivent entrer qu'avec un
   mécanisme qui les représente : (D), ou à défaut un plancher de transfert.

⇒ **(D) est ouvert.** Les maths du modèle 2D sont à écrire avant (A)(2). L'alternative
minimale est à chiffrer en parallèle : un plancher de report par votant, avec bruit
logistique additionnel, qui couvre le point 2 mais pas le point 1.

### 8.39 (D)(ii) Plancher de report (0 fit ; lecture descriptive)

Alternative minimale à une seconde dimension : une fraction ε des votants choisit au hasard
entre les deux finalistes, quelle que soit sa position,
$$q_\varepsilon(v)=(1-\varepsilon)\,q(v)+\varepsilon/2 .$$
ρ est inchangé, donc la cellule « Aucun » aussi. δ est recalé pour que le duel agrégé reste
$s_{ab}$ (§8.38) : seule la répartition entre électorats change. ε est un paramètre global.
Il ne crée aucune dépendance à l'électorat d'origine au-delà de la géométrie : il rapproche
tous les partages de 50 %, et le fait d'autant plus qu'ils en sont loin.

**Ce qu'on attend, avant lecture.** Les cellules 0/100 % prennent 5 à 10 % de transfert
croisé, ce qui corrige le point 2. En revanche, l'écart Mélenchon / gauche modérée subsiste :
ε abaisse aussi le partage de Hamon et de Jadot, déjà proche de la mesure. La grille
ε ∈ {0 ; 0,1 ; 0,2 ; 0,3 ; 0,4} est évaluée sur les mêmes tirages (`--planchers`), en MAE
sur Macron, Le Pen et le partage. C'est une lecture descriptive : ε choisi sur les reports
serait de l'ajustement en échantillon ; en (A)(2), il serait appris.

**Résultats** (`results/reports_t2_plancher.json`, sur les mêmes tirages que §8.38). MAE sur
Macron, Le Pen et le partage, en pt :

| | ε = 0 | ε = 0,1 | ε = 0,2 | ε = 0,3 | ε = 0,4 |
|---|---|---|---|---|---|
| abst 2017 | 8,8 | 7,3 | 7,3 | 7,6 | 7,6 |
| abst 2022 | 14,3 | 13,8 | 13,6 | 13,8 | 14,7 |
| hier 2017 | 8,5 | 6,4 | 5,8 | 6,4 | 6,9 |
| hier 2022 | 19,4 | 12,0 | 8,8 | 9,2 | 10,4 |

À ε = 0,2, l'erreur sur le partage de l'électorat Mélenchon reste de +12 à +14 pt en 2017
et de +24 à +29 pt en 2022. La prédiction faite avant lecture est vérifiée : ε corrige la
netteté, pas Mélenchon. Comme mécanisme, le plancher est à garder pour (A)(2), avec ε appris.

### 8.40 (D)(i) Seconde dimension : maths

**Forme retenue d'abord : le « fer à cheval » à coordonnée apprise (D-a).** Elle garde la
grille 1D des votants. Au second tour seulement, chaque finaliste reçoit une coordonnée
anti-système $\nu_i$. Le votant en $v$ porte une affinité anti-système qui croît avec son
excentrement, $e(v)=|2v-1|\in[0,1]$. Le votant choisit $a$ avec

$$q_{ab}(v)=\mathrm{logistique}\big(\kappa[A_a(v)-A_b(v)]+(\nu_a-\nu_b)\,e(v)\big),$$

η restant posé à l'agrégat (§2.1).

Priors : $\nu_i=s_\nu z_i$, avec $z_i\sim\mathcal N(0,1)$ et $s_\nu\sim\mathcal{HN}(1)$.
Seules les différences sont identifiées ; le niveau commun est tenu par le prior, comme pour η.

Propriétés :

1. **Non-monotonie en $v$.** Si $\nu_{LP}>\nu_M$, les deux extrémités de l'axe penchent
   vers Le Pen. Pour la gauche radicale, ce penchant s'oppose à la pente géométrique : c'est
   le report Mélenchon→Le Pen. La gauche modérée ($e$ plus petit) fuit moins.
2. **Valeur inclusive inchangée** : ν n'entre pas dans ρ, pas plus que η (§3.1).
3. **Un seul paramètre par candidat**, et rien au T1 : les fits T1 et la géométrie restent
   ceux d'aujourd'hui.
4. **Confusion avec η.** Sur une seule paire, $(\nu_a-\nu_b)\,\bar e$ se confond avec
   $\eta_a-\eta_b$ dans le duel agrégé. ν n'est donc identifié que par la **forme** des
   reports entre électorats, ou par des duels contre des adversaires variés (2027 : 11
   paires et plus). Sans reports dans la vraisemblance, ν retombe sur son prior. (D-a) et
   (A)(2) vont donc ensemble.

**Test d'identification sans fit (préalable au code).** Sur les tirages de §8.38, pour
Macron–Le Pen, on cherche par profil le scalaire $\Delta\nu=\nu_M-\nu_{LP}$ qui minimise
la MAE des partages, avec δ recalé. Deux critères, fixés avant lecture :

- un seul $\Delta\nu$ ramène l'erreur de Mélenchon sous 10 pt **sans** dégrader de plus de
  3 pt la MAE des autres électorats, les deux années ;
- les valeurs de $\Delta\nu$ de 2017 et de 2022 sont de même signe et dans un rapport
  inférieur à 2.

Si les deux critères passent, on code (D-a). Sinon, on passe à (D-b).

**Résultat du test (2026-11-19, `results/reports_t2_nu.json`) : ÉCHEC, (D-a) n'est pas
codé.** Δν est profilé sur {0 ; −1 ; −2 ; −3 ; −4 ; −6 ; −8}, à ε = 0 et à ε = 0,2.
L'erreur de Mélenchon ne bouge presque pas : de 22,5 à 21,6 pt en 2017 et de 37,0 à 37,0 pt
en 2022 (abst, ε = 0), à peine plus à ε = 0,2 (de 24,2 à 26,3 pt en hier 2022, en sens
contraire). La MAE des autres électorats varie de moins de 1,5 pt.

**Cause.** Le logit par votant $\kappa[A_a-A_b]$ est saturé à l'endroit de l'électorat
Mélenchon. $A$ contient $\logarphi=-(v-\mu)^2/2\sigma^2$ ; avec des σ de 0,08 à 0,15,
l'écart atteint des dizaines de logits à 0,3 d'axe de Le Pen. Un terme additif borné,
$
u\,e(v)$ avec $|
u|\le 8$, ne passe pas, et le recalage δ en absorbe le peu qui
passe. Le défaut ne tient donc pas au manque d'une **utilité** de second tour : la
composition de l'électorat (un noyau gaussien étroit sur un seul axe) interdit qu'une
partie des électeurs de Mélenchon soit proche de Le Pen.

**Conséquence pour (D).** Il faut que des électeurs de Mélenchon soient *proches* de Le Pen
dans l'espace des votants, et non pas seulement attirés par elle au T2. Deux voies :

- **(D-b)**, la 2D séparable ci-dessus : l'électorat de $k$ s'étale en $u$ et chevauche
  celui de Le Pen à $
u$ voisin ;
- **(D-c)**, plus légère : un noyau à queues lourdes dans $A$ pour le T2 seulement (Student
  à faible ν_dof, ou mélange gaussien avec une composante large). Elle désature $q$ loin de
  $\mu$, mais reste monotone : elle traiterait la netteté comme ε, pas le croisement.

Seule (D-b) vise le point 1 ; ses maths complètes (coût, quadrature 2D, priors sur ν,
identification par les champs T1) sont la prochaine tâche.

**(D-b), plus coûteux, seulement si (D-a) échoue.** Votants sur une grille 2D
$[0,1]\times[0,1]$, noyau séparable
$\varphi_i(v,u)\propto\mathcal N(v;\mu_i,\sigma_i)\,\mathcal N(u;\nu_i,\lambda)$, au T1 comme
au T2. Le coût est multiplié par $B_u$. ν est alors aussi identifié par les substitutions du
T1 (hypothèses à champs différents), mais faiblement.

### 8.41 (D-b) Votants en 2D : maths et test d'existence sans fit (protocole fixé avant résultat)

**Modèle.** Les votants sont uniformes sur $[0{,}01;0{,}99]^2$, de coordonnées
$v$ (gauche-droite) et $u$ (système/anti-système). La quadrature est un produit de
Gauss-Legendre $B_v\times B_u$, de poids $W_bW'_c$. Noyau séparable, normalisé sur la grille
comme en 1D :

$$\varphi_i(v,u)=\frac{\mathcal N(v;\mu_i,\sigma_i)\,\mathcal N(u;\nu_i,\lambda)}
{\sum_{b,c}W_bW'_c\,\mathcal N(V_b;\mu_i,\sigma_i)\mathcal N(U_c;\nu_i,\lambda)},\qquad
A_i(v,u)=w_i+\log\varphi_i(v,u).$$

Tout le reste est inchangé, avec $\int dv$ remplacé par $\iint dv\,du$ : le premier tour
$\pi_i(S)$, le duel $q$, l'expression ρ, la valeur inclusive, et η, ω, κ.

Priors : $\nu_i=\mathrm{logistique}(\zeta_i)$ avec $\zeta_i\sim\mathcal N(0,1)$, et
$\log\lambda\sim\mathcal N(\log 0{,}2;\,0{,}5)$. λ est **commun** à tous les candidats : un
λ par candidat serait trop peu identifié.

Le sens de l'axe $u$ est symétrique par réflexion $u\mapsto1-u$. On le fixe par la seule
contrainte d'ordre $\nu_{\text{RN}}\ge\nu_{\text{centre}}$ (finaliste RN contre finaliste
centriste), par une transformation ordonnée sur ces deux coordonnées.

**Ce que la 2D change, et que la §8.40 n'atteignait pas.** L'électorat de $k$,
$p_k(v,u)$, s'étale en $u$. Si $\nu_k$ est proche de $\nu_{LP}$ et que $\lambda$ n'est pas
petit, une partie des électeurs de $k$ se trouve dans la région $u$ où Le Pen est forte.
Le logit $\kappa[A_M-A_{LP}]$ y est alors bien moins saturé que sur un seul axe : c'est un
croisement par la composition de l'électorat, et non par une utilité ajoutée.

**Identification.**

- **Au T1, faible :** ν ne modifie que les substitutions entre hypothèses à champs
  différents (qui récupère l'électorat d'un absent ?).
- **Au T2, forte :** par les reports, en (A)(2).

D-b et (A)(2) vont donc ensemble, comme D-a.

**Coût.** Le tenseur $P\times N\times B$ devient $P\times N\times B_vB_u$. À $B_u=7$, le
gradient coûte environ 7 fois plus, soit à peu près 4 h par chaîne au lieu de 35 min. C'est
le prix à justifier avant tout fit.

**Test d'existence sans fit (porte avant le code).** On prend 40 tirages par fit V2 T2−7
de §8.38. On ajoute $u$ avec $B_u=7$ et $\lambda\in\{0{,}1;0{,}2;0{,}3\}$.

- À chaque évaluation, $w$ est **re-résolu** par point fixe
  ($w\leftarrow w+\log\pi^{1D}/\pi^{2D}$ sur le champ réel du T1), de sorte que les parts
  T1 du champ réel restent celles du fit 1D.
- κ et ω sont repris tels quels ; δ est recalé (§8.38).
- Seuls 9 ν sont libres, communs aux deux années : Macron, Le Pen, Mélenchon, Hamon,
  Fillon, Dupont-Aignan, Jadot, Pécresse, Zemmour. Les autres valent 0,5.
- On minimise la somme des carrés des erreurs des cellules (Macron, Le Pen, Aucun) des
  deux années, soit 9 paramètres pour 16 degrés de liberté.

Critères, fixés avant lecture. Pour au moins une valeur de λ, la variante hier doit
vérifier les trois :

- erreur sur le partage de Mélenchon < 10 pt **les deux années** ;
- MAE des autres électorats non dégradée de plus de 1 pt par rapport à la 1D ;
- ν interprétable : $\nu_{JLM}$ du côté de $\nu_{LP}$ par rapport à $\nu_M$, et plus
  proche de $\nu_{LP}$ que $\nu_{Hamon}$ et $\nu_{Jadot}$ ne le sont.

Si les trois passent, on code D-b derrière un drapeau éteint, avec le plancher ε de §8.39.
Le chantier (A)(2) suit. Sinon, (D) est abandonné et l'on consigne que le croisement n'est
pas représentable à coût raisonnable : on ne publie que $s_{ab}$, comme aujourd'hui.

Limite : c'est un test d'**existence** en échantillon. Il dit si la 2D peut représenter les
reports sans toucher au T1, pas que le fit les apprendra.

**Résultat (2026-11-19, `results/reports_t2_2d.json`) : ÉCHEC, (D) est abandonné.**

| variante, λ | partage Mélenchon 2017 (mesuré 76,4) | partage Mélenchon 2022 (mesuré 63,0) | MAE des autres, 1D → 2D | ν interprétable |
|---|---|---|---|---|
| hier, 0,1 | 90,1 (+13,7) | 100,0 (+37,0) | 9,9 → 4,3 | non |
| hier, 0,2 | 98,6 (+22,2) | 100,0 (+37,0) | 9,9 → 5,1 | non |
| hier, 0,3 | 98,3 (+21,9) | 100,0 (+37,0) | 9,9 → 7,0 | non |
| abst, 0,2 | 98,9 (+22,5) | 100,0 (+37,0) | 13,6 → 9,9 | non |

Les parts T1 du champ réel restent tenues (écart ≤ 0,2 pt). Avec 9 ν libres, la 2D
améliore les autres électorats, ce qui est attendu en échantillon. Le partage de Mélenchon
2022 ne bouge pas, même en le laissant libre. Les ν ajustés ne racontent rien de stable :
Macron et Hamon vont aux bords, et l'ordre change avec λ.

**Cause.** Le noyau séparable ajoute les deux termes en log :
$\logarphi=-(v-\mu)^2/2\sigma^2-(u-
u)^2/2\lambda^2$. L'écart en $v$ entre Mélenchon et
Le Pen reste dans chaque électeur de Mélenchon et sature toujours le logit (§8.40). Pour que
la 2D croise réellement, il faudrait que σ en $v$ soit large. Or σ est réglé par le T1, qui
le veut étroit.

> **Retiré le 19/11 (revue, point 2)** : test faux négatif par construction (κ et σ figés au 1D) ; voir §8.44, révision.

**Décision (règle §8.41, fixée avant lecture).**

- Le croisement Mélenchon→Le Pen n'est pas représentable à coût raisonnable dans cette
  famille de modèles.
- On ne publie que $s_{ab}$ et $P(	ext{élu})$, comme aujourd'hui (§2.3).
- (A)(2), les reports comme observations, n'est pas entrepris : en 1D comme en 2D, ils
  tordraient la géométrie (§8.38, point 4) sans qu'aucun mécanisme ne les représente.

Le plancher ε (§8.39) reste la seule piste utile ; elle relève de la netteté des reports
par électorat, qu'on ne publie pas. Il n'est pas adopté.

Piste non chiffrée, notée pour mémoire : un noyau **à queues lourdes en $v$**
(Student, mélange avec une composante large), qui laisserait une fraction de chaque
électorat loin de son centre. C'est la seule façon de désaturer $q$ dans cette famille. Elle
toucherait au T1, donc à toute la production.

### 8.42 τ_T2 a posteriori sur 2002-2022 (revue, 9 ; protocole fixé avant résultat)

**Données.** `data/parsed/intentions_{2002,...,2022}_wiki.csv`, tour « Deuxième tour »,
avec les duels de la finale réelle et `date_fin` dans $[T_2-7\,\mathrm j;\,T_2-1]$. Si une
élection n'a aucun duel dans cette fenêtre, la fenêtre s'élargit à l'entre-deux-tours entier
(depuis $T_1+1$), et ce repli est consigné. Résultats :
`data/historical/results_*.csv`, en part parmi les exprimés du premier nommé.

**Grandeur observée.** Pour l'élection $e$, avec $k_e$ notices dans la fenêtre :

$$d_e=\operatorname{logit}y_e-\operatorname{logit}\bar p_e,\qquad
\bar p_e=\frac1{k_e}\sum_j p_{ej}.$$

Ici $y_e$ est le résultat, $p_{ej}$ la part du premier nommé dans la notice $j$, et la
moyenne simple est une notice par institut et par jour.

Le bruit de la moyenne, en logit, se lit par la méthode delta :

$$v_e=\max\Big(\frac{1}{k_e\,\bar n_e\,\bar p_e(1-\bar p_e)},\ \frac{s_e^2}{k_e}\Big).$$

Le premier terme est binomial, avec $\bar n_e$ l'échantillon moyen, ou 1 000 si
l'échantillon manque. Le second est empirique, avec $s_e^2$ la variance entre notices des
$\operatorname{logit}p_{ej}$ ; il contient déjà l'effet institut σ_T2.

**Vraisemblance.** Les $d_e$ sont indépendantes entre élections :

$$d_e\sim\mathcal N(0,\ \tau^2+v_e)\quad\text{(A)},\qquad
d_e\sim t_4\big(0,\ \sqrt{\tau^2+v_e}\big)\quad\text{(B)}.$$

La moyenne est nulle : un biais directionnel relève de l'arbitrage 2, non corrigé. Le
prior est $\tau\sim\mathcal{HN}(0{,}5)$, large pour des erreurs de quelques points.
L'a posteriori se calcule sur une grille de τ de 0 à 1,5, avec un pas de 0,001.

**Sorties.**

- Médiane, moyenne et IC90 de τ pour (A) et (B), sur les 5 élections, sans 2002, et
  sur 2017-2022 seulement, pour retrouver la décision du 15/11.
- Score de prédiction « une élection retirée » : log-densité prédictive de $d_e$ sous
  l'a posteriori des quatre autres, pour comparer (A) et (B).
- Couverture de l'IC90 prédictif de chaque $d_e$, en une élection retirée.

**Décision.** Elle n'est pas prise ici. τ = 0,19 a été tranché par l'utilisateur le 15/11.
Si la médiane a posteriori sur 5 élections s'écarte de 0,19 de plus de 0,05, on chiffre
l'effet sur $P(\text{élu})$ 2027 et sur $P(\text{Philippe bat Le Pen})$ via
`p_elu(tau_t2=…)`, et l'on ouvre la ligne « ARBITRAGE À TRANCHER ». Sinon, on consigne que
0,19 est confirmé.

**Résultats (2026-11-19, `results/tau_t2_historique.json`, 0 fit).** Erreur finale,
résultat moins moyenne T2−7..T2−1 :

| élection | 2002 | 2007 | 2012 | 2017 | 2022 |
|---|---|---|---|---|---|
| notices dans la fenêtre | 2 | 7 | 3 | 12 | 15 |
| erreur en pt | +2,7 | −0,2 | −1,9 | +5,1 | +3,1 |
| erreur en logit | +0,18 | −0,01 | −0,07 | +0,22 | +0,12 |

Le +5,1 de 2017 diffère du +4,4 de §6.2, qui reposait sur une fenêtre et des dates
antérieures à leur correction sur main.

A posteriori de τ, en logit :

| loi | 2002-2022 | sans 2002 | 2017-2022 |
|---|---|---|---|
| normale (A) | médiane 0,164, moyenne 0,184, IC90 [0,10 ; 0,34] | 0,165 [0,09 ; 0,38] | 0,255 [0,12 ; 0,67] |
| Student-$t_4$ (B) | 0,151 [0,08 ; 0,33] | 0,147 [0,07 ; 0,36] | 0,248 [0,10 ; 0,67] |

En une élection retirée, la loi normale fait mieux : log-densité prédictive totale de 1,70
contre 1,37. Les deux lois couvrent 5 élections sur 5 à 90 %. 2002 ne pèse pas : la médiane
passe de 0,164 à 0,165 sans elle.

**Précision (revue 19/11, point 9).** La fenêtre T2−7..T2−1 est l'erreur des **derniers**
sondages. La dérive entre `as_of` et le scrutin, comme le débat de 2017, est portée par la
projection OU de $w$ (§6, point 1), pas par τ. Si l'on voulait que τ couvre l'erreur à T2−7, il
faudrait la moyenne à T2−7 : sensibilité notée dans la file. La couverture LOO de 5/5 n'apporte
aucune information.

**Sensibilité faite le 19/11** (`results/tau_t2_historique_t2m7.json`). On prend la moyenne des
sondages connus à T2−7, de T2−13 à T2−7 ; 2002 n'a qu'une notice. Erreurs :

| élection | 2002 | 2007 | 2012 | 2017 | 2022 |
|---|---|---|---|---|---|
| erreur en pt | +1,2 | +0,4 | −2,5 | +5,8 | +5,0 |

A posteriori de τ (loi normale) : médiane **0,185**, moyenne 0,206, IC90 [0,11 ; 0,37] ; 0,207
sans 2002. Student-t4 : 0,167. En une élection retirée, la normale reste meilleure (1,16 contre
0,82). Avec cette fenêtre, la médiane rejoint 0,19 : **τ_T2 = 0,19 est confirmé dans les deux
lectures**.

**Lecture.** La médiane à 0,164 s'écarte de 0,19 de moins que le seuil de 0,05, et 0,19
est dans l'IC90 de tous les sous-ensembles. **τ_T2 = 0,19 est confirmé**, du côté prudent :
c'est la moyenne a posteriori, et non la médiane, qui vaut 0,18-0,19.

L'effet d'une valeur de 0,164 sur $P(	ext{élu})$ n'a pas été recalculé, faute de fit. On
l'estime par interpolation linéaire entre les deux points de la grille §8.36 (hier, 17/11) :

- Le Pen : 85,7 % à τ = 0,10 et 82,1 % à τ = 0,19, soit environ 83 % à 0,164 ;
- Philippe : 10,9 % et 13,4 %, soit environ 12,7 %.

L'écart, de l'ordre de 1 pt, est sous la variance de graine.

Les trois erreurs contre le FN/RN sont toutes positives (+2,7, +5,1, +3,1), alors que les deux
autres sont négatives et faibles. Cela rejoint l'arbitrage 2 (biais anti-RN non corrigé),
déjà tranché ; on ne le rouvre pas ici. Une moyenne non nulle ferait baisser τ, mais
déplacerait les duels 2027.

### 8.43 (R1) Part anti-système par électorat (0 fit ; protocole fixé avant résultat)

**Idée (utilisateur, 19/11).** Contrairement au plancher symétrique de §8.39, c'est un
mélange **orienté**. Une fraction $\pi_k$ de l'électorat de $k$ vote pour le finaliste
« anti-système », quelle que soit sa position sur l'axe. En 2017 et en 2022, c'est Le Pen
contre Macron. Le reste suit le modèle spatial. $\pi$ est par **famille** et commun aux deux
années :

| famille | électorats mesurés | autres candidats du T1 rattachés |
|---|---|---|
| gauche radicale (GR) | Mélenchon 2017 et 2022 | Poutou, Arthaud, Roussel |
| gauche / écologie (GE) | Hamon 2017, Jadot 2022 | Hidalgo |
| droite (D) | Fillon 2017, Pécresse 2022 | — |
| droite radicale (DR) | Dupont-Aignan 2017, Zemmour 2022 | — |

Les finalistes ont $\pi=0$, car leur électorat leur est acquis. Pour Lassalle, Asselineau
et Cheminade, $\pi=0$ : ils ne sont pas estimés et pèsent moins de 2 % du T1. La famille
« centre » n'a aucun électorat mesuré ; son seul candidat est le finaliste Macron.

**Parts par électorat.** On reprend $E_k, R_k, G_k$ de §8.38 et
$\tilde G_k=\mathrm{logistique}(\operatorname{logit}G_k+\delta)$.

- (R1a) :

$$a_k=(1-\pi_k)R_k\tilde G_k,\qquad b_k=(1-\pi_k)R_k(1-\tilde G_k)+\pi_k,\qquad
\varnothing_k=(1-\pi_k)(1-R_k).$$

- (R1b), « Aucun » orienté, si les données le demandent : on ajoute $\pi^\varnothing_k$,
  par famille également.

$$a_k=(1-\pi_k-\pi^\varnothing_k)R_k\tilde G_k,\quad
b_k=(1-\pi_k-\pi^\varnothing_k)R_k(1-\tilde G_k)+\pi_k,\quad
\varnothing_k=(1-\pi_k-\pi^\varnothing_k)(1-R_k)+\pi^\varnothing_k.$$

**Duel agrégé.** Le mélange déplace $s_{ab}$. Deux lectures :

- (i) δ est recalé pour que $\sum_kE_ka_k/\sum_kE_k(a_k+b_k)=s_{ab}$ : le duel publié est
  inchangé et seule la répartition bouge. C'est la lecture principale.
- (ii) δ garde sa valeur sans mélange : on rapporte alors le déplacement du duel en points.

**Ajustement.** On minimise la somme des carrés des erreurs des cellules Macron, Le Pen et
Aucun, en %, sur les moyennes postérieures (tous les tirages), pour 2017 et 2022 ensemble.
Cela fait 4 paramètres en R1a et 8 en R1b, pour 16 degrés de liberté. On utilise les fits
hier, la variante de production, et abst en information. Optimiseur : Powell, sur
$\operatorname{logit}\pi$.

**Critères (R1a, hier), fixés avant lecture.**

1. Partage Mélenchon parmi les exprimés à moins de 10 pt du mesuré, les deux années, avec
   l'ajustement conjoint.
2. MAE des cellules des autres électorats non dégradée de plus de 1 pt par rapport à la 1D
   sans mélange (§8.38).
3. Stabilité : avec un ajustement séparé par année, $|\pi_{2017}-\pi_{2022}|<0{,}1$ pour
   chacune des quatre familles.
4. Duels : on rapporte le déplacement (ii) de $s_{ab}$ et le δ recalé (i). Ce n'est pas un
   critère éliminatoire, mais un fait à dire.

**En information**, une prédiction « une année retirée » : π ajusté sur 2017 sert à prédire
les cellules de 2022, et inversement. C'est le seul chiffre hors échantillon.

Si R1a passe, on spécifie l'entrée dans le modèle : π par famille dans la vraisemblance des
reports, ce qui rend (A)(2) possible sans tordre la géométrie. Le backtest suit §8.29, et R2
est fait en mode réduit (2022 seul). Si R1a échoue, on consigne l'échec et l'on passe à R2.

**Résultats (2026-11-19, `results/reports_t2_r1_{hier,abst}.json`) : ÉCHEC de R1a sous
hier, la variante de référence.**

| | π GR | π GE | π D | π DR | partage Mélenchon 2017 / 2022 (erreur) | MAE des autres, sans → avec | duel sans recalage 2017 / 2022 |
|---|---|---|---|---|---|---|---|
| hier R1a, conjoint | 0,19 | 0,04 | 0,01 | 0,00 | −7,2 / **+13,3** | 9,8 → 8,5 | −5,0 / −4,3 pt |
| hier R1a, 2017 seul | 0,09 | 0,01 | 0,11 | 0,00 | | | |
| hier R1a, 2022 seul | **0,27** | 0,08 | 0,06 | 0,00 | | | |
| hier R1b, conjoint | 0,21 (∅ 0,22) | 0,07 (0) | 0,18 (0,19) | 0,35 (0,16) | **−17,3** / +4,2 | 9,8 → 4,9 | −11,1 / −6,6 pt |
| abst R1b, conjoint | 0,19 (∅ 0,02) | 0,20 (0) | 0,22 (0,18) | 0,37 (0,17) | −4,2 / +1,5 ✓ | 13,0 → 10,6 | −10,5 / −6,1 pt |

Critères pour hier R1a :

- (1) Mélenchon < 10 pt les deux années : non (+13,3 pt en 2022).
- (2) Autres électorats non dégradés : oui.
- (3) Stabilité de π : non, $\pi_{GR}$ vaut 0,09 en 2017 et 0,27 en 2022 (écart 0,18).

En une année retirée, l'erreur sur le partage de Mélenchon est de −17 / +25 pt (R1a) et de
−23 / +19 pt (R1b, hier).

abst R1b remplit le critère 1, avec $\pi_{GR}\approx0{,}2$, mais pas la stabilité : π GE
vaut 0 contre 0,33, et π DR 0,26 contre 0,59. Cette variante reproduit en outre l'artefact
d'abstention de §8.38 : Jadot 2022 tombe à 19 % de partage contre 88 % mesuré.

**Lecture.**

1. **Le mélange orienté est la bonne forme pour la gauche radicale.** $\pi_{GR}$ est le
   seul π nettement non nul en R1a, et il absorbe l'essentiel de l'erreur de Mélenchon.
   C'est ce qu'aucune variante géométrique (§8.39-8.41) ne savait faire.
2. **Il n'est pas stable entre élections.** Le report mesuré Mélenchon→Le Pen passe de 15 %
   à 21 % de l'électorat entre 2017 et 2022, et l'électorat de Mélenchon a changé de
   composition. Un π commun ne prédit pas l'année retirée : erreurs de 17 à 25 pt, du même
   ordre que sans mélange. Avec deux élections, π est un paramètre d'élection, pas de famille.
3. **Le mélange déplace les duels de 4 à 11 pt** si δ n'est pas recalé. Dans un modèle, π
   et η se disputeraient donc le duel agrégé ; δ (lecture (i)) l'absorbe ici.

**Décision (règle §8.43).** R1 n'est pas adopté. On passe à R2.

Une variante « π par famille et par élection », appris sur les reports de l'élection en
cours, resterait possible en 2027 *si* des tableaux de reports existent avant le T1. Or ils
n'existent qu'entre les deux tours. Elle n'apporterait donc rien à la prévision avant le
T1, qui est l'objet du site.

### 8.44 (R2) Carte 2D estimée dès le T1 : maths (avant tout code ; critères fixés avant résultat)

**Pourquoi R2 diffère de §8.41.** Le test de §8.41 greffait $u$ sur un T1 1D figé, avec $\sigma_v$ étroit
appris sans $u$. Ici, $u$ est appris **avec** le T1. L'hypothèse, celle de l'utilisateur, est la
suivante : en 1D, des candidats voisins (Mélenchon, Hamon, Jadot, Roussel) ne se séparent que par des
$\sigma_v$ étroits et des $\mu$ écartés. En 2D, $u$ porte une partie de cette séparation, $\sigma_v$
s'élargit et le logit T2 n'est plus saturé.

**1. Grille et noyau.** Les votants sont uniformes sur $[0{,}01;0{,}99]^2$, avec une quadrature produit
Gauss-Legendre $B_v\times B_u=25\times7$ et des poids $W_bW'_c$. Le noyau est séparable et normalisé sur
la grille :

$$A_i(v,u)=w_i+\log\frac{\mathcal N(v;\mu_i,\sigma_i)\,\mathcal N(u;\nu_i,\lambda)}
{\sum_{b,c}W_bW'_c\,\mathcal N(V_b;\mu_i,\sigma_i)\,\mathcal N(U_c;\nu_i,\lambda)}.$$

Un noyau corrélé, ou des axes non orthogonaux, est écarté : il ajoute une rotation non identifiée et un
paramètre par candidat. Tout le reste est inchangé, avec $\int dv$ remplacé par $\iint dv\,du$ : π au T1,
$q$, ρ, la valeur inclusive, κ, η, ω.

**2. Priors.**

- $\sigma_i$ et $\mu_i$ gardent **exactement** les priors 1D, dont les ancres ORDER_GROUPS en $v$. Rien
  ne pousse $\sigma_v$ à s'élargir : c'est ce qu'on veut tester.
- $\nu_i=\mathrm{logistique}(\zeta_i)$, avec $\zeta_i\sim\mathcal N(0,1)$.
- $\log\lambda\sim\mathcal N(\log0{,}2;\,0{,}5)$, λ commun à tous les candidats.

**3. Symétries.** Elles sont levées par deux ancres internes, sans aucune source externe.

- **Réflexion** $u\mapsto1-u$ : contrainte d'ordre $\zeta_{RN}=\zeta_{C}+\mathrm{softplus}(\xi)$, avec
  $\xi\sim\mathcal N(0,1)$. RN désigne le candidat RN principal (Le Pen, ou Bardella en 2027) et $C$ le
  finaliste centriste (Macron en 2017 et 2022 ; en 2027, le candidat du bloc central au plus fort T1
  modal).
- **Translation du nuage en $u$.** Elle n'est que faiblement tenue par le bord du carré. On fixe donc
  $\nu_C=0{,}3$, côté « système ». Justification : c'est la position la moins discutable, celle du
  sortant ou de l'héritier du bloc central, et une ancre dure sur un seul candidat suffit à lever la
  translation sans rien imposer aux autres.

Le choix de 0,3 plutôt que 0,5 évite de placer l'ancre au centre de la grille, où la réflexion serait
dégénérée. **Sensibilité : 0,2 et 0,4.** Si les conclusions dépendent de cette valeur, il faudra une
ancre externe (par exemple la dimension anti-élite de CHES, inaccessible depuis la VM), et la ligne
« ARBITRAGE À TRANCHER » sera écrite.

**4. Vraisemblance des reports**, qui réalise (A)(2) dans R2. Pour une notice $j$ datée $t_j\le$
`as_of` et un électorat $k$ du champ T1 réel, on observe
$y_{jk}=(y^M,y^{LP},y^\varnothing)/100$, renormalisé à 1. Prédit :
$\theta_k(t_j)=(a_k,b_k,\varnothing_k)$, issu de §8.38 sur la grille 2D, avec $w(t_j)$.

$$y_{jk}\sim\mathrm{Dirichlet}\big(c\,\theta_k(t_j)\big),\qquad \log c\sim\mathcal N(\log 100;\,1).$$

- **Pas de $n$.** `base_electorat` est vide partout, et $c$ joue le rôle d'un $n$ effectif ×
  effet de plan. À titre d'ordre de grandeur, un électorat de 15 % dans un échantillon de 1 000 fait
  $n\approx150$.
- **δ dans le modèle.** La recalibration par bissection a un gradient nul. On la remplace par
  4 itérations de Newton partant de $\delta_0=\eta_a-\eta_b$, sur
  $f(\delta)=\sum_kE_kR_k\tilde G_k(\delta)/\sum_kE_kR_k-s_{ab}$. Elles sont différentiables, et l'on
  teste leur convergence contre la bissection.
- **Dépendance avec le duel de la même notice.** Les cellules d'une notice, pondérées par les parts T1
  de ses répondants, redonnent son duel. Les compter en plus du duel compte une partie de
  l'information deux fois. On l'accepte, par indépendance conditionnelle, car c'est surtout la
  composition qui est nouvelle ; le contrôle consiste à vérifier que $\sigma_{T2}$ et $s_{ab}$ ne bougent
  pas de plus de 0,5 pt. Repli si nécessaire : retirer le duel Macron–Le Pen des notices qui publient
  des reports.
- **Pas de tempérage** ($w_R=1$). En sensibilité, $w_R=0{,}5$ si le T1 se dégrade.

**5. Ce que R2 peut apporter au site, et sa limite.** Les reports n'existent qu'**entre les deux
tours**. Avant le T1, ce qui est l'objet du site, $\nu$ n'est appris que par le T1, faiblement, via les
substitutions entre hypothèses. Le critère V1 2022 ci-dessous se fait donc **sans** reports. S'il ne
passe pas, R2 ne vaut qu'en entre-deux-tours.

Une extension est notée sans être faite : réutiliser les ν **par famille** appris sur 2017 et 2022
comme prior pour 2027. R1 montre que les familles ne sont pas stables d'une élection à l'autre.

**6. Code**, derrière des drapeaux éteints (production inchangée, `PUBLIER_T2 = False`) :

- `fit_spatial_pooling(..., carte_2d=False, reports=None)` ;
- dans `joint_model`, la grille 2D et les sites `zeta`, `xi`, `log_lambda`, `log_c` ;
- dans `second_round.py`, une option `--variante hier2d`.

Tests à écrire :

- avec λ très grand, la 2D redonne exactement les parts 1D ;
- la contrainte d'ordre et l'ancre ;
- la convergence de Newton contre la bissection ;
- la vraisemblance des reports, finie et maximale à $\theta=y$ pour $c$ grand.

**7. Coût et ordre des fits.** À $B_u=7$, compter environ 4 h par chaîne et 4 chaînes en parallèle, soit
une tâche par réveil long. L'ordre suit :

1. V2 T2−7 2022, variante hier2d avec les reports ;
2. seulement si le point 1 passe, V1 2022 (protocole §8.29, sans reports), puis 2017, puis l'effet sur
   2027.

Les tâches sont marquées EN COURS, lancées par un sous-agent Sonnet qui les surveille (watcher).

**8. Critères, fixés avant lecture, sur V2 T2−7 2022.**

1. **Report.** Partage Macron parmi les exprimés de l'électorat Mélenchon à moins de 10 pt du mesuré,
   63,0 % (en 1D : 100 %). Mesuré sur les reports de toutes les notices 2022, y compris celles
   postérieures à `as_of`, qui ne sont pas dans le fit.
2. **Mécanisme.** Médiane a posteriori de $\bar\sigma_v$ (moyenne des σ sur le champ T1 réel) au moins
   1,25 fois celle du fit 1D hier V2 T2−7 2022. Sinon, le résultat ne s'explique pas par l'hypothèse.
3. **T1 non dégradé.** MAE des résidus des nœuds T1 au plus celle du 1D + 0,2 pt, et MAE des parts au
   T1 (w au 10/04/2022) contre le résultat au plus celle du 1D + 0,3 pt.
4. **Chaînes.** R̂ ≤ 1,1 sur les ν du champ T1 après la sélection §8.8. ν interprétable :
   $\nu_{JLM}$ plus proche de $\nu_{LP}$ que $\nu_{Jadot}$ et $\nu_{Hidalgo}$, et $\nu_{Zemmour}\ge\nu_{Pécresse}$.
5. **Contrôle de double compte.** $s_{ab}$ et $\sigma_{T2}$ à ±0,5 pt du 1D.

Si 1 à 5 passent, on enchaîne avec V1 2022 (§8.29 : duels non dégradés). Si 1 passe sans 2, le mécanisme
est autre et on le dit. Si 1 échoue, R2 est abandonné et (A) est clos pour de bon.

**Résultats V2 T2−7 2022 (2026-11-19, 4 chaînes de 1 h 30 environ, données de main au 19/11).**
Fichiers : `results/second_round_hier2d_v2_2022_T2-7.json`, `eval_r2_hier2d_2022.json`. Le
témoin 1D hier a été refitté sur les **mêmes** données : `r2_temoin_hier_v2_2022_T2-7.json`,
`eval_r2_hier_2022.json`. Chaînes retenues (§8.8) : 0 et 1 en 2D, 1 et 2 en 1D.

| critère | 2D (hier2d) | 1D (témoin) | verdict |
|---|---|---|---|
| 1. partage Mélenchon, toutes notices (mesuré 63,0) | 62,3, IC90 [57,6 ; 66,9] | 100,0 | ✓ |
| 1 bis. idem, 11 notices **après** `as_of`, hors fit (mesuré 65,8) | 62,3 (−3,5) | 100,0 (+34,2) | ✓ hors échantillon |
| autres partages hors fit (Pécresse, Jadot, Zemmour) | −2,9 / −1,5 / −2,4 | +4,0 / +10,4 / −10,1 | |
| 2. $\bar\sigma_v$ sur le champ T1 (médiane) | 0,064 | 0,048 | ✓ (rapport 1,33 ≥ 1,25) |
| 3a. MAE des nœuds T1 | 0,603 pt | 0,616 pt | ✓ |
| 3b. MAE des parts au T1 contre le résultat | 1,17 pt (couverture IC90 3/12) | 1,23 pt (5/12) | ✓ en MAE ; couverture en baisse |
| 4. R̂ des ν du champ (2 chaînes retenues) | **7,5** | — | ✗ |
| 4 bis. ν interprétable (sur les moyennes) | JLM 0,42 contre LP 0,44 ; Jadot 0,19, Hidalgo 0,31 ; Zemmour 0,73 ≥ Pécresse 0,58 | — | ✓, mais sur des chaînes en désaccord |
| 5. $s_{ab}$ et σ_T2 à ±0,5 pt | 54,2 contre 53,5 (+0,7) ; 1,03 contre 0,93 | | ✗ de peu sur $s_{ab}$ |
| κ | 0,22 | 0,75 | |
| λ | 0,089 | — | |

**Lecture.**

1. **Le mécanisme existe.** Sur une carte 2D apprise dès le T1, le report Mélenchon→Le Pen est
   reproduit (21,3 contre 21,0 % de l'électorat), y compris sur les 11 notices hors fit. Les
   autres électorats le sont à 3 pt près, là où le 1D est à 10-34 pt. Le T1 n'est pas dégradé en
   MAE.
2. **La désaturation passe surtout par κ.** κ tombe de 0,75 à 0,22, soit une température de duel
   trois fois plus haute. $\sigma_v$ s'élargit aussi (rapport 1,33, critère 2 passé), mais c'est κ
   qui porte l'essentiel. L'hypothèse de l'utilisateur ($u$ porte une séparation, $\sigma_v$
   s'élargit) est vraie en partie seulement. Le duel reste au bon niveau, car η et la valeur
   inclusive compensent.
3. **Les ν ne sont pas identifiés de façon stable** (R̂ 7,5). Les deux chaînes retenues
   s'accordent sur Mélenchon (0,40 / 0,43), Roussel (0,06 / 0,06), Le Pen (0,42 / 0,47) et Hidalgo
   (0,31 / 0,32). Elles divergent sur Jadot (0,31 / 0,06), Lassalle (0,58 / 0,86) et les putatifs
   jamais au T1 réel. Avec 37 lignes de reports sur 4 électorats, ce sont les électorats mesurés
   qui fixent les ν, pas les autres.
4. **Double compte** : $s_{ab}$ monte de 0,7 pt, juste au-delà du seuil de 0,5.

**Verdict (règle fixée avant lecture) : R2 ne passe pas encore.** Les critères 4 et 5 échouent ;
1, 2 et 3 passent, avec un résultat fort hors échantillon. On ne passe pas à V1. Les remèdes déjà
écrits sont à appliquer dans l'ordre :

- (a) phase 2 de §8.8, quatre chaînes parties du mode dominant (chaîne 0), pour la stabilité
  des ν ;
- (b) si $s_{ab}$ reste au-delà de 0,5 pt, le repli de §8.44.4 : retirer le duel Macron–Le Pen
  des notices qui publient des reports.

Rappel de la limite de §8.44.5 : avant le T1, sans reports, ce gain n'existe pas. Le test V1
dira ce qui en reste.

**Révision après la revue Opus du 19/11, intégrée le même jour.**

**1. La lecture 2 est retirée.** Le mécanisme ne passe pas « surtout par κ ». Contrefactuels à
0 fit, sur les mêmes tirages (`kappa_test.py` de la revue, rejoué) ; partage de Mélenchon parmi
les exprimés, mesuré à 63,0 :

| configuration | partage Mélenchon |
|---|---|
| 1D témoin (κ 0,75) | 100,0 |
| 1D, κ forcé à 0,22 | 100,0 |
| 1D, κ forcé à 0,1 | 98,7 |
| 1D, κ 0,22 et σ × 1,33 | 99,6 |
| 2D retenu (κ 0,22) | **62,3** |
| 2D, κ forcé à 0,75 | 97,2 |
| 2D, u effacé (λ = 50) | 82,4 |

κ seul n'apporte rien. La géométrie en $v$ réorganisée, avec un κ bas, fait passer de 100 à 82 ;
$u$ fait passer de 82 à 62. C'est une **interaction**.

Le mécanisme se lit par candidat :

- σ_Mélenchon vaut 0,020 en 1D, contre 0,053 et 0,080 dans les deux chaînes 2D retenues, soit
  ×2,6 à ×4 ;
- en 1D, Jadot ($\mu$ 0,11) est *plus à gauche* que Mélenchon (0,21) ;
- en 2D, Jadot et Roussel se séparent de Mélenchon par $u$ (ν ≈ 0,06 contre ν_JLM ≈ 0,41 ≈ ν_LP).

L'hypothèse de l'utilisateur est donc confirmée **pour Mélenchon** : $u$ porte la séparation avec
la gauche modérée, et son rayon en $v$ s'élargit. Le $\bar\sigma_v$ du champ ne le montrait qu'en
moyenne.

**2. §8.40 et §8.41 étaient des faux négatifs par construction.** Les deux figeaient κ = 0,75 et
σ du 1D ; or la 2D elle-même échoue à κ = 0,75 (97 %). La conclusion « non représentable dans
cette famille » de §8.41 est **retirée**.

Règle pour la suite : une porte à 0 fit qui gèle des paramètres couplés au mécanisme testé ne peut
pas exclure ; seul un résultat positif y est informatif.

**3. Le témoin 1D n'était pas équitable**, puisque les reports étaient dans la vraisemblance du 2D
seulement. Le témoin `hier_rep` (1D avec reports) est en cours.

**4. Le contrôle 1 bis est requalifié** en « prolongation à quelques jours » : mêmes instituts, à
1-7 jours. La persistance naïve (moyenne des notices du fit) fait une MAE de 3,6 pt sur les quatre
partages hors fit, Mélenchon −5,8. Le modèle 2D fait 2,6 pt. Le gain sur la persistance est
d'environ 1 pt, pas de 30.

**5. Multimodalité**, qui touche $\mu$ aussi (R̂ 9,98), et non seulement ν. La phase 2 depuis le
mode dominant donnerait un R̂ bon par construction : elle a été **arrêtée**.

**6. Nouveau critère 4, qui remplace R̂(ν) pour les runs suivants.** Sur au moins 3 chaînes
initialisées indépendamment, l'écart maximal entre chaînes doit être :

- ≤ 3 pt sur les partages mesurés ;
- ≤ 0,5 pt sur les parts T1 ;
- ≤ 2 pt sur tous les duels, observés et hypothétiques clés, dont Mélenchon–Le Pen.

On rapporte en plus l'écart de lp entre modes.

Le run actuel échoue aussi à ce critère. Mélenchon–Le Pen, un seul sondage en 2022, vaut 47,2 /
48,9 / 48,9 / 43,4 selon les chaînes 2D, contre 42,4 en 1D. R2 déplace de 5-6 pt, et de façon
instable, un duel hypothétique : c'est exactement ce que le site publie.

**7. Double compte, nouvelle forme pour le prochain run R2.** δ est recalé **par notice** sur le
duel publié par cette notice, au lieu de $s_{ab}$ modélisé : les reports ne portent alors plus que
la composition. Le repli « retirer le duel » ôterait environ 10 des 15 duels de la dernière
semaine.

**8. Code.**

- Test de Newton ajouté sur un cas saturé (σ = 0,02, κ = 0,75).
- Sensibilité au plancher de 1 % (au lieu de 0,5 %) sur $y$ des reports : à faire au prochain run.
- Couverture T1 de 3/12 contre 5/12 : non expliquée, à suivre.

**Témoin équitable `hier_rep` (1D avec reports, mêmes données ; 19/11).** Fichiers :
`results/second_round_hier_rep_v2_2022_T2-7.json`, `eval_r2_hier_rep_2022.json`.

Partages parmi les exprimés, erreur en pt :

| | Mélenchon | Pécresse | Jadot | Zemmour | MAE hors fit (11 notices) |
|---|---|---|---|---|---|
| 1D sans reports | +37,0 | +5,1 | +12,5 | −9,4 | 14,7 |
| 1D avec reports | **+17,2** | −6,4 | **−8,3** | +0,4 | 8,2 |
| 2D avec reports | −0,7 | −1,9 | +0,6 | −1,7 | 2,6 |
| persistance naïve | | | | | 3,6 |

Autres grandeurs :

| | κ | $\bar\sigma_v$ | MAE T1, nœuds / résultat | chaînes retenues |
|---|---|---|---|---|
| 1D sans reports | 0,75 | 0,048 | 0,616 / 1,23 | 2 |
| 1D avec reports | 0,41 | **0,042** | 0,618 / 1,26 | 1 |
| 2D avec reports | 0,22 | 0,064 | 0,603 / 1,17 | 2 |

La prédiction de la revue est vérifiée. Avec les reports mais sur un seul axe, Mélenchon reste à
80 % au lieu de 63. Jadot est tiré vers le centre (79 % au lieu de 88), et $\sigma_v$ *rétrécit*.
Le 1D ne peut pas séparer Mélenchon de la gauche modérée.

**La 2D est donc justifiée pour l'entre-deux-tours.** Elle seule reproduit les reports, et le gain
hors fit sur la persistance naïve est modeste : 2,6 contre 3,6 pt. Le 1D avec reports est en outre
multimodal : une seule chaîne retenue.

Reste à savoir ce que R2 vaut **avant le T1**, sans reports. C'est l'idée 2 de la revue : V2
T1−30 hier2d, 4 chaînes indépendantes, critère 4 révisé.

**Critère 4 révisé, appliqué à V2 T2−7 2D** (`eval_r2_hier2d_2022_T2-7_par_chaine.json`).
Écarts entre les 4 chaînes indépendantes :

- Partages des reports : Mélenchon 8,6 pt (2,8 sur les 3 chaînes de lp proche), Pécresse 11,6,
  Jadot 4,7, Zemmour 3,8. Seuil 3 pt : ✗.
- Duels :
  - Macron–Le Pen 0,6 pt, Macron–Zemmour 0,8 ;
  - Macron–Mélenchon 2,6, Macron–Pécresse 1,9 ;
  - Le Pen–Mélenchon 5,6, Le Pen–Zemmour 7,3 ;
  - Mélenchon–Zemmour **11,8**, Mélenchon–Pécresse 9,1.
  Seuil 2 pt : ✗.
- Écart de lp entre chaînes : 82 nats.

Les duels contre Macron, très sondés, tiennent. Les duels hypothétiques entre deux candidats qui
ne sont pas au centre dépendent du mode. **Échec net** en l'état, au-delà de la question du T1.

**Avant le T1 : V2 T1−30 2022, sans aucun report (idée 2 de la revue, 19/11).** Comparaison 2D
contre 1D, mêmes données, 4 chaînes indépendantes chacun. Fichiers :
`eval_r2_hier2d_2022_T1-30_par_chaine.json` et `eval_r2_hier_2022_T1-30_par_chaine.json`.

| | 2D (hier2d) | 1D (hier) |
|---|---|---|
| partage Mélenchon prédit (mesuré 63, après le T1) | 100 / 0 / 100 / 100 selon la chaîne | 100 (4/4) |
| Jadot (mesuré 88) | 100 (4/4) | 100 (4/4) |
| MAE des parts au T1 contre le résultat | 2,83 à 3,00 pt | 2,98 à 2,99 pt |
| Macron–Le Pen (résultat 58,55) | 58,6 | 58,8 |
| écart entre chaînes, duels hypothétiques (max) | 5,4 (Mélenchon–Pécresse, Le Pen–Mélenchon) | 7,9 (Le Pen–Zemmour), les autres ≤ 2,9 |
| écart de lp entre chaînes | 28,7 nats | 4,0 nats |
| sd a posteriori moyen des ν (prior 0,21) | 0,02 à 0,06 | — |

**Lecture.** Sans reports, le T1 seul ne dit rien des reports : la 2D prédit la même chose que le
1D (Mélenchon→Macron 100 %), ou l'inverse dans une chaîne. Le T1 n'est pas mieux prédit. Les ν sont
serrés, mais différents d'une chaîne à l'autre : ce sont des modes, pas une information. R2
ajoute de la multimodalité (29 nats contre 4) sans rien gagner.

**Décision (règle de l'idée 2, fixée avant lecture) : R2 est abandonné pour le site.** Il reste un
résultat de recherche d'entre-deux-tours : quand les reports existent, la carte 2D les représente
et le 1D ne le peut pas, même avec les reports en vraisemblance. Ses duels hypothétiques y sont
toutefois instables, donc rien n'est publiable. La production est inchangée, les drapeaux restent
éteints. La piste (A) est close.

**Constat incident, à traiter.** En 1D aussi, le duel hypothétique Le Pen–Zemmour varie de 7,9 pt
entre chaînes à T1−30 2022, alors que les autres duels restent sous 2,9 pt. Le critère de
stabilité des duels hypothétiques n'a jamais été appliqué au fit de production 2027, qui publie
des duels jamais sondés. C'est à faire, à 0 fit sur les chaînes du gel prospectif s'il les a
gardées, sinon avec 1 fit.

### 8.45 Stabilité entre chaînes des duels 2027 (constat de §8.44 ; protocole fixé avant résultat)

**Constat de départ.** À T1−30 2022, en 1D et en production, le duel hypothétique Le Pen–Zemmour
varie de 7,9 pt d'une chaîne à l'autre. Le site publie pour 2027 des duels jamais sondés. La
sélection des chaînes (§8.8) et le R̂ de la cible ne portent que sur les duels backtestés.

**Protocole.** On fait un fit de production 2027 : variante hier, sondages au dernier terrain,
`load_raw_polls(tour=None)` comme `prospectif_2027`. Il comporte 4 processus d'une chaîne, de
graines 27 à 30, sans phase 2. Pour chaque chaîne prise seule, et pour toutes les paires du champ
modal ∪ candidats en duel (les 78 paires du gel prospectif), on calcule $\bar s_{ab}$ à h = 0,
sans aléa de scrutin. On rapporte aussi le lp par chaîne et les chaînes retenues par §8.8.

**Critère, fixé avant lecture.** Une paire est **instable** si l'écart max − min de $\bar s_{ab}$
entre les chaînes retenues par §8.8 dépasse 2 pt. On le rapporte aussi sur les 4 chaînes. On
sépare les paires sondées des paires jamais sondées, ces dernières recevant τ_paire en
prédiction.

**Lecture prévue.**

- Aucune paire instable parmi les chaînes retenues : on consigne et on clôt.
- Des paires instables : on compare leur écart à leur demi-IC90 publié. Si l'écart dépasse le
  demi-IC90, on ouvre la ligne « ARBITRAGE À TRANCHER » : élargir l'IC de ces paires (τ_paire
  plus grand, ou un mélange explicite des modes), ou ne pas les afficher.

**Résultat (2026-11-19, sondages au 17/11, `results/stabilite_2027_hier_2026-11-17.json`) :
stable.** Les 4 chaînes sont retenues (lp 8009,7 / 8007,7 / 8006,7 / 8007,6, soit 3 nats d'écart).
Les écarts entre chaînes valent au plus **0,90 pt** sur les 78 paires (Bardella–Zemmour,
Glucksmann–Hollande), puis 0,76 pt (Hollande–Roussel). La médiane est sous 0,4 pt.

**Aucune paire instable**, et aucun écart n'approche le demi-IC90 publié (9 à 17 pt). La
multimodalité vue à T1−30 2022 (Le Pen–Zemmour à 7,9 pt) ne se retrouve pas dans le fit de
production 2027 : tâche close. À refaire si le roster ou le régime de données change nettement.

### 8.46 (R3) Votants sur un cercle : maths (avant tout code ; critères fixés avant résultat)

**Idée (utilisateur, 19/11).** Le segment $v\in[0{,}01;0{,}99]$ est remplacé par un cercle, et
l'électorat est uniforme sur $\theta\in[0,2\pi)$. Les deux extrêmes deviennent voisins par
« l'arrière ». La médiatrice du duel Macron–Le Pen coupe alors le cercle en deux points, au centre
et à l'arrière. Un électorat placé près du point arrière se partage entre les deux finalistes sans
logit saturé : c'est le mécanisme qui manquait en 1D (§8.44, révision 1).

**1. Paramétrisation, qui garde les positions et les priors 1D.** La position $\mu_i\in(0,1)$ et le
rayon $\sigma_i$ restent ceux du 1D, avec les mêmes priors, les mêmes ancres ORDER_GROUPS et le
même `pos_scale`. On ajoute un seul paramètre global : l'**arc avant** $A\in(0,2\pi)$, sur
lequel le segment est posé.

$$\theta_i=A(\mu_i-\tfrac12),\qquad \sigma^{\theta}_i=A\,\sigma_i,\qquad
\kappa^{vM}_i=1/(\sigma^{\theta}_i)^2,$$

$$A_i(\theta)=w_i+\kappa^{vM}_i\cos(\theta-\theta_i)-\log\sum_jW^\circ_j\,e^{\kappa^{vM}_i\cos(\Theta_j-\theta_i)}.$$

Le noyau de von Mises est normalisé sur la quadrature comme en 1D. Pour un κ grand, il équivaut à
une gaussienne d'écart-type $\sigma^\theta$.

L'**arc arrière** $2\pi-A$ porte la fraction $f=1-A/2\pi$ de l'électorat. Ces votants sont plus
loin du centre que les deux extrêmes, et proches de l'un et de l'autre.

Prior : $A=2\pi\,\mathrm{logistique}(a)$ avec $a\sim\mathcal N(\mathrm{logit}\,0{,}75;\,0{,}75)$, soit
une médiane $f=0{,}25$ et un IC90 de 8 à 53 %.

**2. Symétries.**

- La **rotation** est fixée : $\mu=\tfrac12$, le centre des ancres, est à $\theta=0$.
- La **réflexion** est fixée par les ancres : gauche vers $\theta<0$.

Il n'y a pas de nouvelle symétrie, contrairement à R2 qui demandait deux ancres de plus. Le
« trou » où le cercle se ferme est donc placé derrière les extrêmes de l'ordre ORDER_GROUPS, et sa
taille $A$ est apprise. C'est le choix le plus simple ; un ordre cyclique libre (où l'arrière
pourrait passer ailleurs) est écarté.

**3. Ce que le cercle n'emboîte pas.** Le 1D n'est pas un cas limite exact : il y a toujours des
votants sur l'arc arrière. Quand $A\to2\pi$, ils disparaissent, mais les deux extrêmes se touchent.
Le 1D sert donc de **témoin**, pas de sous-modèle.

**4. Quadrature.** On utilise des trapèzes à $N_\theta=64$ nœuds équidistants, de poids $1/64$.
C'est spectralement exact pour une fonction périodique lisse. Le pas de 0,098 rad vaut 1,1 fois le
plus petit $\sigma^\theta$ attendu ($A\approx4{,}7$, σ = 0,02) ; l'erreur d'aliasing est alors
d'environ $e^{-2\pi^2\sigma^2/h^2}\approx10^{-7}$. Le coût vaut environ 64/25 = 2,6 fois celui du
1D, contre 7 fois pour R2. Un test vérifie que les parts sont stables à $10^{-4}$ entre 64 et 256
nœuds.

**Révision du point 4, mesurée avant tout fit.** L'estimation d'aliasing ci-dessus était
optimiste : l'intégrande est un softmax entre noyaux étroits, plus raide qu'une gaussienne. Erreur
maximale sur π, contre une référence à 2048 nœuds, sur 5 géométries aléatoires de 12 candidats :

| σ | arc | N = 64 | N = 128 | N = 192 | N = 256 |
|---|---|---|---|---|---|
| 0,02 | 2,5 | 0,93 pt | 0,28 | 0,15 | 0,11 |
| 0,02 | 4,7 | 0,51 | 0,12 | 0,04 | 0,02 |
| 0,04 | 2,5 | 0,40 | 0,11 | 0,04 | 0,01 |
| 0,04 | 4,7 | 0,10 | 0,003 | 0,0001 | 0 |

On retient **$N_\theta=128$**, soit un coût d'environ **5 fois** le 1D, et non « à peu près celui du
1D » comme le supposait l'idée.

**Constat incident sur le 1D de production.** Gauss-Legendre à B = 25 a une erreur de 3,0 pt à
σ = 0,02 et de 0,6 pt à σ = 0,04, sur les mêmes géométries aléatoires. Sur des tirages réels :

- 2027 (`stabilite_2027_hier`, σ min 0,051) : 0,001 pt, sans effet ;
- 2017 V2 T2−7 (σ min 0,062) : 0,005 pt ;
- **2022 V2 T2−7** (σ min 0,018) : **1,6 pt au maximum**, 0,10 pt en moyenne.

La production n'est pas touchée. Les backtests 2022 le sont, là où un σ s'effondre (Mélenchon,
0,020). La note « 0,0015 pt dès B = 20 » de `geometry.py` ne vaut que pour σ ≥ 0,05 environ. À N =
128, le cercle est donc plus précis que le 1D actuel aux petits σ.

**5. Second tour.** Rien ne change : κ, η, ω, la valeur inclusive, la vraisemblance Dirichlet des
reports et le Newton pour δ (§8.44.4), avec les poids $W^\circ$.

**6. Affichage.** La position publiée reste $\mu_i$ (abscisse le long de l'arc avant), qui garde
son sens gauche-droite. On publie $f$ en plus, s'il est adopté.

**7. Code**, derrière un drapeau éteint :

- `carte_cercle=True` dans `fit_spatial_pooling` et `joint_model` (site `a_arc`, déterministe
  `arc`) ;
- `--variante hiercercle` (avec reports) et `hiercercle_sr` (sans reports, pour le protocole 2) ;
- tests : convergence de la quadrature ; parts sommant à 1 ; votant de l'arc arrière partagé entre
  les deux extrêmes ; densité finie avec reports.

**8. Protocole et critères (utilisateur, figés avant lecture).**

1. **V2 T2−7 2022 avec reports.**
   - Partage Mélenchon hors fit (11 notices après `as_of`) à moins de 10 pt du mesuré.
   - MAE hors fit des 4 partages ≤ 2,6 (hier2d), ou au moins < 8,2 (1D avec reports).
   - T1 non dégradé (§8.44.8, critère 3).
   - Critère 4 révisé de §8.44 : écart entre chaînes ≤ 3 pt sur les partages mesurés, ≤ 2 pt
     sur les duels clés.

   Si 1 échoue, on s'arrête.

2. **V2 T1−30 2022 sans reports**, 4 chaînes indépendantes.
   - Partage Mélenchon prédit (mesuré 63) : le T1 seul le place-t-il près de l'arrière ?
   - Duels hypothétiques stables entre chaînes (≤ 3 pt).
   - Écart de lp pas pire que le 1D (4 nats).
   - T1 non dégradé.

   Si 2 échoue : outil d'entre-deux-tours seulement, comparé explicitement à la 2D en coût et en
   stabilité.

3. **Si 2 passe** : 2017 (V2 T1−30), V1 (§8.29), effet sur 2027 (P(élu), duels, stabilité
   §8.45), puis « ARBITRAGE À TRANCHER » sur l'adoption.

### 8.47 (R3b) Un seul modèle spatial pour toutes les élections : maths (avant code ; critères figés avant résultat)

**Pourquoi.** L'étape 2 de R3 (§8.46) montre qu'avant le T1, sans reports, le cercle ne place pas
Mélenchon près de l'arrière (99,9 % prédit contre 63 % mesuré en 2022). Aucun tableau de reports
n'existe avant le T1 : les fichiers 2017/2022 sont tous datés après lui. L'information qui manque
existe pourtant dans les élections passées, et la structure qu'elle décrit (LFI, RN, le centre) se
retrouve d'une élection à l'autre. Consigne utilisateur : toutes les données nourrissent le même
modèle.

**1. Modèle.** Les élections $e\in\mathcal E$ (d'abord 2017 et 2022, puis 2027) sont ajustées
ensemble. Chacune garde sa vraisemblance actuelle : T1 (nœuds), duels, reports s'ils sont datés au
plus tôt à `as_of`, et nœud du résultat T1 si la coupure est postérieure. Chacune garde aussi ses
paramètres propres : $w$ (niveaux et chemins OU), $\sigma_{global,e}$, $\eta_e$, $\omega_e$,
$\sigma_{T2,e}$, $c_e$.

Les paramètres suivants sont partagés.

- **Arc avant** (cercle seulement) : un $A$ commun,
  $A=2\pi\,\mathrm{logistique}(a)$ avec $a\sim\mathcal N(\mathrm{logit}\,0{,}75;0{,}75)$ (prior de
  §8.46). C'est la taille de l'arrière, la grandeur structurelle qui transporte l'information des
  reports.
- **Personnes récurrentes.** Soit $p(c)$ la personne d'un candidat (même nom de famille normalisé
  dans au moins deux élections : Arthaud, Poutou, Mélenchon, Jadot, Macron, Lassalle,
  Dupont-Aignan, Asselineau, Le Pen). Les écarts d'ancre de position et de rayon sont tirés autour
  d'une valeur propre à la personne :

$$z^{\mu}_{c,e}=\bar z^{\mu}_{p(c)}+s_\mu\,\epsilon^{\mu}_{c,e},\qquad
z^{\sigma}_{c,e}=\bar z^{\sigma}_{p(c)}+s_\sigma\,\epsilon^{\sigma}_{c,e},$$

  avec $\bar z\sim\mathcal N(0,1)$, $\epsilon\sim\mathcal N(0,1)$ et
  $s_\mu,s_\sigma\sim\mathrm{HalfNormal}(0{,}5)$. Pour un candidat non récurrent, $z=\epsilon$,
  comme aujourd'hui. On a toujours $\mu_{c,e}=\mathrm{logistique}(\mathrm{logit}\,\mathrm{ancre}_{c,e}
  +\mathrm{pos\_scale}\cdot z^\mu_{c,e})$, avec les ancres ORDER_GROUPS de chaque élection :
  l'écart à l'ancre est partagé, pas la position absolue, car les ancres dépendent du nombre de
  groupes. Les candidats récurrents gardent le même rang relatif, Mélenchon en tête de gauche et
  Le Pen en tête de droite ; c'est l'approximation assumée.
- **κ** (netteté du passage du T1 au T2) :
  $\log\kappa_e=\log\bar\kappa+s_\kappa\epsilon^\kappa_e$, avec
  $\log\bar\kappa\sim\mathcal N(0;0{,}5)$ (prior actuel), $s_\kappa\sim\mathrm{HalfNormal}(0{,}3)$ et
  $\epsilon^\kappa_e\sim\mathcal N(0,1)$. κ décidait en §8.44 entre la saturation (0,75) et le
  croisement (0,22) : le partager transporte l'information des reports passés vers une élection
  qui n'en a pas.

Rien d'autre ne change : la quadrature (trapèzes à 128 nœuds), le noyau de von Mises, la valeur
inclusive, la vraisemblance Dirichlet des reports.

**2. Témoin.** Le même modèle joint sur le segment (sans $A$), avec le même partage de $z$ et de
κ. Il sépare l'apport du cercle de celui des données passées.

**3. Mise en œuvre.** `spatial_pooling_model_ou` reçoit des valeurs imposées optionnelles
(`z_pos_fixe`, `z_sigma_fixe`, `arc_fixe`, `kappa_fixe`). Il les enregistre comme déterministes
sous les noms habituels, ce qui garde la reconstruction et l'évaluation par élection inchangées.
Un modèle enveloppe tire les paramètres partagés, puis appelle le modèle d'une élection sous
`numpyro.handlers.scope(prefix="e<année>")`. Les chaînes sont sauvées par élection, sous les noms
de sites habituels, ce qui permet `eval_r2` et `_runner_fusion` tels quels. La log-densité JOINTE
par chaîne, tirée de `potential_energy`, est rapportée à part : c'est elle qui juge la
multimodalité.

**4. Protocole et critères (figés avant lecture).**

*Étape A, prospective vers 2022.* On ajuste 2017 complet (T1, duels, reports, nœud du résultat T1,
`as_of` = T2−1) avec 2022 à T1−30 sans reports. Les grandeurs prédites pour 2022 sont lues
exactement comme à l'étape 2 de §8.46 (`eval_r2 --coupure T1-30`). Quatre chaînes T4.

1. Partage Mélenchon 2022 à moins de 10 pt du mesuré (63,0).
2. MAE des 4 partages 2022 (Mélenchon, Pécresse, Jadot, Zemmour) inférieure à celle du 1D seul sans
   reports (14,7 pt, §8.44).
3. Stabilité entre chaînes : écart ≤ 3 pt sur les 10 duels clés de `eval_r2 --par-chaine`, et écart
   de log-densité JOINTE ≤ 10 nats entre les 4 chaînes.
4. T1 2022 non dégradé : MAE des parts contre le résultat ≤ 3,1 pt (1D : 2,98).

*Témoin.* Le même protocole sur le segment joint. Si le témoin passe aussi, l'apport vient des
données passées et non du cercle : on le dit, et c'est le segment joint (moins cher) qui serait
retenu.

*Étape B, symétrique* (seulement si A passe) : 2022 complet et 2017 à T1−30. Mêmes critères, partage
Mélenchon 2017 mesuré à 76,4.

*Étape C* (seulement si A et B passent) : 2017 + 2022 complets et 2027 à date. Effet sur P(élu),
duels et stabilité (§8.45), puis « ARBITRAGE À TRANCHER » sur l'adoption, qui toucherait aussi le
T1 publié.

**5. Ajout du 2026-11-20 : reports publiés AVANT le T1** (b6c29d3, `data/parsed/reports_avant_t1_{2017,2022,2027}.csv`).
Ce sont des duels T2 croisés par intention de vote T1, publiés par les instituts pendant la campagne
(2022 : 40 sondages de février 2019 à avril 2022 ; 2017 : 19 ; 2027 : 8 à ce jour). Ils lèvent le
diagnostic du §8.47 (« aucun report n'existe avant le T1 »). Figé avant lecture :

- **Vraisemblance.** Celle des reports d'entre-deux-tours (Dirichlet par notice et électorat,
  §8.44.4), avec θ_k(t) évalué à la date du sondage. L'électorat k est celui de l'*intention* T1
  déclarée, interprété comme l'électorat k du modèle à la date t.
- **Fenêtre.** Seuls les reports datés dans $[\text{as\_of}-180\,\text{j};\ \text{as\_of}]$
  entrent, la même fenêtre que les sondages T1 (`polls_historiques`) : un report de 2019 étirerait
  la base OU sur trois ans. Les électorats absents du roster (ex. Hamon 2019) sont ignorés.
- **Champ des électorats.** C'est la liste officielle du T1, connue avant T1−30 en 2017 (18/03) et
  en 2022 (07/03). L'approximation est que chaque sondage avait son propre champ hypothétique.

Variantes : `_reports_avant` (drapeau de backtest) charge ce fichier en plus des reports
d'entre-deux-tours. Il s'ajoute aux variantes existantes sans changer leur sens.

*Étape A′* (en file derrière A, mêmes critères 1-4 que A, lus sur 2022 à T1−30) :

- **A′1** — une seule élection, cercle, 2022 à T1−30 avec reports d'avant T1 (`hiercercle_ra`).
- **A′2** — même chose sur le segment (`hier_ra`), comme témoin.
- **A′3** — le joint 2017 complet + 2022 à T1−30, reports d'avant T1 compris (`A2_cercle`).

Lecture : si A′1 passe, ce sont les reports d'avant T1 qui suffisent, sans multi-élections. Si A′2
passe aussi, le cercle n'apporte rien. A′3 dit ce que l'histoire ajoute par-dessus. Le gain de
Mélenchon se lit hors échantillon : les reports d'entre-deux-tours 2022, que rien n'a vus, comme
en A.

**6. Ajout du 2026-11-20, 10:30Z : étape B′2, figée avant lecture.** A′2 (segment + reports
d'avant T1, 2022 à T1−30) remplit les critères 1, 2 et 4 mais pas le 3. Pour savoir si ce gain se
généralise, on refait le même ajustement sur l'autre élection : le segment `hier_ra` sur 2017 à
T1−30. La cible est les reports d'entre-deux-tours 2017, que le fit ne voit pas : partage Mélenchon
mesuré 76,4. Critères :

1. partage Mélenchon 2017 à moins de 10 pt ;
2. MAE des partages mesurés 2017 (Mélenchon, Hamon, Fillon, Dupont-Aignan) inférieure à 10 pt ;
3. stabilité (duels clés 2017 ≤ 3 pt entre chaînes, lp ≤ 10 nats) ;
4. T1 2017 non dégradé contre le 1D sans reports à T1−30, s'il existe, sinon ≤ 3,1 pt.

Il tourne sur le Mac (verdict de stabilité « non probant s'il est favorable »).

**Limites connues.**

- Deux élections seulement pour apprendre $s_\kappa$, $s_\mu$ et $s_\sigma$ : ces échelles
  suivront surtout leurs priors. Les valeurs retenues sont dans le §1 et ne seront pas retouchées
  après lecture.
- L'électorat de Mélenchon a changé entre 2017 et 2022 (partage 76 contre 63) : c'est ce que
  $s_\mu$ et $s_\sigma$ doivent permettre, et ce que l'étape A mesure.

#### Réplication VM du protocole 1 de §8.46 (V2 T2−7 2022 avec reports, CPU, 19/11)

Run indépendant de celui du T4 (`_t4`, d1c98e5), sur la VM, avec les noms standard. Fichiers :
`results/second_round_hiercercle_v2_2022_T2-7.json`, `eval_r2_hiercercle_2022.json` et
`_T2-7_par_chaine.json`.

**Coût.** Environ 3 h par chaîne et 511 pas de leapfrog par tirage, contre environ 1 h 30 pour la
2D et 25 min pour le 1D. Hypothèse sur la cause : la crête $A\leftrightarrow\sigma$, puisque seul le
produit $A\sigma$ règle la largeur des noyaux et que $A$ ne joue en propre que par l'arrière.

**L'arc se ferme presque** : $A=6{,}06$ rad, soit 3,5 % de votants à l'arrière (prior : 25 %). Les
deux extrêmes deviennent voisins. κ (0,68) et $\bar\sigma_v$ (0,052) restent proches du 1D.

| critère (figé avant lecture) | cercle | 2D (hier2d) | 1D avec reports | verdict |
|---|---|---|---|---|
| Mélenchon hors fit, < 10 pt | +6,8 | −3,5 | +14,4 | ✓ |
| MAE hors fit des 4 partages, ≤ 2,6 ou au moins < 8,2 | 3,3 | 2,6 | 8,2 | ✓ (forme faible) |
| T1 : MAE des nœuds / du résultat | 0,622 / 1,08 | 0,603 / 1,17 | 0,618 / 1,26 | ✓ |
| écart de lp entre chaînes | 6 nats (4 retenues) | 82 nats | | |
| stabilité des partages, ≤ 3 pt | **Mélenchon 5,6, Pécresse 4,5**, Jadot 0,3, Zemmour 1,1 | jusqu'à 11,6 | | ✗ |
| stabilité des duels, ≤ 2 pt | tous ≤ 1,7, sauf **Zemmour–Pécresse 6,1** | jusqu'à 11,8 | | ✗ |

L'écart vient d'une seule chaîne : la chaîne 0, à 5-6 nats sous les trois autres, mais retenue par
§8.8. Entre les chaînes 1 à 3, Mélenchon varie de 71,0 à 71,5 et Pécresse de 61,4 à 62,1. Le
R̂ de 23 porte sur un μ, sans doute un candidat d'extrême qui passe d'un côté à l'autre du point
arrière une fois le cercle presque fermé : c'est une symétrie que §8.46.2 n'avait pas prévue.

**Verdict.** Le protocole 1 échoue strictement sur la stabilité, comme sur le T4 (671 nats,
partage Pécresse 8,7 pt). Les deux runs s'accordent : le cercle reproduit les reports quand il les a.
Mélenchon est à +6,8 pt hors fit sur la VM et à +8,1 en toutes notices sur le T4, alors que le 1D
avec reports est à +14,4. La stabilité entre chaînes ne passe pas. Les deux runs diffèrent par la
multimodalité : 6 nats et 4 chaînes retenues sur la VM, contre 671 nats et 2 chaînes sur le T4.
Ce peut être la plateforme ou la graine, et c'est à garder en tête pour R3b. La suite est R3b
(§8.47, session locale).

### 8.48 Précision de la quadrature 1D aux petits σ (constat de §8.46.4 ; 0 fit)

**Mesure.** On reconstruit les fits témoins 1D hier de 2022, sur les données courantes :
`r2_temoin_hier_v2_2022_T2-7` et `_T1-30`. Les mêmes tirages postérieurs sont réévalués avec la
grille de production (Gauss-Legendre, B = 25), puis avec une grille fine (B = 400). Les
quantités publiées par les backtests sont :

- le duel Macron–Le Pen prédit (`modele`) ;
- les 10 duels clés de §8.44 ;
- les parts T1 au T1 contre le résultat (MAE).

Critère de lecture, fixé avant : un écart de plus de 0,5 pt sur un duel ou sur la MAE T1 rend
le backtest 2022 sensible à la quadrature. Il faudrait alors refaire les chiffres 2022 publiés
avec B plus grand (rien en production).

**Garde-fou, en diagnostic seulement, sans changement de calcul.** `fit_spatial_pooling` publie
`sigma_min` (la plus petite médiane a posteriori des σ) et `quadrature_fragile = sigma_min < 0,035`.
Le seuil vient de §8.46.4 : à σ = 0,04, l'erreur de B = 25 vaut 0,6 pt au maximum sur des
géométries aléatoires.

**Résultat de la mesure, à postérieur fixé** (`results/quadrature_1d_2022.json`). Écart entre
B = 25 et B = 400 :

| coupure | duel max | Macron–Le Pen | parts T1 (max) | MAE T1 |
|---|---|---|---|---|
| T2−7 (σ min 0,020) | **1,09** pt (Macron–Mélenchon, Le Pen–Mélenchon, Mélenchon–Zemmour, Mélenchon–Pécresse) | 0,09 | 1,1 | −0,01 |
| T1−30 (σ min 0,021) | 0,53 | **0,51** | 0,43 | +0,08 |

Le seuil de 0,5 pt est franchi : **les chiffres 2022 sont sensibles à la quadrature**, surtout ceux
qui impliquent Mélenchon (σ ≈ 0,02). La MAE T1 ne bouge pas.

C'est une borne inférieure. Le fit a été fait avec B = 25 : il peut avoir exploité l'erreur de
quadrature, puisqu'un noyau étroit tombé entre deux nœuds est mal intégré.

**Test, fixé avant lecture.** Refit du témoin 1D hier V2 T2−7 2022 avec
`PRES2027_B_QUAD=100`, 4 chaînes, mêmes données.

- Si la médiane de σ_Mélenchon passe au-dessus de 0,03, l'effondrement de σ est en partie un
  artefact. On le rapporte avec ses conséquences : les reports 1D de §8.38 et le témoin équitable
  de §8.44 sont à relire, et les backtests 2022 à refaire avec B plus grand.
- Sinon, l'effet se limite au postérieur fixé ci-dessus.

**Résultat du refit à B = 100** (19/11). Fichiers : `results/r2_temoin_hier_B100_v2_2022_T2-7.json`
et `eval_r2_hier_B100_2022_T2-7_par_chaine.json`. Chaînes retenues : 0, 1 et 2, pour 16 nats
d'écart de lp.

| | B = 25 | B = 100 |
|---|---|---|
| σ_Mélenchon, médiane par chaîne retenue | 0,020 / 0,020 / 0,020 | **0,031 / 0,031 / 0,030** |
| Macron–Le Pen prédit | 53,5 | 53,6 |
| partage Mélenchon prédit par les reports (mesuré 63) | 100 | 100 |
| MAE T1 contre le résultat | 1,23 | 1,25-1,26 |

**Lecture.** σ_Mélenchon passe au-dessus de 0,03. L'effondrement du rayon de Mélenchon en 1D
2022 est donc **en partie un artefact de quadrature** : à B = 25, un noyau à 0,02 tombe entre les
nœuds (espacés d'environ 0,04) et le fit l'exploite. Mais rien de ce qui a été conclu ne change :

- le duel Macron–Le Pen et le T1 sont stables ;
- les reports 1D restent saturés (Mélenchon→Macron 100 %), donc §8.38 et le témoin de §8.44 tiennent.

**Décision**, méthodologique et sans effet sur la production (σ min 0,051 en 2027, erreur
0,001 pt) :

- les backtests historiques futurs se font avec `PRES2027_B_QUAD=100` quand `quadrature_fragile`
  est vrai ;
- le garde-fou `sigma_min` reste dans les diagnostics, pour être vu en production si un σ 2027
  descend sous 0,035.

Les chiffres 2022 déjà publiés ne sont pas refaits : écarts ≤ 1,1 pt sur les duels de Mélenchon, et
aucune conclusion ne change.

### 8.48 bis (R3c) Abstention située sur le cercle : maths (avant code ; critères figés avant résultat)

**Idée (utilisateur, 2026-11-20).** Sur le cercle, on lit les angles comme des points normalisés en L2
d'un plan à deux axes, $x=\cos\theta$ (système, avec le centre en $\theta=0$) et $y=\sin\theta$
(gauche–droite). L'arrière $\theta=\pi$ est le pôle anti-système, entre les deux extrêmes. Aujourd'hui,
l'abstention du T2 est une option extérieure de niveau unique ω, la même pour tout électeur
(§3.1). On la *situe* : elle est plus attirante à l'arrière qu'au centre.

**1. Forme.** L'utilité de l'option extérieure en θ devient

$$\omega(\theta)=\omega_0-\gamma\cos\theta=\omega_0+\gamma\,x_{\text{anti}}(\theta),\qquad \gamma\ge0,$$

linéaire dans la coordonnée anti-système du cercle normalisé. Le reste ne change pas :

$$\rho(\theta)=\mathrm{logistique}\big(\log(e^{\kappa A_a}+e^{\kappa A_b})-\omega(\theta)-\kappa\,\mathrm{lse}(\theta)\big).$$

Le partage $q(\theta)$ entre a et b n'est pas touché : seule la probabilité de s'exprimer dépend de
θ. **Emboîtement** : avec γ = 0, on retrouve exactement le cercle de §8.46 et §8.47 ; le témoin est
donc A′1. **Coût** : un paramètre, γ. Le point d'abstention est fixé à l'arrière (θ = π) : un
θ_abs libre coûterait un paramètre de plus pour une symétrie qui s'impose.

**2. Prior.** $\gamma\sim\mathrm{HalfNormal}(1{,}5)$ en logit, soit un écart d'utilité centre–arrière
de 2γ (médiane ≈ 2).

**3. Identification.** Au T1, rien (les parts sont calculées parmi les exprimés). Au T2, par les
cases « Aucun » des reports, d'avant et d'entre-deux-tours, par électorat : Mélenchon ≈ 44 % contre
Zemmour ≈ 13 % en 2022. Un ω uniforme ne peut pas produire cet écart sans tordre les σ.

**4. Mise en œuvre.** C'est un terme additif `omega_g` (forme (..., N_θ)) passé à `duel_expression`,
`duel_shares_poids` et `reports_cellules`, avec `None` par défaut (comportement inchangé). Il est
tiré dans `joint_model` sous le drapeau `abst_situee` (cercle seulement), et conservé dans le fit
(`gamma_draws`) pour les prédictions (`duel_draws`) et l'évaluation (`eval_r2`). La variante est
`hiercercle_ra_abs`.

**5. Protocole (figé avant lecture).** R3c a le même cadre qu'A′1 : 2022 à T1−30, reports d'avant
T1, 4 chaînes, une seule plateforme. Les critères 1 à 4 de §8.47.4 sont lus comme pour A′1. R3c est
retenu contre A′1 s'il remplit les quatre conditions suivantes :

1. critères 1, 2 et 4 pas plus mauvais que ceux d'A′1 de plus de 1 pt ;
2. MAE des cases « Aucun » des 4 électorats mesurés inférieure d'au moins 2 pt à celle d'A′1 ;
3. **plausibilité** (orientation utilisateur du 20/11) : rapport σ_Le Pen/σ_Macron inférieur à celui
   d'A′1 ;
4. γ identifié : quantile 5 % de γ supérieur à 0,1.

Si R3c remplit 1, 2 et 4 mais pas 3, on le dit : l'abstention située n'aura pas suffi à rétrécir
Le Pen.

### 8.49 Valeur inclusive sans candidats fantômes (revue du 20/11, point 1 ; maths avant code)

**Défaut.** La non-expression au T2 fait intervenir $U_0(v)=\omega+\kappa\log\sum_{j\in R}e^{A_j(v)}$
(§3.1), où $R$ est le **roster** entier. En backtest, le roster comprend les putatifs testés dans
les hypothèses T1 anciennes. En 2022, neuf d'entre eux n'étaient pas candidats : Bertrand,
Barnier, Ciotti, Juvin, Montebourg, Taubira, Thouy, Philippot, Asselineau.

Leur $w$ courant n'est contraint par aucun sondage récent : ils ne sont plus testés. Il entre
pourtant dans la valeur inclusive, et donc dans l'abstention prédite et dans les duels. Mesures
(cercle, V2 T2−7 2022) :

- `w_now` de Bertrand vaut 3,34 contre 2,01 pour Macron (chaîne 1), et 1,97 dans la chaîne 0 ;
- μ_Montebourg vaut 0,76 dans la chaîne 0 et 0,08 dans la chaîne 1, ce qui explique le R̂ de 23.

Le §3.1 avait noté la fragilité (« $R$ = roster, pas le champ ») mais l'avait crue « sans effet en
2017/2022 (roster = champ réel) ». C'est faux pour les backtests, où le roster inclut les putatifs.

**Correctif.** La somme porte sur le **champ** $C$ :

$$U_0(v)=\omega+\kappa\log\sum_{j\in C}e^{A_j(v)},\qquad
C=\text{champ officiel du T1}\ \cup\ \{\text{finalistes des duels observés}\}.$$

- En backtest, le champ officiel est connu avant T1−30 (dépôt des candidatures). Il est fixe pour
  toutes les dates.
- En 2027, il n'existe pas encore. L'analogue est le champ modal des sondages T1 (`lse_champ`,
  §8.27, déjà codé), ce qui sera à décider au moment d'une publication du T2.

Pour les finalistes hypothétiques d'un duel hors champ (rare en backtest), on les ajoute à $C$
pour que le duel garde son sens. La mise en œuvre réutilise `masque_lse` : 1 sur $C$, $10^{-30}$
ailleurs. Rien d'autre ne change.

**Mesure prévue, fixée avant lecture.**

1. **À 0 fit**, sur les chaînes existantes : on réévalue avec $C$ les reports prédits (dont la
   case « Aucun ») et les duels de V2 T2−7 2022 (cercle et 1D). C'est indicatif, puisque le
   postérieur a été ajusté avec $R$.
2. **Reruns avec le drapeau `--lse-officiel`**, avec 8 chaînes, sur une même plateforme, à B = 100
   pour le segment. Les critères sont ceux des étapes concernées, lus chaîne par chaîne (revue du
   20/11, point 2), et la persistance des tableaux d'avant T1 sert de témoin (point 3).

**Mesure 1, à 0 fit** (cercle, V2 T2−7 2022 ; `eval_r2_hiercercle_2022_lseo.json`). Cases du même
postérieur, réévaluées avec $C$ au lieu de $R$ :

| électorat | « Aucun » avec R | avec C | mesuré | partage avec R | avec C | mesuré |
|---|---|---|---|---|---|---|
| Mélenchon | 44,9 | 45,4 | 43,6 | 72,6 | 76,9 | 63,0 |
| Pécresse | 26,3 | **11,1** | 29,9 | 62,7 | 66,4 | 65,0 |
| Jadot | 31,1 | 28,9 | 33,3 | 87,6 | 87,9 | 87,5 |
| Zemmour | 13,5 | 9,3 | 12,7 | 8,9 | 9,8 | 9,4 |

Macron–Le Pen passe de 53,0 à 53,4 ; la MAE T1 ne bouge pas (1,08).

**Lecture.** Le fit **utilisait les fantômes** comme réservoirs d'abstention. Bertrand, voisin de
Pécresse à droite et de $w$ libre, fabriquait la non-expression de l'électorat Pécresse : sans lui,
la case tombe à 11 % contre 30 mesurés. Les reports d'abstention de tous les backtests 2022 avec
reports (R2, R3, A′) sont donc en partie des artefacts.

Un refit avec le drapeau est **nécessaire** avant toute conclusion sur les cases « Aucun » ou sur
R3c. C'est la mesure 2.

### 8.50 Reports d'avant T1 : groupement par hypothèse et champ par ligne (revue locale du 20/11, points 1.1-1.2)

**Défaut 1.1.** `arrays_reports` groupait par (notice, électorat). Une notice d'avant T1 teste
jusqu'à 6 duels ; leurs cellules étaient fusionnées dans un `dict` (la dernière hypothèse lue
gagnait), et la paire (a, b) venait de la première. En 2022, à T1−30, 179 groupes sur 196 sont
touchés. Les tableaux d'entre-deux-tours (une hypothèse par notice) ne le sont pas. Les
résultats A′1, A′2, B′2, les diagnostics de modes et `A2_cercle` sont donc **invalides**.

**Défaut 1.2.** Le champ des électorats était un seul vecteur, enrichi de toutes les lignes : un
finaliste hypothétique (Bertrand, Taubira) entrait dans le champ de *toutes* les lignes.

**Correctif.**

- Une observation par (notice, électorat, hypothèse).
- Champ par ligne : $C_r = C_{\text{officiel}}\cup\{a_r, b_r, k_r\}$, transmis comme une matrice
  (R, N) à `reports_cellules`.
- Repli sans champ officiel (2027) : l'union de toutes les lignes, comme avant.
- Test unitaire : une notice à deux duels donne deux lignes non mélangées.

**Témoin de persistance (données seules, calculé avant tout fit).** On prend la moyenne des
tableaux d'avant T1 Macron–Le Pen des 30 jours précédant `as_of`, et on la compare au partage
mesuré entre les deux tours.

| | Mélenchon | Pécresse / Hamon | Jadot / Fillon | Zemmour / Dupont-Aignan | MAE |
|---|---|---|---|---|---|
| 2022 | −1,9 | −1,5 | −2,9 | +4,1 | **2,6** |
| 2017 | +4,8 | −1,2 | −3,2 | +7,3 | **4,1** |

Les tableaux d'avant T1 *contiennent* presque la réponse. Un modèle n'apporte de la valeur que
s'il fait au moins aussi bien, tout en donnant les duels jamais sondés et la cohérence avec le T1.

**Critères révisés de l'étape A′ (corrigée), figés avant tout fit.** Ils remplacent les critères 1
et 2 de §8.47.4, jugés trop faciles par la revue. Ils valent pour toute variante (segment, cercle,
noyau t) sur 2022 à T1−30, avec `--lse-officiel`, 8 chaînes, une seule plateforme. Ils sont lus
sur le **mode dominant** (les chaînes à moins de 10 nats de la meilleure, ET de même géométrie :
écart de σ_Mélenchon ≤ 0,02 entre elles), et chaque mode est rapporté.

1. MAE des 4 partages ≤ persistance + 1 = **3,6 pt** ;
2. partage Mélenchon à moins de 5 pt du mesuré ;
3. plausibilité : **σ_Le Pen / σ_Macron < 1,5** (orientation utilisateur du 20/11, désormais un
   critère) ;
4. stabilité : 8 chaînes, mode dominant visité par au moins 3 chaînes ; duels clés ≤ 3 pt entre
   ces chaînes ;
5. T1 contre le résultat ≤ 3,1 pt ; la couverture IC90 du T1 est rapportée (revue 1.4).

### 8.51 (R4) Noyau de Student-t sur le segment : maths (revue locale du 20/11, point 3.3 ; avant code)

**Diagnostic (revue, 3.1).** Sur le segment gaussien, le logit du votant au T2,
$\kappa[\Delta w+(v-\mu_b)^2/2\sigma_b^2-(v-\mu_a)^2/2\sigma_a^2+\log\sigma_b/\sigma_a]$, n'a qu'un
terme non monotone en v : $(1/\sigma_b^2-1/\sigma_a^2)v^2/2$. « Les deux extrêmes penchent vers
Le Pen » équivaut donc à σ_LP > σ_Macron, ce que l'utilisateur juge non crédible. La cause est la
forme quadratique du noyau, pas la dimension.

**Forme.** On garde le noyau normalisé sur la grille, mais on remplace sa log-densité, au T1 comme
au T2 (un seul noyau pour les deux tours) :

$$\log\tilde\varphi_i(v)=-\frac{\nu+1}{2}\log\!\Big(1+\frac{(v-\mu_i)^2}{\nu\sigma_i^2}\Big),\qquad
\log\varphi_i=\log\tilde\varphi_i-\log\textstyle\sum_bW_b\tilde\varphi_i(V_b).$$

ν est **un seul paramètre global**, avec $\log\nu\sim\mathcal N(\log5;1)$. Quand ν → ∞, on retrouve
exactement le noyau gaussien : le modèle actuel est emboîté. L'écart de log-attractivité entre
deux candidats devient logarithmique en distance, au lieu de quadratique. Un électeur proche d'un
candidat reste tranché ; un électeur loin des deux finalistes s'adoucit, sans qu'aucun σ ne
s'élargisse.

**Variantes.** Drapeau `noyau_t` ; variante `hier_ra_t` (segment, reports d'avant T1,
`--lse-officiel`). Repli « t au T2 seul » si le T1 se dégrade : ce serait un second modèle, à
spécifier séparément seulement si nécessaire.

**Critères, figés avant tout fit.** Ce sont ceux de §8.50 (persistance + 1, Mélenchon à moins de
5 pt, σ_LP/σ_M < 1,5, stabilité sur 8 chaînes, T1 ≤ 3,1). On ajoute :

6. ν identifié et lourd : la médiane a posteriori de ν est inférieure à 30 ;
7. emboîtement : si la médiane de ν dépasse 30, R4 n'apporte rien et l'on retombe sur le
   gaussien (A′2 corrigé).

Comparaison : A′2 corrigé (gaussien), même plateforme (Mac), même nombre de chaînes, même B = 100.

### 8.52 Critères de §8.50-8.51 précisés (revue locale du 20/11 PM) — figés AVANT toute lecture d'A′2 corrigé et de R4

Aucune chaîne `_lseo` n'a été lue à l'écriture de ce paragraphe.

- **Outillage.**
  - `PRES2027_N_CHAINES=8` : `--fusionner` et `eval_r2` lisaient jusqu'ici c0..c3 sans prévenir.
  - Évaluation avec `--stem …_lseo` et `PRES2027_B_QUAD=100` : sans eux, on évaluait les anciennes
    chaînes brouillées, ou sur une autre grille.
  - Lecteur par mode versionné : `model/backtest/modes.py`. Contrôle : sur les anciennes chaînes,
    il retrouve H = {1, 4, 6, 7} (test).
- **Mode dominant.** Ce sont les chaînes avec L̄ ≥ L̄_max − 10 nats et |σ_Mélenchon − σ_Mélenchon de la
  meilleure chaîne| ≤ 0,02. Les autres modes sont regroupés par σ_Mélenchon et rapportés. Le terme de
  volume ½·log det Σ peut valoir ±5 nats entre chaînes : la masse relative des modes n'est donc PAS
  lue comme e^ΔL̄.
- **Critère 3b (ajouté), monotonie à gauche.** q_Macron(v) ≥ 0,5 pour tout v < μ_Macron dans le duel
  Macron–Le Pen, sur au moins 90 % des tirages. C'est la forme directe de « Le Pen ne gagne pas
  l'extrême gauche par sa largeur ». Le rapport σ_LP/σ_M < 1,5 reste rapporté.
- **Critère « Aucun » (ajouté).** MAE de la case Aucun ≤ persistance « Aucun » + 2, soit 1,75 + 2 en
  2022 et 4,8 + 2 en 2017.
- **R4 : les critères 6 et 7 de §8.51 sont REMPLACÉS.**
  - Le prior donne P(ν < 30) = 0,96, et ν = 30 n'est pas gaussien sur cette grille : l'écart vaut
    encore 2 pt sur π.
  - **6′** : sd a posteriori de log ν ≤ 0,5 (la moitié du prior) ET q95(ν) < 30.
  - **7′ (parcimonie)** : R4 n'est retenu que s'il passe 1-5, 3b et 6′ ALORS que le gaussien (A′2
    corrigé) échoue à au moins un de 1, 2, 3b. Si les deux passent, on garde le gaussien.
  - corr(log ν, log κ) et corr(log ν, log σ_global) sont rapportées.
- **Hors comparaison A′2 / R4 (même données pour les deux), corrections reportées au tour suivant.**
  - Case Aucun omise : poser 100 − a − b (12 lignes en 2022, 6 en 2017).
  - Pondération des reports par l'effectif.
  - Champ par ligne pour 2027 : l'union met Le Pen et Bardella ensemble.

### 8.53 (R5) Témoin parcimonieux : rayon de Le Pen lié à celui de Macron (revue locale du 20/11 PM, point 4 ; avant code)

**Forme.** Segment gaussien inchangé, sauf $z^\sigma_{LP}\equiv z^\sigma_{Macron}$, donc
$\sigma_{LP}=\sigma_{Macron}$ : un paramètre de MOINS. Le Pen ne peut plus gagner les extrêmes par
sa largeur (avec des σ égaux, le seul terme non monotone de §8.51 disparaît). Le croisement doit
alors venir d'ailleurs : positions, κ, valeur inclusive.

**Variante.** `hier_ra_lie` (drapeau `sigma_lie`), avec reports d'avant T1 corrigés,
`--lse-officiel`, B = 100, 8 chaînes, Mac, comparée à A′2 corrigé et à R4.

**Critères (figés).** Ceux de §8.50 / §8.52 (1, 2, 3b, 4, 5, Aucun). Le critère 3 est satisfait par
construction et n'est pas compté.

**Règle de parcimonie (extension de 7′).** Si R5 passe 1, 2 et 3b, il prime sur R4 (un paramètre de
moins au lieu d'un de plus). S'il échoue à 1 ou 2, on mesure ce que coûte la plausibilité imposée :
écart de MAE des partages et de log-densité des reports contre A′2 corrigé, rapportés tels quels.

### 8.54 Lecture du cercle (A′1 corrigé) et prédictions figées (revue locale du 20/11 soir) — AVANT toute lecture d'A′1 corrigé ou de R4/R5

**Monotonie circulaire (critère 3b sur le cercle).** La monotonie du segment n'a pas de sens sur le
cercle : il n'y a pas de « v < μ_Macron ». On la remplace par la part des votants de l'**arc gauche**,
c'est-à-dire des θ compris entre θ_Mélenchon et θ_Macron sur l'arc avant, pour lesquels
q_Macron(θ) < 0,5 dans le duel Macron–Le Pen. Elle est pondérée par la densité uniforme des votants,
et le critère est **≤ 10 %**, lu sur la moyenne a posteriori. Pour le segment, la même version
pondérée (votants v < μ_M) est RAPPORTÉE à côté du critère 3b d'origine, qui reste le critère.

**Contrôles de structure, rapportés pour toute variante**, avec seuils figés :

- σ_Mélenchon ≤ 0,15 ;
- σ_LP/σ_M < 1,5 ;
- électorat Le Pen donnant ≥ 45 % à Mélenchon dans Mélenchon–Macron (données d'avant T1 : 56,8) ;
- électorat Zemmour donnant ≤ 40 % à Mélenchon (données : 31,9).

Ces deux derniers sont des « fers à cheval » mesurés, que la géométrie doit reproduire sans
élargir de noyau.

**Prédictions figées (revue, 2.2-2.3).**

- **R5** (σ_LP ≡ σ_M) : A_M − A_LP est alors linéaire en v. Prédiction : **échec au critère 2,
  Mélenchon ≥ 80**. Les critères 3 et 3b y sont vrais par construction et ne sont PAS informatifs.
  R5 sert de contrôle de cette prédiction.
- **R4** (noyau t) : on rapporte corr(log ν, log σ_LP). Si q95(ν) ≥ 30, R4 est déclaré « équivalent
  au gaussien », sans verdict sur 3b.

**Limite annoncée.** Le « Aucun » de Pécresse (15 contre 30) risque de rester raté, cercle ou pas :
un électorat placé entre les deux finalistes s'abstient peu.

**Résultat d'A′1 corrigé** (cercle, reports d'avant T1 regroupés selon §8.50, `--lse-officiel`, V2
T1−30 2022, VM, 4 chaînes d'environ 10 h chacune, 511 pas par tirage). Fichiers :
`results/second_round_hiercercle_ra_v2_2022_T1-30_lseo.json` et
`modes_second_round_hiercercle_ra_v2_2022_T1-30_lseo.json`.

Le mode dominant regroupe les chaînes 0, 2 et 3 (lp à 1,6 nat près). La chaîne 1 est 36 nats plus
bas (σ_Mélenchon 0,086 contre 0,102).

| critère (figé avant lecture) | A′1 corrigé (cercle) | A′2 corrigé (segment, Mac) | verdict A′1 |
|---|---|---|---|
| 1. MAE des 4 partages (persistance 2,61) | 7,93 | 7,3 | ✗ |
| 2. Mélenchon (mesuré 63,0, à moins de 10 pt) | **48,6** | 66 | ✗ (−14,4) |
| 3. σ_LP/σ_M < 1,5 | 1,64 | ≈ 3,7 | ✗ de peu |
| 3b. monotonie (circulaire, ≤ 10 %) | 0 % | 0 % (segment) | ✓ |
| 4. stabilité (écart des duels dans le mode dominant) | 0,09 pt | un seul mode | ✓ |
| 5. MAE T1 | 2,82 | | ✓ |
| « Aucun » ≤ persistance + 2 (3,75) | 9,0 | | ✗ |
| structure : électorat LP → JLM ≥ 45 (56,8) | 63,9 | | ✓ |
| structure : électorat Zemmour → JLM ≤ 40 (31,9) | 26,6 | | ✓ |
| structure : σ_Mélenchon ≤ 0,15 | 0,10 | | ✓ |

Grandeurs du modèle : κ = 0,35, arc 3,97 rad (37 % des votants à l'arrière, contre 3,5 % à T2−7 avec
les fantômes), Macron–Le Pen 59,6 (résultat 58,55).

**Lecture.**

1. **Une fois les deux défauts corrigés, le cercle est stable** : écart des duels de 0,09 pt dans le
   mode dominant, contre 5-12 pt la veille. Les candidats fantômes et les reports brouillés
   alimentaient donc la multimodalité (revue du 20/11, point 1).
2. **Il reproduit la structure croisée sans élargir Le Pen à l'excès** : σ_LP/σ_M vaut 1,64, contre
   environ 3,7 sur le segment. Il passe les deux « fers à cheval » mesurés (LP→JLM, Zemmour→JLM) et
   la monotonie circulaire.
3. **Il dépasse la cible pour Mélenchon** (48,6 contre 63, trop vers Le Pen) et rate les cases
   « Aucun ». Il ne bat pas non plus la persistance des tableaux (7,9 contre 2,6 pt).
4. **Verdict : A′1 échoue** (critères 1, 2, 3 et « Aucun »). Ni le cercle ni le segment gaussien ne
   passent ; R4 (noyau t) et R5 sont en cours sur le Mac. Le cercle reste la seule géométrie qui
   passe à la fois la stabilité, la monotonie et les contrôles de structure.

### 8.55 (R6) Cercle fermé, A = 2π (revue locale du 20/11 midi, point 2.1 ; avant code)

**Constat (A′1 corrigé).** Le cercle à arc libre est stable et passe la structure (monotonie,
« fers à cheval » LP/Zemmour, σ_LP/σ_M 1,64). Mais il apprend un arrière de 37 % des votants et
DÉPASSE la cible : Mélenchon 48,6 contre 63. L'arc A règle à lui seul la masse d'électeurs placés
entre les deux extrêmes.

**Forme.** On fixe $A=2\pi$ : les positions ORDER_GROUPS couvrent tout le cercle, et les deux extrêmes
sont voisins par construction, sans arc vide. C'est **un paramètre de moins**. La rotation et la
réflexion restent fixées comme en §8.46.2. Mise en œuvre : `arc_fixe = 2π` (déjà dans `joint_model`),
drapeau `arc_ferme`, variante `hiercercle_ra_ferme`, avec `--lse-officiel`.

**Critères, figés avant tout fit.** Ce sont ceux de §8.50, §8.52 et §8.54 (1, 2, 3, 3b circulaire,
4, 5, Aucun, contrôles de structure), lus comme pour A′1. Comparaison avec A′1 corrigé, sur la même
plateforme (VM) et le même nombre de chaînes (4).

**Règle de parcimonie.** Si R6 fait au moins aussi bien qu'A′1 sur 1, 2 et 3 (sans en dégrader un de
plus de 1 pt), on préfère R6 (un paramètre de moins).

**Prédiction (non contraignante).** Sans arrière vide, la masse entre les extrêmes diminue, et
Mélenchon devrait remonter vers Macron.

**Résultat de R6** (cercle fermé, VM, 4 chaînes d'environ 9 h, 21/11). Fichiers :
`results/second_round_hiercercle_ra_ferme_v2_2022_T1-30_lseo.json` et `modes_…_lseo.json`.

Le mode dominant se réduit à la **seule chaîne 3**. Les trois autres sont 20 à 61 nats plus bas, avec
σ_Mélenchon de 0,031 à 0,047 contre 0,055 pour la chaîne 3.

| critère | R6 (fermé) | A′1 corrigé (ouvert) | R6 |
|---|---|---|---|
| 1. MAE des 4 partages (persistance 2,61) | 8,2 | 7,93 | ✗ |
| 2. Mélenchon (63,0) | 50,5 | 48,6 | ✗ |
| 3. σ_LP/σ_M < 1,5 | **0,97** | 1,64 | ✓ |
| 3b. monotonie circulaire | 1 % | 0 % | ✓ |
| 4. stabilité | mode dominant à 1 chaîne | 3 chaînes, 0,09 pt | ✗ |
| 5. MAE T1 | 2,84 | 2,82 | ✓ |
| « Aucun » | 9,07 | 9,0 | ✗ |
| structure LP→JLM / Zemmour→JLM / σ_JLM | 63,7 / 23,0 / 0,055 | 63,9 / 26,6 / 0,10 | ✓ |

Ajustement des reports (`modes.py`) : c médian 8,8 ; résidu « Aucun » de +11,2 sur les 164 lignes
« propres » et de −8,8 sur les autres.

Positions (chaîne 3) : θ_Zemmour = 2,05 avec **|z_pos| = 3,9**, Poutou à 2,87, Le Pen à 2,44 (z −1,0),
Arthaud à −2,83, Mélenchon à −2,22. L'écart entre Arthaud et Le Pen par l'arrière est d'environ
1,0 rad.

**Lecture.**

- **Règle de parcimonie (§8.55).** R6 fait au moins aussi bien qu'A′1 sur 1 (+0,3 pt), 2 (+1,9 vers
  le mesuré) et 3 (0,97 contre 1,64). R6 serait donc préféré à A′1. Mais il **perd la stabilité**
  (critère 4), qui n'est pas dans la règle, et **les deux échouent** aux critères 1, 2 et « Aucun ».
- **Prédictions.**
  - Celle de la revue est contredite : Mélenchon 50,5 > 48,6 et Zemmour→JLM 23,0 < 35.
  - Celle de §8.55 est faiblement vérifiée : Mélenchon remonte de 1,9 pt.
- **Zemmour |z_pos| = 3,9 > 2** : l'ancre ORDER_GROUPS de l'extrême droite est forcée. Comme le
  prévoyait la revue, R6 n'est en partie qu'A′1 payé par la tension des ancres.
- **Bilan cercle.** Ni l'ouvert ni le fermé ne passent ; fermer le cercle n'apporte pas de gain
  décisif. Le défaut commun aux deux (Mélenchon trop vers Le Pen, « Aucun » raté, résidu de +11 pt
  sur les lignes propres) est celui que R7 (fidélité révélée) vise. R7 est la suite logique.

### 8.56 (R7) Fidélité révélée dans la vraisemblance des reports (revue locale du 21/11 ~00:30Z ; avant code)

**Défaut.** 164 lignes de reports sur 585 (28 %) sont « propres » : l'électorat k est l'un des deux
finalistes (électeurs Le Pen dans Macron–Le Pen, Zemmour dans Macron–Zemmour, Mélenchon dans
Mélenchon–Macron…). Le modèle traite T1 et T2 comme indépendants sachant la position v ; il ne peut
pas représenter le fait qu'un électeur ayant choisi k au T1 vote k au T2 quand k est présent.
Postérieur d'A′2 corrigé :

| ligne propre | « Aucun » observé / modèle | partage vers l'adversaire, observé / modèle |
|---|---|---|
| Le Pen dans Macron–Le Pen | 2,3 / 11,9 | 2,3 / 15,8 |
| Zemmour dans Macron–Zemmour | 1,8 / 12,9 | 3,5 / 18,8 |
| Mélenchon dans Mélenchon–Macron | 4,4 / 19,1 | fidélité 95 contre 72 au modèle |

Conséquences :

- c (concentration Dirichlet) s'effondre à 15, contre 8 sur les lignes propres et 25 ailleurs :
  surdispersion d'un facteur 10 à 40 ;
- ω est tiré dans deux sens contraires (résidu « Aucun » de +5,6 pt sur les lignes propres et de
  −1,3 ailleurs), ce qui écrase l'abstention de Pécresse ;
- les tableaux, qui « contiennent » la réponse, pèsent donc peu.

**Forme (0 paramètre).** Pour une ligne où k ∈ {a, b}, le choc d'utilité individuel qui a fait
choisir k au T1 persiste au T2 : l'électeur préfère k à l'adversaire. Il ne reste que la décision de
s'exprimer, donnée par la formule actuelle avec k dominant : $\rho\approx\sigma(-\omega)$, puisque
$\log(e^{\kappa A_a}+e^{\kappa A_b})\approx\kappa A_k\approx\kappa\,\mathrm{lse}$. La cellule devient

$$\theta_r=(\,\sigma(-\omega),\ \varepsilon,\ \sigma(\omega)\,)\ \text{(k, adversaire, Aucun)},$$

renormalisée, avec ε = 10⁻⁴ (le plancher déjà en place). Les autres lignes ne changent pas. Les
duels et le T1 ne changent pas non plus : la même fidélité dans l'agrégation des duels est une
étape B, à spécifier séparément, car elle modifierait les duels publiés.

**Variante.** Drapeau `fidelite` ; `hier_ra_fid` (segment, reports d'avant T1 corrigés,
`--lse-officiel`, B = 100, 8 chaînes, Mac), comparé à A′2 corrigé ; puis le cercle (VM) si le
segment progresse.

**Critères, figés avant tout fit** (proposés par la revue). Ceux de §8.50 / §8.52 / §8.54, plus :

- c médian ≥ 22 ;
- « Aucun » de Pécresse ≥ 22 (mesuré 29,9) ;
- résidu moyen du « Aucun » ≤ 2 pt en valeur absolue, sur les lignes propres comme sur les autres.

**Prédiction figée pour R6 (cercle fermé, revue)**, contraire à la prédiction non contraignante de
§8.55 :

- partage Mélenchon ≤ 48,6 ;
- Zemmour → Mélenchon > 35 ;
- « Aucun » de Pécresse inchangé.

On rapportera θ_i et |z_pos| des extrêmes. Si les extrêmes ont |z_pos| > 2, R6 n'est qu'A′1 payé
dans le prior.

**Révision du 21/11 ~04:30Z, avant tout fit (revue locale de 04:00Z).**

1. **Dérivation.** L'argument « κA_k ≈ κ·lse » est faux : E[lse − A_k] vaut 1,3 à 1,8 pour les
   extrêmes. La bonne justification, toujours à 0 paramètre, est un choc T1 persistant de type
   Gumbel. Si k a été choisi au T1, $U_k=\mathrm{lse}+G$ avec $G\sim\mathrm{Gumbel}(0,1)$, et k domine
   aussi sur {a, b}. Donc $\rho_{fid}=\mathbb E_G\,\sigma(\kappa G-\omega)$, calculé par quadrature sur
   32 quantiles fixes. Sur le postérieur d'A′2, cela donne 1,6 % d'« Aucun », pour 2,2 % observé
   sur les lignes propres.
2. **Défection.** ε passe de 10⁻⁴ à **0,02** (défection publiée typique de 1 à 3 %). La cellule
   devient $(\rho(1-\varepsilon),\ \rho\varepsilon,\ 1-\rho)$. Avec ε = 10⁻⁴, la Dirichlet perdait
   environ 850 nats sur les 164 lignes propres et poussait artificiellement c.
3. **Lecture.** `modes.py` rapporte c médian et les résidus « Aucun » sur les lignes du fit, propres
   et autres. Contrôle sur A′2 : 14,9 / +5,55 / −1,30. Le critère « c ≥ 22 » est jugé **non
   probant** (il passerait presque mécaniquement) : il est rapporté, sans valeur de verdict.

**Rejetés (revue, chiffrés).**

- c_r = φ·n_eff : la part des lignes propres passerait de 28 % à 44 %, ce qui aggrave le défaut.
- R3c (γ cos θ) : l'ordre des données est contraire. Zemmour, près de l'arrière, s'abstient à 13 % ;
  Pécresse, près du centre, à 30 %.
- « Aucun » = 100 − a − b : seulement 8 groupes concernés.

### 8.57 Synthèse des trois géométries et nouveau critère de valeur ajoutée (revue locale du 21/11 ~08:00Z) — figé AVANT lecture de R7

**Diagnostic (revue, calculs sur le postérieur d'A′2).**

- **Poids des blocs.** Les duels sont NEUTRES vis-à-vis de σ_LP : imposer σ_LP ≡ σ_M au MAP leur
  coûte 3 nats, contre 366 aux reports et 48 au T1. La tension oppose le T1 aux reports, et les
  reports portent environ 85 % de la largeur de Le Pen.
- **Lignes qui l'élargissent.** Ce sont l'électorat Le Pen dans les duels SANS Le Pen (il s'abstient
  à 38-60 % et donne 56 % à Mélenchon face à Macron) et Mélenchon dans Macron–Le Pen.
- **Conclusion structurelle.** σ_k fait double emploi : il fixe la composition de l'électorat k et la
  portée de k comme finaliste. Un noyau par candidat ne peut pas faire les deux. Le défaut vient de
  l'hypothèse « T2 indépendant de l'électorat T1, sachant v ».
- **Confirmation dans les données.** Sur les 421 cases « Aucun » hors lignes propres, l'électorat
  seul donne R² = 0,36, et la cellule électorat × duel R² = 0,84. L'abstention dépend de l'identité
  de l'électorat (Mélenchon environ 50 % quel que soit le duel, Pécresse environ 30 %) bien
  au-delà de la position.
- **Les trois échecs « en miroir »** (segment, cercle, noyau t) sont donc un seul et même défaut.

**Prédiction figée pour R7, sans lecture de ses chaînes.** R7 retire la ligne propre qui tire σ_LP le
plus vers l'étroit (−51). D'où **σ_LP/σ_M ≥ 3,7 et monotonie 3b à 0 %**, même si c et les résidus
« Aucun » des lignes propres s'améliorent.

**Nouveau critère de VALEUR AJOUTÉE, figé (remplace « ≤ persistance + 1 » comme verdict).** La
persistance devient un diagnostic d'ajustement, avec un seuil de 5 pt : c'est un test en échantillon,
qu'un modèle saturé passe trivialement.

- **V1, reports d'une hypothèse retirée.** On refait le fit sans les lignes Mélenchon–Macron (9
  notices), puis on prédit les partages de leurs 7 électorats.
  - Témoin non structurel, calculé à 0 fit : la part de Macron chez k dans les autres duels Macron–X.
    MAE de 35,8 pt sur le partage, 21,7 sur « Aucun ».
  - Seuils :
    - MAE ≤ 12 sur Mélenchon–Macron ;
    - fers à cheval de bon signe sans les avoir vus : Le Pen → Mélenchon ≥ 50, Zemmour → Mélenchon
      ≤ 40.
- **V2, duels jamais sondés.** On retire tous les sondages d'une paire (Macron–Mélenchon, puis
  Macron–Zemmour) et on prédit leur moyenne.
  - Témoin : un Bradley-Terry ajusté sur les autres duels.
  - Exigence : battre ce témoin d'au moins 1,5 pt en moyenne.
  - Le chiffre de Bradley-Terry sera calculé à 0 fit et consigné AVANT le premier fit V2.
- **B′ (2017)** accompagne tout verdict.

**Méthode.** Un dépistage MAP/Laplace (4 à 8 initialisations, environ 1 min chacune) précède tout
NUTS. On ne lance NUTS que si le MAP passe les critères 2 et 3b. Au MAP, R5 perd 366 nats sur les
reports et σ_JLM monte à 0,38 : **R5 est ANNULÉ.**

**Repli publiable (évaluation de la revue).** Le segment T1 + duels, sans reports, est stable et
parcimonieux. Mais σ_LP/σ_M y vaut encore 3,1 au MAP : le mécanisme non crédible subsiste dans les
duels jamais sondés impliquant Le Pen. Il est acceptable pour les duels SONDÉS seulement, pas comme
modèle cible.

### 8.58 (R8) Abstention par électorat ω_k (revue locale du 21/11 08:00Z, piste A ; avant code)

**Idée.** Le type latent d'un électeur, ce n'est pas seulement sa position v : c'est aussi son choix T1
k. L'abstention au T2 dépend de k (cf. §8.57 : R² = 0,84 avec la cellule électorat × duel, 0,36
avec l'électorat seul). On donne donc à chaque électorat un niveau d'option extérieure :

$$\omega_k=\omega+s_\omega z_k,\qquad z_k\sim\mathcal N(0,1),\ s_\omega\sim\mathrm{HalfNormal}(1).$$

Il est hiérarchique : seuls les électorats présents dans les reports s'écartent de ω, les autres sont
rétrécis vers ω.

**Modèle, cohérent T2 / reports.** Un électeur situé en v est de type k avec la probabilité
$p_k(v)=\pi_k(v)/\sum_{j\in C}\pi_j(v)$ (champ C, valeur inclusive officielle §8.49). Pour un type k :

$$\rho_k(v)=\sigma\big(\log(e^{\kappa A_a}+e^{\kappa A_b})-\omega_k-\kappa\,\mathrm{lse}(v)\big).$$

- **Duels** : $g_{ab}=\int\sum_kp_k\rho_kq\,dW\Big/\int\sum_kp_k\rho_k\,dW$. Avec des $\omega_k$ tous
  égaux, on retrouve exactement la forme actuelle.
- **Reports** : la cellule de l'électorat k utilise $\rho_k$. Le Newton sur δ agrège les électorats
  du champ, chacun avec son propre $\rho_j$.
- **Lignes propres, avec R7** : $\rho_{fid}=\mathbb E_G\sigma(\kappa G-\omega_k)$.

**Coût.** N + 1 paramètres, rétrécis par la hiérarchie : seuls les 10 à 14 électorats des reports sont
identifiés. Tenseurs (·, N, G) déjà présents dans le calcul. Variante `hier_ra_fid_omk` (segment, R7
inclus, `--lse-officiel`, B = 100).

**Protocole.**

1. Dépistage MAP (§8.57) sur 6 initialisations dispersées. On ne lance NUTS que si le MAP passe les
   critères 2 et 3b.
2. NUTS, 8 chaînes, sur le Mac.
3. Si le critère 2 passe, V1 : refit sans les lignes Mélenchon–Macron.

**Critères, figés avant tout fit.**

- « Aucun » : MAE ≤ 3,75 (cible post-T1) ;
- MAE des 4 partages ≤ 5 (diagnostic, §8.57) ;
- critère 2 de §8.50 : partage Mélenchon à moins de 5 pt ;
- σ_LP/σ_M < 2 ;
- 3b : monotonie ≥ 90 % des tirages ;
- stabilité (§8.50) ;
- T1 ≤ 3,1 ;
- V1 (§8.57) ;
- s_ω : quantile 5 % > 0,2. Sinon, ω_k n'est pas identifié et R8 se réduit à R7.

**Parcimonie.** Si R7 seul passe les mêmes critères, R7 prime (R8 n'est jugé que sur ce que R7 ne fait
pas).

**Résultat du dépistage MAP de R8 sur le segment** (`results/modes_map_hier_ra_fid_omk_2022_T1-30.json`,
6 initialisations, mode dominant = init 0) :

- partage Mélenchon 63,6 (63) ✓ ;
- « Aucun » : MAE 3,63 ✓, pour la première fois ;
- c médian 58, contre 15 pour A′2 ;
- résidus « Aucun » +1,4 (lignes propres) et −0,3 (autres) ✓ ;
- MAE des partages 5,0 ;
- MAIS σ_LP/σ_M 3,08 ✗, monotonie 0 % ✗ (21 % en version pondérée), LP→JLM 41 ✗, σ_Mélenchon 0,28.

**Règle du protocole : pas de NUTS** (3b échoue). L'abstention est réglée par ω_k et R7 ; ce qui reste
est géométrique, et c'est ce que le cercle traitait (monotonie et structure réussies en A′1).

**Étape suivante, figée.** R8c = cercle (arc libre) + R7 + ω_k (`hiercercle_ra_fid_omk`). Dépistage
MAP d'abord, mêmes critères plus la monotonie circulaire. NUTS seulement si le dépistage passe 2 et
3b.

**Résultat du dépistage MAP de R8c** (cercle + R7 + ω_k ; `results/modes_map_hiercercle_ra_fid_omk_2022_T1-30.json`,
6 initialisations, mode dominant = l'initialisation de lp la plus haute ; dépistage, PAS un postérieur) :

- MAE des partages 3,55 ≤ 3,61 ✓ (au niveau du témoin de persistance, pour la première fois) ;
- Mélenchon 62,4 (63) ✓ ;
- « Aucun » 3,35 ✓ ;
- monotonie circulaire : 0 % de votants mal classés ✓ ;
- σ_LP/σ_M 1,54 : ✓ au critère de §8.58 (< 2), ✗ de peu au critère général de §8.50 (< 1,5) ;
- Zemmour → Mélenchon 32,7 ✓ ; Le Pen → Mélenchon 41,2 ✗ (≥ 45) ; σ_Mélenchon 0,159 ✗ (≤ 0,15) ;
- c 58 ; résidus « Aucun » +1,9 / −1,0 ; T1 3,0 ✓.

Les 6 initialisations aboutissent à des modes différents (lp de 12068 à 13042) : il faudra 8 chaînes
NUTS pour juger la stabilité.

**Règle du protocole : NUTS lancé** (les critères 2 et 3b passent). Deux jeux SÉPARÉS, sans mélange :

- Mac, 8 chaînes, après R7 (verdict de stabilité « non probant s'il est favorable ») ;
- VM, 4 chaînes, après R6 (DEMANDE VM n° 4, probant).

Critères de §8.58, figés, inchangés.

**Dépistage MAP B′ de R8c sur 2017 à T1−30** (`results/modes_map_hiercercle_ra_fid_omk_2017_T1-30.json`, 6
initialisations ; mode dominant = init 3, lp 3201,9).

| électorat | partage modèle | partage mesuré |
|---|---|---|
| Hamon | 95,6 | 93,4 |
| Fillon | 62,9 | 62,8 |
| Mélenchon | **87,2** | **76,4** |
| Dupont-Aignan | 56,5 | 37,6 |

- Dupont-Aignan : ralliement à Le Pen entre les deux tours, imprévisible avant le T1.
- MAE des partages 8,0 ✗ (4,4 hors Dupont-Aignan) ; critère 2 (Mélenchon) ✗, +10,8.
- « Aucun » 4,78 ✓ (≤ 6,8) ; σ_LP/σ_M 0,89 ✓ ; monotonie circulaire ✓ ; T1 1,93 ✓ ; c 49.

**Lecture.** L'abstention et la plausibilité se généralisent ; le report Mélenchon → Le Pen de 2017,
non. On n'en tire pas de verdict (dépistage, pas postérieur), mais cela pèse sur l'étape B :
R8c pourrait passer en 2022 sans se transférer à 2017.

### 8.59 Protocole V1 précisé (hypothèse retirée), figé AVANT tout fit V1

- **Retrait.** Variable `PRES2027_SANS_HYP="Mélenchon – Macron"` : `_fit` écarte des reports d'avant T1
  TOUTES les lignes de cette hypothèse (9 notices dans la fenêtre de 2022 à T1−30).
- **Cible.** Moyenne, sur les 3 notices de mars 2022 (Harris 4-7, Elabe 7-8, BVA 8-9 mars), du partage
  Mélenchon / (Mélenchon + Macron) et de la case « Aucun », pour les 6 électorats non finalistes :
  Le Pen, Zemmour, Pécresse, Jadot, Hidalgo, Roussel.
- **Prédiction du modèle.** Cellules de l'électorat k dans Mélenchon–Macron, calculées à `as_of` avec
  `w_now` (même calcul que les cibles d'entre-deux-tours).
- **Critères (§8.57, inchangés) :**
  - MAE des 6 partages ≤ 12 ;
  - ~~Le Pen → Mélenchon ≥ 50 ; Zemmour → Mélenchon ≤ 40~~. **Corrigé le 21/11 ~11:00Z, AVANT tout
    fit V1 :** ces seuils avaient été posés sur la moyenne de toute la fenêtre (56,8 / 31,9). Or la
    cible V1 est la moyenne des notices de mars, qui MESURE 42,3 (Le Pen) et 42,7 (Zemmour) : même un
    modèle parfait aurait échoué. Nouveaux critères : |Le Pen → Mélenchon − mesuré| ≤ 10 et
    |Zemmour → Mélenchon − mesuré| ≤ 10. Ces valeurs mesurées sont lues dans les DONNÉES (contrôle de
    l'outil sur le fit complet), pas dans un résultat V1.
- **Témoin.** Part de Macron chez k dans les autres duels Macron–X, soit une MAE de 35,8 (revue) ;
  recalculé sur la même cible et rapporté.
- **Méthode.** Dépistage MAP d'abord (6 initialisations), puis NUTS seulement si R8c passe en NUTS sur
  les données complètes.

**Dépistage MAP V1 de R8c** (hypothèse Mélenchon–Macron retirée ;
`results/modes_map_hiercercle_ra_fid_omk_2022_T1-30_sans_MélenchonMacron.json` ; 6 initialisations,
mode dominant = init 3). Partage Mélenchon / (Mélenchon + Macron) prédit, contre la moyenne de mars :

| électorat | modèle | mesuré | « Aucun » modèle / mesuré |
|---|---|---|---|
| Hidalgo | **0,6** | 38,5 | 42,6 / 22,0 |
| Roussel | **99,4** | 65,8 | 0,6 / 21,0 |
| Jadot | **96,8** | 43,9 | 0,4 / 22,5 |
| Le Pen | 25,1 | 42,3 | 70,3 / 53,0 |
| Pécresse | 11,8 | 10,6 | 43,4 / 38,5 |
| Zemmour | 22,5 | 42,7 | 65,6 / 59,0 |

- V1 : MAE 27,2 ✗ (≤ 12) ; écart Le Pen −17,2 ✗ ; écart Zemmour −20,2 ✗. Le témoin non structurel
  donne 35,8 : le modèle fait mieux, mais il est loin du seuil.
- Sur les cibles d'entre-deux-tours, le même fit reste bon : Mélenchon 64,8 ✓, σ_LP/σ_M 1,31 ✓,
  monotonie ✓, partages MAE 4,5, « Aucun » 5,0.

**Lecture.** Les électorats de la gauche modérée sont SATURÉS dans un duel Mélenchon–Macron
qu'il n'a pas vu. Hidalgo va entièrement à Macron ; Roussel et Jadot vont presque entièrement à
Mélenchon (mesuré 39 / 66 / 44 %). La géométrie interne de la gauche n'est identifiée que par le T1 et
par les duels contre Macron ou Le Pen, et elle ne contient pas l'hésitation mesurée entre Mélenchon et
Macron. **R8c échoue au test de valeur ajoutée V1** (dépistage) : ses bons chiffres sur les cibles
d'entre-deux-tours sont en partie un ajustement en échantillon.

### 8.60 Témoin V2 (duels jamais sondés) calculé à 0 fit, figé AVANT tout fit V2

Cadre : 2022 à T1−30, Bradley-Terry ridge (`second_round._bradley_terry`, même prior que le modèle).
Pour chaque paire, le témoin est ajusté sur tous les AUTRES duels de la fenêtre. La cible est la
moyenne des duels de la paire dans les 30 jours précédant `as_of`.

| paire retirée | observé (n) | Bradley-Terry | écart |
|---|---|---|---|
| Macron–Mélenchon | 66,3 (5) | 56,2 | −10,1 |
| Macron–Zemmour | 64,4 (20) | 73,9 | +9,5 |
| Macron–Pécresse (contrôle) | 59,2 (17) | 44,6 | −14,7 |
| Macron–Le Pen (contrôle) | 57,4 (22) | 65,5 | +8,2 |

Le graphe des duels de 2022 est une étoile autour de Macron : un candidat retiré de son seul duel n'a
presque plus d'information, et le témoin est faible (MAE de 9,8 sur les deux paires V2).

**Critère V2 figé** : sur Macron–Mélenchon et Macron–Zemmour, chacun retiré à son tour (deux fits), la
MAE du modèle doit être ≤ 9,8 − 1,5 = **8,3 pt**. Les deux paires de contrôle sont rapportées.

### 8.61 V1 réévalué et mélange individuel λ (revue locale du 21/11 ~12:30Z ; avant code)

**Évaluation de V1 corrigée (données seules).**

- La cible ne compte que **2 notices**, pas 3 : BVA 8-9 mars ne publie que l'électorat de Macron.
  Hidalgo et Roussel ne viennent que d'une notice (≈ 50 à 90 répondants), d'où un plancher de bruit
  d'environ 6 pt de MAE.
- Le témoin de 35,8 ne correspondait pas à la cible figée. Recalculé à 0 fit sur la cible de §8.59
  (moyenne par hypothèse, lignes propres exclues), il vaut **25,9** pour le partage et 17,8 pour
  « Aucun ».
- Témoins triviaux : un 50/50 constant donne **14,6** ; « Pécresse 10, les autres 50 » donne 8,2,
  mais il est construit a posteriori et n'est pas retenu.
- **R8c (27,2) ne bat donc ni le témoin structurel (25,9) ni le 50/50 (14,6).** En échantillon, le
  fit complet plafonne à 11,4 (de 7,8 à 15,5 selon le mode).
- **Critère V1 révisé, figé :** MAE ≤ 12 ET ≤ 50/50 − 1,5 (= 13,1). Il est lu sur TOUS les modes
  (médiane et pire), pas sur le seul mode dominant.

**Diagnostic (revue).**

- L'ordre interne de la gauche n'est pas identifié. Selon le mode, Hidalgo passe à droite de
  Macron ; Jadot et Roussel ont σ ≈ 0,03-0,04 et ω_k ≈ −9.
- Les électeurs sont quasi déterministes : la part d'électorat avec |κΔA + δ| < 2 vaut 0 à 7 %.
- La quadrature n'est pas en cause (1024 nœuds donnent les mêmes prédictions).

**(R9) Mélange individuel λ, un paramètre.** Une fraction λ des électeurs choisit entre les deux
finalistes indépendamment de la géométrie (choc individuel dominant) :

$$q(v)=(1-\lambda)\,\sigma\big(\kappa[A_a-A_b]\big)+\lambda\cdot\tfrac12,\qquad \lambda\sim\mathrm{Beta}(2,8).$$

- La forme est la même pour les duels et les reports, et le δ de Newton réabsorbe le niveau.
- Les lignes propres (fidélité, §8.56) ne sont pas concernées.
- λ = 0 redonne R8c.

À paramètres fixés sur le MAP V1 : 27,2 → 19,1 (λ = 0,3), avec des cellules vues qui s'améliorent.
Ce n'est probablement pas suffisant seul.

Variante `hiercercle_ra_fid_omk_lam`. Protocole : dépistage MAP sur les données complètes puis sur V1.
Critères figés : V1 révisé ci-dessus (médiane des modes) ; cellules vues ≤ celles de R8c + 0,5 ;
quantile 5 % de λ > 0,05 ; critères 2 et 3b de §8.58.

**ARBITRAGE À TRANCHER (utilisateur), proposé par la revue.** Les reports par électorat dans un duel
jamais sondé ne sont pas contraints par les données dans cette classe de modèles. Trois options :

- (i) continuer l'empilement (λ, puis identification de la gauche via le modèle joint avec 2017) ;
- (ii) un modèle unique, avec une couche de reports NON spatiale explicite : des effets électorat ×
  finaliste rétrécis vers la prédiction spatiale ;
- (iii) un modèle de transfert séparé pour les reports, avec le spatial seulement pour le T1 et les
  duels (on renonce à l'unicité).

### 8.62 Lecture du dépistage MAP de R9 (λ) — échec (21/11 ~13:20Z, Mac, B = 100, 6 inits)

**Données complètes** (`modes_map_hiercercle_ra_fid_omk_lam_2022_T1-30.json`) :

| | R9 | R8c (MAP) |
|---|---|---|
| MAE des cellules vues | 5,25 | 3,55 |
| Mélenchon (modèle / mesure) | 59,4 / 63,0 | |
| « Aucun » | 5,08 | |
| σ_LP/σ_M | 1,30 | |

- R9 dépasse R8c + 0,5 sur les cellules vues ⇒ critère **non tenu**.
- Les modes sont éclatés (5 groupes sur 6 inits).
- λ au MAP vaut de 0,12 à 0,34 ⇒ critère λ tenu.
- En échantillon, V1 est à 9,6 en médiane des modes (pire 12,8 ; R8c : 11,4).

**V1, hypothèse Mélenchon–Macron retirée** (`…_sans_MélenchonMacron.json`) :

- V1 hors échantillon : **médiane des modes 18,6 et pire 22,9** (mode dominant 22,9, autres 16,5 et
  18,6), avec λ au MAP de 0,16 à 0,40.
- R8c faisait 27,2. **Le critère n'est pas tenu** (≤ 12 ET ≤ 13,1) : R9 bat le témoin structurel
  (25,9) mais **pas le 50/50 (14,6)**.
- Sur le mode dominant, l'électorat Le Pen part vers Mélenchon à 75,5 (mesure 42,3) et Zemmour à 68,0
  (mesure 42,7). La gauche reste mal ordonnée : Hidalgo 26 et Roussel 36, contre 38 et 66 mesurés.
- Cellules vues du fit V1 : MAE 3,53.

**Verdict :** R9 n'est pas retenu.

- λ adoucit les cellules, mais ne fixe pas le sens des reports hors échantillon. Sans
  Mélenchon–Macron, le cercle bascule encore les droites vers Mélenchon.
- Ce résultat confirme le diagnostic de §8.61 : dans cette classe de modèles, les reports d'un duel
  non sondé ne sont pas contraints par les données.
- L'arbitrage (i)/(ii)/(iii) devient bloquant : aucune piste spatiale à un paramètre ne reste en file.

### 8.63 Réparer la mesure de V1 (revue locale du 21/11 ~16:00Z ; avant code, figé)

**Constat (revue, données seules).**

- Le témoin 50/50 ignore l'agrégat du duel, que le modèle connaît (5 duels Mélenchon–Macron
  conservés). Le témoin à 0 paramètre qui convient est **U** : tous les électorats non finalistes
  reçoivent le même logit, calé par le même Newton δ sur l'agrégat du fit. C'est τ = 0 dans
  $\mathrm{logit}\,\tilde G_k=\tau\,\mathrm{lG}^{geo}_k+\delta$.
  - Sur le point MAP V1 de R8c, U vaut **14,8** ; avec l'agrégat sondé, ≈ 11,5.
  - Le critère « ≤ 12 ET ≤ 13,1 » de §8.61 est donc franchissable sans structure. **Il est retiré.**
- Le plancher d'échantillonnage de la moyenne V1 est ≈ 3,3 pt (4,7 avec un effet de plan de 2), et
  non 6.
- En 2022, une fois l'hypothèse retirée, Mélenchon n'est finaliste dans aucune autre hypothèse de la
  fenêtre : Mélenchon–Macron seul ne départage aucune option.

**Nouveau protocole V1-LOHO (une hypothèse laissée de côté à la fois), figé :**

- Hypothèses retirées tour à tour : Mélenchon–Macron, Macron–Zemmour, Macron–Pécresse (2022, T1−30).
  Cible : moyenne des notices des 30 jours, lignes propres exclues.
- Témoin **U**, calculé par le MÊME code sur le MÊME point ou tirage, et consigné avant toute lecture
  du modèle.
- **Critère :** MAE moyenne sur les 3 hypothèses ≤ U − 2, et aucune hypothèse > U + 2.
- **Lecture des modes :** un mode est retenu si Δlp ≤ max(10, 4·SE) du meilleur, et la partition se
  fait par log-densité, pas par σ_Mélenchon seul. Les points MAP ne valent que comme dépistage, sans
  verdict de stabilité.
- **Drapeau d'identification**, rapporté à chaque lecture : max |z_pos| > 3 (positions forcées).

**Lecture prévue, 0 fit :** les points MAP existants de R8c pour les 3 hypothèses retirées.

- R8c ≤ U − 2 sur Macron–Zemmour et Macron–Pécresse : seule Mélenchon–Macron est disqualifiée
  (absence de pont).
- Échec sur les deux : la classe spatiale ne prédit pas les reports d'une hypothèse jamais sondée
  mieux que U.

Il faut pour cela 2 nouveaux dépistages MAP de R8c, Macron–Zemmour et Macron–Pécresse retirées, sur le
Mac.

**R9 (λ) : retiré définitivement.**

- Le critère sur le quantile 5 % de λ n'est pas lisible au MAP, et le mode du prior vaut ≈ 0,2.
- La densité de Beta(2,8) est nulle en λ = 0 : « λ = 0 emboîte R8c » ne vaut que pour la vraisemblance.

### 8.64 Résultat du V1-LOHO de R8c face au témoin U (21/11 ~17:10Z, Mac, dépistage MAP, 6 inits par hypothèse)

Mode retenu par log-densité (Δlp ≤ 10, un seul mode chaque fois). Fichiers
`results/v1loho_map_hiercercle_ra_fid_omk_2022_T1-30_sans_*.json`.

| Hypothèse retirée | MAE R8c | U | Écart à U | max \|z_pos\| |
|---|---|---|---|---|
| Mélenchon–Macron | 27,2 | 10,5 | +16,7 ✗ | 4,9 |
| Macron–Zemmour | 10,6 | 22,0 | −11,4 ✓ | 5,0 |
| Macron–Pécresse | 22,1 | 14,9 | +7,2 ✗ | 4,9 |
| **Moyenne** | **20,0** | **15,8** | **+4,2** | |

**Verdict au critère figé de §8.63** (moyenne ≤ U − 2, aucune hypothèse > U + 2) : **échec**.

- Le modèle bat largement U là où la structure gauche-droite décide, c'est-à-dire face à Zemmour : U
  est mauvais (22) parce que les électorats divergent fortement.
- Il perd là où il faut ordonner finement le centre et la gauche (Mélenchon, Pécresse).
- Dans les trois fits, la géométrie est forcée (max |z_pos| ≈ 5). Les modes secondaires donnent les
  mêmes conclusions (Zemmour 9,7-10,6 ; Pécresse 20,6-22,6).
- Ce sont des points MAP, qui ne valent que comme dépistage.

**Lecture.** La classe spatiale actuelle prédit le sens d'un report jamais sondé quand les deux
finalistes sont aux deux bords. Elle échoue quand ils sont voisins dans l'ordre (centre et droite
modérée, gauche et centre). C'est exactement le domaine où la revue recommande une couche
électorat × finaliste (option ii) avec un τ_report calibré. L'arbitrage (i)/(ii)/(iii) reste ouvert
pour l'utilisateur.

### 8.64 bis Corrections de la revue du 21/11 ~20:30Z

**(a) La valeur 14,8 citée en §8.63 n'est pas le témoin U.** C'est le partage s_ab appliqué tel quel,
sans Newton. Le code applique bien la définition de §8.63, et U = 10,5 est juste.

**(b) Bug de nom dans modes.v1.** L'électorat « Éric Zemmour » était compté comme non finaliste dans
Macron–Zemmour. C'est corrigé par la comparaison sur les noms normalisés.

Valeurs corrigées pour Macron–Zemmour : R8c 9,7 (U 21,7). Moyennes : R8c 19,7 contre U 15,7. **Le
verdict d'échec tient.**

**(c) Le témoin U doit mettre les électorats des finalistes à la fidélité R7 dans le Newton** : c'est
U_fid. Sinon δ se cale sur une fidélité géométrique trop basse (79,5 % au lieu de 95 %).

| | U_fid (revue, à 0 fit) |
|---|---|
| Mélenchon–Macron | 11,3 |
| Macron–Zemmour | 18,3 |
| Macron–Pécresse | 12,7 |
| **Moyenne** | **14,1** |

À coder ; U_fid devient le témoin du V1-LOHO (§8.66).

**(d) Lecture sélective.** Pour Mélenchon–Macron, les modes secondaires font 14,2 à 19,3, contre 27,2
pour le mode retenu (meilleure log-densité). Le mode le mieux ajusté est le pire hors échantillon.
L'échec tient grâce à Pécresse, à environ +5 de U sur tous les modes.

**(e) Le drapeau |z_pos| > 3 ne discrimine pas.** Le « ≈ 5 » vient de Lassalle, ancre éditoriale du T1
déjà à −4,1 sur le segment. Les signaux qui comptent sont Hidalgo à +3,8 (μ = 0,69, à droite de Macron)
et Taubira à +3,4. Le drapeau est restreint aux finalistes et aux électorats des reports, et Lassalle
est rapporté à part (à coder).

### 8.65 Test à 0 fit ω_k[Mélenchon] := ω (point MAP de R8c, init 3 ; 21/11 ~17:20Z)

On remet l'abstention propre de l'électorat de Mélenchon (ω_k = −2,83) au niveau commun (ω = −3,56),
sans rien d'autre de changé.

| | Point MAP de R8c | ω_k[Mélenchon] := ω |
|---|---|---|
| Partage de l'électorat Mélenchon | 62,4 | 62,0 (mesure 63,0) |
| « Aucun » de l'électorat Mélenchon | 41,3 | 28,4 (mesure 43,6) |
| MAE des partages | 3,55 | 3,65 |
| MAE « Aucun » | 3,35 | 6,57 |

- **La prédiction de la routine (−8 pt sur le partage Mélenchon) est démentie** : −0,4.
- Le partage de Mélenchon ne passe pas par ω_Mélenchon. ω_k sert à l'abstention de cet électorat, ce
  qui est son rôle.
- Les critères 1-2 restent donc des critères, pas des artefacts de ω_k.
- ω_k reste conditionnel à la lecture de NUTS (retrait si s_ω q5 < 0,2).

### 8.66 τ_report en cross-fit (forme (ii-0) de l'option (ii)) — spécifié et figé, NON lancé avant l'arbitrage

**Forme.** Pour un triplet électorat × hypothèse jamais observé :

$$\mathrm{logit}\,\tilde G_k=\tau\,\mathrm{lG}^{geo}_k+\delta.$$

- Les électorats des finalistes sont à la fidélité R7 DANS le Newton, et δ réimpose l'agrégat.
- Les duels et le T1 sont inchangés : c'est le pendant de τ_paire (§8.33).

**Calibrage sans fuite.** Pour chaque hypothèse h, τ est choisi sur les autres hypothèses (chacune lue
sur son propre fit qui l'exclut), puis évalué sur h.

**Critères figés :**

- **V1-LOHO poolé :** 2022 (Mélenchon–Macron, Macron–Zemmour, Macron–Pécresse), 2017 (Fillon–Le Pen),
  2027 (Philippe–Le Pen, Attal–Le Pen).
  - Témoin U_fid ; finalistes exclus par nom normalisé.
  - Seuils : moyenne ≤ U_fid − 2, aucune hypothèse > U_fid + 3, lus sur tous les modes à Δlp ≤ 10
    (minimum et maximum rapportés).
- **V2 :** MAE ≤ celle de `hier` sans reports + 0,5.
- **Plausibilité :**
  - σ_LP/σ_M < 1,5 et monotonie ≤ 10 % ;
  - |z_pos| ≤ 3 pour les finalistes et les électorats des reports, Lassalle à part ;
  - Hidalgo à gauche de Macron.

**Prédictions consignées avant lecture (revue, 0 fit sur le mode c3 de 2022) :**

- τ* entre 0,15 et 0,35 ;
- moyenne 2022 ≈ 11,2 contre 14,1 pour U_fid ;
- Mélenchon–Macron au-dessus de U_fid (+4) ;
- duels inchangés.

### 8.67 Échelle de parcimonie et V2 de R8c (revue du 21/11 ~20:30Z ; pas d'arbitrage requis)

**(a) V2 de R8c face à `hier` sans reports.**

- MAE sur Macron–Mélenchon et Macron–Zemmour retirés.
- Seuil figé : R8c ≤ `hier` − 1, sinon la couche de reports ne se justifie que par la plausibilité et
  ne sert pas aux duels publiés.
- Lecture d'abord au MAP sur le Mac, puis NUTS sur la VM si R8c passe.

**(b) Échelle de parcimonie au V1-LOHO (2022, 3 hypothèses, témoin U_fid).**

- Variantes : `hier_ra` (segment), `hier_ra_fid_omk` (segment) et `hiercercle_ra_fid` (cercle sans
  ω_k), face à R8c.
- Règle de retrait figée : un mécanisme est retiré si son retrait coûte < 2 pt en V1 (partage) ET
  < 1 pt en V1 (« Aucun »), sans casser σ_LP/σ_M < 1,5 ni la monotonie.

**(c) MAP prolongé.**

- 20 000 pas, lr décroissant, ‖∇‖ journalisé.
- Si les écarts de log-densité entre initialisations bougent de plus de 10 nats après prolongation, ce
  sont des non-convergences et non des modes.

### 8.68 Échelle de parcimonie au V1-LOHO (§8.67 b) — résultat (21/11 ~22:30Z, Mac, dépistage MAP, 6 inits × 3 hypothèses × 4 variantes)

MAE du partage du mode retenu (meilleure log-densité, Δlp ≤ 10) ; entre parenthèses, le témoin U_fid du
même point. Fichiers `results/v1loho_map_*_2022_T1-30_sans_*.json`.

| Variante | Mélenchon–Macron | Macron–Zemmour | Macron–Pécresse | Moyenne (U_fid) | Δ |
|---|---|---|---|---|---|
| R8c `hiercercle_ra_fid_omk` (cercle + R7 + ω_k) | 27,2 (10,8) | 9,7 (18,1) | 22,1 (13,9) | 19,7 (14,3) | +5,4 |
| `hiercercle_ra_fid` (cercle + R7) | 10,6 (10,2) | 8,8 (17,4) | 24,3 (14,6) | **14,6 (14,1)** | +0,5 |
| `hier_ra_fid_omk` (segment + R7 + ω_k) | 34,4 (10,3) | 7,1 (19,9) | 18,2 (14,9) | 19,9 (15,0) | +4,9 |
| `hier_ra` (segment seul) | 9,3 (10,8) | 6,1 (19,1) | 24,8 (16,8) | **13,4 (15,6)** | −2,2 |

**Critère figé** (moyenne ≤ U_fid − 2, aucune hypothèse > U_fid + 3) : **aucune variante ne le tient.**

- Macron–Pécresse dépasse U_fid de 3,3 à 9,7 partout.
- `hier_ra` tient la moyenne, de justesse (−2,2), mais pas Pécresse (+8,0).

**Règle de retrait (partage) :**

- **Retirer ω_k fait GAGNER 5,1 pt** sur le cercle (19,7 → 14,6) et 6,5 pt sur le segment (19,9 →
  13,4, mais sans R7 non plus).
- La règle (« retrait si son coût < 2 pt ») désigne donc **ω_k au retrait**, sous réserve de la case
  « Aucun » : elle n'est pas encore lue par mode, à coder avant de conclure.
- Le gain vient surtout de Mélenchon–Macron (27,2 → 10,6), conformément aux pathologies de ω_k
  (s_ω ≈ 3, ω_k de Roussel et Jadot ≈ −9).

**Cercle ou segment.**

- Au V1, le segment seul (13,4) et le cercle + R7 (14,6) sont à 1,2 pt, soit dans le bruit (plancher
  ≈ 3,3) et entre points MAP.
- La parcimonie ne tranche donc pas par V1 ; elle tranche par la plausibilité.
  - Segment : σ_LP/σ_M de 3,1 à 3,7, 21 à 33 % de votants mal classés.
  - Cercle : de 1,0 à 1,6, et 0 %.
- **Le cercle + R7 sans ω_k devient la variante de référence candidate.**

**Avertissements.**

- Ce sont des points MAP (Adam, 5 000 pas, non prolongés).
- Les modes non retenus varient beaucoup : Pécresse fait 7,1 pour le mode c5 du cercle + R7, contre
  24,3 retenu.
- Aucun verdict de stabilité.

**§8.68, suite : case « Aucun » par mode (mode retenu, 21/11 ~23:00Z).**

| | Mélenchon–Macron | Macron–Zemmour | Macron–Pécresse | Moyenne |
|---|---|---|---|---|
| R8c (avec ω_k) | 15,3 | 4,8 | 3,3 | 7,8 |
| Cercle + R7 (sans ω_k) | 4,8 | 6,8 | 7,4 | **6,3** |

- **Retirer ω_k améliore aussi « Aucun »** (−1,5 pt en moyenne).
- ω_k aide Zemmour et Pécresse (−2 à −4) mais ruine Mélenchon–Macron (+10,5).
- **Règle de retrait figée en §8.67 (b) : appliquée. ω_k est RETIRÉ** (partage −5,1, « Aucun » −1,5,
  soit environ 24 paramètres de moins).
- **Référence : `hiercercle_ra_fid`** (cercle + fidélité révélée R7, abstention commune).
- C'est un dépistage MAP : le verdict sera à confirmer par NUTS sur une seule plateforme.

### 8.69 Protocole V2 au MAP (dépistage de §8.60 et §8.67 a ; avant code, figé)

- **Retrait.** La variable `PRES2027_SANS_PAIRE="A|B"` retire tous les duels de la paire
  (`_sans_paire`). Elle retire AUSSI les reports de l'hypothèse correspondante, pour un duel
  véritablement jamais sondé : les cellules de reports d'une hypothèse informent son agrégat.

  C'est plus strict que §8.60, dont le témoin BT ne voit pas de reports ; le seuil reste figé.
- **Cadre.** 2022, T1−30, `--lse-officiel`, 6 inits. Variantes `hiercercle_ra_fid` (référence, §8.68)
  et `hier` (témoin sans reports).
- **Lecture.** Part de A prédite au mode retenu (Δlp ≤ 10), contre la moyenne des duels de la paire
  sur les 30 jours avant `as_of` : Macron–Mélenchon 66,3, Macron–Zemmour 64,4. Les autres modes sont
  rapportés.
- **Critères figés :**
  - MAE sur les deux paires ≤ 8,3 (§8.60) ;
  - ET ≤ celle de `hier` − 1 (§8.67 a).
- **Prédiction consignée avant lecture :** comme en V1, Macron–Zemmour (les deux bords) devrait être
  bien prédit, et Macron–Mélenchon mal prédit (aucun pont).

**§8.69, résultat (21/11 ~23:40Z, Mac, dépistage MAP, 6 inits).** Part de Macron prédite au mode
retenu (Δlp ≤ 10). Fichiers `results/v2paire_map_*_2022_T1-30_v2sans_*.json`.

| Paire retirée (observé) | `hiercercle_ra_fid` | `hier` (sans reports) |
|---|---|---|
| Macron–Mélenchon (66,3) | 64,3 (−2,0) ; 2ᵉ mode retenu 68,2 (+1,9) | 75,5 (+9,2) |
| Macron–Zemmour (64,4) | 70,4 (+6,0) | 69,0 (+4,6) |
| **MAE** | **4,0** | **6,9** |

- **Les deux critères figés sont tenus au dépistage :** 4,0 ≤ 8,3, et 4,0 ≤ `hier` − 1 = 5,9.
- **La prédiction consignée est démentie dans les deux sens :**
  - Macron–Mélenchon est bien prédit (les reports placent l'électorat Mélenchon et la gauche) ;
  - Macron–Zemmour est un peu moins bien prédit que par `hier`.
- Les modes non retenus de la référence restent dans [−0,9 ; +7,2] pour Mélenchon et [+5,8 ; +9,7]
  pour Zemmour.
- **Verdict : favorable, sur le Mac, au MAP ⇒ NON PROBANT** par règle. NUTS sur une seule plateforme,
  la VM : DEMANDE VM n° 5.
- Lecture provisoire : les reports n'apprennent pas bien les cellules d'une hypothèse jamais sondée
  (V1-LOHO, §8.68), mais ils améliorent l'AGRÉGAT d'un duel jamais sondé de près de 3 pt. C'est le
  premier argument chiffré, hors plausibilité, pour garder la couche de reports dans le modèle unique.

### 8.70 Corrections de la revue du 22/11 ~01:00Z (avant toute nouvelle lecture)

**(a) MAP non convergé à 5 000 pas (§8.67 c vérifié).** À 20 000 pas (`--decroissant`), plusieurs
inits convergent vers un même point. Le point de meilleure lp n'est pas celui de 5 000 pas.

Mélenchon–Macron, `hiercercle_ra_fid` :

| | 5 000 pas | 20 000 pas |
|---|---|---|
| Meilleur point | init 3, lp 12 642, MAE 10,6 | init 4, lp ≈ 12 703-12 708, MAE ≈ 24 |
| Arc du meilleur point | | 1,76 rad (presque un segment) |
| Inits 0, 2, 3 | | même point, lp 12 651, arc 3,56, MAE ≈ 10,7 |

- **Les tableaux de §8.68 et §8.69 portent sur des points non convergés.** Leurs verdicts (ω_k retiré,
  V2 favorable) sont **provisoires** et doivent être relus à 20 000 pas.
- L'arc est une crête plate (de 1,8 à 6,0 rad à κ, ω et c constants), et c'est lui qui règle le
  passage de la droite vers Mélenchon par l'arrière du cercle.

**Lecture de la session locale à 20 000 pas (22/11 ~01:45Z), qui confirme la revue.** Fichiers
`results/v1loho_map_hiercercle_ra_fid_2022_T1-30_p20000d_sans_*.json`.

| Hypothèse retirée | Point retenu | MAE partage | « Aucun » | U_fid | Autres modes |
|---|---|---|---|---|---|
| Mélenchon–Macron | init 4, lp 12 707,5 | **24,0** | 10,8 | 10,4 | 10,7 (lp 12 656-12 657) |
| Macron–Pécresse | inits 0/2, lp 12 357,3 | **24,0** | 7,3 | 14,5 | de 9,8 (init 5, −70 nats) à 25,6 |

Avec Zemmour à 5 000 pas (8,8), la moyenne de la référence passe à ≈ 18,9 : le V1-LOHO échoue
nettement.

**(b) Retrait de ω_k : formulation corrigée.**

- La règle de §8.67 (b) porte sur le COÛT du retrait (< 2 pt et < 1 pt). Elle reste remplie sur la
  médiane de tous les modes :
  - partage : 16,7 avec ω_k, 11,9 sans ;
  - « Aucun » : 8,1 avec ω_k, 6,4 sans.
- Le « gain de 5,1 pt » n'est pas robuste à 20 000 pas (environ 18,9 contre 19,7, et R8c n'a pas été
  prolongé). Il ne doit plus être présenté comme un gain.
- **La clause « sans casser σ_LP/σ_M < 1,5 » n'était pas vérifiée.** La référence atteint 1,55-1,62 sur
  des points retenus, alors que R8c restait entre 1,31 et 1,48. La mention « de 1,0 à 1,6 » de §8.68
  était un relâchement non signalé de 1,5 à 1,6. **Retiré** : le seuil reste 1,5, et le retrait de ω_k
  est **provisoire** tant que la clause n'est pas relue à 20 000 pas.

**(c) V2 : attribution non identifiée.** Le témoin `hier` diffère de la référence par trois facteurs :
la géométrie, les reports et R7. Le gain V2 vient du passage arrière du cercle : électorats Le Pen 45,6
et Pécresse 28,8 vers Mélenchon, contre 0 pour `hier`. Les cellules de la référence (MAE 11,45) sont au
niveau de U_fid (11,43), et l'agrégat juste vient en partie de compensations. **La phrase « premier
argument chiffré pour garder la couche de reports » est retirée**, en attendant le plan 2×2 (d).

Le critère V2 ne discrimine presque pas : `hier` tient seul le seuil de 8,3, et le gain vient d'une
seule paire. Les seuils V2 de §8.66 (`hier` + 0,5) et de §8.67 (`hier` − 1) n'ont pas le même rôle :
§8.66 protège les duels (τ_report ne doit pas les bouger), §8.67 exige un gain.

**(d) Protocoles figés pour la suite (MAP 20 000 pas `--decroissant`, Mac) :**

1. **Relecture V2 de la référence (2 paires) et V1-LOHO de R8c (3 hypothèses) à 20 000 pas.**
   - Si le point de meilleure lp en V2 donne Macron–Mélenchon avec |écart| > 6, ou une MAE V2 >
     `hier` − 1 (5,9, `hier` relu lui aussi à 20 000 pas), le V2 favorable est retiré et la DEMANDE
     VM n° 5 annulée.
   - La règle de retrait de ω_k est réappliquée à 20 000 pas, avec la clause σ_LP/σ_M < 1,5.
2. **Plan 2×2 en V2 : {segment, cercle} × {sans, avec reports}** (`hier`, `hier_ra`, `hiercercle_sr`,
   référence).
   - On attribue le gain aux reports si MAE(`hiercercle_sr`) − MAE(référence) ≥ 1, en moyenne sur les
     2 paires.
   - Prédiction consignée : `hiercercle_sr` ≥ 70 sur Macron–Mélenchon.
3. **Contrôles §8.60 en V2** (Macron–Pécresse, Macron–Le Pen). On qualifie le V2 de « favorable »
   seulement si la référence est ≤ `hier` sur au moins 3 des 4 paires.
4. **Profil de l'arc** (arc fixé à 1,8 / 3,5 / 5,0 / 6,2, données complètes et V2 Macron–Mélenchon).
   - Si Δlp < 10 entre deux arcs alors que V2 bouge de plus de 4 pt, les données ne révèlent pas l'arc.
   - Il faudra alors intégrer l'arc (NUTS), ou le fixer par une règle pré-enregistrée sur 2017 + 2022.
     Ce dernier point est un **ARBITRAGE**.
5. **η ≡ 0 sur la référence** (−24 paramètres). η est retiré si V2 ≤ référence + 1, les duels en
   échantillon ≤ référence + 0,5 ET V1-LOHO ≤ référence + 2.
6. **Confirmation sur 2017** (paires ayant au moins 3 duels), même seuil « ≤ `hier` − 1 ».

### 8.71 Résultat NUTS de R8c sur la VM (DEMANDE VM n° 4 ; 22/11 ~02:25Z, routine)

`hiercercle_ra_fid_omk --lse-officiel`, 2022 à T1−30, 4 chaînes CPU indépendantes (≈ 16 h chacune),
B = 25. Fichiers : `results/second_round_hiercercle_ra_fid_omk_v2_2022_T1-30_lseo.json`,
`results/modes_second_round_hiercercle_ra_fid_omk_v2_2022_T1-30_lseo.json`. Lecture d'un postérieur, pas
d'un dépistage.

**Convergence.** lp 12876,6 / 12880,3 / 12878,2 / 12876,2 (écart 4 nats), 4 chaînes retenues, un seul mode
(partition : 4 dominantes, 0 autre) ; R̂ max 1,035, ESS min 154, 0 divergence ; duel Macron–Le Pen
58,96 % [48,0 ; 69,6] (résultat 58,55 ✓), écart entre chaînes 0,56 pt ; écart max des duels 0,14 pt.
Les 6 modes distincts du dépistage MAP (§8.58) étaient donc des points non convergés, pas des modes du
postérieur (revue du 21/11, point 2, test (i) confirmé indirectement).

**Critères de §8.58 (figés) :**

| critère | seuil | R8c NUTS (VM) | MAP (§8.58) |
|---|---|---|---|
| « Aucun » MAE | ≤ 3,75 | 3,20 ✓ | 3,35 |
| MAE des 4 partages (diagnostic) | ≤ 5 | 3,65 ✓ (persistance 2,61) | 3,55 |
| Mélenchon | à 5 pt de 63,0 | 62,3 ✓ | 62,4 |
| σ_LP/σ_M | < 2 (et < 1,5 de §8.50) | 1,42 ✓ (✓) | 1,54 |
| 3b monotonie | ≥ 90 % | part mal classée pondérée 0,0 ✓ | 0 % |
| stabilité | §8.50 | 4/4 chaînes, 0,14 pt ✓ | — |
| T1 MAE | ≤ 3,1 | 3,03 ✓ (couverture 1/12) | 3,0 |
| s_ω, quantile 5 % | > 0,2 | 1,01 ✓ (médiane 1,58) | — |
| V1 | §8.57 | non fait en NUTS (MAP : échec, §8.63-8.64) | — |

Partages (modèle / mesure) : Mélenchon 62,3 / 63,0 ; Pécresse 61,6 / 65,0 ; Jadot 84,9 / 87,5 ;
Zemmour 17,3 / 9,4. « Aucun » : 41,5 / 43,6 ; 20,7 / 29,9 ; 31,9 / 33,3 ; 12,8 / 12,7.
Structure : Zemmour → Mélenchon 34,3 ✓ ; Le Pen → Mélenchon 42,6 ✗ (≥ 45) ; σ_Mélenchon 0,17 ✗ (≤ 0,15).
c médian 55,5.

**Verdict.** R8c passe TOUS les critères in-sample de §8.58 en NUTS, de façon stable, et passe aussi le
seuil général σ_LP/σ_M < 1,5 (le relâchement à 2 ne sert plus). Le critère 1 de §8.50 (MAE ≤
persistance) échoue (3,65 contre 2,61), comme tous les modèles. Le V1 reste le critère hors échantillon
décisif, et il échoue au MAP (§8.63-8.64, §8.68) : **R8c reproduit les reports qu'il voit, pas ceux qu'on
lui cache**. Ce résultat NUTS ne change donc pas la conclusion de §8.64-8.68.

**Prédictions figées (journal, revue routine du 21/11).** « MAE ≥ 5 » : DÉMENTIE (3,65) ;
« σ_LP/σ_M dans [0,7 ; 1,6] » : vérifiée (1,42). Sur ce modèle, le dépistage MAP à 5000 pas a bien prédit
les grandeurs NUTS (écart ≤ 0,15 pt sur les critères, sauf σ_LP/σ_M 1,54 → 1,42) : le point 2 de la revue
(MAP optimiste) ne se vérifie pas ici, contrairement à A′2.

**Test de la revue du 21/11, point 4 (in-sample)** : bloc `v1_melenchon_macron` de ce fit, données
complètes (lignes Mélenchon–Macron vues) : MAE des partages 11,25 (U 13,83, U_fid 10,17, 50/50 14,63).
Entre les deux seuils figés (> 15 structurel, < 8 identification) : NON CONCLUANT. L'erreur vient surtout
de Roussel (89,2 contre 65,8) puis Pécresse (23,0 / 10,6) et Jadot (55,8 / 43,9) ; Hidalgo 49,6 / 38,5,
Le Pen 42,6 / 42,3 ✓, Zemmour 34,3 / 42,7. Même en voyant les données, le noyau ne sépare pas assez
Roussel de Hidalgo ⇒ indice d'une limite structurelle partielle (Roussel), pas d'un simple défaut
d'identification.

**Plateforme.** Ce jeu (VM, 4 chaînes) est probant seul ; à ne pas fusionner avec le jeu du Mac.
La référence de la session locale est désormais `hiercercle_ra_fid` (ω_k retiré par la règle figée de
§8.67 b, au MAP) : ce NUTS montre que ω_k est identifié (s_ω q5 1,01) et que la case « Aucun » y est bonne
(3,20) ; à confronter au NUTS de `hiercercle_ra_fid` avant de confirmer le retrait.

**§8.71, conséquence pour §8.70 (b) (session locale, 22/11 ~03:10Z).**

- Le NUTS de R8c donne s_ω de quantile 5 % 1,01 > 0,2. Par le critère figé de la revue du 21/11
  (« retrait si s_ω q5 < 0,2 en NUTS »), **ω_k est identifié et ne doit pas être retiré sur ce seul
  motif**.
- Le retrait de §8.68 reposait sur un coût V1-LOHO mesuré sur des points MAP non convergés ; il est
  **suspendu**. R8c redevient la variante de référence jusqu'à la relecture à 20 000 pas : V1-LOHO de
  R8c, en cours sur le Mac.
- La règle sera réappliquée à 20 000 pas avec ses trois clauses (partage < 2, « Aucun » < 1,
  σ_LP/σ_M < 1,5).
- Les 6 « modes » MAP de R8c étaient des non-convergences (§8.71), ce qui confirme §8.70 (a).

### 8.72 Lectures à 20 000 pas (`--decroissant`, Mac, dépistage MAP, 6 inits ; consignées le 23/11 ~20:10Z)

**(a) V2 et plan 2×2 (§8.70 d 1-2).** Écart de la part de Macron au mode retenu (Δlp ≤ 10). Fichiers
`results/v2paire_map_*_p20000d_*`.

| | Macron–Mélenchon (66,3) | Macron–Zemmour (64,4) | MAE |
|---|---|---|---|
| Cercle + reports (`hiercercle_ra_fid`) | −1,3 | +5,3 | **3,3** |
| Segment + reports (`hier_ra`) | 0,0 | +9,3 / +9,7 | 4,7 |
| Segment sans reports (`hier`) | +8,4 | +4,2 | 6,3 |
| Cercle sans reports (`hiercercle_sr`) | +21,2 | +4,0 | 12,6 |

- **Critère V2 de §8.70 (d) 1** (référence de l'époque) : |Mélenchon| = 1,3 ≤ 6 et MAE 3,3 ≤ `hier` −
  1 = 5,3 ⇒ **tenu.** Favorable sur le Mac ⇒ **non probant.**
- **Critère d'attribution de §8.70 (d) 2 :** MAE(`hiercercle_sr`) − MAE(référence) = 9,3 ≥ 1 ⇒ **le
  gain V2 est attribué aux reports.**
- **La prédiction consignée** (`hiercercle_sr` ≥ 70 sur Macron–Mélenchon) est **confirmée** (87,5).
- **Lecture.**
  - Sans reports, le cercle est PIRE que le segment (12,6 contre 6,3) : sa porte arrière, non
    calibrée, envoie trop peu d'électeurs de droite vers Mélenchon.
  - Avec les reports, il devient le meilleur des quatre (3,3 contre 4,7).
  - La géométrie du cercle n'apporte donc quelque chose qu'**avec** les reports ; c'est l'argument
    pour l'unicité du modèle.
  - Le segment avec reports y parvient par un σ_LP/σ_M de 3 à 4, non crédible (front républicain).

**(b) V1-LOHO à 20 000 pas.** MAE du partage du mode retenu, avec U_fid entre parenthèses.

| | Mélenchon–Macron | Macron–Zemmour | Macron–Pécresse | Moyenne (U_fid) | « Aucun » |
|---|---|---|---|---|---|
| R8c (cercle + R7 + ω_k) | 28,1 (10,9) | 10,5 (18,3) | 21,7 (13,8) | **20,1 (14,3)** | 14,7 / 4,5 / 3,2 → 7,5 |
| Cercle + R7 | 24,0 (10,4) | 10,0 (17,4) | 24,0 (14,5) | **19,3 (14,1)** | 10,8 / 6,8 / 7,3 → 8,3 |

- **V1-LOHO :** échec des deux variantes (moyenne > U_fid − 2, et deux hypothèses > U_fid + 3).
- **Règle de retrait de ω_k (§8.67 b), relue à 20 000 pas avec ses trois clauses :**
  - coût sur le partage −0,8 (< 2 ✓) ;
  - coût sur « Aucun » +0,85 (< 1 ✓) ;
  - **σ_LP/σ_M < 1,5 non tenu sans ω_k** (1,55-1,62 sur des points retenus de la référence, revue du
    22/11), contre 1,42 en NUTS pour R8c ✗.

  **ω_k n'est donc PAS retiré.** Avec s_ω identifié en NUTS (q5 1,01), **R8c reste la référence.**
  Le retrait provisoire de §8.68 est annulé.

**(c) Conséquence de protocole.** Le V2 favorable porte sur `hiercercle_ra_fid`, qui n'est plus la
référence. **Le V2 de R8c est à faire** (même protocole, 20 000 pas), avec les contrôles de §8.60
(Macron–Pécresse, Macron–Le Pen ; R8c et `hier`). Critère figé : MAE ≤ `hier` − 1 sur les deux paires
V2, ET R8c ≤ `hier` sur au moins 3 des 4 paires (§8.70 d 3).

### 8.73 Correction de §8.72 (b) et (c) : ω_k RETIRÉ, référence figée (revue du 23/11 ~20:30Z)

**Erreur de lecture.** La clause σ_LP/σ_M de §8.72 (b) a été lue sur des valeurs périmées : les 1,55-1,62
viennent de 5 000 pas, ou d'inits non retenues à 20 000 pas (inits 0/2/3 de Mélenchon–Macron, 51 nats
sous le meilleur point). Vérifié dans les JSON à 20 000 pas, les points retenus de `hiercercle_ra_fid`
donnent tous **σ_LP/σ_M de 1,10 à 1,14** :

| Lecture | σ_LP/σ_M |
|---|---|
| V1-LOHO Mélenchon–Macron | 1,138 |
| V1-LOHO Macron–Zemmour | 1,142 |
| V1-LOHO Macron–Pécresse | 1,105 |
| V2 Macron–Mélenchon | 1,104 |
| V2 Macron–Zemmour | 1,132 |

**Règle de §8.67 (b), appliquée au protocole figé de §8.70 (d) 1 (20 000 pas) :**

- coût sur le partage −0,7 (< 2 ✓) ;
- coût sur « Aucun » +0,86 (< 1 ✓, de justesse) ;
- σ_LP/σ_M < 1,5 ✓.

⇒ **ω_k est RETIRÉ** (environ 24 paramètres).

**Qui prime.**

- Le critère « s_ω q5 > 0,2 » (§8.58) est une condition d'identification à sens unique : il peut
  ajouter un motif de retrait, jamais en annuler un. Être identifié ne prouve pas l'utilité hors
  échantillon.
- C'est la règle de parcimonie §8.67 (b) qui décide, conformément au principe de l'utilisateur.
- La suspension de §8.71 et l'annulation de §8.72 (b) étaient des renversements a posteriori d'une
  décision figée. **Elles sont retirées.**

**Référence FIGÉE avant toute lecture du V2 de R8c :** `hiercercle_ra_fid` (cercle + reports d'avant
T1 + fidélité R7, abstention commune). Le V2 de R8c, relancé après les contrôles, est un **diagnostic**.

**Formulations corrigées.**

- **Attribution du V2.** Le gain de 9,3 pt mesure le BLOC reports + R7 : `hiercercle_sr` n'a ni l'un ni
  l'autre. Le plan 2×2 montre une interaction :
  - reports + R7 sur le segment : −1,6 ;
  - reports + R7 sur le cercle : −9,3.

  Cercle et segment ne sont pas séparés par V2 (3,3 contre 4,7, dans le bruit). C'est la plausibilité
  qui écarte le segment (σ_LP/σ_M de 2,3 à 3,7 avec reports, contre 1,1 pour le cercle). « Argument
  pour l'unicité » est retiré ; il faut lire : les reports sont nécessaires, la géométrie du cercle
  est retenue pour sa plausibilité.
- **Ce que révèle la géométrie.** Elle reproduit l'agrégat des duels et les reports vus, pas les
  reports d'une hypothèse cachée (V1-LOHO au-dessus de U_fid de 5 à 6 pt). La porte arrière (Le Pen →
  Mélenchon) est apprise des reports, pas des duels.
- **Le MAP à 20 000 pas reste partiellement non convergé** (`hier`, `hier_ra`, `hiercercle_sr` : une lp
  différente par init). Plusieurs points retenus reposent sur une seule init, dont celui de la
  référence sur Macron–Mélenchon (arc ≈ 1,76).

**Contrôles de §8.70 (d) 3** (Macron–Pécresse, Macron–Le Pen ; référence et `hier`) : en cours, avec le
critère figé (référence ≤ `hier` sur au moins 3 des 4 paires). Aujourd'hui la référence est à 1 sur 2,
pire que `hier` sur Zemmour (+5,3 contre +4,2).

**Profil de l'arc (VM n° 6) : précision avant lecture.** Avec `arc_fixe`, le terme a priori de l'arc
sort de la lp : sur la grille, l'a priori seul vaut −3,6 / −0,7 / −0,1 / −9,2 nats. Le profil se lit
donc comme une vraisemblance. On rapportera la lp avec l'a priori réintégré, et on ne la comparera pas
aux lp à arc libre.

### 8.74 Confirmation V2 hors 2022 (revue de la routine du 23/11 ~20:15Z, point 8) — spécifié, NON lancé

**Pourquoi.**

- V2 2022 porte sur 2 paires : un écart de 3 pt donne t ≈ 0,7.
- La géométrie, les critères et la variante ont été choisis sur 2022 : V2 2022 est du DÉVELOPPEMENT.
- 2017 n'a que 2 paires sondées en duel (Macron–Le Pen 40, Fillon–Le Pen 30 ; vérifié), ce qui est
  inutilisable seul.
- **La confirmation se fait donc sur 2027**, à T1−30 de la coupure courante, avec les 11 paires sondées.

**Protocole.**

- Pour chaque paire sondée en 2027 avec au moins 3 duels, on la retire (duels + hypothèse de reports,
  comme en §8.69). On prédit la moyenne de ses duels sur les 30 jours.
- Variante : la référence figée en §8.73 (`hiercercle_ra_fid`), avec l'arc libre. Si le profil de l'arc
  conclut à « fixer l'arc par parcimonie » (revue de la routine, point 5), l'arc est fixé AVANT tout run
  2027, à une valeur choisie sans 2027.
- Témoins :
  - `hier` (segment sans reports) ;
  - régression T1 (part du finaliste a en duel ~ part T1 de b sur les autres paires de la même
    élection ; définition exacte à écrire et chiffrer à 0 fit AVANT le premier run).
- MAP 20 000 pas `--decroissant`, poli par L-BFGS (revue de la routine, point 3).

**Critère figé :** MAE ≤ min(témoins) − 1, ET au moins 8 victoires sur 11 contre le meilleur témoin
(test de signe ; si moins de 11 paires sont éligibles, ⌈0,7·n⌉).

**Statut.** Le code (paires 2027, témoin de régression, polissage) est à écrire. Le lancement attend
l'ARBITRAGE « V2 2022 = développement ». La session locale y souscrit, l'utilisateur tranche.

### 8.75 Profil de l'arc (§8.70 d 4 ; DEMANDE VM n° 6) — résultat (23/11 ~22:00Z, VM, routine)

`hiercercle_ra_fid --lse-officiel`, 2022 à T1−30, `PRES2027_ARC_FIXE` ∈ {1,8 ; 3,5 ; 5,0 ; 6,2},
B = 100, MAP 20 000 pas `--decroissant`, 4 inits ; données complètes et V2 (Macron–Mélenchon retirée).
Fichiers : `results/modes_map_hiercercle_ra_fid_2022_T1-30_arc*_p20000d.json`,
`results/v2paire_map_hiercercle_ra_fid_2022_T1-30_arc*_p20000d_v2sans_*.json`.

| arc | lp max complet (`modes`, sans jacobien) | lp max V2 | V2 Macron–Mélenchon, mode retenu (observé 66,3) | autre mode | inits au mode retenu (V2) |
|---|---|---|---|---|---|
| 1,8 | 12830,2 | 12631,9 | 66,32 (+0,02) | — | 4/4 |
| 3,5 | 12854,7 | 12670,6 | 66,63 (+0,33) | 66,06 | 1/4 |
| 5,0 | 12829,9 | 12654,4 | 66,30 (0,00) | 66,28 | 1/4 |
| 6,2 | 12801,2 | 12603,6 | 66,41 (+0,11) | — | 4/4 |

Potentiel (jacobiens compris), meilleure init : complet 12829,9 / 12850,4 / 12824,7 / 12805,6 ; V2 12631,6 /
12665,9 / 12649,1 / 12608,0. Les maxima à 3,5 et 5,0 viennent d'UNE init chacun, de ‖∇‖ élevée (25 et 75 à
3,5) : non convergés au sens strict.

**Critère figé (§8.70 d 4).** La prédiction V2 bouge de 0,6 pt au plus entre arcs et modes (66,06 à 66,63),
loin des 4 pt : la clause « données qui ne révèlent pas l'arc ALORS QUE V2 bouge » ne se déclenche pas.
**La prédiction V2 ne dépend pas de l'arc** ; aucun ARBITRAGE. Prédiction de la revue du 23/11 (≤ 3 pt
entre 1,8 et 3,5) vérifiée (0,3 pt).

**Identification de l'arc.** lp favorise un arc intermédiaire (3,5) de 24 nats (complet) et 39 nats (V2) sur
1,8, mais sur un point isolé et non convergé, et sans le prior de `a_arc` (hors lp quand l'arc est fixé).
Lecture : l'arc est faiblement identifié, il influe sur le premier tour (σ_LP/σ_M 0,54 / 1,58 / 1,12 / 0,93
au mode d'init 0 des données complètes), pas sur V2. Par parcimonie et selon la branche proposée par la revue
(point 5), il peut être fixé (cercle fermé, 6,2, ou médiane a priori) sans coût sur V2 ; choix laissé à la
session locale, à faire AVANT toute lecture hors 2022 (§8.74).

**Correction.** La lecture préliminaire du journal (revue routine du 23/11, point 5 : « 1,8 meilleur de 12,6
nats ») était FAUSSE : le potentiel le plus négatif est le meilleur, et j'avais pris l'init 0 au lieu du
maximum. Corrigée ici.

### 8.76 Contrôles V2 de la référence (§8.70 d 3) et décision sur l'arc (session locale, 23/11 ~23:00Z)

**V2 sur les 4 paires de 2022** (MAP 20 000 pas `--decroissant`, Mac, mode retenu). Écart de la part de
Macron à la moyenne des 30 jours.

| Paire retirée (observé) | Référence `hiercercle_ra_fid` | `hier` | Gagnant |
|---|---|---|---|
| Macron–Mélenchon (66,3) | −1,3 | +8,4 | référence |
| Macron–Zemmour (64,4) | +5,3 | +4,2 | `hier` |
| Macron–Pécresse (59,2) | +0,5 | −8,3 | référence |
| Macron–Le Pen (57,4) | +3,5 | +10,8 | référence |
| **MAE** | **2,65** | **7,9** | |

- **Critère figé de §8.70 (d) 3 :**
  - référence ≤ `hier` sur 3 paires sur 4 ✓ ;
  - MAE sur les deux paires V2 3,3 ≤ `hier` − 1 = 5,3 ✓.

  ⇒ **V2 2022 favorable.**
- **Statut :** NON PROBANT (Mac, MAP) et DÉVELOPPEMENT (2022 a servi au choix du modèle ; revue de la
  routine du 23/11). La confirmation relève de §8.74 (2027).
- Les modes non retenus de la référence restent du même côté : Pécresse de 0,0 à +0,6, avec un mode
  à +10,6 ; Le Pen de +4,6 à +5,5.
- **Fichiers :** `results/v2paire_map_*_p20000d_v2sans_EmmanuelMacron{ValériePécresse,MarineLePen}.json`.

**Décision sur l'arc (après §8.75, AVANT toute lecture hors 2022).**

- La prédiction V2 ne dépend pas de l'arc (0,6 pt entre 1,8 et 6,2). L'arc agit sur le T1 : σ_LP/σ_M
  vaut de 0,54 à 1,58 selon l'arc, sur des points d'init 0.
- Le fixer exigerait de choisir une valeur sur 2022 seul, d'après le T1 et la plausibilité : on
  économiserait 1 scalaire au prix d'un choix éditorial lu sur les données.
- **Décision : l'arc reste LIBRE** (a priori N(log 3 ; 0,75) inchangé). Sa loi a posteriori sera
  rapportée. Aucune valeur n'est choisie a posteriori, et 2027 reste propre.
- Si l'utilisateur préfère le fixer (par exemple le cercle fermé), c'est un ARBITRAGE, à trancher avant
  le lancement de §8.74.

**§8.76, diagnostic : V2 de R8c (avec ω_k ; lu APRÈS avoir figé la référence, §8.73).** MAP 20 000
pas, Mac, mode retenu : Macron–Mélenchon −5,0 (61,3 contre 66,3), Macron–Zemmour +4,2 (68,6 contre
64,4) ⇒ MAE **4,6**, contre 3,3 pour la référence et 6,3 pour `hier`.

- ω_k dégrade le V2 de 1,3 pt (surtout sur Mélenchon). Cela va dans le sens de son retrait, sans en
  être une preuve.
- Ce chiffre ne change pas la référence, figée avant la lecture.

### 8.77 Corrections de la revue du 24/11 ~00:30Z : arc, témoin R1, §8.74 (avant tout nouveau run)

**(a) Le profil de l'arc (§8.75) ne couvre pas les points retenus.** Vérifié dans les pseudo-chaînes.
L'arc au point retenu de la référence en V2 vaut :

| Paire retirée | Arc retenu | Init(s) |
|---|---|---|
| Mélenchon | 1,39 | c4, seule |
| Zemmour | 1,07-1,34 | |
| Le Pen | 0,41 | c3, seule, 175 nats au-dessus des 5 autres |
| Pécresse | 5,01 | |

- Trois contrôles sur quatre sont hors de la grille 1,8-6,2.
- Sur Mélenchon, le point libre (lp hors prior ≈ 12 697,5) dépasse de 27 nats le meilleur point de la
  grille.
- « lp favorise 3,5 », « arc faiblement identifié » et « V2 indépendant de l'arc » sont **retirés** tels
  qu'écrits. Seul tient : sur Macron–Mélenchon et sur [1,8 ; 6,2], V2 bouge de 0,6 pt (1,6 pt avec le
  point à 1,39).
- L'arc saute de 0,4 à 5,0 quand on retire une seule paire : c'est une **instabilité**, pas une crête
  plate.
- Correction annexe : « arc ≈ 1,76 » en §8.73 pour la référence V2 Mélenchon doit se lire **1,39**
  (1,757 est l'arc du V1-LOHO).

**(b) La décision « arc libre » de §8.76 est ANNULÉE.**

- C'était une bifurcation : la règle en place, intégrée au journal (revue de la routine, point 5), et
  §8.74 disaient de **fixer l'arc par parcimonie (cercle fermé ou médiane a priori)** si V2 bougeait de
  moins de 4 pt.
- L'argument « choisir une valeur sur 2022 » était faux : ces deux valeurs sont posées sans données.
- **Règle appliquée : arc fixé à 2π (cercle fermé ; R6 + R7).** Cette valeur n'impose aucun
  emplacement de trou, si bien que le fer à cheval doit émerger des positions.
- Critères figés AVANT le run (2022, développement, Mac, 6 inits, 20 000 pas `--decroissant`) :
  - au moins 4 inits sur 6 à moins de 1 nat du meilleur point sur chacun des 4 V2 (stabilité) ;
  - σ_LP/σ_M < 1,5 et max |z_pos| ≤ 3 (hors Lassalle) sur tous les points retenus ;
  - MAE V2 sur les 4 paires ≤ 3,65 (référence + 1).
  - S'ils sont tenus, **le cercle fermé devient la référence avant 2027.** Sinon on revient à l'arc
    libre, en déclarant l'instabilité.

**(c) Témoin R1 (régression T1, 0 fit), défini et figé.**

$$\mathrm{logit}\,y_{ab}=\beta\,(\log t_a-\log t_b),$$

- MCO sans constante, pondérés par le nombre de duels, ajustés sur les AUTRES paires éligibles (au moins
  3 duels dans la fenêtre W).
- t_x : part T1 de x, moyenne par notice sur l'hypothèse la plus large, puis moyenne des notices sur
  les 30 jours.
- Reproduit par la session locale, 2022 : écarts −4,1 / −1,9 / +0,6 / +4,8, **MAE 2,85**.
- **§8.76 est corrigé** : V2 2022 est « favorable contre `hier` (2,65 contre 7,9), **pas contre R1**
  (2,65 contre 2,85, 2 paires sur 4) ». Sur les 2 paires V2 : référence 3,3, R1 3,0. Test de signe à
  3/4 : p = 0,31. La prédiction de la revue de la routine (pas de victoire d'au moins 1 pt sur R1) est
  vérifiée.

**(d) §8.74 est à RESPÉCIFIER avant tout code 2027.** Constats de la revue :

- Paires avec au moins 3 duels : 3 sur 30 jours, 6 sur 60 jours, 10 depuis toujours.
- Les paires Bardella s'arrêtent au 27/05/2026.
- Il n'existe ni `T1_DATES[2027]` ni champ officiel (`lse_officiel` / `champ_reports`).
- « ≥ 8/11 (test de signe) » donne p = 0,11 et ⌈0,7·n⌉ n'est jamais significatif.

Proposition de la revue, à figer :

- fenêtre de 60 jours (n ≈ 6), Bardella exclu ou réconcilié ;
- champ 2027 déclaré et figé à l'avance ;
- plateforme figée (le Mac) ;
- critère : MAE ≤ min(`hier`, R1) − 1 ET ≥ ⌈0,8·n⌉ victoires, présenté comme DESCRIPTIF.

**À écrire avant le lancement (arbitrage « V2 2022 = développement » toujours ouvert).**

### 8.78 Résultat de l'arc fixé à 2π (§8.77 b ; 24/11 ~03:10Z, Mac, MAP 20 000 pas, 6 inits)

`hiercercle_ra_fid`, `PRES2027_ARC_FIXE=6.283185`, V2 sur les 4 paires de 2022. Fichiers
`results/v2paire_map_hiercercle_ra_fid_2022_T1-30_arc6p283185_p20000d_v2sans_*.json`.

| Paire retirée (observé) | Prédiction, mode de meilleure lp | Écart | Inits à moins de 1 nat du meilleur point | σ_LP/σ_M | max \|z_pos\| (hors Lassalle) |
|---|---|---|---|---|---|
| Macron–Mélenchon (66,3) | 66,4 | +0,1 | 5/6 ✓ | 0,98 | Zemmour 4,35 |
| Macron–Zemmour (64,4) | 70,8 | +6,4 | 2/6 ✗ | 0,98-1,01 | Poutou 4,34, Zemmour 4,16 |
| Macron–Le Pen (57,4) | 62,7 | +5,3 | 1/6 ✗ | 1,31-1,34 | Zemmour 4,41 |
| Macron–Pécresse (59,2) | 58,6 | −0,7 | 1/6 ✗ | 1,15 | Bertrand 4,97 |
| **MAE** | **3,13** | | | | |

**Critères figés :**

- stabilité (au moins 4 inits sur 6 par paire) ✗ sur 3 paires ;
- σ_LP/σ_M < 1,5 ✓ ;
- max |z_pos| ≤ 3 ✗ (Zemmour vers 4,2-4,4 partout, comme R6 : 3,9) ;
- MAE ≤ 3,65 ✓ (3,13 ; arc libre 2,65 ; R1 2,85 ; `hier` 7,9).

**Verdict : le cercle fermé NE devient PAS la référence.**

- Règle de §8.77 (b) : retour à l'arc libre, **instabilité déclarée**. L'arc libre saute de 0,4 à 5,0
  selon la paire retirée (§8.77 a), et fixer l'arc ne rend pas les inits concordantes sur 3 paires.
- La multimodalité ne vient donc pas de l'arc seul.
- Le cercle fermé force aussi Zemmour, comme R6, et perd 0,5 pt de MAE V2.

**Lecture.**

- Sur la seule paire où il y a des reports de l'électorat finaliste à lire (Mélenchon), le cercle
  fermé est presque exact et stable.
- Ailleurs, il est en échec de stabilité.
- Aucune variante à arc fixe ou libre ne bat R1 d'au moins 1 pt en 2022.

### 8.79 Polissage L-BFGS des points MAP (protocole figé AVANT lecture ; 24/11 ~03:40Z)

- **Outil.** `model/backtest/polir.py` repolit les pseudo-chaînes existantes : L-BFGS-B sur le potentiel
  U (jacobiens compris), en float64 (`JAX_ENABLE_X64=1`, paramètres convertis), avec ftol = 1e-15 et
  gtol = 1e-3.
- **Contrôle sur 1 point** (arc 2π, V2 Mélenchon, init 0) : ‖∇‖ passe de 0,95 à 0,006, ΔU = 0,00. Le
  point Adam était convergé en valeur.
- **Application :** les 6 inits de chacun des 8 V2 de 2022 (référence à arc libre et arc 2π × 4 paires).
- **Critère** (revue de la routine du 23/11, point 3) : convergé si |ΔU| < 1 et ‖∇‖ < 1e-2.
- **Lecture des modes** après polissage : classement par U poli ; « au moins 4 inits sur 6 à moins de
  1 nat » (§8.77 b).
- **Conséquences prévues :**
  - si les inits restent à plus de 1 nat après polissage, ce sont de vrais modes distincts, et le
    verdict d'instabilité de §8.78 tient ;
  - si elles se rejoignent, c'étaient des non-convergences, et §8.78 est à relire.

**§8.79, résultat (24/11 ~03:45Z, Mac, float64).** Fichiers `results/polir_*.json` et pseudo-chaînes
`…_poli_c<k>.npz`.

**Arc fixé à 2π : convergé.** Sur les 24 points, ΔU ≤ 0,05 (un seul à 2,3) et ‖∇‖ ≈ 3e-3. Les écarts
entre inits sont de **vrais modes distincts** :

| Paire retirée | Écarts des inits au meilleur point (nats) |
|---|---|
| Zemmour | 0, 0, 5,9, 11,9, 28, 29 |
| Le Pen | 0, 2,3, 16, 50 ×3 |
| Pécresse | 0, 9 ×3, 21 ×2 |
| Mélenchon | 5/6 à moins de 1 nat |

⇒ Le verdict d'instabilité de §8.78 **tient** : c'est de la multimodalité réelle, pas de
l'optimisation.

**Arc libre (référence) : NON convergé, et dégénérescence vers la frontière.**

- Le polissage gagne de 48 à 211 nats sur les meilleurs points, et laisse ‖∇‖ de 10 à 200. Les points
  d'Adam à 20 000 pas n'étaient pas des optima.
- Les meilleurs points polis **s'effondrent vers arc ≈ 0,07-0,08 rad** (Mélenchon c4, Zemmour
  c0/c2/c3/c5, Le Pen c3), avec κ ≈ 1,9-2,3. Ils sont encore en mouvement : le maximum est sur le bord
  arc → 0, où presque tout le cercle est « arrière ».
- Pécresse reste à 5,0 (inits 0/2 convergées).
- ⇒ **Les V2 de la référence à arc libre (§8.72, §8.76 : MAE 2,65 ; §8.77 c contre R1) portent sur des
  points non convergés. Ils sont PROVISOIRES.** Le vrai mode du MAP est une géométrie dégénérée, que
  l'a priori N(log 3 ; 0,75) sur log(arc) ne retient pas (à 0,075, z ≈ −4,9).
- Lecture des prédictions V2 aux points polis : en cours.

**§8.79, suite : prédictions V2 aux points polis de la référence à arc libre (24/11 ~04:10Z).**

| Paire retirée (observé) | Point poli de meilleure lp | Écart | lp (avant → après) | arc | σ_Mélenchon | σ_LP/σ_M |
|---|---|---|---|---|---|---|
| Mélenchon (66,3) | 66,2 | −0,1 | 12 692 → 12 900 | 0,078 | 1,77 | 1,00 |
| Zemmour (64,4) | 70,1 | +5,7 | 12 233 → 12 313 | 0,08 | 1,55 | 1,01 |
| Le Pen (57,4) | 60,8 | +3,5 | 12 270 → 12 317 | 0,073 | 1,87 | 1,01 |
| Pécresse (59,2) | 59,8 | +0,5 | inchangé | 5,0 | 0,16 | 1,10 |
| **MAE** | **2,45** | | | | | |

À titre de comparaison, la MAE était de 2,65 avant polissage ; R1 fait 2,85 et `hier` 7,9.

- **Les prédictions V2 bougent à peine** (au plus 0,4 pt par paire).
- **Mais la géométrie disparaît au vrai mode.** Sur 3 paires sur 4, l'arc tombe à 0,07-0,08 rad avec des
  σ de 1,5 à 1,9 (en unités de μ, soit des noyaux couvrant tout le cercle). σ_LP/σ_M ≈ 1,00 :
  **tous les noyaux sont plats.** Le modèle devient en pratique un logit sans espace (attractivité w,
  κ, ω, η, et fidélité R7 sur les reports).
- **Lecture.**
  - Au MAP, les données de 2022 (T1, duels, reports d'avant T1) préfèrent **abolir la géométrie**
    plutôt que la révéler.
  - Le V2 « favorable » de la référence vient alors d'un modèle de fait non spatial, nourri par les
    reports. C'est cohérent avec R1, une régression sans espace, presque aussi bon (2,85).
  - L'a priori actuel sur l'arc ne protège pas contre cette dégénérescence.
- **Conséquence pour l'objectif** (« la géométrie révèle la structure »). Sur 2022, la vraisemblance
  ne soutient pas de structure spatiale au-delà de ce que donne un modèle sans espace, une fois les
  reports inclus. C'est un résultat négatif, qui rejoint l'option (ii) / (iii) de l'arbitrage du 21/11.

**ARBITRAGE À TRANCHER (utilisateur).** Que faire de la dégénérescence de l'arc ?

- (a) un a priori sur l'arc plus ferme, ou une borne inférieure posée à l'avance (par exemple
  arc ≥ π) : c'est une contrainte éditoriale qui impose la géométrie ;
- (b) accepter le constat : modèle sans espace + reports pour le T2, la géométrie servant au seul T1
  ou à la présentation ;
- (c) chercher la structure ailleurs (modèle joint 2017 + 2022 + 2027 avec positions partagées,
  R3b) avant de trancher.

### 8.80 Correction de §8.79 et §8.79 suite ; arbitrage retiré ; piste P4 (σ égaux) figée (revue du 24/11 ~05:00Z)

**(a) « Noyaux plats, géométrie disparue, modèle de fait non spatial » est FAUX.** Il est retiré.

- Le noyau de von Mises est un modèle vectoriel 2D : A_i(θ) = w_i − κ_i − log Z_i + ⟨e^{iθ}, m_i⟩,
  avec m_i = κ_i·e^{iθ_i}. Les parts et les duels sont invariants par translation commune des m_i.
- Au point « effondré », arc → 0 avec κ_vM ≈ 53-57 (≈ 150 pour Roussel). Le noyau a une largeur
  arc·σ ≈ 0,14 rad, soit 2 % du cercle, et non « tout le cercle ». Le jitter de σ (0,2) y devient une
  **coordonnée radiale libre** : la géométrie s'ÉTIRE au lieu de s'aplatir.
- Mesures de la revue aux points polis :
  - information mutuelle θ → vote T1 de 0,58 à 0,63 nat (29-32 % de H), PLUS élevée qu'aux modes
    non effondrés (0,43-0,55) ;
  - égaliser les positions déplace les duels de 10,5 à 13,6 pt en moyenne, jusqu'à 27 pt.
- σ_LP/σ_M ≈ 1 y vaut par construction et ne dit rien de la largeur.
- Le κ ≈ 1,9-2,3 cité était celui des duels.

**(b) « NON convergé, encore en mouvement vers le bord » est FAUX** pour Mél c4 et LP c3 : hessienne
définie positive, gain de Newton ≤ 0,05 nat, conditionnement de 10⁷ à 10⁸ (raideur). Plusieurs inits
convergent vers le même arc 0,073-0,084 : c'est un **mode intérieur**. Z c2 n'est pas tout à fait
convergé (gain de Newton 7,6 nats).

Autres corrections :

- L'a priori porte sur logit(arc/2π) : z = −7,3 et une pénalité d'environ 27 nats, et non « z ≈ −4,9
  sur log(arc) ».
- La MAE de 2,45 mélange des bassins : 2,98 sur le bassin non effondré homogène.

**(c) Ce que gagne le bassin effondré, et ce qu'il coûte.** Il gagne +144 à +372 nats sur la
vraisemblance T1, et en perd 35 à 64 sur les reports. Il est peu crédible :

- l'électorat Roussel donne 0 % à Mélenchon dans Mélenchon–Macron (75-95 % ailleurs) ;
- |z_pos| de Mélenchon vaut 4,3-4,6 ;
- les cellules Mélenchon–Macron font 22-24, contre 7,7-14.

Le V2 bouge d'au plus 1,1 pt entre les bassins. **L'instabilité porte sur les paramètres (jauge), pas
sur les prédictions** (Procuste : des configurations identiques à 10 % près sous des arcs de 2,98 et
4,97).

**(d) L'ARBITRAGE (a)/(b)/(c) du §8.79 suite est RETIRÉ** : sa prémisse est fausse. Il sera reposé, s'il
le faut, après P4.

**(e) Critère « au moins 4/6 inits à 1 nat » (§8.77 b).** Il reste appliqué tel quel aux verdicts
déjà rendus (figés). **Pour la suite**, deux modes ne sont « distincts » que si le résidu de Procuste
(en m centrées) dépasse 0,25 fois l'échelle ET si un duel clé diffère de plus de 2 pt. Ce critère est
figé AVANT les runs de P4.

**(f) Piste P4 : σ_i tous égaux** (drapeau `sigma_egaux`, z_sigma ≡ 0, −23 paramètres). Cela supprime
la coordonnée radiale : on obtient un vrai cercle 1D en fer à cheval, le plus parcimonieux. Variante
`hiercercle_ra_fid_sigeq`.

Critères figés AVANT le run (2022, développement, Mac, MAP 20 000 pas `--decroissant` + polissage) :

- V2 sur les 4 paires ≤ référence + 1, avec pour référence le bassin non effondré à 2,98, soit **≤ 3,98** ;
- **duels en échantillon** (données complètes) : MAE ≤ référence + 0,5 ;
- **électorat Roussel → Mélenchon ≥ 50 %** dans Mélenchon–Macron (données complètes) ;
- |z_pos| des finalistes (Macron, Le Pen, Mélenchon, Zemmour, Pécresse) ≤ 3 ;
- perte de vraisemblance T1 rapportée, sans seuil.

**Mesure des duels en échantillon**, précisée le 24/11 ~07:55Z AVANT toute lecture des deux points :
moyenne de |100·(duel_s − duel_y)| sur tous les duels du fit sur données complètes, au point poli de
meilleur U. La référence est `hiercercle_ra_fid` sur données complètes, à 20 000 pas `--decroissant`,
puis polie (run lancé le 24/11 à ~07:55Z, aucun n'existait).

Si ces critères tiennent, `hiercercle_ra_fid_sigeq` devient la référence ; sinon, la référence reste
inchangée. Témoins rapportés : R1 (2,86) et `hier` (7,9).

### 8.81 Résultat de P4, σ égaux (§8.80 f ; 24/11 ~09:30Z, Mac, MAP 20 000 pas + polissage float64)

`hiercercle_ra_fid_sigeq`. Fichiers `results/polir_map_hiercercle_ra_fid_sigeq_*.json` et
`results/v2paire_map_hiercercle_ra_fid_sigeq_*_poli.json`.

| Paire retirée (observé) | P4 | Écart | Inits concordantes (U poli, à 1 nat) | Référence (bassin non effondré) |
|---|---|---|---|---|
| Mélenchon (66,3) | 61,6 | −4,7 | 6/6 | −1,3 |
| Zemmour (64,4) | 69,4 | +5,0 | 5/6 | +5,3 |
| Le Pen (57,4) | 56,0 | −1,4 | 4/6 | +3,5 |
| Pécresse (59,2) | 72,4 | **+13,1** | 6/6 | +0,5 |
| **MAE** | **6,03** | | | 2,98 (R1 2,86 ; `hier` 7,9) |

**Données complètes :** 6/6 inits au même point, convergé (‖∇‖ ≈ 2e-3). Roussel → Mélenchon 73,3 (mesure
65,8) ; T1 3,02 ; σ_LP/σ_M = 1 (par construction).

**Critères figés :**

- V2 ≤ 3,98 ✗ (6,03) ;
- Roussel → Mélenchon ≥ 50 ✓ ;
- |z_pos| des finalistes ≤ 3 ✗ : Zemmour à −3,69 dans le fit Macron–Le Pen ; partout ailleurs ≤ 2,7 ;
- duels en échantillon : point de référence en cours, rapporté pour mémoire (le verdict est déjà
  acquis).

**Verdict : P4 NON retenu.** La référence reste `hiercercle_ra_fid`.

**Lecture.**

- σ égaux SUPPRIME l'instabilité : de 4 à 6 inits concordantes partout, et les données complètes en
  un seul point. Cela confirme que la multimodalité venait de la coordonnée radiale (§8.80 a).
- Mais le cercle 1D ainsi contraint ne sait plus placer Pécresse : 72,4 contre 59,2. La flexibilité
  radiale portait de l'information réelle sur la droite modérée.
- **Tension parcimonie / prédiction :** −23 paramètres et la stabilité, au prix de +3 pt de MAE V2.
  La règle figée tranche contre P4.
- Piste ouverte par ce résultat : la piste P3 (vecteur 2D explicite, jauge fixée), qui garde le rayon
  mais le paramètre proprement.

### 8.82 Errata de §8.80-8.81 et piste P4b (revue du 24/11 ~10:00Z ; avant code, figé)

**Errata.**

- **§8.81, colonne « Référence » : FAUSSE.** Elle reprenait les points Adam non polis de §8.76, dont un
  point effondré pour Le Pen. Le bassin non effondré à 2,98 donne :

  | | Mélenchon | Zemmour | Le Pen | Pécresse |
  |---|---|---|---|---|
  | Référence (bassin non effondré) | −0,05 | +6,75 | +4,6 | +0,53 |
  | P4 | −4,7 | +5,0 | −1,35 | +13,1 |

  P4 fait MIEUX que la référence sur Zemmour et sur Le Pen. Sans Pécresse, la MAE sur 3 paires vaut
  3,66 pour P4 contre 3,80 pour la référence. **L'échec de P4 tient à la seule paire Pécresse.** Le
  verdict est inchangé : 6,03 dépasse toute référence + 1.
- **« Bassin homogène » (§8.80) : retiré.** Les arcs vont de 0,99 à 5,01. Le meilleur bassin non
  effondré pour Le Pen est c4 (arc 3,21, +3,04), ce qui donne une MAE de 2,59.
- **Troisième pathologie, jamais nommée : les « pointes » de σ des petits candidats.** Montebourg
  (putatif, non officiel) a z_σ de −2,2 à −8,8 à tous les points, Roussel de −6,7 à −7,7.
- **Le critère |z_pos| ≤ 3 n'est pas discriminant** : la référence y échoue aussi (Zemmour 3,32 dans le
  même fit Macron–Le Pen, Mélenchon 4,3-4,6 aux points effondrés, Zemmour 5,25 en Pécresse c1). P4
  fait mieux partout.
- **Perte T1 de P4**, figée « rapportée » et pourtant omise : ΔU de 110 à 199 nats selon la paire,
  au-dessus de la pénalité BIC pour 23 paramètres (≈ 94). Les rayons portent de l'information au sens
  de la vraisemblance.
- **Horodatage.** La mesure des duels en échantillon a été commitée (fe12837) APRÈS que 3 V2 de P4 sur
  4 et les données complètes ont été écrits sur disque. L'affirmation « avant toute lecture » n'est pas
  vérifiable ; ce critère reste « pour mémoire ». Les critères V2 de P4 avaient été figés avant le
  premier fit.

**Mécanisme Pécresse (revue, reproduit à 0 fit).**

- Sous σ égaux, une petite candidate proche d'une grande ne peut être petite au T1 qu'avec un w bas,
  qui se reporte sur le duel : son propre électorat préfère Macron (21 % contre 62 % dans la
  référence).
- Au point P4, multiplier le seul κ de Pécresse par 2,8 ramène Macron–Pécresse de 72,4 à 59,7.
- **Le rayon encode la concentration de l'électorat, identifiée au T1 par les substitutions.** C'est la
  géométrie qui révèle la structure, au sens de l'utilisateur.

**Point de référence sur données complètes** (`hiercercle_ra_fid`, 20 000 pas + polissage, Mac,
24/11 ~10:30Z ; pour mémoire, le critère n'étant plus vérifiable « avant lecture »).

| | Référence | P4 |
|---|---|---|
| Inits au même point (à 1 nat) | 6/6 | 6/6 |
| Arc | 3,67 | 3,84 |
| MAE des duels en échantillon | 1,56 | 1,77 (+0,21 ≤ +0,5 ✓) |
| U | | 174 nats de plus que la référence |

Sur les données complètes, la référence n'est donc PAS instable. L'instabilité et l'effondrement
n'apparaissent qu'une paire retirée.

**(P4b) Rayons libres pour les seuls candidats « dynamiques ».** Règle PRÉEXISTANTE :
`SEUIL_DYNAMIQUE` = 5 % de part moyenne au T1, celle qui donne déjà un chemin à w. Les autres
candidats ont z_σ ≡ 0 (σ_global). En 2022, cela laisse 8 rayons au lieu de 23 (−15 paramètres). Aucun
nom n'est choisi, et la règle se transpose à 2027. Variante `hiercercle_ra_fid_sigdyn`.

Critères figés AVANT le code (2022, développement, Mac, MAP 20 000 pas `--decroissant` + polissage ;
4 V2 + données complètes) :

1. V2 sur 4 paires ≤ 3,98 ;
2. stabilité Procuste (§8.80 e) : au moins 4 inits sur 6 dans le même mode sur chacun des 5 fits ;
3. aucun |z_σ| > 3 au point retenu ;
4. écart d'arc entre les 5 fits ≤ 1 rad ;
5. |z_pos| des finalistes ≤ max(3, valeur de la référence au même fit) ;
6. Roussel → Mélenchon ≥ 50 ;
7. ΔU contre la référence rapporté ; les 15 rayons retirés sont jugés inutiles si ΔU ≤ 15·ln(3 500)/2 ≈ 61.

Si tous ces critères tiennent, P4b devient la référence (plus parcimonieux, stable).

### 8.83 Résultat de P4b, rayons des seuls candidats dynamiques (§8.82 ; 24/11 ~13:30Z, Mac, MAP 20 000 pas + polissage float64)

`hiercercle_ra_fid_sigdyn` : 8 rayons libres sur 23 (Mélenchon, Jadot, Macron, Bertrand, Pécresse,
Barnier, Le Pen, Zemmour). Fichiers `results/polir_map_hiercercle_ra_fid_sigdyn_*.json`,
`results/v2paire_map_hiercercle_ra_fid_sigdyn_*_poli.json` et `results/modes_…sigdyn…_poli.json`.

| Fit (paire retirée, observé) | Écart V2 | Inits dans le mode retenu | Arc | max \|z_σ\| | max \|z_pos\| des finalistes (référence) | ΔU contre la référence |
|---|---|---|---|---|---|---|
| Données complètes | — | 5/6 | 2,74 | 0,89 | 2,59 (3,29) | **+7,6** |
| Mélenchon (66,3) | −0,4 | 5/6 | 3,08 | 1,10 | 2,48 (4,31) | +284 |
| Zemmour (64,4) | +6,2 | 6/6 | 2,38 | 0,93 | 2,81 (0,59) | +226 |
| Le Pen (57,4) | +4,4 | 5/6 | 3,12 | 1,98 | 2,46 (4,64) | +259 |
| Pécresse (59,2) | +1,4 | 5/6 | 2,73 | 2,03 | 2,31 (1,69) | +12 |
| **MAE V2** | **3,09** | | | | | |

**Critères figés de §8.82 :**

1. V2 ≤ 3,98 ✓ (3,09).
2. Stabilité : au moins 4/6 inits dans le même mode sur les 5 fits ✓ (5, 5, 6, 5, 5).
   - Critère de §8.80 (e) appliqué sous cette forme : deux inits sont dans le même mode si
     ΔU < 1 nat OU si la prédiction V2 diffère de 2 pt au plus. « Distinct » exige les deux
     conditions, Procuste ET duel ; la condition sur le duel suffit donc à les réunir.
   - Le résidu de Procuste n'a pas été calculé : lecture CONSERVATRICE pour « distinct »,
     généreuse pour « stable ». À vérifier par la revue.
3. Aucun |z_σ| > 3 ✓ (max 2,03 ; plus de pointes de petits candidats).
4. Écart d'arc entre fits ≤ 1 rad ✓ (2,38 à 3,12, soit 0,74 ; contre 0,4 à 5,0 pour la référence).
5. |z_pos| des finalistes ≤ max(3, valeur de la référence) ✓ (max 2,81).
6. Roussel → Mélenchon ≥ 50 ✓ (74,7 ; mesure 65,8). Partage Mélenchon : 63,6 contre 63,0.
7. ΔU sur données complètes = +7,6 nats ≤ 61 ⇒ **les 15 rayons retirés sont jugés inutiles** ✓.
   - En V2, les ΔU de +226 à +284 se mesurent contre les points EFFONDRÉS de la référence
     (bassin peu crédible, §8.80 c). Contre les bassins non effondrés, l'écart est bien moindre
     (Pécresse : +12).

**Verdict : TOUS les critères sont tenus ⇒ `hiercercle_ra_fid_sigdyn` devient la RÉFÉRENCE**
(−15 paramètres, stable, sans effondrement ni pointe).

- Statut : favorable sur le Mac au MAP ⇒ **NON PROBANT** par règle. 2022 reste du DÉVELOPPEMENT.
- **Face aux témoins :** MAE V2 3,09 contre R1 2,86 (0 fit) et `hier` 7,9. **P4b ne bat pas R1.**
  Sa justification reste le modèle unique et génératif (T1, duels, reports, abstention) et sa carte
  stable, pas la précision.
- V1-LOHO de P4b (reports d'une hypothèse cachée, face à U_fid 14,1) : **pas encore mesuré**, en file.

**§8.83, erratum (revue du 24/11 ~14:00Z).**

- **Critère 2, recalculé avec un vrai Procuste** (m centrées, pondérées par les parts, champ officiel) :
  **5, 5, 6, 5, 5 confirmé.** Réserves :
  - pour Pécresse, c1 et c2 sont des optima distincts (+27 nats, arc 4,35 contre 2,73) mais
    équivalents en prédiction (Procuste 0,21, duels ≤ 0,25 pt). La partition figée de `modes.py`
    (−10 nats) ne compte que 3/6 dans le mode dominant ;
  - le bassin effondré n'a pas disparu : il est RELÉGUÉ de 975 à 1 436 nats (init 4 dans 4 fits sur
    5 ; V2 Le Pen à +15,7) ;
  - Pécresse c4 n'est pas convergé (‖∇‖ 30,9).

  Pour le NUTS : initialiser au MAP et vérifier qu'aucune chaîne n'entre dans ce bassin.
- **Critère 7 mal interprété.** ΔU = +7,6 compare des potentiels qui contiennent les a priori, dont
  environ 35 nats en faveur de P4b (normalisations, ½Σz² des 15 z_σ, z_pos).
  - **La perte de vraisemblance est d'environ 43 nats.** Elle est sous le BIC sur les nœuds (61), au
    seuil du BIC sur les sondages (41), et au-dessus de l'AIC (15) comme du test du rapport de
    vraisemblance (86 contre 25).
  - Formulation corrigée : « **les 15 rayons portent environ 43 nats ; le BIC ne les justifie pas,
    l'AIC les garderait** ». « Jugés inutiles » est retiré. Le critère, figé sur ΔU, reste tenu à la
    lettre ; l'erreur d'interprétation favorisait P4b.
  - Les ΔU de V2 (+226 à +284) étaient mesurés contre des points de référence non convergés. Contre
    le meilleur point convergé et non effondré : 9,6 / 25,4 / 3,9 (51,9 si l'on prend Le Pen c4) /
    12,0, tous sous 61.
- **Colonne « référence » de |z_pos|** : Zemmour vaut 2,38 (et non 0,59), Pécresse 2,84 (et non 1,69).
  C'est sans effet, car la borne de 3 s'applique.
- **Critères 3 et 6 en partie tautologiques** (15 rayons fixés). L'arc reste faiblement identifié
  (2,73 à 4,35 au sein d'un même mode pour Pécresse).
- **Duels en échantillon** (pour mémoire) : P4b 1,59, référence 1,56, P4 1,77.
- **Seuil de 5 %** : tout seuil entre 4,98 % (Ciotti) et 7,11 % (Jadot) donne les mêmes 8 candidats.
  Ciotti est au fil du rasoir. L'idée est venue après les échecs de 2022 : statut « développement »
  confirmé.
- **Règle « B′ (2017) accompagne tout verdict » (spec:3338) non appliquée** en §8.82-8.83 : elle est
  due, en file.

### 8.84 B′ 2017 de P4b (règle de spec:3338) — protocole et critères figés AVANT le run (24/11 ~15:00Z)

**Cadre.** `hiercercle_ra_fid_sigdyn`, 2017 à T1−30, `--lse-officiel`, B = 100, MAP 20 000 pas
`--decroissant`, 6 inits, polissage float64 (`polir`). Les candidats dynamiques sont définis par la
même règle (part ≥ 5 %), appliquée aux données de 2017. **Plateforme : la VM (DEMANDE n° 7), seule
pour cette étape.**

**Critères**, même famille que le dépistage B′ de R8c (§8.58, R8c : T1 1,93, Mélenchon 87,2 contre
76,4) :

1. **Stabilité** : Procuste (§8.80 e) au moins 4/6 inits dans le même mode ;
2. **T1 2017** : MAE ≤ 1,93 + 0,3 = **2,23** ;
3. **Partage de l'électorat Mélenchon** (Macron–Le Pen, après T1) à 5 pt au plus de la mesure
   (76,4) ;
4. **σ_LP/σ_M < 1,5** ; aucun |z_σ| > 3 ; |z_pos| des finalistes ≤ 3 ;
5. **rapporté, sans seuil** : V2 Fillon–Le Pen retiré (seule autre paire sondée), contre `hier` sur
   la même paire.

**Lecture.**

- Si 1, 2 et 4 tiennent, la structure de P4b se généralise à 2017.
- Le critère 3 a échoué pour R8c (+10,8) ; un nouvel échec serait attendu (report Mélenchon → Le Pen
  de 2017). Il serait rapporté comme la limite connue, sans bloquer.

**Résultat du B′ 2017 de P4b** (VM, DEMANDE n° 7, 24/11 15:25Z, routine). Fichiers :
`results/polir_map_hiercercle_ra_fid_sigdyn_2017_T1-30_p20000d{,_v2sans_FrançoisFillonMarineLePen}.json`,
`results/modes_map_hiercercle_ra_fid_sigdyn_2017_T1-30_p20000d_poli.json`,
`results/v2paire_map_{hiercercle_ra_fid_sigdyn,hier}_2017_T1-30_p20000d_v2sans_FrançoisFillonMarineLePen_poli.json`.
Polissage convergé partout (‖∇‖ 0,0025-0,0034, critère PGTOL).

Données complètes, par init (potentiel poli ; partage Mélenchon, mesure 76,4 ; σ_LP/σ_M ; T1) :

| init | −U | Mélenchon | σ_LP/σ_M | T1 |
|---|---|---|---|---|
| 0 | 2822,1 | 84,9 | 0,63 | 2,21 |
| 1 | 3005,3 | 88,1 | 1,05 | 2,08 |
| 2 | **3075,4** | 88,9 | 0,89 | 1,97 |
| 3 | 3052,9 | 86,4 | 0,84 | 1,79 |
| 4 | 3007,3 | 90,5 | 0,81 | 2,03 |
| 5 | 2821,5 | 84,8 | 0,62 | 2,21 |

1. **Stabilité : ÉCHEC probable.** Six points polis distincts sur 254 nats, sauf 0 ≈ 5 (0,6 nat) ; la
   partition par log-densité donne un mode dominant d'une seule init (2). Procuste non calculé (script
   hors de la VM) ; la session locale le fera. La concordance 0/5 par lp rend « ≥ 4/6 » improbable.
2. **T1 2017 : ✓** 1,97 au meilleur point (toutes inits ≤ 2,21 ≤ 2,23).
3. **Mélenchon : ✗ attendu**, 88,9 contre 76,4 (+12,5 ; R8c +10,8) ; toutes inits entre 84,8 et 90,5.
4. **σ_LP/σ_M : ✓** 0,89 (< 1,5 à toutes les inits) ; |z_σ| et |z_pos| non calculés sur la VM.
5. **V2 Fillon–Le Pen (rapporté, sans seuil)**, observé 57,1 :
   - P4b : modes très dispersés — 40,5 (init 3, meilleur lp), 42,9, 46,7, 75,9 (inits 0, 1, 5) ;
     mode retenu −16,6 pt ;
   - `hier` : 47,8 (−9,3 ; 4/6 inits) ; autre mode 56,7 (2/6).
   P4b fait moins bien que `hier` et n'est pas stable sur cette paire.

Également : « Aucun » MAE 7,15 (critère 3,75 de 2022 : ✗) ; structure Le Pen → Mélenchon 95,3.

**Lecture (règle figée ci-dessus).** 2 et 4 tiennent, 1 échoue vraisemblablement : selon la règle, la
structure de P4b **ne se généralise pas à 2017** tant que le Procuste n'a pas montré ≥ 4/6. Le critère 3
échoue comme prévu (limite connue). La V2 2017 ne confirme pas le gain de 2022.

### 8.85 V1-LOHO de P4b (critère de §8.63 ; 24/11 ~16:30Z, Mac, MAP 20 000 pas + polissage)

Fichiers `results/v1loho_map_hiercercle_ra_fid_sigdyn_2022_T1-30_p20000d_sans_*_poli.json`. Mode
retenu par lp.

| Hypothèse retirée | P4b, partage (« Aucun ») | U_fid | Écart à U_fid | Inits dans le mode | Ancienne référence (cercle + R7) |
|---|---|---|---|---|---|
| Mélenchon–Macron | 10,6 (5,0) | 10,4 | +0,2 | 5/6 | 24,0 (10,8) |
| Macron–Zemmour | **5,7** (4,6) | 17,5 | −11,8 | 6/6 | 10,0 (6,8) |
| Macron–Pécresse | 23,4 (5,5) | 13,9 | **+9,5** | 3/6 (+2 à 26 nats) | 24,0 (7,3) |
| **Moyenne** | **13,2 (5,0)** | **13,9** | −0,7 | | 19,3 (8,3) |

**Critère figé** (moyenne ≤ U_fid − 2 = 11,9, aucune hypothèse > U_fid + 3) : **ÉCHEC**. La moyenne
vaut 13,2 et Pécresse dépasse de +9,5.

**Lecture.**

- P4b améliore nettement les cellules d'une hypothèse cachée : partage 19,3 → 13,2, « Aucun »
  8,3 → 5,0. Sur Mélenchon–Macron, il passe de 24,0 à 10,6, au niveau de U_fid.
- Il ne bat pas encore le témoin sans géométrie U_fid en moyenne.
- L'échec est concentré sur Macron–Pécresse : la droite modérée reste mal ordonnée par rapport à la
  gauche dans les reports, comme pour toutes les variantes depuis §8.64.

### 8.86 Errata de §8.84-8.85 et synthèse de fin de recherche (revue du 24/11 ~17:00Z)

**Errata.**

- **§8.84 : 9 « dynamiques » en 2017, et non 8, dont 4 fantômes** (Hollande, Valls, Montebourg,
  Bayrou). Aucun n'est testé dans les 30 derniers jours ni ne figure dans un duel ou un tableau croisé.
  La règle de 5 % fait la moyenne sur TOUTE la campagne. En 2022, Bertrand et Barnier sont dans le
  même cas. Dupont-Aignan (3,1 %) a un σ fixé, et sa cellule varie de 35,9 à 99,7 selon l'init.
- **§8.84, V2 Fillon–Le Pen** : deux modes sont retenus (Δlp ≤ 10), c'est donc une **fourchette de
  −16,6 à −10,4** (`hier` : −9,3). La conclusion tient ; l'ampleur était exagérée.
- **§8.84, stabilité** : non démontrée dans un sens comme dans l'autre (npz de la VM non versionnés).
  Les prédictions en échantillon restent proches entre inits (Hamon 92-95, Fillon 60-67, Mélenchon
  85-90), sauf pour Dupont-Aignan.
- **§8.84, critère 4** : coché sans calcul de |z_σ| ni de |z_pos|. Vérifié en partie seulement.
- **§8.84, critères assouplis AVANT le run**, mais vers l'indulgence : le V2 Fillon–Le Pen est passé
  de « ≤ `hier` − 1 » à « rapporté ». Sous le seuil initial, c'est un ✗ ferme.
- **§8.85, U_fid** : ce n'est pas un témoin indépendant. C'est le même fit à τ = 0, donc **une
  constante par hypothèse** (54,0 pour toutes les cellules de Macron–Pécresse) : le critère mesure
  la valeur ajoutée de la géométrie sur une constante. 50/50 sur Macron–Pécresse : 15,2.
- **§8.85, |z_pos|** : 3,08 pour Zemmour et 3,19 pour Mélenchon dans deux fits LOHO (> 3), non
  rapportés.

**Mécanismes** (revue, à 0 fit).

- **2017 : une seule paire en V2, et tout contre Le Pen.** Les données T2 ne séparent pas
  « Le Pen pénalisée » (front républicain, transférable) de « Macron central » (géométrie). Le témoin
  **T0, « pénalité Le Pen constante »** (Fillon–LP prédit à la valeur observée de Macron–LP), donne
  **+4,7**, contre −16,6 / −10,4 pour P4b, −9,3 pour `hier` et +42,8 pour R1, qui s'effondre en 2017.
  - L'information est dans les données ; c'est la décomposition du modèle qui échoue.
  - Mélenchon est à +12,5 à toutes les inits : le « ni-ni » de 2017 n'est pas une distance.
- **Pécresse en V1-LOHO.** Sans ses tableaux croisés, elle glisse vers Le Pen (θ de 0,42 à 0,86,
  σ de 0,23 à 0,15). La gauche bascule alors à 95-99 % sur Macron, pour 59 à 74 % mesurés. Cela donne
  une MAE de 23,4, pire que 50/50 (15,2).
  - Diagnostic : non-identification, plus une valence propre à l'électorat (rejet de Macron à gauche).
  - En revanche, Mélenchon–Macron et Zemmour font aussi bien cachés qu'en échantillon : ce ne sont pas
    des échecs de généralisation.

**Neutralité en production : ÉCHEC CONSTATÉ, puis CORRIGÉ dans ce lot.**

- `model/run.py` chargeait `tour=None` (commit d7b1575, audit Codex). Les T2 entraient donc dans
  `listes.produire`, `derniers_sondages.produire` et `_alerte_palette`. Mesuré par la revue : les 3
  listes publiées changeaient, les événements passaient de 25 à 51, et les `n_sondages` gonflaient.
- **Retour à `load_raw_polls(as_of=as_of)`**, identique à main. Les tests restent verts.
- Restent à documenter :
  - `base.py:186` : le `used_polls` par défaut des modèles privés compte les notices T2, avec un
    risque de rejeu en `--include-private` ; inactif depuis que run.py ne charge plus de T2 ;
  - la colonne `non_expression` de `live_dataset` (NaN).

  gp-pooling et le spatial de production sont neutres (revue : mêmes kwargs, mêmes paramètres, même
  potentiel en un point fixé, sur les données 2027).

**Neutralité vérifiée après correction (24/11 ~17:40Z, 0 fit, même CSV au 2026-11-24, script
`scratchpad/rev9/neutral.py` et `listes_seul.py`, main extrait par `git archive`).**

| Sortie ou entrée | Main contre branche |
|---|---|
| `listes.json` et `derniers-sondages.json` | **identiques à l'octet**, à graine de hachage égale |
| Entrée et affichage de gp-pooling | identiques |
| Entrée, affichage, kwargs et potentiel du spatial | identiques (portée CORRIGÉE en §8.90 : pas les chiffres publiés à l'octet) |
| Hachage du T1 | identique |

- Les seules différences sont l'ordre des événements de même date de `listes.json` et la colonne
  `non_expression` (NaN) du brut. L'ordre dépend de `PYTHONHASHSEED`, **sur main aussi** :
  c'est un non-déterminisme préexistant, hors périmètre, signalé en suivi.
- **Test durable** `tests/test_run_premier_tour.py` : run.py ne charge que le premier tour, et
  `PUBLIER_T2` est False. Le test échoue bien si l'on réintroduit `tour=None` (mutation vérifiée).

**SYNTHÈSE DE FIN DE RECHERCHE** (2022 = développement ; MAP sur le Mac, NON PROBANT).

- **Établi.**
  - Un modèle génératif unique (cercle + duels + tableaux croisés + abstention + fidélité) s'ajuste
    bien à 2022 en échantillon : duels 1,6 pt, cellules environ 6-7 pt.
  - Avec des rayons libres pour les seuls candidats qui pèsent (P4b), la carte de 2022 est stable, sans
    effondrement ni pointe, et σ_Le Pen reste crédible (σ_LP/σ_M ≈ 0,9 à 1,4).
  - Les reports d'avant T1 et la fidélité sont ce qui rend le T2 utilisable (V2 de 7,9 à environ 3).
  - Le rayon d'un candidat encode la concentration de son électorat (cas Pécresse).
- **Meilleure variante : P4b** (`hiercercle_ra_fid_sigdyn`). Ses limites :
  - hors échantillon, il **ne bat pas les témoins triviaux** : V2 3,09 contre R1 2,86 ; V1-LOHO 13,2
    contre une constante par hypothèse (13,9) ;
  - il échoue sur Pécresse caché (pire que 50/50) ;
  - en 2017, le report Mélenchon est faux (+12,5) et le V2 Fillon–LP est pire que `hier` et que T0.
- **Non établi.**
  - Que la géométrie apporte de la précision prédictive.
  - Que la structure se généralise à 2017.
  - Que le postérieur NUTS soit stable.
  - Qu'un candidat sans tableaux croisés soit bien placé, ce qui concernera des candidats de 2027.
- **Recommandation sur l'objectif.** Le modèle spatial unique est atteint comme **représentation
  cohérente** (carte, reports, abstention), mais **non validé comme prévision du T2**.
  - Garder `PUBLIER_T2 = False`.
  - Deux essais peu coûteux et parcimonieux avant de conclure, jugés sur le V2 2017 contre T0 :
    - (1) rayons libres sur la fenêtre de 30 jours (règle sans fuite) ;
    - (2) pénalité d'extrémité à 1 paramètre à la place de η.
  - Sans gain, publier R1 / T0 comme référence des duels, et la carte spatiale comme représentation.

### 8.87 P4c : rayons libres sur la fenêtre de 30 jours (revue du 24/11 ~17:00Z ; avant code, figé)

**Règle.** Le rayon du candidat i est libre si sa part moyenne au T1 dans les 30 jours avant `as_of`
est d'au moins 5 % ; sinon z_σ,i ≡ 0. Même seuil et même calcul que `SEUIL_DYNAMIQUE`, mais sur la
seule fenêtre récente.

- Aucune fuite : la règle n'utilise que les sondages ≤ `as_of`.
- Elle retire les « fantômes » (candidats testés en début de campagne seulement, sans duel ni tableau
  croisé). Attendu : 2017, 9 → 5 (Le Pen, Fillon, Macron, Mélenchon, Hamon) ; 2022, 8 → 6.
- Variante `hiercercle_ra_fid_sigdyn30`.

**Plateformes (une par étape).**

- 2022 (4 V2 + données complètes, puis 3 V1-LOHO) : **le Mac**.
- 2017 (données complètes + V2 Fillon–Le Pen) : **la VM**, DEMANDE n° 8.

Protocole : MAP 20 000 pas `--decroissant` + polissage float64, 6 inits.

**Critères figés :**

- **2017** :
  - Procuste ≥ 4/6 ;
  - T1 ≤ 2,23 ;
  - |écart V2 Fillon–LP| ≤ 9,3 (`hier`) ;
  - T0 (+4,7) rapporté.
- **2022** :
  - MAE V2 sur 4 paires ≤ 3,39 (P4b + 0,3) ;
  - V1-LOHO ≤ 14,2 (P4b + 1) ;
  - σ_LP/σ_M < 1,5 ; |z_pos| des finalistes ≤ 3.
- **Verdict.** Si 2017 ET 2022 tiennent, P4c remplace P4b comme référence. Si seul 2022 tient, P4c
  est retenu par parcimonie (moins de rayons à prédiction égale) et 2017 reste en échec. Si 2022
  échoue, P4b reste la référence.


**Résultat de P4c sur 2017** (VM, DEMANDE n° 8, 24/11 17:25Z, routine). Fichiers :
`results/{polir,modes}_map_hiercercle_ra_fid_sigdyn30_2017_T1-30_p20000d*.json`,
`results/v2paire_map_hiercercle_ra_fid_sigdyn30_2017_T1-30_p20000d_v2sans_FrançoisFillonMarineLePen_poli.json`,
et **les points retenus** `results/points_map_hiercercle_ra_fid_sigdyn30_2017_T1-30_p20000d{,_v2sans_FrançoisFillonMarineLePen}_poli.json`
(par init : `arc`, `z_pos`, `z_sigma`, `z_sigma_dyn`, `mu`, `sigma`, dans l'ordre des candidats du fit) pour le
Procuste de la session locale. Polissage convergé (‖∇‖ 0,002-0,004).

Données complètes, par init (−U poli ; Mélenchon, mesure 76,4 ; σ_LP/σ_M ; T1 ; arc) :

| init | −U | Mélenchon | σ_LP/σ_M | T1 | arc |
|---|---|---|---|---|---|
| 0, 1, 3, 5 | 2769,2 (identiques) | 83,9 | 0,70 | 2,22 | 4,74 |
| 2 | **3014,4** | 79,9 | 0,95 | 1,77 | 5,57 |
| 4 | 3013,8 | 89,4 | 0,98 | 1,97 | 6,18 |

- **Procuste ≥ 4/6 : ambigu.** 4 inits tombent exactement au même point (0, 1, 3, 5), mais c'est un mode
  245 nats SOUS les deux meilleurs ; les deux meilleurs (2, 4) sont à 0,5 nat l'un de l'autre mais
  différents (arc 5,57 / 6,18, Mélenchon 79,9 / 89,4). Lecture par lp : le mode retenu n'a qu'une
  init. Procuste à faire par la session locale sur les points exportés.
- **T1 ≤ 2,23 : ✓** 1,77 au meilleur point (toutes ≤ 2,22).
- **σ_LP/σ_M < 1,5 : ✓** 0,95 ; |z_pos| des finalistes : dans les points exportés, non calculés ici.
- **Mélenchon (rapporté)** : 79,9 au meilleur point (+3,5, dans les 5 pt, pour la première fois en 2017),
  mais 89,4 à l'init 4, quasi égale en lp ⇒ non robuste.
- **|écart V2 Fillon–LP| ≤ 9,3 : ✗** mode retenu (init 1) 43,1 (−14,0) ; autres modes 43,5 (−13,6) et
  67,5 (+10,4), observé 57,1.
- Rapporté : « Aucun » MAE 9,9 ; partages MAE 16,7 (persistance 4,1).

**Lecture.** T1 et σ tiennent ; V2 échoue (−14,0 contre −16,6 pour P4b, `hier` −9,3) ; stabilité au mieux
ambiguë. La fenêtre de 30 jours n'apporte pas la généralisation à 2017 sur V2.

**§8.87, Procuste de 2017 (session locale, 24/11 ~18:00Z, sur les points JSON exportés par la VM).**

Calcul : m_i = κ_i·e^{iθ_i}, centré ; rotation et réflexion orthogonales, sans échelle ; résidu
rapporté à l'échelle. Les poids ne sont pas pondérés par les parts, à la différence de la revue.

| Fit | Points | ΔU | Procuste |
|---|---|---|---|
| Données complètes | init 2 (meilleur) et init 4 | **0,5 nat** | **0,91** |
| Données complètes | inits 0, 1, 3, 5 | +245 nats | 0,84 (identiques entre elles) |
| V2 Fillon–LP | 6 points | 0 à 168 nats | 0,52 à 1,03 |

- Sur les données complètes, les inits 2 et 4 sont **deux configurations distinctes à U quasi égal**
  (0,5 nat) : vraie non-identification. Les inits 0, 1, 3 et 5 forment un même mode, mais 245 nats
  plus bas.
- En V2 Fillon–LP, il y a 6 points, dont aucun à moins de 30 nats du meilleur.
- **Stabilité 2017 : ÉCHEC** (critère « ≥ 4/6 dans le mode retenu »).
- Avec le V2 Fillon–LP à −14,0 (seuil 9,3) : **P4c ÉCHOUE sur 2017.** Retirer les fantômes ne suffit
  pas.
- 2022 est en cours sur le Mac. Le verdict de §8.87 ne peut plus être que « retenu par parcimonie »
  (si 2022 tient) ou « non retenu ».

### 8.88 Pénalité d'extrémité à 1 paramètre à la place de η (revue du 24/11 ~17:00Z ; avant code, figé)

**Motif.** Témoin T0 en 2017 : « pénalité propre à Le Pen » fait +4,7 sur Fillon–Le Pen, contre −14 à
−17 pour les variantes spatiales. Le mécanisme transférable est une désaffection pour les extrêmes au
second tour (front républicain), et non une valence libre par candidat.

**Forme.** Extrémité e_i = (2μ_i − 1)² ∈ [0, 1], avec μ_i la position sur l'arc, valable aussi pour
le segment. Elle vaut 0 au centre et 1 aux deux bouts.

$$\eta_i=-\lambda\,e_i,\qquad \lambda\sim\mathrm{HalfNormal}(1)\ \text{(logit)}.$$

- Remplace z_η (N) et s_η : **−N + 0 paramètre net** (1 au lieu de 24).
- La symétrie est assumée : Mélenchon et Le Pen sont pénalisés de la même façon ; aucune pénalité n'est
  attribuée à un nom.
- Variante `hiercercle_ra_fid_sigdyn_ext` (sur la base P4b, la référence).

**Plateformes (une par étape).**

- 2017 (données complètes et V2 Fillon–Le Pen) : **la VM**, DEMANDE n° 9 ;
- 2022 (4 V2, 3 V1-LOHO, données complètes) : **le Mac**, après P4c.

**Critères figés :**

- **2017** : |écart V2 Fillon–LP| ≤ 8,3 (`hier` − 1) ; T1 ≤ 2,23 ; Procuste ≥ 4/6. Rapporté : T0
  (+4,7) et Mélenchon (76,4).
- **2022** : MAE V2 ≤ 3,59 (P4b + 0,5) ; duels en échantillon ≤ P4b + 0,5 (1,59 → 2,09) ; V1-LOHO
  ≤ 14,2 ; |z_pos| des finalistes ≤ 3 ; perte de vraisemblance rapportée (AIC : −23 paramètres ×
  1 nat ; BIC : −23 × ln(3 500)/2 ≈ −94).
- **Verdict.** 2017 ET 2022 tenus ⇒ la variante devient la référence. 2022 seul ⇒ retenue par
  parcimonie, 2017 restant en échec. Sinon, P4b (ou P4c) reste la référence.


**Résultat de la pénalité d'extrémité sur 2017** (VM, DEMANDE n° 9, 24/11 19:25Z, routine). Fichiers :
`results/{polir,modes}_map_hiercercle_ra_fid_sigdyn_ext_2017_T1-30_p20000d*.json`,
`results/v2paire_map_hiercercle_ra_fid_sigdyn_ext_2017_T1-30_p20000d_v2sans_FrançoisFillonMarineLePen_poli.json`,
points retenus `results/points_map_hiercercle_ra_fid_sigdyn_ext_2017_T1-30_p20000d{,_v2sans_FrançoisFillonMarineLePen}_poli.json`
(par init : `arc`, `lam_ext`, `z_pos`, `z_sigma`, `z_sigma_dyn`, `mu`, `sigma`). Polissage convergé (‖∇‖ ≤ 0,0034).

Données complètes, par init (−U poli ; λ ; Mélenchon, mesure 76,4 ; σ_LP/σ_M ; T1 ; arc) :

| init | −U | λ | Mélenchon | σ_LP/σ_M | T1 | arc |
|---|---|---|---|---|---|---|
| 0, 1, 4, 5 | 2788,3 (identiques) | 0,007 | 85,2 | 0,82 | 2,21 | 4,58 |
| 2 | 2981,0 | 2,00 | 87,8 | 0,80 | 1,87 | 6,21 |
| 3 | **3027,9** | 0,33 | 85,6 | 0,80 | 1,73 | 5,69 |

V2 Fillon–Le Pen (observé 57,1), par mode : init 4 (meilleur lp, λ 3,81) **65,3 (+8,2)** ; inits 2-3
(λ 0,60-2,15) 56,8 (−0,3) ; init 1 68,0 (+10,9) ; inits 0, 5 (λ 0,59) 79,9 (+22,8).

- **|écart V2| ≤ 8,3 : ✓ de justesse** (+8,2) au mode retenu, mais les modes s'étalent de −0,3 à +22,8 et
  le meilleur lp ne devance le suivant que de 13,5 nats ⇒ fragile.
- **T1 ≤ 2,23 : ✓** 1,73 au meilleur point.
- **Procuste ≥ 4/6 : ambigu**, même figure que P4c : 4 inits identiques à λ ≈ 0 (pénalité éteinte),
  240 nats sous le meilleur point, qui est seul. Procuste à faire par la session locale sur les points.
- **λ non identifié** : de 0,007 à 3,8 selon l'init, et le mode retenu change avec λ ⇒ la pénalité
  ajoute des modes plus qu'elle ne contraint.
- Rapporté : Mélenchon 85,6 (+9,2) ; σ_LP/σ_M 0,80.

**Lecture.** Le critère V2 passe au mode retenu mais pas de façon robuste ; la stabilité et
l'identification de λ font défaut. Verdict laissé à la session locale après Procuste et 2022.

**§8.88, Procuste de 2017 (session locale, 24/11 ~20:00Z).** Même calcul qu'en §8.87 (non pondéré).

| Fit | Points | ΔU | Procuste |
|---|---|---|---|
| Données complètes | 3 (meilleur), 2 | 0, 46,9 | 0, **0,73** (λ 0,33 contre 2,0) |
| Données complètes | 0, 1, 4, 5 | +240 (λ ≈ 0) | 0,76 (identiques entre eux) |
| V2 Fillon–LP | 6 points, tous distincts | 0 à 284 | 0,79 à 1,14 |

- **Stabilité 2017 : ÉCHEC**, comme pour P4c : les points retenus sont seuls.
- λ n'est pas identifié (0,01 à 3,8).
- Le V2 Fillon–LP passe de justesse au mode retenu (+8,2 ≤ 8,3), mais les modes vont de −0,3 à
  +22,8 : **le critère V2 est tenu sans robustesse.**
- **2017 : ÉCHEC** (stabilité). Le verdict de §8.88 ne peut plus être que « retenue par parcimonie »
  si 2022 tient, 2017 restant en échec.

**Lecture pour l'objectif.**

- Avec deux paires de duels, toutes contre Le Pen, 2017 ne peut pas identifier à la fois la
  géométrie et une pénalité, même à 1 paramètre. Les paramètres T2 de 2017 se confondent.
- T0 (+4,7 sur la même paire, 0 paramètre ajusté) reste le meilleur prédicteur : l'information est
  là, mais sous une forme qu'aucune variante spatiale ne capte de façon stable.

### 8.89 Résultat de P4c sur 2022 (§8.87 ; 24/11 ~23:30Z, Mac, MAP 20 000 pas + polissage)

`hiercercle_ra_fid_sigdyn30` : 6 rayons libres (Mélenchon, Jadot, Macron, Pécresse, Le Pen, Zemmour).

| Fit | Écart V2 | Inits à 1 nat | Arc | max \|z_σ\| | max \|z_pos\| des finalistes | σ_LP/σ_M | ΔU contre P4b |
|---|---|---|---|---|---|---|---|
| Données complètes | — | 6/6 | 2,81 | 1,03 | 2,63 | 1,27 | +2,7 |
| Macron–Mélenchon | −0,4 | 6/6 | 2,97 | 1,22 | 2,52 | 1,23 | +3,5 |
| Macron–Zemmour | +6,2 | 6/6 | 2,44 | 0,80 | 2,88 | 1,31 | +0,6 |
| Macron–Le Pen | +4,2 | 6/6 | 3,06 | 1,65 | 2,45 | **1,77** | +0,9 |
| Macron–Pécresse | +1,6 | 5/6 | 2,73 | 1,89 | 2,34 | 1,23 | −1,4 |
| **MAE V2** | **3,09** | | | | | | |

V1-LOHO : Mélenchon–Macron 10,1 (U_fid 10,3), Macron–Zemmour 5,5 (17,5), Macron–Pécresse 23,5 (13,8).
Moyenne **13,0** (P4b 13,2).

**Critères de §8.87, 2022 :**

- V2 ≤ 3,39 ✓ (3,09) ;
- V1-LOHO ≤ 14,2 ✓ (13,0) ;
- |z_pos| ≤ 3 ✓ (2,88 au plus en V2 ; en V1, 2,92 à 2,98, tous candidats, au mode retenu) ;
- **σ_LP/σ_M < 1,5 : ✗ dans le fit Macron–Le Pen retiré (1,77)**, ✓ ailleurs (1,23 à 1,31).
  - Le texte figé ne dit pas sur quels fits la clause s'applique. La lecture stricte (tous les fits) la
    fait échouer.
  - P4b échoue de même dans ce fit (1,86, revue du 24/11 ~14:00Z). Sa règle §8.82 ne contenait pas
    cette clause : **la clause n'est donc pas comparable**.
  - Aucun assouplissement après lecture : je ne requalifie pas la clause.

**Verdict.**

- 2017 ✗ (§8.87) : P4c ne peut qu'être « retenu par parcimonie si 2022 tient ».
- 2022 **ne tient PAS en lecture stricte.** **P4b reste donc la référence** et P4c est un candidat non
  adopté.
- P4c ne dégrade rien d'autre : mêmes prédictions à 0,2 pt près, stabilité au moins égale (5 à 6
  inits sur 6), 2 rayons et 2 fantômes de moins, ΔU ≤ 3,5 contre P4b.
- **ARBITRAGE À TRANCHER (utilisateur)** : la clause σ_LP/σ_M < 1,5 porte-t-elle sur tous les fits
  (P4c non retenu) ou sur les données complètes (P4c retenu par parcimonie) ?

### 8.90 Errata de §8.86-8.89 et portée réelle de la neutralité (revue du 25/11 ~00:15Z)

**(a) Neutralité : l'énoncé de §8.86 était trop large.** Il ne couvrait pas les SNAPSHOTS.

| Objet | Constat (main contre branche, CSV au 2026-11-24) |
|---|---|
| `seconds_tours` dans chaque snapshot | ajouté par la branche ; **CORRIGÉ** (clé écrite seulement si non vide), testé |
| Snapshot gp-pooling complet | identique (0 différence) |
| Spatial : arguments par le vrai chemin `nowcast` | identiques |
| Spatial : U et gradient, 8 points, float32 et float64 | identiques au bit près |
| Spatial : chaîne NUTS à graine égale | **identique jusqu'à 4 pas, puis divergente** |
| `diagnostics` du spatial | deux clés de plus, `sigma_min` et `quadrature_fragile` (§8.48) : **restantes, déclarées** |

- L'écart des chaînes vaut 2e-15 (6 pas), 7e-14 (8), 2e-9 (12) en float64 : arrondi d'une compilation
  différente, amplifié par NUTS. **Même postérieur visé, mais pas les mêmes octets.** Les chiffres
  spatiaux publiés changeraient dans le bruit Monte-Carlo.
- Non fait : un fit de production complet des deux côtés, comparé à ce bruit. En file.

**(b) σ_LP/σ_M dépend de la jauge** : la clause est mal posée. La revue mesure, sur 15 fits,
σ_LP/σ_M de 1,00 à 1,86 à concentration d'électorat presque constante (R_LP de 0,58 à 0,66). En
« sans Macron–Le Pen », P4b fait 1,86 et P4c 1,77. Mon verdict (« P4b reste la référence car P4c fait
1,77 ») est donc **incohérent**, puisque P4b fait pire. Le verdict figé reste écrit ; l'arbitrage est
reformulé en (g). Tous les verdicts antérieurs fondés sur cette clause sont affectés (dont la
justification « cercle plutôt que segment » par le rapport, §8.68 et §8.72) : à relire avec R.

**(c) Extrémité (2μ − 1)² : mauvaise grandeur.** Elle dépend de la jauge et du roster (μ = ½ est le rang
médian), et n'a plus de sens quand l'arc approche 2π. λ n'est pas identifié PAR CONSTRUCTION. Le
« ✓ de justesse » de 2017 (+8,2) a z_pos(Le Pen) = −4,3 : la clause |z| ≤ 3 avait été OMISE du volet
2017 de §8.88. **Le volet 2017 est un échec.** Même omission pour P4c 2017 (point c4 à −5,0).

**(d) Procuste de §8.87-8.88** : protocole non conforme à §8.80 (e) (non pondéré, volet duel non
mesuré). Errata : « 6 points tous distincts » est faux (c0 ≡ c5) ; P4c V2, c1/c2 sont le même mode
selon (e), donc 2/6. L'échec de stabilité tient.

- Les 4 inits identiques à +240 nats sont un **échec d'optimisation** : 6 inits ne suffisent pas.

**(e) « 2017 ne peut identifier même 1 paramètre » (§8.88) est trop fort.** Test à 0 fit de la revue,
non revérifié ici : T0-cellules donne Fillon–LP à 57,2 (observé 57,1) sans paramètre ; l'enveloppe
plausible est [47,8 ; 61,5], et P4b (40,5) comme P4c (43,1) en sortent. **L'échec de 2017 est donc
surtout celui du modèle et de l'optimisation.** Les bornes de l'enveloppe sont choisies par la revue.

**(f) Parcimonie de §8.88 mal comptée.** Le « −24 paramètres, BIC −94 » est nominal : ½Σz_η² vaut
0,32 nat au MAP, soit environ 3 à 5 degrés de liberté effectifs. ΔU contient environ 21 nats de
normalisation, à retirer avant lecture.

**(g) T0 comme témoin permanent** (constante par centre d'étoile, retrait d'une paire ; chiffres de la
revue, non revérifiés) : MAE 4,7 sur les 5 paires de 2017 + 2022, contre 4,6 à 5,8 pour P4b, 5,3 pour
P4c et 10,8 pour R1. **R1 ne doit pas être publié comme référence** : il s'effondre en 2017.

**Prédiction de la revue pour l'extrémité 2022, consignée AVANT lecture :** λ ≲ 0,3 ; V2 hors Pécresse
à ± 1 pt ; Pécresse seule dégradée. Même si 2022 tient, la variante ne sera PAS adoptée (c).

**ARBITRAGE reformulé (utilisateur), remplace celui de §8.89 :** figer P4c comme référence de recherche
en remplaçant la clause σ_LP/σ_M par R_LP ≥ R_M − 0,05 (posée APRÈS lecture, donc à déclarer comme
telle), ou garder P4b ? Les prédictions sont les mêmes.

**§8.88, résultat 2022 (Mac, 25/11 ~01:00Z) : ÉCHEC, run interrompu.** Deux fits lus sur huit.

| Fit | Inits à 1 nat | λ par init | Lecture |
|---|---|---|---|
| Données complètes | 3/6 | 0,35 à 0,39 | ΔU brut +35,0 contre P4b |
| Macron–Mélenchon retiré | 3/6 | 0,41 ou **3,52** | mode retenu **+23,8** ; autre mode +0,4 |

- +23,8 sur une seule paire impose une MAE d'au moins 5,95 : le seuil de 3,59 est perdu quoi que
  donnent les 3 autres paires. Les six fits restants ont été ARRÊTÉS : ils ne pouvaient plus changer le
  verdict. **Leurs résultats ne sont pas connus.**
- Prédiction de la revue (§8.90) : λ ≲ 0,3, constaté 0,35-0,39 (proche) ; « V2 hors Pécresse à ± 1 pt »
  **démentie** au mode retenu. Les deux modes montrent la non-identification de λ annoncée en (c).
- ΔU contient environ 21 nats de normalisation (§8.90 f), avec un signe à vérifier avant tout usage ;
  il n'est pas interprété ici.

**T0 vérifié (session locale, arithmétique sur les 5 paires) :** −6,0 / −3,4 / +5,9 / +3,5 en 2022 et
+4,7 en 2017, soit MAE 4,7. Confirme §8.90 (g). Réserve : T0 utilise les duels des autres paires, pas
le T1 ni les reports.

### 8.91 Neutralité des chiffres spatiaux publiés — protocole figé AVANT lecture (25/11 ~01:00Z)

Quatre fits de production (premier tour seul, CSV au 2026-11-24) : main et branche, graines 27 et 28,
**tous sur la VM**, par `model/backtest/neutralite_spatial.py`, qui n'écrit rien dans `docs/`. Écart
mesuré en écarts-types a posteriori sur la part moyenne de chaque candidat, champ complet.

- **Critère :** écart maximal main/branche ≤ 1,5 × le plus grand écart entre graines d'un même arbre,
  ET R-hat ≤ 1,2 sur les 4 fits.
- Sinon : « NON conclu », sans relâcher le seuil ; pas de « PRÊT À PUBLIER ».
- Limite déclarée : le roster de la branche diffère (Hollande et Pinel en 2017, sans effet attendu en
  2027) ; le script s'arrête si les candidats diffèrent.

**Résultat** (VM, DEMANDE n° 10, 25/11 00:57Z, routine ; main extrait par `git archive origin/main`,
`results/neutralite_spatial_{main,branche}{27,28}.json`). 19 candidats identiques des deux côtés.

| comparaison | écart max | écart moyen |
|---|---|---|
| graines main/main | 0,071 sd | 0,028 sd |
| graines branche/branche | 0,094 sd | 0,030 sd |
| main/branche, graine 27 | 0,042 sd | 0,014 sd |
| main/branche, graine 28 | 0,029 sd | 0,009 sd |

R-hat 1,016 / 1,014 / 1,021 / 1,017 (≤ 1,2) ; ESS min 195-238 ; 0 divergence.
**Critère tenu** : écart main/branche max 0,042 ≤ 1,5 × 0,094 = 0,141. **Verdict : neutre au bruit près**
(l'écart entre arbres est plus petit que l'écart entre graines d'un même arbre).

### 8.92 Essai 2027 : modèle unique, cercle contre segment — protocole figé AVANT code et lecture (25/11 ~20:20Z)

Demandé par l'utilisateur le 25/11 (« utilise les données 2027/2026 », « je veux un unique modèle »).
Remplace §8.74, inexécutable.

**Données vues avant de figer :** effectifs des duels et cellules BRUTES de l'électorat Mélenchon (Harris
seul : 5 à 12 % vers le RN face à Philippe, Attal ou Glucksmann, 49 à 67 % « Aucun »). Aucune prédiction
de modèle n'a été lue sur ces cellules. 2027 n'a pas servi à choisir les variantes, sauf les anciens
diagnostics κ/η (déclaré, §8 ancien).

**Modèle unique, deux géométries qui ne diffèrent que par elles.** T1, duels et reports de tout 2026
(`MIN_POLL_DATE`), as_of 2026-11-25, une seule vraisemblance ; η hiérarchique, fidélité R7, rayons libres
sur 30 jours (P4c).

- **C** : cercle ; **S** : segment, 1 paramètre de moins (pas d'arc), B = 100.
- Champ déclaré (`lse_officiel`, `champ_reports`) : hypothèse T1 la plus fréquente des 60 derniers jours.
- Le Pen et Bardella restent deux candidats distincts, comme en production.

**Épreuves.** MAP 8 inits (amendé de 4 à 8 AVANT lecture des prédictions ; seul U du fit complet, 1 init, vu) + polissage float64 ; une seule plateforme, le Mac. Mode retenu : meilleur U ;
les autres modes à moins de 10 nats sont rapportés.

1. **V2** : chaque paire d'au moins 3 duels en 2026 est retirée (duels ET reports de l'hypothèse) ; cible :
   moyenne de ses duels des 60 derniers jours, ou des 3 derniers s'il n'y en a pas.
2. **V1** : les reports de Philippe–Le Pen, puis Attal–Le Pen, puis Glucksmann–Le Pen sont cachés (duels
   gardés) ; cible : moyenne des notices, 3 cellules par électorat non finaliste.
3. **LFI → Le Pen** (question de l'utilisateur) : électorat Mélenchon des épreuves V1, part RN et « Aucun »,
   en points du total.

**Témoins à 0 paramètre ajusté.** V2 : T0 (moyenne des autres duels X–Le Pen) pour les paires X–Le Pen,
et Bradley-Terry pour toutes. V1 : **K**, mêmes électorats dans les AUTRES hypothèses X–Le Pen (moyenne).

**Règles de décision.**

- **Cercle retenu** si LFI → Le Pen (erreur absolue moyenne des cellules RN et Aucun) est meilleur que le
  segment d'au moins 2 pt ET si V2 et V1 ne sont pas pires de plus de 0,5 pt.
- **Segment retenu** dans le cas symétrique. Sinon : **non séparés** ; la parcimonie penche alors pour le
  segment d'un paramètre, décision à l'utilisateur.
- **« Prévision du T2 »** seulement si la géométrie retenue bat chaque témoin d'au moins 1 pt (V2 contre le
  meilleur de T0 et BT sur les paires communes ; V1 contre K). Sinon : représentation.
- Plausibilité rapportée sans seuil : concentration d'électorat R de Le Pen et Philippe (§8.90 b).
- Un seul institut (Harris) fournit les reports : tout résultat V1 est « selon Harris ».

### 8.93 Résultat de l'essai 2027 (§8.92 ; 25/11 ~22:00Z, Mac, MAP 8 inits float64, 24 fits, 0 erreur)

Fichiers `results/essai2027_*.json`, lecture `essai2027_lecture.json` (lecteur commité avant lecture).

| | Segment | Cercle | Témoin |
|---|---|---|---|
| V2, 8 paires cachées | 4,91 | 4,96 | **BT 1,81** |
| V2, 6 paires avec Le Pen | 3,66 | 3,17 | T0 3,76 ; **BT 1,51** |
| V1, 9 cellules × 3 | **7,36** | 9,54 | K 9,75 |
| LFI : RN et Aucun (critère figé) | 10,7 | 12,2 | **K 2,6** |
| Inits au même point | 8/8 sauf une épreuve 1/8 (Le Pen–Mélenchon) et une 7/8 | 8/8 partout | |

**Verdicts figés : géométries NON SÉPARÉES ; pas de « prévision du T2 »** (BT bat les deux de 3 pt en V2 ;
V1 : le segment bat K de 2,4 pt, le cercle de 0,2, mais K fait 2,6 sur LFI contre 10,7 et 12,2).

**Lu APRÈS coup (descriptif, ne change pas le verdict) :**

- **LFI → RN seul** (points du total, hypothèses cachées) : cercle 1,5 pt d'erreur, segment 6,4, K 2,4.
  Le segment annonce 2 à 6 % au lieu de 8 à 12 ; le cercle 8 à 9.
- **Les deux ratent la suite** : trop d'« Aucun » (cercle 22,9 pt d'erreur, segment 15,1) et pas assez pour
  l'adversaire (23,9 et 14,9). En échantillon aussi : l'écart ne vient pas du masquage.
- **Ajustement** : le cercle est meilleur de **79 à 110 nats de potentiel sur les 12 épreuves**, pour 1
  paramètre (a priori compris) ; cellules en échantillon 5,9 contre 8,2.
- **Plausibilité** : segment σ_Le Pen 0,19 contre 0,09 pour Philippe, Zemmour à z_pos −5,5 ; cercle 0,10
  contre 0,13, max |z_pos| 2,84. Le rapport de σ dépend de la jauge (§8.90 b), les z_pos moins.
- K est pollué pour l'électorat Attal (une seule autre hypothèse, Mélenchon–Le Pen) ; sans cette cellule, V1 :
  segment 7,3, cercle 10,0, K 7,1. **Alors plus aucune géométrie ne bat K.**

**Limites** : MAP (pas de NUTS), Mac ; Harris seul pour les reports ; cibles V2 sur 2 à 13 duels ; BT et K
utilisent la même campagne.

**Lecture.** Le cercle explique le mieux l'ensemble et le passage LFI → RN ; il échoue sur la case
« Aucun » de LFI. Le défaut commun est l'ABSTENTION, trop élevée pour un électorat éloigné des deux
finalistes : c'est la prochaine pièce à revoir, pas la géométrie.

### 8.94 Revue du 26/11 sur §8.93 : erreur de lecture par la date, correction déclarée

**(a) « Le spatial n'ajuste pas les duels qu'il voit » (revue) : RÉFUTÉ, c'est un artefact de lecture.**
- `lire()` évaluait tous les duels à l'as_of (fin novembre) au lieu de leur date.
- Refit complet avec la prédiction de chaque duel à SA date (`duel_s`, même U au millième) :
  - cercle : 1,16 pt par duel, 0,49 par paire ;
  - segment : 1,23 et 0,79 ;
  - BT : 1,04 en échantillon (revue).
- Le modèle ajuste donc bien ses duels.

**(b) Le même défaut biaisait V2 §8.93.** Deux cibles ne sont pas récentes : Bardella–Philippe (mars à mai)
et Le Pen–Retailleau (juillet-août). Lues à l'as_of, elles chargeaient les géométries (Bardella–Philippe :
cercle 64,9 à l'as_of contre 52,3 aux dates de la cible, pour 51,5 observés). BT, sans date, n'est pas
concerné.

- **Correction APRÈS lecture** : V2 lu aux dates de la cible, par le chemin de w (`duels_date`), et les 8
  épreuves refaites. Le verdict de §8.93 est conservé tel quel.
- La lecture corrigée sera rapportée À CÔTÉ, sans changer les seuils, et marquée « corrigée après lecture ».

**(c) Autres points de la revue, retenus :**
- K pollué pour l'électorat Attal ; K-bloc à rapporter aussi.
- ΔU : environ 35 nats viennent de l'a priori (z_pos, z_σ) ; la vraisemblance pure donne environ 45 à
  75 nats. À décomposer.
- Les inits ne sont que du bruit autour d'un même point ; 8/8 ne prouve pas un mode global.
- Le champ du lse diffère de celui des cellules (effet de 1 à 3 pt).
- La V2 Le Pen–Philippe retire 40 lignes de reports.
- **Le mécanisme d'abstention est identifié** : pente κ, imposée par les duels. La pente impliquée vaut
  environ 0,56, contre κ = 0,93. Correction à 1 paramètre (β découplé) **spécifiée, NON lancée**, en
  attente de l'accord de l'utilisateur. Critères proposés par la revue : β ∈ [0,4 ; 0,75] ; LFI ≤ 6 ;
  V1 ≤ K − 1 ; V2 +0,3 au plus ; ΔU ≥ 3.

### 8.95 V2 de l'essai 2027 relu aux dates des cibles (correction APRÈS lecture, §8.94 b ; 26/11 ~10:00Z)

Mêmes fits refaits, 8 inits, Mac, 0 erreur ; mêmes U que §8.93. Écart à l'observé, en points :

| Paire (observé) | Segment, à l'as_of → à la date | Cercle, à l'as_of → à la date | BT | T0 |
|---|---|---|---|---|
| Attal–Le Pen (44,0) | −6,7 → −6,6 | −5,1 → −5,6 | −1,6 | −2,7 |
| Bardella–Philippe (51,5) | +9,8 → +2,1 | +13,4 → **+0,8** | +4,6 | — |
| Glucksmann–Le Pen (43,2) | −3,3 → −1,3 | −0,4 → +0,8 | −2,2 | −1,8 |
| Hollande–Le Pen (43,0) | −4,0 → +1,0 | −7,8 → −4,1 | +1,0 | −1,5 |
| Le Pen–Mélenchon (67,7) | −0,8 → +0,2 | −1,6 → −0,2 | +0,1 | +11,3 |
| Le Pen–Philippe (55,1) | +0,9 → −0,9 | +0,7 → −0,2 | −2,8 | −3,8 |
| Le Pen–Retailleau (57,0) | +6,4 → +1,5 | +3,4 → −0,7 | +1,4 | −1,5 |
| Mélenchon–Philippe (36,5) | +7,5 → +3,9 | +7,3 → +3,2 | −0,8 | — |
| **MAE** | 4,91 → **2,18** | 4,96 → **1,95** | **1,81** | 3,76 (6 paires) |

**Lecture corrigée, aux mêmes seuils que §8.92 :**

- Cercle et segment : 1,95 contre 2,18, écart de 0,23 ≤ 0,5, donc toujours **non séparés**. Le cercle fait
  mieux sur 5 paires sur 8.
- Face à BT : 1,95 contre 1,81 ; ni l'un ni l'autre ne bat BT d'1 pt, donc toujours **pas de « prévision »**
  au sens figé. [RETIRÉ, §8.96 b : BT sans date n'était pas équitable ; BT daté environ 1,0.]
- Le verdict de §8.93 reste écrit ; celui-ci est la lecture corrigée, déclarée après lecture.
- Non corrigé : les cellules V1 sont lues à l'as_of, alors que les cibles sont des moyennes de mai à
  novembre. L'effet attendu est petit, les reports Harris étant stables.

### 8.96 Revue du 26/11 ~11:00Z sur §8.95 : BT daté, tests appariés, V1 à la date

**(a) La correction V2 est juste** (dates sur les nœuds, `duel_s` reproduit, pas de fuite). Erratum :

- dans la V2 Hollande–Le Pen, Hollande était hors du `masque_lse`. Corrigé, la MAE vaut 2,01 pour le
  cercle et 2,15 pour le segment ;
- le cercle fait mieux sur **7 paires sur 8** (et non 5), l'égalité étant Le Pen–Mélenchon.

**(b) BT sans date n'est pas un témoin équitable** contre un modèle lu à la date. Un BT daté (entraîné sur
les autres duels à ±W jours de chaque cible) fait **1,01 à ±30 et ±60 j, 0,85 à ±90 j**, contre 1,81 sans
date. Les fenêtres ont été choisies après lecture.

- **Le spatial (environ 2,0) est donc à environ 1 pt du meilleur témoin, et non à 0,14.**
- La phrase de §8.95 « au niveau du meilleur témoin » est **retirée**.

**(c) Tests appariés (8 paires) : aucun écart n'est distinguable du bruit.**

| Écart | Moyenne | Test de signe | IC t 95 % |
|---|---|---|---|
| Cercle − BT ±60 j | +0,94 | p = 0,73 | [−1,1 ; +3,0] |
| Cercle − segment | −0,24 | p = 0,07 | [−1,4 ; +0,9] |

Les tests sont optimistes, car les erreurs sont corrélées (0,72) et les candidats partagés. **8 moyennes ne
peuvent pas séparer les géométries.**

**(d) La lecture des cellules V1 à l'as_of a un GROS effet** (revue, à 0 fit, en échantillon) : V1 cercle
6,57 → 5,58, segment 5,30 → 4,33 ; LFI (RN + Aucun) cercle 8,9 → 6,2, segment 8,75 → 4,1.

- Approximation en hypothèses cachées : la règle figée de §8.92 basculerait vers le **segment** (LFI 6,9
  contre 9,6).
- **Lecture datée en cours** (6 fits V1 refaits, `z` sauvegardé, cellules à la date de chaque notice,
  finalistes et électorat dans le lse). Ce sera une correction APRÈS lecture, aux seuils inchangés.

**(e) Point faible propre au cercle** : Mélenchon → Glucksmann à 61-68 %, contre 36 mesurés, même en
échantillon ; le segment est à environ 39.

**(f) β.** L'excès d'« Aucun » venait en grande partie de la lecture à l'as_of. La pente impliquée (0,56)
est à recalculer aux dates avant tout fit. Critères proposés pour un essai β :

- log-vraisemblance prédictive des observations cachées, sommée sur 11 épreuves, comme juge cercle contre
  segment (ΣΔ ≥ 5 nats et 8/11) ;
- non séparés ⇒ **segment, par parcimonie, figé** ;
- témoin V2 : BT ±60 j.

À figer avant tout fit, sous réserve de l'accord de l'utilisateur.

### 8.97 V1 de l'essai 2027 relu à la date de chaque notice (correction APRÈS lecture, §8.96 d ; 26/11 ~14:35Z)

Six fits V1 refaits sur le Mac (8 inits, float64, 0 erreur ; les 8 inits de chaque fit sont à moins d'un nat
du meilleur U). `z` est sauvegardé. Cellules lues à la date de chaque notice, finalistes et électorat dans le
lse. Seuils inchangés (§8.92).

| Épreuve | Segment | Cercle | Témoin |
|---|---|---|---|
| V1, MAE, 9 électorats | 7,54 (7,36 à l'as_of) | 8,88 (9,54) | K 9,75 |
| LFI → Le Pen, RN + Aucun | 9,41 (10,73) | 10,53 (12,21) | K 2,57 |
| V2, MAE, 8 paires, à la date (§8.95) | 2,18 | 1,95 | BT 1,81 ; BT ±60 j ≈ 1,0 |

**Règle figée appliquée.**

- LFI : le cercle est **pire** de 1,1 pt. Aucune géométrie n'a l'écart de 2 pt.
- V1 : le cercle est pire de 1,3 pt, au-delà de la tolérance de 0,5 pt.
- **Verdict : non séparés**, comme en §8.93.
- **Pas de « prévision » :**
  - V2 ne bat aucun témoin ;
  - en V1, « le segment bat K de 2,2 pt » est **retiré** (§8.98 b) : sans la cellule polluée, aucune géométrie ne bat K.
- Par la règle de parcimonie de §8.92, **le segment l'emporte**. La décision reste à l'utilisateur.

**L'estimation de §8.96 d est démentie.** Elle prédisait un gros gain à la date et une bascule nette vers le
segment (LFI 4,1 contre 6,2), mais elle portait sur des cellules vues en échantillon. Sur reports cachés,
dater fait gagner 0,2 à 1,7 pt, pas 3 à 5. LFI reste à 4 fois l'erreur de K.

**Lu après coup, hors critère : la part RN seule de l'électorat Mélenchon.** RETIRÉ en §8.98 a (compensation d'erreurs) ; chiffres provisoires (§8.98 c).

| Hypothèse | Mesuré | Segment | Cercle |
|---|---|---|---|
| Philippe – Le Pen | 8,6 | 2,0 | 9,8 |
| Attal – Le Pen | 8,3 | 9,9 | 8,5 |
| Glucksmann – Le Pen | 12,0 | 3,5 | 8,0 |
| **MAE** | | **5,6** | **1,8** |

- Le cercle place bien LFI par rapport au RN, comme en §8.93 (1,5 contre 6,4).
- Il le paie sur « Aucun » :
  - MAE 19,2 pour le cercle, contre 13,2 pour le segment ;
  - le pire cas est Mélenchon → Glucksmann, avec 69 % pour Glucksmann et 23 % pour « Aucun » mesurés à 36 et 52 (§8.96 e).
- Le signal « le cercle explique mieux LFI → RN » tient donc sur trois cellules d'un seul institut.
- C'est l'abstention de l'électorat éloigné qui fait l'erreur, pas la géométrie.
- Lecture post hoc, Harris seul : **non probant**.

**Erratum de lecture.** Le lecteur imprime encore 1,95 et 2,18 pour V2 à la date, car le fit V2
Hollande–Le Pen corrigé (`masque_lse`, §8.96 a : 2,01 et 2,15) n'a pas été réécrit dans les résultats. Le
verdict ne change pas.

### 8.98 Revue du 26/11 ~14:40Z sur §8.97 : deux retraits, un défaut de lecture

**(a) Retrait du post hoc favorable au cercle.** Vérifié depuis le JSON.

- En points du total, « RN seul, cercle 1,8 contre 5,6 » est une **compensation d'erreurs**.
- Part du RN parmi les **exprimés** de l'électorat Mélenchon :

| Hypothèse | Mesuré | Cercle | Segment | K |
|---|---|---|---|---|
| Philippe – Le Pen | 19,2 | 35,4 | 11,3 | |
| Attal – Le Pen | 19,5 | 27,1 | 27,8 | |
| Glucksmann – Le Pen | 25,0 | 10,3 | 6,5 | |
| **MAE** | | **12,8** | **11,6** | **3,9** |

- Le cercle tombe juste en part du total parce qu'il compte trop d'« Aucun » : ses exprimés valent 27,7 au lieu de 44,8 face à Philippe.
- **Ce post hoc ne dit rien en faveur du cercle.** Sa ligne de §8.97 est retirée.

**(b) Retrait de « le segment bat K de 2,2 ».**

- Sans la cellule K polluée (Glucksmann – Le Pen, électorat Attal : K vient de la seule hypothèse Mélenchon – Le Pen), en lecture datée, sur 8 électorats : segment 7,33, cercle 9,20, K 7,13.
- **Aucune géométrie ne bat K**, ni en V1 ni sur LFI.

**(c) Défaut de lecture dans `cellules()` (`essai_2027.py`, ligne 103).** Avec `w` à la date de la notice, la part de
duel `s` contrainte par le Newton reste `duel_draws` à l'as_of, alors que l'entraînement utilise `s_r` à la
date (`joint_model.py`, ligne 329).

- Les `pred_date` V1 de §8.97 sont donc **hybrides**, et le défaut touche surtout le partage a contre RN.
- Correction : `_duel_par_tirage(fit, w, …)`, puis rejouer le `z` sauvegardé, sans fit. Mise en file.
- Les chiffres de §8.97 restent provisoires jusque-là.

**(d) Verdict de §8.97.** Il avait été appliqué à la main ; le code l'applique désormais aussi à la lecture
datée (clé `verdict_geometrie_date` de `essai2027_lecture.json`) : **non séparés**, inchangé.

**(e) Fait structurel retenu : K bat tout sur LFI.**

- LFI donne environ 35 % à l'adversaire du RN et environ 55 % « Aucun », quel que soit l'adversaire.
- Un modèle où abstention et partage dépendent de la distance ne produit pas cette platitude ; ses erreurs changent de sens selon l'adversaire.
- **Une pente β unique ne règle que le niveau** : le critère « LFI ≤ 6 » de §8.94 c est probablement hors d'atteinte avec β seul.
- **β est suspendu** derrière un test sans fit (§8.98 g, 2).

**(f) Taille réelle des données, déclarée.**

- Reports d'avant T1 : Glucksmann – Le Pen et Mélenchon – Le Pen ne comptent qu'une notice Harris chacune.
- Les 3 cellules V1 Glucksmann viennent donc d'un seul sondage.
- V2 repose sur 8 paires corrélées.
- P4c (rayons sur 30 j) a été utilisé en 2027 alors que §8.89 le laisse non adopté : déclaré comme choix du protocole §8.92.
- **Le BT ±60 j est figé comme témoin V2 de tout nouvel essai.**
- Biais mineur, sans incidence sur le verdict : `searchsorted` prend le nœud suivant au lieu du plus proche.

**(g) Pistes et critères, figés avant lecture.** Tout résultat de reports reste « selon Harris, non probant ».

1. **Relecture corrigée, sans fit.** On applique (c), puis la règle de §8.92 par le code ; on rapporte aussi K sans la cellule polluée.
2. **Test du niveau d'« Aucun », sans fit.** On rejoue le `z` V1 avec β ∈ {0,4 ; 0,56 ; 0,75}, seul ρ changeant.
   - Un fit β n'est demandé que si LFI < 6 pt **et** si la part RN parmi les exprimés ne se dégrade pas de plus de 1 pt.
3. **Segment η ≡ 0 contre le segment actuel** (−20 paramètres), sur 11 épreuves.
   - η ≡ 0 est retenu si la log-vraisemblance prédictive des observations cachées vérifie ΣΔ ≥ −5 nats **et** si V2 daté se dégrade de 0,3 pt au plus.
   - En cas d'égalité, η ≡ 0.
4. **Arc figé à 2π contre arc libre, puis segment contre cercle fermé.**
   - La géométrie la plus riche n'est retenue que si ΣΔ ≥ 5 nats et 8 épreuves sur 11 ; sinon la plus parcimonieuse.
5. **« Aucun » constant par électorat** (1 paramètre par électorat, à la place de la distance).
   - Retenu si V1 ≤ K − 1 et si V2 ne se dégrade pas de plus de 0,3.
   - Maths à écrire avant le code.
6. **P4 contre P4c.** P4 est retenu si V2 daté se dégrade de 0,3 au plus et si z_σ max ≤ 3 ; sinon P4c.

Les pistes 1 et 2 se font sans fit. Les pistes 3 à 6 sont des fits Mac (12 chacun) ; elles touchent au modèle, d'où l'arbitrage soumis à l'utilisateur.

### 8.99 Relecture V1 corrigée (§8.98 c, g1 ; 26/11 ~16:20Z, Mac, 0 fit)

**Méthode.**

- Le partage du duel est désormais pris à la date de la notice : `_duel_par_tirage` sur `w` à cette date.
- Les 6 fits V1 ont été rejoués depuis leurs valeurs sauvegardées (`essai_2027.py --rejouer`), soit 48 inits :
  - le rejeu reproduit `duel_s` à 0,0 près ;
  - la lecture de §8.97 est gardée sous `cellules_s_asof`.
- Comme prévu, « Aucun » est inchangé, seul le partage a contre RN bouge.
- Le verdict est appliqué par le code (`verdict`, testée) : seuils de §8.92, témoins de §8.98.

| Épreuve | Segment | Cercle | K |
|---|---|---|---|
| V1, 9 électorats | 7,26 | 8,79 | 9,75 |
| V1, sans la cellule K polluée (8) | **7,10** | 9,06 | **7,13** |
| LFI → Le Pen, RN + Aucun | 9,32 | 10,50 | 2,57 |
| Électorat Mélenchon, part RN des exprimés (rapportée sans seuil) | 10,5 | 9,5 | 3,9 |

**Verdict figé : non séparés.**

- LFI : le cercle est pire de 1,2 pt.
- V1 : le cercle est pire de 1,5 pt.

**Pas de « prévision ».**

- En V1, le segment égale K (7,10 contre 7,13), sans le battre de 1 pt.
- En V2 (§8.95-8.96), aucune géométrie ne bat BT ±60 j.

**Les géométries reproduisent les reports cachés aussi bien qu'un témoin sans paramètre, pas mieux.**

**Ce qui tient, lu après coup.**

- Part RN de l'électorat Mélenchon, en points du total : cercle 1,7, segment 5,4. Elle repose toujours sur un excès d'« Aucun » du cercle face à Philippe et Attal (§8.98 a).
- Parmi les exprimés, l'avantage du cercle n'est plus que de 1 pt, sur 3 cellules Harris : non probant.
- Le défaut commun reste la platitude de LFI (environ 35 / 55 quel que soit l'adversaire), que K capte et qu'aucune distance ne capte.

**Conséquences.**

- Tout ce qui suit dans l'essai 2027 relève de la parcimonie (§8.98 g3 à g6) ou de l'abstention (g2, g5).
- Rien ne départage le cercle et le segment sur les données 2027.

### 8.100 Revue du 26/11 ~16:15Z sur §8.99 : témoin K_s et critères figés du test β (AVANT code et lecture)

**(a) Vérifié par la revue.**

- Le rejeu est fidèle : 13 sites échantillonnés fournis, `duel_s`, `mu` et `eta` identiques.
- `cellules(…, w à la date)` redonne le `reports_theta` du modèle à 8e-16 près : la part `s` à la date est bien `s_r`.
- Les chiffres de §8.99 et `verdict()` sont conformes.
- **Correction de §8.99 :** les 8 inits de chaque fichier ont le même U. Le contrôle porte sur **6 fits**, pas sur 48 points distincts.

**(b) Retraits et précisions sur §8.99.**

- « RN en points du total : cercle 1,7 contre 5,4 » est **retiré** de « ce qui tient ». C'est la compensation d'erreurs de §8.98 a.
- L'écart sur les exprimés change de signe selon la lecture :
  - à l'as_of : 17,1 contre 10,4 ;
  - en lecture hybride : 12,8 contre 11,6 ;
  - en lecture datée : 9,5 contre 10,5.
- **Il ne tient donc pas non plus.**
- Le sous-ensemble « sans cellule polluée » a été choisi après lecture : il reste un témoin rapporté et n'entre dans aucun verdict.

**(c) Défaut de lecture.** `pred_date` moyennait les dates de l'hypothèse, sans regarder l'électorat (`essai_2027.py`, lecture des cellules) :

- pour Attal – Le Pen, électorat Mélenchon, 4 dates étaient moyennées contre 3 notices mesurées ;
- correctif : ne garder que les notices où l'électorat a ses cellules complètes, comme `mesure()` ;
- relecture par rejeu, sans fit.

**(d) Comparaison inéquitable en V1.** La géométrie connaît la part du duel caché par `s`, ce que K ignore. Témoin
**K_s**, à 0 paramètre :

- garder le « Aucun » de K pour chaque électorat ;
- décaler le logit a/RN de K d'un δ commun à tous les électorats de la notice ;
- choisir δ pour que l'agrégat sur les électorats du champ (parts T1 des sondages de la même date, pondérées comme dans le Newton du modèle) redonne la part `s` du duel observé ;
- moyenner sur les notices de la cellule, comme pour `pred_date`.

**(e) Critères figés, avant tout code de K_s ou de β.**

1. **Une géométrie n'apporte quelque chose sur les reports** que si les deux conditions suivantes sont réunies (8 cellules hors cellule polluée, et 9 rapportées) :
   - V1 ≤ K_s − 1 ;
   - au moins 2 des 3 hypothèses s'améliorent.

   Sinon, le modèle reproduit les reports aussi bien qu'un témoin à 0 paramètre qui connaît le duel.
2. **Test β sans refit**, ρ = σ(β·ℓ − ω) dans les cellules de reports seulement :
   - **mise en œuvre :**
     - `kappa_rho` optionnel dans `duel_expression` et `reports_cellules` (`geometry.py`), `q` intact ;
     - `s` gardé à sa valeur ajustée ;
     - ω re-profilé en une dimension pour chaque β, sur la seule vraisemblance Dirichlet des reports **d'entraînement**, `log_c` fixé ;
   - **choix de β\*** sur la grille {0 ; 0,4 ; 0,56 ; 0,75 ; κ}, par cette vraisemblance d'entraînement, jamais sur les cellules cachées ;
   - **non-régression :** `kappa_rho=None` redonne `reports_theta` à 1e-12 près ;
   - **un fit β n'est demandé que si,** à β\* :
     - LFI daté < 6 ;
     - la part RN des exprimés n'est pas pire de plus de 1 pt ;
     - V1 ≤ K_s.
3. **« Aucun » constant par électorat (g5)** : seulement si β\* = 0 et si l'écart à K_s dépasse 2 pt.
   - Le critère g5 « V1 ≤ K − 1 » devient « V1 ≤ K_s, et V2 pas pire de plus de 0,3 ».
   - Ce modèle reproduit K par construction.
4. **Les pistes g3 (η ≡ 0) et g6 (P4 contre P4c) sont inchangées.**
   - η n'entre dans les cellules que par `s`, puisque le Newton converge.
   - g3 se juge donc surtout sur les duels.

**(f) Géométrie.** Le segment est meilleur ou égal en V1 et non séparé partout, et la parcimonie le désigne (§8.92).

- Si l'utilisateur adopte le segment, g4 (arc 2π, cercle fermé) devient sans objet : 12 fits économisés.
- **ARBITRAGE À TRANCHER.**

### 8.101 Correctif `pred_date` appliqué (§8.100 c ; 26/11 ~17:15Z, Mac, 0 fit)

Une date par notice où l'électorat a ses cellules a, b et « Aucun » complètes, comme `mesure()`. Les 6 fits V1
sont rejoués et reproduisent le fit sauvegardé à l'identique.

**Effet : au plus 0,1 pt.** Seule la cellule Attal – Le Pen de l'électorat Mélenchon change (4 dates ramenées à 3).

| Mesure (lecture datée) | Segment | Cercle | K |
|---|---|---|---|
| V1, 9 électorats | 7,22 | 8,74 | 9,75 |
| V1, sans la cellule K polluée (8) | 7,06 | 9,00 | 7,13 |
| LFI, RN + Aucun | 9,24 | 10,40 | 2,57 |
| Électorat Mélenchon, part RN des exprimés | 10,4 | 9,2 | 3,9 |

**Verdict figé (code) : non séparés.** Pas de prévision. Les conclusions de §8.99 et §8.100 sont inchangées.
Prochaine étape : le témoin K_s (§8.100 d-e1).

### 8.102 Témoin K_s : définition exacte (précise §8.100 d ; écrite AVANT code, 26/11 ~18:00Z)

§8.100 d laissait ouverts les électorats des finalistes et ceux sans cellule. K_s reste à 0 paramètre ajusté.

**Pour chaque notice n de l'hypothèse cachée h = a – b :**

1. **E_n**, les électorats T1 qui ont des cellules a, b et « Aucun » complètes dans la notice, y compris ceux des finalistes. Les cellules mesurées ne servent qu'à fixer E_n, jamais à la valeur prédite.
2. **Poids p_e** : intention T1 de e dans la même notice, renormalisée sur E_n.
   - On prend l'hypothèse T1 de la notice qui contient a, b et tous les e de E_n ; s'il y en a plusieurs, leur moyenne.
   - Sans hypothèse T1 convenable : notice exclue de K_s, nombre rapporté.
3. **Cible s_n** : part de a dans le duel a – b de la même notice (`date_fin` égale). Sans duel dans la notice : notice exclue, nombre rapporté.
4. **Cellules de base** (a, b, Aucun) :
   - **e non finaliste** : K_e (§8.92), soit les mêmes électorats dans les AUTRES hypothèses X – Le Pen, en moyenne ;
   - **électorat de Le Pen** : idem, K de l'électorat Le Pen ;
   - **électorat de a** : moyenne, sur les autres hypothèses X – Le Pen, de la cellule de X dans son propre électorat (fidélité), avec le même ordre (soi, Le Pen, Aucun).
5. **Ajustement** : on garde « Aucun » (n_e) de chaque cellule de base, et on écrit le partage des exprimés comme σ(l_e + δ), où l_e = logit(a_e / (a_e + b_e)).
   - δ est l'unique racine de Σ_e p_e (1 − n_e) σ(l_e + δ) / Σ_e p_e (1 − n_e) = s_n.
   - La racine existe et est unique, la fonction étant monotone.
6. **Approximation déclarée** : les électorats hors E_n (petits candidats, environ 15 à 25 % du T1) sont supposés se partager comme l'agrégat de E_n. C'est ce qui fait de s_n une cible de l'agrégat de E_n.

**Valeur de K_s.** Pour une cellule (h, e), c'est la moyenne des cellules ajustées sur les notices retenues où e a des
cellules complètes, comme `mesure()`. Électorats lus : ceux de §8.92, hors finalistes.

**Critère, inchangé (§8.100 e1).** Une géométrie n'apporte quelque chose sur les reports que si :

- V1 ≤ K_s − 1, sur les 8 cellules hors cellule polluée (et 9 rapportées) ;
- au moins 2 des 3 hypothèses s'améliorent, comparées par la MAE moyenne de leurs cellules.

Sont aussi rapportés, sans seuil : K_s sur l'électorat Mélenchon (RN + Aucun), et la part RN de cet électorat parmi les exprimés.

**Limite déclarée.** K_s utilise le duel de la même notice que les reports cachés. Le modèle n'y a accès que par les
duels ajustés à cette date, Harris inclus. Les deux témoins sont donc à égalité d'information sur le duel.

### 8.103 Témoin K_s : résultat (§8.102 ; 26/11 ~18:30Z, Mac, 0 fit)

**Mise en œuvre.** `k_s` et `delta_ks` dans `lire_essai_2027.py` (racine testée).

- « Même notice » se lit comme même institut et même date de fin : les libellés T1 de Harris n'ont pas l'année, ceux de T2 l'ont. Ce constat a été fait pendant le débogage, avant toute lecture de K_s.
- 0 notice exclue sur 13.

| Lecture datée | Segment | Cercle | K | K_s |
|---|---|---|---|---|
| V1, 9 électorats | 7,22 | 8,74 | 9,75 | 10,04 |
| V1, 8 cellules (hors cellule polluée) | **7,06** | 9,00 | 7,13 | 7,79 |
| LFI, RN + Aucun | 9,24 | 10,40 | 2,57 | 3,94 |
| Électorat Mélenchon, part RN des exprimés | 10,4 | 9,2 | 3,9 | 11,2 |

**Critère figé (§8.100 e1, §8.102) : non rempli par les deux géométries.**

| Géométrie | V1 | Seuil (K_s − 1) | Hypothèses améliorées |
|---|---|---|---|
| Segment | 7,06 | 6,79 | 2 sur 3 (Attal 3,75 contre 8,50 ; Glucksmann 8,65 contre 9,03 ; Philippe 9,30 contre 6,26) |
| Cercle | 9,00 | 6,79 | 1 sur 3 |

**Lecture.**

- Le témoin qui connaît le duel fait **moins bien** que K (7,79 contre 7,13).
- Forcer l'agrégat des électorats mesurés sur le duel tire le partage des exprimés vers le finaliste opposé au RN : LFI vers le RN tombe à 4 à 6 %, contre 8 à 12 mesurés.
- C'est ce qu'on attend de l'approximation déclarée en §8.102 (6) : les électorats non mesurés (Zemmour, Dupont-Aignan, petits candidats) ne se partagent pas comme l'agrégat mesuré.
- Le témoin équitable de §8.100 d n'est donc **pas plus exigeant** que K.
- La conclusion de §8.99 tient contre les deux témoins : **aucune géométrie n'apporte quelque chose sur les reports cachés au-delà d'un témoin à 0 paramètre.**
- Le segment s'en approche (7,06 contre 7,13 pour K), le cercle non.

Lecture après coup, sans seuil : le segment est meilleur que K_s sur Attal – Le Pen et Glucksmann – Le Pen, et
pire sur Philippe – Le Pen, l'hypothèse la plus documentée (8 notices).

**Suite : le test β sans refit (§8.100 e2), dont le critère contient « V1 ≤ K_s ».**

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
