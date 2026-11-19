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
