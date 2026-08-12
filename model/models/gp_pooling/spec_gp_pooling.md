# Spec — `gp-pooling`

Notice technique du modèle. Les mesures qui ont motivé ces choix sont dans les
backtests (`model/backtest/`) et le README ; ici, seulement le modèle.

---

## 1. Notations

$K$ candidatures, indice $s$. Le temps est un **horizon** $h \ge 0$, en jours
avant le scrutin ($h=0$ le jour du vote).

Les compositions vivent sur le simplexe et sont manipulées en coordonnées
**CLR** (log-ratio centré) :

$$\mathrm{clr}(y)_s = \log y_s - \frac1K\sum_{j}\log y_j, \qquad \sum_s \mathrm{clr}(y)_s = 0$$

d'inverse $y = \mathrm{softmax}(\mathrm{clr}(y))$. Toutes les hypothèses de
sondage retenues portent sur le **même champ** de $K$ candidatures
(`filter_scenarios_by_exact_slots`) — le CLR n'est défini qu'à champ constant.

## 2. État latent

$$\theta_s(\cdot) \sim \mathcal{GP}(\mu_s,\ k), \qquad
k(h,h') = \sigma^2 \exp\!\big(-|h-h'|/\tau\big)$$

indépendamment par $s$. C'est la covariance d'un processus d'Ornstein-Uhlenbeck
stationnaire, d'où

$$\mathrm{Var}\big(\theta_s(h)-\theta_s(h')\big) = 2\sigma^2\big(1-e^{-|h-h'|/\tau}\big)$$

bornée par $2\sigma^2$ : la dispersion sature, elle ne croît pas indéfiniment.

$\mu_s$ est **inconnu**, de prior plat, et marginalisé analytiquement (krigeage
universel, §4).

## 3. Observations

Sondage $p$, d'horizon $h_p$, d'institut $i(p)$, d'échantillon $n_p$, de parts
observées $\hat y_p$. On pose $z_p = \mathrm{clr}(\hat y_p)$ et

$$z_{p,s} = \theta_s(h_p) + u_{i(p),s} + w_{p,s} + \varepsilon_{p,s}$$

| terme | loi | portée |
|---|---|---|
| $u_{i,s}$ | $\mathcal N(0,\sigma_h^2)$ | partagé par tous les sondages d'un institut |
| $w_{p,s}$ | $\mathcal N(0,\sigma_p^2)$ | propre à un sondage |
| $\varepsilon_{p,s}$ | $\mathcal N(0, v_{p,s})$ | bruit d'échantillonnage |

Bruit d'échantillonnage, par la méthode delta sur un multinomial, avec
$P = I - \tfrac1K\mathbb 1\mathbb 1^\top$ :

$$\mathrm{Cov}(\varepsilon_p) = P\,\mathrm{diag}\!\big(\tfrac1{n_p \hat y_p}\big)\,P,
\qquad
v_{p,s} = \frac{1}{n_p\hat y_{p,s}}\Big(1-\frac2K\Big) + \frac{1}{K^2}\sum_j \frac{1}{n_p\hat y_{p,j}}$$

Seule la diagonale est retenue ; le second terme conserve l'essentiel de l'effet
simplexe (un petit candidat très bruité gonfle la variance de tous les autres).

## 4. Postérieur

Pour un slot $s$, avec $P$ sondages, $z=(z_{p,s})_p$ et

$$\Sigma = \underbrace{\big[\sigma^2 e^{-|h_p-h_q|/\tau}\big]_{pq}}_{K_{\text{OU}}}
+ \sigma_h^2\,\Delta + \mathrm{diag}(v_{\cdot,s}+\sigma_p^2),
\qquad \Delta_{pq} = \mathbb 1\{i(p)=i(q)\}$$

$$k_*(h) = \big(\sigma^2 e^{-|h-h_p|/\tau}\big)_p, \qquad
a = \mathbb 1^\top\Sigma^{-1}\mathbb 1, \qquad
\hat\mu_s = \frac{\mathbb 1^\top\Sigma^{-1}z}{a}$$

$$\boxed{\;
\mathbb E\big[\theta_s(h_*)\mid z\big] = \hat\mu_s + k_*^\top\Sigma^{-1}\big(z-\hat\mu_s\mathbb 1\big),
\qquad
\mathrm{Var} = \sigma^2 - k_*^\top\Sigma^{-1}k_* + \frac{\big(1-\mathbb 1^\top\Sigma^{-1}k_*\big)^2}{a}\;}$$

Le dernier terme est le coût de l'ignorance sur $\mu_s$. Une factorisation de
Cholesky $P\times P$ par slot suffit : pas d'échantillonnage.

**Loi prédictive d'un sondage** (pour les tests de couverture). Si l'on veut la
loi de ce qu'observerait un sondage cible d'institut $i_*$ et de variance
$v_*$, on remplace $k_*$ par $k_*^{\text{obs}} = k_* + \sigma_h^2\,\mathbb 1\{i(p)=i_*\}$
et $\sigma^2$ par $\sigma^2+\sigma_h^2+v_{*,s}$ dans l'expression de la variance.

## 5. Du scrutin au résultat

Deux quantités distinctes, estimées séparément :

$$\underbrace{\theta_s(h_{\text{as\_of}}) \longrightarrow \theta_s(0)}_{\text{diffusion, §2}}
\qquad\text{puis}\qquad
\underbrace{r_s = \theta_s(0) + \delta_s}_{\text{écart sondages-urne}}$$

$\theta_s(0)$ sort du **même** postérieur (§4) évalué en $h_*=0$. L'écart
terminal $\delta$ ne dépend pas de l'horizon :

$$\delta_s \sim \mathrm{SinhArcsinh}(0,\ \sigma_\delta,\ \gamma,\ \eta)$$

famille asymétrique à queues ajustables. Il est calibré comme **résidu**, à
diffusion fixée : sur les mouvements historiques $m_i$ mesurés à l'horizon $h_i$,

$$m_i \sim \mathrm{SinhArcsinh}\Big(0,\ \sqrt{2\sigma^2\big(1-e^{-h_i/\tau}\big) + \sigma_\delta^2},\ \gamma,\ \eta\Big)$$

Moyenne nulle imposée : les $m_i$ sont des CLR, donc de moyenne exactement nulle
par construction.

Parts finales : $\pi = \mathrm{softmax}\big(\theta(0)+\delta\big)$, puis comptage
des probabilités (`model/core/simulate.py`).

## 6. Paramètres

| paramètre | valeur | estimation |
|---|---|---|
| $\tau$ | 261,7 j | REML, `calibration.py` |
| $\sigma$ | 0,340 | REML |
| $\sigma_h$ | 0,126 | REML, IC 95 % ≈ [0,107 ; 0,148] |
| $\sigma_p$ | 0,039 | REML |
| $\sigma_\delta$ | 0,419 | MCMC, `terminal.py` |
| $\gamma$ (asymétrie) | +0,030 | MCMC |
| $\eta$ (queues) | 0,960 | MCMC |

REML sur 15 blocs (élection × champ exact), 215 sondages, 2 410 observations
slot × sondage, $\mu_s$ marginalisé par slot et vraisemblance composite sur les
slots.

```bash
.venv/bin/python -m model.models.gp_pooling.calibration   # tau, sigma, sigma_h, sigma_p
.venv/bin/python -m model.models.gp_pooling.terminal      # sigma_delta, gamma, eta
```

## 7. Approximations et limites

1. Bruit d'échantillonnage réduit à sa **diagonale** (§3).
2. Slots **indépendants a priori**, recouplés seulement par le softmax final :
   la cohérence compositionnelle des tirages n'est pas exacte.
3. **Approximation gaussienne** du bruit en CLR, au lieu d'un multinomial exact ;
   plancher des parts à 0,3 % pour borner le CLR.
4. $\delta$ **indépendant entre candidats** et de dispersion commune.
5. L'ordonnée à l'origine de §5 n'est pas identifiable sur le pool seul : aucun
   mouvement historique n'y est mesuré à moins de 59 jours du scrutin.
