# Theoretical Analysis: Irrelevant Variables and the Curse of Dimensionality

**Course:** MAC 312 — Foundations of Data Science

---

## 1. Data-Generating Model and Comparability Across $r$

Each observation is $x = (z, u) \in \mathbb{R}^d$, with binary label $y \in \{0, 1\}$, $P(y=0) = P(y=1) = 0.5$:
- **Informative coordinates** $z \in \mathbb{R}^r$:
  $$z \mid (y = 0) \sim \mathcal{N}(-\delta \mathbf{1}_r, \sigma_z^2 I_r), \qquad z \mid (y = 1) \sim \mathcal{N}(+\delta \mathbf{1}_r, \sigma_z^2 I_r)$$
- **Irrelevant coordinates** $u \in \mathbb{R}^{d-r}$:
  $$u \sim \mathcal{N}(0, \sigma_u^2 I_{d-r}), \qquad u \perp\!\!\!\perp (z, y)$$

Because $u$ is independent of $(z, y)$, the true posterior depends only on $z$:
$$\log \frac{P(y = 1 \mid x)}{P(y = 0 \mid x)} = \log \frac{P(y = 1 \mid z)}{P(y = 0 \mid z)} = \frac{2\delta}{\sigma_z^2} \sum_{j=1}^r z_j$$

### Keeping the Distribution Comparable When $r$ Changes
If $\delta$ were kept constant as $r$ increases, the class separation $\|\mu_1 - \mu_0\|_2 = 2\delta\sqrt{r}$ would grow with $r$, making higher-$r$ problems artificially easier.

Instead, we set:
$$\delta = \frac{\mu_0}{\sqrt{r}} \sigma_z \qquad (\mu_0 = 1.2)$$

Define the standardized projection $s(z) = \frac{1}{\sigma_z \sqrt{r}} \sum_{j=1}^r z_j$. Then:
$$s(z) \mid (y = 0) \sim \mathcal{N}(-\mu_0, 1), \qquad s(z) \mid (y = 1) \sim \mathcal{N}(+\mu_0, 1)$$
and the posterior is simply $P(y = 1 \mid z) = \sigma(2\mu_0 s(z))$. Consequently:
- The **Bayes optimal error rate** is $\Phi(-\mu_0) = \Phi(-1.2) \approx 11.51\%$ for **every** $r$ and **every** scale $\sigma_z$.
- The **Bayes optimal log-loss** is also identical across all $r$ and $\sigma_z$.

---

## 2. Moments of Squared Distances and Distance Concentration

Consider two independent observations $x = (z, u)$ and $x' = (z', u')$. Their squared Euclidean distance splits cleanly into informative and irrelevant parts:
$$\|x - x'\|^2 = \underbrace{\|z - z'\|^2}_{\text{signal}} + \underbrace{\|u - u'\|^2}_{\text{noise}}$$

### 2.1 Distribution and Moments of the Noise Term
Since $u_j - u'_j \sim \mathcal{N}(0, 2\sigma_u^2)$ independently for $j = 1, \dots, d - r$:
$$\frac{\|u - u'\|^2}{2\sigma_u^2} \sim \chi^2(d - r)$$
Using $\mathbb{E}[\chi^2(k)] = k$ and $\operatorname{Var}(\chi^2(k)) = 2k$:
$$\mathbb{E}\!\left[\|u - u'\|^2\right] = 2(d - r)\sigma_u^2, \qquad \operatorname{Var}\!\left(\|u - u'\|^2\right) = 8(d - r)\sigma_u^4$$

For isotropic Gaussian vectors ($X, X' \sim \mathcal{N}(0, I_d)$), the relative spread (coefficient of variation) shrinks as:
$$\mathrm{CV}\!\left(\|X - X'\|^2\right) = \frac{\sqrt{8d}}{2d} = \sqrt{\frac{2}{d}} \xrightarrow{d \to \infty} 0$$
So pairwise distances concentrate in a thin shell around $\sqrt{2d}$, and the nearest-to-farthest distance contrast
$$C = \frac{D_{\text{far}} - D_{\text{near}}}{D_{\text{near}}}$$
collapses toward $0$ as $d$ grows.

### 2.2 Same-Class vs. Opposite-Class Squared Distances (Why Classification Fails)
Now include the informative coordinates $z$ with $\delta = \frac{\mu_0}{\sqrt{r}}\sigma_z$:

- **Same class ($y = y'$):** $z_j - z'_j \sim \mathcal{N}(0, 2\sigma_z^2)$, so
  $$\mathbb{E}\!\left[\|x - x'\|^2 \mid y = y'\right] = 2r\sigma_z^2 + 2(d - r)\sigma_u^2$$
- **Opposite class ($y \neq y'$):** $z_j - z'_j \sim \mathcal{N}(\pm 2\delta, 2\sigma_z^2)$, so each coordinate adds $(2\delta)^2 = 4\delta^2$ to the mean squared distance:
  $$\mathbb{E}\!\left[\|x - x'\|^2 \mid y \neq y'\right] = \underbrace{4\mu_0^2 \sigma_z^2}_{\text{between-class gap}} + \; 2r\sigma_z^2 + 2(d - r)\sigma_u^2$$

Notice two critical facts from these two equations:
1. The **between-class mean gap** in squared distance is fixed at $4\mu_0^2 \sigma_z^2$ (independent of $d$ and $r$).
2. The **random fluctuation (standard deviation)** of the noise squared distance is $\sqrt{8(d - r)}\,\sigma_u^2$, which grows as $\sqrt{d - r}$ and scales with $\sigma_u^2$.

Define the distance signal-to-noise ratio:
$$\text{SNR}_{\text{dist}} = \frac{\mathbb{E}[\|x - x'\|^2 \mid y \neq y'] - \mathbb{E}[\|x - x'\|^2 \mid y = y']}{\sqrt{\operatorname{Var}(\|u - u'\|^2)}} = \frac{4\mu_0^2 \sigma_z^2}{\sqrt{8(d - r)}\,\sigma_u^2} = \frac{\sqrt{2}\,\mu_0^2}{\sqrt{d - r}} \left(\frac{\sigma_z}{\sigma_u}\right)^2$$

Whenever $\text{SNR}_{\text{dist}} \ll 1$, random fluctuations from the $d - r$ irrelevant coordinates easily exceed the true between-class gap $4\mu_0^2 \sigma_z^2$. As a result, a query point's nearest neighbors (and highest-weight KDE terms) come from the wrong class almost as often as the right class, driving KNN and Gaussian KDE error rates upward toward random guessing ($50\%$).

---

## 3. Regimes of Standardization

Standardizing replaces each coordinate $x_j$ by $\tilde{x}_j = (x_j - \hat{\mu}_j) / \hat{s}_j$ using training mean and standard deviation.

What is the population marginal variance of each coordinate?
- For an irrelevant coordinate: $\operatorname{Var}(u_j) = \sigma_u^2$. After standardization, $\tilde{u}_j$ has variance $1$.
- For an informative coordinate (by the law of total variance across both classes):
  $$\operatorname{Var}(z_j) = \sigma_z^2 + \delta^2 = \sigma_z^2\!\left(1 + \frac{\mu_0^2}{r}\right)$$
  After standardization, the between-class gap in squared distance becomes:
  $$\Delta_{\text{std}} = \frac{4\mu_0^2}{1 + \mu_0^2 / r}$$
  while the noise fluctuation is $\sqrt{8(d - r)}$ — **completely independent of the original $\sigma_z$ and $\sigma_u$**.

This yields three exact regimes:

1. **Noise-Dominated Scales ($\sigma_u \gg \sigma_z$, e.g., $\sigma_u = 10, \sigma_z = 1$):**
   - **Raw:** $\left(\sigma_z/\sigma_u\right)^2 = 1/100$. For $d=50, r=2$, noise contributes $2(48)(100) = 9600$ to expected squared distance versus $4\mu_0^2 \approx 5.76$ for the class gap. Raw KNN and KDE collapse to $\approx 50\%$ error.
   - **Standardization helps dramatically** by removing the $100\times$ variance imbalance, **but cannot remove the 48 irrelevant coordinates**: noise still contributes $\sqrt{8 \cdot 48} \approx 19.6$ standard deviation against $\Delta_{\text{std}} \approx 3.35$, leaving KNN/KDE well above Bayes optimal.
2. **Equal Scales ($\sigma_u = \sigma_z = 1$):**
   - **Standardization cannot help.** Because $\operatorname{Var}(z_j) = \sigma_z^2 + \delta^2 > \sigma_z^2$, dividing $z_j$ by its marginal standard deviation actually shrinks the informative coordinates slightly relative to unit-variance noise ($\Delta_{\text{std}} = 3.35 < 5.76$), slightly *increasing* KNN/KDE error compared to raw equal-scale data.
3. **Signal-Dominated Scales ($\sigma_z \gg \sigma_u$, e.g., $\sigma_z = 10, \sigma_u = 1$):**
   - **Raw:** $\left(\sigma_z/\sigma_u\right)^2 = 100$, so the informative coordinates naturally dominate Euclidean distance and Raw KNN/KDE achieve near-oracle performance.
   - **Standardization actively hurts**, because it scales the 48 small noise coordinates *up* to unit variance, restoring the high-dimensional noise floor.

In all three regimes, only **supervised feature weighting** (down-weighting coordinates with small training ANOVA $F$-scores) or **training-only feature selection** (discarding low-$F$ noise coordinates) removes the $(d-r)$ noise term and recovers near-oracle accuracy.

---

## 4. Why the Four Classifiers Behave Differently

- **KNN & Gaussian KDE (local distance-based):** Depend directly on Euclidean distances across all $d$ coordinates. Both degrade rapidly as $d$ grows or when $\sigma_u \gg \sigma_z$. Increasing sample size $n$ helps only very slowly because filling a $d$-dimensional ball requires $n$ exponential in $d$.
- **Regularized Gaussian Classifier (QDA with shrinkage):** Estimates $\hat{\mu}_c$ and $\hat{\Sigma}_c^{\text{reg}} = (1-\alpha) S_c + \alpha \frac{\operatorname{Tr}(S_c)}{d} I_d$. Because noise coordinates have identical true means ($\mu_{0,j} = \mu_{1,j} = 0$) and equal variances, shrinkage prevents covariance overfitting when $d$ is large relative to $n$. *(Note: isotropic shrinkage assumes comparable feature scales; on raw Noise-Dominated data, $\operatorname{Tr}(S_c)/d \approx 96$ washes out the small-scale informative coordinates unless features are standardized first.)*
- **L2 Logistic Regression:** Learns a linear weight vector $w = (w_z, w_u)$. Because $u \perp y$, L2 regularization shrinks the irrelevant weights $w_u \to 0$ while preserving $w_z$, making it resilient to both high ambient dimension $d$ and large raw noise scales.
