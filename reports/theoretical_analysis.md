# Theoretical Analysis: Irrelevant Variables and the Curse of Dimensionality

**Course:** Foundations of Data Science / Statistical Machine Learning  
**Author:** Research Assistant  
**Topic:** Mathematical Derivations and Theoretical Foundations of High-Dimensional Concentration and Learning Dynamics

---

## 1. Introduction and Mathematical Formulation

In statistical machine learning, the **Curse of Dimensionality** refers to an interrelated set of mathematical and geometric phenomena that emerge when the ambient dimension $d$ of a feature space grows large relative to the number of available observations $n$. When irrelevant (uninformative) features are introduced into a classification problem, the geometry of the input space deteriorates significantly, particularly for distance-based and local non-parametric estimators.

We formalize the data-generating process as follows:
Let the observed feature vector $X \in \mathbb{R}^d$ be partitioned into:
$$
X = \begin{bmatrix} z \\ u \end{bmatrix}
$$
where:
1. $z \in \mathbb{R}^r$ represents the **informative coordinates** ($r \ge 1$), which carry statistical signal about the binary class label $y \in \{0, 1\}$.
2. $u \in \mathbb{R}^{d-r}$ represents the **irrelevant noise coordinates**, which are statistically independent of both the informative features $z$ and the label $y$:
   $$
   u \perp\!\!\!\perp (z, y)
   $$
3. The class conditional distributions for the informative features are given by:
   $$
   z \mid (y = 0) \sim \mathcal{N}(-\boldsymbol{\delta}, \sigma_z^2 I_r), \quad z \mid (y = 1) \sim \mathcal{N}(+\boldsymbol{\delta}, \sigma_z^2 I_r)
   $$
   with equal class priors $P(y = 0) = P(y = 1) = 0.5$.
4. The irrelevant noise features are independently distributed:
   $$
   u_j \sim \mathcal{N}(0, \sigma_u^2), \quad j = 1, \dots, d - r
   $$

---

## 2. Derivation of Squared Euclidean Distance Statistics

### 2.1 Independent Standard Gaussian Vectors

Consider two independent observations $X, X' \in \mathbb{R}^d$ drawn from a standard multivariate Gaussian distribution:
$$
X, X' \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, I_d)
$$

The pairwise difference vector is:
$$
D = X - X'
$$

Because linear combinations of independent Gaussian random vectors remain Gaussian:
$$
\mathbb{E}[D] = \mathbb{E}[X] - \mathbb{E}[X'] = 0 - 0 = 0
$$
$$
\operatorname{Cov}(D) = \operatorname{Cov}(X) + \operatorname{Cov}(X') = I_d + I_d = 2 I_d
$$
Hence:
$$
D = X - X' \sim \mathcal{N}(0, 2 I_d)
$$

Each coordinate $D_j = X_j - X'_j$ is an independent univariate Gaussian:
$$
D_j \sim \mathcal{N}(0, 2) \implies \frac{D_j}{\sqrt{2}} \sim \mathcal{N}(0, 1)
$$

The squared Euclidean distance between $X$ and $X'$ is:
$$
\|X - X'\|^2 = \sum_{j=1}^d (X_j - X'_j)^2 = \sum_{j=1}^d D_j^2
$$

Dividing by 2 yields:
$$
\frac{\|X - X'\|^2}{2} = \sum_{j=1}^d \left( \frac{D_j}{\sqrt{2}} \right)^2
$$

Since each term $\left( \frac{D_j}{\sqrt{2}} \right)^2$ is the square of an independent standard normal random variable, their sum follows a Chi-square distribution with $d$ degrees of freedom:
$$
\frac{\|X - X'\|^2}{2} \sim \chi^2(d)
$$

### 2.2 Expectation and Variance

From the properties of the $\chi^2(d)$ distribution:
- For $W \sim \chi^2(d)$:
  $$
  \mathbb{E}[W] = d, \quad \operatorname{Var}(W) = 2d
  $$

Setting $W = \frac{\|X - X'\|^2}{2}$, we have $\|X - X'\|^2 = 2W$. Therefore:
$$
\mathbb{E}\left[ \|X - X'\|^2 \right] = \mathbb{E}[2W] = 2 \mathbb{E}[W] = 2d
$$
$$
\operatorname{Var}\left( \|X - X'\|^2 \right) = \operatorname{Var}(2W) = 4 \operatorname{Var}(W) = 4(2d) = 8d
$$

The standard deviation is:
$$
\sigma_{\|X - X'\|^2} = \sqrt{8d} = 2\sqrt{2d}
$$

### 2.3 Coefficient of Variation and Distance Concentration

The **Coefficient of Variation** ($\mathrm{CV}$) measures the relative dispersion of the squared distance distribution:
$$
\mathrm{CV}\left( \|X - X'\|^2 \right) = \frac{\sqrt{\operatorname{Var}(\|X - X'\|^2)}}{\mathbb{E}[\|X - X'\|^2]} = \frac{\sqrt{8d}}{2d} = \frac{2\sqrt{2}\sqrt{d}}{2d} = \sqrt{\frac{2}{d}}
$$

As the ambient dimension $d \to \infty$:
$$
\lim_{d \to \infty} \mathrm{CV}\left( \|X - X'\|^2 \right) = \lim_{d \to \infty} \sqrt{\frac{2}{d}} = 0
$$

By Chebyshev's inequality, for any $\varepsilon > 0$:
$$
P\left( \left| \frac{\|X - X'\|^2 - 2d}{2d} \right| \ge \varepsilon \right) \le \frac{\operatorname{Var}(\|X - X'\|^2)}{(2d)^2 \varepsilon^2} = \frac{8d}{4d^2 \varepsilon^2} = \frac{2}{d \varepsilon^2} \xrightarrow{d \to \infty} 0
$$

This proves that as $d \to \infty$, the normalized squared distance concentrates almost entirely around its mean value $2d$:
$$
\frac{\|X - X'\|^2}{2d} \xrightarrow{P} 1
$$

---

## 3. Generalization to Unequal Feature Variances

Now consider the case where coordinates have heterogeneous variances $\sigma_j^2$:
$$
X_j, X'_j \sim \mathcal{N}(0, \sigma_j^2), \quad j = 1, \dots, d
$$
Then:
$$
D_j = X_j - X'_j \sim \mathcal{N}(0, 2\sigma_j^2) \implies \frac{D_j}{\sqrt{2}\sigma_j} \sim \mathcal{N}(0, 1)
$$
The squared distance is a weighted sum of independent $\chi^2(1)$ random variables:
$$
\|X - X'\|^2 = \sum_{j=1}^d D_j^2 = \sum_{j=1}^d 2\sigma_j^2 \left( \frac{D_j}{\sqrt{2}\sigma_j} \right)^2
$$

The expectation and variance are:
$$
\mathbb{E}\left[ \|X - X'\|^2 \right] = \sum_{j=1}^d 2\sigma_j^2 = 2 \sum_{j=1}^d \sigma_j^2
$$
$$
\operatorname{Var}\left( \|X - X'\|^2 \right) = \sum_{j=1}^d 4 \sigma_j^4 \operatorname{Var}\left( \chi^2(1) \right) = 8 \sum_{j=1}^d \sigma_j^4
$$

### Application to Experiment 3: Large Irrelevant Variance
Suppose $r = 2$ informative features have $\sigma_z = 1$, and $d - r = 48$ irrelevant noise features have $\sigma_u = 10$.
- Expected contribution of informative features:
  $$
  2 \sum_{j=1}^2 \sigma_z^2 = 2(1 + 1) = 4
  $$
- Expected contribution of irrelevant features:
  $$
  2 \sum_{j=3}^{50} \sigma_u^2 = 2 \times 48 \times 100 = 9600
  $$
- Variance contribution of informative features:
  $$
  8 \sum_{j=1}^2 \sigma_z^4 = 8(1 + 1) = 16
  $$
- Variance contribution of irrelevant features:
  $$
  8 \sum_{j=3}^{50} \sigma_u^4 = 8 \times 48 \times 10000 = 3{,}840{,}000
  $$

The noise features dominate the expected squared distance by a factor of $\frac{9600}{4} = 2400$, and dominate the variance by a factor of $240{,}000$!  
In raw Euclidean space, the meaningful signal along the first two coordinates is completely swamped by the random fluctuations of the 48 high-variance noise dimensions.

---

## 4. The Beyer et al. (1999) Theorem and Distance Contrast Degradation

Beyer, Goldstein, Ramakrishnan, and Shaft (1999) established the foundational theorem formalizing the loss of contrast in high dimensions.

Let $D_{\min} = \min_{i=1,\dots,n} \|X_{\text{query}} - X_i\|$ and $D_{\max} = \max_{i=1,\dots,n} \|X_{\text{query}} - X_i\|$.

**Theorem (Beyer et al., 1999):**  
If:
$$
\lim_{d \to \infty} \frac{\operatorname{Var}(D)}{\mathbb{E}[D]^2} = 0
$$
then for any fixed sample size $n \ge 2$:
$$
\frac{D_{\max} - D_{\min}}{D_{\min}} \xrightarrow{P} 0 \quad \text{as } d \to \infty
$$

### Definition of Distance Contrast
In our evaluation framework, distance contrast is defined as:
$$
C = \frac{D_{\text{far}} - D_{\text{near}}}{D_{\text{near}}}
$$
As $d$ increases:
1. Both $D_{\text{near}}$ and $D_{\text{far}}$ grow on the order of $\Theta(\sqrt{d})$.
2. The difference $D_{\text{far}} - D_{\text{near}}$ grows at a much slower rate $O(1)$ or $O(\sqrt{\log n})$.
3. Therefore, the relative difference $C \to 0$.

When $C \approx 0$, all training points become equidistant from the query point. The concept of "nearest neighbor" becomes statistically meaningless because the nearest point is barely closer than the farthest point, and the ranking of neighbors is dominated by accumulated noise.

---

## 5. Algorithmic Impact Across the Four Classifiers

### 5.1 K-Nearest Neighbours (KNN)
- **Mechanism:** KNN determines class membership based on the labels of the $k$ training points minimizing Euclidean distance $\|x - x_i\|_2$.
- **Failure Mode:** Decomposing $\|x - x_i\|^2 = \|z - z_i\|^2 + \|u - u_i\|^2$:
  As $d - r \to \infty$, $\|u - u_i\|^2 \sim 2(d-r) \pm \sqrt{8(d-r)}$.
  The random noise $\|u - u_i\|^2$ overwhelms the informative separation $\|z - z_i\|^2$.
  Consequently, the identified $k$ nearest neighbors in $\mathbb{R}^d$ are essentially random uniformly drawn samples from the training set with respect to the informative space $\mathbb{R}^r$.
- **Performance:** Error rate deteriorates toward random guessing ($50\%$ for balanced binary classes).

### 5.2 Gaussian Kernel Density Classifier (GKDC)
- **Mechanism:** Predicts posterior class probabilities via Bayes' rule using kernel density estimates:
  $$
  p(x \mid y=c) = \frac{1}{N_c (2\pi)^{d/2} h^d} \sum_{i \in I_c} \exp\left( -\frac{\|x - x_i\|^2}{2h^2} \right)
  $$
- **Failure Mode:**
  1. *Curse of Density Estimation:* To achieve an estimation error of $\epsilon$, non-parametric KDE requires sample size scaling exponentially with dimension:
     $$
     n = \mathcal{O}\left( \epsilon^{-(d + 4)} \right)
     $$
  2. *Exponent Underflow & Kernel Flattop:* If bandwidth $h$ does not grow with $\sqrt{d}$, all exponent arguments $-\frac{\|x - x_i\|^2}{2h^2}$ become extreme negative numbers, causing severe underflow. If $h$ is tuned very large ($h \sim \sqrt{d}$), the Gaussian kernel becomes completely flat, assigning equal likelihood to all observations regardless of true class differences.
- **Log-sum-exp stabilization:** While computing $\log p(x \mid y=c)$ in log-space prevents floating-point underflow, it cannot restore the loss of local density information in high dimensions.

### 5.3 Gaussian Generative Classifier (Regularized QDA)
- **Mechanism:** Models class conditional distributions as $\mathcal{N}(\mu_c, \Sigma_c)$.
- **Behavior in High Dimensions:**
  1. In the true distribution, $\Sigma_c = I_d$ and the irrelevant coordinates have identical means $\mu_{0, j} = \mu_{1, j} = 0$.
  2. The empirical sample covariance $S_c$ in $d$ dimensions requires estimating $\frac{d(d+1)}{2}$ parameters. When $d=100$, each class has 500 training points and must estimate $5{,}050$ covariance parameters.
  3. Covariance shrinkage $\Sigma_c^{\text{reg}} = (1-\alpha) S_c + \alpha \frac{\operatorname{Tr}(S_c)}{d} I_d$ regularizes the estimated covariance toward spherical geometry.
  4. While empirical sample noise in irrelevant mean estimation introduces variance, the Gaussian classifier can remain relatively robust when covariance is correctly regularized because the generative model averages evidence across all samples.

### 5.4 Logistic Regression (L2 Regularized)
- **Mechanism:** Models the log-odds as a linear hyperplane:
  $$
  \log \frac{P(y=1 \mid x)}{P(y=0 \mid x)} = w^T X + b = w_z^T z + w_u^T u + b
  $$
- **L2 Shrinkage:** The objective is:
  $$
  \min_{w, b} \mathcal{L}_{\text{logistic}}(w, b) + \frac{1}{2C} \|w\|_2^2
  $$
- **Why Logistic Regression is Most Robust:**
  1. Unlike distance-based methods that sum squared distances across all $d$ dimensions, the log-odds is *linear* in the features.
  2. The irrelevant features $u$ have zero correlation with $y$ in population: $\mathbb{E}[u_j (2y - 1)] = 0$.
  3. L2 regularization actively penalizes weights $w_u$ corresponding to irrelevant features, shrinking them toward zero:
     $$
     w_{u, j} \approx 0, \quad \forall j > r
     $$
  4. The model effectively projects out the orthogonal noise subspace, preserving the high classification accuracy of the informative subvector $z$.

---

## 6. Preprocessing and Feature Selection Theory (Experiment 3)

### 6.1 Why Standardization Does NOT Eliminate Irrelevant Dimensions
Let $X_j$ have variance $\sigma_j^2$. Standardizing transforms $X_j \to \tilde{X}_j = \frac{X_j - \mu_j}{\sigma_j}$, yielding unit variance for all features:
$$
\operatorname{Var}(\tilde{X}_j) = 1, \quad \forall j = 1, \dots, d
$$
While standardization equalizes the scales and prevents high-variance noise features ($\sigma_u = 10$) from drowning out informative features by scale alone, the standardized feature vector remains in $\mathbb{R}^d$.
The squared distance on standardized features is:
$$
\|\tilde{X} - \tilde{X}'\|^2 = \sum_{j=1}^r (\tilde{z}_j - \tilde{z}'_j)^2 + \sum_{j=r+1}^d (\tilde{u}_j - \tilde{u}'_j)^2
$$
The noise term still contains $d - r$ independent $\chi^2(1)$ components with expected sum $2(d - r)$. Thus, standardization cures scale imbalance, but cannot eliminate the dimensional curse of irrelevant features.

### 6.2 Feature Weighting and Feature Selection
- **Feature Selection:** By testing statistical independence (e.g., via ANOVA $F$-test $F_j = \frac{\text{MSB}_j}{\text{MSW}_j}$), coordinates with $F$-statistic consistent with the null hypothesis are discarded. In our setting, the top $k=2$ selected features isolate the true coordinates $z_1, z_2$, restoring near-oracle performance.
- **Oracle Baseline:** The theoretical upper bound for classification accuracy, using solely $z \in \mathbb{R}^r$ without exposure to any irrelevant dimensions.

---

## 7. Sample Size Scaling Theory (Experiment 2)

To maintain a constant average distance between adjacent neighbors in $d$ dimensions, the volume of the space scales as $V \propto R^d$.
To fill this volume with equal density, the required sample size must scale as:
$$
n \propto \left( \frac{1}{\epsilon} \right)^d
$$
For $d = 50$, doubling the sample size from $n = 1000$ to $n = 2000$ or even $n = 5000$ achieves a linear gain in sample density of $5\times$, whereas maintaining the $d=2$ neighborhood density would require $n \sim 1000^{25} = 10^{75}$ points!
Hence, in high dimensions ($d=50$), increasing sample size within computationally feasible limits provides modest reductions in variance but cannot overcome the fundamental distance concentration of the 48 irrelevant dimensions for KNN and KDE.

