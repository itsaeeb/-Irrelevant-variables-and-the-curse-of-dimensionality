# Irrelevant Variables and the Curse of Dimensionality

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An academically rigorous statistical machine learning research project investigating how irrelevant noise variables, high ambient dimensionality, unequal feature scales, and sample sizes affect the performance and geometric behavior of classification algorithms.

---

## 1. Project Overview & Research Objectives

In real-world data science problems, practitioners often measure many candidate variables, only a subset of which are relevant to the target response. When irrelevant variables (noise features) are added to a dataset:
1. **Geometric Distance Concentration:** Pairwise Euclidean distances between points converge to an increasingly narrow shell, degrading distance contrast:
   $$\lim_{d \to \infty} \frac{D_{\max} - D_{\min}}{D_{\min}} \xrightarrow{P} 0$$
2. **Local Estimator Breakdown:** Non-parametric classifiers that rely on local neighborhood geometry (e.g., K-Nearest Neighbours, Kernel Density Estimation) deteriorate dramatically because nearest neighbors in high-dimensional space are dominated by accumulated noise.
3. **Parametric & Linear Robustness:** Global generative models (Gaussian QDA) and regularized linear classifiers (L2 Logistic Regression) exhibit distinct degradation characteristics and often display greater resilience against orthogonal irrelevant dimensions.
4. **Mitigation Failure & Success:** Standardization scales features to unit variance but **cannot** eliminate independent noise dimensions. In contrast, supervised feature selection and relevance weighting successfully isolate informative coordinates and restore near-oracle classification performance.

---

## 2. Mathematical Data-Generating Process

We formulate a synthetic binary classification setting with strict data isolation:
- Observed feature vector: $X = [z, u] \in \mathbb{R}^d$.
- Informative coordinates: $z \in \mathbb{R}^r$ carrying signal for binary label $y \in \{0, 1\}$.
- Irrelevant coordinates: $u \in \mathbb{R}^{d-r}$ representing independent noise:
  $$u \perp\!\!\!\perp (z, y), \quad u_j \sim \mathcal{N}(0, \sigma_u^2)$$
- Class conditional distributions for informative features:
  $$z \mid (y = 0) \sim \mathcal{N}(-\boldsymbol{\delta}, \sigma_z^2 I_r)$$
  $$z \mid (y = 1) \sim \mathcal{N}(+\boldsymbol{\delta}, \sigma_z^2 I_r)$$
  with equal class priors $P(y = 0) = P(y = 1) = 0.5$.
- Baseline parameters: $r = 2$, $\boldsymbol{\delta} = [1, 1]^T$, $\sigma_z = 1.0$, $\sigma_u = 1.0$.

---

## 3. Implemented Classification Algorithms

| Algorithm | Model Nature | Hyperparameter & Validation Tuning | Key Implementation Details |
|---|---|---|---|
| **K-Nearest Neighbours (KNN)** | Non-parametric, distance-based | Number of neighbors $k \in [1, 3, 5, 7, 11, 15, 21, 31, 51]$ | Vectorized distance matrices via `cdist`, tie-breaking, neighbor class proportions for posterior probabilities. |
| **Gaussian Kernel Density (GKDC)** | Non-parametric generative Bayes | Kernel bandwidth $h \in [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]$ | Log-density evaluation with `logsumexp` to prevent numerical underflow in high dimensions; exact Bayes posterior. |
| **Gaussian Classifier (QDA)** | Parametric generative | Covariance shrinkage $\alpha \in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]$ | Cholesky factorization $L L^T$ for log-determinant and Mahalanobis solves; regularization jitter for numerical stability. |
| **Logistic Regression** | Parametric discriminative | Inverse regularization $C \in [10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0, 10.0, 100.0, 1000.0]$ | L2 regularization via L-BFGS, shrinks uninformative feature coefficients toward zero. |

---

## 4. Evaluation Metrics

For every test observation $x_{\text{test}}^{(i)}$ evaluated against training set $\{x_{\text{train}}^{(j)}\}_{j=1}^n$:
1. **Classification Error Rate:** $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i \neq y_i)$.
2. **Predictive Log-Loss:**
   $$\text{Log-Loss} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$
   with documented probability clipping $\hat{p} = \text{clip}(P(y=1 \mid x), 10^{-15}, 1 - 10^{-15})$.
3. **Nearest-Neighbour Distance:** $D_{\text{near}} = \min_{j} \|x_{\text{test}}^{(i)} - x_{\text{train}}^{(j)}\|_2$.
4. **Farthest-Neighbour Distance:** $D_{\text{far}} = \max_{j} \|x_{\text{test}}^{(i)} - x_{\text{train}}^{(j)}\|_2$.
5. **Distance Contrast:**
   $$C = \frac{D_{\text{far}} - D_{\text{near}}}{D_{\text{near}}}$$
6. **Distance Ratio:** $R = \frac{D_{\text{near}}}{D_{\text{far}}}$.

---

## 5. Summary of Experiments

- **Experiment 1: Ambient Dimension:**
  Fixes $r = 2, n_{\text{train}} = 1000$; varies total dimension $d \in [2, 5, 10, 20, 50, 100]$. Demonstrates geometric distance concentration and error degradation.
- **Experiment 2: Sample Size:**
  Fixes $r = 2, d = 50$; varies training sample size $n_{\text{train}} \in [100, 250, 500, 1000, 2000, 5000]$. Analyzes whether additional samples mitigate the curse of dimensionality.
- **Experiment 3: Unequal Feature Scales:**
  Sets informative feature standard deviation $\sigma_z = 1.0$ and noise standard deviation $\sigma_u = 10.0$ ($d=50, r=2$). Compares 5 pipelines:
  - *Raw*: Features unscaled.
  - *Standardized*: `StandardScaler` fitted strictly on training data.
  - *Weighted*: ANOVA $F$-statistic relevance weights from training data.
  - *Selected*: Training-only feature selection with `SelectKBest` without oracle knowledge.
  - *Oracle*: Benchmark using solely ground-truth informative features $z$.
- **Experiment 4: Informative Dimension:**
  Fixes $d = 50, n_{\text{train}} = 1000, \delta = 1.0$; varies informative dimension $r \in [1, 2, 4, 8]$. Proves how adding signal contrasts with adding ambient noise.

---

## 6. Project Architecture

```
irrelevant-variables-curse-of-dimensionality/
├── data/                                 # Saved synthetic datasets (.npz archives)
├── src/                                  # Modular source package
│   ├── __init__.py
│   ├── data_generation.py                # Synthetic generator with strict split isolation
│   ├── classifiers.py                    # KNN, GKDC, Gaussian QDA, Logistic Regression
│   ├── preprocessing.py                  # Raw, Standardized, Weighted, Selected, Oracle
│   ├── distance_metrics.py               # Pairwise D_near, D_far, contrast C, ratio R
│   ├── evaluation.py                     # Error rate, log-loss with clipping, metric aggregator
│   ├── plotting.py                       # Publication-quality matplotlib/seaborn plots
│   └── experiment_utils.py               # Validation tuning, seed harness, table exporters
├── experiments/                          # Executable experiment scripts
│   ├── experiment_1_ambient_dimension.py
│   ├── experiment_2_sample_size.py
│   ├── experiment_3_feature_scales.py
│   └── experiment_4_informative_dimension.py
├── results/                              # Generated artifacts
│   ├── csv/                              # Raw and summary CSV tables
│   ├── plots/                            # High-resolution PNG figures
│   └── tables/                           # Formatted markdown summary tables
├── reports/                              # Detailed academic reports
│   ├── theoretical_analysis.md           # Mathematical derivations & concentration proofs
│   └── final_report.md                   # Comprehensive final project report with findings
├── main.py                               # CLI entrypoint to reproduce all experiments
├── requirements.txt                      # Project dependencies
└── README.md                             # Documentation and user guide
```

---

## 7. Installation and Execution

### Prerequisites
Python 3.10+ with standard scientific packages:
```bash
pip install -r requirements.txt
```

### Reproduce All Experiments
To execute the complete benchmark pipeline with 5 independent random seeds:
```bash
python main.py --exp all --seeds 5
```

### Run an Individual Experiment
```bash
# Run Experiment 1 (Ambient Dimension)
python main.py --exp 1 --seeds 5

# Run Experiment 3 (Unequal Feature Scales)
python main.py --exp 3 --seeds 5
```

---

## 8. Theoretical Highlights

For independent Gaussian vectors $X, X' \sim \mathcal{N}(0, I_d)$:
$$\frac{\|X - X'\|^2}{2} \sim \chi^2(d)$$
$$\mathbb{E}[\|X - X'\|^2] = 2d, \quad \operatorname{Var}(\|X - X'\|^2) = 8d$$
$$\text{Coefficient of Variation} = \sqrt{\frac{2}{d}} \xrightarrow{d \to \infty} 0$$

See [reports/theoretical_analysis.md](reports/theoretical_analysis.md) for full proofs and [reports/final_report.md](reports/final_report.md) for empirical results.
