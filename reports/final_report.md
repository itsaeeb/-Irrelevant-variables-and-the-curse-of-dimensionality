# Project 6 Report: Irrelevant Variables and the Curse of Dimensionality

**Course:** MAC 312 — Foundations of Data Science

---

## 1. Objective and Setup

We study how irrelevant noise variables degrade classification and distance geometry by generating synthetic binary data $x = (z, u) \in \mathbb{R}^d$:
- **Informative features** $z \in \mathbb{R}^r$: $z \mid (y=0) \sim \mathcal{N}(-\delta \mathbf{1}_r, \sigma_z^2 I_r)$ and $z \mid (y=1) \sim \mathcal{N}(+\delta \mathbf{1}_r, \sigma_z^2 I_r)$ with $P(y=0) = P(y=1) = 0.5$.
- **Irrelevant features** $u \in \mathbb{R}^{d-r}$: $u \sim \mathcal{N}(0, \sigma_u^2 I_{d-r})$, strictly independent of $(z, y)$.
- **Comparability across $r$ and scales:** We set $\delta = (\mu_0 / \sqrt{r})\sigma_z$ with $\mu_0 = 1.2$. As shown in [theoretical_analysis.md](theoretical_analysis.md), this keeps the class Mahalanobis distance ($2\mu_0 = 2.4$) and **Bayes optimal error rate** ($\Phi(-1.2) \approx 11.51\%$) identical across all informative dimensions $r$ and all scale choices $\sigma_z$.

### Classifiers and Validation Tuning
All experiments use independent train, validation ($n_{\text{val}} = 500$), and test ($n_{\text{test}} = 1000$) splits across **5 random seeds** (42–46):
1. **KNN (`KNNClassifier`):** Euclidean $k$-NN; $k \in \{1, 3, 5, 7, 11, 15, 21, 31, 51\}$ tuned by validation classification error (tie-breaking on log-loss). Posterior probability is the fraction of class-1 neighbors.
2. **Gaussian KDE (`GaussianKDEClassifier`):** Isotropic Gaussian kernel per class computed in log-space via `logsumexp`; bandwidth $h \in \{0.2, 0.5, 1, 2, 3, 5, 10, 20, 50, 100\}$ tuned by validation log-loss.
3. **Regularized Gaussian (`GaussianClassifier`):** Class-conditional Gaussians with covariance shrinkage $\Sigma_c^{\text{reg}} = (1-\alpha) S_c + \alpha \frac{\operatorname{Tr}(S_c)}{d} I_d$; $\alpha \in \{0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0\}$ tuned by validation log-loss.
4. **Logistic Regression (`LogisticRegressionClassifier`):** L2-regularized logistic regression; $C \in \{10^{-4}, 10^{-3}, \dots, 10^2\}$ tuned by validation log-loss.

*Metrics:* Test classification error rate, predictive log-loss (probabilities clipped to $[10^{-15}, 1-10^{-15}]$), mean nearest-neighbor distance $D_{\text{near}}$, mean farthest-neighbor distance $D_{\text{far}}$, and distance contrast $C = (D_{\text{far}} - D_{\text{near}}) / D_{\text{near}}$.

---

## 2. Experiment 1: Increasing Ambient Dimension ($d$)

**Fixed:** $r = 2$, $n_{\text{train}} = 500$, $\sigma_z = \sigma_u = 1.0$. **Varied:** $d \in \{2, 5, 10, 20, 50, 100, 200\}$.

| $d$ | Classifier | Error (Mean ± Std) | Log-Loss (Mean ± Std) | $D_{\text{near}}$ | $D_{\text{far}}$ | Contrast $C$ |
|---:|:---|---:|---:|---:|---:|---:|
| 2 | Gaussian | 0.1132 ± 0.0051 | 0.2710 ± 0.0085 | 0.130 | 5.696 | 84.15 |
| 2 | GaussianKDE | 0.1140 ± 0.0048 | 0.2798 ± 0.0114 | 0.130 | 5.696 | 84.15 |
| 2 | KNN | 0.1174 ± 0.0044 | 0.4623 ± 0.2610 | 0.130 | 5.696 | 84.15 |
| 2 | LogisticRegression | 0.1132 ± 0.0061 | 0.2712 ± 0.0085 | 0.130 | 5.696 | 84.15 |
| 20 | Gaussian | 0.1156 ± 0.0029 | 0.2770 ± 0.0071 | 3.856 | 9.399 | 1.464 |
| 20 | GaussianKDE | 0.1838 ± 0.0041 | 0.3991 ± 0.0099 | 3.856 | 9.399 | 1.464 |
| 20 | KNN | 0.1352 ± 0.0100 | 0.3934 ± 0.0120 | 3.856 | 9.399 | 1.464 |
| 20 | LogisticRegression | 0.1198 ± 0.0029 | 0.2895 ± 0.0049 | 3.856 | 9.399 | 1.464 |
| 50 | Gaussian | 0.1222 ± 0.0082 | 0.2863 ± 0.0112 | 7.477 | 12.863 | 0.725 |
| 50 | GaussianKDE | 0.1602 ± 0.0046 | 0.4767 ± 0.0025 | 7.477 | 12.863 | 0.725 |
| 50 | KNN | 0.1472 ± 0.0122 | 0.4727 ± 0.0027 | 7.477 | 12.863 | 0.725 |
| 50 | LogisticRegression | 0.1306 ± 0.0083 | 0.3020 ± 0.0150 | 7.477 | 12.863 | 0.725 |
| 100 | Gaussian | 0.1326 ± 0.0071 | 0.3085 ± 0.0143 | 11.616 | 16.846 | 0.452 |
| 100 | GaussianKDE | 0.2492 ± 0.0151 | 0.5240 ± 0.0169 | 11.616 | 16.846 | 0.452 |
| 100 | KNN | 0.1790 ± 0.0077 | 0.5256 ± 0.0120 | 11.616 | 16.846 | 0.452 |
| 100 | LogisticRegression | 0.1528 ± 0.0054 | 0.3395 ± 0.0154 | 11.616 | 16.846 | 0.452 |
| 200 | Gaussian | 0.1400 ± 0.0042 | 0.3358 ± 0.0134 | 17.430 | 22.729 | 0.305 |
| 200 | GaussianKDE | 0.2298 ± 0.0211 | 0.5736 ± 0.0084 | 17.430 | 22.729 | 0.305 |
| 200 | KNN | 0.2168 ± 0.0313 | 0.5687 ± 0.0097 | 17.430 | 22.729 | 0.305 |
| 200 | LogisticRegression | 0.1534 ± 0.0093 | 0.3956 ± 0.0094 | 17.430 | 22.729 | 0.305 |

**Key Takeaways:**
- **Distance concentration:** As $d$ increases from 2 to 200, $D_{\text{near}}$ grows from $0.13$ to $17.43$ (roughly $\sqrt{2d}$), and distance contrast $C$ collapses from **$84.15$ down to $0.305$** — the farthest training point is only $30\%$ farther than the nearest neighbor.
- **Local estimators degrade fastest:** At $d=2$, all four models match the Bayes optimal error ($11.5\%$). By $d=200$, KNN rises to **$21.7\%$** error (log-loss $0.569$) and Gaussian KDE rises to **$23.0\%$** error (log-loss $0.574$).
- **Parametric models stay much stronger:** Regularized Gaussian ($14.0\%$, log-loss $0.336$) and L2 Logistic Regression ($15.3\%$, log-loss $0.396$) shrink or average out the zero-signal noise coordinates.
- *(Note on KNN log-loss at $d=2$: raw KNN uses neighbor proportions $m/k$, so small $k$ occasionally outputs $0$ or $1$, causing clipped log-loss variance even when classification error is low.)*

---

## 3. Experiment 2: Increasing Sample Size ($n$)

**Fixed:** $r = 2$, $d = 50$, $\sigma_z = \sigma_u = 1.0$. **Varied:** $n_{\text{train}} \in \{100, 250, 500, 1000, 2500, 5000\}$.

| $n_{\text{train}}$ | Classifier | Error (Mean ± Std) | Log-Loss (Mean ± Std) | $D_{\text{near}}$ | Contrast $C$ |
|---:|:---|---:|---:|---:|---:|
| 100 | Gaussian | 0.1504 ± 0.0054 | 0.3522 ± 0.0065 | 7.895 | 0.568 |
| 100 | GaussianKDE | 0.2388 ± 0.0388 | 0.5176 ± 0.0377 | 7.895 | 0.568 |
| 100 | KNN | 0.1884 ± 0.0226 | 0.5710 ± 0.0167 | 7.895 | 0.568 |
| 100 | LogisticRegression | 0.1626 ± 0.0136 | 0.3777 ± 0.0229 | 7.895 | 0.568 |
| 500 | Gaussian | 0.1222 ± 0.0082 | 0.2863 ± 0.0112 | 7.477 | 0.725 |
| 500 | GaussianKDE | 0.1602 ± 0.0046 | 0.4767 ± 0.0025 | 7.477 | 0.725 |
| 500 | KNN | 0.1472 ± 0.0122 | 0.4727 ± 0.0027 | 7.477 | 0.725 |
| 500 | LogisticRegression | 0.1306 ± 0.0083 | 0.3020 ± 0.0150 | 7.477 | 0.725 |
| 1000 | Gaussian | 0.1206 ± 0.0026 | 0.2789 ± 0.0090 | 7.311 | 0.783 |
| 1000 | GaussianKDE | 0.1340 ± 0.0102 | 0.4755 ± 0.0071 | 7.311 | 0.783 |
| 1000 | KNN | 0.1408 ± 0.0114 | 0.4459 ± 0.0080 | 7.311 | 0.783 |
| 1000 | LogisticRegression | 0.1194 ± 0.0062 | 0.2889 ± 0.0111 | 7.311 | 0.783 |
| 5000 | Gaussian | 0.1146 ± 0.0073 | 0.2717 ± 0.0076 | 6.946 | 0.938 |
| 5000 | GaussianKDE | 0.1228 ± 0.0027 | 0.4717 ± 0.0050 | 6.946 | 0.938 |
| 5000 | KNN | 0.1402 ± 0.0055 | 0.4051 ± 0.0069 | 6.946 | 0.938 |
| 5000 | LogisticRegression | 0.1174 ± 0.0072 | 0.2754 ± 0.0075 | 6.946 | 0.938 |

**Key Takeaways:**
- Increasing $n_{\text{train}}$ by **$50\times$** (from 100 to 5000) allows Regularized Gaussian ($15.0\% \to 11.5\%$) and Logistic Regression ($16.3\% \to 11.7\%$) to reach the Bayes floor ($11.5\%$).
- For local methods, $50\times$ more data barely changes the geometry: $D_{\text{near}}$ only drops from $7.90$ to $6.95$ and contrast $C$ stays below $1.0$. Even at $n=5000$, KNN remains stuck at **$14.0\%$** error (log-loss $0.405$) and KDE log-loss stays at **$0.472$**.

---

## 4. Experiment 3: Unequal Feature Scales & Standardization Regimes

**Fixed:** $r = 2$, $d = 50$, $n_{\text{train}} = 500$, constant Bayes error $11.51\%$.
We compare five pipelines — **Raw**, **Standardized**, **Weighted** (standardize + training ANOVA $F$-score weights), **Selected** (training-only ANOVA `SelectKBest`, $k$ tuned on validation), and **Oracle** ($z$ only) — across three scale regimes:

| Regime | Pipeline | KNN Error | KDE Error | Gaussian Error | LogReg Error | Contrast $C$ |
|:---|:---|---:|---:|---:|---:|---:|
| **Noise-Dominated** ($\sigma_z=1, \sigma_u=10$) | Raw | **0.4906** | **0.4852** | **0.4764** | 0.1384 | 0.73 |
| | Standardized | 0.1852 | 0.2048 | 0.1390 | 0.1324 | 0.71 |
| | Weighted | 0.1210 | 0.1140 | 0.1338 | 0.1150 | 8.74 |
| | Selected | 0.1208 | 0.1134 | 0.1134 | 0.1132 | 56.3–84.4 |
| | Oracle | 0.1174 | 0.1140 | 0.1132 | 0.1132 | 84.15 |
| **Equal-Scale** ($\sigma_z=1, \sigma_u=1$) | Raw | 0.1472 | 0.1602 | 0.1222 | 0.1306 | 0.73 |
| | Standardized | 0.1852 | 0.2048 | 0.1390 | 0.1324 | 0.71 |
| | Weighted | 0.1210 | 0.1140 | 0.1338 | 0.1150 | 8.74 |
| | Selected | 0.1208 | 0.1134 | 0.1134 | 0.1132 | 56.3–84.4 |
| | Oracle | 0.1174 | 0.1140 | 0.1132 | 0.1132 | 84.15 |
| **Signal-Dominated** ($\sigma_z=10, \sigma_u=1$) | Raw | **0.1172** | **0.1144** | 0.1278 | 0.1158 | 5.53 |
| | Standardized | **0.1852** | **0.2048** | 0.1390 | 0.1324 | 0.71 |
| | Weighted | 0.1210 | 0.1140 | 0.1338 | 0.1150 | 8.74 |
| | Selected | 0.1208 | 0.1134 | 0.1134 | 0.1132 | 56.3–84.4 |
| | Oracle | 0.1174 | 0.1140 | 0.1132 | 0.1132 | 84.15 |

**Key Takeaways (directly matching Section 3 of [theoretical_analysis.md](theoretical_analysis.md)):**
1. **Where standardization helps (Noise-Dominated):** Raw KNN ($49.1\%$), KDE ($48.5\%$), and isotropic-shrunk Gaussian ($47.6\%$) collapse to coin-flipping because the 48 noise variables have $100\times$ larger variance. Standardization cuts KNN/KDE error by more than half ($18.5\%$ / $20.5\%$) and restores Gaussian to $13.9\%$.
2. **Where standardization cannot remove irrelevant coordinates (Noise-Dominated & Equal-Scale):** After standardization, all 48 noise dimensions still have unit variance, so KNN ($18.5\%$) and KDE ($20.5\%$) remain far worse than Oracle ($11.7\%$ / $11.4\%$). In Equal-Scale, standardizing even nudges KNN error *up* ($14.7\% \to 18.5\%$) because dividing $z_j$ by its marginal std $\sqrt{\sigma_z^2 + \delta^2} \approx 1.31$ slightly shrinks signal relative to noise.
3. **Where standardization actively hurts (Signal-Dominated):** When informative features already have larger scale ($\sigma_z=10, \sigma_u=1$), Raw KNN ($11.7\%$) and Raw KDE ($11.4\%$) naturally match Oracle. Standardization blows the 48 small noise features up to variance 1, pushing KNN up to $18.5\%$ and KDE up to $20.5\%$.
4. **Feature weighting and selection recover Oracle everywhere:** Across all three regimes, training-only `Weighted` and `Selected` pipelines achieve $11.3\%$–$12.1\%$ error, matching the Oracle baseline.

---

## 5. Experiment 4: Paired Isolation of Informative Dimension ($r$) vs. Ambient Dimension ($d$)

To isolate informative dimension $r \in \{1, 2, 4, 8, 16, 32\}$ without confounding it with easier class separation, we hold $n_{\text{train}} = 500$ and total signal norm $\mu_0 = 1.2$ fixed ($\delta = 1.2/\sqrt{r}$, constant Bayes error $11.51\%$) in two paired settings:

| Mode | $r$ | $d$ | KNN Error | KDE Error | Gaussian Error | LogReg Error | Contrast $C$ |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **Fixed Ambient ($d=50$)** | 1 | 50 | 0.1598 | 0.1688 | 0.1230 | 0.1324 | 0.726 |
| | 2 | 50 | 0.1472 | 0.1602 | 0.1222 | 0.1306 | 0.725 |
| | 8 | 50 | 0.1558 | 0.1682 | 0.1278 | 0.1396 | 0.726 |
| | 32 | 50 | 0.1534 | 0.1562 | 0.1148 | 0.1294 | 0.714 |
| **No Noise ($d=r$)** | 1 | 1 | 0.1194 | 0.1184 | 0.1182 | 0.1184 | 10017.0 |
| | 2 | 2 | 0.1174 | 0.1140 | 0.1132 | 0.1132 | 84.15 |
| | 8 | 8 | 0.1292 | 0.1328 | 0.1236 | 0.1244 | 3.75 |
| | 16 | 16 | 0.1268 | 0.1626 | 0.1146 | 0.1160 | 1.78 |
| | 32 | 32 | 0.1422 | 0.1324 | 0.1156 | 0.1250 | 0.987 |

**Key Takeaways:**
- **Fixed Ambient ($d=50$):** Because total dimension $d=50$ and total class separation $2\mu_0 = 2.4$ are both constant, the squared-distance distribution barely changes ($C \approx 0.72$ everywhere), and error stays roughly flat across $r = 1 \dots 32$ (KNN $\approx 15\%$, KDE $\approx 16\%$, Gaussian $\approx 12\%$, LogReg $\approx 13\%$).
- **No Noise ($d=r$):** Even with **zero irrelevant variables**, spreading the same signal over more informative dimensions $r = 1 \to 32$ causes distance contrast to collapse ($10017 \to 0.99$), KNN error to rise ($11.9\% \to 14.2\%$), and KDE log-loss to rise ($0.283 \to 0.472$).
- **Paired conclusion (Exp 1 vs. Exp 2 vs. Exp 4):** For distance-based classifiers, degradation is driven by **total dimensionality $d$** relative to sample size $n$ and total separation $\mu_0$ — whether those dimensions carry pure noise (Exp 1) or dilute the same signal across many informative coordinates (Exp 4, $d=r$).

---

## 6. Generated Artifacts

- **CSVs:** `results/csv/experiment_{1..4}_*_raw.csv` and `*_summary.csv`
- **Tables:** `results/tables/experiment_{1..4}_*_table.md`
- **Plots:** `results/plots/` (`exp1_*.png`, `exp2_*.png`, `exp3_*.png`, `exp4_*.png`)
- **Reproduce everything:** `python main.py --exp all --seeds 5`
