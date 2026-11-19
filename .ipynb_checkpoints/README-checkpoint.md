# Project 6: Irrelevant Variables and the Curse of Dimensionality

Course: **MAC 312 — Foundations of Data Science**

This project studies how independent irrelevant features affect classification accuracy, predictive log-loss, and Euclidean distance geometry across four classifiers: **KNN**, **Gaussian KDE**, **Regularized Gaussian (QDA)**, and **L2 Logistic Regression**.

---

## Setup

Each observation is $x = (z, u) \in \mathbb{R}^d$, where:
- $z \in \mathbb{R}^r$ are the informative coordinates:
  $$z \mid (y = 0) \sim \mathcal{N}(-\delta \mathbf{1}_r, \sigma_z^2 I_r), \quad z \mid (y = 1) \sim \mathcal{N}(+\delta \mathbf{1}_r, \sigma_z^2 I_r)$$
  with $P(y=0) = P(y=1) = 0.5$ and $\delta = (\mu_0 / \sqrt{r}) \sigma_z$ ($\mu_0 = 1.2$).
- $u \in \mathbb{R}^{d-r}$ are independent irrelevant noise coordinates:
  $$u \sim \mathcal{N}(0, \sigma_u^2 I_{d-r}), \quad u \perp\!\!\!\perp (z, y)$$

Scaling $\delta$ by $1/\sqrt{r}$ keeps the class separation norm and **Bayes optimal error** ($\Phi(-1.2) \approx 11.51\%$) identical for every informative dimension $r$ and every rescaling $\sigma_z$.

---

## Experiments

1. **Experiment 1 — Ambient Dimension (`experiments/experiment_1_ambient_dimension.py`):**
   Fixes $r = 2, n_{\text{train}} = 500$; varies $d \in [2, 5, 10, 20, 50, 100, 200]$.
2. **Experiment 2 — Sample Size (`experiments/experiment_2_sample_size.py`):**
   Fixes $r = 2, d = 50$; varies $n_{\text{train}} \in [100, 250, 500, 1000, 2500, 5000]$.
3. **Experiment 3 — Unequal Feature Scales & Regimes (`experiments/experiment_3_feature_scales.py`):**
   Fixes $r = 2, d = 50, n_{\text{train}} = 500$. Compares **Raw**, **Standardized**, **Weighted** (ANOVA F-scores), **Selected** (training-only `SelectKBest`), and **Oracle** ($z$ only) across three regimes:
   - Noise-Dominated ($\sigma_z = 1, \sigma_u = 10$)
   - Equal-Scale ($\sigma_z = 1, \sigma_u = 1$)
   - Signal-Dominated ($\sigma_z = 10, \sigma_u = 1$)
4. **Experiment 4 — Paired Informative Dimension (`experiments/experiment_4_informative_dimension.py`):**
   Varies $r \in [1, 2, 4, 8, 16, 32]$ at constant Bayes error ($11.51\%$) in two paired modes:
   - Fixed Ambient ($d = 50$)
   - No Noise ($d = r$)

---

## Structure

```
├── src/
│   ├── data_generation.py      # Synthetic (z, u) data & independent train/val/test splits
│   ├── classifiers.py          # KNN, GaussianKDE, Regularized Gaussian, LogisticRegression
│   ├── preprocessing.py        # Raw, Standardized, Weighted, Selected, Oracle
│   ├── distance_metrics.py     # D_near, D_far, contrast C, ratio R
│   ├── evaluation.py           # Error rate and clipped predictive log-loss
│   ├── experiment_utils.py     # Validation tuning, aggregation, CSV/table export
│   └── plotting.py             # Matplotlib/Seaborn figures
├── experiments/                # Scripts for Experiments 1–4
├── results/                    # Generated CSVs, Markdown tables, and PNG plots
├── reports/
│   ├── theoretical_analysis.md # Squared-distance moments & standardization analysis
│   └── final_report.md         # Summary of all experiments and findings
├── main.py                     # Entrypoint to run one or all experiments
├── Project_6_Irrelevant_Variables_Curse_of_Dimensionality.ipynb  # Self-contained Jupyter notebook
└── requirements.txt
```

---

## How to Run

### CLI Scripts
```bash
pip install -r requirements.txt

# Run all 4 experiments (5 seeds each)
python main.py --exp all --seeds 5

# Or run a single experiment (1, 2, 3, or 4)
python main.py --exp 1 --seeds 5
```

### Jupyter Notebook (from global JupyterLab)
Register this project's virtual environment as a user-level Jupyter kernel:
```bash
venv/bin/python -m ipykernel install --user --name fds-project --display-name "Python (FDS Project)"
```
Then open `Project_6_Irrelevant_Variables_Curse_of_Dimensionality.ipynb` from your global `jupyter-lab` environment and select the **Python (FDS Project)** kernel.
