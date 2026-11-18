# Microloan Default Risk Analysis

**Wali Abidi · Machine Learning Portfolio**

A reproducible binary-classification study of microloan repayment risk using a deliberately simulated dataset.

## Executive summary

This project investigates whether borrower and loan attributes can distinguish repaid microloans from loans that default. Five supervised classifiers were evaluated against a majority-class baseline using a stratified hold-out test set and training-only preprocessing.

**Best overall ranking model: Logistic Regression** — test ROC-AUC **0.888**, default-class F1 **0.656**, recall **0.556**, and precision **0.800** at the 0.50 threshold.

> **Scope:** This is an educational modelling exercise on simulated data. These results demonstrate an analytical workflow and model comparison—not validated lending performance or real-world borrower behaviour.

## Results

| Metric | Logistic Regression | Linear SVM | KNN | Decision Tree | Naive Bayes |
|---|---:|---:|---:|---:|---:|
| Accuracy | **0.883** | 0.878 | 0.828 | 0.822 | 0.817 |
| Precision | 0.800 | **0.889** | 0.576 | 0.567 | 0.571 |
| Recall | **0.556** | 0.444 | 0.528 | 0.472 | 0.333 |
| Specificity | 0.965 | **0.986** | 0.903 | 0.910 | 0.938 |
| F1 | **0.656** | 0.593 | 0.551 | 0.515 | 0.421 |
| ROC-AUC | **0.888** | 0.884 | 0.737 | 0.713 | 0.764 |

The majority-class baseline obtains 80% accuracy by predicting every loan as repaid, but detects no defaults. Accuracy alone is therefore insufficient for this imbalanced classification task.

## Key findings

1. **Logistic Regression led overall:** highest ROC-AUC (0.888), F1 (0.656), and recall (0.556). Its confusion matrix was 139 true negatives, 5 false positives, 16 false negatives, and 20 true positives.
2. **Threshold selection changes the trade-off:** reducing the threshold from 0.50 to 0.30 raised recall from 0.556 to 0.722 (20 to 26 defaults detected), while precision fell from 0.800 to 0.591 and false positives rose from 5 to 18. This 0.30 result is illustrative; thresholds should be selected using validation data and explicit error costs.
3. **Scaling helped KNN:** validation recall was 0.414 with scaled features versus 0.207 without scaling.
4. **Savings-group membership was the decision tree's root split:** simulated non-members had a 31.3% default rate (111/355), versus 9.0% (33/365) for members. The information gain was 0.0581. This is an association in generated data, not a causal effect.
5. **Complexity exposed overfitting:** KNN at K=1 reached fit F1=1.000 while validation F1 was 0.414; tree fit F1 rose with depth while validation performance remained much lower.

## Dataset and methodology

- 900-row analysis sample from a 3,000-observation synthetic population.
- 180 defaults and 720 repaid loans (20% default prevalence).
- Features: age, monthly household income, loan amount, term, prior loans, agricultural income share, branch distance, and savings-group membership.
- 24 monthly-income and 14 branch-distance values are missing.
- Stratified 80/20 train/test split; median imputation and standardisation fitted on training data only.
- Models: Logistic Regression, KNN, Gaussian Naive Bayes, entropy Decision Tree, and Linear SVM.
- Evaluation: accuracy, precision, recall, specificity, F1, ROC-AUC, confusion matrices, and threshold analysis.

The generator encodes risk using loan-to-income ratio, agricultural income share, branch distance, savings-group membership, prior loans, loan term, and age. Observed patterns partly reflect these simulation assumptions.

## Project workflow

```text
Synthetic data generation
  → Data audit and exploratory analysis
  → Stratified train/test split
  → Training-only imputation and scaling
  → Baseline + five classifiers
  → Validation-based parameter inspection
  → Held-out evaluation and threshold analysis
```

## Repository structure

```text
.
├── assets/          # Animated SVG cover and hand-authored SVG icons
├── data/            # Synthetic sample and data documentation
├── docs/            # Interactive GitHub Pages project page
├── notebooks/       # Project walkthrough
├── reports/         # Metrics, validation sweeps, and figures
├── src/             # Data generation and model training
├── tests/           # Pipeline tests
└── requirements.txt
```

## Run locally

```bash
git clone https://github.com/WaliAbidi2315/microloan-default-risk-analysis.git
cd microloan-default-risk-analysis
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/generate_data.py
python src/train.py
pytest
```

## Limitations and responsible use

This is synthetic data and a small single hold-out experiment (36 defaults in the test set). It has no temporal or external validation, probability calibration, subgroup fairness audit, or real-world lending evidence. It must **not** be used to make lending decisions. A responsible next iteration would include repeated stratified cross-validation, calibration, cost-based threshold selection, fairness analysis, monitoring, and validation on suitably governed real data.

## Author

**Wali Abidi** · [GitHub](https://github.com/WaliAbidi2315)
