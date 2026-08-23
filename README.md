# ⚡ PowerCo Customer Churn Prediction — BCG X Virtual Experience

**Profit-Driven Churn Prediction in a Liberalised Utility Market: A Comparative Study of Interpretable Machine Learning Techniques**

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-green.svg)](https://shap.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

---

## 📌 Overview

This project was completed as part of the **Boston Consulting Group (BCG) X Data Science Job Simulation**, working with **PowerCo**, a major European gas and electricity utility serving corporate, SME, and residential customers.

Following the liberalisation of the European energy market, PowerCo has seen rising customer churn — particularly among **SME (Small & Medium Enterprise) clients**. BCG's initial hypothesis was that **price sensitivity** is the primary driver of this churn. My role was to validate that hypothesis, and — where it fell short — build a robust, business-aligned machine learning framework to identify customers at genuine risk of churning.

The work was later extended into a formal academic research paper — *"Profit-Driven Churn Prediction in Liberalised Utility Market: A Comparative Study of Interpretable Machine Learning Techniques"* — submitted as a Minor Project at the Manipal School of Commerce and Economics, MAHE (2025–2026).

---

## 🎯 Business Problem

PowerCo partnered with BCG to diagnose why SME customers were leaving. The engagement was broken into three stages:

1. **Business Understanding** — Frame churn as a testable data science hypothesis (price sensitivity → churn) and define the data and modelling approach needed.
2. **Exploratory Data Analysis** — Understand the client and pricing datasets, and test whether price sensitivity actually correlates with churn.
3. **Feature Engineering & Modelling** — Build, tune, and evaluate predictive models, then convert model outputs into actionable, revenue-relevant business insight.

**Key finding:** Contrary to the initial hypothesis, raw price changes showed *no meaningful correlation* with churn — the churn rate held steady at ~9.7% regardless of whether prices rose or fell. This redirected the analysis toward **customer profitability and forward-looking cost expectations** as the true economic drivers of churn.

---

## 🗂️ Repository Contents

| File | Description |
|---|---|
| `BCG_PowerCo_Customer_Churn_Research_Paper.ipynb` | End-to-end Jupyter notebook: data cleaning, EDA, price-sensitivity testing, feature engineering, model training/tuning, evaluation, and SHAP interpretability |
| `Minor_Project_251260260051.pdf` | Full academic research paper documenting the extended methodology, statistical validation, and findings |
| `README.md` | This file |

> **Note:** The raw datasets (`client_data.csv`, `price_data.csv`) supplied by PowerCo/BCG are proprietary to the simulation and are **not included** in this repository.

---

## 🧾 Data

Two datasets were provided for the SME client segment:

- [`client_data.csv`](client_data.csv)` — 14,606 records × 26 features: consumption, contract dates, sales channel, forecasted usage/pricing, margins, and the binary `churn` label.
- [`price_data.csv`](price_data.csv) — 193,002 records: daily electricity pricing (peak, off-peak, mid-peak — variable and fixed) per customer for 2015–2016.

**Target variable:** `churn` — highly imbalanced, with only **9.7%** of clients having churned, which shaped every downstream modelling decision.

---

## 🔬 Methodology

### 1. Data Understanding & Cleaning
- Converted date fields to datetime and engineered `contract_start_year`, `contract_end_year`, and tenure-based features.
- Checked for nulls/duplicates (none found) and validated categorical/numeric feature consistency.
- Label-encoded categorical variables (`channel_sales`, `origin_up`) — chosen for tree-model compatibility.

### 2. Exploratory Data Analysis & Hypothesis Testing
- Visualised churn distribution, sales channel mix, tenure, consumption, and pricing patterns.
- Directly tested the "price sensitivity drives churn" hypothesis by comparing churn rates across customers with rising vs. falling prices (1-month, 3-month, and 6-month windows) — **found no significant relationship**.

### 3. Feature Engineering
- **Price-shock features:** e.g. `diff_Dec_mean_price_off_peak_var` — the gap between a customer's December price and their annual average, designed to capture "sticker shock" ahead of renewal.
- **Profitability features:** `margin_net_pow_ele`, `margin_gross_pow_ele`, `net_margin`.
- **Forward-looking cost features:** `forecast_meter_rent_12m`, `forecast_cons_12m`.
- **Relationship features:** `num_years_antig` (tenure), `nb_prod_act` (active products).

### 4. Modelling
Four model families were benchmarked to represent linear, kernel, bagging, and boosting approaches:

| Model | Role |
|---|---|
| Logistic Regression | Linear baseline |
| Support Vector Machine (SVM) | Kernel-based, non-linear boundary |
| Random Forest | Bagging ensemble |
| **XGBoost** | Gradient boosting ensemble |

- **Validation:** Stratified 5-fold cross-validation preserving the 9.7% churn ratio in every fold.
- **Hyperparameter tuning:** Bayesian optimization (`scikit-optimize`) across model-specific search spaces.
- **Class imbalance handling:** Addressed via `scale_pos_weight` (XGBoost) and class-weight balancing (other models) — **no synthetic oversampling (e.g. SMOTE)** was used, to preserve the real-world class distribution and business-realistic evaluation.
- **Decision threshold:** Tuned (not fixed at 0.5) to maximize the precision–recall trade-off relevant to retention economics.

### 5. Evaluation
Because of the severe class imbalance, **accuracy was treated as a misleading metric** (~90% for all models). Evaluation instead centered on:
- Precision, Recall, F1-score
- Precision–Recall curve & Average Precision (AP)
- Confusion matrix at the tuned threshold
- **McNemar's test** for statistical significance of performance differences between models

### 6. Explainability
- **SHAP (SHapley Additive exPlanations)** applied to the top model for both global feature importance and local, per-customer explanations — critical for producing auditable, regulator-friendly retention decisions rather than a black-box output.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.902 | 0.302 | 0.007 | 0.014 |
| SVM | 0.903 | 0.000 | 0.000 | 0.000 |
| Random Forest | 0.904 | 0.883 | 0.019 | 0.036 |
| **XGBoost** | **0.905** | 0.675 | **0.036** | **0.068** |

- **XGBoost** delivered the best overall precision–recall balance (Average Precision ≈ 0.249) and, at a tuned decision threshold, reached **~40% precision** — meaning roughly 2 in 5 customers flagged as at-risk were genuine churners.
- Random Forest achieved higher raw precision but an impractically low recall, making it too conservative for proactive retention campaigns.
- Linear (Logistic Regression) and kernel-based (SVM) models failed to capture the non-linear, interaction-driven nature of churn behaviour.
- **McNemar's test** confirmed XGBoost's advantage over Logistic Regression and SVM was statistically significant (p < 0.05).

### Top Churn Drivers (SHAP)
1. `margin_gross_pow_ele` / `margin_net_pow_ele` — customer profitability
2. `forecast_meter_rent_12m` — anticipated future costs
3. Price deviations around renewal periods (`diff_Dec_*` variables)
4. `pow_max`, `net_margin`, `cons_12m`

**Bottom line:** Churn in this liberalised utility market is best explained not by price levels themselves, but by **customer profitability and anticipated future costs** — customers behave as rational, forward-looking economic agents rather than reacting to price alone.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Data handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Modelling:** `scikit-learn`, `xgboost`
- **Hyperparameter tuning:** `scikit-optimize` (Bayesian search)
- **Explainability:** `shap`, `lime`
- **Statistical testing:** `statsmodels` (McNemar's test)

---

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/powerco-churn-prediction.git
cd powerco-churn-prediction
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy statsmodels shap lime scikit-optimize
jupyter notebook BCG_PowerCo_Customer_Churn_Research_Paper.ipynb
```

> You will need to supply your own copies of `client_data.csv` and `price_data.csv` (from the BCG X simulation) and update the file paths in the notebook's data-loading cells.

---

## ⚠️ Limitations & Future Work

- Modelled on a single European utility provider — generalisability to other regulatory regimes/markets is untested.
- Hyperparameter search was run on a computationally constrained subset, which may leave some room for further tuning.
- **Future directions:** multi-utility / multi-country validation, incorporating macroeconomic and competitor-tariff data, and deploying the model in a controlled A/B test to measure real-world retention impact.

---

## 📚 Background Reading

The full academic writeup — including formal problem framing, literature review, statistical methodology, and detailed discussion — is available in [`Project Report.pdf`](./Project Report.pdf).

---

