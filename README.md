# 🌐 EasyVisa — Visa Approval Prediction
### Ensemble ML | Bagging | Boosting | Stacking | XGBoost | Classification

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5.2-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Domain](https://img.shields.io/badge/Domain-Government%20%7C%20HR%20%7C%20Immigration-blue)

---

## 🏛️ Business Context

US business communities face high demand for skilled human resources. Companies look for talented individuals both locally and abroad — with the **Immigration and Nationality Act (INA)** allowing foreign workers to work in the US on temporary or permanent basis.

The **Office of Foreign Labor Certification (OFLC)** processes thousands of visa applications annually. Manual review is time-consuming and inconsistent. An AI-powered prediction system can help:

- **Speed up screening** — Identify likely-approved applications faster
- **Reduce processing bottlenecks** — Prioritize cases needing deeper review
- **Provide applicant guidance** — Help applicants understand their approval likelihood

---

## 🎯 Objective

> Build an **ensemble machine learning model** to predict visa approval status and **recommend suitable applicant profiles** that are most likely to be certified.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Rows | 25,480 visa applications |
| Columns | 12 features |
| Target Variable | `case_status` (Certified / Denied) |
| Missing Values | None |
| Data Types | Mix of int64, float64, object |

**Key Features:**
- `continent` — Applicant's continent of origin
- `education_of_employee` — Education level (High School to Doctorate)
- `has_job_experience` — Has relevant job experience (Y/N)
- `requires_job_training` — Position requires training (Y/N)
- `no_of_employees` — Employer company size
- `yr_of_estab` — Year the company was established
- `region_of_employment` — US employment region
- `prevailing_wage` — Prevailing wage for the position
- `unit_of_wage` — Hour / Week / Month / Year
- `full_time_position` — Full-time position (Y/N)

---

## 🔬 Approach & Methodology

```
Raw Data → EDA → Preprocessing → Baseline Models → Ensemble Methods → Hyperparameter Tuning → Insights
```

### 1. Exploratory Data Analysis (EDA)
- Analyzed visa approval rates by education level, continent, and employment region
- Key finding: **Doctorate and Master's degree holders** have significantly higher approval rates
- **High School graduates** face greater denial risks
- Prevailing wage is a strong predictor — higher wages correlate with approval
- Company size (`no_of_employees`) shows interesting approval patterns

### 2. Data Preprocessing
- Encoded categorical variables (continent, education, region, wage unit)
- Handled class imbalance in approval status distribution
- Feature scaling for continuous variables
- Train/test split with stratification to maintain class distribution

### 3. Baseline Models
- **Decision Tree** — Interpretable baseline
- **Logistic Regression** — Linear baseline

### 4. Ensemble Models
Built and compared three ensemble approaches:

**Bagging:**
- Random Forest Classifier
- Reduces variance through bootstrap aggregation

**Boosting:**
- AdaBoostClassifier
- GradientBoostingClassifier
- **XGBoostClassifier** (best performer)
- Reduces bias by correcting previous model errors

**Stacking:**
- Base learners: Decision Tree, Random Forest, XGBoost
- Meta-learner: Logistic Regression
- Combines diverse model perspectives for superior performance

### 5. Hyperparameter Tuning
- Applied GridSearchCV with 5-fold cross-validation
- Tuned key parameters: `n_estimators`, `max_depth`, `learning_rate`, `min_samples_split`
- Selected best model based on F1-Score and ROC-AUC

### 6. Feature Importance Analysis
Top features influencing visa approval:
1. **Education of Employee** — Highest impact feature
2. **Prevailing Wage** — Strong positive correlation with approval
3. **Has Job Experience** — Experienced applicants fare better
4. **Region of Employment** — Geographic factors matter
5. **Company Size** — Larger established companies have better approval rates

---

## 📈 Key Results

| Metric | Value |
|---|---|
| Best Model | XGBoost / Stacking Ensemble |
| Top Predictor | Education level of employee |
| Key Business Insight | Doctorate/Master's holders: highest approval rate |
| Key Risk Factor | High School graduates face highest denial rates |

---

## 💡 Business Insights & Recommendations

1. **Education is the Strongest Predictor** — Doctorate and Master's degree holders have the best approval chances; HR teams should prioritize these profiles for international recruitment
2. **Job Experience Matters** — Applicants with relevant experience have notably better outcomes; require experience documentation for all international candidates
3. **Wage Benchmarking** — Higher prevailing wages strongly correlate with approval; ensure competitive wage offerings to improve certification rates
4. **Regional Strategy** — Approval rates vary significantly by US employment region; factor regional success rates into location decisions for international hires
5. **Training Requirements** — Positions requiring job training face higher scrutiny; pre-employment training programs can strengthen applications

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core programming language |
| Pandas 1.5.3 | Data manipulation |
| NumPy 1.25.2 | Numerical computation |
| Scikit-learn 1.5.2 | Ensemble models, GridSearchCV, evaluation |
| XGBoost 2.0.3 | Gradient boosting implementation |
| Matplotlib 3.7.1 | Visualization |
| Seaborn 0.13.1 | Statistical visualization |
| GridSearchCV | Hyperparameter tuning |
| Google Colab | Development environment |

---

## 📁 Project Structure

```
easyvisa-approval-prediction/
│
├── Sapna-Easyvisa.ipynb   # Full notebook: EDA, models, evaluation
├── README.md               # Project documentation
└── data/                   # Dataset (not included - proprietary)
```

---

## 🚀 How to Run

1. Clone this repository
2. Open `Sapna-Easyvisa.ipynb` in Google Colab or Jupyter Notebook
3. Upload dataset to Google Drive and update file path
4. Run all cells sequentially

```bash
pip install numpy==1.25.2 pandas==1.5.3 scikit-learn==1.5.2 matplotlib==3.7.1 seaborn==0.13.1 xgboost==2.0.3
```

---

## 👩‍💻 Author

**Sapna** | Senior AI Quality Engineer  
Post Graduate in AI/ML — University of Texas at Austin  
GitHub: [@sapna-ai-qe](https://github.com/sapna-ai-qe)

---
*Part of AI/ML Portfolio — UT Austin Post Graduate Program*
