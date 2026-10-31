# RetainIQ
### Customer Intelligence. Predictive Retention. Smarter Growth.

**RetainIQ** is an enterprise-grade AI-powered customer intelligence and retention analytics platform. It bridges exploratory data science, behavioral RFM segmentation, and supervised machine learning into a high-performance commercial SaaS application built with Python and Streamlit.

---

## 🎯 Platform Overview

RetainIQ transforms static customer transaction datasets into proactive commercial retention operations. By analyzing historical purchasing cadences, the platform evaluates customer churn risk up to 90 days in advance, ranks accounts by **Risk-Weighted Historical Spend**, and equips commercial teams with targeted playbooks and CRM campaign exports.

### Core Business Metrics (Verified Ground Truth)

- **Full-Period Verified Commercial Footprint (Dec 2010 – Dec 2011)**:
  - **Identified Customers**: 4,338
  - **Customer-Attributed Revenue**: £8,887,208.89
  - **Total Valid Invoices**: 18,532
  - **Active Catalog Products**: 3,665

- **Historical Churn Evaluation Cohort (Cutoff: Sep 10, 2011)**:
  - **Cohort Size**: 3,370 accounts
  - **Historical Churned Customers**: 1,449 (43.0% churn rate)
  - **Historical Retained Customers**: 1,921 (57.0% active rate)
  - **Out-of-Fold Risk Distribution**:
    - **Low Risk** ($P < 0.40$): 1,076 accounts
    - **Medium Risk** ($0.40 \le P < 0.70$): 1,695 accounts
    - **High Risk** ($P \ge 0.70$): 599 accounts

- **Model Benchmark (Logistic Regression Pipeline)**:
  - **5-Fold Cross-Validation ROC-AUC**: `0.7470 ± 0.0110`
  - **Untouched 20% Holdout ROC-AUC**: `0.7238`
  - **Holdout Accuracy**: `66.02%`
  - **Churn Recall (Sensitivity)**: `77.0%`
  - **Churn Precision**: `58.0%`
  - **Churn F1-Score**: `0.66`

---

## 🏛️ System Architecture & Project Structure

```
RetainIQ/
│
├── app.py                         # Enterprise launchpad & central navigation hub
├── requirements.txt               # Application dependencies
├── README.md                      # Comprehensive platform documentation
├── pytest.ini                     # Test configuration
├── .gitignore                     # Git configuration
│
├── pages/
│   ├── 1_Executive_Dashboard.py   # Macro KPIs, revenue curves, segment & risk distributions
│   ├── 2_Customer_Intelligence.py # Customer 360 lookup, RFM profiles, invoice timeline
│   ├── 3_Churn_Prediction.py      # ML pipeline inference, out-of-fold scoring, what-if simulator
│   ├── 4_Retention_Workspace.py   # Value-at-risk ranking, playbooks, CSV export, ROI simulator
│   ├── 5_Model_Performance.py     # LR vs RF benchmarks, Confusion Matrix, ROC curve, feature attribution
│   └── 6_Data_Quality.py          # Data lineage funnel, missing values, validity audit
│
├── src/
│   ├── data_loader.py             # Cached data ingestion, memory optimization, country mapping
│   ├── preprocessing.py          # Data audit funnels, RFM computation, input validation
│   ├── customer_analytics.py      # Monthly trends, country breakdown, customer 360 timelines
│   ├── churn_engine.py            # Joblib model loading, compatibility patching, live inference
│   ├── segmentation.py            # RFM profiling, cohort vs full-period comparisons, cross-tabulation
│   ├── retention_engine.py        # Playbook matrix, priority ranking, CSV builder, scenario math
│   ├── metrics.py                 # Holdout evaluation, confusion matrix plotting, permutation importance
│   └── utils.py                   # Design system, custom CSS, Plotly templates, KPI cards
│
├── models/
│   └── retainiq_churn_model.joblib # Trained StandardScaler + LogisticRegression artifact
│
├── data/
│   └── processed/
│       ├── clean_transactions.csv
│       ├── customer_transactions.csv
│       ├── rfm_segments.csv
│       ├── churn_dataset.csv
│       ├── segment_revenue.csv
│       ├── customer_risk_predictions.csv
│       ├── retainiq_customer_dashboard.csv
│       └── retainiq_app_customers.csv
│
├── assets/
│   └── logo.png                   # RetainIQ brand asset
│
└── tests/
    └── test_retainiq.py           # Comprehensive pytest suite
```

---

## 🚀 Key Modules & Capabilities

### 1. Executive Dashboard
- **Dual-Population Distinction**: Explicitly separates full-period transactions (4,338 customers) from the historical churn evaluation cohort (3,370 customers).
- **Interactive Visualizations**:
  1. Monthly revenue trajectory (£)
  2. Top 10 country revenue contributions (£)
  3. Cutoff cohort segment distribution (Champions, Loyal, Potential, Recent, At Risk, Lost)
  4. Out-of-fold churn risk distribution (Low, Medium, High)
  5. Revenue contribution by customer segment
  6. 90-day customer retention overview (Retained vs Churned)
- **Multi-Dimensional Filters**: Country, Customer Segment, Churn Risk Tier, and Transaction Date Span.

### 2. Customer Intelligence & Segmentation
- **Customer 360 Search**: Direct lookup by `CustomerID` across 4,338 accounts.
- **Cutoff-Date vs Full-Period Alignment**: Preserves cutoff-date features for model alignment while providing complete transaction history.
- **RFM Scorecard**: Quintile scores (R: 1-5, F: 1-5, M: 1-5) and behavioral segment assignments.
- **Invoice & Line Items Timeline**: Detailed invoice history with drill-down into specific line items (StockCode, Description, UnitPrice, Quantity, Revenue).
- **RFM vs Churn Risk Distinction**: Articulates why backward-looking RFM behavioral labels differ from forward-looking ML churn probabilities.

### 3. Churn Prediction Engine
- **Pre-Trained Pipeline Integration**: Seamlessly loads `retainiq_churn_model.joblib` with automated attribute compatibility patching.
- **Expected Features**: `Recency`, `Frequency`, `Monetary`, `UniqueProducts`, `Tenure`, `AvgOrderValue`.
- **Live What-If Inference**: Interactive tool allowing users to enter custom values or select preset accounts to compute real-time probabilities through the `StandardScaler` + `LogisticRegression` pipeline.
- **Historical Governance**: Explicitly labels out-of-fold scores as historical evaluation metrics and provides operational caveats against uncalibrated live drift.

### 4. Retention Workspace
- **Risk-Weighted Historical Spend Priority Ranking**: Prioritizes accounts by $\text{Monetary} \times P(\text{Churn})$, ranking customers by risk-weighted historical spend to assist commercial teams with prioritization (not actual or future revenue loss).
- **Strategic Action Playbooks**:
  - **High Risk ($P \ge 0.70$)**: Priority personal retention outreach, relationship discovery, margin-conscious win-back credits.
  - **Medium Risk ($0.40 \le P < 0.70$)**: Targeted multi-step re-engagement drips, replenishment reminders, personalized product recommendations.
  - **Low Risk ($P < 0.40$)**: Loyalty milestone perks, category cross-selling, VIP appreciation gifts.
- **CRM Export**: One-click export of filtered retention cohorts to CSV with channel recommendations and incentive tiers.
- **Hypothetical Scenario Simulator**: Explore theoretical campaign budgets, response rates, retention uplifts, and hypothetical ROIs (with explicit non-factual disclaimers).

### 5. Model Performance & Evaluation
- **Benchmark Comparison**: Side-by-side performance comparison of Logistic Regression vs Random Forest.
- **Diagnostic Charts**: Interactive Confusion Matrix heatmap (223 TN, 161 FP, 68 FN, 222 TP) and Receiver Operating Characteristic (ROC) curve.
- **Permutation Feature Importance**: Displays true drop in ROC-AUC when features are shuffled:
  - Recency: `+0.0392`
  - UniqueProducts: `+0.0272`
  - Monetary: `+0.0210`
  - Frequency: `+0.0123`
  - AvgOrderValue: `+0.0039`
  - Tenure: `-0.0014`

### 6. Data Quality & Pipeline Audit
- **Pipeline Lineage Funnel**: Visualizes data cleaning stages:
  - Raw Records Ingested: 541,909
  - Exact Duplicates Purged: 5,268 (0.97%)
  - Deduplicated Records: 536,641
  - Valid Purchase Records: 524,878
  - Customer-Identified Records: 392,692
  - Missing CustomerID Records: 132,186 (25.18% anonymous/guest transactions)
- **Integrity Audits**: Date span validation (2010-12-01 to 2011-12-09), column-level null analysis, and sample dataset CSV exports.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- pip

### 1. Clone or Open the Repository
```bash
cd RetainIQ
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest tests/test_retainiq.py
```
*Expected: 7 passed tests verifying data integrity, model inference, and metric consistency.*

### 4. Launch the Streamlit Platform
```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 🔬 Machine Learning Pipeline Details

- **Model Type**: Scikit-Learn `Pipeline` (`StandardScaler` followed by `LogisticRegression` with `class_weight='balanced'`, `C=1.0`, `penalty='l2'`).
- **Cutoff Date**: `2011-09-10 12:50:00`.
- **Target Definition**: Customer made zero purchases in the 90 days following the cutoff date ($Y = 1$), vs at least one purchase ($Y = 0$).
- **Standardized Coefficients ($\beta$) & Conditional Association**:
  - `Frequency`: `-0.9982` (Strong negative conditional association with churn log-odds, holding spend and recency constant)
  - `UniqueProducts`: `-0.5406` (Catalog diversity conditionally associates with lower churn odds)
  - `Monetary`: `-0.4593` (Higher cumulative spend conditionally associates with lower churn likelihood)
  - `Recency`: `+0.3866` (Lapsed days since last order positively associates with higher churn likelihood)
  - `AvgOrderValue`: `+0.3497` (Holding spend and frequency constant, higher AOV reflects fewer/lumpy purchases, correlating with higher churn odds)
  - `Tenure`: `-0.1325` (Mild negative conditional association once frequency and recency are controlled)

---

## ⚖️ Governance & Compliance

- **No Data Fabrication**: Every metric, chart, and scorecard derives strictly from the actual processed datasets.
- **Preservation of Artifacts**: The original raw Excel dataset and pre-trained joblib model artifact are preserved without modification.
- **Transparent Limitations**: Hypothetical simulations and historical model constraints are clearly communicated across the UI.
