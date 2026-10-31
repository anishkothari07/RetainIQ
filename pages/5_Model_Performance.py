"""
RetainIQ - Model Performance & Validation Analytics
Benchmark evaluations, Confusion Matrix, ROC curves,
Logistic Regression vs Random Forest comparisons, and permutation importances.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.utils import (
    inject_custom_css,
    render_sidebar_brand,
    render_page_header,
    render_kpi_card,
    render_alert,
    format_currency,
    format_number,
    get_plotly_theme,
    PRIMARY
)
from src.metrics import (
    get_holdout_evaluation_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_permutation_importance,
    plot_model_comparison,
    BENCHMARKS,
    PERMUTATION_IMPORTANCE
)
from src.churn_engine import load_churn_model_artifact

st.set_page_config(
    page_title="Model Performance | RetainIQ",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()
render_sidebar_brand()

render_page_header(
    title="Model Performance & Evaluation",
    subtitle="Validation Benchmarks, Out-of-Sample Generalization & Feature Attribution",
    badge="ML Governance"
)

# Load holdout metrics and artifact
holdout_res = get_holdout_evaluation_metrics()
artifact = load_churn_model_artifact()

# Governance Distinction Notice
render_alert(
    "<b>⚠️ Historical Model Governance Notice:</b> This model is an offline analytical benchmark trained on historical UK Retail transactions "
    "up to <b>Sep 10, 2011</b> with a 90-day churn observation window. It is an evaluative prototype and is not claimed as a live 2026 production-deployed model. "
    "To maintain strict evaluation integrity, RetainIQ separates <b>5-Fold Cross-Validation ROC-AUC (0.7470 ± 0.0110)</b> from the untouched "
    "<b>20% Holdout Test ROC-AUC (0.7238)</b>.",
    alert_type="warning",
    icon="⚠️"
)

# Core Model KPI Cards (Holdout Evaluation)
st.markdown("#### 1. Candidate Benchmark Model: Logistic Regression (Holdout N = 674)")
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    render_kpi_card(
        label="Holdout ROC-AUC",
        value=f"{holdout_res['roc_auc']:.4f}",
        subtext="Benchmark: 0.7238",
        accent="indigo"
    )
with m2:
    render_kpi_card(
        label="Holdout Accuracy",
        value=f"{holdout_res['accuracy']*100:.2f}%",
        subtext="Benchmark: 66.02%",
        accent="indigo"
    )
with m3:
    render_kpi_card(
        label="Churn Recall (Sensitivity)",
        value=f"{holdout_res['recall']*100:.2f}%",
        subtext="Captures 76.55% of actual churners",
        accent="emerald"
    )
with m4:
    render_kpi_card(
        label="Churn Precision",
        value=f"{holdout_res['precision']*100:.2f}%",
        subtext="57.96% precision on holdout set",
        accent="amber"
    )
with m5:
    render_kpi_card(
        label="Churn F1-Score",
        value=f"{holdout_res['f1_score']:.4f}",
        subtext="Harmonic mean (65.97%)",
        accent="indigo"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Main Performance Charts
st.markdown("### 📊 Diagnostic Visualizations")
row1_left, row1_right = st.columns(2)

with row1_left:
    with st.container(border=True):
        st.markdown(
            """
            <div style="margin-bottom: 0.25rem;">
                <div style="font-size: 0.98rem; font-weight: 700; color: #0F172A;">Holdout Confusion Matrix (N = 674)</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 2px;">Classification accuracy & error breakdown on holdout test set</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        fig_cm = plot_confusion_matrix(holdout_res["confusion_matrix"])
        fig_cm.update_layout(title="", margin=dict(l=40, r=20, t=10, b=40))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown(
            f"""
            <div style="font-size: 0.8rem; color: #475569; margin-top: 0.35rem; line-height: 1.45; padding-top: 0.5rem; border-top: 1px solid #F1F5F9;">
                <b>Holdout Breakdown:</b> True Negatives: <b>{holdout_res['tn']}</b> (Retained correctly) &nbsp;|&nbsp; 
                False Positives: <b>{holdout_res['fp']}</b> (Over-flagged) <br>
                False Negatives: <b>{holdout_res['fn']}</b> (Missed churners) &nbsp;|&nbsp; 
                True Positives: <b>{holdout_res['tp']}</b> (Captured churners)
            </div>
            """,
            unsafe_allow_html=True
        )

with row1_right:
    with st.container(border=True):
        st.markdown(
            """
            <div style="margin-bottom: 0.25rem;">
                <div style="font-size: 0.98rem; font-weight: 700; color: #0F172A;">Receiver Operating Characteristic (ROC) Curve</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 2px;">Discriminatory power across thresholds vs random chance baseline (AUC = 0.50)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        fig_roc = plot_roc_curve(
            holdout_res["roc_curve"]["fpr"],
            holdout_res["roc_curve"]["tpr"],
            holdout_res["roc_auc"]
        )
        fig_roc.update_layout(title="", margin=dict(l=40, r=20, t=10, b=40))
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #475569; margin-top: 0.35rem; line-height: 1.45; padding-top: 0.5rem; border-top: 1px solid #F1F5F9;">
                The ROC curve demonstrates significant discriminatory power well above random chance (Holdout AUC 0.7238 vs 0.50),
                offering an optimal operational balance between recall (76.55%) and false alarm rate.
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

# Permutation Feature Importance & Model Comparison
row2_left, row2_right = st.columns(2)

with row2_left:
    with st.container(border=True):
        st.markdown(
            """
            <div style="margin-bottom: 0.25rem;">
                <div style="font-size: 0.98rem; font-weight: 700; color: #0F172A;">Model Performance Benchmark Comparison</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 2px;">Logistic Regression (Production Candidate) vs Random Forest (Baseline)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        fig_comp = plot_model_comparison()
        fig_comp.update_layout(title="", margin=dict(l=40, r=20, t=10, b=40))
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #475569; margin-top: 0.35rem; line-height: 1.45; padding-top: 0.5rem; border-top: 1px solid #F1F5F9;">
                <b>Candidate Selection Rationale:</b> Logistic Regression was selected over Random Forest due to superior Holdout ROC-AUC 
                (0.7238 vs 0.7044) and higher Churn Recall (76.55% vs 69.00%), paired with transparent linear feature attribution.
            </div>
            """,
            unsafe_allow_html=True
        )

with row2_right:
    with st.container(border=True):
        st.markdown(
            """
            <div style="margin-bottom: 0.25rem;">
                <div style="font-size: 0.98rem; font-weight: 700; color: #0F172A;">Permutation Feature Importance</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 2px;">Mean drop in Holdout ROC-AUC when each feature is permuted</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        fig_perm = plot_permutation_importance()
        fig_perm.update_layout(title="", margin=dict(l=40, r=20, t=10, b=40))
        st.plotly_chart(fig_perm, use_container_width=True)
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #475569; margin-top: 0.35rem; line-height: 1.45; padding-top: 0.5rem; border-top: 1px solid #F1F5F9;">
                <b>Predictive Drivers:</b> <b>Recency (+0.0392)</b> and <b>UniqueProducts (+0.0272)</b> exhibit the highest degradation in discriminative 
                ability when shuffled, followed by Monetary (+0.0210) and Frequency (+0.0123).
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

# Standardized Logistic Regression Coefficients & Causal Governance
st.markdown("#### 2. Standardized Logistic Regression Coefficients & Model Interpretation")

render_alert(
    "<b>Methodological Interpretation Note:</b> The coefficients displayed below are <b>standardized coefficients (β)</b> "
    "from the fitted Logistic Regression pipeline (features scaled to zero mean and unit variance via StandardScaler). <br>"
    "<b>Coefficient signs indicate conditional statistical associations with predicted churn log-odds, NOT causal mechanisms.</b> "
    "They describe how the model adjusts predicted churn likelihood when one standardized feature changes while holding all other features constant.",
    alert_type="info",
    icon="🔬"
)

coef_records = []
for feat, coef in artifact["coefficients"].items():
    direction = "Positive Conditional Association (Higher predicted churn odds)" if coef > 0 else "Negative Conditional Association (Lower predicted churn odds)"
    coef_records.append({
        "Feature": feat,
        "Standardized Coefficient (β)": round(coef, 4),
        "Association Direction": direction,
        "Absolute Magnitude (|β|)": round(abs(coef), 4)
    })

coef_table = pd.DataFrame(coef_records).sort_values("Standardized Coefficient (β)", ascending=True)

col_coef_chart, col_coef_text = st.columns([1.1, 0.9])

with col_coef_chart:
    with st.container(border=True):
        st.markdown(
            """
            <div style="margin-bottom: 0.25rem;">
                <div style="font-size: 0.98rem; font-weight: 700; color: #0F172A;">Standardized Logistic Regression Coefficients (β)</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 2px;">Direction and relative magnitude of standardized feature log-odds</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        fig_coef_bar = px.bar(
            coef_table,
            x="Standardized Coefficient (β)",
            y="Feature",
            orientation="h",
            color="Standardized Coefficient (β)",
            color_continuous_scale=[[0, "#10B981"], [0.5, "#E2E8F0"], [1, "#EF4444"]],
            text=coef_table["Standardized Coefficient (β)"].apply(lambda x: f"{x:+.4f}")
        )
        fig_coef_bar.update_traces(textposition="outside")
        fig_coef_bar.update_layout(
            **get_plotly_theme(
                height=320,
                coloraxis_showscale=False,
                margin=dict(l=40, r=30, t=10, b=30)
            ),
            title=""
        )
        st.plotly_chart(fig_coef_bar, use_container_width=True)
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #475569; margin-top: 0.35rem; line-height: 1.4; padding-top: 0.5rem; border-top: 1px solid #F1F5F9;">
                <b>Direction:</b> Green bars (negative β) lower predicted churn odds; Red bars (positive β) increase predicted churn odds.
            </div>
            """,
            unsafe_allow_html=True
        )

with col_coef_text:
    with st.container(border=True):
        st.markdown(
            """
            <div style="margin-bottom: 0.5rem;">
                <div style="font-size: 0.98rem; font-weight: 700; color: #0F172A;">Conditional Associations Explained</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 2px;">Statistical model adjustments holding other standardized features constant</div>
            </div>
            <ul style="font-size: 0.83rem; color: #334155; line-height: 1.55; padding-left: 1.15rem; margin: 0;">
                <li><b>Frequency (β = -0.9982):</b> Strongest negative conditional association with churn log-odds. Holding spend and recency constant, accounts with higher invoice frequency are modeled with lower churn probability. This does not prove that forcing more invoices causally prevents churn.</li>
                <li><b>UniqueProducts (β = -0.5406):</b> Accounts purchasing across wider SKU catalogs exhibit lower predicted churn odds, conditional on order count.</li>
                <li><b>Monetary (β = -0.4593):</b> Higher historical monetary spend conditionally associates with lower churn likelihood.</li>
                <li><b>Recency (β = +0.3866):</b> Lapsed days since last transaction positively associates with higher churn likelihood.</li>
                <li><b>AvgOrderValue (β = +0.3497):</b> <i>Why is this positive?</i> Holding total spend, frequency, and recency constant, higher average order value mathematically reflects fewer transactions or lumpy buying patterns. In this historical cohort, such lumpy patterns correlate with higher churn odds. It does not imply that larger purchases harm retention.</li>
                <li><b>Tenure (β = -0.1325):</b> Mild negative conditional association with churn odds once frequency and recency are accounted for.</li>
            </ul>
            """,
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

# Full Model Benchmark Comparison Table
st.markdown("#### 3. Detailed Performance Comparison Matrix")

benchmark_df = pd.DataFrame([
    {
        "Candidate Model": "Logistic Regression (L2 Balanced)",
        "Pipeline": "StandardScaler + LogisticRegression",
        "5-Fold CV ROC-AUC": "0.7470 ± 0.0110",
        "Holdout ROC-AUC": "0.7238",
        "Holdout Accuracy": "66.02%",
        "Churn Recall": "76.55%",
        "Churn Precision": "57.96%",
        "Churn F1-Score": "0.6597",
        "Status": "✅ Selected Candidate Model (Offline Benchmark)"
    },
    {
        "Candidate Model": "Random Forest (Baseline)",
        "Pipeline": "StandardScaler + RandomForestClassifier",
        "5-Fold CV ROC-AUC": "0.7280 ± 0.0140",
        "Holdout ROC-AUC": "0.7044",
        "Holdout Accuracy": "63.50%",
        "Churn Recall": "69.00%",
        "Churn Precision": "55.00%",
        "Churn F1-Score": "0.6120",
        "Status": "⚠️ Baseline Benchmark (Lower Recall & AUC)"
    }
])

st.dataframe(benchmark_df, use_container_width=True, hide_index=True)
