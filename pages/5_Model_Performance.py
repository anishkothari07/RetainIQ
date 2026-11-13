"""
RetainIQ - Model Performance & Validation Analytics
Benchmark evaluations, Confusion Matrix, ROC curves, PR curves,
Calibration Reliability curves, 3-Model comparisons, and feature importance.
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
    plot_pr_curve,
    plot_calibration_curve_chart,
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
    subtitle="Validation Benchmarks, Probability Calibration & Multi-Model Evaluation",
    badge="ML Governance"
)

# Load holdout metrics and artifact
holdout_res = get_holdout_evaluation_metrics()
artifact = load_churn_model_artifact()

# Governance Distinction Notice
render_alert(
    "<b>⚠️ Historical Model Governance Notice:</b> This platform benchmarks 3 machine learning pipelines on historical UK Retail transactions "
    "up to <b>Sep 10, 2011</b> with a 90-day churn observation window (N = 3,370 accounts). "
    "All evaluations strictly separate <b>5-Fold Stratified Cross-Validation on the 80% training split (N = 2,696)</b> from the untouched "
    "<b>20% Stratified Holdout Test Set (N = 674)</b> with zero data leakage.",
    alert_type="warning",
    icon="⚠️"
)

# Core Model KPI Cards (Holdout Evaluation)
st.markdown(f"#### 1. Selected Model: {artifact.get('model_name', 'Calibrated HistGradientBoosting')} (Holdout N = 674)")
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    render_kpi_card(
        label="Holdout ROC-AUC",
        value=f"{holdout_res['roc_auc']:.4f}",
        subtext="CV: 0.7502 ± 0.0101",
        accent="indigo"
    )
with m2:
    render_kpi_card(
        label="PR-AUC (Avg Precision)",
        value=f"{holdout_res['pr_auc']:.4f}",
        subtext="No-skill baseline: 0.4300",
        accent="emerald"
    )
with m3:
    render_kpi_card(
        label="Brier Score (Calibration)",
        value=f"{holdout_res['brier_score']:.4f}",
        subtext="Lower is better (Calibrated)",
        accent="indigo"
    )
with m4:
    render_kpi_card(
        label="Precision@Top15%",
        value=f"{holdout_res['precision_at_top15']*100:.2f}%",
        subtext="Accuracy on highest-risk tier",
        accent="amber"
    )
with m5:
    render_kpi_card(
        label="Churn Recall (Sensitivity)",
        value=f"{holdout_res['recall']*100:.2f}%",
        subtext="Captures actual churners",
        accent="emerald"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Main Performance Charts: 2x2 Diagnostic Matrix
st.markdown("### 📊 Diagnostic Visualizations: Discrimination & Calibration")
row1_left, row1_right = st.columns(2)

with row1_left:
    with st.container(border=True):
        st.markdown(
            """
            <div style="margin-bottom: 0.25rem;">
                <div style="font-size: 0.98rem; font-weight: 700; color: #0F172A;">Holdout Confusion Matrix (N = 674)</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 2px;">Classification error and accuracy breakdown on holdout test set</div>
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
                <b>Holdout Breakdown:</b> True Negatives: <b>{holdout_res['tn']}</b> &nbsp;|&nbsp; 
                False Positives: <b>{holdout_res['fp']}</b> <br>
                False Negatives: <b>{holdout_res['fn']}</b> &nbsp;|&nbsp; 
                True Positives: <b>{holdout_res['tp']}</b>
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
            f"""
            <div style="font-size: 0.8rem; color: #475569; margin-top: 0.35rem; line-height: 1.45; padding-top: 0.5rem; border-top: 1px solid #F1F5F9;">
                The ROC curve demonstrates strong out-of-sample discriminatory power (Holdout AUC = <b>{holdout_res['roc_auc']:.4f}</b> vs 0.50 chance).
            </div>
            """,
            unsafe_allow_html=True
        )

row2_left, row2_right = st.columns(2)

with row2_left:
    with st.container(border=True):
        st.markdown(
            """
            <div style="margin-bottom: 0.25rem;">
                <div style="font-size: 0.98rem; font-weight: 700; color: #0F172A;">Precision-Recall (PR) Curve</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 2px;">Precision vs recall tradeoff across thresholds against prevalence rate</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        fig_pr = plot_pr_curve(
            holdout_res["pr_curve"]["recall"],
            holdout_res["pr_curve"]["precision"],
            holdout_res["pr_auc"]
        )
        fig_pr.update_layout(title="", margin=dict(l=40, r=20, t=10, b=40))
        st.plotly_chart(fig_pr, use_container_width=True)
        st.markdown(
            f"""
            <div style="font-size: 0.8rem; color: #475569; margin-top: 0.35rem; line-height: 1.45; padding-top: 0.5rem; border-top: 1px solid #F1F5F9;">
                <b>PR-AUC = {holdout_res['pr_auc']:.4f}</b> significantly exceeds the random no-skill rate of 0.4300,
                confirming superior positive identification precision.
            </div>
            """,
            unsafe_allow_html=True
        )

with row2_right:
    with st.container(border=True):
        st.markdown(
            """
            <div style="margin-bottom: 0.25rem;">
                <div style="font-size: 0.98rem; font-weight: 700; color: #0F172A;">Calibration Reliability Curve</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 2px;">Mean predicted probability vs empirical fraction of positives (Brier check)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        fig_cal = plot_calibration_curve_chart(
            holdout_res["calibration_curve"]["prob_pred"],
            holdout_res["calibration_curve"]["prob_true"],
            holdout_res["brier_score"]
        )
        fig_cal.update_layout(title="", margin=dict(l=40, r=20, t=10, b=40))
        st.plotly_chart(fig_cal, use_container_width=True)
        st.markdown(
            f"""
            <div style="font-size: 0.8rem; color: #475569; margin-top: 0.35rem; line-height: 1.45; padding-top: 0.5rem; border-top: 1px solid #F1F5F9;">
                Sigmoid calibration achieves a low Brier score (<b>{holdout_res['brier_score']:.4f}</b>), ensuring modeled probabilities align with observed commercial lapse rates.
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

# Permutation Feature Importance & Model Comparison
row3_left, row3_right = st.columns(2)

with row3_left:
    with st.container(border=True):
        st.markdown(
            """
            <div style="margin-bottom: 0.25rem;">
                <div style="font-size: 0.98rem; font-weight: 700; color: #0F172A;">3-Model Performance Benchmark Comparison</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 2px;">Calibrated HistGBM vs Logistic Regression vs Random Forest</div>
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
                <b>Selection Rationale:</b> Calibrated HistGBM achieves the best probability calibration (Brier: 0.2112)
                and top-tier Precision@Top15% (68.63%), while non-linear features like recency-to-IPI ratio and spend velocity
                elevate discrimination across high-risk accounts.
            </div>
            """,
            unsafe_allow_html=True
        )

with row3_right:
    with st.container(border=True):
        st.markdown(
            """
            <div style="margin-bottom: 0.25rem;">
                <div style="font-size: 0.98rem; font-weight: 700; color: #0F172A;">Permutation Feature Importance (11 Features)</div>
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
                <b>Key Behavioral Drivers:</b> <b>recency_to_ipi_ratio (+0.0421)</b> is the single strongest predictive feature,
                followed by Recency (+0.0385), Frequency (+0.0265), and spend_velocity_ratio (+0.0215).
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

# Full Model Benchmark Comparison Table
st.markdown("#### 2. Detailed Multi-Model Performance Benchmark Matrix")

benchmark_records = []
for model_name, info in BENCHMARKS.items():
    benchmark_records.append({
        "Model Candidate": model_name,
        "Architecture": info["model_type"],
        "5-Fold CV ROC-AUC": f"{info['cv_roc_auc_mean']:.4f} ± {info['cv_roc_auc_std']:.4f}",
        "CV PR-AUC": f"{info['cv_pr_auc']:.4f}",
        "CV Brier Score": f"{info['cv_brier']:.4f}",
        "Holdout ROC-AUC": f"{info['holdout_roc_auc']:.4f}",
        "Holdout PR-AUC": f"{info['holdout_pr_auc']:.4f}",
        "Holdout Brier": f"{info['holdout_brier']:.4f}",
        "Precision@Top15%": f"{info['precision_at_top15']*100:.1f}%",
        "Churn Recall": f"{info['recall']*100:.1f}%",
        "Status": info["status"]
    })

benchmark_df = pd.DataFrame(benchmark_records)
st.dataframe(benchmark_df, use_container_width=True, hide_index=True)
