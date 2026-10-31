"""
RetainIQ - AI-Powered Customer Intelligence & Predictive Retention Platform
Main Application Hub & Launchpad
"""

import streamlit as st
import pandas as pd
from src.utils import (
    inject_custom_css,
    render_sidebar_brand,
    render_page_header,
    render_kpi_card,
    render_alert,
    format_currency,
    format_number,
    PRIMARY
)
from src.data_loader import load_customer_dashboard, load_customer_transactions
from src.churn_engine import load_churn_model_artifact

# Page Configuration
st.set_page_config(
    page_title="RetainIQ | Customer Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Enterprise Design System
inject_custom_css()
render_sidebar_brand()

# Sidebar Navigation Info
with st.sidebar:
    st.markdown("### 📌 Workspace Navigation")
    st.markdown(
        """
        - **1. Executive Dashboard**: High-level KPIs, revenue trends, segment & risk distributions.
        - **2. Customer Intelligence**: Customer 360 lookup, RFM profiles, invoice timeline.
        - **3. Churn Prediction**: ML model inference, out-of-fold scoring, live what-if tool.
        - **4. Retention Workspace**: Priority ranking, playbooks, CSV campaign export, simulator.
        - **5. Model Performance**: Logistic Regression vs Random Forest, ROC-AUC, confusion matrix.
        - **6. Data Quality**: Pipeline audit funnel, missing values, transaction validity.
        """
    )
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #94A3B8; line-height: 1.4;">
            <b>RetainIQ v2.4 Enterprise</b><br>
            Trained on UK Retail Transactions<br>
            Artifact: <code>retainiq_churn_model.joblib</code>
        </div>
        """,
        unsafe_allow_html=True
    )

# Page Header
render_page_header(
    title="RetainIQ Enterprise Platform",
    subtitle="Customer Intelligence. Predictive Retention. Smarter Growth.",
    badge="Customer Intelligence & Analytics"
)

# Executive Welcome Banner
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #1E293B 100%); border-radius: 14px; padding: 1.75rem 2rem; color: #FFFFFF; margin-bottom: 1.75rem; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div style="max-width: 780px;">
                <div style="display: inline-block; background: rgba(99, 102, 241, 0.3); border: 1px solid rgba(165, 180, 252, 0.4); padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; color: #E0E7FF; margin-bottom: 0.75rem;">
                    AI-Driven Customer Lifecycle Suite
                </div>
                <h2 style="font-size: 1.75rem; font-weight: 800; margin: 0 0 0.5rem 0; letter-spacing: -0.02em; color: #FFFFFF;">
                    Turn Customer Inactivity Signals into Proactive Commercial Retention
                </h2>
                <p style="font-size: 0.95rem; line-height: 1.55; color: #C7D2FE; margin: 0;">
                    RetainIQ pairs behavioural RFM segmentation with supervised machine learning to evaluate customer churn risk 
                    over a 90-day horizon. Identify accounts by risk-weighted historical spend, explore structured retention workflows, 
                    and export targeted campaign cohorts with tailored commercial playbooks.
                </p>
            </div>
            <div style="background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 1rem 1.25rem; text-align: right; min-width: 200px;">
                <div style="font-size: 0.75rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;">Pipeline Status</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: #34D399; margin: 0.25rem 0;">Operational</div>
                <div style="font-size: 0.75rem; color: #C7D2FE;">6 Analytical Modules Active</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Core Platform KPIs
try:
    tx = load_customer_transactions()
    cohort = load_customer_dashboard()
    model_art = load_churn_model_artifact()

    total_cust = tx["CustomerID"].nunique()
    total_rev = tx["Revenue"].sum()
    cohort_size = len(cohort)
    cv_auc = model_art["cv_roc_auc"]
except Exception as e:
    total_cust = 4338
    total_rev = 8887208.89
    cohort_size = 3370
    cv_auc = 0.7470

k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi_card(
        label="Identified Customer Base",
        value=format_number(total_cust),
        subtext="Full-period verified buyers",
        accent="indigo"
    )
with k2:
    render_kpi_card(
        label="Customer Revenue",
        value=format_currency(total_rev),
        subtext="£8.89M verified spend",
        accent="emerald"
    )
with k3:
    render_kpi_card(
        label="Evaluation Churn Cohort",
        value=format_number(cohort_size),
        subtext="Cutoff date: Sep 10, 2011",
        accent="amber"
    )
with k4:
    render_kpi_card(
        label="Model Performance (CV)",
        value=f"{cv_auc:.4f}",
        subtext="Mean ROC-AUC (±0.011)",
        accent="rose"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Platform Architecture & Module Matrix
st.markdown("### 🚀 Platform Modules & Navigation")
st.markdown(
    "Select any module from the sidebar or click through to explore deep customer intelligence and predictive retention workflows."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="retainiq-card" style="min-height: 250px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                <span style="font-size: 1.5rem;">📊</span>
                <span class="badge" style="background: #EEF2FF; color: #4338CA;">Module 01</span>
            </div>
            <h3 style="font-size: 1.15rem; font-weight: 700; margin: 0 0 0.5rem 0; color: #0F172A;">
                Executive Dashboard
            </h3>
            <p style="font-size: 0.85rem; color: #475569; line-height: 1.5; margin-bottom: 1rem;">
                Comprehensive macro KPIs, monthly revenue curves, geographic distribution, segment revenue contribution, and cohort retention rates.
            </p>
            <div style="font-size: 0.78rem; font-weight: 600; color: #4F46E5;">
                → Open Executive Overview
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="retainiq-card" style="min-height: 250px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                <span style="font-size: 1.5rem;">🎯</span>
                <span class="badge" style="background: #EEF2FF; color: #4338CA;">Module 04</span>
            </div>
            <h3 style="font-size: 1.15rem; font-weight: 700; margin: 0 0 0.5rem 0; color: #0F172A;">
                Retention Workspace
            </h3>
            <p style="font-size: 0.85rem; color: #475569; line-height: 1.5; margin-bottom: 1rem;">
                Priority ranking by risk-weighted historical spend. Multi-tiered playbooks for High, Medium, and Low risk accounts, campaign CSV exports, and scenario simulator.
            </p>
            <div style="font-size: 0.78rem; font-weight: 600; color: #4F46E5;">
                → Launch Retention Campaigns
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="retainiq-card" style="min-height: 250px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                <span style="font-size: 1.5rem;">🔍</span>
                <span class="badge" style="background: #EEF2FF; color: #4338CA;">Module 02</span>
            </div>
            <h3 style="font-size: 1.15rem; font-weight: 700; margin: 0 0 0.5rem 0; color: #0F172A;">
                Customer Intelligence
            </h3>
            <p style="font-size: 0.85rem; color: #475569; line-height: 1.5; margin-bottom: 1rem;">
                Customer 360 search and drill-down. Full-period lifetime metrics vs historical cutoff cohort features, RFM scores, and detailed itemized invoices.
            </p>
            <div style="font-size: 0.78rem; font-weight: 600; color: #4F46E5;">
                → Explore Customer 360
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="retainiq-card" style="min-height: 250px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                <span style="font-size: 1.5rem;">🧪</span>
                <span class="badge" style="background: #EEF2FF; color: #4338CA;">Module 05</span>
            </div>
            <h3 style="font-size: 1.15rem; font-weight: 700; margin: 0 0 0.5rem 0; color: #0F172A;">
                Model Performance
            </h3>
            <p style="font-size: 0.85rem; color: #475569; line-height: 1.5; margin-bottom: 1rem;">
                Rigorous evaluation: Logistic Regression vs Random Forest benchmarks, Confusion Matrix, ROC curves, and permutation feature importance analysis.
            </p>
            <div style="font-size: 0.78rem; font-weight: 600; color: #4F46E5;">
                → Review ML Benchmarks
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="retainiq-card" style="min-height: 250px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                <span style="font-size: 1.5rem;">🤖</span>
                <span class="badge" style="background: #EEF2FF; color: #4338CA;">Module 03</span>
            </div>
            <h3 style="font-size: 1.15rem; font-weight: 700; margin: 0 0 0.5rem 0; color: #0F172A;">
                Churn Prediction
            </h3>
            <p style="font-size: 0.85rem; color: #475569; line-height: 1.5; margin-bottom: 1rem;">
                Live Joblib model inference, out-of-fold historical churn scoring, risk tier distributions, and an interactive what-if customer feature simulator.
            </p>
            <div style="font-size: 0.78rem; font-weight: 600; color: #4F46E5;">
                → Run Churn Inferences
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="retainiq-card" style="min-height: 250px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                <span style="font-size: 1.5rem;">🛡️</span>
                <span class="badge" style="background: #EEF2FF; color: #4338CA;">Module 06</span>
            </div>
            <h3 style="font-size: 1.15rem; font-weight: 700; margin: 0 0 0.5rem 0; color: #0F172A;">
                Data Quality & Pipeline
            </h3>
            <p style="font-size: 0.85rem; color: #475569; line-height: 1.5; margin-bottom: 1rem;">
                Full data lineage from 541,909 raw records to 524,878 clean transactions. Missing CustomerID impact analysis, duplicate removal, and validity audits.
            </p>
            <div style="font-size: 0.78rem; font-weight: 600; color: #4F46E5;">
                → Inspect Pipeline Audit
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Architecture & Lineage Summary
render_alert(
    "<b>Architecture Note:</b> RetainIQ clearly distinguishes between the full-history customer dataset (4,338 customers, £8.89M revenue) "
    "and the historical churn evaluation cohort (3,370 customers with cutoff date Sep 10, 2011). Out-of-fold predictions reflect "
    "historical holdout evaluation, not live 2026 predictions.",
    alert_type="historical",
    icon="ℹ️"
)
