"""
RetainIQ - Executive Dashboard
Comprehensive Macro Overview of Customer Intelligence, Revenue Performance,
Cohort Retention, and Churn Risk Distribution.
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
    PRIMARY,
    RISK_HIGH,
    RISK_MEDIUM,
    RISK_LOW,
    SEGMENT_COLORS
)
from src.data_loader import (
    load_customer_transactions,
    load_customer_dashboard,
    load_rfm_segments,
    load_segment_revenue
)
from src.customer_analytics import (
    get_executive_summary_metrics,
    get_monthly_revenue_trend,
    get_country_breakdown,
    get_segment_revenue_summary
)

st.set_page_config(
    page_title="Executive Dashboard | RetainIQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()
render_sidebar_brand()

render_page_header(
    title="Executive Overview",
    subtitle="Enterprise Macro Analytics, Revenue Performance & Churn Risk Landscape",
    badge="Executive Level"
)

# Load core datasets
tx_df = load_customer_transactions()
cohort_df = load_customer_dashboard()
rfm_df = load_rfm_segments()
seg_rev_df = load_segment_revenue()
metrics = get_executive_summary_metrics()

# Sidebar Interactive Filters
with st.sidebar:
    st.markdown("### 🎛️ Dashboard Filters")

    # Country filter
    all_countries = sorted(tx_df["Country"].unique().tolist())
    selected_countries = st.multiselect(
        "Filter by Country",
        options=all_countries,
        default=[],
        help="Select one or more countries. Leave empty to include all countries."
    )

    # Segment filter (Cutoff cohort segments)
    all_segments = ["Champions", "Loyal Customers", "Potential Customers", "Recent Customers", "At Risk", "Lost Customers"]
    selected_segments = st.multiselect(
        "Filter by Segment",
        options=all_segments,
        default=[],
        help="Select customer segments to filter the cohort views."
    )

    # Risk Category filter
    selected_risks = st.multiselect(
        "Filter by Churn Risk",
        options=["High", "Medium", "Low"],
        default=[],
        help="Filter historical cohort by ML risk category."
    )

    # Date range filter
    min_date = tx_df["InvoiceDate"].min().date()
    max_date = tx_df["InvoiceDate"].max().date()
    date_range = st.date_input(
        "Transaction Date Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter full-period transaction revenue."
    )

# Apply filters safely to full transactions
filtered_tx = tx_df.copy()
if selected_countries:
    filtered_tx = filtered_tx[filtered_tx["Country"].isin(selected_countries)]
if len(date_range) == 2:
    start_d, end_d = date_range
    filtered_tx = filtered_tx[
        (filtered_tx["InvoiceDate"].dt.date >= start_d) &
        (filtered_tx["InvoiceDate"].dt.date <= end_d)
    ]

# Apply filters to historical cohort
filtered_cohort = cohort_df.copy()
if selected_countries and "Country" in filtered_cohort.columns:
    filtered_cohort = filtered_cohort[filtered_cohort["Country"].isin(selected_countries)]
if selected_segments:
    filtered_cohort = filtered_cohort[filtered_cohort["Segment"].isin(selected_segments)]
if selected_risks:
    filtered_cohort = filtered_cohort[filtered_cohort["RiskCategory"].isin(selected_risks)]

# Section Distinction Banner
render_alert(
    "<b>Population Architecture Notice:</b> RetainIQ rigorously maintains two distinct datasets: "
    "<b>(1) Full-Period Identified Transactions</b> (4,338 customers, £8.89M verified revenue, Dec 2010 – Dec 2011) and "
    "<b>(2) Historical Churn Evaluation Cohort</b> (3,370 customers evaluated at Sep 10, 2011 cutoff with 90-day churn window). "
    "Do not aggregate these populations together.",
    alert_type="historical",
    icon="🛡️"
)

# KPI Section: Full-Period Verified Transactions
st.markdown("#### 1. Full-Period Customer & Commercial Footprint")
c1, c2, c3, c4 = st.columns(4)

with c1:
    render_kpi_card(
        label="Total Identified Customers",
        value=format_number(filtered_tx["CustomerID"].nunique()),
        subtext=f"Out of {format_number(metrics['full_period']['total_customers'])} total verified accounts",
        accent="indigo"
    )
with c2:
    render_kpi_card(
        label="Customer-Attributed Revenue",
        value=format_currency(filtered_tx["Revenue"].sum()),
        subtext=f"Benchmark: {format_currency(metrics['full_period']['total_revenue'])}",
        accent="emerald"
    )
with c3:
    render_kpi_card(
        label="Total Valid Invoices",
        value=format_number(filtered_tx["InvoiceNo"].nunique()),
        subtext=f"Benchmark: {format_number(metrics['full_period']['total_invoices'])} transactions",
        accent="indigo"
    )
with c4:
    render_kpi_card(
        label="Unique Catalog Products",
        value=format_number(filtered_tx["StockCode"].nunique()),
        subtext=f"Benchmark: {format_number(metrics['full_period']['total_products'])} active SKUs",
        accent="indigo"
    )

st.markdown("<br>", unsafe_allow_html=True)

# KPI Section: Historical Churn Evaluation Cohort
st.markdown("#### 2. Historical Churn Evaluation Cohort (Cutoff: Sep 10, 2011)")
h1, h2, h3, h4 = st.columns(4)

cohort_total = len(filtered_cohort)
churn_count = (filtered_cohort["ActualChurn"] == 1).sum() if "ActualChurn" in filtered_cohort.columns else 0
churn_pct = (churn_count / cohort_total * 100) if cohort_total > 0 else 43.0
high_risk = (filtered_cohort["RiskCategory"] == "High").sum()
medium_risk = (filtered_cohort["RiskCategory"] == "Medium").sum()
low_risk = (filtered_cohort["RiskCategory"] == "Low").sum()

with h1:
    render_kpi_card(
        label="Evaluation Cohort Size",
        value=format_number(cohort_total),
        subtext=f"Base: {format_number(metrics['historical_cohort']['cohort_customers'])} customers",
        accent="amber"
    )
with h2:
    render_kpi_card(
        label="Historical Churn Rate",
        value=f"{churn_pct:.1f}%",
        subtext=f"{format_number(churn_count)} churned / {format_number(cohort_total - churn_count)} retained",
        accent="rose"
    )
with h3:
    render_kpi_card(
        label="High-Risk Customers",
        value=format_number(high_risk),
        subtext="P(Churn) ≥ 0.70 (Priority outreach)",
        accent="rose"
    )
with h4:
    render_kpi_card(
        label="Risk Tier Split (L / M / H)",
        value=f"{low_risk} / {medium_risk} / {high_risk}",
        subtext=f"Low: {metrics['historical_cohort']['low_risk_count']} | Med: {metrics['historical_cohort']['medium_risk_count']} | High: {metrics['historical_cohort']['high_risk_count']}",
        accent="indigo"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Main Visual Analytics Charts (6 Required Charts)
st.markdown("### 📈 Visual Intelligence & Cohort Analytics")

row1_col1, row1_col2 = st.columns(2)

# Chart 1: Revenue by Month
with row1_col1:
    with st.container(border=True):
        st.markdown(
            """
            <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                <span>1. Revenue Performance by Month (£)</span>
                <span style="font-size: 0.75rem; color: #64748B; font-weight: 500;">Dec 2010 – Dec 2011</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        monthly_trend = get_monthly_revenue_trend(filtered_tx)
        fig_month = px.area(
            monthly_trend,
            x="Month",
            y="Revenue",
            labels={"Revenue": "Customer Revenue (£)", "Month": "Invoice Month"},
            color_discrete_sequence=[PRIMARY]
        )
        fig_month.update_traces(line=dict(width=3), fillcolor="rgba(79, 70, 229, 0.12)")
        fig_month.update_layout(
            **get_plotly_theme(
                height=300,
                margin=dict(l=40, r=20, t=10, b=30)
            )
        )
        st.plotly_chart(fig_month, use_container_width=True)

# Chart 2: Revenue by Country
with row1_col2:
    with st.container(border=True):
        st.markdown(
            """
            <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                <span>2. Top 10 Revenue Contributions by Country (£)</span>
                <span style="font-size: 0.75rem; color: #64748B; font-weight: 500;">Geographic Share</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        country_trend = get_country_breakdown(filtered_tx, top_n=10)
        fig_country = px.bar(
            country_trend,
            x="Revenue",
            y="Country",
            orientation="h",
            labels={"Revenue": "Revenue (£)", "Country": ""},
            color="Revenue",
            color_continuous_scale=[[0, "#C7D2FE"], [1, PRIMARY]]
        )
        fig_country.update_layout(
            **get_plotly_theme(
                height=300,
                coloraxis_showscale=False,
                margin=dict(l=40, r=20, t=10, b=30),
                yaxis=dict(autorange="reversed")
            )
        )
        st.plotly_chart(fig_country, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)

# Chart 3: Customer Segment Distribution
with row2_col1:
    with st.container(border=True):
        st.markdown(
            """
            <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                <span>3. Customer Segment Distribution (Cutoff Cohort)</span>
                <span style="font-size: 0.75rem; color: #64748B; font-weight: 500;">N = 3,370</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        seg_counts = filtered_cohort["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        color_map = {k: SEGMENT_COLORS.get(k, "#64748B") for k in seg_counts["Segment"]}

        fig_seg = px.pie(
            seg_counts,
            values="Count",
            names="Segment",
            hole=0.55,
            color="Segment",
            color_discrete_map=color_map
        )
        fig_seg.update_traces(textposition="inside", textinfo="percent+label")
        fig_seg.update_layout(
            **get_plotly_theme(
                height=300,
                showlegend=False,
                margin=dict(l=20, r=20, t=10, b=20)
            )
        )
        st.plotly_chart(fig_seg, use_container_width=True)

# Chart 4: Churn Risk Distribution
with row2_col2:
    with st.container(border=True):
        st.markdown(
            """
            <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                <span>4. Out-of-Fold Churn Risk Distribution (ML Model)</span>
                <span style="font-size: 0.75rem; color: #64748B; font-weight: 500;">Evaluated Cohort</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        risk_order = ["Low", "Medium", "High"]
        risk_df = filtered_cohort["RiskCategory"].value_counts().reindex(risk_order).fillna(0).reset_index()
        risk_df.columns = ["RiskCategory", "Count"]

        risk_colors = [RISK_LOW, RISK_MEDIUM, RISK_HIGH]

        fig_risk = px.bar(
            risk_df,
            x="RiskCategory",
            y="Count",
            labels={"RiskCategory": "Risk Category", "Count": "Number of Customers"},
            color="RiskCategory",
            color_discrete_sequence=risk_colors,
            text="Count"
        )
        fig_risk.update_traces(textposition="outside")
        fig_risk.update_layout(
            **get_plotly_theme(
                height=300,
                showlegend=False,
                margin=dict(l=40, r=20, t=10, b=30)
            )
        )
        st.plotly_chart(fig_risk, use_container_width=True)

row3_col1, row3_col2 = st.columns(2)

# Chart 5: Revenue Contribution by Customer Segment
with row3_col1:
    with st.container(border=True):
        st.markdown(
            """
            <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                <span>5. Revenue Contribution by Customer Segment (£)</span>
                <span style="font-size: 0.75rem; color: #64748B; font-weight: 500;">Full Period Historical Spend</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        seg_rev_summary = get_segment_revenue_summary()
        fig_seg_rev = px.bar(
            seg_rev_summary,
            x="Segment",
            y="Monetary",
            labels={"Monetary": "Total Segment Revenue (£)", "Segment": ""},
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            text=seg_rev_summary["Monetary"].apply(format_currency)
        )
        fig_seg_rev.update_traces(textposition="outside")
        fig_seg_rev.update_layout(
            **get_plotly_theme(
                height=300,
                showlegend=False,
                margin=dict(l=40, r=20, t=10, b=30)
            )
        )
        st.plotly_chart(fig_seg_rev, use_container_width=True)

# Chart 6: Customer Retention Overview (Cohort Retained vs Churned)
with row3_col2:
    with st.container(border=True):
        st.markdown(
            """
            <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                <span>6. Customer Retention Overview (90-Day Outcome)</span>
                <span style="font-size: 0.75rem; color: #64748B; font-weight: 500;">Historical Cohort Result</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        if "ActualChurn" in filtered_cohort.columns:
            retention_df = pd.DataFrame({
                "Status": ["Retained (Active in 90d)", "Churned (Inactive in 90d)"],
                "Count": [
                    int((filtered_cohort["ActualChurn"] == 0).sum()),
                    int((filtered_cohort["ActualChurn"] == 1).sum())
                ]
            })
        else:
            retention_df = pd.DataFrame({
                "Status": ["Retained (Active in 90d)", "Churned (Inactive in 90d)"],
                "Count": [1921, 1449]
            })

        fig_retention = px.pie(
            retention_df,
            values="Count",
            names="Status",
            hole=0.55,
            color="Status",
            color_discrete_map={
                "Retained (Active in 90d)": "#10B981",
                "Churned (Inactive in 90d)": "#EF4444"
            }
        )
        fig_retention.update_traces(textposition="inside", textinfo="percent+value")
        fig_retention.update_layout(
            **get_plotly_theme(
                height=300,
                margin=dict(l=20, r=20, t=10, b=20)
            )
        )
        st.plotly_chart(fig_retention, use_container_width=True)
