"""
RetainIQ - Customer Intelligence & 360 Profile
Deep granular customer inspection, multi-dimensional search, sorting,
RFM scoring, predictive churn risk, and complete transaction & invoice timeline.
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
    render_risk_badge,
    render_segment_badge,
    format_currency,
    format_number,
    get_plotly_theme,
    PRIMARY,
    SEGMENT_COLORS
)
from src.data_loader import (
    load_customer_dashboard,
    load_customer_transactions,
    load_rfm_segments
)
from src.customer_analytics import get_customer_profile, get_customer_invoices
from src.churn_engine import get_top_risk_drivers_for_customer
from src.segmentation import (
    get_segment_comparison,
    get_segment_risk_crosstab,
    get_segment_profiles,
    SEGMENT_DEFINITIONS
)

st.set_page_config(
    page_title="Customer Intelligence | RetainIQ",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()
render_sidebar_brand()

render_page_header(
    title="Customer Intelligence & Segmentation",
    subtitle="360° Account Profiles, RFM Behavioral Segmentation & Invoice Timeline",
    badge="Account Deep Dive"
)

# Load datasets
cohort_df = load_customer_dashboard()
rfm_df = load_rfm_segments()
tx_df = load_customer_transactions()

# Sidebar: Search, Filters and Sorting
with st.sidebar:
    st.markdown("### 🔍 Customer Explorer")

    # Search by CustomerID
    all_customer_ids = sorted(cohort_df["CustomerID"].unique().tolist())
    search_input = st.text_input(
        "Search CustomerID",
        value="",
        placeholder="e.g. 12346, 17850, 13668",
        help="Type a CustomerID to jump directly to their 360 profile."
    )

    # Segment filter
    segments_list = ["All Segments"] + sorted(cohort_df["Segment"].unique().tolist())
    selected_segment = st.selectbox("Filter Segment", options=segments_list)

    # Country filter
    countries_list = ["All Countries"] + sorted(cohort_df["Country"].unique().tolist())
    selected_country = st.selectbox("Filter Country", options=countries_list)

    # Risk category filter
    risk_list = ["All Risk Tiers", "High", "Medium", "Low"]
    selected_risk = st.selectbox("Filter Risk Category", options=risk_list)

    # Sorting
    sort_by = st.selectbox(
        "Sort Customer Table By",
        options=["Monetary Value (High to Low)", "Frequency (High to Low)", "Recency (Lowest First)", "Churn Probability (High to Low)"]
    )

# Filter cohort table
filtered_table = cohort_df.copy()
if selected_segment != "All Segments":
    filtered_table = filtered_table[filtered_table["Segment"] == selected_segment]
if selected_country != "All Countries":
    filtered_table = filtered_table[filtered_table["Country"] == selected_country]
if selected_risk != "All Risk Tiers":
    filtered_table = filtered_table[filtered_table["RiskCategory"] == selected_risk]

# Apply sorting
if sort_by == "Monetary Value (High to Low)":
    filtered_table = filtered_table.sort_values("Monetary", ascending=False)
elif sort_by == "Frequency (High to Low)":
    filtered_table = filtered_table.sort_values("Frequency", ascending=False)
elif sort_by == "Recency (Lowest First)":
    filtered_table = filtered_table.sort_values("Recency", ascending=True)
elif sort_by == "Churn Probability (High to Low)":
    filtered_table = filtered_table.sort_values("ChurnProbability", ascending=False)

# Resolve target CustomerID for 360 profile
target_customer_id = None
if search_input.strip():
    try:
        input_id = int(search_input.strip())
        if input_id in all_customer_ids or input_id in rfm_df["CustomerID"].values:
            target_customer_id = input_id
        else:
            st.sidebar.warning(f"CustomerID {input_id} not found in database.")
    except ValueError:
        st.sidebar.error("Please enter a valid numeric CustomerID.")

if target_customer_id is None:
    if not filtered_table.empty:
        target_customer_id = int(filtered_table.iloc[0]["CustomerID"])
    else:
        target_customer_id = 12346

profile = get_customer_profile(target_customer_id)

# Header Banner for Selected Customer
st.markdown(
    f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
        <div>
            <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">Customer 360 Profile</div>
            <div style="font-size: 1.65rem; font-weight: 800; color: #0F172A; margin: 0.2rem 0;">CustomerID: #{target_customer_id}</div>
            <div style="font-size: 0.85rem; color: #475569;">
                Primary Market: <b>{profile['country'] if profile else 'United Kingdom'}</b> &nbsp;|&nbsp; 
                Lifetime Transactions: <b>{profile['transaction_count'] if profile else 0}</b> &nbsp;|&nbsp; 
                Total Invoices: <b>{profile['lifetime_invoices'] if profile else 0}</b>
            </div>
        </div>
        <div style="display: flex; gap: 0.75rem; align-items: center;">
            {render_segment_badge(profile['cutoff_profile']['segment']) if profile and profile['cutoff_profile'] else ''}
            {render_risk_badge(profile['cutoff_profile']['risk_category']) if profile and profile['cutoff_profile'] else ''}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Multi-view Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Customer 360 Profile",
    "👥 RFM Segmentation & Behavior",
    "🧾 Transaction & Invoice History",
    "📋 Customer Cohort Directory"
])

with tab1:
    render_alert(
        "<b>Cutoff Date Alignment:</b> Features below reflect the customer's state as of the <b>Sep 10, 2011 cutoff date</b>, "
        "matching the exact feature inputs used by the predictive churn model. Full lifetime history to Dec 2011 is available in the transaction tab.",
        alert_type="info",
        icon="ℹ️"
    )

    if profile and profile["cutoff_profile"]:
        cp = profile["cutoff_profile"]

        # Metric Cards Row 1
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            render_kpi_card(
                label="Recency",
                value=f"{cp['recency']} d",
                subtext="Days since last order",
                accent="indigo"
            )
        with m2:
            render_kpi_card(
                label="Frequency",
                value=str(cp["frequency"]),
                subtext="Unique invoice count",
                accent="indigo"
            )
        with m3:
            render_kpi_card(
                label="Cutoff Spend",
                value=format_currency(cp["monetary"]),
                subtext="Total revenue at cutoff",
                accent="emerald"
            )
        with m4:
            render_kpi_card(
                label="Avg Order Value",
                value=format_currency(cp["avg_order_value"]),
                subtext="Spend per invoice",
                accent="indigo"
            )
        with m5:
            render_kpi_card(
                label="Unique SKUs",
                value=str(cp["unique_products"]),
                subtext="Distinct products bought",
                accent="indigo"
            )
        with m6:
            render_kpi_card(
                label="Tenure",
                value=f"{cp['tenure']} d",
                subtext="Days since first order",
                accent="indigo"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1.1, 0.9])

        with col_left:
            with st.container(border=True):
                st.markdown("#### 🎯 RFM Behavioral Breakdown")
                st.markdown(
                    f"""
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0;">
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; text-align: center;">
                            <div style="font-size: 0.75rem; font-weight: 700; color: #64748B;">RECENCY SCORE</div>
                            <div style="font-size: 1.8rem; font-weight: 800; color: #4F46E5;">{cp['r_score']} <span style="font-size: 0.9rem; color: #94A3B8;">/ 5</span></div>
                            <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">Quintile rating</div>
                        </div>
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; text-align: center;">
                            <div style="font-size: 0.75rem; font-weight: 700; color: #64748B;">FREQUENCY SCORE</div>
                            <div style="font-size: 1.8rem; font-weight: 800; color: #4F46E5;">{cp['f_score']} <span style="font-size: 0.9rem; color: #94A3B8;">/ 5</span></div>
                            <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">Quintile rating</div>
                        </div>
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; text-align: center;">
                            <div style="font-size: 0.75rem; font-weight: 700; color: #64748B;">MONETARY SCORE</div>
                            <div style="font-size: 1.8rem; font-weight: 800; color: #4F46E5;">{cp['m_score']} <span style="font-size: 0.9rem; color: #94A3B8;">/ 5</span></div>
                            <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">Quintile rating</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div style="border-top: 1px solid #E2E8F0; padding-top: 1rem; margin-top: 1rem;">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #0F172A; margin-bottom: 0.4rem;">Segment Classification</div>
                        <div style="font-size: 0.92rem; color: #334155;">
                            Assigned Segment: <b>{cp['segment']}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col_right:
            with st.container(border=True):
                st.markdown("#### 🤖 Predictive Churn Scoring")

                p_val = cp["churn_probability"]
                if p_val == 0.0:
                    prob_str = "0.00%"
                elif p_val * 100 < 0.01:
                    prob_str = "<0.01%"
                else:
                    prob_str = f"{p_val * 100:.2f}%"

                prob_pct = p_val * 100
                risk_color = "#EF4444" if cp["risk_category"] == "High" else ("#F59E0B" if cp["risk_category"] == "Medium" else "#10B981")

                st.markdown(
                    f"""
                    <div style="margin: 1rem 0;">
                        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.4rem;">
                            <span style="font-size: 0.82rem; font-weight: 700; color: #64748B;">ESTIMATED CHURN PROBABILITY</span>
                            <span style="font-size: 1.5rem; font-weight: 800; color: {risk_color};">{prob_str}</span>
                        </div>
                        <div style="font-size: 0.78rem; color: #64748B; margin-bottom: 0.4rem;">
                            Raw Score: <code>{p_val:.4f}</code>
                        </div>
                        <div style="background: #E2E8F0; border-radius: 9999px; height: 10px; overflow: hidden;">
                            <div style="background: {risk_color}; width: {min(100.0, max(0.0, prob_pct))}%; height: 100%; border-radius: 9999px;"></div>
                        </div>
                    </div>

                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                        <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">RECOMMENDED ACTION</div>
                        <div style="font-size: 0.98rem; font-weight: 700; color: #0F172A; margin: 0.35rem 0;">
                            {cp['retention_action']}
                        </div>
                        <div style="font-size: 0.82rem; color: #64748B; margin-top: 0.5rem; line-height: 1.5;">
                            Risk Level: <b>{cp['risk_category']}</b> &nbsp;|&nbsp; 
                            Historical 2011 90-Day Outcome: <b>{'Churned' if cp['actual_churn'] == 1 else 'Retained'} (Cohort Only)</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Fetch and render Top 3 Customer-Specific Risk Drivers
                drivers = get_top_risk_drivers_for_customer(selected_customer_id)
                if drivers:
                    drivers_html = "".join([f"<li style='margin-bottom: 0.35rem;'><b>{d}</b></li>" for d in drivers])
                    st.markdown(
                        f"""
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.85rem 1rem; margin-top: 0.75rem;">
                            <div style="font-size: 0.75rem; font-weight: 700; color: #4338CA; text-transform: uppercase; letter-spacing: 0.04em;">
                                🧭 Primary Risk Drivers (SHAP Local Attribution)
                            </div>
                            <ul style="font-size: 0.83rem; color: #334155; margin: 0.45rem 0 0 0; padding-left: 1.2rem; line-height: 1.5;">
                                {drivers_html}
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    else:
        st.info("ℹ️ Account joined after Sep 10, 2011 cutoff date. Churn Probability: N/A (Not evaluated in historical 2011 cohort).")

with tab2:
    st.markdown("#### 👥 RFM Segmentation & Behavior Analysis")

    render_alert(
        "<b>Conceptual Distinction:</b> <b>RFM Segmentation</b> describes historical purchasing patterns (backward-looking behavior), "
        "grouping customers by how recently, how frequently, and how much they bought. <br>"
        "<b>Churn Probability</b> estimates the forward-looking likelihood that a customer will not purchase in the subsequent 90 days. "
        "A customer may currently be labeled a 'Champion' in RFM, but if their purchase cadence decelerates, their Churn Probability elevates before their RFM bucket downgrades.",
        alert_type="historical",
        icon="💡"
    )

    # Historical cohort segment counts: Lost: 899, Champions: 893, Loyal: 680, At Risk: 449, Recent: 235, Potential: 214
    seg_col1, seg_col2 = st.columns(2)

    with seg_col1:
        with st.container(border=True):
            seg_profiles = get_segment_profiles(use_cutoff_cohort=True)
            fig_sdist = px.bar(
                seg_profiles,
                x="Segment",
                y="CustomerCount",
                title="<b>Historical Cohort Segment Distribution (N = 3,370)</b>",
                color="Segment",
                color_discrete_map=SEGMENT_COLORS,
                text="CustomerCount"
            )
            fig_sdist.update_traces(textposition="outside")
            fig_sdist.update_layout(**get_plotly_theme(), height=320, showlegend=False)
            st.plotly_chart(fig_sdist, use_container_width=True)

    with seg_col2:
        with st.container(border=True):
            fig_srev = px.bar(
                seg_profiles,
                x="Segment",
                y="Monetary",
                title="<b>Average Customer Spend by Segment (£)</b>",
                color="Segment",
                color_discrete_map=SEGMENT_COLORS,
                text=seg_profiles["Monetary"].apply(format_currency)
            )
            fig_srev.update_traces(textposition="outside")
            fig_srev.update_layout(**get_plotly_theme(), height=320, showlegend=False)
            st.plotly_chart(fig_srev, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Segment vs Risk Cross-tabulation
    st.markdown("##### 🔀 Segment vs Churn Risk Cross-Tabulation")
    crosstab_df = get_segment_risk_crosstab()
    st.dataframe(crosstab_df, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Segment Strategic Playbook Matrix
    st.markdown("##### 📚 Segment Strategy & Recommendations")
    s_cols = st.columns(3)
    idx = 0
    for seg_name, seg_info in SEGMENT_DEFINITIONS.items():
        with s_cols[idx % 3]:
            color = SEGMENT_COLORS.get(seg_name, "#4F46E5")
            st.markdown(
                f"""
                <div class="retainiq-card" style="border-left: 4px solid {color}; min-height: 180px;">
                    <div style="font-size: 1rem; font-weight: 700; color: #0F172A; display: flex; align-items: center; gap: 0.5rem;">
                        <span>{seg_info['icon']}</span> <span>{seg_name}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #64748B; margin: 0.4rem 0;">
                        {seg_info['description']}
                    </div>
                    <div style="font-size: 0.78rem; font-weight: 600; color: #334155; margin-top: 0.5rem; background: #F8FAFC; padding: 0.5rem; border-radius: 6px;">
                        <b>Prescribed Action:</b> {seg_info['strategy']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        idx += 1

with tab3:
    st.markdown("#### 🧾 Complete Invoices & Line Items Timeline")
    invoices_df = get_customer_invoices(target_customer_id)

    if not invoices_df.empty:
        col_inv_left, col_inv_right = st.columns([1, 1])
        with col_inv_left:
            st.markdown(f"**Total Invoices ({len(invoices_df)})**")
            st.dataframe(
                invoices_df.rename(columns={
                    "InvoiceNo": "Invoice #",
                    "InvoiceDate": "Date",
                    "TotalItems": "Items Qty",
                    "UniqueProducts": "SKUs",
                    "InvoiceTotal": "Total (£)"
                }),
                use_container_width=True,
                height=280
            )

        with col_inv_right:
            selected_invoice = st.selectbox(
                "Select Invoice to View Line Items",
                options=invoices_df["InvoiceNo"].tolist()
            )
            line_items = tx_df[
                (tx_df["CustomerID"] == target_customer_id) &
                (tx_df["InvoiceNo"].astype(str) == str(selected_invoice))
            ][["StockCode", "Description", "Quantity", "UnitPrice", "Revenue"]]

            st.markdown(f"**Line Items for Invoice #{selected_invoice}**")
            st.dataframe(
                line_items.rename(columns={
                    "StockCode": "SKU",
                    "UnitPrice": "Unit Price (£)",
                    "Revenue": "Line Total (£)"
                }),
                use_container_width=True,
                height=280
            )
    else:
        st.info(f"No transaction records found for CustomerID #{target_customer_id}.")

with tab4:
    st.markdown(f"#### 📋 Filtered Cohort Directory ({len(filtered_table)} Customers)")
    display_cols = [
        "CustomerID", "Segment", "RiskCategory", "ChurnProbability",
        "Monetary", "Frequency", "Recency", "AvgOrderValue", "Country", "RetentionAction"
    ]
    st.dataframe(
        filtered_table[display_cols].style.format({
            "ChurnProbability": "{:.3f}",
            "Monetary": "£{:.2f}",
            "AvgOrderValue": "£{:.2f}"
        }),
        use_container_width=True,
        height=400
    )
