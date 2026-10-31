"""
RetainIQ - Data Quality, Lineage & Validation
Comprehensive Audit of the Data Ingestion Funnel, Missing Value Attribution,
Transaction Integrity Checks, and Data Quality Assurance Reports.
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
    format_number,
    get_plotly_theme,
    PRIMARY
)
from src.preprocessing import get_data_cleaning_audit
from src.data_loader import load_clean_transactions, load_customer_transactions

st.set_page_config(
    page_title="Data Quality & Lineage | RetainIQ",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()
render_sidebar_brand()

render_page_header(
    title="Data Quality & Integrity",
    subtitle="Data Lineage Audit Funnel, Missing Customer Analysis & Transaction Validity",
    badge="Data Governance"
)

audit_counts = get_data_cleaning_audit()
clean_tx = load_clean_transactions()
cust_tx = load_customer_transactions()

# Headline Quality Cards
q1, q2, q3, q4 = st.columns(4)

with q1:
    render_kpi_card(
        label="Raw Transactions Ingested",
        value=format_number(audit_counts["raw_records"]),
        subtext="Online Retail Excel Source",
        accent="indigo"
    )
with q2:
    render_kpi_card(
        label="Exact Duplicates Purged",
        value=format_number(audit_counts["exact_duplicates"]),
        subtext="0.97% duplicate rate removed",
        accent="amber"
    )
with q3:
    render_kpi_card(
        label="Valid Purchase Records",
        value=format_number(audit_counts["valid_purchase_records"]),
        subtext="Positive Quantity & Unit Price",
        accent="emerald"
    )
with q4:
    render_kpi_card(
        label="Customer-Identified Records",
        value=format_number(audit_counts["customer_identified_records"]),
        subtext=f"{audit_counts['missing_customer_id_valid']:,} records lack CustomerID",
        accent="rose"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Visual Audit Funnel
st.markdown("### 🔍 Data Cleaning & Identification Funnel")

funnel_stages = [
    "Raw Ingested Records",
    "Deduplicated Transactions",
    "Valid Purchases (Qty > 0 & Price > 0)",
    "Customer-Attributed Transactions"
]
funnel_values = [
    audit_counts["raw_records"],
    audit_counts["deduplicated_records"],
    audit_counts["valid_purchase_records"],
    audit_counts["customer_identified_records"]
]

fig_funnel = go.Figure(go.Funnel(
    y=funnel_stages,
    x=funnel_values,
    textinfo="value+percent previous",
    marker=dict(color=["#312E81", "#4338CA", "#4F46E5", "#06B6D4"]),
    texttemplate="<b>%{x:,.0f}</b> records (%{percentPrevious:.1%})"
))

fig_funnel.update_layout(
    **get_plotly_theme(),
    title=dict(text="<b>Data Processing Pipeline Retention Funnel</b>", font=dict(size=14)),
    height=330,
)
st.plotly_chart(fig_funnel, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs: Missing Value Analysis vs Transaction Integrity vs Dataset Previews
tab_missing, tab_valid, tab_preview = st.tabs([
    "📊 Missing Value & Identification Audit",
    "✅ Transaction Validity & Summary Stats",
    "🔍 Dataset Previews & Exports"
])

with tab_missing:
    st.markdown("#### Missing CustomerID Impact Analysis")
    st.markdown(
        """
        <div style="font-size: 0.92rem; color: #475569; margin-bottom: 1.25rem; line-height: 1.55;">
            While all purchase records in <code>clean_transactions.csv</code> represent valid economic activity, 
            a substantial portion lacks registered customer identification (guest checkouts or POS unlinked sales).
        </div>
        """,
        unsafe_allow_html=True
    )

    m_col1, m_col2 = st.columns([1, 1.2])

    with m_col1:
        missing_count = audit_counts["missing_customer_id_valid"]
        identified_count = audit_counts["customer_identified_records"]
        total_valid = audit_counts["valid_purchase_records"]

        with st.container(border=True):
            st.markdown(
                """
                <div style="margin-bottom: 0.25rem;">
                    <div style="font-size: 0.98rem; font-weight: 700; color: #0F172A;">Valid Transaction Identification Share</div>
                    <div style="font-size: 0.8rem; color: #64748B; margin-top: 2px;">Share of valid sales with verified CustomerID vs guest transactions</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            fig_pie = px.pie(
                values=[identified_count, missing_count],
                names=["Identified CustomerID", "Missing CustomerID (Guest)"],
                color=["Identified CustomerID", "Missing CustomerID (Guest)"],
                color_discrete_map={
                    "Identified CustomerID": "#10B981",
                    "Missing CustomerID (Guest)": "#F59E0B"
                },
                hole=0.55
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+value")
            fig_pie.update_layout(
                **get_plotly_theme(
                    title="",
                    height=280,
                    margin=dict(l=20, r=20, t=10, b=20)
                )
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with m_col2:
        with st.container(border=True):
            st.markdown(
                f"""
                <div>
                    <div style="font-size: 0.98rem; font-weight: 700; color: #0F172A; margin-bottom: 0.25rem;">Data Governance Principles</div>
                    <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 1rem;">Architectural guidelines for handling unlinked vs verified customer transactions</div>
                    <ul style="font-size: 0.88rem; color: #334155; line-height: 1.65; padding-left: 1.25rem; margin-bottom: 0;">
                        <li style="margin-bottom: 0.75rem;">
                            <b>Separation of Datasets:</b> <code>customer_transactions.csv</code> ({identified_count:,} rows) is used exclusively for customer-level intelligence, RFM modeling, and churn prediction.
                        </li>
                        <li style="margin-bottom: 0.75rem;">
                            <b>No Synthesized Identifiers:</b> Anonymous records ({missing_count:,} rows) are NOT artificially imputed with fake IDs, ensuring strict data integrity.
                        </li>
                        <li>
                            <b>Macro vs Micro Distinction:</b> Macro-level data quality reports capture all {total_valid:,} valid sales, while behavioral models operate strictly on identified accounts.
                        </li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

with tab_valid:
    st.markdown("#### Transaction Validity & Core Parameters")

    v1, v2, v3, v4 = st.columns(4)
    with v1:
        render_kpi_card(
            label="Transaction Date Span",
            value="373 Days",
            subtext="2010-12-01 to 2011-12-09",
            accent="indigo"
        )
    with v2:
        render_kpi_card(
            label="Unique Customers",
            value=format_number(cust_tx["CustomerID"].nunique()),
            subtext="Distinct verified accounts",
            accent="emerald"
        )
    with v3:
        render_kpi_card(
            label="Unique Catalog SKUs",
            value=format_number(cust_tx["StockCode"].nunique()),
            subtext="Distinct products sold",
            accent="indigo"
        )
    with v4:
        render_kpi_card(
            label="Total Invoices",
            value=format_number(cust_tx["InvoiceNo"].nunique()),
            subtext="Unique invoice orders",
            accent="indigo"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Missing value table across columns in clean_transactions
    st.markdown("##### Column-Level Null Analysis (`clean_transactions.csv`)")
    null_data = []
    for col in clean_tx.columns:
        null_count = int(clean_tx[col].isnull().sum())
        null_pct = (null_count / len(clean_tx)) * 100
        null_data.append({
            "Column Name": col,
            "Data Type": str(clean_tx[col].dtype),
            "Total Records": len(clean_tx),
            "Null Count": null_count,
            "Null Percentage": f"{null_pct:.2f}%",
            "Quality Status": "✅ 100% Complete" if null_count == 0 else f"⚠️ {null_pct:.1f}% Missing"
        })

    st.dataframe(pd.DataFrame(null_data), use_container_width=True, hide_index=True)

with tab_preview:
    st.markdown("#### Sample Dataset Previews & Verification Exports")

    ds_option = st.selectbox(
        "Select Dataset to Preview",
        options=["Customer Transactions (392,692 rows)", "Clean Transactions (524,878 rows)"]
    )

    if ds_option.startswith("Customer Transactions"):
        preview_data = cust_tx.head(100)
        csv_bytes = preview_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Sample Customer Transactions (CSV)",
            data=csv_bytes,
            file_name="sample_customer_transactions.csv",
            mime="text/csv"
        )
        st.dataframe(preview_data, use_container_width=True)
    else:
        preview_data = clean_tx.head(100)
        csv_bytes = preview_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Sample Clean Transactions (CSV)",
            data=csv_bytes,
            file_name="sample_clean_transactions.csv",
            mime="text/csv"
        )
        st.dataframe(preview_data, use_container_width=True)
