"""
RetainIQ - Retention Workspace & Campaign Engine
Operational Action Playbooks, Value-at-Risk Priority Ranking,
Custom Campaign CSV Builder, and Hypothetical Scenario Simulator.
"""

import streamlit as st
import pandas as pd
import numpy as np

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
    PRIMARY,
    RISK_HIGH,
    RISK_MEDIUM,
    RISK_LOW
)
from src.retention_engine import (
    RETENTION_PLAYBOOKS,
    prepare_retention_table,
    generate_campaign_export,
    simulate_retention_scenario,
    find_optimal_threshold,
    plot_profit_threshold_curve
)
from src.data_loader import load_customer_dashboard

st.set_page_config(
    page_title="Retention Workspace | RetainIQ",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()
render_sidebar_brand()

render_page_header(
    title="Retention Workspace",
    subtitle="Operational Campaign Builder, Strategic Playbooks & Hypothetical ROI Simulator",
    badge="Action Hub"
)

# Sidebar Filters
with st.sidebar:
    st.markdown("### 🎛️ Retention Filters")

    sel_risk = st.multiselect(
        "Risk Categories",
        options=["High", "Medium", "Low"],
        default=["High", "Medium"],
        help="Filter customers by churn risk level."
    )

    all_segments = ["Champions", "Loyal Customers", "Potential Customers", "Recent Customers", "At Risk", "Lost Customers"]
    sel_segments = st.multiselect(
        "Customer Segments",
        options=all_segments,
        default=[],
        help="Filter by behavioral RFM segment."
    )

    min_spend = st.number_input(
        "Minimum Historical Spend (£)",
        min_value=0.0,
        max_value=100000.0,
        value=100.0,
        step=50.0,
        help="Focus budget on higher-value customer accounts."
    )

    st.markdown("---")
    st.markdown("### 🎯 Profit-Optimal Calibration")
    use_opt_threshold = st.checkbox(
        "Filter by Optimal Threshold (t*)",
        value=False,
        help="Filter retention candidate table to accounts meeting or exceeding the profit-optimal decision threshold."
    )
    opt_threshold_val = st.slider(
        "Applied Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.20,
        step=0.02,
        help="Accounts with ChurnProbability >= this threshold qualify for targeted intervention."
    )

# Prepare filtered retention table
retention_df = prepare_retention_table(
    risk_filter=sel_risk if sel_risk else None,
    segment_filter=sel_segments if sel_segments else None,
    min_monetary=min_spend,
    min_probability=opt_threshold_val if use_opt_threshold else None,
    optimal_threshold=opt_threshold_val if use_opt_threshold else None
)

# Top KPI Summary Cards
s1, s2, s3, s4 = st.columns(4)
total_target = len(retention_df)
total_spend = retention_df["Monetary"].sum() if total_target > 0 else 0
total_at_risk = retention_df["RevenueAtRisk"].sum() if total_target > 0 else 0
avg_prob = (retention_df["ChurnProbability"].mean() * 100) if total_target > 0 else 0

with s1:
    render_kpi_card(
        label="Targeted Accounts",
        value=format_number(total_target),
        subtext="Filtered retention candidates",
        accent="indigo"
    )
with s2:
    render_kpi_card(
        label="Cumulative Spend",
        value=format_currency(total_spend),
        subtext="Total historical revenue",
        accent="emerald"
    )
with s3:
    render_kpi_card(
        label="Risk-Weighted Historical Spend",
        value=format_currency(total_at_risk),
        subtext="Historical Spend × Churn Probability",
        accent="rose"
    )
with s4:
    render_kpi_card(
        label="Average Churn Risk",
        value=f"{avg_prob:.2f}%",
        subtext="Cohort-weighted risk level",
        accent="amber"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Strategic Playbooks Cards
st.markdown("### 📘 Strategic Retention Playbooks")
pb_col1, pb_col2, pb_col3 = st.columns(3)

with pb_col1:
    st.markdown(
        f"""
        <div class="retainiq-card" style="border-top: 4px solid {RISK_HIGH};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-weight: 800; font-size: 1.05rem; color: #991B1B;">HIGH RISK (P ≥ 0.70)</span>
                <span class="badge badge-high">Priority Win-Back</span>
            </div>
            <p style="font-size: 0.84rem; color: #475569; margin-bottom: 0.75rem;">
                Immediate personalized intervention for high-value accounts showing severe inactivity.
            </p>
            <div style="font-size: 0.82rem; font-weight: 700; color: #0F172A; margin-bottom: 0.25rem;">Prescribed Strategy:</div>
            <ul style="font-size: 0.8rem; color: #334155; padding-left: 1.1rem; margin: 0 0 0.75rem 0; line-height: 1.45;">
                <li>Priority personal retention outreach by relationship manager</li>
                <li>Customer relationship follow-up & friction discovery</li>
                <li>Targeted win-back offers & credit vouchers</li>
            </ul>
            <div style="font-size: 0.75rem; color: #64748B;">
                <b>Channels:</b> Phone, Executive Concierge, Direct Mail
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with pb_col2:
    st.markdown(
        f"""
        <div class="retainiq-card" style="border-top: 4px solid {RISK_MEDIUM};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-weight: 800; font-size: 1.05rem; color: #92400E;">MEDIUM RISK (0.40 ≤ P < 0.70)</span>
                <span class="badge badge-medium">Nurturing & Re-engagement</span>
            </div>
            <p style="font-size: 0.84rem; color: #475569; margin-bottom: 0.75rem;">
                Proactive nurture sequences to re-activate interest before churn patterns solidify.
            </p>
            <div style="font-size: 0.82rem; font-weight: 700; color: #0F172A; margin-bottom: 0.25rem;">Prescribed Strategy:</div>
            <ul style="font-size: 0.8rem; color: #334155; padding-left: 1.1rem; margin: 0 0 0.75rem 0; line-height: 1.45;">
                <li>Targeted multi-step re-engagement campaigns</li>
                <li>Personalized product recommendations on past categories</li>
                <li>Timely replenishment reminders & incentives</li>
            </ul>
            <div style="font-size: 0.75rem; color: #64748B;">
                <b>Channels:</b> Automated Email Drip, SMS, On-Site Banners
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with pb_col3:
    st.markdown(
        f"""
        <div class="retainiq-card" style="border-top: 4px solid {RISK_LOW};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-weight: 800; font-size: 1.05rem; color: #065F46;">LOW RISK (P < 0.40)</span>
                <span class="badge badge-low">Loyalty Expansion</span>
            </div>
            <p style="font-size: 0.84rem; color: #475569; margin-bottom: 0.75rem;">
                Deepen customer relationship and maximize customer lifetime value.
            </p>
            <div style="font-size: 0.82rem; font-weight: 700; color: #0F172A; margin-bottom: 0.25rem;">Prescribed Strategy:</div>
            <ul style="font-size: 0.8rem; color: #334155; padding-left: 1.1rem; margin: 0 0 0.75rem 0; line-height: 1.45;">
                <li>Loyalty rewards, tiered perks, and milestone gifts</li>
                <li>Cross-selling complementary merchandise</li>
                <li>Repeat-purchase campaigns & VIP appreciation</li>
            </ul>
            <div style="font-size: 0.75rem; color: #64748B;">
                <b>Channels:</b> VIP Member Portal, Loyalty Newsletter
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Tabs: Campaign Table Builder vs Profit-Optimal Solver vs Action Simulator
tab_campaign, tab_solver, tab_sim = st.tabs([
    "📋 Retention Campaign Builder & Export",
    "🎯 Profit-Optimal Decision Threshold Solver",
    "🧮 Hypothetical Scenario Simulator"
])

with tab_campaign:
    st.markdown(f"#### Priority-Ranked Retention Cohort ({len(retention_df)} Accounts)")
    st.markdown(
        "Customers are ranked by **Risk-Weighted Historical Spend** (Historical Monetary Spend × Churn Probability). "
        "Accounts at the top represent customers ranked by risk-weighted historical spend to assist commercial teams with prioritization. "
        "*(Note: This metric is an internal heuristic prioritization index and does not represent actual, expected, or verified future revenue loss.)*"
    )

    # Export controls
    exp_col1, exp_col2 = st.columns([2, 1])
    with exp_col1:
        campaign_name_input = st.text_input("Campaign Name for CRM Export", value="Q4_High_Value_Winback")
    with exp_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        export_csv_data = generate_campaign_export(
            retention_df,
            campaign_name=campaign_name_input,
            optimal_threshold=opt_threshold_val if use_opt_threshold else None
        ).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Campaign Cohort (CSV)",
            data=export_csv_data,
            file_name=f"{campaign_name_input}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Prepare display dataframe with exact labels
    display_df = retention_df.copy()
    display_df["Historical Spend (£)"] = display_df["Monetary"]
    display_df["Risk-Weighted Spend (£)"] = display_df["RevenueAtRisk"]
    if "TopRiskDriver" not in display_df.columns:
        display_df["TopRiskDriver"] = "Extended Inactivity"

    display_cols = [
        "PriorityRank", "CustomerID", "Segment", "RiskCategory",
        "ChurnProbability", "TopRiskDriver", "Historical Spend (£)", "Risk-Weighted Spend (£)", "RetentionAction", "Country"
    ]
    cols_to_render = [c for c in display_cols if c in display_df.columns]

    # Render Table
    st.dataframe(
        display_df[cols_to_render].style.format({
            "ChurnProbability": "{:.4f}",
            "Historical Spend (£)": "£{:.2f}",
            "Risk-Weighted Spend (£)": "£{:.2f}"
        }),
        use_container_width=True,
        height=400
    )

with tab_solver:
    st.markdown("#### 🎯 Profit-Optimal Decision Threshold Solver")
    st.markdown(
        "Commercial campaigns should not default blindly to an arbitrary 50% statistical cutoff ($t = 0.50$). "
        "RetainIQ maximizes **Net Expected Commercial Profit** by finding the threshold $t^*$ that balances retention revenue against intervention costs: "
        r"$$\text{Net Expected Profit} = \sum_{i \in \text{TP}} (\text{LTV}_i \times \text{SaveRate}) - \sum_{i \in (\text{TP} \cup \text{FP})} \text{OutreachCost}$$",
        unsafe_allow_html=True
    )

    sc1, sc2, sc3 = st.columns([1, 1, 1])
    with sc1:
        solver_cost = st.number_input(
            "Outreach Cost per Account (£)",
            min_value=1.0,
            max_value=250.0,
            value=15.0,
            step=5.0,
            help="Direct unit cost of outreach (concierge call, direct mail voucher, relationship manager outreach)."
        )
    with sc2:
        solver_save_pct = st.slider(
            "Retention Save Rate (%)",
            min_value=1.0,
            max_value=60.0,
            value=20.0,
            step=1.0,
            help="Estimated percentage of contacted churners who are successfully saved."
        )
        solver_save_rate = solver_save_pct / 100.0
    with sc3:
        solver_scope = st.selectbox(
            "Evaluation Dataset",
            options=["Historical Cutoff Cohort (N = 3,370)", "Holdout Test Set (N = 674)"],
            index=0
        )

    # Compute optimal threshold
    if "Holdout" in solver_scope:
        from src.data_loader import load_holdout_predictions
        solver_data = load_holdout_predictions()
    else:
        solver_data = load_customer_dashboard()

    opt_res = find_optimal_threshold(
        y_true=solver_data["ActualChurn"].values,
        y_prob=solver_data["ChurnProbability"].values,
        ltv_series=solver_data["Monetary"].values,
        outreach_cost=solver_cost,
        save_rate=solver_save_rate
    )

    o1, o2, o3, o4 = st.columns(4)
    with o1:
        render_kpi_card(
            label="Profit-Optimal Threshold (t*)",
            value=f"{opt_res['optimal_threshold']:.2f}",
            subtext=f"Default baseline: 0.50",
            accent="emerald"
        )
    with o2:
        render_kpi_card(
            label="Max Expected Net Profit",
            value=format_currency(opt_res["max_net_profit"]),
            subtext="At optimal threshold t*",
            accent="indigo"
        )
    with o3:
        render_kpi_card(
            label="Default Threshold Profit (t=0.50)",
            value=format_currency(opt_res["default_net_profit"]),
            subtext="Baseline uncalibrated profit",
            accent="amber"
        )
    with o4:
        lift_accent = "emerald" if opt_res["profit_lift"] >= 0 else "rose"
        render_kpi_card(
            label="Net Profit Lift",
            value=f"+{format_currency(opt_res['profit_lift'])}",
            subtext="Commercial gain over 0.50 cutoff",
            accent=lift_accent
        )

    # Render Profit Curve
    fig_profit_curve = plot_profit_threshold_curve(opt_res)
    st.plotly_chart(fig_profit_curve, use_container_width=True)

    st.markdown(
        f"""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; margin-top: 0.5rem; font-size: 0.85rem; color: #334155; line-height: 1.55;">
            <b>Decision Strategy Rationale:</b> At an outreach cost of <b>£{solver_cost:.2f}</b> and assumed save rate of <b>{solver_save_pct:.0f}%</b>, 
            the mathematically optimal cutoff is <b>t* = {opt_res['optimal_threshold']:.2f}</b>. 
            Targeting accounts at <code>ChurnProbability ≥ {opt_res['optimal_threshold']:.2f}</code> reaches <b>{opt_res['opt_targeted_count']} accounts</b>, 
            delivering <b>{format_currency(opt_res['max_net_profit'])}</b> in net expected profit—a lift of <b>{format_currency(opt_res['profit_lift'])}</b> 
            over the default 0.50 cutoff.
        </div>
        """,
        unsafe_allow_html=True
    )
        height=400
    )

with tab_sim:
    st.markdown("#### 🧮 Retention Action Simulator (Hypothetical Planning Tool)")

    render_alert(
        "<b>Hypothetical Planning Disclosure:</b> Campaign outcome data (actual responses, conversion uplifts, or redeemed offers) "
        "is not present in this historical dataset. This simulator computes theoretical projections for planning, capacity allocation, "
        "and budgeting scenarios. Do not present these projections as historical fact.",
        alert_type="warning",
        icon="⚠️"
    )

    sim_left, sim_right = st.columns([1, 1.2])

    with sim_left:
        with st.container(border=True):
            st.markdown("##### ⚙️ Simulation Parameters")

            sim_target = st.slider(
                "Number of Targeted Customers",
                min_value=10,
                max_value=max(10, len(retention_df)),
                value=min(250, len(retention_df)),
                step=10,
                help="Number of top priority-ranked customers to include in the campaign."
            )

            sim_response = st.slider(
                "Assumed Campaign Response Rate (%)",
                min_value=1.0,
                max_value=50.0,
                value=12.0,
                step=0.5,
                help="Hypothetical percentage of contacted accounts who open/engage with the campaign."
            )

            sim_uplift = st.slider(
                "Assumed Retention Uplift (%)",
                min_value=1.0,
                max_value=60.0,
                value=25.0,
                step=1.0,
                help="Hypothetical percentage of engaged accounts who are successfully prevented from churning."
            )

            sim_cost = st.slider(
                "Campaign Cost per Customer (£)",
                min_value=0.50,
                max_value=50.00,
                value=5.00,
                step=0.50,
                help="Estimated unit outreach cost (direct mail, sales rep time, voucher cost)."
            )

    with sim_right:
        with st.container(border=True):
            st.markdown("##### 📊 Projected Theoretical Outcomes")

            # Average spend of top target customers
            top_cohort = retention_df.head(sim_target)
            avg_spend_cohort = float(top_cohort["Monetary"].mean()) if len(top_cohort) > 0 else 500.0

            sim_res = simulate_retention_scenario(
                target_count=sim_target,
                avg_customer_value=avg_spend_cohort,
                response_rate_pct=sim_response,
                uplift_rate_pct=sim_uplift,
                cost_per_customer=sim_cost
            )

            r1, r2 = st.columns(2)
            with r1:
                st.metric("Total Campaign Budget", format_currency(sim_res["total_campaign_cost"]))
                st.metric("Hypothetically Retained", f"{sim_res['retained_customers']:.1f} accounts")
            with r2:
                st.metric("Hypothetical Spend Modeled", format_currency(sim_res["gross_revenue_preserved"]))
                st.metric("Hypothetical Scenario Net", format_currency(sim_res["net_benefit"]))

            roi_color = "#10B981" if sim_res["roi_pct"] > 0 else "#EF4444"
            st.markdown(
                f"""
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.85rem; margin-top: 1rem; text-align: center;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #64748B;">THEORETICAL RETURN ON INVESTMENT</div>
                    <div style="font-size: 1.85rem; font-weight: 800; color: {roi_color}; margin: 0.2rem 0;">
                        {sim_res['roi_pct']:.1f}% ROI
                    </div>
                    <div style="font-size: 0.75rem; color: #94A3B8;">
                        Based on assumed £{avg_spend_cohort:.2f} avg spend per account
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
