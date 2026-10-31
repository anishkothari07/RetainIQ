"""
RetainIQ - Churn Prediction Engine
ML Pipeline Artifact Inspection, Out-of-Fold Risk Distributions,
and Live What-If Model Inference Tool.
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
    RISK_LOW
)
from src.churn_engine import (
    load_churn_model_artifact,
    predict_churn,
    classify_risk,
    get_recommended_action,
    FEATURE_NAMES
)
from src.data_loader import load_customer_dashboard

st.set_page_config(
    page_title="Churn Prediction Engine | RetainIQ",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()
render_sidebar_brand()

render_page_header(
    title="Churn Prediction Engine",
    subtitle="Supervised ML Classification, Out-of-Fold Cohort Scoring & What-If Inference",
    badge="Machine Learning"
)

# Load artifact safely
try:
    artifact = load_churn_model_artifact()
except Exception as e:
    st.error(f"Failed to load model artifact: {e}")
    st.stop()

cohort_df = load_customer_dashboard()

# Prominent Historical Model Disclaimer
render_alert(
    "<b>⚠️ Historical Model Governance Notice:</b> This model was trained on UK Retail transactions up to <b>Sep 10, 2011</b>, "
    "with a 90-day inactivity observation window (to Dec 09, 2011). Out-of-fold probabilities displayed in this workspace "
    "evaluate historical holdout cohort performance. Do not apply this 2011-trained model to arbitrary current 2026 data "
    "without feature pipeline realignment and drift re-estimation.",
    alert_type="warning",
    icon="⚠️"
)

# Model Artifact Metadata Overview
st.markdown("#### 1. Machine Learning Model Artifact Specifications")
a1, a2, a3, a4 = st.columns(4)

with a1:
    render_kpi_card(
        label="Algorithm Pipeline",
        value="StandardScaler + LR",
        subtext="Balanced L2 Logistic Regression",
        accent="indigo"
    )
with a2:
    render_kpi_card(
        label="5-Fold CV ROC-AUC",
        value=f"{artifact['cv_roc_auc']:.4f}",
        subtext=f"Std Dev: ±{artifact['cv_std']:.4f}",
        accent="emerald"
    )
with a3:
    render_kpi_card(
        label="Holdout Test ROC-AUC",
        value=f"{artifact['holdout_roc_auc']:.4f}",
        subtext="Evaluated on 674 holdout customers",
        accent="indigo"
    )
with a4:
    render_kpi_card(
        label="Historical Cutoff Date",
        value="2011-09-10",
        subtext="90-day forward churn horizon",
        accent="amber"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Tabs: Out-of-Fold Cohort vs Live What-If Inference vs Model Weights
tab1, tab2, tab3 = st.tabs([
    "📊 Historical Out-of-Fold Scoring",
    "⚡ Live What-If Inference Simulator",
    "🔬 Pipeline Coefficients & Feature Weights"
])

with tab1:
    st.markdown("#### Historical Cohort Predictions (N = 3,370)")
    st.markdown(
        "Out-of-fold cross-validated probabilities for the historical cutoff cohort. "
        "High-risk accounts are identified for immediate retention intervention."
    )

    k1, k2, k3 = st.columns(3)
    high_count = (cohort_df["RiskCategory"] == "High").sum()
    med_count = (cohort_df["RiskCategory"] == "Medium").sum()
    low_count = (cohort_df["RiskCategory"] == "Low").sum()

    with k1:
        render_kpi_card(
            label="High Risk (P ≥ 0.70)",
            value=f"{format_number(high_count)} accounts",
            subtext=f"{(high_count/len(cohort_df)*100):.1f}% of cohort — Priority Outreach",
            accent="rose"
        )
    with k2:
        render_kpi_card(
            label="Medium Risk (0.40 ≤ P < 0.70)",
            value=f"{format_number(med_count)} accounts",
            subtext=f"{(med_count/len(cohort_df)*100):.2f}% of cohort — Targeted Re-engagement",
            accent="amber"
        )
    with k3:
        render_kpi_card(
            label="Low Risk (P < 0.40)",
            value=f"{format_number(low_count)} accounts",
            subtext=f"{(low_count/len(cohort_df)*100):.2f}% of cohort — Loyalty & Repeat",
            accent="emerald"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Risk Filter for Table
    col_filter, col_search = st.columns([1, 1])
    with col_filter:
        risk_sel = st.multiselect(
            "Filter Risk Level",
            options=["High", "Medium", "Low"],
            default=["High"],
            help="Select risk levels to display in the table."
        )
    with col_search:
        cid_filter = st.text_input("Filter by CustomerID", value="", placeholder="e.g. 12346")

    df_view = cohort_df.copy()
    if risk_sel:
        df_view = df_view[df_view["RiskCategory"].isin(risk_sel)]
    if cid_filter.strip():
        try:
            df_view = df_view[df_view["CustomerID"].astype(str).str.contains(cid_filter.strip())]
        except Exception:
            pass

    st.dataframe(
        df_view[[
            "CustomerID", "RiskCategory", "ChurnProbability", "ActualChurn", "PredictedChurn",
            "Recency", "Frequency", "Monetary", "UniqueProducts", "Tenure", "AvgOrderValue", "RetentionAction"
        ]].sort_values("ChurnProbability", ascending=False).style.format({
            "ChurnProbability": "{:.4f}",
            "Monetary": "£{:.2f}",
            "AvgOrderValue": "£{:.2f}"
        }),
        use_container_width=True,
        height=380
    )

with tab2:
    st.markdown("#### ⚡ Live What-If Churn Inference Engine (Hypothetical Simulation)")
    st.markdown(
        "Enter hypothetical or observed feature values to compute live churn probability through the trained "
        "`StandardScaler` + `LogisticRegression` pipeline artifact. <br>"
        "<b>Model Governance Notice:</b> For hypothetical or prospective inputs, actual churn outcomes are fundamentally <b>Unknown (N/A)</b> "
        "and cannot be verified against historical records.",
        unsafe_allow_html=True
    )

    c_left, c_right = st.columns([1.1, 0.9])

    with c_left:
        with st.container(border=True):
            st.markdown("##### ⚙️ Input Feature Parameters")

            # Preset customer selector
            preset_cust = st.selectbox(
                "Load Features from Existing Customer (Optional)",
                options=["Custom Values"] + cohort_df["CustomerID"].head(20).astype(str).tolist()
            )

            default_rec = 45
            default_freq = 3
            default_mon = 850.0
            default_prod = 20
            default_tenure = 180
            default_aov = 283.33

            if preset_cust != "Custom Values":
                row = cohort_df[cohort_df["CustomerID"] == int(preset_cust)].iloc[0]
                default_rec = int(row["Recency"])
                default_freq = int(row["Frequency"])
                default_mon = float(row["Monetary"])
                default_prod = int(row["UniqueProducts"])
                default_tenure = int(row["Tenure"])
                default_aov = float(row["AvgOrderValue"])

            col_in1, col_in2 = st.columns(2)
            with col_in1:
                rec_val = st.number_input("Recency (Days since last order)", min_value=0, max_value=400, value=default_rec, step=1)
                freq_val = st.number_input("Frequency (Unique invoices)", min_value=1, max_value=250, value=default_freq, step=1)
                mon_val = st.number_input("Monetary Value (£)", min_value=1.0, max_value=300000.0, value=default_mon, step=50.0)

            with col_in2:
                prod_val = st.number_input("Unique Products (SKUs)", min_value=1, max_value=2000, value=default_prod, step=1)
                tenure_val = st.number_input("Tenure (Days since first order)", min_value=0, max_value=400, value=default_tenure, step=5)
                aov_val = st.number_input("Average Order Value (£)", min_value=1.0, max_value=100000.0, value=default_aov, step=25.0)

            run_pred = st.button("🚀 Calculate Live Churn Probability", use_container_width=True)

    with c_right:
        with st.container(border=True):
            st.markdown("##### 🔮 Inference Diagnostic Output")

            # Run prediction
            pred_res = predict_churn(
                recency=rec_val,
                frequency=freq_val,
                monetary=mon_val,
                unique_products=prod_val,
                tenure=tenure_val,
                avg_order_value=aov_val
            )

            p_churn = pred_res["churn_probability"]
            # Format probability with at least 2 decimal places, preventing 0.0% rounding
            if p_churn == 0.0:
                formatted_prob = "0.00%"
            elif p_churn * 100 < 0.01:
                formatted_prob = "<0.01%"
            else:
                formatted_prob = f"{p_churn * 100:.2f}%"

            risk_c = pred_res["risk_category"]
            action = pred_res["retention_action"]

            badge_col = "#EF4444" if risk_c == "High" else ("#F59E0B" if risk_c == "Medium" else "#10B981")

            st.markdown(
                f"""
                <div style="text-align: center; padding: 1.25rem 0;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase;">
                        Computed Churn Likelihood
                    </div>
                    <div style="font-size: 2.75rem; font-weight: 800; color: {badge_col}; line-height: 1.1; margin: 0.35rem 0;">
                        {formatted_prob}
                    </div>
                    <div style="font-size: 0.85rem; color: #64748B; margin-top: 2px;">
                        Raw Score: <code>{p_churn:.4f}</code>
                    </div>
                    <div style="margin: 0.75rem 0;">
                        <span style="background: {badge_col}1A; color: {badge_col}; border: 1px solid {badge_col}4D; padding: 0.35rem 1rem; border-radius: 9999px; font-weight: 700; font-size: 0.9rem;">
                            {risk_c} Risk Classification
                        </span>
                    </div>
                </div>

                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; margin-top: 0.5rem;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">RECOMMENDED STRATEGIC ACTION</div>
                    <div style="font-size: 1rem; font-weight: 700; color: #0F172A; margin: 0.25rem 0;">
                        {action}
                    </div>
                    <div style="font-size: 0.82rem; color: #64748B; margin-top: 0.5rem; line-height: 1.5;">
                        Decision Threshold: <b>0.50</b> &nbsp;|&nbsp; Model Classification: <b>{'Churn (1)' if pred_res['predicted_churn'] == 1 else 'Retain (0)'}</b><br>
                        Actual Historical Outcome: <b>N/A (Prospective / What-If Inference)</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

with tab3:
    st.markdown("#### 🔬 Standardized Pipeline Coefficients & Conditional Associations")
    st.markdown(
        "Standardized coefficients from the trained Logistic Regression pipeline. Features were transformed via `StandardScaler`. "
        "<b>Important:</b> These coefficients reflect <i>conditional statistical associations</i> with churn log-odds, "
        "not causal interventions.",
        unsafe_allow_html=True
    )

    coef_data = []
    for feat, coef in artifact["coefficients"].items():
        impact = "Positive Association (Higher churn odds)" if coef > 0 else "Negative Association (Lower churn odds)"
        coef_data.append({
            "Feature": feat,
            "Standardized Coefficient (β)": round(coef, 4),
            "Conditional Association": impact,
            "Magnitude (|β|)": round(abs(coef), 4)
        })

    coef_df = pd.DataFrame(coef_data).sort_values("Standardized Coefficient (β)", ascending=True)

    fig_coef = px.bar(
        coef_df,
        x="Standardized Coefficient (β)",
        y="Feature",
        orientation="h",
        title="<b>Standardized Logistic Regression Coefficients (β)</b>",
        color="Standardized Coefficient (β)",
        color_continuous_scale=[[0, "#10B981"], [0.5, "#E2E8F0"], [1, "#EF4444"]],
        text=coef_df["Standardized Coefficient (β)"].apply(lambda x: f"{x:+.4f}")
    )
    fig_coef.update_traces(textposition="outside")
    fig_coef.update_layout(
        **get_plotly_theme(
            height=340,
            coloraxis_showscale=False
        )
    )
    st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown(
        """
        **Methodological Interpretation of Conditional Coefficients:**
        - **Frequency (β = -0.9982)**: Strongest negative conditional association with churn log-odds. Holding spend and recency constant, accounts with higher invoice frequency are modeled with lower churn probability. This does not prove that forcing more invoices causally prevents churn.
        - **Unique Products (β = -0.5406)**: Accounts purchasing across wider product selections exhibit lower predicted churn odds, holding frequency constant.
        - **Monetary (β = -0.4593)**: High cumulative spend conditionally associates with lower churn probability.
        - **Recency (β = +0.3866)**: Increased days since last transaction positively associates with churn probability.
        - **AvgOrderValue (β = +0.3497)**: *Why is this positive?* Holding total spend, order count, and recency constant, higher average order value mathematically reflects fewer transactions or lumpy buying patterns. In this historical cohort, such lumpy patterns correlate with higher churn odds. It does not imply that larger purchases cause churn.
        - **Tenure (β = -0.1325)**: Mild negative conditional association with churn log-odds.
        """
    )
