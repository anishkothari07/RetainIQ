"""
RetainIQ - Retention Engine Module
Implements multi-tiered retention action playbooks, customer priority ranking,
retention campaign export generator, and hypothetical scenario simulation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from src.data_loader import load_customer_dashboard

# Strategic Playbook Matrix
RETENTION_PLAYBOOKS = {
    "High": {
        "title": "High-Risk Win-Back & Personal Outreach",
        "badge_color": "#EF4444",
        "description": "Urgent intervention for high-value customers showing critical inactivity signals.",
        "strategies": [
            "Priority personal retention outreach by customer relationship manager",
            "High-touch customer relationship follow-up & feedback discovery",
            "Targeted win-back offers with tailored discounts or credit vouchers"
        ],
        "channels": ["Direct Phone Call", "Executive Email", "Dedicated Account Manager", "Direct Mail Concierge"],
        "incentive_guideline": "15% - 25% margin-conscious win-back credit or complimentary premium service"
    },
    "Medium": {
        "title": "Medium-Risk Nurturing & Re-engagement",
        "badge_color": "#F59E0B",
        "description": "Proactive lifecycle campaigns to re-energize slipping customer interest before churn solidifies.",
        "strategies": [
            "Targeted multi-step re-engagement email campaigns",
            "Personalized product recommendations based on past purchase history",
            "Timely replenishment reminders and limited-time engagement incentives"
        ],
        "channels": ["Automated Email Drip", "SMS Push Notification", "Dynamic On-Site Banners"],
        "incentive_guideline": "10% order discount, free delivery threshold, or category voucher"
    },
    "Low": {
        "title": "Low-Risk Loyalty & Value Expansion",
        "badge_color": "#10B981",
        "description": "Relationship deepening and lifetime value expansion for healthy, active customers.",
        "strategies": [
            "Exclusive loyalty reward tiers and milestone perks",
            "Cross-selling complementary merchandise and category discovery",
            "Repeat-purchase cadence incentives & surprise VIP appreciation gifts"
        ],
        "channels": ["Loyalty Newsletter", "VIP Member Portal", "In-App Gamification"],
        "incentive_guideline": "VIP early-access to new collections, bonus loyalty points, or gift with purchase"
    }
}


def prepare_retention_table(
    risk_filter: Optional[List[str]] = None,
    segment_filter: Optional[List[str]] = None,
    country_filter: Optional[List[str]] = None,
    min_monetary: float = 0.0,
    min_probability: Optional[float] = None,
    optimal_threshold: Optional[float] = None
) -> pd.DataFrame:
    """
    Builds the operational retention workspace table with priority ranking.
    Priority Score = Monetary * ChurnProbability (Risk-Weighted Historical Spend).
    Calculated as Historical Monetary Spend × Churn Probability. This is a heuristic
    prioritization index and does not represent actual, expected, or verified future revenue loss.
    """
    df = load_customer_dashboard().copy()

    # Risk-Weighted Historical Spend: Historical Monetary Spend × Churn Probability
    df["RiskWeightedHistoricalSpend"] = (df["Monetary"] * df["ChurnProbability"]).round(2)
    # Retain RevenueAtRisk alias for backward compatibility
    df["RevenueAtRisk"] = df["RiskWeightedHistoricalSpend"]
    df["PriorityScore"] = df["RiskWeightedHistoricalSpend"]

    # Flag qualification under optimal decision threshold t*
    if optimal_threshold is not None:
        df["OptimalTargetQualified"] = df["ChurnProbability"] >= optimal_threshold

    # Apply filters
    if risk_filter:
        df = df[df["RiskCategory"].isin(risk_filter)]
    if segment_filter:
        df = df[df["Segment"].isin(segment_filter)]
    if country_filter and "Country" in df.columns:
        df = df[df["Country"].isin(country_filter)]
    if min_monetary > 0:
        df = df[df["Monetary"] >= min_monetary]
    if min_probability is not None:
        df = df[df["ChurnProbability"] >= min_probability]

    # Sort by priority score descending
    df = df.sort_values("PriorityScore", ascending=False).reset_index(drop=True)
    df["PriorityRank"] = df.index + 1

    return df


def generate_campaign_export(
    selected_df: pd.DataFrame,
    campaign_name: str = "RetainIQ_Campaign",
    optimal_threshold: Optional[float] = None
) -> pd.DataFrame:
    """
    Enriches selected customer cohort with campaign execution fields for CRM ingestion.
    """
    export_df = selected_df.copy()

    export_df["CampaignName"] = campaign_name
    export_df["OutreachChannel"] = export_df["RiskCategory"].map(
        lambda r: RETENTION_PLAYBOOKS.get(r, {}).get("channels", ["Email"])[0]
    )
    export_df["IncentiveTier"] = export_df["RiskCategory"].map(
        lambda r: RETENTION_PLAYBOOKS.get(r, {}).get("incentive_guideline", "Standard Offer")
    )
    if optimal_threshold is not None:
        export_df["OptimalThresholdApplied"] = optimal_threshold
        export_df["OptimalTargetQualified"] = export_df["ChurnProbability"] >= optimal_threshold

    output_cols = [
        "PriorityRank",
        "CustomerID",
        "Segment",
        "RiskCategory",
        "ChurnProbability",
        "TopRiskDriver",
        "Monetary",
        "RiskWeightedHistoricalSpend",
        "RevenueAtRisk",
        "RetentionAction",
        "OutreachChannel",
        "IncentiveTier",
        "OptimalTargetQualified",
        "Country"
    ]
    cols_to_use = [c for c in output_cols if c in export_df.columns]
    return export_df[cols_to_use]


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ltv_series: np.ndarray,
    outreach_cost: float = 15.0,
    save_rate: float = 0.20,
    n_steps: int = 81
) -> Dict[str, Any]:
    """
    Sweeps decision thresholds from 0.10 to 0.90 to find t* maximizing:
    Net Expected Profit = sum_{i in TP} (LTV_i * SaveRate) - sum_{i in (TP U FP)} OutreachCost
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    ltv = np.asarray(ltv_series).astype(float)

    thresholds = np.linspace(0.10, 0.90, n_steps)
    net_profits = []
    saved_revs = []
    costs = []
    targeted_counts = []
    tp_counts = []
    fp_counts = []

    for t in thresholds:
        targeted = y_prob >= t
        tp_mask = targeted & (y_true == 1)
        fp_mask = targeted & (y_true == 0)

        saved_rev = float(np.sum(ltv[tp_mask] * save_rate))
        cost = float(np.sum(targeted) * outreach_cost)
        net_profit = float(saved_rev - cost)

        net_profits.append(net_profit)
        saved_revs.append(saved_rev)
        costs.append(cost)
        targeted_counts.append(int(np.sum(targeted)))
        tp_counts.append(int(np.sum(tp_mask)))
        fp_counts.append(int(np.sum(fp_mask)))

    best_idx = int(np.argmax(net_profits))
    idx_50 = int(np.argmin(np.abs(thresholds - 0.50)))

    opt_t = float(thresholds[best_idx])
    max_profit = float(net_profits[best_idx])
    default_profit = float(net_profits[idx_50])

    return {
        "optimal_threshold": round(opt_t, 2),
        "max_net_profit": round(max_profit, 2),
        "default_threshold": 0.50,
        "default_net_profit": round(default_profit, 2),
        "profit_lift": round(max_profit - default_profit, 2),
        "thresholds": [round(float(x), 4) for x in thresholds],
        "net_profits": [round(float(x), 2) for x in net_profits],
        "saved_revenues": [round(float(x), 2) for x in saved_revs],
        "outreach_costs": [round(float(x), 2) for x in costs],
        "targeted_counts": targeted_counts,
        "tp_counts": tp_counts,
        "fp_counts": fp_counts,
        "opt_targeted_count": targeted_counts[best_idx],
        "default_targeted_count": targeted_counts[idx_50],
        "outreach_cost": float(outreach_cost),
        "save_rate": float(save_rate)
    }


def plot_profit_threshold_curve(opt_results: Dict[str, Any]) -> Any:
    """
    Renders an interactive Net Profit vs Decision Threshold curve using Plotly,
    marking the optimal threshold t* vs default 0.50.
    """
    import plotly.graph_objects as go
    from src.utils import get_plotly_theme, PRIMARY

    thresholds = opt_results["thresholds"]
    profits = opt_results["net_profits"]
    opt_t = opt_results["optimal_threshold"]
    max_p = opt_results["max_net_profit"]
    def_p = opt_results["default_net_profit"]

    fig = go.Figure()

    # Zero profit horizontal line
    fig.add_trace(go.Scatter(
        x=[0.10, 0.90],
        y=[0, 0],
        mode="lines",
        name="Break-Even (£0)",
        line=dict(color="#94A3B8", dash="dash", width=1.5)
    ))

    # Net Profit Curve
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=profits,
        mode="lines",
        name="Expected Net Profit",
        line=dict(color=PRIMARY, width=3),
        fill="tozeroy",
        fillcolor="rgba(79, 70, 229, 0.08)"
    ))

    # Marker for Optimal Threshold t*
    fig.add_trace(go.Scatter(
        x=[opt_t],
        y=[max_p],
        mode="markers+text",
        name=f"Optimal t* = {opt_t:.2f} (£{max_p:,.0f})",
        marker=dict(color="#10B981", size=13, symbol="star"),
        text=[f"  <b>Optimal t* = {opt_t:.2f}</b><br>  Max Profit: £{max_p:,.0f}"],
        textposition="top right",
        textfont=dict(color="#065F46", size=11, family="Plus Jakarta Sans")
    ))

    # Marker for Default 0.50
    fig.add_trace(go.Scatter(
        x=[0.50],
        y=[def_p],
        mode="markers+text",
        name=f"Default 0.50 (£{def_p:,.0f})",
        marker=dict(color="#F59E0B", size=10, symbol="diamond"),
        text=[f"  Default 0.50<br>  Profit: £{def_p:,.0f}"],
        textposition="bottom right",
        textfont=dict(color="#92400E", size=10, family="Plus Jakarta Sans")
    ))

    fig.update_layout(
        **get_plotly_theme(
            title=dict(
                text=f"<b>Net Commercial Profit vs. Decision Threshold (Lift at t*: +£{opt_results['profit_lift']:,.0f})</b>",
                font=dict(size=14)
            ),
            xaxis=dict(title="Decision Threshold (t)", range=[0.08, 0.92]),
            yaxis=dict(title="Net Expected Profit (£)"),
            height=340,
            legend=dict(x=0.03, y=0.97)
        )
    )
    return fig


def simulate_retention_scenario(
    target_count: int,
    avg_customer_value: float,
    response_rate_pct: float,
    uplift_rate_pct: float,
    cost_per_customer: float
) -> Dict[str, Any]:
    """
    Computes a hypothetical scenario simulation for planning purposes.
    Note: Output values are theoretical models as true campaign uplift data
    is not present in the historical transaction dataset.
    """
    r_rate = max(0.0, min(100.0, response_rate_pct)) / 100.0
    u_rate = max(0.0, min(100.0, uplift_rate_pct)) / 100.0

    total_campaign_cost = float(target_count * cost_per_customer)
    responding_customers = float(target_count * r_rate)
    retained_customers = float(responding_customers * u_rate)

    gross_revenue_preserved = float(retained_customers * avg_customer_value)
    net_benefit = float(gross_revenue_preserved - total_campaign_cost)

    roi_pct = float((net_benefit / total_campaign_cost * 100.0)) if total_campaign_cost > 0 else 0.0

    return {
        "target_count": int(target_count),
        "total_campaign_cost": total_campaign_cost,
        "responding_customers": round(responding_customers, 1),
        "retained_customers": round(retained_customers, 1),
        "avg_customer_value": avg_customer_value,
        "gross_revenue_preserved": gross_revenue_preserved,
        "net_benefit": net_benefit,
        "roi_pct": roi_pct
    }
