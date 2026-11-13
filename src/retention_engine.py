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

    METHODOLOGICAL NOTICE:
    1. Out-of-Sample Discipline: This optimization must strictly be evaluated on
       held-out / test customers, NEVER on the dataset used to train model parameters.
    2. LTV Proxy Definition: In this historical non-contractual retail dataset, customer LTV
       is represented by historical cumulative monetary spend (£). It serves as a heuristic
       proxy for expected preserved value under the assumption that retained customers continue
       purchasing at their historical baseline rate. It is not a verified forward-looking contractual value.
    3. Scenario Nature: Net expected profits are hypothetical scenario estimates based on assumed
       unit outreach costs and retention save rates, not verified causal treatment effects (uplift).
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


def compare_retention_strategies(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ltv_series: np.ndarray,
    outreach_cost: float = 15.0,
    save_rate: float = 0.20,
    target_count: Optional[int] = None
) -> pd.DataFrame:
    """
    Rigorously benchmarks 5 distinct intervention strategies against the default 0.50 baseline:
    1. Contact Nobody: Zero outreach cost, zero preserved revenue.
    2. Contact Everyone: Blanket outreach across all accounts regardless of risk.
    3. Contact Top-N by Churn Risk: Prioritizes purely by highest P(churn).
    4. Contact Top-N by Expected Value at Risk: Prioritizes by LTV * P(churn).
    5. Optimal Threshold (t*): Selects accounts meeting the profit-maximizing threshold.
    6. (Reference) Default Cutoff (t=0.50): Arbitrary industry 50% cutoff.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    ltv = np.asarray(ltv_series).astype(float)
    n_total = len(y_true)

    # Solve optimal threshold
    opt_res = find_optimal_threshold(y_true, y_prob, ltv, outreach_cost=outreach_cost, save_rate=save_rate)
    opt_t = opt_res["optimal_threshold"]
    k_star = opt_res["opt_targeted_count"]
    k = target_count if target_count is not None else k_star

    rows = []

    # Strategy 1: Contact Nobody
    rows.append({
        "Strategy": "1. Contact Nobody",
        "Targeting Logic": "No intervention",
        "Accounts Contacted": 0,
        "Contact Rate (%)": 0.0,
        "Campaign Cost (£)": 0.0,
        "Churners Saved": 0.0,
        "Gross Saved (£)": 0.0,
        "Net Expected Profit (£)": 0.0,
        "ROI (%)": 0.0
    })

    # Strategy 2: Contact Everyone
    cost_all = n_total * outreach_cost
    saved_churners_all = float(np.sum(y_true == 1) * save_rate)
    gross_all = float(np.sum(ltv[y_true == 1] * save_rate))
    net_all = gross_all - cost_all
    roi_all = (net_all / cost_all * 100.0) if cost_all > 0 else 0.0
    rows.append({
        "Strategy": "2. Contact Everyone",
        "Targeting Logic": "Blanket outreach to entire cohort",
        "Accounts Contacted": n_total,
        "Contact Rate (%)": 100.0,
        "Campaign Cost (£)": cost_all,
        "Churners Saved": round(saved_churners_all, 1),
        "Gross Saved (£)": gross_all,
        "Net Expected Profit (£)": net_all,
        "ROI (%)": roi_all
    })

    # Strategy 3: Top-N by Risk Probability
    k_clamped = max(1, min(k, n_total))
    risk_rank = np.argsort(y_prob)[::-1][:k_clamped]
    cost_risk = k_clamped * outreach_cost
    saved_churners_risk = float(np.sum(y_true[risk_rank] == 1) * save_rate)
    gross_risk = float(np.sum(ltv[risk_rank][y_true[risk_rank] == 1] * save_rate))
    net_risk = gross_risk - cost_risk
    roi_risk = (net_risk / cost_risk * 100.0) if cost_risk > 0 else 0.0
    rows.append({
        "Strategy": f"3. Top-{k_clamped} by Churn Risk",
        "Targeting Logic": f"Highest {k_clamped} accounts by P(churn)",
        "Accounts Contacted": k_clamped,
        "Contact Rate (%)": round(k_clamped / n_total * 100.0, 1),
        "Campaign Cost (£)": cost_risk,
        "Churners Saved": round(saved_churners_risk, 1),
        "Gross Saved (£)": gross_risk,
        "Net Expected Profit (£)": net_risk,
        "ROI (%)": roi_risk
    })

    # Strategy 4: Top-N by Expected Value at Risk (LTV * P(churn))
    evar = ltv * y_prob
    evar_rank = np.argsort(evar)[::-1][:k_clamped]
    cost_evar = k_clamped * outreach_cost
    saved_churners_evar = float(np.sum(y_true[evar_rank] == 1) * save_rate)
    gross_evar = float(np.sum(ltv[evar_rank][y_true[evar_rank] == 1] * save_rate))
    net_evar = gross_evar - cost_evar
    roi_evar = (net_evar / cost_evar * 100.0) if cost_evar > 0 else 0.0
    rows.append({
        "Strategy": f"4. Top-{k_clamped} by Value at Risk",
        "Targeting Logic": f"Highest {k_clamped} accounts by LTV × P(churn)",
        "Accounts Contacted": k_clamped,
        "Contact Rate (%)": round(k_clamped / n_total * 100.0, 1),
        "Campaign Cost (£)": cost_evar,
        "Churners Saved": round(saved_churners_evar, 1),
        "Gross Saved (£)": gross_evar,
        "Net Expected Profit (£)": net_evar,
        "ROI (%)": roi_evar
    })

    # Strategy 5: Optimal Decision Threshold t*
    mask_opt = y_prob >= opt_t
    n_opt = int(np.sum(mask_opt))
    cost_opt = n_opt * outreach_cost
    saved_churners_opt = float(np.sum(mask_opt & (y_true == 1)) * save_rate)
    gross_opt = float(np.sum(ltv[mask_opt & (y_true == 1)] * save_rate))
    net_opt = gross_opt - cost_opt
    roi_opt = (net_opt / cost_opt * 100.0) if cost_opt > 0 else 0.0
    rows.append({
        "Strategy": f"5. Optimal Threshold (t* = {opt_t:.2f})",
        "Targeting Logic": f"Accounts with P(churn) ≥ {opt_t:.2f}",
        "Accounts Contacted": n_opt,
        "Contact Rate (%)": round(n_opt / n_total * 100.0, 1),
        "Campaign Cost (£)": cost_opt,
        "Churners Saved": round(saved_churners_opt, 1),
        "Gross Saved (£)": gross_opt,
        "Net Expected Profit (£)": net_opt,
        "ROI (%)": roi_opt
    })

    # Reference Strategy 6: Default 0.50 Threshold
    mask_50 = y_prob >= 0.50
    n_50 = int(np.sum(mask_50))
    cost_50 = n_50 * outreach_cost
    saved_churners_50 = float(np.sum(mask_50 & (y_true == 1)) * save_rate)
    gross_50 = float(np.sum(ltv[mask_50 & (y_true == 1)] * save_rate))
    net_50 = gross_50 - cost_50
    roi_50 = (net_50 / cost_50 * 100.0) if cost_50 > 0 else 0.0
    rows.append({
        "Strategy": "Reference: Default (t = 0.50)",
        "Targeting Logic": "Accounts with P(churn) ≥ 0.50",
        "Accounts Contacted": n_50,
        "Contact Rate (%)": round(n_50 / n_total * 100.0, 1),
        "Campaign Cost (£)": cost_50,
        "Churners Saved": round(saved_churners_50, 1),
        "Gross Saved (£)": gross_50,
        "Net Expected Profit (£)": net_50,
        "ROI (%)": roi_50
    })

    return pd.DataFrame(rows)


def compute_profit_sensitivity_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ltv_series: np.ndarray,
    save_rates: Optional[List[float]] = None,
    outreach_costs: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Computes a 4x4 sensitivity grid over save_rates in {5, 10, 20, 30}% and
    outreach_costs in {5, 15, 30, 50} GBP showing optimal threshold t*,
    number of contacted accounts, and expected net profit.
    """
    if save_rates is None:
        save_rates = [0.05, 0.10, 0.20, 0.30]
    if outreach_costs is None:
        outreach_costs = [5.0, 15.0, 30.0, 50.0]

    records = []
    for s in save_rates:
        for c in outreach_costs:
            res = find_optimal_threshold(y_true, y_prob, ltv_series, outreach_cost=c, save_rate=s)
            records.append({
                "save_rate": s,
                "save_rate_pct": int(round(s * 100)),
                "outreach_cost": c,
                "cost_str": f"£{c:.0f}",
                "optimal_threshold": res["optimal_threshold"],
                "opt_targeted_count": res["opt_targeted_count"],
                "max_net_profit": res["max_net_profit"],
                "default_net_profit": res["default_net_profit"],
                "profit_lift": res["profit_lift"],
            })

    df = pd.DataFrame(records)

    profit_pivot = df.pivot(index="save_rate_pct", columns="cost_str", values="max_net_profit")
    threshold_pivot = df.pivot(index="save_rate_pct", columns="cost_str", values="optimal_threshold")
    targeted_pivot = df.pivot(index="save_rate_pct", columns="cost_str", values="opt_targeted_count")

    return {
        "records": records,
        "df": df,
        "profit_pivot": profit_pivot,
        "threshold_pivot": threshold_pivot,
        "targeted_pivot": targeted_pivot,
        "save_rates": save_rates,
        "outreach_costs": outreach_costs
    }


def plot_profit_sensitivity_heatmap(grid_dict: Dict[str, Any]) -> Any:
    """
    Renders an interactive Plotly Heatmap showing net profit across unit costs and save rates,
    annotating each cell with optimal threshold t* and targeted account volume.
    """
    import plotly.graph_objects as go
    from src.utils import get_plotly_theme

    df = grid_dict["df"]
    costs = sorted(df["outreach_cost"].unique())
    saves = sorted(df["save_rate"].unique())

    cost_labels = [f"£{int(c)}" for c in costs]
    save_labels = [f"{int(s*100)}%" for s in saves]

    z_matrix = []
    text_matrix = []

    for s in saves:
        z_row = []
        text_row = []
        for c in costs:
            sub = df[(df["save_rate"] == s) & (df["outreach_cost"] == c)].iloc[0]
            prof = sub["max_net_profit"]
            opt_t = sub["optimal_threshold"]
            k = sub["opt_targeted_count"]
            z_row.append(prof)
            text_row.append(f"<b>£{prof:,.0f}</b><br>t*={opt_t:.2f} (N={k})")
        z_matrix.append(z_row)
        text_matrix.append(text_row)

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=cost_labels,
        y=save_labels,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=11, family="Plus Jakarta Sans"),
        colorscale="Blues",
        colorbar=dict(title="Net Profit (£)"),
        hoverongaps=False
    ))

    fig.update_layout(
        **get_plotly_theme(
            title=dict(
                text="<b>Profit Sensitivity Grid (Net Profit, Optimal t*, & Contacted Accounts)</b>",
                font=dict(size=14)
            ),
            xaxis=dict(title="Unit Outreach Cost (£ per Customer)"),
            yaxis=dict(title="Assumed Retention Save Rate (%)"),
            height=340
        )
    )
    return fig


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
