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
    min_monetary: float = 0.0
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

    # Apply filters
    if risk_filter:
        df = df[df["RiskCategory"].isin(risk_filter)]
    if segment_filter:
        df = df[df["Segment"].isin(segment_filter)]
    if country_filter and "Country" in df.columns:
        df = df[df["Country"].isin(country_filter)]
    if min_monetary > 0:
        df = df[df["Monetary"] >= min_monetary]

    # Sort by priority score descending
    df = df.sort_values("PriorityScore", ascending=False).reset_index(drop=True)
    df["PriorityRank"] = df.index + 1

    return df


def generate_campaign_export(
    selected_df: pd.DataFrame,
    campaign_name: str = "RetainIQ_Campaign"
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

    output_cols = [
        "PriorityRank",
        "CustomerID",
        "Segment",
        "RiskCategory",
        "ChurnProbability",
        "Monetary",
        "RiskWeightedHistoricalSpend",
        "RevenueAtRisk",
        "RetentionAction",
        "OutreachChannel",
        "IncentiveTier",
        "Country"
    ]
    cols_to_use = [c for c in output_cols if c in export_df.columns]
    return export_df[cols_to_use]


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
