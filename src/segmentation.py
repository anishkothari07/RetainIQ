"""
RetainIQ - Customer Segmentation Module
Manages RFM segment profiles, compares historical cutoff vs full-period cohorts,
and articulates the critical distinction between behavioral RFM and predictive churn risk.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.data_loader import load_customer_dashboard, load_rfm_segments, load_segment_revenue

SEGMENT_DEFINITIONS = {
    "Champions": {
        "description": "Top-tier customers who bought recently, buy often, and spend heavily.",
        "strategy": "Reward loyalty, provide VIP access, seek product advocacy, and avoid aggressive discounts.",
        "icon": "👑"
    },
    "Loyal Customers": {
        "description": "Consistent buyers with strong historical spend and regular purchase cycles.",
        "strategy": "Upsell higher-value lines, invite to referral programs, and recommend personalized bundles.",
        "icon": "💎"
    },
    "Potential Customers": {
        "description": "Recent shoppers with moderate spend and engagement; strong growth headroom.",
        "strategy": "Offer targeted category cross-sells, free shipping thresholds, and welcome incentives.",
        "icon": "🚀"
    },
    "Recent Customers": {
        "description": "First-time or very recent buyers who have made few purchases to date.",
        "strategy": "Deliver high-touch onboarding sequences, customer education, and immediate second-order coupons.",
        "icon": "🌱"
    },
    "At Risk": {
        "description": "Historically valuable or frequent customers whose recency has lapsed significantly.",
        "strategy": "Deploy time-sensitive win-back incentives, personalized outreach, and feedback surveys.",
        "icon": "⚠️"
    },
    "Lost Customers": {
        "description": "Lowest recency, frequency, and spend; long period of dormancy.",
        "strategy": "Low-cost reactivation emails, seasonal catalog announcements, or de-prioritize budget.",
        "icon": "💤"
    }
}


def get_segment_comparison() -> pd.DataFrame:
    """
    Compares segment distributions between the historical cutoff cohort (3,370)
    and the full-period customer dataset (4,338).
    """
    cohort = load_customer_dashboard()
    rfm_full = load_rfm_segments()

    c_counts = cohort["Segment"].value_counts().reset_index()
    c_counts.columns = ["Segment", "CutoffCohortCount"]

    f_counts = rfm_full["Segment"].value_counts().reset_index()
    f_counts.columns = ["Segment", "FullPeriodCount"]

    comp = pd.merge(c_counts, f_counts, on="Segment", how="outer").fillna(0)
    comp["CutoffCohortShare"] = (comp["CutoffCohortCount"] / len(cohort) * 100).round(1)
    comp["FullPeriodShare"] = (comp["FullPeriodCount"] / len(rfm_full) * 100).round(1)

    return comp.sort_values("CutoffCohortCount", ascending=False)


def get_segment_risk_crosstab() -> pd.DataFrame:
    """
    Cross-tabulates historical cutoff RFM Segments against ML Churn Risk Categories.
    Demonstrates that RFM behavior and Churn Risk provide complementary intelligence.
    """
    cohort = load_customer_dashboard()
    ct = pd.crosstab(
        cohort["Segment"],
        cohort["RiskCategory"],
        margins=True,
        margins_name="Total"
    )
    # Ensure standard column order
    cols = [c for c in ["Low", "Medium", "High", "Total"] if c in ct.columns]
    return ct[cols]


def get_segment_profiles(use_cutoff_cohort: bool = True) -> pd.DataFrame:
    """
    Computes mean and median RFM characteristics for each segment.
    """
    df = load_customer_dashboard() if use_cutoff_cohort else load_rfm_segments()

    agg_dict = {
        "CustomerID": "count",
        "Recency": "median",
        "Frequency": "mean",
        "Monetary": "mean"
    }
    if "AvgOrderValue" in df.columns:
        agg_dict["AvgOrderValue"] = "mean"
    if "ChurnProbability" in df.columns:
        agg_dict["ChurnProbability"] = "mean"

    profiles = df.groupby("Segment").agg(agg_dict).reset_index()
    profiles.rename(columns={"CustomerID": "CustomerCount"}, inplace=True)
    profiles["Frequency"] = profiles["Frequency"].round(1)
    profiles["Monetary"] = profiles["Monetary"].round(2)

    if "AvgOrderValue" in profiles.columns:
        profiles["AvgOrderValue"] = profiles["AvgOrderValue"].round(2)
    if "ChurnProbability" in profiles.columns:
        profiles["ChurnProbability"] = (profiles["ChurnProbability"] * 100).round(1)

    return profiles.sort_values("CustomerCount", ascending=False)
