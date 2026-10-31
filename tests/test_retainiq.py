"""
RetainIQ - Test Suite
Verifies data loading, metric calculations, ML model artifact inference,
risk thresholds, segment distributions, and retention simulator calculations.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    load_customer_transactions,
    load_clean_transactions,
    load_customer_dashboard,
    load_rfm_segments,
    load_holdout_predictions
)
from src.churn_engine import (
    load_churn_model_artifact,
    predict_churn,
    classify_risk,
    get_recommended_action
)
from src.customer_analytics import get_executive_summary_metrics
from src.retention_engine import simulate_retention_scenario
from src.metrics import get_holdout_evaluation_metrics


def test_data_loading_shapes():
    """Verify all datasets load with exact expected row counts and columns."""
    cust_tx = load_customer_transactions()
    assert len(cust_tx) == 392692
    assert "CustomerID" in cust_tx.columns
    assert "Revenue" in cust_tx.columns

    clean_tx = load_clean_transactions()
    assert len(clean_tx) == 524878
    assert clean_tx["CustomerID"].isnull().sum() == 132186

    cohort = load_customer_dashboard()
    assert len(cohort) == 3370
    assert "RiskCategory" in cohort.columns
    assert "ChurnProbability" in cohort.columns

    rfm = load_rfm_segments()
    assert len(rfm) == 4338
    assert "RFM_Score" in rfm.columns

    holdout = load_holdout_predictions()
    assert len(holdout) == 674


def test_executive_kpi_consistency():
    """Verify macro executive metrics against notebook ground truth."""
    metrics = get_executive_summary_metrics()

    # Full period metrics
    fp = metrics["full_period"]
    assert fp["total_customers"] == 4338
    assert abs(fp["total_revenue"] - 8887208.89) < 1.0
    assert fp["total_invoices"] == 18532
    assert fp["total_products"] == 3665

    # Historical cohort metrics
    hc = metrics["historical_cohort"]
    assert hc["cohort_customers"] == 3370
    assert hc["churned_customers"] == 1449
    assert hc["retained_customers"] == 1921
    assert abs(hc["churn_rate"] - 0.43) < 0.005
    assert hc["high_risk_count"] == 599
    assert hc["medium_risk_count"] == 1695
    assert hc["low_risk_count"] == 1076


def test_historical_cohort_segment_counts():
    """Verify cutoff cohort segment distributions."""
    cohort = load_customer_dashboard()
    counts = cohort["Segment"].value_counts().to_dict()

    assert counts["Lost Customers"] == 899
    assert counts["Champions"] == 893
    assert counts["Loyal Customers"] == 680
    assert counts["At Risk"] == 449
    assert counts["Recent Customers"] == 235
    assert counts["Potential Customers"] == 214


def test_churn_engine_artifact_and_inference():
    """Verify Joblib model artifact loading and live inference."""
    artifact = load_churn_model_artifact()
    assert artifact["model_name"] == "Logistic Regression"
    assert abs(artifact["cv_roc_auc"] - 0.7470) < 0.001
    assert abs(artifact["holdout_roc_auc"] - 0.7238) < 0.001
    assert len(artifact["features"]) == 6

    # Test sample inference
    res = predict_churn(
        recency=15,
        frequency=5,
        monetary=1500.0,
        unique_products=30,
        tenure=200,
        avg_order_value=300.0
    )
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert res["predicted_churn"] in [0, 1]
    assert res["risk_category"] in ["Low", "Medium", "High"]
    assert isinstance(res["retention_action"], str)


def test_risk_classification_thresholds():
    """Verify empirical risk category boundaries (<0.40, 0.40-0.70, >=0.70)."""
    assert classify_risk(0.10) == "Low"
    assert classify_risk(0.399) == "Low"
    assert classify_risk(0.40) == "Medium"
    assert classify_risk(0.699) == "Medium"
    assert classify_risk(0.70) == "High"
    assert classify_risk(0.99) == "High"

    assert get_recommended_action("High") == "Priority personal retention outreach"
    assert get_recommended_action("Medium") == "Targeted re-engagement campaign"
    assert get_recommended_action("Low") == "Loyalty and repeat-purchase campaign"


def test_holdout_evaluation_metrics():
    """Verify holdout test set evaluation benchmarks."""
    res = get_holdout_evaluation_metrics()
    assert res["sample_size"] == 674
    assert abs(res["accuracy"] - 0.6602) < 0.005
    assert abs(res["roc_auc"] - 0.7238) < 0.005
    assert abs(res["precision"] - 0.58) < 0.01
    assert abs(res["recall"] - 0.77) < 0.01
    assert abs(res["f1_score"] - 0.66) < 0.01

    # Confusion matrix elements: 223 TN, 161 FP, 68 FN, 222 TP
    assert res["tn"] == 223
    assert res["fp"] == 161
    assert res["fn"] == 68
    assert res["tp"] == 222


def test_retention_simulator_calculations():
    """Verify simulation math and edge cases."""
    sim = simulate_retention_scenario(
        target_count=100,
        avg_customer_value=500.0,
        response_rate_pct=10.0,
        uplift_rate_pct=20.0,
        cost_per_customer=5.0
    )
    assert sim["total_campaign_cost"] == 500.0
    assert sim["responding_customers"] == 10.0
    assert sim["retained_customers"] == 2.0
    assert sim["gross_revenue_preserved"] == 1000.0
    assert sim["net_benefit"] == 500.0
    assert sim["roi_pct"] == 100.0


def test_risk_weighted_historical_spend_metric():
    """Verify Risk-Weighted Historical Spend equals Monetary * ChurnProbability."""
    from src.retention_engine import prepare_retention_table
    df = prepare_retention_table()
    assert "RiskWeightedHistoricalSpend" in df.columns
    assert "RevenueAtRisk" in df.columns
    expected = (df["Monetary"] * df["ChurnProbability"]).round(2)
    assert (df["RiskWeightedHistoricalSpend"] == expected).all()


def test_probability_formatting_helpers():
    """Verify probability formatting with >=2 decimal places, small non-zero protection, and missing handling."""
    from src.utils import format_probability_pct, format_probability_decimal

    # Zero values
    assert format_probability_pct(0.0) == "0.00%"
    assert format_probability_decimal(0.0) == "0.0000"

    # Small non-zero values (must not round to 0.00%)
    assert format_probability_pct(0.00005) == "<0.01%"
    assert format_probability_decimal(0.00005) == "<0.0001"

    # Standard values (at least 2 decimal places)
    assert format_probability_pct(0.1234) == "12.34%"
    assert format_probability_decimal(0.1234) == "0.1234"

    # Missing / None values (distinguished from zero)
    assert format_probability_pct(None) == "N/A"
    assert format_probability_decimal(None) == "N/A"
    assert format_probability_pct(np.nan) == "N/A"
    assert format_probability_decimal(np.nan) == "N/A"
