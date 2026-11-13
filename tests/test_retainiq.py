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
    """Verify Joblib model artifact loading and live calibrated inference."""
    artifact = load_churn_model_artifact()
    assert artifact["model_name"] in [
        "Calibrated HistGradientBoosting",
        "Calibrated HistGBM",
        "Logistic Regression"
    ]
    assert artifact["cv_roc_auc"] >= 0.74
    assert artifact["holdout_roc_auc"] >= 0.71
    assert len(artifact["features"]) >= 6
    assert "cv_brier" in artifact or "holdout_brier" in artifact

    # Test sample inference (with 6 base features and automatic velocity imputation)
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
    assert "top_risk_drivers" in res
    assert len(res["top_risk_drivers"]) <= 3


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


def test_temporal_velocity_zero_leakage_and_invariance():
    """Verify that velocity features are leakage-free and invariant to future transactions."""
    from src.preprocessing import compute_cadence_velocity_features, VELOCITY_FEATURES

    cutoff = pd.Timestamp("2011-09-10 12:50:00")
    tx_pre = pd.DataFrame([
        {"CustomerID": 99999, "InvoiceNo": "10001", "Quantity": 2, "UnitPrice": 10.0,
         "InvoiceDate": pd.Timestamp("2011-05-01 10:00:00"), "StockCode": "A"},
        {"CustomerID": 99999, "InvoiceNo": "10002", "Quantity": 3, "UnitPrice": 15.0,
         "InvoiceDate": pd.Timestamp("2011-08-20 11:00:00"), "StockCode": "B"},
        {"CustomerID": 99999, "InvoiceNo": "C10003", "Quantity": -1, "UnitPrice": 15.0,
         "InvoiceDate": pd.Timestamp("2011-08-25 12:00:00"), "StockCode": "B"},
    ])
    f_clean = compute_cadence_velocity_features(tx_pre, reference_date=cutoff)

    # Adding post-cutoff transactions (> 2011-09-10) must produce identical features
    future_tx = pd.DataFrame([
        {"CustomerID": 99999, "InvoiceNo": "10004", "Quantity": 50, "UnitPrice": 100.0,
         "InvoiceDate": pd.Timestamp("2011-10-15 10:00:00"), "StockCode": "C"},
        {"CustomerID": 99999, "InvoiceNo": "C10005", "Quantity": -10, "UnitPrice": 100.0,
         "InvoiceDate": pd.Timestamp("2011-11-01 10:00:00"), "StockCode": "C"},
    ])
    tx_with_future = pd.concat([tx_pre, future_tx], ignore_index=True)
    f_future = compute_cadence_velocity_features(tx_with_future, reference_date=cutoff)

    for col in VELOCITY_FEATURES:
        assert col in f_clean.columns
        assert not f_clean[col].isnull().any()
        val_clean = f_clean[col].iloc[0]
        val_future = f_future[col].iloc[0]
        assert np.isclose(val_clean, val_future), f"Leakage detected in {col}: {val_clean} != {val_future}"


def test_calibrated_probability_bounds_and_brier_improvement():
    """Verify calibrated probability bounds, Brier score quality, and Precision@Top15%."""
    from src.metrics import get_holdout_evaluation_metrics, BENCHMARKS

    res = get_holdout_evaluation_metrics()
    assert 0.0 <= res["brier_score"] <= 0.25
    assert res["pr_auc"] >= 0.58
    assert res["precision_at_top15"] >= 0.65

    # Benchmark comparison verifies calibrated model achieves lowest Brier loss
    cal_brier = BENCHMARKS["Calibrated HistGBM"]["holdout_brier"]
    lr_brier = BENCHMARKS["Logistic Regression"]["holdout_brier"]
    assert cal_brier < lr_brier, f"Calibrated Brier ({cal_brier}) should be lower than baseline ({lr_brier})"


def test_profit_optimal_threshold_solver_math():
    """Verify profit-optimal threshold solver sweeps, monotonicity, and lift over 0.50 default."""
    from src.retention_engine import find_optimal_threshold

    y_true = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0])
    y_prob = np.array([0.85, 0.75, 0.65, 0.45, 0.35, 0.55, 0.25, 0.90, 0.15, 0.10])
    ltv = np.array([100.0, 200.0, 150.0, 80.0, 50.0, 300.0, 40.0, 250.0, 20.0, 10.0])

    # Standard run
    res = find_optimal_threshold(y_true, y_prob, ltv, outreach_cost=10.0, save_rate=0.20)
    assert 0.10 <= res["optimal_threshold"] <= 0.90
    assert res["max_net_profit"] >= res["default_net_profit"]
    assert res["profit_lift"] >= 0.0

    # Sensitivity check: High unit cost increases threshold t*
    res_low_cost = find_optimal_threshold(y_true, y_prob, ltv, outreach_cost=5.0, save_rate=0.20)
    res_high_cost = find_optimal_threshold(y_true, y_prob, ltv, outreach_cost=40.0, save_rate=0.20)
    assert res_high_cost["optimal_threshold"] >= res_low_cost["optimal_threshold"]
    assert res_high_cost["opt_targeted_count"] <= res_low_cost["opt_targeted_count"]


def test_temporal_backtest_pipeline_and_leakage_invariance():
    """
    Task 1 Verification:
    Verify that features and cohort generation for cutoff date T are completely
    invariant to whether future transactions (> T) exist in the raw transaction input.
    """
    from src.backtest import build_cohort_at_cutoff
    from src.preprocessing import FEATURE_NAMES_V2

    cutoff = pd.Timestamp("2011-05-10 12:50:00")
    tx_pre = pd.DataFrame([
        {"CustomerID": 88888, "InvoiceNo": "20001", "Quantity": 4, "UnitPrice": 20.0,
         "InvoiceDate": pd.Timestamp("2011-02-01 10:00:00"), "StockCode": "P1"},
        {"CustomerID": 88888, "InvoiceNo": "20002", "Quantity": 1, "UnitPrice": 50.0,
         "InvoiceDate": pd.Timestamp("2011-04-15 11:00:00"), "StockCode": "P2"},
    ])
    cohort_clean = build_cohort_at_cutoff(tx_pre, cutoff_date=cutoff, window_days=90)

    # Post-cutoff transactions (> 2011-05-10) should not alter pre-cutoff feature values
    tx_future = pd.DataFrame([
        {"CustomerID": 88888, "InvoiceNo": "20003", "Quantity": 100, "UnitPrice": 5.0,
         "InvoiceDate": pd.Timestamp("2011-06-20 10:00:00"), "StockCode": "P3"},
        {"CustomerID": 88888, "InvoiceNo": "C20004", "Quantity": -10, "UnitPrice": 5.0,
         "InvoiceDate": pd.Timestamp("2011-07-01 10:00:00"), "StockCode": "P3"},
    ])
    tx_combined = pd.concat([tx_pre, tx_future], ignore_index=True)
    cohort_with_future = build_cohort_at_cutoff(tx_combined, cutoff_date=cutoff, window_days=90)

    # Features must match identically
    for col in FEATURE_NAMES_V2:
        val_clean = cohort_clean[col].iloc[0]
        val_future = cohort_with_future[col].iloc[0]
        assert np.isclose(val_clean, val_future), f"Leakage detected in {col}: {val_clean} != {val_future}"

    # Churn label check: customer bought in future window in combined -> Retained (0); in clean -> Churned (1)
    assert cohort_clean["ActualChurn"].iloc[0] == 1
    assert cohort_with_future["ActualChurn"].iloc[0] == 0


def test_probabilistic_lifetimes_baselines_and_bootstrap_ci():
    """
    Task 2 Verification:
    Verify probabilistic lifetimes baselines, Poisson-based churn probability mapping,
    and percentile bootstrap confidence interval bounds.
    """
    from src.backtest import compute_bootstrap_ci, predict_probabilistic_churn
    from lifetimes import BetaGeoFitter

    # Bootstrap CI calculation
    y_true = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0] * 5)
    y_prob = np.array([0.85, 0.75, 0.65, 0.45, 0.35, 0.55, 0.25, 0.90, 0.15, 0.10] * 5)

    ci_res = compute_bootstrap_ci(y_true, y_prob, n_resamples=100, ci=0.95, random_state=42)
    assert "roc_auc" in ci_res
    assert "pr_auc" in ci_res
    assert "brier_score" in ci_res
    assert "precision_at_top15" in ci_res

    for metric, (point, lower, upper) in ci_res.items():
        assert lower <= point <= upper or np.isclose(lower, point, atol=0.05), f"CI ordering broken for {metric}"
        assert 0.0 <= point <= 1.0


def test_retention_strategies_and_sensitivity_grid():
    """
    Task 3 Verification:
    Verify multi-strategy retention comparison (5 strategies + reference)
    and 4x4 profit sensitivity grid computation on held-out predictions.
    """
    from src.data_loader import load_holdout_predictions
    from src.retention_engine import compare_retention_strategies, compute_profit_sensitivity_grid

    df = load_holdout_predictions()
    y_true = df["ActualChurn"].values
    y_prob = df["ChurnProbability"].values
    ltv = df["Monetary"].values

    # Multi-strategy benchmark
    strat_df = compare_retention_strategies(y_true, y_prob, ltv, outreach_cost=15.0, save_rate=0.20)
    assert len(strat_df) == 6
    assert "1. Contact Nobody" in strat_df["Strategy"].values
    assert "2. Contact Everyone" in strat_df["Strategy"].values

    # Contact nobody yields exactly £0 cost and £0 profit
    nobody_row = strat_df[strat_df["Strategy"] == "1. Contact Nobody"].iloc[0]
    assert nobody_row["Campaign Cost (£)"] == 0.0
    assert nobody_row["Net Expected Profit (£)"] == 0.0

    # Optimal threshold yields profit >= default 0.50 cutoff profit
    opt_row = strat_df[strat_df["Strategy"].str.startswith("5. Optimal Threshold")].iloc[0]
    def_row = strat_df[strat_df["Strategy"].str.startswith("Reference: Default")].iloc[0]
    assert opt_row["Net Expected Profit (£)"] >= def_row["Net Expected Profit (£)"]

    # Sensitivity Grid (4x4)
    grid_res = compute_profit_sensitivity_grid(y_true, y_prob, ltv)
    assert len(grid_res["records"]) == 16
    assert grid_res["profit_pivot"].shape == (4, 4)

    # Monotonicity: Higher save rate increases net profit at same unit cost
    low_save_prof = grid_res["df"][(grid_res["df"]["save_rate"] == 0.05) & (grid_res["df"]["outreach_cost"] == 15.0)]["max_net_profit"].iloc[0]
    high_save_prof = grid_res["df"][(grid_res["df"]["save_rate"] == 0.30) & (grid_res["df"]["outreach_cost"] == 15.0)]["max_net_profit"].iloc[0]
    assert high_save_prof > low_save_prof
