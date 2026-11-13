"""
RetainIQ - Churn Prediction Engine
Loads the saved scikit-learn model artifact, provides robust calibrated inference,
risk classification, and local feature explainability (SHAP & feature log-odds).
"""

import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import shap
from typing import Dict, Any, Tuple, List, Optional
from src.preprocessing import BASE_FEATURES, VELOCITY_FEATURES, FEATURE_NAMES_V2

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "retainiq_churn_model.joblib"
)

# Standard feature order expected by the trained pipeline
FEATURE_NAMES = FEATURE_NAMES_V2

# Risk Category Thresholds (empirically aligned with the historical evaluation cohort)
RISK_THRESHOLD_LOW = 0.40
RISK_THRESHOLD_HIGH = 0.70


@st.cache_resource(show_spinner=False)
def load_churn_model_artifact() -> Dict[str, Any]:
    """
    Loads and caches the trained Joblib model artifact.
    Includes backward-compatibility handling for scikit-learn estimator attributes.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact.get("model")
    base_estimator = artifact.get("base_estimator")
    logistic_pipe = artifact.get("logistic_pipeline")

    # Compatibility patch: scikit-learn 1.7.x expects multi_class on LogisticRegression
    if hasattr(pipeline, "steps"):
        for name, step in pipeline.steps:
            if hasattr(step, "get_params") and not hasattr(step, "multi_class"):
                step.multi_class = "auto"

    if logistic_pipe is not None and hasattr(logistic_pipe, "steps"):
        for name, step in logistic_pipe.steps:
            if hasattr(step, "get_params") and not hasattr(step, "multi_class"):
                step.multi_class = "auto"

    # Extract logistic regression coefficients and intercept if available
    coef_dict = {}
    intercept = 0.0
    try:
        source_pipe = logistic_pipe if logistic_pipe is not None else pipeline
        if hasattr(source_pipe, "named_steps"):
            classifier = source_pipe.named_steps.get("classifier", source_pipe.steps[-1][1])
            if hasattr(classifier, "coef_"):
                features_list = artifact.get("features", FEATURE_NAMES)
                for feat, coef in zip(features_list, classifier.coef_[0]):
                    coef_dict[feat] = float(coef)
                intercept = float(classifier.intercept_[0])
    except Exception:
        pass

    return {
        "pipeline": pipeline,
        "model": pipeline,
        "base_estimator": base_estimator,
        "logistic_pipeline": logistic_pipe,
        "features": artifact.get("features", FEATURE_NAMES),
        "cutoff_date": artifact.get("cutoff_date", "2011-09-10 12:50:00"),
        "churn_definition": artifact.get("churn_definition", "No purchase in the 90 days following cutoff"),
        "model_name": artifact.get("model_name", "Calibrated HistGradientBoosting"),
        "cv_roc_auc": float(artifact.get("cross_validation_roc_auc", 0.7502)),
        "cv_std": float(artifact.get("cross_validation_std", 0.0101)),
        "cv_pr_auc": float(artifact.get("cross_validation_pr_auc", 0.6465)),
        "cv_brier": float(artifact.get("cross_validation_brier", 0.1992)),
        "holdout_roc_auc": float(artifact.get("holdout_roc_auc", 0.7179)),
        "holdout_pr_auc": float(artifact.get("holdout_pr_auc", 0.6052)),
        "holdout_brier": float(artifact.get("holdout_brier", 0.2112)),
        "precision_at_top15": float(artifact.get("precision_at_top15", 0.6863)),
        "coefficients": coef_dict,
        "intercept": intercept,
    }


def classify_risk(probability: float) -> str:
    """Classifies churn probability into Low, Medium, or High risk."""
    if probability < RISK_THRESHOLD_LOW:
        return "Low"
    elif probability < RISK_THRESHOLD_HIGH:
        return "Medium"
    else:
        return "High"


def get_recommended_action(risk_category: str) -> str:
    """Returns primary retention intervention action for a given risk category."""
    if risk_category == "High":
        return "Priority personal retention outreach"
    elif risk_category == "Medium":
        return "Targeted re-engagement campaign"
    else:
        return "Loyalty and repeat-purchase campaign"


def format_risk_driver_text(feature: str, val: float, contrib: float) -> str:
    """Converts a feature value and contribution into human-readable business context."""
    if feature == "recency_to_ipi_ratio":
        return f"Overdue by {val:.1f}x personal purchase cadence"
    elif feature == "spend_velocity_ratio":
        if val < 0.5:
            return f"Spend velocity collapsed to {val:.2f}x of 90-day baseline"
        return f"Recent spend velocity at {val:.2f}x of baseline"
    elif feature == "frequency_decay":
        if val < 0:
            return f"Purchase cadence fell by {abs(int(val))} orders vs prior month"
        return f"Order velocity change: {int(val):+d} orders"
    elif feature == "return_cancellation_rate":
        return f"Elevated cancellation rate ({val*100:.1f}%) on prior orders"
    elif feature == "Recency":
        return f"Extended inactivity ({int(val)} days since last order)"
    elif feature == "Frequency":
        return f"Low order volume ({int(val)} historical invoices)"
    elif feature == "Monetary":
        return f"Historical cumulative spend: £{val:,.2f}"
    elif feature == "UniqueProducts":
        return f"Narrow SKU diversity ({int(val)} unique products)"
    elif feature == "avg_inter_purchase_days":
        return f"Slow typical buying cycle ({val:.1f} days between orders)"
    elif feature == "AvgOrderValue":
        return f"Lumpy order sizing (AOV £{val:,.2f})"
    elif feature == "Tenure":
        return f"Account tenure: {int(val)} days"
    return f"{feature}: {val:.2f}"


def compute_local_feature_contributions(input_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Computes local feature contributions for an input record using SHAP TreeExplainer
    on the base estimator (or fallback to standardized linear coefficients).
    Returns (contributions_list, top_3_risk_drivers_text).
    """
    artifact = load_churn_model_artifact()
    base_estimator = artifact.get("base_estimator")
    feature_order = artifact["features"]

    X = input_df[feature_order].values

    contributions = []
    shap_vals = None

    if base_estimator is not None:
        try:
            explainer = shap.TreeExplainer(base_estimator)
            sv = explainer.shap_values(X)
            shap_vals = sv[0] if len(sv.shape) == 2 else sv
        except Exception:
            shap_vals = None

    if shap_vals is not None:
        for feat, val, s in zip(feature_order, X[0], shap_vals):
            contributions.append({
                "feature": feat,
                "value": float(val),
                "contribution": float(s),
                "direction": "Elevates Churn Risk" if s > 0 else "Protects Retention",
                "magnitude": abs(float(s))
            })
    else:
        # Fallback to linear coefficients if SHAP is unavailable
        coefs = artifact.get("coefficients", {})
        for feat in feature_order:
            val = float(input_df[feat].iloc[0])
            coef = coefs.get(feat, 0.0)
            contrib = val * coef
            contributions.append({
                "feature": feat,
                "value": val,
                "contribution": contrib,
                "direction": "Elevates Churn Risk" if contrib > 0 else "Protects Retention",
                "magnitude": abs(contrib)
            })

    # Sort descending by risk elevation
    risk_increasing = [c for c in contributions if c["contribution"] > 0]
    risk_increasing.sort(key=lambda x: x["contribution"], reverse=True)

    top_drivers = []
    for item in risk_increasing[:3]:
        driver_text = format_risk_driver_text(item["feature"], item["value"], item["contribution"])
        top_drivers.append(driver_text)

    # If fewer than 3 risk-elevating features, fill with highest magnitude features
    if len(top_drivers) < 3:
        all_sorted = sorted(contributions, key=lambda x: x["magnitude"], reverse=True)
        for item in all_sorted:
            txt = format_risk_driver_text(item["feature"], item["value"], item["contribution"])
            if txt not in top_drivers:
                top_drivers.append(txt)
            if len(top_drivers) >= 3:
                break

    return contributions, top_drivers[:3]


def predict_churn(
    recency: float,
    frequency: float,
    monetary: float,
    unique_products: float,
    tenure: float,
    avg_order_value: float,
    avg_inter_purchase_days: Optional[float] = None,
    recency_to_ipi_ratio: Optional[float] = None,
    spend_velocity_ratio: Optional[float] = None,
    frequency_decay: Optional[float] = None,
    return_cancellation_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Executes live calibrated model inference with local feature explainability.
    Supports base 6 features with intelligent fallbacks for velocity metrics.
    """
    artifact = load_churn_model_artifact()
    pipe = artifact["pipeline"]
    feature_order = artifact["features"]

    # Intelligent default fallbacks for velocity features when omitted
    if avg_inter_purchase_days is None:
        avg_inter_purchase_days = float(tenure) / max(float(frequency) - 1.0, 1.0)
    if recency_to_ipi_ratio is None:
        recency_to_ipi_ratio = float(recency) / (float(avg_inter_purchase_days) + 1.0)
    if spend_velocity_ratio is None:
        spend_velocity_ratio = 1.0  # Assumes stable spend
    if frequency_decay is None:
        frequency_decay = 0.0  # Neutral decay
    if return_cancellation_rate is None:
        return_cancellation_rate = 0.0  # Clean history

    record = {
        "Recency": float(recency),
        "Frequency": float(frequency),
        "Monetary": float(monetary),
        "UniqueProducts": float(unique_products),
        "Tenure": float(tenure),
        "AvgOrderValue": float(avg_order_value),
        "avg_inter_purchase_days": float(avg_inter_purchase_days),
        "recency_to_ipi_ratio": float(recency_to_ipi_ratio),
        "spend_velocity_ratio": float(spend_velocity_ratio),
        "frequency_decay": float(frequency_decay),
        "return_cancellation_rate": float(return_cancellation_rate),
    }

    input_df = pd.DataFrame([record])[feature_order]

    # Predict calibrated probability of Churn
    X_val = input_df[feature_order].values
    proba = float(pipe.predict_proba(X_val)[0][1])
    pred_label = int(pipe.predict(X_val)[0]) if hasattr(pipe, "predict") else int(proba >= 0.5)
    risk_cat = classify_risk(proba)
    action = get_recommended_action(risk_cat)

    contributions, top_drivers = compute_local_feature_contributions(input_df)

    return {
        "churn_probability": proba,
        "predicted_churn": pred_label,
        "risk_category": risk_cat,
        "retention_action": action,
        "features": record,
        "feature_contributions": contributions,
        "top_risk_drivers": top_drivers,
    }


def get_top_risk_drivers_for_customer(customer_id: int) -> List[str]:
    """Fetches cohort data for a CustomerID and returns their top 3 plain-English risk drivers."""
    from src.data_loader import load_customer_dashboard
    cohort = load_customer_dashboard()
    cust_rows = cohort[cohort["CustomerID"] == customer_id]
    if len(cust_rows) == 0:
        return ["Customer not found in historical cutoff cohort."]

    row = cust_rows.iloc[0]
    artifact = load_churn_model_artifact()
    feature_order = artifact["features"]

    record = {}
    for f in feature_order:
        if f in row:
            record[f] = float(row[f])
        else:
            record[f] = 0.0

    input_df = pd.DataFrame([record])[feature_order]
    _, top_drivers = compute_local_feature_contributions(input_df)
    return top_drivers


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Computes churn predictions for a batch DataFrame."""
    artifact = load_churn_model_artifact()
    pipe = artifact["pipeline"]
    feature_order = artifact["features"]

    X = df[feature_order].values
    probas = pipe.predict_proba(X)[:, 1]
    preds = (probas >= 0.5).astype(int)

    res = df.copy()
    res["ChurnProbability"] = probas
    res["PredictedChurn"] = preds
    res["RiskCategory"] = res["ChurnProbability"].apply(classify_risk)
    res["RetentionAction"] = res["RiskCategory"].apply(get_recommended_action)
    return res
