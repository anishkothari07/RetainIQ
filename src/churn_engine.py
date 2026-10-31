"""
RetainIQ - Churn Prediction Engine
Loads the saved scikit-learn model artifact, provides robust inference,
risk classification, and feature contribution inspection.
"""

import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "retainiq_churn_model.joblib"
)

# Standard feature order expected by the trained pipeline
FEATURE_NAMES = ["Recency", "Frequency", "Monetary", "UniqueProducts", "Tenure", "AvgOrderValue"]

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

    # Compatibility patch: scikit-learn 1.7.x expects multi_class on LogisticRegression
    if hasattr(pipeline, "steps"):
        for name, step in pipeline.steps:
            if hasattr(step, "get_params") and not hasattr(step, "multi_class"):
                step.multi_class = "auto"

    # Extract logistic regression coefficients and intercept
    coef_dict = {}
    intercept = 0.0
    try:
        classifier = pipeline.named_steps.get("classifier", pipeline.steps[-1][1])
        if hasattr(classifier, "coef_"):
            for feat, coef in zip(artifact.get("features", FEATURE_NAMES), classifier.coef_[0]):
                coef_dict[feat] = float(coef)
            intercept = float(classifier.intercept_[0])
    except Exception:
        pass

    return {
        "pipeline": pipeline,
        "features": artifact.get("features", FEATURE_NAMES),
        "cutoff_date": artifact.get("cutoff_date", "2011-09-10 12:50:00"),
        "churn_definition": artifact.get("churn_definition", "No purchase in the 90 days following cutoff"),
        "model_name": artifact.get("model_name", "Logistic Regression"),
        "cv_roc_auc": float(artifact.get("cross_validation_roc_auc", 0.7470)),
        "cv_std": float(artifact.get("cross_validation_std", 0.0110)),
        "holdout_roc_auc": float(artifact.get("holdout_roc_auc", 0.7238)),
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


def predict_churn(
    recency: float,
    frequency: float,
    monetary: float,
    unique_products: float,
    tenure: float,
    avg_order_value: float
) -> Dict[str, Any]:
    """
    Executes live model inference for a single customer record through
    the StandardScaler + LogisticRegression pipeline.
    """
    artifact = load_churn_model_artifact()
    pipe = artifact["pipeline"]
    feature_order = artifact["features"]

    input_df = pd.DataFrame([{
        "Recency": float(recency),
        "Frequency": float(frequency),
        "Monetary": float(monetary),
        "UniqueProducts": float(unique_products),
        "Tenure": float(tenure),
        "AvgOrderValue": float(avg_order_value),
    }])[feature_order]

    # Predict probability of Churn (class 1)
    proba = pipe.predict_proba(input_df)[0][1]
    pred_label = int(pipe.predict(input_df)[0])
    risk_cat = classify_risk(proba)
    action = get_recommended_action(risk_cat)

    return {
        "churn_probability": float(proba),
        "predicted_churn": pred_label,
        "risk_category": risk_cat,
        "retention_action": action,
        "features": input_df.to_dict(orient="records")[0]
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes churn predictions for a batch DataFrame containing the 6 required features.
    """
    artifact = load_churn_model_artifact()
    pipe = artifact["pipeline"]
    feature_order = artifact["features"]

    X = df[feature_order].copy()
    probas = pipe.predict_proba(X)[:, 1]
    preds = pipe.predict(X)

    res = df.copy()
    res["ChurnProbability"] = probas
    res["PredictedChurn"] = preds
    res["RiskCategory"] = res["ChurnProbability"].apply(classify_risk)
    res["RetentionAction"] = res["RiskCategory"].apply(get_recommended_action)
    return res
