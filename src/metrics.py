"""
RetainIQ - Model Performance & Validation Metrics Module
Provides historical benchmark metrics, dynamic evaluation of the holdout test set,
confusion matrix plotting, ROC curves, PR curves, calibration curves, and feature importance analyses.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    classification_report
)
from sklearn.calibration import calibration_curve
from typing import Dict, Any, Tuple, Optional
from src.data_loader import load_holdout_predictions
from src.utils import get_plotly_theme, PRIMARY, TEXT_PRIMARY, TEXT_SECONDARY, RISK_HIGH, RISK_LOW, RISK_MEDIUM

# 3-Model Benchmark Metrics (5-Fold Stratified CV on N=2,696; Holdout on N=674)
BENCHMARKS = {
    "Calibrated HistGBM": {
        "model_type": "HistGradientBoosting + CalibratedClassifierCV (Sigmoid)",
        "cv_roc_auc_mean": 0.7502,
        "cv_roc_auc_std": 0.0101,
        "cv_pr_auc": 0.6465,
        "cv_brier": 0.1992,
        "holdout_accuracy": 0.6691,
        "holdout_roc_auc": 0.7179,
        "holdout_pr_auc": 0.6052,
        "holdout_brier": 0.2112,
        "precision": 0.64,
        "recall": 0.62,
        "f1": 0.63,
        "precision_at_top15": 0.6863,
        "status": "Winning Model (Selected & Calibrated)"
    },
    "Logistic Regression": {
        "model_type": "StandardScaler + Logistic Regression (L2 Balanced)",
        "cv_roc_auc_mean": 0.7483,
        "cv_roc_auc_std": 0.0093,
        "cv_pr_auc": 0.6399,
        "cv_brier": 0.2031,
        "holdout_accuracy": 0.6602,
        "holdout_roc_auc": 0.7207,
        "holdout_pr_auc": 0.6027,
        "holdout_brier": 0.2137,
        "precision": 0.58,
        "recall": 0.77,
        "f1": 0.66,
        "precision_at_top15": 0.6471,
        "status": "Linear Baseline Candidate"
    },
    "Random Forest": {
        "model_type": "Random Forest (Baseline)",
        "cv_roc_auc_mean": 0.7518,
        "cv_roc_auc_std": 0.0094,
        "cv_pr_auc": 0.6458,
        "cv_brier": 0.2006,
        "holdout_accuracy": 0.6721,
        "holdout_roc_auc": 0.7220,
        "holdout_pr_auc": 0.6127,
        "holdout_brier": 0.2133,
        "precision": 0.62,
        "recall": 0.75,
        "f1": 0.68,
        "precision_at_top15": 0.6961,
        "status": "Ensemble Comparison"
    }
}

PERMUTATION_IMPORTANCE = {
    "recency_to_ipi_ratio": 0.0421,
    "Recency": 0.0385,
    "Frequency": 0.0265,
    "spend_velocity_ratio": 0.0215,
    "UniqueProducts": 0.0195,
    "frequency_decay": 0.0162,
    "return_cancellation_rate": 0.0145,
    "Monetary": 0.0120,
    "avg_inter_purchase_days": 0.0085,
    "AvgOrderValue": 0.0039,
    "Tenure": -0.0012
}


def get_precision_at_top_k(y_true: np.ndarray, y_prob: np.ndarray, k_pct: float = 0.15) -> float:
    """Computes precision on the top k percent highest predicted risk customers."""
    k = max(1, int(np.ceil(len(y_true) * k_pct)))
    top_indices = np.argsort(y_prob)[::-1][:k]
    return float(np.mean(y_true[top_indices]))


def get_holdout_evaluation_metrics() -> Dict[str, Any]:
    """
    Computes exact evaluation metrics directly from the holdout dataset (674 samples),
    including ROC-AUC, PR-AUC, Brier score, and Precision@Top15%.
    """
    df = load_holdout_predictions()
    y_true = df["ActualChurn"].values
    y_pred = df["PredictedChurn"].values
    y_prob = df["ChurnProbability"].values

    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    prec = float(precision_score(y_true, y_pred))
    rec = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    p15 = get_precision_at_top_k(y_true, y_prob, 0.15)
    cm = confusion_matrix(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy="uniform")

    return {
        "sample_size": len(df),
        "accuracy": acc,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "precision_at_top15": p15,
        "confusion_matrix": cm,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": precision_curve.tolist(), "recall": recall_curve.tolist()},
        "calibration_curve": {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()},
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def plot_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """
    Renders an interactive, branded Confusion Matrix heatmap.
    """
    labels = ["Retained (0)", "Churned (1)"]
    z_text = [[str(cm[0, 0]), str(cm[0, 1])], [str(cm[1, 0]), str(cm[1, 1])]]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=z_text,
        texttemplate="<b>%{text}</b>",
        textfont={"size": 18, "family": "Plus Jakarta Sans"},
        colorscale=[[0, "#EEF2FF"], [0.5, "#818CF8"], [1, "#4338CA"]],
        showscale=False,
    ))

    fig.update_layout(
        **get_plotly_theme(
            title=dict(text="<b>Holdout Confusion Matrix (N = 674)</b>", font=dict(size=14)),
            xaxis=dict(title="<b>Predicted Class</b>", side="bottom"),
            yaxis=dict(title="<b>Actual Class</b>", autorange="reversed"),
            height=320,
        )
    )
    return fig


def plot_roc_curve(fpr: list, tpr: list, auc_val: float) -> go.Figure:
    """
    Renders an interactive ROC Curve with the 45-degree chance baseline.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Chance (AUC = 0.50)",
        line=dict(color="#94A3B8", dash="dash", width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"ROC Curve (AUC = {auc_val:.4f})",
        line=dict(color=PRIMARY, width=3)
    ))

    fig.update_layout(
        **get_plotly_theme(
            title=dict(text="<b>Receiver Operating Characteristic (ROC) Curve</b>", font=dict(size=14)),
            xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[-0.02, 1.02]),
            yaxis=dict(title="True Positive Rate (Recall)", range=[-0.02, 1.02]),
            height=340,
            legend=dict(x=0.45, y=0.08)
        )
    )
    return fig


def plot_pr_curve(recall_vals: list, precision_vals: list, pr_auc: float, no_skill_rate: float = 0.430) -> go.Figure:
    """
    Renders an interactive Precision-Recall Curve with the no-skill baseline.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[no_skill_rate, no_skill_rate],
        mode="lines",
        name=f"No-Skill Baseline (Rate = {no_skill_rate:.2f})",
        line=dict(color="#94A3B8", dash="dash", width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=recall_vals,
        y=precision_vals,
        mode="lines",
        name=f"PR Curve (PR-AUC = {pr_auc:.4f})",
        line=dict(color="#0D9488", width=3)
    ))

    fig.update_layout(
        **get_plotly_theme(
            title=dict(text="<b>Precision-Recall (PR) Curve</b>", font=dict(size=14)),
            xaxis=dict(title="Recall (Sensitivity)", range=[-0.02, 1.02]),
            yaxis=dict(title="Precision (PPV)", range=[-0.02, 1.02]),
            height=340,
            legend=dict(x=0.45, y=0.88)
        )
    )
    return fig


def plot_calibration_curve_chart(prob_pred: list, prob_true: list, brier_score: float) -> go.Figure:
    """
    Renders an interactive Calibration Reliability Curve with the 45-degree perfect calibration line.
    """
    fig = go.Figure()

    # Perfectly calibrated 45-degree diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect Calibration",
        line=dict(color="#94A3B8", dash="dash", width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=prob_pred,
        y=prob_true,
        mode="lines+markers",
        name=f"Model Calibration (Brier = {brier_score:.4f})",
        line=dict(color="#4F46E5", width=2.5),
        marker=dict(size=8, color="#4338CA")
    ))

    fig.update_layout(
        **get_plotly_theme(
            title=dict(text="<b>Calibration Reliability Curve (Brier Loss Check)</b>", font=dict(size=14)),
            xaxis=dict(title="Mean Predicted Probability", range=[-0.02, 1.02]),
            yaxis=dict(title="Fraction of Positives (Actual Churn)", range=[-0.02, 1.02]),
            height=340,
            legend=dict(x=0.05, y=0.88)
        )
    )
    return fig


def plot_permutation_importance() -> go.Figure:
    """
    Renders a horizontal bar chart of permutation feature importance across expanded features.
    """
    df = pd.DataFrame(list(PERMUTATION_IMPORTANCE.items()), columns=["Feature", "Importance"])
    df = df.sort_values("Importance", ascending=True)

    colors = [RISK_HIGH if v < 0 else PRIMARY for v in df["Importance"]]

    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Feature"],
        orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=df["Importance"].apply(lambda x: f"{x:+.4f}"),
        textposition="outside",
        textfont=dict(size=11, family="Plus Jakarta Sans")
    ))

    fig.update_layout(
        **get_plotly_theme(
            title=dict(text="<b>Permutation Feature Importance (Mean Drop in Holdout ROC-AUC)</b>", font=dict(size=14)),
            xaxis=dict(title="Importance Score (Mean Score Loss)"),
            yaxis=dict(title=""),
            height=380,
        )
    )
    return fig


def plot_model_comparison() -> go.Figure:
    """
    Renders grouped bar chart comparing all 3 benchmarked models:
    Logistic Regression, Calibrated HistGBM, and Random Forest.
    """
    metrics = ["Holdout ROC-AUC", "Holdout PR-AUC", "Holdout Accuracy", "Recall", "Precision@Top15%", "Brier Score"]
    cal_gbm_vals = [0.7179, 0.6052, 0.6691, 0.6207, 0.6863, 0.2112]
    lr_vals = [0.7207, 0.6027, 0.6602, 0.7655, 0.6471, 0.2137]
    rf_vals = [0.7220, 0.6127, 0.6721, 0.7483, 0.6961, 0.2133]

    fig = go.Figure(data=[
        go.Bar(name="Calibrated HistGBM (Winning)", x=metrics, y=cal_gbm_vals, marker_color="#4F46E5"),
        go.Bar(name="Logistic Regression", x=metrics, y=lr_vals, marker_color="#0D9488"),
        go.Bar(name="Random Forest", x=metrics, y=rf_vals, marker_color="#94A3B8")
    ])

    fig.update_layout(
        **get_plotly_theme(
            barmode="group",
            title=dict(text="<b>3-Model Performance Benchmark Comparison</b>", font=dict(size=14)),
            yaxis=dict(title="Score", range=[0, 1.0]),
            height=340,
        )
    )
    return fig
