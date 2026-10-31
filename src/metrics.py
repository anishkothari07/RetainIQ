"""
RetainIQ - Model Performance & Validation Metrics Module
Provides historical benchmark metrics, dynamic evaluation of the holdout test set,
confusion matrix plotting, ROC curves, and feature importance analyses.
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
    classification_report
)
from typing import Dict, Any, Tuple
from src.data_loader import load_holdout_predictions
from src.utils import get_plotly_theme, PRIMARY, TEXT_PRIMARY, TEXT_SECONDARY, RISK_HIGH, RISK_LOW

# Historical Model Evaluation Benchmarks (Notebook Ground Truth)
BENCHMARKS = {
    "Logistic Regression": {
        "model_type": "Logistic Regression (L2 Balanced)",
        "cv_roc_auc_mean": 0.7470,
        "cv_roc_auc_std": 0.0110,
        "holdout_accuracy": 0.6602,
        "holdout_roc_auc": 0.7238,
        "precision": 0.58,
        "recall": 0.77,
        "f1": 0.66,
        "status": "Production Candidate (Selected)"
    },
    "Random Forest": {
        "model_type": "Random Forest (Baseline)",
        "cv_roc_auc_mean": 0.7280,
        "cv_roc_auc_std": 0.0140,
        "holdout_accuracy": 0.6350,
        "holdout_roc_auc": 0.7044,
        "precision": 0.55,
        "recall": 0.69,
        "f1": 0.61,
        "status": "Baseline Comparison"
    }
}

PERMUTATION_IMPORTANCE = {
    "Recency": 0.0392,
    "UniqueProducts": 0.0272,
    "Monetary": 0.0210,
    "Frequency": 0.0123,
    "AvgOrderValue": 0.0039,
    "Tenure": -0.0014
}


def get_holdout_evaluation_metrics() -> Dict[str, Any]:
    """
    Computes exact evaluation metrics directly from the holdout dataset (674 samples).
    """
    df = load_holdout_predictions()
    y_true = df["ActualChurn"]
    y_pred = df["PredictedChurn"]
    y_prob = df["ChurnProbability"]

    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob))
    prec = float(precision_score(y_true, y_pred))
    rec = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    return {
        "sample_size": len(df),
        "accuracy": acc,
        "roc_auc": auc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
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

    # Chance line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Chance (AUC = 0.50)",
        line=dict(color="#94A3B8", dash="dash", width=1.5)
    ))

    # Model ROC
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"Logistic Regression (AUC = {auc_val:.4f})",
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


def plot_permutation_importance() -> go.Figure:
    """
    Renders a bar chart of permutation feature importance.
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
            title=dict(text="<b>Permutation Feature Importance (Mean Drop in ROC-AUC)</b>", font=dict(size=14)),
            xaxis=dict(title="Importance Score (Mean Score Loss)"),
            yaxis=dict(title=""),
            height=320,
        )
    )
    return fig


def plot_model_comparison() -> go.Figure:
    """
    Renders side-by-side bar chart comparing Logistic Regression vs Random Forest.
    """
    metrics = ["Holdout ROC-AUC", "Holdout Accuracy", "Recall", "Precision", "F1-Score"]
    lr_vals = [0.7238, 0.6602, 0.77, 0.58, 0.66]
    rf_vals = [0.7044, 0.6350, 0.69, 0.55, 0.61]

    fig = go.Figure(data=[
        go.Bar(name="Logistic Regression", x=metrics, y=lr_vals, marker_color=PRIMARY),
        go.Bar(name="Random Forest", x=metrics, y=rf_vals, marker_color="#94A3B8")
    ])

    fig.update_layout(
        **get_plotly_theme(
            barmode="group",
            title=dict(text="<b>Model Performance Benchmark: Logistic Regression vs Random Forest</b>", font=dict(size=14)),
            yaxis=dict(title="Score", range=[0, 1.0]),
            height=320,
        )
    )
    return fig
