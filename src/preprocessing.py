"""
RetainIQ - Preprocessing & Pipeline Validation Module
Provides data cleaning audit funnels, RFM computation algorithms,
cadence and velocity feature engineering, and feature validation.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

# Exact ground-truth counts from exploratory and cleaning notebook
CLEANING_AUDIT_COUNTS = {
    "raw_records": 541909,
    "exact_duplicates": 5268,
    "deduplicated_records": 536641,
    "valid_purchase_records": 524878,
    "customer_identified_records": 392692,
    "missing_customer_id_valid": 132186,
}

BASE_FEATURES = ["Recency", "Frequency", "Monetary", "UniqueProducts", "Tenure", "AvgOrderValue"]
VELOCITY_FEATURES = [
    "avg_inter_purchase_days",
    "recency_to_ipi_ratio",
    "spend_velocity_ratio",
    "frequency_decay",
    "return_cancellation_rate"
]
FEATURE_NAMES_V2 = BASE_FEATURES + VELOCITY_FEATURES


def get_data_cleaning_audit() -> Dict[str, Any]:
    """Returns the data cleaning pipeline audit figures."""
    return CLEANING_AUDIT_COUNTS


def compute_rfm_from_transactions(
    df: pd.DataFrame,
    reference_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Computes standard RFM metrics from transactional data relative to a reference date.
    Recency: Days since most recent transaction
    Frequency: Number of unique invoices
    Monetary: Total customer revenue
    """
    df = df[df["CustomerID"].notnull()].copy()
    if not np.issubdtype(df["InvoiceDate"].dtype, np.datetime64):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Filter to transactions on or before reference date
    df_window = df[df["InvoiceDate"] <= reference_date]

    rfm = df_window.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda d: (reference_date - d.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        Tenure=("InvoiceDate", lambda d: (reference_date - d.min()).days),
    ).reset_index()

    rfm["AvgOrderValue"] = (rfm["Monetary"] / rfm["Frequency"]).round(2)
    return rfm


def compute_cadence_velocity_features(
    raw_df: pd.DataFrame,
    reference_date: pd.Timestamp = pd.Timestamp("2011-09-10 12:50:00")
) -> pd.DataFrame:
    """
    Computes leakage-free behavioral dynamics strictly on transactions <= reference_date:
    1. avg_inter_purchase_days: Mean days between consecutive invoices per customer (fallback to tenure).
    2. recency_to_ipi_ratio: Recency / (avg_inter_purchase_days + 1) measuring overdue cycles.
    3. spend_velocity_ratio: Spend in final 30 days pre-cutoff divided by average 30-day spend in prior 90 days.
    4. frequency_decay: Invoice count in days [-30, 0] minus invoice count in days [-60, -31].
    5. return_cancellation_rate: Ratio of cancelled invoices (starting with 'C') to total invoices pre-cutoff.
    """
    # Strict customer identification and historical cutoff enforcement
    df = raw_df[raw_df["CustomerID"].notnull()].copy()
    if not np.issubdtype(df["InvoiceDate"].dtype, np.datetime64):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)

    # Filter strictly to transactions <= reference_date (zero post-cutoff leakage)
    df_pre = df[df["InvoiceDate"] <= reference_date].copy()
    df_pre["CustomerID"] = df_pre["CustomerID"].astype(int)

    # 5. Return / cancellation rate (computed before filtering out cancellations)
    df_pre["IsCancelled"] = df_pre["InvoiceNo"].str.startswith("C")
    inv_canc = df_pre.groupby("CustomerID").agg(
        total_invoices=("InvoiceNo", "nunique"),
        cancelled_invoices=("InvoiceNo", lambda s: s[s.str.startswith("C")].nunique())
    ).reset_index()
    inv_canc["return_cancellation_rate"] = (
        inv_canc["cancelled_invoices"] / inv_canc["total_invoices"]
    ).fillna(0.0).round(4)

    # Valid purchases for RFM, cadence, and spend features
    valid_pre = df_pre[
        (df_pre["Quantity"] > 0) &
        (df_pre["UnitPrice"] > 0) &
        (~df_pre["IsCancelled"])
    ].copy()
    valid_pre["Revenue"] = valid_pre["Quantity"] * valid_pre["UnitPrice"]

    # Base RFM
    features = valid_pre.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda d: int((reference_date - d.max()).days)),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        Tenure=("InvoiceDate", lambda d: int((reference_date - d.min()).days))
    ).reset_index()
    features["AvgOrderValue"] = (features["Monetary"] / features["Frequency"]).round(2)
    features["Monetary"] = features["Monetary"].round(2)

    # 1. avg_inter_purchase_days
    inv_times = valid_pre.groupby(["CustomerID", "InvoiceNo"])["InvoiceDate"].min().reset_index()
    inv_times = inv_times.sort_values(["CustomerID", "InvoiceDate"])
    diffs = inv_times.groupby("CustomerID")["InvoiceDate"].diff().dt.total_seconds() / 86400.0
    avg_ipi = diffs.groupby(inv_times["CustomerID"]).mean()
    features["avg_inter_purchase_days"] = features["CustomerID"].map(avg_ipi)
    features["avg_inter_purchase_days"] = features["avg_inter_purchase_days"].fillna(features["Tenure"]).round(2)

    # 2. recency_to_ipi_ratio
    features["recency_to_ipi_ratio"] = (
        features["Recency"] / (features["avg_inter_purchase_days"] + 1.0)
    ).round(4)

    # 3. spend_velocity_ratio
    t_30 = reference_date - pd.Timedelta(days=30)
    t_90 = reference_date - pd.Timedelta(days=90)
    s30 = valid_pre[valid_pre["InvoiceDate"] > t_30].groupby("CustomerID")["Revenue"].sum()
    s90 = valid_pre[valid_pre["InvoiceDate"] > t_90].groupby("CustomerID")["Revenue"].sum()
    spend_30d = features["CustomerID"].map(s30).fillna(0.0)
    spend_90d = features["CustomerID"].map(s90).fillna(0.0)
    avg_30d = spend_90d / 3.0
    features["spend_velocity_ratio"] = np.where(
        avg_30d > 0,
        (spend_30d / avg_30d).round(4),
        0.0
    )

    # 4. frequency_decay
    t_60 = reference_date - pd.Timedelta(days=60)
    inv30 = valid_pre[valid_pre["InvoiceDate"] > t_30].groupby("CustomerID")["InvoiceNo"].nunique()
    inv60_31 = valid_pre[
        (valid_pre["InvoiceDate"] > t_60) & (valid_pre["InvoiceDate"] <= t_30)
    ].groupby("CustomerID")["InvoiceNo"].nunique()
    f30 = features["CustomerID"].map(inv30).fillna(0).astype(int)
    f60_31 = features["CustomerID"].map(inv60_31).fillna(0).astype(int)
    features["frequency_decay"] = f30 - f60_31

    # 5. return_cancellation_rate merge
    canc_map = inv_canc.set_index("CustomerID")["return_cancellation_rate"].to_dict()
    features["return_cancellation_rate"] = features["CustomerID"].map(canc_map).fillna(0.0).round(4)

    return features


def validate_customer_features(
    df: pd.DataFrame,
    required_features: Optional[List[str]] = None
) -> Tuple[bool, list]:
    """
    Validates whether an input dataframe contains all required features
    with non-null, valid numeric types for churn inference.
    """
    if required_features is None:
        required_features = BASE_FEATURES

    errors = []
    for col in required_features:
        if col not in df.columns:
            errors.append(f"Missing required feature column: {col}")
        elif df[col].isnull().any():
            errors.append(f"Column '{col}' contains {df[col].isnull().sum()} null values")
        elif not np.issubdtype(df[col].dtype, np.number):
            errors.append(f"Column '{col}' must be numeric (found {df[col].dtype})")

    return len(errors) == 0, errors
