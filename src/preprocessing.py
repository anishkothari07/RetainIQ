"""
RetainIQ - Preprocessing & Pipeline Validation Module
Provides data cleaning audit funnels, RFM computation algorithms,
and feature engineering definitions aligned with the training notebook.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

# Exact ground-truth counts from exploratory and cleaning notebook
CLEANING_AUDIT_COUNTS = {
    "raw_records": 541909,
    "exact_duplicates": 5268,
    "deduplicated_records": 536641,
    "valid_purchase_records": 524878,
    "customer_identified_records": 392692,
    "missing_customer_id_valid": 132186,
}


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


def validate_customer_features(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validates whether an input dataframe contains all required features
    with non-null, valid numeric types for churn inference.
    """
    required = ["Recency", "Frequency", "Monetary", "UniqueProducts", "Tenure", "AvgOrderValue"]
    errors = []

    for col in required:
        if col not in df.columns:
            errors.append(f"Missing required feature column: {col}")
        elif df[col].isnull().any():
            errors.append(f"Column '{col}' contains {df[col].isnull().sum()} null values")
        elif not np.issubdtype(df[col].dtype, np.number):
            errors.append(f"Column '{col}' must be numeric (found {df[col].dtype})")

    return len(errors) == 0, errors
