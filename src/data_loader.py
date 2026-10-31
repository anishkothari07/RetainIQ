"""
RetainIQ - Data Loader Module
Handles efficient, cached ingestion and preparation of all processed datasets
with strict validation, memory optimization, and customer-country mapping.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

# Base directory for processed data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")

@st.cache_data(show_spinner=False)
def load_customer_transactions() -> pd.DataFrame:
    """
    Loads customer-identified transactions (392,692 rows).
    Contains: InvoiceNo, StockCode, Description, Quantity, InvoiceDate,
              UnitPrice, CustomerID, Country, Revenue, Month.
    """
    filepath = os.path.join(DATA_DIR, "customer_transactions.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Customer transactions file not found: {filepath}")

    df = pd.read_csv(
        filepath,
        dtype={
            "InvoiceNo": str,
            "StockCode": str,
            "Description": str,
            "Quantity": "int32",
            "UnitPrice": "float32",
            "CustomerID": "int64",
            "Country": str,
            "Revenue": "float32",
            "Month": str,
        },
        parse_dates=["InvoiceDate"]
    )
    return df


@st.cache_data(show_spinner=False)
def load_clean_transactions() -> pd.DataFrame:
    """
    Loads deduplicated valid purchase transactions (524,878 rows),
    including those with missing CustomerID (132,186 rows).
    Used primarily for Data Quality and Funnel analysis.
    """
    filepath = os.path.join(DATA_DIR, "clean_transactions.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Clean transactions file not found: {filepath}")

    df = pd.read_csv(
        filepath,
        dtype={
            "InvoiceNo": str,
            "StockCode": str,
            "Description": str,
            "Quantity": "int32",
            "UnitPrice": "float32",
            "CustomerID": "float64",
            "Country": str,
        },
        parse_dates=["InvoiceDate"],
        low_memory=False
    )
    return df


@st.cache_data(show_spinner=False)
def load_customer_country_map() -> Dict[int, str]:
    """
    Computes a deterministic primary country mapping for every CustomerID
    based on their most frequent transaction country.
    """
    tx = load_customer_transactions()
    country_counts = tx.groupby(["CustomerID", "Country"]).size().reset_index(name="count")
    idx_max = country_counts.groupby("CustomerID")["count"].idxmax()
    primary_countries = country_counts.loc[idx_max].set_index("CustomerID")["Country"].to_dict()
    return primary_countries


@st.cache_data(show_spinner=False)
def load_customer_dashboard() -> pd.DataFrame:
    """
    Loads the historical churn cohort (3,370 customers evaluated at cutoff 2011-09-10).
    Enriched with primary country mapping.
    Columns: CustomerID, Recency, Frequency, Monetary, UniqueProducts, Tenure,
             AvgOrderValue, ChurnProbability, ActualChurn, PredictedChurn,
             RiskCategory, R_Score, F_Score, M_Score, Segment, RetentionAction, Country.
    """
    filepath = os.path.join(DATA_DIR, "retainiq_customer_dashboard.csv")
    if not os.path.exists(filepath):
        # Fallback to retainiq_app_customers if needed
        filepath = os.path.join(DATA_DIR, "retainiq_app_customers.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Customer dashboard file not found: {filepath}")

    df = pd.read_csv(filepath)
    country_map = load_customer_country_map()
    df["Country"] = df["CustomerID"].map(country_map).fillna("United Kingdom")
    return df


@st.cache_data(show_spinner=False)
def load_rfm_segments() -> pd.DataFrame:
    """
    Loads the full-period customer RFM segmentation (4,338 customers).
    Enriched with primary country mapping.
    Columns: CustomerID, Recency, Frequency, Monetary, R_Score, F_Score,
             M_Score, RFM_Score, Segment, Country.
    """
    filepath = os.path.join(DATA_DIR, "rfm_segments.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"RFM segments file not found: {filepath}")

    df = pd.read_csv(filepath)
    country_map = load_customer_country_map()
    df["Country"] = df["CustomerID"].map(country_map).fillna("United Kingdom")
    return df


@st.cache_data(show_spinner=False)
def load_segment_revenue() -> pd.DataFrame:
    """Loads precomputed aggregate revenue by customer segment."""
    filepath = os.path.join(DATA_DIR, "segment_revenue.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Segment revenue file not found: {filepath}")
    return pd.read_csv(filepath)


@st.cache_data(show_spinner=False)
def load_churn_dataset() -> pd.DataFrame:
    """Loads raw churn evaluation cohort (3,370 records with binary Churn label)."""
    filepath = os.path.join(DATA_DIR, "churn_dataset.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Churn dataset file not found: {filepath}")
    return pd.read_csv(filepath)


@st.cache_data(show_spinner=False)
def load_holdout_predictions() -> pd.DataFrame:
    """
    Loads the 20% holdout test set predictions (674 customers).
    Contains: CustomerID, Recency, Frequency, Monetary, UniqueProducts, Tenure,
              AvgOrderValue, ChurnProbability, PredictedChurn, ActualChurn,
              RiskCategory, RetentionAction.
    """
    filepath = os.path.join(DATA_DIR, "customer_risk_predictions.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Holdout predictions file not found: {filepath}")
    return pd.read_csv(filepath)
