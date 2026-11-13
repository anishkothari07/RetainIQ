"""
RetainIQ - Temporal Backtesting & Probabilistic Baseline Module
Implements leakage-free multi-cutoff cohort generation, out-of-time walk-forward
evaluation, probabilistic repeat-purchasing baselines (BG/NBD & Pareto/NBD via lifetimes),
and non-parametric percentile bootstrap confidence intervals.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Callable
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from lifetimes import BetaGeoFitter, ParetoNBDFitter
from lifetimes.utils import summary_data_from_transaction_data

from src.preprocessing import compute_cadence_velocity_features, FEATURE_NAMES_V2
from src.metrics import get_precision_at_top_k


def build_cohort_at_cutoff(
    raw_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    window_days: int = 90
) -> pd.DataFrame:
    """
    Constructs a leakage-free customer evaluation cohort at a historical cutoff date T.
    Features: 11 base RFM and velocity features computed strictly on transactions <= T.
    Label: ActualChurn = 1 if customer makes 0 qualifying purchases in (T, T + window_days], else 0.
    """
    if not isinstance(cutoff_date, pd.Timestamp):
        cutoff_date = pd.Timestamp(cutoff_date)

    # 1. Feature extraction strictly <= cutoff_date (zero future data leakage)
    cohort_df = compute_cadence_velocity_features(raw_df, reference_date=cutoff_date)

    # 2. Forward churn observation window strictly (cutoff_date, cutoff_date + window_days]
    window_end = cutoff_date + pd.Timedelta(days=window_days)
    df_raw = raw_df[raw_df["CustomerID"].notnull()].copy()
    if not np.issubdtype(df_raw["InvoiceDate"].dtype, np.datetime64):
        df_raw["InvoiceDate"] = pd.to_datetime(df_raw["InvoiceDate"])

    # Valid qualifying purchases in future window
    future_tx = df_raw[
        (df_raw["InvoiceDate"] > cutoff_date) &
        (df_raw["InvoiceDate"] <= window_end) &
        (df_raw["Quantity"] > 0) &
        (df_raw["UnitPrice"] > 0) &
        (~df_raw["InvoiceNo"].astype(str).str.startswith("C"))
    ]
    retained_customers = set(future_tx["CustomerID"].astype(int).unique())

    # Label: Inactive in observation window -> Churned (1), Active -> Retained (0)
    cohort_df["ActualChurn"] = (~cohort_df["CustomerID"].isin(retained_customers)).astype(int)
    return cohort_df


def extract_lifetimes_summary(
    raw_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    customer_ids: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Extracts standard RFM-T summary data (frequency, recency, T, monetary_value)
    strictly from transactions <= cutoff_date for lifetimes probabilistic fitters.
    """
    if not isinstance(cutoff_date, pd.Timestamp):
        cutoff_date = pd.Timestamp(cutoff_date)

    df = raw_df[raw_df["CustomerID"].notnull()].copy()
    if not np.issubdtype(df["InvoiceDate"].dtype, np.datetime64):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    pre = df[
        (df["InvoiceDate"] <= cutoff_date) &
        (df["Quantity"] > 0) &
        (df["UnitPrice"] > 0) &
        (~df["InvoiceNo"].astype(str).str.startswith("C"))
    ].copy()
    pre["CustomerID"] = pre["CustomerID"].astype(int)
    pre["Revenue"] = pre["Quantity"] * pre["UnitPrice"]

    summary = summary_data_from_transaction_data(
        pre,
        customer_id_col="CustomerID",
        datetime_col="InvoiceDate",
        monetary_value_col="Revenue",
        observation_period_end=cutoff_date,
        freq="D"
    )

    if customer_ids is not None:
        summary = summary.reindex(customer_ids).fillna(0.0)

    return summary


def predict_probabilistic_churn(
    fitter: Any,
    summary_df: pd.DataFrame,
    window_days: int = 90
) -> np.ndarray:
    """
    Converts lifetimes model expected purchase rate over window_days into
    P(churn in next window_days) = exp(-E[Y(window_days)]).
    Under the standard Poisson repeat-purchasing arrival process conditional on latency,
    the probability of zero transactions in window t is exp(-lambda * t) = exp(-E[Y(t)]).
    """
    exp_purch = fitter.predict(
        window_days,
        summary_df["frequency"],
        summary_df["recency"],
        summary_df["T"]
    )
    # Handle rare numerical indeterminacy for T ~ 0 in hypergeometric series
    if hasattr(exp_purch, "isna") and exp_purch.isna().any():
        if hasattr(fitter, "params_") and "r" in fitter.params_ and "alpha" in fitter.params_:
            fallback = (fitter.params_["r"] / (fitter.params_["alpha"] + summary_df["T"])) * window_days
            exp_purch = exp_purch.fillna(fallback)
        else:
            exp_purch = exp_purch.fillna(0.0)

    exp_purch = np.asarray(exp_purch)
    # Bound expected purchases to prevent numerical underflow
    churn_prob = np.exp(-np.clip(exp_purch, 0.0, 100.0))
    return np.clip(churn_prob, 0.0, 1.0)


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    random_state: int = 42
) -> Dict[str, Tuple[float, float, float]]:
    """
    Computes empirical non-parametric percentile bootstrap confidence intervals
    for ROC-AUC, PR-AUC, Brier Score, and Precision@Top15%.
    Returns dict mapping metric_name -> (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    alpha_low = ((1.0 - ci) / 2.0) * 100.0
    alpha_high = (1.0 - (1.0 - ci) / 2.0) * 100.0

    point_auc = float(roc_auc_score(y_true, y_prob))
    point_pr = float(average_precision_score(y_true, y_prob))
    point_brier = float(brier_score_loss(y_true, y_prob))
    point_p15 = float(get_precision_at_top_k(y_true, y_prob, 0.15))

    boot_auc, boot_pr, boot_brier, boot_p15 = [], [], [], []

    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        yb = y_true[idx]
        pb = y_prob[idx]

        # In extreme bootstrap samples where only one class is sampled, skip AUC
        if len(np.unique(yb)) > 1:
            boot_auc.append(float(roc_auc_score(yb, pb)))
        boot_pr.append(float(average_precision_score(yb, pb)))
        boot_brier.append(float(brier_score_loss(yb, pb)))
        boot_p15.append(float(get_precision_at_top_k(yb, pb, 0.15)))

    return {
        "roc_auc": (point_auc, float(np.percentile(boot_auc, alpha_low)), float(np.percentile(boot_auc, alpha_high))),
        "pr_auc": (point_pr, float(np.percentile(boot_pr, alpha_low)), float(np.percentile(boot_pr, alpha_high))),
        "brier_score": (point_brier, float(np.percentile(boot_brier, alpha_low)), float(np.percentile(boot_brier, alpha_high))),
        "precision_at_top15": (point_p15, float(np.percentile(boot_p15, alpha_low)), float(np.percentile(boot_p15, alpha_high))),
    }


def run_temporal_backtest(
    raw_df: pd.DataFrame,
    train_cutoff: str = "2011-05-10 12:50:00",
    test_cutoff: str = "2011-09-10 12:50:00",
    n_boot: int = 1000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes a strict out-of-time temporal backtest:
    1. Trains ML pipelines on train_cutoff (transactions <= train_cutoff).
    2. Fits probabilistic baselines (Pareto/NBD and BG/NBD) on feature window.
    3. Evaluates all models on untouched future cohort at test_cutoff.
    4. Computes empirical 95% bootstrap confidence intervals for all holdout metrics.
    """
    # 1. Build training cohort
    train_cohort = build_cohort_at_cutoff(raw_df, pd.Timestamp(train_cutoff))
    X_train = train_cohort[FEATURE_NAMES_V2].values
    y_train = train_cohort["ActualChurn"].values

    # 2. Build testing cohort
    test_cohort = build_cohort_at_cutoff(raw_df, pd.Timestamp(test_cutoff))
    X_test = test_cohort[FEATURE_NAMES_V2].values
    y_test = test_cohort["ActualChurn"].values

    # 3. Fit ML models on training cohort
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000))
        ]),
        "Calibrated HistGBM": CalibratedClassifierCV(
            estimator=HistGradientBoostingClassifier(
                max_depth=3,
                min_samples_leaf=30,
                l2_regularization=5.0,
                random_state=random_state
            ),
            method="sigmoid",
            cv=5
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state
        )
    }

    predictions = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        predictions[name] = m.predict_proba(X_test)[:, 1]

    # 4. Fit Probabilistic Repeat-Purchase Baselines (lifetimes)
    # Summary data strictly up to test_cutoff feature window
    sum_test = extract_lifetimes_summary(raw_df, pd.Timestamp(test_cutoff), test_cohort["CustomerID"])

    # Pareto/NBD
    pnbd = ParetoNBDFitter(penalizer_coef=0.0)
    pnbd.fit(sum_test["frequency"], sum_test["recency"], sum_test["T"])
    predictions["Pareto/NBD"] = predict_probabilistic_churn(pnbd, sum_test, window_days=90)

    # BetaGeo/NBD (BG/NBD)
    bgf = BetaGeoFitter(penalizer_coef=0.1)
    bgf.fit(sum_test["frequency"], sum_test["recency"], sum_test["T"])
    predictions["BetaGeo/NBD"] = predict_probabilistic_churn(bgf, sum_test, window_days=90)

    # 5. Compute metrics and 95% bootstrap confidence intervals
    metrics_summary = {}
    table_rows = []

    for name, y_prob in predictions.items():
        ci_res = compute_bootstrap_ci(y_test, y_prob, n_resamples=n_boot, ci=0.95, random_state=random_state)
        metrics_summary[name] = {
            "probabilities": y_prob,
            "metrics": ci_res
        }
        auc_p, auc_l, auc_u = ci_res["roc_auc"]
        pr_p, pr_l, pr_u = ci_res["pr_auc"]
        br_p, br_l, br_u = ci_res["brier_score"]
        p15_p, p15_l, p15_u = ci_res["precision_at_top15"]

        model_type = "Probabilistic Baseline" if "NBD" in name else "Supervised ML"
        table_rows.append({
            "Model": name,
            "Type": model_type,
            "Holdout ROC-AUC": f"{auc_p:.4f} [{auc_l:.4f}–{auc_u:.4f}]",
            "Holdout PR-AUC": f"{pr_p:.4f} [{pr_l:.4f}–{pr_u:.4f}]",
            "Brier Score": f"{br_p:.4f} [{br_l:.4f}–{br_u:.4f}]",
            "Precision@Top15%": f"{p15_p*100:.2f}% [{p15_l*100:.2f}–{p15_u*100:.2f}%]",
            "raw_auc": auc_p,
            "raw_pr": pr_p,
            "raw_brier": br_p,
            "raw_p15": p15_p,
        })

    summary_table = pd.DataFrame(table_rows)

    return {
        "train_cutoff": train_cutoff,
        "test_cutoff": test_cutoff,
        "train_sample_size": len(train_cohort),
        "test_sample_size": len(test_cohort),
        "train_churn_rate": float(train_cohort["ActualChurn"].mean()),
        "test_churn_rate": float(test_cohort["ActualChurn"].mean()),
        "metrics_summary": metrics_summary,
        "benchmark_table": summary_table,
        "y_test": y_test,
        "test_cohort": test_cohort
    }
