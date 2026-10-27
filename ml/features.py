"""Feature definitions shared by training and serving.

Both `ml/train_propensity.py` and the API read features from the table this
module builds, so the model is scored on exactly the columns it was trained on.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# Feature name -> plain-English description (shown as "top factors" in the API)
FEATURES: dict[str, str] = {
    "tenure_months": "months since the policy started",
    "premium_amount": "premium amount",
    "payment_gaps": "missed premium payments on this policy",
    "surrender_charge_pct": "current surrender charge (%)",
    "fund_value_change_3m": "fund value change over the last 3 months",
    "fund_to_sum_ratio": "fund value relative to sum assured",
    "in_lock_period": "policy still in its lock-in period",
    "days_since_last_payment": "days since the last premium payment",
    "avg_sentiment_12m": "average sentiment of contacts in the last year",
    "contacts_12m": "contacts in the last year (capped at 10)",
    "surrender_queries_12m": "surrender-related contacts in the last year",
    "complaints_12m": "complaints in the last year",
    "annual_income": "annual income",
    "credit_score": "credit score",
    "risk_score": "customer risk score",
    "customer_tenure_days": "days as a customer",
    "total_policies": "number of policies held",
    "avg_payment_gaps_all_policies": "average missed payments across the customer's policies",
    "premium_to_income_ratio": "premium relative to monthly income",
}
FEATURE_NAMES = list(FEATURES)


def build_feature_table(
    customers: pd.DataFrame,
    policies: pd.DataFrame,
    interactions: pd.DataFrame,
    as_of: date,
) -> pd.DataFrame:
    """One row per policy: IDs, status and every feature in FEATURES."""
    as_of_ts = pd.Timestamp(as_of)

    contacts = interactions.groupby("customer_id").agg(
        avg_sentiment_12m=("sentiment_score", "mean"),
        contacts_12m=("interaction_id", "count"),
        surrender_queries_12m=("reason", lambda r: int((r == "Surrender Query").sum())),
        complaints_12m=("reason", lambda r: int((r == "Complaint").sum())),
    )
    contacts["contacts_12m"] = contacts["contacts_12m"].clip(upper=10)

    per_customer = policies.groupby("customer_id").agg(
        total_policies=("policy_id", "count"),
        avg_payment_gaps_all_policies=("payment_gaps", "mean"),
    )

    cust = customers.set_index("customer_id")[["annual_income", "credit_score", "risk_score"]].copy()
    cust["customer_tenure_days"] = (
        as_of_ts - pd.to_datetime(customers.set_index("customer_id")["customer_since"])
    ).dt.days

    df = policies[
        [
            "policy_id",
            "customer_id",
            "policy_status",
            "tenure_months",
            "premium_amount",
            "payment_gaps",
            "surrender_charge_pct",
            "fund_value_change_3m",
        ]
    ].copy()
    df["fund_to_sum_ratio"] = policies["current_fund_value"] / policies["sum_assured"]
    df["in_lock_period"] = (policies["tenure_months"] < policies["lock_in_period_months"]).astype(int)
    df["days_since_last_payment"] = (as_of_ts - pd.to_datetime(policies["last_premium_paid"])).dt.days

    df = df.join(contacts, on="customer_id").join(per_customer, on="customer_id").join(cust, on="customer_id")
    for col in ["avg_sentiment_12m", "contacts_12m", "surrender_queries_12m", "complaints_12m"]:
        df[col] = df[col].fillna(0)
    df["premium_to_income_ratio"] = df["premium_amount"] / np.maximum(df["annual_income"] / 12, 1)

    return df[["policy_id", "customer_id", "policy_status", *FEATURE_NAMES]].reset_index(drop=True)
