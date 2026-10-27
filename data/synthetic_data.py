"""Synthetic customers, policies, contact history and surrender labels.

Everything here is made up. No names, emails or other personal details are
generated: customers are identified only by IDs like C000123.

The surrender label is drawn from a hand-written rule (see `surrender_probability`),
the same rule the original Databricks generator used. A model trained on this data
can at best recover that rule, so its scores say nothing about real customers.

The output is fully determined by `seed` and `as_of`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

AS_OF = date(2026, 1, 1)

POLICY_TYPES = ["Term Life", "Whole Life", "ULIP", "Endowment", "Health", "Motor"]
POLICY_STATUSES = ["Active", "Lapsed", "Surrendered", "Matured", "Pending"]
STATUS_WEIGHTS = [0.70, 0.10, 0.05, 0.10, 0.05]
PREMIUM_FREQUENCIES = ["Monthly", "Quarterly", "Semi-Annual", "Annual"]
CHANNELS = ["Call", "Email", "Chat", "Branch Visit", "App"]
CONTACT_REASONS = [
    "Premium Payment",
    "Policy Inquiry",
    "Claim Status",
    "Surrender Query",
    "Document Upload",
    "Complaint",
    "New Product",
    "Renewal",
]


@dataclass
class SyntheticData:
    customers: pd.DataFrame
    policies: pd.DataFrame
    interactions: pd.DataFrame
    labels: pd.DataFrame
    as_of: date


def surrender_probability(
    payment_gaps: np.ndarray,
    fund_value_change_3m: np.ndarray,
    avg_sentiment: np.ndarray,
    surrender_queries: np.ndarray,
    tenure_months: np.ndarray,
    surrender_charge_pct: np.ndarray,
) -> np.ndarray:
    """The rule that generates labels (before noise is added)."""
    p = np.full(len(payment_gaps), 0.10)
    p += 0.20 * (payment_gaps > 1)
    p += 0.15 * (fund_value_change_3m < -0.05)
    p += 0.10 * (avg_sentiment < -0.3)
    p += 0.25 * (surrender_queries > 0)
    p -= 0.10 * ((tenure_months < 36) & (surrender_charge_pct > 5))
    return p


def generate(n_customers: int = 20_000, seed: int = 42, as_of: date = AS_OF) -> SyntheticData:
    rng = np.random.default_rng(seed)
    as_of_ts = pd.Timestamp(as_of)

    # Customers
    customer_ids = np.array([f"C{i:06d}" for i in range(1, n_customers + 1)])
    days_as_customer = rng.integers(1, 15 * 365, n_customers)
    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "customer_since": as_of_ts - pd.to_timedelta(days_as_customer, unit="D"),
            "annual_income": rng.lognormal(11, 0.8, n_customers).astype(int),
            "credit_score": np.clip(rng.normal(700, 100, n_customers), 300, 850).astype(int),
            "risk_score": rng.beta(2, 5, n_customers).round(3),
            "preferred_channel": rng.choice(CHANNELS, n_customers),
        }
    )

    # Policies: 1-5 per customer, started on or after the customer joined
    n_policies_per_customer = np.clip(rng.poisson(2.4, n_customers), 1, 5)
    owner = np.repeat(np.arange(n_customers), n_policies_per_customer)
    n_policies = len(owner)
    days_since_start = (rng.random(n_policies) * days_as_customer[owner]).astype(int)
    tenure_months = days_since_start // 30
    premium = rng.uniform(500, 50_000, n_policies).round(2)
    sum_assured = (premium * rng.uniform(10, 100, n_policies)).round(2)
    term_years = rng.choice([5, 10, 15, 20, 25, 30], n_policies)
    start_date = as_of_ts - pd.to_timedelta(days_since_start, unit="D")
    policies = pd.DataFrame(
        {
            "policy_id": [f"P{i:08d}" for i in range(1, n_policies + 1)],
            "customer_id": customer_ids[owner],
            "policy_type": rng.choice(POLICY_TYPES, n_policies),
            "policy_status": rng.choice(POLICY_STATUSES, n_policies, p=STATUS_WEIGHTS),
            "start_date": start_date,
            "term_years": term_years,
            "premium_amount": premium,
            "premium_frequency": rng.choice(PREMIUM_FREQUENCIES, n_policies),
            "sum_assured": sum_assured,
            "current_fund_value": (sum_assured * rng.uniform(0.8, 1.3, n_policies)).round(2),
            "fund_value_change_3m": rng.uniform(-0.15, 0.25, n_policies).round(3),
            "tenure_months": tenure_months,
            "surrender_charge_pct": np.clip(10 - tenure_months / 12, 0, 10).round(2),
            "lock_in_period_months": rng.choice([36, 48, 60], n_policies),
            "payment_gaps": rng.choice([0, 1, 2, 3], n_policies, p=[0.70, 0.15, 0.10, 0.05]),
            "last_premium_paid": as_of_ts - pd.to_timedelta(rng.integers(0, 61, n_policies), unit="D"),
        }
    )

    # Contact history over the last year: about 5 contacts per customer
    n_contacts = rng.poisson(5, n_customers)
    contact_owner = np.repeat(np.arange(n_customers), n_contacts)
    n_int = len(contact_owner)
    interactions = pd.DataFrame(
        {
            "interaction_id": [f"INT{i:08d}" for i in range(1, n_int + 1)],
            "customer_id": customer_ids[contact_owner],
            "interaction_date": as_of_ts - pd.to_timedelta(rng.integers(0, 366, n_int), unit="D"),
            "channel": rng.choice(CHANNELS, n_int),
            "reason": rng.choice(CONTACT_REASONS, n_int),
            "sentiment_score": rng.uniform(-1, 1, n_int).round(3),
        }
    )

    # Labels: per policy, from the customer's contact history and the policy itself
    sentiment_sum = np.bincount(contact_owner, weights=interactions["sentiment_score"], minlength=n_customers)
    avg_sentiment = np.divide(sentiment_sum, n_contacts, out=np.zeros(n_customers), where=n_contacts > 0)
    is_surrender_query = (interactions["reason"] == "Surrender Query").to_numpy()
    surrender_queries = np.bincount(contact_owner, weights=is_surrender_query, minlength=n_customers)

    p = surrender_probability(
        policies["payment_gaps"].to_numpy(),
        policies["fund_value_change_3m"].to_numpy(),
        avg_sentiment[owner],
        surrender_queries[owner],
        tenure_months,
        policies["surrender_charge_pct"].to_numpy(),
    )
    p = np.clip(p + rng.normal(0, 0.05, n_policies), 0.05, 0.95)
    labels = pd.DataFrame(
        {
            "policy_id": policies["policy_id"],
            "true_probability": p.round(4),
            "surrendered_in_next_90d": (rng.random(n_policies) < p).astype(int),
        }
    )

    return SyntheticData(customers, policies, interactions, labels, as_of)
