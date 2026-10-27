"""Serve the surrender-propensity model and the synthetic policy records.

Reads what `python -m ml.train_propensity` writes: the model in models/ and the
policy and feature tables in data/generated/.

Top factors are per-prediction: for each feature, how much the score drops when that
feature is replaced by its training median. It is a simple, model-agnostic
explanation, not SHAP; it ignores interactions between features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.decision import ACTIONS, needs_human_review, risk_tier
from ml.features import FEATURES

MODEL_FILE = "propensity.joblib"


class ModelNotTrained(RuntimeError):
    pass


def _py(value: Any) -> Any:
    """numpy scalars -> plain Python for JSON."""
    if isinstance(value, np.generic):
        return value.item()
    return value


class PropensityService:
    def __init__(self, model_dir: Path, data_dir: Path, high: float, medium: float):
        self.high, self.medium = high, medium
        model_path = model_dir / MODEL_FILE
        features_path = data_dir / "policy_features.csv"
        policies_path = data_dir / "policies.csv"
        self.available = model_path.exists() and features_path.exists() and policies_path.exists()
        if not self.available:
            return
        # joblib uses pickle: only load model files you produced yourself.
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.feature_names: list[str] = artifact["features"]
        self.medians: dict[str, float] = artifact["medians"]
        self.model_version: str = artifact["model_version"]
        self.features = pd.read_csv(features_path).set_index("policy_id", drop=False)
        self.policies = pd.read_csv(policies_path).set_index("policy_id", drop=False)

    def _require(self) -> None:
        if not self.available:
            raise ModelNotTrained("No trained model found. Run: python -m ml.train_propensity")

    # --- scoring -------------------------------------------------------------------------

    def _explain(self, X: pd.DataFrame, base: np.ndarray, top_n: int = 3) -> list[list[dict[str, Any]]]:
        n, names = len(X), self.feature_names
        # One row per (policy, feature) with that feature set to its median
        perturbed = pd.concat([X] * len(names), ignore_index=True)
        for j, name in enumerate(names):
            perturbed.loc[j * n : (j + 1) * n - 1, name] = self.medians[name]
        p = self.model.predict_proba(perturbed[names])[:, 1].reshape(len(names), n).T  # (n, features)
        effects = base[:, None] - p
        out = []
        for i in range(n):
            order = np.argsort(-effects[i])[:top_n]
            out.append(
                [
                    {
                        "feature": names[j],
                        "description": FEATURES[names[j]],
                        "value": round(_py(X.iloc[i][names[j]]), 3),
                        "typical_value": round(float(self.medians[names[j]]), 3),
                        "effect": round(float(effects[i, j]), 3),
                    }
                    for j in order
                    if effects[i, j] > 0.005
                ]
            )
        return out

    def score_customer(self, customer_id: str) -> dict[str, Any] | None:
        """Score every active policy of a customer. None if the customer is unknown."""
        self._require()
        rows = self.features[self.features["customer_id"] == customer_id]
        if rows.empty:
            return None
        active = rows[rows["policy_status"] == "Active"]
        result: dict[str, Any] = {
            "customer_id": customer_id,
            "model_version": self.model_version,
            "policies": [],
            "decision": None,
        }
        if active.empty:
            return result

        X = active[self.feature_names].astype(float)
        probs = self.model.predict_proba(X)[:, 1]
        factors = self._explain(X, probs)
        for policy_id, p, f in zip(active.index, probs, factors, strict=True):
            tier = risk_tier(float(p), self.high, self.medium)
            result["policies"].append(
                {
                    "policy_id": policy_id,
                    "policy_type": self.policies.loc[policy_id, "policy_type"],
                    "probability": round(float(p), 3),
                    "risk_tier": tier,
                    "top_factors": f,
                }
            )
        result["policies"].sort(key=lambda r: -r["probability"])
        top = result["policies"][0]
        result["decision"] = {
            "policy_id": top["policy_id"],
            "probability": top["probability"],
            "risk_tier": top["risk_tier"],
            "recommended_action": ACTIONS[top["risk_tier"]],
            "needs_human_review": needs_human_review(top["risk_tier"]),
        }
        return result

    # --- policy records ------------------------------------------------------------------

    def policy_summary(self, policy_id: str) -> dict[str, Any] | None:
        self._require()
        if policy_id not in self.policies.index:
            return None
        p = self.policies.loc[policy_id]
        return {
            "policy_id": policy_id,
            "customer_id": p["customer_id"],
            "policy_type": p["policy_type"],
            "status": p["policy_status"],
            "start_date": p["start_date"],
            "term_years": _py(p["term_years"]),
            "tenure_months": _py(p["tenure_months"]),
            "premium_amount": _py(p["premium_amount"]),
            "premium_frequency": p["premium_frequency"],
            "sum_assured": _py(p["sum_assured"]),
            "current_fund_value": _py(p["current_fund_value"]),
            "surrender_charge_pct": _py(p["surrender_charge_pct"]),
            "lock_in_months_remaining": max(0, _py(p["lock_in_period_months"]) - _py(p["tenure_months"])),
            "missed_payments": _py(p["payment_gaps"]),
            "last_premium_paid": p["last_premium_paid"],
        }

    def surrender_value(self, policy_id: str) -> dict[str, Any] | None:
        summary = self.policy_summary(policy_id)
        if summary is None:
            return None
        fund = float(summary["current_fund_value"])
        charge = round(fund * summary["surrender_charge_pct"] / 100, 2)
        lock_in = summary["lock_in_months_remaining"]
        return {
            "policy_id": policy_id,
            "current_fund_value": fund,
            "surrender_charge_pct": summary["surrender_charge_pct"],
            "surrender_charge_amount": charge,
            "indicative_surrender_value": round(fund - charge, 2),
            "lock_in_months_remaining": lock_in,
            "note": (
                "Indicative only: fund value minus the surrender charge on the synthetic policy record. "
                + (f"The policy is still in its lock-in period ({lock_in} months left). " if lock_in else "")
                + "Not a quote."
            ),
        }
