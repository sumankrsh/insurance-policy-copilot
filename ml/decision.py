"""Decision policy: how a surrender probability turns into a recommended action.

- high   (p >= 0.50): more likely than not to surrender in the next 90 days.
                      Recommend a retention call, but hold it for human review first.
- medium (p >= 0.25): above the synthetic portfolio's average surrender rate (about 26%).
                      Recommend sending the policy's surrender-charge and benefits summary.
- low:                no action.
"""

from __future__ import annotations

HIGH_RISK = 0.50
MEDIUM_RISK = 0.25

ACTIONS = {
    "high": "retention_call",
    "medium": "send_surrender_charge_summary",
    "low": "no_action",
}


def risk_tier(probability: float, high: float = HIGH_RISK, medium: float = MEDIUM_RISK) -> str:
    if probability >= high:
        return "high"
    if probability >= medium:
        return "medium"
    return "low"


def needs_human_review(tier: str) -> bool:
    return tier == "high"
