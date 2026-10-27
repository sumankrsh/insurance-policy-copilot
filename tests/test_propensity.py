import pandas as pd
import pytest

from app.propensity import ModelNotTrained, PropensityService
from data.synthetic_data import generate
from ml.decision import ACTIONS, needs_human_review, risk_tier
from ml.features import FEATURE_NAMES, build_feature_table


def test_generator_is_deterministic_and_has_no_personal_details():
    a, b = generate(n_customers=200, seed=1), generate(n_customers=200, seed=1)
    pd.testing.assert_frame_equal(a.policies, b.policies)
    pd.testing.assert_frame_equal(a.labels, b.labels)
    assert not {"first_name", "last_name", "email", "phone", "address"} & set(a.customers.columns)


def test_feature_table_has_one_row_per_policy_and_no_label_leak():
    data = generate(n_customers=300, seed=3)
    features = build_feature_table(data.customers, data.policies, data.interactions, data.as_of)
    assert len(features) == len(data.policies)
    assert list(features.columns[3:]) == FEATURE_NAMES
    assert features[FEATURE_NAMES].notna().all().all()
    assert "true_probability" not in features and "surrendered_in_next_90d" not in features


def test_trained_model_beats_baseline_and_nears_ceiling(artifacts):
    _, report = artifacts
    metrics = report["test_metrics"]
    chosen = metrics[report["chosen_model"]]["roc_auc"]
    ceiling = metrics["generator_probability (ceiling)"]["roc_auc"]
    assert chosen > 0.65
    assert chosen <= ceiling + 0.03  # can't meaningfully beat the rule that made the labels


@pytest.mark.parametrize("p, tier", [(0.9, "high"), (0.5, "high"), (0.49, "medium"), (0.25, "medium"), (0.1, "low")])
def test_risk_tiers(p, tier):
    assert risk_tier(p) == tier
    assert needs_human_review(tier) == (tier == "high")
    assert tier in ACTIONS


@pytest.fixture
def service(settings):
    return PropensityService(settings.model_dir, settings.data_dir, 0.5, 0.25)


def active_customer(service):
    return service.features[service.features["policy_status"] == "Active"]["customer_id"].iloc[0]


def test_scores_are_probabilities_sorted_with_explanations(service):
    result = service.score_customer(active_customer(service))
    probs = [p["probability"] for p in result["policies"]]
    assert probs and all(0 <= p <= 1 for p in probs)
    assert probs == sorted(probs, reverse=True)
    assert result["decision"]["policy_id"] == result["policies"][0]["policy_id"]
    for p in result["policies"]:
        assert all(f["feature"] in FEATURE_NAMES and f["effect"] > 0 for f in p["top_factors"])


def test_scoring_is_repeatable(service):
    cid = active_customer(service)
    assert service.score_customer(cid) == service.score_customer(cid)


def test_unknown_customer_returns_none(service):
    assert service.score_customer("C999999") is None


def test_surrender_value_is_fund_value_minus_charge(service):
    pid = service.policies.index[0]
    v = service.surrender_value(pid)
    expected = round(v["current_fund_value"] * (1 - v["surrender_charge_pct"] / 100), 2)
    assert v["indicative_surrender_value"] == pytest.approx(expected, abs=0.01)
    assert "Not a quote" in v["note"]


def test_missing_model_raises(tmp_path):
    service = PropensityService(tmp_path / "models", tmp_path / "data", 0.5, 0.25)
    assert not service.available
    with pytest.raises(ModelNotTrained):
        service.score_customer("C000001")
