"""Train the surrender-propensity model on synthetic data and write an eval report.

    python -m ml.train_propensity                 # 20,000 synthetic customers
    python -m ml.train_propensity --mlflow        # also log the run to MLflow
    python -m ml.train_propensity --check         # fail if test AUC drops below the floor

Writes:
    data/generated/policies.csv, data/generated/policy_features.csv   (read by the API)
    models/propensity.joblib, models/propensity_meta.json
    reports/propensity_eval.json, reports/propensity_eval.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.synthetic_data import generate
from ml.decision import HIGH_RISK, MEDIUM_RISK
from ml.features import FEATURE_NAMES, build_feature_table

ROOT = Path(__file__).resolve().parent.parent
AUC_FLOOR = 0.70  # used by --check in CI


def candidates(seed: int) -> dict:
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }


def split_by_customer(df: pd.DataFrame, seed: int):
    """60/20/20 train/validation/test split. A customer's policies stay in one split."""
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(outer.split(df, groups=df["customer_id"]))
    trainval = df.iloc[trainval_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, val_idx = next(inner.split(trainval, groups=trainval["customer_id"]))
    return trainval.iloc[train_idx], trainval.iloc[val_idx], df.iloc[test_idx]


def score_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    top_decile = p >= np.quantile(p, 0.9)
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
        "lift_top_decile": round(float(y[top_decile].mean() / y.mean()), 2),
    }


def tier_table(y: np.ndarray, p: np.ndarray) -> list[dict]:
    rows = []
    tiers = [("high", p >= HIGH_RISK), ("medium", (p >= MEDIUM_RISK) & (p < HIGH_RISK)), ("low", p < MEDIUM_RISK)]
    for name, mask in tiers:
        n = int(mask.sum())
        rows.append(
            {
                "tier": name,
                "policies": n,
                "share": round(n / len(y), 4),
                "mean_predicted": round(float(p[mask].mean()), 4) if n else None,
                "observed_surrender_rate": round(float(y[mask].mean()), 4) if n else None,
                "share_of_all_surrenders": round(float(y[mask].sum() / y.sum()), 4),
            }
        )
    return rows


def calibration_bins(y: np.ndarray, p: np.ndarray) -> list[dict]:
    edges = np.linspace(0, 1, 11)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if mask.sum():
            rows.append(
                {
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "policies": int(mask.sum()),
                    "mean_predicted": round(float(p[mask].mean()), 4),
                    "observed_rate": round(float(y[mask].mean()), 4),
                }
            )
    return rows


def train(
    n_customers: int = 20_000,
    seed: int = 42,
    data_dir: Path = ROOT / "data" / "generated",
    model_dir: Path = ROOT / "models",
    report_dir: Path | None = ROOT / "reports",
    log=print,
) -> dict:
    data = generate(n_customers=n_customers, seed=seed)
    features = build_feature_table(data.customers, data.policies, data.interactions, data.as_of)
    df = features.merge(data.labels, on="policy_id")
    df = df[df["policy_status"] == "Active"].reset_index(drop=True)
    log(f"{n_customers:,} customers, {len(data.policies):,} policies, {len(df):,} active policies used for training")

    train_df, val_df, test_df = split_by_customer(df, seed)
    y = "surrendered_in_next_90d"

    # Pick the model on the validation split
    val_auc = {}
    for name, model in candidates(seed).items():
        model.fit(train_df[FEATURE_NAMES], train_df[y])
        val_auc[name] = round(float(roc_auc_score(val_df[y], model.predict_proba(val_df[FEATURE_NAMES])[:, 1])), 4)
    chosen = max(val_auc, key=val_auc.get)
    log(f"validation AUC: {val_auc} -> using {chosen}")

    # Refit each candidate on train+validation, report on the untouched test split
    trainval = pd.concat([train_df, val_df])
    fitted, test_metrics = {}, {}
    y_test = test_df[y].to_numpy()
    for name, model in candidates(seed).items():
        model.fit(trainval[FEATURE_NAMES], trainval[y])
        fitted[name] = model
        test_metrics[name] = score_metrics(y_test, model.predict_proba(test_df[FEATURE_NAMES])[:, 1])
    test_metrics["generator_probability (ceiling)"] = score_metrics(y_test, test_df["true_probability"].to_numpy())

    model = fitted[chosen]
    p_test = model.predict_proba(test_df[FEATURE_NAMES])[:, 1]
    model_version = f"{chosen}-seed{seed}-n{n_customers}"

    report = {
        "model_version": model_version,
        "chosen_model": chosen,
        "data": {
            "synthetic_customers": n_customers,
            "seed": seed,
            "as_of": data.as_of.isoformat(),
            "active_policies": len(df),
            "train_policies": len(train_df),
            "validation_policies": len(val_df),
            "test_policies": len(test_df),
            "test_base_rate": round(float(y_test.mean()), 4),
        },
        "validation_roc_auc": val_auc,
        "test_metrics": test_metrics,
        "risk_tiers_on_test": tier_table(y_test, p_test),
        "calibration_on_test": calibration_bins(y_test, p_test),
        "thresholds": {"high": HIGH_RISK, "medium": MEDIUM_RISK},
        "library_versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }

    # Artifacts for the API
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    policies_out = data.policies.copy()
    for col in ["start_date", "last_premium_paid"]:
        policies_out[col] = pd.to_datetime(policies_out[col]).dt.date.astype(str)
    policies_out.to_csv(data_dir / "policies.csv", index=False)
    features.to_csv(data_dir / "policy_features.csv", index=False)
    joblib.dump(
        {
            "model": model,
            "features": FEATURE_NAMES,
            "medians": trainval[FEATURE_NAMES].median().to_dict(),
            "model_version": model_version,
        },
        model_dir / "propensity.joblib",
    )
    (model_dir / "propensity_meta.json").write_text(json.dumps(report, indent=2))

    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "propensity_eval.json").write_text(json.dumps(report, indent=2) + "\n")
        (report_dir / "propensity_eval.md").write_text(render_markdown(report))

    t = test_metrics[chosen]
    ceiling = test_metrics["generator_probability (ceiling)"]["roc_auc"]
    log(f"test ROC AUC {t['roc_auc']} (ceiling {ceiling}), PR AUC {t['pr_auc']}, Brier {t['brier']}")
    return report


def render_markdown(r: dict) -> str:
    d = r["data"]
    lines = [
        "# Surrender propensity model: evaluation on synthetic data",
        "",
        "Generated by `python -m ml.train_propensity`. All data is synthetic (see `data/synthetic_data.py`).",
        "The labels come from a hand-written rule plus noise, so these numbers show how well the model",
        "recovers that rule. They say nothing about how it would perform on real policies.",
        "",
        f"- Model: `{r['model_version']}` (chosen on validation ROC AUC: "
        + ", ".join(f"{k} {v}" for k, v in r["validation_roc_auc"].items())
        + ")",
        f"- Data: {d['synthetic_customers']:,} synthetic customers, {d['active_policies']:,} active policies, "
        f"snapshot date {d['as_of']}, seed {d['seed']}",
        f"- Split by customer: {d['train_policies']:,} train / {d['validation_policies']:,} validation / "
        f"{d['test_policies']:,} test policies",
        f"- Surrender rate in the test split: {d['test_base_rate']:.1%}",
        "",
        "## Test split",
        "",
        "| Scores from | ROC AUC | PR AUC | Brier | Log loss | Lift in top 10% |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in r["test_metrics"].items():
        lines.append(
            f"| {name} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {m['brier']:.3f} | {m['log_loss']:.3f} "
            f"| {m['lift_top_decile']:.2f}x |"
        )
    lines += [
        "",
        "The ceiling row scores the test split with the generator's own probabilities (the rule before the",
        "coin flip). No model can beat it on average, so the gap to it is the useful number.",
        "",
        f"## Risk tiers on the test split ({r['chosen_model']})",
        "",
        "| Tier | Policies | Share | Mean predicted | Observed surrender rate | Share of all surrenders |",
        "|---|---|---|---|---|---|",
    ]
    for t in r["risk_tiers_on_test"]:
        if not t["policies"]:
            lines.append(f"| {t['tier']} | 0 | 0% | - | - | - |")
            continue
        lines.append(
            f"| {t['tier']} | {t['policies']:,} | {t['share']:.1%} | {t['mean_predicted']:.1%} "
            f"| {t['observed_surrender_rate']:.1%} | {t['share_of_all_surrenders']:.1%} |"
        )
    lines += [
        "",
        f"Tiers use the API's thresholds (`ml/decision.py`): high >= {r['thresholds']['high']} "
        "(more likely than not to surrender),",
        f"medium >= {r['thresholds']['medium']} (above the average surrender rate). High-risk scores are queued",
        "for human review before any retention outreach.",
        "",
        "## Calibration on the test split",
        "",
        "| Predicted | Policies | Mean predicted | Observed rate |",
        "|---|---|---|---|",
    ]
    for b in r["calibration_on_test"]:
        lines.append(f"| {b['bin']} | {b['policies']:,} | {b['mean_predicted']:.1%} | {b['observed_rate']:.1%} |")
    v = r["library_versions"]
    lines += ["", f"Library versions: scikit-learn {v['scikit-learn']}, numpy {v['numpy']}, pandas {v['pandas']}.", ""]
    return "\n".join(lines)


def log_to_mlflow(report: dict, model_dir: Path) -> None:
    """Optional: record the run in MLflow (local ./mlruns, or Databricks via MLFLOW_TRACKING_URI)."""
    import inspect

    import mlflow  # optional dependency: requirements-databricks.txt
    import mlflow.sklearn

    artifact = joblib.load(model_dir / "propensity.joblib")
    chosen = report["chosen_model"]
    log_kwargs = {}
    if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
        # Newer MLflow saves with skops and asks which types to trust. We trained this model ourselves.
        log_kwargs["skops_trusted_types"] = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]
    # Databricks needs a workspace path here, e.g. MLFLOW_EXPERIMENT_NAME=/Shared/insurance-policy-copilot
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "insurance-policy-copilot-propensity"))
    with mlflow.start_run(run_name=report["model_version"]):
        mlflow.log_params({**report["data"], "chosen_model": chosen})
        mlflow.log_metrics({f"test_{k}": v for k, v in report["test_metrics"][chosen].items()})
        mlflow.log_dict(report, "propensity_eval.json")
        mlflow.sklearn.log_model(artifact["model"], name="model", **log_kwargs)
    print(f"Logged run to MLflow at {mlflow.get_tracking_uri()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--customers", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-report", action="store_true", help="don't write reports/")
    parser.add_argument("--mlflow", action="store_true", help="also log the run to MLflow")
    parser.add_argument("--check", action="store_true", help=f"exit 1 if test ROC AUC < {AUC_FLOOR}")
    args = parser.parse_args(argv)

    report = train(
        n_customers=args.customers,
        seed=args.seed,
        report_dir=None if args.no_report else ROOT / "reports",
    )
    if args.mlflow:
        log_to_mlflow(report, ROOT / "models")
    if args.check:
        auc = report["test_metrics"][report["chosen_model"]]["roc_auc"]
        if auc < AUC_FLOOR:
            print(f"FAIL: test ROC AUC {auc} < {AUC_FLOOR}")
            return 1
        print(f"OK: test ROC AUC {auc} >= {AUC_FLOOR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
