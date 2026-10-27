# Databricks version (optional)

These notebooks are the first version of the propensity pipeline, written for Azure
Databricks: synthetic data in Delta tables, feature tables in Unity Catalog, an XGBoost
model with isotonic calibration, and the MLflow model registry.

The local pipeline in `ml/` replaced them for everything the API uses. They are kept
for anyone who wants the same idea on Databricks.

**Status: not run in CI, and I have not re-run them since the local rewrite.** After moving
them here I fixed two calls that newer libraries removed (XGBoost 2 now takes
`early_stopping_rounds` in the constructor; scikit-learn 1.6+ replaces `cv="prefit"` with
`FrozenEstimator`) and the import paths. Expect to adjust catalog names and cluster
settings for your workspace.

| File | What it does |
|---|---|
| `01_databricks_setup.py` | Creates the `insurance_copilot` catalog and schemas, generates synthetic data, writes Delta tables, builds feature tables and views |
| `02_train_surrender_model.py` | Trains XGBoost with SMOTE and calibration, logs to MLflow, registers `surrender_propensity_model` |
| `03_upload_data.py` | Alternative to step 1: loads parquet files you uploaded yourself into Delta tables |
| `setup.sql` | The same DDL as plain SQL |
| `synthetic_data_generator.py` | The original generator these notebooks import (uses Faker for names; names are masked before writing) |

## Running them

1. Add this repository as a Databricks Git folder, so the notebooks can import
   `synthetic_data_generator.py` from the same folder.
2. Run `01_databricks_setup` on a cluster with Unity Catalog enabled, then `02_train_surrender_model`.
   Each notebook installs its own packages with `%pip`.

## Logging the local model to Databricks MLflow instead

If you only want experiment tracking, the local training script can log to a Databricks
workspace's MLflow without these notebooks. I have tested this against a local MLflow
store, not against a Databricks workspace:

```bash
pip install -r requirements-databricks.txt
export MLFLOW_TRACKING_URI=databricks
export DATABRICKS_HOST=https://<workspace>.azuredatabricks.net
export DATABRICKS_TOKEN=<token>
export MLFLOW_EXPERIMENT_NAME=/Shared/insurance-policy-copilot
python -m ml.train_propensity --mlflow
```
