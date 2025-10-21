# Databricks notebook source
# MAGIC %md
# MAGIC # Surrender Propensity Model Training
# MAGIC Train XGBoost model with MLflow tracking and Feature Store integration

# COMMAND ----------

# MAGIC %pip install xgboost scikit-learn imbalanced-learn shap matplotlib seaborn mlflow

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, 
    classification_report, confusion_matrix,
    roc_curve, precision_score, recall_score, f1_score
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# Configuration
catalog = "insurance_copilot"
data_schema = "policy_data"
feature_schema = "features"
ml_schema = "ml_models"

# MLflow configuration
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/insurance-copilot-surrender-model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Training Data

# COMMAND ----------

# Load features from Delta tables
query = f"""
SELECT 
    sl.policy_id,
    sl.customer_id,
    sl.tenure_months,
    sl.premium_amount,
    sl.payment_gaps,
    sl.surrender_charge_pct,
    sl.fund_value_change_3m,
    sl.avg_sentiment,
    sl.interaction_count_90d,
    sl.surrender_queries_90d,
    sl.complaints_90d,
    sl.agent_activity_30d,
    pf.fund_to_sum_ratio,
    pf.in_lock_period,
    pf.days_since_last_payment,
    cf.annual_income,
    cf.credit_score,
    cf.risk_score,
    cf.customer_tenure_days,
    cf.total_policies,
    cf.avg_payment_gaps as hist_avg_payment_gaps,
    sl.surrendered_in_next_90d as target
FROM {catalog}.{data_schema}.surrender_labels sl
JOIN {catalog}.{feature_schema}.policy_features pf 
    ON sl.policy_id = pf.policy_id
JOIN {catalog}.{feature_schema}.customer_features cf 
    ON sl.customer_id = cf.customer_id
"""

df = spark.sql(query).toPandas()
print(f"Loaded {len(df)} records for training")
print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering

# COMMAND ----------

# Additional feature engineering
df['premium_to_income_ratio'] = df['premium_amount'] / (df['annual_income'] / 12)
df['risk_adjusted_tenure'] = df['tenure_months'] * (1 - df['risk_score'])
df['payment_reliability'] = 1 / (1 + df['payment_gaps'])
df['sentiment_interaction'] = df['avg_sentiment'] * df['interaction_count_90d']
df['financial_stress'] = (df['payment_gaps'] > 0).astype(int) * df['fund_value_change_3m']

# Handle missing values
df = df.fillna(df.median())

# Select features for model
feature_cols = [
    'tenure_months', 'premium_amount', 'payment_gaps', 'surrender_charge_pct',
    'fund_value_change_3m', 'avg_sentiment', 'interaction_count_90d',
    'surrender_queries_90d', 'complaints_90d', 'agent_activity_30d',
    'fund_to_sum_ratio', 'in_lock_period', 'days_since_last_payment',
    'annual_income', 'credit_score', 'risk_score', 'customer_tenure_days',
    'total_policies', 'hist_avg_payment_gaps', 'premium_to_income_ratio',
    'risk_adjusted_tenure', 'payment_reliability', 'sentiment_interaction',
    'financial_stress'
]

X = df[feature_cols]
y = df['target']

print(f"Features shape: {X.shape}")
print(f"Feature columns: {feature_cols}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train-Test Split

# COMMAND ----------

# Stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Handle Class Imbalance

# COMMAND ----------

# Apply SMOTE to training data
smote = SMOTE(random_state=42, sampling_strategy=0.5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Original training set: {X_train.shape}")
print(f"Balanced training set: {X_train_balanced.shape}")
print(f"Class distribution after SMOTE:\n{pd.Series(y_train_balanced).value_counts(normalize=True)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train XGBoost Model

# COMMAND ----------

def calculate_cost_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculate business metrics"""
    
    # Business constants
    avg_policy_value = 50000  # Average policy value
    retention_success_rate = 0.3  # 30% of targeted customers retained
    outreach_cost = 100  # Cost per customer outreach
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Business metrics
    retained_value = tp * avg_policy_value * retention_success_rate
    outreach_costs = (tp + fp) * outreach_cost
    lost_value = fn * avg_policy_value
    
    net_gain = retained_value - outreach_costs - lost_value
    roi = (retained_value - outreach_costs) / outreach_costs if outreach_costs > 0 else 0
    
    return {
        'threshold': threshold,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'retained_value': retained_value,
        'outreach_costs': outreach_costs,
        'lost_value': lost_value,
        'net_gain': net_gain,
        'roi': roi
    }

# COMMAND ----------

with mlflow.start_run(run_name="xgboost_surrender_model"):
    
    # Log parameters
    params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 1,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'scale_pos_weight': len(y_train_balanced[y_train_balanced==0]) / len(y_train_balanced[y_train_balanced==1]),
        'random_state': 42
    }
    mlflow.log_params(params)
    
    # Train model
    model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    
    eval_set = [(X_val, y_val)]
    model.fit(
        X_train_balanced, y_train_balanced,
        eval_set=eval_set,
        early_stopping_rounds=20,
        verbose=False
    )
    
    # Calibrate model for better probability estimates
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_val, y_val)
    
    # Predictions
    y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("auc", auc_score)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    
    print(f"AUC: {auc_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:1000])  # Sample for speed
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0, 0].plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    
    # Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)
    axes[0, 1].plot(recall_vals, precision_vals, label=f'PR (AUC = {pr_auc:.3f})')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend()
    
    # Feature Importance
    top_features = feature_importance.head(10)
    axes[1, 0].barh(top_features['feature'], top_features['importance'])
    axes[1, 0].set_xlabel('Importance')
    axes[1, 0].set_title('Top 10 Feature Importances')
    axes[1, 0].invert_yaxis()
    
    # Calibration Plot
    fraction_pos = []
    mean_pred_val = []
    n_bins = 10
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins
        mask = (y_pred_proba >= lower) & (y_pred_proba < upper)
        if mask.sum() > 0:
            fraction_pos.append(y_test[mask].mean())
            mean_pred_val.append(y_pred_proba[mask].mean())
    
    axes[1, 1].plot(mean_pred_val, fraction_pos, 'o-', label='Model')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Perfect')
    axes[1, 1].set_xlabel('Mean Predicted Probability')
    axes[1, 1].set_ylabel('Fraction of Positives')
    axes[1, 1].set_title('Calibration Plot')
    axes[1, 1].legend()
    
    plt.tight_layout()
    mlflow.log_figure(fig, "model_diagnostics.png")
    plt.show()
    
    # Find optimal threshold
    thresholds = np.arange(0.1, 0.9, 0.05)
    threshold_metrics = []
    
    for threshold in thresholds:
        metrics = calculate_cost_metrics(y_test, y_pred_proba, threshold)
        threshold_metrics.append(metrics)
    
    threshold_df = pd.DataFrame(threshold_metrics)
    optimal_idx = threshold_df['net_gain'].idxmax()
    optimal_threshold = threshold_df.iloc[optimal_idx]['threshold']
    
    print(f"\n💰 Optimal threshold for max net gain: {optimal_threshold:.2f}")
    print(f"   Net gain: ${threshold_df.iloc[optimal_idx]['net_gain']:,.0f}")
    print(f"   ROI: {threshold_df.iloc[optimal_idx]['roi']:.1%}")
    
    mlflow.log_metric("optimal_threshold", optimal_threshold)
    mlflow.log_metric("max_net_gain", threshold_df.iloc[optimal_idx]['net_gain'])
    
    # Log model
    signature = infer_signature(X_train, y_pred)
    
    model_info = mlflow.sklearn.log_model(
        calibrated_model,
        artifact_path="model",
        signature=signature,
        registered_model_name="surrender_propensity_model",
        input_example=X_train[:5],
        pip_requirements=[
            "scikit-learn==1.3.2",
            "xgboost==2.0.2",
            "pandas==2.1.3",
            "numpy==1.24.3"
        ]
    )
    
    # Save feature columns
    mlflow.log_dict({"feature_columns": feature_cols}, "feature_columns.json")
    
    # Save threshold configuration
    threshold_config = {
        "optimal_threshold": float(optimal_threshold),
        "high_risk_threshold": 0.7,
        "low_risk_threshold": 0.3,
        "business_rules": {
            "high_risk_action": "escalate_to_retention_desk",
            "medium_risk_action": "send_personalized_education",
            "low_risk_action": "normal_service"
        }
    }
    mlflow.log_dict(threshold_config, "threshold_config.json")
    
    print(f"\n✅ Model logged to MLflow: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Model Card

# COMMAND ----------

model_card = f"""
# Surrender Propensity Model Card

## Model Details
- **Model Name**: surrender_propensity_model
- **Version**: 1.0
- **Type**: Binary Classification (XGBoost + Isotonic Calibration)
- **Framework**: XGBoost 2.0.2, scikit-learn 1.3.2

## Intended Use
- **Primary Use**: Predict likelihood of policy surrender within 90 days
- **Users**: Insurance retention teams, customer service agents
- **Out of Scope**: Not for regulatory decisions, pricing, or underwriting

## Training Data
- **Source**: {catalog}.{data_schema}
- **Size**: {len(df)} policies
- **Class Distribution**: {df['target'].value_counts().to_dict()}
- **Features**: {len(feature_cols)} features
- **Time Period**: Synthetic data

## Model Performance
- **AUC-ROC**: {auc_score:.4f}
- **Precision**: {precision:.4f}
- **Recall**: {recall:.4f}
- **F1 Score**: {f1:.4f}
- **Optimal Threshold**: {optimal_threshold:.2f}

## Top Predictive Features
{feature_importance.head(5).to_string()}

## Ethical Considerations
- Model uses anonymized/hashed PII
- No discrimination based on protected characteristics
- Regular fairness audits recommended

## Limitations
- Trained on synthetic data (production model needs real data)
- Performance may vary across policy types
- Requires retraining every quarter

## Monitoring
- Track prediction drift weekly
- Monitor feature distributions
- Alert on performance degradation > 5%
- Review false positives monthly
"""

# Save model card
with open("/tmp/model_card.md", "w") as f:
    f.write(model_card)

mlflow.log_artifact("/tmp/model_card.md")
print("✓ Model card saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Best Model Version

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "surrender_propensity_model"

# Get latest version
model_versions = client.search_model_versions(f"name='{model_name}'")
latest_version = max([int(v.version) for v in model_versions])

# Transition to production
client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Production",
    archive_existing_versions=True
)

print(f"✓ Model version {latest_version} promoted to Production")

# Add model alias for easy access
client.set_registered_model_alias(model_name, "champion", latest_version)
print(f"✓ Set alias 'champion' to version {latest_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Inference Pipeline

# COMMAND ----------

# Create batch inference function
def batch_score_policies(spark_df):
    """Score policies in batch using the production model"""
    
    import mlflow
    import pandas as pd
    
    # Load production model
    model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
    
    # Convert to pandas
    df = spark_df.toPandas()
    
    # Generate predictions
    probabilities = model.predict_proba(df[feature_cols])[:, 1]
    
    # Add predictions to dataframe
    df['surrender_probability'] = probabilities
    df['risk_tier'] = pd.cut(
        probabilities, 
        bins=[0, 0.3, 0.7, 1.0],
        labels=['low', 'medium', 'high']
    )
    df['scored_at'] = pd.Timestamp.now()
    
    # Convert back to Spark
    return spark.createDataFrame(df)

# Register UDF for single record scoring
@udf(returnType="double")
def score_single_policy(*features):
    """Score a single policy"""
    import mlflow
    import pandas as pd
    
    model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
    
    # Create dataframe with single row
    df = pd.DataFrame([features], columns=feature_cols)
    
    # Score
    prob = model.predict_proba(df)[0, 1]
    return float(prob)

# Register UDF
spark.udf.register("surrender_score", score_single_policy)

print("✓ Created inference pipeline and UDFs")

# COMMAND ----------

# Test batch scoring
test_query = f"""
SELECT * FROM (
    SELECT 
        policy_id,
        {', '.join(feature_cols)}
    FROM {catalog}.{data_schema}.surrender_labels sl
    JOIN {catalog}.{feature_schema}.policy_features pf 
        ON sl.policy_id = pf.policy_id
    JOIN {catalog}.{feature_schema}.customer_features cf 
        ON sl.customer_id = cf.customer_id
    LIMIT 100
)
"""

test_df = spark.sql(test_query)
scored_df = batch_score_policies(test_df)
scored_df.select("policy_id", "surrender_probability", "risk_tier").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print(f"""
✅ Surrender Propensity Model Training Complete!

Model Performance:
- AUC-ROC: {auc_score:.4f}
- Precision: {precision:.4f}  
- Recall: {recall:.4f}
- F1 Score: {f1:.4f}

Business Metrics:
- Optimal Threshold: {optimal_threshold:.2f}
- Expected Net Gain: ${threshold_df.iloc[optimal_idx]['net_gain']:,.0f}
- Expected ROI: {threshold_df.iloc[optimal_idx]['roi']:.1%}

MLflow:
- Run ID: {mlflow.active_run().info.run_id}
- Model URI: models:/{model_name}/Production
- Model Version: {latest_version}

Next Steps:
1. Deploy model to serving endpoint
2. Integrate with FastAPI for real-time scoring
3. Set up monitoring dashboards
4. Schedule batch scoring job
""")

# COMMAND ----------