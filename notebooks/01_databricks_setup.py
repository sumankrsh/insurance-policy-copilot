# Databricks notebook source
# MAGIC %md
# MAGIC # Insurance Copilot - Databricks Setup
# MAGIC This notebook creates the necessary tables and loads synthetic data

# COMMAND ----------

# MAGIC %pip install databricks-sdk mlflow delta-spark faker pandas numpy

# COMMAND ----------

import sys
import os
sys.path.append('/Workspace/ai-policy-copilot')

from config.databricks_config import databricks_config
from data.synthetic_data_generator import InsuranceDataGenerator
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from delta.tables import DeltaTable
import mlflow
import pandas as pd

# COMMAND ----------

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Insurance Copilot Setup") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

print("Spark session initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Catalog and Schemas

# COMMAND ----------

catalog = "insurance_copilot"
data_schema = "policy_data"
ml_schema = "ml_models"
feature_schema = "features"

# Create catalog
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
print(f"✓ Created catalog: {catalog}")

# Create schemas
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{data_schema}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{ml_schema}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{feature_schema}")
print(f"✓ Created schemas: {data_schema}, {ml_schema}, {feature_schema}")

# Set default catalog and schema
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {data_schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Synthetic Data

# COMMAND ----------

# Generate synthetic data
generator = InsuranceDataGenerator(num_customers=50000)
all_data = generator.generate_all_data()

print("Generated datasets:")
for table_name, df in all_data.items():
    print(f"  - {table_name}: {len(df)} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Delta Tables

# COMMAND ----------

# Define table schemas
customer_schema = StructType([
    StructField("customer_id", StringType(), False),
    StructField("first_name", StringType(), True),
    StructField("last_name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("phone", StringType(), True),
    StructField("date_of_birth", DateType(), True),
    StructField("gender", StringType(), True),
    StructField("address", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("zip_code", StringType(), True),
    StructField("annual_income", LongType(), True),
    StructField("credit_score", IntegerType(), True),
    StructField("customer_since", DateType(), True),
    StructField("customer_lifetime_value", DoubleType(), True),
    StructField("preferred_channel", StringType(), True),
    StructField("risk_score", DoubleType(), True),
    StructField("is_active", BooleanType(), True),
    StructField("last_updated", TimestampType(), True),
    StructField("first_name_hash", StringType(), True),
    StructField("last_name_hash", StringType(), True)
])

policy_schema = StructType([
    StructField("policy_id", StringType(), False),
    StructField("customer_id", StringType(), False),
    StructField("policy_type", StringType(), True),
    StructField("policy_status", StringType(), True),
    StructField("start_date", DateType(), True),
    StructField("end_date", DateType(), True),
    StructField("term_years", IntegerType(), True),
    StructField("premium_amount", DoubleType(), True),
    StructField("premium_frequency", StringType(), True),
    StructField("sum_assured", DoubleType(), True),
    StructField("current_fund_value", DoubleType(), True),
    StructField("fund_value_change_3m", DoubleType(), True),
    StructField("tenure_months", IntegerType(), True),
    StructField("surrender_charge_pct", DoubleType(), True),
    StructField("lock_in_period_months", IntegerType(), True),
    StructField("payment_gaps", IntegerType(), True),
    StructField("last_premium_paid", DateType(), True),
    StructField("agent_code", StringType(), True),
    StructField("rider_count", IntegerType(), True),
    StructField("created_at", DateType(), True),
    StructField("updated_at", TimestampType(), True)
])

claim_schema = StructType([
    StructField("claim_id", StringType(), False),
    StructField("policy_id", StringType(), False),
    StructField("customer_id", StringType(), False),
    StructField("claim_type", StringType(), True),
    StructField("claim_amount", DoubleType(), True),
    StructField("claim_date", DateType(), True),
    StructField("claim_status", StringType(), True),
    StructField("approved_amount", DoubleType(), True),
    StructField("rejection_reason", StringType(), True),
    StructField("processing_days", IntegerType(), True),
    StructField("documents_submitted", BooleanType(), True),
    StructField("investigation_required", BooleanType(), True),
    StructField("claim_notes", StringType(), True),
    StructField("created_at", DateType(), True),
    StructField("updated_at", TimestampType(), True)
])

interaction_schema = StructType([
    StructField("interaction_id", StringType(), False),
    StructField("customer_id", StringType(), False),
    StructField("interaction_type", StringType(), True),
    StructField("interaction_date", DateType(), True),
    StructField("interaction_reason", StringType(), True),
    StructField("sentiment_score", DoubleType(), True),
    StructField("resolution_status", StringType(), True),
    StructField("agent_id", StringType(), True),
    StructField("call_duration_seconds", IntegerType(), True),
    StructField("notes", StringType(), True),
    StructField("created_at", TimestampType(), True)
])

surrender_label_schema = StructType([
    StructField("policy_id", StringType(), False),
    StructField("customer_id", StringType(), False),
    StructField("tenure_months", IntegerType(), True),
    StructField("premium_amount", DoubleType(), True),
    StructField("payment_gaps", IntegerType(), True),
    StructField("surrender_charge_pct", DoubleType(), True),
    StructField("fund_value_change_3m", DoubleType(), True),
    StructField("avg_sentiment", DoubleType(), True),
    StructField("interaction_count_90d", IntegerType(), True),
    StructField("surrender_queries_90d", IntegerType(), True),
    StructField("complaints_90d", IntegerType(), True),
    StructField("agent_activity_30d", IntegerType(), True),
    StructField("surrender_probability", DoubleType(), True),
    StructField("surrendered_in_next_90d", IntegerType(), True),
    StructField("model_version", StringType(), True),
    StructField("created_at", TimestampType(), True)
])

# COMMAND ----------

# Function to create and load Delta table
def create_delta_table(df_pandas, table_name, schema, partition_cols=None):
    """Create a Delta table from pandas DataFrame"""
    
    # Convert pandas to Spark DataFrame
    df_spark = spark.createDataFrame(df_pandas, schema=schema)
    
    # Table path
    table_path = f"{catalog}.{data_schema}.{table_name}"
    
    # Drop table if exists
    spark.sql(f"DROP TABLE IF EXISTS {table_path}")
    
    # Write as Delta table with partitioning if specified
    writer = df_spark.write.mode("overwrite").format("delta")
    
    if partition_cols:
        writer = writer.partitionBy(partition_cols)
    
    # Save table
    writer.saveAsTable(table_path)
    
    # Add table properties and comments
    spark.sql(f"""
        ALTER TABLE {table_path}
        SET TBLPROPERTIES (
            'delta.minReaderVersion' = '2',
            'delta.minWriterVersion' = '5',
            'delta.columnMapping.mode' = 'name',
            'delta.autoOptimize.optimizeWrite' = 'true',
            'delta.autoOptimize.autoCompact' = 'true'
        )
    """)
    
    # Optimize table
    spark.sql(f"OPTIMIZE {table_path}")
    
    # Analyze table for statistics
    spark.sql(f"ANALYZE TABLE {table_path} COMPUTE STATISTICS")
    
    print(f"✓ Created Delta table: {table_path}")
    
    # Show sample data
    spark.sql(f"SELECT * FROM {table_path} LIMIT 5").show(truncate=False)
    
    return table_path

# COMMAND ----------

# Create all Delta tables
tables_created = {}

# Create customers table
tables_created['customers'] = create_delta_table(
    all_data['customers'],
    'customers',
    customer_schema,
    partition_cols=['state']
)

# Create policies table  
tables_created['policies'] = create_delta_table(
    all_data['policies'],
    'policies',
    policy_schema,
    partition_cols=['policy_type', 'policy_status']
)

# Create claims table
tables_created['claims'] = create_delta_table(
    all_data['claims'],
    'claims',
    claim_schema,
    partition_cols=['claim_status']
)

# Create interactions table
tables_created['interactions'] = create_delta_table(
    all_data['interactions'],
    'interactions',
    interaction_schema,
    partition_cols=['interaction_type']
)

# Create surrender labels table
tables_created['surrender_labels'] = create_delta_table(
    all_data['surrender_labels'],
    'surrender_labels',
    surrender_label_schema
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Feature Store Tables

# COMMAND ----------

# Create customer features table for Feature Store
customer_features_query = f"""
CREATE OR REPLACE TABLE {catalog}.{feature_schema}.customer_features AS
SELECT 
    c.customer_id,
    c.annual_income,
    c.credit_score,
    c.risk_score,
    DATEDIFF(CURRENT_DATE, c.customer_since) as customer_tenure_days,
    c.customer_lifetime_value,
    COUNT(DISTINCT p.policy_id) as total_policies,
    SUM(p.premium_amount) as total_premium,
    AVG(p.payment_gaps) as avg_payment_gaps,
    MAX(p.payment_gaps) as max_payment_gaps,
    COUNT(DISTINCT cl.claim_id) as total_claims,
    SUM(cl.approved_amount) as total_claims_approved,
    AVG(i.sentiment_score) as avg_sentiment,
    COUNT(i.interaction_id) as total_interactions,
    current_timestamp() as feature_timestamp
FROM {catalog}.{data_schema}.customers c
LEFT JOIN {catalog}.{data_schema}.policies p ON c.customer_id = p.customer_id
LEFT JOIN {catalog}.{data_schema}.claims cl ON c.customer_id = cl.customer_id
LEFT JOIN {catalog}.{data_schema}.interactions i ON c.customer_id = i.customer_id
GROUP BY c.customer_id, c.annual_income, c.credit_score, c.risk_score, 
         c.customer_since, c.customer_lifetime_value
"""

spark.sql(customer_features_query)
print(f"✓ Created feature table: {catalog}.{feature_schema}.customer_features")

# COMMAND ----------

# Create policy features table
policy_features_query = f"""
CREATE OR REPLACE TABLE {catalog}.{feature_schema}.policy_features AS
SELECT 
    p.policy_id,
    p.customer_id,
    p.tenure_months,
    p.premium_amount,
    p.payment_gaps,
    p.surrender_charge_pct,
    p.fund_value_change_3m,
    p.current_fund_value / p.sum_assured as fund_to_sum_ratio,
    CASE WHEN p.tenure_months < p.lock_in_period_months THEN 1 ELSE 0 END as in_lock_period,
    DATEDIFF(CURRENT_DATE, p.last_premium_paid) as days_since_last_payment,
    current_timestamp() as feature_timestamp
FROM {catalog}.{data_schema}.policies p
WHERE p.policy_status = 'Active'
"""

spark.sql(policy_features_query)
print(f"✓ Created feature table: {catalog}.{feature_schema}.policy_features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Views for Analytics

# COMMAND ----------

# Create analytical views
views = [
    f"""
    CREATE OR REPLACE VIEW {catalog}.{data_schema}.v_customer_360 AS
    SELECT 
        c.*,
        COUNT(DISTINCT p.policy_id) as active_policies,
        SUM(p.premium_amount) as total_monthly_premium,
        AVG(p.payment_gaps) as avg_payment_delays,
        MAX(cl.claim_date) as last_claim_date,
        AVG(i.sentiment_score) as recent_sentiment
    FROM {catalog}.{data_schema}.customers c
    LEFT JOIN {catalog}.{data_schema}.policies p 
        ON c.customer_id = p.customer_id AND p.policy_status = 'Active'
    LEFT JOIN {catalog}.{data_schema}.claims cl 
        ON c.customer_id = cl.customer_id
    LEFT JOIN {catalog}.{data_schema}.interactions i 
        ON c.customer_id = i.customer_id 
        AND i.interaction_date > DATE_SUB(CURRENT_DATE, 90)
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.phone,
             c.date_of_birth, c.gender, c.address, c.city, c.state, c.zip_code,
             c.annual_income, c.credit_score, c.customer_since, 
             c.customer_lifetime_value, c.preferred_channel, c.risk_score,
             c.is_active, c.last_updated, c.first_name_hash, c.last_name_hash
    """,
    
    f"""
    CREATE OR REPLACE VIEW {catalog}.{data_schema}.v_surrender_risk AS
    SELECT 
        sl.*,
        p.policy_type,
        p.premium_frequency,
        p.current_fund_value,
        c.risk_score as customer_risk_score,
        c.customer_lifetime_value
    FROM {catalog}.{data_schema}.surrender_labels sl
    JOIN {catalog}.{data_schema}.policies p ON sl.policy_id = p.policy_id
    JOIN {catalog}.{data_schema}.customers c ON sl.customer_id = c.customer_id
    WHERE p.policy_status = 'Active'
    """,
    
    f"""
    CREATE OR REPLACE VIEW {catalog}.{data_schema}.v_claims_analytics AS
    SELECT 
        cl.*,
        p.policy_type,
        p.sum_assured,
        p.tenure_months as policy_tenure_at_claim,
        c.annual_income,
        c.risk_score,
        cl.approved_amount / cl.claim_amount as approval_ratio
    FROM {catalog}.{data_schema}.claims cl
    JOIN {catalog}.{data_schema}.policies p ON cl.policy_id = p.policy_id
    JOIN {catalog}.{data_schema}.customers c ON cl.customer_id = c.customer_id
    """
]

for view_sql in views:
    spark.sql(view_sql)
    
print("✓ Created analytical views")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table Statistics

# COMMAND ----------

# Show table statistics
print("\n📊 Table Statistics:")
print("-" * 50)

for table_name in ['customers', 'policies', 'claims', 'interactions', 'surrender_labels']:
    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {catalog}.{data_schema}.{table_name}").collect()[0]['cnt']
    print(f"{table_name}: {count:,} rows")

print("\n📈 Policy Distribution:")
spark.sql(f"""
    SELECT policy_type, policy_status, COUNT(*) as count
    FROM {catalog}.{data_schema}.policies
    GROUP BY policy_type, policy_status
    ORDER BY count DESC
""").show()

print("\n🎯 Surrender Risk Distribution:")
spark.sql(f"""
    SELECT 
        surrendered_in_next_90d,
        COUNT(*) as count,
        AVG(surrender_probability) as avg_probability
    FROM {catalog}.{data_schema}.surrender_labels
    GROUP BY surrendered_in_next_90d
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup MLflow Experiment

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI
mlflow.set_tracking_uri("databricks")

# Create experiment
experiment_name = "/Shared/insurance-copilot-surrender-model"
mlflow.set_experiment(experiment_name)

# Get experiment details
experiment = mlflow.get_experiment_by_name(experiment_name)
print(f"✓ MLflow experiment created: {experiment_name}")
print(f"  Experiment ID: {experiment.experiment_id}")
print(f"  Artifact Location: {experiment.artifact_location}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Model Registry

# COMMAND ----------

# Register placeholder for model
model_name = "surrender_propensity_model"
client = MlflowClient()

try:
    client.create_registered_model(
        name=model_name,
        description="Surrender propensity model for insurance policies",
        tags={
            "task": "binary_classification",
            "framework": "xgboost",
            "business_unit": "insurance",
            "data_source": f"{catalog}.{data_schema}"
        }
    )
    print(f"✓ Created model in registry: {model_name}")
except:
    print(f"Model {model_name} already exists in registry")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("""
✅ Databricks Setup Complete!

Created:
- Catalog: insurance_copilot
- Schemas: policy_data, ml_models, features
- Tables:
  * customers (50,000 rows)
  * policies (120,000 rows)  
  * claims (15,000 rows)
  * interactions (250,000 rows)
  * surrender_labels (120,000 rows)
  * customer_features
  * policy_features
- Views: v_customer_360, v_surrender_risk, v_claims_analytics
- MLflow Experiment: /Shared/insurance-copilot-surrender-model
- Model Registry: surrender_propensity_model

Next steps:
1. Run 02_train_surrender_model notebook to train the ML model
2. Run 03_create_policy_documents notebook to generate policy PDFs
3. Run 04_setup_rag_pipeline notebook to index documents
4. Start the FastAPI server for the chatbot
""")

# COMMAND ----------