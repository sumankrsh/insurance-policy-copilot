"""
Databricks Integration Script
Upload notebooks and data to Databricks workspace
"""

import os
import json
from datetime import datetime

print("""
=====================================
🛡️ Insurance Copilot - Databricks Integration
=====================================

This script will help you set up the Insurance Copilot in your Databricks workspace.

IMPORTANT: You'll need to:
1. Log in to Databricks Community Edition: https://community.cloud.databricks.com
   Email: smn.mukherjee@gmail.com

2. Create a new cluster:
   - Go to Compute → Create Cluster
   - Name: insurance-copilot-cluster
   - Runtime: 13.3 LTS (or latest)
   - Click "Create Cluster"

3. Upload the notebooks:
   - Go to Workspace → Users → Your Email
   - Create folder: insurance-copilot
   - Upload these notebooks from the 'notebooks' folder:
     • 01_databricks_setup.py
     • 02_train_surrender_model.py

4. Run the notebooks in order:
   - First: 01_databricks_setup.py (creates tables and schemas)
   - Second: 02_train_surrender_model.py (trains ML model)

5. Get your access token:
   - Click your email in top right → User Settings
   - Go to Developer → Access Tokens
   - Generate New Token → Copy it

6. Update .env file with your token:
   DATABRICKS_HOST=https://community.cloud.databricks.com
   DATABRICKS_TOKEN=<your-token-here>
""")

# Create SQL commands for Databricks
sql_commands = """
-- Run these commands in a Databricks SQL notebook

-- 1. Create Catalog (if not using default)
CREATE CATALOG IF NOT EXISTS insurance_copilot;
USE CATALOG insurance_copilot;

-- 2. Create Schemas
CREATE SCHEMA IF NOT EXISTS policy_data;
CREATE SCHEMA IF NOT EXISTS ml_models;
CREATE SCHEMA IF NOT EXISTS features;

-- 3. Create Tables
USE SCHEMA policy_data;

-- Customers Table
CREATE TABLE IF NOT EXISTS customers (
    customer_id STRING PRIMARY KEY,
    first_name STRING,
    last_name STRING,
    email STRING,
    phone STRING,
    date_of_birth DATE,
    gender STRING,
    address STRING,
    city STRING,
    state STRING,
    zip_code STRING,
    annual_income BIGINT,
    credit_score INT,
    customer_since DATE,
    customer_lifetime_value DOUBLE,
    preferred_channel STRING,
    risk_score DOUBLE,
    is_active BOOLEAN,
    last_updated TIMESTAMP
) USING DELTA;

-- Policies Table
CREATE TABLE IF NOT EXISTS policies (
    policy_id STRING PRIMARY KEY,
    customer_id STRING,
    policy_type STRING,
    policy_status STRING,
    start_date DATE,
    end_date DATE,
    term_years INT,
    premium_amount DOUBLE,
    premium_frequency STRING,
    sum_assured DOUBLE,
    current_fund_value DOUBLE,
    fund_value_change_3m DOUBLE,
    tenure_months INT,
    surrender_charge_pct DOUBLE,
    lock_in_period_months INT,
    payment_gaps INT,
    last_premium_paid DATE,
    agent_code STRING,
    rider_count INT,
    created_at DATE,
    updated_at TIMESTAMP
) USING DELTA;

-- Claims Table
CREATE TABLE IF NOT EXISTS claims (
    claim_id STRING PRIMARY KEY,
    policy_id STRING,
    customer_id STRING,
    claim_type STRING,
    claim_amount DOUBLE,
    claim_date DATE,
    claim_status STRING,
    approved_amount DOUBLE,
    rejection_reason STRING,
    processing_days INT,
    documents_submitted BOOLEAN,
    investigation_required BOOLEAN,
    claim_notes STRING,
    created_at DATE,
    updated_at TIMESTAMP
) USING DELTA;

-- Interactions Table
CREATE TABLE IF NOT EXISTS interactions (
    interaction_id STRING PRIMARY KEY,
    customer_id STRING,
    interaction_type STRING,
    interaction_date DATE,
    interaction_reason STRING,
    sentiment_score DOUBLE,
    resolution_status STRING,
    agent_id STRING,
    call_duration_seconds INT,
    notes STRING,
    created_at TIMESTAMP
) USING DELTA;

-- Surrender Labels Table
CREATE TABLE IF NOT EXISTS surrender_labels (
    policy_id STRING PRIMARY KEY,
    customer_id STRING,
    tenure_months INT,
    premium_amount DOUBLE,
    payment_gaps INT,
    surrender_charge_pct DOUBLE,
    fund_value_change_3m DOUBLE,
    avg_sentiment DOUBLE,
    interaction_count_90d INT,
    surrender_queries_90d INT,
    complaints_90d INT,
    agent_activity_30d INT,
    surrender_probability DOUBLE,
    surrendered_in_next_90d INT,
    model_version STRING,
    created_at TIMESTAMP
) USING DELTA;

-- 4. Load Sample Data (Python code to run in notebook)
"""

# Save SQL commands
with open("databricks_setup.sql", "w") as f:
    f.write(sql_commands)

# Create data upload script
upload_script = '''
# Databricks notebook source
# MAGIC %md
# MAGIC # Upload Data to Delta Tables
# MAGIC Run this after creating tables

# COMMAND ----------

import pandas as pd
from pyspark.sql import SparkSession
import os

# COMMAND ----------

# Read parquet files from local
base_path = "/Workspace/insurance-copilot/data"

# Load data
customers_df = spark.read.parquet(f"{base_path}/customers.parquet")
policies_df = spark.read.parquet(f"{base_path}/policies.parquet")
claims_df = spark.read.parquet(f"{base_path}/claims.parquet")
interactions_df = spark.read.parquet(f"{base_path}/interactions.parquet")
surrender_labels_df = spark.read.parquet(f"{base_path}/surrender_labels.parquet")

# COMMAND ----------

# Write to Delta tables
catalog = "insurance_copilot"
schema = "policy_data"

customers_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.customers")
policies_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.policies")
claims_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.claims")
interactions_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.interactions")
surrender_labels_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.surrender_labels")

print("✓ Data uploaded to Delta tables")

# COMMAND ----------

# Verify counts
for table in ["customers", "policies", "claims", "interactions", "surrender_labels"]:
    count = spark.sql(f"SELECT COUNT(*) FROM {catalog}.{schema}.{table}").collect()[0][0]
    print(f"{table}: {count:,} rows")
'''

with open("notebooks/03_upload_data.py", "w") as f:
    f.write(upload_script)

print("""
Files created:
- databricks_setup.sql - SQL commands to create tables
- notebooks/03_upload_data.py - Python script to upload data

Next steps:
1. Upload the parquet files from 'data/' folder to Databricks
2. Run the SQL commands in databricks_setup.sql
3. Run the notebook 03_upload_data.py to load data

Your services are now running at:
=====================================
✅ Chat UI: http://localhost:8501
✅ API Docs: http://localhost:8001/docs
✅ Health Check: http://localhost:8001/health
=====================================

The application is running in MOCK DATA mode.
To connect to real Databricks data:
1. Complete the Databricks setup above
2. Update .env with your Databricks token
3. Set USE_MOCK_DATA=false in .env
4. Restart the services
""")