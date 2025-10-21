
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
