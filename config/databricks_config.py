"""
Databricks Configuration and Connection Manager
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import SchemaInfo, TableInfo
from pyspark.sql import SparkSession
import mlflow
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class DatabricksConfig:
    """Databricks configuration and connection manager"""
    
    def __init__(self):
        self.host = os.getenv("DATABRICKS_HOST", "https://adb-<workspace-id>.azuredatabricks.net")
        self.token = os.getenv("DATABRICKS_TOKEN")
        self.catalog = "insurance_copilot"
        self.schema = "policy_data"
        self.ml_schema = "ml_models"
        
        # MLflow configuration
        self.mlflow_tracking_uri = f"databricks://{self.host}"
        self.mlflow_experiment_name = "/Shared/insurance-copilot-experiments"
        
        # Storage configuration
        self.storage_account = os.getenv("AZURE_STORAGE_ACCOUNT", "insurancecopilot")
        self.container_name = "policy-documents"
        
    def get_spark_session(self) -> SparkSession:
        """Get or create Spark session with Databricks connection"""
        return SparkSession.builder \
            .appName("InsuranceCopilot") \
            .config("spark.databricks.service.server.enabled", "true") \
            .config("spark.databricks.service.token", self.token) \
            .config("spark.databricks.service.address", self.host) \
            .config("spark.sql.catalog.spark_catalog", "hive_metastore") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.implementation", "io.delta.sql.catalog.DeltaCatalog") \
            .enableHiveSupport() \
            .getOrCreate()
    
    def get_workspace_client(self) -> WorkspaceClient:
        """Get Databricks workspace client"""
        return WorkspaceClient(
            host=self.host,
            token=self.token
        )
    
    def setup_mlflow(self):
        """Configure MLflow for Databricks"""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)
        return mlflow
    
    def create_catalog_schema(self):
        """Create catalog and schemas if they don't exist"""
        spark = self.get_spark_session()
        
        # Create catalog
        spark.sql(f"CREATE CATALOG IF NOT EXISTS {self.catalog}")
        
        # Create schemas
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.catalog}.{self.schema}")
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.catalog}.{self.ml_schema}")
        
        print(f"✓ Created catalog '{self.catalog}' with schemas '{self.schema}' and '{self.ml_schema}'")
    
    def get_table_path(self, table_name: str) -> str:
        """Get full table path"""
        return f"{self.catalog}.{self.schema}.{table_name}"
    
    def get_model_path(self, model_name: str) -> str:
        """Get full model path"""
        return f"{self.catalog}.{self.ml_schema}.{model_name}"

# Global instance
databricks_config = DatabricksConfig()