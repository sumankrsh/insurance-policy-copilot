"""
Insurance-specific tools for policy operations and ML model inference
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import mlflow
from pyspark.sql import SparkSession
import logging
import asyncio

from config.databricks_config import databricks_config

logger = logging.getLogger(__name__)

class InsuranceTools:
    """Tools for insurance operations"""
    
    def __init__(self):
        self.spark = None
        self.model = None
        self.model_name = "surrender_propensity_model"
        self.catalog = "insurance_copilot"
        self.schema = "policy_data"
        self.use_mock_data = os.getenv("USE_MOCK_DATA", "true").lower() == "true"
        self._initialize()
        
    def _initialize(self):
        """Initialize Spark and ML model"""
        if self.use_mock_data:
            logger.info("Using mock data mode")
            return
            
        try:
            # Initialize Spark session
            self.spark = databricks_config.get_spark_session()
            logger.info("Spark session initialized")
            
            # Load ML model
            mlflow.set_tracking_uri("databricks")
            self.model = mlflow.sklearn.load_model(f"models:/{self.model_name}/Production")
            logger.info(f"Loaded model: {self.model_name}")
            self.use_mock_data = False
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            # Use mock data if Databricks not available
            self.use_mock_data = True
            logger.info("Falling back to mock data mode")
            
    async def get_surrender_propensity(self, customer_id: str) -> Dict[str, Any]:
        """Get surrender propensity score for a customer"""
        
        try:
            if self.use_mock_data:
                return self._mock_surrender_propensity(customer_id)
                
            # Query features from Databricks
            query = f"""
            SELECT 
                sl.*,
                pf.*,
                cf.*
            FROM {self.catalog}.{self.schema}.surrender_labels sl
            JOIN {self.catalog}.features.policy_features pf 
                ON sl.policy_id = pf.policy_id
            JOIN {self.catalog}.features.customer_features cf 
                ON sl.customer_id = cf.customer_id
            WHERE sl.customer_id = '{customer_id}'
            LIMIT 1
            """
            
            df = self.spark.sql(query).toPandas()
            
            if df.empty:
                return {
                    "error": "Customer not found",
                    "customer_id": customer_id
                }
            
            # Prepare features
            feature_cols = [
                'tenure_months', 'premium_amount', 'payment_gaps', 'surrender_charge_pct',
                'fund_value_change_3m', 'avg_sentiment', 'interaction_count_90d',
                'surrender_queries_90d', 'complaints_90d', 'agent_activity_30d',
                'fund_to_sum_ratio', 'in_lock_period', 'days_since_last_payment',
                'annual_income', 'credit_score', 'risk_score', 'customer_tenure_days',
                'total_policies', 'hist_avg_payment_gaps'
            ]
            
            # Add derived features
            df['premium_to_income_ratio'] = df['premium_amount'] / (df['annual_income'] / 12)
            df['risk_adjusted_tenure'] = df['tenure_months'] * (1 - df['risk_score'])
            df['payment_reliability'] = 1 / (1 + df['payment_gaps'])
            df['sentiment_interaction'] = df['avg_sentiment'] * df['interaction_count_90d']
            df['financial_stress'] = (df['payment_gaps'] > 0).astype(int) * df['fund_value_change_3m']
            
            feature_cols.extend([
                'premium_to_income_ratio', 'risk_adjusted_tenure', 
                'payment_reliability', 'sentiment_interaction', 'financial_stress'
            ])
            
            X = df[feature_cols].fillna(0)
            
            # Get prediction
            probability = self.model.predict_proba(X)[0, 1]
            
            # Get feature importance for this prediction
            feature_importance = self._get_feature_importance(X)
            top_factors = feature_importance[:5]
            
            # Determine action
            if probability >= 0.7:
                action = "escalate_to_retention_desk"
                recommendation = "Immediate retention intervention required"
            elif probability >= 0.3:
                action = "send_personalized_education"
                recommendation = "Send surrender charge information and benefits summary"
            else:
                action = "normal_service"
                recommendation = "Continue regular service"
            
            return {
                "customer_id": customer_id,
                "probability": float(probability),
                "risk_level": "high" if probability >= 0.7 else "medium" if probability >= 0.3 else "low",
                "action": action,
                "recommendation": recommendation,
                "top_factors": [f[0] for f in top_factors],
                "model_version": "v1.0",
                "scored_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Surrender propensity error: {e}")
            return self._mock_surrender_propensity(customer_id)
            
    def _mock_surrender_propensity(self, customer_id: str) -> Dict[str, Any]:
        """Mock surrender propensity for demo"""
        
        # Generate consistent mock probability based on customer ID
        hash_val = hash(customer_id) % 100
        probability = hash_val / 100
        
        factors = [
            "payment_gaps", "fund_value_change", "sentiment_score",
            "surrender_queries", "tenure_months"
        ]
        
        if probability >= 0.7:
            action = "escalate_to_retention_desk"
            recommendation = "Immediate retention intervention required"
            risk_level = "high"
        elif probability >= 0.3:
            action = "send_personalized_education"
            recommendation = "Send surrender charge information"
            risk_level = "medium"
        else:
            action = "normal_service"
            recommendation = "Continue regular service"
            risk_level = "low"
            
        return {
            "customer_id": customer_id,
            "probability": probability,
            "risk_level": risk_level,
            "action": action,
            "recommendation": recommendation,
            "top_factors": factors[:3],
            "model_version": "v1.0-mock",
            "scored_at": datetime.now().isoformat()
        }
        
    async def get_policy_summary(self, policy_id: str) -> Dict[str, Any]:
        """Get policy summary"""
        
        try:
            if self.use_mock_data:
                return self._mock_policy_summary(policy_id)
                
            query = f"""
            SELECT 
                p.*,
                c.first_name,
                c.last_name,
                c.email
            FROM {self.catalog}.{self.schema}.policies p
            JOIN {self.catalog}.{self.schema}.customers c
                ON p.customer_id = c.customer_id
            WHERE p.policy_id = '{policy_id}'
            """
            
            df = self.spark.sql(query).toPandas()
            
            if df.empty:
                return {"error": "Policy not found", "policy_id": policy_id}
                
            policy = df.iloc[0].to_dict()
            
            # Calculate derived fields
            tenure_years = (datetime.now() - pd.to_datetime(policy['start_date'])).days / 365
            remaining_term = policy['term_years'] - tenure_years
            
            return {
                "policy_id": policy_id,
                "customer_name": f"{policy['first_name']} {policy['last_name']}",
                "policy_type": policy['policy_type'],
                "status": policy['policy_status'],
                "start_date": policy['start_date'].isoformat(),
                "end_date": policy['end_date'].isoformat(),
                "term_years": policy['term_years'],
                "tenure_years": round(tenure_years, 1),
                "remaining_term": max(0, round(remaining_term, 1)),
                "premium_amount": policy['premium_amount'],
                "premium_frequency": policy['premium_frequency'],
                "sum_assured": policy['sum_assured'],
                "current_fund_value": policy['current_fund_value'],
                "surrender_charge_pct": policy['surrender_charge_pct'],
                "lock_in_months_remaining": max(0, policy['lock_in_period_months'] - policy['tenure_months']),
                "payment_gaps": policy['payment_gaps'],
                "last_premium_paid": policy['last_premium_paid'].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Policy summary error: {e}")
            return self._mock_policy_summary(policy_id)
            
    def _mock_policy_summary(self, policy_id: str) -> Dict[str, Any]:
        """Mock policy summary for demo"""
        
        return {
            "policy_id": policy_id,
            "customer_name": "John Doe",
            "policy_type": "Term Life",
            "status": "Active",
            "start_date": "2020-01-15",
            "end_date": "2040-01-15",
            "term_years": 20,
            "tenure_years": 4.8,
            "remaining_term": 15.2,
            "premium_amount": 5000.00,
            "premium_frequency": "Monthly",
            "sum_assured": 1000000.00,
            "current_fund_value": 240000.00,
            "surrender_charge_pct": 2.5,
            "lock_in_months_remaining": 8,
            "payment_gaps": 0,
            "last_premium_paid": datetime.now().date().isoformat()
        }
        
    async def calculate_surrender_value(self, policy_id: str) -> Dict[str, Any]:
        """Calculate surrender value for a policy"""
        
        try:
            # Get policy details
            policy = await self.get_policy_summary(policy_id)
            
            if "error" in policy:
                return policy
                
            # Calculate surrender value
            fund_value = policy['current_fund_value']
            surrender_charge_pct = policy['surrender_charge_pct']
            
            surrender_charge = fund_value * (surrender_charge_pct / 100)
            surrender_value = fund_value - surrender_charge
            
            # Check lock-in period
            if policy['lock_in_months_remaining'] > 0:
                can_surrender = False
                message = f"Policy is in lock-in period. {policy['lock_in_months_remaining']} months remaining."
            else:
                can_surrender = True
                message = "Policy can be surrendered without lock-in restrictions."
                
            return {
                "policy_id": policy_id,
                "current_fund_value": fund_value,
                "surrender_charge_pct": surrender_charge_pct,
                "surrender_charge_amount": round(surrender_charge, 2),
                "net_surrender_value": round(surrender_value, 2),
                "can_surrender": can_surrender,
                "lock_in_months_remaining": policy['lock_in_months_remaining'],
                "message": message,
                "calculated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Surrender calculation error: {e}")
            return {
                "error": "Could not calculate surrender value",
                "policy_id": policy_id
            }
            
    async def get_customer_policies(self, customer_id: str) -> Dict[str, Any]:
        """Get all policies for a customer"""
        
        try:
            if self.use_mock_data:
                return self._mock_customer_policies(customer_id)
                
            query = f"""
            SELECT 
                policy_id,
                policy_type,
                policy_status,
                premium_amount,
                sum_assured,
                start_date
            FROM {self.catalog}.{self.schema}.policies
            WHERE customer_id = '{customer_id}'
            """
            
            df = self.spark.sql(query).toPandas()
            
            if df.empty:
                return {
                    "customer_id": customer_id,
                    "policies": [],
                    "total_policies": 0
                }
                
            policies = df.to_dict('records')
            
            # Convert dates to strings
            for policy in policies:
                if 'start_date' in policy:
                    policy['start_date'] = policy['start_date'].isoformat()
                    
            return {
                "customer_id": customer_id,
                "policies": policies,
                "total_policies": len(policies),
                "total_premium": float(df['premium_amount'].sum()),
                "total_sum_assured": float(df['sum_assured'].sum())
            }
            
        except Exception as e:
            logger.error(f"Customer policies error: {e}")
            return self._mock_customer_policies(customer_id)
            
    def _mock_customer_policies(self, customer_id: str) -> Dict[str, Any]:
        """Mock customer policies for demo"""
        
        policies = [
            {
                "policy_id": "P00000001",
                "policy_type": "Term Life",
                "policy_status": "Active",
                "premium_amount": 5000.00,
                "sum_assured": 1000000.00,
                "start_date": "2020-01-15"
            },
            {
                "policy_id": "P00000002",
                "policy_type": "Health",
                "policy_status": "Active",
                "premium_amount": 2000.00,
                "sum_assured": 500000.00,
                "start_date": "2021-06-01"
            }
        ]
        
        return {
            "customer_id": customer_id,
            "policies": policies,
            "total_policies": 2,
            "total_premium": 7000.00,
            "total_sum_assured": 1500000.00
        }
        
    def _get_feature_importance(self, X: pd.DataFrame) -> List[tuple]:
        """Get feature importance for a prediction"""
        
        # For XGBoost, get feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = X.columns.tolist()
            
            # Sort by importance
            feature_importance = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
            
            return feature_importance
        else:
            # Return default if model doesn't have feature_importances
            return [
                ("payment_gaps", 0.25),
                ("fund_value_change_3m", 0.20),
                ("surrender_queries_90d", 0.15),
                ("sentiment_score", 0.10),
                ("tenure_months", 0.08)
            ]

# Tool definitions for LLM
TOOL_DEFINITIONS = [
    {
        "name": "get_surrender_propensity",
        "description": "Calculate surrender propensity score for a customer",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The customer ID (format: C######)"
                }
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "get_policy_summary",
        "description": "Get detailed summary of a policy",
        "parameters": {
            "type": "object",
            "properties": {
                "policy_id": {
                    "type": "string",
                    "description": "The policy ID (format: P########)"
                }
            },
            "required": ["policy_id"]
        }
    },
    {
        "name": "calculate_surrender_value",
        "description": "Calculate the surrender value of a policy",
        "parameters": {
            "type": "object",
            "properties": {
                "policy_id": {
                    "type": "string",
                    "description": "The policy ID (format: P########)"
                }
            },
            "required": ["policy_id"]
        }
    },
    {
        "name": "get_customer_policies",
        "description": "Get all policies for a customer",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The customer ID (format: C######)"
                }
            },
            "required": ["customer_id"]
        }
    }
]

if __name__ == "__main__":
    # Test tools
    import asyncio
    
    tools = InsuranceTools()
    
    async def test_tools():
        print("Testing Insurance Tools")
        print("=" * 60)
        
        # Test surrender propensity
        result = await tools.get_surrender_propensity("C000001")
        print("\nSurrender Propensity:")
        print(result)
        
        # Test policy summary
        result = await tools.get_policy_summary("P00000001")
        print("\nPolicy Summary:")
        print(result)
        
        # Test surrender value
        result = await tools.calculate_surrender_value("P00000001")
        print("\nSurrender Value:")
        print(result)
        
        # Test customer policies
        result = await tools.get_customer_policies("C000001")
        print("\nCustomer Policies:")
        print(result)
        
    asyncio.run(test_tools())