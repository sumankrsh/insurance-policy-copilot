"""
Synthetic Insurance Data Generator for Databricks
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import string
from typing import Dict, List, Tuple
from faker import Faker
import hashlib

fake = Faker()
np.random.seed(42)
random.seed(42)

class InsuranceDataGenerator:
    """Generate synthetic insurance data"""
    
    def __init__(self, num_customers: int = 50000):
        self.num_customers = num_customers
        self.num_policies = int(num_customers * 2.4)  # Avg 2.4 policies per customer
        self.num_claims = int(num_customers * 0.3)  # 30% have claims
        self.num_interactions = num_customers * 5  # 5 interactions per customer avg
        
        self.policy_types = ['Term Life', 'Whole Life', 'ULIP', 'Endowment', 'Health', 'Motor']
        self.interaction_types = ['Call', 'Email', 'Chat', 'Branch Visit', 'App']
        self.claim_statuses = ['Pending', 'Approved', 'Rejected', 'Processing', 'Closed']
        
    def generate_customers(self) -> pd.DataFrame:
        """Generate customer data"""
        customers = []
        
        for i in range(self.num_customers):
            # Generate PII (will be masked later)
            first_name = fake.first_name()
            last_name = fake.last_name()
            email = f"{first_name.lower()}.{last_name.lower()}@{fake.domain_name()}"
            
            customer = {
                'customer_id': f'C{str(i+1).zfill(6)}',
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'phone': fake.phone_number(),
                'date_of_birth': fake.date_of_birth(minimum_age=18, maximum_age=75),
                'gender': random.choice(['M', 'F', 'Other']),
                'address': fake.address().replace('\n', ', '),
                'city': fake.city(),
                'state': fake.state(),
                'zip_code': fake.zipcode(),
                'annual_income': int(np.random.lognormal(11, 0.8)),  # Log-normal income distribution
                'credit_score': int(np.random.normal(700, 100)),
                'customer_since': fake.date_between(start_date='-15y', end_date='-1d'),
                'customer_lifetime_value': round(np.random.lognormal(9, 1.2), 2),
                'preferred_channel': random.choice(self.interaction_types),
                'risk_score': round(np.random.beta(2, 5), 3),  # Right-skewed risk
                'is_active': random.choices([True, False], weights=[0.85, 0.15])[0],
                'last_updated': datetime.now()
            }
            
            # Ensure credit score is in valid range
            customer['credit_score'] = max(300, min(850, customer['credit_score']))
            
            customers.append(customer)
        
        return pd.DataFrame(customers)
    
    def generate_policies(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Generate policy data"""
        policies = []
        policy_counter = 0
        
        for _, customer in customers_df.iterrows():
            num_policies = np.random.poisson(2.4)  # Poisson distribution for policy count
            num_policies = max(1, min(5, num_policies))  # Between 1 and 5 policies
            
            for _ in range(num_policies):
                policy_counter += 1
                policy_type = random.choice(self.policy_types)
                
                # Policy dates
                start_date = fake.date_between(
                    start_date=customer['customer_since'], 
                    end_date='today'
                )
                term_years = random.choice([5, 10, 15, 20, 25, 30])
                end_date = start_date + timedelta(days=term_years*365)
                
                # Financial details
                premium_amount = round(random.uniform(500, 50000), 2)
                sum_assured = premium_amount * random.uniform(10, 100)
                
                # Surrender calculations
                tenure_months = (datetime.now().date() - start_date).days // 30
                surrender_charge_pct = max(0, min(10, 10 - tenure_months/12))  # Decreases over time
                
                policy = {
                    'policy_id': f'P{str(policy_counter).zfill(8)}',
                    'customer_id': customer['customer_id'],
                    'policy_type': policy_type,
                    'policy_status': random.choices(
                        ['Active', 'Lapsed', 'Surrendered', 'Matured', 'Pending'],
                        weights=[0.7, 0.1, 0.05, 0.1, 0.05]
                    )[0],
                    'start_date': start_date,
                    'end_date': end_date,
                    'term_years': term_years,
                    'premium_amount': premium_amount,
                    'premium_frequency': random.choice(['Monthly', 'Quarterly', 'Semi-Annual', 'Annual']),
                    'sum_assured': round(sum_assured, 2),
                    'current_fund_value': round(sum_assured * random.uniform(0.8, 1.3), 2),
                    'fund_value_change_3m': round(random.uniform(-0.15, 0.25), 3),
                    'tenure_months': tenure_months,
                    'surrender_charge_pct': round(surrender_charge_pct, 2),
                    'lock_in_period_months': random.choice([36, 48, 60]),
                    'payment_gaps': random.choices([0, 1, 2, 3], weights=[0.7, 0.15, 0.1, 0.05])[0],
                    'last_premium_paid': fake.date_between(start_date='-60d', end_date='today'),
                    'agent_code': f'AG{random.randint(1000, 9999)}',
                    'rider_count': random.choices([0, 1, 2], weights=[0.5, 0.35, 0.15])[0],
                    'created_at': start_date,
                    'updated_at': datetime.now()
                }
                
                policies.append(policy)
        
        return pd.DataFrame(policies)
    
    def generate_claims(self, policies_df: pd.DataFrame) -> pd.DataFrame:
        """Generate claims data"""
        claims = []
        claim_counter = 0
        
        # Select subset of policies for claims
        policies_with_claims = policies_df.sample(n=self.num_claims, replace=True)
        
        for _, policy in policies_with_claims.iterrows():
            claim_counter += 1
            
            claim_date = fake.date_between(start_date=policy['start_date'], end_date='today')
            
            claim = {
                'claim_id': f'CL{str(claim_counter).zfill(7)}',
                'policy_id': policy['policy_id'],
                'customer_id': policy['customer_id'],
                'claim_type': random.choice(['Death', 'Hospitalization', 'Accident', 'Critical Illness', 'Maturity']),
                'claim_amount': round(random.uniform(1000, policy['sum_assured']*0.5), 2),
                'claim_date': claim_date,
                'claim_status': random.choice(self.claim_statuses),
                'approved_amount': None,
                'rejection_reason': None,
                'processing_days': random.randint(1, 90),
                'documents_submitted': random.choice([True, False]),
                'investigation_required': random.choices([True, False], weights=[0.1, 0.9])[0],
                'claim_notes': fake.text(max_nb_chars=200),
                'created_at': claim_date,
                'updated_at': datetime.now()
            }
            
            # Set approved amount for approved claims
            if claim['claim_status'] == 'Approved':
                claim['approved_amount'] = round(claim['claim_amount'] * random.uniform(0.7, 1.0), 2)
            elif claim['claim_status'] == 'Rejected':
                claim['rejection_reason'] = random.choice([
                    'Incomplete documentation',
                    'Policy lapsed',
                    'Exclusion clause',
                    'Fraudulent claim',
                    'Waiting period not met'
                ])
            
            claims.append(claim)
        
        return pd.DataFrame(claims)
    
    def generate_interactions(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Generate customer interaction data"""
        interactions = []
        interaction_counter = 0
        
        for _, customer in customers_df.iterrows():
            num_interactions = np.random.poisson(5)  # Average 5 interactions
            
            for _ in range(num_interactions):
                interaction_counter += 1
                
                interaction = {
                    'interaction_id': f'INT{str(interaction_counter).zfill(8)}',
                    'customer_id': customer['customer_id'],
                    'interaction_type': random.choice(self.interaction_types),
                    'interaction_date': fake.date_between(start_date='-365d', end_date='today'),
                    'interaction_reason': random.choice([
                        'Premium Payment',
                        'Policy Inquiry',
                        'Claim Status',
                        'Surrender Query',
                        'Document Upload',
                        'Complaint',
                        'New Product',
                        'Renewal'
                    ]),
                    'sentiment_score': round(random.uniform(-1, 1), 3),  # -1 to 1
                    'resolution_status': random.choice(['Resolved', 'Pending', 'Escalated']),
                    'agent_id': f'AGT{random.randint(100, 999)}',
                    'call_duration_seconds': random.randint(30, 1800) if random.random() > 0.5 else None,
                    'notes': fake.text(max_nb_chars=150),
                    'created_at': datetime.now()
                }
                
                interactions.append(interaction)
        
        return pd.DataFrame(interactions)
    
    def generate_surrender_labels(self, policies_df: pd.DataFrame, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate surrender propensity labels based on features"""
        surrender_data = []
        
        # Group interactions by customer
        interaction_stats = interactions_df.groupby('customer_id').agg({
            'sentiment_score': 'mean',
            'interaction_id': 'count',
            'interaction_reason': lambda x: (x == 'Surrender Query').sum()
        }).rename(columns={
            'sentiment_score': 'avg_sentiment',
            'interaction_id': 'interaction_count',
            'interaction_reason': 'surrender_queries'
        })
        
        for _, policy in policies_df.iterrows():
            customer_id = policy['customer_id']
            
            # Get interaction stats
            if customer_id in interaction_stats.index:
                cust_stats = interaction_stats.loc[customer_id]
                avg_sentiment = cust_stats['avg_sentiment']
                interaction_count = cust_stats['interaction_count']
                surrender_queries = cust_stats['surrender_queries']
            else:
                avg_sentiment = 0
                interaction_count = 0
                surrender_queries = 0
            
            # Calculate surrender probability based on multiple factors
            base_prob = 0.1  # 10% base probability
            
            # Factors that increase surrender probability
            if policy['payment_gaps'] > 1:
                base_prob += 0.2
            if policy['fund_value_change_3m'] < -0.05:
                base_prob += 0.15
            if avg_sentiment < -0.3:
                base_prob += 0.1
            if surrender_queries > 0:
                base_prob += 0.25
            if policy['tenure_months'] < 36 and policy['surrender_charge_pct'] > 5:
                base_prob -= 0.1  # High surrender charge reduces probability
            
            # Add some randomness
            base_prob = min(0.95, max(0.05, base_prob + np.random.normal(0, 0.05)))
            
            # Generate binary label based on probability
            surrendered = np.random.random() < base_prob
            
            surrender_record = {
                'policy_id': policy['policy_id'],
                'customer_id': customer_id,
                'tenure_months': policy['tenure_months'],
                'premium_amount': policy['premium_amount'],
                'payment_gaps': policy['payment_gaps'],
                'surrender_charge_pct': policy['surrender_charge_pct'],
                'fund_value_change_3m': policy['fund_value_change_3m'],
                'avg_sentiment': avg_sentiment,
                'interaction_count_90d': min(interaction_count, 10),  # Cap at 10
                'surrender_queries_90d': surrender_queries,
                'complaints_90d': random.choices([0, 1, 2], weights=[0.7, 0.2, 0.1])[0],
                'agent_activity_30d': random.choices([0, 1, 2, 3], weights=[0.3, 0.4, 0.2, 0.1])[0],
                'surrender_probability': round(base_prob, 3),
                'surrendered_in_next_90d': int(surrendered),
                'model_version': 'v1.0',
                'created_at': datetime.now()
            }
            
            surrender_data.append(surrender_record)
        
        return pd.DataFrame(surrender_data)
    
    def mask_pii(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mask PII data for privacy"""
        df_masked = df.copy()
        
        # Mask email
        if 'email' in df_masked.columns:
            df_masked['email'] = df_masked['email'].apply(
                lambda x: x.split('@')[0][:3] + '****@' + x.split('@')[1] if '@' in str(x) else x
            )
        
        # Mask phone
        if 'phone' in df_masked.columns:
            df_masked['phone'] = df_masked['phone'].apply(
                lambda x: str(x)[:3] + '****' + str(x)[-2:] if len(str(x)) > 5 else x
            )
        
        # Hash names
        for col in ['first_name', 'last_name']:
            if col in df_masked.columns:
                df_masked[f'{col}_hash'] = df_masked[col].apply(
                    lambda x: hashlib.md5(str(x).encode()).hexdigest()[:8]
                )
                df_masked[col] = df_masked[col].apply(lambda x: x[0] + '***')
        
        return df_masked
    
    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """Generate all datasets"""
        print("Generating synthetic insurance data...")
        
        # Generate base data
        customers = self.generate_customers()
        print(f"✓ Generated {len(customers)} customers")
        
        policies = self.generate_policies(customers)
        print(f"✓ Generated {len(policies)} policies")
        
        claims = self.generate_claims(policies)
        print(f"✓ Generated {len(claims)} claims")
        
        interactions = self.generate_interactions(customers)
        print(f"✓ Generated {len(interactions)} interactions")
        
        surrender_labels = self.generate_surrender_labels(policies, interactions)
        print(f"✓ Generated {len(surrender_labels)} surrender labels")
        
        # Mask PII
        customers_masked = self.mask_pii(customers)
        print("✓ Masked PII data")
        
        return {
            'customers': customers_masked,
            'policies': policies,
            'claims': claims,
            'interactions': interactions,
            'surrender_labels': surrender_labels
        }

if __name__ == "__main__":
    generator = InsuranceDataGenerator(num_customers=50000)
    all_data = generator.generate_all_data()
    
    # Save to parquet files
    for table_name, df in all_data.items():
        df.to_parquet(f"data/{table_name}.parquet", index=False)
        print(f"Saved {table_name} to data/{table_name}.parquet")