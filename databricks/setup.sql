
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
