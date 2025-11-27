-- =====================================================
-- Migration 001: Initial Schema
-- =====================================================
-- Version: 001
-- Description: Criação das tabelas principais
-- Date: 2025-11-27
-- =====================================================

-- Up Migration
BEGIN;

-- Transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id VARCHAR(50) PRIMARY KEY,
    amount DECIMAL(15,2) NOT NULL CHECK (amount >= 0),
    currency VARCHAR(3) NOT NULL DEFAULT 'BRL',
    merchant_id VARCHAR(100) NOT NULL,
    customer_id VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    risk_score FLOAT NOT NULL DEFAULT 0.0 CHECK (risk_score >= 0 AND risk_score <= 1),
    risk_level VARCHAR(20) NOT NULL DEFAULT 'low',
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_merchant_id ON transactions(merchant_id);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status);
CREATE INDEX IF NOT EXISTS idx_transactions_risk_level ON transactions(risk_level);
CREATE INDEX IF NOT EXISTS idx_transactions_amount ON transactions(amount);

-- Composite indexes
CREATE INDEX IF NOT EXISTS idx_transactions_customer_timestamp ON transactions(customer_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_status_timestamp ON transactions(status, timestamp DESC);

-- Register migration
INSERT INTO schema_migrations (version, applied_by) VALUES ('001_initial_schema', 'migration_script')
ON CONFLICT DO NOTHING;

COMMIT;

-- Down Migration (rollback)
-- BEGIN;
-- DROP TABLE IF EXISTS transactions CASCADE;
-- DELETE FROM schema_migrations WHERE version = '001_initial_schema';
-- COMMIT;
