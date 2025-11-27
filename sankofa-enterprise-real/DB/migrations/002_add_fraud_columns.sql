-- =====================================================
-- Migration 002: Add Fraud Detection Columns
-- =====================================================
-- Version: 002
-- Description: Adiciona colunas para detecção de fraude
-- Date: 2025-11-27
-- =====================================================

BEGIN;

-- Adicionar colunas de fraude se não existirem
DO $$
BEGIN
    -- is_fraud column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'transactions' AND column_name = 'is_fraud') THEN
        ALTER TABLE transactions ADD COLUMN is_fraud BOOLEAN DEFAULT FALSE;
    END IF;
    
    -- fraud_score column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'transactions' AND column_name = 'fraud_score') THEN
        ALTER TABLE transactions ADD COLUMN fraud_score DECIMAL(5,4) DEFAULT 0.0;
    END IF;
    
    -- model_version column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'transactions' AND column_name = 'model_version') THEN
        ALTER TABLE transactions ADD COLUMN model_version VARCHAR(20);
    END IF;
    
    -- processing_time_ms column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'transactions' AND column_name = 'processing_time_ms') THEN
        ALTER TABLE transactions ADD COLUMN processing_time_ms INTEGER;
    END IF;
    
    -- explanation_text column (LGPD)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'transactions' AND column_name = 'explanation_text') THEN
        ALTER TABLE transactions ADD COLUMN explanation_text TEXT;
    END IF;
END $$;

-- Índice para consultas de fraude
CREATE INDEX IF NOT EXISTS idx_transactions_is_fraud ON transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_transactions_fraud_score ON transactions(fraud_score);

-- Register migration
INSERT INTO schema_migrations (version, applied_by) VALUES ('002_add_fraud_columns', 'migration_script')
ON CONFLICT DO NOTHING;

COMMIT;
