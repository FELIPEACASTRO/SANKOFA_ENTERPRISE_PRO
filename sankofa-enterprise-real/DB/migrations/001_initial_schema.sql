-- =====================================================
-- Migration 001: Initial Schema
-- =====================================================
-- Version: 001
-- Description: Criação das tabelas principais do sistema de fraude
-- Date: 2025-11-27
-- Updated: 2025-12-12 (CORRECAO 10/10: Consolidação com schema principal)
-- =====================================================

-- Up Migration
BEGIN;

-- CORRECAO 10/10: Criar tabela de migrations primeiro (se não existir)
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(50) PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    applied_by VARCHAR(100),
    checksum VARCHAR(64)
);

-- Extensões necessárias
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- =====================================================
-- TABELA: transactions
-- Transações financeiras (PIX, TED, Cartão, etc.)
-- CORRECAO 10/10: Alinhado com schema principal (DB/schema.sql v12.0)
-- =====================================================
CREATE TABLE IF NOT EXISTS transactions (
    -- Identificadores
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id VARCHAR(100) UNIQUE NOT NULL,

    -- Dados do Cliente (LGPD: mascarados)
    cliente_cpf VARCHAR(14) NOT NULL,
    customer_id VARCHAR(100),

    -- Dados da Transação
    amount DECIMAL(15, 2) NOT NULL CHECK (amount >= 0),
    valor DECIMAL(15, 2) NOT NULL CHECK (valor >= 0),
    currency VARCHAR(3) DEFAULT 'BRL',
    tipo_transacao VARCHAR(50) NOT NULL,
    canal VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'PENDING',

    -- Localização
    cidade VARCHAR(100),
    estado VARCHAR(2),
    pais VARCHAR(3) DEFAULT 'BRA',
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),

    -- Dispositivo e Rede
    ip_address INET,
    device_id VARCHAR(100),
    device_type VARCHAR(50),
    user_agent TEXT,

    -- Dados do Recebedor (para PIX/TED)
    conta_recebedor VARCHAR(100),
    banco_recebedor VARCHAR(10),
    pix_key VARCHAR(100),
    pix_key_type VARCHAR(20),

    -- Resultados da Análise de Fraude
    is_fraud BOOLEAN DEFAULT FALSE,
    fraud_score DECIMAL(5, 4) CHECK (fraud_score >= 0 AND fraud_score <= 1),
    risk_level VARCHAR(20) DEFAULT 'LOW',
    risk_score FLOAT DEFAULT 0.0,

    -- Metadados ML
    model_version VARCHAR(20),
    processing_time_ms INTEGER,
    features_used JSONB DEFAULT '{}',
    explanation_text TEXT,

    -- Timestamps
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Metadados adicionais
    metadata JSONB DEFAULT '{}',
    version INTEGER DEFAULT 1
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_cliente_cpf ON transactions(cliente_cpf);
CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_is_fraud ON transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_transactions_risk_level ON transactions(risk_level);
CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status);
CREATE INDEX IF NOT EXISTS idx_transactions_canal ON transactions(canal);
CREATE INDEX IF NOT EXISTS idx_transactions_tipo ON transactions(tipo_transacao);
CREATE INDEX IF NOT EXISTS idx_transactions_amount ON transactions(amount);

-- Composite indexes
CREATE INDEX IF NOT EXISTS idx_transactions_customer_timestamp ON transactions(customer_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_fraud_timestamp ON transactions(is_fraud, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_canal_timestamp ON transactions(canal, timestamp DESC);

-- Trigger para updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_transactions_updated_at ON transactions;
CREATE TRIGGER update_transactions_updated_at
    BEFORE UPDATE ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Register migration
INSERT INTO schema_migrations (version, applied_by, checksum)
VALUES ('001_initial_schema', 'migration_script', 'sha256_consolidated_v12')
ON CONFLICT (version) DO UPDATE SET
    applied_at = NOW(),
    checksum = 'sha256_consolidated_v12';

COMMIT;

-- =====================================================
-- Down Migration (rollback)
-- =====================================================
-- BEGIN;
-- DROP TRIGGER IF EXISTS update_transactions_updated_at ON transactions;
-- DROP FUNCTION IF EXISTS update_updated_at_column();
-- DROP TABLE IF EXISTS transactions CASCADE;
-- DELETE FROM schema_migrations WHERE version = '001_initial_schema';
-- COMMIT;
