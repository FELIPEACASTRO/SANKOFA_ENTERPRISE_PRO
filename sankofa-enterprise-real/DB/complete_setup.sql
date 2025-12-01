-- =====================================================
-- SANKOFA ENTERPRISE PRO - SETUP COMPLETO DO BANCO
-- =====================================================
-- 
-- Este arquivo contém TODO o schema e dados necessários
-- para subir a aplicação Sankofa Enterprise Pro
--
-- Versão: 12.0
-- Data: Dezembro 2025
-- =====================================================

-- =====================================================
-- CONFIGURAÇÃO INICIAL
-- =====================================================

SET statement_timeout = '120s';
SET lock_timeout = '30s';
SET timezone = 'America/Sao_Paulo';

-- =====================================================
-- 1. TABELA: users
-- Usuários do sistema com autenticação
-- =====================================================

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255),
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'viewer',
    is_active BOOLEAN DEFAULT TRUE,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);

-- =====================================================
-- 2. TABELA: transactions
-- Transações financeiras processadas
-- =====================================================

CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(100) UNIQUE NOT NULL,
    amount NUMERIC(15,2) NOT NULL,
    channel VARCHAR(50) NOT NULL,
    type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    risk_score NUMERIC(5,4),
    is_fraud BOOLEAN DEFAULT FALSE,
    cpf VARCHAR(20),
    location VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms NUMERIC(10,2),
    model_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_is_fraud ON transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status);
CREATE INDEX IF NOT EXISTS idx_transactions_channel ON transactions(channel);
CREATE INDEX IF NOT EXISTS idx_transactions_cpf ON transactions(cpf);

-- =====================================================
-- 3. TABELA: hard_rules
-- Regras duras para detecção de fraude
-- =====================================================

CREATE TABLE IF NOT EXISTS hard_rules (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    condition TEXT NOT NULL,
    action VARCHAR(50) NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    conditions_json JSONB DEFAULT '[]',
    logic_operator VARCHAR(10) DEFAULT 'AND',
    priority INTEGER DEFAULT 1,
    description TEXT,
    action_config JSONB DEFAULT '{}',
    rule_type VARCHAR(50) DEFAULT 'blocking'
);

CREATE INDEX IF NOT EXISTS idx_hard_rules_enabled ON hard_rules(enabled);
CREATE INDEX IF NOT EXISTS idx_hard_rules_action ON hard_rules(action);
CREATE INDEX IF NOT EXISTS idx_hard_rules_priority ON hard_rules(priority);

-- =====================================================
-- 4. TABELA: vip_list
-- Lista VIP (whitelist) de clientes confiáveis
-- =====================================================

CREATE TABLE IF NOT EXISTS vip_list (
    id SERIAL PRIMARY KEY,
    identifier VARCHAR(255) NOT NULL,
    identifier_type VARCHAR(50) NOT NULL,
    reason TEXT,
    added_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_vip_list_identifier ON vip_list(identifier);
CREATE INDEX IF NOT EXISTS idx_vip_list_type ON vip_list(identifier_type);

-- =====================================================
-- 5. TABELA: hot_list
-- Lista Hot (blacklist) de entidades suspeitas
-- =====================================================

CREATE TABLE IF NOT EXISTS hot_list (
    id SERIAL PRIMARY KEY,
    identifier VARCHAR(255) NOT NULL,
    identifier_type VARCHAR(50) NOT NULL,
    reason TEXT,
    added_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_hot_list_identifier ON hot_list(identifier);
CREATE INDEX IF NOT EXISTS idx_hot_list_type ON hot_list(identifier_type);

-- =====================================================
-- 6. TABELA: alerts
-- Alertas de fraude gerados pelo sistema
-- =====================================================

CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(100) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'novo',
    transaction_id VARCHAR(100),
    amount_involved NUMERIC(15,2),
    recommended_action TEXT,
    investigator VARCHAR(100),
    tags TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(type);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at DESC);

-- =====================================================
-- 7. TABELA: feedback
-- Feedback de analistas sobre transações
-- =====================================================

CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(100) NOT NULL,
    is_fraud BOOLEAN NOT NULL,
    analyst_notes TEXT,
    analyst_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_feedback_transaction_id ON feedback(transaction_id);
CREATE INDEX IF NOT EXISTS idx_feedback_is_fraud ON feedback(is_fraud);

-- =====================================================
-- 8. TABELA: audit_logs
-- Logs de auditoria (compliance LGPD/BACEN)
-- =====================================================

CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    action VARCHAR(100) NOT NULL,
    user_id VARCHAR(100),
    details TEXT,
    ip_address VARCHAR(45),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at DESC);

-- =====================================================
-- 9. TABELA: model_metrics
-- Métricas de performance dos modelos ML
-- =====================================================

CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(20) NOT NULL,
    accuracy NUMERIC(5,4),
    precision_score NUMERIC(5,4),
    recall NUMERIC(5,4),
    f1_score NUMERIC(5,4),
    roc_auc NUMERIC(5,4),
    threshold NUMERIC(5,4),
    samples_used INTEGER,
    fraud_ratio NUMERIC(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_model_metrics_version ON model_metrics(model_version);

-- =====================================================
-- 10. TABELA: system_configs
-- Configurações do sistema
-- =====================================================

CREATE TABLE IF NOT EXISTS system_configs (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_system_configs_key ON system_configs(config_key);

-- =====================================================
-- 11. TABELAS RBAC (Role-Based Access Control)
-- =====================================================

CREATE TABLE IF NOT EXISTS rbac_roles (
    id VARCHAR(100) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    permissions JSONB NOT NULL DEFAULT '[]',
    is_system_role BOOLEAN DEFAULT FALSE,
    parent_role VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rbac_user_roles (
    user_id VARCHAR(100) NOT NULL,
    role_name VARCHAR(100) NOT NULL,
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    granted_by VARCHAR(100),
    expires_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (user_id, role_name)
);

CREATE TABLE IF NOT EXISTS rbac_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS rbac_permissions_override (
    user_id VARCHAR(100) NOT NULL,
    permission VARCHAR(100) NOT NULL,
    is_granted BOOLEAN NOT NULL,
    reason TEXT,
    granted_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (user_id, permission)
);

-- =====================================================
-- 12. TABELAS DE TOKENIZAÇÃO CPF (LGPD)
-- =====================================================

CREATE TABLE IF NOT EXISTS cpf_tokens (
    token VARCHAR(100) PRIMARY KEY,
    encrypted_cpf BYTEA NOT NULL,
    cpf_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_cpf_tokens_hash ON cpf_tokens(cpf_hash);

CREATE TABLE IF NOT EXISTS cpf_access_log (
    id VARCHAR(100) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    token VARCHAR(100) NOT NULL,
    action VARCHAR(50) NOT NULL,
    purpose TEXT,
    user_id VARCHAR(100),
    ip_address INET,
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_cpf_access_log_token ON cpf_access_log(token);
CREATE INDEX IF NOT EXISTS idx_cpf_access_log_accessed_at ON cpf_access_log(accessed_at DESC);

-- =====================================================
-- TRIGGER: Atualizar updated_at automaticamente
-- =====================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_hard_rules_updated_at ON hard_rules;
CREATE TRIGGER update_hard_rules_updated_at
    BEFORE UPDATE ON hard_rules
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_alerts_updated_at ON alerts;
CREATE TRIGGER update_alerts_updated_at
    BEFORE UPDATE ON alerts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- FIM DO SCHEMA - INÍCIO DOS DADOS
-- =====================================================

