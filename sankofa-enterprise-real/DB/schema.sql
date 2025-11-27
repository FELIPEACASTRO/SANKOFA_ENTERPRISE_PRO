-- =====================================================
-- Sankofa Enterprise Pro - Database Schema v12.0
-- PostgreSQL Production-Ready Schema
-- =====================================================
-- 
-- Este arquivo contém o schema completo do banco de dados
-- para o sistema de detecção de fraude Sankofa Enterprise Pro
--
-- Autor: Sankofa Team
-- Data: Novembro 2025
-- Versão: 12.0
-- =====================================================

-- =====================================================
-- EXTENSÕES NECESSÁRIAS
-- =====================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- Para busca fuzzy

-- =====================================================
-- TABELA: transactions
-- Transações financeiras (PIX, TED, Cartão, etc.)
-- =====================================================

CREATE TABLE IF NOT EXISTS transactions (
    -- Identificadores
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- Dados do Cliente (LGPD: mascarados)
    cliente_cpf VARCHAR(14) NOT NULL,  -- Armazenado mascarado: XXX.XXX.XXX-XX
    customer_id VARCHAR(100),
    
    -- Dados da Transação
    amount DECIMAL(15, 2) NOT NULL CHECK (amount >= 0),
    valor DECIMAL(15, 2) NOT NULL CHECK (valor >= 0),
    currency VARCHAR(3) DEFAULT 'BRL',
    tipo_transacao VARCHAR(50) NOT NULL,  -- PIX, TED, DOC, CARTAO_CREDITO, CARTAO_DEBITO
    canal VARCHAR(50) NOT NULL,           -- APP, WEB, ATM, AGENCIA, API
    status VARCHAR(20) DEFAULT 'PENDING', -- PENDING, APPROVED, BLOCKED, REVIEW
    
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
    pix_key_type VARCHAR(20),  -- CPF, CNPJ, EMAIL, TELEFONE, ALEATORIO
    
    -- Resultados da Análise de Fraude
    is_fraud BOOLEAN DEFAULT FALSE,
    fraud_score DECIMAL(5, 4) CHECK (fraud_score >= 0 AND fraud_score <= 1),
    risk_level VARCHAR(20) DEFAULT 'LOW',  -- LOW, MEDIUM, HIGH, CRITICAL
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

-- =====================================================
-- TABELA: fraud_detections
-- Detalhes das detecções de fraude
-- =====================================================

CREATE TABLE IF NOT EXISTS fraud_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id UUID REFERENCES transactions(id) ON DELETE CASCADE,
    
    -- Scores e Probabilidades
    fraud_probability DECIMAL(5, 4) NOT NULL CHECK (fraud_probability >= 0 AND fraud_probability <= 1),
    risk_score DECIMAL(5, 4) NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
    risk_level VARCHAR(20) NOT NULL,
    
    -- Explicação (LGPD Art. 20)
    detection_reason TEXT[],
    top_risk_factors JSONB DEFAULT '[]',
    top_protective_factors JSONB DEFAULT '[]',
    explanation_text TEXT,
    lgpd_compliant BOOLEAN DEFAULT TRUE,
    
    -- Modelo usado
    model_version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50),  -- ENSEMBLE, LSTM, GNN, etc.
    ensemble_votes JSONB DEFAULT '{}',
    
    -- Performance
    processing_time_ms DECIMAL(10, 2),
    cache_hit BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    reviewed_by VARCHAR(100)
);

-- =====================================================
-- TABELA: customers
-- Perfil de risco dos clientes
-- =====================================================

CREATE TABLE IF NOT EXISTS customers (
    id VARCHAR(100) PRIMARY KEY,
    cpf_hash VARCHAR(64) NOT NULL,  -- Hash do CPF para privacidade
    
    -- Perfil
    risk_profile VARCHAR(20) NOT NULL DEFAULT 'LOW',
    is_active BOOLEAN DEFAULT TRUE,
    account_age_days INTEGER DEFAULT 0,
    
    -- Estatísticas de transação
    transaction_count INTEGER DEFAULT 0,
    total_amount DECIMAL(15, 2) DEFAULT 0.00,
    avg_amount DECIMAL(15, 2) DEFAULT 0.00,
    last_transaction_at TIMESTAMP WITH TIME ZONE,
    
    -- Histórico de fraude
    fraud_count INTEGER DEFAULT 0,
    fraud_rate DECIMAL(5, 4) DEFAULT 0.0,
    last_fraud_at TIMESTAMP WITH TIME ZONE,
    
    -- Dispositivos conhecidos
    known_devices JSONB DEFAULT '[]',
    known_locations JSONB DEFAULT '[]',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Metadados
    metadata JSONB DEFAULT '{}',
    version INTEGER DEFAULT 1
);

-- =====================================================
-- TABELA: audit_trail
-- Log de auditoria (compliance LGPD/BACEN)
-- =====================================================

CREATE TABLE IF NOT EXISTS audit_trail (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Evento
    event_type VARCHAR(100) NOT NULL,
    action VARCHAR(50) NOT NULL,  -- CREATE, READ, UPDATE, DELETE, PREDICT, REVIEW
    status VARCHAR(20) NOT NULL,  -- SUCCESS, FAILURE, PENDING
    
    -- Recurso
    resource_type VARCHAR(100),  -- transaction, customer, model, etc.
    resource_id VARCHAR(100),
    
    -- Usuário/Sistema
    user_id VARCHAR(100),
    user_role VARCHAR(50),
    system_component VARCHAR(100),
    
    -- Detalhes
    details JSONB DEFAULT '{}',
    changes_before JSONB,
    changes_after JSONB,
    
    -- Rede
    ip_address INET,
    user_agent TEXT,
    session_id VARCHAR(100),
    
    -- Timestamps
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Compliance
    retention_until TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '7 years'),
    is_sensitive BOOLEAN DEFAULT FALSE
);

-- =====================================================
-- TABELA: users
-- Usuários do sistema
-- =====================================================

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Credenciais
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    
    -- Perfil
    full_name VARCHAR(255),
    role VARCHAR(50) NOT NULL,  -- ADMIN, ANALYST, VIEWER, API
    department VARCHAR(100),
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    
    -- Segurança
    last_login TIMESTAMP WITH TIME ZONE,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret VARCHAR(100),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- TABELA: model_versions
-- Versões dos modelos ML
-- =====================================================

CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Identificação
    version VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,  -- ENSEMBLE, RANDOM_FOREST, XGBOOST, LIGHTGBM, LSTM, GNN
    
    -- Métricas
    metrics JSONB DEFAULT '{}',
    accuracy DECIMAL(5, 4),
    precision_score DECIMAL(5, 4),
    recall DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    roc_auc DECIMAL(5, 4),
    
    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'TRAINING',  -- TRAINING, VALIDATION, PRODUCTION, DEPRECATED
    is_active BOOLEAN DEFAULT FALSE,
    
    -- Dados de treinamento
    training_data_size INTEGER,
    feature_count INTEGER,
    training_duration_seconds INTEGER,
    
    -- Timestamps
    trained_at TIMESTAMP WITH TIME ZONE,
    validated_at TIMESTAMP WITH TIME ZONE,
    deployed_at TIMESTAMP WITH TIME ZONE,
    deprecated_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Armazenamento
    model_path VARCHAR(500),
    model_size_bytes BIGINT
);

-- =====================================================
-- TABELA: api_keys
-- Chaves de API para autenticação
-- =====================================================

CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Chave
    key_hash VARCHAR(128) NOT NULL UNIQUE,
    key_prefix VARCHAR(10) NOT NULL,  -- Primeiros caracteres para identificação
    name VARCHAR(100) NOT NULL,
    description TEXT,
    
    -- Permissões
    permissions JSONB DEFAULT '["read"]',
    rate_limit INTEGER DEFAULT 1000,  -- Requisições por minuto
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Uso
    last_used_at TIMESTAMP WITH TIME ZONE,
    usage_count BIGINT DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(100)
);

-- =====================================================
-- TABELA: events
-- Event Sourcing
-- =====================================================

CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id UUID NOT NULL,
    aggregate_id VARCHAR(100) NOT NULL,
    
    -- Evento
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    
    -- Versionamento
    version INTEGER NOT NULL,
    
    -- Timestamps
    occurred_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- TABELA: alerts
-- Alertas do sistema
-- =====================================================

CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Alerta
    alert_type VARCHAR(50) NOT NULL,  -- FRAUD, ANOMALY, SYSTEM, MODEL
    severity VARCHAR(20) NOT NULL,    -- INFO, WARNING, CRITICAL
    title VARCHAR(255) NOT NULL,
    message TEXT,
    
    -- Referência
    transaction_id UUID REFERENCES transactions(id),
    customer_id VARCHAR(100),
    
    -- Status
    status VARCHAR(20) DEFAULT 'OPEN',  -- OPEN, ACKNOWLEDGED, RESOLVED, CLOSED
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(100),
    resolution_notes TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- TABELA: compliance_reports
-- Relatórios de compliance
-- =====================================================

CREATE TABLE IF NOT EXISTS compliance_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Relatório
    report_type VARCHAR(50) NOT NULL,  -- BACEN, LGPD, PCI_DSS, INTERNAL
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Dados
    report_data JSONB NOT NULL,
    summary JSONB,
    
    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',  -- PENDING, GENERATED, SUBMITTED, APPROVED
    
    -- Timestamps
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    submitted_at TIMESTAMP WITH TIME ZONE,
    approved_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadados
    generated_by VARCHAR(100),
    file_path VARCHAR(500)
);

-- =====================================================
-- TABELA: rate_limits
-- Controle de rate limiting
-- =====================================================

CREATE TABLE IF NOT EXISTS rate_limits (
    key_hash VARCHAR(128) NOT NULL,
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    request_count INTEGER DEFAULT 0,
    PRIMARY KEY (key_hash, window_start)
);

-- =====================================================
-- TABELA: schema_migrations
-- Controle de migrações
-- =====================================================

CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(50) PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    applied_by VARCHAR(100),
    checksum VARCHAR(64)
);

-- =====================================================
-- ÍNDICES
-- =====================================================

-- Transactions
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_cliente_cpf ON transactions(cliente_cpf);
CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_is_fraud ON transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_transactions_risk_level ON transactions(risk_level);
CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status);
CREATE INDEX IF NOT EXISTS idx_transactions_canal ON transactions(canal);
CREATE INDEX IF NOT EXISTS idx_transactions_tipo ON transactions(tipo_transacao);
CREATE INDEX IF NOT EXISTS idx_transactions_amount ON transactions(amount);

-- Índices compostos para queries frequentes
CREATE INDEX IF NOT EXISTS idx_transactions_customer_timestamp ON transactions(customer_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_fraud_timestamp ON transactions(is_fraud, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_canal_timestamp ON transactions(canal, timestamp DESC);

-- Fraud Detections
CREATE INDEX IF NOT EXISTS idx_fraud_detections_transaction_id ON fraud_detections(transaction_id);
CREATE INDEX IF NOT EXISTS idx_fraud_detections_risk_level ON fraud_detections(risk_level);
CREATE INDEX IF NOT EXISTS idx_fraud_detections_detected_at ON fraud_detections(detected_at DESC);

-- Customers
CREATE UNIQUE INDEX IF NOT EXISTS idx_customers_cpf_hash ON customers(cpf_hash);
CREATE INDEX IF NOT EXISTS idx_customers_risk_profile ON customers(risk_profile);
CREATE INDEX IF NOT EXISTS idx_customers_last_transaction ON customers(last_transaction_at);

-- Audit Trail
CREATE INDEX IF NOT EXISTS idx_audit_trail_timestamp ON audit_trail(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_trail_user_id ON audit_trail(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_trail_resource ON audit_trail(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_trail_event_type ON audit_trail(event_type);

-- Users
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

-- Model Versions
CREATE INDEX IF NOT EXISTS idx_model_versions_status ON model_versions(status);
CREATE INDEX IF NOT EXISTS idx_model_versions_active ON model_versions(is_active);

-- API Keys
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active);
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix);

-- Events
CREATE INDEX IF NOT EXISTS idx_events_aggregate_id ON events(aggregate_id, version);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events(occurred_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_unique ON events(event_id, aggregate_id);

-- Alerts
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_transaction_id ON alerts(transaction_id);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at DESC);

-- Rate Limits
CREATE INDEX IF NOT EXISTS idx_rate_limits_window ON rate_limits(window_start);

-- =====================================================
-- TRIGGERS
-- =====================================================

-- Função para atualizar updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger para transactions
DROP TRIGGER IF EXISTS update_transactions_updated_at ON transactions;
CREATE TRIGGER update_transactions_updated_at
    BEFORE UPDATE ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger para customers
DROP TRIGGER IF EXISTS update_customers_updated_at ON customers;
CREATE TRIGGER update_customers_updated_at
    BEFORE UPDATE ON customers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger para users
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Função para atualizar estatísticas do cliente
CREATE OR REPLACE FUNCTION update_customer_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO customers (id, cpf_hash, transaction_count, total_amount, last_transaction_at)
        VALUES (
            NEW.customer_id,
            encode(digest(NEW.cliente_cpf, 'sha256'), 'hex'),
            1,
            NEW.amount,
            NEW.timestamp
        )
        ON CONFLICT (id) DO UPDATE SET
            transaction_count = customers.transaction_count + 1,
            total_amount = customers.total_amount + NEW.amount,
            avg_amount = (customers.total_amount + NEW.amount) / (customers.transaction_count + 1),
            last_transaction_at = NEW.timestamp,
            updated_at = NOW();
            
        IF NEW.is_fraud THEN
            UPDATE customers SET
                fraud_count = fraud_count + 1,
                fraud_rate = (fraud_count + 1)::DECIMAL / (transaction_count + 1),
                last_fraud_at = NEW.timestamp
            WHERE id = NEW.customer_id;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger para atualizar estatísticas do cliente
DROP TRIGGER IF EXISTS update_customer_stats_trigger ON transactions;
CREATE TRIGGER update_customer_stats_trigger
    AFTER INSERT ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_customer_stats();

-- =====================================================
-- VIEWS
-- =====================================================

-- View: Estatísticas de fraude por dia
CREATE OR REPLACE VIEW fraud_statistics AS
SELECT
    DATE(timestamp) as date,
    COUNT(*) as total_transactions,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count,
    ROUND(AVG(CASE WHEN is_fraud THEN 1 ELSE 0 END) * 100, 2) as fraud_rate_percent,
    SUM(amount) as total_amount,
    SUM(CASE WHEN is_fraud THEN amount ELSE 0 END) as fraud_amount,
    AVG(fraud_score) as avg_fraud_score
FROM transactions
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- View: Estatísticas por canal
CREATE OR REPLACE VIEW channel_statistics AS
SELECT
    canal,
    COUNT(*) as total_transactions,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count,
    ROUND(AVG(CASE WHEN is_fraud THEN 1 ELSE 0 END) * 100, 2) as fraud_rate_percent,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount
FROM transactions
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY canal
ORDER BY total_transactions DESC;

-- View: Top clientes de risco
CREATE OR REPLACE VIEW high_risk_customers AS
SELECT
    c.id,
    c.risk_profile,
    c.transaction_count,
    c.total_amount,
    c.fraud_count,
    c.fraud_rate,
    c.last_fraud_at
FROM customers c
WHERE c.fraud_rate > 0.05 OR c.fraud_count > 0
ORDER BY c.fraud_rate DESC, c.fraud_count DESC
LIMIT 100;

-- View: Performance dos modelos
CREATE OR REPLACE VIEW model_performance AS
SELECT
    version,
    name,
    model_type,
    status,
    accuracy,
    precision_score,
    recall,
    f1_score,
    roc_auc,
    deployed_at,
    is_active
FROM model_versions
ORDER BY deployed_at DESC NULLS LAST;

-- =====================================================
-- COMENTÁRIOS PARA DOCUMENTAÇÃO
-- =====================================================

COMMENT ON TABLE transactions IS 'Tabela principal de transações financeiras com resultados de detecção de fraude';
COMMENT ON TABLE fraud_detections IS 'Detalhes das detecções de fraude pelos modelos ML';
COMMENT ON TABLE customers IS 'Perfil de risco e histórico dos clientes';
COMMENT ON TABLE audit_trail IS 'Log de auditoria append-only para compliance (BACEN, LGPD)';
COMMENT ON TABLE users IS 'Usuários do sistema para autenticação e autorização';
COMMENT ON TABLE model_versions IS 'Registro e lifecycle dos modelos de ML';
COMMENT ON TABLE api_keys IS 'Chaves de API para autenticação de sistemas externos';
COMMENT ON TABLE events IS 'Event sourcing para rastreabilidade completa';
COMMENT ON TABLE alerts IS 'Alertas de fraude e sistema';
COMMENT ON TABLE compliance_reports IS 'Relatórios regulatórios (BACEN, LGPD, PCI DSS)';

COMMENT ON COLUMN transactions.cliente_cpf IS 'CPF mascarado (XXX.XXX.XXX-XX) para LGPD';
COMMENT ON COLUMN transactions.fraud_score IS 'Score de fraude entre 0 e 1';
COMMENT ON COLUMN audit_trail.retention_until IS 'Data até quando manter registro (7 anos para BACEN)';
COMMENT ON COLUMN customers.cpf_hash IS 'Hash SHA-256 do CPF para busca sem expor dado sensível';
