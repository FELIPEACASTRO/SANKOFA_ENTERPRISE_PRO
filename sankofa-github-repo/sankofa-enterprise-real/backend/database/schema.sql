-- =====================================================
-- Sankofa Enterprise Pro - Database Schema
-- PostgreSQL Production-Ready Schema
-- =====================================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id VARCHAR(100) UNIQUE NOT NULL,
    cliente_cpf VARCHAR(14) NOT NULL,
    valor DECIMAL(15, 2) NOT NULL,
    tipo_transacao VARCHAR(50) NOT NULL,
    canal VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    cidade VARCHAR(100),
    estado VARCHAR(2),
    pais VARCHAR(3),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    ip_address INET,
    device_id VARCHAR(100),
    conta_recebedor VARCHAR(100),
    is_fraud BOOLEAN DEFAULT FALSE,
    fraud_score DECIMAL(5, 4),
    risk_level VARCHAR(20),
    model_version VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Fraud detections table
CREATE TABLE IF NOT EXISTS fraud_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id UUID REFERENCES transactions(id),
    fraud_probability DECIMAL(5, 4) NOT NULL,
    risk_score DECIMAL(5, 4) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    detection_reason TEXT[],
    model_version VARCHAR(20) NOT NULL,
    processing_time_ms DECIMAL(10, 2),
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit trail table (append-only for compliance)
CREATE TABLE IF NOT EXISTS audit_trail (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    user_id VARCHAR(100),
    resource_type VARCHAR(100),
    resource_id VARCHAR(100),
    action VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Users table (for authentication)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP WITH TIME ZONE,
    failed_login_attempts INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model versions table
CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version VARCHAR(20) UNIQUE NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    metrics JSONB,
    status VARCHAR(20) NOT NULL,  -- training, validation, production, deprecated
    trained_at TIMESTAMP WITH TIME ZONE,
    deployed_at TIMESTAMP WITH TIME ZONE,
    deprecated_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Compliance reports table
CREATE TABLE IF NOT EXISTS compliance_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    report_type VARCHAR(50) NOT NULL,  -- bacen, lgpd, pci_dss
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    report_data JSONB,
    status VARCHAR(20) NOT NULL,  -- pending, generated, submitted
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    submitted_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_cliente_cpf ON transactions(cliente_cpf);
CREATE INDEX IF NOT EXISTS idx_transactions_is_fraud ON transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_fraud_detections_transaction_id ON fraud_detections(transaction_id);
CREATE INDEX IF NOT EXISTS idx_audit_trail_timestamp ON audit_trail(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_trail_user_id ON audit_trail(user_id);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_model_versions_status ON model_versions(status);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_transactions_updated_at
    BEFORE UPDATE ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- View for fraud statistics
CREATE OR REPLACE VIEW fraud_statistics AS
SELECT
    DATE(timestamp) as date,
    COUNT(*) as total_transactions,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count,
    ROUND(AVG(CASE WHEN is_fraud THEN 1 ELSE 0 END) * 100, 2) as fraud_rate,
    SUM(valor) as total_amount,
    SUM(CASE WHEN is_fraud THEN valor ELSE 0 END) as fraud_amount
FROM transactions
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- Comments for documentation
COMMENT ON TABLE transactions IS 'Main transactions table with fraud detection results';
COMMENT ON TABLE fraud_detections IS 'Detailed fraud detection results from ML models';
COMMENT ON TABLE audit_trail IS 'Append-only audit log for compliance (BACEN, LGPD)';
COMMENT ON TABLE users IS 'System users for authentication and authorization';
COMMENT ON TABLE model_versions IS 'ML model version tracking and lifecycle management';
COMMENT ON TABLE compliance_reports IS 'Regulatory compliance reports (BACEN, LGPD, PCI DSS)';
