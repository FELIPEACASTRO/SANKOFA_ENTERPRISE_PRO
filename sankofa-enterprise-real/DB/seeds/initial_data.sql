-- =====================================================
-- Sankofa Enterprise Pro - Dados Iniciais
-- =====================================================
-- 
-- Este script insere dados iniciais necessários para o funcionamento
-- do sistema, incluindo usuários, modelos e configurações.
-- =====================================================

-- =====================================================
-- USUÁRIO ADMIN PADRÃO
-- =====================================================

INSERT INTO users (id, username, email, password_hash, full_name, role, is_active, is_verified)
VALUES (
    uuid_generate_v4(),
    'admin',
    'admin@sankofa.com.br',
    -- Password: admin123 (bcrypt hash) - TROCAR EM PRODUÇÃO!
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4.W8FnkqHQK3fIzy',
    'Administrador Sistema',
    'ADMIN',
    TRUE,
    TRUE
)
ON CONFLICT (username) DO NOTHING;

INSERT INTO users (id, username, email, password_hash, full_name, role, is_active, is_verified)
VALUES (
    uuid_generate_v4(),
    'analista',
    'analista@sankofa.com.br',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4.W8FnkqHQK3fIzy',
    'Analista de Fraude',
    'ANALYST',
    TRUE,
    TRUE
)
ON CONFLICT (username) DO NOTHING;

-- =====================================================
-- VERSÃO INICIAL DO MODELO
-- =====================================================

INSERT INTO model_versions (
    id, version, name, model_type, status, is_active,
    accuracy, precision_score, recall, f1_score, roc_auc,
    feature_count, trained_at, deployed_at
)
VALUES (
    uuid_generate_v4(),
    '1.0.0',
    'Sankofa Fraud Engine',
    'ENSEMBLE',
    'PRODUCTION',
    TRUE,
    0.9500,
    0.9362,
    0.8980,
    0.9167,
    0.9952,
    30,
    NOW(),
    NOW()
)
ON CONFLICT (version) DO NOTHING;

-- =====================================================
-- API KEY PARA DESENVOLVIMENTO
-- =====================================================

INSERT INTO api_keys (
    id, key_hash, key_prefix, name, description,
    permissions, rate_limit, is_active
)
VALUES (
    uuid_generate_v4(),
    encode(digest('dev_api_key_sankofa_2025', 'sha256'), 'hex'),
    'dev_',
    'Development API Key',
    'Chave de API para ambiente de desenvolvimento',
    '["read", "write", "predict"]',
    10000,
    TRUE
)
ON CONFLICT DO NOTHING;

-- =====================================================
-- REGISTRO DE MIGRAÇÃO
-- =====================================================

INSERT INTO schema_migrations (version, applied_by, checksum)
VALUES 
    ('001_initial_schema', 'init_script', 'initial'),
    ('002_customers_table', 'init_script', 'initial'),
    ('003_events_table', 'init_script', 'initial'),
    ('004_audit_table', 'init_script', 'initial'),
    ('005_ml_models_table', 'init_script', 'initial'),
    ('006_security_tables', 'init_script', 'initial'),
    ('007_triggers_and_functions', 'init_script', 'initial')
ON CONFLICT (version) DO NOTHING;

-- =====================================================
-- AUDITORIA DA INICIALIZAÇÃO
-- =====================================================

INSERT INTO audit_trail (
    event_type, action, status,
    resource_type, details, system_component
)
VALUES (
    'SYSTEM_INIT',
    'CREATE',
    'SUCCESS',
    'database',
    jsonb_build_object(
        'version', '12.0',
        'initialized_at', NOW(),
        'environment', 'development'
    ),
    'init_script'
);

-- =====================================================
-- TRANSAÇÕES DE EXEMPLO (APENAS DESENVOLVIMENTO)
-- =====================================================

-- Transação normal (não fraude)
INSERT INTO transactions (
    transaction_id, cliente_cpf, customer_id,
    amount, valor, currency, tipo_transacao, canal, status,
    cidade, estado, pais,
    is_fraud, fraud_score, risk_level, model_version,
    timestamp
)
VALUES (
    'TXN_SAMPLE_001',
    'XXX.XXX.XXX-01',
    'CUST_001',
    1500.00, 1500.00, 'BRL', 'PIX', 'APP', 'APPROVED',
    'São Paulo', 'SP', 'BRA',
    FALSE, 0.12, 'LOW', '1.0.0',
    NOW() - INTERVAL '1 hour'
),
(
    'TXN_SAMPLE_002',
    'XXX.XXX.XXX-02',
    'CUST_002',
    350.00, 350.00, 'BRL', 'CARTAO_DEBITO', 'APP', 'APPROVED',
    'Rio de Janeiro', 'RJ', 'BRA',
    FALSE, 0.08, 'LOW', '1.0.0',
    NOW() - INTERVAL '2 hours'
),
-- Transação suspeita (fraude)
(
    'TXN_SAMPLE_003',
    'XXX.XXX.XXX-03',
    'CUST_003',
    15000.00, 15000.00, 'BRL', 'PIX', 'APP', 'BLOCKED',
    'São Paulo', 'SP', 'BRA',
    TRUE, 0.89, 'CRITICAL', '1.0.0',
    NOW() - INTERVAL '30 minutes'
)
ON CONFLICT (transaction_id) DO NOTHING;

-- Detecção de fraude para transação suspeita
INSERT INTO fraud_detections (
    transaction_id,
    fraud_probability, risk_score, risk_level,
    detection_reason, model_version, model_type,
    explanation_text, lgpd_compliant,
    processing_time_ms
)
SELECT 
    t.id,
    0.89, 0.89, 'CRITICAL',
    ARRAY['Valor alto (R$ 15.000)', 'Horário atípico', 'Novo destinatário'],
    '1.0.0', 'ENSEMBLE',
    'Transação de alto valor (R$ 15.000) realizada em horário atípico para novo destinatário. Padrão consistente com fraude PIX.',
    TRUE,
    28.5
FROM transactions t
WHERE t.transaction_id = 'TXN_SAMPLE_003'
ON CONFLICT DO NOTHING;

-- =====================================================
-- ALERTA DE EXEMPLO
-- =====================================================

INSERT INTO alerts (
    alert_type, severity, title, message,
    transaction_id, status
)
SELECT
    'FRAUD', 'CRITICAL',
    'Transação PIX de Alto Risco Bloqueada',
    'Transação de R$ 15.000 bloqueada automaticamente. Score de risco: 89%. Revisão manual recomendada.',
    t.id,
    'OPEN'
FROM transactions t
WHERE t.transaction_id = 'TXN_SAMPLE_003'
ON CONFLICT DO NOTHING;

-- =====================================================
-- MENSAGEM FINAL
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '=====================================================';
    RAISE NOTICE 'Dados iniciais inseridos com sucesso!';
    RAISE NOTICE '';
    RAISE NOTICE 'Usuários criados:';
    RAISE NOTICE '  - admin (senha: admin123)';
    RAISE NOTICE '  - analista (senha: admin123)';
    RAISE NOTICE '';
    RAISE NOTICE 'IMPORTANTE: Altere as senhas em produção!';
    RAISE NOTICE '=====================================================';
END $$;
