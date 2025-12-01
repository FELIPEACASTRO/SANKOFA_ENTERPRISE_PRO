-- =====================================================
-- SANKOFA ENTERPRISE PRO - DADOS DE SEED
-- =====================================================
-- 
-- Este arquivo contém TODOS os dados necessários para
-- popular o banco de dados e subir a aplicação
--
-- Versão: 12.0
-- Data: Dezembro 2025
-- =====================================================

-- =====================================================
-- 1. USUÁRIOS INICIAIS
-- =====================================================

INSERT INTO users (username, email, password_hash, name, role, is_active) VALUES
('admin', 'admin@sankofa.com.br', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4.W8FnkqHQK3fIzy', 'Administrador Sistema', 'admin', TRUE),
('analista', 'analista@sankofa.com.br', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4.W8FnkqHQK3fIzy', 'Analista de Fraude', 'analyst', TRUE),
('viewer', 'viewer@sankofa.com.br', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4.W8FnkqHQK3fIzy', 'Visualizador', 'viewer', TRUE)
ON CONFLICT (username) DO NOTHING;

-- =====================================================
-- 2. CONFIGURAÇÕES DO SISTEMA
-- =====================================================

INSERT INTO system_configs (config_key, config_value) VALUES
('ml_threshold', '{"value": 0.5, "description": "Threshold de detecção ML"}'),
('bacen_pix_limit_night', '{"value": 1000, "description": "Limite PIX noturno BACEN", "hours": {"start": 20, "end": 6}}'),
('bacen_pix_limit_day', '{"value": 20000, "description": "Limite PIX diurno BACEN"}'),
('fraud_alert_threshold', '{"critical": 0.9, "high": 0.75, "medium": 0.5}'),
('velocity_limits', '{"1h": 5, "24h": 20, "description": "Limites de velocidade de transações"}'),
('enabled_channels', '{"channels": ["PIX", "TED", "CARTAO_CREDITO", "CARTAO_DEBITO", "BOLETO", "ATM"]}'),
('cache_ttl', '{"rules": 30, "transactions": 60, "dashboard": 30}')
ON CONFLICT (config_key) DO UPDATE SET config_value = EXCLUDED.config_value;

-- =====================================================
-- 3. MÉTRICAS DO MODELO ML
-- =====================================================

INSERT INTO model_metrics (model_version, accuracy, precision_score, recall, f1_score, roc_auc, threshold, samples_used, fraud_ratio) VALUES
('1.0.0', 0.9500, 0.9362, 0.8980, 0.9167, 0.9952, 0.5000, 100000, 0.0150),
('2.0.0', 0.9650, 0.9485, 0.9220, 0.9351, 0.9975, 0.5000, 150000, 0.0145)
ON CONFLICT DO NOTHING;

-- =====================================================
-- 4. TRANSAÇÕES DE EXEMPLO
-- =====================================================

INSERT INTO transactions (transaction_id, amount, channel, type, status, risk_score, is_fraud, cpf, location, timestamp) VALUES
('TXN_SAMPLE_001', 1500.00, 'PIX', 'PAYMENT', 'APPROVED', 0.12, FALSE, '***.***.*01-**', 'Sao Paulo, SP', NOW() - INTERVAL '1 hour'),
('TXN_SAMPLE_002', 350.00, 'CARTAO_DEBITO', 'PAYMENT', 'APPROVED', 0.08, FALSE, '***.***.*02-**', 'Rio de Janeiro, RJ', NOW() - INTERVAL '2 hours'),
('TXN_SAMPLE_003', 15000.00, 'PIX', 'TRANSFER', 'BLOCKED', 0.89, TRUE, '***.***.*03-**', 'Sao Paulo, SP', NOW() - INTERVAL '30 minutes'),
('TXN_SAMPLE_004', 500.00, 'TED', 'TRANSFER', 'APPROVED', 0.15, FALSE, '***.***.*04-**', 'Belo Horizonte, MG', NOW() - INTERVAL '3 hours'),
('TXN_SAMPLE_005', 8500.00, 'PIX', 'PAYMENT', 'REVIEW', 0.72, FALSE, '***.***.*05-**', 'Curitiba, PR', NOW() - INTERVAL '45 minutes')
ON CONFLICT (transaction_id) DO NOTHING;

-- =====================================================
-- 5. ALERTAS DE EXEMPLO
-- =====================================================

INSERT INTO alerts (alert_id, title, description, type, severity, status, transaction_id, amount_involved, recommended_action, tags) VALUES
('ALERT_001', 'PIX Alto Risco Bloqueado', 'Transação de R$ 15.000 bloqueada automaticamente. Score: 89%', 'FRAUD', 'CRITICAL', 'novo', 'TXN_SAMPLE_003', 15000.00, 'Revisar e confirmar com cliente', ARRAY['pix', 'alto_valor', 'automatico']),
('ALERT_002', 'Transação em Revisão', 'PIX de R$ 8.500 para análise manual', 'FRAUD', 'HIGH', 'em_analise', 'TXN_SAMPLE_005', 8500.00, 'Validar com cliente via telefone', ARRAY['pix', 'review'])
ON CONFLICT (alert_id) DO NOTHING;

-- =====================================================
-- 6. VIP LIST (WHITELIST)
-- =====================================================

INSERT INTO vip_list (identifier, identifier_type, reason, added_by) VALUES
('11111111111', 'CPF', 'Cliente VIP - histórico excelente', 'admin'),
('22222222222', 'CPF', 'Funcionário do banco', 'admin'),
('00000000000001', 'CNPJ', 'Empresa parceira', 'admin')
ON CONFLICT DO NOTHING;

-- =====================================================
-- 7. HOT LIST (BLACKLIST)
-- =====================================================

INSERT INTO hot_list (identifier, identifier_type, reason, added_by) VALUES
('99999999999', 'CPF', 'CPF envolvido em fraude confirmada', 'sistema'),
('88888888888', 'CPF', 'Múltiplas tentativas de fraude', 'analista'),
('192.168.1.100', 'IP', 'IP usado em ataques', 'sistema')
ON CONFLICT DO NOTHING;

-- =====================================================
-- 8. ROLES RBAC
-- =====================================================

INSERT INTO rbac_roles (name, description, permissions, is_system_role) VALUES
('admin', 'Administrador do sistema com acesso total', 
 '["transactions:read", "transactions:write", "transactions:delete", "alerts:read", "alerts:write", "alerts:delete", "rules:read", "rules:write", "rules:delete", "users:read", "users:write", "users:delete", "settings:read", "settings:write", "reports:read", "reports:write", "audit:read"]'::jsonb, 
 TRUE),
('analyst', 'Analista de fraude com acesso a transações e alertas', 
 '["transactions:read", "transactions:write", "alerts:read", "alerts:write", "rules:read", "reports:read"]'::jsonb, 
 TRUE),
('viewer', 'Visualizador com acesso somente leitura', 
 '["transactions:read", "alerts:read", "rules:read", "reports:read"]'::jsonb, 
 TRUE),
('manager', 'Gerente com acesso a relatórios e configurações', 
 '["transactions:read", "alerts:read", "alerts:write", "rules:read", "rules:write", "settings:read", "reports:read", "reports:write"]'::jsonb, 
 TRUE),
('compliance', 'Compliance com acesso a auditoria e relatórios', 
 '["transactions:read", "alerts:read", "rules:read", "audit:read", "reports:read", "reports:write"]'::jsonb, 
 TRUE)
ON CONFLICT (name) DO UPDATE SET permissions = EXCLUDED.permissions;

-- =====================================================
-- 9. AUDIT LOG INICIAL
-- =====================================================

INSERT INTO audit_logs (action, user_id, details, ip_address) VALUES
('SYSTEM_INIT', 'system', 'Sistema inicializado com sucesso - Versão 12.0', '127.0.0.1'),
('DATA_SEED', 'system', 'Dados de seed inseridos', '127.0.0.1')
ON CONFLICT DO NOTHING;

-- =====================================================
-- MENSAGEM FINAL
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '=====================================================';
    RAISE NOTICE 'Sankofa Enterprise Pro - Seed Data Loaded';
    RAISE NOTICE '';
    RAISE NOTICE 'Usuarios:';
    RAISE NOTICE '  - admin (senha: admin123)';
    RAISE NOTICE '  - analista (senha: admin123)';
    RAISE NOTICE '  - viewer (senha: admin123)';
    RAISE NOTICE '';
    RAISE NOTICE 'IMPORTANTE: Altere as senhas em producao!';
    RAISE NOTICE '=====================================================';
END $$;
