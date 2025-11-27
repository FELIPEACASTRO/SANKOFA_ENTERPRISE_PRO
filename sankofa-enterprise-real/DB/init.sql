-- =====================================================
-- Sankofa Enterprise Pro - Inicialização do Banco
-- =====================================================
-- 
-- Execute este script para inicializar um banco novo
-- 
-- Uso: psql -U postgres -f init.sql
-- =====================================================

-- Criar banco de dados (se necessário)
-- SELECT 'CREATE DATABASE sankofa_fraud' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'sankofa_fraud')\gexec

-- Configurações de sessão
SET statement_timeout = '60s';
SET lock_timeout = '10s';
SET timezone = 'America/Sao_Paulo';

-- Informações
DO $$
BEGIN
    RAISE NOTICE '=====================================================';
    RAISE NOTICE 'Sankofa Enterprise Pro - Database Initialization';
    RAISE NOTICE 'Version: 12.0';
    RAISE NOTICE 'Date: %', NOW();
    RAISE NOTICE '=====================================================';
END $$;

-- Verificar extensões
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'uuid-ossp') THEN
        CREATE EXTENSION "uuid-ossp";
        RAISE NOTICE 'Extension uuid-ossp created';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pgcrypto') THEN
        CREATE EXTENSION "pgcrypto";
        RAISE NOTICE 'Extension pgcrypto created';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm') THEN
        CREATE EXTENSION "pg_trgm";
        RAISE NOTICE 'Extension pg_trgm created';
    END IF;
END $$;

-- Criar role para aplicação (se não existir)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'sankofa_app') THEN
        CREATE ROLE sankofa_app WITH LOGIN PASSWORD 'change_me_in_production';
        RAISE NOTICE 'Role sankofa_app created';
    END IF;
END $$;

-- Criar role para leitura apenas (relatórios)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'sankofa_readonly') THEN
        CREATE ROLE sankofa_readonly WITH LOGIN PASSWORD 'change_me_in_production';
        RAISE NOTICE 'Role sankofa_readonly created';
    END IF;
END $$;

-- Aplicar schema
\echo 'Applying schema...'
\i schema.sql

-- Aplicar seeds (dados iniciais)
\echo 'Applying initial data...'
\i seeds/initial_data.sql

-- Conceder permissões
GRANT USAGE ON SCHEMA public TO sankofa_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO sankofa_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO sankofa_app;

GRANT USAGE ON SCHEMA public TO sankofa_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO sankofa_readonly;

-- Estatísticas
DO $$
DECLARE
    table_count INTEGER;
    index_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO table_count FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
    
    SELECT COUNT(*) INTO index_count FROM pg_indexes 
    WHERE schemaname = 'public';
    
    RAISE NOTICE '=====================================================';
    RAISE NOTICE 'Initialization Complete!';
    RAISE NOTICE 'Tables created: %', table_count;
    RAISE NOTICE 'Indexes created: %', index_count;
    RAISE NOTICE '=====================================================';
END $$;
