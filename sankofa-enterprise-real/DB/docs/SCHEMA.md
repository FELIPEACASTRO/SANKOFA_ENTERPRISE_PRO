# Sankofa Enterprise Pro - Database Schema Documentation

## Visão Geral

Este documento descreve o schema do banco de dados PostgreSQL utilizado pelo sistema Sankofa Enterprise Pro para detecção de fraude bancária.

## Diagrama ER (Simplificado)

```
┌─────────────────┐     ┌──────────────────────┐
│   customers     │     │    transactions      │
├─────────────────┤     ├──────────────────────┤
│ id (PK)         │◄────│ customer_id (FK)     │
│ cpf_hash        │     │ id (PK)              │
│ risk_profile    │     │ transaction_id       │
│ fraud_count     │     │ amount               │
│ transaction_cnt │     │ is_fraud             │
└─────────────────┘     │ fraud_score          │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  fraud_detections    │
                        ├──────────────────────┤
                        │ id (PK)              │
                        │ transaction_id (FK)  │
                        │ fraud_probability    │
                        │ explanation_text     │
                        └──────────────────────┘

┌─────────────────┐     ┌──────────────────────┐
│     users       │     │    audit_trail       │
├─────────────────┤     ├──────────────────────┤
│ id (PK)         │────►│ user_id              │
│ username        │     │ id (PK)              │
│ email           │     │ event_type           │
│ role            │     │ action               │
│ is_active       │     │ timestamp            │
└─────────────────┘     └──────────────────────┘

┌─────────────────┐     ┌──────────────────────┐
│ model_versions  │     │      alerts          │
├─────────────────┤     ├──────────────────────┤
│ id (PK)         │     │ id (PK)              │
│ version         │     │ transaction_id (FK)  │
│ model_type      │     │ alert_type           │
│ metrics         │     │ severity             │
│ is_active       │     │ status               │
└─────────────────┘     └──────────────────────┘
```

## Tabelas Principais

### 1. transactions

Tabela central que armazena todas as transações financeiras.

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `id` | UUID | Identificador único interno |
| `transaction_id` | VARCHAR(100) | ID externo da transação |
| `cliente_cpf` | VARCHAR(14) | CPF mascarado (LGPD) |
| `customer_id` | VARCHAR(100) | ID do cliente |
| `amount` / `valor` | DECIMAL(15,2) | Valor da transação |
| `tipo_transacao` | VARCHAR(50) | PIX, TED, CARTAO_CREDITO, etc. |
| `canal` | VARCHAR(50) | APP, WEB, ATM, AGENCIA |
| `status` | VARCHAR(20) | PENDING, APPROVED, BLOCKED, REVIEW |
| `is_fraud` | BOOLEAN | Resultado da análise |
| `fraud_score` | DECIMAL(5,4) | Score de risco (0-1) |
| `risk_level` | VARCHAR(20) | LOW, MEDIUM, HIGH, CRITICAL |
| `timestamp` | TIMESTAMPTZ | Data/hora da transação |

**Índices:**
- `idx_transactions_timestamp` - Consultas por período
- `idx_transactions_cliente_cpf` - Consultas por cliente
- `idx_transactions_is_fraud` - Filtro de fraudes
- `idx_transactions_canal` - Filtro por canal

### 2. fraud_detections

Detalhes das detecções de fraude.

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `id` | UUID | Identificador único |
| `transaction_id` | UUID | FK para transactions |
| `fraud_probability` | DECIMAL(5,4) | Probabilidade de fraude |
| `risk_score` | DECIMAL(5,4) | Score de risco |
| `detection_reason` | TEXT[] | Razões da detecção |
| `explanation_text` | TEXT | Explicação LGPD |
| `model_version` | VARCHAR(20) | Versão do modelo |
| `processing_time_ms` | DECIMAL(10,2) | Tempo de processamento |

### 3. customers

Perfil de risco dos clientes.

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `id` | VARCHAR(100) | Identificador do cliente |
| `cpf_hash` | VARCHAR(64) | Hash SHA-256 do CPF |
| `risk_profile` | VARCHAR(20) | LOW, MEDIUM, HIGH |
| `transaction_count` | INTEGER | Total de transações |
| `total_amount` | DECIMAL(15,2) | Valor total movimentado |
| `fraud_count` | INTEGER | Quantidade de fraudes |
| `fraud_rate` | DECIMAL(5,4) | Taxa de fraude |

### 4. audit_trail

Log de auditoria para compliance (LGPD/BACEN).

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `id` | UUID | Identificador único |
| `event_type` | VARCHAR(100) | Tipo do evento |
| `action` | VARCHAR(50) | CREATE, READ, UPDATE, DELETE |
| `user_id` | VARCHAR(100) | Usuário que executou |
| `resource_type` | VARCHAR(100) | Tipo do recurso |
| `resource_id` | VARCHAR(100) | ID do recurso |
| `details` | JSONB | Detalhes do evento |
| `timestamp` | TIMESTAMPTZ | Data/hora do evento |
| `retention_until` | TIMESTAMPTZ | Retenção até (7 anos) |

**Importante:** Esta tabela é append-only para compliance.

### 5. users

Usuários do sistema.

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `id` | UUID | Identificador único |
| `username` | VARCHAR(100) | Nome de usuário |
| `email` | VARCHAR(255) | Email |
| `password_hash` | VARCHAR(255) | Hash da senha (bcrypt) |
| `role` | VARCHAR(50) | ADMIN, ANALYST, VIEWER |
| `is_active` | BOOLEAN | Usuário ativo |
| `mfa_enabled` | BOOLEAN | MFA habilitado |

### 6. model_versions

Registro de versões dos modelos ML.

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `id` | UUID | Identificador único |
| `version` | VARCHAR(20) | Versão (ex: 1.0.0) |
| `model_type` | VARCHAR(50) | ENSEMBLE, LSTM, GNN |
| `metrics` | JSONB | Métricas do modelo |
| `status` | VARCHAR(20) | TRAINING, PRODUCTION |
| `is_active` | BOOLEAN | Modelo em produção |

## Views

### fraud_statistics

Estatísticas diárias de fraude.

```sql
SELECT * FROM fraud_statistics ORDER BY date DESC LIMIT 7;
```

### channel_statistics

Estatísticas por canal (últimos 30 dias).

```sql
SELECT * FROM channel_statistics;
```

### high_risk_customers

Top 100 clientes de alto risco.

```sql
SELECT * FROM high_risk_customers;
```

## Triggers

### update_updated_at_column

Atualiza automaticamente o campo `updated_at` em UPDATE.

Aplicado em:
- `transactions`
- `customers`
- `users`

### update_customer_stats

Atualiza estatísticas do cliente após INSERT em `transactions`.

## Índices de Performance

### Índices Simples

| Tabela | Índice | Colunas |
|--------|--------|---------|
| transactions | idx_transactions_timestamp | timestamp DESC |
| transactions | idx_transactions_cliente_cpf | cliente_cpf |
| transactions | idx_transactions_is_fraud | is_fraud |
| audit_trail | idx_audit_trail_timestamp | timestamp DESC |
| users | idx_users_email | email |

### Índices Compostos

| Tabela | Índice | Colunas |
|--------|--------|---------|
| transactions | idx_transactions_customer_timestamp | customer_id, timestamp DESC |
| transactions | idx_transactions_fraud_timestamp | is_fraud, timestamp DESC |
| audit_trail | idx_audit_trail_resource | resource_type, resource_id |

## Particionamento (Recomendado para Produção)

Para volumes de 300M+ req/dia, particionar por mês:

```sql
-- Criar tabela particionada
CREATE TABLE transactions_partitioned (
    LIKE transactions INCLUDING ALL
) PARTITION BY RANGE (timestamp);

-- Criar partições mensais
CREATE TABLE transactions_2025_11 PARTITION OF transactions_partitioned
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
```

## Retenção de Dados

| Tabela | Retenção | Justificativa |
|--------|----------|---------------|
| audit_trail | 7 anos | BACEN Resolução 6/2023 |
| transactions | 5 anos | Compliance fiscal |
| fraud_detections | 5 anos | Análise forense |
| rate_limits | 24 horas | Temporário |

## Considerações de Segurança

1. **CPF Mascarado:** Armazenado como XXX.XXX.XXX-XX
2. **CPF Hash:** SHA-256 para buscas sem expor dado
3. **Senhas:** bcrypt com salt
4. **Audit Trail:** Append-only, sem DELETE/UPDATE
5. **API Keys:** Hash SHA-256, nunca em plaintext

## Backup

### Automático (recomendado)

```bash
# Backup diário às 3h
0 3 * * * /path/to/DB/backup/backup.sh /path/to/backups
```

### Manual

```bash
./DB/backup/backup.sh
```

## Monitoramento

### Queries de Saúde

```sql
-- Tamanho do banco
SELECT pg_size_pretty(pg_database_size(current_database()));

-- Transações por hora (última hora)
SELECT COUNT(*) FROM transactions 
WHERE timestamp > NOW() - INTERVAL '1 hour';

-- Taxa de fraude (últimas 24h)
SELECT 
    COUNT(*) as total,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as frauds,
    ROUND(AVG(CASE WHEN is_fraud THEN 1 ELSE 0 END) * 100, 2) as fraud_rate
FROM transactions
WHERE timestamp > NOW() - INTERVAL '24 hours';
```
