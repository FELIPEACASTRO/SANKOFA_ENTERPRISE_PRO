# INVENTÁRIO COMPLETO POSTGRESQL - ULTRA MILITAR
## Protocolo MODO MILITAR 3X - DATABASE - FASE 1
## Data: 29/11/2025

---

## RESUMO EXECUTIVO

| Métrica | Quantidade | Status |
|---------|------------|--------|
| **Total de Tabelas** | 16 | ✅ |
| **Total de Colunas** | 116 | ✅ |
| **Índices** | 32 | ⚠️ Faltam índices críticos |
| **Foreign Keys** | 0 | ❌ PROBLEMA CRÍTICO |
| **Triggers** | 0 | ⚠️ Definidos em código |
| **Views** | 0 | OK |

---

## 1. INVENTÁRIO DE TABELAS

### 1.1 Tabelas Core (4)

| Tabela | Registros | Tamanho | Propósito |
|--------|-----------|---------|-----------|
| `transactions` | 4.467 | 880 KB | Transações financeiras |
| `users` | 5 | 112 KB | Usuários do sistema |
| `alerts` | 0 | 24 KB | Alertas de fraude |
| `audit_logs` | 0 | 16 KB | Logs de auditoria |

### 1.2 Tabelas RBAC (5)

| Tabela | Registros | Tamanho | Propósito |
|--------|-----------|---------|-----------|
| `rbac_roles` | 6 | 48 KB | Papéis do sistema |
| `rbac_user_roles` | 5 | 24 KB | Associação usuário-papel |
| `rbac_sessions` | 0 | 40 KB | Sessões ativas |
| `rbac_permissions_override` | 0 | 16 KB | Override de permissões |

### 1.3 Tabelas ML/Fraude (4)

| Tabela | Registros | Tamanho | Propósito |
|--------|-----------|---------|-----------|
| `model_metrics` | 0 | 8 KB | Métricas de modelos ML |
| `feedback` | 0 | 16 KB | Feedback de analistas |
| `hard_rules` | 0 | 16 KB | Regras determinísticas |
| `system_configs` | 0 | 24 KB | Configurações do sistema |

### 1.4 Tabelas de Segurança/Compliance (3)

| Tabela | Registros | Tamanho | Propósito |
|--------|-----------|---------|-----------|
| `cpf_tokens` | 0 | 40 KB | Tokenização de CPF (LGPD) |
| `cpf_access_log` | 0 | 32 KB | Log de acesso a CPF |
| `vip_list` | 0 | 16 KB | Lista VIP |
| `hot_list` | 0 | 16 KB | Lista de alto risco |

---

## 2. DETALHAMENTO DE COLUNAS

### 2.1 Tabela: `transactions` (Principal)

| Coluna | Tipo | Nullable | Default | Índice |
|--------|------|----------|---------|--------|
| `id` | integer | NO | sequence | PK |
| `transaction_id` | varchar(100) | NO | - | UNIQUE |
| `amount` | numeric | NO | - | ❌ |
| `channel` | varchar(50) | NO | - | ❌ FALTANDO |
| `type` | varchar(50) | NO | - | ❌ |
| `status` | varchar(50) | NO | - | ❌ FALTANDO |
| `risk_score` | numeric | YES | - | ❌ |
| `is_fraud` | boolean | YES | false | ❌ |
| `cpf` | varchar(20) | YES | - | ❌ FALTANDO |
| `location` | varchar(100) | YES | - | ❌ |
| `timestamp` | timestamp | YES | CURRENT_TIMESTAMP | ❌ |
| `processing_time_ms` | numeric | YES | - | ❌ |
| `model_version` | varchar(20) | YES | - | ❌ |
| `created_at` | timestamp | YES | CURRENT_TIMESTAMP | ❌ FALTANDO |

**Problemas Identificados:**
- ❌ `channel` não indexado (usado em filtros)
- ❌ `status` não indexado (usado em filtros)
- ❌ `cpf` não indexado (usado em buscas)
- ❌ `created_at` não indexado (usado em ordenação)
- ⚠️ `risk_score` deveria ser NOT NULL
- ⚠️ `timestamp` deveria ser NOT NULL

---

### 2.2 Tabela: `users`

| Coluna | Tipo | Nullable | Default | Índice |
|--------|------|----------|---------|--------|
| `id` | integer | NO | sequence | PK |
| `username` | varchar(50) | NO | - | UNIQUE + IDX |
| `email` | varchar(255) | YES | - | UNIQUE + IDX |
| `password_hash` | varchar(255) | NO | - | ❌ |
| `name` | varchar(100) | NO | - | ❌ |
| `role` | varchar(50) | NO | 'viewer' | IDX |
| `is_active` | boolean | YES | true | ❌ |
| `failed_login_attempts` | integer | YES | 0 | ❌ |
| `locked_until` | timestamp | YES | - | ❌ |
| `last_login` | timestamp | YES | - | ❌ |
| `created_at` | timestamp | YES | now() | ❌ |
| `updated_at` | timestamp | YES | now() | ❌ |

**Status:** ✅ Bem estruturada

---

### 2.3 Tabela: `alerts`

| Coluna | Tipo | Nullable | Default | Índice |
|--------|------|----------|---------|--------|
| `id` | integer | NO | sequence | PK |
| `alert_id` | varchar(50) | NO | - | UNIQUE |
| `title` | varchar(255) | NO | - | ❌ |
| `description` | text | YES | - | ❌ |
| `type` | varchar(50) | NO | - | ❌ |
| `severity` | varchar(20) | NO | - | ❌ FALTANDO |
| `status` | varchar(50) | YES | 'novo' | ❌ FALTANDO |
| `transaction_id` | varchar(100) | YES | - | ❌ FALTANDO (FK) |
| `amount_involved` | numeric | YES | - | ❌ |
| `recommended_action` | text | YES | - | ❌ |
| `investigator` | varchar(100) | YES | - | ❌ |
| `tags` | ARRAY | YES | - | ❌ |
| `created_at` | timestamp | YES | CURRENT_TIMESTAMP | ❌ FALTANDO |
| `updated_at` | timestamp | YES | CURRENT_TIMESTAMP | ❌ |

**Problemas Identificados:**
- ❌ `status` não indexado
- ❌ `severity` não indexado
- ❌ `created_at` não indexado
- ❌ `transaction_id` deveria ser FK para `transactions`

---

### 2.4 Tabela: `cpf_tokens` (LGPD)

| Coluna | Tipo | Nullable | Default | Índice |
|--------|------|----------|---------|--------|
| `token` | varchar(50) | NO | - | PK |
| `encrypted_cpf` | bytea | NO | - | ❌ |
| `cpf_hash` | varchar(64) | NO | - | UNIQUE + IDX |
| `created_at` | timestamptz | YES | now() | ❌ |
| `expires_at` | timestamptz | YES | - | IDX |
| `access_count` | integer | YES | 0 | ❌ |
| `last_accessed` | timestamptz | YES | - | ❌ |
| `metadata` | jsonb | YES | '{}' | ❌ |

**Problemas Identificados:**
- ⚠️ Índice redundante: `cpf_hash` tem UNIQUE + índice comum

---

### 2.5 Tabela: `rbac_roles`

| Coluna | Tipo | Nullable | Default | Índice |
|--------|------|----------|---------|--------|
| `id` | varchar(36) | NO | gen_random_uuid() | PK |
| `name` | varchar(100) | NO | - | UNIQUE |
| `description` | text | YES | - | ❌ |
| `permissions` | jsonb | NO | '[]' | ❌ |
| `is_system_role` | boolean | YES | false | ❌ |
| `parent_role` | varchar(100) | YES | - | ❌ FALTANDO (FK) |
| `created_at` | timestamptz | YES | now() | ❌ |
| `updated_at` | timestamptz | YES | now() | ❌ |

**Problemas Identificados:**
- ❌ `parent_role` deveria ser FK para `rbac_roles.name`

---

### 2.6 Tabela: `rbac_sessions`

| Coluna | Tipo | Nullable | Default | Índice |
|--------|------|----------|---------|--------|
| `session_id` | varchar(100) | NO | - | PK |
| `user_id` | varchar(36) | NO | - | IDX |
| `ip_address` | inet | YES | - | ❌ |
| `user_agent` | text | YES | - | ❌ |
| `created_at` | timestamptz | YES | now() | ❌ |
| `expires_at` | timestamptz | NO | - | IDX |
| `last_activity` | timestamptz | YES | now() | ❌ |
| `is_active` | boolean | YES | true | IDX |
| `metadata` | jsonb | YES | '{}' | ❌ |

**Status:** ✅ Bem indexada

---

## 3. ANÁLISE DE ÍNDICES

### 3.1 Índices Existentes (32)

| Tabela | Índice | Tipo | Colunas |
|--------|--------|------|---------|
| `alerts` | alerts_pkey | PRIMARY | id |
| `alerts` | alerts_alert_id_key | UNIQUE | alert_id |
| `audit_logs` | audit_logs_pkey | PRIMARY | id |
| `cpf_access_log` | cpf_access_log_pkey | PRIMARY | id |
| `cpf_access_log` | idx_cpf_access_log_time | BTREE | accessed_at DESC |
| `cpf_access_log` | idx_cpf_access_log_token | BTREE | token |
| `cpf_tokens` | cpf_tokens_pkey | PRIMARY | token |
| `cpf_tokens` | cpf_tokens_cpf_hash_key | UNIQUE | cpf_hash |
| `cpf_tokens` | idx_cpf_tokens_expires | BTREE | expires_at |
| `cpf_tokens` | idx_cpf_tokens_hash | BTREE | cpf_hash |
| `feedback` | feedback_pkey | PRIMARY | id |
| `hard_rules` | hard_rules_pkey | PRIMARY | id |
| `hot_list` | hot_list_pkey | PRIMARY | id |
| `model_metrics` | model_metrics_pkey | PRIMARY | id |
| `rbac_permissions_override` | rbac_permissions_override_pkey | PRIMARY | (user_id, permission) |
| `rbac_roles` | rbac_roles_pkey | PRIMARY | id |
| `rbac_roles` | rbac_roles_name_key | UNIQUE | name |
| `rbac_sessions` | rbac_sessions_pkey | PRIMARY | session_id |
| `rbac_sessions` | idx_rbac_sessions_user_id | BTREE | user_id |
| `rbac_sessions` | idx_rbac_sessions_expires | BTREE | expires_at |
| `rbac_sessions` | idx_rbac_sessions_active | BTREE | is_active |
| `rbac_user_roles` | rbac_user_roles_pkey | PRIMARY | (user_id, role_name) |
| `system_configs` | system_configs_pkey | PRIMARY | id |
| `system_configs` | system_configs_config_key_key | UNIQUE | config_key |
| `transactions` | transactions_pkey | PRIMARY | id |
| `transactions` | transactions_transaction_id_key | UNIQUE | transaction_id |
| `users` | users_pkey | PRIMARY | id |
| `users` | users_username_key | UNIQUE | username |
| `users` | users_email_key | UNIQUE | email |
| `users` | idx_users_username | BTREE | username |
| `users` | idx_users_email | BTREE | email |
| `users` | idx_users_role | BTREE | role |
| `vip_list` | vip_list_pkey | PRIMARY | id |

### 3.2 Índices FALTANTES (Críticos)

| Tabela | Coluna | Tipo Sugerido | Prioridade |
|--------|--------|---------------|------------|
| `transactions` | channel | BTREE | **ALTA** |
| `transactions` | status | BTREE | **ALTA** |
| `transactions` | cpf | BTREE | **ALTA** |
| `transactions` | created_at | BTREE DESC | **ALTA** |
| `transactions` | (channel, status) | COMPOSITE | MÉDIA |
| `transactions` | (status, created_at) | COMPOSITE | MÉDIA |
| `alerts` | status | BTREE | **ALTA** |
| `alerts` | severity | BTREE | MÉDIA |
| `alerts` | created_at | BTREE DESC | **ALTA** |
| `audit_logs` | action | BTREE | MÉDIA |
| `audit_logs` | created_at | BTREE DESC | **ALTA** |
| `audit_logs` | user_id | BTREE | MÉDIA |
| `feedback` | transaction_id | BTREE | **ALTA** |
| `feedback` | created_at | BTREE DESC | MÉDIA |
| `hot_list` | identifier | BTREE | **ALTA** |
| `vip_list` | identifier | BTREE | **ALTA** |

---

## 4. ANÁLISE DE FOREIGN KEYS

### 4.1 Foreign Keys Inexistentes (PROBLEMA CRÍTICO)

| Tabela | Coluna | Deveria referenciar |
|--------|--------|---------------------|
| `alerts` | transaction_id | transactions(transaction_id) |
| `feedback` | transaction_id | transactions(transaction_id) |
| `rbac_user_roles` | role_name | rbac_roles(name) |
| `rbac_roles` | parent_role | rbac_roles(name) |
| `audit_logs` | user_id | users(id) |
| `cpf_access_log` | user_id | users(id) |

### 4.2 Impacto da Falta de FKs

| Risco | Descrição | Severidade |
|-------|-----------|------------|
| Dados Órfãos | Registros podem referenciar IDs inexistentes | **CRÍTICO** |
| Inconsistência | DELETE em transactions não propaga para alerts/feedback | **ALTO** |
| Integridade | Sem garantia de referência válida | **ALTO** |
| Compliance | Pode violar requisitos de auditoria | **MÉDIO** |

---

## 5. ANÁLISE DE NORMALIZAÇÃO

### 5.1 Primeira Forma Normal (1FN)
| Tabela | Status | Observação |
|--------|--------|------------|
| `transactions` | ✅ | OK |
| `alerts` | ⚠️ | `tags` é ARRAY (aceitável) |
| `rbac_roles` | ⚠️ | `permissions` é JSONB (aceitável) |
| `users` | ✅ | OK |

### 5.2 Segunda Forma Normal (2FN)
| Tabela | Status | Observação |
|--------|--------|------------|
| Todas | ✅ | Sem dependências parciais |

### 5.3 Terceira Forma Normal (3FN)
| Tabela | Status | Observação |
|--------|--------|------------|
| Todas | ✅ | Sem dependências transitivas |

---

## 6. DADOS DE TRANSAÇÕES

### 6.1 Distribuição por Canal/Status

| Canal | Tipo | Status | Count | Avg Amount | Avg Risk |
|-------|------|--------|-------|------------|----------|
| PIX | PAYMENT | FRAUD | 3.080 | R$ 4.630,85 | 0.494 |
| PIX | PAYMENT | APPROVED | 1.204 | R$ 1.739,84 | 0.010 |
| BOLETO | PAYMENT | APPROVED | 74 | R$ 2.340,26 | 0.010 |
| TED | PAYMENT | APPROVED | 72 | R$ 2.216,61 | 0.008 |
| TED | PAYMENT | FRAUD | 14 | R$ 2.299,00 | 0.476 |
| BOLETO | PAYMENT | FRAUD | 14 | R$ 2.227,32 | 0.465 |
| mobile | PAYMENT | FRAUD | 5 | R$ 500,00 | 0.500 |
| mobile | PAYMENT | APPROVED | 1 | R$ 15.000,00 | 0.010 |
| web | PAYMENT | APPROVED | 1 | R$ 50.000,00 | 0.010 |

### 6.2 Observações
- Taxa de fraude alta para testes (69% PIX)
- Latência de processamento = 0 (não registrada)
- Canais inconsistentes: `PIX` vs `mobile` vs `web`

---

## 7. PROBLEMAS E RECOMENDAÇÕES

### 7.1 Problemas Críticos (P0)

| # | Problema | Impacto | Ação |
|---|----------|---------|------|
| 1 | Sem Foreign Keys | Integridade de dados | Adicionar FKs |
| 2 | Índices faltantes em transactions | Performance degradada | Criar índices |
| 3 | Índices faltantes em alerts | Queries lentas | Criar índices |

### 7.2 Problemas Altos (P1)

| # | Problema | Impacto | Ação |
|---|----------|---------|------|
| 4 | Campos nullable que deveriam ser NOT NULL | Dados inconsistentes | ALTER TABLE |
| 5 | Índice redundante cpf_hash | Overhead de escrita | Remover duplicado |
| 6 | Canais de transação inconsistentes | Relatórios incorretos | Normalizar dados |

### 7.3 Melhorias (P2)

| # | Melhoria | Benefício | Ação |
|---|----------|-----------|------|
| 7 | Índices compostos | Query performance | Criar índices |
| 8 | Particionamento de audit_logs | Melhor gestão 7 anos | Implementar |
| 9 | Triggers de updated_at | Rastreabilidade | Já no código |

---

## 8. SCRIPTS DE CORREÇÃO

### 8.1 Criar Índices Faltantes

```sql
-- Índices para transactions
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_channel 
    ON transactions(channel);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_status 
    ON transactions(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_cpf 
    ON transactions(cpf);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_created_at 
    ON transactions(created_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_channel_status 
    ON transactions(channel, status);

-- Índices para alerts
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_status 
    ON alerts(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_severity 
    ON alerts(severity);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_created_at 
    ON alerts(created_at DESC);

-- Índices para audit_logs
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_created_at 
    ON audit_logs(created_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_action 
    ON audit_logs(action);

-- Índices para feedback
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feedback_transaction_id 
    ON feedback(transaction_id);

-- Índices para listas
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_hot_list_identifier 
    ON hot_list(identifier);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vip_list_identifier 
    ON vip_list(identifier);
```

### 8.2 Remover Índice Redundante

```sql
DROP INDEX CONCURRENTLY IF EXISTS idx_cpf_tokens_hash;
```

---

## 9. CONCLUSÃO FASE 1

| Aspecto | Status | Observação |
|---------|--------|------------|
| Inventário Tabelas | ✅ | 16 tabelas mapeadas |
| Inventário Colunas | ✅ | 116 colunas detalhadas |
| Análise de Índices | ⚠️ | 15+ índices faltantes |
| Análise de FKs | ❌ | 0 FKs - CRÍTICO |
| Normalização | ✅ | 3FN respeitada |
| Dados | ⚠️ | Inconsistências menores |

**PRÓXIMA FASE:** Análise de Queries e Regras de Negócio (FASE 2)

---

*Documento gerado pelo Protocolo MODO MILITAR 3X - DATABASE*
*Rigor Absoluto. Zero Gaps. 100% Compliance.*
