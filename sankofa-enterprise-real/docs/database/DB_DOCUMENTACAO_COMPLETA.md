# Documentacao Completa do Banco de Dados

## Sankofa Enterprise Pro v12.0

**Versao:** 12.0  
**Banco:** PostgreSQL (Neon-backed)  
**Cache:** Redis Cluster  
**Ultima Atualizacao:** 29 de Novembro de 2025

---

## 1. Introducao

O Sankofa Enterprise Pro utiliza PostgreSQL como banco de dados principal e Redis para cache. O sistema foi projetado para suportar 300M+ transacoes/dia com latencia PIX <50ms.

### 1.1. Visao Geral da Arquitetura

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|  Frontend React  +---->+  Backend Flask   +---->+  PostgreSQL DB   |
|   (16 Paginas)   |     |  (78+ Endpoints) |     |  (14 Tabelas)    |
|                  |     |                  |     |                  |
+------------------+     +--------+---------+     +------------------+
                                  |
                                  v
                         +------------------+
                         |                  |
                         |   Redis Cache    |
                         |  (Session/Cache) |
                         |                  |
                         +------------------+
```

### 1.2. Caracteristicas do Banco

- **Tipo:** PostgreSQL 15+
- **Provider:** Neon (serverless)
- **Conexao:** DATABASE_URL (env variable)
- **Extensoes:** uuid-ossp, pgcrypto, pg_trgm

---

## 2. Dicionario Completo de Tabelas

### 2.1. Tabela: `transactions`

**Proposito:** Armazena todas as transacoes financeiras processadas pelo sistema.

| Coluna | Tipo | Nullable | Default | Descricao |
|--------|------|----------|---------|-----------|
| id | integer | NO | autoincrement | ID unico da transacao |
| transaction_id | varchar(100) | NO | - | ID externo da transacao |
| amount | numeric | NO | - | Valor da transacao |
| channel | varchar(50) | NO | - | Canal (APP, WEB, ATM, API) |
| type | varchar(50) | NO | - | Tipo (PIX, TED, CREDITO, DEBITO) |
| status | varchar(50) | NO | - | Status (PENDING, APPROVED, BLOCKED) |
| risk_score | numeric | YES | - | Score de risco (0-100) |
| is_fraud | boolean | YES | false | Flag de fraude |
| cpf | varchar(20) | YES | - | CPF mascarado (XXX.XXX.XXX-XX) |
| location | varchar(100) | YES | - | Localizacao da transacao |
| timestamp | timestamp | YES | CURRENT_TIMESTAMP | Data/hora da transacao |
| processing_time_ms | numeric | YES | - | Tempo de processamento em ms |
| model_version | varchar(20) | YES | - | Versao do modelo usado |
| created_at | timestamp | YES | CURRENT_TIMESTAMP | Data de criacao |

**Indices:**
- `idx_transactions_id` (id) - PK
- `idx_transactions_transaction_id` (transaction_id) - UNIQUE
- `idx_transactions_timestamp` (timestamp DESC)
- `idx_transactions_status` (status)
- `idx_transactions_is_fraud` (is_fraud)

**Regras de Negocio:**
- Transacoes com risk_score >= 70 vao para revisao manual
- Transacoes com risk_score >= 90 sao bloqueadas automaticamente
- CPF sempre armazenado mascarado para LGPD

---

### 2.2. Tabela: `alerts`

**Proposito:** Armazena alertas gerados pelo sistema de deteccao.

| Coluna | Tipo | Nullable | Default | Descricao |
|--------|------|----------|---------|-----------|
| id | integer | NO | autoincrement | ID unico do alerta |
| alert_id | varchar(50) | NO | - | ID externo do alerta |
| title | varchar(255) | NO | - | Titulo do alerta |
| description | text | YES | - | Descricao detalhada |
| type | varchar(50) | NO | - | Tipo (FRAUD, ANOMALY, SYSTEM) |
| severity | varchar(20) | NO | - | Severidade (INFO, WARNING, CRITICAL) |
| status | varchar(50) | YES | 'novo' | Status (novo, acknowledged, resolved) |
| transaction_id | varchar(100) | YES | - | ID da transacao relacionada |
| amount_involved | numeric | YES | - | Valor envolvido |
| recommended_action | text | YES | - | Acao recomendada |
| investigator | varchar(100) | YES | - | Investigador atribuido |
| tags | ARRAY | YES | - | Tags para categorizacao |
| created_at | timestamp | YES | CURRENT_TIMESTAMP | Data de criacao |
| updated_at | timestamp | YES | CURRENT_TIMESTAMP | Data de atualizacao |

**Indices:**
- `idx_alerts_status` (status)
- `idx_alerts_severity` (severity)
- `idx_alerts_created_at` (created_at DESC)

**Regras de Negocio:**
- Alertas CRITICAL devem ser resolvidos em <4 horas (SLA)
- Alertas WARNING devem ser resolvidos em <24 horas
- Alertas nao resolvidos sao escalados automaticamente

---

### 2.3. Tabela: `audit_logs`

**Proposito:** Trilha de auditoria para compliance LGPD/BACEN (append-only).

| Coluna | Tipo | Nullable | Default | Descricao |
|--------|------|----------|---------|-----------|
| id | integer | NO | autoincrement | ID unico do log |
| action | varchar(100) | NO | - | Acao realizada |
| user_id | varchar(100) | YES | - | ID do usuario |
| details | text | YES | - | Detalhes da acao |
| ip_address | varchar(50) | YES | - | IP do usuario |
| created_at | timestamp | YES | CURRENT_TIMESTAMP | Data da acao |

**Indices:**
- `idx_audit_logs_user_id` (user_id)
- `idx_audit_logs_action` (action)
- `idx_audit_logs_created_at` (created_at DESC)

**Regras de Negocio:**
- Logs sao imutaveis (append-only)
- Retencao minima de 7 anos (BACEN)
- Nao pode haver DELETE nesta tabela

---

### 2.4. Tabela: `feedback`

**Proposito:** Armazena feedbacks dos analistas sobre as predicoes (Human-in-the-Loop).

| Coluna | Tipo | Nullable | Default | Descricao |
|--------|------|----------|---------|-----------|
| id | integer | NO | autoincrement | ID unico |
| transaction_id | varchar(100) | NO | - | ID da transacao |
| is_fraud | boolean | NO | - | Classificacao real (analista) |
| analyst_notes | text | YES | - | Notas do analista |
| analyst_id | varchar(100) | YES | - | ID do analista |
| created_at | timestamp | YES | CURRENT_TIMESTAMP | Data do feedback |

**Indices:**
- `idx_feedback_transaction_id` (transaction_id)
- `idx_feedback_analyst_id` (analyst_id)

**Regras de Negocio:**
- Feedbacks sao usados para retraining do modelo
- Cada transacao pode ter multiplos feedbacks
- Feedbacks discordantes geram revisao adicional

---

### 2.5. Tabela: `users`

**Proposito:** Usuarios do sistema com autenticacao e RBAC.

| Coluna | Tipo | Nullable | Default | Descricao |
|--------|------|----------|---------|-----------|
| id | integer | NO | autoincrement | ID unico |
| username | varchar(50) | NO | - | Nome de usuario (unico) |
| email | varchar(255) | YES | - | Email do usuario |
| password_hash | varchar(255) | NO | - | Hash bcrypt da senha |
| name | varchar(100) | NO | - | Nome completo |
| role | varchar(50) | NO | 'viewer' | Role principal |
| is_active | boolean | YES | true | Usuario ativo |
| failed_login_attempts | integer | YES | 0 | Tentativas falhas |
| locked_until | timestamp | YES | - | Bloqueado ate |
| last_login | timestamp | YES | - | Ultimo login |
| created_at | timestamp | YES | now() | Data de criacao |
| updated_at | timestamp | YES | now() | Data de atualizacao |

**Indices:**
- `idx_users_username` (username) - UNIQUE
- `idx_users_email` (email)
- `idx_users_role` (role)

**Regras de Negocio:**
- 5 tentativas falhas = bloqueio por 30 minutos
- Senhas devem ter minimo 8 caracteres
- Roles disponiveis: admin, analyst, ml_engineer, auditor, viewer

---

### 2.6. Tabela: `hard_rules`

**Proposito:** Regras rigidas de bloqueio (sobrepoe ML).

| Coluna | Tipo | Nullable | Default | Descricao |
|--------|------|----------|---------|-----------|
| id | integer | NO | autoincrement | ID unico |
| name | varchar(255) | NO | - | Nome da regra |
| condition | text | NO | - | Condicao (JSON ou expressao) |
| action | varchar(50) | NO | - | Acao (BLOCK, FLAG, ALERT) |
| enabled | boolean | YES | true | Regra ativa |
| created_at | timestamp | YES | CURRENT_TIMESTAMP | Data de criacao |
| updated_at | timestamp | YES | CURRENT_TIMESTAMP | Data de atualizacao |

**Indices:**
- `idx_hard_rules_enabled` (enabled)
- `idx_hard_rules_action` (action)

**Regras de Negocio:**
- Regras sao avaliadas ANTES do modelo ML
- Regras BLOCK impedem a transacao imediatamente
- Historico de alteracoes e auditado

---

### 2.7. Tabela: `vip_list` (Lista Branca)

**Proposito:** Lista de entidades VIP com tratamento diferenciado.

| Coluna | Tipo | Nullable | Default | Descricao |
|--------|------|----------|---------|-----------|
| id | integer | NO | autoincrement | ID unico |
| identifier | varchar(100) | NO | - | CPF, CNPJ ou outro identificador |
| identifier_type | varchar(20) | NO | - | Tipo (CPF, CNPJ, DEVICE_ID) |
| reason | text | YES | - | Motivo da inclusao |
| added_by | varchar(100) | YES | - | Usuario que adicionou |
| created_at | timestamp | YES | CURRENT_TIMESTAMP | Data de adicao |

**Indices:**
- `idx_vip_list_identifier` (identifier, identifier_type) - UNIQUE

**Regras de Negocio:**
- Transacoes de VIPs recebem score_adjustment -20
- Inclusao requer aprovacao de admin
- Revisao anual obrigatoria

---

### 2.8. Tabela: `hot_list` (Lista Negra)

**Proposito:** Lista de entidades bloqueadas.

| Coluna | Tipo | Nullable | Default | Descricao |
|--------|------|----------|---------|-----------|
| id | integer | NO | autoincrement | ID unico |
| identifier | varchar(100) | NO | - | CPF, CNPJ ou outro identificador |
| identifier_type | varchar(20) | NO | - | Tipo (CPF, CNPJ, DEVICE_ID) |
| reason | text | YES | - | Motivo da inclusao |
| added_by | varchar(100) | YES | - | Usuario que adicionou |
| created_at | timestamp | YES | CURRENT_TIMESTAMP | Data de adicao |

**Indices:**
- `idx_hot_list_identifier` (identifier, identifier_type) - UNIQUE

**Regras de Negocio:**
- Transacoes de HOT LIST sao bloqueadas automaticamente
- Alerta CRITICAL e gerado automaticamente
- Inclusao requer auditoria

---

### 2.9. Tabela: `model_metrics`

**Proposito:** Historico de metricas do modelo ML.

| Coluna | Tipo | Nullable | Default | Descricao |
|--------|------|----------|---------|-----------|
| id | integer | NO | autoincrement | ID unico |
| model_version | varchar(20) | NO | - | Versao do modelo |
| accuracy | numeric | YES | - | Accuracy |
| precision_score | numeric | YES | - | Precision |
| recall | numeric | YES | - | Recall |
| f1_score | numeric | YES | - | F1-Score |
| roc_auc | numeric | YES | - | ROC-AUC |
| threshold | numeric | YES | - | Threshold usado |
| samples_used | integer | YES | - | Amostras de treinamento |
| fraud_ratio | numeric | YES | - | Proporcao de fraudes |
| created_at | timestamp | YES | CURRENT_TIMESTAMP | Data do registro |

**Indices:**
- `idx_model_metrics_version` (model_version)
- `idx_model_metrics_created_at` (created_at DESC)

**Regras de Negocio:**
- Metricas abaixo do threshold geram alerta
- Historico usado para drift detection
- Retencao minima de 2 anos

---

### 2.10. Tabela: `cpf_tokens`

**Proposito:** Tokenizacao de CPFs para compliance LGPD.

| Coluna | Tipo | Nullable | Default | Descricao |
|--------|------|----------|---------|-----------|
| token | varchar(50) | NO | - | Token unico (PK) |
| encrypted_cpf | bytea | NO | - | CPF criptografado (AES-256) |
| cpf_hash | varchar(64) | NO | - | Hash SHA-256 do CPF |
| created_at | timestamp | YES | now() | Data de criacao |
| expires_at | timestamp | YES | - | Data de expiracao |
| access_count | integer | YES | 0 | Contador de acessos |
| last_accessed | timestamp | YES | - | Ultimo acesso |
| metadata | jsonb | YES | '{}' | Metadados adicionais |

**Indices:**
- `idx_cpf_tokens_hash` (cpf_hash)
- `idx_cpf_tokens_expires` (expires_at)

**Regras de Negocio:**
- CPFs nunca sao armazenados em texto claro
- Tokens expiram em 90 dias por padrao
- Desencriptacao requer auditoria

---

### 2.11. Tabela: `cpf_access_log`

**Proposito:** Log de acessos a CPFs desencriptados.

| Coluna | Tipo | Nullable | Default | Descricao |
|--------|------|----------|---------|-----------|
| id | varchar(36) | NO | gen_random_uuid() | ID unico |
| token | varchar(50) | NO | - | Token do CPF acessado |
| action | varchar(50) | NO | - | Acao (decrypt, view, export) |
| purpose | text | YES | - | Proposito do acesso |
| user_id | varchar(100) | YES | - | Usuario que acessou |
| ip_address | inet | YES | - | IP do acesso |
| accessed_at | timestamp | YES | now() | Data/hora do acesso |
| metadata | jsonb | YES | '{}' | Metadados adicionais |

**Indices:**
- `idx_cpf_access_token` (token)
- `idx_cpf_access_user` (user_id)
- `idx_cpf_access_time` (accessed_at DESC)

**Regras de Negocio:**
- Todo acesso a CPF desencriptado e logado
- Logs sao imutaveis
- Retencao de 7 anos (LGPD)

---

### 2.12. Tabela: `rbac_roles`

**Proposito:** Definicao de roles do sistema RBAC.

| Coluna | Tipo | Nullable | Default | Descricao |
|--------|------|----------|---------|-----------|
| id | varchar(36) | NO | gen_random_uuid() | ID unico |
| name | varchar(100) | NO | - | Nome do role (unico) |
| description | text | YES | - | Descricao do role |
| permissions | jsonb | NO | '[]' | Lista de permissoes |
| is_system_role | boolean | YES | false | Role de sistema |
| parent_role | varchar(100) | YES | - | Role pai (heranca) |
| created_at | timestamp | YES | now() | Data de criacao |
| updated_at | timestamp | YES | now() | Data de atualizacao |

**Roles Padrao:**
- `admin`: Acesso total
- `analyst`: Analise e revisao
- `ml_engineer`: Gestao de modelos
- `auditor`: Apenas leitura + auditoria
- `viewer`: Apenas leitura

---

### 2.13. Tabela: `rbac_user_roles`

**Proposito:** Atribuicao de roles a usuarios.

| Coluna | Tipo | Nullable | Default | Descricao |
|--------|------|----------|---------|-----------|
| user_id | varchar(36) | NO | - | ID do usuario |
| role_name | varchar(100) | NO | - | Nome do role |
| granted_at | timestamp | YES | now() | Data de atribuicao |
| granted_by | varchar(100) | YES | - | Quem atribuiu |
| expires_at | timestamp | YES | - | Expiracao (opcional) |

**Indices:**
- `idx_rbac_user_roles_user` (user_id, role_name) - PK
- `idx_rbac_user_roles_role` (role_name)

---

### 2.14. Tabela: `rbac_sessions`

**Proposito:** Sessoes ativas dos usuarios.

| Coluna | Tipo | Nullable | Default | Descricao |
|--------|------|----------|---------|-----------|
| session_id | varchar(100) | NO | - | ID da sessao (PK) |
| user_id | varchar(36) | NO | - | ID do usuario |
| ip_address | inet | YES | - | IP da sessao |
| user_agent | text | YES | - | User agent |
| created_at | timestamp | YES | now() | Inicio da sessao |
| expires_at | timestamp | NO | - | Expiracao da sessao |
| last_activity | timestamp | YES | now() | Ultima atividade |
| is_active | boolean | YES | true | Sessao ativa |
| metadata | jsonb | YES | '{}' | Metadados |

**Indices:**
- `idx_rbac_sessions_user` (user_id)
- `idx_rbac_sessions_active` (is_active)
- `idx_rbac_sessions_expires` (expires_at)

---

### 2.15. Tabela: `system_configs`

**Proposito:** Configuracoes do sistema.

| Coluna | Tipo | Nullable | Default | Descricao |
|--------|------|----------|---------|-----------|
| id | integer | NO | autoincrement | ID unico |
| config_key | varchar(100) | NO | - | Chave de configuracao |
| config_value | jsonb | NO | - | Valor (JSON) |
| updated_at | timestamp | YES | CURRENT_TIMESTAMP | Ultima atualizacao |

**Indices:**
- `idx_system_configs_key` (config_key) - UNIQUE

---

## 3. Relacionamentos

```
+----------------+       +----------------+       +----------------+
|  transactions  |       |    alerts      |       |   feedback     |
+----------------+       +----------------+       +----------------+
        |                       |                        |
        +-------+-------+-------+                        |
                |                                        |
                v                                        v
        +----------------+                       +----------------+
        |   audit_logs   |                       |     users      |
        +----------------+                       +----------------+
                                                        |
                                                        v
                                                +----------------+
                                                | rbac_user_roles|
                                                +----------------+
                                                        |
                                                        v
                                                +----------------+
                                                |   rbac_roles   |
                                                +----------------+
```

### 3.1. Relacionamentos Principais

1. **transactions -> alerts**: Uma transacao pode gerar multiplos alertas
2. **transactions -> feedback**: Uma transacao pode ter multiplos feedbacks
3. **users -> rbac_user_roles**: Um usuario pode ter multiplos roles
4. **rbac_user_roles -> rbac_roles**: Cada atribuicao referencia um role
5. **cpf_tokens -> cpf_access_log**: Cada token pode ter multiplos acessos

---

## 4. Redis Cache

### 4.1. Arquitetura Redis

```
Redis Cluster
+-- session:*           # Sessoes de usuario (TTL: 24h)
+-- token:*             # Tokens JWT (TTL: 1h)
+-- cache:transaction:* # Cache de transacoes (TTL: 5min)
+-- cache:model:*       # Cache do modelo ML (TTL: 10min)
+-- rate_limit:*        # Rate limiting (TTL: 1min)
+-- lock:*              # Locks distribuidos (TTL: 30s)
```

### 4.2. Chaves Redis

| Padrao | TTL | Descricao |
|--------|-----|-----------|
| `session:{user_id}` | 24h | Dados da sessao do usuario |
| `token:{jwt_id}` | 1h | Token JWT ativo |
| `cache:transaction:{id}` | 5min | Cache de transacao |
| `cache:model:metrics` | 10min | Cache de metricas do modelo |
| `cache:dashboard:summary` | 1min | Cache do dashboard |
| `rate_limit:{ip}` | 1min | Contador de rate limit |
| `lock:{resource}` | 30s | Lock distribuido |

### 4.3. Politicas de Cache

1. **Write-through**: Transacoes sao escritas no DB e cache simultaneamente
2. **Cache-aside**: Leituras verificam cache primeiro
3. **TTL-based invalidation**: Cache expira automaticamente
4. **Manual invalidation**: Atualizacoes invalidam cache relacionado

### 4.4. Sincronismo DB <-> Redis

```python
# Fluxo de escrita
def save_transaction(transaction):
    # 1. Salva no PostgreSQL
    db.insert(transaction)
    
    # 2. Atualiza cache Redis
    redis.set(f"cache:transaction:{transaction.id}", transaction, ttl=300)
    
    # 3. Invalida cache de dashboard
    redis.delete("cache:dashboard:summary")
```

---

## 5. Seguranca do Banco

### 5.1. Criptografia

- **Em repouso**: Neon encrypted at rest (AES-256)
- **Em transito**: TLS 1.3 obrigatorio
- **Dados sensiveis**: CPFs tokenizados + criptografados (AES-256-GCM)

### 5.2. Acesso

- **Conexao**: Via DATABASE_URL com SSL
- **Usuarios DB**: Application user com privilegios minimos
- **RBAC**: Controle de acesso via aplicacao

### 5.3. Auditoria

- Todas as operacoes CRUD sao logadas
- Logs sao imutaveis (append-only)
- Retencao de 7 anos (BACEN)

---

## 6. Performance

### 6.1. Indices Criticos

Todos os indices foram projetados para as queries mais frequentes:

```sql
-- Query mais frequente: listar transacoes recentes
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp DESC);

-- Query de filtro: transacoes fraudulentas
CREATE INDEX idx_transactions_is_fraud ON transactions(is_fraud) WHERE is_fraud = true;

-- Query de busca: por CPF
CREATE INDEX idx_transactions_cpf ON transactions(cpf);
```

### 6.2. Vacuum e Analyze

```sql
-- Configuracao recomendada
ALTER TABLE transactions SET (autovacuum_vacuum_scale_factor = 0.01);
ALTER TABLE transactions SET (autovacuum_analyze_scale_factor = 0.005);
```

### 6.3. Conexoes

- **Pool size**: 20 conexoes
- **Max connections**: 100
- **Timeout**: 30 segundos

---

## 7. Queries Criticas do Backend

### 7.1. Dashboard Summary

```sql
SELECT 
    COUNT(*) as total_transactions,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count,
    AVG(risk_score) as avg_risk_score,
    AVG(processing_time_ms) as avg_processing_time
FROM transactions
WHERE timestamp >= NOW() - INTERVAL '24 hours';
```

### 7.2. Lista de Transacoes com Paginacao

```sql
SELECT t.*, a.severity as alert_severity
FROM transactions t
LEFT JOIN alerts a ON t.transaction_id = a.transaction_id
WHERE t.status = $1
ORDER BY t.timestamp DESC
LIMIT $2 OFFSET $3;
```

### 7.3. Feedback Analytics

```sql
SELECT 
    f.analyst_id,
    COUNT(*) as total_feedbacks,
    AVG(CASE WHEN t.is_fraud = f.is_fraud THEN 1 ELSE 0 END) as accuracy
FROM feedback f
JOIN transactions t ON f.transaction_id = t.transaction_id
GROUP BY f.analyst_id;
```

---

## 8. Migracoes

### 8.1. Estrutura de Migracoes

```
DB/migrations/
+-- 001_initial_schema.sql
+-- 002_add_fraud_columns.sql
+-- 003_add_rbac_tables.sql
+-- 004_add_feedback_table.sql
```

### 8.2. Executar Migracoes

```bash
cd DB/scripts
python migrate.py --up
```

---

## 9. Backup e Recuperacao

### 9.1. Backup

```bash
# Backup automatico (Neon)
# Neon faz backup point-in-time automaticamente

# Backup manual
cd DB/backup
./backup.sh
```

### 9.2. Restore

```bash
cd DB/backup
./restore.sh backup_file.sql
```

---

## 10. Melhorias Recomendadas

### 10.1. Curto Prazo

1. Adicionar indices parciais para queries especificas
2. Implementar particionamento por data em transactions
3. Adicionar materialized views para dashboard

### 10.2. Medio Prazo

1. Implementar sharding horizontal
2. Adicionar read replicas
3. Migrar para TimescaleDB para time-series

### 10.3. Longo Prazo

1. Event sourcing completo
2. CQRS (Command Query Responsibility Segregation)
3. Multi-region deployment

---

## 11. Problemas Conhecidos

1. **Tabela transactions** crescendo rapidamente - considerar particionamento
2. **Indices grandes** em transactions - monitorar tamanho
3. **Bloat** em audit_logs - vacuum regular necessario

---

## 12. Contato

Para duvidas sobre o banco de dados, consulte a equipe de DBA ou abra um ticket.

---

**Documentacao atualizada em:** 29 de Novembro de 2025  
**Versao do documento:** 1.0
