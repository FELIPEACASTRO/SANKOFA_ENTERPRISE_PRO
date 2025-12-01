# Sankofa Enterprise Pro - Setup do Banco de Dados

**Versão:** 2.0.0  
**Última Atualização:** Dezembro 2025  
**Status:** Produção

---

## Visão Geral

O Sankofa Enterprise Pro utiliza PostgreSQL como banco de dados principal e Redis como cache distribuído para processamento de alta performance (300M+ transações/dia com latência <50ms).

---

## Arquivos Disponíveis

| Arquivo | Descrição | Linhas |
|---------|-----------|--------|
| `complete_setup.sql` | Schema completo de todas as 16 tabelas | ~240 |
| `seed_data.sql` | Dados iniciais (usuários, configs, exemplos) | ~120 |
| `hard_rules_insert.sql` | 216 regras duras de detecção de fraude | ~220 |
| `REDIS_CONFIG.md` | Configuração completa do cache Redis | ~150 |

---

## Setup Rápido (Replit)

O banco PostgreSQL já está configurado automaticamente no Replit.
A variável `DATABASE_URL` está disponível para conexão.

### 1. Criar as tabelas:
```bash
psql "$DATABASE_URL" -f DB/complete_setup.sql
```

### 2. Inserir dados de seed:
```bash
psql "$DATABASE_URL" -f DB/seed_data.sql
```

### 3. Inserir regras duras (216 regras):
```bash
psql "$DATABASE_URL" -f DB/hard_rules_insert.sql
```

### 4. Verificar instalação:
```bash
psql "$DATABASE_URL" -c "SELECT COUNT(*) as tabelas FROM information_schema.tables WHERE table_schema = 'public';"
psql "$DATABASE_URL" -c "SELECT COUNT(*) as regras FROM hard_rules;"
```

---

## Setup Manual (Ambiente Local)

### 1. Criar banco de dados:
```bash
createdb sankofa_fraud
```

### 2. Configurar variável de ambiente:
```bash
export DATABASE_URL="postgresql://usuario:senha@localhost:5432/sankofa_fraud"
```

### 3. Executar scripts:
```bash
psql "$DATABASE_URL" -f DB/complete_setup.sql
psql "$DATABASE_URL" -f DB/seed_data.sql
psql "$DATABASE_URL" -f DB/hard_rules_insert.sql
```

---

## Arquitetura de Tabelas (16 Tabelas)

### Tabelas Principais (6)

| Tabela | Descrição | Registros Iniciais | Índices |
|--------|-----------|-------------------|---------|
| `users` | Usuários do sistema | 3 | username, email |
| `transactions` | Transações processadas | 5 | id, cpf, created_at |
| `hard_rules` | Regras de detecção | 216 | name, action, enabled |
| `alerts` | Alertas de fraude | 2 | status, created_at |
| `vip_list` | Lista branca (whitelist) | 3 | cpf, account_id |
| `hot_list` | Lista negra (blacklist) | 3 | cpf, account_id |

### Tabelas de Suporte (6)

| Tabela | Descrição | Uso |
|--------|-----------|-----|
| `audit_logs` | Log de auditoria | LGPD/BACEN compliance |
| `feedback` | Feedback de analistas | Treinamento ML |
| `model_metrics` | Métricas do ML | Monitoramento de modelos |
| `system_configs` | Configurações | Parâmetros do sistema |
| `rbac_roles` | Roles de acesso | Controle de permissões |
| `rbac_user_roles` | Associação usuário-role | RBAC |

### Tabelas de Segurança (4)

| Tabela | Descrição | Compliance |
|--------|-----------|------------|
| `rbac_sessions` | Sessões ativas | Segurança |
| `rbac_permissions_override` | Permissões especiais | Auditoria |
| `cpf_tokens` | Tokenização CPF | LGPD |
| `cpf_access_log` | Log de acesso CPF | LGPD/Auditoria |

---

## Usuários Padrão

| Username | Senha | Role | Permissões |
|----------|-------|------|------------|
| admin | admin123 | admin | Acesso total ao sistema |
| analista | admin123 | analyst | Transações, alertas, regras |
| viewer | admin123 | viewer | Somente leitura |

**⚠️ IMPORTANTE:** Altere as senhas imediatamente em produção!

---

## 216 Regras Duras (HardRulesEngine v2.0)

### Distribuição por Ação

| Ação | Quantidade | Score | Descrição |
|------|------------|-------|-----------|
| **block** | 63 regras | 0.95 | Bloqueio imediato (CRITICAL) |
| **step_up** | 19 regras | 0.80 | Verificação extra (HIGH) |
| **review** | 106 regras | 0.75 | Análise manual (MEDIUM) |
| **alert** | 28 regras | 0.50 | Monitoramento (MEDIUM) |

### Categorias de Regras (17 Categorias)

| Categoria | Qtd | Exemplos |
|-----------|-----|----------|
| REGULAÇÃO BACEN | 10 | BCB 403/2024, Limites noturnos, COAF |
| CARD-NOT-PRESENT | 10 | AVS, CVV, 3DS, Card Testing |
| DEVICE/LOCATION | 12 | Fingerprinting, VPN, Emulador |
| ENGENHARIA SOCIAL | 6 | WhatsApp, Falsa Central, QR Code |
| MALWARE | 5 | Mão Fantasma, BrasDex, ATS |
| SEQUESTRO | 4 | ATM madrugada, Coação |
| VELOCITY | 18 | Card Testing, Impossible Travel |
| ML PATTERNS | 10 | Anomalia, Behavioral |
| VALOR | 14 | Faixas críticas R$50-10.000 |
| HORÁRIO | 12 | 00h-06h, 13h, 20h-23h |
| PIX KEY | 10 | Aleatória, Telefone, CNPJ |
| COMBINADAS | 14 | Multi-fator, Tríade |
| COMPLIANCE | 3 | PCI DSS, LGPD, COAF |
| CANAL | 12 | Mobile, Web, ATM, E-commerce |
| GOLPES ESPECÍFICOS | 12 | Romântico, Investimento, Pirâmide |
| AUTENTICAÇÃO | 3 | 3DS, Step-up biométrico |
| NOVO CLIENTE | 5 | Conta nova, Cartão recém-emitido |

### Formato de Resposta Unificado (Idêntico ao ML)

```python
{
    "transaction_id": "TXN_001",
    "is_fraud": true/false,
    "fraud_probability": 0.0-1.0,
    "risk_score": 0.0-1.0,
    "risk_level": "LOW/MEDIUM/HIGH/CRITICAL",
    "confidence": 0.0-1.0,
    "processing_time_ms": float,
    "model_version": "HARD_RULES_2.0.0",
    "detection_reason": ["Razão 1", "Razão 2"],
    "timestamp": "ISO8601"
}
```

---

## Queries Úteis

### Verificar Status do Sistema
```sql
-- Contar tabelas
SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';

-- Verificar regras duras
SELECT action, COUNT(*) as total FROM hard_rules GROUP BY action ORDER BY total DESC;

-- Verificar regras ativas
SELECT COUNT(*) FROM hard_rules WHERE enabled = true;

-- Verificar usuários
SELECT username, role, created_at FROM users;

-- Últimas transações
SELECT id, amount, channel, risk_score, status FROM transactions ORDER BY created_at DESC LIMIT 10;

-- Alertas pendentes
SELECT COUNT(*) FROM alerts WHERE status = 'pending';
```

### Monitoramento
```sql
-- Logs de auditoria (últimas 24h)
SELECT action, entity_type, user_id, created_at 
FROM audit_logs 
WHERE created_at > NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC;

-- Performance do ML
SELECT model_name, accuracy, precision_score, recall, f1_score, timestamp
FROM model_metrics
ORDER BY timestamp DESC LIMIT 5;

-- Transações por canal (hoje)
SELECT channel, COUNT(*) as total, AVG(amount) as valor_medio
FROM transactions
WHERE created_at::date = CURRENT_DATE
GROUP BY channel;
```

---

## Backup e Restore

### Backup Completo
```bash
pg_dump "$DATABASE_URL" > backup_sankofa_$(date +%Y%m%d).sql
```

### Backup Apenas Dados
```bash
pg_dump "$DATABASE_URL" --data-only > backup_dados_$(date +%Y%m%d).sql
```

### Restore
```bash
psql "$DATABASE_URL" < backup_sankofa.sql
```

---

## Índices Recomendados

```sql
-- Transações (performance crítica)
CREATE INDEX idx_transactions_cpf ON transactions(cpf);
CREATE INDEX idx_transactions_created_at ON transactions(created_at);
CREATE INDEX idx_transactions_status ON transactions(status);
CREATE INDEX idx_transactions_channel ON transactions(channel);

-- Alertas
CREATE INDEX idx_alerts_status ON alerts(status);
CREATE INDEX idx_alerts_created_at ON alerts(created_at);

-- Regras
CREATE INDEX idx_hard_rules_enabled ON hard_rules(enabled);
CREATE INDEX idx_hard_rules_action ON hard_rules(action);

-- Auditoria (LGPD)
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);
```

---

## Troubleshooting

### Verificar Conexão
```bash
psql "$DATABASE_URL" -c "SELECT 1;"
```

### Verificar Locks
```sql
SELECT pid, usename, query, state FROM pg_stat_activity WHERE state != 'idle';
```

### Limpar Cache de Regras
```sql
-- Força reload das regras no próximo request
UPDATE hard_rules SET updated_at = NOW() WHERE id = 1;
```

### Verificar Tamanho das Tabelas
```sql
SELECT relname as tabela, 
       pg_size_pretty(pg_total_relation_size(relid)) as tamanho
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;
```

---

## Compliance

### LGPD
- Tokenização de CPF via tabela `cpf_tokens`
- Log de acesso a dados sensíveis em `cpf_access_log`
- Auditoria completa em `audit_logs`

### BACEN
- Limites de transação configuráveis em `system_configs`
- Regras BCB 403/2024 em `hard_rules`
- Relatórios STR via `alerts`

### PCI DSS
- Dados de cartão nunca armazenados em texto
- Sessões com timeout em `rbac_sessions`
- Logs de acesso em `audit_logs`

---

## Suporte

Para problemas com o banco de dados:

1. **Verifique logs:**
```bash
psql "$DATABASE_URL" -c "SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT 10;"
```

2. **Verifique conexão:**
```bash
psql "$DATABASE_URL" -c "SELECT version();"
```

3. **Verifique regras:**
```bash
psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM hard_rules WHERE enabled = true;"
```

4. **Reinicie cache:**
```bash
# Via API
curl -X POST http://localhost:5000/api/cache/invalidate
```
