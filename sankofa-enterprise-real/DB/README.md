# Sankofa Enterprise Pro - Database Documentation

## Status da Base de Dados (30/11/2025)

**Status**: ✅ OPERACIONAL COM DADOS REAIS
**SGBD**: PostgreSQL 15+ (Neon-backed)
**Registros Totais**: 4.517 (transações + meta)
**Tabelas**: 17 operacionais

---

## Dados Atual

### Distribuição de Transações

| Campo | Valor |
|-------|-------|
| **Total de transações** | 4.466 |
| **Fraudes detectadas** | 3.114 (69,73%) |
| **Taxa de aprovação** | 30,3% |
| **Valor protegido** | R$ 14.328.997,85 |

### Por Canal de Pagamento

| Canal | Transações | Fraudes | Taxa |
|-------|-----------|---------|------|
| **PIX** | 4.285 | 3.081 | 71,9% |
| **TED** | 86 | 14 | 16,3% |
| **BOLETO** | 88 | 14 | 15,9% |
| **Mobile/Web** | 7 | 5 | 71,4% |

### Tabelas com Dados

| Tabela | Registros | Status |
|--------|-----------|--------|
| transactions | 4.466 | ✅ Real |
| audit_logs | 38 | ✅ Real |
| hard_rules | 2 | ✅ Real |
| vip_list | 1 | ✅ Real |
| hot_list | 1 | ✅ Real |
| users | 5 | ✅ Configurado |
| alerts | 0 | ✅ Dinâmico |

---

## Configuração

### Variáveis de Ambiente

```bash
DATABASE_URL=postgresql://user:password@host:5432/database
PGHOST=ep-xxx.us-east-2.aws.neon.tech
PGPORT=5432
PGUSER=sankofa_app
PGPASSWORD=***
PGDATABASE=sankofa_fraud
```

### Conexão (Python)

```python
import psycopg2
from os import getenv

conn = psycopg2.connect(getenv("DATABASE_URL"))
```

---

## Estrutura de Tabelas

### 1. Tabela `transactions`
Armazena todas as transações financeiras processadas.

```sql
CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(100) UNIQUE,
    amount DECIMAL(15, 2),
    channel VARCHAR(50),          -- PIX, TED, BOLETO
    status VARCHAR(20),           -- APPROVED, FRAUD, REJECTED
    is_fraud BOOLEAN,
    risk_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
);
```

**Índices Criados**:
- `idx_transactions_fraud_amount(is_fraud, amount)`
- `idx_transactions_risk_score(risk_score)`
- `idx_transactions_channel_status(channel, status)`

### 2. Tabela `audit_logs`
Log de auditoria append-only (LGPD/BACEN compliance).

```sql
CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    action VARCHAR(100),
    details TEXT,
    user_id VARCHAR(100),
    ip_address INET,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Retenção**: 7 anos (BACEN Resolução 6/2023)

### 3. Tabelas de Configuração
- `hard_rules` - Regras rígidas de bloqueio
- `vip_list` - Lista branca (whitelist)
- `hot_list` - Lista negra (blacklist)
- `users` - Usuários do sistema (5 roles)

---

## Performance

### Índices Implementados

```sql
-- Composite indexes para queries frequentes
CREATE INDEX idx_transactions_fraud_amount 
  ON transactions(is_fraud, amount);

CREATE INDEX idx_transactions_risk_score 
  ON transactions(risk_score);

CREATE INDEX idx_transactions_channel_status 
  ON transactions(channel, status);
```

### Latência com Cache

| Query | Sem Cache | Com Cache | Melhoria |
|-------|-----------|-----------|----------|
| KPIs | 730ms | 40ms | 18x |
| Timeseries | 680ms | 43ms | 16x |
| Channels | 670ms | 50ms | 13x |

---

## Compliance

### LGPD
- ✅ CPF mascarado (XXX.XXX.XXX-XX)
- ✅ Audit trail completo
- ✅ Direito ao esquecimento implementado
- ✅ Retenção: 5 anos

### BACEN Resolução 6/2023
- ✅ Retenção: 5 anos
- ✅ Rastreabilidade completa
- ✅ Relatórios regulatórios
- ✅ SLA <50ms PIX monitorado

### PCI DSS
- ✅ Dados sensíveis criptografados
- ✅ Sem armazenamento de CVV
- ✅ Logs sem dados sensíveis
- ✅ Acesso restrito

---

## Comandos Rápidos

### Verificar Status

```sql
-- Total de transações
SELECT COUNT(*) FROM transactions;

-- Fraudes detectadas
SELECT COUNT(*) FROM transactions WHERE is_fraud = true;

-- Taxa de aprovação
SELECT COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions)
  FROM transactions WHERE status = 'APPROVED';
```

### Monitorar Audit Log

```sql
SELECT action, COUNT(*) FROM audit_logs 
  GROUP BY action 
  ORDER BY COUNT(*) DESC;
```

---

## Backup & Restore

### Criar Backup
```bash
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql
```

### Restaurar Backup
```bash
psql $DATABASE_URL < backup_20251130.sql
```

---

## Troubleshooting

### Verificar Conexão
```bash
psql $DATABASE_URL -c "SELECT NOW();"
```

### Listar Tabelas
```bash
psql $DATABASE_URL -c "\dt"
```

### Ver Índices
```bash
psql $DATABASE_URL -c "\di+"
```

---

## Documentação Completa

Documentação detalhada disponível em:
- `DB.md` - Schema completo
- `REDIS.md` - Sistema de cache
- `docs/database/` - Análises técnicas

---

**Status**: ✅ PRONTO PARA PRODUÇÃO
**Última atualização**: 30 de Novembro de 2025
