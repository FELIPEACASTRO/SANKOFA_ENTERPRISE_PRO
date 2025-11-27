# Sankofa Enterprise Pro - Documentação Completa do Banco de Dados

## Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura de Dados](#arquitetura-de-dados)
3. [Tabelas do Sistema](#tabelas-do-sistema)
4. [Diagrama Entidade-Relacionamento](#diagrama-entidade-relacionamento)
5. [Fluxo de Dados](#fluxo-de-dados)
6. [Integrações](#integrações)
7. [Camadas de Persistência](#camadas-de-persistência)
8. [Pool de Conexões](#pool-de-conexões)
9. [Índices e Performance](#índices-e-performance)
10. [Triggers e Functions](#triggers-e-functions)
11. [Views](#views)
12. [Compliance e Auditoria](#compliance-e-auditoria)
13. [Backup e Recuperação](#backup-e-recuperação)
14. [Monitoramento](#monitoramento)
15. [Troubleshooting](#troubleshooting)

---

## Visão Geral

O Sankofa Enterprise Pro utiliza **PostgreSQL** como banco de dados principal, aproveitando recursos avançados como:

- **JSONB** para dados semi-estruturados
- **UUID** para identificadores únicos distribuídos
- **Extensões** (uuid-ossp, pgcrypto, pg_trgm)
- **Triggers** para automação
- **Views** para relatórios

### Características

| Característica | Valor |
|----------------|-------|
| SGBD | PostgreSQL 15+ |
| Hosting | Neon (Serverless) |
| Capacidade | 300M+ transações/dia |
| Retenção | 5-7 anos (compliance) |
| Backup | Automático + Manual |

### Variáveis de Ambiente

```bash
DATABASE_URL=postgresql://user:password@host:5432/database
PGHOST=ep-xxx.us-east-2.aws.neon.tech
PGPORT=5432
PGUSER=sankofa_app
PGPASSWORD=***
PGDATABASE=sankofa_fraud
```

---

## Arquitetura de Dados

### Diagrama de Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                     APLICAÇÃO SANKOFA                           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌─────────────────┐ ┌───────────┐ ┌─────────────────┐
│   PRODUCTION    │ │   CACHE   │ │   CONTINUOUS    │
│   API (Flask)   │ │  (Redis)  │ │   LEARNING      │
└────────┬────────┘ └─────┬─────┘ └────────┬────────┘
         │                │                │
         │                │                │
         ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CAMADA DE REPOSITÓRIO                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ CompositeTransactionRepository (Write-Through Cache)    │    │
│  │   ├── PostgreSQLTransactionRepository (Primário)       │    │
│  │   └── RedisTransactionRepository (Cache)               │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│  │ PostgreSQLCusto │ │ PostgreSQLEvent │ │ PostgreSQLPersi │    │
│  │ merRepository   │ │ Store           │ │ stence (API)    │    │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONNECTION POOL                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ThreadedConnectionPool (psycopg2)                       │    │
│  │   - min_connections: 2                                  │    │
│  │   - max_connections: 20                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ asyncpg.Pool (async operations)                         │    │
│  │   - min_size: 5                                         │    │
│  │   - max_size: 20                                        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POSTGRESQL (Neon)                            │
│                                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ transactions │ │ customers    │ │ audit_trail  │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ fraud_       │ │ users        │ │ model_       │            │
│  │ detections   │ │              │ │ versions     │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ events       │ │ alerts       │ │ compliance_  │            │
│  │              │ │              │ │ reports      │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ api_keys     │ │ rate_limits  │ │ schema_      │            │
│  │              │ │              │ │ migrations   │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tabelas do Sistema

### Resumo das Tabelas

| # | Tabela | Descrição | Volume Esperado |
|---|--------|-----------|-----------------|
| 1 | `transactions` | Transações financeiras | 300M+/dia |
| 2 | `fraud_detections` | Resultados de ML | 3-5% das transações |
| 3 | `customers` | Perfil de risco | Milhões |
| 4 | `audit_trail` | Log de auditoria | Todos os eventos |
| 5 | `users` | Usuários do sistema | ~100-1000 |
| 6 | `model_versions` | Versões dos modelos ML | ~10-50 |
| 7 | `api_keys` | Chaves de API | ~50-100 |
| 8 | `events` | Event sourcing | Milhões |
| 9 | `alerts` | Alertas de fraude | ~1% das transações |
| 10 | `compliance_reports` | Relatórios BACEN/LGPD | ~12/ano |
| 11 | `rate_limits` | Controle de rate limiting | Temporário |
| 12 | `schema_migrations` | Controle de migrações | ~10-50 |

---

### 1. Tabela: `transactions`

A tabela principal do sistema, armazena todas as transações financeiras.

#### Estrutura

```sql
CREATE TABLE transactions (
    -- Identificadores
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- Dados do Cliente (LGPD: mascarados)
    cliente_cpf VARCHAR(14) NOT NULL,      -- XXX.XXX.XXX-XX
    customer_id VARCHAR(100),
    
    -- Dados da Transação
    amount DECIMAL(15, 2) NOT NULL,
    valor DECIMAL(15, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'BRL',
    tipo_transacao VARCHAR(50) NOT NULL,   -- PIX, TED, CARTAO_CREDITO
    canal VARCHAR(50) NOT NULL,            -- APP, WEB, ATM, AGENCIA
    status VARCHAR(20) DEFAULT 'PENDING',  -- PENDING, APPROVED, BLOCKED
    
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
    
    -- Dados do Recebedor (PIX/TED)
    conta_recebedor VARCHAR(100),
    banco_recebedor VARCHAR(10),
    pix_key VARCHAR(100),
    pix_key_type VARCHAR(20),
    
    -- Resultados da Análise de Fraude
    is_fraud BOOLEAN DEFAULT FALSE,
    fraud_score DECIMAL(5, 4),
    risk_level VARCHAR(20) DEFAULT 'LOW',
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
```

#### Valores Válidos

| Coluna | Valores |
|--------|---------|
| `tipo_transacao` | PIX, TED, DOC, CARTAO_CREDITO, CARTAO_DEBITO, BOLETO, SAQUE |
| `canal` | APP, WEB, ATM, AGENCIA, API, TELEFONE |
| `status` | PENDING, APPROVED, BLOCKED, REVIEW, CANCELLED |
| `risk_level` | LOW, MEDIUM, HIGH, CRITICAL |
| `pix_key_type` | CPF, CNPJ, EMAIL, TELEFONE, ALEATORIO |

#### Índices

```sql
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp DESC);
CREATE INDEX idx_transactions_cliente_cpf ON transactions(cliente_cpf);
CREATE INDEX idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX idx_transactions_is_fraud ON transactions(is_fraud);
CREATE INDEX idx_transactions_risk_level ON transactions(risk_level);
CREATE INDEX idx_transactions_status ON transactions(status);
CREATE INDEX idx_transactions_canal ON transactions(canal);
CREATE INDEX idx_transactions_customer_timestamp ON transactions(customer_id, timestamp DESC);
```

#### Exemplo de Uso

```python
# Inserir transação
cur.execute("""
    INSERT INTO transactions (
        transaction_id, cliente_cpf, amount, valor, 
        tipo_transacao, canal, status, is_fraud, fraud_score
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
""", (
    'TXN_123456',
    'XXX.XXX.XXX-01',
    1500.00, 1500.00,
    'PIX', 'APP', 'APPROVED',
    False, 0.12
))
```

---

### 2. Tabela: `fraud_detections`

Detalhes das detecções de fraude pelos modelos ML.

#### Estrutura

```sql
CREATE TABLE fraud_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id UUID REFERENCES transactions(id) ON DELETE CASCADE,
    
    -- Scores e Probabilidades
    fraud_probability DECIMAL(5, 4) NOT NULL,
    risk_score DECIMAL(5, 4) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    
    -- Explicação (LGPD Art. 20)
    detection_reason TEXT[],
    top_risk_factors JSONB DEFAULT '[]',
    top_protective_factors JSONB DEFAULT '[]',
    explanation_text TEXT,
    lgpd_compliant BOOLEAN DEFAULT TRUE,
    
    -- Modelo usado
    model_version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50),
    ensemble_votes JSONB DEFAULT '{}',
    
    -- Performance
    processing_time_ms DECIMAL(10, 2),
    cache_hit BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    reviewed_by VARCHAR(100)
);
```

#### Exemplo de Detecção

```json
{
  "fraud_probability": 0.89,
  "risk_score": 0.89,
  "risk_level": "CRITICAL",
  "detection_reason": [
    "Valor alto (R$ 15.000)",
    "Horário atípico (03:00)",
    "Novo destinatário"
  ],
  "top_risk_factors": [
    {"feature": "amount_normalized", "impact": 0.45},
    {"feature": "hour_sin", "impact": 0.25}
  ],
  "top_protective_factors": [
    {"feature": "device_risk_score", "impact": -0.15}
  ],
  "explanation_text": "Transação de alto valor em horário atípico para novo destinatário."
}
```

---

### 3. Tabela: `customers`

Perfil de risco e histórico dos clientes.

#### Estrutura

```sql
CREATE TABLE customers (
    id VARCHAR(100) PRIMARY KEY,
    cpf_hash VARCHAR(64) NOT NULL,           -- SHA-256 do CPF
    
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
    
    -- Dispositivos e locais conhecidos
    known_devices JSONB DEFAULT '[]',
    known_locations JSONB DEFAULT '[]',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    metadata JSONB DEFAULT '{}',
    version INTEGER DEFAULT 1
);
```

#### Atualização Automática via Trigger

O trigger `update_customer_stats` atualiza automaticamente as estatísticas do cliente após cada transação:

```sql
CREATE OR REPLACE FUNCTION update_customer_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO customers (id, cpf_hash, transaction_count, total_amount, last_transaction_at)
        VALUES (
            NEW.customer_id,
            encode(digest(NEW.cliente_cpf, 'sha256'), 'hex'),
            1, NEW.amount, NEW.timestamp
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
```

---

### 4. Tabela: `audit_trail`

Log de auditoria append-only para compliance LGPD/BACEN.

#### Estrutura

```sql
CREATE TABLE audit_trail (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Evento
    event_type VARCHAR(100) NOT NULL,
    action VARCHAR(50) NOT NULL,           -- CREATE, READ, UPDATE, DELETE, PREDICT
    status VARCHAR(20) NOT NULL,           -- SUCCESS, FAILURE, PENDING
    
    -- Recurso
    resource_type VARCHAR(100),
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
```

#### Eventos Auditados

| Event Type | Descrição |
|------------|-----------|
| `TRANSACTION_CREATED` | Nova transação registrada |
| `FRAUD_DETECTED` | Fraude detectada pelo modelo |
| `FRAUD_REVIEWED` | Analista revisou fraude |
| `USER_LOGIN` | Login de usuário |
| `USER_LOGIN_FAILED` | Tentativa de login falhou |
| `MODEL_DEPLOYED` | Novo modelo implantado |
| `DATA_EXPORTED` | Dados exportados (LGPD) |
| `DATA_DELETED` | Dados deletados (LGPD) |

#### Política de Retenção

```
┌────────────────────┬──────────────┬─────────────────────────────────┐
│ Tipo de Dado       │ Retenção     │ Justificativa                   │
├────────────────────┼──────────────┼─────────────────────────────────┤
│ audit_trail        │ 7 anos       │ BACEN Resolução 6/2023          │
│ transactions       │ 5 anos       │ Compliance fiscal               │
│ fraud_detections   │ 5 anos       │ Análise forense                 │
│ rate_limits        │ 24 horas     │ Temporário                      │
└────────────────────┴──────────────┴─────────────────────────────────┘
```

---

### 5. Tabela: `users`

Usuários do sistema com autenticação segura.

#### Estrutura

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Credenciais
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,   -- bcrypt
    
    -- Perfil
    full_name VARCHAR(255),
    role VARCHAR(50) NOT NULL,
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
```

#### Roles (Papéis)

| Role | Permissões |
|------|------------|
| `ADMIN` | Acesso total, gerenciamento de usuários |
| `ANALYST` | Revisão de fraudes, relatórios |
| `VIEWER` | Apenas visualização |
| `API` | Acesso via API apenas |

---

### 6. Tabela: `model_versions`

Registro e lifecycle dos modelos de ML.

#### Estrutura

```sql
CREATE TABLE model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Identificação
    version VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    
    -- Métricas
    metrics JSONB DEFAULT '{}',
    accuracy DECIMAL(5, 4),
    precision_score DECIMAL(5, 4),
    recall DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    roc_auc DECIMAL(5, 4),
    
    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'TRAINING',
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
```

#### Lifecycle do Modelo

```
TRAINING → VALIDATION → PRODUCTION → DEPRECATED
    ↑                                    │
    └────────────────────────────────────┘
              (Re-treinamento)
```

---

### 7-12. Tabelas Auxiliares

#### `api_keys`
Chaves de API para autenticação de sistemas externos.

#### `events`
Event sourcing para rastreabilidade completa.

#### `alerts`
Alertas de fraude e sistema.

#### `compliance_reports`
Relatórios regulatórios (BACEN, LGPD, PCI DSS).

#### `rate_limits`
Controle de rate limiting por API key.

#### `schema_migrations`
Controle de versão do schema.

---

## Diagrama Entidade-Relacionamento

```
┌─────────────────┐     1:N     ┌──────────────────────┐
│   customers     │◄────────────│    transactions      │
├─────────────────┤             ├──────────────────────┤
│ id (PK)         │             │ id (PK)              │
│ cpf_hash        │             │ customer_id (FK)     │
│ risk_profile    │             │ transaction_id       │
│ fraud_count     │             │ amount               │
│ transaction_cnt │             │ is_fraud             │
│ known_devices   │             │ fraud_score          │
└─────────────────┘             └──────────┬───────────┘
                                           │
                                           │ 1:1
                                           ▼
                                ┌──────────────────────┐
                                │  fraud_detections    │
                                ├──────────────────────┤
                                │ id (PK)              │
                                │ transaction_id (FK)  │
                                │ fraud_probability    │
                                │ explanation_text     │
                                │ model_version        │
                                └──────────────────────┘

┌─────────────────┐             ┌──────────────────────┐
│     users       │─────────────│    audit_trail       │
├─────────────────┤    1:N      ├──────────────────────┤
│ id (PK)         │             │ id (PK)              │
│ username        │             │ user_id (FK)         │
│ email           │             │ event_type           │
│ role            │             │ action               │
│ is_active       │             │ timestamp            │
└─────────────────┘             └──────────────────────┘

┌─────────────────┐             ┌──────────────────────┐
│ model_versions  │─────────────│   fraud_detections   │
├─────────────────┤    1:N      ├──────────────────────┤
│ id (PK)         │             │ model_version (FK)   │
│ version         │             │ model_type           │
│ is_active       │             │ ensemble_votes       │
└─────────────────┘             └──────────────────────┘

┌─────────────────┐             ┌──────────────────────┐
│   api_keys      │─────────────│    rate_limits       │
├─────────────────┤    1:N      ├──────────────────────┤
│ id (PK)         │             │ key_hash (FK)        │
│ key_hash        │             │ window_start         │
│ permissions     │             │ request_count        │
└─────────────────┘             └──────────────────────┘

┌─────────────────┐             ┌──────────────────────┐
│  transactions   │─────────────│       alerts         │
├─────────────────┤    1:N      ├──────────────────────┤
│ id (PK)         │             │ id (PK)              │
│ is_fraud        │             │ transaction_id (FK)  │
│ fraud_score     │             │ severity             │
└─────────────────┘             └──────────────────────┘
```

---

## Fluxo de Dados

### Fluxo de uma Transação

```
┌──────────────┐
│   Cliente    │
│   (APP/Web)  │
└──────┬───────┘
       │ 1. Envia transação
       ▼
┌──────────────┐
│  API Gateway │
│  (Flask)     │
└──────┬───────┘
       │ 2. Valida request
       ▼
┌──────────────┐     ┌──────────────┐
│  Redis Cache │◄───►│  Blacklist   │
│  (Lookup)    │     │  Check       │
└──────┬───────┘     └──────────────┘
       │ 3. Cache miss? Consulta DB
       ▼
┌──────────────────────────────────────┐
│         PostgreSQL                   │
│  ┌───────────────┐ ┌───────────────┐ │
│  │  customers    │ │  transactions │ │
│  │  (histórico)  │ │  (inserir)    │ │
│  └───────────────┘ └───────────────┘ │
└──────────────┬───────────────────────┘
               │ 4. Dados para ML
               ▼
┌──────────────────────────────────────┐
│         ML Engine                    │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│  │ XGBoost │ │ LightGBM│ │ RF      │ │
│  └────┬────┘ └────┬────┘ └────┬────┘ │
│       └───────────┼───────────┘      │
│                   ▼                  │
│           ┌─────────────┐            │
│           │  Ensemble   │            │
│           │  (Stacking) │            │
│           └─────────────┘            │
└──────────────┬───────────────────────┘
               │ 5. Predição
               ▼
┌──────────────────────────────────────┐
│         PostgreSQL                   │
│  ┌───────────────┐ ┌───────────────┐ │
│  │  fraud_       │ │  audit_trail  │ │
│  │  detections   │ │  (log)        │ │
│  └───────────────┘ └───────────────┘ │
└──────────────┬───────────────────────┘
               │ 6. Resposta
               ▼
┌──────────────┐
│   Cliente    │
│  (Resposta)  │
└──────────────┘
```

---

## Integrações

### 1. Integração com API de Produção (`production_api.py`)

A classe `PostgreSQLPersistence` gerencia a conexão:

```python
class PostgreSQLPersistence:
    def __init__(self, fail_closed: bool = False):
        self._pool = None
        self._initialized = False
        self._fail_closed = fail_closed
        self._init_pool()
    
    def _init_pool(self):
        database_url = os.getenv("DATABASE_URL")
        pool_min = int(os.getenv("DB_POOL_MIN", "2"))
        pool_max = int(os.getenv("DB_POOL_MAX", "20"))
        
        self._pool = pool.ThreadedConnectionPool(
            minconn=pool_min,
            maxconn=pool_max,
            dsn=database_url
        )
    
    def save_transaction(self, transaction_data: Dict, prediction: Dict) -> bool:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO transactions (
                        transaction_id, amount, channel, type, status,
                        risk_score, is_fraud, cpf, location, timestamp,
                        processing_time_ms, model_version
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (transaction_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        risk_score = EXCLUDED.risk_score,
                        is_fraud = EXCLUDED.is_fraud
                """, (...))
                conn.commit()
                return True
        finally:
            self._pool.putconn(conn)
```

### 2. Integração com Repositórios (`repositories.py`)

Padrão Repository com PostgreSQL e Redis:

```python
class PostgreSQLTransactionRepository(TransactionRepository):
    def __init__(self, connection_pool: asyncpg.Pool):
        self._pool = connection_pool
    
    async def save(self, transaction: Transaction) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO transactions (...) VALUES ($1, $2, ...)
                ON CONFLICT (id) DO UPDATE SET ...
            """, ...)
    
    async def find_by_id(self, transaction_id: TransactionId) -> Optional[Transaction]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM transactions WHERE id = $1", 
                transaction_id.value
            )
            return self._row_to_transaction(row) if row else None

class CompositeTransactionRepository(TransactionRepository):
    """Write-Through Cache Pattern"""
    def __init__(self, primary_repo: PostgreSQLTransactionRepository, 
                 cache_repo: RedisTransactionRepository):
        self._primary = primary_repo
        self._cache = cache_repo
    
    async def save(self, transaction: Transaction) -> None:
        # 1. Salva no PostgreSQL primeiro (consistência)
        await self._primary.save(transaction)
        # 2. Atualiza cache (best effort)
        try:
            await self._cache.save(transaction)
        except Exception:
            pass  # Cache failure não falha a operação
```

### 3. Integração com Continuous Learning (`continuous_learning_system.py`)

SQLite para dados de ML (separado do PostgreSQL principal):

```python
class ContinuousLearningSystem:
    def __init__(self, db_path: str = "data/continuous_learning.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de transações processadas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id TEXT PRIMARY KEY,
                valor REAL,
                predicted_fraud_prob REAL,
                actual_is_fraud INTEGER DEFAULT NULL,
                analyst_feedback TEXT DEFAULT NULL
            )
        """)
        
        # Tabela de métricas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT,
                auc_score REAL,
                training_date TEXT
            )
        """)
```

---

## Camadas de Persistência

### Arquitetura em Camadas

```
┌─────────────────────────────────────────────────────────┐
│                   CAMADA DE APLICAÇÃO                   │
│  - production_api.py                                    │
│  - Flask endpoints                                      │
└─────────────────────────────────┬───────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────┐
│                   CAMADA DE REPOSITÓRIO                 │
│  - TransactionRepository (interface)                    │
│  - CustomerRepository (interface)                       │
│  - EventStore (interface)                               │
└─────────────────────────────────┬───────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────▼─────────┐   ┌────────▼────────┐   ┌─────────▼─────────┐
│   PostgreSQL      │   │     Redis       │   │    Composite      │
│   Repository      │   │   Repository    │   │    Repository     │
│   (Persistência)  │   │   (Cache)       │   │ (Write-Through)   │
└─────────┬─────────┘   └────────┬────────┘   └─────────┬─────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────┐
│                   CONNECTION POOL                       │
│  - ThreadedConnectionPool (psycopg2, síncrono)          │
│  - asyncpg.Pool (assíncrono)                            │
└────────────────────────────────┬────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────┐
│                   POSTGRESQL (Neon)                     │
│  - Serverless                                           │
│  - Auto-scaling                                         │
│  - Point-in-time recovery                               │
└─────────────────────────────────────────────────────────┘
```

---

## Pool de Conexões

### Configuração do Pool (Síncrono)

```python
from psycopg2 import pool

# Configuração via variáveis de ambiente
pool_min = int(os.getenv("DB_POOL_MIN", "2"))
pool_max = int(os.getenv("DB_POOL_MAX", "20"))

connection_pool = pool.ThreadedConnectionPool(
    minconn=pool_min,
    maxconn=pool_max,
    dsn=os.getenv("DATABASE_URL")
)
```

### Configuração do Pool (Assíncrono)

```python
import asyncpg

pool = await asyncpg.create_pool(
    host=config["db_host"],
    port=config["db_port"],
    database=config["db_name"],
    user=config["db_user"],
    password=config["db_password"],
    min_size=5,
    max_size=20,
    max_queries=50000,
    max_inactive_connection_lifetime=300.0,
    command_timeout=60,
    server_settings={
        "jit": "off",  # Desabilita JIT para performance consistente
        "application_name": "sankofa_fraud_detection",
    },
)
```

### Monitoramento do Pool

```python
# Estatísticas do pool
pool_stats = {
    "size": pool.get_size(),
    "max_size": pool.get_max_size(),
    "min_size": pool.get_min_size(),
    "idle": pool.get_idle_size(),
}
```

---

## Índices e Performance

### Índices Principais

| Tabela | Índice | Colunas | Tipo | Uso |
|--------|--------|---------|------|-----|
| transactions | idx_transactions_timestamp | timestamp DESC | B-tree | Consultas por período |
| transactions | idx_transactions_cliente_cpf | cliente_cpf | B-tree | Consultas por cliente |
| transactions | idx_transactions_is_fraud | is_fraud | B-tree | Filtro de fraudes |
| transactions | idx_transactions_customer_timestamp | customer_id, timestamp DESC | B-tree Composto | Histórico por cliente |
| audit_trail | idx_audit_trail_timestamp | timestamp DESC | B-tree | Auditoria por período |
| customers | idx_customers_cpf_hash | cpf_hash | B-tree Único | Busca por CPF (hash) |

### Análise de Performance

```sql
-- Verificar uso de índices
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Queries mais lentas
SELECT 
    query,
    calls,
    total_time / 1000 as total_seconds,
    mean_time as avg_ms
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;
```

### Particionamento (Recomendado para Produção)

```sql
-- Criar tabela particionada por mês
CREATE TABLE transactions_partitioned (
    LIKE transactions INCLUDING ALL
) PARTITION BY RANGE (timestamp);

-- Criar partições
CREATE TABLE transactions_2025_11 PARTITION OF transactions_partitioned
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

CREATE TABLE transactions_2025_12 PARTITION OF transactions_partitioned
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');
```

---

## Triggers e Functions

### 1. Atualização Automática de `updated_at`

```sql
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Aplicar em tabelas
CREATE TRIGGER update_transactions_updated_at
    BEFORE UPDATE ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_customers_updated_at
    BEFORE UPDATE ON customers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

### 2. Atualização de Estatísticas do Cliente

```sql
CREATE OR REPLACE FUNCTION update_customer_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- Insere ou atualiza estatísticas do cliente
        INSERT INTO customers (id, cpf_hash, transaction_count, total_amount, last_transaction_at)
        VALUES (
            NEW.customer_id,
            encode(digest(NEW.cliente_cpf, 'sha256'), 'hex'),
            1, NEW.amount, NEW.timestamp
        )
        ON CONFLICT (id) DO UPDATE SET
            transaction_count = customers.transaction_count + 1,
            total_amount = customers.total_amount + NEW.amount,
            avg_amount = (customers.total_amount + NEW.amount) / (customers.transaction_count + 1),
            last_transaction_at = NEW.timestamp,
            updated_at = NOW();
            
        -- Se é fraude, atualiza contadores de fraude
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

CREATE TRIGGER update_customer_stats_trigger
    AFTER INSERT ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_customer_stats();
```

---

## Views

### 1. Estatísticas de Fraude por Dia

```sql
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
```

### 2. Estatísticas por Canal

```sql
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
```

### 3. Clientes de Alto Risco

```sql
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
```

### 4. Performance dos Modelos

```sql
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
```

---

## Compliance e Auditoria

### LGPD (Lei Geral de Proteção de Dados)

| Requisito | Implementação |
|-----------|---------------|
| Consentimento | Gerenciado pela aplicação |
| CPF Mascarado | `XXX.XXX.XXX-XX` em todas as tabelas |
| CPF Hash | SHA-256 para buscas sem expor dado |
| Direito ao Esquecimento | DELETE com audit trail |
| Explicabilidade (Art. 20) | `explanation_text` em fraud_detections |
| Portabilidade | Endpoint de exportação JSON |

### BACEN Resolução 6/2023

| Requisito | Implementação |
|-----------|---------------|
| Retenção 5 anos | `retention_until` em audit_trail |
| Rastreabilidade | Event sourcing em `events` |
| Relatórios | `compliance_reports` |
| Tempo de resposta | Monitorado via observabilidade |

### PCI DSS

| Requisito | Implementação |
|-----------|---------------|
| Dados sensíveis | Criptografados via pgcrypto |
| CVV | Nunca armazenado |
| Logs | Sem dados sensíveis |
| Acesso | API keys com permissões granulares |

---

## Backup e Recuperação

### Backup Automático

```bash
#!/bin/bash
# DB/backup/backup.sh

BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="sankofa_backup_${TIMESTAMP}.dump"

pg_dump "$DATABASE_URL" \
    --format=custom \
    --file="${BACKUP_DIR}/${BACKUP_FILE}" \
    --no-owner \
    --no-privileges

# Manter últimos 7 backups
ls -t ${BACKUP_DIR}/sankofa_backup_*.dump | tail -n +8 | xargs -r rm
```

### Restore

```bash
#!/bin/bash
# DB/backup/restore.sh

BACKUP_FILE="$1"

pg_restore "$DATABASE_URL" \
    --clean \
    --if-exists \
    --no-owner \
    --no-privileges \
    "$BACKUP_FILE"
```

### Point-in-Time Recovery (Neon)

O Neon oferece PITR automático:
- Retenção de 7 dias (plano gratuito)
- Retenção de 30 dias (plano pago)

---

## Monitoramento

### Queries de Saúde

```sql
-- Tamanho do banco
SELECT pg_size_pretty(pg_database_size(current_database())) as size;

-- Tamanho por tabela
SELECT 
    tablename,
    pg_size_pretty(pg_total_relation_size(quote_ident(tablename))) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(quote_ident(tablename)) DESC;

-- Transações por hora (última hora)
SELECT COUNT(*) as txn_count
FROM transactions 
WHERE timestamp > NOW() - INTERVAL '1 hour';

-- Taxa de fraude (últimas 24h)
SELECT 
    COUNT(*) as total,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as frauds,
    ROUND(AVG(CASE WHEN is_fraud THEN 1 ELSE 0 END) * 100, 2) as fraud_rate
FROM transactions
WHERE timestamp > NOW() - INTERVAL '24 hours';

-- Conexões ativas
SELECT 
    state,
    COUNT(*) as count
FROM pg_stat_activity
WHERE datname = current_database()
GROUP BY state;
```

### Métricas Expostas

A API expõe métricas em `/api/observability/metrics`:

```json
{
  "database": {
    "status": "healthy",
    "pool_size": 20,
    "pool_idle": 18,
    "connections_active": 2,
    "avg_query_time_ms": 12.5
  }
}
```

---

## Troubleshooting

### Problema: Conexão Recusada

```
psycopg2.OperationalError: could not connect to server: Connection refused
```

**Solução:**
1. Verificar `DATABASE_URL`
2. Verificar se o Neon está ativo
3. Verificar firewall/rede

### Problema: Pool Esgotado

```
psycopg2.pool.PoolError: connection pool exhausted
```

**Solução:**
1. Aumentar `DB_POOL_MAX`
2. Verificar conexões não fechadas
3. Implementar retry com backoff

### Problema: Query Lenta

```sql
-- Identificar queries lentas
SELECT 
    query,
    calls,
    total_time / calls as avg_time_ms
FROM pg_stat_statements
WHERE calls > 10
ORDER BY total_time / calls DESC
LIMIT 10;

-- Analisar plano de execução
EXPLAIN ANALYZE SELECT * FROM transactions WHERE customer_id = 'CUST_001';
```

**Solução:**
1. Adicionar índice apropriado
2. Otimizar query
3. Considerar particionamento

### Problema: Disco Cheio

```sql
-- Verificar tamanho
SELECT pg_size_pretty(pg_database_size(current_database()));

-- Identificar tabelas grandes
SELECT 
    tablename,
    pg_size_pretty(pg_total_relation_size(quote_ident(tablename)))
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(quote_ident(tablename)) DESC;
```

**Solução:**
1. Executar VACUUM FULL
2. Implementar particionamento
3. Arquivar dados antigos

---

## Comandos Úteis

```bash
# Conectar ao banco
psql $DATABASE_URL

# Executar migrações
python DB/scripts/migrate.py

# Ver status das migrações
python DB/scripts/migrate.py status

# Estatísticas do banco
python DB/scripts/db_utils.py stats

# Listar tabelas
python DB/scripts/db_utils.py tables

# VACUUM ANALYZE
python DB/scripts/db_utils.py vacuum

# Backup
./DB/backup/backup.sh

# Restore
./DB/backup/restore.sh backups/sankofa_backup_20251127.dump
```

---

*Documentação gerada em Novembro 2025 - Sankofa Enterprise Pro v12.0*
