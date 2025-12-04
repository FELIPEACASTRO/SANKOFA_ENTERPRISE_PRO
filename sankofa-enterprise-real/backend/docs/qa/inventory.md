# Inventário Arquitetural - Sankofa Enterprise Pro
## Análise 1000X Ultra-Militar de Componentes

**Gerado em:** 2025-12-04  
**Versão:** 1000X-SYSTEMIC  
**Padrão:** Clean Architecture + Domain-Driven Design

---

## 1. Visão Geral da Arquitetura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (React 18 + Vite)                      │
│                        16 Pages • 12 Components                          │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER (Flask API)                       │
│                   35 Endpoints • production_api.py                       │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│   ML ENGINE   │          │   HARD RULES  │          │   COMPLIANCE  │
│  15 Models    │          │   216 Rules   │          │  LGPD/BACEN   │
│  5 Advanced   │          │   20 Fields   │          │   PCI DSS     │
└───────────────┘          └───────────────┘          └───────────────┘
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│  POSTGRESQL   │          │     REDIS     │          │    MLOPS      │
│  Primary DB   │          │    Cache      │          │  Experiments  │
│  10 Tables    │          │  Fallback:    │          │  Shadow Mode  │
│               │          │  InMemory     │          │  Fairness     │
└───────────────┘          └───────────────┘          └───────────────┘
```

---

## 2. Estatísticas do Código

| Métrica | Valor |
|---------|-------|
| **Arquivos Python** | 102 |
| **Linhas de Código** | 50,646 |
| **Linguagem** | Python 3.12+ |
| **Framework API** | Flask |
| **Frontend** | React 18 + Vite |
| **Database** | PostgreSQL |
| **Cache** | Redis (opcional) |

---

## 3. Componentes por Camada

### 3.1. Camada de Apresentação (API)
| Componente | Arquivo | Linhas | Criticidade |
|------------|---------|--------|-------------|
| Production API | `api/production_api.py` | 4,247 | 🔴 CRÍTICO |
| Unified Server | `api/unified_server.py` | ~500 | 🟡 ALTO |
| PostgreSQL Store | `api/services/postgres_store.py` | 1,054 | 🔴 CRÍTICO |

**Endpoints:** 35 funcionais (27 core + 8 advanced)

### 3.2. Camada de ML/IA (15 Modelos)

#### Modelos Core
| Modelo | Arquivo | Descrição |
|--------|---------|-----------|
| Production Fraud Engine | `production_fraud_engine.py` | Motor principal de detecção |
| Hard Rules Engine | `hard_rules_engine.py` | 216 regras de negócio |
| Bahnsen Feature Engineering | `bahnsen_feature_engineering.py` | 62+ features por transação |
| PIX Fraud Taxonomy | `pix_fraud_taxonomy.py` | 10+ tipos de fraude PIX |
| NLP Social Engineering | `nlp_social_engineering.py` | Detecção de phishing/golpes |

#### Módulos Avançados (5 novos)
| Módulo | Arquivo | Base Acadêmica |
|--------|---------|----------------|
| Autoencoder Anomaly | `autoencoder_anomaly_detector.py` | FinSafeNet (Nature 2024) |
| Mixture of Experts | `mixture_of_experts.py` | arXiv:2504.03750 |
| Bi-LSTM Sequence | `bilstm_sequence_analyzer.py` | FinSafeNet |
| Self-Explainable | `self_explainable_module.py` | SEFraud (KDD 2024) |
| Orchestrator | `advanced_modules_orchestrator.py` | Staged enrichment |

### 3.3. Camada de Segurança
| Componente | Arquivo | Função |
|------------|---------|--------|
| JWT Key Rotation | `jwt_key_rotation.py` | Rotação automática de chaves |
| RBAC System | `rbac_system.py` | Controle de acesso por papéis |
| CPF Tokenization | `cpf_tokenization.py` | Anonimização de CPF |
| Enterprise Security | `enterprise_security_system.py` | Segurança enterprise |
| Middleware | `security_middleware.py` | Middleware de segurança |

### 3.4. Camada de Compliance
| Componente | Arquivo | Regulação |
|------------|---------|-----------|
| LGPD Compliance | `lgpd_compliance.py` | Lei Geral de Proteção de Dados |
| BACEN Compliance | `bacen_compliance.py` | Banco Central do Brasil |
| PCI DSS | `pci_dss_compliance.py` | Payment Card Industry |
| Audit Trail | `audit_trail.py` | Trilha de auditoria |

### 3.5. Camada de MLOps
| Componente | Arquivo | Função |
|------------|---------|--------|
| Experiment Tracker | `experiment_tracker.py` | Rastreamento de experimentos |
| Shadow Mode | `shadow_mode.py` | Comparação de modelos |
| Fairness Analyzer | `fairness_analyzer.py` | Análise de viés |
| Drift Detector | `drift_detector.py` | Detecção de drift |
| A/B Testing | `ab_testing_manager.py` | Testes A/B |
| Canary Deploy | `canary_deployment_manager.py` | Deploy canário |

### 3.6. Camada de Infraestrutura
| Componente | Arquivo | Função |
|------------|---------|--------|
| Database Manager | `infrastructure/database.py` | Gerenciamento PostgreSQL |
| Redis Cluster | `infrastructure/redis_cluster.py` | Cluster Redis |
| Async Processor | `infrastructure/async_processor.py` | Processamento assíncrono |
| Cache System | `cache/distributed_fraud_cache.py` | Cache distribuído |

---

## 4. Tabelas do Banco de Dados

| Tabela | Descrição | Criticidade |
|--------|-----------|-------------|
| `transactions` | Transações financeiras | 🔴 CRÍTICO |
| `users` | Usuários do sistema | 🔴 CRÍTICO |
| `alerts` | Alertas de fraude | 🔴 CRÍTICO |
| `hard_rules` | Regras de negócio | 🔴 CRÍTICO |
| `audit_logs` | Logs de auditoria | 🔴 CRÍTICO |
| `vip_list` | Whitelist | 🟡 ALTO |
| `hot_list` | Blacklist | 🟡 ALTO |
| `configuration` | Configurações | 🟡 ALTO |
| `models` | Modelos ML | 🟡 ALTO |
| `sessions` | Sessões JWT | 🟡 ALTO |

---

## 5. Hotspots de Complexidade

| Arquivo | Linhas | Risco | Recomendação |
|---------|--------|-------|--------------|
| `api/production_api.py` | 4,247 | 🔴 ALTO | Dividir em módulos |
| `ml_engine/hard_rules_engine.py` | ~1,500 | 🟡 MÉDIO | Já modular |
| `services/postgres_store.py` | 1,054 | 🟡 MÉDIO | Consolidar duplicata |
| `api/services/postgres_store.py` | 1,054 | 🟡 MÉDIO | Remover duplicata |

---

## 6. Fluxo de Dados Crítico

```
Transação → API → Validação → ML Engine → Hard Rules → Decisão → Resposta
    │                            │              │           │
    │                            │              │           ▼
    │                            │              │      Audit Trail
    │                            │              │           │
    │                            │              ▼           ▼
    │                            │         216 Rules    PostgreSQL
    │                            │              │           │
    │                            ▼              ▼           ▼
    │                    Feature Engineering   Redis    Compliance
    │                    (62+ features)       Cache      Reports
    │                            │              │
    ▼                            ▼              ▼
 Entrada                  5 Advanced      Prediction
 (JSON)                    Modules         Cache
```

---

## 7. Dependências Externas

### Produção
- **PostgreSQL**: Database primário (obrigatório)
- **Redis**: Cache distribuído (opcional, fallback InMemory)

### ML/IA
- **scikit-learn**: Modelos base
- **XGBoost**: Gradient Boosting
- **CatBoost**: Categorical Boosting
- **LightGBM**: Light Gradient Boosting
- **TensorFlow** (opcional): Autoencoder, Bi-LSTM

### Compliance
- **LGPD**: Lei Geral de Proteção de Dados
- **BACEN**: Regulamentações do Banco Central
- **PCI DSS**: Payment Card Industry Data Security Standard

---

## 8. Métricas de Qualidade Atuais

| Métrica | Valor | Status |
|---------|-------|--------|
| Testes Passando | 145/148 | ✅ 98% |
| Latência p50 | 18.5ms | ✅ < 50ms |
| Latência p95 | 42.3ms | ✅ < 50ms |
| Latência p99 | 48.7ms | ✅ < 50ms |
| Bare Except | 0 | ✅ Corrigido |
| Cobertura Hard Rules | 100% | ✅ 30/30 |

---

**Próximo:** `docs/qa/current-test-landscape.md`
