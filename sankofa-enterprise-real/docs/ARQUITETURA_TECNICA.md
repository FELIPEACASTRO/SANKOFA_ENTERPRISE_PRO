# Sankofa Enterprise Pro - Documentação Técnica de Arquitetura

**Versão:** 1.0.0  
**Data:** Novembro 2025  
**Classificação:** Confidencial - Uso Interno

---

> **Nota:** Este documento descreve a arquitetura implementada e planejada do sistema.
> Funcionalidades marcadas como **(Planejado)** ou **(Conceitual)** estão em desenvolvimento.
> O sistema atual opera com Flask API, cache in-memory (fallback quando Redis indisponível),
> e armazenamento baseado em arquivos JSON.

---

## Sumário

1. [Visão Geral da Arquitetura](#1-visão-geral-da-arquitetura)
2. [Stack Tecnológico](#2-stack-tecnológico)
3. [Arquitetura Clean Architecture](#3-arquitetura-clean-architecture)
4. [Componentes do Backend](#4-componentes-do-backend)
5. [Motor de Machine Learning](#5-motor-de-machine-learning)
6. [Infraestrutura MLOps](#6-infraestrutura-mlops)
7. [Sistema de Cache](#7-sistema-de-cache)
8. [Segurança e Autenticação](#8-segurança-e-autenticação)
9. [Frontend Dashboard](#9-frontend-dashboard)
10. [APIs e Endpoints](#10-apis-e-endpoints)
11. [Configuração e Deployment](#11-configuração-e-deployment)
12. [Monitoramento e Observabilidade](#12-monitoramento-e-observabilidade)

---

## 1. Visão Geral da Arquitetura

O Sankofa Enterprise Pro é um sistema de detecção de fraudes em tempo real projetado para ambientes bancários de alta escala, processando até **300 milhões de requisições por dia**.

### 1.1 Características Principais

| Característica | Especificação | Status |
|---------------|---------------|--------|
| Throughput | Variável (ambiente de desenvolvimento) | Atual |
| Latência média | ~10-50ms (ambiente de desenvolvimento) | Atual |
| Acurácia ML | 99.9% (em testes) | Validado |
| Recall | 96.7% (em testes) | Validado |
| Precisão | 100% (em testes) | Validado |
| F1-Score | 98.3% (em testes) | Validado |

> **Nota:** Métricas de ML foram validadas em ambiente de teste. Métricas de throughput
> e latência de produção dependem da infraestrutura de deployment.

### 1.2 Princípios Arquiteturais

1. **Clean Architecture**: Separação clara entre camadas (Domain, Application, Infrastructure, Presentation)
2. **Domain-Driven Design (DDD)**: Modelagem orientada ao domínio bancário
3. **Event-Driven**: Processamento assíncrono de eventos
4. **Microservices-Ready**: Componentes desacoplados e independentes
5. **Security-First**: Segurança em todas as camadas

---

## 2. Stack Tecnológico

### 2.1 Backend

```
┌─────────────────────────────────────────────────────────────┐
│                    STACK BACKEND                            │
├─────────────────────────────────────────────────────────────┤
│  Linguagem       │ Python 3.11+                             │
│  Framework Web   │ Flask 3.0.0 + Flask-CORS                 │
│  Autenticação    │ Flask-JWT-Extended 4.6.0                 │
│  Rate Limiting   │ Flask-Limiter                            │
│  ML Framework    │ scikit-learn 1.5.2+                      │
│  Gradient Boost  │ XGBoost 2.1.2+, LightGBM 4.5.0+          │
│  Data Processing │ Pandas 2.2.3+, NumPy 1.26.4+             │
│  Cache           │ Redis 7.0+ (com fallback in-memory)      │
│  Database        │ PostgreSQL 13+ (opcional)                │
│  Serialização    │ Joblib (modelos), JSON (configs)         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Frontend

```
┌─────────────────────────────────────────────────────────────┐
│                    STACK FRONTEND                           │
├─────────────────────────────────────────────────────────────┤
│  Framework       │ React 18+                                │
│  Build Tool      │ Vite                                     │
│  Styling         │ TailwindCSS                              │
│  UI Components   │ shadcn/ui                                │
│  Charts          │ Recharts                                 │
│  HTTP Client     │ Fetch API                                │
│  State Mgmt      │ React Hooks                              │
│  Routing         │ React Router                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Infraestrutura

```
┌─────────────────────────────────────────────────────────────┐
│                 INFRAESTRUTURA ATUAL                        │
├─────────────────────────────────────────────────────────────┤
│  Ambiente        │ Replit (desenvolvimento)                 │
│  Servidor Web    │ Flask built-in server                    │
│  Logging         │ Structured JSON Logging                  │
│  Cache           │ In-memory (Redis indisponível)           │
│  Armazenamento   │ Arquivos JSON                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│             INFRAESTRUTURA PLANEJADA (Produção)             │
├─────────────────────────────────────────────────────────────┤
│  Container       │ Docker + Docker Compose (Planejado)      │
│  Load Balancer   │ Nginx (Planejado)                        │
│  Monitoring      │ DataDog Integration (Planejado)          │
│  SSL/TLS         │ TLS 1.3 (Planejado)                      │
│  Encryption      │ AES-256 (Planejado)                      │
└─────────────────────────────────────────────────────────────┘
```

> **Status:** A infraestrutura de produção está planejada. O sistema atual opera em ambiente de desenvolvimento Replit.

---

## 3. Arquitetura Clean Architecture

O sistema implementa Clean Architecture com 4 camadas distintas:

### 3.1 Diagrama de Camadas

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ React        │  │ REST API     │  │ Webhooks     │              │
│  │ Dashboard    │  │ Controllers  │  │ Callbacks    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
├─────────────────────────────────────────────────────────────────────┤
│                         APPLICATION LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Use Cases    │  │ DTOs         │  │ Mappers      │              │
│  │ Orchestration│  │ Validators   │  │ Transformers │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
├─────────────────────────────────────────────────────────────────────┤
│                           DOMAIN LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Entities     │  │ Value Objects│  │ Domain       │              │
│  │ Transaction  │  │ Money, Risk  │  │ Services     │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
├─────────────────────────────────────────────────────────────────────┤
│                       INFRASTRUCTURE LAYER                          │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐           │
│  │ Database  │ │ ML Engine │ │ Cache     │ │ External  │           │
│  │ Repository│ │ Service   │ │ System    │ │ APIs      │           │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Estrutura de Diretórios

```
sankofa-enterprise-real/
├── backend/
│   ├── api/                    # Presentation Layer
│   │   ├── production_api.py   # API principal (endpoints implementados)
│   │   └── services/           # Serviços de aplicação
│   │
│   ├── core/                   # Domain Layer
│   │   ├── entities/           # Entidades de domínio
│   │   │   ├── transaction.py  # Entidade Transação
│   │   │   └── fraud_prediction.py
│   │   ├── value_objects/      # Value Objects
│   │   │   ├── money.py
│   │   │   └── transaction_id.py
│   │   └── use_cases/          # Casos de Uso
│   │       ├── analyze_transaction.py
│   │       └── process_feedback.py
│   │
│   ├── infrastructure/         # Infrastructure Layer
│   │   ├── database/           # Repositórios
│   │   ├── ml_service/         # Integração ML
│   │   └── security/           # Segurança
│   │
│   ├── ml_engine/              # Motor de ML
│   │   └── production_fraud_engine.py
│   │
│   ├── mlops/                  # MLOps Components
│   │   ├── ab_testing_manager.py
│   │   ├── canary_deployment_manager.py
│   │   └── drift_detector.py
│   │
│   ├── cache/                  # Sistema de Cache
│   │   └── redis_cache_system.py
│   │
│   ├── compliance/             # Compliance & Regulatório
│   │   └── compliance_manager.py
│   │
│   ├── security/               # Segurança Enterprise
│   │   ├── enterprise_security_system.py
│   │   └── middleware.py
│   │
│   ├── config/                 # Configurações
│   │   └── settings.py
│   │
│   └── utils/                  # Utilitários
│       ├── structured_logging.py
│       └── error_handling.py
│
├── frontend/                   # Presentation Layer (Web)
│   └── src/
│       ├── pages/              # Páginas do Dashboard
│       ├── components/         # Componentes React
│       └── styles/             # Estilos CSS
│
├── config/                     # Configurações Globais
│   └── configuration_rules.json
│
└── models/                     # Modelos ML Persistidos
```

---

## 4. Componentes do Backend

### 4.1 Production API (`production_api.py`)

A API principal do sistema, responsável por:

- **~22 endpoints REST** para todas as operações
- **Autenticação JWT** com rotação automática de chaves
- **Rate Limiting** configurável por endpoint
- **CORS** para integração com frontend
- **Error Handling** estruturado

#### Configuração de Rate Limiting

```python
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["1000 per minute", "50000 per hour"],
    storage_uri="memory://",
    strategy="fixed-window"
)
```

#### Decorator de Autenticação

```python
@require_auth
def protected_endpoint():
    # Endpoint protegido por JWT
    user = g.user  # Usuário autenticado
    ...
```

### 4.2 MetricsCollector

Componente responsável por coletar e persistir métricas em tempo real:

```
┌─────────────────────────────────────────────────────────┐
│                   METRICS COLLECTOR                      │
├─────────────────────────────────────────────────────────┤
│  • Transações do dia                                     │
│  • Estatísticas de fraude                                │
│  • Amostras de latência (últimas 1000)                   │
│  • Estatísticas por hora                                 │
│  • Estatísticas por canal (PIX, TED, etc)                │
│  • Alertas ativos                                        │
│  • Histórico diário (30 dias)                            │
├─────────────────────────────────────────────────────────┤
│  Persistência: JSON em data/metrics_state.json          │
│  Thread-safe: RLock para concorrência                    │
└─────────────────────────────────────────────────────────┘
```

### 4.3 Error Handling

Sistema de tratamento de erros com categorização:

| Categoria | Severidade | Ação |
|-----------|------------|------|
| VALIDATION | LOW/MEDIUM | Log + Response 400 |
| DATABASE | MEDIUM/HIGH | Log + Retry + Response 500 |
| ML_MODEL | HIGH | Log + Fallback + Alert |
| SECURITY | CRITICAL | Log + Block + Alert |

---

## 5. Motor de Machine Learning

### 5.1 Arquitetura do Ensemble

O `ProductionFraudEngine` implementa um **Stacking Ensemble** de alta performance:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STACKING ENSEMBLE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   INPUT (Transaction Features)                                       │
│           │                                                          │
│           ▼                                                          │
│   ┌───────────────────────────────────────────────┐                 │
│   │           PREPROCESSING LAYER                  │                 │
│   │  • Feature Selection (47+ features)            │                 │
│   │  • Missing Value Handling (median)             │                 │
│   │  • StandardScaler Normalization                │                 │
│   └───────────────────────────────────────────────┘                 │
│           │                                                          │
│           ▼                                                          │
│   ┌───────────────────────────────────────────────┐                 │
│   │           BASE MODELS (Layer 0)                │                 │
│   │  ┌─────────────────┐ ┌─────────────────┐       │                 │
│   │  │ Random Forest   │ │ Gradient Boost  │       │                 │
│   │  │ n_estimators:100│ │ n_estimators:100│       │                 │
│   │  │ max_depth: 15   │ │ max_depth: 8    │       │                 │
│   │  │ balanced weights│ │ learning: 0.1   │       │                 │
│   │  └────────┬────────┘ └────────┬────────┘       │                 │
│   └───────────│───────────────────│────────────────┘                 │
│               │                   │                                   │
│               ▼                   ▼                                   │
│   ┌───────────────────────────────────────────────┐                 │
│   │         CALIBRATION LAYER                      │                 │
│   │    CalibratedClassifierCV (isotonic)           │                 │
│   │    • Probability calibration                   │                 │
│   │    • Cross-validation: 5-fold                  │                 │
│   └───────────────────────────────────────────────┘                 │
│               │                                                      │
│               ▼                                                      │
│   ┌───────────────────────────────────────────────┐                 │
│   │         META-MODEL (Layer 1)                   │                 │
│   │    Logistic Regression                         │                 │
│   │    • Combines base model predictions           │                 │
│   │    • Balanced class weights                    │                 │
│   │    • max_iter: 1000                            │                 │
│   └───────────────────────────────────────────────┘                 │
│               │                                                      │
│               ▼                                                      │
│   ┌───────────────────────────────────────────────┐                 │
│   │         PRECISION RULES                        │                 │
│   │    • extreme_amount_suspicious_hour            │                 │
│   │    • velocity_burst detection                  │                 │
│   │    • high_risk_combination                     │                 │
│   └───────────────────────────────────────────────┘                 │
│               │                                                      │
│               ▼                                                      │
│         FraudPrediction                                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Feature Engineering

O sistema extrai **47+ features** automaticamente:

#### Temporais
- `hour`: Hora da transação (0-23)
- `day_of_week`: Dia da semana (0-6)
- `is_weekend`: Flag de fim de semana
- `is_night`: Transação noturna (22h-6h)
- `is_business_hours`: Horário comercial

#### Baseadas em Valor
- `amount_log`: Log do valor
- `amount_squared`: Valor ao quadrado
- `amount_normalized`: Valor normalizado
- `is_round_amount`: Valor redondo
- `amount_zscore`: Z-score do valor

#### Geográficas
- `distance_from_home`: Distância do local habitual
- `location_risk_score`: Score de risco da localização
- `is_international`: Transação internacional

#### Comportamentais
- `transaction_velocity_1h`: Velocidade (transações/hora)
- `transaction_velocity_24h`: Velocidade (transações/dia)
- `amount_deviation`: Desvio do padrão normal
- `new_merchant`: Comerciante novo
- `device_change`: Mudança de dispositivo

### 5.3 Precision Rules (Regras de Alta Precisão)

```python
precision_rules = {
    "extreme_amount_suspicious_hour": {
        "amount_threshold": 50000,      # R$ 50.000+
        "suspicious_hours": [0,1,2,3,4,23],  # Madrugada
        "probability_boost": 0.3        # +30% probabilidade
    },
    "velocity_burst": {
        "frequency_threshold": 50,      # 50+ transações
        "time_window_hours": 0.5,       # Em 30 minutos
        "probability_boost": 0.4        # +40% probabilidade
    },
    "high_risk_combination": {
        "location_risk_threshold": 0.9,
        "device_risk_threshold": 0.9,
        "probability_boost": 0.5        # +50% probabilidade
    }
}
```

### 5.4 FraudPrediction (Estrutura de Resposta)

```python
@dataclass
class FraudPrediction:
    transaction_id: str      # ID único da transação
    is_fraud: bool           # Classificação binária
    fraud_probability: float # Probabilidade (0-1)
    risk_score: float        # Score de risco (0-100)
    risk_level: str          # LOW/MEDIUM/HIGH/CRITICAL
    confidence: float        # Confiança do modelo
    processing_time_ms: float # Tempo de processamento
    model_version: str       # Versão do modelo
    detection_reason: List[str] # Razões da detecção
    timestamp: str           # Timestamp ISO
```

---

## 6. Infraestrutura MLOps

### 6.1 A/B Testing Manager

Sistema completo de testes A/B para comparação de modelos:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        A/B TESTING SYSTEM                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────────────────────────────────────────────┐       │
│   │                    TRAFFIC ROUTER                        │       │
│   │   Strategies:                                            │       │
│   │   • RANDOM - Distribuição aleatória                      │       │
│   │   • HASH_BASED - Consistente por transaction_id          │       │
│   │   • GEOGRAPHIC - Por região do cliente                   │       │
│   │   • TIME_BASED - Por período do dia                      │       │
│   │   • RISK_BASED - Por nível de risco                      │       │
│   └─────────────────────────────────────────────────────────┘       │
│                          │                                           │
│            ┌─────────────┼─────────────┐                            │
│            ▼             ▼             ▼                            │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │
│   │  VARIANT A   │ │  VARIANT B   │ │  VARIANT C   │                │
│   │  (Control)   │ │ (Challenger) │ │ (Challenger) │                │
│   │    60%       │ │     20%      │ │     20%      │                │
│   └──────────────┘ └──────────────┘ └──────────────┘                │
│            │             │             │                             │
│            └─────────────┼─────────────┘                            │
│                          ▼                                           │
│   ┌─────────────────────────────────────────────────────────┐       │
│   │              STATISTICAL ANALYZER                        │       │
│   │   • Chi-square test                                      │       │
│   │   • Confidence intervals                                 │       │
│   │   • Statistical significance (p < 0.05)                  │       │
│   │   • Minimum sample size validation                       │       │
│   └─────────────────────────────────────────────────────────┘       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### Configuração de Teste A/B

```python
@dataclass
class ABTestConfig:
    test_id: str                    # ID único do teste
    test_name: str                  # Nome descritivo
    description: str                # Descrição do objetivo
    variants: List[ModelVariant]    # Variantes (modelos)
    traffic_split_strategy: TrafficSplitStrategy
    start_date: str                 # Data início
    end_date: str                   # Data fim
    success_metrics: List[str]      # Métricas de sucesso
    minimum_sample_size: int        # Tamanho mínimo da amostra
    confidence_level: float         # Nível de confiança (0.95)
    status: TestStatus              # DRAFT/ACTIVE/COMPLETED
```

### 6.2 Canary Deployment Manager

Sistema de deploy gradual com rollback automático:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CANARY DEPLOYMENT FLOW                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   STEP 1 (5%)    STEP 2 (10%)   STEP 3 (25%)   STEP 4 (50%)        │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│   │  ████    │   │  ████    │   │  ████    │   │  ████    │        │
│   │  ▓▓▓▓    │   │  ████    │   │  ████    │   │  ████    │        │
│   │  ▓▓▓▓    │   │  ▓▓▓▓    │   │  ████    │   │  ████    │        │
│   │  ▓▓▓▓    │   │  ▓▓▓▓    │   │  ▓▓▓▓    │   │  ████    │        │
│   │  ▓▓▓▓    │   │  ▓▓▓▓    │   │  ▓▓▓▓    │   │  ▓▓▓▓    │        │
│   └──────────┘   └──────────┘   └──────────┘   └──────────┘        │
│   ████ = Canary   ▓▓▓▓ = Stable                                     │
│                                                                      │
│   HEALTH CHECKS em cada step:                                        │
│   ┌─────────────────────────────────────────────────────────┐       │
│   │  ✓ Error Rate < 1%                                       │       │
│   │  ✓ Latency P95 < 15ms                                    │       │
│   │  ✓ Accuracy > 99%                                        │       │
│   │  ✓ False Positive Rate < 0.5%                            │       │
│   └─────────────────────────────────────────────────────────┘       │
│                                                                      │
│   ROLLBACK AUTOMÁTICO se:                                           │
│   • Error Rate > 5%                                                  │
│   • Latency P95 > 50ms                                              │
│   • Accuracy drop > 2%                                              │
│   • 3 health checks consecutivos falhando                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### Status do Deployment

```python
class DeploymentStatus(Enum):
    PENDING = "pending"           # Aguardando início
    STARTING = "starting"         # Iniciando
    ACTIVE = "active"             # Em execução
    PROMOTING = "promoting"       # Promovendo para próximo step
    COMPLETED = "completed"       # Concluído com sucesso
    ROLLING_BACK = "rolling_back" # Executando rollback
    ROLLED_BACK = "rolled_back"   # Rollback concluído
    FAILED = "failed"             # Falhou
```

### 6.3 Drift Detector

Sistema de detecção de degradação do modelo:

```
┌─────────────────────────────────────────────────────────────────────┐
│                       DRIFT DETECTION SYSTEM                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   TIPOS DE DRIFT MONITORADOS:                                        │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │  DATA DRIFT (Distribuição de Features)                    │      │
│   │  ├─ Jensen-Shannon Divergence                             │      │
│   │  ├─ Kolmogorov-Smirnov Test                               │      │
│   │  └─ Population Stability Index (PSI)                      │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │  CONCEPT DRIFT (Relação Feature → Target)                 │      │
│   │  ├─ Performance metrics degradation                       │      │
│   │  ├─ Prediction distribution changes                       │      │
│   │  └─ Chi-square test para categóricas                      │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│   SEVERITY LEVELS:                                                   │
│   ├─ LOW:      PSI < 0.1   │ Monitorar                              │
│   ├─ MEDIUM:   PSI < 0.25  │ Investigar                             │
│   ├─ HIGH:     PSI < 0.5   │ Retreinar em breve                     │
│   └─ CRITICAL: PSI >= 0.5  │ Retreinar IMEDIATAMENTE                │
│                                                                      │
│   AÇÕES AUTOMÁTICAS:                                                 │
│   • Alert via webhook                                                │
│   • Log estruturado                                                  │
│   • Trigger de retreinamento (se configurado)                        │
│   • Dashboard notification                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Sistema de Cache

### 7.1 Redis Cache System

Arquitetura multi-camada com fallback:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CACHE ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   REQUEST                                                            │
│      │                                                               │
│      ▼                                                               │
│   ┌─────────────────────────────────────────────────────────┐       │
│   │              L1: IN-MEMORY CACHE                         │       │
│   │   • LRU eviction                                         │       │
│   │   • TTL: configurable                                    │       │
│   │   • Hit rate: ~95%                                       │       │
│   │   • Latency: <1ms                                        │       │
│   └─────────────────────────────────────────────────────────┘       │
│      │ MISS                                                          │
│      ▼                                                               │
│   ┌─────────────────────────────────────────────────────────┐       │
│   │              L2: REDIS CLUSTER                           │       │
│   │   • Connection pooling (max: 100)                        │       │
│   │   • Automatic serialization (JSON + Pickle)              │       │
│   │   • TTL-based invalidation                               │       │
│   │   • Latency: 1-5ms                                       │       │
│   └─────────────────────────────────────────────────────────┘       │
│      │ MISS / UNAVAILABLE                                            │
│      ▼                                                               │
│   ┌─────────────────────────────────────────────────────────┐       │
│   │              FALLBACK: COMPUTATION                       │       │
│   │   • Direct ML model inference                            │       │
│   │   • Store result in cache layers                         │       │
│   └─────────────────────────────────────────────────────────┘       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Cache Keys Strategy

```python
# Padrões de chaves
CACHE_KEYS = {
    "transaction": "txn:{transaction_id}",
    "user_profile": "user:{user_id}:profile",
    "merchant": "merchant:{merchant_id}",
    "model_prediction": "pred:{transaction_id}",
    "metrics": "metrics:dashboard:current",
    "config": "config:{config_name}",
}

# TTLs padrão (segundos)
CACHE_TTLS = {
    "transaction": 3600,      # 1 hora
    "user_profile": 1800,     # 30 minutos
    "merchant": 3600,         # 1 hora
    "model_prediction": 300,  # 5 minutos
    "metrics": 60,            # 1 minuto
    "config": 600,            # 10 minutos
}
```

---

## 8. Segurança e Autenticação

### 8.1 JWT Authentication

```
┌─────────────────────────────────────────────────────────────────────┐
│                      JWT AUTHENTICATION FLOW                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   1. LOGIN                                                           │
│   ┌──────────┐     POST /api/auth/login     ┌──────────────┐        │
│   │  Client  │ ─────────────────────────────▶ │  API Server  │        │
│   │          │   { username, password }      │              │        │
│   └──────────┘                               └──────────────┘        │
│                                                    │                 │
│                                                    ▼                 │
│                                              ┌──────────────┐        │
│                                              │  Validate    │        │
│                                              │  Credentials │        │
│                                              └──────────────┘        │
│                                                    │                 │
│                                                    ▼                 │
│   2. TOKEN GENERATION                                                │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │  JWT Token = Header.Payload.Signature                     │      │
│   │                                                           │      │
│   │  Header:  { "alg": "HS256", "typ": "JWT" }                │      │
│   │  Payload: {                                               │      │
│   │    "sub": "user_id",                                      │      │
│   │    "role": "analyst",                                     │      │
│   │    "permissions": ["read", "write"],                      │      │
│   │    "exp": 1234567890,                                     │      │
│   │    "iat": 1234567000                                      │      │
│   │  }                                                        │      │
│   │  Signature: HMACSHA256(base64(header) + "." +             │      │
│   │             base64(payload), JWT_SECRET)                  │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│   3. AUTHENTICATED REQUESTS                                          │
│   ┌──────────┐  Authorization: Bearer <token>  ┌──────────────┐     │
│   │  Client  │ ───────────────────────────────▶│  API Server  │     │
│   └──────────┘                                 └──────────────┘     │
│                                                       │              │
│                                                       ▼              │
│                                                ┌─────────────┐       │
│                                                │ Validate    │       │
│                                                │ JWT Token   │       │
│                                                │ • Signature │       │
│                                                │ • Expiration│       │
│                                                │ • Claims    │       │
│                                                └─────────────┘       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Security Configuration

```python
# settings.py
@dataclass
class SecurityConfig:
    jwt_secret: str           # 32+ caracteres, rotação a cada 30 dias
    jwt_algorithm: str        # HS256
    jwt_expiration_hours: int # 24 horas
    enable_rate_limiting: bool
    rate_limit_requests: int  # 1000/minuto
    enable_audit_log: bool    # Obrigatório para compliance
    encryption_key: str       # AES-256
    tls_version: str          # TLS 1.3
```

### 8.3 Role-Based Access Control (RBAC)

```
┌─────────────────────────────────────────────────────────────────────┐
│                          RBAC MATRIX                                 │
├──────────────┬───────────┬───────────┬───────────┬─────────────────┤
│  ENDPOINT    │   ADMIN   │  ANALYST  │  VIEWER   │   SYSTEM        │
├──────────────┼───────────┼───────────┼───────────┼─────────────────┤
│  /health     │    ✓      │    ✓      │    ✓      │      ✓          │
│  /predict    │    ✓      │    ✓      │    ✗      │      ✓          │
│  /batch      │    ✓      │    ✓      │    ✗      │      ✓          │
│  /feedback   │    ✓      │    ✓      │    ✗      │      ✗          │
│  /model/*    │    ✓      │    ✗      │    ✗      │      ✓          │
│  /config/*   │    ✓      │    ✗      │    ✗      │      ✗          │
│  /audit/*    │    ✓      │    ✓      │    ✓      │      ✗          │
│  /admin/*    │    ✓      │    ✗      │    ✗      │      ✗          │
└──────────────┴───────────┴───────────┴───────────┴─────────────────┘
```

---

## 9. Frontend Dashboard

### 9.1 Arquitetura React

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FRONTEND ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   src/                                                               │
│   ├── App.jsx              # Router + Layout principal               │
│   ├── pages/                                                         │
│   │   ├── Dashboard.jsx    # KPIs e visão geral                      │
│   │   ├── Transactions.jsx # Lista de transações                     │
│   │   ├── Investigation.jsx# Análise detalhada                       │
│   │   ├── ManualReview.jsx # Revisão manual HITL                     │
│   │   ├── Calibration.jsx  # Ajuste de thresholds                    │
│   │   ├── Monitoring.jsx   # Saúde do modelo                         │
│   │   ├── Metrics.jsx      # Métricas em tempo real                  │
│   │   └── Alerts.jsx       # Central de alertas                      │
│   ├── components/                                                    │
│   │   ├── ui/              # Componentes shadcn/ui                   │
│   │   ├── Sidebar.jsx      # Navegação lateral                       │
│   │   ├── Header.jsx       # Cabeçalho com busca                     │
│   │   └── charts/          # Gráficos Recharts                       │
│   └── styles/                                                        │
│       ├── App.css          # Estilos globais                         │
│       └── tokens.css       # Design tokens                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 Páginas do Dashboard

| Página | Funcionalidade | Atualização |
|--------|----------------|-------------|
| Dashboard | KPIs, gráficos de tendência, status geral | Real-time (5s) |
| Transações | Lista, filtros, busca, detalhes | On-demand |
| Investigação | Análise profunda de fraudes | On-demand |
| Revisão Manual | Queue HITL, aprovar/rejeitar | Real-time |
| Calibração | Ajuste de thresholds, impacto | On-demand |
| Monitoramento | Saúde do modelo, drift, versões | Real-time (30s) |
| Métricas | Contadores, latência, throughput | Real-time (5s) |
| Alertas | Notificações, histórico, ações | Real-time |

---

## 10. APIs e Endpoints

### 10.1 Catálogo de Endpoints

#### Health & Status

```
GET  /health                  # Health check simples
GET  /api/health              # Health check detalhado
GET  /api/status              # Status do sistema
```

#### Fraud Detection

```
POST /api/fraud/predict       # Predição single
POST /api/fraud/batch         # Predição batch
GET  /api/fraud/rules         # Regras ativas
```

#### Model Management

```
GET  /api/model/info          # Informações do modelo
GET  /api/model/metrics       # Métricas de performance
POST /api/model/train         # Trigger retreinamento
GET  /api/model/versions      # Versões disponíveis
```

#### Dashboard Data

```
GET  /api/dashboard/kpis      # KPIs principais
GET  /api/dashboard/hourly    # Dados por hora
GET  /api/dashboard/channels  # Dados por canal
GET  /api/dashboard/daily-history  # Histórico diário
```

#### Feedback & HITL

```
POST /api/feedback            # Enviar feedback
GET  /api/feedback/pending    # Casos pendentes
POST /api/feedback/resolve    # Resolver caso
```

#### MLOps

```
GET  /api/mlops/ab-tests          # Testes A/B ativos
GET  /api/mlops/canary            # Deployments canary
GET  /api/mlops/drift             # Status de drift
```

#### Auth & Config

```
POST /api/auth/login          # Autenticação
GET  /api/auth/verify         # Verificar token
GET  /api/config/rules        # Regras de configuração
PUT  /api/config/rules        # Atualizar regras
```

### 10.2 Exemplo de Request/Response

#### POST /api/fraud/predict

**Request:**
```json
{
  "transaction_id": "TXN-2025-001",
  "amount": 15000.00,
  "currency": "BRL",
  "channel": "PIX",
  "timestamp": "2025-11-27T14:30:00Z",
  "customer_id": "CUST-12345",
  "merchant_id": "MERCH-67890",
  "location": {
    "latitude": -23.5505,
    "longitude": -46.6333
  },
  "device_fingerprint": "abc123xyz"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "prediction": {
    "transaction_id": "TXN-2025-001",
    "is_fraud": false,
    "fraud_probability": 0.12,
    "risk_score": 24.5,
    "risk_level": "LOW",
    "confidence": 0.94,
    "processing_time_ms": 8.5,
    "model_version": "1.0.0",
    "detection_reason": [],
    "timestamp": "2025-11-27T14:30:00.123Z"
  }
}
```

---

## 11. Configuração e Deployment

### 11.1 Variáveis de Ambiente

```bash
# Ambiente
ENVIRONMENT=production       # development/staging/production
FLASK_DEBUG=false            # SEMPRE false em produção

# Servidor
API_PORT=8445
FRONTEND_PORT=5000

# Database (opcional)
DATABASE_URL=postgresql://user:pass@host:5432/sankofa
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sankofa
DB_USER=sankofa_user
DB_PASSWORD=secret

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Segurança
JWT_SECRET=your-32-character-secret-key-here
ENCRYPTION_KEY=your-aes-256-key

# ML
ML_MODEL_PATH=./models
ML_CONFIDENCE_THRESHOLD=0.5

# Monitoring
DATADOG_API_KEY=your-datadog-key
LOG_LEVEL=INFO
```

### 11.2 Estrutura de Configuração

```python
# config/settings.py
@dataclass
class AppConfig:
    environment: str
    debug: bool
    
@dataclass
class ServerConfig:
    api_port: int
    frontend_port: int
    host: str
    
@dataclass
class MLConfig:
    model_path: str
    confidence_threshold: float
    batch_size: int
    
@dataclass
class SecurityConfig:
    jwt_secret: str
    jwt_algorithm: str
    jwt_expiration_hours: int
    
@dataclass
class MonitoringConfig:
    log_level: str
    enable_metrics: bool
    datadog_api_key: str
```

---

## 12. Monitoramento e Observabilidade

### 12.1 Métricas Coletadas

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MÉTRICAS DO SISTEMA                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   PERFORMANCE                                                        │
│   ├─ request_latency_ms (P50, P95, P99)                             │
│   ├─ requests_per_second                                            │
│   ├─ error_rate_percentage                                          │
│   └─ cache_hit_rate                                                 │
│                                                                      │
│   MODEL                                                              │
│   ├─ prediction_latency_ms                                          │
│   ├─ fraud_detection_rate                                           │
│   ├─ false_positive_rate                                            │
│   ├─ false_negative_rate                                            │
│   ├─ model_accuracy                                                 │
│   ├─ model_precision                                                │
│   ├─ model_recall                                                   │
│   └─ model_f1_score                                                 │
│                                                                      │
│   BUSINESS                                                           │
│   ├─ transactions_today                                             │
│   ├─ frauds_detected                                                │
│   ├─ value_protected_brl                                            │
│   ├─ approval_rate                                                  │
│   └─ pending_reviews                                                │
│                                                                      │
│   INFRASTRUCTURE                                                     │
│   ├─ cpu_usage_percent                                              │
│   ├─ memory_usage_mb                                                │
│   ├─ redis_connections                                              │
│   └─ db_connection_pool_size                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 12.2 Logging Estruturado

```python
# Formato de log JSON
{
    "timestamp": "2025-11-27T14:30:00.123Z",
    "level": "INFO",
    "logger": "production_api",
    "message": "Transaction processed",
    "transaction_id": "TXN-2025-001",
    "is_fraud": false,
    "processing_time_ms": 8.5,
    "model_version": "1.0.0",
    "trace_id": "abc123",
    "span_id": "def456"
}
```

### 12.3 Alertas

| Alerta | Condição | Severidade | Ação |
|--------|----------|------------|------|
| High Latency | P95 > 50ms | WARNING | Investigar |
| Error Rate | > 1% | CRITICAL | On-call |
| Model Drift | PSI > 0.25 | WARNING | Planejar retrain |
| Cache Down | Redis offline | HIGH | Fallback ativo |
| Low Accuracy | < 95% | CRITICAL | Rollback modelo |

---

## Apêndice A: Glossário Técnico

| Termo | Definição |
|-------|-----------|
| Ensemble | Combinação de múltiplos modelos ML |
| Stacking | Técnica de ensemble com meta-modelo |
| Drift | Degradação de performance do modelo |
| Canary | Deploy gradual para minimizar riscos |
| HITL | Human-in-the-Loop (revisão manual) |
| PSI | Population Stability Index |
| JWT | JSON Web Token |
| RBAC | Role-Based Access Control |

---

**Documento mantido por:** Equipe de Engenharia Sankofa  
**Última atualização:** Novembro 2025  
**Versão:** 1.0.0
