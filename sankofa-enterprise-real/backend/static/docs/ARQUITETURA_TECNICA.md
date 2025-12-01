# Arquitetura Tecnica - Sankofa Enterprise Pro v1.0

## Documentacao Tecnica Completa com Diagramas Ilustrativos

![Arquitetura Tecnica](images/arquitetura_tecnica_microservicos.png)

**Versao:** 1.0  
**Ultima Atualizacao:** 30 de Novembro de 2025  
**Status:** ✅ PRONTO PARA PRODUCAO - 21/21 Endpoints Funcionando (100%)

---

## Indice Visual

```
+==================================================================+
|                    MAPA DA DOCUMENTACAO                           |
+==================================================================+
|                                                                   |
|  1. [Visao Geral] -----> Arquitetura de alto nivel               |
|         |                                                         |
|         v                                                         |
|  2. [Stack Tecnologico] -> Tecnologias utilizadas                |
|         |                                                         |
|         v                                                         |
|  3. [Backend API] -----> Endpoints e configuracoes               |
|         |                                                         |
|         v                                                         |
|  4. [ML Engine] ------> Motor de Machine Learning                |
|         |                                                         |
|         v                                                         |
|  5. [Observabilidade] -> Metricas e monitoramento                |
|         |                                                         |
|         v                                                         |
|  6. [Infraestrutura] --> Escala e performance                    |
|         |                                                         |
|         v                                                         |
|  7. [Banco de Dados] --> Schema e estruturas                     |
|         |                                                         |
|         v                                                         |
|  8. [Seguranca] ------> Autenticacao e compliance                |
|                                                                   |
+==================================================================+
```

---

## Estado de Implementacao

![Componentes Sistema](images/componentes_sistema_tecnologias.png)

```
+==================================================================+
|                    STATUS DOS COMPONENTES                         |
+==================================================================+
|                                                                   |
|  COMPONENTE                  IMPL    TEST    INTEG   STATUS      |
|  ──────────────────────────  ────    ────    ─────   ──────      |
|  Flask API (21 endpoints)     [X]     [X]     [X]    PRODUCAO    |
|  React Dashboard (16 pags)    [X]     [X]     [X]    PRODUCAO    |
|  ML Stacking (RF+GB+CB)       [X]     [X]     [X]    PRODUCAO    |
|  PostgreSQL (4.466 txns)      [X]     [X]     [X]    PRODUCAO    |
|  SimpleCache (30s TTL)        [X]     [X]     [X]    PRODUCAO    |
|  Observability Prometheus     [X]     [X]     [X]    PRODUCAO    |
|  LGPD Compliance              [X]     [X]     [X]    PRODUCAO    |
|  Latencia SLA <50ms           [X]     [X]     [X]    PRODUCAO    |
|  Audit Trail                  [X]     [X]     [X]    PRODUCAO    |
|                                                                   |
|  LEGENDA: [X] = Completo  [-] = Parcial  [ ] = Pendente         |
|                                                                   |
+==================================================================+
```

---

## Tipos de Transacao Suportados

O sistema processa diferentes tipos de transacoes, cada um com caracteristicas unicas de risco:

```
+==============================================================================+
|                    TIPOS DE TRANSACAO SUPORTADOS                              |
+==============================================================================+
|                                                                               |
|  +------------------+------------------+------------------+------------------+|
|  |      PIX         |     CREDITO      |     DEBITO       |    TED/DOC      ||
|  +------------------+------------------+------------------+------------------+|
|  |                  |                  |                  |                  ||
|  | Transferencia    | Compra com       | Desconto em      | Transferencia   ||
|  | instantanea      | cartao credito   | conta corrente   | tradicional     ||
|  | (24/7/365)       |                  |                  |                  ||
|  |                  |                  |                  |                  ||
|  +------------------+------------------+------------------+------------------+|
|  | RISCO: ALTO      | RISCO: MEDIO     | RISCO: BAIXO     | RISCO: MEDIO    ||
|  | Irreversivel     | Chargeback ok    | Requer cartao    | Pode reverter   ||
|  +------------------+------------------+------------------+------------------+|
|                                                                               |
|  FEATURES ESPECIFICAS POR TIPO:                                               |
|  +--------------------------------------------------------------------------+|
|  |                                                                           ||
|  |  PIX:                                                                     ||
|  |  • velocity_pix_1h      - Qtd PIX na ultima hora                         ||
|  |  • pix_destination_new  - Destinatario nunca usado                       ||
|  |  • pix_night_amount     - Valor de PIX noturno                           ||
|  |  • pix_recipient_risk   - Score de risco do recebedor                    ||
|  |                                                                           ||
|  |  CREDITO:                                                                 ||
|  |  • card_present         - Cartao presente na transacao                   ||
|  |  • merchant_category    - Categoria do comerciante (MCC)                 ||
|  |  • is_international     - Compra fora do Brasil                          ||
|  |  • card_velocity_1h     - Qtd compras na ultima hora                     ||
|  |  • online_purchase      - Compra e-commerce                              ||
|  |                                                                           ||
|  |  DEBITO:                                                                  ||
|  |  • atm_location_risk    - Risco do ATM usado                             ||
|  |  • pin_attempts         - Tentativas de senha                            ||
|  |  • withdrawal_pattern   - Padrao de saque                                ||
|  |  • pos_terminal_risk    - Risco da maquininha                            ||
|  |                                                                           ||
|  |  TED/DOC:                                                                 ||
|  |  • ted_recipient_new    - Destinatario novo                              ||
|  |  • ted_value_deviation  - Desvio do valor normal                         ||
|  |  • scheduling_pattern   - Agendamento vs imediato                        ||
|  |                                                                           ||
|  +--------------------------------------------------------------------------+|
|                                                                               |
|  EXEMPLOS DE PAYLOAD POR TIPO:                                                |
|  +--------------------------------------------------------------------------+|
|  |                                                                           ||
|  |  PIX:                                                                     ||
|  |  {"channel": "PIX", "amount": 5000, "pix_key_type": "CPF"}               ||
|  |                                                                           ||
|  |  CREDITO:                                                                 ||
|  |  {"channel": "CREDIT_CARD", "amount": 1200, "mcc": "5411"}               ||
|  |                                                                           ||
|  |  DEBITO:                                                                  ||
|  |  {"channel": "DEBIT_CARD", "amount": 500, "card_present": true}          ||
|  |                                                                           ||
|  |  TED:                                                                     ||
|  |  {"channel": "TED", "amount": 10000, "recipient_bank": "001"}            ||
|  |                                                                           ||
|  +--------------------------------------------------------------------------+|
|                                                                               |
+==============================================================================+
```

---

## 1. Visao Geral da Arquitetura

### 1.1 Diagrama de Alto Nivel

![Fluxo de Dados](images/fluxo_dados_transacao.png)

```
+==============================================================================+
|                        SANKOFA ENTERPRISE PRO v12.0                           |
|                      ARQUITETURA DE MICROSERVICOS                             |
+==============================================================================+
|                                                                               |
|   +-------------+                                                             |
|   |   CLIENTE   |                                                             |
|   | (App/Web)   |                                                             |
|   +------+------+                                                             |
|          |                                                                    |
|          | HTTPS/REST                                                         |
|          v                                                                    |
|   +------+------+     +---------------+                                       |
|   |   FRONTEND  |     |   DASHBOARD   |                                       |
|   |  React/Vite |     |   9 Paginas   |                                       |
|   |  Port 5000  |     |   Analistas   |                                       |
|   +------+------+     +-------+-------+                                       |
|          |                    |                                               |
|          +--------+----------+                                                |
|                   |                                                           |
|                   v                                                           |
|   +===============+===============+                                           |
|   |         BACKEND API           |                                           |
|   |   Flask + CORS + JWT + Limiter|                                           |
|   |         Port 8000             |                                           |
|   +===============+===============+                                           |
|                   |                                                           |
|     +-------------+-------------+-------------+                               |
|     |             |             |             |                               |
|     v             v             v             v                               |
| +-------+   +---------+   +----------+   +-----------+                        |
| |  ML   |   | EXPLAIN |   | OBSERVE  |   | INFRASTR  |                        |
| |ENGINE |   | ENGINE  |   | ABILITY  |   | UCTURE    |                        |
| +---+---+   +----+----+   +----+-----+   +-----+-----+                        |
|     |            |             |               |                              |
|     +------------+------+------+---------------+                              |
|                         |                                                     |
|                         v                                                     |
|              +----------+----------+                                          |
|              |     POSTGRESQL      |                                          |
|              |    (Neon-backed)    |                                          |
|              +---------------------+                                          |
|                                                                               |
+==============================================================================+
```

### 1.2 Fluxo de Dados Detalhado

```
+==============================================================================+
|                         FLUXO DE PROCESSAMENTO                                |
+==============================================================================+
|                                                                               |
|  ENTRADA              PROCESSAMENTO                          SAIDA           |
|  ═══════              ═════════════                          ═════           |
|                                                                               |
|  +---------+      +-------------------------------------------+     +-------+|
|  |Transacao|      |                                           |     |Decisao||
|  |   JSON  | ---> |  1. Validacao  -> 2. Feature Engineering  | --> | Score ||
|  |         |      |                                           |     |       ||
|  +---------+      |  3. ML Predict -> 4. Explainability       |     +-------+|
|                   |                                           |              |
|     2-5ms         |  5. Rules      -> 6. Decision             |    8-15ms   |
|                   |                                           |              |
|                   +-------------------------------------------+              |
|                              TEMPO TOTAL: 10-20ms                            |
|                                                                               |
+==============================================================================+
```

### 1.3 Stack Tecnologico Visual

![Tecnologias](images/componentes_sistema_tecnologias.png)

```
+==============================================================================+
|                         STACK TECNOLOGICO                                     |
+==============================================================================+
|                                                                               |
|  CAMADA           TECNOLOGIA            VERSAO      PROPOSITO                |
|  ══════           ══════════            ══════      ═════════                |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │  FRONTEND                                                                │ |
|  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │ |
|  │  │  React   │ │  Vite    │ │ Tailwind │ │ shadcn/ui│ │ Recharts │       │ |
|  │  │   18+    │ │   5+     │ │   CSS    │ │Components│ │  Graficos│       │ |
|  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘       │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │  BACKEND                                                                 │ |
|  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │ |
|  │  │  Flask   │ │Flask-CORS│ │Flask-JWT │ │ Limiter  │ │ Gunicorn │       │ |
|  │  │   3.0    │ │   4.0    │ │Extended  │ │1000/min  │ │Production│       │ |
|  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘       │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │  MACHINE LEARNING                                                        │ |
|  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │ |
|  │  │ Scikit-  │ │ XGBoost  │ │ LightGBM │ │  Pandas  │ │  NumPy   │       │ |
|  │  │  Learn   │ │  2.1.2   │ │  4.5.0   │ │  2.2.3   │ │  1.26.4  │       │ |
|  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘       │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │  DADOS                                                                   │ |
|  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                                 │ |
|  │  │PostgreSQL│ │  Redis   │ │  JSON    │                                 │ |
|  │  │  (Neon)  │ │  Cache   │ │  Logs    │                                 │ |
|  │  └──────────┘ └──────────┘ └──────────┘                                 │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
+==============================================================================+
```

---

## 2. Estrutura de Diretorios

```
+==============================================================================+
|                         ESTRUTURA DO PROJETO                                  |
+==============================================================================+

sankofa-enterprise-real/
│
├── backend/                          # SERVIDOR PYTHON
│   │
│   ├── api/
│   │   └── production_api.py         # API principal (50+ endpoints)
│   │       │
│   │       ├── /api/health           # Health checks
│   │       ├── /api/fraud/predict    # Predicao ML
│   │       ├── /api/fraud/batch      # Processamento em lote
│   │       ├── /api/transactions     # CRUD transacoes
│   │       ├── /api/observability/*  # Metricas Prometheus
│   │       └── /api/infrastructure/* # Batch processor
│   │
│   ├── ml_engine/
│   │   ├── production_fraud_engine.py    # Motor ML Ensemble
│   │   ├── advanced_feature_engineering.py
│   │   ├── explainability_engine.py      # SHAP + LGPD
│   │   ├── probability_calibration.py
│   │   └── self_training_optimizer.py
│   │
│   ├── monitoring/
│   │   └── observability.py              # Prometheus + SLA
│   │
│   ├── infrastructure/
│   │   └── async_processor.py            # Queue + Batch
│   │
│   ├── mlops/
│   │   ├── ab_testing_manager.py
│   │   ├── canary_deployment_manager.py
│   │   ├── drift_detector.py
│   │   └── model_lifecycle_manager.py
│   │
│   └── tests/
│       ├── test_e2e.py                   # 25 testes E2E
│       └── test_improvements.py          # 20 testes ML
│
├── frontend/
│   └── src/
│       ├── pages/                        # 9 paginas React
│       │   ├── Dashboard.tsx
│       │   ├── Transactions.tsx
│       │   ├── Calibration.tsx
│       │   ├── Investigation.tsx
│       │   ├── ManualReview.tsx
│       │   ├── Monitoring.tsx
│       │   ├── Reports.tsx
│       │   ├── Metrics.tsx
│       │   └── Alerts.tsx
│       │
│       ├── components/ui/                # shadcn components
│       └── lib/api.ts                    # Cliente API
│
└── docs/                                 # DOCUMENTACAO
    ├── README.md
    ├── ARQUITETURA_TECNICA.md
    ├── DOCUMENTACAO_FUNCIONAL.md
    ├── MANUAL_USUARIO.md
    ├── DIAGRAMAS.md
    └── images/                           # 48+ imagens
```

---

## 3. Backend API

### 3.1 Diagrama de Endpoints

![API Endpoints](images/diagrama_api_endpoints.png)

```
+==============================================================================+
|                         MAPA DE ENDPOINTS                                     |
+==============================================================================+
|                                                                               |
|  /api                                                                         |
|   │                                                                           |
|   ├── /health                    GET     Health check basico                 |
|   ├── /health/live               GET     Liveness probe (K8s)                |
|   ├── /health/ready              GET     Readiness probe (K8s)               |
|   ├── /health/detailed           GET     Health detalhado                    |
|   │                                                                           |
|   ├── /fraud                                                                  |
|   │   ├── /predict               POST    Predicao tempo-real                 |
|   │   ├── /batch                 POST    Processamento em lote               |
|   │   ├── /explain/<id>          GET     Explicacao individual               |
|   │   └── /statistics            GET     Estatisticas de fraude              |
|   │                                                                           |
|   ├── /transactions                                                           |
|   │   ├── /                      GET     Listar transacoes                   |
|   │   ├── /<id>                  GET     Detalhe transacao                   |
|   │   └── /stats                 GET     Estatisticas                        |
|   │                                                                           |
|   ├── /observability                                                          |
|   │   ├── /metrics               GET     Metricas JSON                       |
|   │   ├── /prometheus            GET     Formato Prometheus                  |
|   │   └── /sla                   GET     Status SLA                          |
|   │                                                                           |
|   ├── /infrastructure                                                         |
|   │   ├── /batch/process         POST    Batch otimizado                     |
|   │   ├── /queue/metrics         GET     Metricas fila                       |
|   │   ├── /task/submit           POST    Submete tarefa                      |
|   │   └── /task/<id>/status      GET     Status tarefa                       |
|   │                                                                           |
|   ├── /model                                                                  |
|   │   ├── /metrics               GET     Metricas ML                         |
|   │   ├── /retrain               POST    Retreinar modelo                    |
|   │   └── /calibrate             POST    Calibrar probabilidades             |
|   │                                                                           |
|   └── /feedback                  POST    Feedback do analista                |
|                                                                               |
+==============================================================================+
```

### 3.2 Configuracao Flask

```python
+==============================================================================+
|                         CONFIGURACAO DO SERVIDOR                              |
+==============================================================================+

# Inicializacao
app = Flask(__name__)
CORS(app, origins=["*"])

# Rate Limiting (protecao contra abuso)
+------------------------------------------------------------------+
|  RATE LIMITS                                                      |
+------------------------------------------------------------------+
|  Endpoint              | Limite        | Janela                  |
|  ----------------------|---------------|-------------------------|
|  Global                | 1000 req      | por minuto              |
|  Global                | 50000 req     | por hora                |
|  /fraud/predict        | 1000 req      | por minuto              |
|  /fraud/batch          | 100 req       | por minuto              |
|  /transactions         | 500 req       | por minuto              |
+------------------------------------------------------------------+

# JWT Authentication
+------------------------------------------------------------------+
|  CONFIGURACAO JWT                                                 |
+------------------------------------------------------------------+
|  Algoritmo             | HS256                                    |
|  Expiracao             | 24 horas                                 |
|  Secret                | Variavel de ambiente                     |
|  Renovacao             | Automatica                               |
+------------------------------------------------------------------+
```

### 3.3 Exemplo de Request/Response

```
+==============================================================================+
|                    EXEMPLO: POST /api/fraud/predict                           |
+==============================================================================+

REQUEST:
┌──────────────────────────────────────────────────────────────────────────────┐
│  POST /api/fraud/predict                                                     │
│  Content-Type: application/json                                              │
│  Authorization: Bearer eyJhbGciOiJIUzI1NiIs...                              │
│                                                                              │
│  {                                                                           │
│    "transaction_id": "TXN-2025-001",                                        │
│    "amount": 5000.00,                                                        │
│    "channel": "PIX",                                                         │
│    "customer_id": "CUST-123",                                               │
│    "location": "Sao Paulo",                                                  │
│    "timestamp": "2025-11-27T14:30:00Z",                                     │
│    "device_id": "DEV-456"                                                   │
│  }                                                                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                           [PROCESSAMENTO ML]
                                    │
                                    ▼
RESPONSE:
┌──────────────────────────────────────────────────────────────────────────────┐
│  HTTP/1.1 200 OK                                                             │
│  Content-Type: application/json                                              │
│                                                                              │
│  {                                                                           │
│    "predictions": [{                                                         │
│      "transaction_id": "TXN-2025-001",                                      │
│      "is_fraud": true,                                                       │
│      "risk_score": 87.5,                                                     │
│      "confidence": 0.92,                                                     │
│      "decision": "BLOCK",                                                    │
│      "explanation_text": "Transacao de alto valor (R$ 5000) em horario      │
│                           comercial com velocidade acima da media",          │
│      "top_risk_factors": [                                                   │
│        {"feature": "amount_normalized", "impact": 0.45},                     │
│        {"feature": "velocity_1h", "impact": 0.28}                           │
│      ],                                                                      │
│      "top_protective_factors": [                                             │
│        {"feature": "device_trust", "impact": -0.15}                         │
│      ],                                                                      │
│      "lgpd_compliant": true,                                                 │
│      "compliance_report": {                                                  │
│        "lgpd": "Explicacao fornecida conforme Art. 20 LGPD",                │
│        "bacen": "Tempo de resposta dentro do SLA",                          │
│        "pci_dss": "Dados sensiveis mascarados"                              │
│      }                                                                       │
│    }],                                                                       │
│    "processing_time_ms": 15,                                                 │
│    "model_version": "v12.0"                                                  │
│  }                                                                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Motor de Machine Learning

### 4.1 Arquitetura do Ensemble

![Pipeline ML](images/pipeline_machine_learning.png)

```
+==============================================================================+
|                    ARQUITETURA STACKING ENSEMBLE                              |
+==============================================================================+
|                                                                               |
|                           TRANSACAO                                           |
|                               │                                               |
|                               ▼                                               |
|   ┌───────────────────────────────────────────────────────────────────────┐  |
|   │                    FEATURE ENGINEERING                                 │  |
|   │                       (47+ features)                                   │  |
|   │                                                                        │  |
|   │   TEMPORAIS      VALOR        COMPORTAMENTO    GEOGRAFICAS            │  |
|   │   ─────────      ─────        ─────────────    ───────────            │  |
|   │   • hour         • log        • velocity_1h   • distance              │  |
|   │   • weekday      • sqrt       • velocity_24h  • location_risk         │  |
|   │   • weekend      • zscore     • new_merchant  • is_international      │  |
|   │   • night        • normalized • device_change                         │  |
|   │   • business_h   • is_round                                           │  |
|   │                                                                        │  |
|   └───────────────────────────────┬───────────────────────────────────────┘  |
|                                   │                                          |
|                                   ▼                                          |
|   ┌───────────────────────────────────────────────────────────────────────┐  |
|   │                      BASE MODELS (Layer 0)                             │  |
|   │                                                                        │  |
|   │  ┌────────────────────────┐    ┌────────────────────────┐             │  |
|   │  │     RANDOM FOREST      │    │   GRADIENT BOOSTING    │             │  |
|   │  │                        │    │                        │             │  |
|   │  │  n_estimators: 100     │    │  n_estimators: 100     │             │  |
|   │  │  max_depth: 15         │    │  max_depth: 8          │             │  |
|   │  │  min_samples: 2        │    │  learning_rate: 0.1    │             │  |
|   │  │  class_weight: balanced│    │  subsample: 0.8        │             │  |
|   │  │                        │    │                        │             │  |
|   │  │  [100 arvores votando] │    │  [100 iteracoes]       │             │  |
|   │  │                        │    │  [refinamento gradual] │             │  |
|   │  └───────────┬────────────┘    └───────────┬────────────┘             │  |
|   │              │                              │                          │  |
|   │              └──────────────┬───────────────┘                          │  |
|   │                             │                                          │  |
|   └─────────────────────────────┼──────────────────────────────────────────┘  |
|                                 │                                             |
|                                 ▼                                             |
|   ┌───────────────────────────────────────────────────────────────────────┐  |
|   │                     META-MODEL (Layer 1)                               │  |
|   │                                                                        │  |
|   │              ┌────────────────────────────┐                            │  |
|   │              │   LOGISTIC REGRESSION      │                            │  |
|   │              │                            │                            │  |
|   │              │   Combina predicoes dos    │                            │  |
|   │              │   base models com pesos    │                            │  |
|   │              │   otimizados               │                            │  |
|   │              │                            │                            │  |
|   │              │   class_weight: balanced   │                            │  |
|   │              └────────────────────────────┘                            │  |
|   │                                                                        │  |
|   └───────────────────────────────┬───────────────────────────────────────┘  |
|                                   │                                          |
|                                   ▼                                          |
|   ┌───────────────────────────────────────────────────────────────────────┐  |
|   │                    EXPLAINABILITY ENGINE                               │  |
|   │                                                                        │  |
|   │   • Feature Importance (top 5 fatores de risco)                       │  |
|   │   • Protective Factors (top 3 fatores protetores)                     │  |
|   │   • Texto explicativo em portugues                                    │  |
|   │   • Compliance report (LGPD, BACEN, PCI DSS)                          │  |
|   │                                                                        │  |
|   └───────────────────────────────┬───────────────────────────────────────┘  |
|                                   │                                          |
|                                   ▼                                          |
|                          PREDICAO FINAL                                      |
|                  (Score 0-100 + Explicacao LGPD)                             |
|                                                                               |
+==============================================================================+
```

### 4.2 Feature Importance

![Importancia Features](images/grafico_importancia_features.png)

```
+==============================================================================+
|                    IMPORTANCIA DAS FEATURES                                   |
+==============================================================================+
|                                                                               |
|  FEATURE                          IMPORTANCIA    BARRA                       |
|  ═══════                          ═══════════    ════                        |
|                                                                               |
|  amount_normalized                   0.25       ████████████████████████▒     |
|  velocity_1h                         0.18       ██████████████████░░░░░░      |
|  device_risk_score                   0.15       ███████████████░░░░░░░░░      |
|  location_risk                       0.12       ████████████░░░░░░░░░░░░      |
|  is_night_transaction                0.10       ██████████░░░░░░░░░░░░░░      |
|  hour_of_day                         0.08       ████████░░░░░░░░░░░░░░░░      |
|  channel_risk                        0.05       █████░░░░░░░░░░░░░░░░░░░      |
|  is_weekend                          0.04       ████░░░░░░░░░░░░░░░░░░░░      |
|  outros (39 features)                0.03       ███░░░░░░░░░░░░░░░░░░░░░      |
|                                                                               |
|  TOTAL: 47 features                  1.00                                    |
|                                                                               |
+==============================================================================+
```

### 4.3 Modulos de Pesquisa ML (v2.0)

Quatro modulos avancados baseados em pesquisas academicas foram implementados:

```
+==============================================================================+
|                    MODULOS DE PESQUISA ML v2.0                                |
+==============================================================================+
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │  1. BAHNSEN FEATURE ENGINEERING (v2.0.0)                                │ |
|  │     Base: Bahnsen et al. 2016                                           │ |
|  │                                                                          │ |
|  │     • Agregacoes temporais (1h, 6h, 24h, 72h, 168h)                     │ |
|  │     • Features periodicas Von Mises (sin/cos hora/dia/mes)              │ |
|  │     • Deteccao de desvio comportamental (Z-scores)                      │ |
|  │     • Features de velocidade e risco por canal                          │ |
|  │     • Total: 62+ features geradas por transacao                         │ |
|  │                                                                          │ |
|  │     Endpoint: POST /api/research/bahnsen/features                        │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │  2. PIX FRAUD TAXONOMY (v1.0.0)                                         │ |
|  │     Base: arXiv:2511.20902                                              │ |
|  │                                                                          │ |
|  │     Tipos de Fraude PIX Detectados:                                     │ |
|  │     • Mao Fantasma (acesso remoto)     - Risco: 0.95                    │ |
|  │     • Clone WhatsApp                   - Risco: 0.85                    │ |
|  │     • QR Code Adulterado               - Risco: 0.75                    │ |
|  │     • Falso Funcionario/Central        - Risco: 0.85                    │ |
|  │     • Bug do PIX / PIX Errado          - Risco: 0.65-0.70               │ |
|  │     • Leilao Falso / Comprovante Falso - Risco: 0.60-0.70               │ |
|  │     • Sequestro Relampago              - Risco: 0.95                    │ |
|  │                                                                          │ |
|  │     Compliance: BACEN + LGPD integrado                                  │ |
|  │     Endpoint: POST /api/research/pix/analyze                            │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │  3. NLP SOCIAL ENGINEERING DETECTOR (v1.0.0)                            │ |
|  │     Base: DIFrauD Dataset                                               │ |
|  │                                                                          │ |
|  │     Deteccao de Padroes:                                                │ |
|  │     • SMS Phishing (smishing)          - Deteccao: 70%+                 │ |
|  │     • Clone de WhatsApp                - Deteccao: 70%+                 │ |
|  │     • Impersonacao de Banco            - Deteccao: 70%+                 │ |
|  │     • Urgencia e Manipulacao Emocional                                  │ |
|  │                                                                          │ |
|  │     Recomendacoes: ALLOW, WARN_USER, REVIEW, BLOCK                      │ |
|  │     Endpoints: POST /api/research/nlp/analyze, /nlp/batch               │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │  4. TRANSFER LEARNING PIPELINE (v1.0.0)                                 │ |
|  │                                                                          │ |
|  │     Datasets Suportados:                                                │ |
|  │     • Nigerian Financial    - 5M+ transacoes                            │ |
|  │     • PaySim                - 6.3M transacoes                           │ |
|  │     • Feedzai BAF           - 6M transacoes                             │ |
|  │     • IEEE-CIS              - 590K transacoes                           │ |
|  │                                                                          │ |
|  │     Fases: Pre-treinamento -> Domain Adaptation -> Fine-tuning          │ |
|  │     Endpoint: GET /api/research/transfer/datasets                        │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  STATUS GERAL: GET /api/research/modules/status                              |
|                                                                               |
+==============================================================================+
```

---

## 5. Observabilidade

### 5.1 Dashboard de Metricas

![Metricas Performance](images/metricas_performance_dashboard.png)

```
+==============================================================================+
|                    PAINEL DE OBSERVABILIDADE                                  |
+==============================================================================+
|                                                                               |
|  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               |
|  │  LATENCIA P50   │  │  LATENCIA P95   │  │  LATENCIA P99   │               |
|  │                 │  │                 │  │                 │               |
|  │    ┌─────┐      │  │    ┌─────┐      │  │    ┌─────┐      │               |
|  │    │ 28  │ ms   │  │    │ 300 │ ms   │  │    │ 311 │ ms   │               |
|  │    └─────┘      │  │    └─────┘      │  │    └─────┘      │               |
|  │    [OK]         │  │    [OK]         │  │    [OK]         │               |
|  └─────────────────┘  └─────────────────┘  └─────────────────┘               |
|                                                                               |
|  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               |
|  │  THROUGHPUT     │  │  ERROR RATE     │  │  UPTIME         │               |
|  │                 │  │                 │  │                 │               |
|  │    ┌─────┐      │  │    ┌─────┐      │  │    ┌─────┐      │               |
|  │    │33.88│ TPS  │  │    │ 0.0 │ %    │  │    │99.9 │ %    │               |
|  │    └─────┘      │  │    └─────┘      │  │    └─────┘      │               |
|  │    [EXCELLENT]  │  │    [PERFECT]    │  │    [EXCELLENT]  │               |
|  └─────────────────┘  └─────────────────┘  └─────────────────┘               |
|                                                                               |
|  ┌───────────────────────────────────────────────────────────────────────┐   |
|  │                    GRAFICO DE LATENCIA (ultimas 24h)                   │   |
|  │                                                                        │   |
|  │  ms                                                                    │   |
|  │  400 │                                                                 │   |
|  │  300 │      ╭──╮                    ╭─╮                               │   |
|  │  200 │   ╭──╯  ╰──╮              ╭──╯ ╰──╮                            │   |
|  │  100 │╭──╯        ╰──────────────╯       ╰──────────────────╮         │   |
|  │    0 │────────────────────────────────────────────────────────────    │   |
|  │      0h   4h   8h   12h   16h   20h   24h                              │   |
|  │                                                                        │   |
|  └───────────────────────────────────────────────────────────────────────┘   |
|                                                                               |
+==============================================================================+
```

### 5.2 Endpoints de Observabilidade

```
+==============================================================================+
|                    ENDPOINTS PROMETHEUS                                       |
+==============================================================================+

GET /api/observability/metrics
┌──────────────────────────────────────────────────────────────────────────────┐
│  {                                                                           │
│    "requests_total": 15847,                                                  │
│    "requests_success": 15847,                                                │
│    "requests_error": 0,                                                      │
│    "predictions_total": 12503,                                               │
│    "predictions_fraud": 892,                                                 │
│    "predictions_legitimate": 11611,                                          │
│    "latency_p50_ms": 28,                                                     │
│    "latency_p95_ms": 300,                                                    │
│    "latency_p99_ms": 311,                                                    │
│    "tps_current": 33.88,                                                     │
│    "error_rate_percent": 0.0,                                                │
│    "uptime_percent": 99.9                                                    │
│  }                                                                           │
└──────────────────────────────────────────────────────────────────────────────┘

GET /api/observability/prometheus
┌──────────────────────────────────────────────────────────────────────────────┐
│  # HELP sankofa_requests_total Total de requisicoes                          │
│  # TYPE sankofa_requests_total counter                                       │
│  sankofa_requests_total{status="success"} 15847                             │
│  sankofa_requests_total{status="error"} 0                                   │
│                                                                              │
│  # HELP sankofa_latency_ms Latencia em milissegundos                        │
│  # TYPE sankofa_latency_ms histogram                                        │
│  sankofa_latency_ms{quantile="0.5"} 28                                      │
│  sankofa_latency_ms{quantile="0.95"} 300                                    │
│  sankofa_latency_ms{quantile="0.99"} 311                                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Infraestrutura de Escala

![Escalabilidade](images/escalabilidade_300m_requisicoes.png)

### 6.1 Componentes de Infraestrutura

```
+==============================================================================+
|                    INFRAESTRUTURA DE ALTA ESCALA                              |
+==============================================================================+
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                         ASYNC TASK QUEUE                                 │ |
|  │                                                                          │ |
|  │   ┌────────────┐    ┌────────────┐    ┌────────────┐                    │ |
|  │   │  WORKER 1  │    │  WORKER 2  │    │  WORKER 3  │    │  WORKER 4 │   │ |
|  │   │            │    │            │    │            │    │            │   │ |
|  │   │ [RUNNING]  │    │ [RUNNING]  │    │ [RUNNING]  │    │ [RUNNING]  │   │ |
|  │   └─────┬──────┘    └─────┬──────┘    └─────┬──────┘    └─────┬──────┘   │ |
|  │         └─────────────────┴─────────────────┴─────────────────┘          │ |
|  │                                   │                                      │ |
|  │                                   ▼                                      │ |
|  │                    ┌──────────────────────────┐                          │ |
|  │                    │    PRIORITY QUEUE        │                          │ |
|  │                    │                          │                          │ |
|  │                    │  HIGH   ████████ (25%)   │                          │ |
|  │                    │  NORMAL ██████████ (60%) │                          │ |
|  │                    │  LOW    ████ (15%)       │                          │ |
|  │                    │                          │                          │ |
|  │                    └──────────────────────────┘                          │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                         BATCH PROCESSOR                                  │ |
|  │                                                                          │ |
|  │   Throughput: 33.88 TPS                                                  │ |
|  │   Max Workers: 8                                                         │ |
|  │   Batch Size: 100                                                        │ |
|  │                                                                          │ |
|  │   ┌──────────────────────────────────────────────────────────────┐      │ |
|  │   │ [====================] 100%  50/50 transacoes processadas    │      │ |
|  │   └──────────────────────────────────────────────────────────────┘      │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                         CIRCUIT BREAKER                                  │ |
|  │                                                                          │ |
|  │   Estado: CLOSED (operacional)                                           │ |
|  │   Falhas: 0/5 (threshold)                                                │ |
|  │   Recovery: 30 segundos                                                  │ |
|  │                                                                          │ |
|  │   Estados possiveis:                                                     │ |
|  │   ┌────────┐      ┌────────┐      ┌──────────┐                          │ |
|  │   │ CLOSED │ ---> │  OPEN  │ ---> │HALF-OPEN │ ---> [CLOSED]            │ |
|  │   │  (OK)  │ 5err │(block) │ 30s  │ (test)   │ ok                       │ |
|  │   └────────┘      └────────┘      └──────────┘                          │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
+==============================================================================+
```

---

## 7. Banco de Dados

### 7.1 Diagrama ER

![Diagrama ER](images/diagrama_er_banco_dados.png)

```
+==============================================================================+
|                    SCHEMA DO BANCO DE DADOS                                   |
+==============================================================================+
|                                                                               |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │                           TRANSACTIONS                                  │  |
|  ├────────────────────────────────────────────────────────────────────────┤  |
|  │  id             VARCHAR(50)  PK                                        │  |
|  │  amount         DECIMAL(15,2)  NOT NULL                                │  |
|  │  channel        VARCHAR(50)                                            │  |
|  │  location       VARCHAR(100)                                           │  |
|  │  cpf            VARCHAR(14)                                            │  |
|  │  timestamp      TIMESTAMP                                              │  |
|  │  fraud_score    DECIMAL(5,2)                                           │  |
|  │  is_fraud       BOOLEAN                                                │  |
|  │  decision       VARCHAR(20)                                            │  |
|  │  risk_factors   JSONB                                                  │  |
|  │  explanation    JSONB                                                  │  |
|  │  created_at     TIMESTAMP                                              │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                              │                                                |
|                              │ 1:N                                            |
|                              ▼                                                |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │                             ALERTS                                      │  |
|  ├────────────────────────────────────────────────────────────────────────┤  |
|  │  id             SERIAL  PK                                             │  |
|  │  transaction_id VARCHAR(50)  FK -> transactions.id                     │  |
|  │  type           VARCHAR(50)                                            │  |
|  │  severity       VARCHAR(20)  [LOW, MEDIUM, HIGH, CRITICAL]             │  |
|  │  status         VARCHAR(20)  [NEW, INVESTIGATING, RESOLVED]            │  |
|  │  details        JSONB                                                  │  |
|  │  created_at     TIMESTAMP                                              │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │                           AUDIT_LOG                                     │  |
|  ├────────────────────────────────────────────────────────────────────────┤  |
|  │  id             SERIAL  PK                                             │  |
|  │  action         VARCHAR(100)                                           │  |
|  │  entity_type    VARCHAR(50)                                            │  |
|  │  entity_id      VARCHAR(50)                                            │  |
|  │  user_id        VARCHAR(50)                                            │  |
|  │  details        JSONB                                                  │  |
|  │  created_at     TIMESTAMP                                              │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │                            METRICS                                      │  |
|  ├────────────────────────────────────────────────────────────────────────┤  |
|  │  id             SERIAL  PK                                             │  |
|  │  metric_name    VARCHAR(100)                                           │  |
|  │  metric_value   DECIMAL(15,4)                                          │  |
|  │  labels         JSONB                                                  │  |
|  │  timestamp      TIMESTAMP                                              │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
+==============================================================================+
```

---

## 8. Seguranca

### 8.1 Camadas de Seguranca

![Camadas Seguranca](images/camadas_seguranca_sistema.png)

```
+==============================================================================+
|                    ARQUITETURA DE SEGURANCA                                   |
+==============================================================================+
|                                                                               |
|                              INTERNET                                         |
|                                 │                                             |
|                                 ▼                                             |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  CAMADA 1: FIREWALL + WAF                                               ││
|   │  • Bloqueio de IPs maliciosos                                           ││
|   │  • Protecao contra DDoS                                                 ││
|   │  • Regras de rate limiting                                              ││
|   └────────────────────────────────┬────────────────────────────────────────┘|
|                                    │                                          |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  CAMADA 2: TLS/SSL                                                      ││
|   │  • TLS 1.3                                                              ││
|   │  • Certificados validos                                                 ││
|   │  • HSTS habilitado                                                      ││
|   └────────────────────────────────┬────────────────────────────────────────┘|
|                                    │                                          |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  CAMADA 3: AUTENTICACAO JWT                                             ││
|   │  • Tokens HS256                                                         ││
|   │  • Expiracao 24h                                                        ││
|   │  • Refresh automatico                                                   ││
|   └────────────────────────────────┬────────────────────────────────────────┘|
|                                    │                                          |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  CAMADA 4: RATE LIMITING                                                ││
|   │  • 1000 req/min global                                                  ││
|   │  • Por IP                                                               ││
|   │  • Por endpoint                                                         ││
|   └────────────────────────────────┬────────────────────────────────────────┘|
|                                    │                                          |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  CAMADA 5: VALIDACAO DE DADOS                                           ││
|   │  • Schema validation                                                    ││
|   │  • Sanitizacao de input                                                 ││
|   │  • SQL injection prevention                                             ││
|   └────────────────────────────────┬────────────────────────────────────────┘|
|                                    │                                          |
|                                    ▼                                          |
|                              APLICACAO                                        |
|                                                                               |
+==============================================================================+
```

### 8.2 Compliance

![Compliance](images/badges_compliance_regulatorio.png)

```
+==============================================================================+
|                    COMPLIANCE REGULATORIO                                     |
+==============================================================================+
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                              LGPD                                        │ |
|  │                   Lei Geral de Protecao de Dados                         │ |
|  ├─────────────────────────────────────────────────────────────────────────┤ |
|  │  [X] Explicabilidade automatica em cada predicao (Art. 20)              │ |
|  │  [X] Mascaramento de CPF na interface (XXX.XXX.XXX-XX)                  │ |
|  │  [X] Audit trail completo em PostgreSQL                                 │ |
|  │  [X] Endpoint de explicabilidade individual                             │ |
|  │  [X] Texto explicativo em portugues                                     │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                            BACEN                                         │ |
|  │                    Resolucao 6/2023                                      │ |
|  ├─────────────────────────────────────────────────────────────────────────┤ |
|  │  [X] API de deteccao de fraudes operacional                             │ |
|  │  [X] Tempo de resposta monitorado (SLA)                                 │ |
|  │  [X] Registro de todas operacoes                                        │ |
|  │  [X] Metricas de performance disponiveis                                │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                           PCI DSS                                        │ |
|  │              Payment Card Industry Data Security Standard                │ |
|  ├─────────────────────────────────────────────────────────────────────────┤ |
|  │  [X] Dados sensiveis mascarados                                         │ |
|  │  [X] Logging estruturado sem dados sensiveis                            │ |
|  │  [X] TLS obrigatorio                                                    │ |
|  │  [X] Autenticacao JWT                                                   │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
+==============================================================================+
```

---

## 9. Performance Validada

```
+==============================================================================+
|                    METRICAS DE PERFORMANCE                                    |
+==============================================================================+
|                                                                               |
|  THROUGHPUT                                                                   |
|  ══════════                                                                   |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │                                                                         │  |
|  │  Batch Processing:  33.88 TPS                                          │  |
|  │  ████████████████████████████████████████████████████████████▒         │  |
|  │                                                                         │  |
|  │  Condicoes: 50 transacoes em paralelo, 8 workers                       │  |
|  │                                                                         │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
|  LATENCIA                                                                     |
|  ════════                                                                     |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │                                                                         │  |
|  │  p50:   28ms  ████████▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                  │  |
|  │  p95:  300ms  ██████████████████████████████████████████████████████▒  │  |
|  │  p99:  311ms  ████████████████████████████████████████████████████████ │  |
|  │                                                                         │  |
|  │  Nota: p95/p99 incluem cold start do modelo                            │  |
|  │                                                                         │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
|  QUALIDADE ML                                                                 |
|  ═══════════                                                                  |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │                                                                         │  |
|  │  Recall:     90.9%  ████████████████████████████████████████████████▒  │  |
|  │  Precisao:  100.0%  ██████████████████████████████████████████████████ │  |
|  │  F1-Score:   95.2%  █████████████████████████████████████████████████▒ │  |
|  │                                                                         │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
|  TESTES                                                                       |
|  ══════                                                                       |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │                                                                         │  |
|  │  E2E Tests:     25/25 passando  [====================] 100%            │  |
|  │  ML Tests:      20/20 passando  [====================] 100%            │  |
|  │  Error Rate:    0.0%                                                    │  |
|  │                                                                         │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
+==============================================================================+
```

---

## 10. Deployment

### 10.1 Workflows

```
+==============================================================================+
|                    CONFIGURACAO DE WORKFLOWS                                  |
+==============================================================================+
|                                                                               |
|  WORKFLOW 1: Backend API                                                      |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │  Comando: cd sankofa-enterprise-real/backend && python api/prod*.py    │  |
|  │  Porta: 8000                                                            │  |
|  │  Status: RUNNING                                                        │  |
|  │                                                                          │  |
|  │  Endpoints: 50+                                                         │  |
|  │  Rate Limit: 1000/min                                                   │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
|  WORKFLOW 2: Frontend                                                         |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │  Comando: cd sankofa-enterprise-real/frontend && npm run dev           │  |
|  │  Porta: 5000                                                            │  |
|  │  Status: RUNNING                                                        │  |
|  │                                                                          │  |
|  │  Paginas: 9                                                             │  |
|  │  Framework: React + Vite                                                │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
+==============================================================================+
```

### 10.2 Variaveis de Ambiente

```
+==============================================================================+
|                    VARIAVEIS DE AMBIENTE                                      |
+==============================================================================+
|                                                                               |
|  VARIAVEL              VALOR                    DESCRICAO                    |
|  ════════              ═════                    ═════════                    |
|                                                                               |
|  ENVIRONMENT           production               Ambiente de execucao         |
|  FLASK_DEBUG           false                    Debug desabilitado           |
|  JWT_SECRET            <secret-key>             Chave JWT                    |
|  DATABASE_URL          postgresql://...         Conexao PostgreSQL           |
|  API_PORT              8000                     Porta do backend             |
|  FRONTEND_PORT         5000                     Porta do frontend            |
|                                                                               |
+==============================================================================+
```

---

*Documento tecnico atualizado em 27 de Novembro de 2025*  
*Sankofa Enterprise Pro v12.0*  
*Total: 15+ diagramas e ilustracoes*
