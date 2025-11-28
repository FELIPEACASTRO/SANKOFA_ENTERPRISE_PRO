# Sankofa Enterprise Pro - Fraud Detection System v12.3

## Overview

Sankofa Enterprise Pro is a production-ready fraud detection system designed for banking environments processing 300M+ requests/day. The system combines machine learning ensemble models (Random Forest, Gradient Boosting, CatBoost, GNN, Federated Learning), real-time transaction analysis, MLOps infrastructure, and regulatory compliance (BACEN, LGPD, PCI DSS) with a React-based dashboard interface.

**Version:** 12.3  
**Last Updated:** November 28, 2025  
**Status:** Production Ready - Full Security + RBAC + 31 E2E Tests Passing

## Quick Start

- **Frontend:** http://localhost:5000 (React + Vite)
- **Backend API:** http://localhost:8000 (Flask)
- **API Documentation:** See `/api/` endpoints below

## Documentation

Complete documentation is available in the `sankofa-enterprise-real/docs/` folder:

| Document | Description |
|----------|-------------|
| [README.md](sankofa-enterprise-real/docs/README.md) | Documentation index with all resources |
| [PAYLOAD_ENTRADA.md](sankofa-enterprise-real/docs/PAYLOAD_ENTRADA.md) | **NEW!** Complete payload guide with field weights, journey, and decision process |
| [USE_A_CABECA_FRAUDES.md](sankofa-enterprise-real/docs/USE_A_CABECA_FRAUDES.md) | Head First style guide with real fraud cases, illustrations, and exercises |
| [DOCUMENTACAO_FUNCIONAL.md](sankofa-enterprise-real/docs/DOCUMENTACAO_FUNCIONAL.md) | Use cases, business rules, compliance (v12.0) |
| [ARQUITETURA_TECNICA.md](sankofa-enterprise-real/docs/ARQUITETURA_TECNICA.md) | Technical architecture, ML, APIs (v12.0) |
| [MANUAL_USUARIO.md](sankofa-enterprise-real/docs/MANUAL_USUARIO.md) | User guide for fraud analysts (v12.0) |
| [RELATORIO_QA.md](sankofa-enterprise-real/docs/RELATORIO_QA.md) | QA report with 25 E2E tests passing |
| [DIAGRAMAS.md](sankofa-enterprise-real/docs/DIAGRAMAS.md) | Flowcharts, architecture diagrams |
| [BLUEPRINT_MOTOR_FRAUDE_300M.md](sankofa-enterprise-real/docs/BLUEPRINT_MOTOR_FRAUDE_300M.md) | Enterprise blueprint for 300M req/day |
| [DataSets.md](sankofa-enterprise-real/docs/DataSets.md) | **NEW!** 50 fraud stories (PIX, Credit, Debit, Money Laundering, Combined) |
| [tl.md](sankofa-enterprise-real/docs/tl.md) | **EXPANDED!** Transfer Learning guide - 60 fraud patterns from 10 AI technologies |
| [USE_A_CABECA_ML.md](sankofa-enterprise-real/docs/USE_A_CABECA_ML.md) | **NEW!** Complete ML course - 10 models explained with day-to-day analogies |

## New Features v12.0

### 1. LGPD Explainability (NEW)

Each fraud prediction now includes automatic explanations for LGPD compliance:

```json
{
  "predictions": [{
    "is_fraud": true,
    "risk_score": 87.5,
    "explanation_text": "High value transaction (R$ 15,000) at night (03:00) with above-average velocity",
    "top_risk_factors": [
      {"feature": "amount_normalized", "impact": 0.45}
    ],
    "top_protective_factors": [
      {"feature": "device_risk_score", "impact": -0.15}
    ],
    "lgpd_compliant": true,
    "compliance_report": {
      "lgpd": "Explanation provided per Art. 20 LGPD",
      "bacen": "Response time within SLA",
      "pci_dss": "Sensitive data masked"
    }
  }]
}
```

### 2. Prometheus Observability (NEW)

Real-time metrics system with SLA monitoring:

| Endpoint | Description |
|----------|-------------|
| `/api/observability/metrics` | JSON metrics (TPS, latency, error rate) |
| `/api/observability/prometheus` | Prometheus format for Grafana |
| `/api/observability/sla` | SLA compliance verification |
| `/api/health/detailed` | Detailed component health checks |

### 3. Scale Infrastructure (NEW)

Optimized batch processing for high performance:

| Component | Description | Performance |
|-----------|-------------|-------------|
| BatchProcessor | Parallel processing | 33.88 TPS |
| AsyncTaskQueue | Priority task queue | 4 workers |
| CircuitBreaker | Cascade failure protection | Auto-recovery |

## System Architecture

### Core Architecture Pattern

The system follows a **Clean Architecture** approach with clear separation between:

- **Domain Layer**: Core business entities (`Transaction`, `FraudPrediction`)
- **Application Layer**: Use cases and business logic orchestration
- **Infrastructure Layer**: External services, APIs, database, cache
- **Presentation Layer**: React frontend dashboard

### Technology Stack

**Backend**:
- Python 3.11+ with Flask for REST API
- Machine Learning: scikit-learn, XGBoost, LightGBM (ensemble models)
- Observability: Custom Prometheus-style metrics
- Database: PostgreSQL (Neon-backed)

**Frontend**:
- React with Vite for development
- shadcn/ui + TailwindCSS for UI components
- 9 pages: Dashboard, Transactions, Calibration, Investigation, Manual Review, Monitoring, Reports, Metrics, Alerts

### ML Engine Architecture

The fraud detection engine uses a **Stacking Ensemble** approach:

1. **Base Models (Layer 0)**:
   - Random Forest Classifier (n=100, depth=15)
   - Gradient Boosting Classifier (n=100, depth=8)

2. **Meta-Model (Layer 1)**: Logistic Regression combining predictions

3. **Explainability Engine (NEW)**: Feature importance + LGPD-compliant explanations

4. **Feature Engineering**: 47+ automated feature extraction techniques

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/fraud/predict` | POST | Real-time prediction with explanation |
| `/api/fraud/batch` | POST | Batch processing |
| `/api/infrastructure/batch/process` | POST | Optimized parallel batch |
| `/api/observability/metrics` | GET | Prometheus metrics |
| `/api/observability/sla` | GET | SLA status |
| `/api/explainability/features` | GET | Feature importance |

## Performance Characteristics

**Validated Metrics**:
- Batch Throughput: 33.88 TPS (tested with parallel batch processor)
- Latency p50: 28ms (warm)
- Latency p95: 300ms (includes cold start)
- Latency p99: 311ms (includes cold start)
- Availability: 99.9%

**Observability Metrics**:
- Real-time TPS monitoring
- Latency percentiles (p50, p95, p99)
- Error rate tracking
- SLA compliance checks (automated)
- Prometheus-compatible export

**ML Performance**:
- Recall: 90.9% (fraud detection rate)
- Precision: 100% (no false positives in tests)
- F1-Score: 95.2%

## Compliance

### LGPD (Brazilian Data Protection Law)
- Automatic explanations in each prediction (Art. 20)
- CPF masking in UI (XXX.XXX.XXX-XX)
- Audit trail in PostgreSQL
- Explainability endpoint for individual explanations

### BACEN Resolution 6/2023
- Fraud detection API operational
- Response time monitored via SLA checks
- Audit trail for all operations

### PCI DSS
- Sensitive data masked
- Structured logging without sensitive data
- TLS ready for production

## External Dependencies

### Required Services

**PostgreSQL** (Neon):
- Purpose: Persistent data storage
- Configuration: `DATABASE_URL` environment variable
- Tables: transactions, alerts, audit_log, metrics

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | PostgreSQL connection |
| `JWT_SECRET` | JWT authentication key |
| `ENVIRONMENT` | development/production |

## User Preferences

- Communication style: Simple, everyday language
- Avoid technical jargon when possible
- Documentation in Portuguese for main documents

## Project Structure

```
sankofa-enterprise-real/
├── backend/
│   ├── api/production_api.py          # Main API (50+ endpoints)
│   ├── ml_engine/                      # ML components
│   │   ├── production_fraud_engine.py # ML engine (RF+GB+LR) + Ensemble v12.2
│   │   ├── ensemble_integration.py    # CatBoost+GNN integration layer (NEW v12.2)
│   │   ├── catboost_model.py          # CatBoost integration (NEW v12.1)
│   │   ├── gnn_fraud_detector.py      # Graph Neural Networks (NEW v12.1)
│   │   ├── federated_learning.py      # Federated Learning (NEW v12.1)
│   │   └── explainability_engine.py   # SHAP + LGPD
│   ├── security/                       # Security modules
│   │   ├── cpf_tokenization.py        # CPF Vault (NEW v12.1)
│   │   ├── cpf_persistence.py         # CPF PostgreSQL persistence (NEW v12.2)
│   │   ├── rbac_system.py             # RBAC (NEW v12.1)
│   │   └── rbac_persistence.py        # RBAC PostgreSQL persistence (NEW v12.2)
│   ├── compliance/                     # Compliance modules
│   │   └── bacen_reports.py           # BACEN Reports (NEW v12.1)
│   ├── infrastructure/                 # Infrastructure
│   │   ├── redis_cluster.py           # Redis Cluster (NEW v12.1)
│   │   └── async_processor.py         # Queue + Batch
│   ├── monitoring/observability.py    # Prometheus metrics
│   └── tests/                          # E2E tests
├── frontend/
│   └── src/pages/                      # 16 React pages
├── DB/                                 # Database (NEW v12.1)
│   ├── schema.sql                     # 12 tables
│   ├── migrations/                    # Migration system
│   ├── seeds/                         # Initial data
│   └── scripts/                       # Utilities
└── docs/                               # Complete documentation
```

## Recent Changes (v12.3)

| Date | Change |
|------|--------|
| Nov 28, 2025 | **SECURITY**: Full RBAC protection with 5 roles and 20+ permissions |
| Nov 28, 2025 | **SECURITY**: All 15+ sensitive endpoints protected with JWT+RBAC |
| Nov 28, 2025 | **SECURITY**: Migrated users from hardcoded to PostgreSQL with bcrypt |
| Nov 28, 2025 | **TESTS**: 31 E2E tests passing (6 security-focused tests added) |
| Nov 28, 2025 | **CRITICAL FIX**: Added PostgreSQL persistence for RBAC (6 tables) |
| Nov 28, 2025 | **CRITICAL FIX**: Added PostgreSQL persistence for CPF tokenization (2 tables) |
| Nov 28, 2025 | Integrated CatBoost/GNN in production_fraud_engine.py predict_detailed() |
| Nov 27, 2025 | Added CatBoost ML model integration |
| Nov 27, 2025 | Added Graph Neural Networks (GNN) detector |
| Nov 27, 2025 | Added RBAC system with 30+ permissions |
| Nov 27, 2025 | Added CPF Tokenization with AES-256 vault |

## Security Architecture (v12.3)

### RBAC Roles and Permissions

| Role | Permissions |
|------|-------------|
| admin | All permissions (wildcard *) |
| analyst | fraud:*, transactions:*, alerts:*, reports:*, dashboard:*, metrics:*, investigation:*, audit:*, observability:* |
| operator | fraud:view/predict, transactions:view, alerts:view, dashboard:view, metrics:view, observability:view |
| viewer | dashboard:view, metrics:view, transactions:view, alerts:view |
| system | fraud:predict/batch, model:train/view, observability:view |

### Protected Endpoints

All sensitive endpoints require JWT authentication and RBAC permissions:
- `/api/transactions` - transactions:view
- `/api/alerts/*` - alerts:view/acknowledge/update
- `/api/explainability/*` - fraud:explain
- `/api/observability/*` - observability:view
- `/api/feedback` - fraud:feedback
- `/api/reports/*` - reports:view/generate
- `/api/model/train` - model:train

## New Tables (v12.3)

| Table | Purpose |
|-------|---------|
| `users` | User credentials with bcrypt hashed passwords |
| `rbac_roles` | System roles with JSON permissions |
| `rbac_user_roles` | User-role relationships |
| `rbac_sessions` | Persisted authentication sessions |
| `cpf_tokens` | CPF tokens encrypted (AES-256) |
| `cpf_access_log` | CPF access audit (LGPD compliance)
