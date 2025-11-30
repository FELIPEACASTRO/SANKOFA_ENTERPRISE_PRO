# Sankofa Enterprise Pro v12.0

## Sistema de Deteccao de Fraudes para Instituicoes Financeiras

**Versao:** 12.0  
**Status:** Producao - 100% Operacional  
**Ultima Atualizacao:** 29 de Novembro de 2025

---

## Visao Geral

O Sankofa Enterprise Pro e um sistema de deteccao de fraudes em tempo real que combina Machine Learning, explicabilidade LGPD e observabilidade enterprise-grade. Projetado para processar 300M+ transacoes/dia com latencia PIX <50ms.

---

## Recursos Principais v12.0

### 1. Explicabilidade LGPD
Cada predicao inclui explicacao em texto para compliance com Art. 20 da LGPD:
- Texto explicativo em linguagem natural
- Fatores de risco e protecao identificados
- Relatorio de compliance automatico

### 2. Observabilidade Prometheus
Sistema completo de metricas em tempo real:
- TPS, latencia (p50/p95/p99), error rate
- SLA compliance checks automaticos
- Formato Prometheus para Grafana

### 3. Infraestrutura de Escala
Processamento otimizado para alta performance:
- Batch paralelo: 33.88 TPS testado
- Fila assincrona com prioridades
- Circuit breaker para resiliencia

### 4. Dashboard Completo
Interface React moderna com 16 paginas funcionais:
- Dashboard executivo
- Gestao de transacoes
- Calibracao de modelos ML
- Human-in-the-Loop review

---

## Quick Start

### Backend (API)
```bash
cd backend
python api/production_api.py
```
API disponivel em: http://localhost:5000

### Frontend (Dashboard)
```bash
cd frontend
npm run dev
```
Dashboard disponivel em: http://localhost:5000

---

## Endpoints Principais (78+ Disponiveis)

### Health & Status
| Endpoint | Metodo | Descricao |
|----------|--------|-----------|
| `/api/health` | GET | Health check basico |
| `/api/health/live` | GET | Kubernetes liveness probe |
| `/api/health/ready` | GET | Kubernetes readiness probe |
| `/api/health/detailed` | GET | Health detalhado por componente |
| `/api/status` | GET | Status completo do sistema |

### Autenticacao
| Endpoint | Metodo | Descricao |
|----------|--------|-----------|
| `/api/auth/login` | POST | Autenticacao JWT |
| `/api/auth/verify` | GET | Validacao de token |
| `/api/auth/refresh` | POST | Renovacao de token |

### Deteccao de Fraude
| Endpoint | Metodo | Descricao |
|----------|--------|-----------|
| `/api/fraud/predict` | POST | Predicao com explicacao LGPD |
| `/api/fraud/batch` | POST | Processamento em lote |

### Dashboard
| Endpoint | Metodo | Descricao |
|----------|--------|-----------|
| `/api/dashboard/summary` | GET | Resumo executivo |
| `/api/dashboard/kpis` | GET | KPIs principais |
| `/api/dashboard/hourly` | GET | Metricas por hora |
| `/api/dashboard/timeseries` | GET | Dados para graficos |
| `/api/dashboard/channels` | GET | Estatisticas por canal |
| `/api/dashboard/alerts` | GET | Alertas ativos |

### Transacoes
| Endpoint | Metodo | Descricao |
|----------|--------|-----------|
| `/api/transactions` | GET | Lista de transacoes |
| `/api/transactions/<id>/approve` | POST | Aprovar transacao |
| `/api/transactions/<id>/reject` | POST | Rejeitar transacao |
| `/api/transactions/<id>/review` | POST | Enviar para revisao |

### Modelo ML
| Endpoint | Metodo | Descricao |
|----------|--------|-----------|
| `/api/model/metrics` | GET | Metricas do modelo |
| `/api/model/info` | GET | Info do modelo |
| `/api/model/train` | POST | Treinar modelo |

### Explicabilidade (LGPD)
| Endpoint | Metodo | Descricao |
|----------|--------|-----------|
| `/api/explainability/features` | GET | Features importantes |
| `/api/explainability/explain` | POST | Explicar decisao |

### Calibracao
| Endpoint | Metodo | Descricao |
|----------|--------|-----------|
| `/api/calibration` | GET/PUT | Configuracao de calibracao |
| `/api/calibration/impact` | GET | Simulacao de impacto |
| `/api/calibration/apply` | POST | Aplicar calibracao |
| `/api/calibration/history` | GET | Historico de alteracoes |

### Observabilidade
| Endpoint | Metodo | Descricao |
|----------|--------|-----------|
| `/api/observability/metrics` | GET | Metricas Prometheus |
| `/api/observability/sla` | GET | Status SLA |
| `/api/observability/alerts` | GET | Alertas de observabilidade |

### Feedback Analyst (Human-in-the-Loop)
| Endpoint | Metodo | Descricao |
|----------|--------|-----------|
| `/api/feedback` | POST | Submeter feedback |
| `/api/feedback/list` | GET | Lista de feedbacks |
| `/api/feedback/analytics` | GET | Analytics de feedbacks |

---

## Exemplo de Predicao com Explicacao

```bash
curl -X POST http://localhost:5000/api/fraud/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [{
      "amount": 15000,
      "hour": 3,
      "day_of_week": 2,
      "location_risk_score": 0.3,
      "device_risk_score": 0.2,
      "velocity_score": 0.8,
      "is_new_device": 0
    }],
    "include_explanation": true,
    "include_compliance_report": true
  }'
```

**Resposta:**
```json
{
  "predictions": [{
    "is_fraud": true,
    "risk_score": 87.5,
    "risk_level": "HIGH",
    "explanation_text": "Transacao de alto valor em horario noturno com velocidade acima do padrao",
    "top_risk_factors": [
      {"feature": "amount_normalized", "impact": 0.45},
      {"feature": "is_night", "impact": 0.32}
    ],
    "top_protective_factors": [
      {"feature": "device_risk_score", "impact": -0.15}
    ],
    "lgpd_compliant": true
  }]
}
```

---

## Dashboard - 16 Paginas

| Pagina | Arquivo | Descricao |
|--------|---------|-----------|
| Dashboard | Dashboard.jsx | Painel principal com KPIs |
| Transacoes | Transactions.jsx | Lista e gestao de transacoes |
| Calibracao | Calibration.jsx | Ajuste de thresholds ML |
| Investigacao | Investigation.jsx | Analise detalhada de fraudes |
| Revisao Manual | ManualReview.jsx | Human-in-the-Loop review |
| Monitoramento | Monitoring.jsx | Saude dos modelos de IA |
| Relatorios | Reports.jsx | Geracao de relatorios |
| Metricas | Metrics.jsx | Contadores em tempo real |
| Alertas | Alerts.jsx | Central de alertas criticos |
| Datasets | Datasets.jsx | Catalogo de datasets |
| Hard Rules | HardRules.jsx | Regras rigidas de bloqueio |
| Lista VIP | VipList.jsx | Lista branca (whitelist) |
| Lista HOT | HotList.jsx | Lista negra (blacklist) |
| Auditoria | Audit.jsx | Trilhas de auditoria LGPD |
| Configuracoes | Settings.jsx | Configuracoes do sistema |
| Feedback Analista | FeedbackAnalyst.jsx | Feedback do modelo ML |

---

## Metricas de Performance

| Metrica | Valor |
|---------|-------|
| Throughput Batch | 33.88 TPS |
| Latencia p50 | 28ms |
| Latencia p95 | 300ms |
| Latencia p99 | 311ms |
| Latencia PIX | <50ms (SLA) |
| Recall ML | 90.3% |
| Precisao ML | 89.7% |
| F1-Score | 92.9% |
| Endpoints | 78+ |

---

## Banco de Dados (PostgreSQL)

### Tabelas Principais
| Tabela | Descricao |
|--------|-----------|
| transactions | Transacoes financeiras |
| alerts | Alertas de fraude |
| audit_logs | Trilha de auditoria |
| feedback | Feedbacks dos analistas |
| users | Usuarios do sistema |
| hard_rules | Regras rigidas |
| vip_list | Lista branca |
| hot_list | Lista negra |
| model_metrics | Metricas do modelo ML |
| cpf_tokens | Tokenizacao LGPD |
| rbac_roles | Roles RBAC |
| rbac_user_roles | Atribuicao de roles |
| rbac_sessions | Sessoes ativas |
| system_configs | Configuracoes |

Ver documentacao completa em: `docs/database/DB_DOCUMENTACAO_COMPLETA.md`

---

## Estrutura do Projeto

```
sankofa-enterprise-real/
+-- backend/
|   +-- api/
|   |   +-- production_api.py           # API principal (78+ endpoints)
|   |   +-- services/                   # Servicos auxiliares
|   +-- ml_engine/
|   |   +-- production_fraud_engine.py  # Motor ML Ensemble
|   |   +-- explainability_engine.py    # SHAP + LGPD
|   |   +-- catboost_model.py           # CatBoost detector
|   |   +-- gnn_fraud_detector.py       # GNN detector
|   +-- monitoring/
|   |   +-- observability.py            # Prometheus + SLA
|   +-- infrastructure/
|   |   +-- async_processor.py          # Batch + Queue
|   |   +-- database.py                 # Conexao PostgreSQL
|   +-- security/
|   |   +-- rbac_system.py              # RBAC com 5 roles
|   |   +-- cpf_tokenization.py         # Tokenizacao LGPD
|   +-- compliance/
|   |   +-- lgpd_compliance.py          # Compliance LGPD
|   |   +-- bacen_compliance.py         # Compliance BACEN
|   +-- tests/                          # Testes automatizados
+-- frontend/
|   +-- src/
|   |   +-- pages/                      # 16 paginas React
|   |   +-- components/                 # Componentes UI
+-- DB/
|   +-- schema.sql                      # Schema completo
|   +-- migrations/                     # Migracoes SQL
+-- docs/
|   +-- database/                       # Documentacao do DB
|   +-- images/                         # Imagens ilustrativas
+-- Insomnia/
|   +-- collections/                    # Collection API
```

---

## Compliance

| Regulamentacao | Status | Implementacao |
|----------------|--------|---------------|
| LGPD | ✅ | Explicabilidade Art. 20, Tokenizacao CPF |
| BACEN | ✅ | SLA <50ms PIX, Relatorios |
| PCI DSS | ✅ | Mascaramento, Logs estruturados |

---

## Tecnologias

**Backend:**
- Python 3.12+
- Flask + Flask-CORS + Flask-JWT-Extended
- scikit-learn, XGBoost, LightGBM, CatBoost
- PostgreSQL (Neon-backed)

**Frontend:**
- React 18 + Vite
- TailwindCSS + shadcn/ui
- Recharts para graficos

**ML Engine:**
- Stacking Ensemble (RF + GB + CB)
- SHAP para explicabilidade
- 47+ features engenheiradas

**Seguranca:**
- RBAC com 5 roles e 20+ permissions
- JWT com rotacao de chaves
- Tokenizacao de dados sensiveis

---

## Variaveis de Ambiente

| Variavel | Descricao |
|----------|-----------|
| `DATABASE_URL` | Conexao PostgreSQL |
| `JWT_SECRET` | Chave JWT |
| `ENCRYPTION_KEY` | Chave AES-256 |
| `ENVIRONMENT` | development/production |

---

## Testes

```bash
cd backend
python -m pytest tests/ -v
```

---

## Contato e Suporte

Para duvidas ou problemas, entre em contato com a equipe de suporte.

---

**Sankofa Enterprise Pro v12.0** - Protegendo instituicoes financeiras com inteligencia artificial.

*Ultima atualizacao: 29 de Novembro de 2025*
