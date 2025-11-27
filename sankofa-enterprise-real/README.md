# Sankofa Enterprise Pro v12.0

## Sistema de Deteccao de Fraudes para Instituicoes Financeiras

**Versao:** 12.0  
**Status:** Producao - 25 Testes E2E Passando  
**Ultima Atualizacao:** 27 de Novembro de 2025

---

## Visao Geral

O Sankofa Enterprise Pro e um sistema de deteccao de fraudes em tempo real que combina Machine Learning, explicabilidade LGPD e observabilidade enterprise-grade.

---

## Novos Recursos v12.0

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

---

## Quick Start

### Backend (API)
```bash
cd backend
python api/production_api.py
```
API disponivel em: http://localhost:8000

### Frontend (Dashboard)
```bash
cd frontend
npm run dev
```
Dashboard disponivel em: http://localhost:5000

---

## Endpoints Principais

| Endpoint | Metodo | Descricao |
|----------|--------|-----------|
| `/api/health` | GET | Health check |
| `/api/fraud/predict` | POST | Predicao com explicacao LGPD |
| `/api/fraud/batch` | POST | Batch tradicional |
| `/api/infrastructure/batch/process` | POST | Batch otimizado (33.88 TPS) |
| `/api/observability/metrics` | GET | Metricas Prometheus |
| `/api/observability/sla` | GET | Status SLA |
| `/api/explainability/features` | GET | Importancia das features |

---

## Exemplo de Predicao com Explicacao

```bash
curl -X POST http://localhost:8000/api/fraud/predict \
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
    "lgpd_compliant": true,
    "compliance_report": {
      "lgpd": "Explicacao fornecida conforme Art. 20 LGPD",
      "bacen": "Tempo de resposta dentro do SLA",
      "pci_dss": "Dados sensiveis mascarados"
    }
  }]
}
```

---

## Metricas de Performance

| Metrica | Valor |
|---------|-------|
| Throughput Batch | 33.88 TPS |
| Latencia p50 | 28ms |
| Latencia p95 | 300ms |
| Latencia p99 | 311ms |
| Recall ML | 90.9% |
| Precisao ML | 100% |
| Testes E2E | 25/25 passando |

---

## Documentacao

Documentacao completa em `docs/`:

| Documento | Descricao |
|-----------|-----------|
| [README.md](docs/README.md) | Indice da documentacao |
| [DOCUMENTACAO_FUNCIONAL.md](docs/DOCUMENTACAO_FUNCIONAL.md) | Casos de uso e regras |
| [ARQUITETURA_TECNICA.md](docs/ARQUITETURA_TECNICA.md) | Arquitetura tecnica |
| [MANUAL_USUARIO.md](docs/MANUAL_USUARIO.md) | Manual do usuario |
| [RELATORIO_QA.md](docs/RELATORIO_QA.md) | Relatorio de testes |
| [DIAGRAMAS.md](docs/DIAGRAMAS.md) | Diagramas e fluxogramas |

---

## Estrutura do Projeto

```
sankofa-enterprise-real/
+-- backend/
|   +-- api/production_api.py           # API principal (50+ endpoints)
|   +-- ml_engine/
|   |   +-- production_fraud_engine.py  # Motor ML
|   |   +-- explainability_engine.py    # SHAP + LGPD
|   +-- monitoring/observability.py     # Prometheus + SLA
|   +-- infrastructure/async_processor.py # Batch + Queue
|   +-- tests/                          # 25 testes E2E
+-- frontend/
|   +-- src/pages/                      # 9 paginas React
+-- docs/                               # Documentacao completa
```

---

## Compliance

| Regulamentacao | Status | Implementacao |
|----------------|--------|---------------|
| LGPD | ✅ | Explicabilidade automatica (Art. 20) |
| BACEN | ✅ | SLA monitorado em tempo real |
| PCI DSS | ✅ | Dados sensiveis mascarados |

---

## Tecnologias

**Backend:**
- Python 3.11+
- Flask + Flask-CORS
- scikit-learn, XGBoost, LightGBM
- PostgreSQL (Neon)

**Frontend:**
- React + Vite
- TailwindCSS + shadcn/ui
- 9 paginas de dashboard

**Observabilidade:**
- Metricas Prometheus-style
- SLA checks automaticos
- Alert manager

---

## Variaveis de Ambiente

| Variavel | Descricao |
|----------|-----------|
| `DATABASE_URL` | Conexao PostgreSQL |
| `JWT_SECRET` | Chave JWT |
| `ENVIRONMENT` | development/production |

---

## Testes

```bash
cd backend
python -m pytest tests/test_e2e.py -v
```

Resultado: 25 testes passando (100%)

---

## Contato e Suporte

Para duvidas ou problemas, entre em contato com a equipe de suporte.

---

**Sankofa Enterprise Pro v12.0** - Protegendo instituicoes financeiras com inteligencia artificial.

*Ultima atualizacao: 27 de Novembro de 2025*
