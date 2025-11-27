# Documentacao Funcional - Sankofa Enterprise Pro v12.0
## Sistema de Deteccao de Fraudes para Instituicoes Financeiras

**Versao:** 12.0  
**Ultima Atualizacao:** 27 de Novembro de 2025  
**Status:** Producao - 25 Testes E2E Passando

---

## Estado do Sistema

| Componente | Status | Detalhes |
|------------|--------|----------|
| API Backend | ✅ Producao | 50+ endpoints Flask |
| Frontend Dashboard | ✅ Producao | 9 paginas React |
| ML Stacking Ensemble | ✅ Producao | RF + GB + LR |
| Explicabilidade SHAP | ✅ Integrado | API retorna explicacoes LGPD |
| Observabilidade | ✅ Producao | Prometheus, SLA, alertas |
| Infraestrutura Escala | ✅ Producao | Batch 33.88 TPS, async queue |
| PostgreSQL | ✅ Integrado | Transacoes persistidas |
| Testes Automatizados | ✅ 25 E2E | 100% passando |

---

## 1. Visao Geral do Sistema

### 1.1 O que e o Sankofa?

O **Sankofa Enterprise Pro** e um sistema de deteccao de fraudes financeiras em producao que analisa transacoes em tempo real usando Machine Learning. O nome "Sankofa" vem de um simbolo africano que significa "voltar e buscar" - representando a capacidade do sistema de aprender com padroes passados.

### 1.2 Para Quem e Este Sistema?

| Perfil | Uso Principal |
|--------|---------------|
| **Analistas de Fraude** | Investigar alertas, revisar transacoes suspeitas |
| **Gestores de Risco** | Monitorar KPIs, ajustar thresholds |
| **Equipe de Compliance** | Gerar relatorios, auditorias, LGPD |
| **Administradores de TI** | Configurar sistema, monitorar observabilidade |

### 1.3 Capacidades do Sistema

**Funcionalidades de Producao:**
```
+-------------------------------------------------------------+
|  API de Predicao em Tempo Real (/api/fraud/predict)         |
|  Explicabilidade LGPD com SHAP (explanation_text)           |
|  Dashboard 9 Paginas React                                  |
|  ML Stacking (RandomForest + GradientBoosting + LR)         |
|  PostgreSQL para transacoes (psycopg2)                      |
|  25 Testes E2E Passando                                     |
|  Observabilidade Prometheus (TPS, latencia, alertas)        |
|  Processamento Batch (33.88 TPS)                            |
|  Fila Assincrona com Prioridades                            |
|  Circuit Breaker para Resiliencia                           |
+-------------------------------------------------------------+
```

---

## 2. Novos Recursos v12.0

### 2.1 Explicabilidade LGPD (NOVO)

Cada predicao agora inclui explicacoes automaticas em texto para compliance LGPD:

**Endpoint:** `POST /api/fraud/predict`

**Parametros:**
- `include_explanation`: true/false (default: true)
- `include_compliance_report`: true/false (default: false)

**Resposta com Explicabilidade:**
```json
{
  "predictions": [{
    "transaction_id": "TXN_001",
    "is_fraud": true,
    "risk_score": 87.5,
    "risk_level": "HIGH",
    "explanation_text": "Transacao de alto valor (R$ 15.000) em horario noturno (03:00) com velocidade acima do padrao",
    "top_risk_factors": [
      {"feature": "amount_normalized", "impact": 0.45, "direction": "increases_risk"},
      {"feature": "is_night", "impact": 0.32, "direction": "increases_risk"}
    ],
    "top_protective_factors": [
      {"feature": "device_risk_score", "impact": -0.15, "direction": "decreases_risk"}
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

### 2.2 Observabilidade Prometheus (NOVO)

Sistema completo de metricas em tempo real:

| Endpoint | Descricao |
|----------|-----------|
| `/api/observability/metrics` | Metricas JSON completas |
| `/api/observability/prometheus` | Formato Prometheus |
| `/api/observability/sla` | Verificacao de SLA |
| `/api/health/detailed` | Health check detalhado |
| `/api/health/live` | Liveness probe |
| `/api/health/ready` | Readiness probe |

**Metricas Disponiveis:**
- `sankofa_requests_total` - Total de requisicoes
- `sankofa_predictions_total` - Total de predicoes
- `sankofa_predictions_fraud` - Predicoes de fraude
- `sankofa_latency_p50/p95/p99` - Percentis de latencia
- `sankofa_error_rate` - Taxa de erro
- `sankofa_tps` - Transacoes por segundo

### 2.3 Infraestrutura de Escala (NOVO)

| Endpoint | Descricao | Performance |
|----------|-----------|-------------|
| `/api/infrastructure/batch/process` | Batch paralelo | 33.88 TPS |
| `/api/infrastructure/task/submit` | Fila assincrona | Prioridades |
| `/api/infrastructure/queue/metrics` | Metricas fila | Circuit breaker |

**Componentes:**
- `AsyncTaskQueue`: Fila com prioridades (CRITICAL, HIGH, NORMAL, LOW)
- `BatchProcessor`: Processamento paralelo (8 workers)
- `CircuitBreaker`: Protecao contra falhas em cascata

---

## 3. Casos de Uso Principais

### 3.1 UC01: Analise de Transacao em Tempo Real

**Ator:** Sistema Bancario (Core Banking)  
**Objetivo:** Avaliar risco de fraude antes de aprovar transacao

**Fluxo Principal:**
```
   SISTEMA           SANKOFA                          RESPOSTA
   BANCARIO          API
      |                |
      |  POST /api/    |
      |  fraud/predict |
      +--------------->|
      |                |  1. Validar Payload
      |                |  2. Extrair Features
      |                |  3. Stacking (RF+GB)
      |                |  4. Meta-model (LR)
      |                |  5. Gerar Explicacao SHAP
      |                |  6. Salvar no BD
      |                |
      |  200 OK        |
      |<---------------+
      |  {is_fraud,    |
      |   score,       |
      |   explanation} |

   TEMPO TOTAL: ~30ms (aquecido)
```

**Exemplo de Requisicao:**
```json
{
  "transactions": [{
    "transaction_id": "TXN_001",
    "amount": 15000.00,
    "hour": 3,
    "day_of_week": 2,
    "location_risk_score": 0.3,
    "device_risk_score": 0.2,
    "velocity_score": 0.8,
    "is_new_device": 0
  }],
  "include_explanation": true,
  "include_compliance_report": true
}
```

### 3.2 UC02: Processamento em Batch

**Ator:** Sistema de Reconciliacao  
**Objetivo:** Processar grande volume de transacoes

**Endpoint:** `POST /api/infrastructure/batch/process`

```json
{
  "transactions": [/* lista de 50+ transacoes */],
  "batch_size": 100,
  "include_explanation": false
}
```

**Resposta:**
```json
{
  "success": true,
  "data": {
    "total": 50,
    "successful": 50,
    "failed": 0,
    "processing_time_ms": 1475.7,
    "throughput_tps": 33.88,
    "results": [/* predicoes */]
  }
}
```

### 3.3 UC03: Monitoramento de SLA

**Ator:** Equipe de Operacoes  
**Objetivo:** Verificar compliance de SLA

**Endpoint:** `GET /api/observability/sla`

```json
{
  "success": true,
  "data": {
    "latency_p95": {
      "current": 28.5,
      "threshold": 100.0,
      "compliant": true
    },
    "error_rate": {
      "current": 0.0,
      "threshold": 0.1,
      "compliant": true
    },
    "tps": {
      "current": 33.88,
      "threshold": 100.0,
      "compliant": true
    }
  }
}
```

---

## 4. Modulos do Sistema

### 4.1 Dashboard Executivo (`/`)

**Funcionalidades:**
- Cards de KPIs: Transacoes, Fraudes, Taxa de Aprovacao, Latencia
- Graficos de transacoes por hora
- Alertas recentes
- Status dos modelos ML

### 4.2 Central de Transacoes (`/transactions`)

**Funcionalidades:**
- Tabela com transacoes recentes (via API /api/transactions)
- Colunas: ID, Valor, Tipo, Canal, Localizacao, CPF mascarado, Data/Hora
- Filtros por status e tipo

### 4.3 Revisao Manual (`/manual-review`)

**Funcionalidades:**
- Lista de transacoes pendentes de revisao
- Contadores (total, pendentes, completadas)
- Botoes de acao (aprovar/rejeitar)
- SLA timers

### 4.4 Monitoramento (`/monitoring`)

**Funcionalidades (ATUALIZADO):**
- Status em tempo real via `/api/observability/metrics`
- Metricas de TPS, latencia, error rate
- Health checks por componente
- Alertas automaticos de SLA

### 4.5 Calibragem (`/calibration`)

**Funcionalidades:**
- Ajuste de thresholds por tier
- Controle de pesos do ensemble
- Configuracao de regras

---

## 5. Regras de Negocio

### 5.1 Classificacao de Risco

| Score | Classificacao | Acao | Cor |
|-------|---------------|------|-----|
| 0-30 | **Baixo Risco** | Aprovacao automatica | Verde |
| 31-50 | **Risco Moderado** | Aprovacao com monitoramento | Amarelo |
| 51-70 | **Alto Risco** | Encaminha para revisao manual | Laranja |
| 71-100 | **Risco Critico** | Bloqueio automatico | Vermelho |

### 5.2 Features de ML (47+)

**Temporais (5):**
- hour, day_of_week, is_weekend, is_night, is_business_hours

**Valor (5):**
- amount_log, amount_squared, amount_normalized, is_round_amount, amount_zscore

**Geograficas (3):**
- distance_from_home, location_risk_score, is_international

**Comportamentais (4):**
- transaction_velocity_1h, transaction_velocity_24h, amount_deviation, new_merchant

**Location Entropy (11):**
- location_entropy, unique_locations, location_diversity_score, etc.

---

## 6. APIs e Integracao

### 6.1 Endpoints Principais

| Endpoint | Metodo | Funcao | Rate Limit |
|----------|--------|--------|------------|
| `/api/health` | GET | Health check | - |
| `/api/fraud/predict` | POST | Predicao com explicacao | 1000/min |
| `/api/fraud/batch` | POST | Batch tradicional | 100/min |
| `/api/infrastructure/batch/process` | POST | Batch otimizado | 50/min |
| `/api/observability/metrics` | GET | Metricas Prometheus | 500/min |
| `/api/observability/sla` | GET | Status SLA | 500/min |
| `/api/health/detailed` | GET | Health detalhado | - |

### 6.2 Endpoints de Explicabilidade

| Endpoint | Metodo | Funcao |
|----------|--------|--------|
| `/api/explainability/features` | GET | Lista features e importancia |
| `/api/explainability/explain` | POST | Explicacao individual |

### 6.3 Endpoints de Infraestrutura

| Endpoint | Metodo | Funcao |
|----------|--------|--------|
| `/api/infrastructure/queue/metrics` | GET | Metricas da fila |
| `/api/infrastructure/task/submit` | POST | Submete tarefa async |
| `/api/infrastructure/task/<id>/status` | GET | Status da tarefa |

---

## 7. Compliance e Regulamentacao

### 7.1 LGPD

| Requisito | Status | Implementacao |
|-----------|--------|---------------|
| Dados pessoais mascarados | ✅ Implementado | CPF exibido como XXX.XXX.XXX-XX |
| Logs de auditoria | ✅ Implementado | Tabela audit_log PostgreSQL |
| Explicabilidade (Art. 20) | ✅ Implementado | SHAP + explanation_text |
| Direito a explicacao | ✅ Implementado | Endpoint /api/explainability/explain |

### 7.2 BACEN Resolucao 6/2023

| Requisito | Status | Implementacao |
|-----------|--------|---------------|
| API de deteccao | ✅ Implementado | /api/fraud/predict |
| Tempo de resposta | ✅ Monitorado | /api/observability/sla |
| Disponibilidade | ✅ Monitorado | Health checks |
| Audit trail | ✅ Implementado | PostgreSQL + logs |

### 7.3 PCI DSS

| Requisito | Status | Implementacao |
|-----------|--------|---------------|
| Dados sensiveis | ✅ Mascarados | CPF, cartao |
| Logs seguros | ✅ Implementado | Structured logging |
| Monitoramento | ✅ Implementado | Observabilidade |

---

## 8. Metricas de Performance

### 8.1 Metricas Validadas

| Metrica | Valor | Condicao |
|---------|-------|----------|
| Throughput Batch | 33.88 TPS | 50 transacoes paralelas |
| Latencia p50 | 28ms | Modelo aquecido |
| Latencia p95 | 300ms | Inclui cold start |
| Latencia p99 | 311ms | Inclui cold start |
| Error Rate | 0% | Testes E2E |

### 8.2 ML Performance

| Metrica | Valor |
|---------|-------|
| Recall | 90.9% |
| Precisao | 100% |
| F1-Score | 95.2% |

---

## 9. Glossario

| Termo | Definicao |
|-------|-----------|
| **Ensemble** | Combinacao de multiplos modelos de ML |
| **Feature** | Caracteristica extraida da transacao |
| **Threshold** | Limite de corte para decisao |
| **SHAP** | SHapley Additive exPlanations - explicabilidade |
| **TPS** | Transacoes por segundo |
| **HITL** | Human-in-the-Loop - revisao humana |
| **Circuit Breaker** | Padrao de resiliencia contra falhas |
| **SLA** | Service Level Agreement |

---

*Documento gerado automaticamente pelo Sankofa Enterprise Pro v12.0*  
*Ultima atualizacao: 27 de Novembro de 2025*
