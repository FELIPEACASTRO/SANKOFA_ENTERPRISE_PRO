# 04 - Contratos de API Frontend-Backend

**Data da Analise:** 29/11/2025  
**Metodologia:** MODO DOUBLE CHECK ULTRA - Fase 3

---

## 1. Dashboard Endpoints

### 1.1 GET /api/dashboard/kpis

| Campo | Valor |
|-------|-------|
| **Quem chama** | Dashboard.jsx |
| **Metodo** | GET |
| **Payload** | - |
| **Response esperado** | `{ total_transactions, fraud_count, fraud_rate, false_positive_rate, avg_risk_score, model_accuracy }` |
| **Tratamento sucesso** | Atualiza estado `kpis` |
| **Tratamento erro** | Fallback para valores zerados |
| **Latencia observada** | 3.9ms ✅ |

### 1.2 GET /api/dashboard/timeseries

| Campo | Valor |
|-------|-------|
| **Quem chama** | Dashboard.jsx |
| **Response** | `{ data: [{ date, transactions, frauds, ... }] }` |
| **Latencia** | < 5ms ✅ |

### 1.3 GET /api/dashboard/channels

| Campo | Valor |
|-------|-------|
| **Quem chama** | Dashboard.jsx |
| **Response** | `{ data: [{ name, value, ... }] }` |
| **Latencia** | < 5ms ✅ |

### 1.4 GET /api/dashboard/recent-alerts

| Campo | Valor |
|-------|-------|
| **Quem chama** | Dashboard.jsx |
| **Response** | `{ alerts: [...] }` |
| **Latencia** | < 5ms ✅ |

### 1.5 GET /api/dashboard/model-status

| Campo | Valor |
|-------|-------|
| **Quem chama** | Dashboard.jsx |
| **Response** | `{ models: [{ name, status, accuracy, ... }] }` |
| **Latencia** | < 5ms ✅ |

---

## 2. Fraud Prediction Endpoints

### 2.1 POST /api/fraud/predict

| Campo | Valor |
|-------|-------|
| **Quem chama** | Transactions.jsx (acao analisar) |
| **Metodo** | POST |
| **Content-Type** | application/json |
| **Payload** | `{ transactions: [{ amount, type, channel, ... }], include_explanation: true }` |
| **Response** | `{ success: true, data: { predictions: [{ is_fraud, risk_score, explanation }], summary } }` |
| **Tratamento 400** | Exibe mensagem de validacao |
| **Tratamento 500** | Exibe "Erro ao processar" |
| **Latencia** | 2691ms ❌ **CRITICO** (SLA: 50ms) |

### 2.2 POST /api/fraud/batch

| Campo | Valor |
|-------|-------|
| **Quem chama** | (nao implementado no frontend) |
| **Payload** | `{ transactions: [...], batch_size: 100 }` |
| **Response** | `{ predictions: [...], summary }` |
| **Latencia** | < 100ms por lote ✅ |

---

## 3. Transactions Endpoints

### 3.1 GET /api/transactions

| Campo | Valor |
|-------|-------|
| **Quem chama** | Transactions.jsx |
| **Query params** | `page`, `limit`, `status`, `type`, `period`, `search`, `sort`, `order` |
| **Response** | `{ transactions: [...], total, page, total_pages }` |
| **Tratamento erro** | Empty state ou erro generico |
| **Latencia** | 5.5ms ✅ |

### 3.2 POST /api/transactions/{id}/approve

| Campo | Valor |
|-------|-------|
| **Quem chama** | Transactions.jsx (botao Aprovar) |
| **Payload** | `{}` ou `{ notes }` |
| **Response** | `{ success: true, transaction }` |
| **Latencia** | < 50ms ✅ |

### 3.3 POST /api/transactions/{id}/reject

| Campo | Valor |
|-------|-------|
| **Quem chama** | Transactions.jsx (botao Rejeitar) |
| **Payload** | `{ reason }` |
| **Response** | `{ success: true, transaction }` |

### 3.4 POST /api/transactions/{id}/flag

| Campo | Valor |
|-------|-------|
| **Quem chama** | Transactions.jsx (botao Flag) |
| **Payload** | `{ reason }` |
| **Response** | `{ success: true, alert_id }` |

### 3.5 POST /api/investigations

| Campo | Valor |
|-------|-------|
| **Quem chama** | Transactions.jsx (botao Investigar) |
| **Payload** | `{ transaction_id, priority, notes }` |
| **Response** | `{ success: true, investigation_id }` |

---

## 4. Calibration Endpoints

### 4.1 GET /api/calibration/config

| Campo | Valor |
|-------|-------|
| **Quem chama** | Calibration.jsx |
| **Response** | `{ ensemble: {...}, rules: [...], weights: {...}, thresholds: {...} }` |

### 4.2 GET /api/calibration/impact

| Campo | Valor |
|-------|-------|
| **Quem chama** | Calibration.jsx (preview) |
| **Response** | `{ predicted_fraud_rate, predicted_fp_rate, predicted_tp_rate }` |

### 4.3 POST /api/calibration/apply

| Campo | Valor |
|-------|-------|
| **Quem chama** | Calibration.jsx (botao Aplicar) |
| **Payload** | `{ config: {...} }` |
| **Response** | `{ success: true, applied_at }` |

### 4.4 POST /api/calibration/reset

| Campo | Valor |
|-------|-------|
| **Quem chama** | Calibration.jsx (botao Resetar) |
| **Response** | `{ success: true, config }` |

---

## 5. Alerts Endpoints

### 5.1 GET /api/alerts

| Campo | Valor |
|-------|-------|
| **Quem chama** | Alerts.jsx, Dashboard.jsx |
| **Query params** | `type`, `severity`, `status`, `search` |
| **Response** | `{ alerts: [{ id, titulo, descricao, tipo, severidade, status, timestamp, ... }] }` |

### 5.2 PUT /api/alerts/{id}/status

| Campo | Valor |
|-------|-------|
| **Quem chama** | Alerts.jsx |
| **Payload** | `{ status: "investigando" | "resolvido" | "ignorado" }` |
| **Response** | `{ success: true }` |

---

## 6. List Management Endpoints

### 6.1 VIP List

| Endpoint | Metodo | Payload | Response |
|----------|--------|---------|----------|
| GET /api/vip-list | GET | - | `{ entries: [...] }` |
| POST /api/vip-list | POST | `{ cpf, name, reason }` | `{ success, entry }` |
| DELETE /api/vip-list/{id} | DELETE | - | `{ success }` |

### 6.2 HOT List

| Endpoint | Metodo | Payload | Response |
|----------|--------|---------|----------|
| GET /api/hot-list | GET | - | `{ entries: [...] }` |
| POST /api/hot-list | POST | `{ cpf, name, reason }` | `{ success, entry }` |
| DELETE /api/hot-list/{id} | DELETE | - | `{ success }` |

### 6.3 Hard Rules

| Endpoint | Metodo | Payload | Response |
|----------|--------|---------|----------|
| GET /api/hard-rules | GET | - | `{ rules: [...] }` |
| POST /api/hard-rules | POST | `{ name, condition, action, active }` | `{ success, rule }` |
| PUT /api/hard-rules/{id} | PUT | `{ ...updates }` | `{ success, rule }` |
| DELETE /api/hard-rules/{id} | DELETE | - | `{ success }` |

---

## 7. Observability Endpoints

### 7.1 GET /api/health

| Campo | Valor |
|-------|-------|
| **Quem chama** | (deveria ser Monitoring.jsx) |
| **Response** | `{ status: "healthy" | "degraded" | "unhealthy" }` |

### 7.2 GET /api/health/detailed

| Campo | Valor |
|-------|-------|
| **Response** | `{ database, redis, ml_engine, queue, ... }` |

### 7.3 GET /api/observability/metrics

| Campo | Valor |
|-------|-------|
| **Response** | `{ tps, latency_p50, latency_p95, error_rate, ... }` |

### 7.4 GET /api/observability/sla

| Campo | Valor |
|-------|-------|
| **Response** | `{ sla_status, violations: [...] }` |

**NOTA**: Monitoring.jsx NAO consome esses endpoints - usa dados mocados.

---

## 8. Other Endpoints

### 8.1 GET /api/metrics/dashboard

| Campo | Valor |
|-------|-------|
| **Quem chama** | Metrics.jsx |
| **Response** | `{ transactions_processed, fraud_detected, accuracy, ... }` |

### 8.2 GET /api/datasets

| Campo | Valor |
|-------|-------|
| **Quem chama** | Datasets.jsx |
| **Response** | `{ datasets: [{ id, name, size, fraud_ratio, ... }] }` |

### 8.3 GET /api/audit

| Campo | Valor |
|-------|-------|
| **Quem chama** | Audit.jsx |
| **Response** | `{ events: [{ id, action, user, timestamp, ... }] }` |

### 8.4 GET /api/settings

| Campo | Valor |
|-------|-------|
| **Quem chama** | Settings.jsx |
| **Response** | `{ general: {...}, security: {...}, notifications: {...} }` |

### 8.5 POST /api/feedback

| Campo | Valor |
|-------|-------|
| **Quem chama** | FeedbackAnalyst.jsx |
| **Payload** | `{ transaction_id, analyst_id, decision, notes }` |
| **Response** | `{ success: true }` |

---

## 9. Problemas de Contrato Identificados

### 9.1 Inconsistencias

| Problema | Impacto | Solucao |
|----------|---------|---------|
| Monitoring.jsx nao usa APIs reais | Alto | Integrar com /api/health/* e /api/observability/* |
| Fallback para dados mocados em Metrics.jsx | Medio | Exibir erro ao usuario |
| Datasets exibe NaN% | Baixo | Tratar valores nulos |

### 9.2 Tratamento de Erros Incompleto

| Endpoint | Problema |
|----------|----------|
| /api/fraud/predict | Erro 500 exibe mensagem generica |
| /api/transactions | Timeout nao tratado |

### 9.3 Campos Assumidos

| Endpoint | Campo | Risco |
|----------|-------|-------|
| /api/alerts | `valor_envolvido` | Pode ser null |
| /api/alerts | `tags` | Pode ser null |
| /api/dashboard/model-status | `models` | Pode ser array vazio |

---

## 10. Recomendacoes

1. **Adicionar null checks** em todos os acessos a campos opcionais
2. **Exibir mensagens de erro especificas** (nao apenas "Erro")
3. **Tratar timeouts** com retry automatico
4. **Integrar Monitoring.jsx** com endpoints reais
5. **Documentar contratos** em OpenAPI/Swagger

---

*Documento gerado conforme MODO DOUBLE CHECK ULTRA - Fase 3*
