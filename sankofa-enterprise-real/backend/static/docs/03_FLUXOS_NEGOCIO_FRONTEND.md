# 03 - Fluxos de Negocio do Frontend

**Data da Analise:** 29/11/2025  
**Metodologia:** MODO DOUBLE CHECK ULTRA - Fase 2

---

## 1. Fluxo: Dashboard Executivo

### Diagrama Textual

```
[Usuario acessa /]
    ↓
[Dashboard.jsx monta]
    ↓
[useEffect dispara loadDashboard()]
    ↓
[Promise.all - 5 chamadas paralelas]
    ├── GET /api/dashboard/kpis
    ├── GET /api/dashboard/timeseries
    ├── GET /api/dashboard/channels
    ├── GET /api/dashboard/recent-alerts
    └── GET /api/dashboard/model-status
    ↓
[Estados atualizados: kpis, timeSeriesData, channelData, recentAlerts, modelStatus]
    ↓
[Renderiza: KPICards + Graficos + Tabela Alertas + Status Modelos]
    ↓
[setInterval 30s → refresh automatico]
```

### Componentes Envolvidos

- `Dashboard.jsx`
- `KPICard.jsx`
- `SimpleChart.jsx` (Line, Area, Pie)
- `Badge.jsx`

### Tratamento de Erros

- Loading state com skeleton
- Error state com mensagem generica
- Fallback para dados vazios

---

## 2. Fluxo: Predicao Unitaria de Fraude

### Diagrama Textual

```
[Usuario em /transactions]
    ↓
[Seleciona transacao → clica "Analisar"]
    ↓
[Transactions.jsx chama analyzeTransaction()]
    ↓
[POST /api/fraud/predict]
    Body: { transactions: [{ amount, type, channel, ... }] }
    ↓
[Backend processa com ML Engine]
    ↓
[Resposta: { predictions: [{ is_fraud, risk_score, explanation }] }]
    ↓
[Atualiza estado da transacao]
    ↓
[Exibe resultado com RiskScoreBadge]
```

### Problema Identificado

- **Latencia**: 2.69s (SLA: < 50ms para PIX)
- **Causa**: Geracao de explicacoes SHAP e muito lenta

### Componentes Envolvidos

- `Transactions.jsx`
- `RiskScoreBadge.jsx`
- Modal de detalhes

---

## 3. Fluxo: Predicao em Batch

### Diagrama Textual

```
[Usuario em tela de batch (nao implementada no frontend)]
    ↓
[Upload de arquivo CSV/JSON]
    ↓
[POST /api/fraud/batch ou /api/infrastructure/batch/process]
    Body: { transactions: [...], batch_size: 100 }
    ↓
[Backend processa em lotes]
    ↓
[Resposta: { results: [...], summary: {...} }]
    ↓
[Exibe resultados + estatisticas]
```

### Status

- Endpoint funcional no backend
- **Frontend nao possui UI dedicada para batch**
- Recomendacao: Criar pagina BatchPrediction.jsx

---

## 4. Fluxo: Lista de Transacoes

### Diagrama Textual

```
[Usuario acessa /transactions]
    ↓
[Transactions.jsx monta]
    ↓
[useEffect dispara loadTransactions()]
    ↓
[GET /api/transactions?page=1&limit=50&period=24h]
    ↓
[Estado atualizado: transactions, totalPages, totalTransactions]
    ↓
[Renderiza tabela com TransactionStatusBadge, RiskScoreBadge]
    ↓
[Usuario pode:]
    ├── Filtrar por status/tipo/periodo
    ├── Buscar por texto
    ├── Ordenar por coluna
    ├── Paginar
    └── Acoes: Aprovar/Rejeitar/Flag/Investigar
```

### Componentes Envolvidos

- `Transactions.jsx`
- `Badge.jsx` (TransactionStatusBadge, RiskScoreBadge)
- `Button.jsx`
- `Input.jsx` (busca)
- Modal de detalhes

### Tratamento de Erros

- Loading state
- Empty state ("Nenhuma transacao encontrada")
- Error state com retry

---

## 5. Fluxo: Calibracao de Modelos

### Diagrama Textual

```
[Usuario acessa /calibration]
    ↓
[Calibration.jsx monta]
    ↓
[GET /api/calibration/config]
    ↓
[Renderiza tabs: Ensemble | Regras | Pesos | Limiares | Features]
    ↓
[Usuario ajusta sliders]
    ↓
[Estado local atualizado: config, hasChanges=true]
    ↓
[GET /api/calibration/impact (calculo de impacto)]
    ↓
[Exibe preview de impacto]
    ↓
[Usuario clica "Aplicar"]
    ↓
[POST /api/calibration/apply]
    ↓
[Confirmacao de sucesso + timestamp]
```

### Componentes Envolvidos

- `Calibration.jsx`
- `Slider.jsx`
- `Switch.jsx`
- `Tabs` (inline)
- `Button.jsx`

---

## 6. Fluxo: Metricas de Modelo

### Diagrama Textual

```
[Usuario acessa /metrics]
    ↓
[Metrics.jsx monta]
    ↓
[GET /api/metrics/dashboard]
    ↓
[Renderiza KPIs: Transacoes, Fraudes, Precisao, Tempo, Hard Rules, VIP/HOT Hits]
    ↓
[Auto-refresh 30s se ativado]
```

### Problema Identificado

- Em caso de erro, usa dados mocados (fallback)
- Nao exibe mensagem de erro ao usuario

---

## 7. Fluxo: Observabilidade e Health

### Diagrama Textual

```
[Usuario acessa /monitoring]
    ↓
[Monitoring.jsx monta]
    ↓
[PROBLEMA: Dados mocados localmente]
    ↓
[setInterval 3s atualiza valores aleatorios]
    ↓
[Exibe: CPU, Memoria, Disco, Latencia, TPS, Alertas]
```

### Problema Critico

- **Nao consome endpoints reais**:
  - `/api/health/detailed`
  - `/api/observability/metrics`
  - `/api/observability/sla`
- Usuario ve dados falsos

### Recomendacao

Integrar com endpoints reais do backend.

---

## 8. Fluxo: Central de Alertas

### Diagrama Textual

```
[Usuario acessa /alerts]
    ↓
[Alerts.jsx monta]
    ↓
[GET /api/alerts]
    ↓
[Renderiza: Stats Cards + Lista de Alertas + Painel de Detalhes]
    ↓
[Usuario pode:]
    ├── Filtrar por tipo/severidade/status
    ├── Buscar por texto
    ├── Ver detalhes (clique)
    └── Atualizar status (Novo → Investigando → Resolvido)
    ↓
[PUT /api/alerts/{id}/status]
    ↓
[Estado atualizado + stats recalculadas]
```

### Componentes Envolvidos

- `Alerts.jsx`
- `Card.jsx`
- `Badge.jsx`
- `Button.jsx`
- `Input.jsx`

---

## 9. Fluxo: Gestao de Listas (VIP/HOT/HardRules)

### Diagrama Textual (Generico)

```
[Usuario acessa /vip-list ou /hot-list ou /hard-rules]
    ↓
[GET /api/{lista}]
    ↓
[Renderiza tabela com itens]
    ↓
[Usuario pode:]
    ├── Adicionar (POST /api/{lista})
    ├── Editar (PUT /api/{lista}/{id}) [apenas HardRules]
    ├── Excluir (DELETE /api/{lista}/{id})
    └── Buscar
```

---

## 10. Fluxo: Auditoria

### Diagrama Textual

```
[Usuario acessa /audit]
    ↓
[GET /api/audit]
    ↓
[Renderiza: Filtros + Timeline de Eventos]
    ↓
[Usuario pode:]
    ├── Filtrar por periodo/tipo/usuario
    └── Exportar (POST /api/audit/export)
```

---

## 11. Resumo de Fluxos

| Fluxo | Status | Problemas |
|-------|--------|-----------|
| Dashboard | ✅ OK | - |
| Predicao Unitaria | ⚠️ Lento | Latencia 2.69s |
| Predicao Batch | ❌ Sem UI | Frontend nao implementado |
| Lista Transacoes | ✅ OK | - |
| Calibracao | ✅ OK | Arquivo grande |
| Metricas | ⚠️ | Fallback mocado |
| Observabilidade | ❌ Mocado | Nao usa backend |
| Alertas | ✅ OK | - |
| Listas | ✅ OK | - |
| Auditoria | ✅ OK | - |

---

*Documento gerado conforme MODO DOUBLE CHECK ULTRA - Fase 2*
