# 02 - Inventario Detalhado de Componentes, Hooks e Paginas

**Data da Analise:** 29/11/2025  
**Metodologia:** MODO DOUBLE CHECK ULTRA - Fase 1.2

---

## 1. Paginas - Analise Detalhada

### 1.1 Dashboard.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Dashboard executivo com KPIs, graficos e alertas |
| **Props** | Nenhuma |
| **useState** | `loading`, `kpis`, `timeSeriesData`, `channelData`, `recentAlerts`, `modelStatus`, `lastUpdate` |
| **useEffect** | 1 (carregamento inicial + polling 30s) |
| **Side Effects** | Fetch paralelo de 5 endpoints |
| **Endpoints** | `/api/dashboard/kpis`, `/api/dashboard/timeseries`, `/api/dashboard/channels`, `/api/dashboard/recent-alerts`, `/api/dashboard/model-status` |
| **Faz mais de uma coisa?** | Nao - foco em exibicao de dados |
| **Depende de HTTP diretamente?** | Sim - fetch inline |
| **Riscos** | Nenhum (cleanup de interval OK) |
| **Avaliacao** | ✅ Bem estruturado |

### 1.2 Transactions.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Lista, busca, filtros e acoes em transacoes |
| **Props** | Nenhuma |
| **useState** | 16 estados (transactions, loading, searchQuery, filters, pagination, modals, etc.) |
| **useEffect** | 1 (carregamento com dependencias de filtros) |
| **useMemo** | 1 (transacoes filtradas) |
| **Side Effects** | GET transacoes, POST approve/reject/flag/investigate |
| **Endpoints** | `/api/transactions`, `/api/transactions/{id}/approve`, `/api/transactions/{id}/reject`, `/api/transactions/{id}/flag`, `/api/investigations` |
| **Faz mais de uma coisa?** | Sim - lista + acoes + filtros + modais |
| **Riscos** | Muitos estados - useReducer recomendado |
| **Avaliacao** | ⚠️ Funcional mas complexo |

### 1.3 Calibration.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Calibracao de 17+ algoritmos de ML |
| **Props** | Nenhuma |
| **useState** | `config`, `hasChanges`, `applying`, `lastApplied`, `activeTab`, `activeGlobalTab`, `impactData`, `loading` |
| **useEffect** | 2 (carregamento config + calculo impacto) |
| **Side Effects** | GET config, POST apply/reset |
| **Endpoints** | `/api/calibration/config`, `/api/calibration/impact`, `/api/calibration/apply`, `/api/calibration/reset` |
| **Faz mais de uma coisa?** | Sim - muitas tabs e controles |
| **Riscos** | Arquivo muito grande (1400+ linhas) |
| **Avaliacao** | ⚠️ Precisa modularizacao |

### 1.4 Investigation.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Central de investigacao de fraudes |
| **Props** | Nenhuma |
| **useState** | Multiplos (investigations, loading, filters, selectedCase) |
| **Side Effects** | GET/POST investigations |
| **Endpoints** | `/api/investigations`, `/api/investigation/{id}` |
| **Avaliacao** | ✅ OK |

### 1.5 ManualReview.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Fila de revisao Human-in-the-Loop |
| **useState** | `queue`, `loading`, `selectedItem`, `decision`, `notes` |
| **Side Effects** | GET/PUT manual-review |
| **Endpoints** | `/api/manual-review` |
| **Avaliacao** | ✅ OK |

### 1.6 Monitoring.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Monitoramento de sistema e recursos |
| **Props** | Nenhuma |
| **useState** | `systemHealth` (objeto com CPU, memoria, etc.), `alerts`, `autoRefresh` |
| **useEffect** | 1 (interval 3s para simulacao) |
| **Side Effects** | NENHUM - dados mocados localmente |
| **Endpoints** | NENHUM |
| **Faz mais de uma coisa?** | Nao |
| **Riscos** | **CRITICO**: Dados hardcoded, nao reflete sistema real |
| **Avaliacao** | ❌ Precisa integracao com backend |

### 1.7 Metrics.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Metricas em tempo real |
| **useState** | `metrics`, `loading`, `autoRefresh` |
| **Side Effects** | GET metrics/dashboard |
| **Endpoints** | `/api/metrics/dashboard` |
| **Riscos** | Fallback para dados mocados |
| **Avaliacao** | ⚠️ Melhorar tratamento de erro |

### 1.8 Alerts.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Central de alertas |
| **useState** | `alerts`, `loading`, `searchQuery`, `typeFilter`, `severityFilter`, `statusFilter`, `selectedAlert`, `stats` |
| **useEffect** | 1 (carregamento + polling 30s) |
| **Side Effects** | GET alerts, PUT alert status |
| **Endpoints** | `/api/alerts`, `/api/alerts/{id}/status` |
| **Avaliacao** | ✅ Bem estruturado |

### 1.9-1.16 Demais Paginas

| Pagina | Avaliacao | Notas |
|--------|-----------|-------|
| Reports | ✅ OK | GET/POST reports |
| Datasets | ⚠️ | NaN% em valores nulos |
| HardRules | ✅ OK | CRUD completo |
| VipList | ✅ OK | CRUD completo |
| HotList | ✅ OK | CRUD completo |
| Audit | ✅ OK | GET + export |
| Settings | ✅ OK | GET/PUT |
| FeedbackAnalyst | ✅ OK | POST feedback |

---

## 2. Componentes UI - Analise

### 2.1 Button.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Props** | `variant`, `size`, `asChild`, `className`, ...rest |
| **Variants** | primary, secondary, outline, ghost, destructive |
| **Sizes** | default, sm, lg, icon |
| **Acessibilidade** | ✅ Usa Radix Slot |
| **Avaliacao** | ✅ OK |

### 2.2 Card.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Componentes** | Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter |
| **Composicao** | Componentes compostos |
| **Acessibilidade** | ✅ Semantica HTML correta |
| **Avaliacao** | ✅ OK |

### 2.3 Badge.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Componentes** | Badge, TransactionStatusBadge, RiskScoreBadge |
| **Variants** | default, secondary, destructive, success, warning, info, outline |
| **Acessibilidade** | ⚠️ Falta aria-label |
| **Avaliacao** | ⚠️ Melhorar a11y |

### 2.4 Input.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Componentes** | Input, FormField |
| **Props** | label, error, helper, etc. |
| **Acessibilidade** | ⚠️ Falta aria-describedby |
| **Avaliacao** | ⚠️ Melhorar a11y |

### 2.5 Slider.jsx / Switch.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Base** | Radix UI primitives |
| **Acessibilidade** | ✅ Radix garante a11y |
| **Avaliacao** | ✅ OK |

---

## 3. Componentes Layout - Analise

### 3.1 Sidebar.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Props** | `isOpen`, `onToggle`, `currentPath`, `className` |
| **Features** | Colapsavel, overlay mobile, active state |
| **Navegacao** | 15 itens de menu |
| **Acessibilidade** | ✅ aria-label, aria-current |
| **Avaliacao** | ✅ OK |

### 3.2 AppBar.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Features** | Logo, search, theme toggle, notifications, user menu |
| **Acessibilidade** | ⚠️ Falta aria-label em alguns botoes |
| **Avaliacao** | ⚠️ Melhorar a11y |

---

## 4. Hooks - Analise

### 4.1 use-mobile.js

| Aspecto | Detalhes |
|---------|----------|
| **Retorno** | `boolean` (isMobile) |
| **Logica** | window.innerWidth < 768 |
| **Cleanup** | ✅ Remove listener |
| **Avaliacao** | ✅ OK |

---

## 5. Resumo de Problemas por Prioridade

### Criticos (P0)

| Componente | Problema |
|------------|----------|
| Monitoring.jsx | Dados mocados - nao reflete sistema real |

### Altos (P1)

| Componente | Problema |
|------------|----------|
| Calibration.jsx | Arquivo muito grande (1400+ linhas) |
| Transactions.jsx | Muitos estados (16) - useReducer recomendado |

### Medios (P2)

| Componente | Problema |
|------------|----------|
| Badge.jsx | Falta aria-labels |
| Input.jsx | Falta aria-describedby |
| AppBar.jsx | Falta aria-labels em botoes |
| Metrics.jsx | Fallback para dados mocados |
| Datasets.jsx | NaN% em valores nulos |

### Baixos (P3)

| Componente | Problema |
|------------|----------|
| Sidebar badges | Valores estaticos (nao dinamicos) |

---

*Documento gerado conforme MODO DOUBLE CHECK ULTRA - Fase 1.2*
