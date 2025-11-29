# Inventario de Componentes - Sankofa Enterprise Pro

**Versao:** 11.0  
**Data da Analise:** 29/11/2025  
**Escopo:** Todos os componentes em `src/pages/`, `src/components/`, `src/hooks/`

---

## 1. Paginas (src/pages/)

### 1.1 Dashboard.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Dashboard executivo com KPIs e graficos |
| **Props** | Nenhuma |
| **Estado Interno** | `loading`, `kpis`, `timeSeriesData`, `channelData`, `recentAlerts`, `modelStatus`, `lastUpdate` |
| **Hooks** | `useState`, `useEffect` |
| **Side Effects** | Fetch paralelo de 5 endpoints, polling de 30s |
| **Endpoints** | `/api/dashboard/kpis`, `/api/dashboard/timeseries`, `/api/dashboard/channels`, `/api/dashboard/recent-alerts`, `/api/dashboard/model-status` |
| **Riscos** | Nenhum intervalo de limpeza observado (OK - limpo em useEffect) |
| **Avaliacao** | ✅ Bem estruturado |

### 1.2 Transactions.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Lista, busca e acoes em transacoes |
| **Props** | Nenhuma |
| **Estado Interno** | `transactions`, `loading`, `searchQuery`, `statusFilter`, `typeFilter`, `sortField`, `sortDirection`, `currentPage`, `totalPages`, `totalTransactions`, `selectedTransaction`, `showDetailsModal`, `showActionsMenu`, `actionLoading`, `periodFilter`, `showPeriodMenu`, `exportLoading` |
| **Hooks** | `useState`, `useEffect`, `useMemo` |
| **Side Effects** | Fetch de transacoes, acoes POST (approve, reject, flag, investigate) |
| **Endpoints** | `/api/transactions`, `/api/transactions/{id}/approve`, `/api/transactions/{id}/reject`, `/api/transactions/{id}/review`, `/api/transactions/{id}/flag`, `/api/investigations` |
| **Riscos** | Muitos estados - considerar useReducer |
| **Avaliacao** | ⚠️ Funcional, mas complexo |

### 1.3 Calibration.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Calibracao de 17+ algoritmos de ML |
| **Props** | Nenhuma |
| **Estado Interno** | `config`, `hasChanges`, `applying`, `lastApplied`, `activeTab`, `activeGlobalTab`, `impactData`, `loading` |
| **Hooks** | `useState`, `useEffect` |
| **Side Effects** | Fetch de config, POST de apply/reset |
| **Endpoints** | `/api/calibration/config`, `/api/calibration/impact`, `/api/calibration/apply`, `/api/calibration/reset` |
| **Riscos** | Arquivo muito grande (1400+ linhas) |
| **Avaliacao** | ⚠️ Funcional, precisa modularizacao |

### 1.4 Investigation.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Central de investigacao de fraudes |
| **Estado Interno** | Multiplos estados para filtros e selecao |
| **Endpoints** | `/api/investigations`, `/api/investigation/{id}` |
| **Avaliacao** | ✅ OK |

### 1.5 ManualReview.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Revisao manual Human-in-the-Loop |
| **Endpoints** | `/api/manual-review` |
| **Avaliacao** | ✅ OK |

### 1.6 Monitoring.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Monitoramento de sistema e recursos |
| **Estado Interno** | `systemHealth` (objeto grande), `alerts`, `autoRefresh` |
| **Side Effects** | Interval de 3s para simular dados |
| **Endpoints** | NENHUM - Dados mocados localmente |
| **Riscos** | ❌ **CRITICO**: Dados hardcoded, nao reflete sistema real |
| **Avaliacao** | ❌ Precisa integracao com backend |

### 1.7 Metrics.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Metricas e contadores em tempo real |
| **Estado Interno** | `metrics`, `loading`, `autoRefresh` |
| **Endpoints** | `/api/metrics/dashboard` |
| **Riscos** | Fallback para dados mocados em caso de erro |
| **Avaliacao** | ⚠️ Precisa tratamento de erro melhor |

### 1.8 Alerts.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Central de alertas |
| **Estado Interno** | `alerts`, `loading`, `searchQuery`, filtros multiplos, `selectedAlert`, `stats` |
| **Endpoints** | `/api/alerts`, `/api/alerts/{id}/status` |
| **Side Effects** | Polling de 30s |
| **Avaliacao** | ✅ Bem estruturado |

### 1.9 Reports.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Geracao de relatorios |
| **Endpoints** | `/api/reports`, `/api/reports/generate` |
| **Avaliacao** | ✅ OK |

### 1.10 Datasets.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Catalogo de datasets |
| **Endpoints** | `/api/datasets` |
| **Riscos** | Mostra "NaN%" quando dados nulos |
| **Avaliacao** | ⚠️ Precisa tratamento de valores nulos |

### 1.11-1.13 HardRules.jsx, VipList.jsx, HotList.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | CRUD de listas de bloqueio/aprovacao |
| **Endpoints** | `/api/hard-rules`, `/api/vip-list`, `/api/hot-list` |
| **Avaliacao** | ✅ OK |

### 1.14 Audit.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Trilhas de auditoria |
| **Endpoints** | `/api/audit`, `/api/audit/export` |
| **Avaliacao** | ✅ OK |

### 1.15 Settings.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Configuracoes do sistema |
| **Endpoints** | `/api/settings` |
| **Avaliacao** | ✅ OK |

### 1.16 FeedbackAnalyst.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Responsabilidade** | Feedback de analistas |
| **Endpoints** | `/api/feedback` |
| **Avaliacao** | ✅ OK |

---

## 2. Componentes de UI (src/components/ui/)

### 2.1 Button.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Variants** | `primary`, `secondary`, `outline`, `ghost`, `destructive` |
| **Sizes** | `default`, `sm`, `lg`, `icon` |
| **Props** | `variant`, `size`, `asChild`, `className`, ...rest |
| **Acessibilidade** | ✅ Usa Radix Slot |

### 2.2 Card.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Componentes** | `Card`, `CardHeader`, `CardTitle`, `CardDescription`, `CardContent`, `CardFooter` |
| **Acessibilidade** | ✅ Semantica correta |

### 2.3 Badge.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Componentes** | `Badge`, `TransactionStatusBadge`, `RiskScoreBadge` |
| **Variants** | `default`, `secondary`, `destructive`, `success`, `warning`, `info`, `outline` |
| **Acessibilidade** | ⚠️ Falta aria-label em badges de status |

### 2.4 Input.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Componentes** | `Input`, `FormField` |
| **Props** | `label`, `error`, `helper`, etc. |
| **Acessibilidade** | ⚠️ Labels presentes, mas falta aria-describedby |

### 2.5 Slider.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Componentes** | `Slider`, `SliderControl` |
| **Props** | `label`, `value`, `onValueChange`, `min`, `max`, `step`, `format`, `description` |
| **Acessibilidade** | ✅ Usa Radix Slider |

### 2.6 Switch.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Componentes** | `Switch`, `SwitchControl` |
| **Acessibilidade** | ✅ Usa Radix Switch |

---

## 3. Componentes de Charts (src/components/charts/)

### 3.1 KPICard.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Props** | `title`, `value`, `previousValue`, `format`, `icon` |
| **Formatos** | `number`, `percentage`, `currency`, `decimal` |
| **Features** | Calculo de variacao percentual |

### 3.2 SimpleChart.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Componentes** | `SimpleLineChart`, `SimpleAreaChart`, `SimpleBarChart`, `SimplePieChart` |
| **Biblioteca** | Recharts |
| **Props** | `title`, `data`, `dataKey`, `xAxisKey`, `color`, `formatter` |

---

## 4. Componentes de Layout (src/components/layout/)

### 4.1 Layout.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Props** | `children`, `currentPath` |
| **Estrutura** | AppBar + Sidebar + Main Content |

### 4.2 Sidebar.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Props** | `isOpen`, `onToggle`, `currentPath`, `className` |
| **Features** | Collapsible, mobile overlay, active state |
| **Acessibilidade** | ✅ `aria-label`, `aria-current` implementados |

### 4.3 AppBar.jsx

| Aspecto | Detalhes |
|---------|----------|
| **Features** | Logo, search, theme toggle, notifications, user menu |
| **Acessibilidade** | ⚠️ Falta aria-label em alguns botoes |

---

## 5. Hooks (src/hooks/)

### 5.1 use-mobile.js

| Aspecto | Detalhes |
|---------|----------|
| **Retorno** | `boolean` indicando se e mobile |
| **Breakpoint** | 768px |
| **Features** | Listener de resize com cleanup |

---

## 6. Resumo de Problemas por Componente

| Componente | Severidade | Problema |
|------------|------------|----------|
| Monitoring.jsx | ❌ CRITICO | Dados mocados |
| Calibration.jsx | ⚠️ MEDIO | Arquivo muito grande |
| Transactions.jsx | ⚠️ MEDIO | Muitos estados |
| Datasets.jsx | ⚠️ BAIXO | NaN% em valores nulos |
| Metrics.jsx | ⚠️ BAIXO | Fallback para dados mocados |
| Badge.jsx | ⚠️ BAIXO | Falta aria-labels |
| Input.jsx | ⚠️ BAIXO | Falta aria-describedby |
| AppBar.jsx | ⚠️ BAIXO | Falta aria-labels |

---

*Documento gerado conforme diretrizes MODO ULTRA*
