# 07 - Relatorio de Performance do Frontend

**Data da Analise:** 29/11/2025  
**Metodologia:** MODO DOUBLE CHECK ULTRA - Fase 5.4

---

## 1. Resumo Executivo

| Metrica | Valor | Target | Status |
|---------|-------|--------|--------|
| Bundle Size (JS) | 878 KB | < 500 KB | ⚠️ Alto |
| Bundle Total | 3.8 MB | < 2 MB | ❌ Alto |
| FCP (est.) | ~1.5s | < 1.8s | ✅ OK |
| LCP (est.) | ~2.5s | < 2.5s | ⚠️ Limite |
| API Latencia P95 | 2691ms | < 50ms | ❌ CRITICO |
| Re-renders | Alguns | Minimo | ⚠️ |

---

## 2. Analise de Bundle

### 2.1 Composicao do Bundle

| Modulo | Tamanho | % | Notas |
|--------|---------|---|-------|
| recharts | 475 KB | 54% | Maior dependencia |
| framer-motion | 138 KB | 16% | Animacoes |
| react-dom | 157 KB | 18% | Necessario |
| app code | 108 KB | 12% | Codigo proprio |
| **Total JS** | **878 KB** | 100% | - |

### 2.2 Oportunidades de Otimizacao

| Acao | Economia | Esforco |
|------|----------|---------|
| Lazy load Recharts | ~400 KB | Medio |
| Lazy load Framer Motion | ~100 KB | Baixo |
| Code split rotas | ~200 KB inicial | Medio |
| Tree shake Radix | ~50 KB | Baixo |

---

## 3. Analise de Rotas

### 3.1 Carregamento Atual

**Todas as rotas sao carregadas no bundle principal:**

```javascript
// App.jsx - Carregamento sincrono
import Dashboard from './pages/Dashboard'
import Transactions from './pages/Transactions'
import Calibration from './pages/Calibration'
// ... mais 13 paginas
```

### 3.2 Recomendacao: Lazy Loading

```javascript
// App.jsx - Com lazy loading
import { lazy, Suspense } from 'react'

const Dashboard = lazy(() => import('./pages/Dashboard'))
const Transactions = lazy(() => import('./pages/Transactions'))
const Calibration = lazy(() => import('./pages/Calibration'))
// ...

function App() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        {/* ... */}
      </Routes>
    </Suspense>
  )
}
```

**Economia estimada:** 400-500 KB no carregamento inicial.

---

## 4. Analise de Renderizacao

### 4.1 Componentes com Potencial de Re-render

| Componente | Estado | Risco | Solucao |
|------------|--------|-------|---------|
| Transactions.jsx | 16 estados | Alto | useReducer |
| Calibration.jsx | 8 estados | Medio | Memoizacao |
| Dashboard.jsx | 7 estados | Baixo | OK |

### 4.2 Uso de Memoizacao

```bash
grep -r "useMemo\|useCallback\|React.memo" src/
# Resultado: 
# - useMemo: 2 usos (Transactions, Dashboard)
# - useCallback: 0 usos
# - React.memo: 0 usos
```

### 4.3 Recomendacoes

```javascript
// KPICard.jsx - Memoizar componente puro
const KPICard = React.memo(function KPICard({ title, value, change, icon }) {
  // ...
});

// Transactions.jsx - Memoizar handlers
const handleSearch = useCallback((e) => {
  setSearchQuery(e.target.value);
}, []);

// Dashboard.jsx - Memoizar calculos
const fraudRate = useMemo(() => {
  return (kpis.fraud_count / kpis.total_transactions * 100).toFixed(2);
}, [kpis.fraud_count, kpis.total_transactions]);
```

---

## 5. Listas e Virtualizacao

### 5.1 Componentes com Listas Grandes

| Componente | Lista | Tamanho Max | Virtualizacao |
|------------|-------|-------------|---------------|
| Transactions.jsx | Transacoes | 1000+ | ❌ Nao |
| Alerts.jsx | Alertas | 500+ | ❌ Nao |
| Audit.jsx | Eventos | 10000+ | ❌ Nao |
| Calibration.jsx | Algoritmos | 17 | N/A |

### 5.2 Recomendacao: Virtualizar Listas Grandes

```javascript
// Instalar: npm install @tanstack/react-virtual

import { useVirtualizer } from '@tanstack/react-virtual'

function VirtualizedTransactionList({ transactions }) {
  const parentRef = useRef(null)
  
  const rowVirtualizer = useVirtualizer({
    count: transactions.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 60,
    overscan: 5,
  })

  return (
    <div ref={parentRef} style={{ height: 600, overflow: 'auto' }}>
      <div style={{ height: rowVirtualizer.getTotalSize() }}>
        {rowVirtualizer.getVirtualItems().map(virtualRow => (
          <TransactionRow 
            key={virtualRow.key}
            style={{ transform: `translateY(${virtualRow.start}px)` }}
            transaction={transactions[virtualRow.index]}
          />
        ))}
      </div>
    </div>
  )
}
```

---

## 6. Performance de APIs

### 6.1 Latencias Medidas

| Endpoint | Latencia | SLA | Status |
|----------|----------|-----|--------|
| GET /api/dashboard/kpis | 3.9ms | 100ms | ✅ |
| GET /api/transactions | 5.5ms | 100ms | ✅ |
| GET /api/alerts | 4.2ms | 100ms | ✅ |
| POST /api/fraud/predict | **2691ms** | 50ms | ❌ CRITICO |
| GET /api/calibration/config | 8.3ms | 100ms | ✅ |

### 6.2 Causa da Latencia em /api/fraud/predict

**Identificado:** O endpoint chama `explainability_engine.explain_prediction()` que usa SHAP (computacionalmente caro) para cada transacao.

```python
# Codigo atual (lento)
for i, pred in enumerate(predictions):
    explanation = explainability_engine.explain_prediction(...)  # ~2.5s
```

### 6.3 Solucao Proposta

```python
# Opcao 1: Tornar explicacoes opcionais
include_explanation = request.json.get("include_explanation", False)  # Default FALSE

# Opcao 2: Cache de explicacoes
cached = redis_cache.get(f"explain:{txn_id}")
if cached:
    explanation = cached
else:
    explanation = explainability_engine.explain_prediction(...)
    redis_cache.set(f"explain:{txn_id}", explanation, ttl=3600)

# Opcao 3: Fallback rapido (sem SHAP)
explanation = explainability_engine.get_fast_explanation(X_features)  # ~5ms
```

---

## 7. Polling e Atualizacoes

### 7.1 Intervalos de Polling

| Componente | Intervalo | Endpoint | Impacto |
|------------|-----------|----------|---------|
| Dashboard | 30s | 5 endpoints | Medio |
| Alerts | 30s | 1 endpoint | Baixo |
| Monitoring | 3s | (mocado) | Nenhum |
| Metrics | 30s | 1 endpoint | Baixo |

### 7.2 Recomendacoes

1. **Unificar polling:** Usar um unico endpoint `/api/dashboard/all` que retorna todos os dados
2. **WebSocket:** Para atualizacoes em tempo real de alertas criticos
3. **Visibility API:** Pausar polling quando tab inativa

```javascript
useEffect(() => {
  const handleVisibility = () => {
    if (document.hidden) {
      clearInterval(intervalRef.current);
    } else {
      fetchData();
      intervalRef.current = setInterval(fetchData, 30000);
    }
  };
  
  document.addEventListener('visibilitychange', handleVisibility);
  return () => document.removeEventListener('visibilitychange', handleVisibility);
}, []);
```

---

## 8. Metricas de Core Web Vitals (Estimadas)

| Metrica | Valor Estimado | Target | Status |
|---------|----------------|--------|--------|
| LCP | 2.5s | < 2.5s | ⚠️ Limite |
| FID | ~50ms | < 100ms | ✅ OK |
| CLS | ~0.05 | < 0.1 | ✅ OK |
| FCP | 1.5s | < 1.8s | ✅ OK |
| TTI | 3.0s | < 3.8s | ✅ OK |

---

## 9. Plano de Otimizacao

### 9.1 Prioridade Alta (Impacto Imediato)

| Acao | Impacto | Tempo |
|------|---------|-------|
| Corrigir latencia /api/fraud/predict | -2.6s | 2h |
| Lazy load rotas | -300KB inicial | 1h |
| Memoizar componentes puros | Menos re-renders | 1h |

### 9.2 Prioridade Media

| Acao | Impacto | Tempo |
|------|---------|-------|
| Virtualizar listas | Scroll suave | 2h |
| Otimizar polling | Menos requests | 1h |
| Tree shake Recharts | -200KB | 1h |

### 9.3 Prioridade Baixa

| Acao | Impacto | Tempo |
|------|---------|-------|
| Compression (gzip) | -60% transfer | Config |
| Preload fontes | -100ms FCP | Config |
| Service Worker | Offline support | 4h |

---

## 10. Conclusao

**Status Geral:** ⚠️ Performance aceitavel com gargalo critico no backend.

**Acoes Imediatas:**
1. Corrigir latencia de 2.69s no endpoint de predicao
2. Implementar lazy loading de rotas
3. Adicionar virtualizacao em listas grandes

---

*Documento gerado conforme MODO DOUBLE CHECK ULTRA - Fase 5.4*
