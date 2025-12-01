# 🧪 PLANO DE TESTES - DASHBOARD EXECUTIVO
## Sankofa Enterprise Pro - Tela de Detecção de Fraudes

**Data**: Dezembro 01, 2025  
**Versão**: 1.0  
**Status**: Pronto para Execução

---

## 📋 ÍNDICE
1. [Mapeamento de Componentes](#1-mapeamento-de-componentes)
2. [Arquitetura Front-End + Back-End](#2-arquitetura-front-end--back-end)
3. [Plano de Testes Funcionais](#3-plano-de-testes-funcionais)
4. [Testes de Integração](#4-testes-de-integração)
5. [Testes de Gráficos](#5-testes-de-gráficos)
6. [Testes de Estados e UX](#6-testes-de-estados-e-ux)
7. [Estratégia de Automação](#7-estratégia-de-automação)
8. [Checklist Final](#8-checklist-final)

---

## 1. MAPEAMENTO DE COMPONENTES

### 1.1 Estrutura React (Frontend)

```
src/pages/Dashboard.jsx (MAIN COMPONENT)
├── Header (linha 119-139)
│   ├── Título: "Dashboard Executivo"
│   ├── Subtítulo: "Visão geral do sistema de detecção de fraudes"
│   ├── Badge "Sistema Online" (status do sistema)
│   ├── Badge "N Algoritmos Ativos" (status do ML)
│   ├── Timestamp "Atualizado: HH:MM:SS"
│   └── Botão "Atualizar" (refetch de dados)
│
├── KPI Cards (linha 142-171)
│   ├── KPICard "Transações Hoje" (CreditCard icon)
│   ├── KPICard "Fraudes Detectadas" (Shield icon)
│   ├── KPICard "Taxa de Aprovação" (TrendingUp icon)
│   └── KPICard "Latência Média" (Clock icon)
│
├── Charts Row 1 (linha 173-190)
│   ├── SimpleLineChart "Transações por Hora"
│   └── SimpleAreaChart "Latência do Sistema"
│
├── Charts Row 2 (linha 192-248)
│   ├── SimpleBarChart "Fraudes por Canal"
│   ├── SimplePieChart "Distribuição por Canal"
│   └── Card "Alertas Recentes"
│
└── System Status (linha 250-318)
    ├── Card "Status dos Modelos"
    └── Card "Valor Protegido"
```

### 1.2 Componentes Reutilizáveis

| Componente | Localização | Responsabilidade |
|-----------|-----------|------------------|
| `KPICard` | `/components/charts/KPICard.jsx` | Renderizar card com valor, trend, ícone |
| `SimpleLineChart` | `/components/charts/SimpleChart.jsx` | Gráfico de linha com tooltips |
| `SimpleAreaChart` | `/components/charts/SimpleChart.jsx` | Gráfico de área com preenchimento |
| `SimpleBarChart` | `/components/charts/SimpleChart.jsx` | Gráfico de barras com labels |
| `SimplePieChart` | `/components/charts/SimpleChart.jsx` | Gráfico de pizza com percentuais |
| `Card` | `/components/ui/Card.jsx` | Container genérico com espaçamento |
| `Badge` | `/components/ui/Badge.jsx` | Status badges com cores (success, error, warning) |
| `Button` | `/components/ui/Button.jsx` | Botão com variantes e estados |

### 1.3 Fluxo de Dados (API → State → Render)

```
fetchDashboardData() (line 28-62)
│
├─→ Promise.all([
│   ├─ /api/dashboard/kpis          → setKpis()
│   ├─ /api/dashboard/timeseries    → setTimeSeriesData()
│   ├─ /api/dashboard/channels      → setChannelData()
│   ├─ /api/dashboard/recent-alerts → setRecentAlerts()
│   └─ /api/dashboard/model-status  → setModelStatus()
│
└─→ setLastUpdate(new Date())
    └─→ Re-render com novos dados
```

---

## 2. ARQUITETURA FRONT-END + BACK-END

### 2.1 Endpoints da API

| Endpoint | Método | Retorno | Latência Esperada | Cache |
|----------|--------|---------|------------------|-------|
| `/api/dashboard/kpis` | GET | `{transacoes_hoje, fraudes_detectadas, taxa_aprovacao, latencia_media, ...}` | <50ms | 30s TTL |
| `/api/dashboard/timeseries` | GET | `{timeseries: [{time, transactions, latency}, ...]}` | <50ms | 30s TTL |
| `/api/dashboard/channels` | GET | `{channels: [{name, frauds, value}, ...]}` | <50ms | 30s TTL |
| `/api/dashboard/recent-alerts` | GET | `{alerts: [{id, message, severity, timestamp}, ...]}` | <50ms | - |
| `/api/dashboard/model-status` | GET | `{models: [{name, accuracy, status}, ...]}` | <50ms | - |

### 2.2 Estados da Requisição (Frontend)

```javascript
loading (inicial)
  ↓
loading = true (fetching)
  ↓
loading = false (sucesso) → render com dados
  ↓
OU
  ↓
loading = false (erro) → render com fallback
```

### 2.3 Formatadores de Dados

| Função | Entrada | Saída | Usado em |
|--------|---------|-------|----------|
| `formatCurrency(value)` | 172100000 | "R$ 172.1M" | Valor Protegido |
| `formatNumber(value)` | 3115 | "3.115" (pt-BR) | Transações, Fraudes |
| Percentage (KPICard) | 30.3 | "30.3%" | Taxa de Aprovação |
| Decimal (KPICard) | 0.045 | "0.05" | Latência Média |

---

## 3. PLANO DE TESTES FUNCIONAIS

### 3.1 TESTE 1: Header - Botão "Atualizar"

**Objetivo**: Verificar se o botão dispara requisição correta e atualiza todos os dados

**Pré-condições**:
- Dashboard carregado
- Dados anteriores disponíveis na tela
- Conexão com API funcionando

**Passos**:
1. Anotar horário atual do timestamp "Atualizado: HH:MM:SS"
2. Anotar valores dos KPIs (ex: Transações Hoje = 4.467)
3. Clicar no botão "Atualizar"
4. Observar spinner no ícone do botão
5. Aguardar até 2 segundos para completar

**Resultado Esperado**:
- ✅ Ícone do botão exibe spinner/loading durante requisição
- ✅ Timestamp "Atualizado" muda para novo horário
- ✅ Todos os KPIs atualizam com novos valores
- ✅ Gráficos atualizam com novos dados
- ✅ Lista de alertas refaz o fetch
- ✅ Botão volta ao estado normal (sem spinner) após sucesso
- ✅ Latência: <2 segundos para completar atualização

**Cenários de Erro**:
- ❌ API retorna 500: Exibir toast "Erro ao atualizar dashboard"
- ❌ Timeout (>5s): Mostrar mensagem "Tempo limite excedido"
- ❌ Sem internet: Manter dados anteriores, exibir badge "Offline"

---

### 3.2 TESTE 2: KPI Card - "Transações Hoje"

**Objetivo**: Verificar renderização correta do valor, variação e trend

**Pré-condições**:
- API retorna: `{transacoes_hoje: 4467, transacoes_ontem: 4000}`
- Componente KPICard renderizado com props

**Passos**:
1. Observar valor principal do card
2. Observar ícone (CreditCard)
3. Observar badge de variação no canto direito
4. Observar texto abaixo do valor

**Resultado Esperado**:
- ✅ Valor exibido: "4.467" (com separador de milhar pt-BR)
- ✅ Ícone: CreditCard em fundo azul claro
- ✅ Badge verde com seta para cima (↑): "11.7%"
- ✅ Texto: "Aumento em relação ao período anterior"
- ✅ Cor da seta: green-600
- ✅ Fundo do badge: bg-green-50

**Casos Limite**:
| Cenário | Input | Esperado |
|---------|-------|----------|
| Sem variação | today=5000, yesterday=5000 | "0.0%" com ícone Minus, texto "Estável" |
| Diminuição | today=3000, yesterday=4000 | "25.0%" com seta vermelha (↓) |
| Valor muito alto | 1234567890 | "1.234.567.890" formatado |
| Zero | today=0 | "0" sem badge de variação |

---

### 3.3 TESTE 3: KPI Card - "Fraudes Detectadas"

**Objetivo**: Mesmo que teste 3.2, mas com dados de fraude

**Dados de Teste**:
```json
{
  "fraudes_detectadas": 3115,
  "fraudes_ontem": 2800,
  "taxa_reducao": 11.3
}
```

**Resultado Esperado**:
- ✅ Valor: "3.115"
- ✅ Trend: "Aumento" (↑) 11.3% em verde (mais fraudes = tendência de cima para baixo!)
- ✅ Ícone: Shield em fundo azul
- ✅ Badge: bg-green-50 (aumentou detecção = bom)

---

### 3.4 TESTE 4: KPI Card - "Taxa de Aprovação"

**Objetivo**: Verificar formatação percentual correta

**Dados de Teste**:
```json
{
  "taxa_aprovacao": 30.3,
  "taxa_aprovacao_ontem": 30.5,
  "format": "percentage"
}
```

**Resultado Esperado**:
- ✅ Valor: "30.3%" (não "0.303")
- ✅ Trend: "Diminuição" (↓) 0.2% em vermelho (redução de aprovação)
- ✅ Ícone: TrendingUp em fundo azul
- ✅ Precisão: 1 casa decimal

---

### 3.5 TESTE 5: KPI Card - "Latência Média"

**Objetivo**: Verificar formatação decimal e SLA

**Dados de Teste**:
```json
{
  "latencia_media": 0.0,
  "latencia_ontem": 0.05,
  "format": "decimal"
}
```

**Resultado Esperado**:
- ✅ Valor: "0.00ms" (2 casas decimais)
- ✅ Trend: "Diminuição" em vermelho (melhor performance!)
- ✅ Ícone: Clock em fundo azul
- ✅ Alerta visual se > 50ms (SLA breached)

---

### 3.6 TESTE 6: Card - "Valor Protegido"

**Objetivo**: Verificar formatação de moeda grande

**Dados de Teste**:
```json
{
  "valor_protegido_hoje": 0,
  "valor_protegido_ano": 172100000,
  "familias_protegidas": 0
}
```

**Resultado Esperado**:
- ✅ Hoje: "R$ 0" (vermelho/warning se zero)
- ✅ Este ano: "R$ 172.1M" (em verde - success)
- ✅ Famílias: "0" com texto "Famílias protegidas"
- ✅ Formatação: apenas 1 casa decimal para bilhões/milhões

---

### 3.7 TESTE 7: Status dos Modelos

**Objetivo**: Verificar renderização de lista de modelos com health status

**Dados de Teste**:
```json
{
  "models": [
    {"name": "Production Ensemble (RF+GB+LR)", "accuracy": 100, "status": "healthy"},
    {"name": "Decision Tree Backup", "accuracy": 95, "status": "warning"}
  ]
}
```

**Resultado Esperado**:
- ✅ Título: "Status dos Modelos"
- ✅ Lista renderizada com 2 items
- ✅ Modelo 1: "Production Ensemble" + "Precisão: 100%" + Badge "Saudável" (verde)
- ✅ Modelo 2: "Decision Tree Backup" + "Precisão: 95%" + Badge "Atenção" (amarelo)
- ✅ Ícone Activity no header

**Cenário de Erro**:
- ❌ Array vazio: Exibir "Carregando status dos modelos..."

---

### 3.8 TESTE 8: Alertas Recentes - Estado Vazio

**Objetivo**: Verificar comportamento quando sem alertas

**Dados de Teste**:
```json
{
  "alerts": []
}
```

**Resultado Esperado**:
- ✅ Card exibido com título "Alertas Recentes"
- ✅ Conteúdo: "Nenhum alerta recente"
- ✅ Texto centralizado e em cor secundária
- ✅ Sem quebra de layout

---

### 3.9 TESTE 9: Alertas Recentes - Com Dados

**Objetivo**: Verificar rendering de alertas com severidade

**Dados de Teste**:
```json
{
  "alerts": [
    {
      "id": 1,
      "message": "Tentativa de login com IP suspeito",
      "severity": "critico",
      "timestamp": "2025-12-01T08:15:00Z"
    },
    {
      "id": 2,
      "message": "Fraude detectada em PIX",
      "severity": "alto",
      "timestamp": "2025-12-01T08:10:00Z"
    },
    {
      "id": 3,
      "message": "Modelo ML com performance degradada",
      "severity": "medio",
      "timestamp": "2025-12-01T08:05:00Z"
    }
  ]
}
```

**Resultado Esperado**:
- ✅ 3 alertas renderizados
- ✅ Alert 1: Mensagem + Timestamp + Badge "Crítico" (vermelho)
- ✅ Alert 2: Mensagem + Timestamp + Badge "Alto" (vermelho)
- ✅ Alert 3: Mensagem + Timestamp + Badge "Médio" (amarelo)
- ✅ Background cinza suave (bg-neutral-50)
- ✅ Espaçamento vertical consistente (space-y-3)

**Validações**:
- ✅ Timestamp formatado em pt-BR: "01/12/2025 08:15:00"
- ✅ Severity mapeado corretamente: critico/alto→destructive, medio→warning, baixo→secondary
- ✅ Tooltips ao passar mouse com timestamp completo (se implementado)

---

## 4. TESTES DE INTEGRAÇÃO

### 4.1 CENÁRIO: Fluxo Completo de Carregamento

**Objetivo**: Verificar se dados fluem corretamente da API ao render

**Pré-condições**:
- Backend rodando em http://localhost:5000
- Banco de dados com dados reais
- JWT token válido (se autenticação ativada)

**Passos**:

```
1. Carregar Dashboard.jsx
   ↓
2. useEffect() dispara fetchDashboardData()
   ↓
3. setState(loading=true)
   ↓
4. Promise.all([5 endpoints]) iniciado
   ↓
5. Componente renderiza skeleton (KPI cards em pulse)
   ↓
6. Todas as 5 promises resolvem
   ↓
7. setState(loading=false) + todos os states atualizados
   ↓
8. Componente re-render com dados reais
```

**Validações**:

```javascript
// Teste 1: Skeleton loading visível
expect(document.querySelector('.animate-pulse')).toBeVisible()

// Teste 2: Spin de loading visível
expect(document.querySelector('.animate-spin')).toBeVisible()

// Teste 3: Após fetch, skeleton desaparece
await waitFor(() => {
  expect(document.querySelector('.animate-pulse')).not.toBeVisible()
})

// Teste 4: Dados renderizados
expect(screen.getByText('4.467')).toBeInTheDocument() // Transações Hoje
expect(screen.getByText('3.115')).toBeInTheDocument() // Fraudes Detectadas

// Teste 5: Timestamp atualizado
const timestamp = screen.getByText(/Atualizado:/i)
expect(timestamp).toBeVisible()
```

---

### 4.2 CENÁRIO: Erro de API - 500 Internal Server Error

**Objetivo**: Verificar tratamento robusto de falhas

**Setup**:
```bash
Mock /api/dashboard/kpis para retornar 500
```

**Esperado**:
- ✅ catch() em fetchDashboardData() captura erro
- ✅ console.error() loga erro
- ✅ setState(loading=false) mesmo com erro
- ✅ Dados anteriores mantidos (se existirem)
- ✅ UI não quebra (fallback para valores default)
- ✅ Toast/notificação amigável exibida

**Código Esperado**:
```javascript
catch (error) {
  console.error('Erro ao buscar dados do dashboard:', error);
  // Manter dados anteriores
  // Mostrar mensagem de erro
}
finally {
  setLoading(false); // CRÍTICO: sempre executar
}
```

---

### 4.3 CENÁRIO: Dados Incompletos na API

**Objetivo**: Verificar se aplicação lidida com respostas parciais

**Setup**:
```json
GET /api/dashboard/kpis → 200 OK
{
  "data": {
    "transacoes_hoje": 4467
    // Faltam: fraudes_detectadas, taxa_aprovacao, latencia_media, etc.
  }
}
```

**Esperado**:
- ✅ `|| 0` fallback usado em cada campo
- ✅ KPI cards renderizam com "0" para campos ausentes
- ✅ Sem erros de undefined/null
- ✅ Layout não quebra

**Validação**:
```javascript
// No Dashboard.jsx linha 145-146
value={kpis.transacoes_hoje || 0}  // ✅ Fallback
previousValue={kpis.transacoes_ontem || 0}  // ✅ Fallback
```

---

### 4.4 CENÁRIO: Timeout de Rede (>5 segundos)

**Objetivo**: Verificar comportamento em conexão lenta

**Setup**:
```javascript
fetch() com delay simulado de 10 segundos
```

**Esperado**:
- ✅ Spinner de loading permanece visível
- ✅ Botão "Atualizar" disabled (para evitar múltiplas requisições)
- ✅ Após timeout (5s), mensagem de erro exibida
- ✅ Usuário pode clicar "Tentar novamente"

**Validação**:
```javascript
await waitFor(() => {
  expect(screen.getByText(/Tempo limite/i)).toBeInTheDocument()
}, { timeout: 6000 })
```

---

### 4.5 CENÁRIO: Auto-refresh a cada 30 segundos

**Objetivo**: Verificar se setInterval() funciona corretamente

**Setup**:
```javascript
// Dashboard.jsx linha 68
const interval = setInterval(fetchDashboardData, 30000);
```

**Esperado**:
- ✅ 1ª chamada: useEffect() inicia
- ✅ 2ª chamada: setTimeout 30s
- ✅ 3ª chamada: setTimeout 30s
- ✅ ...N chamadas a cada 30s
- ✅ Ao desmontar: clearInterval() executado

**Validação**:
```javascript
// Mock fetch
const fetchSpy = jest.fn()

// Renderizar por 70 segundos
render(<Dashboard />)

await waitFor(() => {
  // Deve ter 2+ chamadas (inicial + 1 auto-refresh)
  expect(fetchSpy).toHaveBeenCalledTimes(2)
}, { timeout: 70000 })

// Limpar
unmountComponent()
expect(fetchSpy).not.toBeCalled() // Sem mais chamadas
```

---

## 5. TESTES DE GRÁFICOS

### 5.1 TESTE: SimpleLineChart - "Transações por Hora"

**Objetivo**: Verificar se dados do gráfico batem com API

**Dados de Teste**:
```json
GET /api/dashboard/timeseries → 200 OK
{
  "data": {
    "timeseries": [
      {"time": "08:00", "transactions": 500, "latency": 45},
      {"time": "09:00", "transactions": 650, "latency": 48},
      {"time": "10:00", "transactions": 720, "latency": 50},
      ...
    ]
  }
}
```

**Validações**:

1. **Eixo X (Time)**:
   - ✅ Labels: "08:00", "09:00", "10:00", ...
   - ✅ Espaçamento uniforme
   - ✅ Formatação: HH:MM

2. **Eixo Y (Transações)**:
   - ✅ Min: 0 (ou valor mínimo)
   - ✅ Max: 720 (ou valor máximo +10%)
   - ✅ Incrementos: automáticos
   - ✅ Labels numéricos

3. **Série de Dados**:
   - ✅ Linha azul/cor definida
   - ✅ Pontos nos valores corretos
   - ✅ Conectados por linha reta

4. **Interatividade**:
   - ✅ Tooltip ao passar mouse: "08:00: 500 transações"
   - ✅ Highlight do ponto ao hover
   - ✅ Sem erro se mudar size (responsivo)

**Cenários Limite**:

| Cenário | Dados | Esperado |
|---------|-------|----------|
| Sem dados | `timeseries: []` | Mensagem "Sem dados para exibir" |
| Um ponto | `[{time: "08:00", transactions: 500}]` | Um ponto renderizado |
| Valores altos | `transactions: 999999` | Escala ajustada, labels legíveis |
| Valores zero | `[{..., transactions: 0}]` | Ponto na base do gráfico |

---

### 5.2 TESTE: SimpleAreaChart - "Latência do Sistema"

**Objetivo**: Verificar gráfico de área com preenchimento

**Dados de Teste**:
```json
{
  "time": ["08:00", "09:00", "10:00"],
  "latency": [45, 48, 50]  // em ms
}
```

**Validações**:

1. **Visual**:
   - ✅ Cor: amber/amarelo (var(--accent-amber-400))
   - ✅ Preenchimento sob a linha (área)
   - ✅ Opacidade: 0.3-0.5 (semitransparente)
   - ✅ Linha superior: mais opaca

2. **Dados Corretos**:
   - ✅ Eixo Y: 45, 48, 50 em ms
   - ✅ Linha sube de 45→48→50 (ascendente)
   - ✅ SLA visual: linha em 50ms, alerta se ultrapassa

3. **Alerta SLA**:
   - ✅ Se qualquer latency > 50ms: linha vermelha
   - ✅ Tooltip aviso: "⚠️ SLA breached: 52ms > 50ms"

---

### 5.3 TESTE: SimpleBarChart - "Fraudes por Canal"

**Objetivo**: Verificar gráfico de barras com canais

**Dados de Teste**:
```json
GET /api/dashboard/channels → 200 OK
{
  "data": {
    "channels": [
      {"name": "PIX", "frauds": 3081},
      {"name": "TED", "frauds": 14},
      {"name": "BOLETO", "frauds": 14},
      {"name": "CREDITO", "frauds": 6}
    ]
  }
}
```

**Validações**:

1. **Eixo X (Canais)**:
   - ✅ Labels: "PIX", "TED", "BOLETO", "CREDITO"
   - ✅ Sem sobreposição (responsivo)
   - ✅ Ícones do canal (se implementado)

2. **Eixo Y (Fraudes)**:
   - ✅ Escala: 0 → ~3081
   - ✅ Incrementos automáticos
   - ✅ Labels legíveis

3. **Barras**:
   - ✅ Cor: error-500 (vermelho para fraude)
   - ✅ Altura proporcional ao valor
   - ✅ PIX: barra muito maior (3081 vs 14)

4. **Interatividade**:
   - ✅ Tooltip: "PIX: 3081 fraudes"
   - ✅ Highlight da barra no hover
   - ✅ Sem overlay se tela pequena

---

### 5.4 TESTE: SimplePieChart - "Distribuição por Canal"

**Objetivo**: Verificar gráfico de pizza com percentuais

**Dados de Teste**:
```json
{
  "channels": [
    {"name": "PIX", "value": 3081},
    {"name": "TED", "value": 14},
    {"name": "BOLETO", "value": 14},
    {"name": "CREDITO", "value": 6}
  ]
}
// Total: 3115 fraudes
// PIX: 98.9%, TED: 0.4%, BOLETO: 0.4%, CREDITO: 0.2%
```

**Validações**:

1. **Fatias**:
   - ✅ PIX: ~99% (fatia gigante)
   - ✅ TED: ~0.4% (fatia minúscula)
   - ✅ BOLETO: ~0.4% (fatia minúscula)
   - ✅ CREDITO: ~0.2% (fatia minúscula)

2. **Cores**:
   - ✅ Cada canal: cor diferente
   - ✅ Degradação visual clara (PIX destaca)

3. **Labels/Legenda**:
   - ✅ Nome do canal
   - ✅ Percentual: "98.9%"
   - ✅ Legenda abaixo ou ao lado

4. **Interatividade**:
   - ✅ Tooltip: "PIX: 3081 (98.9%)"
   - ✅ Highlight da fatia no hover

**Caso Limite**:
- Se todas as fatias muito pequenas (< 1%): Agrupar em "Outros"

---

## 6. TESTES DE ESTADOS E UX

### 6.1 TESTE: Estado "Sistema Online" vs "Offline"

**Objetivo**: Verificar badge de status muda corretamente

**Pré-condições**:
- Badge renderizado em linha 127: `<Badge variant="success">Sistema Online</Badge>`

**Cenários**:

| Estado | Condição | Badge | Cor |
|--------|----------|-------|-----|
| Online | API respondendo | "Sistema Online" | Verde |
| Offline | Sem conexão | "Sistema Offline" | Cinza |
| Degradado | API lenta (>2s) | "Sistema Degradado" | Amarelo |

**Validação**:
```javascript
// Online
await fetchDashboardData() // sucesso
expect(screen.getByText('Sistema Online')).toHaveClass('bg-green-500')

// Offline
mockFetch.mockRejectedValue(new Error('Network error'))
await fetchDashboardData()
expect(screen.getByText('Sistema Offline')).toHaveClass('bg-gray-500')

// Degradado (latência > 2s)
mockFetch.mockImplementation(() => 
  new Promise(resolve => setTimeout(resolve, 2500))
)
await fetchDashboardData()
expect(screen.getByText('Sistema Degradado')).toHaveClass('bg-yellow-500')
```

---

### 6.2 TESTE: Badge "N Algoritmos Ativos"

**Objetivo**: Verificar dinamicidade da contagem de modelos

**Dados de Teste**:
```json
GET /api/dashboard/model-status
{
  "models": [
    {"name": "Random Forest", ...},
    {"name": "Gradient Boosting", ...},
    {"name": "CatBoost", ...}
  ]
}
// models.length = 3
```

**Esperado**:
- ✅ Badge exibe: "3 Algoritmos Ativos"
- ✅ Atualiza se novo modelo adicionado: "4 Algoritmos Ativos"
- ✅ Se vazio: "0 Algoritmos Ativos"

**Validação**:
```javascript
expect(screen.getByText('3 Algoritmos Ativos')).toBeInTheDocument()
```

---

### 6.3 TESTE: Responsividade - Layout em Mobile

**Objetivo**: Verificar se dashboard fica legível em telas pequenas

**Viewports de Teste**:
```
Mobile: 375x667 (iPhone SE)
Tablet: 768x1024 (iPad)
Desktop: 1920x1080 (Full HD)
```

**Validações**:

1. **Mobile (375x667)**:
   - ✅ KPI Cards: 1 coluna (grid-cols-1)
   - ✅ Gráficos: 1 coluna cada
   - ✅ Sem scroll horizontal
   - ✅ Alertas legível (sem truncate abusivo)

2. **Tablet (768x1024)**:
   - ✅ KPI Cards: 2 colunas (md:grid-cols-2)
   - ✅ Gráficos: 2 colunas (lg:grid-cols-2)
   - ✅ Layout equilibrado

3. **Desktop (1920x1080)**:
   - ✅ KPI Cards: 4 colunas (lg:grid-cols-4)
   - ✅ Gráficos: 2 e 3 colunas
   - ✅ Espaçamento ótimo

**Código Esperado** (tailwind):
```jsx
// KPI Cards (linha 142)
<div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">

// Gráficos Row 1 (linha 174)
<div className="grid gap-6 lg:grid-cols-2">

// Gráficos Row 2 (linha 193)
<div className="grid gap-6 lg:grid-cols-3">
```

---

### 6.4 TESTE: Overflow de Textos Longos

**Objetivo**: Verificar se layout quebra com textos longos

**Dados de Teste**:
```javascript
message: "Fraude detectada em transação de cartão crédito internacional com múltiplas tentativas de contato do cliente para contestação"
```

**Esperado**:
- ✅ Texto truncado com "..." (text-ellipsis)
- ✅ Tooltip ao hover mostra texto completo
- ✅ Sem quebra de layout
- ✅ Card mantém tamanho consistente

---

### 6.5 TESTE: Valores Numéricos Extremos

**Objetivo**: Verificar formatação com números muito grandes

**Dados de Teste**:

| Valor | Esperado |
|-------|----------|
| 999999999999 | "R$ 1.0T" (trilhão) |
| 0 | "R$ 0" |
| -100000 | "R$ -100.0K" (negativo) |
| 0.001 | "R$ 0.00" (centavos) |

**Validação**:
```javascript
expect(formatCurrency(999999999999)).toBe('R$ 1.0T')
expect(formatCurrency(0)).toBe('R$ 0')
```

---

## 7. ESTRATÉGIA DE AUTOMAÇÃO

### 7.1 Stack de Testes Recomendado

```
Frontend:
├─ Unit Tests: Vitest + React Testing Library
│  └─ Componentes: KPICard, formatters, badge variants
├─ Integration Tests: Vitest + MSW (Mock Service Worker)
│  └─ Dashboard.jsx com API mockada
└─ E2E Tests: Playwright ou Cypress
   └─ Fluxo completo com real backend

Backend:
├─ Unit Tests: pytest
│  └─ Lógica de KPI, cálculos, formatação
├─ Integration Tests: pytest + PostgreSQL container
│  └─ Endpoints reais + DB
└─ Performance Tests: locust
   └─ <50ms latency validation
```

### 7.2 Exemplo: Teste de Unidade (Vitest)

```javascript
// src/components/charts/__tests__/KPICard.test.jsx

import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { KPICard } from '../KPICard'

describe('KPICard', () => {
  it('renderiza valor formatado corretamente', () => {
    render(
      <KPICard
        title="Transações Hoje"
        value={4467}
        previousValue={4000}
        format="number"
      />
    )
    
    expect(screen.getByText('4.467')).toBeInTheDocument()
    expect(screen.getByText('11.7%')).toBeInTheDocument()
  })

  it('exibe trend correto para aumento', () => {
    const { container } = render(
      <KPICard value={500} previousValue={400} />
    )
    
    // Check if TrendingUp icon is rendered
    expect(container.querySelector('[data-trend="up"]')).toBeInTheDocument()
  })

  it('formatação percentual correta', () => {
    render(
      <KPICard
        title="Taxa"
        value={30.3}
        format="percentage"
      />
    )
    
    expect(screen.getByText('30.3%')).toBeInTheDocument()
  })

  it('valor zero sem badge de variação', () => {
    render(
      <KPICard
        title="Proteção"
        value={0}
      />
    )
    
    expect(screen.getByText('0')).toBeInTheDocument()
    expect(screen.queryByText('%')).not.toBeInTheDocument()
  })
})
```

### 7.3 Exemplo: Teste de Integração (Vitest + MSW)

```javascript
// src/pages/__tests__/Dashboard.integration.test.jsx

import { describe, it, expect, beforeAll } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'
import { Dashboard } from '../Dashboard'

const mockData = {
  kpis: {
    transacoes_hoje: 4467,
    fraudes_detectadas: 3115,
    taxa_aprovacao: 30.3,
    latencia_media: 0.05
  },
  timeseries: [
    { time: '08:00', transactions: 500, latency: 45 },
    { time: '09:00', transactions: 650, latency: 48 }
  ],
  channels: [
    { name: 'PIX', frauds: 3081, value: 98.9 },
    { name: 'TED', frauds: 14, value: 0.4 }
  ],
  alerts: [
    {
      id: 1,
      message: 'Fraude detectada',
      severity: 'alto',
      timestamp: '2025-12-01T08:00:00Z'
    }
  ],
  models: [
    { name: 'Random Forest', accuracy: 95, status: 'healthy' }
  ]
}

const server = setupServer(
  http.get('/api/dashboard/kpis', () => HttpResponse.json({ data: mockData.kpis })),
  http.get('/api/dashboard/timeseries', () => HttpResponse.json({ data: { timeseries: mockData.timeseries } })),
  http.get('/api/dashboard/channels', () => HttpResponse.json({ data: { channels: mockData.channels } })),
  http.get('/api/dashboard/recent-alerts', () => HttpResponse.json({ alerts: mockData.alerts })),
  http.get('/api/dashboard/model-status', () => HttpResponse.json({ models: mockData.models }))
)

beforeAll(() => server.listen())

describe('Dashboard Integration', () => {
  it('carrega e renderiza todos os dados', async () => {
    render(<Dashboard />)
    
    // Loader visível inicialmente
    expect(screen.getByText('Carregando dados do dashboard...')).toBeInTheDocument()
    
    // Aguardar dados
    await waitFor(() => {
      expect(screen.queryByText('Carregando dados do dashboard...')).not.toBeInTheDocument()
    })
    
    // Validar KPIs
    expect(screen.getByText('4.467')).toBeInTheDocument() // Transações
    expect(screen.getByText('3.115')).toBeInTheDocument() // Fraudes
    expect(screen.getByText('30.3%')).toBeInTheDocument() // Taxa
    
    // Validar timestamp
    expect(screen.getByText(/Atualizado:/)).toBeInTheDocument()
  })

  it('exibe alertas recentes', async () => {
    render(<Dashboard />)
    
    await waitFor(() => {
      expect(screen.getByText('Fraude detectada')).toBeInTheDocument()
      expect(screen.getByText('Alto')).toBeInTheDocument() // Severity badge
    })
  })

  it('trata erro de API corretamente', async () => {
    server.use(
      http.get('/api/dashboard/kpis', () => new HttpResponse(null, { status: 500 }))
    )
    
    render(<Dashboard />)
    
    await waitFor(() => {
      expect(screen.getByText(/Erro ao buscar/i)).toBeInTheDocument()
    })
  })
})
```

### 7.4 Exemplo: Teste E2E (Playwright)

```javascript
// e2e/dashboard.spec.js

import { test, expect } from '@playwright/test'

test.describe('Dashboard Executivo', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:5000')
    // Aguardar dados carregarem
    await page.waitForLoadState('networkidle')
  })

  test('renderiza header com timestamp', async ({ page }) => {
    const header = page.locator('text=Dashboard Executivo')
    await expect(header).toBeVisible()

    const timestamp = page.locator('text=Atualizado:')
    await expect(timestamp).toBeVisible()

    const updateBtn = page.locator('button:has-text("Atualizar")')
    await expect(updateBtn).toBeEnabled()
  })

  test('botão atualizar refaz fetch', async ({ page }) => {
    const initialTime = await page.locator('text=Atualizado:').textContent()
    
    // Aguardar 1 segundo
    await page.waitForTimeout(1000)
    
    // Clicar atualizar
    await page.locator('button:has-text("Atualizar")').click()
    
    // Aguardar atualização
    await page.waitForLoadState('networkidle')
    
    const newTime = await page.locator('text=Atualizado:').textContent()
    expect(newTime).not.toEqual(initialTime)
  })

  test('gráficos renderizam com dados', async ({ page }) => {
    const transChart = page.locator('text=Transações por Hora')
    await expect(transChart).toBeVisible()
    
    const latencyChart = page.locator('text=Latência do Sistema')
    await expect(latencyChart).toBeVisible()
    
    // Verifica se SVG (gráfico) existe
    const chartSVG = page.locator('svg').first()
    await expect(chartSVG).toBeVisible()
  })

  test('KPI cards exibem valores formatados', async ({ page }) => {
    // Transações Hoje
    const txns = page.locator('text=Transações Hoje').locator('..').locator('text=/\\d{1,3}(\\.\\d{3})?/')
    await expect(txns).toBeVisible()
    
    // Taxa de Aprovação em %
    const rate = page.locator('text=Taxa de Aprovação').locator('..').locator('text=/%/')
    await expect(rate).toBeVisible()
  })

  test('alertas renderizam com severidade', async ({ page }) => {
    const alertsCard = page.locator('text=Alertas Recentes')
    await expect(alertsCard).toBeVisible()
    
    // Se houver alertas
    const alerts = page.locator('[class*="bg-neutral-50"]')
    if (await alerts.count() > 0) {
      const firstAlert = alerts.first()
      const badge = firstAlert.locator('[class*="badge"]')
      await expect(badge).toBeVisible()
    }
  })

  test('responsivo em mobile', async ({ page }) => {
    // Redimensionar para mobile
    await page.setViewportSize({ width: 375, height: 667 })
    
    // Elementos ainda visíveis
    expect(page.locator('text=Dashboard Executivo')).toBeVisible()
    expect(page.locator('text=Transações Hoje')).toBeVisible()
    
    // Sem scroll horizontal
    const bodyWidth = await page.locator('body').evaluate(el => el.offsetWidth)
    const windowWidth = await page.evaluate(() => window.innerWidth)
    expect(bodyWidth).toBeLessThanOrEqual(windowWidth)
  })
})
```

### 7.5 Exemplo: Teste de Performance (Backend - pytest)

```python
# backend/tests/test_dashboard_performance.py

import pytest
import time
from api.production_api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

class TestDashboardLatency:
    """Validar SLA < 50ms para endpoints do dashboard"""
    
    def test_kpis_endpoint_latency(self, client):
        """GET /api/dashboard/kpis deve responder em <50ms"""
        start = time.time()
        response = client.get('/api/dashboard/kpis')
        elapsed = (time.time() - start) * 1000  # ms
        
        assert response.status_code == 200
        assert elapsed < 50, f"Latency {elapsed}ms exceeds SLA (50ms)"
    
    def test_timeseries_endpoint_latency(self, client):
        """GET /api/dashboard/timeseries deve responder em <50ms"""
        start = time.time()
        response = client.get('/api/dashboard/timeseries')
        elapsed = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert elapsed < 50
    
    def test_channels_endpoint_latency(self, client):
        """GET /api/dashboard/channels deve responder em <50ms"""
        start = time.time()
        response = client.get('/api/dashboard/channels')
        elapsed = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert elapsed < 50
    
    def test_all_endpoints_parallel(self, client):
        """Simular 5 requests paralelos (como frontend faz)"""
        import concurrent.futures
        
        endpoints = [
            '/api/dashboard/kpis',
            '/api/dashboard/timeseries',
            '/api/dashboard/channels',
            '/api/dashboard/recent-alerts',
            '/api/dashboard/model-status'
        ]
        
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(client.get, ep) for ep in endpoints]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        elapsed = (time.time() - start) * 1000
        
        # Todos 200 OK
        assert all(r.status_code == 200 for r in results)
        
        # Tempo total < 200ms (50ms + margem para paralelo)
        assert elapsed < 200
```

---

## 8. CHECKLIST FINAL

### ✅ Pré-requisitos
- [ ] Backend rodando em http://localhost:5000
- [ ] Database PostgreSQL conectado e populado com dados
- [ ] Frontend built e servido em /api/
- [ ] JWT token válido (se autenticação ativada)
- [ ] Browser (Chrome, Firefox, Safari) atualizado

### ✅ Renderização Visual (Manual)
- [ ] Dashboard carrega sem erros
- [ ] Skeleton loading visível inicialmente
- [ ] Todos 4 KPI cards renderizam
- [ ] 2 gráficos superiores visíveis
- [ ] 3 cards inferiores visíveis (Fraudes, Distribuição, Alertas)
- [ ] 2 cards de Status (Modelos, Valor Protegido)
- [ ] Header com timestamp "Atualizado: HH:MM:SS"
- [ ] Badge "Sistema Online" verde
- [ ] Badge "N Algoritmos Ativos" correto

### ✅ KPI Cards
- [ ] **Transações Hoje**: valor formatado, trend up/down/neutral, percentual
- [ ] **Fraudes Detectadas**: valor formatado, trend, percentual
- [ ] **Taxa de Aprovação**: percentual formatado, trend
- [ ] **Latência Média**: decimal formatado, trend
- [ ] Todos 4 com ícones corretos
- [ ] Fallback para 0 se dados ausentes

### ✅ Gráficos
- [ ] **Transações por Hora**: linha com 24+ pontos, eixos corretos
- [ ] **Latência do Sistema**: área com preenchimento, cor amber
- [ ] **Fraudes por Canal**: barras PIX>>TED=BOLETO, cores corretas
- [ ] **Distribuição por Canal**: pizza com fatias proporcionais, legenda
- [ ] Todos tooltips funcionam ao hover
- [ ] Responsivos em mobile/tablet

### ✅ Alertas
- [ ] Estado vazio: "Nenhum alerta recente" visível
- [ ] Com dados: lista renderizada corretamente
- [ ] Severity badges com cores corretas (critico=vermelho, alto=vermelho, medio=amarelo, baixo=cinza)
- [ ] Timestamp formatado em pt-BR

### ✅ Status dos Modelos
- [ ] Lista renderizada com modelos
- [ ] Nomes dos modelos corretos
- [ ] Accuracy % exibida
- [ ] Badge health (Saudável=verde, Atenção=amarelo)
- [ ] Ícone Activity visível

### ✅ Valor Protegido
- [ ] "Valor protegido hoje" em R$ formatado
- [ ] "Este ano" em R$ formatado (M/B/K)
- [ ] "Famílias protegidas" em número formatado
- [ ] Cores corretas (verde para sucesso)

### ✅ Interatividade
- [ ] Botão "Atualizar" funciona
- [ ] Spinner visível durante fetch
- [ ] Timestamp atualiza após refresh
- [ ] Todos os KPIs/gráficos atualizam
- [ ] Botão desabilitado durante loading

### ✅ Responsividade
- [ ] Mobile (375px): 1 coluna KPIs, layout vertical
- [ ] Tablet (768px): 2 colunas KPIs
- [ ] Desktop (1920px): 4 colunas KPIs
- [ ] Sem scroll horizontal em nenhuma resolução
- [ ] Gráficos legíveis em todas as resoluções

### ✅ Tratamento de Erros
- [ ] Erro 500 na API: fallback para dados anteriores
- [ ] Timeout (5s+): mensagem de erro amigável
- [ ] Sem internet: badge "Sistema Offline"
- [ ] Dados parciais: sem erros de undefined
- [ ] Never quebra layout

### ✅ Performance
- [ ] 1ª requisição: <2 segundos (com cache hit)
- [ ] 2ª requisição: <70ms (com cache)
- [ ] Auto-refresh a cada 30s funciona
- [ ] Sem memory leaks (cleanup em unmount)
- [ ] Skeleton animation suave

### ✅ Testes Automatizados
- [ ] Unit tests: KPICard formatação ✅
- [ ] Unit tests: Trends (up/down/neutral) ✅
- [ ] Integration tests: fetchDashboardData() ✅
- [ ] Integration tests: Estado de erro ✅
- [ ] E2E tests: Fluxo completo ✅
- [ ] Performance tests: Latency <50ms ✅
- [ ] Coverage: >80% para componentes críticos

### ✅ Compliance & Security
- [ ] Sem logs de dados sensíveis
- [ ] CORS configurado para frontend
- [ ] Rate limiting ativo (se presente)
- [ ] JWT validation (se presente)
- [ ] XSS prevention (sanitização)

### ✅ Documentação
- [ ] README atualizado com endpoints
- [ ] Swagger/OpenAPI documentado
- [ ] Exemplos de requests/responses
- [ ] Casos de erro documentados
- [ ] SLA (<50ms) documentado

### ✅ CI/CD & Deployment
- [ ] Tests passando em CI pipeline
- [ ] Build sem warnings
- [ ] Frontend otimizado (bundle size <500KB)
- [ ] Backend otimizado (cold start <2s)
- [ ] Logs estruturados (estrutlog) configurados

---

## 📌 PRÓXIMAS AÇÕES

1. **Execute Teste Manual Inicial**: Abra http://localhost:5000/dashboard e valide renderização
2. **Configure MSW para Mocking**: `npm install msw`
3. **Execute Testes Unitários**: `npm run test`
4. **Execute Testes E2E**: `npx playwright install && npm run test:e2e`
5. **Validar Performance**: `npm run test:perf` ou usar DevTools
6. **Corrigir Issues**: Priorizar por severity (critical > major > minor)
7. **Documenter Results**: Registrar screenshots/vídeos de testes

---

**Documento preparado em**: Dezembro 01, 2025  
**Stack**: React 18 + Vite + Vitest + Playwright + Python/pytest  
**SLA Alvo**: <50ms por requisição  
**Cobertura Mínima**: 80% (componentes críticos)
