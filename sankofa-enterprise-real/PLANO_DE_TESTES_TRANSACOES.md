# 🧪 PLANO DE TESTES - TELA DE TRANSAÇÕES
## Sankofa Enterprise Pro - Lista e Busca de Transações

**Data**: Dezembro 01, 2025  
**Versão**: 1.0  
**Status**: Pronto para Execução  
**Arquivo Frontend**: `src/pages/Transactions.jsx` (1108 linhas)

---

## 📋 ÍNDICE
1. [Mapeamento de Componentes](#1-mapeamento-de-componentes)
2. [Arquitetura Front-End + Back-End](#2-arquitetura-front-end--back-end)
3. [Testes Funcionais - Header e Ações](#3-testes-funcionais---header-e-ações)
4. [Testes Funcionais - Filtros](#4-testes-funcionais---filtros)
5. [Testes Funcionais - Tabela](#5-testes-funcionais---tabela)
6. [Testes do Modal de Detalhes](#6-testes-do-modal-de-detalhes)
7. [Testes de Integração](#7-testes-de-integração)
8. [Estratégia de Automação](#8-estratégia-de-automação)
9. [Checklist Final](#9-checklist-final)

---

## 1. MAPEAMENTO DE COMPONENTES

### 1.1 Estrutura React

```
src/pages/Transactions.jsx (MAIN COMPONENT - 1108 linhas)
├── Header (linhas 298-321)
│   ├── Título: "Transações"
│   ├── Subtítulo: "Lista e busca de transações processadas em tempo real"
│   ├── Botão "Exportar" (CSV)
│   └── Botão "Atualizar"
│
├── Card de Filtros (linhas 323-403)
│   ├── Buscar por: ID, CPF, cidade...
│   ├── Status: Dropdown (Todos, Aprovada, Rejeitada, Pendente, Em Revisão)
│   ├── Tipo: Dropdown (Todos, PIX, Crédito, Débito, TED, DOC)
│   └── Período: Menu suspenso (1h, 24h, 7d, 30d, Todo período)
│
├── Resultados Summary (linhas 405-429)
│   ├── Texto: "Mostrando X de Y transações"
│   └── Dropdown de Ordenação (6 opções)
│
├── Tabela de Transações (linhas 431-569)
│   ├── Colunas: ID, Valor, Tipo, Canal, Localização, CPF, Data/Hora, Status, Risco, Ações
│   ├── Linhas dinâmicas com handleViewDetails() e handleToggleActionsMenu()
│   └── Menu de Ações (Aprovar, Rejeitar, Revisão, Marcar Suspeito, Investigar)
│
└── Modal de Detalhes (linhas 571-900+)
    ├── Status + Risco Score
    ├── Informações Principais (Valor, Tipo, Canal, Data)
    ├── Dados do Cliente (CPF, Localização)
    ├── Análise de Risco (Barra de progresso, percentual)
    └── Explicação Detalhada (Seções 1-4 ultra-didáticas)
```

### 1.2 Estados React (linhas 36-54)

| Estado | Tipo | Propósito |
|--------|------|----------|
| `transactions` | Array | Transações do backend |
| `loading` | Boolean | Estado de carregamento |
| `searchQuery` | String | Busca livre |
| `statusFilter` | String | Filtro de status |
| `typeFilter` | String | Filtro de tipo |
| `sortField` | String | Campo para ordenação |
| `sortDirection` | String | 'asc' ou 'desc' |
| `currentPage` | Number | Página atual (paginação) |
| `totalPages` | Number | Total de páginas |
| `totalTransactions` | Number | Total de registros |
| `selectedTransaction` | Object | Transação selecionada para modal |
| `showDetailsModal` | Boolean | Modal aberto/fechado |
| `showActionsMenu` | String/null | ID da transação com menu aberto |
| `periodFilter` | String | '1h', '24h', '7d', '30d', 'all' |
| `exportLoading` | Boolean | Exportação em progresso |
| `explanation` | Object | Explicação detalhada da transação |

### 1.3 Funções Principais

| Função | Linhas | Responsabilidade |
|--------|--------|-----------------|
| `loadTransactions()` | 68-97 | Fetch de /api/transactions com filtros |
| `handleSort()` | 123-130 | Alternar ordenação (asc/desc) |
| `handleRefresh()` | 132-134 | Recarregar dados |
| `handleExport()` | 136-171 | Exportar CSV com dados filtrados |
| `handleViewDetails()` | 178-186 | Abrir modal com detalhes |
| `loadExplanation()` | 188-206 | Fetch /api/explainability/explain |
| `handleAction()` | 218-271 | Executar ação (aprovar, rejeitar, etc.) |
| `formatCurrency()` | 273-278 | Formatar valores em R$ |

---

## 2. ARQUITETURA FRONT-END + BACK-END

### 2.1 Endpoints da API

| Endpoint | Método | Params | Retorno | Latência |
|----------|--------|--------|---------|----------|
| `/api/transactions` | GET | `page`, `limit`, `search`, `status`, `type`, `period` | `{success, data: [], stats: {total}}` | <50ms |
| `/api/transactions/{id}/approve` | POST | - | `{success, data}` | <100ms |
| `/api/transactions/{id}/reject` | POST | - | `{success, data}` | <100ms |
| `/api/transactions/{id}/review` | POST | - | `{success, data}` | <100ms |
| `/api/transactions/{id}/flag` | POST | `{flagged: true}` | `{success, data}` | <100ms |
| `/api/investigations` | POST | `{transaction_id, priority}` | `{success, data}` | <100ms |
| `/api/explainability/explain` | POST | `{transaction_id}` | `{success, explanation}` | <500ms |

### 2.2 Fluxo de Carregamento

```javascript
useEffect (linhas 64-66) →
  dispara loadTransactions() ao mudar: currentPage, searchQuery, statusFilter, typeFilter, periodFilter
  ↓
  setLoading(true) → renderiza skeleton
  ↓
  fetch('/api/transactions?{params}')
  ↓
  setTransactions(data.data)
  setTotalPages(Math.ceil(data.stats.total / 50))
  setTotalTransactions(data.stats.total)
  ↓
  setLoading(false) → renderiza tabela real
```

### 2.3 Mapeamento de Filtros → Query Params

| Filtro Frontend | Query Param | Valores Possíveis |
|-----------------|-------------|------------------|
| `searchQuery` | `search` | String (ID, CPF, cidade) |
| `statusFilter` | `status` | TODOS, APROVADA, REJEITADA, PENDENTE, EM_REVISAO |
| `typeFilter` | `type` | TODOS, PIX, CREDITO, DEBITO, TED, DOC |
| `periodFilter` | `period` | 1h, 24h, 7d, 30d, all |
| `currentPage` | `page` | Integer ≥ 1 |
| Fixo | `limit` | 50 (hardcoded) |

---

## 3. TESTES FUNCIONAIS - HEADER E AÇÕES

### 3.1 TESTE: Botão "Atualizar"

**Objetivo**: Verificar se refetch funciona e atualiza todos os dados

**Pré-condições**:
- Transações carregadas
- Backend respondendo corretamente

**Passos**:
1. Anotar número total: "Mostrando X de Y transações"
2. Clicar "Atualizar"
3. Observar spinner no botão
4. Aguardar até 2 segundos

**Resultado Esperado**:
- ✅ Spinner visível durante fetch
- ✅ Tabela atualiza com possíveis novos dados
- ✅ Contador "Mostrando X de Y" pode mudar
- ✅ Botão volta ao normal
- ✅ Sem erros no console

**Tratamento de Erro**:
- ❌ API 500: Exibir toast de erro
- ❌ Timeout: Mensagem "Tempo limite excedido"

---

### 3.2 TESTE: Botão "Exportar"

**Objetivo**: Verificar geração e download de arquivo CSV

**Pré-condições**:
- Transações renderizadas
- Filtros aplicados ou não

**Passos**:
1. Clique em "Exportar"
2. Aguardar download
3. Abrir arquivo em editor de texto
4. Verificar formato e conteúdo

**Resultado Esperado**:

✅ **Arquivo gerado com nome**: `transacoes_YYYY-MM-DD.csv`
✅ **Cabeçalho**: `ID;Valor;Tipo;Canal;Localização;CPF;Data/Hora;Status;Score de Risco`
✅ **Delimitador**: Ponto-e-vírgula (;)
✅ **Encoding**: UTF-8 com BOM (\ufeff)
✅ **Linhas**: Uma por transação
✅ **Dados**: Correspondem aos da tabela

**Exemplo de conteúdo**:
```csv
﻿ID;Valor;Tipo;Canal;Localização;CPF;Data/Hora;Status;Score de Risco
TXN17644513896743000;15000.00;PIX;PIX;São Paulo;***.***.***-**;2025-11-30 14:30:36;APROVADA;0.05
TXN17644513896743001;8500.50;CREDITO;CREDITO;Rio de Janeiro;***.***.***-**;2025-11-30 15:45:22;REJEITADA;0.95
```

**Cenários de Erro**:
- ❌ Nenhuma transação: Botão desabilitado
- ❌ Exportação falha: Alert "Erro ao exportar arquivo"

**Validação de Filtros em CSV**:
- Se Status="APROVADA" selecionado, CSV deve conter APENAS APROVADAS
- Se Período="24h", CSV deve conter APENAS últimas 24h
- Combinação de filtros deve ser respeitada

---

## 4. TESTES FUNCIONAIS - FILTROS

### 4.1 TESTE: Busca Livre (Search)

**Objetivo**: Verificar busca por ID, CPF, cidade

**Dados de Teste**:
```
Transaction ID: TXN17644513896743000
CPF: 123.456.789-00 (ou mascarado)
Localização: "São Paulo"
```

**Casos de Teste**:

#### Caso 1: Busca Exata por ID
- **Input**: `TXN17644513896743000`
- **Esperado**: 1 resultado exato
- **Validação**: Tabela mostra apenas esse ID

#### Caso 2: Busca Parcial
- **Input**: `TXN176`
- **Esperado**: Todos os IDs começados com TXN176
- **Validação**: Multiple resultados com mesmo prefixo

#### Caso 3: Busca por CPF
- **Input**: `123.456` ou `123456`
- **Esperado**: Transações desse CPF
- **Validação**: Coluna CPF contém valor

#### Caso 4: Busca por Localização
- **Input**: `São Paulo`
- **Esperado**: Transações de São Paulo
- **Validação**: Coluna Localização mostra "São Paulo"

#### Caso 5: Sem Resultados
- **Input**: `XXXXXX999999999` (inexistente)
- **Esperado**: "Nenhuma transação encontrada com os filtros aplicados"
- **Validação**: Tabela vazia, mensagem visível

#### Caso 6: Busca com Caracteres Especiais
- **Input**: `!@#$%&*()`
- **Esperado**: Sem resultados ou tratamento seguro
- **Validação**: Sem erro no console, sem XSS

---

### 4.2 TESTE: Filtro Status

**Objetivo**: Verificar aplicação do filtro de status

**Dados de Teste**:
```json
Transações com diferentes status:
- 100 com status APROVADA
- 50 com status REJEITADA
- 30 com status PENDENTE
- 20 com status EM_REVISAO
```

**Casos de Teste**:

| Opção | Esperado | Validação |
|-------|----------|-----------|
| "Todos" | 200 transações | Todas as status na tabela |
| "Aprovada" | ~100 transações | Apenas com `status === "APROVADA"` |
| "Rejeitada" | ~50 transações | Apenas com `status === "REJEITADA"` |
| "Pendente" | ~30 transações | Apenas com `status === "PENDENTE"` |
| "Em Revisão" | ~20 transações | Apenas com `status === "EM_REVISAO"` |

**Validação Visual**:
- ✅ Badge de status colorida corretamente
- ✅ Cores consistentes com design system
- ✅ Sem mistura de status

---

### 4.3 TESTE: Filtro Tipo

**Objetivo**: Verificar filtro por tipo de transação

**Dados de Teste**:
```json
Distribuição de tipos:
- 1000 PIX
- 200 CREDITO
- 150 DEBITO
- 50 TED
- 30 DOC
```

**Casos de Teste**:

| Opção | Badge Esperado | Count Esperado |
|-------|---|---|
| "Todos" | Misturados | 1430 |
| "PIX" | "PIX" (azul) | 1000 |
| "Crédito" | "CREDITO" (verde) | 200 |
| "Débito" | "DEBITO" (laranja) | 150 |
| "TED" | "TED" (roxo) | 50 |
| "DOC" | "DOC" (cinza) | 30 |

**Validação**:
- ✅ Badge renderizada com tipo correto
- ✅ Count atualiza após filtro
- ✅ Sem mistura de tipos

---

### 4.4 TESTE: Filtro Período

**Objetivo**: Verificar filtro temporal com 5 opções

**Dados de Teste**:
```
Agora: 2025-12-01 10:00:00 UTC
Transações espalhadas nos últimos 90 dias
```

**Períodos Esperados**:

| Opção | Range | Esperado |
|-------|-------|----------|
| "Última hora" | Agora - 1h | Transações desde 09:00 |
| "Últimas 24h" | Agora - 24h | Transações desde ontem 10:00 |
| "Últimos 7 dias" | Agora - 7d | Transações desde 24-Nov |
| "Últimos 30 dias" | Agora - 30d | Transações desde 01-Nov |
| "Todo período" | Sem limite | Todas as transações |

**Validação**:
- ✅ Coluna Data/Hora respeita range
- ✅ Sem transações fora do período selecionado
- ✅ API recebe parâmetro `period` correto

**Teste de Limite de Período**:
```javascript
// Se Período="Última hora" selecionado
// Todas as transações devem ter:
const hourAgo = new Date(Date.now() - 3600000)
assert(transaction.timestamp >= hourAgo)
```

---

### 4.5 TESTE: Combinação de Filtros

**Objetivo**: Verificar se múltiplos filtros funcionam juntos

**Cenário 1**: Status=REJEITADA + Tipo=PIX + Período=24h
```
Esperado: Transações rejeitadas via PIX nas últimas 24h
Validação:
  - status === "REJEITADA"
  - tipo === "PIX"
  - timestamp >= (agora - 24h)
```

**Cenário 2**: Busca="São Paulo" + Status=APROVADA + Tipo=CREDITO
```
Esperado: Créditos aprovados de São Paulo
Validação:
  - localizacao contém "São Paulo"
  - status === "APROVADA"
  - tipo === "CREDITO"
```

**Cenário 3**: Busca="CPF" + Período=7d + Status=TODOS
```
Esperado: CPF específico nos últimos 7 dias
Validação:
  - cpf contém valor buscado
  - timestamp >= (agora - 7d)
```

**Validação Técnica** (linhas 71-78):
```javascript
// Verificar que params corretos são enviados
const params = new URLSearchParams({
  page: 1,
  limit: 50,
  search: "São Paulo",
  status: "REJEITADA",
  type: "PIX",
  period: "24h"
});
// URL final: /api/transactions?page=1&limit=50&search=São Paulo&status=REJEITADA&type=PIX&period=24h
```

---

## 5. TESTES FUNCIONAIS - TABELA

### 5.1 TESTE: Contagem e Summary

**Objetivo**: Verificar se texto "Mostrando X de Y" está correto

**Pré-condições**:
- API retorna 250 transações totais
- Página 1 mostra 50 por página (hardcoded)

**Validação**:

| Página | Display Esperado | Validação |
|--------|---|---|
| 1 | "Mostrando 50 de 250 transações" | 50 linhas, total=250 |
| 2 | "Mostrando 50 de 250 transações" | 50 linhas, total=250 (mesmo) |
| 5 | "Mostrando 50 de 250 transações" | 50 linhas, total=250 (mesmo) |
| 6 | "Mostrando 0 de 250 transações" | Vazio (fora do range) |

**Fórmula** (linhas 405-409):
```javascript
Mostrando {filteredTransactions.length} de {totalTransactions} transações
// filteredTransactions.length = dados atuais da página (50)
// totalTransactions = data.stats.total da API
```

---

### 5.2 TESTE: Ordenação

**Objetivo**: Verificar 6 opções de sort

**Opções**:
```javascript
[
  { value: "timestamp-desc", label: "Mais recentes" },      // Linhas 421
  { value: "timestamp-asc", label: "Mais antigas" },        // Linhas 422
  { value: "valor-desc", label: "Maior valor" },            // Linhas 423
  { value: "valor-asc", label: "Menor valor" },             // Linhas 424
  { value: "risk_score-desc", label: "Maior risco" },       // Linhas 425
  { value: "risk_score-asc", label: "Menor risco" }         // Linhas 426
]
```

**Validação para "Mais Recentes"**:
```javascript
// Dados devem estar em ordem DESC por timestamp
assert(transações[0].timestamp > transações[1].timestamp)
assert(transações[1].timestamp > transações[2].timestamp)
```

**Validação para "Maior Valor"**:
```javascript
// Dados devem estar em ordem DESC por valor
assert(transações[0].valor > transações[1].valor)
assert(transações[1].valor > transações[2].valor)
```

---

### 5.3 TESTE: Formatação de Dados na Tabela

**Objetivo**: Verificar formatação de cada coluna

#### Coluna: Valor (línea 459)
```javascript
formatCurrency(transaction.valor)
// Formato: Intl.NumberFormat('pt-BR', {style: 'currency', currency: 'BRL'})
```

**Casos**:
| Valor API | Esperado | Formato |
|-----------|----------|---------|
| 15000.00 | R$ 15.000,00 | pt-BR currency |
| 150.50 | R$ 150,50 | Vírgula decimal |
| 0 | R$ 0,00 | Zero |
| 999999.99 | R$ 999.999,99 | Separador milhar |

#### Coluna: Tipo (linhas 462-464)
```javascript
<Badge variant="default" size="sm">
  {transaction.tipo}  // Renderiza como está (PIX, CREDITO, etc.)
</Badge>
```

**Esperado**: Badge com texto correto, sem transformação

#### Coluna: Canal (linhas 466-468)
```javascript
{transaction.canal.toUpperCase()}  // Converte para maiúsculas
```

**Casos**:
| Entrada | Esperado |
|---------|----------|
| "pix" | "PIX" |
| "PIX" | "PIX" |
| "credito" | "CREDITO" |

#### Coluna: Localização (linhas 469-470)
```javascript
{transaction.localizacao}  // Renderiza como está
```

**Esperado**: Cidade/Estado completo sem transformação

#### Coluna: CPF (linhas 472-474)
```javascript
{transaction.cpf}  // Pode estar mascarado ou em branco
```

**LGPD Compliance**:
- ❌ CPF nunca exibido completo (ex: mascarado como **.***.***-**)
- ❌ Se vazio: exibir "***.***.***-**" ou traço

#### Coluna: Data/Hora (linhas 475-476)
```javascript
{transaction.data_hora}  // Já vem formatada da API
```

**Esperado**:
- Exemplo: "Sun, 30 Nov 2025 14:30:36 GMT"
- Formato: `day, DD Mon YYYY HH:MM:SS GMT`
- Timezone: UTC/GMT

#### Coluna: Status (linhas 478-479)
```javascript
<TransactionStatusBadge status={transaction.status} size="sm" />
```

**Cores Esperadas**:
| Status | Cor | Ícone |
|--------|-----|-------|
| APROVADA | Verde | CheckCircle |
| REJEITADA | Vermelho | XCircle |
| PENDENTE | Amarelo | Clock |
| EM_REVISAO | Amarelo | AlertTriangle |

#### Coluna: Risco (linhas 481-482)
```javascript
<RiskScoreBadge score={transaction.fraud_score} size="sm" />
```

**Cores Esperadas**:
| Score | Cor | Label |
|-------|-----|-------|
| 0.0 - 0.3 | Verde | "Baixo Risco" |
| 0.4 - 0.7 | Amarelo | "Médio Risco" |
| 0.7 - 1.0 | Vermelho | "Alto Risco" |

---

### 5.4 TESTE: Ações da Linha

**Objetivo**: Verificar ícone olho e menu de ações

#### Sub-teste: Ícone Olho (Detalhes)

**Passos** (linhas 486-493):
1. Clicar no ícone olho de uma transação
2. Observar comportamento

**Resultado Esperado**:
- ✅ Modal de detalhes abre
- ✅ `selectedTransaction` = dados da linha
- ✅ `showDetailsModal` = true
- ✅ Função `loadExplanation()` chamada
- ✅ Spinner de "Analisando..." visível

#### Sub-teste: Menu de Ações (Linhas 495-551)

**Passos**:
1. Clicar MoreHorizontal (⋯) em uma transação
2. Menu suspenso aparece com 5 opções:
   - Aprovar
   - Rejeitar
   - Enviar p/ Revisão
   - Marcar como Suspeito
   - Abrir Investigação

**Validação de Ação: Aprovar**:
- POST `/api/transactions/{id}/approve`
- Body: `{status: 'APROVADA'}`
- Esperado: Success toast + tabela atualiza

**Validação de Ação: Rejeitar**:
- POST `/api/transactions/{id}/reject`
- Body: `{status: 'REJEITADA'}`
- Esperado: Success toast + status muda para vermelho

**Validação de Ação: Enviar p/ Revisão**:
- POST `/api/transactions/{id}/review`
- Body: `{status: 'EM_REVISAO'}`
- Esperado: Status muda para amarelo

**Validação de Ação: Marcar Suspeito**:
- POST `/api/transactions/{id}/flag`
- Body: `{flagged: true}`
- Esperado: Flag visual adicionada ou notificação

**Validação de Ação: Abrir Investigação**:
- POST `/api/investigations`
- Body: `{transaction_id, priority: 'high'}`
- Esperado: Investigação criada, notification exibida

**Estados de Erro**:
- ❌ API 500: Alert "Erro: {error}"
- ❌ Timeout: Alert "Erro ao executar ação"
- ❌ Durante ação: Botões desabilitados

---

### 5.5 TESTE: Tabela Vazia

**Objetivo**: Verificar comportamento quando sem resultados

**Pré-condições**:
- Filtros aplicados que resultam em 0 transações
- Ou API retorna `data: []`

**Esperado** (linhas 561-567):
```jsx
{filteredTransactions.length === 0 && (
  <div className="text-center py-12">
    <p className="text-[var(--color-text-secondary)]">
      Nenhuma transação encontrada com os filtros aplicados.
    </p>
  </div>
)}
```

**Validações**:
- ✅ Mensagem centralizada
- ✅ Texto em cor secundária (cinza)
- ✅ Padding generoso (py-12)
- ✅ Tabela não renderizada
- ✅ Nenhum erro no console

---

## 6. TESTES DO MODAL DE DETALHES

### 6.1 Estrutura do Modal

```
Modal (linhas 572-900+)
├── Header (linhas 578-587)
│   ├── Título: "Detalhes da Transação"
│   ├── Subtítulo: ID da transação
│   └── Botão X fechar
│
├── Status + Risco (linhas 592-595)
│   ├── TransactionStatusBadge
│   └── RiskScoreBadge
│
├── Informações Principais (linhas 598-615)
│   ├── Valor
│   ├── Tipo
│   ├── Canal
│   └── Data/Hora
│
├── Dados do Cliente (linhas 618-633)
│   ├── CPF (mascarado)
│   └── Localização
│
├── Análise de Risco (linhas 636-663)
│   ├── Score de Fraude (%)
│   ├── Barra de progresso visual
│   └── Interpretação (Alto/Médio/Baixo)
│
└── Explicação Detalhada - SEÇÕES DIDÁTICAS (linhas 666-900+)
    ├── Seção 1: O que significa o status (linhas 682-734)
    ├── Seção 2: Como o sistema chegou (linhas 737-777)
    ├── Seção 3: Termômetro de Risco (linhas 780-818)
    └── Seção 4: Fatores que chamaram atenção (linhas 821-900+)
```

### 6.2 TESTE: Fechamento do Modal

**Objetivo**: Verificar se modal fecha corretamente

**Métodos de Fechamento**:

1. **Clique no X** (linhas 585-586):
   - Esperado: Modal desaparece, `showDetailsModal=false`

2. **Clique fora do modal** (linhas 573):
   - Background tem `onClick={handleCloseDetails}`
   - Esperado: Modal desaparece sem propagar

3. **Validação** (linhas 208-212):
   ```javascript
   const handleCloseDetails = () => {
     setShowDetailsModal(false)
     setSelectedTransaction(null)
     setExplanation(null)
   }
   ```

---

### 6.3 TESTE: Barra de Risco Visual

**Objetivo**: Verificar barra de progresso e cores

**Score → Cor**:
```javascript
// Linhas 647-654
const progressColor = 
  fraud_score > 0.7 ? 'bg-red-500' :      // Vermelho: Alto risco
  fraud_score > 0.4 ? 'bg-yellow-500' :   // Amarelo: Médio risco
  'bg-green-500'                          // Verde: Baixo risco

// Largura da barra: {fraud_score * 100}%
// Se fraud_score = 0.75, barra = 75%
```

**Casos de Teste**:

| Score | Esperado | Cor | Largura |
|-------|----------|-----|---------|
| 0.0 | 0% | Verde | 0% |
| 0.3 | 30% | Verde | 30% |
| 0.5 | 50% | Amarelo | 50% |
| 0.7 | 70% | Amarelo/Vermelho | 70% |
| 0.95 | 95% | Vermelho | 95% |
| 1.0 | 100% | Vermelho | 100% |

---

### 6.4 TESTE: Seção 1 - Interpretação do Status

**Objetivo**: Verificar se texto muda baseado no score

**Linhas 706-731**:

| Score | Ícone | Título | Texto |
|-------|-------|--------|-------|
| > 0.7 | XCircle (vermelho) | "ALTO RISCO - Possível Fraude Detectada" | Padrão anormal, recomenda análise |
| 0.4-0.7 | AlertTriangle (amarelo) | "RISCO MODERADO - Requer Atenção" | Alguns pontos merecem atenção |
| < 0.4 | CheckCircle (verde) | "BAIXO RISCO - Transação Normal" | Segue padrão esperado |

---

### 6.5 TESTE: Seção 2 - Explicação do Sistema

**Objetivo**: Verificar 4 passos da análise

**Linhas 748-774**:

1. **Análise do Valor** (linhas 748-753):
   - Mostra valor da transação
   - Aviso se > R$ 5.000

2. **Análise do Horário e Local** (linhas 755-759):
   - Descrição genérica de temporalidade

3. **Padrão de Comportamento** (linhas 761-766):
   - Análise histórica do cliente

4. **Canal e Tipo de Transação** (linhas 768-773):
   - Mostra tipo e canal
   - Nota especial para PIX

---

### 6.6 TESTE: Seção 3 - Termômetro de Risco

**Objetivo**: Verificar slider visual verde → amarelo → vermelho

**Componente** (linhas 786-817):
- Gradient: green → yellow → red
- Indicador branco movível
- Percentual exibido
- Zona interpretada (verde/amarela/vermelha)

---

### 6.7 TESTE: Seção 4 - Fatores de Risco

**Objetivo**: Verificar se aparecem fatores relevantes

**Condição** (linha 821):
```javascript
{selectedTransaction.fraud_score > 0.4 && (
```

Somente mostra se score > 0.4 (médio ou alto)

**Fatores Exibidos**:

1. **Valor muito alto** (linhas 831-840):
   - Condição: `valor > 10000`
   - Ícone: DollarSign
   - Texto: Explicação sobre valores acima de R$ 10k

2. **Valor elevado** (linhas 842-849):
   - Condição: `valor > 5000 && valor <= 10000`
   - Ícone: DollarSign
   - Texto: Explicação sobre valores entre R$ 5k-10k

---

### 6.8 TESTE: Explicação Carregando

**Objetivo**: Verificar spinner de carregamento

**Linhas 672-677**:
```jsx
{loadingExplanation ? (
  <div className="bg-blue-50 rounded-lg p-6 text-center">
    <RefreshCw className="h-6 w-6 animate-spin" />
    <p>Analisando os detalhes...</p>
  </div>
)
```

**Esperado**:
- ✅ Spinner animado
- ✅ Texto informativo
- ✅ Background azul
- ✅ Sem erros se falhar

---

## 7. TESTES DE INTEGRAÇÃO

### 7.1 Fluxo Completo: Filtrar → Exportar

**Cenário**: Exportar transações de alto risco das últimas 24h

**Passos**:
1. Status = REJEITADA
2. Período = 24h
3. Clique em Exportar

**Validações**:
- ✅ CSV contém APENAS status REJEITADA
- ✅ CSV contém APENAS últimas 24h
- ✅ Nome arquivo: `transacoes_YYYY-MM-DD.csv`
- ✅ Encoding UTF-8

---

### 7.2 Fluxo: Detalhes → Ação → Refresh

**Cenário**: Aprovar transação e verificar atualização

**Passos**:
1. Clique olho em transação (status PENDENTE)
2. Modal abre
3. Clique fora (fecha modal)
4. Clique MoreHorizontal
5. "Aprovar"
6. Alert "sucesso"
7. `loadTransactions()` dispara

**Validações**:
- ✅ POST `/api/transactions/{id}/approve` chamado
- ✅ Status na tabela muda para APROVADA
- ✅ Badge cor muda para verde
- ✅ Tabela inteira recarrega

---

### 7.3 Cenário de Erro: API 500

**Setup**: Mock `/api/transactions` retorna 500

**Esperado**:
- ✅ Loading finaliza
- ✅ Tabela vazia (fallback para [])
- ✅ Mensagem de erro exibida
- ✅ Botões funcionam

---

### 7.4 Cenário: Dados Incompletos

**Setup**: API retorna transações com campos faltando

```json
{
  "success": true,
  "data": [
    {
      "id": "TXN123",
      "valor": 1000,
      // Faltam: tipo, canal, localizacao, cpf, data_hora, status, fraud_score
    }
  ]
}
```

**Esperado**:
- ✅ Sem erros de undefined
- ✅ Campos vazios renderizam como "-" ou vazio
- ✅ Tabela não quebra

---

## 8. ESTRATÉGIA DE AUTOMAÇÃO

### 8.1 Testes Unitários (Vitest)

```javascript
// formatCurrency.test.js
describe('formatCurrency', () => {
  it('formata BRL corretamente', () => {
    expect(formatCurrency(15000)).toBe('R$ 15.000,00')
    expect(formatCurrency(150.50)).toBe('R$ 150,50')
  })
})

// RiskScoreBadge.test.js
describe('RiskScoreBadge', () => {
  it('renderiza cor correta por score', () => {
    const { container } = render(<RiskScoreBadge score={0.85} />)
    expect(container).toHaveClass('bg-red-500')
  })
})
```

### 8.2 Testes E2E (Playwright)

```javascript
test.describe('Transactions Page', () => {
  test('apply filters and export', async ({ page }) => {
    await page.goto('/transactions')
    
    // Selecionar status
    await page.selectOption('select[name="status"]', 'REJEITADA')
    
    // Aguardar tabela atualizar
    await page.waitForLoadState('networkidle')
    
    // Exportar
    const downloadPromise = page.waitForEvent('download')
    await page.click('button:has-text("Exportar")')
    const download = await downloadPromise
    
    // Verificar arquivo
    expect(download.suggestedFilename()).toMatch(/transacoes_\d{4}-\d{2}-\d{2}\.csv/)
  })
  
  test('open details and verify data', async ({ page }) => {
    await page.goto('/transactions')
    await page.click('[data-testid="view-details-first"]')
    
    // Modal deve abrir
    await expect(page.locator('text=Detalhes da Transação')).toBeVisible()
    
    // Explicação deve carregar
    await expect(page.locator('text=Analisando')).not.toBeVisible({ timeout: 2000 })
    await expect(page.locator('text=Como o sistema chegou')).toBeVisible()
  })
  
  test('approve transaction from actions menu', async ({ page }) => {
    await page.goto('/transactions')
    
    // Abrir menu
    await page.click('[data-testid="actions-menu-first"]')
    
    // Clicar Aprovar
    await page.click('button:has-text("Aprovar")')
    
    // Verificar atualização
    await page.waitForLoadState('networkidle')
    await expect(page.locator('text=sucesso').first()).toBeVisible()
  })
})
```

### 8.3 Testes de Performance

```javascript
// performance.test.js
describe('Transactions Performance', () => {
  test('load transactions in <2 seconds', async () => {
    const start = performance.now()
    const response = await fetch('/api/transactions?page=1&limit=50')
    const elapsed = performance.now() - start
    
    expect(elapsed).toBeLessThan(2000)
    expect(response.status).toBe(200)
  })
})
```

---

## 9. CHECKLIST FINAL

### ✅ Header e Ações
- [ ] Título "Transações" visível
- [ ] Subtítulo descritivo exibido
- [ ] Botão "Exportar" funciona
- [ ] Botão "Atualizar" refaz fetch
- [ ] Spinner visível durante ações
- [ ] Tratamento de erro implementado

### ✅ Filtros
- [ ] Busca funciona (ID, CPF, cidade)
- [ ] Status dropdown com 5 opções
- [ ] Tipo dropdown com 6 opções
- [ ] Período dropdown com 5 opções
- [ ] Filtros combinam corretamente
- [ ] Sem resultados mostra mensagem

### ✅ Tabela
- [ ] 10 colunas renderizadas corretamente
- [ ] Contagem "Mostrando X de Y" correta
- [ ] Ordenação: 6 opções funcionam
- [ ] Formatação moeda (pt-BR)
- [ ] Formatação data/hora
- [ ] Badges de status com cores corretas
- [ ] Badges de risco com cores corretas
- [ ] CPF mascarado (LGPD)

### ✅ Ações de Linha
- [ ] Ícone olho abre modal
- [ ] Menu (...) abre com 5 opções
- [ ] Aprovar → POST API + atualiza
- [ ] Rejeitar → POST API + atualiza
- [ ] Revisão → POST API + atualiza
- [ ] Marcar Suspeito → POST API
- [ ] Abrir Investigação → POST API

### ✅ Modal de Detalhes
- [ ] Abre com dados corretos
- [ ] Fecha com X ou clique fora
- [ ] Status badge renderizado
- [ ] Risco badge renderizado
- [ ] Informações principais exibidas
- [ ] Dados cliente completos
- [ ] Barra de risco visual funciona
- [ ] Cores mudam por score
- [ ] 4 seções didáticas carregam
- [ ] Explicação não quebra com dados faltando

### ✅ Exportação CSV
- [ ] Arquivo gerado com nome correto
- [ ] Cabeçalho correto (9 colunas)
- [ ] Delimitador ponto-e-vírgula
- [ ] UTF-8 com BOM
- [ ] Filtros respeitados (status, período, etc.)
- [ ] Sem linhas extras/vazias

### ✅ Integração Front + Back
- [ ] GET /api/transactions com params corretos
- [ ] Resposta mapeada para state corretamente
- [ ] POST approve/reject/review/flag funcionam
- [ ] POST investigations funciona
- [ ] POST explainability/explain funciona
- [ ] Erros tratados com mensagens amigáveis

### ✅ Performance
- [ ] Carregamento inicial < 2s
- [ ] Ordenação local rápida
- [ ] Filtros não causam lag
- [ ] Exportação não trava UI
- [ ] Modal carrega explicação < 500ms

### ✅ Responsividade
- [ ] Desktop: Layout com espaço
- [ ] Tablet: Tabela com scroll horizontal se necessário
- [ ] Mobile: Colunas críticas visíveis

### ✅ Estados Especiais
- [ ] Loading skeleton exibido
- [ ] Sem transações: mensagem clara
- [ ] Erro API: mensagem amigável
- [ ] Timeout: "Tempo limite excedido"

### ✅ Compliance
- [ ] CPF nunca exibido completo
- [ ] Sem logs de dados sensíveis
- [ ] CORS configurado
- [ ] Rate limiting ativo (se houver)

---

**Documento Completo Preparado**: Dezembro 01, 2025  
**Total de Casos de Teste**: 80+  
**Stack**: React 18 + Vite + Vitest + Playwright  
**Cobertura Alvo**: 80%+ funcionalidade crítica
