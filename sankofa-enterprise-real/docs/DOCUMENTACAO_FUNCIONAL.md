# Documentação Funcional - Sankofa Enterprise Pro v11.0
## Sistema de Detecção de Fraudes para Instituições Financeiras

**Versão:** 11.0  
**Última Atualização:** 27 de Novembro de 2025  
**Status:** Desenvolvimento/Staging - 45 Testes Automatizados Passando

---

## Estado do Sistema

| Componente | Status | Persistência |
|------------|--------|--------------|
| API Backend | ✅ Implementado | 50+ endpoints Flask |
| Frontend Dashboard | ✅ Implementado | 9 páginas React (dados estáticos/mock) |
| ML Stacking Ensemble | ✅ Implementado | RF + GB + LR |
| Transações (PostgreSQL) | ✅ Integrado | INSERT via psycopg2 quando DATABASE_URL existe |
| Transações (In-Memory) | ✅ Implementado | Cache para consultas recentes |
| Configurações | ✅ Implementado | JSON files (system_config.json) |
| Explainability (SHAP) | ⚠️ Módulo separado | Testado, não na API |
| Probability Calibration | ⚠️ Módulo separado | Testado, uso via código |
| Location Entropy | ⚠️ Módulo separado | Features testadas |
| Self-Training | ⚠️ Módulo separado | Testado, uso via código |

**Nota sobre persistência:**
- **PostgreSQL:** Usado para persistir transações quando `DATABASE_URL` está configurado
- **TransactionStore:** Cache em memória para consultas recentes (máximo 1000 itens)
- **ConfigStore:** JSON files para hard rules, VIP list, HOT list, settings

---

## 1. Visão Geral do Sistema

### 1.1 O que é o Sankofa?

O **Sankofa Enterprise Pro** é um sistema de detecção de fraudes financeiras em desenvolvimento que analisa transações em tempo real usando Machine Learning. O nome "Sankofa" vem de um símbolo africano que significa "voltar e buscar" - representando a capacidade do sistema de aprender com padrões passados.

### 1.2 Para Quem é Este Sistema?

| Perfil | Uso Principal |
|--------|---------------|
| **Analistas de Fraude** | Investigar alertas, revisar transações suspeitas |
| **Gestores de Risco** | Monitorar KPIs, ajustar thresholds |
| **Equipe de Compliance** | Gerar relatórios, auditorias |
| **Administradores de TI** | Configurar sistema, gerenciar integrações |

### 1.3 Capacidades do Sistema

**Implementado e Funcional:**
```
┌─────────────────────────────────────────────────────────────┐
│  ✅ Análise em Tempo Real (API /api/fraud/predict)          │
│  ✅ Dashboard 9 Páginas (React)                             │
│  ✅ ML Stacking (RandomForest + GradientBoosting + LR)      │
│  ✅ PostgreSQL para transações (psycopg2)                   │
│  ✅ 45 Testes Automatizados Passando                        │
│  ✅ Endpoints de Calibração (GET/PUT /api/calibration)      │
│  ✅ Lista de Transações, Alertas, Revisão Manual (UI)       │
└─────────────────────────────────────────────────────────────┘
```

**Módulos Disponíveis (não integrados na API):**
```
┌─────────────────────────────────────────────────────────────┐
│  ⚠️ SHAP Explainability (explainability_engine.py)         │
│  ⚠️ Probability Calibration (probability_calibration.py)   │
│  ⚠️ Location Entropy Features (advanced_feature_eng.py)    │
│  ⚠️ Self-Training Optimizer (self_training_optimizer.py)   │
└─────────────────────────────────────────────────────────────┘
```

**Roadmap (Planejado):**
```
┌─────────────────────────────────────────────────────────────┐
│  📋 SHAP explanations nas respostas da API                  │
│  📋 Timers de SLA automáticos na revisão manual             │
│  📋 Métricas de produção reais (TPS, latência)              │
│  📋 Redis cache (atualmente in-memory)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Cobertura de Testes

### 2.1 Resultados dos Testes Automatizados (45 testes)

| Categoria | Testes | Status |
|-----------|--------|--------|
| E2E Infrastructure | 4 | ✅ Passando |
| E2E API Endpoints | 5 | ✅ Passando |
| E2E Fraud Prediction | 4 | ✅ Passando |
| E2E Data Persistence | 2 | ✅ Passando |
| E2E ML Pipeline | 3 | ✅ Passando |
| E2E Performance | 3 | ✅ Passando |
| E2E Validation | 3 | ✅ Passando |
| E2E Integration | 1 | ✅ Passando |
| ML Improvements | 20 | ✅ Passando |
| **TOTAL** | **45** | **100%** |

**Nota:** Métricas de produção (TPS, latência real, uptime) requerem instrumentação adicional que não está implementada. Os valores exibidos no dashboard são simulados para demonstração.

---

## 3. Casos de Uso Principais

### 3.1 UC01: Análise de Transação em Tempo Real

**Ator:** Sistema Bancário (Core Banking)  
**Objetivo:** Avaliar risco de fraude antes de aprovar transação

**Fluxo Principal:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FLUXO DE ANÁLISE EM TEMPO REAL                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   SISTEMA           SANKOFA                          RESPOSTA        │
│   BANCÁRIO          API                                              │
│      │                │                                              │
│      │  POST /api/    │                                              │
│      │  fraud/predict │                                              │
│      ├───────────────►│                                              │
│      │                │  ┌──────────────────────┐                    │
│      │                │  │ 1. Validar Payload   │                    │
│      │                │  │ 2. Extrair Features  │                    │
│      │                │  │ 3. Stacking (RF+GB)  │                    │
│      │                │  │ 4. Meta-model (LR)   │                    │
│      │                │  │ 5. Gerar Prediction  │                    │
│      │                │  │ 6. Salvar no BD      │                    │
│      │                │  └──────────────────────┘                    │
│      │                │                                              │
│      │  200 OK        │                                              │
│      │◄───────────────┤                                              │
│      │  {is_fraud,    │                                              │
│      │   score,       │                                              │
│      │   decision,    │                                              │
│      │   risk_factors}│                                              │
│                                                                      │
│   TEMPO TOTAL: ~250ms                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Exemplo de Requisição:**
```json
{
  "transactions": [{
    "transaction_id": "TXN_001",
    "amount": 15000.00,
    "channel": "PIX",
    "hour": 3,
    "location": "São Paulo"
  }]
}
```

**Exemplo de Resposta (Atual):**
```json
{
  "predictions": [{
    "transaction_id": "TXN_001",
    "is_fraud": true,
    "fraud_probability": 0.87,
    "risk_score": 87,
    "risk_level": "HIGH",
    "confidence": 0.92,
    "processing_time_ms": 25.4,
    "model_version": "1.0.0",
    "detection_reason": [
      "Transação de alto valor",
      "Horário noturno"
    ],
    "timestamp": "2025-11-27T14:30:00Z"
  }]
}
```

**Nota:** SHAP explanations estão disponíveis no módulo `explainability_engine.py` mas não são retornadas automaticamente pela API. Para integrar, veja a documentação técnica.

### 3.2 UC02: Investigação de Fraude

**Ator:** Analista de Fraude  
**Objetivo:** Investigar alerta e decidir ação

**Fluxo (Atual):**
1. Analista acessa Central de Investigação (`/investigation`)
2. Visualiza casos com status "Novo" ou "Em Investigação"
3. Abre detalhes da transação suspeita
4. Analisa informações disponíveis (valor, hora, local, score)
5. Registra decisão: Confirma Fraude ou Falso Positivo

**Roadmap:** Integração de SHAP explanations para justificar decisões do modelo.

### 3.3 UC03: Revisão Manual (Human-in-the-Loop)

**Ator:** Revisor Sênior  
**Objetivo:** Decidir sobre transações em "zona cinza"

**Critérios de Encaminhamento:**
- Score entre 40-70 (zona de incerteza)
- Valor acima de R$ 10.000

**Fluxo (Atual):**
1. Transação aparece na fila de revisão (`/manual-review`)
2. Revisor visualiza lista de transações pendentes
3. Analisa detalhes e toma decisão
4. Registra aprovação ou rejeição

**Roadmap:** Timers de SLA automáticos, priorização visual, feedback loop para retreino.

### 3.4 UC04: Calibragem de Algoritmos

**Ator:** Gestor de Risco  
**Objetivo:** Ajustar sensibilidade do sistema por tier

**Tiers Disponíveis:**

| Tier | Nome | Características | Threshold Padrão |
|------|------|-----------------|------------------|
| 1 | Velocistas | Baixo valor, decisão <50ms | 80% |
| 2 | Rápidos | Valor médio, análise padrão | 70% |
| 3 | Avançados | Alto valor, análise completa | 60% |
| 4 | Supremos | VIP/Corporate, regras customizadas | 50% |

**Parâmetros por Tier:**

| Parâmetro | Descrição | Faixa |
|-----------|-----------|-------|
| Threshold | Limite para classificar como fraude | 0-100% |
| Peso no Ensemble | Importância relativa do algoritmo | 0-0.5 |
| Valor Máximo | Limite para aprovação automática | R$ 1k - 100k |
| Janela de Tempo | Período para análise de velocidade | 60s - 86400s |
| Cache Timeout | Tempo de cache das listas | 60s - 3600s |

### 3.5 UC05: Geração de Relatórios

**Ator:** Equipe de Compliance  
**Objetivo:** Gerar documentação para órgãos reguladores

**Templates Disponíveis:**

| Relatório | Tempo | Conteúdo |
|-----------|-------|----------|
| Mensal de Fraudes | 5-10 min | Fraudes detectadas, valores, canais |
| Performance Trimestral | 3-5 min | Métricas ML, SLAs, uptime |
| Análise de Tendências | 7-12 min | Padrões emergentes, previsões |
| Impacto Financeiro | 4-8 min | Perdas evitadas, ROI |

---

## 4. Módulos do Sistema

### 4.1 Dashboard Executivo (`/`)

**Implementado (UI estática):**
- Cards de KPIs: Transações, Fraudes, Taxa de Aprovação, Latência
- Layout para gráficos e alertas
- Cards de status dos modelos

**Nota:** Os valores exibidos são dados estáticos/mock. Não há polling ou atualização em tempo real implementada.

**Roadmap:** Integração com API para dados em tempo real, atualização automática.

### 4.2 Central de Transações (`/transactions`)

**Implementado:**
- Tabela com transações recentes (via API /api/transactions)
- Colunas: ID, Valor, Tipo, Canal, Localização, CPF mascarado, Data/Hora
- Filtros básicos de status e tipo (UI)

**Nota:** A paginação e exportação CSV são roadmap. A tabela exibe dados retornados pela API.

**Roadmap:** Paginação server-side, filtros avançados, exportação CSV.

### 4.3 Calibragem Manual (`/calibration`)

**Implementado (UI):**
- Layout com cards para diferentes algoritmos
- Sliders visuais para threshold
- Botões de salvar/resetar

**Nota:** A interface existe mas a integração com backend (GET/PUT /api/calibration) pode não refletir todas as opções visuais. Os endpoints de calibração existem na API.

**Roadmap:** Integração completa de todos os controles com backend.

### 4.4 Central de Investigação (`/investigation`)

**Implementado (UI):**
- Layout com cards de estatísticas
- Lista de casos
- Campos de busca e filtro (UI)

**Nota:** A interface exibe layout estático. Os dados de investigação dependem de transações flagadas no sistema.

**Roadmap:** Integração com fluxo de investigação completo.

### 4.5 Revisão Manual (`/manual-review`)

**Implementado:**
- Lista de transações pendentes de revisão
- Contadores estáticos (total, pendentes, completadas)
- Tabela com ID, valor, CPF mascarado, score de risco
- Botões de ação (aprovar/rejeitar)

**Roadmap:** Timers de SLA automáticos, contagem de expiradas.

### 4.6 Monitoramento (`/monitoring`)

**Implementado (UI com dados simulados):**
- Cards de status do sistema
- Indicadores visuais de saúde
- Layout para métricas de recursos

**Nota:** Os valores exibidos são simulados para demonstração. Métricas reais de CPU, memória, TPS e latência requerem instrumentação backend não implementada.

**Roadmap:** Integração com métricas reais, monitoramento Redis, filas.

### 4.7 Central de Relatórios (`/reports`)

**Implementado:**
- Lista de templates de relatórios
- Interface de seleção
- Botão de geração

**Nota:** A geração de relatórios retorna dados simulados. Integração com dados reais é roadmap.

### 4.8 Métricas e Contadores (`/metrics`)

**Implementado:**
- Cards de contadores (transações, fraudes, precisão, tempo)
- Layout de hard rules e listas VIP/HOT
- Toggle de auto-refresh (UI)

**Nota:** Valores são simulados para demonstração. Contadores reais requerem integração com backend de métricas.

### 4.9 Central de Alertas (`/alerts`)

**Implementado:**
- Lista de alertas (via API /api/alerts)
- Filtros por status
- Ações de acknowledge

**Tipos de Alerta (API):**
| Tipo | Descrição |
|------|-----------|
| Fraude Detectada | Transação bloqueada |
| Sistema | Erros de backend |

---

## 5. Regras de Negócio

### 5.1 Classificação de Risco

| Score | Classificação | Ação | Cor |
|-------|---------------|------|-----|
| 0-30 | **Baixo Risco** | Aprovação automática | 🟢 Verde |
| 31-50 | **Risco Moderado** | Aprovação com monitoramento | 🟡 Amarelo |
| 51-70 | **Alto Risco** | Encaminha para revisão manual | 🟠 Laranja |
| 71-100 | **Risco Crítico** | Bloqueio automático | 🔴 Vermelho |

### 5.2 Hard Rules (Regras Absolutas)

Regras que bloqueiam independente do score de ML:

1. **Lista Negra:** CPF/CNPJ em lista de fraudadores conhecidos
2. **Valor Extremo:** Transação > R$ 50.000 em conta nova
3. **Velocidade:** Mais de 10 transações em 5 minutos
4. **Horário Noturno:** Transações 22h-6h + valor alto + conta nova
5. **Location Entropy Alta:** Muitas localizações diferentes em curto período

### 5.3 Features de ML (47+)

**Temporais (5):**
- hour, day_of_week, is_weekend, is_night, is_business_hours

**Valor (5):**
- amount_log, amount_squared, amount_normalized, is_round_amount, amount_zscore

**Geográficas (3):**
- distance_from_home, location_risk_score, is_international

**Comportamentais (4):**
- transaction_velocity_1h, transaction_velocity_24h, amount_deviation, new_merchant

**Location Entropy (11):**
- location_entropy, unique_locations, location_diversity_score, etc.

**Padrões (19):**
- V1-V28 do dataset Kaggle (PCA components)

---

## 6. Integrações e APIs

### 6.1 Endpoints Principais

| Endpoint | Método | Função | Auth |
|----------|--------|--------|------|
| `/api/health` | GET | Health check | Não |
| `/api/fraud/predict` | POST | Análise de transação | Sim |
| `/api/fraud/batch` | POST | Análise em lote | Sim |
| `/api/transactions` | GET | Listar transações | Sim |
| `/api/feedback` | POST | Feedback do analista | Sim |
| `/api/model/metrics` | GET | Métricas do modelo | Sim |
| `/api/dashboard/*` | GET | Dados do dashboard | Sim |
| `/api/manual-review` | GET | Fila de revisão | Sim |
| `/api/alerts` | GET | Listar alertas | Sim |

### 6.2 Rate Limiting

| Endpoint | Limite |
|----------|--------|
| `/api/fraud/predict` | 1000/min |
| `/api/fraud/batch` | 100/min |
| Outros | 500/min |

### 6.3 Autenticação

- **Tipo:** JWT Bearer Token
- **Expiração:** 24 horas
- **Rotação:** Automática a cada 30 dias

---

## 7. Compliance e Regulamentação

**Nota:** Esta seção descreve requisitos de compliance. A implementação atual atende parcialmente estes requisitos.

### 7.1 LGPD

| Requisito | Status | Notas |
|-----------|--------|-------|
| Dados pessoais mascarados (CPF) | ✅ Implementado | UI exibe XXX.XXX.XXX-XX |
| Logs de auditoria | ✅ Implementado | Tabela audit_log no PostgreSQL |
| Explicabilidade (SHAP) | ⚠️ Módulo existe | Não integrado na API |
| Exportação de dados | 📋 Roadmap | Não implementado |

### 7.2 BACEN Resolução 6/2023

| Requisito | Status | Notas |
|-----------|--------|-------|
| API de detecção de fraude | ✅ Implementado | /api/fraud/predict |
| Tempo de resposta | ⚠️ Não monitorado | Sem instrumentação de latência |
| Disponibilidade | ⚠️ Não monitorado | Sem SLA enforcement |
| Audit trail | ✅ Implementado | Via banco de dados |

### 7.3 PCI DSS

| Requisito | Status | Notas |
|-----------|--------|-------|
| CPF/dados sensíveis | ✅ Mascarados na UI | Não tokenizados no backend |
| Logs seguros | ✅ Sem dados sensíveis | Structured logging |
| TLS | ⚠️ Ambiente dev | Produção requer configuração |

---

## 8. Glossário

| Termo | Definição |
|-------|-----------|
| **Ensemble** | Combinação de múltiplos modelos de ML |
| **Feature** | Característica extraída da transação |
| **Threshold** | Limite de corte para decisão |
| **Drift** | Degradação do modelo por mudança nos dados |
| **SHAP** | SHapley Additive exPlanations - técnica de explicabilidade |
| **TPS** | Transações por segundo |
| **HITL** | Human-in-the-Loop - revisão humana |
| **Score** | Pontuação de risco (0-100) |
| **Location Entropy** | Medida de diversidade geográfica |
| **Calibração** | Ajuste de probabilidades do modelo |

---

*Documento gerado automaticamente pelo Sankofa Enterprise Pro v11.0*  
*Última atualização: 27 de Novembro de 2025*
