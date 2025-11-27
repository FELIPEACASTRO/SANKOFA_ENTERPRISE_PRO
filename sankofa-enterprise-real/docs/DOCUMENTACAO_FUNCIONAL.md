# Sankofa Enterprise Pro - Documentação Funcional

**Versão:** 1.0.0  
**Data:** Novembro 2025  
**Classificação:** Confidencial - Uso Interno

---

> **Nota:** Este documento descreve as funcionalidades implementadas e planejadas do sistema.
> Funcionalidades marcadas como **(Em Desenvolvimento)** ou **(Planejado)** estão em progresso.
> O sistema atual está em ambiente de desenvolvimento/homologação.

---

## Sumário

1. [Introdução](#1-introdução)
2. [Visão Geral do Sistema](#2-visão-geral-do-sistema)
3. [Casos de Uso](#3-casos-de-uso)
4. [Fluxos de Negócio](#4-fluxos-de-negócio)
5. [Regras de Negócio](#5-regras-de-negócio)
6. [Compliance e Regulamentação](#6-compliance-e-regulamentação)
7. [Integrações](#7-integrações)
8. [SLAs e Níveis de Serviço](#8-slas-e-níveis-de-serviço)
9. [Glossário de Negócio](#9-glossário-de-negócio)

---

## 1. Introdução

### 1.1 Propósito

O Sankofa Enterprise Pro é uma plataforma de detecção de fraudes em tempo real projetada para instituições financeiras brasileiras. O sistema utiliza inteligência artificial avançada para analisar transações e identificar atividades fraudulentas antes que causem prejuízos.

### 1.2 Escopo

O sistema abrange:

- **Detecção de Fraudes em Tempo Real**: Análise de transações PIX, TED, DOC, cartões
- **Machine Learning**: Modelos de IA continuamente atualizados
- **Revisão Manual**: Interface para analistas revisarem casos suspeitos
- **Compliance**: Conformidade com BACEN, LGPD e PCI-DSS
- **Dashboard Executivo**: Visualização de KPIs e métricas em tempo real

### 1.3 Público-Alvo

| Perfil | Responsabilidades |
|--------|-------------------|
| **Analista de Fraude** | Revisar casos suspeitos, aprovar/bloquear transações |
| **Gerente de Operações** | Monitorar KPIs, ajustar configurações |
| **Compliance Officer** | Garantir conformidade regulatória |
| **Equipe de TI** | Manutenção técnica, integrações |
| **Diretoria** | Visão executiva de resultados |

---

## 2. Visão Geral do Sistema

### 2.1 Capacidades Principais

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SANKOFA ENTERPRISE PRO                           │
│                 Sistema de Detecção de Fraudes                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│   │  DETECÇÃO EM     │  │  MACHINE         │  │  REVISÃO         │  │
│   │  TEMPO REAL      │  │  LEARNING        │  │  MANUAL          │  │
│   │                  │  │                  │  │                  │  │
│   │  • Análise <15ms │  │  • Ensemble ML   │  │  • Queue HITL    │  │
│   │  • 300M txn/dia  │  │  • 99.9% acurácia│  │  • Workflow      │  │
│   │  • Multi-canal   │  │  • Auto-retrain  │  │  • Histórico     │  │
│   └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                      │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│   │  MONITORAMENTO   │  │  COMPLIANCE      │  │  DASHBOARD       │  │
│   │  CONTÍNUO        │  │  REGULATÓRIO     │  │  EXECUTIVO       │  │
│   │                  │  │                  │  │                  │  │
│   │  • Drift detect  │  │  • BACEN         │  │  • KPIs tempo    │  │
│   │  • Alertas       │  │  • LGPD          │  │    real          │  │
│   │  • Health checks │  │  • PCI-DSS       │  │  • Relatórios    │  │
│   └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Métricas de Negócio

| Métrica | Valor Alvo | Descrição |
|---------|------------|-----------|
| Taxa de Detecção | > 95% | Fraudes identificadas / Total de fraudes |
| Falsos Positivos | < 1% | Transações legítimas bloqueadas |
| Tempo de Resposta | < 15ms | Latência P95 de análise |
| Disponibilidade | 99.9% | Uptime do sistema |
| Valor Protegido | R$ XXM/mês | Fraudes evitadas em reais |

### 2.3 Canais Suportados

| Canal | Descrição | Volume Estimado |
|-------|-----------|-----------------|
| **PIX** | Pagamentos instantâneos | 60% |
| **TED** | Transferência eletrônica | 20% |
| **DOC** | Documento de crédito | 5% |
| **Cartão Crédito** | Compras com cartão | 10% |
| **Cartão Débito** | Compras com débito | 5% |

---

## 3. Casos de Uso

### 3.1 UC01 - Análise de Transação em Tempo Real

**Ator Principal:** Sistema de Pagamentos (Core Banking)

**Pré-condições:**
- Sistema Sankofa operacional
- Modelo ML treinado e disponível
- Conexão com sistema de origem ativa

**Fluxo Principal:**

```
┌────────────────────────────────────────────────────────────────────┐
│ UC01: ANÁLISE DE TRANSAÇÃO EM TEMPO REAL                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Sistema de origem envia transação para análise                  │
│     │                                                               │
│     ▼                                                               │
│  2. Sankofa valida formato e campos obrigatórios                   │
│     │                                                               │
│     ├── [Inválido] → Retorna erro de validação                     │
│     │                                                               │
│     ▼                                                               │
│  3. Motor ML extrai features da transação                          │
│     │                                                               │
│     ▼                                                               │
│  4. Modelo ensemble gera predição                                  │
│     │                                                               │
│     ▼                                                               │
│  5. Regras de precisão aplicadas                                   │
│     │                                                               │
│     ▼                                                               │
│  6. Sistema calcula score de risco final                           │
│     │                                                               │
│     ├── [BAIXO: 0-30] → Aprovar automaticamente                    │
│     ├── [MÉDIO: 31-70] → Aprovar com monitoramento                 │
│     ├── [ALTO: 71-90] → Enviar para revisão manual                 │
│     └── [CRÍTICO: 91-100] → Bloquear automaticamente               │
│                                                                     │
│  7. Retorna resposta para sistema de origem                        │
│                                                                     │
│  TEMPO TOTAL: < 15ms                                                │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

**Pós-condições:**
- Transação classificada com score de risco
- Decisão registrada para auditoria
- Métricas atualizadas em tempo real

---

### 3.2 UC02 - Revisão Manual de Transação (HITL)

**Ator Principal:** Analista de Fraude

**Pré-condições:**
- Analista autenticado no sistema
- Transação na fila de revisão manual

**Fluxo Principal:**

```
┌────────────────────────────────────────────────────────────────────┐
│ UC02: REVISÃO MANUAL (HUMAN-IN-THE-LOOP)                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Analista acessa página "Revisão Manual"                        │
│     │                                                               │
│     ▼                                                               │
│  2. Sistema exibe lista de casos pendentes                         │
│     │ (ordenados por prioridade/tempo na fila)                     │
│     │                                                               │
│     ▼                                                               │
│  3. Analista seleciona caso para análise                           │
│     │                                                               │
│     ▼                                                               │
│  4. Sistema exibe:                                                 │
│     │ • Detalhes da transação                                      │
│     │ • Score de risco e razões                                    │
│     │ • Histórico do cliente                                       │
│     │ • Transações relacionadas                                    │
│     │ • Mapa de localização                                        │
│     │                                                               │
│     ▼                                                               │
│  5. Analista toma decisão:                                         │
│     │                                                               │
│     ├── [APROVAR] → Libera transação                               │
│     │              → Registra como legítima                        │
│     │              → Feedback para retreino                        │
│     │                                                               │
│     ├── [BLOQUEAR] → Bloqueia transação                            │
│     │              → Notifica cliente                              │
│     │              → Registra como fraude confirmada               │
│     │                                                               │
│     └── [ESCALAR] → Envia para supervisor                          │
│                   → Mantém em hold                                 │
│                                                                     │
│  6. Sistema registra decisão e justificativa                       │
│                                                                     │
│  SLA: < 5 minutos por caso                                         │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

**Pós-condições:**
- Decisão registrada com justificativa
- Transação processada conforme decisão
- Feedback enviado para modelo ML
- Auditoria completa do caso

---

### 3.3 UC03 - Monitoramento de Saúde do Modelo

**Ator Principal:** Data Scientist / ML Engineer

**Fluxo Principal:**

```
┌────────────────────────────────────────────────────────────────────┐
│ UC03: MONITORAMENTO DE MODELO                                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Sistema coleta métricas continuamente:                         │
│     │ • Acurácia, Precisão, Recall, F1                             │
│     │ • Distribuição de predições                                  │
│     │ • Latência de inferência                                     │
│     │ • Distribuição de features                                   │
│     │                                                               │
│     ▼                                                               │
│  2. Drift Detector analisa mudanças:                               │
│     │                                                               │
│     ├── Data Drift: Mudança nas features de entrada                │
│     │   (Jensen-Shannon, PSI)                                      │
│     │                                                               │
│     └── Concept Drift: Mudança na relação feature→target           │
│         (Performance degradation)                                  │
│     │                                                               │
│     ▼                                                               │
│  3. Sistema classifica severidade:                                 │
│     │                                                               │
│     ├── [LOW] → Log + Dashboard update                             │
│     ├── [MEDIUM] → Alerta + Investigação                           │
│     ├── [HIGH] → Alerta urgente + Planejar retrain                 │
│     └── [CRITICAL] → Alerta crítico + Retrain imediato             │
│                                                                     │
│  4. Página Monitoramento exibe:                                    │
│     │ • Gráficos de tendência                                      │
│     │ • Alertas ativos                                             │
│     │ • Comparativo com baseline                                   │
│     │ • Recomendações de ação                                      │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

### 3.4 UC04 - Configuração de Regras

**Ator Principal:** Gerente de Operações

**Fluxo Principal:**

```
┌────────────────────────────────────────────────────────────────────┐
│ UC04: CONFIGURAÇÃO DE REGRAS                                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Gerente acessa página "Calibração"                             │
│     │                                                               │
│     ▼                                                               │
│  2. Sistema exibe configurações atuais:                            │
│     │ • Threshold de risco                                         │
│     │ • Limites por canal                                          │
│     │ • Horários suspeitos                                         │
│     │ • Regras de velocidade                                       │
│     │                                                               │
│     ▼                                                               │
│  3. Gerente ajusta parâmetros                                      │
│     │                                                               │
│     ▼                                                               │
│  4. Sistema simula impacto:                                        │
│     │ • Estimativa de novos bloqueios                              │
│     │ • Impacto em falsos positivos                                │
│     │ • Comparativo com período anterior                           │
│     │                                                               │
│     ▼                                                               │
│  5. Gerente confirma alterações                                    │
│     │                                                               │
│     ▼                                                               │
│  6. Sistema aplica novas regras                                    │
│     │ • Registra alteração para auditoria                          │
│     │ • Notifica equipe                                            │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

### 3.5 UC05 - Processamento em Batch

**Ator Principal:** Sistema Agendado

**Fluxo Principal:**

```
┌────────────────────────────────────────────────────────────────────┐
│ UC05: PROCESSAMENTO BATCH                                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Sistema recebe lote de transações                              │
│     │ (até 10.000 transações por request)                          │
│     │                                                               │
│     ▼                                                               │
│  2. Validação inicial do lote                                      │
│     │ • Formato JSON válido                                        │
│     │ • Campos obrigatórios presentes                              │
│     │                                                               │
│     ▼                                                               │
│  3. Processamento paralelo                                         │
│     │ • Chunks de 100 transações                                   │
│     │ • Multi-threading                                            │
│     │                                                               │
│     ▼                                                               │
│  4. Agregação de resultados                                        │
│     │                                                               │
│     ▼                                                               │
│  5. Retorno com:                                                   │
│     │ • Predição para cada transação                               │
│     │ • Estatísticas do lote                                       │
│     │ • Tempo total de processamento                               │
│                                                                     │
│  PERFORMANCE: ~10.000 TPS em batch                                  │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 4. Fluxos de Negócio

### 4.1 Fluxo Completo de Detecção de Fraude

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FLUXO COMPLETO DE DETECÇÃO                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ORIGEM                     SANKOFA                     DESTINO          │
│  ──────                     ───────                     ───────          │
│                                                                          │
│  ┌──────────┐                                                            │
│  │  Core    │                                                            │
│  │ Banking  │───┐                                                        │
│  └──────────┘   │                                                        │
│                 │    ┌─────────────────────────────────────┐             │
│  ┌──────────┐   │    │                                     │             │
│  │  Mobile  │───┼───▶│           API GATEWAY               │             │
│  │   App    │   │    │   (Rate Limit + Auth + Logging)     │             │
│  └──────────┘   │    │                                     │             │
│                 │    └──────────────────┬──────────────────┘             │
│  ┌──────────┐   │                       │                                │
│  │  PIX     │───┘                       ▼                                │
│  │ Gateway  │               ┌─────────────────────┐                      │
│  └──────────┘               │   FEATURE ENGINE    │                      │
│                             │ (47+ features)      │                      │
│                             └──────────┬──────────┘                      │
│                                        │                                 │
│                                        ▼                                 │
│                             ┌─────────────────────┐                      │
│                             │   ML ENSEMBLE       │                      │
│                             │ (RF + GB + LR)      │                      │
│                             └──────────┬──────────┘                      │
│                                        │                                 │
│                                        ▼                                 │
│                             ┌─────────────────────┐                      │
│                             │  DECISION ENGINE    │                      │
│                             │  (Rules + Score)    │                      │
│                             └──────────┬──────────┘                      │
│                                        │                                 │
│                     ┌──────────────────┼──────────────────┐              │
│                     │                  │                  │              │
│                     ▼                  ▼                  ▼              │
│              ┌───────────┐     ┌───────────┐     ┌───────────┐          │
│              │  APROVAR  │     │  REVISAR  │     │ BLOQUEAR  │          │
│              │   AUTO    │     │  MANUAL   │     │   AUTO    │          │
│              └─────┬─────┘     └─────┬─────┘     └─────┬─────┘          │
│                    │                 │                 │                 │
│                    ▼                 ▼                 ▼                 │
│              ┌───────────┐     ┌───────────┐     ┌───────────┐          │
│              │  Processa │     │  Fila de  │     │ Notifica  │          │
│              │ Transação │     │  Analista │     │  Cliente  │          │
│              └───────────┘     └───────────┘     └───────────┘          │
│                    │                 │                 │                 │
│                    │                 ▼                 │                 │
│                    │          ┌───────────┐           │                 │
│                    │          │ DECISÃO   │           │                 │
│                    │          │ ANALISTA  │           │                 │
│                    │          └─────┬─────┘           │                 │
│                    │                │                 │                 │
│                    └────────────────┼─────────────────┘                 │
│                                     │                                    │
│                                     ▼                                    │
│                             ┌───────────────┐                            │
│                             │   FEEDBACK    │                            │
│                             │   LOOP        │                            │
│                             │ (Retreino ML) │                            │
│                             └───────────────┘                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Fluxo de Feedback e Aprendizado

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FLUXO DE FEEDBACK E APRENDIZADO                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  FONTES DE FEEDBACK                                                      │
│                                                                          │
│  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐          │
│  │   Analista    │     │   Cliente     │     │   Chargeback  │          │
│  │   Aprova/     │     │   Contesta    │     │   Recebido    │          │
│  │   Bloqueia    │     │   Transação   │     │               │          │
│  └───────┬───────┘     └───────┬───────┘     └───────┬───────┘          │
│          │                     │                     │                   │
│          └─────────────────────┼─────────────────────┘                   │
│                                │                                         │
│                                ▼                                         │
│                      ┌─────────────────────┐                             │
│                      │   FEEDBACK STORE    │                             │
│                      │                     │                             │
│                      │  • Transaction ID   │                             │
│                      │  • True Label       │                             │
│                      │  • Source           │                             │
│                      │  • Timestamp        │                             │
│                      │  • Reason           │                             │
│                      └──────────┬──────────┘                             │
│                                 │                                        │
│                                 ▼                                        │
│                      ┌─────────────────────┐                             │
│                      │  AGGREGATION JOB    │                             │
│                      │  (Daily/Weekly)     │                             │
│                      └──────────┬──────────┘                             │
│                                 │                                        │
│                                 ▼                                        │
│                      ┌─────────────────────┐                             │
│                      │   MODEL TRAINING    │                             │
│                      │                     │                             │
│                      │  • Dados históricos │                             │
│                      │  • Novos feedbacks  │                             │
│                      │  • Validação        │                             │
│                      └──────────┬──────────┘                             │
│                                 │                                        │
│                                 ▼                                        │
│                      ┌─────────────────────┐                             │
│                      │   A/B TEST ou       │                             │
│                      │   CANARY DEPLOY     │                             │
│                      └──────────┬──────────┘                             │
│                                 │                                        │
│                                 ▼                                        │
│                      ┌─────────────────────┐                             │
│                      │   PRODUCTION        │                             │
│                      │   (New Model)       │                             │
│                      └─────────────────────┘                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Regras de Negócio

### 5.1 Classificação de Risco

| Score | Nível | Ação Automática | Cor no Dashboard |
|-------|-------|-----------------|------------------|
| 0-30 | BAIXO | Aprovar | Verde |
| 31-50 | MÉDIO-BAIXO | Aprovar + Log | Amarelo claro |
| 51-70 | MÉDIO | Aprovar + Monitorar | Amarelo |
| 71-85 | ALTO | Revisão Manual | Laranja |
| 86-95 | MUITO ALTO | Revisão Manual Urgente | Vermelho |
| 96-100 | CRÍTICO | Bloquear | Vermelho escuro |

### 5.2 Regras de Velocidade (Velocity Rules)

| Regra | Limite | Janela | Ação |
|-------|--------|--------|------|
| Transações por hora | > 20 | 1 hora | +20 score |
| Transações por dia | > 50 | 24 horas | +15 score |
| Valor acumulado dia | > R$ 50.000 | 24 horas | +25 score |
| Novos destinatários | > 5 | 24 horas | +10 score |

### 5.3 Regras de Valor

| Condição | Ação |
|----------|------|
| Valor > R$ 50.000 em horário suspeito (0h-6h) | +30 score |
| Valor > R$ 100.000 (qualquer horário) | +20 score |
| Valor redondo exato (ex: R$ 10.000,00) | +5 score |
| Primeiro PIX > R$ 5.000 | +15 score |

### 5.4 Regras de Comportamento

| Condição | Ação |
|----------|------|
| Novo dispositivo | +10 score |
| Nova localização (> 100km) | +15 score |
| Transação internacional | +20 score |
| Horário incomum para cliente | +10 score |
| Comerciante de alto risco | +25 score |

### 5.5 Regras de Combinação (High Risk)

```python
# Combinações que disparam alerta máximo
HIGH_RISK_COMBINATIONS = [
    {
        "conditions": ["novo_dispositivo", "nova_localizacao", "valor_alto"],
        "score_boost": 50,
        "action": "BLOCK_AUTO"
    },
    {
        "conditions": ["horario_madrugada", "valor_muito_alto", "novo_destinatario"],
        "score_boost": 45,
        "action": "MANUAL_REVIEW_URGENT"
    },
    {
        "conditions": ["velocidade_anormal", "multiplos_destinatarios"],
        "score_boost": 40,
        "action": "MANUAL_REVIEW"
    }
]
```

---

## 6. Compliance e Regulamentação

### 6.1 BACEN - Resolução 6/2023

**Requisitos Atendidos:**

| Requisito | Implementação |
|-----------|---------------|
| Compartilhamento de dados de fraude | API de exportação para DICT |
| Tempo de resposta PIX | < 10 segundos (SLA: 15ms) |
| Registro de transações suspeitas | Auditoria completa |
| Mecanismos antifraude | Motor ML + Regras |
| Comunicação ao cliente | Notificações configuráveis |

### 6.2 LGPD - Lei Geral de Proteção de Dados

**Medidas Implementadas:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CONFORMIDADE LGPD                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  COLETA E TRATAMENTO                                                     │
│  ├─ Base legal: Legítimo interesse (Art. 10) + Prevenção fraude         │
│  ├─ Minimização: Apenas dados necessários para análise                  │
│  └─ Finalidade: Exclusivamente detecção de fraude                       │
│                                                                          │
│  ARMAZENAMENTO                                                           │
│  ├─ Criptografia AES-256 em repouso                                     │
│  ├─ TLS 1.3 em trânsito                                                 │
│  ├─ Acesso restrito por RBAC                                            │
│  └─ Logs de acesso a dados pessoais                                     │
│                                                                          │
│  RETENÇÃO                                                                │
│  ├─ Transações: 5 anos (regulatório)                                    │
│  ├─ Logs de acesso: 2 anos                                              │
│  └─ Dados de treinamento: Anonimizados                                  │
│                                                                          │
│  DIREITOS DO TITULAR                                                     │
│  ├─ Acesso: API de consulta disponível                                  │
│  ├─ Correção: Endpoint de atualização                                   │
│  ├─ Eliminação: Processo definido (respeitando retenção legal)          │
│  └─ Portabilidade: Exportação em formato padrão                         │
│                                                                          │
│  SEGURANÇA                                                               │
│  ├─ DPO designado                                                       │
│  ├─ DPIA realizado                                                      │
│  ├─ Incidentes: Processo de notificação 72h                             │
│  └─ Treinamento: Equipe capacitada                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 PCI-DSS

**Controles Implementados:**

| Requisito PCI | Implementação |
|---------------|---------------|
| Req. 1: Firewall | Configuração de rede segura |
| Req. 2: Senhas padrão | Sem senhas default |
| Req. 3: Proteção dados | AES-256 + Tokenização |
| Req. 4: Transmissão segura | TLS 1.3 obrigatório |
| Req. 5: Antivírus | Scanning contínuo |
| Req. 6: Sistemas seguros | SAST/DAST, patches |
| Req. 7: Acesso restrito | RBAC implementado |
| Req. 8: ID único | Autenticação individual |
| Req. 9: Acesso físico | N/A (Cloud) |
| Req. 10: Monitoramento | Logs + SIEM |
| Req. 11: Testes | Pentest anual |
| Req. 12: Política | Documentação completa |

---

## 7. Integrações

### 7.1 Integrações de Entrada

| Sistema | Protocolo | Descrição |
|---------|-----------|-----------|
| Core Banking | REST API | Transações para análise |
| PIX Gateway | REST API | Transações PIX |
| Mobile App | REST API | Transações mobile |
| Batch Systems | REST API | Lotes de transações |

### 7.2 Integrações de Saída

| Sistema | Protocolo | Descrição |
|---------|-----------|-----------|
| DICT (BACEN) | REST API | Compartilhamento fraudes |
| Notificações | Webhook | Alertas para sistemas |
| SIEM | Syslog/API | Logs de segurança |
| DataDog | API | Métricas e APM |

### 7.3 Formato de Mensagem Padrão

```json
{
  "header": {
    "message_id": "uuid",
    "timestamp": "ISO8601",
    "source_system": "string",
    "version": "1.0"
  },
  "transaction": {
    "transaction_id": "string",
    "type": "PIX|TED|DOC|CARD",
    "amount": 0.00,
    "currency": "BRL",
    "timestamp": "ISO8601",
    "customer": {
      "id": "string",
      "document": "CPF/CNPJ"
    },
    "merchant": {
      "id": "string",
      "category": "MCC"
    },
    "location": {
      "latitude": 0.0,
      "longitude": 0.0
    },
    "device": {
      "fingerprint": "string"
    }
  }
}
```

---

## 8. SLAs e Níveis de Serviço

### 8.1 SLAs de Performance

| Métrica | Bronze | Silver | Gold | Platinum |
|---------|--------|--------|------|----------|
| Latência P95 | < 50ms | < 30ms | < 15ms | < 10ms |
| Disponibilidade | 99% | 99.5% | 99.9% | 99.99% |
| Throughput | 10K TPS | 50K TPS | 100K TPS | 300K TPS |
| Tempo recuperação | 4h | 2h | 1h | 15min |

### 8.2 SLAs de Qualidade

| Métrica | Mínimo | Alvo | Excelência |
|---------|--------|------|------------|
| Taxa Detecção | 90% | 95% | 99% |
| Falsos Positivos | < 5% | < 2% | < 0.5% |
| Acurácia Modelo | 95% | 98% | 99.9% |

### 8.3 SLAs de Operação

| Processo | SLA |
|----------|-----|
| Revisão Manual | < 5 minutos por caso |
| Resposta Suporte N1 | < 15 minutos |
| Resolução Incidente P1 | < 1 hora |
| Deploy Nova Versão | < 30 minutos |

---

## 9. Limitações Conhecidas

### Funcionalidades em Desenvolvimento

| Funcionalidade | Status | Impacto |
|---|---|---|
| **A/B Testing de Modelos** | 📋 Conceitual | Não afeta produção atual |
| **Canary Deployment** | 📋 Conceitual | Não afeta produção atual |
| **Monitoramento de Drift** | ⚠️ Básico | Monitorado manualmente |
| **Integração DICT (BACEN)** | 📋 Planejado | Exportação manual possível |
| **Endpoints de Calibração** | ⚠️ Parcial | Ajustes via arquivo JSON |
| **Retreinamento Automático** | 📋 Planejado | Retreinamento manual possível |

### Conformidade Atual

- ✅ **LGPD**: Implementado (acesso/eliminação de dados)
- ✅ **Audit Trail**: Implementado (logs estruturados)
- ⚠️ **BACEN**: Funcionalidades core, integração com DICT em progresso
- 📋 **PCI-DSS**: Estrutura pronta, certificação pendente

### Recomendações para Produção

1. Implementar A/B Testing antes de usar em ambiente bancário
2. Adicionar Docker/Nginx para escalabilidade
3. Integrar certificação PCI-DSS
4. Implementar failover de Redis para alta disponibilidade

---

## 10. Glossário de Negócio

| Termo | Definição |
|-------|-----------|
| **Chargeback** | Estorno de transação por contestação do cliente |
| **Falso Positivo** | Transação legítima incorretamente classificada como fraude |
| **Falso Negativo** | Fraude não detectada pelo sistema |
| **HITL** | Human-in-the-Loop - Revisão manual por analista |
| **KYC** | Know Your Customer - Conhecimento do cliente |
| **MCC** | Merchant Category Code - Código de categoria do comerciante |
| **PIX** | Sistema de pagamentos instantâneos do Brasil |
| **Score de Risco** | Pontuação 0-100 indicando probabilidade de fraude |
| **TED** | Transferência Eletrônica Disponível |
| **Threshold** | Limite de corte para classificação |
| **Velocity** | Velocidade de transações em período |

---

**Documento mantido por:** Equipe de Negócios Sankofa  
**Última atualização:** Novembro 2025  
**Versão:** 1.0.0
