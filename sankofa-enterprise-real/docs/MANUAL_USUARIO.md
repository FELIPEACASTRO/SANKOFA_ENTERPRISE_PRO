# Sankofa Enterprise Pro - Manual do Usuário

**Versão:** 1.0.0  
**Data:** Novembro 2025  
**Público:** Analistas de Fraude, Gerentes de Operações, Compliance Officers

---

> **Nota sobre o Status do Sistema:**
> Este manual descreve as funcionalidades do dashboard Sankofa Enterprise Pro.
> Algumas páginas podem apresentar limitações no ambiente de desenvolvimento:
> - **Transações e Alertas:** Podem apresentar erros menores de renderização
> - **Dados em tempo real:** Dependem do backend estar em execução
> - **Endpoints de calibração:** Em desenvolvimento

---

## Sumário

1. [Introdução](#1-introdução)
2. [Acessando o Sistema](#2-acessando-o-sistema)
3. [Visão Geral do Dashboard](#3-visão-geral-do-dashboard)
4. [Navegação Principal](#4-navegação-principal)
5. [Dashboard Executivo](#5-dashboard-executivo)
6. [Transações](#6-transações)
7. [Investigação](#7-investigação)
8. [Revisão Manual](#8-revisão-manual)
9. [Calibração](#9-calibração)
10. [Monitoramento](#10-monitoramento)
11. [Métricas](#11-métricas)
12. [Alertas](#12-alertas)
13. [Interpretação de Indicadores](#13-interpretação-de-indicadores)
14. [Perguntas Frequentes (FAQ)](#14-perguntas-frequentes-faq)
15. [Solução de Problemas](#15-solução-de-problemas)
16. [Glossário](#16-glossário)

---

## 1. Introdução

### 1.1 O que é o Sankofa Enterprise Pro?

O Sankofa Enterprise Pro é uma plataforma de detecção de fraudes em tempo real que protege sua instituição financeira contra atividades fraudulentas. Utilizando inteligência artificial avançada, o sistema analisa cada transação em milissegundos e identifica comportamentos suspeitos antes que causem prejuízos.

### 1.2 Para quem é este manual?

Este manual foi desenvolvido para:

- **Analistas de Fraude**: Profissionais que revisam casos suspeitos diariamente
- **Gerentes de Operações**: Responsáveis por monitorar KPIs e ajustar configurações
- **Compliance Officers**: Profissionais que garantem conformidade regulatória

### 1.3 Benefícios do Sistema

| Benefício | Descrição |
|-----------|-----------|
| **Proteção em Tempo Real** | Análise de transações em menos de 15 milissegundos |
| **Alta Precisão** | 99.9% de acurácia na detecção de fraudes |
| **Baixos Falsos Positivos** | Menos de 1% de transações legítimas bloqueadas |
| **Conformidade Total** | Atende BACEN, LGPD e PCI-DSS |

---

## 2. Acessando o Sistema

### 2.1 Requisitos do Navegador

| Navegador | Versão Mínima |
|-----------|---------------|
| Google Chrome | 90+ |
| Mozilla Firefox | 88+ |
| Microsoft Edge | 90+ |
| Safari | 14+ |

### 2.2 Como Fazer Login

1. Abra o navegador e acesse o endereço do sistema
2. Na tela de login, insira suas credenciais:
   - **Usuário**: Seu e-mail corporativo
   - **Senha**: Sua senha de acesso
3. Clique em **"Entrar"**

```
┌─────────────────────────────────────────────────┐
│                                                 │
│              🦅 SANKOFA                         │
│              Análise de Risco                   │
│                                                 │
│         ┌─────────────────────────────┐         │
│         │ E-mail                      │         │
│         └─────────────────────────────┘         │
│                                                 │
│         ┌─────────────────────────────┐         │
│         │ Senha                       │         │
│         └─────────────────────────────┘         │
│                                                 │
│         ┌─────────────────────────────┐         │
│         │         ENTRAR              │         │
│         └─────────────────────────────┘         │
│                                                 │
│         Esqueci minha senha                     │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 2.3 Primeiro Acesso

No primeiro acesso, você será solicitado a:
1. Alterar sua senha temporária
2. Configurar autenticação de dois fatores (2FA)
3. Aceitar os termos de uso

---

## 3. Visão Geral do Dashboard

### 3.1 Layout Principal

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🦅 Sankofa   [🔍 Buscar transações, CPF, ID...]       🌙  🔔  👤 Analista   │
├───────────────┬─────────────────────────────────────────────────────────────┤
│               │                                                              │
│  📊 Dashboard │                     ÁREA DE CONTEÚDO                        │
│               │                                                              │
│  📋 Transações│                                                              │
│               │                                                              │
│  🔧 Calibragem│                                                              │
│               │                                                              │
│  🔍 Investigação                                                             │
│               │                                                              │
│  👁️ Revisão   │                                                              │
│    Manual     │                                                              │
│               │                                                              │
│  📈 Monitor   │                                                              │
│               │                                                              │
│  📊 Relatórios│                                                              │
│               │                                                              │
│  📉 Métricas  │                                                              │
│               │                                                              │
│  🔔 Alertas   │                                                              │
│               │                                                              │
│  ─────────────│                                                              │
│  Sankofa v11.0│                                                              │
└───────────────┴─────────────────────────────────────────────────────────────┘
```

### 3.2 Elementos da Interface

| Elemento | Descrição |
|----------|-----------|
| **Barra Superior** | Logo, busca global, modo escuro, notificações, perfil |
| **Menu Lateral** | Navegação entre as páginas do sistema |
| **Área de Conteúdo** | Conteúdo principal da página selecionada |
| **Badges (NEW, LIVE)** | Indicadores de funcionalidades novas ou em tempo real |

---

## 4. Navegação Principal

### 4.1 Menu do Sistema

| Página | Ícone | Descrição |
|--------|-------|-----------|
| **Dashboard** | 📊 | Visão geral e KPIs principais |
| **Transações** | 📋 | Lista e busca de transações |
| **Calibragem** | 🔧 | Ajuste de thresholds e regras |
| **Investigação** | 🔍 | Análise detalhada de fraudes |
| **Revisão Manual** | 👁️ | Fila de casos para revisão humana |
| **Monitoramento** | 📈 | Saúde dos modelos de IA |
| **Relatórios** | 📊 | Análises e métricas históricas |
| **Métricas** | 📉 | Contadores em tempo real |
| **Alertas** | 🔔 | Notificações e avisos do sistema |

### 4.2 Busca Global

A barra de busca no topo permite encontrar rapidamente:
- Transações por ID
- Clientes por CPF/CNPJ
- Casos por número de protocolo

**Dica**: Use Ctrl+K para acessar a busca rapidamente.

---

## 5. Dashboard Executivo

### 5.1 Visão Geral

O Dashboard é a página inicial do sistema e apresenta uma visão consolidada de todos os indicadores importantes.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Dashboard Executivo                                   │
│            Sistema Online   1 Algoritmo Ativo   Atualizado: 12:02:49        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐ │
│  │ Transações    │  │ Fraudes       │  │ Taxa de       │  │ Latência      │ │
│  │ Hoje          │  │ Detectadas    │  │ Aprovação     │  │ Média         │ │
│  │               │  │               │  │               │  │               │ │
│  │    15.432     │  │      23       │  │    98.5%      │  │    8.5ms      │ │
│  │    ↑ 12.5%    │  │    ↓ 5.2%     │  │    → 0.0%     │  │    ↓ 2.1%     │ │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐   │
│  │     Transações por Hora         │  │     Latência do Sistema         │   │
│  │                                 │  │                                 │   │
│  │  📈 [Gráfico de barras]         │  │  📉 [Gráfico de linha]          │   │
│  │                                 │  │                                 │   │
│  └─────────────────────────────────┘  └─────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Cartões de KPI

#### Transações Hoje
- **O que mostra**: Número total de transações processadas no dia
- **Comparativo**: Variação percentual em relação ao dia anterior
- **Cor verde**: Indica aumento (mais transações processadas)

#### Fraudes Detectadas
- **O que mostra**: Número de fraudes identificadas pelo sistema
- **Comparativo**: Variação em relação ao dia anterior
- **Cor vermelha/amarela**: Indica aumento de fraudes

#### Taxa de Aprovação
- **O que mostra**: Percentual de transações aprovadas automaticamente
- **Valor ideal**: > 95%
- **Alerta**: Se cair abaixo de 90%, verifique as regras

#### Latência Média
- **O que mostra**: Tempo médio de processamento por transação
- **Valor ideal**: < 15ms
- **Alerta**: Se ultrapassar 50ms, pode haver problema técnico

### 5.3 Gráficos

#### Transações por Hora
Mostra a distribuição de transações ao longo do dia, permitindo identificar:
- Horários de pico
- Padrões anormais
- Comparativo com média histórica

#### Latência do Sistema
Exibe a performance do sistema em tempo real:
- Linha azul: Latência atual
- Linha pontilhada: Limite aceitável (15ms)
- Área vermelha: Zona de alerta

---

## 6. Transações

### 6.1 Lista de Transações

A página de Transações permite visualizar e filtrar todas as transações processadas.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Transações                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Filtros:  [Canal ▼]  [Status ▼]  [Risco ▼]  [Data ▼]  [🔍 Buscar]          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ ID         │ Valor      │ Canal │ Status    │ Risco │ Data/Hora        ││
│  ├────────────┼────────────┼───────┼───────────┼───────┼──────────────────┤│
│  │ TXN-001    │ R$ 5.000   │ PIX   │ ✅ Aprovado│ BAIXO │ 27/11 14:30:00  ││
│  │ TXN-002    │ R$ 15.000  │ TED   │ ⏳ Revisão │ ALTO  │ 27/11 14:28:15  ││
│  │ TXN-003    │ R$ 50.000  │ PIX   │ ❌ Bloqueado│CRÍTICO│ 27/11 14:25:00  ││
│  │ TXN-004    │ R$ 2.500   │ PIX   │ ✅ Aprovado│ BAIXO │ 27/11 14:22:30  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  Exibindo 1-10 de 15.432 transações    [← Anterior] [1] [2] [3] [Próximo →] │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Filtros Disponíveis

| Filtro | Opções |
|--------|--------|
| **Canal** | PIX, TED, DOC, Cartão Crédito, Cartão Débito, Todos |
| **Status** | Aprovado, Bloqueado, Em Revisão, Todos |
| **Risco** | Baixo, Médio, Alto, Crítico, Todos |
| **Data** | Hoje, Últimos 7 dias, Últimos 30 dias, Personalizado |

### 6.3 Detalhes da Transação

Clique em uma transação para ver detalhes completos:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Detalhes da Transação TXN-002                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INFORMAÇÕES BÁSICAS                                                         │
│  ─────────────────────                                                       │
│  ID: TXN-002                    Canal: TED                                   │
│  Valor: R$ 15.000,00            Data: 27/11/2025 14:28:15                   │
│  Status: Em Revisão             Risco: ALTO (Score: 78)                      │
│                                                                              │
│  CLIENTE                                                                     │
│  ───────                                                                     │
│  Nome: Maria Silva              CPF: ***.***.789-00                          │
│  Cliente desde: 2020            Transações anteriores: 342                   │
│                                                                              │
│  DESTINATÁRIO                                                                │
│  ───────────                                                                 │
│  Nome: Empresa XYZ Ltda         CNPJ: **.***.567/0001-**                     │
│  Primeira transação: Sim        Categoria: Serviços                          │
│                                                                              │
│  RAZÕES DO ALERTA                                                            │
│  ────────────────                                                            │
│  ⚠ Horário incomum para este cliente (14:28 - fora do padrão habitual)      │
│  ⚠ Valor 3x maior que a média do cliente                                     │
│  ⚠ Novo destinatário                                                         │
│                                                                              │
│  [Ver Histórico Completo]  [Investigar]  [Aprovar]  [Bloquear]              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Investigação

### 7.1 Análise Profunda

A página de Investigação permite realizar análises detalhadas de casos suspeitos.

### 7.2 Ferramentas de Investigação

| Ferramenta | Descrição |
|------------|-----------|
| **Timeline** | Linha do tempo com todas as transações do cliente |
| **Mapa de Localização** | Visualização geográfica das transações |
| **Rede de Relacionamentos** | Conexões entre contas e destinatários |
| **Análise de Padrões** | Comparativo com comportamento histórico |

### 7.3 Como Investigar um Caso

1. Acesse a página **Investigação**
2. Selecione o caso a ser investigado
3. Revise a **Timeline** para entender a sequência de eventos
4. Verifique o **Mapa** para identificar anomalias geográficas
5. Analise a **Rede** para encontrar conexões suspeitas
6. Documente suas conclusões
7. Tome a decisão (Aprovar/Bloquear/Escalar)

---

## 8. Revisão Manual

### 8.1 Fila de Casos

A página de Revisão Manual exibe todos os casos que precisam de análise humana.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Revisão Manual                                       │
│                    Human-in-the-Loop Review                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Casos Pendentes: 12    Tempo Médio de Resolução: 3.5 min                   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Prioridade │ ID      │ Valor      │ Score │ Na Fila  │ SLA            ││
│  ├────────────┼─────────┼────────────┼───────┼──────────┼────────────────┤│
│  │ 🔴 CRÍTICO │ TXN-005 │ R$ 85.000  │  95   │ 00:30    │ ⚠ 0:30 restam ││
│  │ 🟠 ALTO    │ TXN-002 │ R$ 15.000  │  82   │ 02:15    │ ✓ 2:45 restam ││
│  │ 🟠 ALTO    │ TXN-003 │ R$ 22.000  │  78   │ 03:45    │ ✓ 1:15 restam ││
│  │ 🟡 MÉDIO   │ TXN-008 │ R$ 8.500   │  65   │ 08:20    │ ✓ 6:40 restam ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  [Pegar Próximo Caso]                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Níveis de Prioridade

| Prioridade | Cor | Score | SLA |
|------------|-----|-------|-----|
| **CRÍTICO** | 🔴 | 90-100 | 1 minuto |
| **ALTO** | 🟠 | 70-89 | 5 minutos |
| **MÉDIO** | 🟡 | 50-69 | 15 minutos |
| **BAIXO** | 🟢 | 30-49 | 30 minutos |

### 8.3 Processo de Revisão

1. Clique em **"Pegar Próximo Caso"** ou selecione um caso específico
2. O sistema exibe todas as informações relevantes
3. Analise os dados apresentados
4. Escolha uma das ações:
   - **Aprovar**: Libera a transação
   - **Bloquear**: Impede a transação
   - **Escalar**: Envia para supervisor
5. Digite uma justificativa (obrigatório)
6. Confirme a decisão

### 8.4 Dicas para Revisão Eficiente

- Comece pelos casos **CRÍTICOS** (SLA mais curto)
- Use atalhos de teclado: **A** (Aprovar), **B** (Bloquear), **E** (Escalar)
- Documente sempre a razão da sua decisão
- Em caso de dúvida, **escale** o caso

---

## 9. Calibração

### 9.1 Ajuste de Parâmetros

A página de Calibração permite ajustar os parâmetros do sistema de detecção.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Calibração                                        │
│                    Ajuste manual dos algoritmos                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  THRESHOLD DE RISCO                                                          │
│  ─────────────────                                                           │
│  Score para Revisão Manual: [70] ──────────●────────── [0-100]              │
│  Score para Bloqueio:       [90] ────────────────●──── [0-100]              │
│                                                                              │
│  LIMITES POR CANAL                                                           │
│  ─────────────────                                                           │
│  PIX - Limite Suspeito:    R$ [50.000,00]                                    │
│  TED - Limite Suspeito:    R$ [100.000,00]                                   │
│  Cartão - Limite Suspeito: R$ [10.000,00]                                    │
│                                                                              │
│  HORÁRIOS SUSPEITOS                                                          │
│  ─────────────────                                                           │
│  Início: [00:00]   Fim: [06:00]   Boost de Score: [+20]                     │
│                                                                              │
│  IMPACTO ESTIMADO (baseado nos últimos 30 dias)                             │
│  ────────────────                                                            │
│  • Transações para Revisão: 450 → 520 (+15.5%)                               │
│  • Falsos Positivos: 1.2% → 0.9% (-0.3%)                                     │
│  • Tempo Médio de Revisão: +2.5 min/dia                                      │
│                                                                              │
│  [Simular Alterações]  [Aplicar]  [Cancelar]                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Parâmetros Ajustáveis

| Parâmetro | Descrição | Impacto |
|-----------|-----------|---------|
| **Score para Revisão** | Limite mínimo para enviar à fila manual | Menor = mais revisões |
| **Score para Bloqueio** | Limite para bloqueio automático | Menor = mais bloqueios |
| **Limite por Canal** | Valor que aumenta o score de risco | Menor = mais alertas |
| **Horários Suspeitos** | Faixa horária com boost de score | Maior boost = mais alertas |

### 9.3 Boas Práticas

- Sempre **simule** antes de aplicar alterações
- Faça mudanças **graduais** (máximo 10% por vez)
- Monitore os resultados por **pelo menos 24h** após mudanças
- Documente o motivo de cada alteração

---

## 10. Monitoramento

### 10.1 Saúde do Modelo

A página de Monitoramento exibe informações sobre a saúde dos modelos de IA.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Monitoramento                                       │
│                     Saúde dos modelos de IA                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STATUS GERAL: ✅ SAUDÁVEL                                                   │
│                                                                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐ │
│  │ Acurácia      │  │ Precisão      │  │ Recall        │  │ F1-Score      │ │
│  │   99.9%       │  │   100%        │  │   96.7%       │  │   98.3%       │ │
│  │   ✓ OK        │  │   ✓ OK        │  │   ✓ OK        │  │   ✓ OK        │ │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘ │
│                                                                              │
│  DRIFT DETECTION                                                             │
│  ───────────────                                                             │
│  Data Drift:    ✅ Normal (PSI: 0.05)                                        │
│  Concept Drift: ✅ Normal (Accuracy stable)                                  │
│  Última verificação: 27/11/2025 12:00:00                                     │
│                                                                              │
│  VERSÃO DO MODELO                                                            │
│  ────────────────                                                            │
│  Versão atual: 1.0.0                                                         │
│  Data do treinamento: 15/11/2025                                             │
│  Próximo retreino: 15/12/2025 (agendado)                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Indicadores de Saúde

| Indicador | Valor Saudável | Alerta |
|-----------|---------------|--------|
| Acurácia | > 95% | < 90% |
| Precisão | > 95% | < 90% |
| Recall | > 90% | < 85% |
| PSI (Drift) | < 0.1 | > 0.25 |

### 10.3 O que fazer quando há alerta

1. **PSI Alto** (Drift detectado):
   - Verifique se houve mudança no perfil das transações
   - Comunique à equipe de Data Science
   - Pode ser necessário retreinar o modelo

2. **Acurácia baixa**:
   - Revise os últimos feedbacks (podem estar incorretos)
   - Verifique se há mudança no padrão de fraudes
   - Comunique à equipe técnica

---

## 11. Métricas

### 11.1 Métricas em Tempo Real

A página de Métricas exibe contadores e estatísticas em tempo real.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Métricas                                         │
│                   Contadores e métricas em tempo real                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PERFORMANCE DO SISTEMA                                                      │
│  ─────────────────────                                                       │
│  TPS atual:          1.250 transações/segundo                                │
│  Latência P50:       5.2ms                                                   │
│  Latência P95:       9.8ms                                                   │
│  Latência P99:       11.2ms                                                  │
│  Taxa de erros:      0.01%                                                   │
│                                                                              │
│  TRANSAÇÕES HOJE                                                             │
│  ────────────────                                                            │
│  Total processado:   45.832                                                  │
│  Aprovadas:          44.912 (98.0%)                                          │
│  Bloqueadas:         287 (0.6%)                                              │
│  Em revisão:         633 (1.4%)                                              │
│                                                                              │
│  VALOR PROTEGIDO                                                             │
│  ───────────────                                                             │
│  Fraudes evitadas:   R$ 2.345.678,00                                         │
│  Este mês:           R$ 15.892.340,00                                        │
│  Este ano:           R$ 187.543.210,00                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Alertas

### 12.1 Central de Alertas

A página de Alertas exibe todas as notificações do sistema.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Alertas                                          │
│                    Alertas e notificações                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  🔴 CRÍTICOS (2)   🟠 ALTOS (5)   🟡 MÉDIOS (12)   🟢 BAIXOS (28)            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 🔴 Pico de transações suspeitas detectado                               ││
│  │    Há 5 minutos • Sistema                                                ││
│  │    15 transações de alto risco nos últimos 10 minutos                   ││
│  │    [Investigar] [Ignorar]                                               ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │ 🟠 SLA de revisão em risco                                              ││
│  │    Há 12 minutos • Operações                                            ││
│  │    3 casos próximos de estourar o SLA                                   ││
│  │    [Ver Fila] [Ignorar]                                                 ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │ 🟡 Novo padrão de fraude identificado                                   ││
│  │    Há 1 hora • Machine Learning                                         ││
│  │    Sistema identificou novo padrão em cartões de crédito                ││
│  │    [Ver Detalhes] [Marcar como Lido]                                    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 12.2 Tipos de Alertas

| Tipo | Cor | Ação Requerida |
|------|-----|----------------|
| **CRÍTICO** | 🔴 | Ação imediata necessária |
| **ALTO** | 🟠 | Ação em até 30 minutos |
| **MÉDIO** | 🟡 | Ação em até 4 horas |
| **BAIXO** | 🟢 | Informativo |

---

## 13. Interpretação de Indicadores

### 13.1 Score de Risco

O score de risco varia de 0 a 100:

```
0 ─────────────────────────────────────────────────────────────────────── 100
     BAIXO      MÉDIO-BAIXO     MÉDIO         ALTO         CRÍTICO
    (0-30)       (31-50)       (51-70)       (71-90)       (91-100)
    
   ✅ Aprovar   ✅ Aprovar    ⚠ Monitorar   ⏳ Revisar    ❌ Bloquear
      Auto         Auto                       Manual          Auto
```

### 13.2 Cores no Dashboard

| Cor | Significado |
|-----|-------------|
| 🟢 Verde | Positivo / Normal / OK |
| 🟡 Amarelo | Atenção / Alerta leve |
| 🟠 Laranja | Importante / Risco elevado |
| 🔴 Vermelho | Crítico / Ação imediata |
| 🔵 Azul | Informativo / Neutro |

### 13.3 Setas de Tendência

| Símbolo | Significado |
|---------|-------------|
| ↑ Seta verde | Aumento positivo |
| ↓ Seta verde | Diminuição positiva |
| ↑ Seta vermelha | Aumento negativo |
| ↓ Seta vermelha | Diminuição negativa |
| → Seta cinza | Estável |

---

## 14. Perguntas Frequentes (FAQ)

### P: O que fazer quando vejo muitos falsos positivos?

**R:** Se você notar muitas transações legítimas sendo bloqueadas:
1. Acesse a página de **Calibração**
2. Aumente levemente o **Score para Bloqueio** (ex: de 90 para 92)
3. **Simule** o impacto antes de aplicar
4. Monitore por 24-48 horas após a mudança

### P: Como sei se o modelo está funcionando corretamente?

**R:** Verifique a página de **Monitoramento**:
- Acurácia > 95%
- PSI < 0.1 (sem drift)
- Status: SAUDÁVEL

### P: Posso aprovar uma transação bloqueada?

**R:** Não diretamente. Transações bloqueadas automaticamente precisam de:
1. Análise na página de **Investigação**
2. Documentação da justificativa
3. Aprovação do supervisor (se necessário)
4. Liberação via processo específico

### P: Com que frequência os dados são atualizados?

**R:** Depende da página:
- **Dashboard**: A cada 5 segundos
- **Métricas**: Em tempo real
- **Monitoramento**: A cada 30 segundos
- **Transações**: On-demand (ao navegar)

### P: O que significa PSI no Monitoramento?

**R:** PSI (Population Stability Index) mede se as transações estão mudando:
- < 0.1: Normal
- 0.1 - 0.25: Mudança leve
- > 0.25: Mudança significativa (alerta)

---

## 15. Solução de Problemas

### 15.1 Problemas Comuns

| Problema | Causa Provável | Solução |
|----------|----------------|---------|
| Dashboard não carrega | Problema de conexão | Verifique sua internet, atualize a página |
| Dados desatualizados | Cache do navegador | Limpe o cache ou use Ctrl+F5 |
| Lentidão no sistema | Pico de processamento | Aguarde alguns minutos |
| Erro de login | Credenciais expiradas | Redefina sua senha |
| Gráficos vazios | Sem dados no período | Verifique os filtros de data |

### 15.2 Contato com Suporte

Se o problema persistir:

1. **Nível 1 (Help Desk)**: suporte@sankofa.com.br
2. **Nível 2 (Técnico)**: tecnico@sankofa.com.br
3. **Emergências**: 0800-XXX-XXXX (24/7)

Ao contatar o suporte, informe:
- Descrição do problema
- Horário em que ocorreu
- Página/funcionalidade afetada
- Screenshot do erro (se possível)

---

## 16. Glossário

| Termo | Definição |
|-------|-----------|
| **Acurácia** | Percentual de previsões corretas do modelo |
| **Drift** | Mudança no padrão de dados que pode degradar o modelo |
| **Falso Positivo** | Transação legítima incorretamente classificada como fraude |
| **Falso Negativo** | Fraude que não foi detectada pelo sistema |
| **F1-Score** | Média harmônica entre precisão e recall |
| **HITL** | Human-in-the-Loop (revisão manual por analista) |
| **KPI** | Key Performance Indicator (indicador-chave) |
| **Latência** | Tempo de resposta do sistema |
| **P95/P99** | Percentil 95/99 (latência de 95%/99% das transações) |
| **PIX** | Sistema de pagamentos instantâneos do Brasil |
| **Precisão** | Percentual de alertas que são realmente fraudes |
| **PSI** | Population Stability Index (mede drift) |
| **Recall** | Percentual de fraudes que foram detectadas |
| **Score** | Pontuação de risco (0-100) |
| **SLA** | Service Level Agreement (tempo máximo para ação) |
| **TPS** | Transações por segundo |

---

**Dúvidas? Acesse a Central de Ajuda ou contate o suporte.**

**Documento mantido por:** Equipe de Treinamento Sankofa  
**Última atualização:** Novembro 2025  
**Versão:** 1.0.0
