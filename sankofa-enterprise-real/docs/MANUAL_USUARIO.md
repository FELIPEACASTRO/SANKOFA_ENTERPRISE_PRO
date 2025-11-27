# Manual do Usuário - Sankofa Enterprise Pro v11.0
## Guia Completo para Analistas de Fraude

**Versão:** 11.0  
**Última Atualização:** 27 de Novembro de 2025  
**Público:** Analistas de Fraude, Gerentes de Operações, Compliance Officers

---

## Bem-vindo ao Sankofa! 👋

Este manual vai te guiar passo a passo no uso do sistema de detecção de fraudes mais avançado do mercado. Não se preocupe se você não é técnico - este guia foi feito pensando em você!

---

## Índice

1. [Primeiros Passos](#1-primeiros-passos)
2. [Conhecendo o Dashboard](#2-conhecendo-o-dashboard)
3. [Analisando Transações](#3-analisando-transações)
4. [Investigando Fraudes](#4-investigando-fraudes)
5. [Revisão Manual](#5-revisão-manual)
6. [Calibrando o Sistema](#6-calibrando-o-sistema)
7. [Monitorando a Saúde](#7-monitorando-a-saúde)
8. [Gerando Relatórios](#8-gerando-relatórios)
9. [Entendendo os Alertas](#9-entendendo-os-alertas)
10. [Dicas e Truques](#10-dicas-e-truques)
11. [Perguntas Frequentes](#11-perguntas-frequentes)
12. [Glossário](#12-glossário)

---

## 1. Primeiros Passos

### 1.1 Como Acessar

1. Abra seu navegador (Chrome, Firefox, Edge ou Safari)
2. Digite o endereço do sistema na barra de endereços
3. Você verá a tela inicial do Sankofa

### 1.2 Navegadores Suportados

| Navegador | Versão Mínima | Recomendado |
|-----------|---------------|-------------|
| Chrome | 90+ | ✅ Sim |
| Firefox | 88+ | ✅ Sim |
| Edge | 90+ | ✅ Sim |
| Safari | 14+ | ⚠️ OK |

### 1.3 Primeira Coisa que Você Verá

Ao acessar, você cairá direto no **Dashboard Executivo** - a central de comando do sistema:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🦅 Sankofa   [🔍 Buscar...]                    🌙  🔔(3)  👤 Analista  │
├────────────────┬────────────────────────────────────────────────────────┤
│                │                                                         │
│  📊 Dashboard  │              Dashboard Executivo                       │
│  ◀ selecionado │                                                         │
│                │         Sistema Online   1 Algoritmo Ativo             │
│  📋 Transações │                                                         │
│                │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  ⚙️ Calibragem │   │   518   │  │   23    │  │  95.6%  │  │ 33.50ms │  │
│                │   │Transações│  │ Fraudes │  │Aprovação│  │Latência │  │
│  🔍 Investigação│   └─────────┘  └─────────┘  └─────────┘  └─────────┘  │
│                │                                                         │
│  👁️ Revisão    │                                                         │
│                │                                                         │
│  📈 Monitoramento                                                        │
│                │                                                         │
│  📊 Relatórios │                                                         │
│                │                                                         │
│  📉 Métricas   │                                                         │
│                │                                                         │
│  🔔 Alertas    │                                                         │
│                │                                                         │
│  ──────────────│                                                         │
│  Sankofa v11.0 │                                                         │
└────────────────┴────────────────────────────────────────────────────────┘
```

---

## 2. Conhecendo o Dashboard

### 2.1 O Que Significam os Números?

#### 📊 Transações Hoje
**O que é:** Quantas transações passaram pelo sistema hoje.
**Por que importa:** Mostra o volume de trabalho do dia.
**Normal:** Varia conforme o dia, mas geralmente entre 10.000-50.000.

#### 🛡️ Fraudes Detectadas
**O que é:** Quantas transações o sistema identificou como suspeitas.
**Por que importa:** Se aumentar muito, pode indicar ataque.
**Normal:** Geralmente 2-5% do total de transações.

#### ✅ Taxa de Aprovação
**O que é:** Percentual de transações aprovadas automaticamente.
**Por que importa:** Se cair muito, pode estar bloqueando clientes legítimos.
**Normal:** Deve ficar acima de 95%.

#### ⏱️ Latência Média
**O que é:** Quanto tempo o sistema leva para analisar uma transação.
**Por que importa:** Se aumentar muito, pode atrasar pagamentos.
**Normal:** Menos de 50ms (0.05 segundos).

### 2.2 As Cores dos Indicadores

| Cor | Significado | Ação |
|-----|-------------|------|
| 🟢 Verde | Tudo normal | Nenhuma |
| 🟡 Amarelo | Atenção | Monitorar |
| 🔴 Vermelho | Problema | Investigar imediatamente |

### 2.3 Os Gráficos

**Transações por Hora:** Mostra quando o sistema está mais ocupado.
- Picos de manhã e tarde são normais
- Picos de madrugada podem indicar fraude

**Latência do Sistema:** Mostra a velocidade do sistema ao longo do dia.
- Linha azul: latência atual
- Se subir muito, pode haver problema técnico

---

## 3. Analisando Transações

### 3.1 Acessando a Lista

1. Clique em **📋 Transações** no menu lateral
2. Você verá uma lista com todas as transações do dia

### 3.2 Entendendo a Lista

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Transações                                    │
│        Lista e busca de transações processadas em tempo real             │
├─────────────────────────────────────────────────────────────────────────┤
│  🔍 Filtros                                                              │
│  [Buscar: ID, CPF, cidade...]   [Status: Todos ▼]   [Tipo: Todos ▼]    │
├─────────────────────────────────────────────────────────────────────────┤
│  Mostrando 50 de 250 transações                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  ID                    │ Valor      │ Tipo   │ Canal │ Local   │ Data   │
├────────────────────────┼────────────┼────────┼───────┼─────────┼────────┤
│  TXN1764254880868000   │ R$ 1.234   │  PIX   │  TED  │São Paulo│ 14:48  │
│  TXN1764254880604000   │ -R$ 100    │CREDITO │  PIX  │Rio de J.│ 14:48  │
│  ...                   │            │        │       │         │        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Os Tipos de Transação

| Tipo | O que é |
|------|---------|
| **PIX** | Pagamento instantâneo (mais comum hoje) |
| **TED** | Transferência bancária tradicional |
| **CREDITO** | Compra no cartão de crédito |
| **DEBITO** | Compra no cartão de débito |

### 3.4 Filtrando Transações

**Por Status:**
- **Todos:** Mostra tudo
- **Aprovado:** Transações que passaram
- **Bloqueado:** Transações barradas pelo sistema
- **Revisão:** Aguardando análise humana

**Por Tipo:**
- Selecione PIX, TED, CREDITO ou DEBITO

**Por Busca:**
- Digite o ID da transação
- Digite o CPF do cliente
- Digite a cidade

---

## 4. Investigando Fraudes

### 4.1 Quando Investigar?

Você deve investigar quando:
- Receber um alerta de fraude
- Ver uma transação com score alto
- Cliente reclamar de bloqueio indevido

### 4.2 Acessando a Central de Investigação

1. Clique em **🔍 Investigação** no menu
2. Você verá os casos que precisam de atenção

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Central de Investigação                               │
│            Análise detalhada de fraudes e casos suspeitos                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │   0      │  │    0     │  │    0     │  │    0%    │                │
│  │  Casos   │  │    Em    │  │Resolvidos│  │  Taxa de │                │
│  │  Ativos  │  │Investigação         │  │ Resolução│                │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                │
├─────────────────────────────────────────────────────────────────────────┤
│  [Buscar investigações...]   [Todos os Status ▼]                        │
│                              [Todas as Prioridades ▼]                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│        🛡️ Nenhuma investigação encontrada                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 O Que Analisar em um Caso

1. **Valor da transação:** É compatível com o perfil do cliente?
2. **Horário:** O cliente costuma transacionar nesse horário?
3. **Local:** A transação foi feita de onde o cliente mora?
4. **Histórico:** O cliente já fez transações similares?
5. **Razões do alerta:** Por que o sistema flagrou?

### 4.4 Tomando uma Decisão

Após analisar, você deve:

| Decisão | Quando Usar | O Que Acontece |
|---------|-------------|----------------|
| **Confirmar Fraude** | Quando tem certeza que é fraude | Transação é bloqueada permanentemente |
| **Falso Positivo** | Quando a transação é legítima | Libera o cliente, sistema aprende |
| **Escalar** | Quando tem dúvida | Vai para supervisor |

---

## 5. Revisão Manual

### 5.1 O Que é a Revisão Manual?

Algumas transações ficam na "zona cinza" - não são claramente fraude nem claramente legítimas. Essas vão para a fila de revisão manual, onde você decide.

### 5.2 Acessando a Fila

1. Clique em **👁️ Revisão Manual** no menu
2. Você verá todas as transações aguardando

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  Revisão Manual (Human-in-the-Loop)                      │
│       Sistema de revisão manual para transações flagadas                 │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │   0      │  │    0     │  │    0     │  │    0     │                │
│  │  Total   │  │Pendentes │  │Completadas│ │ Expiradas│                │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                │
├─────────────────────────────────────────────────────────────────────────┤
│       Transações para Revisão (0)                                        │
│  ─────────────────────────────────────────────────────────────────────   │
│  │ ID │ VALOR │ CPF │ RISCO │ STATUS │ AÇÕES │                          │
│  ─────────────────────────────────────────────────────────────────────   │
│                                                                          │
│        Nenhuma transação pendente de revisão manual.                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Prioridades

| Cor | Prioridade | SLA | O Que Fazer |
|-----|------------|-----|-------------|
| 🔴 | CRÍTICO | 1 min | Resolver imediatamente! |
| 🟠 | ALTO | 5 min | Priorizar |
| 🟡 | MÉDIO | 15 min | Resolver quando possível |
| 🟢 | BAIXO | 30 min | Pode aguardar |

### 5.4 Como Revisar

1. Clique na transação para ver detalhes
2. Analise as informações apresentadas
3. Clique em **Aprovar** ou **Rejeitar**
4. Digite uma justificativa (obrigatório)
5. Confirme sua decisão

**⚡ Dica:** Use atalhos de teclado!
- **A** = Aprovar
- **B** = Bloquear
- **E** = Escalar

---

## 6. Calibrando o Sistema

### 6.1 O Que é Calibragem?

Calibragem é ajustar a "sensibilidade" do sistema. Se está bloqueando muitos clientes legítimos, você pode diminuir a sensibilidade. Se está deixando passar fraudes, pode aumentar.

### 6.2 Acessando a Calibragem

1. Clique em **⚙️ Calibragem** no menu
2. Você verá os controles de ajuste

### 6.3 Os Controles Disponíveis

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Calibragem Manual                                 │
│          Ajuste em tempo real dos parâmetros dos algoritmos             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [Aplicar Mudanças ao Motor]   [Resetar Padrões]   [Ver Histórico]      │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ Tier 1 - Velocistas │ Tier 2 - Rápidos │ Tier 3 - Avançados │ ...  ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │ Motor de Regras │  │ Listas Negras   │  │ Velocidade      │         │
│  │ Básicas      🔘 │  │              🔘 │  │              🔘 │         │
│  │                 │  │                 │  │                 │         │
│  │ Threshold: 80%  │  │ Threshold: 100% │  │ Threshold: 70%  │         │
│  │ ────●────────── │  │ ──────────────● │  │ ──────●──────── │         │
│  │                 │  │                 │  │                 │         │
│  │ Peso: 0.150     │  │ Peso: 0.200     │  │ Peso: 0.120     │         │
│  │ ●─────────────  │  │ ─●────────────  │  │ ●─────────────  │         │
│  │                 │  │                 │  │                 │         │
│  │ Valor Máx:      │  │ Cache: 300s     │  │ Janela: 3600s   │         │
│  │ R$ 50.000       │  │                 │  │                 │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.4 O Que Cada Controle Faz

| Controle | O Que Faz | Aumentar | Diminuir |
|----------|-----------|----------|----------|
| **Threshold** | Limite para considerar fraude | Menos bloqueios | Mais bloqueios |
| **Peso** | Importância do algoritmo | Mais influente | Menos influente |
| **Valor Máx** | Limite de aprovação auto | Mais aprovações | Menos aprovações |
| **Janela** | Tempo de análise | Análise mais longa | Análise mais curta |

### 6.5 Cuidados ao Calibrar

⚠️ **ATENÇÃO:**
- Mudanças afetam TODAS as transações
- Sempre teste antes em ambiente seguro
- Documente suas alterações
- Monitore os resultados após mudanças

---

## 7. Monitorando a Saúde

### 7.1 Para Que Serve?

A página de Monitoramento mostra se o sistema está funcionando bem. É como o "painel de instrumentos" de um carro.

### 7.2 Acessando o Monitor

1. Clique em **📈 Monitoramento** no menu
2. Você verá o status de todos os componentes

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Monitoramento do Sistema                            │
│           Saúde dos modelos de IA e performance em tempo real           │
├─────────────────────────────────────────────────────────────────────────┤
│                                      [Auto-refresh ON]   [Atualizar]    │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ Status   │  │ Modelos  │  │Trans/seg │  │  Tempo   │                │
│  │ ✅       │  │    5     │  │   127    │  │  0.15s   │                │
│  │ Saudável │  │  Ativos  │  │          │  │ Resposta │                │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │  Taxa    │  │  Falsos  │  │Processadas│ │  Uptime  │                │
│  │ Detecção │  │Positivos │  │   Hoje   │  │          │                │
│  │  94.2%   │  │   2.1%   │  │  15.420  │  │ 15d 8h   │                │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                │
│                                                                          │
│                     Recursos do Sistema                                  │
│           Monitoramento em tempo real dos recursos                       │
│                                                                          │
│  [💻 CPU]  [💾 Memória]  [📀 Disco]  [🌐 Rede]                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.3 O Que Cada Indicador Significa

| Indicador | Bom | Atenção | Crítico |
|-----------|-----|---------|---------|
| **Status Geral** | Saudável ✅ | Degradado ⚠️ | Crítico 🔴 |
| **Modelos Ativos** | 5 | 3-4 | <3 |
| **Trans/seg** | >100 | 50-100 | <50 |
| **Tempo Resposta** | <0.15s | 0.15-0.5s | >0.5s |
| **Taxa Detecção** | >90% | 80-90% | <80% |
| **Falsos Positivos** | <3% | 3-5% | >5% |

---

## 8. Gerando Relatórios

### 8.1 Tipos de Relatório

| Relatório | Para Quê | Tempo |
|-----------|----------|-------|
| **Mensal de Fraudes** | Resumo do mês para diretoria | 5-10 min |
| **Performance Trimestral** | Avaliação de performance | 3-5 min |
| **Análise de Tendências** | Identificar padrões | 7-12 min |
| **Impacto Financeiro** | Calcular economia | 4-8 min |

### 8.2 Como Gerar

1. Clique em **📊 Relatórios** no menu
2. Escolha o template desejado
3. Configure o período
4. Clique em **Gerar Relatório**
5. Aguarde a conclusão
6. Faça download do arquivo

---

## 9. Entendendo os Alertas

### 9.1 Tipos de Alerta

| Tipo | Severidade | Exemplo |
|------|------------|---------|
| 🔴 **Crítico** | Requer ação imediata | Sistema fora do ar |
| 🟠 **Alto** | Investigar hoje | Pico de fraudes |
| 🟡 **Médio** | Monitorar | Latência elevada |
| 🔵 **Baixo** | Informativo | Novo modelo disponível |

### 9.2 Acessando os Alertas

1. Clique em **🔔 Alertas** no menu
2. Ou clique no ícone de sino no topo (🔔)

### 9.3 Gerenciando Alertas

- **Novo:** Acabou de chegar
- **Em Investigação:** Alguém está olhando
- **Resolvido:** Problema corrigido

---

## 10. Dicas e Truques

### 10.1 Atalhos de Teclado

| Atalho | Ação |
|--------|------|
| `Ctrl + K` | Abrir busca rápida |
| `A` | Aprovar transação (em revisão) |
| `B` | Bloquear transação (em revisão) |
| `E` | Escalar caso |
| `R` | Atualizar página |

### 10.2 Modo Escuro

Clique no ícone 🌙 no topo para alternar entre modo claro e escuro.

### 10.3 Notificações

O sino 🔔 mostra quantos alertas novos você tem. Clique para ver.

### 10.4 Boas Práticas

1. **Comece pelo Dashboard:** Veja a situação geral antes de mergulhar nos detalhes
2. **Priorize os críticos:** Sempre resolva alertas vermelhos primeiro
3. **Documente suas decisões:** Escreva justificativas claras
4. **Monitore após mudanças:** Sempre verifique se calibrações surtiram efeito
5. **Escale quando em dúvida:** Melhor perguntar que errar

---

## 11. Perguntas Frequentes

### "Por que uma transação legítima foi bloqueada?"

O sistema analisa padrões. Se o cliente fez algo diferente do normal (compra maior, horário diferente, local novo), pode ser bloqueado por segurança. Você pode aprovar na Revisão Manual.

### "O score de risco mudou sem eu fazer nada?"

O sistema aprende continuamente. Conforme recebe feedback, os modelos se ajustam. Isso é normal e esperado.

### "Como sei se o sistema está funcionando?"

Vá em Monitoramento. Se o status estiver "Saudável" com check verde, está tudo ok.

### "Posso desfazer uma aprovação/bloqueio?"

Não diretamente. Mas você pode registrar feedback para o sistema aprender.

### "O que fazer se o sistema ficar lento?"

1. Verifique a página de Monitoramento
2. Procure alertas sobre performance
3. Se persistir, contate o suporte

---

## 12. Glossário

| Termo | O Que Significa |
|-------|-----------------|
| **Score** | Nota de 0 a 100 que indica chance de fraude |
| **Threshold** | Limite - acima dele, é considerado fraude |
| **Latência** | Tempo que o sistema leva para responder |
| **Falso Positivo** | Transação legítima marcada como fraude |
| **HITL** | Human-in-the-Loop - quando você decide |
| **Ensemble** | Combinação de vários modelos de IA |
| **Drift** | Quando o modelo fica "desatualizado" |
| **SLA** | Tempo máximo para resolver algo |
| **PIX** | Sistema de pagamento instantâneo do Banco Central |
| **SHAP** | Técnica que explica "por que" o sistema decidiu algo |

---

## Precisa de Ajuda?

Se tiver dúvidas ou problemas:

1. **Consulte este manual** - A maioria das respostas está aqui
2. **Verifique os alertas** - Pode haver um aviso relevante
3. **Pergunte ao colega** - Alguém pode já ter passado por isso
4. **Contate o suporte** - Para problemas técnicos

---

*Manual do Usuário - Sankofa Enterprise Pro v11.0*  
*Última atualização: 27 de Novembro de 2025*
