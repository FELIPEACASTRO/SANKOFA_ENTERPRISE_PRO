import { useState } from 'react';
import { ChevronDown, ChevronUp, BookOpen, Home } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';

const manualContent = {
  introducao: {
    title: '🎯 Bem-vindo ao Manual do Sankofa',
    sections: [
      {
        id: 'intro-main',
        title: 'O Que é o Sankofa?',
        content: `
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║               🚀 SANKOFA - PROTEÇÃO EM TEMPO REAL              ║
║                                                                ║
║   Detecta fraudes bancárias em milissegundos usando IA         ║
║                                                                ║
║   PIX • CARTÃO • TED • BOLETO                                  ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

**O Que Significa Sankofa?**
Vem de um provérbio africano: "Voltar para buscar". Simboliza aprender com o passado para construir um futuro melhor. É exatamente o que fazemos aqui: analisamos padrões históricos de fraude para proteger suas transações AGORA.

**Como Funciona?**

        Transação Chega
              ↓
        ┌─────────────┐
        │   Sankofa   │  Analisa 40+ características
        │  Algoritmo  │  em milissegundos
        └─────────────┘
              ↓
        ┌─────────────────────┐
        │ FRAUDE? ou LEGÍTIMA?│
        └─────────────────────┘
              ↓         ↓
           ✅ OK    ⚠️ ALERTAR

**Qual é Seu Papel?**
- Monitorar o sistema (Dashboard)
- Validar decisões questionáveis (Revisão Manual)
- Ensinar ao sistema quando erra (Feedback)
- Configurar regras específicas do seu banco (Calibragem)
- Investigar fraudes confirmadas (Investigação)

**Resultado?**
💰 Você protege milhões em reais
🛡️ Seus clientes dormem tranquilos
📊 Sua instituição cumpre compliance`
      },
      {
        id: 'mapa-visual',
        title: '🗺️ Mapa Visual do Sistema',
        content: `
O Sankofa é organizado em 4 áreas estratégicas:

┌─────────────────────────────────────────────────────────┐
│                                                         │
│  📊 ANÁLISE EM TEMPO REAL                               │
│  ├─ Dashboard: Seu painel de controle                   │
│  ├─ Transações: Busque qualquer operação                │
│  └─ Alertas: Notificações de fraudes                    │
│                                                         │
│  🔍 INVESTIGAÇÃO E REVISÃO                              │
│  ├─ Investigação: Analise profunda de fraudes           │
│  ├─ Revisão Manual: Valide as decisões do sistema       │
│  └─ Feedback: Treine o modelo com sua experiência       │
│                                                         │
│  ⚙️ CONFIGURAÇÃO E CONTROLE                             │
│  ├─ Calibragem: Ajuste a sensibilidade                  │
│  ├─ Regras Duras: Bloqueios automáticos                 │
│  ├─ Lista VIP: Clientes confiáveis (aprovação direta)   │
│  └─ Lista HOT: Bloqueados (rejeição automática)         │
│                                                         │
│  📈 OBSERVABILIDADE E CONFORMIDADE                      │
│  ├─ Métricas: Números do sistema em tempo real          │
│  ├─ Monitoramento: Saúde dos modelos de IA              │
│  ├─ Relatórios: Análises para gerência                  │
│  ├─ Datasets: Catálogo de dados disponíveis             │
│  └─ Auditoria: Registro de TUDO (LGPD)                  │
│                                                         │
└─────────────────────────────────────────────────────────┘

**Dica de Ouro:** Sempre comece no Dashboard para ter visão geral!`
      }
    ]
  },
  telas: {
    title: '📚 Guia Completo das Telas',
    sections: [
      {
        id: 'dashboard',
        title: '📊 Dashboard - Seu Painel de Controle',
        content: `
**Aonde Encontrar:** Menu principal (primeiro item)

**O Que Você Vê:**

╔═══════════════════════════════════════════════════════╗
║              DASHBOARD SANKOFA                        ║
╠═══════════════════════════════════════════════════════╣
║                                                       ║
║  ┌─────────────┬─────────────┬─────────────────┐    ║
║  │ Transações  │   Fraudes   │  Taxa de Fraude │    ║
║  │   4.466     │   3.114     │      69,7%      │    ║
║  └─────────────┴─────────────┴─────────────────┘    ║
║                                                       ║
║  ┌────────────────────────────────────────────┐     ║
║  │ Valor Protegido Hoje: R$ 14.328.997,85     │     ║
║  └────────────────────────────────────────────┘     ║
║                                                       ║
║  📈 Gráfico de Série Temporal (últimas 24h)         ║
║  ─── Mostra evolução de fraudes ao longo do dia      ║
║                                                       ║
║  🎯 Distribuição por Canal                          ║
║  ├─ PIX: 4.285 txns  (95%)                          ║
║  ├─ TED: 86 txns     (2%)                           ║
║  └─ BOLETO: 88 txns  (3%)                           ║
║                                                       ║
║  🚨 Alertas Recentes                                ║
║  └─ Últimas fraudes críticas                        ║
║                                                       ║
║  🤖 Status dos Modelos de IA                        ║
║  ├─ Random Forest: ✅ Online                        ║
║  ├─ Gradient Boosting: ✅ Online                    ║
║  └─ CatBoost: ✅ Online                             ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝

**Para Que Serve?**
É como o painel de bordo de um avião. Você vê TUDO em uma tela:
- Quantas fraudes entraram hoje
- Qual foi a hora de pico
- Se há alguma anomalia
- Se os algoritmos estão OK

**Quando Usar?**
- Começo do turno: saber o cenário da noite/madrugada
- Operações críticas: verificar SLA em tempo real
- Antes de tomar decisões: validar contexto do sistema

**Elementos Principais:**

🔢 **KPIs (Indicadores Principais)**
Números maiores na parte superior. Mostram totalizadores do dia.
- Total de Transações: quanto entrou
- Fraudes Detectadas: quantas eram suspeitas
- Taxa de Fraude: percentual de fraudes
- Valor Protegido: quanto dinheiro foi salvo

📈 **Série Temporal**
Gráfico de linha mostrando ao longo do dia como subiu/desceu o número de fraudes.
→ Se sobe muito rápido: pode ser ataque
→ Se cai: período calmo
→ Se estável: padrão normal

🍰 **Gráfico por Canal**
Pizza mostrando como ficou distribuído:
- PIX domina (transferências instantâneas = maior risco)
- Cartão tem risco médio
- TED tem menor volume
- Boleto é tradicional

🚨 **Alertas Recentes**
Últimos avisos do sistema. Se vê algo vermelho aqui, investigue!

🤖 **Status dos Modelos**
Mostra se os 3 algoritmos que rodam em paralelo estão vivos:
- Todos online = sistema rodando bem
- Um offline = sistema segue, mas com 2 modelos

**Como Usar:**
1. Abra o Dashboard
2. Olhe os KPIs (números)
3. Se algo estranho, clique em um alerta para investigar
4. Verifique status dos modelos (todos devem estar verdes)
5. Monitore a série temporal (detecte picos)

**Dica Importante:**
O Dashboard atualiza a cada 30 segundos automaticamente. Não precisa atualizar manualmente. Ideal para deixar em um monitor durante todo o turno!

**Cuidado:**
Não tome decisões baseado APENAS no Dashboard. Sempre investigue antes de fazer calibragens ou bloqueios.`
      },
      {
        id: 'transacoes',
        title: '💳 Transações - Busque Qualquer Operação',
        content: `
**Aonde Encontrar:** Menu > Transações

**O Que Você Vê:**

╔════════════════════════════════════════════════════════════════╗
║               TELA DE TRANSAÇÕES                              ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  🔍 FILTROS NO TOPO                                           ║
║  ┌────────────────────────────────────────────────────────┐  ║
║  │ Buscar CPF: [_______] | Status: [Todos ▼]             │  ║
║  │ Data De: [__/__/__] Até: [__/__/__]                    │  ║
║  │ Canal: [Todos ▼] | Período: [Últimas 24h ▼]           │  ║
║  │ [ 🔍 Buscar ] [ ⬇️ Exportar ]                          │  ║
║  └────────────────────────────────────────────────────────┘  ║
║                                                                ║
║  📋 TABELA DE RESULTADOS (50 por página)                    ║
║  ┌──────────────────────────────────────────────────────┐   ║
║  │ ID │ CPF │ Valor │ Data │ Canal │ Status │ Score │...│  ║
║  ├──────────────────────────────────────────────────────┤   ║
║  │ 1  │***.***.789-01 │ R$ 1.500 │ 14:32 │ PIX   │ ✅    │ 15│  ║
║  │ 2  │***.***.345-67 │ R$ 50.000│ 14:35 │ TED   │ ⚠️    │ 72│  ║
║  │ 3  │***.***.123-45 │ R$ 200   │ 14:38 │ PIX   │ ❌ 🔴 │ 95│  ║
║  │    │ ... mais 47 linhas ...                          │   ║
║  └──────────────────────────────────────────────────────┘   ║
║                                                                ║
║  Página 1 de 89 | [ ◀ ] [ ▶ ]                               ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

**Para Que Serve?**
Encontrar uma transação específica que um cliente reclamou, ou buscar padrões. É seu "banco de dados visual".

**Quando Usar?**

Cenário 1: Cliente ligou reclamando
"Meu PIX de R$ 2.000 foi bloqueado às 14:35"
→ Vem aqui, busca por CPF + data/hora → Vê por que foi bloqueado

Cenário 2: Quer estudar fraudes
"Qual é o padrão de fraude de cartão de hoje?"
→ Filtra: Status = Fraude, Canal = Cartão → Vê todos

Cenário 3: Validar decisão
"Essa transação que deixei em feedback, qual era seu score?"
→ Busca pelo ID → Vê score e motivos

**Elementos Principais:**

🔎 **Campo de Busca (CPF)**
Digite os dígitos do CPF (mascarado automaticamente no sistema).

📅 **Filtro de Data**
Selecione período. Útil para:
- "Últimas 24h" - padrão mais comum
- "Últimos 7 dias" - tendência semanal
- "Todo período" - análise histórica

⚡ **Filtro de Canal**
- PIX: maior volume e risco
- Cartão: compras e saques
- TED: transferências bancárias
- Boleto: pagamentos tradicionais

📊 **Filtro de Status**
- Todos: mostra tudo
- Legítimas: ✅ aprovadas
- Suspeitas: ⚠️ analisadas
- Fraudes: ❌ confirmadas

📋 **Tabela de Resultados**

Cada coluna significa:
- **ID**: identificador único da transação
- **CPF**: cliente (mascarado por privacidade: ***.***789-01)
- **Valor**: quanto em reais
- **Data/Hora**: quando aconteceu
- **Canal**: tipo de transação
- **Status**: resultado (✅ ⚠️ ❌)
- **Score**: 0-100, quanto maior mais suspeita

Cores na tabela:
- Verde ✅ = Legítima (passou tranquilo)
- Amarelo ⚠️ = Analisada (em revisão)
- Vermelho ❌ = Fraude (bloqueada)

🔘 **Botão "Ver Detalhes"**
Clique em uma linha para abrir Investigação da transação.

⬇️ **Botão "Exportar"**
Baixa os resultados em CSV/Excel para relatórios.

**Como Usar - Passo a Passo:**
1. Filtre o que procura (CPF, data, canal, status)
2. Clique "Buscar"
3. Revise os resultados
4. Se precisa entender uma transação, clique "Ver Detalhes"
5. Se quer levar dados para fora, clique "Exportar"

**Dica:**
Combine filtros! Exemplo: "PIX + Últimas 24h + Fraudes" = mostra todas as fraudes de PIX de hoje. Excelente para padrões!

**Cuidado com Privacidade:**
CPF sempre aparece mascarado. Você vê essas informações porque tem permissão. Não compartilhe printscreens com ninguém!`
      },
      {
        id: 'investigacao',
        title: '🔍 Investigação - Análise Profunda de Fraudes',
        content: `
**Aonde Encontrar:** Menu > Investigação (ou clique "Ver Detalhes" em Transações)

**O Que Você Vê:**

╔══════════════════════════════════════════════════════════════════╗
║              TELA DE INVESTIGAÇÃO                               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1️⃣  DADOS DA TRANSAÇÃO                                         ║
║  ┌─────────────────────────────────────────────────────────┐   ║
║  │ CPF: ***.***.789-01                                     │   ║
║  │ Valor: R$ 5.000,00                                      │   ║
║  │ Canal: PIX                                              │   ║
║  │ Hora: 23:45 (NOTURNO - Alto risco!)                    │   ║
║  │ Destinatário Novo: Sim                                  │   ║
║  │ Estado: São Paulo → Bahia (1.500 km)                    │   ║
║  └─────────────────────────────────────────────────────────┘   ║
║                                                                  ║
║  2️⃣  EXPLICABILIDADE (Por que foi bloqueada?)                  ║
║  ┌─────────────────────────────────────────────────────────┐   ║
║  │ 🔴 Motivo 1: Transação Noturna                          │   ║
║  │    "PIX fora do horário normal (23h45). Risco Alto."   │   ║
║  │    Peso: ████████░░ (80%)                              │   ║
║  │                                                         │   ║
║  │ 🟠 Motivo 2: Destinatário Novo                          │   ║
║  │    "CPF de destino não tem histórico com este cliente" │   ║
║  │    Peso: ██████░░░░ (60%)                              │   ║
║  │                                                         │   ║
║  │ 🟠 Motivo 3: Desvio Geográfico                          │   ║
║  │    "Transação para estado distante. Cliente em SP?"    │   ║
║  │    Peso: ████░░░░░░ (40%)                              │   ║
║  │                                                         │   ║
║  │ 🟡 Motivo 4: Valor Acima da Média                       │   ║
║  │    "Cliente normalmente transfere até R$ 2.000"        │   ║
║  │    Peso: ██░░░░░░░░ (20%)                              │   ║
║  │                                                         │   ║
║  └─────────────────────────────────────────────────────────┘   ║
║                                                                  ║
║  3️⃣  SCORE E CONFIANÇA                                         ║
║  ┌─────────────────────────────────────────────────────────┐   ║
║  │ Score de Risco: ████████████████░░ 82/100              │   ║
║  │ Confiança do Modelo: 94% (MUITO CONFIANTE)             │   ║
║  │ Recomendação: BLOQUEIO                                  │   ║
║  └─────────────────────────────────────────────────────────┘   ║
║                                                                  ║
║  4️⃣  HISTÓRICO DO CLIENTE                                      ║
║  ┌─────────────────────────────────────────────────────────┐   ║
║  │ Transferências por dia: 2-3 em média                    │   ║
║  │ Horário preferido: 09h-18h (comercial)                  │   ║
║  │ Valor médio: R$ 1.500                                   │   ║
║  │ Destinos: Sempre para SP, RJ ou MG                      │   ║
║  │ Comportamento: MUITO estável (baixo risco base)        │   ║
║  └─────────────────────────────────────────────────────────┘   ║
║                                                                  ║
║  5️⃣  AÇÕES DISPONÍVEIS                                         ║
║  [ 👍 Confirmar Fraude ] [ 👎 Discordo ] [ 💬 Deixar Feedback] ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

**Para Que Serve?**
Entender POR QUE uma transação foi bloqueada. Você vira um "detetive" armado com dados.

**Quando Usar?**

Cenário 1: Cliente reclama
"Por que meu PIX foi bloqueado?"
→ Você vem aqui e mostra os motivos (educado e técnico)

Cenário 2: Auditoria
"Justifique por que bloqueou essa transação"
→ Tem aqui tudo explicado. Cumpre LGPD (Art. 20)

Cenário 3: Aprender padrões
"Que características fazem um PIX ser suspeito?"
→ Vê aqui vários exemplos reais

**Seção 1: Dados da Transação**

Informações básicas da operação:
- CPF (mascarado)
- Valor e hora
- Canal (PIX, Cartão, etc)
- Localização (de → para)

**Seção 2: Explicabilidade (LGPD Art. 20)**

Este é o "coração" da investigação. Para cada motivo de suspeita, mostra:

📊 **Motivo**: descrição em português claro
📈 **Peso**: quanto contribuiu para a decisão (em percentual)

Exemplo real:
```
🔴 Motivo: Transação Noturna
"PIX realizado às 23h45. Clientes normais não transferem nessa hora."
Peso: 80% (muito importante)
```

O Sankofa combina esses pesos para chegar ao score final.

**Seção 3: Score e Confiança**

- **Score**: 0-100, quanto maior = mais suspeita
  - 0-30: Muito legítima
  - 30-50: Incerta (vai para Revisão Manual)
  - 50-70: Suspeita
  - 70-100: Muito suspeita (bloqueada)

- **Confiança**: 0-100%, quanto o modelo tem certeza
  - 90%+ = Modelo muito confiante
  - 50-90% = Moderado (requer análise)
  - <50% = Incerto (sempre revisa)

**Seção 4: Histórico do Cliente**

Mostra o "padrão normal" do cliente:
- Quantas transferências faz por dia
- Em que horários (comercial vs noturno)
- Quanto transfere (média e máximo)
- Para onde transfere (mesmos estados?)
- Comportamento geral (estável ou errático)

Se a transação DESVIA MUITO disso → Suspeita

**Seção 5: Ações Disponíveis**

👍 **Confirmar Fraude**
"Sim, isso realmente era uma fraude"
→ Você está validando a decisão do sistema

👎 **Discordo (Era Legítima)**
"Não, isso era legítimo, modelo errou"
→ Feedback importante para treinar

💬 **Deixar Comentário**
"Cliente estava viajando para RJ"
→ Contexto que ajuda a entender

**Como Usar:**
1. Abra uma transação (via Transações → Ver Detalhes)
2. Leia os MOTIVOS (explicabilidade)
3. Veja o SCORE e CONFIANÇA
4. Revise o HISTÓRICO do cliente
5. Se faz sentido: Confirme ou deixe Feedback
6. Se tem dúvida: deixe comentário

**Exemplo Real (Caso de Sucesso):**

Cliente Marcus ligou reclamando: "Bloquearam minha transferência!"

Você vai em Investigação e vê:
- Motivo 1 (80%): Noturno
- Motivo 2 (60%): Destinatário novo
- Histórico: Marcus sempre transfere 9h-17h, para SP

Você entende: o contexto combinou tudo contra ele. Mas era legítima.

Você deixa feedback: "Viajando para Bahia esse mês"

Sistema aprende → Próxima vez, não bloqueia tão rápido.

**Dica de Ouro:**
A Explicabilidade é uma VANTAGEM competitiva. Concorrentes não têm isso. Use para ganhar confiança dos clientes!

**Cuidado:**
Não assuma que o modelo SEMPRE acerta. Seu julgamento humano é importante. Se a explicação não faz sentido → Deixe feedback!`
      },
      {
        id: 'revisao-manual',
        title: '👁️ Revisão Manual - Human-in-the-Loop',
        content: `
**Aonde Encontrar:** Menu > Revisão Manual

**Para Que Serve?**

O Sankofa é inteligente MAS tem decisões incertas (score ~ 50%). Aqui você, como especialista, revisa essas transações e diz: "Isso é fraude" ou "Isso é legítimo".

**Analogia:**

Imagine um juiz:
- Casos claros: condena logo (fraude certa = bloqueio)
- Casos duvidosos: chama perito (score incerto = vai para você)
- Perito analisa: valida ou discorda
- Resultado refinado: decisão final mais acurada

**O Que Você Vê:**

╔══════════════════════════════════════════════════════════════════╗
║            FILA DE REVISÃO MANUAL                               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  📊 ESTATÍSTICAS DA FILA                                        ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ Pendentes: 45 | Revisadas: 1.203 | Taxa: 94,1%          │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  💼 TRANSAÇÃO 1 (Prioridade: ALTA - Score 48)                 ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ CPF: ***.***.123-45                                      │  ║
║  │ Valor: R$ 3.200                                          │  ║
║  │ Canal: PIX                                               │  ║
║  │ Hora: 14:32                                              │  ║
║  │ Score: ████████░░ 48/100 (INCERTO)                       │  ║
║  │ Confiança: 52% (BAIXA - por isso está aqui)             │  ║
║  │                                                          │  ║
║  │ Motivos:                                                │  ║
║  │ ⚠️ Valor 2x acima da média deste cliente                │  ║
║  │ ✅ Horário comercial (baixo risco)                       │  ║
║  │ ✅ Destinatário conhecido                                │  ║
║  │                                                          │  ║
║  │ Sua Decisão:                                            │  ║
║  │ [ ✅ Legítima ] [ ❌ Fraude ] [ 💬 Deixar Comentário ]  │  ║
║  │                                                          │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  💼 TRANSAÇÃO 2 (Prioridade: MÉDIA - Score 51)                │  ║
║  [ ... similar ... ]                                           ║
║                                                                  ║
║  [ ▶ Próxima ] [ ◀ Anterior ]                                 ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

**Quando Usar?**

**Cenário 1: Rotina Diária**
"Vou revisar 50 transações incertas hoje"
→ Dedica 1 hora da manhã para isso
→ Revisa ~50 casos
→ Cada validação TREINA o modelo

**Cenário 2: Melhoria Contínua**
"Model accuracy caiu de 95% para 92%"
→ Aumenta volume de revisões
→ Mais feedback = modelo melhora

**Cenário 3: Investigação**
"Clientes estão reclamando de bloqueios"
→ Filtra as incertas que foram bloqueadas
→ Valida como legítimas
→ Feedback faz modelo aprender

**Elementos Principais:**

📊 **Estatísticas no Topo**
- Pendentes: quantas esperando por você
- Revisadas: quantas você já fez
- Taxa: % de conclusão

💼 **Transação em Revisão**

Mostra:
- CPF (mascarado)
- Valor e hora
- Canal (PIX, TED, etc)
- Score (0-100)
- Confiança do modelo
- Motivos de suspeita (marcados com ✅ ⚠️)

🎯 **Score e Prioridade**

- Score 40-60: INCERTOS (sua decisão é crítica)
- Score 30-40 ou 60-70: MODERADOS
- Prioridade ALTA: quando score está MUITO perto do threshold

✅ **Botão "Legítima"**
"Isso é uma transação normal, modelo errou em marcar como suspeita"
→ Feedback aumenta confiança do modelo em casos similares

❌ **Botão "Fraude"**
"Isso realmente é uma fraude, modelo fez a escolha certa"
→ Reforça que o modelo está no caminho correto

💬 **Comentário Opcional**
"Cliente confirmou ao telefone que fez"
"Era teste de cartão novo"
"Cliente em viagem de negócios"
→ Contexto importante para análise

**Como Usar - Passo a Passo:**

1. Abra Revisão Manual
2. Leia os dados da transação
3. Analise os motivos listados
4. Faça sua decisão:
   - ✅ Se faz sentido ser legítima
   - ❌ Se realmente parece fraude
   - 💬 Deixe comentário se tiver contexto
5. Clique em sua escolha
6. Próxima! (carrega a próxima automáticamente)

**Exemplo Real:**

Carlos, seu colega analista, está em Revisão Manual:

Vê uma transação:
- Score: 49 (bem no meio!)
- Valor: R$ 7.000
- Motivo: "Horário noturno"
- Mas: Horário 23h30 é noturno? Sim...

Carlos pensa: "Mas isso pode ser roaming noturno, cliente do exterior"
Carlos deixa feedback: "Legítima - cliente é viajante"

No dia seguinte, o modelo melhora e não bloqueia mais viajantes a noite.

**Por Que Isso Importa?**

Toda validação que você faz:
1. Melhora o modelo
2. Reduz falsos positivos
3. Melhora experiência do cliente
4. Você aprimora sua experiência
5. Sistema aprende com especialista humano

É Machine Learning + Human Intelligence = IA + você!

**Dica Profissional:**
Se você revisar 100 transações/dia consistentemente, em 1 mês o modelo fica muito mais preciso. É seu "investimento" no sistema!`
      },
      {
        id: 'calibragem',
        title: '⚙️ Calibragem - Ajuste de Sensibilidade',
        content: `
**Aonde Encontrar:** Menu > Calibragem

**Para Que Serve?**

Você controla o quanto "rigoroso" ou "permissivo" o Sankofa é.

**Analogia do Mundo Real:**

Imagine um detector de metais em um aeroporto:
- SENSÍVEL ALTO: detecta até uma moeda no bolso (bloqueias muitas pessoas)
- SENSÍVEL MÉDIO: detecta armas (balanço)
- SENSÍVEL BAIXO: detecta apenas objetos grandes (deixa passar muita coisa)

**O Mesmo Vale para Fraude:**
- CALIBRAGEM ALTA (>70): Muito rigoroso, bloqueia muito (bom clientes reclamam)
- CALIBRAGEM MÉDIA (40-60): Balanço recomendado
- CALIBRAGEM BAIXA (<30): Muito permissivo, deixa fraude passar

**O Que Você Vê:**

╔══════════════════════════════════════════════════════════════════╗
║                  TELA DE CALIBRAGEM                             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  📊 THRESHOLD ATUAL                                             ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ Configuração Ativa: 45/100                               │  ║
║  │                                                          │  ║
║  │ 0         25         45         75         100          │  ║
║  │ |---------|---------|●---------|---------|              │  ║
║  │ MuitoPermissivo    RECOMENDADO   Muito Rigoroso         │  ║
║  │                                                          │  ║
║  │ [Usar Slider para Ajustar] ◀──●──▶                      │  ║
║  │                                                          │  ║
║  │ Novo Valor: 50 (será 45 → 50)                          │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  🎯 IMPACTO ESPERADO (se mudar para 50)                       ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ Fraudes deixadas passar (False Negative): -5% menos      │  ║
║  │ Clientes legítimos bloqueados (False Positive): +3%mais  │  ║
║  │ Valor em reais desprotegido: -R$ 500.000 aprox          │  ║
║  │ Reclamações esperadas: +12 clientes/dia                 │  ║
║  │                                                          │  ║
║  │ ⚠️ Recomendação: Manter em 45 (já está otimizado)        │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  📈 PERFORMANCE COMPARATIVA                                    ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ Acurácia em 30:  81%                                     │  ║
║  │ Acurácia em 45:  93% ⭐ (ATUAL)                         │  ║
║  │ Acurácia em 60:  87%                                     │  ║
║  │ Acurácia em 75:  79%                                     │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  📋 HISTÓRICO DE CALIBRAGENS                                   ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ 2024-11-30 14:30 │ 45 → 42 │ João Silva │ Melhora  │   │  ║
║  │ 2024-11-28 09:15 │ 42 → 45 │ Maria Rosa │ Reduzir │   │  ║
║  │ 2024-11-25 16:45 │ 48 → 42 │ Carlos    │ Ataque  │   │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  [ 💾 Salvar Novo Threshold ] [ ↩️ Desfazer Última ] [ ⚠️ Reset ]║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

**Quando Usar?**

**Cenário 1: Reclamações em Massa**
"Clientes estão reclamando que estamos bloqueando muitos PIX"
→ Taxa de falsos positivos alta
→ DIMINUA o threshold (45 → 35)
→ Monitore no dia seguinte

**Cenário 2: Fraudes Aumentando**
"Fraudes de cartão clonado explodiram essa semana"
→ Deixando muita coisa passar
→ AUMENTE o threshold (45 → 55)
→ Mais rígido = menos fraude passa

**Cenário 3: Análise de Performance**
"Preciso otimizar para máxima acurácia"
→ Veja histórico de performance
→ Teste diferentes valores (45, 50, 55)
→ Mantenha o com melhor resultado

**Elementos Principais:**

🎚️ **Slider de Threshold**

O control principal. Vai de 0 a 100:
- 0-25: Extremamente permissivo (quase nada bloqueia)
- 25-50: Permissivo para balanço (RECOMENDADO: 40-50)
- 50-75: Rigoroso
- 75-100: Extremamente rigoroso (bloqueia quase tudo)

**Padrão Recomendado: 45**
(máxima acurácia geral)

📊 **Impacto Esperado**

Quando você muda o slider, o sistema calcula:
- Quantas mais/menos fraudes vão passar
- Quantos mais/menos clientes legítimos vão ser bloqueados
- Valor em reais afetado
- Estimativa de reclamações

Exemplo:
"Se mudo de 45 para 55, bloqueio +5% de fraudes, mas também bloqueio +8% de clientes legítimos"

📈 **Comparação de Performance**

Mostra acurácia em diferentes thresholds:
- 30: 81%
- 45: 93% (melhor)
- 60: 87%
- 75: 79%

Use isso para entender qual está otimizado!

📋 **Histórico**

Todas as mudanças que foram feitas:
- Quando
- Por quem
- De quanto para quanto
- Motivo

Excelente para auditoria e para entender decisões passadas.

**Como Usar:**

1. Abra Calibragem
2. Analise a performance atual (está em 45, 93% acurácia)
3. Se precisa mudar:
   - Clientes reclamando? Diminua (45 → 35)
   - Fraudes passando? Aumente (45 → 55)
4. Estude o impacto esperado
5. Se faz sentido, clique "Salvar Novo Threshold"
6. Monitore no dia seguinte
7. Se piorou, "Desfazer" volta ao anterior

**Dica de Ouro:**

Não mude drasticamente! Estratégia recomendada:
- Mudanças pequenas: 5 pontos (45 → 50)
- Esperar 1 dia de monitoramento
- Ver resultados no Dashboard
- Ajustar novamente se necessário

Assim você evita grandes erros!

**Exemplo Real:**

Você observa:
- Quinta passada: 10 reclamações (clientes bloqueados)
- Sexta: 15 reclamações
- Segunda: 18 reclamações

Você decide: "Vou abaixar de 45 para 40"

Na terça:
- 5 reclamações (sucesso!)
- Taxa de fraude: +2% (aceitável)

Mantém em 40. Problema resolvido!

**Cuidado:**

Não use calibragem como "solução mágica". Se o problema é:
- Fraudes novas e desconhecidas → Calibrar não resolve
- Dados ruins → Calibrar não resolve
- Sistema offline → Calibrar não resolve

Calibragem é para AJUSTAR comportamento conhecido. Para problemas profundos, use Hard Rules ou Feedback!`
      },
      {
        id: 'alertas',
        title: '🚨 Alertas - Notificações Críticas',
        content: `
**Aonde Encontrar:** Menu > Alertas

**Para Que Serve?**

Quando algo FORA DO NORMAL acontece, você recebe um ALERTA aqui. É como uma sirene de incêndio: algo merece atenção AGORA.

**Exemplos de Alertas Reais:**

🔴 CRÍTICO: "Spike de fraudes de cartão: +250% vs média"
🟠 AVISO: "Modelo offline há 15 minutos"
🟡 INFORMAÇÃO: "Recalibração automática recomendada"

**O Que Você Vê:**

╔══════════════════════════════════════════════════════════════════╗
║                  TELA DE ALERTAS                                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ⚡ ALERTAS ATIVOS (Precisa de Ação)                           ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ 🔴 [CRÍTICO] Spike de Fraudes em PIX                     │  ║
║  │    Detectado 45 fraudes em 30min (vs 10 de média)        │  ║
║  │    Iniciado: 14:35 | Status: ATIVO                      │  ║
║  │    [ 🔍 Investigar ] [ ✅ Resolvido ] [ 📌 Fixar ]       │  ║
║  │                                                          │  ║
║  │ 🟠 [AVISO] Taxa de Falso Positivo Alta                  │  ║
║  │    23% de clientes legítimos bloqueados (vs 5% alvo)    │  ║
║  │    Iniciado: 09:22 | Status: ATIVO                      │  ║
║  │    Sugestão: Diminuir threshold de 45 para 35           │  ║
║  │    [ ⚙️ Ir para Calibragem ] [ ✅ Resolvido ]            │  ║
║  │                                                          │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  📋 HISTÓRICO DE ALERTAS (Já Resolvidos)                      ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ 2024-11-30 13:00 │ Modelo CatBoost Offline    │ Resolvido  │  ║
║  │ 2024-11-29 22:30 │ Drift Detectado em Features│ Resolvido  │  ║
║  │ 2024-11-28 16:45 │ TPS Abaixo do Esperado     │ Resolvido  │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

**Quando Usar?**

✅ **Toda vez que ABRE o sistema**
"Posso trabalhar ou há algo crítico?"
→ Abre Alertas primeiro
→ Se há algo vermelho: resolve antes

✅ **Se recebe notificação do sistema**
Você pode receber email/SMS com alerta crítico
→ Vem aqui, vê qual é
→ Toma ação

✅ **Rotina de encerramento**
"Vou encerrar o turno, há algo pendente?"
→ Revisa alertas ativos
→ Se há, resolve ou passa para colega

**Elementos Principais:**

🔴 🟠 🟡 **Níveis de Severidade**

- 🔴 CRÍTICO: Ação urgente necessária (afeta SLA, segurança)
- 🟠 AVISO: Atenção recomendada (próximas horas)
- 🟡 INFORMAÇÃO: Notificação (para conhecimento)

📊 **Descrição do Alerta**

Cada alerta explica:
- O que aconteceu
- Quando começou
- Dados quantitativos (diferença vs normal)
- Sugestão de ação

⏱️ **Status e Duração**

- ATIVO: ainda acontecendo, precisa ação
- RESOLVIDO: problema já passou

🔘 **Botões de Ação**

- 🔍 Investigar: vai para tela relacionada (Transações, Investigação, etc)
- ✅ Marcar Resolvido: você tratou o problema
- 📌 Fixar: importante, deixa em destaque
- 💬 Comentário: deixa nota para colega

**Como Usar - Passo a Passo:**

1. Abra Alertas
2. Veja se há alertas ativos (vermelhos/alaranjados)
3. Para cada alerta:
   - Leia a descrição
   - Entenda o problema
   - Clique no botão de ação (Investigar, Ir para Calibragem, etc)
   - Resolva na tela específica
   - Volte e marque como "Resolvido"
4. Se não sabe o que fazer: deixe comentário para colega/gerente

**Exemplo Real - Spike de Fraudes:**

14:35 - Alerta aparece: "Spike de fraudes em PIX"

Você vê:
- 45 fraudes em 30 minutos
- Normal é ~10
- Diferença: +350%

Você:
1. Clica "Investigar"
2. Va para Transações
3. Filtra: PIX + Últimos 30 minutos + Fraudes
4. Vê que todas vêm de um CPF (clonado)
5. Vai para Regras Duras
6. Cria hard rule: "CPF XXX.XXX.789-01: BLOQUEAR TUDO"
7. Volta e marca alerta como "Resolvido"

Pronto! Bloqueou um clone em ~5 minutos.

**Exemplo Real - Falso Positivo Alto:**

09:22 - Alerta aparece: "Taxa de falso positivo acima do limite"

Você vê:
- 23% de clientes legítimos bloqueados
- Alvo é 5%
- Sugestão: diminuir threshold

Você:
1. Clica "Ir para Calibragem"
2. Muda de 45 para 35
3. Salva
4. Volta e marca como "Resolvido"
5. Monitora Dashboard próximas 2 horas

Resultado: Reclamações caem drasticamente.

**Dica de Ouro:**

Alertas são como "canários na mina". Se está vendo alertas, significa que há algo mudando no comportamento do sistema. SEMPRE investigue!

**Cuidado:**

Não ignore alertas vermelhos só porque está ocupado:
- Podem indicar ataque em progresso
- Podem indicar sistema falhando
- Cada minuto conta em fraude

Se não pode resolver agora, chame seu gerente/colega!`
      },
      {
        id: 'hard-rules',
        title: '🔒 Regras Duras - Bloqueio Automático',
        content: `
**Aonde Encontrar:** Menu > Regras Duras

**Para Que Serve?**

Hard Rules são decisões AUTOMÁTICAS e PERMANENTES:
"SE [condição], ENTÃO [ação]"

Exemplos:
- "SE CPF na lista negra ENTÃO BLOQUEIO"
- "SE valor > R$ 100k E horário 23h-5h ENTÃO ALERTA"
- "SE IP do Exterior ENTÃO INVESTIGAR"

É como um "sinal de PARADA" de trânsito. Quem vê, obedece.

**O Que Você Vê:**

╔══════════════════════════════════════════════════════════════════╗
║             TELA DE REGRAS DURAS (HARD RULES)                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  📌 REGRAS ATIVAS (12 regras rodando agora)                    ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │                                                          │  ║
║  │ Regra #1: Bloqueio Lista Negra                          │  ║
║  │ SE: CPF em HOT_LIST = SIM                               │  ║
║  │ AÇÃO: BLOQUEIO PERMANENTE                               │  ║
║  │ Ativa: SIM | Criada: 2024-10-01 | Por: Admin            │  ║
║  │ Impacto: 156 transações bloqueadas                      │  ║
║  │ [ ✏️ Editar ] [ 🗑️ Deletar ] [ 📊 Estatísticas ]        │  ║
║  │                                                          │  ║
║  │ Regra #2: Alerta Valor Alto Noturno                    │  ║
║  │ SE: Valor > R$ 50.000 E Hora entre 23h-05h             │  ║
║  │ AÇÃO: GERAR ALERTA (não bloqueia)                      │  ║
║  │ Ativa: SIM | Criada: 2024-11-15 | Por: Ana Silva       │  ║
║  │ Impacto: 34 alertas gerados                             │  ║
║  │ [ ✏️ Editar ] [ 🗑️ Deletar ] [ 📊 Estatísticas ]        │  ║
║  │                                                          │  ║
║  │ Regra #3: Bloqueio IP Exterior                          │  ║
║  │ SE: Origem IP = Fora do Brasil                          │  ║
║  │ AÇÃO: INVESTIGAÇÃO                                      │  ║
║  │ Ativa: SIM | Criada: 2024-11-20 | Por: Carlos          │  ║
║  │ Impacto: 8 investigações iniciadas                      │  ║
║  │ [ ✏️ Editar ] [ 🗑️ Deletar ] [ 📊 Estatísticas ]        │  ║
║  │                                                          │  ║
║  │ [ ... 9 mais regras ... ]                               │  ║
║  │                                                          │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  ✏️ CRIAR NOVA REGRA                                           ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ Nome: [Bloquear Cartão Duplicado_________]              │  ║
║  │                                                          │  ║
║  │ Condições (SE):                                         │  ║
║  │ [ Canal ] [ = ] [ Cartão ▼ ]  [ + Adicionar ]          │  ║
║  │ [ Tentativas ] [ > ] [ 5 ]     [ + Adicionar ]          │  ║
║  │ [ Em ] [ 1 hora ]                                       │  ║
║  │                                                          │  ║
║  │ Ação (ENTÃO):                                           │  ║
║  │ [ BLOQUEIO ▼ ]                                          │  ║
║  │                                                          │  ║
║  │ [ 💾 Salvar Regra ] [ ❌ Cancelar ]                     │  ║
║  │                                                          │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

**Quando Usar?**

**Cenário 1: Detectou novo padrão de fraude**
"Detectei que fraude de cartão clonado sempre acontece em SP entre 14h-18h"
→ Cria regra: SE [Cartão + SP + 14h-18h + 5+ tentativas] ENTÃO [ALERTA]
→ Próxima vez que acontecer: você sabe IMEDIATAMENTE

**Cenário 2: CPF Comprometido**
"Esse CPF teve fraude confirmada, precisa bloquear TUDO"
→ Adiciona à HOT list
→ Hard rule já existe para isso
→ Automático!

**Cenário 3: Promoção ou Evento**
"Black Friday: espero spike legítimo, não é fraude"
→ Cria regra: SE [Horário 9h-21h + Black Friday] ENTÃO [Permitir valores até 2x]
→ Clientes aprovam mais rápido

**Elementos Principais:**

📌 **Lista de Regras Ativas**

Cada regra mostra:
- Nome descritivo
- Condições (SE...)
- Ação (ENTÃO...)
- Se está ativa ou não
- Quando foi criada
- Quem criou
- Impacto (quantas transações afetou)

🎯 **Acções Possíveis:**

- BLOQUEIO: Nega a transação permanentemente
- ALERTA: Notifica, mas deixa passar (vai para humano revisar)
- INVESTIGAÇÃO: Trata como suspeita (análise profunda)
- PERMITIR: Aprova automaticamente (para VIPs)

📝 **Criar Nova Regra**

Bem simples:
1. Dê um nome descritivo
2. Defina CONDIÇÕES (SE...)
   - Pode combinar vários "E"
   - Exemplo: Canal=PIX E Horário=23h-5h E Valor>5000
3. Defina AÇÃO (ENTÃO...)
   - BLOQUEIO / ALERTA / INVESTIGAÇÃO / PERMITIR
4. Salve!

**Como Usar - Passo a Passo:**

1. Abra Regras Duras
2. Analise as regras ativas (ficam desatualizadas?)
3. Se precisa criar:
   - Clique "Criar Nova Regra"
   - Nome: algo descritivo ("Bloquear Cartão SP 14h")
   - Condições: combina os campos
   - Ação: escolhe ação
   - Salva

4. Monitore o impacto:
   - Volte em alguns dias
   - Veja quantas transações foram afetadas
   - Se muitas bloqueadas: talvez regra muito rigorosa

**Exemplo Real - Fraude de Cartão Clonado:**

Você percebeu:
- Última semana: 50 fraudes de cartão
- Local: São Paulo
- Horário: 14h-18h
- Padrão: 5-10 tentativas no mesmo comerciante

Você cria regra:
```
Nome: Bloquear Teste de Cartão Clonado SP
SE:
  - Canal = Cartão
  - Localização = São Paulo
  - Horário entre 14h-18h
  - Múltiplas tentativas (>3) em 1 hora
ENTÃO:
  - BLOQUEIO
```

Resultado:
- Fraudes SP caem de 50 para 5/semana
- Clientes não reclamam (só testadores estão bravos!)

**Dica de Ouro:**

Crie regras baseada em PADRÕES OBSERVADOS:
- Não crie por "achismo"
- Sempre analise Transações primeiro
- Confirme padrão
- Depois cria regra

Assim garante que regra funciona!

**Cuidado:**

Hard Rules são PERMANENTES e AUTOMÁTICAS. Erros são críticos:
- Regra muito ampla = bloqueia clientes legítimos
- Regra muito específica = não pega fraudes

Teste em ALERTA primeiro, depois muda para BLOQUEIO!

**Manutenção:**

A cada mês, revise regras antigas:
- Ainda estão sendo usadas?
- Impacto ainda faz sentido?
- Padrão mudou?

Se não mais necessária: delete ou desative!`
      },
      {
        id: 'listas-vip-hot',
        title: '✨ Listas VIP e HOT - Whitelist e Blacklist',
        content: `
**Aonde Encontrar:** Menu > Lista VIP (ou Menu > Lista HOT)

**Para Que Servem?**

São duas listas simples MAS PODEROSAS:

📝 **VIP (Whitelist)** - Aprovação Direta
"São cliente que CONFIO 100%. Deixa passar tudo"
- Diretores da empresa
- Clientes top tiers
- Contas internas
- Pessoas muito confiáveis

📝 **HOT (Blacklist)** - Bloqueio Direto
"São contas que SÃO PROBLEMÁTICAS. Bloqueia TUDO"
- CPFs com fraude confirmada
- Documentos clonados
- Contas comprometidas

**O Que Você Vê (VIP):**

╔══════════════════════════════════════════════════════════════════╗
║               LISTA VIP (WHITELIST)                             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ✨ VIP CADASTRADOS (1 cliente)                                ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ CPF: ***.***.111-00                                      │  ║
║  │ Nome: Diretor Geral - João Silva                         │  ║
║  │ Adicionado: 2024-10-01 às 10:30                         │  ║
║  │ Por: Gerente - Maria Rosa                               │  ║
║  │ Motivo: "Diretor Executivo - Confiança Total"           │  ║
║  │ Status: Ativo                                            │  ║
║  │ [ 🗑️ Remover ] [ ✏️ Editar ]                            │  ║
║  │                                                          │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  ➕ ADICIONAR NOVA VIP                                         ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ CPF: [_______________________]                           │  ║
║  │ Nome (opcional): [____________________________]          │  ║
║  │ Motivo: [________________________________]              │  ║
║  │ Exemplo: "Gerente Regional", "Cliente VIP"              │  ║
║  │                                                          │  ║
║  │ [ ✅ Adicionar ] [ ❌ Cancelar ]                        │  ║
║  │                                                          │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

**O Que Você Vê (HOT):**

╔══════════════════════════════════════════════════════════════════╗
║               LISTA HOT (BLACKLIST)                             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ❌ BLOQUEADOS (1 CPF)                                          ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ CPF: ***.***.999-99                                      │  ║
║  │ Status: Ativo (BLOQUEADO)                                │  ║
║  │ Adicionado: 2024-11-20 às 15:45                         │  ║
║  │ Por: Analista - Carlos Santos                           │  ║
║  │ Motivo: "Fraude confirmada - CPF clonado"               │  ║
║  │ Transações bloqueadas: 8                                 │  ║
║  │ [ ❌ Remover ] [ ⏸️ Desativar Temporariamente ]         │  ║
║  │                                                          │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  ➕ ADICIONAR NOVA BLOQUEADA                                   ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ CPF/Conta: [_______________________]                    │  ║
║  │ Motivo: [____________________________]                  │  ║
║  │ Exemplo: "Fraude confirmada", "Documento Clonado"       │  ║
║  │                                                          │  ║
║  │ [ ✅ Adicionar ] [ ❌ Cancelar ]                        │  ║
║  │                                                          │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  📤 IMPORTAR EM MASSA                                          ║
║  [ Fazer upload de arquivo CSV ] [ Baixar Modelo ]             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

**Quando Usar (VIP):**

✅ **Adicionar**
"Novo diretor entrou na empresa"
"Cliente premium pediu aceleração"
"Gerente regional quer confiança total"

✅ **Remover**
"Diretor foi demitido"
"Cliente não é mais VIP"
"Pessoa foi reclassificada"

✅ **Revisar Regularmente**
"Vou auditar quem está na VIP"
A cada 3 meses, revisa se realmente todos continuam merecendo.

**Quando Usar (HOT):**

✅ **Adicionar**
"Fraude confirmada neste CPF"
"Cliente tentou múltiplas fraudes"
"Documento clonado"
"Conta comprometida"

✅ **Remover**
"Investigação descobriu ser erro nosso"
"Cliente resolveu problema com segurança"
"CPF recuperado do clone"

⚠️ **NUNCA remova sem investigação!**

**Como Usar - Passo a Passo (VIP):**

1. Abra Lista VIP
2. Analise quem está lá (faz sentido?)
3. Para ADICIONAR:
   - Clique "Adicionar Nova VIP"
   - Digite CPF
   - Deixe motivo claro
   - Clique "Adicionar"
4. Para REMOVER:
   - Encontre a pessoa
   - Clique "Remover"
   - Confirme

**Como Usar - Passo a Passo (HOT):**

1. Abra Lista HOT
2. Revise CPFs bloqueados
3. Para ADICIONAR:
   - Clique "Adicionar Nova Bloqueada"
   - Digite CPF
   - Motivo bem documentado
   - Clique "Adicionar"
4. Para REMOVER:
   - Encontre o CPF
   - ANTES: revise no Investigação por que foi adicionado
   - Clique "Remover"
   - Deixe comentário explicando por quê

**Exemplo Real - VIP:**

João é novo Gerente Regional.
Ele faz transferências grandes toda semana (R$ 100k+).
Sankofa, sem conhecê-lo, começaria a bloquear.

Solução:
→ Adicionas ele à VIP: "Gerente Regional - Confiança Total"
→ Próxima transferência: passa automático
→ João fica satisfeito
→ Sem reclamações

**Exemplo Real - HOT:**

Você revisa transações e vê:
- CPF XXX.XXX.789-01 tentou 5 fraudes em 1 hora
- Todas de canais diferentes
- Padrão 100% suspeito = conta clonada

Solução:
→ Adiciona à HOT: "Fraude confirmada - Múltiplas tentativas"
→ Próxima transação: BLOQUEIO automático
→ Cliente precisa ir ao banco recuperar
→ Fraude para

**Dica de Ouro:**

VIP e HOT são SIMPLES mas PODEROSAS:
- VIP: economiza tempo (não precisa esperar análise)
- HOT: protege 100% (nada passa)

Use com sabedoria!

**Cuidado:**

- VIP MAS fraudador = problema! Revise regularmente
- HOT MAS removido rápido = cliente sofre
- Se tem dúvida: deixe em investigação normal, não force VIP/HOT`
      },
      {
        id: 'monitoramento-metricas',
        title: '📊 Monitoramento & Métricas - Saúde do Sistema',
        content: `
**Aonde Encontrar:** Menu > Monitoramento (ou Menu > Métricas)

**Para Que Servem?**

**Métricas**: Números do sistema EM TEMPO REAL
- Quantas transações/segundo
- Qual é a latência agora
- Taxa de fraude nesse momento

**Monitoramento**: Saúde dos algoritmos
- Os 3 modelos de IA estão online?
- Acurácia está normal?
- Há drift de dados?

**O Que Você Vê (Métricas):**

╔══════════════════════════════════════════════════════════════════╗
║                DASHBOARD DE MÉTRICAS                            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ⚡ CONTADORES EM TEMPO REAL (Atualiza a cada 1 segundo)       ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ TPS (Tx/segundo): 12.3  │ Taxa Fraude: 68% │ Uptime: 99,9% │  ║
║  │ Latência P95: 72ms      │ Latência P99: 125ms             │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  📈 HISTÓRICO (últimas 6 horas)                                ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ Gráfico 1: TPS ao longo do tempo                         │  ║
║  │  ▁▂▃▄▅▆▇███▇▆▅▄▃▂▁  (pico às 14h)                      │  ║
║  │                                                          │  ║
║  │ Gráfico 2: Latência (deve estar < 50ms)                 │  ║
║  │  ▂▂▂▂▂▂▂▂██▂▂▂▂▂▂▂  (spike às 14h30)                   │  ║
║  │                                                          │  ║
║  │ Gráfico 3: Taxa de Fraude (%)                           │  ║
║  │  ▅▅▅▅▅▅▅███▆▆▇███  (padrão 65-70%)                    │  ║
║  │                                                          │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  ⚠️ LIMITES E ALERTAS                                         ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ TPS: 12.3 (Limite: 100) ✅                               │  ║
║  │ Latência P95: 72ms (Limite: 100ms) ✅                    │  ║
║  │ Uptime: 99.9% (Limite: 99.5%) ✅                        │  ║
║  │                                                          │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

**O Que Você Vê (Monitoramento):**

╔══════════════════════════════════════════════════════════════════╗
║             PAINEL DE MONITORAMENTO                              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  🤖 STATUS DOS MODELOS DE IA                                   ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ Model 1: Random Forest                                   │  ║
║  │ Status: ✅ ONLINE | Acurácia: 91% | Latência: 23ms      │  ║
║  │ Versão: 1.2.3 | Última atualização: 2024-11-28          │  ║
║  │                                                          │  ║
║  │ Model 2: Gradient Boosting                              │  ║
║  │ Status: ✅ ONLINE | Acurácia: 88% | Latência: 31ms      │  ║
║  │ Versão: 2.1.0 | Última atualização: 2024-11-25          │  ║
║  │                                                          │  ║
║  │ Model 3: CatBoost                                       │  ║
║  │ Status: ✅ ONLINE | Acurácia: 94% | Latência: 28ms      │  ║
║  │ Versão: 1.0.5 | Última atualização: 2024-11-29          │  ║
║  │                                                          │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  📊 DATA DRIFT (Mudança na distribuição de dados)             ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ Data Drift Score: 0.08 (Limite: 0.1) ✅                 │  ║
║  │ Interpretação: Mudança leve nos dados (normal)           │  ║
║  │ Ação recomendada: Monitorar proximamente                 │  ║
║  │                                                          │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  🎯 CONCEPT DRIFT (Mudança no significado dos dados)           ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │ Concept Drift Score: 0.05 (Limite: 0.15) ✅             │  ║
║  │ Interpretação: Fraudes mudaram pouco de padrão           │  ║
║  │ Ação recomendada: Continuar normalmente                  │  ║
║  │                                                          │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

**Conceitos Importantes:**

📊 **TPS (Transactions Per Second)**
Quantas transações o sistema processa por segundo.
- Esperado: 10-50 TPS em operações normais
- Pico: 100+ TPS em períodos críticos
- Se cair: problema de performance ou carga

⏱️ **Latência (P95, P99)**
Tempo que leva para ter uma resposta.
- P95: 95% das transações saem em < 72ms
- P99: 99% das transações saem em < 125ms
- SLA: deve estar < 100ms

📈 **Acurácia do Modelo**
Percentual de acertos do algoritmo.
- Objetivo: > 90%
- Se cair < 85%: problema, chamar data science
- Pode cair por drift de dados

🔄 **Data Drift**
Os DADOS mudaram (nova distribuição).
Exemplo: antes clientes transferiam <R$1000, agora transferem R$5000.
- Limite: 0.1
- Se > 0.1: modelo pode ficar menos preciso
- Solução: retreinar modelo

🎯 **Concept Drift**
O SIGNIFICADO dos dados mudou (novo tipo de fraude).
Exemplo: antes fraudes eram noturnas, agora são às 14h.
- Limite: 0.15
- Se > 0.15: padrão fundamentalmente mudou
- Solução: retreinar com novos dados

**Quando Usar?**

✅ **Inicio do turno**
"Como está a saúde geral?"
→ Abre Monitoramento
→ Se tudo verde: pode trabalhar normal
→ Se algo vermelho: investiga

✅ **Se recebe alerta**
"Latência subiu muito"
→ Vem aqui
→ Ve o que está acontecendo
→ Se problema persistir: chama DevOps

✅ **Rotina de Análise**
"Vou revisar como foi o dia"
→ Ve os gráficos históricos
→ Identifica padrões
→ Documenta para relatório

**Como Usar - Passo a Passo:**

1. Abra Métricas ou Monitoramento
2. Revise os contadores AGORA:
   - TPS está normal? (10-50)
   - Latência < 100ms? Sim
   - Uptime > 99%? Sim
3. Se tudo verde: trabalhe normal
4. Se algo alaranjado/vermelho:
   - Clique em detalhes
   - Ve o gráfico histórico
   - Entenda se é pico ou anomalia
5. Se precisa tomar ação:
   - Cale seu gerente
   - Ou escale para DevOps

**Exemplo Real - Latência Spike:**

14:32 - Você vê que Latência P95 subiu para 300ms (era 72ms)

Você:
1. Abre Monitoramento
2. Ve o gráfico: pico às 14:32 exatamente
3. Abre Dashboard
4. Ve: "Spike de fraudes de PIX +250%"
5. Entende: o sistema está sobrecarregado processando o ataque
6. Ativa ALERTAS em todas as centrais
7. Coordena bloqueio de CPF fraudador

Resultado: volta ao normal em 15 minutos.

**Dica de Ouro:**

Monitore regularmente:
- Você entende patterns
- Consegue prever problemas
- Fica expert no sistema

Quem conhece bem o "normal" identifica rápido o "anormal"!

**Cuidado:**

- Não assuste com spike pequeno (5 segundos é normal)
- Estranho é quando latência FICA alta por 10+ minutos
- Se drift muito alto: FALE com data science, não mexe sozinho!`
      },
      {
        id: 'outros-recursos',
        title: '📚 Outros Recursos Importantes',
        content: `
**Feedback Analista** (Menu > Feedback Analista)
Você deixa feedback para o modelo aprender.
- Quando discorda de uma decisão
- Deixa: "Isso era legítimo" ou "Isso era fraude"
- Modelo usa feedback para melhorar
- Quanto mais feedback, melhor o modelo

**Investigação Detalhada** → Veja a seção acima ✓

**Datasets** (Menu > Datasets)
Catálogo de dados disponíveis para análise.
- Histórico de fraudes
- Transações legítimas
- Padrões de clientes
- Use para criar relatórios customizados

**Relatórios** (Menu > Relatórios)
Gera análises para gerência/compliance.
- Template de performance
- Fraudes por período
- Fraudes por canal
- Exporta em PDF/Excel

**Auditoria** (Menu > Auditoria)
Registro LGPD de TUDO que aconteceu.
- Quem acessou quais dados
- Quem fez qual ação
- Quando foi feito
- Necessário para compliance

**Configurações** (Menu > Configurações)
Preferências pessoais e segurança.
- Tema (claro/escuro)
- Notificações
- Trocar senha
- Permissões (se admin)`
      },
      {
        id: 'rotina-completa',
        title: '⏰ Sua Rotina Diária Recomendada',
        content: `
**INÍCIO DO TURNO (5 minutos)**
1. Abra Alertas
2. Se há alertas vermelhos/laranja: resolva antes de qualquer coisa
3. Abra Dashboard
4. Revise KPIs e gráficos
5. Veja status dos modelos

**TRABALHO NORMAL (6 horas)**

Durante o turno, você:
- Responde reclamações de clientes (Transações → Investigação)
- Valida decisões (Revisão Manual)
- Cria relatórios (Relatórios)
- Monitora alertas (Alertas)
- Deixa feedback (Feedback Analista)

**CHECAGEM DE MID-TURNO (3 minutos)**
Metade do turno, pausa rápida:
- Abra Dashboard
- Há anomalias?
- Alertas novos?
- Continue trabalhando normalmente

**ENCERRAMENTO DO TURNO (10 minutos)**
1. Revise todos os alertas abertos
2. Se há: resolve ou passa para colega
3. Gere relatório do dia (Relatórios)
4. Documente ações importantes em comentário
5. Passe informações ao próximo turno

**AÇÕES ESPECIAIS (quando necessário)**
- **Fraude em massa**: Cria Hard Rule
- **Modelo falhando**: Chama gerente
- **Recalibração**: Vai em Calibragem
- **Auditoria**: Consulta Auditoria`
      }
    ]
  }
};

const ManualSection = ({ section, isOpen, onToggle }) => {
  return (
    <Card>
      <button
        onClick={onToggle}
        className="w-full"
      >
        <CardHeader className="cursor-pointer hover:bg-gray-50 transition-colors">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg text-left">
              {section.title}
            </CardTitle>
            {isOpen ? (
              <ChevronUp className="h-5 w-5 text-blue-600" />
            ) : (
              <ChevronDown className="h-5 w-5 text-gray-400" />
            )}
          </div>
        </CardHeader>
      </button>

      {isOpen && (
        <CardContent>
          {Array.isArray(section.sections) ? (
            section.sections.map((sub) => (
              <div key={sub.id} className="mb-8 pb-8 border-b last:border-b-0">
                <h3 className="text-base font-semibold text-gray-900 mb-3">
                  {sub.title}
                </h3>
                <div className="prose prose-sm max-w-none whitespace-pre-wrap text-gray-700 leading-relaxed font-mono text-xs">
                  {sub.content}
                </div>
              </div>
            ))
          ) : (
            <div className="prose prose-sm max-w-none whitespace-pre-wrap text-gray-700 leading-relaxed font-mono text-xs">
              {section.content}
            </div>
          )}
        </CardContent>
      )}
    </Card>
  );
};

export function Manual() {
  const [expandedSections, setExpandedSections] = useState({
    'introducao-intro-main': true,
    'introducao-mapa-visual': false,
    'telas-dashboard': false,
    'telas-transacoes': false,
  });

  const toggleSection = (key) => {
    setExpandedSections(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const allSections = [
    manualContent.introducao,
    manualContent.telas
  ];

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 via-blue-700 to-blue-800 rounded-lg p-8 text-white shadow-lg">
        <div className="flex items-center gap-3 mb-4">
          <BookOpen className="h-10 w-10" />
          <h1 className="text-4xl font-bold">📘 Manual do Sankofa v1.0</h1>
        </div>
        <p className="text-lg opacity-95 mb-2">
          Guia Completo e Profissional para Entender e Usar o Sistema de Detecção de Fraudes
        </p>
        <div className="flex items-center gap-2 text-sm opacity-80 mt-4">
          <Home className="h-4 w-4" />
          <span>Última atualização: 30 de Novembro de 2025</span>
        </div>
      </div>

      {/* Índice de Conteúdos */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span>📋</span> Índice de Conteúdos (Clique para Expandir)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {allSections.map((section) =>
              section.sections?.map((sub) => (
                <button
                  key={sub.id}
                  onClick={() => {
                    const element = document.getElementById(`section-${sub.id}`);
                    if (element) {
                      element.scrollIntoView({ behavior: 'smooth' });
                      setTimeout(() => {
                        const key = `telas-${sub.id.split('-')[0]}`;
                        setExpandedSections(prev => ({ ...prev, [key]: true }));
                      }, 100);
                    }
                  }}
                  className="text-left p-3 rounded-lg hover:bg-blue-50 hover:text-blue-700 transition-colors text-sm font-medium border border-transparent hover:border-blue-200"
                >
                  {sub.title}
                </button>
              ))
            )}
          </div>
        </CardContent>
      </Card>

      {/* Conteúdo das Seções */}
      <div className="space-y-4">
        {allSections.map((section) => {
          const sectionKey = section === manualContent.introducao ? 'introducao' : 'telas';
          const isOpen = expandedSections[`${sectionKey}-main`];
          
          return (
            <div key={sectionKey} id={`section-${sectionKey}`}>
              <ManualSection
                section={section}
                isOpen={isOpen}
                onToggle={() => {
                  setExpandedSections(prev => ({
                    ...prev,
                    [`${sectionKey}-main`]: !prev[`${sectionKey}-main`]
                  }));
                }}
              />
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-100">
        <CardContent className="pt-6">
          <div className="text-center space-y-3">
            <p className="text-sm font-medium text-gray-900">
              📞 Dúvidas não respondidas neste manual?
            </p>
            <p className="text-xs text-gray-600">
              Fale com seu gerente ou time de suporte imediatamente.
            </p>
            <p className="text-xs text-blue-700 font-semibold">
              🔐 Lembre-se: Todos os dados aqui são confidenciais e monitorados por auditoria LGPD (Art. 20).
            </p>
            <p className="text-xs text-gray-500 pt-3 border-t border-blue-200">
              Sankofa Enterprise Pro v1.0 | Manual Didático Completo | 30 de Novembro de 2025
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
