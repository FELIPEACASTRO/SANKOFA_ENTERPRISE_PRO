# 🧠 Use a Cabeça! Sankofa
## Aprenda Detecção de Fraudes de Forma Divertida e Eficaz

**Versão:** 11.0  
**Última Atualização:** 27 de Novembro de 2025  
**Metodologia:** Baseado no estilo "Head First"

---

# Como Usar Este Documento

Este documento foi criado pensando em VOCÊ! Usamos a metodologia **Head First** (Use a Cabeça) que combina:

- 🎯 **Perguntas provocativas** para ativar seu cérebro
- 🖼️ **Diagramas visuais** para facilitar a compreensão
- 💬 **Conversas** para tornar o conteúdo mais humano
- ✏️ **Exercícios práticos** para fixar o aprendizado
- 🎭 **Cenários reais** para você se imaginar na situação
- 🧩 **Quebra-cabeças** para desafiar seu pensamento

**Dica:** Não leia de forma passiva! Participe, responda as perguntas, faça os exercícios.

---

# Capítulo 1: O Que Diabos é Detecção de Fraude?

## 🤔 Pare e Pense

> Você está no caixa do supermercado. Passa seu cartão e... "TRANSAÇÃO NÃO AUTORIZADA".
> 
> Mas você TEM dinheiro na conta! O que aconteceu?

A resposta provavelmente é: **um sistema de detecção de fraudes achou que não era você**.

## O Grande Problema

```
        Imagine que você é um banco...
        
        ┌─────────────────────────────────────────────────────────┐
        │                                                          │
        │   300 MILHÕES de transações por dia passam por você     │
        │                                                          │
        │   Como saber quais são                                   │
        │                                                          │
        │        FRAUDE 🦹 vs LEGÍTIMA 👤 ?                        │
        │                                                          │
        └─────────────────────────────────────────────────────────┘
```

**É impossível** um humano analisar cada uma. Você precisaria de...

```
        300.000.000 transações/dia
        ÷ 86.400 segundos/dia
        ───────────────────────
        = 3.472 transações por SEGUNDO
        
        Mesmo que cada análise levasse 1 segundo,
        você precisaria de 3.472 analistas trabalhando
        24 horas por dia, sem parar!
```

## A Solução: Inteligência Artificial

É aí que entra o **SANKOFA**! 🦅

```
        ┌─────────────────────────────────────────────────────────┐
        │                                                          │
        │   Transação ──►  SANKOFA  ──► Decisão em 33ms           │
        │                    🦅                                    │
        │                                                          │
        │   "Esse PIX de R$ 50.000 às 3 da manhã                  │
        │    de um celular novo em outra cidade...                │
        │    parece FRAUDE!"                                      │
        │                                                          │
        └─────────────────────────────────────────────────────────┘
```

## 💡 Momento Eureka!

O Sankofa não é uma bola de cristal mágica. Ele **APRENDE** com o passado!

```
        ONTEM                          HOJE
        ─────                          ────
        
        João fez PIX de R$ 100         João faz PIX de R$ 100
        às 10h da manhã                às 10h da manhã
        de São Paulo                   de São Paulo
            │                              │
            ▼                              ▼
        NORMAL!                        NORMAL! ✅
        
        
        João fez PIX de R$ 100         João faz PIX de R$ 50.000
        às 10h da manhã                às 3h da manhã
        de São Paulo                   de Moscou
            │                              │
            ▼                              ▼
        NORMAL!                        ESPERA AÍ! 🚨
```

---

## ✏️ Exercício Rápido: Você é o Sankofa!

Analise estas transações. Qual parece mais suspeita?

| # | Cliente | Valor | Horário | Local | Histórico |
|---|---------|-------|---------|-------|-----------|
| A | Maria | R$ 200 | 15:00 | São Paulo | Sempre compra às 15h |
| B | Pedro | R$ 5.000 | 03:00 | Rio de Janeiro | Nunca transacionou de noite |
| C | Ana | R$ 150 | 20:00 | Belo Horizonte | Faz compras todo dia |

**Sua resposta:** ___

<details>
<summary>👉 Clique para ver a resposta</summary>

**Resposta: B (Pedro)**

Por quê?
- Horário incomum (3 da manhã)
- Valor relativamente alto
- Quebra do padrão histórico

Maria e Ana estão agindo conforme seu padrão normal!

</details>

---

# Capítulo 2: Como o Sankofa Pensa?

## 🧠 Não É Um Modelo, São TRÊS!

O Sankofa não confia em apenas uma opinião. Ele usa **Stacking** com 3 modelos:

```
        ┌─────────────────────────────────────────────────────────────┐
        │                STACKING ENSEMBLE (Comitê)                    │
        ├─────────────────────────────────────────────────────────────┤
        │                                                              │
        │            🌲               🎯                               │
        │          Random          Gradient                            │
        │          Forest          Boosting                            │
        │            │                │                                │
        │            ▼                ▼                                │
        │          85%              88%                                │
        │                                                              │
        │              ──────────────────────────                     │
        │                        │                                     │
        │                        ▼                                     │
        │                       📈                                     │
        │                   Logistic                                   │
        │                  Regression                                  │
        │               (Meta-Modelo)                                  │
        │                        │                                     │
        │                        ▼                                     │
        │              DECISÃO FINAL: 86%                             │
        │                                                              │
        └─────────────────────────────────────────────────────────────┘
```

## 💬 Conversa Entre Modelos

Imagine se os modelos pudessem conversar:

```
        🌲 Random Forest: "Essa transação me lembra 
            de outras fraudes que vi. 85% de chance!"
        
        🎯 Gradient Boosting: "Concordo! O horário 
            é muito suspeito. Digo 88%."
        
        📈 Logistic Regression (Meta): "Deixa eu 
            combinar essas opiniões de forma inteligente...
            Minha conclusão: 86% de chance de fraude."
        
        ────────────────────────────────────────────────
        
        🦅 Sankofa: "Como 86% é maior que 70%, 
            vou BLOQUEAR essa transação e 
            mandar para revisão humana!"
```

## 🎭 Cenário: O Caso do PIX da Madrugada

Era 3 da manhã quando o celular do Sr. Roberto vibrou:

> "PIX de R$ 15.000 bloqueado. Entre em contato com seu banco."

**O que aconteceu nos bastidores:**

```
        1️⃣ CHEGADA DA TRANSAÇÃO
        ───────────────────────────────────────────
        {
            "cliente": "Roberto Silva",
            "valor": 15000,
            "hora": 3,
            "canal": "PIX",
            "device": "iPhone 15 (NOVO!)",
            "local": "Curitiba" (Roberto mora em SP)
        }
        
        
        2️⃣ EXTRAÇÃO DE CARACTERÍSTICAS (Features)
        ───────────────────────────────────────────
        
        O Sankofa extraiu 47 características:
        
        ✓ is_night = TRUE (é de madrugada)
        ✓ amount_zscore = 4.2 (valor 4x acima do normal)
        ✓ new_device = TRUE (celular nunca visto)
        ✓ location_entropy = 0.8 (local diferente do habitual)
        ✓ hour_pattern_deviation = HIGH (nunca fez PIX às 3h)
        
        
        3️⃣ VOTAÇÃO DOS MODELOS
        ───────────────────────────────────────────
        
        🌲 Random Forest: 92% - "Padrão claro de fraude!"
        🎯 Gradient Boosting: 89% - "Device novo + madrugada = perigo"
        🌳 Extra Trees: 87% - "Muito diferente do histórico"
        📈 Logistic Regression: 85% - "Combinação perigosa"
        🔮 SVC: 91% - "Vários red flags juntos"
        
        RESULTADO: 88.8% de probabilidade de fraude
        
        
        4️⃣ DECISÃO
        ───────────────────────────────────────────
        
        Como 88.8% > 70% (threshold)...
        
        🚫 TRANSAÇÃO BLOQUEADA
        📱 SMS enviado para Roberto
        🔔 Alerta criado para analista
        
        
        5️⃣ INVESTIGAÇÃO HUMANA
        ───────────────────────────────────────────
        
        O analista ligou para Roberto:
        
        📞 "Sr. Roberto, tentaram fazer um PIX 
            de R$ 15.000 do seu celular às 3h?"
        
        👤 "Não! Eu estava dormindo! Meu celular 
            foi roubado ontem!"
        
        ✅ FRAUDE CONFIRMADA
        💰 R$ 15.000 salvos!
```

---

## 🧩 Quebra-Cabeça: Monte as Peças!

Ordene as etapas corretas do processo de detecção:

| Peça | Etapa |
|------|-------|
| 🅰️ | Modelos votam na probabilidade |
| 🅱️ | Transação chega na API |
| 🅲️ | Features são extraídas |
| 🅳️ | Analista investiga (se necessário) |
| 🅴️ | Decisão: Aprovar/Bloquear/Revisar |

**Ordem correta:** __ → __ → __ → __ → __

<details>
<summary>👉 Ver resposta</summary>

**Ordem: 🅱️ → 🅲️ → 🅰️ → 🅴️ → 🅳️**

1. Transação chega
2. Features extraídas
3. Modelos votam
4. Sistema decide
5. Humano revisa (se score médio)

</details>

---

# Capítulo 3: O Que o Sankofa Analisa?

## 47+ Características Sob a Lupa 🔍

O Sankofa não olha só o valor. Ele analisa **47 características** de cada transação!

## As 5 Categorias de Features

```
        ┌───────────────────────────────────────────────────────────┐
        │                                                            │
        │   ⏰ TEMPORAIS          🏦 VALOR           📍 GEOGRÁFICAS  │
        │   ────────────         ─────────          ─────────────   │
        │   - Hora do dia        - Valor em R$      - Cidade        │
        │   - Dia da semana      - Log do valor     - Estado        │
        │   - É fim de semana?   - É valor redondo? - País          │
        │   - É noite?           - Z-score          - Distância     │
        │   - É horário comercial?                                   │
        │                                                            │
        │                                                            │
        │   🏃 COMPORTAMENTAIS                 📊 ENTROPIA          │
        │   ──────────────────                 ────────────          │
        │   - Velocidade (txn/hora)            - Diversidade de     │
        │   - Desvio do padrão                   locais             │
        │   - É comerciante novo?              - Locais únicos      │
        │   - Mudou de dispositivo?            - Padrão geográfico  │
        │                                                            │
        └───────────────────────────────────────────────────────────┘
```

## 💡 Feature Destaque: Location Entropy

**O que é isso?** Mede o quão "espalhado" são os locais de transação de uma pessoa.

```
        CLIENTE A: Baixa Entropia           CLIENTE B: Alta Entropia
        ─────────────────────────           ──────────────────────────
        
              São Paulo                          São Paulo
                 ⬤                                  ⬤
                 │                          Curitiba  │  Rio
                 │                              ⬤ ── ⬤ ── ⬤
                 │                              │     │     │
             Sempre aqui!                   Recife  Brasília  Salvador
                                               ⬤      ⬤       ⬤
                                               
        Entropia = 0.0                      Entropia = 0.9
        (Previsível)                        (Imprevisível)
        
        
        Se Cliente A de repente fizer
        transação em Recife... 🚨 ALERTA!
        
        Se Cliente B fizer transação
        em Recife... Normal para ele!
```

## 🎯 Feature Destaque: is_night (Correção Crítica!)

Em versões anteriores, tínhamos um bug! Veja:

```
        ❌ ANTES (BUG):
        ─────────────────────────────────────────
        is_night = hour.between(22, 6)
        
        Problema: between(22, 6) não funciona!
        22 < 6 é FALSE, então NINGUÉM era noturno!
        
        
        ✅ DEPOIS (CORRIGIDO na v11):
        ─────────────────────────────────────────
        is_night = (hour >= 22) | (hour <= 6)
        
        Agora funciona! 
        22h, 23h, 0h, 1h, 2h, 3h, 4h, 5h, 6h → NOITE! ✓
```

---

# Capítulo 4: O Dashboard - Sua Central de Comando

## 🎮 Conheça Seu Cockpit

Quando você abre o Sankofa, você é o **piloto**. O Dashboard é seu painel de instrumentos.

```
        ╔═══════════════════════════════════════════════════════════╗
        ║                     DASHBOARD EXECUTIVO                    ║
        ╠═══════════════════════════════════════════════════════════╣
        ║                                                            ║
        ║   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────┐ ║
        ║   │    518      │ │     23      │ │   95.6%     │ │33.5 │ ║
        ║   │ Transações  │ │   Fraudes   │ │  Aprovação  │ │ ms  │ ║
        ║   │    Hoje     │ │ Detectadas  │ │    Taxa     │ │Lat. │ ║
        ║   │     ↑       │ │      =      │ │      ↓      │ │  =  │ ║
        ║   └─────────────┘ └─────────────┘ └─────────────┘ └─────┘ ║
        ║                                                            ║
        ║        🟢 Tudo OK    🟢 Normal    🟢 Bom    🟢 Rápido     ║
        ║                                                            ║
        ╚═══════════════════════════════════════════════════════════╝
```

## 🚦 O Semáforo dos Indicadores

| Indicador | 🟢 Verde | 🟡 Amarelo | 🔴 Vermelho |
|-----------|----------|------------|-------------|
| Taxa Aprovação | >95% | 90-95% | <90% |
| Latência | <50ms | 50-100ms | >100ms |
| Detecção | >90% | 80-90% | <80% |
| Falso Positivo | <3% | 3-5% | >5% |

## 💬 Se o Dashboard Pudesse Falar...

```
        ┌────────────────────────────────────────────────────┐
        │                                                     │
        │  "Olá! Hoje processamos 518 transações.            │
        │   Dessas, bloqueei 23 que pareciam fraude.         │
        │                                                     │
        │   95.6% dos seus clientes foram aprovados          │
        │   automaticamente - eles nem perceberam            │
        │   que eu estava ali protegendo!                    │
        │                                                     │
        │   Minha velocidade média foi de 33.5 milissegundos.│
        │   Isso é mais rápido que um piscar de olhos!"      │
        │                                                     │
        │                                 - Seu Sankofa 🦅    │
        │                                                     │
        └────────────────────────────────────────────────────┘
```

---

# Capítulo 5: Quando Você Entra em Ação

## 👤 O Papel do Humano

O Sankofa é muito bom, mas não é perfeito. Por isso existe a **Revisão Manual**.

```
        ZONA DE DECISÃO AUTOMÁTICA        ZONA CINZA        ZONA DE BLOQUEIO
        ─────────────────────────        ─────────        ────────────────────
        
        Score: 0 ────────── 40 ────────── 70 ────────── 100
                    │                 │                │
                    ▼                 ▼                ▼
               APROVAR            REVISAR          BLOQUEAR
               automático         HUMANO!          automático
```

## 🎭 Cenário: Você é o Revisor

São 10h da manhã. Você abre a fila de revisão e vê:

```
        ┌──────────────────────────────────────────────────────────┐
        │              FILA DE REVISÃO MANUAL                       │
        ├──────────────────────────────────────────────────────────┤
        │                                                           │
        │  🟠 TXN-001 │ R$ 8.500 │ Score: 65 │ PIX │ SLA: 4min     │
        │     ├── Cliente: José Santos                              │
        │     ├── Horário: 09:45 (normal para ele)                  │
        │     ├── Local: São Paulo (onde mora)                      │
        │     └── Motivo: Valor 2x maior que média                  │
        │                                                           │
        │  [APROVAR]  [BLOQUEAR]  [ESCALAR]                        │
        │                                                           │
        └──────────────────────────────────────────────────────────┘
```

**O que você faz?**

🤔 **Pensando...**

- Horário é normal para o cliente ✅
- Local é onde ele mora ✅
- Só o valor que é maior que o habitual ⚠️

**Decisão provável:** APROVAR

Por quê? Apenas UM fator de risco, e não é muito grave.

---

## ✏️ Exercício: Você Decide!

Analise estes casos e decida:

**Caso 1:**
- Score: 72
- Valor: R$ 50.000
- Horário: 3h da manhã
- Local: Diferente do habitual
- Device: Novo
- Cliente há: 2 anos

**Sua decisão:** _______________

**Caso 2:**
- Score: 55
- Valor: R$ 1.200
- Horário: 14h
- Local: Mesmo de sempre
- Device: Mesmo de sempre
- Cliente há: 5 anos

**Sua decisão:** _______________

<details>
<summary>👉 Ver respostas</summary>

**Caso 1: BLOQUEAR ou ESCALAR**
- Muitos red flags juntos
- Valor muito alto + madrugada + local diferente + device novo
- Melhor errar bloqueando que liberar fraude de R$ 50k

**Caso 2: APROVAR**
- Score moderado
- Valor baixo
- Tudo consistente com histórico
- Cliente antigo e confiável

</details>

---

# Capítulo 6: Calibrando Seu Sankofa

## ⚙️ O Que é Calibragem?

É como ajustar a "sensibilidade" de um detector de metais:

```
        MUITO SENSÍVEL                      POUCO SENSÍVEL
        ──────────────                      ────────────────
        
        "BIIIP!" para                       Só apita para
        qualquer coisa                      metais grandes
        
        Detecta TUDO,                       Pode deixar
        mas muito alarme                    passar algo
        falso!                              perigoso!
        
              │                                   │
              ▼                                   ▼
        Muitos Falsos                       Pode perder
        Positivos                           Fraudes Reais
```

## 🎚️ Os Controles de Ajuste

Na página de Calibragem, você tem sliders para ajustar:

```
        THRESHOLD (Limite)
        ─────────────────────────────────────────────────
        
        0% ════════════════════●══════════════════ 100%
                              70%
        
        ↑ Baixar = Mais bloqueios, menos fraudes passam
        ↓ Subir = Menos bloqueios, mais fraudes podem passar
        
        
        PESO NO ENSEMBLE
        ─────────────────────────────────────────────────
        
        0.0 ════●═══════════════════════════════════ 0.5
              0.15
        
        Quanto "voz" esse algoritmo específico tem na votação
```

## ⚠️ CUIDADO!

```
        ╔════════════════════════════════════════════════════╗
        ║                     ⚠️ ATENÇÃO                      ║
        ╠════════════════════════════════════════════════════╣
        ║                                                     ║
        ║  Mudar a calibragem afeta TODAS as transações!     ║
        ║                                                     ║
        ║  Antes de mexer:                                    ║
        ║  1. Entenda o impacto                               ║
        ║  2. Faça mudanças pequenas                          ║
        ║  3. Monitore os resultados                          ║
        ║  4. Documente o que fez                             ║
        ║                                                     ║
        ║  Na dúvida, pergunte ao supervisor!                 ║
        ║                                                     ║
        ╚════════════════════════════════════════════════════╝
```

---

# Capítulo 7: Quando Algo Dá Errado

## 🔔 Alertas - Seu Sistema de Aviso

O Sankofa te avisa quando algo precisa de atenção:

```
        ┌────────────────────────────────────────────────────┐
        │                                                     │
        │  🔴 CRÍTICO - Sistema fora do ar                   │
        │     → Ação: Ligar para TI AGORA                    │
        │                                                     │
        │  🟠 ALTO - Pico de fraudes detectado               │
        │     → Ação: Investigar imediatamente               │
        │                                                     │
        │  🟡 MÉDIO - Latência acima do normal               │
        │     → Ação: Monitorar, pode ser temporário         │
        │                                                     │
        │  🔵 BAIXO - Nova versão do modelo disponível       │
        │     → Ação: Agendar atualização                    │
        │                                                     │
        └────────────────────────────────────────────────────┘
```

## 🔍 Monitoramento - A Saúde do Sistema

Assim como você faz check-up médico, o Sankofa tem sua própria "saúde":

```
        STATUS GERAL: ✅ SAUDÁVEL
        ───────────────────────────────────────
        
        💻 CPU .......... [████████░░] 80% OK
        💾 Memória ...... [██████░░░░] 60% OK
        📀 Disco ........ [████░░░░░░] 40% OK
        🌐 Rede ......... [██░░░░░░░░] 20% OK
        
        🤖 Modelos ...... 5/5 ativos ✅
        ⚡ Latência ..... 33ms (< 50ms) ✅
        🎯 Precisão ..... 94.2% (> 90%) ✅
        📊 Uptime ....... 15 dias ✅
```

---

# Capítulo 8: Resumão - Tudo Que Você Precisa Lembrar

## 📋 Checklist Diário do Analista

```
        ☐ Verificar Dashboard - tudo verde?
        ☐ Checar fila de Revisão Manual
        ☐ Resolver alertas pendentes
        ☐ Investigar casos críticos
        ☐ Documentar decisões importantes
```

## 🧠 As 5 Coisas Mais Importantes

```
        1️⃣ O SANKOFA usa STACKING com 3 modelos (2 base + 1 meta)
        
        2️⃣ Random Forest + Gradient Boosting → Logistic Regression
        
        3️⃣ SCORE > 70 = Bloqueio | 40-70 = Revisão | < 40 = Aprova
        
        4️⃣ VOCÊ decide nos casos de zona cinza (Human-in-the-Loop)
        
        5️⃣ MUDANÇAS na calibragem afetam TODO mundo
```

## 🎯 Seu Mantra

```
        ╔══════════════════════════════════════════════════════╗
        ║                                                       ║
        ║   "Na dúvida, é melhor bloquear e investigar         ║
        ║    do que liberar e lamentar."                       ║
        ║                                                       ║
        ║                    - Sabedoria de Analista de Fraude ║
        ║                                                       ║
        ╚══════════════════════════════════════════════════════╝
```

---

# Quiz Final: Teste Seus Conhecimentos!

## Pergunta 1
Quantos modelos de ML compõem o stacking ensemble do Sankofa?
- A) 1
- B) 2 base + 1 meta = 3
- C) 5
- D) 10

## Pergunta 2
Qual é o threshold padrão para bloqueio automático?
- A) 50%
- B) 60%
- C) 70%
- D) 80%

## Pergunta 3
O que significa "Location Entropy alta"?
- A) Cliente está em local perigoso
- B) Cliente transaciona de muitos locais diferentes
- C) Sistema de GPS está com problema
- D) Local não identificado

## Pergunta 4
Quando uma transação vai para Revisão Manual?
- A) Sempre
- B) Quando score é maior que 100
- C) Quando score está entre 40-70
- D) Nunca, tudo é automático

## Pergunta 5
O que você deve fazer PRIMEIRO ao abrir o Sankofa?
- A) Ir direto para Calibragem
- B) Verificar o Dashboard
- C) Gerar relatórios
- D) Mudar as configurações

<details>
<summary>👉 Ver respostas</summary>

1. **B) 2 base + 1 meta = 3** - Random Forest + Gradient Boosting (base) e Logistic Regression (meta)
2. **C) 70%** - Acima disso é bloqueio automático
3. **B) Cliente transaciona de muitos locais diferentes** - Mede diversidade geográfica
4. **C) Quando score está entre 40-70** - A "zona cinza"
5. **B) Verificar o Dashboard** - Sempre comece vendo o panorama geral!

</details>

---

# Parabéns! 🎉

Você completou o guia "Use a Cabeça! Sankofa"!

Agora você entende:
- ✅ Como o Sankofa detecta fraudes
- ✅ Por que usa múltiplos modelos
- ✅ O que cada feature significa
- ✅ Quando o humano entra em ação
- ✅ Como calibrar o sistema
- ✅ Como monitorar a saúde

**Próximo passo:** Coloque em prática! Abra o Sankofa e explore.

---

*"Aprender é uma jornada, não um destino."*

*Use a Cabeça! Sankofa v11.0*  
*27 de Novembro de 2025*
