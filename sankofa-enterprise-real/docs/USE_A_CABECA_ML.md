# Use a Cabeca: Machine Learning para Deteccao de Fraude

## Uma Aula Completa sobre Como a IA Protege Seu Dinheiro

```
+==============================================================================+
|                                                                               |
|     ██╗   ██╗███████╗███████╗     █████╗                                     |
|     ██║   ██║██╔════╝██╔════╝    ██╔══██╗                                    |
|     ██║   ██║███████╗█████╗      ███████║                                    |
|     ██║   ██║╚════██║██╔══╝      ██╔══██║                                    |
|     ╚██████╔╝███████║███████╗    ██║  ██║                                    |
|      ╚═════╝ ╚══════╝╚══════╝    ╚═╝  ╚═╝                                    |
|                                                                               |
|      ██████╗ █████╗ ██████╗ ███████╗ ██████╗ █████╗ ██╗                      |
|     ██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗██║                      |
|     ██║     ███████║██████╔╝█████╗  ██║     ███████║██║                      |
|     ██║     ██╔══██║██╔══██╗██╔══╝  ██║     ██╔══██║╚═╝                      |
|     ╚██████╗██║  ██║██████╔╝███████╗╚██████╗██║  ██║██╗                      |
|      ╚═════╝╚═╝  ╚═╝╚═════╝ ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝                      |
|                                                                               |
|              MACHINE LEARNING PARA DETECCAO DE FRAUDE                        |
|              ═══════════════════════════════════════════                     |
|                                                                               |
|              Uma Aula Completa com Exemplos do Dia a Dia                     |
|                                                                               |
+==============================================================================+
```

**Autor:** Equipe Sankofa Enterprise Pro  
**Versao:** 12.1  
**Data:** 27 de Novembro de 2025  
**Publico:** Analistas de fraude, desenvolvedores, curiosos sobre IA

---

# INDICE DA AULA

```
+==============================================================================+
|                         MAPA DA SUA JORNADA                                  |
+==============================================================================+
|                                                                               |
|   ATO 0: ANTES DE COMECAR                                                    |
|   ════════════════════════                                                   |
|   • O que voce vai aprender                                                  |
|   • Pre-requisitos (nenhum!)                                                 |
|   • Como usar esta aula                                                      |
|                                                                               |
|   ATO 1: A JORNADA DE UMA TRANSACAO                                          |
|   ═══════════════════════════════════                                        |
|   • Do PIX ao Score: 300 milissegundos                                       |
|   • Os 3 Guardas da Primeira Linha (Random Forest, XGBoost, LightGBM)       |
|   • O Juiz Final: Stacking Ensemble                                          |
|                                                                               |
|   ATO 2: OS ESPECIALISTAS ENTRAM EM CENA                                     |
|   ═══════════════════════════════════════                                    |
|   • LSTM/GRU: O Detetive com Memoria                                         |
|   • TabTransformer: O Caso Stripe de $6 Bilhoes                              |
|   • Autoencoders/VAE: Os Cacadores de Anomalias                              |
|   • GNN: O Investigador de Redes Criminosas                                  |
|   • Federated Learning: A Alianca dos Bancos                                 |
|                                                                               |
|   ATO 3: A SALA DE GUERRA                                                    |
|   ═══════════════════════                                                    |
|   • Metricas que Importam                                                    |
|   • Cronologia Completa: Quem Faz o Que                                      |
|   • Dashboard do Analista                                                    |
|   • Exercicios Praticos                                                      |
|                                                                               |
|   ANEXOS                                                                     |
|   ══════                                                                     |
|   • AS 47 FEATURES EXPLICADAS (NOVO!)                                        |
|   • Glossario Visual                                                         |
|   • Tabela de Metricas por Modelo                                            |
|   • Referencias e Leitura Adicional                                          |
|                                                                               |
+==============================================================================+
```

---

# AS 47 FEATURES: O FORMULARIO DE SEGURANCA

## O Que Sao Features?

```
+==============================================================================+
|                         FEATURES = PERGUNTAS DE SEGURANCA                    |
+==============================================================================+
|                                                                               |
|   Antes dos modelos de IA analisarem uma transacao, o sistema               |
|   EXTRAI 47 informacoes (features) sobre ela.                               |
|                                                                               |
|   ANALOGIA: E como o formulario que voce preenche na imigracao:             |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  FORMULARIO DE IMIGRACAO              FORMULARIO DE TRANSACAO            │|
|   │  ═══════════════════════              ═══════════════════════            │|
|   │                                                                          │|
|   │  Nome: ____________                   CPF: ____________                  │|
|   │  Pais de origem: ______               Conta origem: ______               │|
|   │  Destino: ____________                Conta destino: ______              │|
|   │  Motivo da viagem: ____               Tipo transacao: ____               │|
|   │  Quanto dinheiro traz: __             Valor: R$ ________                 │|
|   │  Ja visitou antes? ____               Ja transferiu antes? __            │|
|   │  Vem de onde? _________               Qual dispositivo? ____             │|
|   │  Vai ficar quanto tempo?              Qual horario? ________             │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   Cada resposta ajuda a IA a decidir se a transacao e suspeita.             |
|                                                                               |
+==============================================================================+
```

## Diagrama: Features Alimentando os Modelos

```
+==============================================================================+
|                    COMO AS FEATURES ALIMENTAM OS MODELOS                     |
+==============================================================================+
|                                                                               |
|                           TRANSACAO                                          |
|                       (PIX de R$ 5.000)                                      |
|                              │                                               |
|                              ▼                                               |
|   ┌──────────────────────────────────────────────────────────────────────┐   |
|   │                    EXTRACAO DE 47 FEATURES                            │   |
|   │                                                                       │   |
|   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │   |
|   │  │TEMPORAIS│ │  VALOR  │ │ CLIENTE │ │ DEVICE  │ │  LOCAL  │         │   |
|   │  │ 7 feat  │ │ 6 feat  │ │ 8 feat  │ │ 4 feat  │ │ 5 feat  │         │   |
|   │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘         │   |
|   │       │           │           │           │           │              │   |
|   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │   |
|   │  │ CANAL   │ │VELOCID. │ │MERCHANT │ │  PIX    │ │ CARTAO  │         │   |
|   │  │ 3 feat  │ │ 4 feat  │ │ 3 feat  │ │ 4 feat  │ │ 3 feat  │         │   |
|   │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘         │   |
|   │       │           │           │           │           │              │   |
|   │       └───────────┴───────────┴───────────┴───────────┘              │   |
|   │                              │                                        │   |
|   │                     47 FEATURES PRONTAS                               │   |
|   │                              │                                        │   |
|   └──────────────────────────────┼───────────────────────────────────────┘   |
|                                  │                                            |
|          ┌───────────────────────┼───────────────────────┐                   |
|          │                       │                       │                   |
|          ▼                       ▼                       ▼                   |
|   ┌──────────────┐       ┌──────────────┐       ┌──────────────┐            |
|   │    RANDOM    │       │   XGBOOST    │       │   LIGHTGBM   │            |
|   │    FOREST    │       │              │       │              │            |
|   │  Usa 47 feat │       │  Usa 47 feat │       │  Usa 47 feat │            |
|   │  Score: 0.72 │       │  Score: 0.68 │       │  Score: 0.75 │            |
|   └──────────────┘       └──────────────┘       └──────────────┘            |
|                                                                               |
+==============================================================================+
```

## Tabela Completa: As 47 Features

### Grupo 1: Features Temporais (7 features)

```
+==============================================================================+
|                    GRUPO 1: FEATURES TEMPORAIS                               |
+==============================================================================+
|                                                                               |
|   Essas features respondem: "QUANDO a transacao aconteceu?"                  |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  #  FEATURE            DESCRICAO                 PESO    EXEMPLO        │|
|   │  ══════════════════════════════════════════════════════════════════════│|
|   │                                                                          │|
|   │  1  hour               Hora do dia (0-23)        0.08    14 (2h tarde)  │|
|   │                                                                          │|
|   │  2  day_of_week        Dia da semana (0-6)       0.03    1 (terca)      │|
|   │                                                                          │|
|   │  3  is_weekend         E final de semana?        0.05    0 (nao)        │|
|   │                        1=sim, 0=nao                                      │|
|   │                                                                          │|
|   │  4  is_night           E horario noturno?        0.12    0 (nao)        │|
|   │                        (22h-6h)                         ⚠️ ALTO PESO    │|
|   │                                                                          │|
|   │  5  is_business_hours  Horario comercial?        0.04    1 (sim)        │|
|   │                        (9h-18h)                                          │|
|   │                                                                          │|
|   │  6  is_early_morning   Madrugada? (0h-6h)        0.15    0 (nao)        │|
|   │                                                         ⚠️ ALTO PESO    │|
|   │                                                                          │|
|   │  7  timestamp          Data/hora exata           0.02    2025-11-27...  │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   💡 INSIGHT: Transacoes de madrugada tem 8x mais chance de ser fraude!     |
|                                                                               |
+==============================================================================+
```

### Grupo 2: Features de Valor (6 features)

```
+==============================================================================+
|                    GRUPO 2: FEATURES DE VALOR                                |
+==============================================================================+
|                                                                               |
|   Essas features respondem: "QUANTO dinheiro esta envolvido?"                |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  #  FEATURE            DESCRICAO                 PESO    EXEMPLO        │|
|   │  ══════════════════════════════════════════════════════════════════════│|
|   │                                                                          │|
|   │  8  amount             Valor da transacao        0.18    5000.00        │|
|   │                        (em reais)                       ⚠️ MAIOR PESO   │|
|   │                                                                          │|
|   │  9  log_value          Log do valor              0.08    8.52           │|
|   │                        (suaviza valores altos)                           │|
|   │                                                                          │|
|   │  10 amount_zscore      Desvio do padrao          0.14    2.3 (alto!)    │|
|   │                        (vs historico do cliente)        ⚠️ ALTO PESO    │|
|   │                                                                          │|
|   │  11 value_rounded      Valor e "redondo"?        0.04    1 (sim, R$5000)│|
|   │                        (1000, 5000, 10000)                               │|
|   │                                                                          │|
|   │  12 is_high_value      Valor > R$ 5.000?         0.10    1 (sim)        │|
|   │                                                                          │|
|   │  13 is_very_high_value Valor > R$ 10.000?        0.12    0 (nao)        │|
|   │                                                         ⚠️ ALTO PESO    │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   💡 INSIGHT: "amount" e a feature MAIS importante! 18% do peso total.      |
|                                                                               |
|   EXEMPLO PRATICO:                                                           |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  Maria geralmente faz PIX de R$ 200-500.                                │|
|   │  Hoje ela tenta fazer PIX de R$ 8.000.                                  │|
|   │                                                                          │|
|   │  amount = 8000 (alto)                                                   │|
|   │  amount_zscore = 4.2 (MUITO acima do padrao!)                           │|
|   │  is_high_value = 1                                                      │|
|   │                                                                          │|
|   │  → Sistema aciona alerta para verificacao!                              │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

### Grupo 3: Features de Comportamento do Cliente (8 features)

```
+==============================================================================+
|                    GRUPO 3: FEATURES DE COMPORTAMENTO                        |
+==============================================================================+
|                                                                               |
|   Essas features respondem: "Isso COMBINA com o historico do cliente?"       |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  #  FEATURE                 DESCRICAO                PESO   EXEMPLO     │|
|   │  ══════════════════════════════════════════════════════════════════════│|
|   │                                                                          │|
|   │  14 user_history            Score de historico       0.10   0.85        │|
|   │                             (0=novo, 1=confiavel)                        │|
|   │                                                                          │|
|   │  15 amount_deviation_zscore Desvio do valor          0.14   2.3         │|
|   │                             vs media historica              ⚠️ ALTO     │|
|   │                                                                          │|
|   │  16 tx_count_7d             Qtd transacoes           0.06   12          │|
|   │                             nos ultimos 7 dias                           │|
|   │                                                                          │|
|   │  17 amount_normalized_hour  Valor vs media           0.05   1.8         │|
|   │                             daquele horario                              │|
|   │                                                                          │|
|   │  18 amount_channel_ratio    Valor vs media           0.04   0.9         │|
|   │                             daquele canal                                │|
|   │                                                                          │|
|   │  19 high_amount_suspicious  Valor alto + hora        0.15   1 (alerta!) │|
|   │     _hour                   suspeita? (0h-4h)               ⚠️ ALTO     │|
|   │                                                                          │|
|   │  20 account_age_days        Idade da conta           0.08   2920        │|
|   │                             (em dias)                       (8 anos)     │|
|   │                                                                          │|
|   │  21 previous_frauds         Fraudes anteriores       0.20   0           │|
|   │                             nessa conta                     ⚠️ CRITICO  │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   💡 INSIGHT: Contas com fraudes anteriores tem 15x mais chance de fraude!  |
|                                                                               |
+==============================================================================+
```

### Grupo 4: Features de Dispositivo (4 features)

```
+==============================================================================+
|                    GRUPO 4: FEATURES DE DISPOSITIVO                          |
+==============================================================================+
|                                                                               |
|   Essas features respondem: "DE ONDE a transacao esta sendo feita?"          |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  #  FEATURE            DESCRICAO                 PESO    EXEMPLO        │|
|   │  ══════════════════════════════════════════════════════════════════════│|
|   │                                                                          │|
|   │  22 device_risk        Score de risco do         0.12    0.15           │|
|   │                        dispositivo (0-1)                ⚠️ ALTO         │|
|   │                                                                          │|
|   │  23 device_trust       Confianca no device       0.10    0.92           │|
|   │                        (0=novo, 1=conhecido)                             │|
|   │                                                                          │|
|   │  24 device_fingerprint Device ja usado?          0.08    1 (sim)        │|
|   │                        (1=sim, 0=novo)                                   │|
|   │                                                                          │|
|   │  25 device_change      Trocou de device          0.14    0 (nao)        │|
|   │                        recentemente?                    ⚠️ ALTO         │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   EXEMPLO PRATICO:                                                           |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  Maria sempre usa iPhone 13 (device_trust = 0.95)                       │|
|   │  Hoje a transacao vem de um Android desconhecido:                       │|
|   │                                                                          │|
|   │  device_trust = 0.10 (MUITO BAIXO!)                                     │|
|   │  device_change = 1 (SIM!)                                               │|
|   │  device_fingerprint = 0 (NUNCA VISTO!)                                  │|
|   │                                                                          │|
|   │  → Sistema bloqueia e pede biometria!                                   │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

### Grupo 5: Features de Localizacao (5 features)

```
+==============================================================================+
|                    GRUPO 5: FEATURES DE LOCALIZACAO                          |
+==============================================================================+
|                                                                               |
|   Essas features respondem: "A LOCALIZACAO faz sentido?"                     |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  #  FEATURE              DESCRICAO               PESO    EXEMPLO        │|
|   │  ══════════════════════════════════════════════════════════════════════│|
|   │                                                                          │|
|   │  26 location_risk        Score de risco da       0.10    0.20           │|
|   │                          localizacao (0-1)                               │|
|   │                                                                          │|
|   │  27 location_entropy     Variedade de locais     0.06    0.45           │|
|   │                          (0=sempre mesmo,                                │|
|   │                           1=muito variado)                               │|
|   │                                                                          │|
|   │  28 unique_locations     Qtd locais unicos       0.04    3              │|
|   │     _count               nos ultimos 30 dias                             │|
|   │                                                                          │|
|   │  29 is_new_location      Localizacao nova?       0.12    0 (nao)        │|
|   │                          (nunca usou antes)             ⚠️ ALTO         │|
|   │                                                                          │|
|   │  30 impossible_travel    "Viagem impossivel"?    0.25    0 (nao)        │|
|   │                          (ex: SP→RJ em 5min)            ⚠️ CRITICO      │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   💡 INSIGHT: "Viagem impossivel" e o maior indicador de cartao clonado!    |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  EXEMPLO: VIAGEM IMPOSSIVEL                                             │|
|   │                                                                          │|
|   │  14:30 - Transacao em BRASILIA (Fernanda comprando cafe)                │|
|   │  14:35 - Transacao em RECIFE (criminoso usando cartao clonado)          │|
|   │                                                                          │|
|   │  Distancia: 1.600 km                                                    │|
|   │  Tempo: 5 minutos                                                       │|
|   │  Velocidade necessaria: 19.200 km/h (IMPOSSIVEL!)                       │|
|   │                                                                          │|
|   │  impossible_travel = 1 → BLOQUEIO IMEDIATO!                             │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

### Grupo 6: Features de Canal e Tipo (3 features)

```
+==============================================================================+
|                    GRUPO 6: FEATURES DE CANAL                                |
+==============================================================================+
|                                                                               |
|   Essas features respondem: "POR QUAL CANAL a transacao foi feita?"          |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  #  FEATURE            DESCRICAO                 PESO    EXEMPLO        │|
|   │  ══════════════════════════════════════════════════════════════════════│|
|   │                                                                          │|
|   │  31 channel_risk       Score de risco do         0.08    0.35           │|
|   │                        canal (0-1)                                       │|
|   │                                                                          │|
|   │  32 transaction_type   Tipo (PIX, TED, etc)      0.05    "PIX"          │|
|   │                        (codificado numericamente)                        │|
|   │                                                                          │|
|   │  33 online_purchase    Compra online?            0.10    0 (nao)        │|
|   │                        (maior risco)                                     │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   RANKING DE RISCO POR CANAL:                                                |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  CANAL              RISCO        MOTIVO                                 │|
|   │  ═════════════════════════════════════════════════════════════════════  │|
|   │  PIX                ALTO         Irreversivel, instantaneo              │|
|   │  E-commerce         ALTO         Cartao nao presente                    │|
|   │  TED/DOC            MEDIO        Pode ser revertido                     │|
|   │  Debito presencial  BAIXO        Cartao + senha                         │|
|   │  Credito presencial BAIXO        Chargeback possivel                    │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

### Grupo 7: Features de Velocidade (4 features)

```
+==============================================================================+
|                    GRUPO 7: FEATURES DE VELOCIDADE                           |
+==============================================================================+
|                                                                               |
|   Essas features respondem: "QUANTAS transacoes em POUCO TEMPO?"             |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  #  FEATURE            DESCRICAO                 PESO    EXEMPLO        │|
|   │  ══════════════════════════════════════════════════════════════════════│|
|   │                                                                          │|
|   │  34 velocity_score     Score de velocidade       0.15    0.20           │|
|   │                        geral (0-1)                      ⚠️ ALTO         │|
|   │                                                                          │|
|   │  35 velocity_1h        Qtd transacoes na         0.12    2              │|
|   │                        ultima hora                      ⚠️ ALTO         │|
|   │                                                                          │|
|   │  36 velocity_5min      Qtd transacoes nos        0.18    0              │|
|   │                        ultimos 5 minutos                ⚠️ CRITICO      │|
|   │                                                                          │|
|   │  37 inter_event_time   Tempo desde a             0.08    3600           │|
|   │                        ultima transacao (seg)           (1 hora)        │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   💡 INSIGHT: Criminosos tentam agir RAPIDO antes do bloqueio!              |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  EXEMPLO: CARD TESTING (Caso Stripe)                                    │|
|   │                                                                          │|
|   │  23:45:00 - tx $1.00 (velocity_5min = 1)                                │|
|   │  23:45:01 - tx $1.00 (velocity_5min = 2)                                │|
|   │  23:45:02 - tx $1.00 (velocity_5min = 3)                                │|
|   │  ...                                                                     │|
|   │  23:55:00 - tx $1.00 (velocity_5min = 600!)                             │|
|   │                                                                          │|
|   │  inter_event_time = 1 segundo (IMPOSSIVEL para humano!)                 │|
|   │  → BLOQUEIO + Alerta de CARD TESTING                                    │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

### Grupo 8: Features de Merchant (3 features)

```
+==============================================================================+
|                    GRUPO 8: FEATURES DE MERCHANT                             |
+==============================================================================+
|                                                                               |
|   Essas features respondem: "PARA QUEM o dinheiro esta indo?"                |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  #  FEATURE            DESCRICAO                 PESO    EXEMPLO        │|
|   │  ══════════════════════════════════════════════════════════════════════│|
|   │                                                                          │|
|   │  38 merchant_risk      Score de risco do         0.12    0.15           │|
|   │                        comerciante (0-1)                ⚠️ ALTO         │|
|   │                                                                          │|
|   │  39 merchant_category  Categoria (MCC)           0.06    5411           │|
|   │                        (supermercado, posto...)         (mercado)        │|
|   │                                                                          │|
|   │  40 is_new_merchant    Comerciante novo?         0.10    0 (nao)        │|
|   │                        (nunca comprou antes)                             │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   CATEGORIAS DE ALTO RISCO (MCC):                                            |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  MCC      CATEGORIA            RISCO    MOTIVO                          │|
|   │  ═══════════════════════════════════════════════════════════════════    │|
|   │  7995     Apostas/Gambling     ALTO     Lavagem de dinheiro             │|
|   │  5967     Direct Marketing     ALTO     Golpes comuns                   │|
|   │  6051     Criptomoedas         ALTO     Irreversivel                    │|
|   │  6211     Corretoras           MEDIO    Grandes valores                 │|
|   │  5411     Supermercados        BAIXO    Uso diario normal               │|
|   │  5812     Restaurantes         BAIXO    Uso diario normal               │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

### Grupo 9: Features Especificas PIX (4 features)

```
+==============================================================================+
|                    GRUPO 9: FEATURES ESPECIFICAS PIX                         |
+==============================================================================+
|                                                                               |
|   Essas features sao usadas APENAS para transacoes PIX:                      |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  #  FEATURE             DESCRICAO                PESO    EXEMPLO        │|
|   │  ══════════════════════════════════════════════════════════════════════│|
|   │                                                                          │|
|   │  41 velocity_pix_1h     Qtd PIX na ultima        0.15    1              │|
|   │                         hora                            ⚠️ ALTO         │|
|   │                                                                          │|
|   │  42 pix_destination_new Destinatario nunca       0.18    0 (nao)        │|
|   │                         usado antes?                    ⚠️ CRITICO      │|
|   │                                                                          │|
|   │  43 pix_night_amount    Valor do PIX se          0.12    0              │|
|   │                         for noturno                     (nao e noturno) │|
|   │                                                                          │|
|   │  44 pix_recipient_risk  Score de risco do        0.14    0.20           │|
|   │                         recebedor                       ⚠️ ALTO         │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   💡 INSIGHT: PIX para destinatario NOVO + valor ALTO = maior alerta!       |
|                                                                               |
+==============================================================================+
```

### Grupo 10: Features Especificas Cartao (3 features)

```
+==============================================================================+
|                    GRUPO 10: FEATURES ESPECIFICAS CARTAO                     |
+==============================================================================+
|                                                                               |
|   Essas features sao usadas para transacoes de CARTAO:                       |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  #  FEATURE            DESCRICAO                 PESO    EXEMPLO        │|
|   │  ══════════════════════════════════════════════════════════════════════│|
|   │                                                                          │|
|   │  45 card_present       Cartao fisico presente?   0.12    1 (sim)        │|
|   │                        (chip/tarja)                     (mais seguro)    │|
|   │                                                                          │|
|   │  46 is_international   Compra internacional?     0.15    0 (nao)        │|
|   │                        (fora do Brasil)                 ⚠️ ALTO         │|
|   │                                                                          │|
|   │  47 card_velocity_1h   Qtd compras na            0.10    2              │|
|   │                        ultima hora                                       │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

## Resumo Visual: Importancia das Features

```
+==============================================================================+
|                    RANKING DE IMPORTANCIA DAS FEATURES                       |
+==============================================================================+
|                                                                               |
|   TOP 10 FEATURES MAIS IMPORTANTES PARA DETECTAR FRAUDE:                     |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  #1  amount (valor)                    ████████████████████  18%        │|
|   │  #2  impossible_travel                 █████████████████     25%*       │|
|   │  #3  pix_destination_new               ██████████████        18%        │|
|   │  #4  velocity_5min                     ██████████████        18%        │|
|   │  #5  is_early_morning                  ████████████          15%        │|
|   │  #6  high_amount_suspicious_hour       ████████████          15%        │|
|   │  #7  amount_deviation_zscore           ███████████           14%        │|
|   │  #8  device_change                     ███████████           14%        │|
|   │  #9  is_international                  ████████████          15%        │|
|   │  #10 pix_recipient_risk                ███████████           14%        │|
|   │                                                                          │|
|   │  * impossible_travel e calculado, nao e input direto                    │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   COMBINACOES PERIGOSAS (triggers automaticos):                              |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  🚨 BLOQUEIO IMEDIATO:                                                  │|
|   │     • impossible_travel = 1                                             │|
|   │     • previous_frauds >= 2                                              │|
|   │                                                                          │|
|   │  ⚠️ ALERTA ALTO (pede confirmacao):                                     │|
|   │     • is_early_morning + is_high_value + pix_destination_new            │|
|   │     • device_change + is_new_location + amount_deviation > 3            │|
|   │                                                                          │|
|   │  📊 MONITORAMENTO:                                                      │|
|   │     • velocity_1h > 5 transacoes                                        │|
|   │     • is_new_merchant + is_high_value                                   │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

## Exercicio: Calcule as Features!

```
+==============================================================================+
|                    🧠 EXERCICIO: CALCULE AS FEATURES                         |
+==============================================================================+
|                                                                               |
|   SITUACAO:                                                                  |
|   Maria (conta de 8 anos, media de R$300/transacao) faz:                    |
|   - PIX de R$ 12.000                                                        |
|   - As 3h da manha                                                          |
|   - Para conta que nunca usou antes                                         |
|   - De um celular novo                                                      |
|                                                                               |
|   PREENCHA AS FEATURES:                                                      |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  is_early_morning = ___  (0 ou 1?)                                      │|
|   │  is_high_value = ___     (0 ou 1?)                                      │|
|   │  is_very_high_value = ___ (0 ou 1?)                                     │|
|   │  pix_destination_new = ___ (0 ou 1?)                                    │|
|   │  device_change = ___     (0 ou 1?)                                      │|
|   │  amount_deviation_zscore = ___ (alto ou baixo?)                         │|
|   │                                                                          │|
|   │  SCORE ESPERADO: ___ (baixo, medio ou alto?)                            │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   .                                                                          |
|   .                                                                          |
|   (role para ver a resposta)                                                |
|   .                                                                          |
|   .                                                                          |
|                                                                               |
|   RESPOSTA:                                                                  |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  is_early_morning = 1        (3h da manha = madrugada)                  │|
|   │  is_high_value = 1           (R$12.000 > R$5.000)                       │|
|   │  is_very_high_value = 1      (R$12.000 > R$10.000)                      │|
|   │  pix_destination_new = 1     (nunca transferiu para essa conta)         │|
|   │  device_change = 1           (celular novo)                             │|
|   │  amount_deviation_zscore = 40x! (12000/300 = 40 desvios!)               │|
|   │                                                                          │|
|   │  SCORE: MUITO ALTO! (provavelmente > 0.90)                              │|
|   │                                                                          │|
|   │  ACAO: BLOQUEIO + SMS + Ligacao para Maria                              │|
|   │                                                                          │|
|   │  (Se for Maria mesmo, ela confirma e transacao e liberada)              │|
|   │  (Se for golpista, bloqueio salvou R$ 12.000!)                          │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

---

# ATO 0: ANTES DE COMECAR

## O Que Voce Vai Aprender

```
+==============================================================================+
|                         OBJETIVOS DESTA AULA                                 |
+==============================================================================+
|                                                                               |
|   Ao final desta aula, voce sera capaz de:                                   |
|                                                                               |
|   [✓] Entender como 10 modelos de IA trabalham JUNTOS                        |
|   [✓] Explicar a jornada de uma transacao em 300ms                           |
|   [✓] Interpretar metricas (Precision, Recall, F1)                           |
|   [✓] Identificar qual modelo detecta qual tipo de fraude                    |
|   [✓] Usar analogias do dia a dia para explicar ML                           |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  IMPORTANTE: Voce NAO precisa saber programar!                          │|
|   │  Esta aula usa analogias e exemplos visuais.                            │|
|   │  Se voce sabe usar um caixa eletronico, voce consegue entender.         │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

## A Grande Analogia: O Aeroporto

Antes de mergulharmos nos modelos, vamos usar uma analogia que vai te acompanhar por toda a aula:

```
+==============================================================================+
|                    DETECCAO DE FRAUDE = SEGURANCA DE AEROPORTO               |
+==============================================================================+
|                                                                               |
|   Imagine que cada TRANSACAO bancaria e como um PASSAGEIRO                   |
|   tentando embarcar em um aviao (seu dinheiro).                              |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │   PASSAGEIRO                              TRANSACAO                      │|
|   │   (voce no aeroporto)                     (PIX que voce faz)             │|
|   │                                                                          │|
|   │   Documento de identidade      =         CPF/Conta de origem            │|
|   │   Bagagem                      =         Valor da transacao             │|
|   │   Destino                      =         Conta de destino               │|
|   │   Horario do voo               =         Horario da transacao           │|
|   │   Historico de viagens         =         Historico de transacoes        │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   Assim como o aeroporto tem VARIOS checkpoints de seguranca,                |
|   o sistema de fraude tem VARIOS modelos de IA analisando.                   |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │   AEROPORTO                               SISTEMA DE FRAUDE              │|
|   │                                                                          │|
|   │   Check-in (documentos)        =         Validacao inicial               │|
|   │   Raio-X da bagagem            =         Random Forest                   │|
|   │   Detector de metais           =         XGBoost                         │|
|   │   Revista manual (se suspeito) =         LightGBM                        │|
|   │   Lista de procurados          =         Autoencoders                    │|
|   │   Camera de reconhecimento     =         LSTM (memoria)                  │|
|   │   Cooperacao internacional     =         Federated Learning              │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

## Os 10 Modelos que Voce Vai Conhecer

```
+==============================================================================+
|                         ELENCO PRINCIPAL DA AULA                             |
+==============================================================================+
|                                                                               |
|   ┌────────────────────────────────────────────────────────────────────────┐ |
|   │  GRUPO 1: OS GUARDAS DA PRIMEIRA LINHA (Ensemble Base)                 │ |
|   │  ══════════════════════════════════════════════════════                │ |
|   │                                                                         │ |
|   │  🌲 Random Forest     - O Comite de Arvores                            │ |
|   │  🚀 XGBoost           - O Aluno que Aprende com Erros                  │ |
|   │  ⚡ LightGBM          - O Velocista Eficiente                          │ |
|   │                                                                         │ |
|   │  Esses 3 trabalham JUNTOS em 95% das transacoes.                       │ |
|   │  Tempo de resposta: 28ms (mais rapido que um piscar de olhos!)         │ |
|   └────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|   ┌────────────────────────────────────────────────────────────────────────┐ |
|   │  GRUPO 2: OS ESPECIALISTAS (Chamados em Casos Complexos)               │ |
|   │  ═══════════════════════════════════════════════════════               │ |
|   │                                                                         │ |
|   │  🧠 LSTM/GRU          - O Detetive com Memoria de Elefante             │ |
|   │  🔄 Autoencoders      - O Cacador de Coisas Estranhas                  │ |
|   │  📊 TabTransformer    - O Leitor de Contexto (Caso Stripe)             │ |
|   │  🕸️ GNN               - O Mapeador de Redes Criminosas                 │ |
|   │                                                                         │ |
|   │  Esses sao acionados quando algo parece suspeito.                      │ |
|   └────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|   ┌────────────────────────────────────────────────────────────────────────┐ |
|   │  GRUPO 3: OS ESTRATEGISTAS (Visao Sistemica)                           │ |
|   │  ═══════════════════════════════════════════                           │ |
|   │                                                                         │ |
|   │  🤝 Federated Learning - A Alianca Secreta dos Bancos                  │ |
|   │  📈 VAE               - O Gerador de Cenarios                          │ |
|   │  🔗 Stacking          - O Juiz Final                                   │ |
|   │                                                                         │ |
|   │  Esses operam em nivel sistemico, combinando informacoes.              │ |
|   └────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
+==============================================================================+
```

---

# ATO 1: A JORNADA DE UMA TRANSACAO

## Cena 1: Maria Faz um PIX

Vamos acompanhar uma transacao REAL do inicio ao fim. Conheca Maria:

```
+==============================================================================+
|                         CONHECA A MARIA                                      |
+==============================================================================+
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │        ╭──────────────────────────────────────────────────────╮         │|
|   │        │                                                      │         │|
|   │        │   PERFIL: Maria Santos, 42 anos                      │         │|
|   │        │   ════════════════════════════                       │         │|
|   │        │                                                      │         │|
|   │        │   Profissao: Professora de matematica                │         │|
|   │        │   Cidade: Belo Horizonte, MG                         │         │|
|   │        │   Banco: Banco Sankofa (conta ha 8 anos)             │         │|
|   │        │                                                      │         │|
|   │        │   Comportamento tipico:                              │         │|
|   │        │   • Salario: R$ 6.500/mes (dia 5)                    │         │|
|   │        │   • Aluguel: R$ 1.800/mes (dia 10)                   │         │|
|   │        │   • Supermercado: R$ 800/mes (sabados)               │         │|
|   │        │   • Uber: R$ 200/mes (dias uteis)                    │         │|
|   │        │   • Poupanca: R$ 500/mes (dia 5)                     │         │|
|   │        │                                                      │         │|
|   │        │   Transacoes por mes: ~45                            │         │|
|   │        │   Valor medio: R$ 180                                │         │|
|   │        │   Horario tipico: 7h-21h                             │         │|
|   │        │   Localizacao: BH e regiao                           │         │|
|   │        │                                                      │         │|
|   │        ╰──────────────────────────────────────────────────────╯         │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

### A Transacao de Maria

```
+==============================================================================+
|                    TERÇA-FEIRA, 14:32:15                                     |
+==============================================================================+
|                                                                               |
|   Maria abre o app do banco no celular.                                      |
|   Ela quer fazer um PIX de R$ 350 para sua filha que esta na faculdade.      |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                          DADOS DA TRANSACAO                              │|
|   │  ═══════════════════════════════════════════════════════════════════    │|
|   │                                                                          │|
|   │  De:        Maria Santos (CPF: 123.456.789-00)                          │|
|   │  Para:      Julia Santos (CPF: 987.654.321-00)                          │|
|   │  Valor:     R$ 350,00                                                   │|
|   │  Tipo:      PIX                                                         │|
|   │  Horario:   14:32:15 (terca-feira)                                      │|
|   │  Device:    iPhone 13, iOS 17.2                                         │|
|   │  IP:        189.45.123.xxx (BH, MG)                                     │|
|   │  Biometria: Face ID confirmado                                          │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   Maria clica em "Confirmar PIX"...                                          |
|                                                                               |
|   O que acontece nos proximos 300 MILISSEGUNDOS?                             |
|                                                                               |
+==============================================================================+
```

## Cena 2: Os 300 Milissegundos Mais Importantes

```
+==============================================================================+
|                    LINHA DO TEMPO: 300 MILISSEGUNDOS                         |
+==============================================================================+
|                                                                               |
|   TEMPO      EVENTO                                    MODELO                 |
|   ═════════════════════════════════════════════════════════════════════════  |
|                                                                               |
|   0ms        Maria clica "Confirmar"                   -                      |
|   │                                                                           |
|   │   ┌─────────────────────────────────────────────────────────────────┐    |
|   ▼   │  ETAPA 1: COLETA DE FEATURES (0-15ms)                           │    |
|   15ms│  ═══════════════════════════════════════                        │    |
|   │   │  O sistema coleta 30 informacoes sobre a transacao:             │    |
|   │   │  • Valor normalizado (0.05 - baixo para Maria)                  │    |
|   │   │  • Hora do dia (14h - dentro do padrao)                         │    |
|   │   │  • Dia da semana (terca - normal)                               │    |
|   │   │  • Destinatario conhecido? (SIM - filha)                        │    |
|   │   │  • Device score (0.95 - celular habitual)                       │    |
|   │   │  • Geolocalizacao (BH - normal)                                 │    |
|   │   │  • Velocidade (0 - nenhuma tx nos ultimos 5min)                 │    |
|   │   │  • ... mais 23 features                                         │    |
|   │   └─────────────────────────────────────────────────────────────────┘    |
|   │                                                                           |
|   │   ┌─────────────────────────────────────────────────────────────────┐    |
|   ▼   │  ETAPA 2: RANDOM FOREST ANALISA (15-22ms)                       │    |
|   22ms│  ═══════════════════════════════════════════                    │    |
|   │   │                                                                  │    |
|   │   │  100 "arvores" votam se e fraude:                               │    |
|   │   │                                                                  │    |
|   │   │  Arvore 1:  "NAO e fraude" (confianca 92%)                      │    |
|   │   │  Arvore 2:  "NAO e fraude" (confianca 88%)                      │    |
|   │   │  Arvore 3:  "NAO e fraude" (confianca 95%)                      │    |
|   │   │  ...                                                             │    |
|   │   │  Arvore 100: "NAO e fraude" (confianca 91%)                     │    |
|   │   │                                                                  │    |
|   │   │  VOTACAO FINAL: 98/100 dizem NAO FRAUDE                         │    |
|   │   │  Score Random Forest: 0.02 (muito baixo = seguro)               │    |
|   │   │                                                                  │    |
|   │   └─────────────────────────────────────────────────────────────────┘    |
|   │                                                                           |
|   │   ┌─────────────────────────────────────────────────────────────────┐    |
|   ▼   │  ETAPA 3: XGBOOST ANALISA (22-26ms)                             │    |
|   26ms│  ═══════════════════════════════════════                        │    |
|   │   │                                                                  │    |
|   │   │  XGBoost "aprende com erros" de modelos anteriores:             │    |
|   │   │                                                                  │    |
|   │   │  Rodada 1: Score 0.15 (incerto)                                 │    |
|   │   │  Rodada 2: Corrige erro → Score 0.08                            │    |
|   │   │  Rodada 3: Corrige erro → Score 0.04                            │    |
|   │   │  ...                                                             │    |
|   │   │  Rodada 100: Score final = 0.03                                 │    |
|   │   │                                                                  │    |
|   │   │  Score XGBoost: 0.03 (muito baixo = seguro)                     │    |
|   │   │                                                                  │    |
|   │   └─────────────────────────────────────────────────────────────────┘    |
|   │                                                                           |
|   │   ┌─────────────────────────────────────────────────────────────────┐    |
|   ▼   │  ETAPA 4: LIGHTGBM ANALISA (26-28ms)                            │    |
|   28ms│  ═════════════════════════════════════                          │    |
|   │   │                                                                  │    |
|   │   │  LightGBM e o mais RAPIDO, otimizado para velocidade:           │    |
|   │   │                                                                  │    |
|   │   │  Analise em 2ms (vs 7ms do XGBoost)                             │    |
|   │   │  Mesma precisao, menor consumo de recursos                      │    |
|   │   │                                                                  │    |
|   │   │  Score LightGBM: 0.02 (muito baixo = seguro)                    │    |
|   │   │                                                                  │    |
|   │   └─────────────────────────────────────────────────────────────────┘    |
|   │                                                                           |
|   │   ┌─────────────────────────────────────────────────────────────────┐    |
|   ▼   │  ETAPA 5: STACKING COMBINA TUDO (28-30ms)                       │    |
|   30ms│  ════════════════════════════════════════                       │    |
|   │   │                                                                  │    |
|   │   │  O "Juiz Final" (Regressao Logistica) combina:                  │    |
|   │   │                                                                  │    |
|   │   │  Random Forest: 0.02 × peso 0.35 = 0.007                        │    |
|   │   │  XGBoost:       0.03 × peso 0.40 = 0.012                        │    |
|   │   │  LightGBM:      0.02 × peso 0.25 = 0.005                        │    |
|   │   │  ─────────────────────────────────────────                      │    |
|   │   │  SCORE FINAL:   0.024 (2.4% de chance de fraude)                │    |
|   │   │                                                                  │    |
|   │   │  THRESHOLD: 0.35 (35%)                                          │    |
|   │   │  DECISAO: 0.024 < 0.35 → APROVAR!                               │    |
|   │   │                                                                  │    |
|   │   └─────────────────────────────────────────────────────────────────┘    |
|   │                                                                           |
|   ▼                                                                           |
|   30ms      PIX APROVADO! ✓                                                   |
|                                                                               |
|   300ms     Dinheiro na conta da Julia                                        |
|                                                                               |
+==============================================================================+
```

## Cena 3: O Diagrama Visual da Jornada

```
+==============================================================================+
|                    FLUXOGRAMA: JORNADA DA TRANSACAO                          |
+==============================================================================+
|                                                                               |
|   ┌──────────────┐                                                           |
|   │   TRANSACAO  │                                                           |
|   │   (PIX R$350)│                                                           |
|   └──────┬───────┘                                                           |
|          │                                                                    |
|          ▼                                                                    |
|   ┌──────────────────────────────────────────────────────────────────────┐   |
|   │                     EXTRACAO DE FEATURES                              │   |
|   │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   |
|   │   │  Valor  │ │  Hora   │ │  Local  │ │ Device  │ │Historico│ ...    │   |
|   │   │  0.05   │ │  0.60   │ │  0.95   │ │  0.98   │ │  0.12   │        │   |
|   │   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   |
|   │                        30 FEATURES NO TOTAL                           │   |
|   └──────────────────────────────┬───────────────────────────────────────┘   |
|                                  │                                            |
|          ┌───────────────────────┼───────────────────────┐                   |
|          │                       │                       │                   |
|          ▼                       ▼                       ▼                   |
|   ┌──────────────┐       ┌──────────────┐       ┌──────────────┐            |
|   │    RANDOM    │       │   XGBOOST    │       │   LIGHTGBM   │            |
|   │    FOREST    │       │              │       │              │            |
|   │   ┌──────┐   │       │  ┌──────┐    │       │  ┌──────┐    │            |
|   │   │ 🌲🌲 │   │       │  │ 🚀   │    │       │  │ ⚡   │    │            |
|   │   │ 🌲🌲 │   │       │  │      │    │       │  │      │    │            |
|   │   └──────┘   │       │  └──────┘    │       │  └──────┘    │            |
|   │  100 arvores │       │ 100 rodadas  │       │ Otimizado    │            |
|   │              │       │ de melhoria  │       │ para GPU     │            |
|   │  Score: 0.02 │       │  Score: 0.03 │       │  Score: 0.02 │            |
|   └──────┬───────┘       └──────┬───────┘       └──────┬───────┘            |
|          │                      │                      │                     |
|          └──────────────────────┼──────────────────────┘                     |
|                                 │                                            |
|                                 ▼                                            |
|                    ┌────────────────────────┐                                |
|                    │   STACKING ENSEMBLE    │                                |
|                    │   (Meta-Modelo)        │                                |
|                    │                        │                                |
|                    │   RF × 0.35  = 0.007   │                                |
|                    │   XGB × 0.40 = 0.012   │                                |
|                    │   LGB × 0.25 = 0.005   │                                |
|                    │   ──────────────────   │                                |
|                    │   TOTAL:     0.024     │                                |
|                    │                        │                                |
|                    └───────────┬────────────┘                                |
|                                │                                             |
|                                ▼                                             |
|                    ┌────────────────────────┐                                |
|                    │      DECISAO           │                                |
|                    │                        │                                |
|                    │   0.024 < 0.35 ?       │                                |
|                    │                        │                                |
|                    │      ✓ SIM             │                                |
|                    │                        │                                |
|                    └───────────┬────────────┘                                |
|                                │                                             |
|                                ▼                                             |
|                    ┌────────────────────────┐                                |
|                    │                        │                                |
|                    │   ✅ PIX APROVADO!     │                                |
|                    │                        │                                |
|                    │   Tempo total: 30ms    │                                |
|                    │                        │                                |
|                    └────────────────────────┘                                |
|                                                                               |
+==============================================================================+
```

## Exercicio 1: Use a Cabeca!

```
+==============================================================================+
|                    🧠 EXERCICIO 1: PENSE COMO O MODELO                       |
+==============================================================================+
|                                                                               |
|   SITUACAO:                                                                  |
|   Maria faz outro PIX, mas dessa vez as 3h da manha, de R$ 8.000,           |
|   para uma conta que ela NUNCA transferiu antes.                            |
|                                                                               |
|   PERGUNTA: O que voce acha que vai acontecer?                              |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  ANTES DE VER A RESPOSTA, TENTE PENSAR:                                 │|
|   │                                                                          │|
|   │  [ ] O valor e alto para Maria?                                         │|
|   │  [ ] O horario e normal para ela?                                       │|
|   │  [ ] O destinatario e conhecido?                                        │|
|   │  [ ] Quantas "flags vermelhas" voce contou?                             │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   .                                                                          |
|   .                                                                          |
|   .                                                                          |
|   (role para ver a resposta)                                                |
|   .                                                                          |
|   .                                                                          |
|   .                                                                          |
|                                                                               |
|   RESPOSTA:                                                                  |
|   ════════                                                                   |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  Random Forest:  Score 0.72 (ALTO - 72 arvores votaram "fraude")        │|
|   │  XGBoost:        Score 0.68 (ALTO)                                      │|
|   │  LightGBM:       Score 0.75 (ALTO)                                      │|
|   │                                                                          │|
|   │  Stacking Final: 0.72 (72% de chance de fraude)                         │|
|   │                                                                          │|
|   │  0.72 > 0.35 (threshold) → BLOQUEADO! 🚫                                │|
|   │                                                                          │|
|   │  MOTIVOS:                                                                │|
|   │  • Horario atipico (3h vs padrao 7h-21h)                                │|
|   │  • Valor 22x maior que a media de Maria (R$350)                         │|
|   │  • Destinatario desconhecido                                            │|
|   │  • Combinacao de 3 fatores de risco                                     │|
|   │                                                                          │|
|   │  ACAO: SMS + Ligacao para Maria confirmar                               │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

---

# ATO 2: OS ESPECIALISTAS ENTRAM EM CENA

## Quando os 3 Guardas Nao Sao Suficientes

```
+==============================================================================+
|                    QUANDO CHAMAR OS ESPECIALISTAS?                           |
+==============================================================================+
|                                                                               |
|   Os 3 guardas da primeira linha (RF, XGBoost, LightGBM) resolvem            |
|   95% dos casos. Mas existem fraudes SOFISTICADAS que exigem mais:           |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  TIPO DE FRAUDE                    ESPECIALISTA CHAMADO                  │|
|   │  ═════════════════════════════════════════════════════════              │|
|   │                                                                          │|
|   │  Sequencia de transacoes           LSTM/GRU                              │|
|   │  suspeitas ao longo de dias        (O Detetive com Memoria)             │|
|   │                                                                          │|
|   │  Transacao "estranha" que          Autoencoders/VAE                      │|
|   │  nao se parece com nada            (Cacadores de Anomalias)             │|
|   │                                                                          │|
|   │  Card testing em massa             TabTransformer                        │|
|   │  (muitas transacoes pequenas)      (Leitor de Contexto)                 │|
|   │                                                                          │|
|   │  Rede de contas laranja            GNN                                   │|
|   │  (lavagem de dinheiro)             (Mapeador de Redes)                  │|
|   │                                                                          │|
|   │  Fraude que um banco nao ve        Federated Learning                    │|
|   │  (crime internacional)             (Alianca dos Bancos)                 │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

## Especialista 1: LSTM/GRU - O Detetive com Memoria

```
+==============================================================================+
|                    LSTM/GRU: O DETETIVE COM MEMORIA                          |
+==============================================================================+
|                                                                               |
|   ANALOGIA DO DIA A DIA:                                                     |
|   ════════════════════════                                                   |
|                                                                               |
|   Imagine um detetive que LEMBRA de tudo que voce fez nos ultimos 90 dias.   |
|   Ele sabe:                                                                  |
|   • Onde voce compra cafe toda manha                                         |
|   • Quanto voce gasta em media                                               |
|   • Em que horarios voce faz transacoes                                      |
|   • Quais lojas voce frequenta                                               |
|                                                                               |
|   Quando algo foge do padrao, ele percebe IMEDIATAMENTE.                     |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │              COMO O LSTM "LEMBRA" DAS TRANSACOES                        │|
|   │                                                                          │|
|   │   Transacao 1     Transacao 2     Transacao 3     Transacao 4           │|
|   │   (ha 3 dias)     (ha 2 dias)     (ontem)         (AGORA)               │|
|   │                                                                          │|
|   │   [cafe R$15] ──→ [almoco R$45] ──→ [uber R$30] ──→ [PIX R$8000 ?]      │|
|   │        │               │               │               │                │|
|   │        ▼               ▼               ▼               ▼                │|
|   │   ┌────────┐     ┌────────┐     ┌────────┐     ┌────────┐              │|
|   │   │ Celula │ ──→ │ Celula │ ──→ │ Celula │ ──→ │ Celula │              │|
|   │   │  LSTM  │     │  LSTM  │     │  LSTM  │     │  LSTM  │              │|
|   │   └────────┘     └────────┘     └────────┘     └────────┘              │|
|   │        │               │               │               │                │|
|   │        └───────────────┴───────────────┴───────────────┘                │|
|   │                           │                                              │|
|   │                           ▼                                              │|
|   │                    MEMORIA DE LONGO PRAZO                                │|
|   │                    "Esse PIX NAO combina com o historico!"              │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

### Historia Real: O Cartao Clonado Detectado pelo LSTM

```
+==============================================================================+
|                    CASO REAL: FERNANDA E O CARTAO CLONADO                    |
+==============================================================================+
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │  PERFIL: Fernanda, 34 anos, advogada em Brasilia                        │|
|   │  PADRAO NORMAL: cafe, almoco, uber, academia (2-3 tx/dia)               │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   SEGUNDA-FEIRA, 14:32                                                       |
|   ══════════════════════                                                     |
|                                                                               |
|   O cartao de Fernanda foi clonado em um restaurante (maquininha adulterada)|
|   Criminosos em RECIFE tentam usar o cartao:                                 |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  14:32  cafe R$15         [OK - padrao normal]                          │|
|   │         │                                                                │|
|   │         ▼                                                                │|
|   │  14:35  ATM saque R$1000  [ALERTA - Fernanda NUNCA usa ATM!]            │|
|   │         │                 [ALERTA - Localizacao: RECIFE, nao Brasilia!] │|
|   │         ▼                                                                │|
|   │  14:38  eletronicos R$4500 [ALERTA - categoria NOVA]                    │|
|   │         │                                                                │|
|   │         ▼                                                                │|
|   │  14:41  eletronicos R$3800 [CRITICO - 2 compras em 3 minutos!]          │|
|   │         │                                                                │|
|   │         ▼                                                                │|
|   │  14:43  gift card R$2000  [BLOQUEADO!]                                  │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   O QUE O LSTM DETECTOU:                                                     |
|   ═══════════════════════                                                    |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  1. INTER-EVENT TIME (tempo entre transacoes)                           │|
|   │     Normal de Fernanda: 3+ horas                                        │|
|   │     Observado: 3 minutos                                                │|
|   │     → ANOMALIA SEVERA                                                   │|
|   │                                                                          │|
|   │  2. CATEGORIA NOVA                                                       │|
|   │     Historico 90 dias: cafe, restaurante, uber, academia                │|
|   │     Observado: ATM, eletronicos, gift card                              │|
|   │     → ANOMALIA SEVERA                                                   │|
|   │                                                                          │|
|   │  3. LOCALIZACAO IMPOSSIVEL                                              │|
|   │     Ultima tx real: Brasilia, 14:30                                     │|
|   │     TX suspeitas: Recife, 14:32+                                        │|
|   │     Distancia: 1.600km em 2 minutos? IMPOSSIVEL!                        │|
|   │     → ANOMALIA CRITICA                                                  │|
|   │                                                                          │|
|   │  HIDDEN STATE ACUMULADO:                                                │|
|   │  [0.12] → [0.45] → [0.78] → [0.91] → [0.97] BLOQUEIO!                   │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   RESULTADO: R$ 5.800 em fraudes EVITADAS                                    |
|              Cartao bloqueado, novo emitido                                 |
|              Fernanda recebeu SMS em 30 segundos                            |
|                                                                               |
+==============================================================================+
```

### Metricas do LSTM/GRU

```
+==============================================================================+
|                    METRICAS: LSTM/GRU                                        |
+==============================================================================+
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │   PRECISION:  98.2%   ← De cada 100 alertas, 98 sao fraudes reais      │|
|   │   RECALL:     94.7%   ← Detecta 95 de cada 100 fraudes                 │|
|   │   F1-SCORE:   96.4%   ← Media harmonica (equilibrio)                   │|
|   │   LATENCIA:   15ms    ← Tempo para analisar sequencia de 7 tx          │|
|   │                                                                          │|
|   │   MELHOR PARA:                                                          │|
|   │   ✓ Cartoes clonados                                                    │|
|   │   ✓ Fraudes de "aquecimento" (testam antes de golpe grande)            │|
|   │   ✓ Interceptacao de boletos                                            │|
|   │   ✓ Padroes temporais (velocidade, horario)                             │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

---

## Especialista 2: TabTransformer - O Caso Stripe de $6 Bilhoes

```
+==============================================================================+
|                    TABTRANSFORMER: O LEITOR DE CONTEXTO                      |
+==============================================================================+
|                                                                               |
|   ANALOGIA DO DIA A DIA:                                                     |
|   ════════════════════════                                                   |
|                                                                               |
|   Imagine que voce le a frase: "Banco de praca"                             |
|   O que significa? Depende do CONTEXTO!                                      |
|                                                                               |
|   • "Vou sentar no banco de praca" → movel para sentar                      |
|   • "Vou ao banco de praca principal" → instituicao financeira              |
|                                                                               |
|   O TabTransformer faz a mesma coisa com transacoes:                         |
|   Ele ENTENDE O CONTEXTO de cada feature, nao apenas o valor.               |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │   MODELO TRADICIONAL (XGBoost):                                         │|
|   │   ══════════════════════════════                                        │|
|   │   BIN = 411111  →  "E um cartao Visa" (so isso!)                        │|
|   │   CEP = 01310   →  "E de Sao Paulo" (so isso!)                          │|
|   │                                                                          │|
|   │   TABTRANSFORMER:                                                       │|
|   │   ═══════════════                                                       │|
|   │   BIN = 411111 + CEP = 01310 + Merchant = "Loja Premium"                │|
|   │   →  "Cliente Visa de SP comprando em loja de luxo = NORMAL"            │|
|   │                                                                          │|
|   │   BIN = 411111 + CEP = 99999 + Merchant = "Gift Card Online"            │|
|   │   →  "Cartao premium + CEP falso + gift card = SUSPEITO!"               │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

### O Caso Stripe: De 59% para 97% em Uma Noite

```
+==============================================================================+
|                    CASO REAL: STRIPE - $6 BILHOES RECUPERADOS                |
+==============================================================================+
|                                                                               |
|   CONTEXTO:                                                                  |
|   ═══════════                                                                |
|   A Stripe processa $1.4 TRILHAO em pagamentos por ano.                     |
|   Um problema serio: CARD TESTING.                                          |
|                                                                               |
|   O QUE E CARD TESTING?                                                      |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │   Criminosos compram listas de cartoes roubados na dark web.            │|
|   │   Precisam descobrir quais AINDA FUNCIONAM.                             │|
|   │                                                                          │|
|   │   METODO:                                                                │|
|   │   1. Fazer milhares de compras de $1                                    │|
|   │   2. Se aprovar → cartao valido → vender por mais                       │|
|   │   3. Se negar → cartao cancelado → descartar                            │|
|   │                                                                          │|
|   │   ATAQUE TIPICO:                                                        │|
|   │   23:45:00 - tx $1.00 cartao XXX4532                                    │|
|   │   23:45:01 - tx $1.00 cartao XXX7821                                    │|
|   │   23:45:02 - tx $1.00 cartao XXX9103                                    │|
|   │   ...                                                                    │|
|   │   23:55:00 - tx $1.00 cartao XXX2847 (50.000 transacoes em 10 min!)     │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   ANTES DO TABTRANSFORMER (XGBoost):                                         |
|   ════════════════════════════════════                                       |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │   XGBoost olhava CADA transacao isoladamente:                           │|
|   │                                                                          │|
|   │   tx $1.00 → Valor baixo? SIM → Normal                                  │|
|   │   Cartao valido? SIM → Normal                                           │|
|   │   Merchant legitimo? SIM → Normal                                       │|
|   │                                                                          │|
|   │   RESULTADO: 59% de deteccao (41% passavam!)                            │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   DEPOIS DO TABTRANSFORMER:                                                  |
|   ═══════════════════════════                                                |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │   TabTransformer analisou o CONTEXTO:                                   │|
|   │                                                                          │|
|   │   SELF-ATTENTION descobriu:                                             │|
|   │   • Mesmo IP para todas as transacoes (BOT!)                            │|
|   │   • User-Agent identico (script automatizado)                           │|
|   │   • Intervalo de 1 segundo (impossivel humano)                          │|
|   │   • BINs sequenciais (lista ordenada de cartoes)                        │|
|   │   • Valor sempre $1 (padrao de teste)                                   │|
|   │   • Horario 23:45 sexta (baixa vigilancia)                              │|
|   │                                                                          │|
|   │   RESULTADO: 97% de deteccao!                                           │|
|   │                                                                          │|
|   │   MELHORIA: De 59% para 97% EM UMA NOITE!                               │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   IMPACTO FINANCEIRO (2024):                                                 |
|   ═══════════════════════════                                                |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │   💰 $6 BILHOES em transacoes recuperadas                               │|
|   │      (transacoes que eram falsamente recusadas)                         │|
|   │                                                                          │|
|   │   📉 80% de reducao em ataques de card testing                          │|
|   │                                                                          │|
|   │   📈 35% menos tentativas de re-submit                                  │|
|   │      (cliente nao precisa tentar de novo)                               │|
|   │                                                                          │|
|   │   ⚡ 70% de melhoria em precisao                                        │|
|   │      (menos falsos positivos)                                           │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

### Diagrama: Como o TabTransformer "Pensa"

```
+==============================================================================+
|                    DIAGRAMA: SELF-ATTENTION DO TABTRANSFORMER                |
+==============================================================================+
|                                                                               |
|   Transacao: Maria compra bolsa na Macy's (NY) com cartao brasileiro        |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  FEATURES ISOLADAS (como XGBoost veria):                                │|
|   │                                                                          │|
|   │  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐                 │|
|   │  │ BIN   │  │ Pais  │  │ Valor │  │ Hora  │  │Merchant│                 │|
|   │  │Itau BR│  │  USA  │  │ $500  │  │ 14h   │  │ Macy's │                 │|
|   │  └───────┘  └───────┘  └───────┘  └───────┘  └───────┘                 │|
|   │      │          │          │          │          │                      │|
|   │      ▼          ▼          ▼          ▼          ▼                      │|
|   │    RISCO      RISCO      MEDIO     NORMAL    NORMAL                     │|
|   │  (estrangeiro)(diferente)                                               │|
|   │                                                                          │|
|   │  RESULTADO XGBOOST: "Talvez fraude?" → RECUSAR                          │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  SELF-ATTENTION (como TabTransformer ve):                               │|
|   │                                                                          │|
|   │  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐                 │|
|   │  │ BIN   │  │ Pais  │  │ Valor │  │ Hora  │  │Merchant│                 │|
|   │  │Itau BR│  │  USA  │  │ $500  │  │ 14h   │  │ Macy's │                 │|
|   │  └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘                 │|
|   │      │          │          │          │          │                      │|
|   │      └──────────┴──────────┴──────────┴──────────┘                      │|
|   │                          │                                               │|
|   │                          ▼                                               │|
|   │               ┌────────────────────┐                                    │|
|   │               │   TRANSFORMER      │                                    │|
|   │               │   SELF-ATTENTION   │                                    │|
|   │               │                    │                                    │|
|   │               │ "Como essas        │                                    │|
|   │               │  features se       │                                    │|
|   │               │  RELACIONAM?"      │                                    │|
|   │               └─────────┬──────────┘                                    │|
|   │                         │                                                │|
|   │                         ▼                                                │|
|   │   ┌─────────────────────────────────────────────────────────────────┐   │|
|   │   │  CONTEXTO COMBINADO:                                            │   │|
|   │   │                                                                  │   │|
|   │   │  • BIN Itau Personnalite = cliente PREMIUM                      │   │|
|   │   │  • Pais USA + IP de hotel em NY = TURISTA                       │   │|
|   │   │  • Historico Stripe: cartao usado em 47 paises = viajante       │   │|
|   │   │  • Macy's = loja de departamento CONFIAVEL                      │   │|
|   │   │  • 14h sabado = horario TIPICO de compras                       │   │|
|   │   │                                                                  │   │|
|   │   │  CONCLUSAO: "Turista brasileiro premium fazendo compras"        │   │|
|   │   │                                                                  │   │|
|   │   └─────────────────────────────────────────────────────────────────┘   │|
|   │                                                                          │|
|   │  RESULTADO TABTRANSFORMER: "Transacao legitima" → APROVAR ✓             │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

---

## Especialista 3: Autoencoders/VAE - Os Cacadores de Anomalias

```
+==============================================================================+
|                    AUTOENCODERS: OS CACADORES DE ANOMALIAS                   |
+==============================================================================+
|                                                                               |
|   ANALOGIA DO DIA A DIA:                                                     |
|   ════════════════════════                                                   |
|                                                                               |
|   Imagine uma maquina de fotocopias IMPERFEITA.                             |
|                                                                               |
|   Voce coloca um documento NORMAL e ela faz uma copia QUASE perfeita.       |
|   Voce coloca um documento ESTRANHO e ela faz uma copia MUITO RUIM.         |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │   DOCUMENTO NORMAL (transacao de rotina):                               │|
|   │   ┌─────────────┐     COPIADORA      ┌─────────────┐                    │|
|   │   │ cafe R$15   │  ─────────────→   │ cafe R$14.8 │                    │|
|   │   │ 8h manha    │     (encoder +    │ 8h manha    │                    │|
|   │   │ padaria SP  │      decoder)     │ padaria SP  │                    │|
|   │   └─────────────┘                    └─────────────┘                    │|
|   │                                                                          │|
|   │   ERRO = |15-14.8| = 0.2 (BAIXO = NORMAL)                               │|
|   │                                                                          │|
|   │   ─────────────────────────────────────────────────────────────────     │|
|   │                                                                          │|
|   │   DOCUMENTO ESTRANHO (fraude):                                          │|
|   │   ┌─────────────┐     COPIADORA      ┌─────────────┐                    │|
|   │   │crypto R$50k │  ─────────────→   │ ??? R$8k   │                    │|
|   │   │ 3h manha    │     (encoder +    │ ???        │                    │|
|   │   │ Nigeria     │      decoder)     │ ???        │                    │|
|   │   └─────────────┘                    └─────────────┘                    │|
|   │                                                                          │|
|   │   ERRO = MUITO ALTO! (NAO SABE RECONSTRUIR = ANOMALIA!)                 │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

### Diagrama: Arquitetura do Autoencoder

```
+==============================================================================+
|                    ARQUITETURA VISUAL DO AUTOENCODER                         |
+==============================================================================+
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  TRANSACAO                                             RECONSTRUCAO      │|
|   │  ORIGINAL                                              (copia)           │|
|   │                                                                          │|
|   │  ┌─────────┐                                          ┌─────────┐       │|
|   │  │ valor   │─┐                                    ┌──│ valor'  │       │|
|   │  │ hora    │─┤                                    ├──│ hora'   │       │|
|   │  │ local   │─┤     ENCODER        DECODER         ├──│ local'  │       │|
|   │  │ device  │─┤    (comprime)    (reconstroi)      ├──│ device' │       │|
|   │  │ destino │─┤         │              │           ├──│ destino'│       │|
|   │  │  ...    │─┤         │              │           ├──│  ...    │       │|
|   │  │(30 feat)│─┘         │              │           └──│(30 feat)│       │|
|   │  └─────────┘           │              │              └─────────┘       │|
|   │       │                │              │                   │             │|
|   │       │                ▼              ▼                   │             │|
|   │       │           ┌────────┐    ┌────────┐               │             │|
|   │       │           │  10    │    │  10    │               │             │|
|   │       │           │ neuron │    │ neuron │               │             │|
|   │       │           └────┬───┘    └───┬────┘               │             │|
|   │       │                │            │                     │             │|
|   │       │                ▼            ▼                     │             │|
|   │       │           ┌────────────────────┐                 │             │|
|   │       │           │                    │                 │             │|
|   │       │           │   ESPACO LATENTE   │                 │             │|
|   │       │           │   (2 dimensoes)    │                 │             │|
|   │       │           │                    │                 │             │|
|   │       │           │   "Essencia" da    │                 │             │|
|   │       │           │    transacao       │                 │             │|
|   │       │           │                    │                 │             │|
|   │       │           └────────────────────┘                 │             │|
|   │       │                                                   │             │|
|   │       └───────────────────────────────────────────────────┘             │|
|   │                               │                                          │|
|   │                               ▼                                          │|
|   │                    ┌────────────────────┐                               │|
|   │                    │   ERRO DE RECON.   │                               │|
|   │                    │                    │                               │|
|   │                    │   erro < 15 → OK   │                               │|
|   │                    │   erro > 25 → ALERTA│                              │|
|   │                    │   erro > 50 → BLOQ │                               │|
|   │                    │                    │                               │|
|   │                    └────────────────────┘                               │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

### Historia Real: O Golpe da Identidade Sintetica

```
+==============================================================================+
|                    CASO REAL: IDENTIDADE SINTETICA                           |
+==============================================================================+
|                                                                               |
|   O que e Identidade Sintetica?                                              |
|   ═══════════════════════════════                                            |
|                                                                               |
|   Criminosos criam "pessoas" FALSAS usando dados REAIS misturados:          |
|   • CPF de um idoso que nunca usou credito                                  |
|   • Endereco de um apartamento alugado                                       |
|   • Email e telefone novos                                                   |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  O GOLPE EM 6 MESES:                                                    │|
|   │  ══════════════════                                                     │|
|   │                                                                          │|
|   │  Mes 1: farmacia R$45, supermercado R$120, uber R$35                    │|
|   │  Mes 2: farmacia R$48, supermercado R$115, uber R$40                    │|
|   │  Mes 3: farmacia R$42, supermercado R$125, uber R$32                    │|
|   │  Mes 4: farmacia R$50, supermercado R$118, uber R$38                    │|
|   │  Mes 5: farmacia R$44, supermercado R$122, uber R$36                    │|
|   │  Mes 6: farmacia R$47, supermercado R$119, uber R$34                    │|
|   │                                                                          │|
|   │  → Pede cartao de credito com limite alto                              │|
|   │  → Estoura limite e desaparece                                         │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   Por que modelos tradicionais NAO pegaram?                                  |
|   ─────────────────────────────────────────                                  |
|   • CPF valido na Receita Federal? SIM                                      |
|   • Endereco valido? SIM                                                    |
|   • Historico de pagamento? PERFEITO                                        |
|   • Score de credito? 720 (excelente!)                                      |
|                                                                               |
|   Como o AUTOENCODER detectou?                                               |
|   ─────────────────────────────                                              |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  O Autoencoder aprendeu como clientes REAIS se comportam:               │|
|   │                                                                          │|
|   │  CLIENTES REAIS:                                                        │|
|   │  • Variancia NATURAL nos gastos (as vezes mais, as vezes menos)         │|
|   │  • Compram em lugares DIFERENTES                                        │|
|   │  • Horarios VARIAM                                                       │|
|   │  • Categorias DIVERSAS (roupa, lazer, saude...)                         │|
|   │                                                                          │|
|   │  IDENTIDADE SINTETICA:                                                  │|
|   │  • Variancia MUITO BAIXA (valores quase identicos!)                     │|
|   │  • Apenas 3 categorias de gastos                                        │|
|   │  • Horarios MUITO regulares                                             │|
|   │  • Locais MUITO repetidos                                               │|
|   │                                                                          │|
|   │  ERRO DE RECONSTRUCAO: 87.3 (normal: <15)                               │|
|   │                                                                          │|
|   │  CONCLUSAO: "Esse padrao e PERFEITO demais para ser humano!"            │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   RESULTADO: Cartao NEGADO antes de ser emitido                              |
|              R$ 2.3 milhoes em fraudes evitadas                              |
|              23 outras identidades sinteticas do mesmo grupo descobertas     |
|                                                                               |
+==============================================================================+
```

---

## Especialista 4: GNN - O Mapeador de Redes Criminosas

```
+==============================================================================+
|                    GNN: O MAPEADOR DE REDES CRIMINOSAS                       |
+==============================================================================+
|                                                                               |
|   ANALOGIA DO DIA A DIA:                                                     |
|   ════════════════════════                                                   |
|                                                                               |
|   Imagine um detetive que cria um MAPA DE CONEXOES entre criminosos.        |
|                                                                               |
|   Ele nao olha cada pessoa isoladamente.                                     |
|   Ele olha QUEM CONHECE QUEM, QUEM ENVIA DINHEIRO PARA QUEM.                |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │   VISAO TRADICIONAL (por conta):                                        │|
|   │   Conta A → OK (baixo volume)                                           │|
|   │   Conta B → OK (baixo volume)                                           │|
|   │   Conta C → OK (baixo volume)                                           │|
|   │   Cada conta parece normal isoladamente!                                │|
|   │                                                                          │|
|   │   ─────────────────────────────────────────────────────────────────     │|
|   │                                                                          │|
|   │   VISAO GNN (grafo de conexoes):                                        │|
|   │                                                                          │|
|   │            ┌───────┐     R$5000      ┌───────┐                          │|
|   │            │       │ ─────────────→  │       │                          │|
|   │            │ CONTA │                 │ CONTA │                          │|
|   │            │   A   │                 │   B   │                          │|
|   │            └───────┘                 └───┬───┘                          │|
|   │                                          │ R$4800                       │|
|   │                                          ▼                              │|
|   │                                     ┌───────┐                           │|
|   │                                     │ CONTA │                           │|
|   │                                     │   C   │                           │|
|   │                                     └───┬───┘                           │|
|   │                                         │ R$4600                        │|
|   │                     ┌───────────────────┼───────────────────┐           │|
|   │                     ▼                   ▼                   ▼           │|
|   │               ┌───────┐           ┌───────┐           ┌───────┐        │|
|   │               │ ATM   │           │CRYPTO │           │GIFTCARD│        │|
|   │               └───────┘           └───────┘           └───────┘        │|
|   │                                                                          │|
|   │   GNN VE:                                                               │|
|   │   • Estrutura de "cascata" (dinheiro fluindo em cadeia)                 │|
|   │   • Valores diminuindo (R$200 de "comissao" em cada passo)              │|
|   │   • Destino final: ATM/Crypto/Gift Card (dificil rastrear)              │|
|   │   → PADRAO CLASSICO DE LAVAGEM DE DINHEIRO!                             │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

### Historia Real: A Rede de 200 Contas Laranja

```
+==============================================================================+
|                    CASO REAL: OPERACAO "LARANJAL"                            |
+==============================================================================+
|                                                                               |
|   CENARIO: Banco digital brasileiro, Janeiro-Marco 2025                      |
|   VOLUME SUSPEITO: R$ 45 milhoes movimentados                                |
|                                                                               |
|   O ESQUEMA:                                                                 |
|   ═══════════                                                                |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  ETAPA 1: RECRUTAMENTO                                                  │|
|   │  ─────────────────────                                                  │|
|   │  • Criminosos recrutam pessoas em vulnerabilidade                       │|
|   │  • Oferecem R$500 para "emprestar" a conta por 1 mes                    │|
|   │  • Coletam selfies, documentos, senhas                                  │|
|   │  • 200+ contas coletadas em 3 meses                                     │|
|   │                                                                          │|
|   │  ETAPA 2: ESTRUTURACAO EM CAMADAS                                       │|
|   │  ─────────────────────────────────                                      │|
|   │  • Camada 0: 20 contas recebem dinheiro sujo (golpes PIX)               │|
|   │  • Camada 1: 40 contas (primeira dispersao)                             │|
|   │  • Camada 2: 60 contas (segunda dispersao)                              │|
|   │  • Camada 3: 50 contas (terceira dispersao)                             │|
|   │  • Camada 4: 30 contas (saida: ATM/crypto)                              │|
|   │                                                                          │|
|   │  ETAPA 3: MOVIMENTACAO (4-6 horas por ciclo)                            │|
|   │  ─────────────────────────────────────────────                          │|
|   │  • Vitima de golpe envia PIX para Camada 0                              │|
|   │  • Dinheiro passa por todas as camadas                                  │|
|   │  • Sai como saque ATM ou compra de crypto                               │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   POR QUE NAO FOI DETECTADO ANTES?                                           |
|   ═══════════════════════════════════                                        |
|                                                                               |
|   • Cada transacao individual era < R$5.000 (abaixo do threshold)           |
|   • Cada conta tinha poucos movimentos                                      |
|   • Nenhum padrao obvio em cada transacao isolada                           |
|                                                                               |
|   COMO O GNN DETECTOU:                                                       |
|   ═════════════════════                                                      |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  1. DETECTOU CLUSTER ISOLADO                                            │|
|   │     • 200 contas com 99% de transacoes APENAS entre si                  │|
|   │     • Clientes normais: 80% transacoes com contas EXTERNAS              │|
|   │                                                                          │|
|   │  2. DETECTOU ESTRUTURA EM CAMADAS                                       │|
|   │     • Dinheiro sempre flui na mesma direcao (nunca volta)               │|
|   │     • Camadas bem definidas (1→2→3→4→5)                                 │|
|   │                                                                          │|
|   │  3. DETECTOU SINCRONIZACAO TEMPORAL                                     │|
|   │     • Todas as transacoes em janelas de 4-6 horas                       │|
|   │     • Clientes normais: transacoes ao longo do dia                      │|
|   │                                                                          │|
|   │  4. DETECTOU FEATURES SUSPEITAS                                         │|
|   │     • 95% das contas criadas nos ultimos 90 dias                        │|
|   │     • Mesmos IPs para multiplas contas                                  │|
|   │                                                                          │|
|   │  VISUALIZACAO DO GRAFO:                                                 │|
|   │                                                                          │|
|   │      CAMADA 0        CAMADA 1       CAMADA 2       CAMADA 4             │|
|   │      (entrada)      (dispersao)    (dispersao)      (saida)             │|
|   │                                                                          │|
|   │       ● ● ●            ●●●●          ●●●●●●           ●●●               │|
|   │      ●     ● ───────→ ●●●●● ───────→ ●●●●●● ───────→ ●●●               │|
|   │       ● ● ●            ●●●●          ●●●●●●           ●●●               │|
|   │                                                                          │|
|   │  Score de anomalia do cluster: 0.97                                     │|
|   │  Probabilidade de crime organizado: 99.2%                               │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   RESULTADO:                                                                 |
|   ══════════                                                                 |
|   • 200 contas bloqueadas SIMULTANEAMENTE                                   |
|   • R$ 12 milhoes retidos antes de sair do sistema                          |
|   • 8 organizadores presos                                                  |
|   • Esquema desmantelado em 72 horas                                        |
|                                                                               |
+==============================================================================+
```

---

## Especialista 5: Federated Learning - A Alianca dos Bancos

```
+==============================================================================+
|                    FEDERATED LEARNING: A ALIANCA DOS BANCOS                  |
+==============================================================================+
|                                                                               |
|   ANALOGIA DO DIA A DIA:                                                     |
|   ════════════════════════                                                   |
|                                                                               |
|   Imagine que 12 hospitais querem criar um modelo para diagnosticar cancer. |
|   Cada hospital tem milhares de exames de pacientes.                        |
|                                                                               |
|   PROBLEMA: Eles NAO PODEM compartilhar dados dos pacientes (LGPD!)         |
|                                                                               |
|   SOLUCAO TRADICIONAL: Impossivel treinar modelo bom                        |
|   SOLUCAO FEDERATED: Cada hospital treina LOCALMENTE                        |
|                      e compartilha apenas o CONHECIMENTO (pesos do modelo)  |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │                        SERVIDOR CENTRAL (Swift)                         │|
|   │                   ┌─────────────────────────────┐                       │|
|   │                   │    MODELO GLOBAL DE FRAUDE  │                       │|
|   │                   │    (combinacao de todos)    │                       │|
|   │                   └──────────────┬──────────────┘                       │|
|   │                                  │                                       │|
|   │              ┌───────────────────┼───────────────────┐                  │|
|   │              │                   │                   │                  │|
|   │              ▼                   ▼                   ▼                  │|
|   │       ┌──────────┐        ┌──────────┐        ┌──────────┐             │|
|   │       │  BANCO   │        │  BANCO   │        │  BANCO   │             │|
|   │       │  ITAU    │        │BRADESCO  │        │ NUBANK   │             │|
|   │       │          │        │          │        │          │             │|
|   │       │ [dados]  │        │ [dados]  │        │ [dados]  │             │|
|   │       │ locais   │        │ locais   │        │ locais   │             │|
|   │       └──────────┘        └──────────┘        └──────────┘             │|
|   │                                                                          │|
|   │   CICLO:                                                                │|
|   │   1. Servidor envia modelo global para cada banco                       │|
|   │   2. Cada banco treina LOCALMENTE (dados nunca saem!)                   │|
|   │   3. Bancos enviam apenas PESOS do modelo (nao dados)                   │|
|   │   4. Servidor combina pesos e cria novo modelo global                   │|
|   │   5. Repete...                                                          │|
|   │                                                                          │|
|   │   RESULTADO:                                                            │|
|   │   Modelo treinado em 500 MILHOES de transacoes                          │|
|   │   SEM nenhum dado sair de cada banco!                                   │|
|   │   100% compliant com LGPD/GDPR!                                         │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

### Caso Real: Swift + Google Cloud (2025)

```
+==============================================================================+
|                    CASO REAL: SWIFT + GOOGLE (12 BANCOS)                     |
+==============================================================================+
|                                                                               |
|   INICIATIVA:                                                                |
|   ════════════                                                               |
|   Em 2025, Swift e Google Cloud lancaram iniciativa com 12 bancos globais   |
|   para treinar modelos de fraude SEM compartilhar dados de clientes.        |
|                                                                               |
|   PARTICIPANTES:                                                             |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  • 3 bancos europeus (UK, Alemanha, Franca)                             │|
|   │  • 3 bancos americanos (EUA, Canada)                                    │|
|   │  • 3 bancos asiaticos (Japao, Singapura, Hong Kong)                     │|
|   │  • 2 bancos australianos                                                │|
|   │  • 1 banco latino-americano (Brasil)                                    │|
|   │                                                                          │|
|   │  VOLUME COMBINADO: 2+ bilhoes de transacoes/ano                         │|
|   │  COBERTURA: 197 paises                                                  │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   CASO DE SUCESSO: Quadrilha Internacional Detectada                         |
|   ════════════════════════════════════════════════════                       |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  ANTES DO FEDERATED LEARNING:                                           │|
|   │  ─────────────────────────────                                          │|
|   │  Quadrilha operava assim:                                               │|
|   │                                                                          │|
|   │  1. Roubavam cartoes no BRASIL (skimming em caixas)                     │|
|   │  2. Vendiam dados para parceiros na EUROPA                              │|
|   │  3. Parceiros faziam compras na ALEMANHA                                │|
|   │  4. Produtos enviados para HONG KONG                                    │|
|   │  5. Revendiam e lavavam dinheiro em SINGAPURA                           │|
|   │                                                                          │|
|   │  PROBLEMA:                                                              │|
|   │  • Banco brasileiro via roubo, mas NAO via uso                          │|
|   │  • Banco alemao via compra estranha, mas NAO via origem                 │|
|   │  • Banco de HK via movimentacao, mas NAO via contexto                   │|
|   │                                                                          │|
|   │  CADA BANCO via apenas UMA PARTE do crime!                              │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  COM FEDERATED LEARNING:                                                │|
|   │  ────────────────────────                                               │|
|   │  O modelo global APRENDEU o padrao completo:                            │|
|   │                                                                          │|
|   │  "Cartao brasileiro → compra Alemanha → frete HK → depositos SG"        │|
|   │  = 99.7% de probabilidade de crime organizado internacional             │|
|   │                                                                          │|
|   │  SEM compartilhar nenhum dado individual!                               │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   RESULTADO:                                                                 |
|   • Quadrilha detectada em 72 horas                                         |
|   • 47 membros presos em 4 paises                                           |
|   • $12 milhoes recuperados                                                 |
|   • 30% de melhoria na acuracia de todos os bancos participantes            |
|                                                                               |
+==============================================================================+
```

---

# ATO 3: A SALA DE GUERRA

## Cronologia Completa: Quem Faz o Que

```
+==============================================================================+
|                    CRONOLOGIA: ORQUESTRA DE MODELOS                          |
+==============================================================================+
|                                                                               |
|   TEMPO         MODELO                 FUNCAO                   METRICAS     |
|   ═════════════════════════════════════════════════════════════════════════  |
|                                                                               |
|   0-15ms        EXTRACAO DE FEATURES   Coleta 30 informacoes    -            |
|                                                                               |
|   15-22ms       RANDOM FOREST          Votacao de 100 arvores   Prec: 95%    |
|                 🌲🌲🌲                 "E fraude ou nao?"       Rec: 92%     |
|                                                                               |
|   22-26ms       XGBOOST                Aprende com erros        Prec: 94%    |
|                 🚀                     100 rodadas de melhoria  Rec: 93%     |
|                                                                               |
|   26-28ms       LIGHTGBM               Analise super rapida     Prec: 93%    |
|                 ⚡                     Otimizado para GPU       Rec: 91%     |
|                                                                               |
|   28-30ms       STACKING               Combina os 3 scores      Prec: 97%    |
|                 🎯                     Peso ponderado           Rec: 94%     |
|                                                                               |
|   ─────────────────────────────────────────────────────────────────────────  |
|   SE SCORE > 0.35, ACIONA ESPECIALISTAS:                                     |
|   ─────────────────────────────────────────────────────────────────────────  |
|                                                                               |
|   30-45ms       LSTM/GRU               Analisa sequencia        Prec: 98%    |
|                 🧠                     de transacoes            Rec: 95%     |
|                                                                               |
|   30-45ms       AUTOENCODER            Erro de reconstrucao     Prec: 99%    |
|   (paralelo)    🔍                     "E estranho?"            Rec: 85%     |
|                                                                               |
|   45-60ms       TABTRANSFORMER         Contexto combinado       Prec: 97%    |
|   (se card test)📊                     BIN+CEP+Merchant         Rec: 97%     |
|                                                                               |
|   ─────────────────────────────────────────────────────────────────────────  |
|   MODELOS DE ANALISE POSTERIOR (nao em tempo real):                          |
|   ─────────────────────────────────────────────────────────────────────────  |
|                                                                               |
|   1-5min        GNN                    Mapeamento de redes      Prec: 99%    |
|                 🕸️                     Deteccao de laranjas     Rec: 96%     |
|                                                                               |
|   Background    FEDERATED LEARNING     Treino colaborativo      +30%         |
|                 🤝                     entre bancos             acuracia     |
|                                                                               |
|   Background    VAE                    Geracao de cenarios      Bal: 95%     |
|                 📈                     Dados sinteticos         dados        |
|                                                                               |
+==============================================================================+
```

## Tabela de Metricas por Modelo

```
+==============================================================================+
|                    TABELA COMPLETA DE METRICAS                               |
+==============================================================================+
|                                                                               |
|   MODELO           | PRECISION | RECALL | F1     | LATENCIA | MELHOR PARA   |
|   ═════════════════|═══════════|════════|════════|══════════|═══════════════|
|                    |           |        |        |          |               |
|   Random Forest    |   95.0%   | 92.0%  | 93.5%  |   7ms    | Primeira      |
|   🌲               |           |        |        |          | triagem       |
|                    |           |        |        |          |               |
|   XGBoost          |   94.0%   | 93.0%  | 93.5%  |   7ms    | Aprendizado   |
|   🚀               |           |        |        |          | de erros      |
|                    |           |        |        |          |               |
|   LightGBM         |   93.0%   | 91.0%  | 92.0%  |   2ms    | Velocidade    |
|   ⚡               |           |        |        |          | (GPU)         |
|                    |           |        |        |          |               |
|   Stacking         |   97.0%   | 94.0%  | 95.5%  |   2ms    | Decisao       |
|   🎯               |           |        |        |          | final         |
|                    |           |        |        |          |               |
|   LSTM/GRU         |   98.2%   | 94.7%  | 96.4%  |  15ms    | Sequencias    |
|   🧠               |           |        |        |          | temporais     |
|                    |           |        |        |          |               |
|   Autoencoder      |   99.5%   | 85.3%  | 91.8%  |  10ms    | Anomalias     |
|   🔍               |           |        |        |          | raras         |
|                    |           |        |        |          |               |
|   TabTransformer   |   97.0%   | 97.0%  | 97.0%  |  20ms    | Card testing  |
|   📊               |           |        |        |          | contexto      |
|                    |           |        |        |          |               |
|   GNN              |   98.7%   | 96.2%  | 97.4%  | 5min     | Redes de      |
|   🕸️               |           |        |        |          | laranjas      |
|                    |           |        |        |          |               |
|   Federated        |   +30%    |  base  |  -     | async    | Multi-banco   |
|   🤝               |           |        |        |          | privacidade   |
|                    |           |        |        |          |               |
|   VAE              |   99.0%   | 82.0%  | 89.6%  |  12ms    | Dados         |
|   📈               |           |        |        |          | sinteticos    |
|                    |           |        |        |          |               |
+==============================================================================+
|                                                                               |
|   LEGENDA:                                                                   |
|   ─────────                                                                  |
|   PRECISION = De cada 100 alertas, quantos sao fraudes REAIS?               |
|   RECALL    = De cada 100 fraudes, quantas o modelo DETECTA?                |
|   F1-SCORE  = Media equilibrada entre Precision e Recall                    |
|   LATENCIA  = Tempo para processar uma transacao                            |
|                                                                               |
+==============================================================================+
```

## Entendendo as Metricas (Use a Cabeca!)

```
+==============================================================================+
|                    🧠 ENTENDENDO PRECISION vs RECALL                         |
+==============================================================================+
|                                                                               |
|   ANALOGIA DO DETECTOR DE METAIS NO AEROPORTO:                              |
|   ═══════════════════════════════════════════════                           |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  ALTA PRECISION, BAIXO RECALL (detector "conservador"):                 │|
|   │  ────────────────────────────────────────────────────                   │|
|   │  • Apita POUCO (so quando tem CERTEZA)                                  │|
|   │  • Quando apita, SEMPRE tem algo errado                                 │|
|   │  • MAS: deixa passar algumas armas escondidas!                          │|
|   │                                                                          │|
|   │  Exemplo: Autoencoder (Precision 99.5%, Recall 85%)                     │|
|   │  → Quase nunca da falso positivo                                        │|
|   │  → Mas perde 15% das fraudes                                            │|
|   │                                                                          │|
|   │  ─────────────────────────────────────────────────────────────────      │|
|   │                                                                          │|
|   │  ALTO RECALL, MENOR PRECISION (detector "paranóico"):                   │|
|   │  ──────────────────────────────────────────────────                     │|
|   │  • Apita MUITO (na duvida, apita!)                                      │|
|   │  • Pega TODAS as armas                                                  │|
|   │  • MAS: muitos alarmes falsos (irritante!)                              │|
|   │                                                                          │|
|   │  Exemplo: GNN (Precision 98.7%, Recall 96.2%)                           │|
|   │  → Pega quase todas as redes de laranjas                                │|
|   │  → Alguns falsos positivos                                              │|
|   │                                                                          │|
|   │  ─────────────────────────────────────────────────────────────────      │|
|   │                                                                          │|
|   │  EQUILIBRIO IDEAL (F1-Score alto):                                      │|
|   │  ─────────────────────────────────                                      │|
|   │  • Apita quando precisa                                                 │|
|   │  • Poucos erros em ambas direcoes                                       │|
|   │                                                                          │|
|   │  Exemplo: TabTransformer (Precision 97%, Recall 97%, F1 97%)            │|
|   │  → Equilibrio perfeito!                                                 │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
+==============================================================================+
```

## Exercicio 2: Escolha o Modelo Certo

```
+==============================================================================+
|                    🧠 EXERCICIO 2: QUAL MODELO USAR?                         |
+==============================================================================+
|                                                                               |
|   Para cada situacao, escolha o modelo MAIS adequado:                        |
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  SITUACAO 1:                                                            │|
|   │  Um e-commerce sofre 50.000 tentativas de compra de $1 em 10 minutos.   │|
|   │  Qual modelo e melhor para detectar isso?                               │|
|   │                                                                          │|
|   │  ( ) Random Forest                                                       │|
|   │  ( ) LSTM/GRU                                                           │|
|   │  ( ) TabTransformer                                                     │|
|   │  ( ) GNN                                                                │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  SITUACAO 2:                                                            │|
|   │  Um cliente que sempre compra cafe as 8h faz um PIX de R$50.000         │|
|   │  as 3h da manha para uma conta na Nigeria.                              │|
|   │  Qual modelo e melhor para detectar isso?                               │|
|   │                                                                          │|
|   │  ( ) Random Forest                                                       │|
|   │  ( ) LSTM/GRU                                                           │|
|   │  ( ) Autoencoder                                                        │|
|   │  ( ) Federated Learning                                                 │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   ┌─────────────────────────────────────────────────────────────────────────┐|
|   │                                                                          │|
|   │  SITUACAO 3:                                                            │|
|   │  Suspeita-se que 200 contas estao sendo usadas para lavar dinheiro      │|
|   │  de golpes de PIX, com dinheiro passando de uma para outra.             │|
|   │  Qual modelo e melhor para detectar isso?                               │|
|   │                                                                          │|
|   │  ( ) XGBoost                                                            │|
|   │  ( ) TabTransformer                                                     │|
|   │  ( ) GNN                                                                │|
|   │  ( ) VAE                                                                │|
|   │                                                                          │|
|   └─────────────────────────────────────────────────────────────────────────┘|
|                                                                               |
|   .                                                                          |
|   .                                                                          |
|   (role para ver as respostas)                                              |
|   .                                                                          |
|   .                                                                          |
|                                                                               |
|   RESPOSTAS:                                                                 |
|   ═══════════                                                                |
|                                                                               |
|   1. TabTransformer ✓                                                        |
|      → Card testing em massa! TabTransformer detecta padroes de contexto    |
|        como mesmo IP, mesma hora, valores repetidos.                        |
|                                                                               |
|   2. LSTM/GRU ✓ (Autoencoder tambem seria bom!)                             |
|      → LSTM lembra do padrao de comportamento (cafe 8h) e detecta           |
|        a mudanca radical (3h, Nigeria, valor alto).                         |
|                                                                               |
|   3. GNN ✓                                                                   |
|      → GNN e especialista em mapear REDES de contas e detectar              |
|        estruturas de lavagem de dinheiro (cascata entre contas).            |
|                                                                               |
+==============================================================================+
```

---

# ANEXOS

## Glossario Visual

```
+==============================================================================+
|                    GLOSSARIO ILUSTRADO                                       |
+==============================================================================+
|                                                                               |
|   TERMO              SIGNIFICADO                    ANALOGIA                 |
|   ═════════════════════════════════════════════════════════════════════════  |
|                                                                               |
|   Feature            Uma informacao sobre           Um campo no              |
|                      a transacao                    formulario               |
|                                                                               |
|   Threshold          Limite de corte                Nota de corte            |
|   (Limiar)           para decidir fraude            no vestibular            |
|                                                                               |
|   Score              Nota de 0 a 1                  Nota na prova            |
|                      (0=seguro, 1=fraude)           (0 a 10)                 |
|                                                                               |
|   Ensemble           Varios modelos                 Juri de jurados          |
|                      votando juntos                 votando                  |
|                                                                               |
|   Stacking           Combinar scores                Media ponderada          |
|                      de varios modelos              das notas                |
|                                                                               |
|   Precision          % de alertas corretos          Tiros no alvo            |
|                                                     que acertam              |
|                                                                               |
|   Recall             % de fraudes detectadas        % de criminosos          |
|                                                     capturados               |
|                                                                               |
|   F1-Score           Media entre Precision          Nota final               |
|                      e Recall                       equilibrada              |
|                                                                               |
|   Latencia           Tempo de resposta              Tempo de espera          |
|                                                     no caixa                 |
|                                                                               |
|   Overfitting        Modelo "decorou" dados         Decorar respostas        |
|                      e nao generaliza               da prova                 |
|                                                                               |
|   Underfitting       Modelo muito simples           Estudar pouco            |
|                      para o problema                para prova               |
|                                                                               |
|   Embedding          Representacao numerica         Codigo de barras         |
|                      de uma categoria               de um produto            |
|                                                                               |
|   Atencao            Mecanismo para                 Grifar partes            |
|   (Attention)        focar no importante            importantes              |
|                                                                               |
|   Espaco Latente     Representacao                  "Essencia" de            |
|                      comprimida dos dados           uma informacao           |
|                                                                               |
|   Grafo              Mapa de conexoes               Mapa de metro            |
|                      entre entidades                (estacoes=nos)           |
|                                                                               |
|   Federated          Treino colaborativo            Reuniao virtual          |
|   Learning           sem compartilhar dados         com sigilo               |
|                                                                               |
+==============================================================================+
```

## Mapa Mental: Todos os Modelos

```
+==============================================================================+
|                    MAPA MENTAL: ECOSSISTEMA DE MODELOS                       |
+==============================================================================+
|                                                                               |
|                                  ┌────────────────┐                          |
|                                  │   TRANSACAO    │                          |
|                                  │   (entrada)    │                          |
|                                  └───────┬────────┘                          |
|                                          │                                    |
|                                          ▼                                    |
|                          ┌───────────────────────────────┐                   |
|                          │   PRIMEIRA LINHA DE DEFESA    │                   |
|                          │   ═══════════════════════════ │                   |
|                          │                               │                   |
|                          │  ┌─────────┐ ┌─────────┐ ┌─────────┐            │
|                          │  │Random   │ │XGBoost  │ │LightGBM │            │
|                          │  │Forest   │ │  🚀     │ │  ⚡     │            │
|                          │  │  🌲     │ │         │ │         │            │
|                          │  └────┬────┘ └────┬────┘ └────┬────┘            │
|                          │       │           │           │                  │
|                          │       └───────────┼───────────┘                  │
|                          │                   │                               │
|                          │                   ▼                               │
|                          │           ┌─────────────┐                        │
|                          │           │  STACKING   │                        │
|                          │           │     🎯      │                        │
|                          │           └──────┬──────┘                        │
|                          │                  │                                │
|                          └──────────────────┼────────────────────────────────┘
|                                             │                                 |
|                              ┌──────────────┴──────────────┐                 |
|                              │                             │                 |
|                              ▼                             ▼                 |
|                       Score < 0.35               Score >= 0.35               |
|                              │                             │                 |
|                              ▼                             ▼                 |
|                      ┌─────────────┐         ┌─────────────────────────────┐|
|                      │   APROVAR   │         │   ESPECIALISTAS ACIONADOS   │|
|                      │     ✓       │         │   ═══════════════════════   │|
|                      └─────────────┘         │                             │|
|                                              │ ┌────────┐  ┌────────────┐  │|
|                                              │ │LSTM/GRU│  │TabTransform│  │|
|                                              │ │  🧠    │  │    📊      │  │|
|                                              │ └────────┘  └────────────┘  │|
|                                              │                             │|
|                                              │ ┌────────┐  ┌────────┐      │|
|                                              │ │Autoenc │  │  GNN   │      │|
|                                              │ │  🔍    │  │  🕸️   │      │|
|                                              │ └────────┘  └────────┘      │|
|                                              │                             │|
|                                              └──────────────┬──────────────┘|
|                                                             │                |
|                                              ┌──────────────┴──────────────┐|
|                                              │                             │|
|                                              ▼                             ▼|
|                                       Score >= 0.50               Score < 0.50
|                                              │                             │|
|                                              ▼                             ▼|
|                                      ┌─────────────┐              ┌─────────────┐
|                                      │   BLOQUEAR  │              │   APROVAR   │
|                                      │  + REVISAR  │              │  (monitorar)│
|                                      │      🚫     │              │      ✓      │
|                                      └─────────────┘              └─────────────┘
|                                                                               |
|   ═══════════════════════════════════════════════════════════════════════════|
|                                                                               |
|   EM BACKGROUND (nao em tempo real):                                         |
|                                                                               |
|   ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐    |
|   │ Federated Learning │  │        VAE         │  │   Analise Batch    │    |
|   │        🤝          │  │        📈          │  │      (diaria)      │    |
|   │                    │  │                    │  │                    │    |
|   │ Treino colaborativo│  │ Geracao de dados   │  │ Revisao manual     │    |
|   │ entre bancos       │  │ sinteticos         │  │ de casos           │    |
|   └────────────────────┘  └────────────────────┘  └────────────────────┘    |
|                                                                               |
+==============================================================================+
```

---

## Resumo da Aula

```
+==============================================================================+
|                    RESUMO: O QUE VOCE APRENDEU                               |
+==============================================================================+
|                                                                               |
|   ✅ A jornada de uma transacao leva apenas 30ms                            |
|                                                                               |
|   ✅ 3 modelos votam na primeira linha: RF, XGBoost, LightGBM               |
|                                                                               |
|   ✅ Stacking combina os votos com pesos                                    |
|                                                                               |
|   ✅ Especialistas sao acionados para casos complexos:                      |
|      • LSTM: sequencias temporais (cartao clonado)                          |
|      • TabTransformer: contexto combinado (card testing)                    |
|      • Autoencoder: anomalias raras (identidade sintetica)                  |
|      • GNN: redes de contas (lavagem de dinheiro)                           |
|      • Federated Learning: aprendizado multi-banco                          |
|                                                                               |
|   ✅ Precision = "De cada alerta, quantos sao corretos?"                    |
|                                                                               |
|   ✅ Recall = "De cada fraude, quantas detectamos?"                         |
|                                                                               |
|   ✅ F1 = equilibrio entre Precision e Recall                               |
|                                                                               |
+==============================================================================+
|                                                                               |
|               PARABENS! VOCE COMPLETOU A AULA! 🎓                           |
|                                                                               |
|      Agora voce sabe como a IA protege seu dinheiro todos os dias.          |
|                                                                               |
+==============================================================================+
```

---

## Referencias

| Recurso | Link | Descricao |
|---------|------|-----------|
| Stripe Blog | stripe.com/blog | Caso TabTransformer |
| IBM ai-on-z | github.com/IBM/ai-on-z-fraud-detection | LSTM/GRU |
| Flower | flower.dev | Federated Learning |
| PyTorch Geometric | pyg.org | GNN |
| Fraud Detection Handbook | fraud-detection-handbook.github.io | Autoencoders |

---

*Use a Cabeca: Machine Learning para Deteccao de Fraude*  
*Sankofa Enterprise Pro v12.1*  
*27 de Novembro de 2025*
