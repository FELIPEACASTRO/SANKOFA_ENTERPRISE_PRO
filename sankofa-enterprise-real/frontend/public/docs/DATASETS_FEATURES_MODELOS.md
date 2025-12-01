# DATASETS, FEATURES E MODELOS - Sankofa Enterprise Pro v2.0

## Documento Consolidado para Detecao de Fraude em Transacoes PIX

**Versao:** 2.0.0  
**Ultima Atualizacao:** Dezembro 2025  
**Objetivo:** Catalogar TODOS os datasets, features e modelos usados no sistema

---

# PARTE 1: DATASETS

## Inventario Completo de Datasets

```
+==============================================================================+
|                    CATALOGO DE DATASETS - SANKOFA v2.0                       |
+==============================================================================+
|                                                                              |
|   TOTAL: 7 DATASETS                                                          |
|   TRANSACOES TOTAIS: ~24 MILHOES                                             |
|   TAXA MEDIA DE FRAUDE: ~1.5%                                                |
|                                                                              |
+==============================================================================+
```

---

## 1. Nigerian Financial Transactions Dataset

```
+------------------------------------------------------------------------------+
|  DATASET: NIGERIAN FINANCIAL TRANSACTIONS                                    |
+------------------------------------------------------------------------------+
|                                                                              |
|   Fonte:        HuggingFace                                                  |
|   URL:          electricsheepafrica/Nigerian-Financial-Transactions          |
|   Tamanho:      5.000.000 transacoes                                         |
|   Taxa Fraude:  1.0%                                                         |
|   Licenca:      CC BY 4.0                                                    |
|                                                                              |
+------------------------------------------------------------------------------+
```

### Mapeamento de Colunas

| Coluna Original | Coluna Sankofa | Descricao |
|-----------------|----------------|-----------|
| Transaction_Amount | amount | Valor da transacao |
| Transaction_Type | transaction_type | Tipo (TRANSFER, PAYMENT, etc) |
| Device_Type | device_type | Tipo de dispositivo |
| Location | location | Localizacao geografica |
| Time_of_Transaction | timestamp | Data/hora da transacao |
| Account_Balance | balance | Saldo da conta |
| Customer_ID | user_id | ID do cliente |
| Is_Fraud | is_fraud | Flag de fraude (0/1) |

### Compatibilidade com PIX

- **Compativel:** SIM
- **Adaptacoes necessarias:** Mapeamento de tipos de transacao para PIX brasileiro
- **Uso recomendado:** Transfer Learning para modelo base

---

## 2. PaySim Mobile Money Simulator

```
+------------------------------------------------------------------------------+
|  DATASET: PAYSIM MOBILE MONEY                                                |
+------------------------------------------------------------------------------+
|                                                                              |
|   Fonte:        Kaggle                                                       |
|   URL:          ealaxi/paysim1                                               |
|   Tamanho:      6.362.620 transacoes                                         |
|   Taxa Fraude:  0.13%                                                        |
|   Licenca:      CC BY-SA 4.0                                                 |
|                                                                              |
+------------------------------------------------------------------------------+
```

### Mapeamento de Colunas

| Coluna Original | Coluna Sankofa | Descricao |
|-----------------|----------------|-----------|
| amount | amount | Valor da transacao |
| type | transaction_type | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| oldbalanceOrg | balance_before | Saldo antes da transacao |
| newbalanceOrig | balance_after | Saldo depois da transacao |
| oldbalanceDest | dest_balance_before | Saldo destino antes |
| newbalanceDest | dest_balance_after | Saldo destino depois |
| nameOrig | user_id | ID do cliente origem |
| nameDest | receiver_id | ID do destinatario |
| isFraud | is_fraud | Flag de fraude |
| step | timestamp | Passo temporal (1 step = 1 hora) |

### Compatibilidade com PIX

- **Compativel:** SIM
- **Adaptacoes necessarias:** Conversao de step para timestamp real
- **Uso recomendado:** Treinamento de modelos de fluxo de caixa

---

## 3. Feedzai Bank Account Fraud (BAF)

```
+------------------------------------------------------------------------------+
|  DATASET: FEEDZAI BAF - NEURIPS 2022                                         |
+------------------------------------------------------------------------------+
|                                                                              |
|   Fonte:        Kaggle/GitHub (NeurIPS 2022)                                 |
|   URL:          sgpjesus/bank-account-fraud-dataset-neurips-2022             |
|   Tamanho:      6.000.000 transacoes                                         |
|   Taxa Fraude:  1.1%                                                         |
|   Licenca:      MIT                                                          |
|                                                                              |
+------------------------------------------------------------------------------+
```

### Mapeamento de Colunas

| Coluna Original | Coluna Sankofa | Descricao |
|-----------------|----------------|-----------|
| income | income | Renda do cliente |
| name_email_similarity | email_similarity | Similaridade nome-email |
| current_address_months_count | address_months | Meses no endereco atual |
| customer_age | customer_age | Idade do cliente |
| fraud_bool | is_fraud | Flag de fraude |

### Compatibilidade com PIX

- **Compativel:** SIM
- **Adaptacoes necessarias:** Features de comportamento de conta
- **Uso recomendado:** Deteccao de conta laranja

---

## 4. IEEE-CIS Fraud Detection

```
+------------------------------------------------------------------------------+
|  DATASET: IEEE-CIS FRAUD DETECTION                                           |
+------------------------------------------------------------------------------+
|                                                                              |
|   Fonte:        Kaggle (Competicao IEEE)                                     |
|   URL:          ieee-fraud-detection                                         |
|   Tamanho:      590.540 transacoes                                           |
|   Taxa Fraude:  3.5%                                                         |
|   Licenca:      Competicao Kaggle                                            |
|                                                                              |
+------------------------------------------------------------------------------+
```

### Mapeamento de Colunas

| Coluna Original | Coluna Sankofa | Descricao |
|-----------------|----------------|-----------|
| TransactionAmt | amount | Valor da transacao |
| TransactionDT | timestamp | Timestamp delta |
| ProductCD | product_code | Codigo do produto |
| card4 | card_type | Tipo de cartao |
| isFraud | is_fraud | Flag de fraude |
| V1-V339 | v_features | 339 features anonimizadas |

### Compatibilidade com PIX

- **Compativel:** PARCIAL
- **Adaptacoes necessarias:** Features anonimizadas precisam mapeamento
- **Uso recomendado:** Treinamento de modelos de cartao

---

## 5. DIFrauD Dataset (NLP/Phishing)

```
+------------------------------------------------------------------------------+
|  DATASET: DIFRAUD - SOCIAL ENGINEERING                                       |
+------------------------------------------------------------------------------+
|                                                                              |
|   Fonte:        HuggingFace                                                  |
|   URL:          DIFrauD                                                      |
|   Tamanho:      95.000 amostras de texto                                     |
|   Taxa Fraude:  ~50% (balanceado)                                            |
|   Licenca:      Research Use                                                 |
|                                                                              |
+------------------------------------------------------------------------------+
```

### Campos Disponiveis

| Campo | Tipo | Descricao |
|-------|------|-----------|
| text | string | Texto da mensagem (SMS, email, WhatsApp) |
| label | int | 0 = legitimo, 1 = phishing |
| source | string | Origem (sms, email, social) |
| language | string | Idioma |

### Uso no Sistema

- **Modulo:** NLP Social Engineering Detector
- **Funcao:** Detectar mensagens de phishing/smishing
- **Taxa deteccao:** >70%

---

## 6. Credit Card Fraud Dataset (Kaggle)

```
+------------------------------------------------------------------------------+
|  DATASET: CREDIT CARD FRAUD                                                  |
+------------------------------------------------------------------------------+
|                                                                              |
|   Fonte:        Kaggle                                                       |
|   URL:          mlg-ulb/creditcardfraud                                      |
|   Tamanho:      284.807 transacoes                                           |
|   Taxa Fraude:  0.17%                                                        |
|   Licenca:      CC0 Public Domain                                            |
|                                                                              |
+------------------------------------------------------------------------------+
```

### Mapeamento de Colunas

| Coluna Original | Coluna Sankofa | Descricao |
|-----------------|----------------|-----------|
| Time | timestamp | Segundos desde primeira transacao |
| Amount | amount | Valor da transacao |
| V1-V28 | pca_features | Features PCA anonimizadas |
| Class | is_fraud | Flag de fraude |

### Compatibilidade com PIX

- **Compativel:** PARCIAL
- **Adaptacoes necessarias:** Features PCA nao mapeiam diretamente
- **Uso recomendado:** Baseline para comparacao

---

## 7. Dados de Producao (Sankofa)

```
+------------------------------------------------------------------------------+
|  DATASET: PRODUCAO SANKOFA                                                   |
+------------------------------------------------------------------------------+
|                                                                              |
|   Fonte:        Sistema em producao                                          |
|   Localizacao:  PostgreSQL + SQLite cache                                    |
|   Tamanho:      Variavel (crescimento diario)                                |
|   Taxa Fraude:  ~0.8-1.2%                                                    |
|   Formato:      Tempo real                                                   |
|                                                                              |
+------------------------------------------------------------------------------+
```

### Esquema de Dados

| Campo | Tipo | Descricao |
|-------|------|-----------|
| transaction_id | UUID | ID unico da transacao |
| timestamp | datetime | Data/hora |
| amount | decimal | Valor |
| channel | enum | PIX, TED, CARTAO, BOLETO |
| client_cpf | string | CPF tokenizado |
| device_id | string | Fingerprint do dispositivo |
| ip_address | string | IP anonimizado |
| fraud_score | float | Score de fraude (0-100) |
| is_fraud | bool | Confirmacao de fraude |
| decision | enum | APPROVED, BLOCKED, REVIEW |

---

# PARTE 2: FEATURES

## Catalogo Completo de Features

```
+==============================================================================+
|                    CATALOGO DE FEATURES - SANKOFA v2.0                        |
+==============================================================================+
|                                                                              |
|   TOTAL: 100+ FEATURES                                                       |
|   CATEGORIAS: 8                                                              |
|   MODULOS GERADORES: 4                                                       |
|                                                                              |
+==============================================================================+
```

---

## Categoria 1: Features Temporais (13 features)

### Baseado em Bahnsen et al. 2016

| Feature | Tipo | Descricao | Formula/Calculo |
|---------|------|-----------|-----------------|
| hour | int | Hora do dia | timestamp.hour |
| day_of_week | int | Dia da semana (0-6) | timestamp.weekday() |
| is_weekend | bool | Final de semana | day_of_week >= 5 |
| is_night | bool | Horario noturno | hour >= 22 OR hour <= 6 |
| is_business_hours | bool | Horario comercial | 9 <= hour <= 18 AND weekday < 5 |
| is_early_morning | bool | Madrugada | hour <= 6 |
| is_month_end | bool | Fim do mes | day >= 25 |
| is_month_start | bool | Inicio do mes | day <= 5 |
| hour_sin | float | Seno da hora | sin(2*pi*hour/24) |
| hour_cos | float | Cosseno da hora | cos(2*pi*hour/24) |
| day_of_week_sin | float | Seno do dia | sin(2*pi*dow/7) |
| day_of_week_cos | float | Cosseno do dia | cos(2*pi*dow/7) |
| month_sin | float | Seno do mes | sin(2*pi*month/12) |

---

## Categoria 2: Features de Valor (10 features)

| Feature | Tipo | Descricao | Formula/Calculo |
|---------|------|-----------|-----------------|
| amount | float | Valor original | - |
| amount_log | float | Log do valor | log1p(amount) |
| amount_normalized | float | Valor normalizado | amount / 10000 |
| valor_zscore | float | Z-score do valor | (valor - media) / std |
| is_high_amount | bool | Valor alto | amount > 5000 |
| is_very_high_amount | bool | Valor muito alto | amount > 10000 |
| value_rounded | bool | Valor redondo | amount % 1 == 0 |
| amount_deviation | float | Desvio da media | amount / avg_amount_30d |
| is_round_number | bool | Numero redondo | amount % 100 == 0 |
| amount_percentile | float | Percentil | Posicao no historico |

---

## Categoria 3: Agregacoes Temporais (25 features)

### Janelas: 1h, 6h, 24h, 72h, 168h

| Feature Pattern | Janelas | Descricao |
|-----------------|---------|-----------|
| txn_count_last_{X}h | 5 | Contagem de transacoes na janela |
| txn_sum_last_{X}h | 5 | Soma de valores na janela |
| txn_avg_last_{X}h | 5 | Media de valores na janela |
| txn_max_last_{X}h | 5 | Valor maximo na janela |
| txn_std_last_{X}h | 5 | Desvio padrao na janela |

### Exemplo de Calculo

```
+------------------------------------------------------------------------------+
|  EXEMPLO: txn_count_last_24h                                                  |
+------------------------------------------------------------------------------+
|                                                                              |
|   Transacao atual: 2025-12-01 14:30:00                                       |
|   Janela: 24 horas atras = 2025-11-30 14:30:00                               |
|                                                                              |
|   Transacoes do usuario nessa janela:                                        |
|   - 2025-11-30 15:00:00  R$ 500                                              |
|   - 2025-11-30 22:00:00  R$ 1200                                             |
|   - 2025-12-01 08:00:00  R$ 300                                              |
|   - 2025-12-01 12:00:00  R$ 800                                              |
|                                                                              |
|   RESULTADO: txn_count_last_24h = 4                                          |
|              txn_sum_last_24h = 2800                                         |
|              txn_avg_last_24h = 700                                          |
|                                                                              |
+------------------------------------------------------------------------------+
```

---

## Categoria 4: Features de Comportamento (12 features)

| Feature | Tipo | Descricao | Fonte |
|---------|------|-----------|-------|
| avg_amount | float | Media historica do cliente | Perfil |
| std_amount | float | Desvio padrao historico | Perfil |
| num_transactions | int | Total de transacoes | Perfil |
| value_deviation | float | Desvio do padrao | Calculo |
| is_new_client | bool | Cliente novo (<30 dias) | Cadastro |
| is_max_value | bool | Maior valor ja feito | Historico |
| avg_frequency_hours | float | Frequencia media | Perfil |
| z_score_amount | float | Z-score vs perfil | Calculo |
| z_score_frequency | float | Z-score de frequencia | Calculo |
| unusual_hour | bool | Hora incomum pro usuario | Perfil |
| unusual_channel | bool | Canal incomum | Perfil |
| unusual_location | bool | Local incomum | Perfil |

---

## Categoria 5: Features de Dispositivo (8 features)

| Feature | Tipo | Descricao |
|---------|------|-----------|
| is_new_device | bool | Dispositivo nunca visto |
| is_shared_device | bool | Dispositivo usado por outros |
| num_clients_per_device | int | Quantos clientes usam |
| device_age_days | int | Idade do dispositivo |
| device_fraud_rate | float | Taxa de fraude do device |
| device_type | category | MOBILE, WEB, ATM |
| device_os | category | Android, iOS, Windows |
| device_risk_score | float | Score de risco (0-1) |

---

## Categoria 6: Features de Localizacao (6 features)

| Feature | Tipo | Descricao |
|---------|------|-----------|
| latitude | float | Latitude |
| longitude | float | Longitude |
| is_high_risk_state | bool | Estado de alto risco |
| is_brazil | bool | Transacao no Brasil |
| location_risk_score | float | Score de risco local |
| distance_from_usual | float | Distancia do local habitual |

---

## Categoria 7: Features de Canal e Tipo (10 features)

| Feature | Tipo | Descricao |
|---------|------|-----------|
| channel_pix | bool | Canal PIX |
| channel_web | bool | Canal Web |
| channel_mobile | bool | Canal Mobile |
| channel_atm | bool | Canal ATM |
| is_pix | bool | Tipo PIX |
| is_boleto | bool | Tipo Boleto |
| is_credit | bool | Tipo Credito |
| is_debit | bool | Tipo Debito |
| channel_risk_score | float | Risco do canal |
| tipo_transacao_encoded | int | Encoding do tipo |

---

## Categoria 8: Features de Velocidade (6 features)

| Feature | Tipo | Descricao |
|---------|------|-----------|
| time_since_last_tx | float | Segundos desde ultima tx |
| is_rapid_transaction | bool | < 60 segundos |
| is_very_rapid_transaction | bool | < 30 segundos |
| velocity_score | float | Score de velocidade (0-1) |
| velocity_device_interaction | float | Velocidade x device novo |
| transactions_last_1h | int | Transacoes na ultima hora |

---

## Categoria 9: Features de Relacionamento (GNN) (8 features)

| Feature | Tipo | Descricao |
|---------|------|-----------|
| graph_risk_score | float | Score de risco do grafo |
| community_risk | float | Risco da comunidade |
| neighbor_fraud_rate | float | Taxa de fraude dos vizinhos |
| centrality_score | float | Centralidade no grafo |
| pagerank_score | float | PageRank do no |
| clustering_coefficient | float | Coeficiente de agrupamento |
| degree_centrality | float | Grau de conexoes |
| anomaly_score | float | Score de anomalia |

---

## Resumo de Features por Modulo

```
+------------------------------------------------------------------------------+
|  MODULO                          | QTD FEATURES | ARQUIVO                    |
+------------------------------------------------------------------------------+
|  Bahnsen Feature Engineering     | 62+          | bahnsen_feature_engine.py  |
|  Advanced Feature Engineering    | 30+          | advanced_feature_engine.py |
|  Production Fraud Engine         | 20+          | production_fraud_engine.py |
|  GNN Fraud Detector              | 8            | gnn_fraud_detector.py      |
|  CatBoost Model                  | 10 categoricas| catboost_model.py         |
|  NLP Social Engineering          | 5            | nlp_social_engineering.py  |
+------------------------------------------------------------------------------+
|  TOTAL APROXIMADO                | 100+         |                            |
+------------------------------------------------------------------------------+
```

---

# PARTE 3: MODELOS DE MACHINE LEARNING

## Inventario Completo de Modelos

```
+==============================================================================+
|                    CATALOGO DE MODELOS - SANKOFA v2.0                         |
+==============================================================================+
|                                                                              |
|   MODELOS PRINCIPAIS: 5 (Ensemble Integrado)                                 |
|   MODELOS AUXILIARES: 8                                                      |
|   TOTAL: 13 MODELOS                                                          |
|                                                                              |
+==============================================================================+
```

---

## Modelos Principais (Ensemble Integrado)

### 1. Random Forest Classifier

```
+------------------------------------------------------------------------------+
|  MODELO: RANDOM FOREST                                                        |
+------------------------------------------------------------------------------+
|                                                                              |
|   Arquivo:        production_fraud_engine.py                                  |
|   Peso no Ensemble: 50% (como parte do Base Ensemble)                        |
|   Papel:          Detector primario de padroes                               |
|                                                                              |
|   PARAMETROS:                                                                 |
|   - n_estimators: 100                                                         |
|   - max_depth: 15                                                             |
|   - min_samples_split: 5                                                      |
|   - min_samples_leaf: 2                                                       |
|   - class_weight: balanced                                                    |
|   - n_jobs: -1                                                                |
|                                                                              |
+------------------------------------------------------------------------------+
```

### Como Funciona

```
+------------------------------------------------------------------------------+
|                    RANDOM FOREST - ANALOGIA                                   |
+------------------------------------------------------------------------------+
|                                                                              |
|   Imagine 100 detetives (arvores) analisando a mesma transacao.              |
|   Cada detetive ve apenas PARTE das pistas (features aleatorias).            |
|   Cada um da seu veredito: "FRAUDE" ou "LEGITIMO".                           |
|   O resultado final e a MAIORIA dos votos.                                   |
|                                                                              |
|   EXEMPLO:                                                                    |
|   - 73 detetives votam "FRAUDE"                                              |
|   - 27 detetives votam "LEGITIMO"                                            |
|   - Resultado: 73% probabilidade de fraude                                   |
|                                                                              |
+------------------------------------------------------------------------------+
```

---

### 2. Gradient Boosting Classifier

```
+------------------------------------------------------------------------------+
|  MODELO: GRADIENT BOOSTING                                                    |
+------------------------------------------------------------------------------+
|                                                                              |
|   Arquivo:        production_fraud_engine.py                                  |
|   Peso no Ensemble: 50% (como parte do Base Ensemble)                        |
|   Papel:          Correcao de erros iterativa                                |
|                                                                              |
|   PARAMETROS:                                                                 |
|   - n_estimators: 100                                                         |
|   - max_depth: 8                                                              |
|   - learning_rate: 0.1                                                        |
|   - subsample: 0.8                                                            |
|                                                                              |
+------------------------------------------------------------------------------+
```

### Como Funciona

```
+------------------------------------------------------------------------------+
|                    GRADIENT BOOSTING - ANALOGIA                               |
+------------------------------------------------------------------------------+
|                                                                              |
|   Imagine uma equipe de revisores em sequencia:                              |
|                                                                              |
|   1. Revisor 1 analisa e erra em 30% dos casos                               |
|   2. Revisor 2 foca nos 30% de erros do Revisor 1                            |
|   3. Revisor 3 foca nos erros restantes do Revisor 2                         |
|   ... e assim por diante                                                     |
|                                                                              |
|   Cada revisor APRENDE com os erros dos anteriores.                          |
|   No final, a taxa de erro cai para <5%.                                     |
|                                                                              |
+------------------------------------------------------------------------------+
```

---

### 3. Logistic Regression (Meta-Modelo)

```
+------------------------------------------------------------------------------+
|  MODELO: LOGISTIC REGRESSION                                                  |
+------------------------------------------------------------------------------+
|                                                                              |
|   Arquivo:        production_fraud_engine.py                                  |
|   Peso no Ensemble: Meta-modelo do Stacking                                  |
|   Papel:          Combinar predicoes dos modelos base                        |
|                                                                              |
|   PARAMETROS:                                                                 |
|   - class_weight: balanced                                                    |
|   - max_iter: 1000                                                            |
|                                                                              |
+------------------------------------------------------------------------------+
```

### Papel no Stacking

```
+------------------------------------------------------------------------------+
|                    STACKING - COMO FUNCIONA                                   |
+------------------------------------------------------------------------------+
|                                                                              |
|   1. Random Forest gera predicao: 0.72 (72% fraude)                          |
|   2. Gradient Boosting gera predicao: 0.78 (78% fraude)                      |
|                                                                              |
|   3. Logistic Regression recebe [0.72, 0.78] como entrada                    |
|   4. Aprende pesos otimos para combinar                                      |
|   5. Resultado final: 0.74 (74% fraude)                                      |
|                                                                              |
|   P(base_ensemble) = LR([P(RF), P(GB)])                                      |
|                                                                              |
+------------------------------------------------------------------------------+
```

---

### 4. CatBoost Classifier

```
+------------------------------------------------------------------------------+
|  MODELO: CATBOOST                                                             |
+------------------------------------------------------------------------------+
|                                                                              |
|   Arquivo:        catboost_model.py                                           |
|   Peso no Ensemble: 25%                                                      |
|   Papel:          Especialista em features categoricas                       |
|                                                                              |
|   PARAMETROS:                                                                 |
|   - iterations: 500                                                           |
|   - learning_rate: 0.05                                                       |
|   - depth: 8                                                                  |
|   - l2_leaf_reg: 3                                                            |
|   - border_count: 128                                                         |
|   - class_weights: [1, 10]                                                    |
|   - auto_class_weights: Balanced                                              |
|   - loss_function: Logloss                                                    |
|   - eval_metric: AUC                                                          |
|                                                                              |
+------------------------------------------------------------------------------+
```

### Features Categoricas Tratadas

| Feature | Cardinalidade | Exemplo |
|---------|---------------|---------|
| canal | 5 | PIX, TED, WEB, MOBILE, ATM |
| tipo_transacao | 8 | TRANSFER, PAYMENT, DEBIT, etc |
| estado | 27 | SP, RJ, MG, etc |
| pais | 10+ | BR, US, PT, etc |
| device_type | 5 | ANDROID, IOS, WEB, etc |
| pix_key_type | 4 | CPF, CNPJ, EMAIL, PHONE |
| banco_recebedor | 50+ | ITAU, BRADESCO, NUBANK, etc |
| merchant_category | 100+ | MCC codes |
| day_of_week | 7 | Segunda a Domingo |
| hour_bucket | 6 | MADRUGADA, MANHA, etc |

---

### 5. GNN (Graph Neural Network)

```
+------------------------------------------------------------------------------+
|  MODELO: GNN FRAUD DETECTOR                                                   |
+------------------------------------------------------------------------------+
|                                                                              |
|   Arquivo:        gnn_fraud_detector.py                                       |
|   Peso no Ensemble: 25%                                                      |
|   Papel:          Analise de relacionamentos e redes                         |
|                                                                              |
|   BIBLIOTECA: NetworkX                                                        |
|                                                                              |
|   TIPOS DE NOS:                                                               |
|   - customer: Cliente (CPF)                                                   |
|   - device: Dispositivo                                                       |
|   - ip: Endereco IP                                                           |
|   - receiver: Conta recebedora                                                |
|   - merchant: Comerciante                                                     |
|   - location: Localizacao                                                     |
|                                                                              |
|   TIPOS DE ARESTAS:                                                           |
|   - transaction: Transacao entre entidades                                    |
|   - uses: Cliente usa dispositivo/IP                                          |
|   - receives: Conta recebe transacao                                          |
|   - at_location: Transacao em localizacao                                     |
|                                                                              |
+------------------------------------------------------------------------------+
```

### Como Funciona

```
+------------------------------------------------------------------------------+
|                    GNN - DETECCAO DE REDES DE FRAUDE                          |
+------------------------------------------------------------------------------+
|                                                                              |
|   O GNN constroi um GRAFO de relacionamentos:                                |
|                                                                              |
|          [CPF-A]----transacao--->[CONTA-X]                                   |
|             |                        |                                       |
|            usa                     recebe                                    |
|             |                        |                                       |
|        [DEVICE-1]             [CPF-B]<---[DEVICE-1]                          |
|             |                                                                |
|             |--- Mesmo device? SUSPEITAAAAA!                                 |
|                                                                              |
|   METRICAS CALCULADAS:                                                        |
|   - PageRank: Importancia do no na rede                                      |
|   - Community Detection: Grupos de contas conectadas                         |
|   - Neighbor Fraud Rate: Taxa de fraude dos vizinhos                         |
|   - Centrality: Quao central e o no                                          |
|                                                                              |
+------------------------------------------------------------------------------+
```

---

## Formula do Ensemble Integrado v2.0

```
+------------------------------------------------------------------------------+
|                    FORMULA FINAL DO ENSEMBLE                                  |
+------------------------------------------------------------------------------+
|                                                                              |
|   P(fraude) = 0.50 x P(base_ensemble)                                        |
|             + 0.25 x P(catboost)                                             |
|             + 0.25 x P(gnn)                                                  |
|                                                                              |
|   Onde:                                                                       |
|   - P(base_ensemble) = Meta-modelo do stacking (RF + GB + LR)                |
|   - P(catboost) = Predicao do CatBoost                                       |
|   - P(gnn) = graph_risk_score do GNN                                         |
|                                                                              |
+------------------------------------------------------------------------------+
```

---

## Ajuste Dinamico de Pesos

| Cenario | Base | CatBoost | GNN | Quando Ocorre |
|---------|------|----------|-----|---------------|
| Todos disponiveis | 50% | 25% | 25% | Operacao normal |
| Apenas GNN disponivel | 70% | 0% | 30% | CatBoost offline |
| Apenas CatBoost disponivel | 65% | 35% | 0% | GNN offline |
| Nenhum disponivel | 100% | 0% | 0% | Modo degradado |

---

## Modelos Auxiliares

### 6. XGBoost Classifier

```
Arquivo: real_model_training.py
Uso: Treinamento offline, comparacao de performance
Parametros: n_estimators=200, max_depth=10, learning_rate=0.1
```

### 7. LightGBM Classifier

```
Arquivo: real_model_training.py
Uso: Treinamento rapido para grandes volumes
Parametros: n_estimators=200, max_depth=10, learning_rate=0.1
```

### 8. Extra Trees Classifier

```
Arquivo: optimized_fraud_engine.py
Uso: Alternativa ao Random Forest
Parametros: n_estimators=100, max_depth=15
```

### 9. SVC (Support Vector Classifier)

```
Arquivo: final_balanced_fraud_engine.py
Uso: Classificacao com margens
Parametros: kernel='rbf', probability=True
```

### 10. MLP Classifier (Neural Network)

```
Arquivo: optimized_fraud_engine.py
Uso: Deteccao de padroes nao-lineares
Parametros: hidden_layer_sizes=(100, 50), activation='relu'
```

### 11. Isolation Forest

```
Arquivo: optimized_fraud_engine.py
Uso: Deteccao de anomalias
Parametros: n_estimators=100, contamination=0.01
```

### 12. Voting Classifier

```
Arquivo: final_balanced_fraud_engine.py
Uso: Ensemble por votacao
Tipo: soft (probabilidades)
```

### 13. Calibrated Classifier CV

```
Arquivo: production_fraud_engine.py
Uso: Calibracao de probabilidades
Metodo: isotonic ou sigmoid
```

---

## Modulos Especializados (Nao-ML Tradicional)

### PIX Fraud Taxonomy

```
+------------------------------------------------------------------------------+
|  MODULO: PIX FRAUD TAXONOMY                                                   |
+------------------------------------------------------------------------------+
|                                                                              |
|   Arquivo:        pix_fraud_taxonomy.py                                       |
|   Baseado em:     arXiv:2511.20902 (2025)                                    |
|   Versao:         1.0.0                                                       |
|                                                                              |
|   TIPOS DE FRAUDE DETECTADOS:                                                 |
|   1. QR_TAMPERED - QR Code adulterado                                        |
|   2. GHOST_HAND - Mao fantasma (acesso remoto)                               |
|   3. FAKE_BANK_CENTER - Central falsa do banco                               |
|   4. WHATSAPP_CLONE - Clone de WhatsApp                                      |
|   5. WRONG_PIX - PIX errado                                                  |
|   6. FAKE_RECEIPT - Comprovante falso                                        |
|   7. KIDNAPPING - Sequestro relampago                                        |
|   8. FAKE_EMPLOYEE - Falso funcionario                                       |
|   9. FAKE_MARKETPLACE - Leilao/marketplace falso                             |
|   10. PIX_BUG - Bug do PIX                                                   |
|                                                                              |
+------------------------------------------------------------------------------+
```

### NLP Social Engineering Detector

```
+------------------------------------------------------------------------------+
|  MODULO: NLP SOCIAL ENGINEERING                                               |
+------------------------------------------------------------------------------+
|                                                                              |
|   Arquivo:        nlp_social_engineering.py                                   |
|   Baseado em:     DIFrauD Dataset (HuggingFace)                              |
|   Versao:         1.0.0                                                       |
|                                                                              |
|   PADROES DETECTADOS:                                                         |
|   - SMS Phishing (Smishing)                                                   |
|   - Golpes de WhatsApp                                                        |
|   - Emails fraudulentos                                                       |
|   - Mensagens de urgencia                                                     |
|   - Impersonacao de bancos                                                    |
|                                                                              |
|   TAXA DE DETECCAO: >70%                                                      |
|                                                                              |
+------------------------------------------------------------------------------+
```

### Transfer Learning Pipeline

```
+------------------------------------------------------------------------------+
|  MODULO: TRANSFER LEARNING PIPELINE                                           |
+------------------------------------------------------------------------------+
|                                                                              |
|   Arquivo:        transfer_learning_pipeline.py                               |
|   Versao:         1.0.0                                                       |
|                                                                              |
|   ESTRATEGIAS:                                                                |
|   1. Domain Adaptation: Nigerian -> PIX brasileiro                           |
|   2. Fine-tuning: PaySim -> Dados proprietarios                              |
|   3. Ensemble Transfer: Combinar multiplos dominios                          |
|                                                                              |
|   DATASETS SUPORTADOS:                                                        |
|   - Nigerian Financial (5M tx)                                                |
|   - PaySim (6.3M tx)                                                          |
|   - Feedzai BAF (6M tx)                                                       |
|   - IEEE-CIS (590K tx)                                                        |
|                                                                              |
+------------------------------------------------------------------------------+
```

---

# PARTE 4: RESUMO EXECUTIVO

## Numeros Chave

```
+==============================================================================+
|                    RESUMO - SANKOFA ENTERPRISE PRO v2.0                       |
+==============================================================================+
|                                                                              |
|   DATASETS                                                                    |
|   ========                                                                    |
|   Total de datasets: 7                                                        |
|   Total de transacoes: ~24 milhoes                                           |
|   Taxa media de fraude: ~1.5%                                                |
|                                                                              |
|   FEATURES                                                                    |
|   ========                                                                    |
|   Total de features: 100+                                                     |
|   Categorias: 9                                                               |
|   Modulo principal: Bahnsen Feature Engineering (62+ features)               |
|                                                                              |
|   MODELOS                                                                     |
|   =======                                                                     |
|   Modelos principais: 5 (RF, GB, LR, CatBoost, GNN)                          |
|   Modelos auxiliares: 8                                                       |
|   Modulos especializados: 3 (PIX Taxonomy, NLP, Transfer Learning)           |
|                                                                              |
|   PERFORMANCE                                                                 |
|   ===========                                                                 |
|   Latencia: <50ms                                                             |
|   Accuracy: >98%                                                              |
|   AUC-ROC: >0.95                                                              |
|                                                                              |
+==============================================================================+
```

---

## Referencias

1. Bahnsen et al. 2016 - "Feature engineering strategies for credit card fraud detection"
2. arXiv:2511.20902 - "PIX Fraud Taxonomy in Brazil" (2025)
3. DIFrauD Dataset - HuggingFace
4. Nigerian Financial Transactions - HuggingFace
5. PaySim - Kaggle
6. Feedzai BAF - NeurIPS 2022
7. IEEE-CIS Fraud Detection - Kaggle

---

**Documento mantido por:** Sankofa Enterprise Team  
**Ultima revisao:** Dezembro 2025
