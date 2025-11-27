# DataSets de Fraude Financeira - Guia Completo

## Guia Didatico para Analistas e Desenvolvedores

**Fontes:** Kaggle, Hugging Face, GitHub (AIForge, Amazon Science, AI4Risk)  
**Ultima Atualizacao:** 27 de Novembro de 2025

---

## Resumo Executivo

```
+==============================================================================+
|                    MAPA COMPLETO DOS DATASETS                                 |
+==============================================================================+
|                                                                               |
|   KAGGLE (7 Datasets)                                                         |
|   ━━━━━━━━━━━━━━━━━━                                                          |
|   1. Credit Card Fraud (MLG-ULB)     284.807 transacoes reais                |
|   2. Credit Card Fraud 2023          550.000+ transacoes (NOVO!)             |
|   3. IEEE-CIS Fraud Detection        590.540 transacoes                      |
|   4. PaySim                          6.362.620 transacoes sinteticas         |
|   5. Sparkov Simulated               1.316.675 transacoes                    |
|   6. Fraud E-commerce                151.112 transacoes                      |
|   7. Financial Transactions 2024     Atualizado outubro 2024                 |
|                                                                               |
|   HUGGING FACE (4 Datasets)                                                   |
|   ━━━━━━━━━━━━━━━━━━━━━━                                                      |
|   1. CiferAI Fraud Detection         6 milhoes de transacoes                 |
|   2. Nigerian Financial Transactions 5 milhoes de transacoes                 |
|   3. Synthetic Financial Cleaned     6.36 milhoes de transacoes              |
|   4. kmasiak FraudDetection          532.909 transacoes                      |
|                                                                               |
|   GITHUB ESPECIALIZADOS (3 Repositorios)                                      |
|   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                           |
|   1. Amazon FDB (9 datasets)         Benchmark padrao da industria           |
|   2. AI4Risk AntiFraud               Modelos de grafos (GTAN, RGTAN)         |
|   3. AIForge Collection              172 recursos de fraude                  |
|                                                                               |
+==============================================================================+
```

---

## Parte 1: Datasets do Kaggle

### 1.1 Credit Card Fraud Detection (MLG-ULB) - O Classico

**URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

| Informacao | Valor |
|------------|-------|
| Total de Transacoes | 284.807 |
| Transacoes Fraudulentas | 492 (0,17%) |
| Transacoes Normais | 284.315 (99,83%) |
| Periodo | 2 dias (Setembro 2013) |
| Origem | Cartoes europeus |
| Tamanho | ~150 MB |

**Campos do Dataset:**
```
+==============================================================================+
|                    ESTRUTURA DO DATASET MLG-ULB                               |
+==============================================================================+
|                                                                               |
|  CAMPO    │ TIPO    │ DESCRICAO                                              |
|  ━━━━━━━━━│━━━━━━━━━│━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  |
|  Time     │ int     │ Segundos desde primeira transacao                      |
|  V1-V28   │ float   │ Features anonimizadas (PCA)                            |
|  Amount   │ float   │ Valor da transacao                                     |
|  Class    │ int     │ 0 = Normal, 1 = Fraude                                 |
|                                                                               |
|  NOTA: V1-V28 foram transformadas por PCA para proteger                      |
|  a privacidade dos clientes. Nao sabemos o que cada V representa!            |
|                                                                               |
+==============================================================================+
```

**Exemplo Pratico - Clonagem de Cartao de Credito:**

```
+==============================================================================+
|                    CASO: COMPRA FRAUDULENTA INTERNACIONAL                     |
+==============================================================================+
|                                                                               |
|  CLIENTE: Maria Silva, Sao Paulo                                              |
|  PADRAO NORMAL: Compras em supermercado, farmacia, postos de gasolina        |
|  VALOR MEDIO: R$ 80 - R$ 300                                                  |
|                                                                               |
|  TRANSACAO NORMAL (9h da manha, sabado):                                      |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │ Time: 45321                                                             │  |
|  │ Amount: R$ 156,00                                                       │  |
|  │ Local: Supermercado Zona Sul - SP                                       │  |
|  │ Class: 0 (APROVADA)                                                     │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
|  TRANSACAO FRAUDULENTA (9h03, mesmo dia - 3 minutos depois!):                 |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │ Time: 45500                                                             │  |
|  │ Amount: R$ 4.899,00                                                     │  |
|  │ Local: Best Buy - Miami, Florida                                        │  |
|  │ Class: 1 (FRAUDE!)                                                      │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
|  [!] ALERTAS DO SISTEMA:                                                      |
|  • Impossivel estar em SP e Miami em 3 minutos                               |
|  • Valor 31x maior que a media da cliente                                    |
|  • Primeira compra internacional (Maria nunca viajou)                        |
|  • Categoria de alto risco (eletronicos)                                     |
|                                                                               |
|  RESULTADO: Transacao BLOQUEADA pelo modelo ML                                |
|                                                                               |
+==============================================================================+
```

---

### 1.2 Credit Card Fraud Detection 2023 - O Mais Recente

**URL:** https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

| Informacao | Valor |
|------------|-------|
| Total de Transacoes | 568.630 |
| Periodo | Ano de 2023 |
| Origem | Cartoes europeus |
| Tamanho | ~325 MB |
| Atualizacao | Anual |

**Por Que Usar Este Dataset?**
- Dados MAIS RECENTES (2023 vs 2013 do classico)
- Padroes de fraude atualizados
- Mesma estrutura do classico (V1-V28)
- Ideal para treinar modelos modernos

---

### 1.3 IEEE-CIS Fraud Detection - O Mais Completo

**URL:** https://www.kaggle.com/competitions/ieee-fraud-detection

| Informacao | Valor |
|------------|-------|
| Total de Transacoes | 590.540 |
| Taxa de Fraude | 3,5% |
| Features | 393 originais (67 apos limpeza) |
| Fornecedor | Vesta Corporation |
| Tipo | Card-Not-Present (compras online) |

**Campos Principais:**
```
+==============================================================================+
|                    ESTRUTURA DO DATASET IEEE-CIS                              |
+==============================================================================+
|                                                                               |
|  GRUPO              │ CAMPOS                                                  |
|  ━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  |
|  Transacao          │ TransactionID, TransactionDT, TransactionAmt           |
|  Produto            │ ProductCD (W, H, C, S, R)                               |
|  Cartao             │ card1-card6 (tipo, categoria, banco)                    |
|  Endereco           │ addr1, addr2 (billing region)                           |
|  Email              │ P_emaildomain, R_emaildomain                            |
|  Contagem           │ C1-C14 (contadores de eventos)                          |
|  Valores            │ D1-D15 (diferenças temporais)                           |
|  Match              │ M1-M9 (correspondencias de dados)                       |
|  Vesta Features     │ V1-V339 (features proprietarias)                        |
|  Identidade         │ id_01-id_38 (device fingerprint)                        |
|  Device             │ DeviceType, DeviceInfo                                  |
|                                                                               |
+==============================================================================+
```

**Exemplo Pratico - Fraude em E-commerce:**

```
+==============================================================================+
|                    CASO: COMPRA ONLINE COM CARTAO ROUBADO                     |
+==============================================================================+
|                                                                               |
|  SITUACAO:                                                                    |
|  Fraudador obteve dados de cartao em vazamento de dados.                     |
|  Tenta comprar celular de R$ 5.000 em loja online.                           |
|                                                                               |
|  DADOS DA TRANSACAO:                                                          |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │ TransactionAmt: R$ 5.000,00                                             │  |
|  │ ProductCD: W (Wireless - celular)                                       │  |
|  │ card4: mastercard                                                       │  |
|  │ card6: credit                                                           │  |
|  │ P_emaildomain: gmail.com                                                │  |
|  │ R_emaildomain: gmail.com                                                │  |
|  │ DeviceType: mobile                                                      │  |
|  │ DeviceInfo: Samsung Galaxy A14                                          │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
|  SINAIS DE FRAUDE DETECTADOS:                                                 |
|  [!] D1 (dias desde ultima transacao): 0 - primeira compra!                  |
|  [!] C1 (contagem de enderecos): 5 - muitos enderecos diferentes            |
|  [!] M4 (match nome/cartao): 0 - nome nao confere                           |
|  [!] V12 (velocidade): muito alto - varias tentativas                       |
|  [!] id_31 (browser): navegador em modo privado                             |
|                                                                               |
|  SCORE DE RISCO: 94/100 → BLOQUEADA                                           |
|                                                                               |
+==============================================================================+
```

---

### 1.4 PaySim - O Melhor Para PIX

**URL:** https://www.kaggle.com/datasets/ealaxi/paysim1v

| Informacao | Valor |
|------------|-------|
| Total de Transacoes | 6.362.620 |
| Transacoes Fraudulentas | 8.213 (0,13%) |
| Tipos | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| Periodo Simulado | 30 dias |
| Tamanho | ~470 MB |

**Mapeamento Para o Sistema Brasileiro:**

```
+==============================================================================+
|                    PAYSIM → SISTEMA BRASILEIRO                                |
+==============================================================================+
|                                                                               |
|  PAYSIM          │ BRASIL           │ RISCO    │ % FRAUDES                   |
|  ━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━│━━━━━━━━━│━━━━━━━━━━━━━━━━━━━━━━━━━━━━ |
|  TRANSFER        │ PIX, TED         │ ALTO     │ 0,76% de todas as fraudes   |
|  CASH_OUT        │ Saque, Pix Saque │ ALTO     │ 0,18% de todas as fraudes   |
|  PAYMENT         │ Pix Pagamento    │ BAIXO    │ 0%                          |
|  CASH_IN         │ Deposito, TED In │ BAIXO    │ 0%                          |
|  DEBIT           │ Debito Auto      │ BAIXO    │ 0%                          |
|                                                                               |
|  CONCLUSAO: TRANSFER e CASH_OUT concentram 100% das fraudes!                 |
|                                                                               |
+==============================================================================+
```

**Exemplo Pratico - Golpe do PIX:**

```
+==============================================================================+
|                    CASO: GOLPE DO FALSO SEQUESTRO                             |
+==============================================================================+
|                                                                               |
|  CENARIO:                                                                     |
|  Criminosos ligam para Dona Fatima dizendo que sequestraram                  |
|  sua filha e pedem R$ 10.000 via PIX "ou ela morre".                         |
|                                                                               |
|  COMO O GOLPE APARECE NO DATASET:                                             |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │ step: 1 (primeira hora)                                                 │  |
|  │ type: TRANSFER                                                          │  |
|  │ amount: 10000.00                                                        │  |
|  │ nameOrig: C1234567890 (Dona Fatima)                                     │  |
|  │ oldbalanceOrg: 15000.00                                                 │  |
|  │ newbalanceOrig: 5000.00                                                 │  |
|  │ nameDest: C9876543210 (conta laranja)                                   │  |
|  │ oldbalanceDest: 0.00  ← conta recem criada!                             │  |
|  │ newbalanceDest: 0.00  ← dinheiro saiu IMEDIATAMENTE!                    │  |
|  │ isFraud: 1                                                              │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
|  ALERTAS QUE O SISTEMA DEVERIA GERAR:                                         |
|  [!] Conta destino criada ha menos de 24h                                    |
|  [!] Valor muito acima da media da cliente                                   |
|  [!] Horario incomum (fora do padrao)                                        |
|  [!] Dinheiro sacado imediatamente da conta destino                          |
|  [!] Primeira transferencia para este destinatario                           |
|                                                                               |
|  ACAO IDEAL: Travar por 30 min + ligar para confirmar                        |
|                                                                               |
+==============================================================================+
```

---

### 1.5 Amazon Fraud Dataset Benchmark (FDB)

**URL:** https://github.com/amazon-science/fraud-dataset-benchmark

O FDB e o **benchmark padrao da industria** desenvolvido pela Amazon. Contem 9 datasets diferentes!

```
+==============================================================================+
|                    AMAZON FDB - 9 DATASETS                                    |
+==============================================================================+
|                                                                               |
|  #  │ DATASET          │ CATEGORIA           │ TREINO    │ TAXA FRAUDE      |
|  ━━━│━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━│━━━━━━━━━━━━━━━━ |
|  1  │ IEEE-CIS         │ Cartao Online       │ 561.013   │ 3,50%            |
|  2  │ CCFraud          │ Cartao Online       │ 227.845   │ 0,18%            |
|  3  │ Fraud Ecommerce  │ Cartao Online       │ 120.889   │ 10,60%           |
|  4  │ Sparkov          │ Cartao Simulado     │ 1.296.675 │ 5,70%            |
|  5  │ Twitter Bots     │ Ataques de Bots     │ 29.950    │ 33,10%           |
|  6  │ Malicious URLs   │ Trafego Malicioso   │ 586.072   │ 34,20%           |
|  7  │ Fake Job         │ Moderacao Conteudo  │ 14.304    │ 4,70%            |
|  8  │ Vehicle Loan     │ Risco de Credito    │ 186.523   │ 21,60%           |
|  9  │ IP Blocklist     │ Trafego Malicioso   │ 172.000   │ 7%               |
|                                                                               |
+==============================================================================+
```

**Como Usar o FDB:**

```python
# Instalacao
pip install git+https://github.com/amazon-science/fraud-dataset-benchmark.git

# Uso
from fdb import FraudDatasetBenchmark

# Carregar dataset
fdb = FraudDatasetBenchmark(dataset_key="ieeecis")
X_train, X_test, y_train, y_test = fdb.get_splits()

# Avaliar modelo
metrics = fdb.evaluate(y_test, predictions)
print(metrics)  # AUC, F1, Precision, Recall
```

---

## Parte 2: Datasets do Hugging Face

### 2.1 CiferAI Fraud Detection - O Maior

**URL:** https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF

| Informacao | Valor |
|------------|-------|
| Total de Transacoes | 6.000.000 |
| Formato | CSV (Parquet disponivel) |
| Particoes | 4 x 1.5M cada |
| Uso | Federated Learning, ML |
| Modelo Pre-treinado | 99,93% de acuracia! |

**Por Que Usar?**
- Dataset ENORME (6 milhoes de transacoes)
- Modelo pre-treinado disponivel
- Otimizado para Federated Learning (privacidade)
- Baseado no PaySim

```python
# Como carregar
from datasets import load_dataset

dataset = load_dataset("CiferAI/Cifer-Fraud-Detection-Dataset-AF")
print(dataset['train'][0])
```

---

### 2.2 Nigerian Financial Transactions

**URL:** https://huggingface.co/datasets/electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset

| Informacao | Valor |
|------------|-------|
| Total de Transacoes | 5.000.000 |
| Contexto | Africa (Nigeria) |
| Formato | CSV, Parquet |

**Por Que e Importante?**
- Padroes de fraude DIFERENTES dos europeus
- Mobile money predominante (similar ao PIX)
- Util para sistemas que atendem mercados emergentes

---

### 2.3 Synthetic Financial Cleaned

**URL:** https://huggingface.co/datasets/purulalwani/Synthetic-Financial-Datasets-For-Fraud-Detection-Cleaned

| Informacao | Valor |
|------------|-------|
| Total de Transacoes | 6.360.000 |
| Formato | Parquet |
| Status | Limpo e pre-processado |

**Vantagem:**
- Dados ja limpos (sem valores nulos)
- Pronto para uso imediato
- Mesmo formato do PaySim

---

## Parte 3: Repositorios Especializados (GitHub)

### 3.1 AI4Risk AntiFraud - Modelos de Grafos

**URL:** https://github.com/AI4Risk/antifraud

O AntiFraud e um framework completo para deteccao de fraude usando **grafos de transacoes**.

**Datasets Incluidos:**

```
+==============================================================================+
|                    DATASETS DO ANTIFRAUD                                      |
+==============================================================================+
|                                                                               |
|  DATASET    │ DESCRICAO                          │ MELHOR MODELO  │ AUC      |
|  ━━━━━━━━━━│━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━│━━━━━━━━ |
|  YelpChi    │ Fraude em avaliacoes do Yelp       │ Grad           │ 0.9908   |
|  Amazon     │ Fraude em avaliacoes da Amazon     │ HOGRL          │ 0.9800   |
|  S-FFSD     │ Fraude bancaria semi-supervisionada│ RGTAN          │ 0.8461   |
|                                                                               |
+==============================================================================+
```

**Modelos Implementados:**

| Modelo | Publicacao | Ano | Descricao |
|--------|------------|-----|-----------|
| MCNN | ICONIP | 2016 | CNN para fraude de cartao |
| STAN | AAAI | 2020 | Atencao espaco-temporal |
| STAGN | TKDE | 2020 | GNN com atencao |
| GTAN | AAAI | 2023 | Semi-supervisionado |
| RGTAN | TKDE | 2025 | Risk-aware GNN |
| HOGRL | IJCAI | 2024 | High-order graphs |
| Grad | WWW | 2025 | Diffusion graphs |

**Estrutura do S-FFSD:**

```
+==============================================================================+
|                    CAMPOS DO S-FFSD                                           |
+==============================================================================+
|                                                                               |
|  CAMPO    │ TIPO    │ RANGE              │ DESCRICAO                         |
|  ━━━━━━━━│━━━━━━━━━│━━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  |
|  Time     │ int32   │ 0 a N              │ Ordem da transacao                |
|  Source   │ string  │ S0 a Sns           │ Remetente                         |
|  Target   │ string  │ T0 a Tnt           │ Destinatario                      |
|  Amount   │ float32 │ 0.00 a infinito    │ Valor                             |
|  Location │ string  │ L0 a Lnl           │ Local da transacao                |
|  Type     │ string  │ TP0 a TPnp         │ Tipo de transacao                 |
|  Labels   │ int32   │ 0, 1, 2            │ 0=normal, 1=fraude, 2=desconhecido|
|                                                                               |
+==============================================================================+
```

---

## Parte 4: Estatisticas Comparativas

```
+==============================================================================+
|                    COMPARACAO DE TODOS OS DATASETS                            |
+==============================================================================+
|                                                                               |
|  DATASET              │ TAMANHO    │ TAXA FRAUDE │ TIPO             │ GRATIS |
|  ━━━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━│━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━│━━━━━━ |
|  CiferAI              │ 6.000.000  │ ~0.1%       │ Sintetico        │ Sim    |
|  PaySim               │ 6.362.620  │ 0.13%       │ Sintetico        │ Sim    |
|  Synthetic Cleaned    │ 6.360.000  │ ~0.1%       │ Sintetico        │ Sim    |
|  Nigerian Financial   │ 5.000.000  │ Variavel    │ Sintetico        │ Sim    |
|  Sparkov              │ 1.316.675  │ 5.70%       │ Simulado         │ Sim    |
|  IEEE-CIS             │ 590.540    │ 3.50%       │ Real             │ Sim*   |
|  CC Fraud 2023        │ 568.630    │ ~0.2%       │ Real             │ Sim    |
|  CC Fraud (classico)  │ 284.807    │ 0.17%       │ Real             │ Sim    |
|  Vehicle Loan         │ 233.154    │ 21.60%      │ Real             │ Sim    |
|  IP Blocklist         │ 215.000    │ 7%          │ Real             │ Sim    |
|  Fraud Ecommerce      │ 151.112    │ 10.60%      │ Real             │ Sim    |
|                                                                               |
|  * IEEE-CIS requer aceitar termos da competicao no Kaggle                    |
|                                                                               |
+==============================================================================+
```

---

## Parte 5: Guia de Escolha do Dataset

```
+==============================================================================+
|                    QUAL DATASET USAR?                                         |
+==============================================================================+
|                                                                               |
|  VOCE QUER DETECTAR...          │ USE ESTES DATASETS                         |
|  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ |
|                                  │                                            |
|  Fraude de PIX                   │ PaySim, CiferAI, Nigerian Financial        |
|  (transferencias instantaneas)   │ Foco em type=TRANSFER                      |
|                                  │                                            |
|  Fraude de Cartao CREDITO        │ CC Fraud 2023, IEEE-CIS, CC Fraud classico |
|  (compras online/fisicas)        │ Dados reais europeus                       |
|                                  │                                            |
|  Fraude de Cartao DEBITO         │ PaySim (CASH_OUT), Sparkov                 |
|  (saques, compras debito)        │ Simula saques em ATM                       |
|                                  │                                            |
|  Fraude de TED/DOC               │ PaySim, CiferAI                            |
|  (transferencias tradicionais)   │ type=TRANSFER                              |
|                                  │                                            |
|  Lavagem de Dinheiro             │ S-FFSD, YelpChi, Amazon (grafos)           |
|  (cadeias de transacoes)         │ Modelos GTAN, RGTAN, HOGRL                 |
|                                  │                                            |
|  Risco de Credito                │ Vehicle Loan, Sparkov                      |
|  (emprestimos, financiamentos)   │ Taxa de default alta (21%)                 |
|                                  │                                            |
|  Bots e Ataques                  │ Twitter Bots, Malicious URLs               |
|  (automacao maliciosa)           │ Taxa de fraude ~33%                        |
|                                  │                                            |
+==============================================================================+
```

---

## Parte 6: Exemplo Pratico - Debito

```
+==============================================================================+
|                    CASO: CLONAGEM DE CARTAO DE DEBITO                         |
+==============================================================================+
|                                                                               |
|  CENARIO:                                                                     |
|  Criminoso instala "chupa-cabra" em caixa eletronico.                        |
|  Clona cartao de debito de Jose Carlos.                                       |
|                                                                               |
|  TRANSACAO NORMAL DO JOSE (sexta-feira, 18h):                                 |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │ type: CASH_OUT                                                          │  |
|  │ amount: R$ 500,00                                                       │  |
|  │ oldbalanceOrg: R$ 3.500,00                                              │  |
|  │ newbalanceOrig: R$ 3.000,00                                             │  |
|  │ Local: Bradesco Av. Paulista - SP                                       │  |
|  │ isFraud: 0 (normal)                                                     │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
|  TRANSACAO FRAUDULENTA (sabado, 3h da manha):                                 |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │ type: CASH_OUT                                                          │  |
|  │ amount: R$ 1.000,00                                                     │  |
|  │ oldbalanceOrg: R$ 3.000,00                                              │  |
|  │ newbalanceOrig: R$ 2.000,00                                             │  |
|  │ Local: Bradesco Shopping Aricanduva - SP (15km de distancia)            │  |
|  │ isFraud: 1 (FRAUDE!)                                                    │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
|  ALERTAS GERADOS:                                                             |
|  [!] Saque as 3h da manha (Jose nunca sacou nesse horario)                   |
|  [!] Caixa eletronico diferente do habitual                                  |
|  [!] Segundo saque em menos de 12 horas                                      |
|  [!] Valor dobrado em relacao ao padrao                                      |
|                                                                               |
|  ACAO: Bloquear cartao + SMS para Jose confirmar                              |
|                                                                               |
+==============================================================================+
```

---

## Parte 7: Como Integrar no Sankofa

### Codigo de Mapeamento Universal

```python
# Mapeamento universal para o Sankofa Enterprise Pro

MAPEAMENTO_CANAL = {
    # PaySim / CiferAI
    'TRANSFER': 'PIX',
    'CASH_OUT': 'DEBITO',
    'PAYMENT': 'CREDITO',
    'DEBIT': 'DEBITO',
    'CASH_IN': 'TED',
    
    # IEEE-CIS ProductCD
    'W': 'CREDITO',  # Wireless (celular)
    'H': 'CREDITO',  # Home
    'C': 'CREDITO',  # Clothing
    'S': 'CREDITO',  # Sports
    'R': 'CREDITO',  # Restaurant
}

def converter_para_sankofa(dataset_tipo, linha):
    """
    Converte qualquer dataset para o formato Sankofa.
    
    Args:
        dataset_tipo: 'paysim', 'ieeecis', 'ccfraud'
        linha: dicionario com dados da transacao
    
    Returns:
        dict no formato Sankofa
    """
    if dataset_tipo == 'paysim':
        return {
            'transaction_id': f'TXN-{linha["step"]}-{linha["nameOrig"]}',
            'amount': linha['amount'],
            'channel': MAPEAMENTO_CANAL.get(linha['type'], 'PIX'),
            'customer_id': linha['nameOrig'],
            'merchant_id': linha['nameDest'],
            'is_fraud': linha['isFraud'] == 1,
            'balance_before': linha['oldbalanceOrg'],
            'balance_after': linha['newbalanceOrig']
        }
    
    elif dataset_tipo == 'ieeecis':
        return {
            'transaction_id': str(linha['TransactionID']),
            'amount': linha['TransactionAmt'],
            'channel': MAPEAMENTO_CANAL.get(linha.get('ProductCD', 'W'), 'CREDITO'),
            'customer_id': str(linha.get('card1', 'unknown')),
            'is_fraud': linha['isFraud'] == 1,
            'device_type': linha.get('DeviceType', 'unknown'),
            'email_domain': linha.get('P_emaildomain', 'unknown')
        }
    
    elif dataset_tipo == 'ccfraud':
        return {
            'transaction_id': f'CC-{linha.name}',
            'amount': linha['Amount'],
            'channel': 'CREDITO',
            'time_seconds': linha['Time'],
            'is_fraud': linha['Class'] == 1
        }
    
    return None
```

---

## Parte 8: Proximos Passos

```
+==============================================================================+
|                    CHECKLIST DE IMPLEMENTACAO                                 |
+==============================================================================+
|                                                                               |
|  [ ] 1. Baixar PaySim do Kaggle (mais similar ao PIX brasileiro)             |
|  [ ] 2. Baixar CC Fraud 2023 (mais recente para cartao)                      |
|  [ ] 3. Configurar conta Kaggle CLI                                          |
|  [ ] 4. Usar script de mapeamento para converter dados                       |
|  [ ] 5. Treinar modelos separados para cada canal:                           |
|         - Modelo PIX (PaySim TRANSFER)                                       |
|         - Modelo Credito (IEEE-CIS + CC Fraud)                               |
|         - Modelo Debito (PaySim CASH_OUT)                                    |
|  [ ] 6. Validar performance com metricas do Sankofa                          |
|  [ ] 7. Ajustar thresholds por canal                                         |
|                                                                               |
|  DICA: Comece com 100.000 transacoes para testes rapidos!                    |
|                                                                               |
+==============================================================================+
```

---

## Referencias Completas

| Fonte | Dataset | URL | Estrelas |
|-------|---------|-----|----------|
| Kaggle | Credit Card Fraud 2023 | [Link](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) | 621 |
| Kaggle | IEEE-CIS | [Link](https://www.kaggle.com/competitions/ieee-fraud-detection) | - |
| Kaggle | PaySim | [Link](https://www.kaggle.com/datasets/ealaxi/paysim1v) | - |
| Kaggle | CC Fraud (classico) | [Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | - |
| Hugging Face | CiferAI | [Link](https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF) | - |
| Hugging Face | Nigerian Financial | [Link](https://huggingface.co/datasets/electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset) | - |
| GitHub | Amazon FDB | [Link](https://github.com/amazon-science/fraud-dataset-benchmark) | 226 |
| GitHub | AI4Risk AntiFraud | [Link](https://github.com/AI4Risk/antifraud) | 298 |
| GitHub | AIForge | [Link](https://github.com/FELIPEACASTRO/AIForge) | 3 |

---

## Glossario

| Termo | Significado |
|-------|-------------|
| AUC | Area Under Curve - metrica de performance (quanto maior, melhor) |
| PCA | Principal Component Analysis - tecnica de anonimizacao |
| GNN | Graph Neural Network - rede neural para grafos |
| Federated Learning | Treinamento distribuido preservando privacidade |
| Card-Not-Present | Transacao online sem cartao fisico |
| Conta Laranja | Conta usada para receber dinheiro de fraudes |

---

*Documento criado para o Sankofa Enterprise Pro v12.0*  
*Fontes: Kaggle, Hugging Face, GitHub (Amazon Science, AI4Risk, AIForge)*
