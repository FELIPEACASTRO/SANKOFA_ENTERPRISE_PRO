# DataSets de Fraude Financeira - Guia Pratico

## Guia Didatico para Analistas e Desenvolvedores

**Fonte:** Repositorio [FELIPEACASTRO/AIForge](https://github.com/FELIPEACASTRO/AIForge)  
**Ultima Atualizacao:** 27 de Novembro de 2025

---

## O Que Voce Vai Aprender

```
+==============================================================================+
|                         MAPA DOS DATASETS                                     |
+==============================================================================+
|                                                                               |
|   1. CREDIT CARD FRAUD       Fraude em cartao de credito                     |
|      (Kaggle Dataset)        284.807 transacoes reais                        |
|                                                                               |
|   2. PAYSIM                  Simulacao de transacoes moveis                  |
|      (Mobile Money)          PIX, transferencias, pagamentos                 |
|                                                                               |
|   3. S-FFSD                  Fraude bancaria semi-supervisionada             |
|      (Anti-Fraud)            Grafos de transacoes                            |
|                                                                               |
|   4. YELP/AMAZON             Fraude em avaliacoes                            |
|      (Review Fraud)          Deteccao de padroes                             |
|                                                                               |
+==============================================================================+
```

---

## 1. Credit Card Fraud Dataset (Kaggle)

### Descricao Simples

Este dataset contem transacoes REAIS de cartao de credito feitas por europeus em setembro de 2013. E o dataset mais usado para estudar fraudes de cartao.

### Numeros Importantes

| Informacao | Valor |
|------------|-------|
| Total de Transacoes | 284.807 |
| Transacoes Fraudulentas | 492 (0,17%) |
| Transacoes Normais | 284.315 (99,83%) |
| Periodo | 2 dias |

### Por Que E Desbalanceado?

```
+==============================================================================+
|                    VISUALIZANDO O DESBALANCEAMENTO                            |
+==============================================================================+
|                                                                               |
|  Imagine uma caixa com 1000 bolinhas:                                         |
|                                                                               |
|  🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢   |
|  🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢   |
|  🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢   |
|  🟢🟢🟢🟢🟢🟢🟢🟢🟢🔴🔴                                                   |
|                                                                               |
|  🟢 = Transacao normal (998 de 1000)                                         |
|  🔴 = Fraude (apenas 2 de 1000!)                                             |
|                                                                               |
|  Esse desbalanceamento e REAL no mundo bancario!                             |
|  Por isso usamos tecnicas especiais para treinar modelos.                    |
|                                                                               |
+==============================================================================+
```

### Exemplo Real do Dia a Dia - CREDITO

```
+==============================================================================+
|                    CASO PRATICO: CLONAGEM DE CARTAO                           |
+==============================================================================+
|                                                                               |
|  SITUACAO REAL:                                                               |
|  ━━━━━━━━━━━━━━                                                               |
|  Maria, cliente do banco, sempre faz compras no supermercado perto           |
|  de casa em Sao Paulo, geralmente aos sabados de manha.                       |
|                                                                               |
|  COMO APARECE NO DATASET:                                                     |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │ Time: 45321 (segundos desde inicio do dataset)                         │  |
|  │ V1-V28: Valores transformados por PCA (anonimizados)                   │  |
|  │ Amount: R$ 150.00                                                       │  |
|  │ Class: 0 (normal)                                                       │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
|  TRANSACAO NORMAL DA MARIA:                                                   |
|  • Sabado, 10h da manha                                                       |
|  • Supermercado Zona Sul - SP                                                 |
|  • R$ 150,00                                                                  |
|  • Padrao habitual = APROVADA                                                 |
|                                                                               |
|  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  |
|                                                                               |
|  E ENTAO... O CARTAO FOI CLONADO!                                             |
|                                                                               |
|  TRANSACAO FRAUDULENTA:                                                       |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │ Time: 45500 (3 minutos depois)                                         │  |
|  │ V1-V28: Valores muito diferentes do padrao                             │  |
|  │ Amount: R$ 2.500.00                                                     │  |
|  │ Class: 1 (FRAUDE!)                                                      │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
|  SINAIS DE ALERTA:                                                            |
|  [!] Localizacao: Miami, EUA (Maria nunca viajou!)                           |
|  [!] Valor: 16x maior que o normal                                           |
|  [!] Horario: 3 minutos apos compra em SP                                    |
|  [!] Tipo: Loja de eletronicos de luxo                                       |
|                                                                               |
|  RESULTADO: BLOQUEADA pelo sistema de ML!                                     |
|                                                                               |
+==============================================================================+
```

### Link do Dataset

- **URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Formato:** CSV
- **Tamanho:** ~150 MB

---

## 2. PaySim Dataset (Transacoes Moveis / PIX)

### Descricao Simples

Dataset SINTETICO que simula transacoes de dinheiro movel (como PIX no Brasil). Foi criado baseado em dados reais de uma empresa financeira africana.

### Numeros Importantes

| Informacao | Valor |
|------------|-------|
| Total de Transacoes | 6.362.620 |
| Transacoes Fraudulentas | 8.213 (0,13%) |
| Tipos de Transacao | 5 (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER) |
| Periodo Simulado | 30 dias |

### Tipos de Transacao

```
+==============================================================================+
|                    TIPOS DE TRANSACAO NO PAYSIM                               |
+==============================================================================+
|                                                                               |
|  CASH_IN (Deposito)                                                           |
|  ━━━━━━━━━━━━━━━━━                                                           |
|  Cliente deposita dinheiro na conta                                           |
|  Equivalente no Brasil: Deposito em conta, recarga                           |
|                                                                               |
|  CASH_OUT (Saque)                                                             |
|  ━━━━━━━━━━━━━━━                                                             |
|  Cliente saca dinheiro da conta                                               |
|  Equivalente no Brasil: Saque em ATM, "Pix Saque"                            |
|                                                                               |
|  TRANSFER (Transferencia)                 ← MAIOR RISCO DE FRAUDE!           |
|  ━━━━━━━━━━━━━━━━━━━━━━━━                                                     |
|  Envia dinheiro para outra conta                                              |
|  Equivalente no Brasil: PIX, TED, DOC                                        |
|                                                                               |
|  PAYMENT (Pagamento)                                                          |
|  ━━━━━━━━━━━━━━━━━━                                                          |
|  Paga conta de comerciante                                                    |
|  Equivalente no Brasil: Pagamento com PIX, boleto                            |
|                                                                               |
|  DEBIT (Debito)                                                               |
|  ━━━━━━━━━━━━━━                                                               |
|  Debito automatico ou compra                                                  |
|  Equivalente no Brasil: Cartao de debito                                     |
|                                                                               |
+==============================================================================+
```

### Exemplo Real do Dia a Dia - PIX

```
+==============================================================================+
|                    CASO PRATICO: GOLPE DO FALSO FUNCIONARIO                   |
+==============================================================================+
|                                                                               |
|  SITUACAO REAL:                                                               |
|  ━━━━━━━━━━━━━━                                                               |
|  Joao recebe uma ligacao de alguem se passando por funcionario do banco.     |
|  O golpista convence Joao a fazer um "PIX de teste" para "verificar          |
|  a seguranca da conta".                                                       |
|                                                                               |
|  COMO APARECE NO DATASET:                                                     |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │ step: 1 (primeira hora do mes)                                         │  |
|  │ type: TRANSFER                                                          │  |
|  │ amount: 5000.00                                                         │  |
|  │ nameOrig: C1234567890 (Joao)                                           │  |
|  │ oldbalanceOrg: 8000.00                                                  │  |
|  │ newbalanceOrig: 3000.00                                                 │  |
|  │ nameDest: C0987654321 (conta do golpista)                              │  |
|  │ oldbalanceDest: 0.00                                                    │  |
|  │ newbalanceDest: 0.00  ← ZEROU IMEDIATAMENTE!                           │  |
|  │ isFraud: 1                                                              │  |
|  │ isFlaggedFraud: 0 (sistema nao pegou)                                  │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
|  SINAIS DE ALERTA QUE O SISTEMA DEVERIA CAPTURAR:                             |
|  [!] Conta destino criada recentemente (oldbalanceDest = 0)                  |
|  [!] Valor alto para primeira transacao do dia                               |
|  [!] Dinheiro saiu imediatamente da conta destino (newbalanceDest = 0)       |
|  [!] Joao nunca transferiu para essa conta antes                             |
|                                                                               |
|  LICAO: O sistema deve aprender esses padroes!                                |
|                                                                               |
+==============================================================================+
```

### Link do Dataset

- **URL:** https://www.kaggle.com/datasets/ealaxi/paysim1v
- **Formato:** CSV
- **Tamanho:** ~470 MB

---

## 3. S-FFSD Dataset (Fraude Bancaria Semi-Supervisionada)

### Descricao Simples

Dataset criado pelo projeto AntiFraud para treinar modelos de deteccao de fraude usando **grafos** (redes de conexoes entre contas).

### Estrutura dos Dados

```
+==============================================================================+
|                    CAMPOS DO DATASET S-FFSD                                   |
+==============================================================================+
|                                                                               |
|  CAMPO      │ TIPO      │ DESCRICAO                                          |
|  ━━━━━━━━━━━│━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ |
|  Time       │ int       │ Ordem da transacao (0 a N)                         |
|  Source     │ string    │ Conta que enviou (S0, S1, S2...)                   |
|  Target     │ string    │ Conta que recebeu (T0, T1, T2...)                  |
|  Amount     │ float     │ Valor da transacao                                 |
|  Location   │ string    │ Local da transacao (L0, L1, L2...)                 |
|  Type       │ string    │ Tipo de transacao (TP0, TP1...)                    |
|  Labels     │ int       │ 0=Normal, 1=Fraude, 2=Nao rotulado                 |
|                                                                               |
+==============================================================================+
```

### O Que Sao Grafos de Transacoes?

```
+==============================================================================+
|                    ENTENDENDO GRAFOS DE FRAUDE                                |
+==============================================================================+
|                                                                               |
|  Imagine as contas como pontos e as transacoes como linhas:                  |
|                                                                               |
|          CONTA A                                                              |
|            |                                                                  |
|            | R$ 5.000                                                         |
|            v                                                                  |
|          CONTA B ──R$ 4.900──> CONTA C ──R$ 4.800──> CONTA D                 |
|                                   |                                           |
|                                   | R$ 4.700                                 |
|                                   v                                           |
|                                SAQUE ATM                                      |
|                                                                               |
|  PADRAO SUSPEITO DETECTADO:                                                   |
|  • Dinheiro passou por 4 contas em minutos                                   |
|  • Valores decrescentes (taxa de "lavagem")                                  |
|  • Terminou em saque (dinheiro vivo = difícil rastrear)                     |
|                                                                               |
|  Isso e um GRAFO! O modelo de ML analisa a REDE de conexoes.                 |
|                                                                               |
+==============================================================================+
```

### Exemplo Real do Dia a Dia - TRANSFERENCIA/TED

```
+==============================================================================+
|                    CASO PRATICO: LAVAGEM DE DINHEIRO                          |
+==============================================================================+
|                                                                               |
|  SITUACAO REAL:                                                               |
|  ━━━━━━━━━━━━━━                                                               |
|  Um criminoso recebe R$ 50.000 de origem ilicita e precisa                   |
|  "limpar" esse dinheiro passando por varias contas.                          |
|                                                                               |
|  COMO APARECE NO DATASET:                                                     |
|  ┌────────────────────────────────────────────────────────────────────────┐  |
|  │ Time: 1    Source: S45   Target: T12   Amount: 50000   Type: TP1       │  |
|  │ Time: 2    Source: S12   Target: T33   Amount: 49500   Type: TP1       │  |
|  │ Time: 3    Source: S33   Target: T77   Amount: 49000   Type: TP1       │  |
|  │ Time: 4    Source: S77   Target: T99   Amount: 48500   Type: TP2       │  |
|  │ Time: 5    Source: S99   (SAQUE)       Amount: 48000   Type: TP3       │  |
|  └────────────────────────────────────────────────────────────────────────┘  |
|                                                                               |
|  O QUE O MODELO DE GRAFO VE:                                                  |
|                                                                               |
|       S45 ──$50k──> S12 ──$49.5k──> S33 ──$49k──> S77 ──$48.5k──> S99       |
|                                                                    │         |
|                                                               SAQUE $48k     |
|                                                                               |
|  ALERTAS AUTOMATICOS:                                                         |
|  [!] Cadeia de 5 transacoes em sequencia rapida                              |
|  [!] Valores muito proximos (perda de ~R$500 por transacao)                  |
|  [!] Contas intermediarias sao "laranjas"                                    |
|  [!] Termina em saque de alto valor                                          |
|                                                                               |
+==============================================================================+
```

### Link do Dataset

- **URL:** https://github.com/AI4Risk/antifraud
- **Formato:** CSV (dentro do repositorio)
- **Uso:** Treinar modelos de deteccao de fraude baseados em grafos

---

## 4. Comparativo: Qual Dataset Usar?

```
+==============================================================================+
|                    ESCOLHENDO O DATASET CERTO                                 |
+==============================================================================+
|                                                                               |
|  VOCE QUER DETECTAR...        │ USE ESTE DATASET                             |
|  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ |
|                               │                                               |
|  Fraude de CARTAO DE CREDITO  │ Credit Card Fraud (Kaggle)                   |
|  Compras online, clonagem     │ 284.807 transacoes reais                     |
|                               │                                               |
|  Fraude em PIX/TRANSFERENCIA  │ PaySim                                       |
|  Golpes instantaneos          │ 6.3 milhoes de transacoes                    |
|                               │                                               |
|  LAVAGEM DE DINHEIRO          │ S-FFSD + YelpChi/Amazon                      |
|  Cadeias de transacoes        │ Analise de grafos                            |
|                               │                                               |
|  FRAUDE DE DEBITO             │ PaySim (CASH_OUT) +                          |
|  Saques, clonagem ATM         │ Credit Card Fraud (adaptado)                 |
|                               │                                               |
+==============================================================================+
```

---

## 5. Como Usar Esses Datasets no Sankofa

### Passo a Passo Simples

```
+==============================================================================+
|                    INTEGRANDO DATASETS NO SISTEMA                             |
+==============================================================================+
|                                                                               |
|  PASSO 1: BAIXAR O DATASET                                                    |
|  ━━━━━━━━━━━━━━━━━━━━━━━━━━                                                   |
|  • Acesse o link do Kaggle ou GitHub                                         |
|  • Faca download do arquivo CSV                                              |
|                                                                               |
|  PASSO 2: MAPEAR OS CAMPOS                                                    |
|  ━━━━━━━━━━━━━━━━━━━━━━━━━                                                    |
|  O Sankofa espera estes campos:                                               |
|  • transaction_id   → Criar um ID unico                                      |
|  • amount          → Amount do dataset                                       |
|  • channel         → Mapear Type para PIX/CREDITO/DEBITO/TED                 |
|  • timestamp       → Converter Time para datetime                            |
|  • is_fraud        → Class ou isFraud do dataset                             |
|                                                                               |
|  PASSO 3: TREINAR O MODELO                                                    |
|  ━━━━━━━━━━━━━━━━━━━━━━━━━                                                    |
|  Use o endpoint /api/calibration/train com os dados mapeados                 |
|                                                                               |
+==============================================================================+
```

### Exemplo de Mapeamento - PaySim para Sankofa

```python
# Mapeamento PaySim → Sankofa

MAPEAMENTO_TIPO = {
    'TRANSFER': 'PIX',       # Transferencia instantanea
    'CASH_OUT': 'DEBITO',    # Saque
    'PAYMENT': 'CREDITO',    # Pagamento
    'DEBIT': 'DEBITO',       # Debito automatico
    'CASH_IN': 'TED'         # Deposito/entrada
}

def converter_transacao(linha_paysim):
    return {
        'transaction_id': f'TXN-{linha_paysim["step"]}-{linha_paysim["nameOrig"]}',
        'amount': linha_paysim['amount'],
        'channel': MAPEAMENTO_TIPO[linha_paysim['type']],
        'customer_id': linha_paysim['nameOrig'],
        'is_fraud': linha_paysim['isFraud'] == 1
    }
```

---

## 6. Estatisticas de Fraude por Tipo

```
+==============================================================================+
|                    ESTATISTICAS DOS DATASETS                                  |
+==============================================================================+
|                                                                               |
|  CREDIT CARD FRAUD (Kaggle)                                                   |
|  ┌───────────────────────────────────────────────────────────────────────┐   |
|  │ Tipo          │ Qtd Transacoes  │ Taxa Fraude │ Valor Medio          │   |
|  ├───────────────┼─────────────────┼─────────────┼──────────────────────┤   |
|  │ CREDITO       │ 284.807         │ 0,17%       │ $88.35               │   |
|  │ (unico tipo)  │                 │             │                      │   |
|  └───────────────┴─────────────────┴─────────────┴──────────────────────┘   |
|                                                                               |
|  PAYSIM (Mobile Money)                                                        |
|  ┌───────────────────────────────────────────────────────────────────────┐   |
|  │ Tipo          │ Qtd Transacoes  │ Taxa Fraude │ Onde Acontece        │   |
|  ├───────────────┼─────────────────┼─────────────┼──────────────────────┤   |
|  │ TRANSFER      │ 532.909         │ 0,76%       │ Engenharia social    │   |
|  │ CASH_OUT      │ 2.237.500       │ 0,18%       │ Clonagem ATM         │   |
|  │ PAYMENT       │ 2.151.495       │ 0%          │ Risco baixo          │   |
|  │ CASH_IN       │ 1.399.284       │ 0%          │ Deposito seguro      │   |
|  │ DEBIT         │ 41.432          │ 0%          │ Debito automatico    │   |
|  └───────────────┴─────────────────┴─────────────┴──────────────────────┘   |
|                                                                               |
|  CONCLUSAO IMPORTANTE:                                                        |
|  • TRANSFER (PIX) e CASH_OUT (Debito) concentram 100% das fraudes!          |
|  • Pagamentos e depositos sao operacoes de baixo risco                       |
|                                                                               |
+==============================================================================+
```

---

## 7. Proximos Passos

```
+==============================================================================+
|                    O QUE FAZER AGORA?                                         |
+==============================================================================+
|                                                                               |
|  [ ] 1. Baixe o dataset PaySim do Kaggle                                     |
|  [ ] 2. Use o script de mapeamento para converter para formato Sankofa       |
|  [ ] 3. Treine o modelo com os novos dados                                   |
|  [ ] 4. Compare a performance com o modelo atual                             |
|  [ ] 5. Ajuste os thresholds conforme necessario                             |
|                                                                               |
|  DICA: Comece com um subset de 100.000 transacoes para testes rapidos!       |
|                                                                               |
+==============================================================================+
```

---

## Referencias

| Dataset | Fonte | Estrelas GitHub |
|---------|-------|-----------------|
| Credit Card Fraud | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | N/A |
| PaySim | [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1v) | N/A |
| AntiFraud (S-FFSD) | [GitHub](https://github.com/AI4Risk/antifraud) | 298 |
| Credit Card Detection App | [GitHub](https://github.com/Nneji123/Credit-Card-Fraud-Detection) | 56 |
| AIForge Collection | [GitHub](https://github.com/FELIPEACASTRO/AIForge) | 3 |

---

*Documento criado para o Sankofa Enterprise Pro v12.0*
