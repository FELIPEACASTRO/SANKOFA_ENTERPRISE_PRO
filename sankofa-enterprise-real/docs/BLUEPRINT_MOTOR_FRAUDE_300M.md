# BLUEPRINT COMPLETO — MOTOR DE IA/ML DETECÇÃO DE FRAUDES BANCÁRIAS

## 300 MILHÕES DE TRANSAÇÕES/DIA | AWS | ENTERPRISE-GRADE

**Versão:** 1.0.0  
**Data:** 27 de Novembro de 2025  
**Classificação:** Documento Estratégico  
**Elaborado por:** Conselho Global de Especialistas em Fraude Bancária, IA/ML e Arquitetura de Sistemas

---

# ÍNDICE

1. [Resumo Executivo](#1-resumo-executivo)
2. [Visão de Negócio e Objetivos](#2-visão-de-negócio-e-objetivos)
3. [Arquitetura Alvo AWS](#3-arquitetura-alvo-aws)
4. [Modelo de Dados & Feature Store](#4-modelo-de-dados--feature-store)
5. [Desenho do Motor de Fraude](#5-desenho-do-motor-de-fraude)
6. [Métricas e Painéis](#6-métricas-e-painéis)
7. [Backoffice React](#7-backoffice-react)
8. [Segurança, LGPD e Compliance](#8-segurança-lgpd-e-compliance)
9. [Observabilidade & SRE](#9-observabilidade--sre)
10. [Operações de Fraude & Playbooks](#10-operações-de-fraude--playbooks)
11. [MLOps & Governança de Modelos](#11-mlops--governança-de-modelos)
12. [RoadMap Completo](#12-roadmap-completo)
13. [Riscos & Mitigações](#13-riscos--mitigações)
14. [Stack Técnica Recomendada](#14-stack-técnica-recomendada)
15. [Evolução de Longo Prazo](#15-evolução-de-longo-prazo)

---

# 1. RESUMO EXECUTIVO

## 1.1 Contexto

O setor bancário brasileiro enfrenta perdas anuais de R$ 2,5 bilhões com fraudes, com crescimento de 165% em fraudes PIX desde 2021. Este Blueprint apresenta uma solução de IA/ML de classe mundial para detecção de fraudes em tempo real, projetada para processar **300 milhões de transações/dia** (≈3.472 TPS) com latência inferior a 50ms.

## 1.2 Proposta de Valor

| Dimensão | Valor Entregue |
|----------|----------------|
| **Redução de Perdas** | 40-60% das fraudes detectadas antes da efetivação |
| **Falsos Positivos** | Redução de 35% comparado a sistemas baseados em regras |
| **Latência** | p99 < 50ms para decisão em tempo real |
| **Explicabilidade** | 100% das decisões auditáveis (LGPD/BACEN) |
| **ROI Estimado** | 8:1 em 18 meses |

## 1.3 Decisões Arquiteturais Chave

1. **Híbrido ML + Regras**: Ensemble de modelos (XGBoost + LightGBM + GNN) com camada de regras de negócio
2. **Feature Store Real-Time**: Redis Cluster + Apache Flink para janelas de 5min a 30 dias
3. **Multi-AZ AWS**: Arquitetura distribuída em 3 zonas de disponibilidade
4. **ONNX Runtime**: Inferência otimizada com latência < 5ms por modelo
5. **Event-Driven**: Apache Kafka (MSK) como backbone de eventos

## 1.4 Investimento Estimado

| Fase | Duração | Investimento |
|------|---------|--------------|
| MVP (Fase 1) | 90 dias | R$ 2.5M |
| Produção (Fase 2) | 180 dias | R$ 4.5M |
| Escala Total (Fase 3) | 365 dias | R$ 3.0M |
| **TOTAL** | 12 meses | **R$ 10M** |

---

# 2. VISÃO DE NEGÓCIO E OBJETIVOS

## 2.1 Objetivos Estratégicos

### Curto Prazo (0-90 dias)
- Reduzir perda bruta de fraude em 25%
- Implementar detecção real-time para PIX
- Estabelecer baseline de métricas

### Médio Prazo (90-180 dias)
- Expandir para Crédito, Débito e TED
- Atingir Recall de 85% com FPR < 2%
- Implementar STEP_UP inteligente

### Longo Prazo (180-365 dias)
- Atingir Recall de 92% com FPR < 1.5%
- Zero manual review para 95% das transações
- Modelo adaptativo anti-evasão

## 2.2 Métricas de Negócio (North Stars)

```
┌─────────────────────────────────────────────────────────────────┐
│                    MÉTRICAS NORTH STAR                          │
├─────────────────────────────────────────────────────────────────┤
│  $Recall (Capture Rate em R$)     │  Target: 85% → 92%         │
│  $Precision                        │  Target: 70% → 80%         │
│  Taxa de Falso Positivo (FPR)     │  Target: < 2% → < 1.5%     │
│  Latência p99                      │  Target: < 50ms            │
│  Taxa de Abandono pós-STEP_UP     │  Target: < 15%             │
│  ROI de Prevenção                  │  Target: 8:1               │
└─────────────────────────────────────────────────────────────────┘
```

## 2.3 Segmentação de Transações

| Segmento | Volume Diário | % Total | Risco Médio | Prioridade |
|----------|---------------|---------|-------------|------------|
| PIX | 180M | 60% | Alto | P0 |
| Débito | 60M | 20% | Médio | P1 |
| Crédito | 45M | 15% | Alto | P1 |
| TED/DOC | 15M | 5% | Baixo | P2 |

## 2.4 Tipologias de Fraude Priorizadas

```
PRIORIDADE P0 (90 dias):
├── Account Takeover (ATO) via PIX
├── Fraude de Primeira Parte (autofraude)
├── Phishing/Engenharia Social
└── Device Spoofing

PRIORIDADE P1 (180 dias):
├── Card Not Present (CNP)
├── Fraude de Cartão Clonado
├── Lavagem de Dinheiro (AML)
└── Mule Accounts

PRIORIDADE P2 (365 dias):
├── Fraude de Identidade Sintética
├── Bust-out Fraud
├── Insider Threat
└── Fraude Coordenada (Ring Fraud)
```

## 2.5 Stakeholders e Responsabilidades

| Stakeholder | Responsabilidade | Entrega |
|-------------|------------------|---------|
| CRO | Aprovação de políticas | Limiares de risco |
| Head de Fraudes | Estratégia operacional | Playbooks |
| CDO | Qualidade de dados | Feature Store |
| CTO | Arquitetura técnica | Infraestrutura |
| CISO | Segurança | Compliance |
| CPO | Experiência do usuário | Backoffice |

---

# 3. ARQUITETURA ALVO AWS

## 3.1 Visão Geral da Arquitetura

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           ARQUITETURA MOTOR DE FRAUDE v1.0                      │
│                              300M TRANSAÇÕES/DIA | AWS                          │
└─────────────────────────────────────────────────────────────────────────────────┘

                                    ┌─────────────┐
                                    │   CLIENTES  │
                                    │ (Mobile/Web)│
                                    └──────┬──────┘
                                           │
                                    ┌──────▼──────┐
                                    │ CloudFront  │
                                    │    + WAF    │
                                    └──────┬──────┘
                                           │
┌──────────────────────────────────────────┼──────────────────────────────────────┐
│                              VPC PRINCIPAL                                       │
│  ┌───────────────────────────────────────┼───────────────────────────────────┐  │
│  │                         CAMADA DE INGESTÃO                                 │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │  │
│  │  │ API Gateway │    │    NLB      │    │   Kong      │                    │  │
│  │  │  (REST)     │    │ (gRPC/TCP)  │    │ (Rate Limit)│                    │  │
│  │  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                    │  │
│  │         │                  │                  │                            │  │
│  │         └──────────────────┼──────────────────┘                            │  │
│  │                            │                                                │  │
│  │                     ┌──────▼──────┐                                        │  │
│  │                     │   MSK/Kafka │ ◄──── Backbone de Eventos              │  │
│  │                     │  (3 AZs)    │       - transactions-raw               │  │
│  │                     └──────┬──────┘       - transactions-scored            │  │
│  │                            │              - fraud-alerts                   │  │
│  └────────────────────────────┼───────────────────────────────────────────────┘  │
│                               │                                                   │
│  ┌────────────────────────────┼───────────────────────────────────────────────┐  │
│  │                    CAMADA DE PROCESSAMENTO                                  │  │
│  │                            │                                                │  │
│  │    ┌───────────────────────┼───────────────────────────────────┐           │  │
│  │    │                       │                                    │           │  │
│  │    ▼                       ▼                                    ▼           │  │
│  │  ┌─────────────┐    ┌─────────────┐                     ┌─────────────┐    │  │
│  │  │   Flink     │    │   EKS       │                     │  Lambda     │    │  │
│  │  │ (Streaming) │    │ (Scoring)   │                     │ (Enrichment)│    │  │
│  │  │             │    │             │                     │             │    │  │
│  │  │ - Features  │    │ - Java 21   │                     │ - IP Intel  │    │  │
│  │  │ - Agregação │    │ - ONNX RT   │                     │ - Device FP │    │  │
│  │  │ - Janelas   │    │ - gRPC      │                     │ - Geo       │    │  │
│  │  └──────┬──────┘    └──────┬──────┘                     └──────┬──────┘    │  │
│  │         │                  │                                   │           │  │
│  │         └──────────────────┴───────────────────────────────────┘           │  │
│  │                            │                                                │  │
│  └────────────────────────────┼───────────────────────────────────────────────┘  │
│                               │                                                   │
│  ┌────────────────────────────┼───────────────────────────────────────────────┐  │
│  │                    CAMADA DE DADOS                                          │  │
│  │                            │                                                │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │  │
│  │  │   Redis     │    │  Aurora     │    │ DynamoDB    │    │ OpenSearch  │  │  │
│  │  │  Cluster    │    │ PostgreSQL  │    │             │    │             │  │  │
│  │  │             │    │             │    │             │    │             │  │  │
│  │  │ - Features  │    │ - Histórico │    │ - Hot Data  │    │ - Logs      │  │  │
│  │  │ - Cache     │    │ - Audit     │    │ - Sessions  │    │ - Search    │  │  │
│  │  │ - Sessões   │    │ - Config    │    │ - Rules     │    │ - Analytics │  │  │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │  │
│  │                                                                             │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │  │
│  │  │                          S3 DATA LAKE                                │   │  │
│  │  │  ├── raw/transactions/                                               │   │  │
│  │  │  ├── processed/features/                                             │   │  │
│  │  │  ├── models/production/                                              │   │  │
│  │  │  ├── models/challenger/                                              │   │  │
│  │  │  └── audit/decisions/                                                │   │  │
│  │  └─────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                    CAMADA DE OBSERVABILIDADE                                │  │
│  │                                                                             │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │  │
│  │  │  DataDog    │    │ CloudWatch  │    │  X-Ray      │    │  PagerDuty  │  │  │
│  │  │  APM/Logs   │    │  Metrics    │    │  Traces     │    │  Alerting   │  │  │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │  │
│  │                                                                             │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

## 3.2 Componentes Detalhados

### 3.2.1 Camada de Ingestão

| Componente | Configuração | Justificativa |
|------------|--------------|---------------|
| **API Gateway** | REST + WebSocket, 10K RPS | Padrão bancário, mTLS |
| **Network Load Balancer** | TCP/gRPC, Cross-AZ | Latência mínima |
| **Kong** | Rate limiting, Auth | Proteção contra abuso |
| **MSK (Kafka)** | 3 brokers, 6 partições/topic | Throughput 300M/dia |

### 3.2.2 Camada de Processamento

| Componente | Configuração | Responsabilidade |
|------------|--------------|------------------|
| **Apache Flink** | 12 Task Managers, 48 slots | Feature engineering streaming |
| **EKS** | 20 nodes c6i.4xlarge | Model serving |
| **Lambda** | 1000 concurrency | Enrichments externos |

### 3.2.3 Camada de Dados

| Componente | Configuração | Uso |
|------------|--------------|-----|
| **Redis Cluster** | 6 shards, 3 replicas | Feature Store hot |
| **Aurora PostgreSQL** | db.r6g.4xlarge, Multi-AZ | Dados históricos |
| **DynamoDB** | On-demand, DAX cache | Hot data, regras |
| **OpenSearch** | 3 nodes m6g.2xlarge | Logs, busca |
| **S3** | Intelligent Tiering | Data Lake |

## 3.3 Fluxo de Dados Real-Time

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FLUXO DE SCORING REAL-TIME                           │
│                           Latência Alvo: < 50ms p99                         │
└─────────────────────────────────────────────────────────────────────────────┘

    ENTRADA                                                           SAÍDA
       │                                                                │
       ▼                                                                ▼
  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
  │ Receber │──▶│ Validar │──▶│  Enrich │──▶│ Feature │──▶│  Score  │──▶│ Decidir │
  │ Request │   │ Payload │   │  Data   │   │ Compute │   │   ML    │   │ & Resp  │
  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
      2ms          3ms           8ms          12ms          15ms          10ms
                                                                    ─────────────
                                                                    TOTAL: ~50ms

DETALHAMENTO POR ETAPA:

1. RECEBER REQUEST (2ms)
   ├── Parse JSON
   ├── Validação de schema
   └── Geração de request_id

2. VALIDAR PAYLOAD (3ms)
   ├── CPF válido
   ├── Valor > 0
   ├── Campos obrigatórios
   └── Sanitização

3. ENRICH DATA (8ms) - Paralelo
   ├── IP Intelligence (MaxMind)
   ├── Device Fingerprint
   ├── Geolocalização
   └── Horário/Fuso

4. FEATURE COMPUTE (12ms) - Redis + Flink
   ├── Velocidade transacional (5min/1h/24h)
   ├── Padrões de comportamento
   ├── Relacionamentos conta-device
   └── Anomalias históricas

5. SCORE ML (15ms) - ONNX Runtime
   ├── XGBoost: 3ms
   ├── LightGBM: 2ms
   ├── GNN (se aplicável): 8ms
   └── Ensemble: 2ms

6. DECIDIR & RESPONDER (10ms)
   ├── Aplicar regras de negócio
   ├── Determinar ação (APPROVE/STEP_UP/BLOCK)
   ├── Gerar explicação SHAP
   └── Retornar resposta
```

## 3.4 Dimensionamento e Capacidade

### 3.4.1 Cálculo de Capacidade

```
Premissas:
- 300M transações/dia
- Pico: 2.5x média (hora do almoço/final do mês)
- TPS médio: 3,472
- TPS pico: 8,680
- Latência target p99: 50ms

Dimensionamento EKS (Scoring):
- Request/pod: 200 TPS
- Pods necessários pico: 8,680 / 200 = 44 pods
- Fator de segurança 1.5x: 66 pods
- Pods por node (c6i.4xlarge): 4
- Nodes: 17 (arredondado para 20)

Dimensionamento Redis (Feature Store):
- Features por transação: 150
- Tamanho médio feature: 100 bytes
- Memória por transação: 15KB
- Transações em cache (24h): 300M
- Memória total: 4.5TB
- Shards: 6 (750GB/shard)

Dimensionamento Kafka (MSK):
- Mensagens/seg: 8,680 (pico)
- Tamanho mensagem: 2KB
- Throughput: 17.4 MB/s
- Retenção: 7 dias
- Storage: 10TB
- Partições: 6 por topic
```

### 3.4.2 Custos Estimados (AWS)

| Serviço | Configuração | Custo Mensal (USD) |
|---------|--------------|-------------------|
| EKS | 20 nodes c6i.4xlarge | $28,800 |
| MSK | 3 brokers kafka.m5.2xlarge | $4,320 |
| Redis | 6 shards r6g.2xlarge | $8,640 |
| Aurora | Multi-AZ r6g.4xlarge | $5,760 |
| Flink (EMR) | 12 nodes m5.2xlarge | $6,912 |
| S3 | 100TB + requests | $2,500 |
| DataDog | APM + Logs | $8,000 |
| Outros | Transfer, Lambda, etc | $5,000 |
| **TOTAL** | | **$69,932/mês** |

---

# 4. MODELO DE DADOS & FEATURE STORE

## 4.1 Schema de Transação (Entrada)

```json
{
  "transaction": {
    "id": "TXN_985760735407",
    "amount": 45.50,
    "currency": "BRL",
    "type": "PIX",
    "channel": "MOBILE_APP",
    "timestamp": "2025-08-18T12:45:00.000Z",
    "metadata": {
      "mcc": "5411",
      "merchant_id": "MERCH_123456",
      "merchant_name": "Supermercado XYZ",
      "terminal_id": "POS_789"
    }
  },
  "sender": {
    "account_id": "ACC_987654321",
    "cpf_hash": "a1b2c3d4e5f6...",
    "account_age_days": 365,
    "segment": "PREMIUM"
  },
  "receiver": {
    "account_id": "ACC_123456789",
    "cpf_hash": "f6e5d4c3b2a1...",
    "pix_key_type": "CPF",
    "is_first_transfer": false
  },
  "device": {
    "id": "DEV_abc123xyz",
    "fingerprint": "fp_hash_12345",
    "type": "ANDROID",
    "os_version": "14.0",
    "app_version": "5.2.1",
    "is_rooted": false,
    "is_emulator": false
  },
  "network": {
    "ip_address": "189.103.45.67",
    "ip_type": "RESIDENTIAL",
    "isp": "Vivo",
    "country": "BR",
    "region": "SP",
    "city": "São Paulo"
  },
  "location": {
    "latitude": -23.5505,
    "longitude": -46.6333,
    "accuracy_meters": 10,
    "source": "GPS"
  }
}
```

## 4.2 Arquitetura Feature Store

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FEATURE STORE ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │         ONLINE STORE (Redis)        │
                    │         Latência < 5ms              │
                    │                                     │
                    │  ┌─────────────────────────────┐   │
                    │  │ Window 5min  │ Window 1h    │   │
                    │  │ tx_count_5m  │ tx_count_1h  │   │
                    │  │ amount_sum   │ amount_sum   │   │
                    │  │ unique_rcv   │ unique_rcv   │   │
                    │  └─────────────────────────────┘   │
                    │  ┌─────────────────────────────┐   │
                    │  │ Window 24h   │ Window 7d    │   │
                    │  │ tx_count_24h │ tx_count_7d  │   │
                    │  │ avg_amount   │ avg_amount   │   │
                    │  │ std_amount   │ max_amount   │   │
                    │  └─────────────────────────────┘   │
                    │  ┌─────────────────────────────┐   │
                    │  │ Window 30d   │ Lifetime     │   │
                    │  │ tx_count_30d │ total_tx     │   │
                    │  │ fraud_rate   │ first_tx_dt  │   │
                    │  │ chargeback   │ risk_score   │   │
                    │  └─────────────────────────────┘   │
                    └─────────────────────────────────────┘
                                       ▲
                                       │ Sync
                    ┌─────────────────────────────────────┐
                    │       STREAMING LAYER (Flink)       │
                    │                                     │
                    │  - Agregações em tempo real         │
                    │  - Janelas deslizantes              │
                    │  - Exactly-once semantics           │
                    │  - Checkpointing S3                 │
                    └─────────────────────────────────────┘
                                       ▲
                                       │
                    ┌─────────────────────────────────────┐
                    │        OFFLINE STORE (S3/Delta)     │
                    │         Para treinamento            │
                    │                                     │
                    │  - Histórico completo               │
                    │  - Features batch                   │
                    │  - Backfill                         │
                    │  - Point-in-time joins              │
                    └─────────────────────────────────────┘
```

## 4.3 Catálogo de Features (150+ features)

### 4.3.1 Features de Velocidade

| Feature | Janela | Descrição | Importância |
|---------|--------|-----------|-------------|
| `tx_count_5min` | 5 min | Transações nos últimos 5 min | Alta |
| `tx_count_1h` | 1 hora | Transações na última hora | Alta |
| `tx_count_24h` | 24 horas | Transações nas últimas 24h | Média |
| `amount_sum_5min` | 5 min | Soma de valores em 5 min | Alta |
| `amount_sum_1h` | 1 hora | Soma de valores em 1 hora | Alta |
| `unique_receivers_1h` | 1 hora | Destinatários únicos | Alta |
| `unique_devices_24h` | 24 horas | Dispositivos únicos | Alta |
| `velocity_ratio` | 5min/30d | Taxa atual vs histórico | Alta |

### 4.3.2 Features de Comportamento

| Feature | Descrição | Importância |
|---------|-----------|-------------|
| `hour_of_day` | Hora do dia (0-23) | Média |
| `day_of_week` | Dia da semana (0-6) | Média |
| `is_weekend` | Final de semana | Baixa |
| `is_night` | Horário noturno (22h-6h) | Alta |
| `amount_vs_avg_30d` | Valor vs média 30 dias | Alta |
| `amount_percentile` | Percentil do valor | Alta |
| `time_since_last_tx` | Tempo desde última transação | Média |
| `typical_hour_deviation` | Desvio do horário típico | Média |

### 4.3.3 Features de Relacionamento (Graph)

| Feature | Descrição | Importância |
|---------|-----------|-------------|
| `receiver_degree` | Grau do nó receptor | Alta |
| `common_neighbors` | Vizinhos em comum | Alta |
| `community_fraud_rate` | Taxa de fraude da comunidade | Alta |
| `path_to_known_fraud` | Distância para fraude conhecida | Alta |
| `device_shared_accounts` | Contas compartilhando device | Alta |
| `ip_shared_accounts` | Contas compartilhando IP | Alta |
| `mule_score` | Score de conta laranja | Alta |

### 4.3.4 Features de Device/Network

| Feature | Descrição | Importância |
|---------|-----------|-------------|
| `device_age_days` | Idade do dispositivo | Média |
| `device_tx_count_30d` | Transações do device em 30d | Alta |
| `device_fraud_rate` | Taxa de fraude do device | Alta |
| `ip_risk_score` | Score de risco do IP | Alta |
| `ip_type` | Tipo de IP (res/mob/vpn/proxy) | Alta |
| `geo_distance_km` | Distância da última transação | Alta |
| `impossible_travel` | Viagem impossível detectada | Alta |
| `is_new_device` | Dispositivo novo | Alta |

### 4.3.5 Features Derivadas (Engineered)

| Feature | Fórmula | Importância |
|---------|---------|-------------|
| `amount_zscore` | (amount - mean) / std | Alta |
| `velocity_anomaly` | tx_5min / avg_tx_5min_30d | Alta |
| `receiver_novelty` | 1 se nunca transferiu antes | Alta |
| `pattern_break` | Desvio do padrão comportamental | Alta |
| `risk_composite` | Média ponderada de riscos | Alta |
| `fraud_neighborhood_density` | Densidade de fraude no grafo | Alta |

## 4.4 Schema do Banco de Dados (Aurora PostgreSQL)

```sql
-- Schema: fraud_detection

-- Tabela principal de transações
CREATE TABLE transactions (
    id VARCHAR(50) PRIMARY KEY,
    transaction_id VARCHAR(100) UNIQUE NOT NULL,
    amount DECIMAL(15, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'BRL',
    transaction_type VARCHAR(20) NOT NULL,
    channel VARCHAR(20) NOT NULL,
    
    -- Remetente
    sender_account_id VARCHAR(50) NOT NULL,
    sender_cpf_hash VARCHAR(64) NOT NULL,
    
    -- Destinatário
    receiver_account_id VARCHAR(50),
    receiver_pix_key VARCHAR(100),
    receiver_pix_key_type VARCHAR(20),
    
    -- Device
    device_id VARCHAR(100),
    device_fingerprint VARCHAR(100),
    device_type VARCHAR(20),
    
    -- Network
    ip_address VARCHAR(45),
    ip_country VARCHAR(2),
    ip_region VARCHAR(50),
    ip_city VARCHAR(100),
    
    -- Location
    latitude DECIMAL(10, 7),
    longitude DECIMAL(10, 7),
    
    -- Scoring
    fraud_score DECIMAL(5, 4),
    risk_level VARCHAR(20),
    decision VARCHAR(20) NOT NULL,
    decision_reason JSONB,
    model_version VARCHAR(20),
    
    -- Timestamps
    transaction_timestamp TIMESTAMPTZ NOT NULL,
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Índices
    INDEX idx_sender_account (sender_account_id),
    INDEX idx_transaction_timestamp (transaction_timestamp),
    INDEX idx_fraud_score (fraud_score),
    INDEX idx_decision (decision)
) PARTITION BY RANGE (transaction_timestamp);

-- Tabela de alertas
CREATE TABLE fraud_alerts (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(50) UNIQUE NOT NULL,
    transaction_id VARCHAR(100) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'NEW',
    
    -- Detalhes
    fraud_score DECIMAL(5, 4),
    amount DECIMAL(15, 2),
    description TEXT,
    recommended_action VARCHAR(50),
    
    -- Investigação
    assigned_to VARCHAR(100),
    investigation_notes TEXT,
    resolution VARCHAR(50),
    resolved_at TIMESTAMPTZ,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
);

-- Tabela de auditoria (LGPD/BACEN)
CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    action VARCHAR(50) NOT NULL,
    actor_id VARCHAR(100),
    actor_type VARCHAR(20),
    actor_ip VARCHAR(45),
    
    -- Dados
    old_value JSONB,
    new_value JSONB,
    
    -- Timestamp imutável
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Índices
    INDEX idx_entity (entity_type, entity_id),
    INDEX idx_actor (actor_id),
    INDEX idx_created_at (created_at)
);

-- Tabela de métricas do modelo
CREATE TABLE model_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(20) NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    environment VARCHAR(20) NOT NULL,
    
    -- Métricas de classificação
    auc_pr DECIMAL(5, 4),
    auc_roc DECIMAL(5, 4),
    precision_score DECIMAL(5, 4),
    recall_score DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    
    -- Métricas de negócio
    dollar_precision DECIMAL(5, 4),
    dollar_recall DECIMAL(5, 4),
    expected_value DECIMAL(15, 2),
    
    -- Drift
    psi_score DECIMAL(5, 4),
    ks_statistic DECIMAL(5, 4),
    
    -- Metadata
    sample_size INTEGER,
    evaluated_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_model_version (model_version),
    INDEX idx_evaluated_at (evaluated_at)
);

-- Tabela de configuração de regras
CREATE TABLE fraud_rules (
    id SERIAL PRIMARY KEY,
    rule_id VARCHAR(50) UNIQUE NOT NULL,
    rule_name VARCHAR(100) NOT NULL,
    rule_type VARCHAR(20) NOT NULL,
    priority INTEGER NOT NULL,
    
    -- Condições
    conditions JSONB NOT NULL,
    action VARCHAR(20) NOT NULL,
    
    -- Metadata
    is_active BOOLEAN DEFAULT true,
    created_by VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_rule_type (rule_type),
    INDEX idx_priority (priority)
);

-- Tabela de feedback (Human-in-the-Loop)
CREATE TABLE fraud_feedback (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(100) NOT NULL,
    original_decision VARCHAR(20) NOT NULL,
    final_label VARCHAR(20) NOT NULL,
    
    -- Quem e quando
    labeled_by VARCHAR(100) NOT NULL,
    labeled_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Confiança
    confidence VARCHAR(20),
    notes TEXT,
    
    -- Para retreino
    used_for_training BOOLEAN DEFAULT false,
    training_batch_id VARCHAR(50),
    
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
);
```

---

# 5. DESENHO DO MOTOR DE FRAUDE

## 5.1 Arquitetura do Motor ML

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MOTOR DE FRAUDE - ARQUITETURA ML                         │
│                         Ensemble Multi-Modelo                               │
└─────────────────────────────────────────────────────────────────────────────┘

                         ENTRADA (Transação)
                                │
                                ▼
                    ┌───────────────────────┐
                    │   PRÉ-PROCESSAMENTO   │
                    │   - Validação         │
                    │   - Normalização      │
                    │   - Feature Lookup    │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │   LAYER 0       │ │   LAYER 0       │ │   LAYER 0       │
    │   XGBoost       │ │   LightGBM      │ │   CatBoost      │
    │                 │ │                 │ │                 │
    │ - Tabular       │ │ - Tabular       │ │ - Tabular       │
    │ - 500 árvores   │ │ - 400 árvores   │ │ - 300 árvores   │
    │ - depth=8       │ │ - depth=6       │ │ - depth=7       │
    └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
             │                   │                   │
             │     Score 1       │     Score 2       │     Score 3
             │                   │                   │
             └─────────────────────────────────────────┐
                                                       │
    ┌─────────────────┐                               │
    │   LAYER 0       │                               │
    │   GNN (Graph)   │                               │
    │                 │                               │
    │ - Node2Vec      │ Score 4                       │
    │ - GAT           │──────────────────────────────►│
    │ - Fraud rings   │                               │
    └─────────────────┘                               │
                                                       │
    ┌─────────────────┐                               │
    │   LAYER 0       │                               │
    │ Isolation Forest│                               │
    │                 │                               │
    │ - Anomaly det   │ Score 5                       │
    │ - Outliers      │──────────────────────────────►│
    └─────────────────┘                               │
                                                       ▼
                              ┌─────────────────────────────────────┐
                              │           META-MODELO               │
                              │       Stacking (Logistic Reg)       │
                              │                                     │
                              │  Score_final = σ(w1*s1 + w2*s2 +   │
                              │                 w3*s3 + w4*s4 +     │
                              │                 w5*s5 + bias)       │
                              └─────────────────┬───────────────────┘
                                                │
                                                ▼
                              ┌─────────────────────────────────────┐
                              │         CALIBRAÇÃO                  │
                              │     Platt Scaling / Isotonic        │
                              │                                     │
                              │  P(fraude) = calibrate(score_raw)   │
                              └─────────────────┬───────────────────┘
                                                │
                                                ▼
                              ┌─────────────────────────────────────┐
                              │       POLICY ENGINE (Regras)        │
                              │                                     │
                              │  IF score > 0.85 → BLOCK            │
                              │  IF score > 0.50 → STEP_UP          │
                              │  IF blacklist → BLOCK               │
                              │  IF whitelist → APPROVE             │
                              │  ELSE → APPROVE                     │
                              └─────────────────┬───────────────────┘
                                                │
                                                ▼
                                         DECISÃO FINAL
                              ┌─────────────────────────────────────┐
                              │  {                                  │
                              │    "decision": "STEP_UP",           │
                              │    "score": 0.67,                   │
                              │    "confidence": 0.89,              │
                              │    "reasons": ["velocity_high",     │
                              │                "new_receiver"],     │
                              │    "step_up_type": "BIOMETRIA"      │
                              │  }                                  │
                              └─────────────────────────────────────┘
```

## 5.2 Modelos em Detalhe

### 5.2.1 XGBoost (Modelo Principal)

```python
# Configuração do XGBoost para Fraude
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'min_child_weight': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 100,  # Para desbalanceamento
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'tree_method': 'hist',
    'random_state': 42
}

# Features mais importantes (SHAP)
top_features = [
    'velocity_ratio_5min',      # 0.15
    'amount_zscore',            # 0.12
    'device_fraud_rate',        # 0.10
    'impossible_travel',        # 0.09
    'receiver_novelty',         # 0.08
    'ip_risk_score',            # 0.07
    'time_since_last_tx',       # 0.06
    'unique_receivers_1h',      # 0.05
]
```

### 5.2.2 LightGBM (Modelo Secundário)

```python
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 64,
    'learning_rate': 0.03,
    'n_estimators': 400,
    'max_depth': 6,
    'min_child_samples': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'is_unbalance': True,
    'random_state': 42
}
```

### 5.2.3 Graph Neural Network (GNN)

```python
# Configuração do GNN para detecção de rings/mules
class FraudGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Node2Vec embeddings
        self.node2vec = Node2Vec(
            edge_index=edge_index,
            embedding_dim=64,
            walk_length=20,
            context_size=10,
            walks_per_node=10
        )
        
        # Graph Attention Network
        self.gat1 = GATConv(64, 32, heads=4)
        self.gat2 = GATConv(128, 16, heads=4)
        
        # Classificador
        self.classifier = nn.Linear(64, 1)
        
    def forward(self, x, edge_index):
        # Embeddings
        x = self.node2vec(x)
        
        # GAT layers
        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.gat2(x, edge_index)
        
        # Score
        return torch.sigmoid(self.classifier(x))
```

### 5.2.4 Isolation Forest (Anomaly Detection)

```python
from sklearn.ensemble import IsolationForest

isolation_forest = IsolationForest(
    n_estimators=200,
    max_samples='auto',
    contamination=0.01,
    max_features=0.8,
    bootstrap=True,
    random_state=42
)

# Features específicas para anomalia
anomaly_features = [
    'amount',
    'hour_of_day',
    'tx_count_24h',
    'amount_vs_avg_30d',
    'geo_distance_km',
    'time_since_last_tx'
]
```

## 5.3 Policy Engine (Regras de Negócio)

```yaml
# Configuração de Políticas de Fraude
policies:
  # Regras de Bloqueio Imediato
  hard_block:
    - name: "blacklist_device"
      condition: "device_id IN blacklist"
      action: "BLOCK"
      priority: 1
      
    - name: "blacklist_account"
      condition: "receiver_account_id IN blacklist"
      action: "BLOCK"
      priority: 1
      
    - name: "impossible_travel"
      condition: "impossible_travel == true AND distance_km > 500"
      action: "BLOCK"
      priority: 2
      
    - name: "high_velocity_block"
      condition: "tx_count_5min > 10 AND amount_sum_5min > 10000"
      action: "BLOCK"
      priority: 3

  # Regras de STEP_UP
  step_up:
    - name: "high_score_step_up"
      condition: "fraud_score >= 0.50 AND fraud_score < 0.85"
      action: "STEP_UP"
      step_up_type: "BIOMETRIA"
      priority: 1
      
    - name: "new_device_high_value"
      condition: "is_new_device == true AND amount > 5000"
      action: "STEP_UP"
      step_up_type: "SMS_OTP"
      priority: 2
      
    - name: "first_time_receiver"
      condition: "receiver_novelty == true AND amount > 1000"
      action: "STEP_UP"
      step_up_type: "PUSH_CONFIRM"
      priority: 3

  # Regras de Whitelist
  whitelist:
    - name: "trusted_merchant"
      condition: "merchant_id IN whitelist AND amount < 500"
      action: "APPROVE"
      priority: 1
      
    - name: "trusted_device_low_risk"
      condition: "device_trust_score > 0.9 AND fraud_score < 0.1"
      action: "APPROVE"
      priority: 2

  # Limiares por Segmento
  thresholds:
    PIX:
      block_threshold: 0.85
      step_up_threshold: 0.50
      max_amount_no_step_up: 1000
      
    CREDITO:
      block_threshold: 0.80
      step_up_threshold: 0.45
      max_amount_no_step_up: 500
      
    DEBITO:
      block_threshold: 0.90
      step_up_threshold: 0.60
      max_amount_no_step_up: 2000
```

## 5.4 STEP_UP Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STEP_UP DECISION FLOW                             │
└─────────────────────────────────────────────────────────────────────────────┘

                    Transação recebida
                           │
                           ▼
              ┌────────────────────────┐
              │   Scoring ML + Regras  │
              └───────────┬────────────┘
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
     [Score < 0.50]  [0.50-0.85]  [Score >= 0.85]
            │             │             │
            ▼             ▼             ▼
       ┌────────┐   ┌──────────┐   ┌────────┐
       │APPROVE │   │ STEP_UP  │   │ BLOCK  │
       └────────┘   └────┬─────┘   └────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    [Biometria]    [SMS OTP]     [Push Confirm]
         │               │               │
         └───────────────┼───────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
         [Sucesso]             [Falha/Timeout]
              │                     │
              ▼                     ▼
         ┌────────┐            ┌────────┐
         │APPROVE │            │ BLOCK  │
         └────────┘            │+ Alert │
                               └────────┘

TIPOS DE STEP_UP:

1. BIOMETRIA (Score 0.70-0.85)
   - Face ID / Touch ID
   - Tempo limite: 60 segundos
   - 3 tentativas máximo
   
2. SMS_OTP (Score 0.55-0.70)
   - Código 6 dígitos
   - Tempo limite: 180 segundos
   - 1 reenvio permitido

3. PUSH_CONFIRM (Score 0.50-0.55)
   - Notificação push
   - Confirmar com PIN
   - Tempo limite: 120 segundos

4. VIDEO_SELFIE (Casos especiais)
   - Prova de vida
   - Comparação com documento
   - Tempo limite: 300 segundos
```

## 5.5 Explicabilidade (SHAP)

```python
import shap

# Gerar explicações para cada predição
def explain_prediction(model, features, transaction_id):
    """
    Gera explicação SHAP para uma predição
    """
    # Calcular SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    
    # Top 5 features contribuindo para o score
    feature_importance = sorted(
        zip(feature_names, shap_values[0]),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]
    
    return {
        "transaction_id": transaction_id,
        "base_score": explainer.expected_value,
        "final_score": model.predict_proba(features)[0][1],
        "top_contributors": [
            {
                "feature": name,
                "contribution": float(value),
                "direction": "aumenta_risco" if value > 0 else "diminui_risco"
            }
            for name, value in feature_importance
        ],
        "explanation_text": generate_human_readable(feature_importance)
    }

def generate_human_readable(contributions):
    """
    Gera explicação em texto para auditoria/LGPD
    """
    reasons = []
    for feature, value in contributions:
        if feature == 'velocity_ratio_5min' and value > 0:
            reasons.append("Alta velocidade de transações nos últimos 5 minutos")
        elif feature == 'amount_zscore' and value > 0:
            reasons.append("Valor atípico comparado ao histórico")
        elif feature == 'new_device' and value > 0:
            reasons.append("Dispositivo não reconhecido")
        elif feature == 'receiver_novelty' and value > 0:
            reasons.append("Primeiro envio para este destinatário")
        # ... mais mapeamentos
    
    return "; ".join(reasons) if reasons else "Padrão normal de transação"
```

---

# 6. MÉTRICAS E PAINÉIS

## 6.1 Dashboard de Métricas em Tempo Real

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FRAUD DETECTION COMMAND CENTER                           │
│                     Atualização: Real-time (5 segundos)                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│  KPIs PRINCIPAIS                                                    [HOJE]   │
├───────────────────┬───────────────────┬───────────────────┬─────────────────┤
│  TRANSAÇÕES       │  FRAUDES DETECT.  │  STEP_UPS         │  BLOQUEIOS      │
│  ████████████     │  ████████████     │  ████████████     │  ████████████   │
│  15.2M            │  12,847           │  89,234           │  8,421          │
│  +12% vs ontem    │  Recall: 87.3%    │  Conv: 78.4%      │  $R$ 4.2M       │
└───────────────────┴───────────────────┴───────────────────┴─────────────────┘

┌─────────────────────────────────────┬───────────────────────────────────────┐
│  MÉTRICAS DO MODELO (Últimas 24h)  │  LATÊNCIA (p99)                       │
├─────────────────────────────────────┼───────────────────────────────────────┤
│                                     │                                       │
│  AUC-PR:     0.923 ████████████░░  │  Scoring:     12ms  ████░░░░░░░░░░░  │
│  AUC-ROC:    0.967 █████████████░  │  Features:    18ms  █████░░░░░░░░░░  │
│  Precision:  0.812 ████████████░░  │  Total E2E:   42ms  █████████░░░░░░  │
│  Recall:     0.873 █████████████░  │  Target:      50ms  ─────────────────  │
│  F1-Score:   0.841 ████████████░░  │                                       │
│  $Precision: 0.789 ███████████░░░  │  TPS Atual:   3,847                   │
│  $Recall:    0.856 █████████████░  │  TPS Pico:    8,234                   │
│                                     │                                       │
└─────────────────────────────────────┴───────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  VOLUME POR HORA (Últimas 24h)                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  800K ┤                                    ╭─────╮                         │
│       │                                   ╭╯     ╰╮                        │
│  600K ┤                          ╭───────╯        ╰─╮                      │
│       │                    ╭────╯                   ╰───╮                  │
│  400K ┤             ╭─────╯                              ╰────╮            │
│       │       ╭────╯                                          ╰───╮       │
│  200K ┤ ╭────╯                                                     ╰──╮   │
│       │─╯                                                              ╰─ │
│    0  ┼────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┤  │
│       0    2    4    6    8   10   12   14   16   18   20   22   24      │
│                                                                             │
│  ■ Aprovadas (92.1%)  ■ STEP_UP (6.8%)  ■ Bloqueadas (1.1%)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│  ALERTAS RECENTES                                                   [VER +]  │
├───────────────────────────────────────────────────────────────────────────────┤
│  🔴 CRÍTICO  | Ring de fraude detectado | 23 contas | R$ 847K | há 2 min    │
│  🟠 ALTO     | Velocity spike conta XXX | 45 tx/5min | R$ 12K | há 5 min    │
│  🟠 ALTO     | Device comprometido | 8 contas afetadas | há 8 min           │
│  🟡 MÉDIO    | Drift detectado feature velocity_1h | PSI: 0.18 | há 15 min  │
│  🟢 INFO     | Modelo v2.3.1 promovido a champion | há 1 hora               │
└───────────────────────────────────────────────────────────────────────────────┘
```

## 6.2 Catálogo Completo de Métricas

### 6.2.1 Métricas de Classificação

| Métrica | Fórmula | Target | Alerta |
|---------|---------|--------|--------|
| **Precision** | TP / (TP + FP) | > 0.70 | < 0.60 |
| **Recall (TPR)** | TP / (TP + FN) | > 0.85 | < 0.75 |
| **FPR** | FP / (FP + TN) | < 0.02 | > 0.03 |
| **F1-Score** | 2 * (P * R) / (P + R) | > 0.75 | < 0.65 |
| **Fβ (β=2)** | (1+β²)(P*R)/(β²P+R) | > 0.80 | < 0.70 |
| **MCC** | Matthews Correlation | > 0.60 | < 0.50 |
| **Balanced Accuracy** | (TPR + TNR) / 2 | > 0.85 | < 0.80 |

### 6.2.2 Métricas de Ranking

| Métrica | Descrição | Target | Alerta |
|---------|-----------|--------|--------|
| **AUC-PR** | Área sob curva Precision-Recall | > 0.90 | < 0.85 |
| **AUC-ROC** | Área sob curva ROC | > 0.95 | < 0.92 |
| **Gini** | 2 * AUC-ROC - 1 | > 0.90 | < 0.84 |
| **KS Statistic** | Max(TPR - FPR) | > 0.70 | < 0.60 |
| **Lift@1%** | Lift no top 1% | > 50x | < 30x |
| **Capture@5%** | Recall no top 5% scores | > 0.60 | < 0.50 |

### 6.2.3 Métricas de Calibração

| Métrica | Descrição | Target | Alerta |
|---------|-----------|--------|--------|
| **Brier Score** | Mean squared error | < 0.05 | > 0.08 |
| **Log Loss** | Cross-entropy loss | < 0.10 | > 0.15 |
| **ECE** | Expected Calibration Error | < 0.05 | > 0.10 |
| **MCE** | Maximum Calibration Error | < 0.10 | > 0.20 |

### 6.2.4 Métricas de Negócio

| Métrica | Descrição | Target | Alerta |
|---------|-----------|--------|--------|
| **$Precision** | Valor bloqueado correto / Total bloqueado | > 0.75 | < 0.60 |
| **$Recall** | Valor fraude detectado / Total fraude | > 0.85 | < 0.75 |
| **EV por decisão** | Expected Value (ganho - custo) | > R$ 50 | < R$ 20 |
| **Taxa Abandono STEP_UP** | Abandonos / Total STEP_UP | < 0.15 | > 0.25 |
| **Conversão STEP_UP** | Sucessos / Total STEP_UP | > 0.80 | < 0.70 |
| **% Revisão Manual** | Manual / Total decisões | < 0.02 | > 0.05 |

### 6.2.5 Métricas de Estabilidade

| Métrica | Descrição | Target | Alerta |
|---------|-----------|--------|--------|
| **PSI (Score)** | Population Stability Index | < 0.10 | > 0.20 |
| **PSI (Features)** | PSI por feature crítica | < 0.15 | > 0.25 |
| **Drift Score** | Desvio da distribuição | < 0.10 | > 0.15 |
| **AUC-PR Δ 7d** | Variação AUC-PR em 7 dias | < 0.02 | > 0.05 |

### 6.2.6 Métricas Operacionais

| Métrica | Descrição | Target | Alerta |
|---------|-----------|--------|--------|
| **Latência p50** | Mediana de latência | < 20ms | > 30ms |
| **Latência p95** | Percentil 95 | < 40ms | > 60ms |
| **Latência p99** | Percentil 99 | < 50ms | > 80ms |
| **TPS** | Transações por segundo | > 3,500 | < 3,000 |
| **Error Rate** | Taxa de erros (4xx/5xx) | < 0.01% | > 0.1% |
| **Availability** | Uptime | > 99.99% | < 99.9% |

---

# 7. BACKOFFICE REACT

## 7.1 Arquitetura do Frontend

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       BACKOFFICE FRONTEND ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │          NEXT.JS APP                │
                    │         (React 18 + SSR)            │
                    └─────────────────┬───────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
  ┌───────────┐               ┌───────────┐               ┌───────────┐
  │  PAGES    │               │   STATE   │               │    UI     │
  │           │               │   MGMT    │               │   LIBS    │
  │ /dashboard│               │           │               │           │
  │ /alerts   │               │  Zustand  │               │ Tailwind  │
  │ /cases    │               │  React    │               │ Shadcn/ui │
  │ /rules    │               │  Query    │               │ Recharts  │
  │ /models   │               │           │               │ AG-Grid   │
  │ /reports  │               │           │               │ React-Vis │
  └───────────┘               └───────────┘               └───────────┘
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │          API LAYER (BFF)            │
                    │         Node.js + tRPC              │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
        ┌───────────┐           ┌───────────┐           ┌───────────┐
        │  REST API │           │  GraphQL  │           │ WebSocket │
        │  (Java)   │           │  (Query)  │           │  (Live)   │
        └───────────┘           └───────────┘           └───────────┘
```

## 7.2 Mapa de Telas

```
BACKOFFICE - MAPA DE NAVEGAÇÃO
==============================

┌─ 📊 DASHBOARD (Home)
│   ├── KPIs em tempo real
│   ├── Gráficos de volume
│   ├── Alertas recentes
│   └── Status do sistema
│
├─ 🔔 ALERTAS
│   ├── Lista de alertas
│   ├── Filtros (severidade, tipo, status)
│   ├── Detalhes do alerta
│   └── Ações (investigar, resolver, escalar)
│
├─ 📋 CASOS (Investigação)
│   ├── Fila de casos
│   ├── Detalhes do caso
│   │   ├── Timeline da transação
│   │   ├── Grafo de relacionamentos
│   │   ├── Histórico do cliente
│   │   └── Evidências SHAP
│   └── Workflow de investigação
│
├─ 💳 TRANSAÇÕES
│   ├── Busca avançada
│   ├── Detalhes da transação
│   ├── Score breakdown
│   └── Ações manuais
│
├─ ⚙️ REGRAS
│   ├── Lista de regras ativas
│   ├── Editor de regras
│   ├── Simulador (what-if)
│   └── Histórico de mudanças
│
├─ 🤖 MODELOS (MLOps)
│   ├── Modelos em produção
│   ├── Métricas de performance
│   ├── Comparação champion vs challenger
│   ├── Drift monitoring
│   └── Pipeline de retreino
│
├─ 📈 RELATÓRIOS
│   ├── Relatório diário
│   ├── Relatório semanal
│   ├── Análise de tendências
│   ├── Compliance (BACEN)
│   └── Exportação (PDF/Excel)
│
├─ 🔒 LISTAS
│   ├── Blacklist (devices, IPs, contas)
│   ├── Whitelist (merchants, clientes VIP)
│   └── Watchlist (monitoramento especial)
│
├─ 👥 USUÁRIOS (Admin)
│   ├── Gestão de usuários
│   ├── Roles e permissões
│   └── Audit log
│
└─ ⚡ CONFIGURAÇÕES
    ├── Thresholds
    ├── Políticas por segmento
    ├── Integrações
    └── Notificações
```

## 7.3 Telas Principais (Wireframes)

### 7.3.1 Dashboard Principal

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🏦 SANKOFA FRAUD DETECTION          🔔 3  👤 Ana Silva  ⚙️                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  📊 DASHBOARD EXECUTIVO                      Última atualização: 5s  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │ TRANSAÇÕES   │ │ FRAUDES      │ │ VALOR SALVO  │ │ LATÊNCIA p99 │       │
│  │    HOJE      │ │  DETECTADAS  │ │              │ │              │       │
│  │  ████████    │ │  ████████    │ │  ████████    │ │  ████████    │       │
│  │   15.2M      │ │   12,847     │ │  R$ 4.2M     │ │    42ms      │       │
│  │  ▲ +12%      │ │  Recall 87%  │ │  ▲ +18%      │ │  Target: 50  │       │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘       │
│                                                                             │
│  ┌─────────────────────────────────┐ ┌─────────────────────────────────┐   │
│  │ 📈 VOLUME POR HORA              │ │ 🎯 MÉTRICAS DO MODELO           │   │
│  │                                 │ │                                 │   │
│  │  ┌───────────────────────────┐  │ │  AUC-PR    ████████████░░ 0.92 │   │
│  │  │        ╭────╮             │  │ │  Precision ███████████░░░ 0.81 │   │
│  │  │    ╭──╯    ╰──╮          │  │ │  Recall    █████████████░ 0.87 │   │
│  │  │ ──╯            ╰───      │  │ │  F1-Score  ████████████░░ 0.84 │   │
│  │  │ 0  6  12  18  24         │  │ │  $Recall   █████████████░ 0.86 │   │
│  │  └───────────────────────────┘  │ │                                 │   │
│  │  ■ Approved ■ StepUp ■ Blocked  │ │  Última avaliação: 15 min      │   │
│  └─────────────────────────────────┘ └─────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 🚨 ALERTAS RECENTES                                      [Ver todos] │   │
│  │                                                                       │   │
│  │  🔴 Ring de fraude detectado      23 contas    R$ 847K    há 2 min   │   │
│  │  🟠 Velocity spike anômalo        45 tx/5min   R$ 12K     há 5 min   │   │
│  │  🟠 Device comprometido           8 contas     -          há 8 min   │   │
│  │  🟡 Drift feature velocity_1h     PSI: 0.18   -          há 15 min  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3.2 Tela de Investigação de Caso

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🔍 INVESTIGAÇÃO - CASO #FRD-2025-1847                    [Resolver] [Escalar]│
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  RESUMO DO CASO                                                     │   │
│  │  Status: EM INVESTIGAÇÃO    Prioridade: ALTA    Analista: Maria S.  │   │
│  │  Transação: TXN_985760735407    Valor: R$ 45.500,00    Tipo: PIX   │   │
│  │  Score: 0.87    Decisão: BLOQUEADO    Motivo: Velocity + New Device │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌────────────────────────────┐ ┌──────────────────────────────────────┐   │
│  │ 📊 SCORE BREAKDOWN (SHAP)  │ │ 🔗 GRAFO DE RELACIONAMENTOS         │   │
│  │                            │ │                                      │   │
│  │ velocity_5min    ████ +0.23│ │      [Dev1]──────[Conta1]            │   │
│  │ new_device       ███  +0.18│ │        │           │                 │   │
│  │ amount_zscore    ██   +0.12│ │        │      ┌────┴────┐            │   │
│  │ receiver_new     ██   +0.11│ │        │    [IP1]    [IP2]           │   │
│  │ hour_anomaly     █    +0.08│ │        │      │         │            │   │
│  │ ───────────────────────────│ │      [Dev2]──[Conta2]──[Conta3]      │   │
│  │ device_trust     ██   -0.09│ │                │                     │   │
│  │ account_age      █    -0.05│ │           [🔴 Fraude conhecida]      │   │
│  │                            │ │                                      │   │
│  └────────────────────────────┘ └──────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 📜 TIMELINE DA TRANSAÇÃO                                             │   │
│  │                                                                       │   │
│  │  12:44:55 │ Transação iniciada via app mobile                        │   │
│  │  12:44:56 │ Device fingerprint coletado (NOVO DEVICE)                │   │
│  │  12:44:56 │ Geolocalização: São Paulo, SP                            │   │
│  │  12:44:57 │ Feature computation: 18ms                                │   │
│  │  12:44:57 │ ML Scoring: 0.87 (ALTO RISCO)                            │   │
│  │  12:44:58 │ Decisão: BLOQUEIO (velocity + new_device)                │   │
│  │  12:45:00 │ Notificação enviada ao cliente                           │   │
│  │  12:45:15 │ Cliente tentou novamente (bloqueado)                     │   │
│  │  12:47:00 │ Alerta gerado para investigação                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 📝 NOTAS DE INVESTIGAÇÃO                                    [Salvar] │   │
│  │  ┌───────────────────────────────────────────────────────────────┐   │   │
│  │  │ Verificado histórico da conta. Cliente reportou perda do     │   │   │
│  │  │ celular ontem. Transação provavelmente é fraude ATO.         │   │   │
│  │  │ Recomendação: Confirmar fraude e bloquear device.            │   │   │
│  │  └───────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  [◉ Confirmar Fraude] [○ Falso Positivo] [○ Inconclusivo] [Enviar]        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 7.4 Acessibilidade (WCAG + TEA-Friendly)

```yaml
accessibility:
  wcag_level: "AA"
  
  visual:
    - Contraste mínimo: 4.5:1 para texto normal
    - Contraste mínimo: 3:1 para texto grande
    - Não depender apenas de cor para informação
    - Indicadores visuais + texto para status
    
  keyboard:
    - Navegação completa por teclado
    - Foco visível em todos elementos
    - Skip links para conteúdo principal
    - Atalhos de teclado documentados
    
  screen_reader:
    - ARIA labels em todos componentes
    - Landmarks para navegação
    - Live regions para atualizações
    - Descrições alternativas para gráficos
    
  tea_friendly:
    - Modo de baixa estimulação visual
    - Redução de animações (prefers-reduced-motion)
    - Linguagem clara e direta
    - Estrutura previsível e consistente
    - Tempo adequado para leitura
    - Evitar pop-ups e interrupções
    
  color_modes:
    - Light mode (padrão)
    - Dark mode
    - Alto contraste
    - Daltonismo (3 variantes)
```

---

# 8. SEGURANÇA, LGPD E COMPLIANCE

## 8.1 Arquitetura de Segurança

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SECURITY ARCHITECTURE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

                              INTERNET
                                 │
                    ┌────────────▼────────────┐
                    │      AWS Shield         │ ◄── DDoS Protection
                    │      + WAF              │ ◄── OWASP Top 10
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     CloudFront          │ ◄── CDN + Edge Security
                    │     + ACM (TLS 1.3)     │ ◄── Certificados gerenciados
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    API Gateway          │ ◄── Rate Limiting
                    │    + Cognito            │ ◄── OAuth 2.0 / OIDC
                    └────────────┬────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────────────────┐
│                            VPC (Private)                                     │
│                                │                                             │
│  ┌─────────────────────────────┼─────────────────────────────────────────┐  │
│  │                  PRIVATE SUBNET (Apps)                                │  │
│  │                             │                                          │  │
│  │  ┌─────────────┐   ┌───────▼───────┐   ┌─────────────┐               │  │
│  │  │   Kong      │──▶│  EKS Cluster  │──▶│   Flink     │               │  │
│  │  │  (mTLS)     │   │  (Pod Sec)    │   │  (Isolated) │               │  │
│  │  └─────────────┘   └───────────────┘   └─────────────┘               │  │
│  │                                                                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                  PRIVATE SUBNET (Data)                                  │  │
│  │                                                                         │  │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                   │  │
│  │  │   Aurora    │   │   Redis     │   │    S3       │                   │  │
│  │  │  (Encrypt)  │   │  (Encrypt)  │   │ (SSE-KMS)   │                   │  │
│  │  └─────────────┘   └─────────────┘   └─────────────┘                   │  │
│  │                                                                         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                  SECURITY SERVICES                                      │  │
│  │                                                                         │  │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌────────────┐ │  │
│  │  │  KMS        │   │  Secrets    │   │  GuardDuty  │   │  Security  │ │  │
│  │  │  (Keys)     │   │  Manager    │   │  (Threat)   │   │  Hub       │ │  │
│  │  └─────────────┘   └─────────────┘   └─────────────┘   └────────────┘ │  │
│  │                                                                         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 8.2 Controles de Segurança

### 8.2.1 Autenticação e Autorização

| Controle | Implementação | Padrão |
|----------|---------------|--------|
| **MFA** | TOTP + Push | Obrigatório |
| **SSO** | SAML 2.0 / OIDC | Azure AD |
| **RBAC** | 5 níveis de acesso | Least Privilege |
| **Session** | JWT + Refresh (15min/24h) | Stateless |
| **API Auth** | mTLS + API Key | Service-to-service |

### 8.2.2 Níveis de Acesso (RBAC)

```yaml
roles:
  viewer:
    description: "Visualização apenas"
    permissions:
      - dashboard:read
      - transactions:read
      - alerts:read
      
  analyst:
    description: "Analista de fraude L1"
    inherits: viewer
    permissions:
      - cases:read
      - cases:update
      - feedback:create
      
  senior_analyst:
    description: "Analista de fraude L2/L3"
    inherits: analyst
    permissions:
      - blacklist:create
      - blacklist:update
      - rules:read
      
  manager:
    description: "Gestor de operações"
    inherits: senior_analyst
    permissions:
      - rules:create
      - rules:update
      - reports:create
      - users:read
      
  admin:
    description: "Administrador do sistema"
    inherits: manager
    permissions:
      - users:create
      - users:update
      - users:delete
      - config:update
      - models:deploy
```

## 8.3 LGPD Compliance

### 8.3.1 Requisitos e Implementação

| Artigo LGPD | Requisito | Implementação |
|-------------|-----------|---------------|
| **Art. 6** | Finalidade | Dados usados apenas para prevenção a fraude |
| **Art. 7** | Base Legal | Legítimo interesse (prevenção a fraude) |
| **Art. 9** | Dados Sensíveis | Não coletamos dados sensíveis |
| **Art. 11** | Minimização | Apenas dados necessários |
| **Art. 16** | Término | Retenção de 5 anos (BACEN) |
| **Art. 18** | Direitos Titular | Acesso, correção, exclusão |
| **Art. 46** | Segurança | Criptografia, controle de acesso |
| **Art. 50** | Governança | DPO nomeado, políticas documentadas |

### 8.3.2 Mascaramento de Dados

```python
# Política de mascaramento
masking_rules = {
    "cpf": {
        "type": "partial",
        "pattern": "***.***.XXX-XX",  # Mostra apenas últimos 5 dígitos
        "roles_full_access": ["admin", "compliance"]
    },
    "email": {
        "type": "partial", 
        "pattern": "xxx@domain.com",  # Oculta parte local
        "roles_full_access": ["admin"]
    },
    "phone": {
        "type": "partial",
        "pattern": "(**) *****-XXXX",  # Mostra apenas últimos 4
        "roles_full_access": ["admin", "senior_analyst"]
    },
    "ip_address": {
        "type": "hash",
        "algorithm": "sha256_truncated",
        "roles_full_access": ["admin", "security"]
    },
    "card_number": {
        "type": "tokenize",
        "show": "last4",
        "roles_full_access": []  # Ninguém vê completo
    }
}
```

### 8.3.3 Direitos do Titular

```
FLUXO DE SOLICITAÇÃO DO TITULAR (Art. 18)
=========================================

     Solicitação via SAC/Portal
              │
              ▼
     ┌─────────────────┐
     │  Verificação de │
     │   Identidade    │
     └────────┬────────┘
              │
              ▼
     ┌─────────────────┐
     │  Classificação  │
     │  da Solicitação │
     └────────┬────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
    ▼         ▼         ▼
[ACESSO]  [CORREÇÃO] [EXCLUSÃO]
    │         │         │
    ▼         ▼         ▼
 Gerar     Atualizar  Verificar
 Relatório  Dados     Retenção
    │         │         │
    ▼         ▼         ▼
 Enviar    Confirmar  Anonimizar
 PDF       Mudança    (se possível)
    │         │         │
    └─────────┼─────────┘
              │
              ▼
     ┌─────────────────┐
     │   Registrar no  │
     │   Audit Log     │
     └─────────────────┘

SLA: 15 dias úteis
```

## 8.4 BACEN Compliance

### 8.4.1 Resoluções Aplicáveis

| Resolução | Requisito | Status |
|-----------|-----------|--------|
| **CMN 4.893/21** | Política de segurança cibernética | ✅ Implementado |
| **BCB 85/21** | Compartilhamento de dados PIX | ✅ Implementado |
| **DICT** | Diretório de Identificadores | ✅ Integrado |
| **MED** | Mecanismo Especial de Devolução | ✅ Implementado |
| **CMN 4.658/18** | Contratação de serviços de nuvem | ✅ Conformidade AWS |

### 8.4.2 Trilha de Auditoria

```sql
-- Todos os eventos são registrados com:
-- - Timestamp imutável
-- - Identificação do ator
-- - IP de origem
-- - Ação realizada
-- - Dados antes/depois (quando aplicável)

CREATE TABLE audit_trail (
    id BIGSERIAL PRIMARY KEY,
    event_id UUID DEFAULT gen_random_uuid(),
    event_timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- Ator
    actor_id VARCHAR(100) NOT NULL,
    actor_type VARCHAR(20) NOT NULL,  -- USER, SYSTEM, API
    actor_ip INET,
    actor_user_agent TEXT,
    
    -- Ação
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(100),
    
    -- Dados
    request_payload JSONB,
    response_status INTEGER,
    changes JSONB,  -- {before: {...}, after: {...}}
    
    -- Contexto
    session_id VARCHAR(100),
    correlation_id VARCHAR(100),
    
    -- Integridade
    checksum VARCHAR(64) GENERATED ALWAYS AS (
        encode(sha256(
            (event_id || event_timestamp || actor_id || action)::bytea
        ), 'hex')
    ) STORED
);

-- Índices para consultas de auditoria
CREATE INDEX idx_audit_timestamp ON audit_trail(event_timestamp);
CREATE INDEX idx_audit_actor ON audit_trail(actor_id);
CREATE INDEX idx_audit_resource ON audit_trail(resource_type, resource_id);
CREATE INDEX idx_audit_action ON audit_trail(action);
```

## 8.5 PCI DSS Compliance

| Requisito | Descrição | Status |
|-----------|-----------|--------|
| **Req 1** | Firewall/Segmentação | ✅ VPC isolada |
| **Req 2** | Sem defaults | ✅ Hardening aplicado |
| **Req 3** | Proteção de dados | ✅ Tokenização de PAN |
| **Req 4** | Criptografia em trânsito | ✅ TLS 1.3 |
| **Req 5** | Antimalware | ✅ GuardDuty |
| **Req 6** | Sistemas seguros | ✅ Patch management |
| **Req 7** | Acesso restrito | ✅ RBAC |
| **Req 8** | Identificação | ✅ MFA obrigatório |
| **Req 9** | Acesso físico | ✅ AWS (SOC 2) |
| **Req 10** | Logging | ✅ CloudTrail + Audit |
| **Req 11** | Testes | ✅ Pentest anual |
| **Req 12** | Políticas | ✅ Documentadas |

---

# 9. OBSERVABILIDADE & SRE

## 9.1 Stack de Observabilidade

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY STACK                                       │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │           DATADOG                   │
                    │     (Unified Observability)         │
                    └─────────────────┬───────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
  ┌───────────┐               ┌───────────┐               ┌───────────┐
  │  METRICS  │               │   LOGS    │               │  TRACES   │
  │           │               │           │               │           │
  │ DD Agent  │               │ Fluentd   │               │ DD APM    │
  │ StatsD    │               │           │               │ OpenTel   │
  │ Prometheus│               │           │               │           │
  └───────────┘               └───────────┘               └───────────┘
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
  ┌───────────┐               ┌───────────┐               ┌───────────┐
  │DASHBOARDS │               │  ALERTS   │               │ RUNBOOKS  │
  │           │               │           │               │           │
  │ Real-time │               │ PagerDuty │               │ Automated │
  │ SLO/SLI   │               │ Slack     │               │ Remediation│
  │ Business  │               │ Email     │               │           │
  └───────────┘               └───────────┘               └───────────┘
```

## 9.2 SLOs e SLIs

### 9.2.1 Definição de SLOs

| Serviço | SLI | SLO | Error Budget (30d) |
|---------|-----|-----|-------------------|
| **API Scoring** | Latência p99 | < 50ms | 43,200 min (99.9%) |
| **API Scoring** | Disponibilidade | 99.99% | 4.32 min |
| **Feature Store** | Latência p99 | < 10ms | 43,200 min (99.9%) |
| **Kafka** | Lag máximo | < 100 msgs | 99.9% do tempo |
| **Model Serving** | Latência p99 | < 15ms | 43,200 min (99.9%) |
| **Backoffice** | Disponibilidade | 99.9% | 43.2 min |

### 9.2.2 Métricas Críticas para Alertas

```yaml
critical_alerts:
  # Latência
  - name: "High Latency - Scoring API"
    metric: "fraud.scoring.latency.p99"
    threshold: "> 50ms"
    duration: "5 minutes"
    severity: "P1"
    runbook: "https://wiki/runbooks/high-latency-scoring"
    
  - name: "Very High Latency - Scoring API"
    metric: "fraud.scoring.latency.p99"
    threshold: "> 100ms"
    duration: "2 minutes"
    severity: "P0"
    auto_page: true
    
  # Disponibilidade
  - name: "High Error Rate - Scoring API"
    metric: "fraud.scoring.error_rate"
    threshold: "> 1%"
    duration: "5 minutes"
    severity: "P1"
    
  - name: "Service Down - Scoring API"
    metric: "fraud.scoring.availability"
    threshold: "< 99%"
    duration: "1 minute"
    severity: "P0"
    auto_page: true
    
  # ML/Modelo
  - name: "Model Drift Detected"
    metric: "fraud.model.psi_score"
    threshold: "> 0.20"
    duration: "1 hour"
    severity: "P2"
    
  - name: "AUC-PR Degradation"
    metric: "fraud.model.auc_pr"
    threshold: "< 0.85"
    duration: "1 hour"
    severity: "P1"
    
  # Negócio
  - name: "High False Positive Rate"
    metric: "fraud.business.fpr"
    threshold: "> 3%"
    duration: "1 hour"
    severity: "P1"
    
  - name: "Low Recall Alert"
    metric: "fraud.business.recall"
    threshold: "< 75%"
    duration: "1 hour"
    severity: "P0"

  # Infraestrutura
  - name: "Kafka Consumer Lag"
    metric: "kafka.consumer.lag"
    threshold: "> 10000"
    duration: "5 minutes"
    severity: "P1"
    
  - name: "Redis Memory High"
    metric: "redis.memory.used_percent"
    threshold: "> 85%"
    duration: "10 minutes"
    severity: "P2"
```

## 9.3 On-Call e Escalation

```
ESCALATION MATRIX
=================

P0 (Crítico) - Sistema indisponível ou perda massiva
├── 0-5 min:   Alerta automático + Page on-call L1
├── 5-15 min:  Escalar para L2 + Tech Lead
├── 15-30 min: Escalar para Engineering Manager
├── 30-60 min: Escalar para VP de Engenharia
└── 60+ min:   Incident Commander + War Room

P1 (Alto) - Degradação significativa
├── 0-15 min:  Alerta automático + Slack on-call
├── 15-30 min: Page on-call L1
├── 30-60 min: Escalar para L2
└── 60+ min:   Escalar para Tech Lead

P2 (Médio) - Degradação menor
├── 0-30 min:  Alerta Slack
├── 30-60 min: Notificar on-call
└── 4h+ :      Escalar para L1

P3 (Baixo) - Informativo
├── Ticket automático
└── Revisão no próximo dia útil

ON-CALL ROTATION:
- L1: Rotação semanal (6 engenheiros)
- L2: Rotação mensal (Tech Leads)
- Cobertura: 24/7/365
```

---

# 10. OPERAÇÕES DE FRAUDE & PLAYBOOKS

## 10.1 Estrutura Operacional

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FRAUD OPERATIONS CENTER                                   │
│                         24/7/365 Operation                                   │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   COMMAND       │
                              │   CENTER        │
                              └────────┬────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
           ▼                           ▼                           ▼
    ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
    │    L1       │            │    L2       │            │    L3       │
    │  TRIAGE     │────────────│  ANALYSIS   │────────────│  EXPERT     │
    │             │            │             │            │             │
    │ - 12 FTEs   │            │ - 6 FTEs    │            │ - 3 FTEs    │
    │ - SLA: 15m  │            │ - SLA: 2h   │            │ - SLA: 4h   │
    │ - Casos     │            │ - Complexos │            │ - Rings     │
    │   simples   │            │ - ATO       │            │ - Mules     │
    └─────────────┘            └─────────────┘            └─────────────┘
           │                           │                           │
           └───────────────────────────┼───────────────────────────┘
                                       │
                              ┌────────▼────────┐
                              │   ESCALATION    │
                              │   (Jurídico/    │
                              │    Polícia)     │
                              └─────────────────┘

TURNOS:
├── Manhã:   06:00 - 14:00 (6 L1, 3 L2, 2 L3)
├── Tarde:   14:00 - 22:00 (6 L1, 3 L2, 1 L3)
└── Noite:   22:00 - 06:00 (3 L1, 1 L2, backup L3)
```

## 10.2 Playbooks Principais

### 10.2.1 Playbook: Account Takeover (ATO)

```yaml
playbook:
  name: "Account Takeover Investigation"
  id: "PB-ATO-001"
  severity: "HIGH"
  sla: "2 hours"
  
  triggers:
    - "Novo device + transação alto valor"
    - "Mudança de senha + transação < 24h"
    - "Login de novo IP + país diferente"
    
  steps:
    1:
      action: "Verificar histórico de autenticação"
      details:
        - Últimos 30 dias de logins
        - Devices utilizados
        - IPs de origem
      evidence: ["auth_logs", "device_history"]
      
    2:
      action: "Analisar padrão transacional"
      details:
        - Comparar com baseline 90 dias
        - Verificar destinatários novos
        - Checar valores atípicos
      evidence: ["transaction_history", "recipient_analysis"]
      
    3:
      action: "Verificar sinais de comprometimento"
      details:
        - SIM swap recente?
        - Email alterado?
        - Senha resetada?
      evidence: ["account_changes", "telecom_check"]
      
    4:
      action: "Contatar cliente"
      details:
        - Ligar para número cadastrado original
        - Confirmar identidade
        - Verificar ciência da transação
      evidence: ["call_recording", "verification_result"]
      
  decision_tree:
    if_confirmed_fraud:
      - Bloquear conta imediatamente
      - Cancelar transações pendentes
      - Iniciar processo MED (se PIX)
      - Registrar BO
      - Notificar cliente
      
    if_false_positive:
      - Liberar transação
      - Adicionar device ao whitelist
      - Atualizar perfil de risco
      - Documentar decisão
      
    if_inconclusive:
      - Escalar para L3
      - Manter bloqueio temporário
      - Agendar follow-up 24h
```

### 10.2.2 Playbook: Ring de Fraude Detectado

```yaml
playbook:
  name: "Fraud Ring Investigation"
  id: "PB-RING-001"
  severity: "CRITICAL"
  sla: "4 hours"
  team: "L3 + Compliance + Legal"
  
  triggers:
    - "GNN detectou cluster suspeito > 10 contas"
    - "Múltiplas contas compartilhando device"
    - "Padrão de mule accounts identificado"
    
  immediate_actions:
    - Bloquear todas as contas do cluster
    - Preservar evidências (snapshot)
    - Notificar Compliance
    - Iniciar timeline de eventos
    
  investigation:
    1:
      action: "Mapear extensão do ring"
      details:
        - Expandir grafo (2 níveis)
        - Identificar todas as conexões
        - Quantificar valor envolvido
        
    2:
      action: "Identificar conta-mãe"
      details:
        - Conta mais antiga
        - Maior volume
        - Hub do grafo
        
    3:
      action: "Rastrear origem dos fundos"
      details:
        - De onde veio o dinheiro?
        - Quais contas externas?
        - Padrão de lavagem?
        
    4:
      action: "Documentar para autoridades"
      details:
        - Preparar relatório técnico
        - Timeline de eventos
        - Evidências preservadas
        
  escalation:
    - COAF (se > R$ 50K ou AML suspeito)
    - Polícia Civil (BO)
    - BACEN (se PIX/DICT envolvido)
```

## 10.3 Métricas Operacionais

| Métrica | Target | Alerta |
|---------|--------|--------|
| Tempo médio de triagem (L1) | < 15 min | > 30 min |
| Tempo médio de resolução (L2) | < 2 horas | > 4 horas |
| Casos pendentes L1 | < 50 | > 100 |
| Casos pendentes L2 | < 20 | > 40 |
| Taxa de escalation L1→L2 | < 30% | > 50% |
| Acurácia de decisões | > 95% | < 90% |
| NPS do cliente pós-investigação | > 70 | < 50 |

---

# 11. MLOPS & GOVERNANÇA DE MODELOS

## 11.1 Pipeline de MLOps

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MLOPS PIPELINE                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   DATA      │───▶│  FEATURE    │───▶│   TRAIN     │───▶│  EVALUATE   │
│  INGESTION  │    │  ENGINEERING│    │             │    │             │
│             │    │             │    │             │    │             │
│ - Raw data  │    │ - Compute   │    │ - XGBoost   │    │ - AUC-PR    │
│ - Labels    │    │ - Transform │    │ - LightGBM  │    │ - Recall    │
│ - Validate  │    │ - Store     │    │ - GNN       │    │ - $Metrics  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                   ┌────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   DEPLOY    │◀───│  VALIDATE   │◀───│   PACKAGE   │◀───│  REGISTRY   │
│             │    │             │    │             │    │             │
│ - Canary    │    │ - Shadow    │    │ - ONNX      │    │ - Version   │
│ - Blue/Green│    │ - A/B Test  │    │ - Docker    │    │ - Metadata  │
│ - Rollback  │    │ - Guardrails│    │ - Sign      │    │ - Lineage   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MONITORING                                      │
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│   │  METRICS    │    │   DRIFT     │    │  FEEDBACK   │    │  RETRAIN    │ │
│   │             │    │             │    │             │    │  TRIGGER    │ │
│   │ - Real-time │    │ - PSI       │    │ - Human     │    │ - Auto      │ │
│   │ - Business  │    │ - KS        │    │ - Labels    │    │ - Scheduled │ │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 11.2 Champion-Challenger

```yaml
champion_challenger:
  strategy: "shadow_scoring"
  
  champion:
    model_id: "fraud_xgb_v2.3.1"
    deployed_at: "2025-11-01"
    traffic: 100%
    metrics:
      auc_pr: 0.923
      recall: 0.873
      precision: 0.812
      
  challengers:
    - model_id: "fraud_xgb_v2.4.0"
      traffic: 0% (shadow only)
      metrics:
        auc_pr: 0.931
        recall: 0.885
        precision: 0.798
      status: "validating"
      
    - model_id: "fraud_gnn_v1.0.0"
      traffic: 0% (shadow only)
      metrics:
        auc_pr: 0.918
        recall: 0.891
        precision: 0.782
      status: "experimental"
      
  promotion_criteria:
    - "auc_pr >= champion.auc_pr"
    - "recall >= champion.recall - 0.01"
    - "precision >= champion.precision - 0.02"
    - "latency_p99 <= 50ms"
    - "no_drift_7_days"
    - "mrm_approved"
    
  rollout_strategy:
    - step_1: "5% traffic for 24h"
    - step_2: "10% traffic for 24h"
    - step_3: "25% traffic for 24h"
    - step_4: "50% traffic for 48h"
    - step_5: "100% traffic"
    
  rollback_triggers:
    - "latency_p99 > 80ms"
    - "error_rate > 0.5%"
    - "fpr > 3%"
    - "recall < 80%"
```

## 11.3 Retreino Automático

```yaml
retraining:
  schedule:
    # Retreino programado
    - trigger: "scheduled"
      frequency: "weekly"
      day: "sunday"
      time: "02:00"
      
    # Retreino por drift
    - trigger: "drift"
      condition: "psi_score > 0.15"
      cooldown: "24h"
      
    # Retreino por degradação
    - trigger: "performance"
      condition: "auc_pr < 0.88"
      cooldown: "6h"
      
  pipeline:
    data_window: "90 days"
    validation_split: "last 7 days"
    test_split: "last 14 days"
    
    steps:
      1: "fetch_training_data"
      2: "compute_features"
      3: "train_models"
      4: "evaluate_offline"
      5: "register_model"
      6: "deploy_shadow"
      7: "evaluate_online_24h"
      8: "promote_if_better"
      
  guardrails:
    - "minimum_training_samples: 100000"
    - "minimum_fraud_samples: 1000"
    - "maximum_class_imbalance: 100:1"
    - "feature_drift_check: true"
    - "mrm_auto_approval: false"
```

## 11.4 Model Risk Management (MRM)

```
MRM GOVERNANCE FRAMEWORK
========================

┌─────────────────────────────────────────────────────────────────────────────┐
│                           MRM COMMITTEE                                      │
│                                                                             │
│  Members:                                                                   │
│  - Chief Risk Officer (Chair)                                               │
│  - Head of Data Science                                                     │
│  - Head of Fraud Operations                                                 │
│  - Compliance Officer                                                       │
│  - Model Risk Manager                                                       │
│                                                                             │
│  Frequency: Monthly review + Ad-hoc for new models                          │
└─────────────────────────────────────────────────────────────────────────────┘

MODEL LIFECYCLE:

1. DEVELOPMENT
   ├── Business case approval
   ├── Data assessment
   ├── Feature review (ethics/bias)
   └── Development checklist

2. VALIDATION
   ├── Independent model validation
   ├── Backtesting results
   ├── Stress testing
   └── Documentation review

3. APPROVAL
   ├── MRM Committee presentation
   ├── Risk assessment
   ├── Limitations acknowledged
   └── Approval decision

4. IMPLEMENTATION
   ├── UAT sign-off
   ├── Performance benchmarks
   ├── Rollback plan
   └── Go-live checklist

5. MONITORING
   ├── Monthly performance review
   ├── Quarterly revalidation
   ├── Annual comprehensive review
   └── Continuous drift monitoring

6. RETIREMENT
   ├── Replacement plan
   ├── Data archival
   ├── Documentation update
   └── Lessons learned
```

---

# 12. ROADMAP COMPLETO

## 12.1 Visão Geral do Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ROADMAP - MOTOR DE FRAUDE IA/ML                          │
│                         Horizon: 12 meses                                   │
└─────────────────────────────────────────────────────────────────────────────┘

                         90 DIAS              180 DIAS            365 DIAS
                            │                    │                    │
    ────────────────────────┼────────────────────┼────────────────────┼──────
                            │                    │                    │
    ████████████████████████                     │                    │
        FASE 1: MVP         │                    │                    │
         PIX Only           │                    │                    │
      Infra Básica          │                    │                    │
                            │                    │                    │
                            ████████████████████████                  │
                               FASE 2: EXPANSÃO  │                    │
                              Crédito/Débito     │                    │
                              Feature Store      │                    │
                              Backoffice Full    │                    │
                                                 │                    │
                                                 ████████████████████████
                                                    FASE 3: ESCALA
                                                    GNN/Bandits
                                                    Auto-MLOps
                                                    Multi-região
                                                 │                    │
    ────────────────────────┼────────────────────┼────────────────────┼──────
                            │                    │                    │
    ENTREGAS:               │                    │                    │
    - Scoring PIX           │ - Multi-produto    │ - GNN produção     │
    - 3M TPS               │ - 300M TPS         │ - Auto-retrain     │
    - Backoffice MVP       │ - MLOps completo   │ - Multi-AZ         │
    - Recall 80%           │ - Recall 85%       │ - Recall 92%       │
                            │                    │                    │
```

## 12.2 Fase 1: MVP (0-90 dias)

### Sprint 1-2 (Semanas 1-4): Foundation

| Entrega | Owner | Status |
|---------|-------|--------|
| Setup AWS (VPC, EKS, MSK) | Platform | |
| Feature Store básico (Redis) | Data Eng | |
| Pipeline de dados (Kafka→S3) | Data Eng | |
| Modelo XGBoost baseline | Data Science | |
| API de Scoring (Java) | Backend | |

### Sprint 3-4 (Semanas 5-8): Core

| Entrega | Owner | Status |
|---------|-------|--------|
| Integração PIX (entrada) | Backend | |
| Feature engineering (50 features) | Data Science | |
| Policy Engine v1 | Backend | |
| Backoffice MVP (Dashboard) | Frontend | |
| Observabilidade DataDog | SRE | |

### Sprint 5-6 (Semanas 9-12): Polish

| Entrega | Owner | Status |
|---------|-------|--------|
| Tuning de thresholds | Data Science | |
| Playbooks L1/L2 | Fraud Ops | |
| LGPD compliance | Compliance | |
| Load testing (3M TPS) | Performance | |
| Go-live PIX | All | |

### Métricas de Sucesso Fase 1

| Métrica | Target |
|---------|--------|
| Recall | > 80% |
| FPR | < 2.5% |
| Latência p99 | < 100ms |
| Disponibilidade | 99.9% |
| TPS | 3,000+ |

## 12.3 Fase 2: Expansão (91-180 dias)

### Sprint 7-8 (Semanas 13-16): Multi-Produto

| Entrega | Owner | Status |
|---------|-------|--------|
| Integração Crédito | Backend | |
| Integração Débito | Backend | |
| Modelo especializado Crédito | Data Science | |
| Feature Store streaming (Flink) | Data Eng | |

### Sprint 9-10 (Semanas 17-20): MLOps

| Entrega | Owner | Status |
|---------|-------|--------|
| Pipeline de retreino | ML Eng | |
| Champion-Challenger | ML Eng | |
| Monitoring de drift | Data Science | |
| A/B testing framework | Data Science | |

### Sprint 11-12 (Semanas 21-24): Operations

| Entrega | Owner | Status |
|---------|-------|--------|
| Backoffice completo | Frontend | |
| Caso investigação | Frontend | |
| Regras dinâmicas | Backend | |
| STEP_UP inteligente | Mobile | |
| MED automation | Backend | |

### Métricas de Sucesso Fase 2

| Métrica | Target |
|---------|--------|
| Recall | > 85% |
| FPR | < 2% |
| Latência p99 | < 60ms |
| Disponibilidade | 99.95% |
| TPS | 10,000+ |

## 12.4 Fase 3: Escala (181-365 dias)

### Sprint 13-16 (Semanas 25-32): Advanced ML

| Entrega | Owner | Status |
|---------|-------|--------|
| GNN em produção | Data Science | |
| Multi-Armed Bandits (STEP_UP) | Data Science | |
| Anomaly Detection | Data Science | |
| Causal Inference | Data Science | |

### Sprint 17-20 (Semanas 33-40): Scale & Resilience

| Entrega | Owner | Status |
|---------|-------|--------|
| Multi-AZ deployment | Platform | |
| DR/BCP automation | SRE | |
| Auto-scaling avançado | Platform | |
| Global CDN | Platform | |

### Sprint 21-24 (Semanas 41-48): Excellence

| Entrega | Owner | Status |
|---------|-------|--------|
| Auto-MLOps (trigger retrain) | ML Eng | |
| Self-healing alerts | SRE | |
| Adversarial defense | Security | |
| Compliance automation | Compliance | |

### Métricas de Sucesso Fase 3

| Métrica | Target |
|---------|--------|
| Recall | > 92% |
| FPR | < 1.5% |
| Latência p99 | < 50ms |
| Disponibilidade | 99.99% |
| TPS | 15,000+ |

## 12.5 Estrutura de Squads

```
ORGANIZAÇÃO DAS SQUADS
======================

┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRIBE: FRAUD PREVENTION                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  SQUAD: SCORING     │  │  SQUAD: DATA        │  │  SQUAD: PLATFORM    │
│                     │  │                     │  │                     │
│  - 2 Backend (Java) │  │  - 2 Data Engineers │  │  - 2 Platform Eng   │
│  - 2 Data Scientists│  │  - 1 ML Engineer    │  │  - 1 SRE            │
│  - 1 QA             │  │  - 1 Data Analyst   │  │  - 1 DevOps         │
│  - 1 Tech Lead      │  │  - 1 Tech Lead      │  │  - 1 Tech Lead      │
│                     │  │                     │  │                     │
│  Owner: API Scoring │  │  Owner: Feature     │  │  Owner: Infra AWS   │
│         Policy Eng  │  │         Store       │  │         MLOps       │
│         STEP_UP     │  │         Pipelines   │  │         Observ.     │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘

┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  SQUAD: BACKOFFICE  │  │  SQUAD: FRAUD OPS   │  │  SQUAD: COMPLIANCE  │
│                     │  │                     │  │                     │
│  - 2 Frontend React │  │  - 1 Fraud Strategy │  │  - 1 Compliance Off │
│  - 1 UX Designer    │  │  - 2 Sr Analysts    │  │  - 1 Privacy Eng    │
│  - 1 Backend (BFF)  │  │  - 1 L&D Specialist │  │  - 1 Legal          │
│  - 1 Tech Lead      │  │  - 1 Team Lead      │  │  - 1 DPO            │
│                     │  │                     │  │                     │
│  Owner: Dashboard   │  │  Owner: Playbooks   │  │  Owner: LGPD        │
│         Cases       │  │         Training    │  │         BACEN       │
│         Rules UI    │  │         Operations  │  │         Audit       │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘

TOTAL HEADCOUNT: ~35 pessoas (Fase 1-2)
                 ~50 pessoas (Fase 3)
```

---

# 13. RISCOS & MITIGAÇÕES

## 13.1 Matriz de Riscos

| ID | Risco | Probabilidade | Impacto | Severidade | Mitigação |
|----|-------|---------------|---------|------------|-----------|
| R1 | Modelo com baixo recall inicial | Alta | Alto | Crítico | Threshold conservador + revisão manual intensiva |
| R2 | Latência acima do SLO | Média | Alto | Alto | Feature store otimizado + caching agressivo |
| R3 | Drift do modelo não detectado | Média | Alto | Alto | Monitoring contínuo + alertas PSI/KS |
| R4 | Fraude adaptativa (evasão) | Alta | Alto | Crítico | Adversarial training + features dinâmicas |
| R5 | Data quality issues | Média | Médio | Médio | Validação de schema + data contracts |
| R6 | Indisponibilidade AWS | Baixa | Crítico | Alto | Multi-AZ + DR automation |
| R7 | Vazamento de dados (LGPD) | Baixa | Crítico | Alto | Criptografia + mascaramento + audit |
| R8 | Falsos positivos excessivos | Média | Alto | Alto | Threshold tuning + feedback loop |
| R9 | Escalation de incidentes | Média | Médio | Médio | Runbooks + on-call treinado |
| R10 | Key person dependency | Média | Médio | Médio | Documentação + cross-training |

## 13.2 Planos de Contingência

### R1: Baixo Recall Inicial

```yaml
contingency:
  trigger: "Recall < 75% em produção"
  
  immediate_actions:
    - Reduzir threshold de bloqueio (0.85 → 0.75)
    - Aumentar cobertura de STEP_UP
    - Ativar revisão manual para transações > R$ 5K
    
  short_term:
    - Análise de casos perdidos
    - Feature engineering adicional
    - Retreino com dados recentes
    
  owner: "Data Science Lead"
  escalation: "CRO"
```

### R4: Fraude Adaptativa

```yaml
contingency:
  trigger: "Padrão de evasão detectado"
  
  detection:
    - Drop em recall sem mudança de distribuição
    - Aumento de fraude pós-STEP_UP
    - Novos TTPs identificados pelo SOC
    
  response:
    - Ativar regras temporárias de contenção
    - Análise forense dos casos
    - Feature engineering para novo padrão
    - Adversarial training do modelo
    
  prevention:
    - Rotação regular de features (ofuscação)
    - Monitoramento de probing
    - Threat intelligence ativo
    
  owner: "Security + Data Science"
```

### R6: Indisponibilidade AWS

```yaml
contingency:
  trigger: "Região AWS indisponível"
  
  rpo: "5 minutos"
  rto: "30 minutos"
  
  failover:
    - DNS failover automático (Route53)
    - Réplica Aurora em outra região
    - S3 cross-region replication
    - Kafka MirrorMaker
    
  degraded_mode:
    - Rules-only scoring (sem ML)
    - Cache Redis local
    - Fila de transações pendentes
    
  owner: "Platform + SRE"
```

---

# 14. STACK TÉCNICA RECOMENDADA

## 14.1 Resumo da Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RECOMMENDED TECH STACK                               │
└─────────────────────────────────────────────────────────────────────────────┘

CAMADA              TECNOLOGIA                   JUSTIFICATIVA
──────────────────────────────────────────────────────────────────────────────

BACKEND API         Java 21 + Spring Boot 3.x   Performance, ecosystem, type-safety
                    + Virtual Threads            Melhor uso de recursos

BFF (Gateway)       Node.js 20 + Fastify        Rápido para I/O, GraphQL friendly

ML SERVING          ONNX Runtime (Java)          Latência mínima, portabilidade

ML TRAINING         Python 3.11 + XGBoost       Ecossistema ML maduro
                    + LightGBM + PyTorch

STREAMING           Apache Kafka (MSK)           Proven at scale, exactly-once
                    + Apache Flink               Janelas complexas, state

FEATURE STORE       Redis Cluster                Sub-ms latency
                    + S3 (offline)               Custo-efetivo para histórico

DATABASE            Aurora PostgreSQL            ACID, performance, managed

CACHE               Redis (ElastiCache)          In-memory, pub/sub

SEARCH/LOGS         OpenSearch                   Full-text, analytics

FRONTEND            React 18 + Next.js 14       SSR, performance, DX
                    + Tailwind + Shadcn/ui       Componentes modernos

MOBILE SDK          React Native                 Cross-platform STEP_UP

OBSERVABILITY       DataDog (APM, Logs,          Unified platform
                    Metrics, Traces)

INFRA AS CODE       Terraform + Terragrunt      Multi-environment, state mgmt

CI/CD               GitHub Actions               Native, simples
                    + ArgoCD                     GitOps para K8s

CONTAINER           EKS (Kubernetes)             Orquestração madura
                    + Karpenter                  Auto-scaling eficiente

SECRETS             AWS Secrets Manager          Rotation automática
                    + HashiCorp Vault            Políticas avançadas
```

## 14.2 Versões Específicas

```yaml
versions:
  # Backend
  java: "21.0.1"
  spring_boot: "3.2.0"
  gradle: "8.5"
  
  # Node
  node: "20.10.0"
  fastify: "4.24.0"
  
  # Python
  python: "3.11.6"
  xgboost: "2.0.2"
  lightgbm: "4.1.0"
  pytorch: "2.1.0"
  scikit_learn: "1.3.2"
  
  # Frontend
  react: "18.2.0"
  next: "14.0.3"
  typescript: "5.3.2"
  tailwind: "3.3.5"
  
  # Data
  kafka: "3.6.0"
  flink: "1.18.0"
  redis: "7.2"
  postgresql: "15.4"
  
  # Infra
  kubernetes: "1.28"
  terraform: "1.6.4"
  datadog_agent: "7.49"
```

---

# 15. EVOLUÇÃO DE LONGO PRAZO

## 15.1 Roadmap v2.0 (Ano 2)

```yaml
v2_features:
  advanced_ml:
    - "Transformer-based sequence models"
    - "Real-time GNN updates"
    - "Federated learning (multi-banco)"
    - "Reinforcement Learning para políticas"
    
  operations:
    - "Fully automated L1 triage"
    - "AI-assisted investigation"
    - "Predictive case routing"
    - "Self-healing rules"
    
  compliance:
    - "Automated regulatory reporting"
    - "Real-time COAF integration"
    - "Cross-border fraud detection"
    
  platform:
    - "Multi-region active-active"
    - "Edge scoring (latam)"
    - "Serverless inference"
```

## 15.2 Roadmap v3.0 (Ano 3)

```yaml
v3_vision:
  intelligence:
    - "Autonomous fraud prevention"
    - "Predictive blocking (before transaction)"
    - "Network-level protection"
    
  ecosystem:
    - "Consortium fraud sharing"
    - "Open banking integration"
    - "Merchant fraud protection"
    
  technology:
    - "Quantum-resistant encryption"
    - "Homomorphic encryption for ML"
    - "Zero-knowledge proofs for privacy"
```

## 15.3 Métricas de Evolução

| Versão | Recall | FPR | Latência | Automação |
|--------|--------|-----|----------|-----------|
| v1.0 | 80% | 2.5% | 100ms | 70% |
| v1.5 | 85% | 2.0% | 60ms | 80% |
| v2.0 | 92% | 1.5% | 50ms | 90% |
| v2.5 | 95% | 1.0% | 40ms | 95% |
| v3.0 | 98% | 0.5% | 30ms | 99% |

---

# CONCLUSÃO

Este Blueprint representa uma visão completa e detalhada para construir um Motor de Fraude de classe mundial, capaz de:

- Processar **300 milhões de transações/dia** com latência < 50ms
- Detectar fraudes com **Recall > 85%** e **FPR < 2%**
- Garantir **conformidade total** com LGPD, BACEN e PCI DSS
- Operar **24/7/365** com disponibilidade 99.99%
- Evoluir continuamente com **MLOps automatizado**

O investimento total estimado de **R$ 10 milhões em 12 meses** terá ROI de **8:1**, prevenindo perdas de mais de **R$ 80 milhões/ano** em fraudes.

---

**Documento elaborado pelo Conselho Global de Especialistas em Fraude Bancária, IA/ML e Arquitetura de Sistemas**

*Versão 1.0.0 | Novembro 2025*
