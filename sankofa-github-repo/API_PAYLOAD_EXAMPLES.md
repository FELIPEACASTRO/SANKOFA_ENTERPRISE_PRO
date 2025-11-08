# 📡 PAYLOAD DE ENTRADA - SANKOFA ENTERPRISE PRO API

**Data**: 08 de Novembro de 2025  
**Versão da API**: 1.0.0  

---

## 📋 SUMÁRIO

Este documento descreve os payloads de entrada esperados pela API de detecção de fraude do SANKOFA_ENTERPRISE_PRO.

---

## 🔍 ENDPOINT PRINCIPAL: `/api/fraud/predict`

### Método
`POST`

### Headers
```http
Content-Type: application/json
```

### Payload de Entrada

```json
{
  "transactions": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "timestamp": "2025-11-08T14:30:00",
      "value": 1500.00,
      "transaction_type": "PIX",
      "channel": "MOBILE",
      "city": "SP_City_1",
      "state": "SP",
      "country": "BR",
      "ip_address": "192.168.1.100",
      "device_id": "device_42",
      "receiver_account": "merchant_5678",
      "client_cpf": "12345678901"
    }
  ]
}
```

### Campos Obrigatórios

| Campo | Tipo | Descrição | Exemplo |
|-------|------|-----------|---------|
| `transactions` | Array | Lista de transações a serem analisadas | `[{...}]` |

### Estrutura de Cada Transação

| Campo | Tipo | Obrigatório | Descrição | Exemplo |
|-------|------|-------------|-----------|---------|
| `id` | String (UUID) | Sim | Identificador único da transação | `"550e8400-e29b-41d4-a716-446655440000"` |
| `timestamp` | String (ISO 8601) | Sim | Data e hora da transação | `"2025-11-08T14:30:00"` |
| `value` | Float | Sim | Valor da transação em reais (BRL) | `1500.00` |
| `transaction_type` | String | Sim | Tipo de transação | `"PIX"`, `"CREDITO"`, `"DEBITO"`, `"BOLETO"` |
| `channel` | String | Sim | Canal de origem | `"MOBILE"`, `"WEB"`, `"POS"`, `"ATM"` |
| `city` | String | Sim | Cidade de origem | `"SP_City_1"` |
| `state` | String | Sim | Estado (UF) | `"SP"`, `"RJ"`, `"MG"`, etc. |
| `country` | String | Sim | País (código ISO) | `"BR"` |
| `ip_address` | String | Sim | Endereço IP do cliente | `"192.168.1.100"` |
| `device_id` | String | Sim | Identificador do dispositivo | `"device_42"` |
| `receiver_account` | String | Sim | Conta de destino/merchant | `"merchant_5678"` |
| `client_cpf` | String | Sim | CPF do cliente (11 dígitos) | `"12345678901"` |

---

## 📤 PAYLOAD DE SAÍDA (Resposta)

### Resposta de Sucesso (200 OK)

```json
{
  "success": true,
  "data": {
    "predictions": [
      {
        "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
        "is_fraud": false,
        "fraud_probability": 0.23,
        "risk_score": 0.23,
        "risk_level": "LOW",
        "confidence": 0.89,
        "processing_time_ms": 12.5,
        "model_version": "1.0.0",
        "detection_reason": [
          "Normal transaction amount",
          "Known device",
          "Standard business hours"
        ],
        "timestamp": "2025-11-08T14:30:05Z"
      }
    ],
    "summary": {
      "total": 1,
      "frauds_detected": 0,
      "avg_risk_score": 0.23,
      "model_version": "1.0.0"
    }
  }
}
```

### Campos da Resposta

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `success` | Boolean | Indica se a requisição foi bem-sucedida |
| `data.predictions` | Array | Lista de predições, uma para cada transação |
| `data.summary` | Object | Resumo das predições |

### Estrutura de Cada Predição

| Campo | Tipo | Descrição | Valores Possíveis |
|-------|------|-----------|-------------------|
| `transaction_id` | String | ID da transação analisada | UUID |
| `is_fraud` | Boolean | Indica se é fraude | `true`, `false` |
| `fraud_probability` | Float | Probabilidade de fraude (0-1) | `0.0` a `1.0` |
| `risk_score` | Float | Score de risco (0-1) | `0.0` a `1.0` |
| `risk_level` | String | Nível de risco | `"LOW"`, `"MEDIUM"`, `"HIGH"`, `"CRITICAL"` |
| `confidence` | Float | Confiança da predição (0-1) | `0.0` a `1.0` |
| `processing_time_ms` | Float | Tempo de processamento em ms | `12.5` |
| `model_version` | String | Versão do modelo usado | `"1.0.0"` |
| `detection_reason` | Array[String] | Razões da detecção | Lista de strings |
| `timestamp` | String | Timestamp da análise | ISO 8601 |

---

## 🔴 EXEMPLOS DE TRANSAÇÕES FRAUDULENTAS

### Exemplo 1: Transação de Alto Valor à Noite

```json
{
  "transactions": [
    {
      "id": "fraud-001",
      "timestamp": "2025-11-08T02:30:00",
      "value": 15000.00,
      "transaction_type": "PIX",
      "channel": "MOBILE",
      "city": "SP_City_1",
      "state": "SP",
      "country": "BR",
      "ip_address": "192.168.1.100",
      "device_id": "new_unregistered_device",
      "receiver_account": "merchant_9999",
      "client_cpf": "12345678901"
    }
  ]
}
```

**Resposta Esperada:**
```json
{
  "success": true,
  "data": {
    "predictions": [
      {
        "transaction_id": "fraud-001",
        "is_fraud": true,
        "fraud_probability": 0.92,
        "risk_score": 0.92,
        "risk_level": "CRITICAL",
        "confidence": 0.95,
        "processing_time_ms": 15.2,
        "model_version": "1.0.0",
        "detection_reason": [
          "High value transaction (>10000)",
          "Unusual hour (02:30)",
          "New/unregistered device",
          "Suspicious receiver account"
        ],
        "timestamp": "2025-11-08T02:30:05Z"
      }
    ],
    "summary": {
      "total": 1,
      "frauds_detected": 1,
      "avg_risk_score": 0.92,
      "model_version": "1.0.0"
    }
  }
}
```

### Exemplo 2: Múltiplas Transações PIX Pequenas

```json
{
  "transactions": [
    {
      "id": "fraud-002",
      "timestamp": "2025-11-08T10:15:00",
      "value": 50.00,
      "transaction_type": "PIX",
      "channel": "MOBILE",
      "city": "RJ_City_2",
      "state": "RJ",
      "country": "BR",
      "ip_address": "10.0.0.50",
      "device_id": "new_unregistered_device",
      "receiver_account": "merchant_1234",
      "client_cpf": "98765432100"
    }
  ]
}
```

---

## 📊 ENDPOINT DE LOTE: `/api/fraud/batch`

Para processar **grandes volumes** de transações (>1000), use o endpoint de lote:

### Método
`POST`

### Payload de Entrada

```json
{
  "transactions": [
    { /* transação 1 */ },
    { /* transação 2 */ },
    { /* ... */ },
    { /* transação N */ }
  ],
  "options": {
    "parallel": true,
    "batch_size": 1000
  }
}
```

### Campos Adicionais

| Campo | Tipo | Obrigatório | Descrição | Padrão |
|-------|------|-------------|-----------|--------|
| `options.parallel` | Boolean | Não | Processar em paralelo | `true` |
| `options.batch_size` | Integer | Não | Tamanho do lote | `1000` |

---

## ⚠️ TRATAMENTO DE ERROS

### Erro de Validação (400 Bad Request)

```json
{
  "success": false,
  "error": {
    "id": "ERR_20251108_143000_001",
    "category": "VALIDATION",
    "severity": "MEDIUM",
    "message": "transactions field is required",
    "recovery_action": "Please provide a valid 'transactions' array in the request body"
  }
}
```

### Erro de Modelo Não Treinado (500 Internal Server Error)

```json
{
  "success": false,
  "error": {
    "id": "ERR_20251108_143000_002",
    "category": "ML_MODEL",
    "severity": "HIGH",
    "message": "Fraud detection model is not trained. Please train the model first.",
    "recovery_action": "Contact system administrator to train the model"
  }
}
```

---

## 🧪 EXEMPLOS DE TESTE (cURL)

### Transação Legítima

```bash
curl -X POST http://localhost:8445/api/fraud/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "id": "test-001",
        "timestamp": "2025-11-08T14:30:00",
        "value": 150.00,
        "transaction_type": "DEBITO",
        "channel": "POS",
        "city": "SP_City_1",
        "state": "SP",
        "country": "BR",
        "ip_address": "192.168.1.100",
        "device_id": "device_10",
        "receiver_account": "merchant_5000",
        "client_cpf": "12345678901"
      }
    ]
  }'
```

### Transação Fraudulenta

```bash
curl -X POST http://localhost:8445/api/fraud/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "id": "test-fraud-001",
        "timestamp": "2025-11-08T03:00:00",
        "value": 20000.00,
        "transaction_type": "PIX",
        "channel": "MOBILE",
        "city": "SP_City_1",
        "state": "SP",
        "country": "BR",
        "ip_address": "192.168.1.100",
        "device_id": "new_device",
        "receiver_account": "suspicious_account",
        "client_cpf": "12345678901"
      }
    ]
  }'
```

---

## 📚 REFERÊNCIAS

- **Documentação da API**: `/api/docs` (quando disponível)
- **Health Check**: `GET /api/health`
- **Status**: `GET /api/status`

---

**Documento preparado por**: Análise Automatizada  
**Data**: 08 de Novembro de 2025  
**Versão**: 1.0  
