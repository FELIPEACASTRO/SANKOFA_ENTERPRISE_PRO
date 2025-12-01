# Redis - Configuração e Estrutura de Cache

## Overview
O Sankofa Enterprise Pro utiliza Redis como camada de cache distribuído para alta performance.
Quando Redis não está disponível, o sistema usa cache em memória como fallback.

## Variáveis de Ambiente

```bash
# Conexão Redis (opcional - sistema funciona sem Redis)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=sua_senha_aqui
REDIS_DB=0
REDIS_SSL=false

# Para Upstash ou serviços cloud:
# REDIS_HOST=rediss://default:senha@host.upstash.io:6379
```

## Estrutura de Chaves

### 1. Cache de Regras Duras
```
sankofa:rules:hard_rules           TTL: 30s
sankofa:rules:vip_list             TTL: 60s
sankofa:rules:hot_list             TTL: 60s
```

### 2. Cache de Dashboard
```
sankofa:dashboard:kpis             TTL: 30s
sankofa:dashboard:timeseries       TTL: 30s
sankofa:dashboard:channels         TTL: 30s
sankofa:dashboard:alerts           TTL: 30s
```

### 3. Cache de Transações
```
sankofa:tx:{transaction_id}        TTL: 60s
sankofa:tx:customer:{cpf_hash}     TTL: 300s
```

### 4. Cache de Sessões
```
sankofa:session:{session_id}       TTL: 3600s (1 hora)
```

### 5. Rate Limiting
```
sankofa:rate:{ip_address}          TTL: 60s
sankofa:rate:api:{api_key}         TTL: 60s
```

### 6. Cache de ML
```
sankofa:ml:model_status            TTL: 300s
sankofa:ml:features:{customer_id}  TTL: 600s
```

## Configuração de Produção

### Para Upstash (Recomendado para Replit)
```python
# Variáveis necessárias:
REDIS_HOST=rediss://default:senha@endpoint.upstash.io:6379
REDIS_SSL=true
```

### Para Redis Local/Docker
```python
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=senha_segura
REDIS_DB=0
```

## Fallback: Cache em Memória

Quando Redis não está configurado, o sistema usa `InMemoryCache`:
- Limite de 10.000 entradas
- Política LRU (Least Recently Used) para eviction
- TTL configurável por chave
- Thread-safe com locks

## Comandos Úteis

### Verificar Status
```bash
redis-cli ping
redis-cli info
```

### Limpar Cache
```bash
redis-cli FLUSHDB          # Limpa DB atual
redis-cli KEYS "sankofa:*" # Lista chaves Sankofa
redis-cli DEL $(redis-cli KEYS "sankofa:rules:*")  # Limpa cache de regras
```

### Monitorar
```bash
redis-cli MONITOR          # Ver comandos em tempo real
redis-cli INFO memory      # Uso de memória
```

## Performance

| Operação | Latência Esperada |
|----------|-------------------|
| GET | < 1ms |
| SET | < 1ms |
| MGET (batch) | < 5ms |
| Cache hit | < 0.5ms |

## Integração com Sankofa

O sistema carrega automaticamente a configuração do Redis:

```python
from cache.redis_cache_system import redis_cache_system, CacheConfig

# Carregar de variáveis de ambiente
config = CacheConfig.from_env()

# Ou configurar manualmente
config = CacheConfig(
    host="localhost",
    port=6379,
    password="senha",
    db=0,
    default_ttl=3600
)
```
