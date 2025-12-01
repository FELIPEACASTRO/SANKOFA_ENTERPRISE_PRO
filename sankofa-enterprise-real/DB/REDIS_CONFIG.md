# Redis - Configuração e Estrutura de Cache

**Versão:** 2.0.0  
**Última Atualização:** Dezembro 2025  
**Status:** Produção

---

## Visão Geral

O Sankofa Enterprise Pro utiliza Redis como camada de cache distribuído para alta performance.
Quando Redis não está disponível, o sistema usa automaticamente cache em memória como fallback.

### Benefícios do Cache
- **Latência:** Reduz tempo de resposta de 200ms para <50ms
- **Throughput:** Suporta 300M+ transações/dia
- **Disponibilidade:** Fallback automático para cache em memória

---

## Variáveis de Ambiente

### Para Upstash (Recomendado para Replit)
```bash
REDIS_HOST=rediss://default:senha@endpoint.upstash.io:6379
REDIS_SSL=true
```

### Para Redis Local/Docker
```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=sua_senha_segura
REDIS_DB=0
REDIS_SSL=false
```

### Variáveis Completas
```bash
# Conexão
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=sua_senha_aqui
REDIS_DB=0
REDIS_SSL=false

# Pool de Conexões
REDIS_MAX_CONNECTIONS=50
REDIS_MIN_IDLE=5
REDIS_TIMEOUT=5000

# TTL Padrão
REDIS_DEFAULT_TTL=3600
```

---

## Estrutura de Chaves

### 1. Cache de Regras Duras (Crítico)
```
sankofa:rules:hard_rules           TTL: 30s    # 216 regras ativas
sankofa:rules:vip_list             TTL: 60s    # Lista branca
sankofa:rules:hot_list             TTL: 60s    # Lista negra
sankofa:rules:metadata             TTL: 300s   # Campos, operadores, ações
```

### 2. Cache de Dashboard
```
sankofa:dashboard:kpis             TTL: 30s    # KPIs principais
sankofa:dashboard:timeseries       TTL: 30s    # Dados de gráficos
sankofa:dashboard:channels         TTL: 30s    # Distribuição por canal
sankofa:dashboard:alerts           TTL: 30s    # Alertas recentes
sankofa:dashboard:model_status     TTL: 60s    # Status dos modelos ML
```

### 3. Cache de Transações
```
sankofa:tx:{transaction_id}        TTL: 60s    # Transação individual
sankofa:tx:customer:{cpf_hash}     TTL: 300s   # Histórico do cliente
sankofa:tx:velocity:{cpf_hash}     TTL: 3600s  # Contadores de velocidade
```

### 4. Cache de Sessões
```
sankofa:session:{session_id}       TTL: 3600s  # Sessão do usuário (1h)
sankofa:session:user:{user_id}     TTL: 3600s  # Sessões por usuário
```

### 5. Rate Limiting
```
sankofa:rate:{ip_address}          TTL: 60s    # Limite por IP
sankofa:rate:api:{api_key}         TTL: 60s    # Limite por API key
sankofa:rate:user:{user_id}        TTL: 60s    # Limite por usuário
```

### 6. Cache de ML
```
sankofa:ml:model_status            TTL: 300s   # Status dos modelos
sankofa:ml:features:{customer_id}  TTL: 600s   # Features calculadas
sankofa:ml:predictions:{tx_id}     TTL: 60s    # Predições recentes
```

### 7. Cache de Compliance
```
sankofa:compliance:bacen_limits    TTL: 3600s  # Limites BACEN
sankofa:compliance:pix_limits      TTL: 3600s  # Limites PIX noturnos
```

---

## Configuração de Produção

### Upstash (Serverless Redis)
```python
import redis

# Conexão Upstash
redis_client = redis.from_url(
    os.environ.get('REDIS_HOST'),
    decode_responses=True,
    ssl=True
)
```

### Redis Cluster (Alta Disponibilidade)
```python
from redis.cluster import RedisCluster

redis_client = RedisCluster(
    host="cluster.redis.local",
    port=6379,
    password="senha_segura",
    decode_responses=True
)
```

---

## Fallback: Cache em Memória

Quando Redis não está configurado, o sistema usa `SimpleCache`:

```python
class SimpleCache:
    """
    Cache em memória com TTL e LRU eviction.
    - Limite: 10.000 entradas
    - Política: LRU (Least Recently Used)
    - Thread-safe: Sim
    """
    def __init__(self, max_size=10000, default_ttl=30):
        self._cache = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
```

### Características do Fallback
| Recurso | Redis | InMemoryCache |
|---------|-------|---------------|
| Persistência | Sim | Não |
| Distribuído | Sim | Não |
| TTL | Sim | Sim |
| Eviction | Configurável | LRU |
| Limite | Ilimitado | 10.000 |

---

## Integração com Sankofa

### Carregamento Automático
```python
from cache.redis_cache_system import CacheConfig, get_cache

# Carrega automaticamente de variáveis de ambiente
cache = get_cache()

# Uso básico
cache.set("key", value, ttl=60)
value = cache.get("key")

# Batch operations
cache.mset({"k1": v1, "k2": v2})
values = cache.mget(["k1", "k2"])
```

### Invalidação de Cache
```python
# Invalidar chave específica
cache.delete("sankofa:dashboard:kpis")

# Invalidar padrão (wildcard)
cache.delete_pattern("sankofa:tx:*")

# Invalidar tudo
cache.flush()
```

---

## Comandos Úteis

### Verificar Status
```bash
redis-cli ping                          # PONG
redis-cli info server                   # Versão, uptime
redis-cli info memory                   # Uso de memória
redis-cli info stats                    # Estatísticas
```

### Monitorar
```bash
redis-cli MONITOR                       # Ver comandos em tempo real
redis-cli CLIENT LIST                   # Conexões ativas
redis-cli SLOWLOG GET 10                # Comandos lentos
```

### Gerenciar Chaves
```bash
redis-cli KEYS "sankofa:*"              # Listar chaves Sankofa
redis-cli TTL "sankofa:rules:hard_rules" # Ver TTL restante
redis-cli TYPE "sankofa:rules:hard_rules" # Tipo da chave
redis-cli MEMORY USAGE "key"            # Memória usada
```

### Limpar Cache
```bash
redis-cli FLUSHDB                       # Limpa DB atual
redis-cli DEL $(redis-cli KEYS "sankofa:rules:*")  # Limpa cache de regras
redis-cli DEL $(redis-cli KEYS "sankofa:dashboard:*")  # Limpa dashboard
```

---

## Performance

### Latência Esperada
| Operação | Latência | Throughput |
|----------|----------|------------|
| GET | < 1ms | 100K ops/s |
| SET | < 1ms | 100K ops/s |
| MGET (batch) | < 5ms | 50K ops/s |
| Cache hit | < 0.5ms | 200K ops/s |
| Cache miss + DB | < 50ms | 10K ops/s |

### Taxa de Hit Esperada
| Recurso | Hit Rate | Impacto |
|---------|----------|---------|
| Hard Rules | 99%+ | Crítico |
| Dashboard KPIs | 95%+ | Alto |
| Transações | 80%+ | Médio |
| Sessões | 99%+ | Crítico |

---

## Alertas e Monitoramento

### Métricas a Monitorar
```python
# Via API
GET /api/observability/cache-stats

# Resposta
{
    "hit_rate": 0.95,
    "miss_rate": 0.05,
    "memory_used_mb": 128,
    "keys_count": 5432,
    "evictions_count": 12,
    "connections_active": 10
}
```

### Alertas Críticos
| Condição | Ação |
|----------|------|
| Hit rate < 80% | Investigar queries lentas |
| Memory > 80% | Aumentar memória ou TTL |
| Evictions alto | Aumentar memória |
| Latência > 10ms | Verificar rede/CPU |

---

## Troubleshooting

### Redis não conecta
```bash
# Verificar se Redis está rodando
redis-cli ping

# Verificar variáveis de ambiente
echo $REDIS_HOST $REDIS_PORT

# Testar conexão manual
redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD ping
```

### Cache não atualiza
```bash
# Forçar invalidação
redis-cli DEL "sankofa:rules:hard_rules"

# Via API
curl -X POST http://localhost:5000/api/cache/invalidate
```

### Memória alta
```bash
# Verificar uso
redis-cli INFO memory

# Limpar chaves expiradas
redis-cli DEBUG SLEEP 0

# Analisar chaves grandes
redis-cli --bigkeys
```

---

## Segurança

### Boas Práticas
1. **Sempre use senha** em produção
2. **TLS/SSL** para conexões remotas
3. **Limite comandos** perigosos (FLUSHALL, DEBUG)
4. **Monitore** conexões suspeitas
5. **Backup** antes de manutenção

### Configuração Segura
```bash
# redis.conf
requirepass sua_senha_muito_segura
rename-command FLUSHALL ""
rename-command DEBUG ""
maxmemory 256mb
maxmemory-policy allkeys-lru
```

---

## Referências

- [Redis Documentation](https://redis.io/documentation)
- [Upstash Docs](https://docs.upstash.com/)
- [Redis Best Practices](https://redis.io/docs/management/optimization/)
