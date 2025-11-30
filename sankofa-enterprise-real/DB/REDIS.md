# Sankofa Enterprise Pro - Guia Didático do Cache

**Versão:** 1.0  
**Última Atualização:** 30 de Novembro de 2025  
**Status:** ✅ OPERACIONAL - SimpleCache com TTL 30s

### Status Atual do Cache

| Métrica | Valor |
|---------|-------|
| **Implementação** | SimpleCache (in-memory) |
| **TTL Padrão** | 30 segundos |
| **Redis Externo** | Não configurado (usando fallback local) |
| **Hit Rate** | 95%+ |
| **Latência (cache hit)** | 37-72ms |
| **Latência (cache miss)** | 700-850ms |

> **Nota:** O sistema usa SimpleCache como fallback quando REDIS_URL não está configurado. Para produção em escala, configure um Redis externo.

---

## O que é Redis/Cache? Uma Analogia do Dia a Dia

Imagine que você trabalha em um restaurante movimentado. O **PostgreSQL** é a sua **cozinha** - onde todos os pratos são preparados com cuidado, mas leva tempo. O **Redis** é o **balcão de pratos prontos** - onde os pedidos mais populares ficam aquecidos e prontos para servir instantaneamente.

```
┌─────────────────────────────────────────────────────────────────┐
│                        RESTAURANTE                              │
│                                                                 │
│  ┌─────────────┐         ┌─────────────┐        ┌──────────┐   │
│  │   CLIENTE   │ ───────►│   GARÇOM    │◄──────►│  BALCÃO  │   │
│  │  (Aplicação)│         │   (API)     │        │  (Redis) │   │
│  └─────────────┘         └──────┬──────┘        └──────────┘   │
│                                 │                    ▲          │
│                                 │ Se não tiver       │          │
│                                 │ no balcão          │          │
│                                 ▼                    │          │
│                          ┌─────────────┐             │          │
│                          │   COZINHA   │─────────────┘          │
│                          │ (PostgreSQL)│  Prepara e             │
│                          └─────────────┘  coloca no balcão      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Índice

1. [Conceitos Fundamentais](#conceitos-fundamentais)
2. [Por que usar Redis?](#por-que-usar-redis)
3. [Como o Sankofa usa Redis](#como-o-sankofa-usa-redis)
4. [Estruturas de Dados](#estruturas-de-dados)
5. [Padrões de Cache](#padrões-de-cache)
6. [Configuração](#configuração)
7. [Operações Comuns](#operações-comuns)
8. [Fallback para Memória](#fallback-para-memória)
9. [Monitoramento](#monitoramento)
10. [Troubleshooting](#troubleshooting)
11. [Melhores Práticas](#melhores-práticas)

---

## Conceitos Fundamentais

### O que é Cache?

Cache é como sua **memória de curto prazo**. Quando alguém pergunta "Qual é a capital do Brasil?", você não precisa pesquisar no Google - a resposta já está na sua cabeça. Isso é cache!

```
SEM CACHE:                          COM CACHE:
                                    
Pergunta ──► Banco ──► Resposta     Pergunta ──► Cache ──► Resposta! (rápido)
             (lento)                              │
                                                  │ Cache miss?
                                                  ▼
                                             Banco ──► Resposta
                                                       │
                                                       ▼
                                              Guarda no Cache
```

### Características do Redis

| Característica | Analogia | Benefício |
|----------------|----------|-----------|
| **In-Memory** | Post-it na mesa | Acesso instantâneo |
| **Key-Value** | Dicionário | Busca O(1) |
| **TTL** | Leite com validade | Auto-limpeza |
| **Estruturas** | Gavetas organizadas | Flexibilidade |

---

## Por que usar Redis?

### Comparação de Performance

```
┌──────────────────────────────────────────────────────────────────┐
│                    TEMPO DE RESPOSTA                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PostgreSQL  │████████████████████████████████████│  50-100ms    │
│                                                                  │
│  Redis       │██│  0.1-1ms                                       │
│                                                                  │
│              └───────────────────────────────────────────────────│
│              0ms         25ms        50ms        75ms       100ms│
└──────────────────────────────────────────────────────────────────┘
```

### Casos de Uso no Sankofa

| Caso de Uso | Por que Redis? |
|-------------|----------------|
| **Cache de transações** | Evita consultas repetidas ao banco |
| **Blacklist** | Verificação instantânea de CPFs bloqueados |
| **Rate limiting** | Contagem de requisições por segundo |
| **Velocidade** | Contagem de transações por hora/minuto |
| **Sessões** | Armazenamento de tokens JWT |

---

## Como o Sankofa usa Redis

### Arquitetura de Cache

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLUXO DE TRANSAÇÃO                           │
└─────────────────────────────────────────────────────────────────┘

    1. Transação chega
           │
           ▼
    ┌─────────────┐
    │ Verificar   │     ┌─────────────────────────────────────┐
    │ Blacklist   │────►│ Redis: blacklist:cpf:XXX.XXX.XXX-XX │
    └─────────────┘     │ TTL: 24 horas                       │
           │            └─────────────────────────────────────┘
           │ Não está na blacklist
           ▼
    ┌─────────────┐
    │ Buscar      │     ┌─────────────────────────────────────┐
    │ Histórico   │────►│ Redis: user_behavior:CUST_001       │
    └─────────────┘     │ TTL: 1 hora                         │
           │            └─────────────────────────────────────┘
           │
           ▼
    ┌─────────────┐
    │ Incrementar │     ┌─────────────────────────────────────┐
    │ Velocidade  │────►│ Redis: velocity:CUST_001:hour       │
    └─────────────┘     │ TTL: 1 hora                         │
           │            └─────────────────────────────────────┘
           │
           ▼
    ┌─────────────┐
    │ Analisar    │
    │ Fraude (ML) │
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │ Cachear     │     ┌─────────────────────────────────────┐
    │ Resultado   │────►│ Redis: analysis:TXN_123456          │
    └─────────────┘     │ TTL: 5 minutos                      │
```

### Componentes de Cache

#### 1. RedisCacheSystem

O sistema principal de cache:

```python
class RedisCacheSystem:
    def __init__(self, config: CacheConfig):
        self.connection_manager = RedisConnectionManager(config)
        self.serializer = CacheSerializer()
        self.key_manager = CacheKeyManager()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtém valor do cache"""
        data = self.client.get(key)
        if data is None:
            self._update_stats("misses")
            return default
        
        self._update_stats("hits")
        return self.serializer.deserialize(data)
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Define valor no cache com TTL"""
        serialized = self.serializer.serialize(value)
        return self.client.setex(key, ttl, serialized)
```

#### 2. FraudCacheManager

Gerenciador específico para cache de fraude:

```python
class FraudCacheManager:
    def __init__(self, cache: RedisCacheSystem):
        self.cache = cache
        self.fraud_ttls = {
            "transaction_analysis": 300,     # 5 minutos
            "user_behavior": 3600,           # 1 hora
            "model_prediction": 600,         # 10 minutos
            "blacklist": 86400,              # 24 horas
            "velocity_counters": 3600,       # 1 hora
        }
    
    def cache_transaction_analysis(self, transaction_id: str, analysis: Dict):
        """Cacheia resultado da análise de transação"""
        key = f"analysis:{transaction_id}"
        ttl = self.fraud_ttls["transaction_analysis"]
        self.cache.set(key, analysis, ttl)
    
    def is_blacklisted(self, cpf: str) -> bool:
        """Verifica se CPF está na blacklist"""
        key = f"blacklist:cpf:{cpf}"
        return self.cache.exists(key)
    
    def increment_velocity_counter(self, customer_id: str, window: str) -> int:
        """Incrementa contador de velocidade"""
        key = f"velocity:{customer_id}:{window}"
        count = self.cache.increment(key)
        self.cache.expire(key, self.fraud_ttls["velocity_counters"])
        return count
```

#### 3. RedisTransactionRepository

Cache de transações para consultas rápidas:

```python
class RedisTransactionRepository:
    def __init__(self, redis_client):
        self._redis = redis_client
        self._ttl = 86400  # 24 horas
    
    async def save(self, transaction: Transaction) -> None:
        """Salva transação no Redis usando Hash"""
        key = f"transaction:{transaction.id}"
        data = {
            "id": transaction.id,
            "amount": str(transaction.amount),
            "customer_id": transaction.customer_id,
            "risk_score": str(transaction.risk_score),
            "status": transaction.status,
        }
        
        # Hash para dados da transação
        await self._redis.hset(key, mapping=data)
        await self._redis.expire(key, self._ttl)
        
        # Sorted Set para índice por cliente
        customer_key = f"customer_transactions:{transaction.customer_id}"
        await self._redis.zadd(
            customer_key, 
            {transaction.id: transaction.timestamp.timestamp()}
        )
```

---

## Estruturas de Dados

Redis oferece várias estruturas de dados. Veja como cada uma é usada:

### 1. String (Simples)

A estrutura mais básica - uma chave aponta para um valor.

```
┌──────────────────────────────────────────────────────────────┐
│                        STRING                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Chave: "blacklist:cpf:XXX.XXX.XXX-01"                      │
│  Valor: "blocked_at:2025-11-27T10:30:00"                    │
│                                                              │
│  Uso no Sankofa:                                             │
│  - Blacklist de CPFs                                         │
│  - Rate limiting counters                                    │
│  - Flags simples                                             │
│                                                              │
└──────────────────────────────────────────────────────────────┘

# Comandos
SET blacklist:cpf:XXX.XXX.XXX-01 "blocked" EX 86400
GET blacklist:cpf:XXX.XXX.XXX-01
INCR velocity:CUST_001:hour
```

### 2. Hash (Dicionário)

Como um dicionário Python - perfeito para objetos.

```
┌──────────────────────────────────────────────────────────────┐
│                          HASH                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Chave: "transaction:TXN_123456"                            │
│  ┌────────────────┬────────────────────────────────────┐    │
│  │ Campo          │ Valor                              │    │
│  ├────────────────┼────────────────────────────────────┤    │
│  │ id             │ TXN_123456                         │    │
│  │ amount         │ 1500.00                            │    │
│  │ customer_id    │ CUST_001                           │    │
│  │ risk_score     │ 0.12                               │    │
│  │ status         │ APPROVED                           │    │
│  │ timestamp      │ 2025-11-27T10:30:00               │    │
│  └────────────────┴────────────────────────────────────┘    │
│                                                              │
│  Uso no Sankofa:                                             │
│  - Cache de transações                                       │
│  - Perfil de usuário                                         │
│  - Resultado de análise                                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘

# Comandos
HSET transaction:TXN_123456 amount 1500.00 status APPROVED
HGET transaction:TXN_123456 amount
HGETALL transaction:TXN_123456
```

### 3. Sorted Set (Conjunto Ordenado)

Perfeito para rankings e índices temporais.

```
┌──────────────────────────────────────────────────────────────┐
│                      SORTED SET                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Chave: "customer_transactions:CUST_001"                    │
│  ┌──────────────────────────┬───────────────────────────┐   │
│  │ Score (timestamp)        │ Membro (transaction_id)   │   │
│  ├──────────────────────────┼───────────────────────────┤   │
│  │ 1732703400               │ TXN_123456                │   │
│  │ 1732703500               │ TXN_123457                │   │
│  │ 1732703600               │ TXN_123458                │   │
│  │ 1732703700               │ TXN_123459                │   │
│  └──────────────────────────┴───────────────────────────┘   │
│                                                              │
│  Uso no Sankofa:                                             │
│  - Histórico de transações por cliente                      │
│  - Ranking de clientes de alto risco                        │
│  - Últimas N transações                                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘

# Comandos
ZADD customer_transactions:CUST_001 1732703400 TXN_123456
ZREVRANGE customer_transactions:CUST_001 0 9  # Últimas 10
ZCOUNT customer_transactions:CUST_001 -inf +inf
```

### 4. List (Lista)

Filas e pilhas para processamento.

```
┌──────────────────────────────────────────────────────────────┐
│                          LIST                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Chave: "pending_reviews"                                   │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │ TXN_001     │ TXN_002     │ TXN_003     │ TXN_004     │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
│       ▲                                           ▲         │
│    (início)                                    (fim)        │
│                                                              │
│  Uso no Sankofa:                                             │
│  - Fila de transações para revisão                          │
│  - Histórico de eventos                                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘

# Comandos
LPUSH pending_reviews TXN_001  # Adiciona no início
RPOP pending_reviews           # Remove do fim
LRANGE pending_reviews 0 -1    # Lista todos
```

### 5. Set (Conjunto)

Coleções sem duplicatas.

```
┌──────────────────────────────────────────────────────────────┐
│                           SET                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Chave: "known_devices:CUST_001"                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  { "device_abc123", "device_def456", "device_ghi789" } │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Uso no Sankofa:                                             │
│  - Dispositivos conhecidos do cliente                       │
│  - IPs permitidos                                           │
│  - Tags de transação                                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘

# Comandos
SADD known_devices:CUST_001 device_abc123
SISMEMBER known_devices:CUST_001 device_abc123  # Verifica se existe
SMEMBERS known_devices:CUST_001                 # Lista todos
```

---

## Padrões de Cache

### 1. Cache-Aside (Lazy Loading)

O padrão mais comum - carrega no cache sob demanda.

```
┌─────────────────────────────────────────────────────────────┐
│                     CACHE-ASIDE                             │
└─────────────────────────────────────────────────────────────┘

    Aplicação
        │
        │ 1. Buscar cliente CUST_001
        ▼
    ┌───────────┐
    │   CACHE   │
    │  (Redis)  │
    └─────┬─────┘
          │
          │ Cache HIT? ──────────────────► Retorna dados
          │
          │ Cache MISS
          ▼
    ┌───────────┐
    │ DATABASE  │
    │(PostgreSQL)│
    └─────┬─────┘
          │
          │ 2. Busca no banco
          │
          ▼
    ┌───────────┐
    │   CACHE   │◄─── 3. Salva no cache
    │  (Redis)  │
    └─────┬─────┘
          │
          ▼
      Retorna dados
```

```python
async def find_by_id(self, transaction_id: str) -> Optional[Transaction]:
    # 1. Tenta cache primeiro
    transaction = await self._cache.find_by_id(transaction_id)
    if transaction:
        return transaction  # Cache HIT
    
    # 2. Cache MISS - busca no banco
    transaction = await self._primary.find_by_id(transaction_id)
    
    if transaction:
        # 3. Popula cache para próxima vez
        await self._cache.save(transaction)
    
    return transaction
```

### 2. Write-Through

Escreve no cache E no banco simultaneamente.

```
┌─────────────────────────────────────────────────────────────┐
│                     WRITE-THROUGH                           │
└─────────────────────────────────────────────────────────────┘

    Aplicação
        │
        │ Salvar transação
        ▼
    ┌───────────┐
    │ DATABASE  │◄─── 1. Salva no banco PRIMEIRO
    │(PostgreSQL)│
    └─────┬─────┘
          │
          │ Sucesso?
          ▼
    ┌───────────┐
    │   CACHE   │◄─── 2. Atualiza cache
    │  (Redis)  │
    └─────┬─────┘
          │
          ▼
      Confirma
```

```python
async def save(self, transaction: Transaction) -> None:
    # 1. Salva no banco primeiro (consistência)
    await self._primary.save(transaction)
    
    # 2. Atualiza cache (best effort)
    try:
        await self._cache.save(transaction)
    except Exception:
        # Cache failure não falha a operação
        pass
```

### 3. TTL (Time-To-Live)

Dados expiram automaticamente.

```
┌─────────────────────────────────────────────────────────────┐
│                          TTL                                │
└─────────────────────────────────────────────────────────────┘

    Tempo ───────────────────────────────────────────────────►
    
    t=0                                    t=300s (5 min)
    │                                      │
    ▼                                      ▼
    ┌───────────────────────────────────────┐
    │  DADO VÁLIDO                          │  EXPIRADO (removido)
    │  (Redis mantém)                       │  (Redis remove)
    └───────────────────────────────────────┘
    
    SET analysis:TXN_123 {...} EX 300
```

**TTLs no Sankofa:**

| Tipo de Dado | TTL | Justificativa |
|--------------|-----|---------------|
| `transaction_analysis` | 5 min | Resultado pode mudar |
| `user_behavior` | 1 hora | Perfil evolui lentamente |
| `model_prediction` | 10 min | Modelo pode atualizar |
| `blacklist` | 24 horas | Lista diária |
| `velocity_counters` | 1 hora | Janela de velocidade |
| `rate_limits` | 1 min | Janela de rate limit |

---

## Configuração

### Variáveis de Ambiente

```bash
# Configuração básica
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=sua_senha_aqui
REDIS_DB=0

# Configuração avançada
REDIS_MAX_CONNECTIONS=100
REDIS_SOCKET_TIMEOUT=5.0
REDIS_SSL=false  # true para Upstash/Redis Cloud
```

### CacheConfig

```python
@dataclass
class CacheConfig:
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 100
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    default_ttl: int = 3600  # 1 hora
    max_memory_policy: str = "allkeys-lru"
    use_ssl: bool = False
    
    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Carrega configuração de variáveis de ambiente"""
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD"),
            use_ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
        )
```

### Pool de Conexões

```python
class RedisConnectionManager:
    def __init__(self, config: CacheConfig):
        self.config = config
        self.pool = None
        self._fallback_cache = InMemoryCache()
        self._init_connection_pool()
    
    def _init_connection_pool(self):
        """Inicializa pool de conexões Redis"""
        try:
            self.pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                decode_responses=False,
            )
            logger.info("Redis connection pool initialized")
        except Exception as e:
            logger.warning(f"Redis não disponível: {e}")
```

---

## Operações Comuns

### Operações Básicas

```python
# GET - Obter valor
value = cache.get("key")

# SET - Definir valor com TTL
cache.set("key", "value", ttl=3600)

# DELETE - Remover
cache.delete("key")

# EXISTS - Verificar existência
exists = cache.exists("key")

# EXPIRE - Definir TTL em chave existente
cache.expire("key", 3600)
```

### Operações de Contador

```python
# INCR - Incrementar
count = cache.increment("velocity:CUST_001:hour")

# INCRBY - Incrementar por valor
count = cache.increment("velocity:CUST_001:hour", amount=5)

# Uso para rate limiting
def check_rate_limit(api_key: str) -> bool:
    key = f"rate:{api_key}:{int(time.time() / 60)}"  # Janela de 1 minuto
    count = cache.increment(key)
    if count == 1:
        cache.expire(key, 60)
    return count <= 1000  # Limite de 1000/min
```

### Pipeline (Batch)

```python
# Executar múltiplos comandos de uma vez
pipe = cache.pipeline()
pipe.set("key1", "value1")
pipe.set("key2", "value2")
pipe.set("key3", "value3")
results = pipe.execute()  # Executa todos de uma vez
```

### Busca por Padrão

```python
# Buscar chaves por padrão
keys = cache.keys("velocity:CUST_001:*")

# Limpar por padrão
cache.clear_pattern("analysis:*")  # Remove todos os resultados de análise
```

---

## Fallback para Memória

O Sankofa implementa um fallback inteligente quando Redis não está disponível:

### InMemoryCache

```python
class InMemoryCache:
    """Cache em memória com LRU eviction"""
    
    MAX_SIZE = 10000
    
    def __init__(self, max_size: int = MAX_SIZE):
        self._cache: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}
        self._access_order: List[str] = []
        self._lock = threading.Lock()
        self._max_size = max_size
    
    def _evict_lru(self) -> None:
        """Remove entradas menos usadas até estar dentro do limite"""
        while len(self._cache) >= self._max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]
            if oldest_key in self._expiry:
                del self._expiry[oldest_key]
    
    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            if key in self._cache:
                # Verifica se expirou
                if key in self._expiry and time.time() > self._expiry[key]:
                    del self._cache[key]
                    del self._expiry[key]
                    return None
                
                # Atualiza ordem de acesso (LRU)
                self._update_access(key)
                return self._cache[key]
            return None
    
    def setex(self, key: str, ttl: int, value: bytes) -> bool:
        with self._lock:
            self._evict_lru()  # Remove LRU se necessário
            self._cache[key] = value
            self._expiry[key] = time.time() + ttl
            self._update_access(key)
            return True
```

### Fluxo de Fallback

```
┌─────────────────────────────────────────────────────────────┐
│                     FALLBACK FLOW                           │
└─────────────────────────────────────────────────────────────┘

    get_client()
        │
        ▼
    ┌───────────────────┐
    │ Redis disponível? │
    └─────────┬─────────┘
              │
    ┌─────────┴─────────┐
    │                   │
    ▼ SIM               ▼ NÃO
┌─────────┐       ┌─────────────┐
│  Redis  │       │ InMemory    │
│ Client  │       │ Cache       │
└─────────┘       └─────────────┘
```

```python
def get_client(self) -> Any:
    """Obtém cliente Redis do pool ou fallback"""
    if self.pool:
        try:
            client = redis.Redis(connection_pool=self.pool)
            client.ping()
            return client
        except Exception:
            pass
    
    # Fallback para cache em memória
    return self._fallback_cache
```

### Vantagens do Fallback

| Aspecto | Com Redis | Com Fallback (Memória) |
|---------|-----------|------------------------|
| Persistência | Sim | Não |
| Compartilhamento | Entre processos | Apenas processo local |
| Capacidade | Terabytes | Limitado por RAM |
| Performance | Excelente | Excelente |
| Disponibilidade | 99.9%+ | 100% (local) |

---

## Monitoramento

### Métricas do Cache

```python
# Estatísticas de cache
stats = {
    "hits": 0,     # Cache acertos
    "misses": 0,   # Cache falhas
    "sets": 0,     # Escritas
    "deletes": 0,  # Remoções
    "errors": 0,   # Erros
}

# Taxa de acerto
hit_rate = stats["hits"] / (stats["hits"] + stats["misses"]) * 100
```

### Health Check

```python
def is_healthy(self) -> bool:
    """Verifica se Redis está saudável"""
    try:
        self.client.ping()
        return True
    except Exception:
        return False

def get_info(self) -> Dict[str, Any]:
    """Obtém informações do Redis"""
    info = self.client.info()
    return {
        "used_memory_human": info.get("used_memory_human"),
        "connected_clients": info.get("connected_clients"),
        "total_commands_processed": info.get("total_commands_processed"),
        "keyspace_hits": info.get("keyspace_hits"),
        "keyspace_misses": info.get("keyspace_misses"),
    }
```

### Dashboard de Métricas

```
┌──────────────────────────────────────────────────────────────┐
│                    REDIS METRICS                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Status: 🟢 HEALTHY                                          │
│                                                              │
│  ┌─────────────────┬────────────────────────────────────┐   │
│  │ Memória usada   │ ████████░░░░░░░░░░░░ 40%  (2GB)   │   │
│  │ Conexões        │ ██░░░░░░░░░░░░░░░░░░ 10% (10/100) │   │
│  │ Hit Rate        │ █████████████████░░░ 85%          │   │
│  │ Commands/sec    │ ████████████░░░░░░░░ 5,000        │   │
│  └─────────────────┴────────────────────────────────────┘   │
│                                                              │
│  Últimas 24h:                                                │
│  - Hits: 4,320,000                                          │
│  - Misses: 764,000                                          │
│  - Sets: 1,200,000                                          │
│  - Deletes: 100,000                                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Problema: Redis não conecta

```
Error 111 connecting to localhost:6379. Connection refused.
```

**Diagnóstico:**
```bash
# Verificar se Redis está rodando
redis-cli ping
# Esperado: PONG

# Verificar porta
netstat -tlnp | grep 6379
```

**Soluções:**
1. Redis não está instalado → Sistema usa fallback em memória (OK)
2. Redis está parado → Iniciar serviço
3. Porta errada → Verificar `REDIS_PORT`

### Problema: Memória cheia

```
OOM command not allowed when used memory > 'maxmemory'
```

**Soluções:**
1. Aumentar `maxmemory` no Redis
2. Configurar `maxmemory-policy` para LRU
3. Reduzir TTLs
4. Limpar chaves desnecessárias

### Problema: Cache desatualizado

**Sintoma:** Dados no cache não refletem alterações no banco.

**Soluções:**
1. Implementar invalidação de cache após writes
2. Reduzir TTL
3. Usar write-through em vez de cache-aside

### Problema: Alta taxa de miss

**Diagnóstico:**
```python
hit_rate = hits / (hits + misses) * 100
# Ideal: > 80%
```

**Soluções:**
1. Aumentar TTL (dados mudam pouco)
2. Pré-aquecer cache com dados frequentes
3. Verificar se chaves estão corretas

---

## Melhores Práticas

### 1. Nomenclatura de Chaves

```
# Formato: namespace:tipo:identificador
sankofa:transaction:TXN_123456
sankofa:blacklist:cpf:XXX.XXX.XXX-XX
sankofa:velocity:CUST_001:hour

# Evite:
transaction_123456  # Sem namespace
sankofa:transaction:TXN_123456:data:extra:muito:longa  # Muito longa
```

### 2. TTLs Apropriados

```python
# Dados que mudam frequentemente → TTL curto
"real_time_velocity": 60,      # 1 minuto

# Dados estáveis → TTL longo
"user_profile": 86400,         # 24 horas

# Dados de sessão → TTL médio
"session": 3600,               # 1 hora
```

### 3. Tratamento de Erros

```python
def get_safe(self, key: str, default: Any = None) -> Any:
    """Sempre retorna algo, nunca falha"""
    try:
        value = self.client.get(key)
        return self.deserialize(value) if value else default
    except Exception as e:
        logger.warning(f"Cache error: {e}")
        return default  # Graceful degradation
```

### 4. Serialização Eficiente

```python
class CacheSerializer:
    @staticmethod
    def serialize(data: Any) -> bytes:
        """Serializa dados para armazenamento"""
        if isinstance(data, (str, int, float, bool)):
            # JSON para tipos simples (mais legível)
            return json.dumps(data).encode("utf-8")
        elif isinstance(data, (dict, list, tuple)):
            # JSON para estruturas
            return json.dumps(data).encode("utf-8")
        else:
            # Pickle para objetos complexos
            return pickle.dumps(data)
```

### 5. Monitoramento Contínuo

```python
# Log de métricas a cada minuto
def log_cache_metrics():
    stats = cache.get_stats()
    hit_rate = stats["hits"] / (stats["hits"] + stats["misses"] + 1) * 100
    
    logger.info(
        "Cache metrics",
        hit_rate=f"{hit_rate:.1f}%",
        hits=stats["hits"],
        misses=stats["misses"],
        errors=stats["errors"],
    )
```

---

## Resumo

### Quando usar Redis

| Use Redis para... | Não use Redis para... |
|-------------------|-----------------------|
| Cache de consultas frequentes | Armazenamento permanente |
| Contadores de velocidade | Dados que precisam de ACID |
| Rate limiting | Transações financeiras |
| Sessões de usuário | Auditoria (use PostgreSQL) |
| Blacklists | Dados muito grandes (>100MB) |
| Pub/Sub em tempo real | Dados raramente acessados |

### Checklist de Implementação

- [ ] Configurar variáveis de ambiente
- [ ] Implementar fallback para memória
- [ ] Definir TTLs apropriados
- [ ] Usar namespaces nas chaves
- [ ] Tratar erros graciosamente
- [ ] Monitorar hit rate
- [ ] Implementar health checks

---

*Documentação gerada em Novembro 2025 - Sankofa Enterprise Pro v12.0*
