"""
Sankofa Enterprise Pro - Redis Cluster Configuration
Configuração de Redis Cluster para cache distribuído e rate limiting
"""

import os
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class RedisNode:
    """Nó do cluster Redis"""
    host: str
    port: int
    role: str = "master"
    is_healthy: bool = True
    last_health_check: datetime = field(default_factory=datetime.now)
    slots: List[int] = field(default_factory=list)


@dataclass
class RedisClusterConfig:
    """Configuração do cluster Redis"""
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    password: Optional[str] = None
    db: int = 0
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    max_connections: int = 100
    health_check_interval: int = 30
    decode_responses: bool = True
    ssl: bool = False


class CacheBackend(ABC):
    """Interface abstrata para backends de cache"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        pass
    
    @abstractmethod
    def incr(self, key: str, amount: int = 1) -> int:
        pass
    
    @abstractmethod
    def expire(self, key: str, seconds: int) -> bool:
        pass


class MemoryCache(CacheBackend):
    """Cache em memória (fallback quando Redis não disponível)"""
    
    def __init__(self, max_size: int = 10000):
        self._cache: Dict[str, Any] = {}
        self._expiry: Dict[str, datetime] = {}
        self._max_size = max_size
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def _cleanup_expired(self):
        """Remove itens expirados"""
        now = datetime.now()
        expired = [k for k, exp in self._expiry.items() if exp < now]
        for key in expired:
            self._cache.pop(key, None)
            self._expiry.pop(key, None)
    
    def _evict_if_needed(self):
        """Evicta itens se cache estiver cheio"""
        if len(self._cache) >= self._max_size:
            to_remove = list(self._cache.keys())[:self._max_size // 10]
            for key in to_remove:
                self._cache.pop(key, None)
                self._expiry.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            self._cleanup_expired()
            
            if key in self._cache:
                if key in self._expiry and self._expiry[key] < datetime.now():
                    self._cache.pop(key, None)
                    self._expiry.pop(key, None)
                    self._misses += 1
                    return None
                
                self._hits += 1
                return self._cache[key]
            
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        with self._lock:
            self._evict_if_needed()
            self._cache[key] = value
            
            if ttl:
                self._expiry[key] = datetime.now() + timedelta(seconds=ttl)
            
            return True
    
    def delete(self, key: str) -> bool:
        with self._lock:
            self._cache.pop(key, None)
            self._expiry.pop(key, None)
            return True
    
    def exists(self, key: str) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            
            if key in self._expiry and self._expiry[key] < datetime.now():
                self._cache.pop(key, None)
                self._expiry.pop(key, None)
                return False
            
            return True
    
    def incr(self, key: str, amount: int = 1) -> int:
        with self._lock:
            value = self._cache.get(key, 0)
            if isinstance(value, str):
                value = int(value)
            value += amount
            self._cache[key] = value
            return value
    
    def expire(self, key: str, seconds: int) -> bool:
        with self._lock:
            if key in self._cache:
                self._expiry[key] = datetime.now() + timedelta(seconds=seconds)
                return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "type": "memory",
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
        }


class RedisCache(CacheBackend):
    """Cache Redis com suporte a cluster e fallback inteligente.
    
    COMPORTAMENTO POR AMBIENTE:
    - Desenvolvimento: Usa MemoryCache automaticamente (não tenta conectar ao Redis)
    - Produção: Tenta conectar ao Redis, fallback para MemoryCache se falhar
    
    CONFIGURAÇÃO DE PRODUÇÃO:
    Defina REDIS_URL no ambiente para usar Redis real:
    - REDIS_URL=redis://localhost:6379
    - REDIS_URL=redis://user:pass@host:6379/0
    
    SEM REDIS_URL:
    - Em produção: Loga warning e usa MemoryCache
    - Em desenvolvimento: Usa MemoryCache silenciosamente
    """
    
    def __init__(self, config: RedisClusterConfig):
        self.config = config
        self._client = None
        self._is_cluster = len(config.nodes) > 1
        self._fallback = MemoryCache()
        self._connected = False
        self._environment = os.getenv("ENVIRONMENT", "development")
        self._redis_url = os.getenv("REDIS_URL")
        
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Inicializa o cache baseado no ambiente e configuração."""
        if self._environment == "development" and not self._redis_url:
            logger.info(
                "Development mode: Using in-memory cache. "
                "Set REDIS_URL for Redis in production."
            )
            self._connected = False
            return
        
        if self._redis_url:
            self._try_connect_with_url()
        elif self.config.nodes:
            self._try_connect_with_config()
        else:
            if self._environment == "production":
                logger.warning(
                    "REDIS_URL not configured in production. Using MemoryCache. "
                    "For 300M req/day, configure Redis: REDIS_URL=redis://host:6379"
                )
            self._connected = False
    
    def _try_connect_with_url(self):
        """Conecta ao Redis usando REDIS_URL."""
        try:
            import redis
            
            self._client = redis.from_url(
                self._redis_url,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                decode_responses=self.config.decode_responses
            )
            
            self._client.ping()
            self._connected = True
            logger.info("Connected to Redis via REDIS_URL")
            
        except ImportError:
            logger.warning("Redis package not installed, using memory fallback")
            self._connected = False
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Using memory fallback")
            self._connected = False
    
    def _try_connect_with_config(self):
        """Tenta conectar ao Redis usando configuração de nós."""
        try:
            import redis
            
            if self._is_cluster:
                from redis.cluster import RedisCluster
                
                startup_nodes = [
                    {"host": n['host'], "port": n['port']}
                    for n in self.config.nodes
                ]
                
                self._client = RedisCluster(
                    startup_nodes=startup_nodes,
                    password=self.config.password,
                    decode_responses=self.config.decode_responses,
                    skip_full_coverage_check=True
                )
            else:
                node = self.config.nodes[0] if self.config.nodes else {"host": "localhost", "port": 6379}
                
                self._client = redis.Redis(
                    host=node.get('host', 'localhost'),
                    port=node.get('port', 6379),
                    password=self.config.password,
                    db=self.config.db,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.socket_connect_timeout,
                    retry_on_timeout=self.config.retry_on_timeout,
                    max_connections=self.config.max_connections,
                    decode_responses=self.config.decode_responses,
                    ssl=self.config.ssl
                )
            
            self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis {'Cluster' if self._is_cluster else 'Standalone'}")
            
        except ImportError:
            logger.warning("Redis package not installed, using memory fallback")
            self._connected = False
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Using memory fallback")
            self._connected = False
    
    def _use_fallback(self) -> bool:
        """Verifica se deve usar fallback"""
        return not self._connected or self._client is None
    
    def get(self, key: str) -> Optional[Any]:
        if self._use_fallback():
            return self._fallback.get(key)
        
        try:
            value = self._client.get(key)
            if value is None:
                return None
            
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            return self._fallback.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if self._use_fallback():
            return self._fallback.set(key, value, ttl)
        
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            if ttl:
                self._client.setex(key, ttl, value)
            else:
                self._client.set(key, value)
            return True
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            return self._fallback.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        if self._use_fallback():
            return self._fallback.delete(key)
        
        try:
            self._client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return self._fallback.delete(key)
    
    def exists(self, key: str) -> bool:
        if self._use_fallback():
            return self._fallback.exists(key)
        
        try:
            return bool(self._client.exists(key))
        except Exception as e:
            logger.warning(f"Redis exists error: {e}")
            return self._fallback.exists(key)
    
    def incr(self, key: str, amount: int = 1) -> int:
        if self._use_fallback():
            return self._fallback.incr(key, amount)
        
        try:
            return self._client.incr(key, amount)
        except Exception as e:
            logger.warning(f"Redis incr error: {e}")
            return self._fallback.incr(key, amount)
    
    def expire(self, key: str, seconds: int) -> bool:
        if self._use_fallback():
            return self._fallback.expire(key, seconds)
        
        try:
            return bool(self._client.expire(key, seconds))
        except Exception as e:
            logger.warning(f"Redis expire error: {e}")
            return self._fallback.expire(key, seconds)
    
    def get_stats(self) -> Dict[str, Any]:
        if self._use_fallback():
            stats = self._fallback.get_stats()
            stats["redis_connected"] = False
            return stats
        
        try:
            info = self._client.info()
            return {
                "type": "redis_cluster" if self._is_cluster else "redis",
                "connected": True,
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
            }
        except Exception as e:
            logger.warning(f"Could not get Redis stats: {e}")
            return {"type": "redis", "connected": False, "error": str(e)}


class DistributedRateLimiter:
    """
    Rate limiter distribuído usando Redis
    
    Algoritmos:
    - Token Bucket
    - Sliding Window
    - Fixed Window
    """
    
    def __init__(
        self,
        cache: CacheBackend,
        algorithm: str = "sliding_window"
    ):
        self.cache = cache
        self.algorithm = algorithm
        self._limits: Dict[str, Dict[str, Any]] = {}
    
    def configure_limit(
        self,
        name: str,
        max_requests: int,
        window_seconds: int,
        burst_size: Optional[int] = None
    ):
        """Configura um limite"""
        self._limits[name] = {
            "max_requests": max_requests,
            "window_seconds": window_seconds,
            "burst_size": burst_size or max_requests
        }
    
    def check_limit(
        self,
        limit_name: str,
        identifier: str
    ) -> Dict[str, Any]:
        """
        Verifica se o limite foi atingido
        
        Args:
            limit_name: Nome do limite configurado
            identifier: Identificador (IP, user_id, etc.)
            
        Returns:
            Dict com allowed, remaining, reset_at
        """
        if limit_name not in self._limits:
            return {"allowed": True, "remaining": -1, "reset_at": None}
        
        config = self._limits[limit_name]
        
        if self.algorithm == "sliding_window":
            return self._sliding_window_check(
                limit_name, identifier,
                config["max_requests"],
                config["window_seconds"]
            )
        else:
            return self._fixed_window_check(
                limit_name, identifier,
                config["max_requests"],
                config["window_seconds"]
            )
    
    def _sliding_window_check(
        self,
        limit_name: str,
        identifier: str,
        max_requests: int,
        window_seconds: int
    ) -> Dict[str, Any]:
        """Sliding window rate limiting"""
        now = time.time()
        window_start = now - window_seconds
        
        key = f"ratelimit:{limit_name}:{identifier}"
        count_key = f"{key}:count"
        
        count = self.cache.get(count_key) or 0
        count = int(count)
        
        if count >= max_requests:
            return {
                "allowed": False,
                "remaining": 0,
                "reset_at": int(now + window_seconds),
                "retry_after": window_seconds
            }
        
        new_count = self.cache.incr(count_key)
        
        if new_count == 1:
            self.cache.expire(count_key, window_seconds)
        
        return {
            "allowed": True,
            "remaining": max(0, max_requests - new_count),
            "reset_at": int(now + window_seconds)
        }
    
    def _fixed_window_check(
        self,
        limit_name: str,
        identifier: str,
        max_requests: int,
        window_seconds: int
    ) -> Dict[str, Any]:
        """Fixed window rate limiting"""
        now = int(time.time())
        window_key = now // window_seconds
        
        key = f"ratelimit:{limit_name}:{identifier}:{window_key}"
        
        count = self.cache.incr(key)
        
        if count == 1:
            self.cache.expire(key, window_seconds)
        
        reset_at = (window_key + 1) * window_seconds
        
        if count > max_requests:
            return {
                "allowed": False,
                "remaining": 0,
                "reset_at": reset_at,
                "retry_after": reset_at - now
            }
        
        return {
            "allowed": True,
            "remaining": max(0, max_requests - count),
            "reset_at": reset_at
        }


class SessionStore:
    """Armazenamento de sessões distribuído"""
    
    def __init__(self, cache: CacheBackend, prefix: str = "session"):
        self.cache = cache
        self.prefix = prefix
        self.default_ttl = 86400
    
    def _key(self, session_id: str) -> str:
        return f"{self.prefix}:{session_id}"
    
    def create(
        self,
        session_id: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cria uma sessão"""
        session_data = {
            **data,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat()
        }
        return self.cache.set(
            self._key(session_id),
            session_data,
            ttl or self.default_ttl
        )
    
    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Recupera uma sessão"""
        data = self.cache.get(self._key(session_id))
        if data:
            data["last_accessed"] = datetime.now().isoformat()
            self.cache.set(self._key(session_id), data, self.default_ttl)
        return data
    
    def update(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Atualiza uma sessão"""
        data = self.get(session_id)
        if not data:
            return False
        
        data.update(updates)
        data["last_accessed"] = datetime.now().isoformat()
        return self.cache.set(self._key(session_id), data, self.default_ttl)
    
    def delete(self, session_id: str) -> bool:
        """Remove uma sessão"""
        return self.cache.delete(self._key(session_id))
    
    def exists(self, session_id: str) -> bool:
        """Verifica se sessão existe"""
        return self.cache.exists(self._key(session_id))


def create_cache_from_env() -> CacheBackend:
    """Cria cache a partir de variáveis de ambiente"""
    redis_url = os.getenv("REDIS_URL")
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD")
    
    if redis_url or redis_host != "localhost":
        config = RedisClusterConfig(
            nodes=[{"host": redis_host, "port": redis_port}],
            password=redis_password
        )
        return RedisCache(config)
    
    logger.info("Using in-memory cache (Redis not configured)")
    return MemoryCache()


_cache_instance: Optional[CacheBackend] = None


def get_cache() -> CacheBackend:
    """Retorna instância singleton do cache"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = create_cache_from_env()
    return _cache_instance


def get_rate_limiter() -> DistributedRateLimiter:
    """Retorna rate limiter configurado"""
    cache = get_cache()
    limiter = DistributedRateLimiter(cache)
    
    limiter.configure_limit("fraud_predict", max_requests=500, window_seconds=60)
    limiter.configure_limit("fraud_batch", max_requests=100, window_seconds=60)
    limiter.configure_limit("auth_login", max_requests=10, window_seconds=60)
    limiter.configure_limit("api_general", max_requests=1000, window_seconds=60)
    
    return limiter


def get_session_store() -> SessionStore:
    """Retorna session store"""
    return SessionStore(get_cache())
