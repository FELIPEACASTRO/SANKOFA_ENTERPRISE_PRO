#!/usr/bin/env python3
"""
Sankofa Enterprise Pro - Cache Service
Serviço de cache abstrato com suporte a Redis e in-memory

Padrões implementados:
- Cache-Aside Pattern
- Write-Through (opcional)
- TTL-based expiration
- Graceful degradation
"""

import os
import json
import time
import asyncio
import logging
import hashlib
from typing import Any, Dict, Optional, Union, List
from datetime import datetime, timezone
from abc import ABC, abstractmethod
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Tentar importar Redis
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import aioredis
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False
        logger.warning("Redis não disponível. Usando cache in-memory.")


class CacheBackend(ABC):
    """Interface abstrata para backends de cache."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Obtém valor do cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Define valor no cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove valor do cache."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Verifica se chave existe."""
        pass

    @abstractmethod
    async def clear(self) -> int:
        """Limpa todo o cache."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Verifica saúde do backend."""
        pass


class InMemoryBackend(CacheBackend):
    """
    Backend de cache in-memory com LRU eviction.

    Útil para:
    - Desenvolvimento local
    - Testes
    - Fallback quando Redis não está disponível
    """

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Verificar expiração
            if entry["expires_at"] and time.time() > entry["expires_at"]:
                del self._cache[key]
                self._misses += 1
                return None

            # Move para o final (LRU)
            self._cache.move_to_end(key)
            self._hits += 1

            return entry["value"]

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        with self._lock:
            # Eviction se necessário
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl
            elif self._default_ttl:
                expires_at = time.time() + self._default_ttl

            self._cache[key] = {
                "value": value,
                "expires_at": expires_at,
                "created_at": time.time()
            }

            return True

    async def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        with self._lock:
            if key not in self._cache:
                return False

            entry = self._cache[key]
            if entry["expires_at"] and time.time() > entry["expires_at"]:
                del self._cache[key]
                return False

            return True

    async def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    async def health_check(self) -> bool:
        return True

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                "type": "in_memory",
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 2)
            }


class RedisBackend(CacheBackend):
    """
    Backend de cache Redis.

    Features:
    - Conexão assíncrona
    - Connection pooling
    - Serialização JSON
    - Graceful degradation
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        default_ttl: int = 3600,
        key_prefix: str = "sankofa:"
    ):
        self._host = host
        self._port = port
        self._password = password
        self._db = db
        self._default_ttl = default_ttl
        self._key_prefix = key_prefix
        self._client: Optional[Any] = None
        self._connected = False

    async def _ensure_connected(self) -> bool:
        if self._connected and self._client:
            return True

        if not REDIS_AVAILABLE:
            return False

        try:
            self._client = aioredis.Redis(
                host=self._host,
                port=self._port,
                password=self._password,
                db=self._db,
                decode_responses=True
            )
            await self._client.ping()
            self._connected = True
            logger.info(f"Redis conectado: {self._host}:{self._port}")
            return True
        except Exception as e:
            logger.error(f"Falha ao conectar ao Redis: {e}")
            self._connected = False
            return False

    def _make_key(self, key: str) -> str:
        return f"{self._key_prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        if not await self._ensure_connected():
            return None

        try:
            full_key = self._make_key(key)
            value = await self._client.get(full_key)
            if value is None:
                return None
            return json.loads(value)
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if not await self._ensure_connected():
            return False

        try:
            full_key = self._make_key(key)
            serialized = json.dumps(value)
            ex = ttl if ttl is not None else self._default_ttl
            await self._client.set(full_key, serialized, ex=ex)
            return True
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        if not await self._ensure_connected():
            return False

        try:
            full_key = self._make_key(key)
            result = await self._client.delete(full_key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        if not await self._ensure_connected():
            return False

        try:
            full_key = self._make_key(key)
            return await self._client.exists(full_key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error: {e}")
            return False

    async def clear(self) -> int:
        if not await self._ensure_connected():
            return 0

        try:
            pattern = f"{self._key_prefix}*"
            keys = await self._client.keys(pattern)
            if keys:
                return await self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis CLEAR error: {e}")
            return 0

    async def health_check(self) -> bool:
        try:
            if not await self._ensure_connected():
                return False
            await self._client.ping()
            return True
        except Exception:
            return False

    async def close(self):
        if self._client:
            await self._client.close()
            self._connected = False


class CacheService:
    """
    Serviço de cache de alto nível com fallback automático.

    Features:
    - Fallback automático de Redis para in-memory
    - Serialização automática
    - Métricas
    - Health check
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        redis_host: Optional[str] = None,
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        redis_db: int = 0,
        default_ttl: int = 3600,
        max_memory_items: int = 10000,
        key_prefix: str = "sankofa:"
    ):
        """
        Inicializa o serviço de cache.

        Args:
            redis_url: URL completa do Redis (redis://host:port/db)
            redis_host: Host do Redis (alternativa a redis_url)
            redis_port: Porta do Redis
            redis_password: Senha do Redis
            redis_db: Database do Redis
            default_ttl: TTL padrão em segundos
            max_memory_items: Máximo de itens no cache in-memory
            key_prefix: Prefixo para chaves
        """
        self._default_ttl = default_ttl
        self._key_prefix = key_prefix

        # Tentar Redis primeiro
        redis_host = redis_host or os.environ.get("REDIS_HOST", "localhost")
        redis_port = int(os.environ.get("REDIS_PORT", redis_port))
        redis_password = redis_password or os.environ.get("REDIS_PASSWORD")

        if REDIS_AVAILABLE:
            self._primary_backend = RedisBackend(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                db=redis_db,
                default_ttl=default_ttl,
                key_prefix=key_prefix
            )
            logger.info("CacheService: Redis backend configurado")
        else:
            self._primary_backend = None

        # Fallback in-memory sempre disponível
        self._fallback_backend = InMemoryBackend(
            max_size=max_memory_items,
            default_ttl=default_ttl
        )

        self._using_fallback = False

    async def _get_backend(self) -> CacheBackend:
        """Retorna o backend apropriado."""
        if self._primary_backend and not self._using_fallback:
            if await self._primary_backend.health_check():
                return self._primary_backend
            else:
                logger.warning("CacheService: Redis indisponível, usando fallback in-memory")
                self._using_fallback = True

        return self._fallback_backend

    async def get(self, key: str) -> Optional[Any]:
        """Obtém valor do cache."""
        backend = await self._get_backend()
        return await backend.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Define valor no cache."""
        backend = await self._get_backend()
        return await backend.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """Remove valor do cache."""
        backend = await self._get_backend()
        return await backend.delete(key)

    async def exists(self, key: str) -> bool:
        """Verifica se chave existe."""
        backend = await self._get_backend()
        return await backend.exists(key)

    async def clear(self) -> int:
        """Limpa todo o cache."""
        backend = await self._get_backend()
        return await backend.clear()

    async def get_or_set(
        self,
        key: str,
        factory: callable,
        ttl: Optional[int] = None
    ) -> Any:
        """
        Obtém do cache ou executa factory e armazena.

        Padrão Cache-Aside.
        """
        value = await self.get(key)
        if value is not None:
            return value

        # Cache miss - executar factory
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        await self.set(key, value, ttl)
        return value

    async def health_check(self) -> Dict[str, Any]:
        """Verifica saúde do cache."""
        primary_ok = False
        if self._primary_backend:
            primary_ok = await self._primary_backend.health_check()

        fallback_ok = await self._fallback_backend.health_check()

        return {
            "healthy": primary_ok or fallback_ok,
            "primary_backend": {
                "type": "redis" if self._primary_backend else "none",
                "available": primary_ok
            },
            "fallback_backend": {
                "type": "in_memory",
                "available": fallback_ok,
                "stats": self._fallback_backend.get_stats()
            },
            "using_fallback": self._using_fallback
        }

    async def close(self):
        """Fecha conexões."""
        if self._primary_backend and hasattr(self._primary_backend, 'close'):
            await self._primary_backend.close()


# Singleton global
_cache_service: Optional[CacheService] = None
_cache_service_lock = threading.Lock()


def get_cache_service() -> CacheService:
    """
    Retorna instância singleton do CacheService.

    Thread-safe com double-checked locking.
    """
    global _cache_service
    if _cache_service is None:
        with _cache_service_lock:
            if _cache_service is None:
                _cache_service = CacheService()
    return _cache_service


# Exemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def test_cache():
        print("=== Teste do CacheService ===")

        cache = CacheService()

        # Teste básico
        print("\n1. Set/Get básico:")
        await cache.set("test_key", {"data": "valor"}, ttl=60)
        result = await cache.get("test_key")
        print(f"   Set: test_key = {{'data': 'valor'}}")
        print(f"   Get: test_key = {result}")

        # Teste get_or_set
        print("\n2. Get or Set:")
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"computed": True, "call": call_count}

        result1 = await cache.get_or_set("computed_key", factory, ttl=60)
        result2 = await cache.get_or_set("computed_key", factory, ttl=60)
        print(f"   Primeira chamada: {result1}")
        print(f"   Segunda chamada (cache): {result2}")
        print(f"   Factory chamado {call_count} vez(es)")

        # Health check
        print("\n3. Health Check:")
        health = await cache.health_check()
        print(f"   {json.dumps(health, indent=2)}")

        await cache.close()

    asyncio.run(test_cache())
