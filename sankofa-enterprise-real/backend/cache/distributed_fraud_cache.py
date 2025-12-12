#!/usr/bin/env python3
"""
Sistema de Cache Distribuído para Análise de Fraude em Tempo Real
Implementa cache multi-camadas com Redis Cluster e cache local
"""

import time
import hashlib
import threading
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
import logging
import json
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entrada de cache com metadados"""

    value: Any
    created_at: float
    ttl: int
    access_count: int = 0
    last_accessed: float = 0
    size_bytes: int = 0


class LRUCache:
    """Cache LRU thread-safe para cache local"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "size": 0}

    def get(self, key: str) -> Optional[CacheEntry]:
        """Obtém item do cache LRU"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]

                # Verifica TTL
                if time.time() - entry.created_at > entry.ttl:
                    del self.cache[key]
                    self.stats["misses"] += 1
                    return None

                # Move para o final (mais recente)
                self.cache.move_to_end(key)
                entry.access_count += 1
                entry.last_accessed = time.time()

                self.stats["hits"] += 1
                return entry

            self.stats["misses"] += 1
            return None

    def set(self, key: str, entry: CacheEntry):
        """Define item no cache LRU"""
        with self.lock:
            # Remove item existente se houver
            if key in self.cache:
                del self.cache[key]

            # Adiciona novo item
            self.cache[key] = entry

            # Remove itens antigos se necessário
            while len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats["evictions"] += 1

            self.stats["size"] = len(self.cache)

    def delete(self, key: str) -> bool:
        """Remove item do cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats["size"] = len(self.cache)
                return True
            return False

    def clear(self):
        """Limpa todo o cache"""
        with self.lock:
            self.cache.clear()
            self.stats["size"] = 0

    def get_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do cache"""
        with self.lock:
            hit_rate = 0
            if self.stats["hits"] + self.stats["misses"] > 0:
                hit_rate = self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])

            return {
                **self.stats,
                "hit_rate": hit_rate,
                "memory_usage_mb": sum(entry.size_bytes for entry in self.cache.values())
                / 1024
                / 1024,
            }


class DistributedFraudCache:
    """Sistema de cache distribuído multi-camadas para detecção de fraude"""

    def __init__(self, redis_cache_system, local_cache_size: int = 50000):
        self.redis_cache = redis_cache_system
        self.local_cache = LRUCache(local_cache_size)
        self.executor = ThreadPoolExecutor(max_workers=20)

        # Configurações específicas para fraude
        self.cache_layers = {
            "local": {"enabled": True, "priority": 1},
            "redis": {"enabled": True, "priority": 2},
        }

        # TTLs otimizados para diferentes tipos de dados de fraude
        self.fraud_ttls = {
            # Dados de transação
            "transaction_features": 300,  # 5 minutos
            "transaction_risk": 180,  # 3 minutos
            "transaction_velocity": 600,  # 10 minutos
            # Perfis de entidades
            "user_behavior": 3600,  # 1 hora
            "merchant_profile": 7200,  # 2 horas
            "device_fingerprint": 1800,  # 30 minutos
            "ip_reputation": 3600,  # 1 hora
            # Modelos e predições
            "model_prediction": 900,  # 15 minutos
            "ensemble_result": 600,  # 10 minutos
            "feature_importance": 1800,  # 30 minutos
            # Listas e regras
            "blacklist_check": 86400,  # 24 horas
            "whitelist_check": 86400,  # 24 horas
            "rule_evaluation": 1800,  # 30 minutos
            # Contadores e agregações
            "velocity_counter": 3600,  # 1 hora
            "aggregated_stats": 1800,  # 30 minutos
            "time_window_data": 900,  # 15 minutos
            # Dados de sessão
            "user_session": 1800,  # 30 minutos
            "device_session": 3600,  # 1 hora
            # Dados geográficos
            "geo_location": 7200,  # 2 horas
            "geo_velocity": 1800,  # 30 minutos
            # Análises complexas
            "network_analysis": 3600,  # 1 hora
            "pattern_matching": 1800,  # 30 minutos
            "anomaly_detection": 900,  # 15 minutos
        }

        # Prefixos para organização
        self.prefixes = {
            "fraud": "fraud",
            "user": "user",
            "merchant": "merchant",
            "transaction": "txn",
            "model": "ml",
            "rule": "rule",
            "velocity": "vel",
            "geo": "geo",
            "session": "sess",
            "device": "dev",
        }

        logger.info("Sistema de Cache Distribuído para Fraude inicializado")

    def _generate_cache_key(self, category: str, subcategory: str, *identifiers) -> str:
        """Gera chave de cache hierárquica"""
        prefix = self.prefixes.get(category, category)

        # Cria hash dos identificadores para chave determinística
        id_string = ":".join(str(id) for id in identifiers)
        id_hash = hashlib.sha256(id_string.encode()).hexdigest()[:12]

        return f"{prefix}:{subcategory}:{id_hash}"

    def _calculate_size(self, data: Any) -> int:
        """Calcula tamanho aproximado dos dados"""
        try:
            return len(json.dumps(data, default=str).encode("utf-8"))
        except (TypeError, ValueError, OverflowError):
            return len(str(data).encode("utf-8"))

    def _create_cache_entry(self, value: Any, ttl: int) -> CacheEntry:
        """Cria entrada de cache com metadados"""
        return CacheEntry(
            value=value, created_at=time.time(), ttl=ttl, size_bytes=self._calculate_size(value)
        )

    def get(self, category: str, subcategory: str, *identifiers, default=None) -> Any:
        """Obtém valor do cache multi-camadas"""
        key = self._generate_cache_key(category, subcategory, *identifiers)

        # Tenta cache local primeiro
        if self.cache_layers["local"]["enabled"]:
            local_entry = self.local_cache.get(key)
            if local_entry is not None:
                return local_entry.value

        # Tenta Redis
        if self.cache_layers["redis"]["enabled"]:
            redis_value = self.redis_cache.get(key, default)

            if redis_value is not default:
                # Popula cache local
                ttl = self.fraud_ttls.get(subcategory, 3600)
                local_entry = self._create_cache_entry(redis_value, ttl)
                self.local_cache.set(key, local_entry)

                return redis_value

        return default

    def set(
        self, category: str, subcategory: str, *identifiers, value: Any, ttl: int = None
    ) -> bool:
        """Define valor no cache multi-camadas"""
        key = self._generate_cache_key(category, subcategory, *identifiers)
        ttl = ttl or self.fraud_ttls.get(subcategory, 3600)

        success = True

        # Define no cache local
        if self.cache_layers["local"]["enabled"]:
            local_entry = self._create_cache_entry(value, ttl)
            self.local_cache.set(key, local_entry)

        # Define no Redis
        if self.cache_layers["redis"]["enabled"]:
            success = self.redis_cache.set(key, value, ttl)

        return success

    def delete(self, category: str, subcategory: str, *identifiers) -> bool:
        """Remove valor do cache multi-camadas"""
        key = self._generate_cache_key(category, subcategory, *identifiers)

        # Remove do cache local
        local_deleted = self.local_cache.delete(key)

        # Remove do Redis
        redis_deleted = self.redis_cache.delete(key)

        return local_deleted or redis_deleted

    # Métodos específicos para detecção de fraude

    def cache_transaction_features(self, transaction_id: str, features: Dict[str, Any]) -> bool:
        """Cache features de transação"""
        return self.set("transaction", "transaction_features", transaction_id, value=features)

    def get_transaction_features(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Obtém features de transação"""
        return self.get("transaction", "transaction_features", transaction_id)

    def cache_user_behavior(self, user_id: str, behavior_data: Dict[str, Any]) -> bool:
        """Cache comportamento do usuário"""
        return self.set("user", "user_behavior", user_id, value=behavior_data)

    def get_user_behavior(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtém comportamento do usuário"""
        return self.get("user", "user_behavior", user_id)

    def cache_model_prediction(
        self, model_name: str, input_hash: str, prediction: Dict[str, Any]
    ) -> bool:
        """Cache predição de modelo"""
        return self.set("model", "model_prediction", model_name, input_hash, value=prediction)

    def get_model_prediction(self, model_name: str, input_hash: str) -> Optional[Dict[str, Any]]:
        """Obtém predição de modelo"""
        return self.get("model", "model_prediction", model_name, input_hash)

    # CORRECAO 10/10: Lua script para operação atômica INCR+EXPIRE
    # Evita race condition onde TTL nunca é definido se outro thread incrementa entre INCR e EXPIRE
    _ATOMIC_INCR_SCRIPT = """
    local count = redis.call('INCR', KEYS[1])
    if count == 1 then
        redis.call('EXPIRE', KEYS[1], ARGV[1])
    end
    return count
    """
    _atomic_incr_sha = None

    def increment_velocity_counter(
        self, counter_type: str, entity_id: str, time_window: str
    ) -> int:
        """
        Incrementa contador de velocidade de forma atômica e thread-safe.

        CORRECAO 10/10: Usa Lua script para garantir atomicidade da operação
        INCR + EXPIRE, evitando race condition onde múltiplas threads incrementam
        simultaneamente e apenas uma delas define o TTL corretamente.

        Args:
            counter_type: Tipo do contador (ex: 'card_usage', 'user_transactions')
            entity_id: ID da entidade sendo contada
            time_window: Janela de tempo (ex: '1h', '24h')

        Returns:
            Valor atual do contador após incremento
        """
        key = self._generate_cache_key("velocity", counter_type, entity_id, time_window)
        ttl = self.fraud_ttls.get("velocity_counter", 3600)

        client = self.redis_cache.connection_manager.get_client()

        try:
            # Tenta usar Lua script para operação atômica
            # Registra o script se ainda não foi registrado
            if DistributedFraudCache._atomic_incr_sha is None:
                DistributedFraudCache._atomic_incr_sha = client.script_load(
                    self._ATOMIC_INCR_SCRIPT
                )

            # Executa script atômico
            count = client.evalsha(
                DistributedFraudCache._atomic_incr_sha,
                1,  # número de keys
                key,
                ttl
            )
            return int(count)

        except Exception as e:
            # Fallback: Se Lua scripts não estiverem disponíveis,
            # usa pipeline para minimizar (mas não eliminar) race condition
            logger.warning(f"Lua script falhou, usando fallback com pipeline: {e}")

            try:
                pipe = client.pipeline(transaction=True)
                pipe.incr(key)
                pipe.expire(key, ttl)
                results = pipe.execute()
                return results[0]  # Retorna o resultado do INCR
            except Exception as pipe_error:
                # Último fallback: operação simples (pode ter race condition)
                logger.error(f"Pipeline falhou, usando operação simples: {pipe_error}")
                count = client.incr(key)
                client.expire(key, ttl)  # Sempre define TTL para garantir expiração
                return count

    def get_velocity_counter(self, counter_type: str, entity_id: str, time_window: str) -> int:
        """Obtém contador de velocidade"""
        return self.get("velocity", counter_type, entity_id, time_window, default=0)

    def cache_geo_location(self, ip_address: str, location_data: Dict[str, Any]) -> bool:
        """Cache localização geográfica"""
        return self.set("geo", "geo_location", ip_address, value=location_data)

    def get_geo_location(self, ip_address: str) -> Optional[Dict[str, Any]]:
        """Obtém localização geográfica"""
        return self.get("geo", "geo_location", ip_address)

    def cache_device_fingerprint(self, device_id: str, fingerprint: Dict[str, Any]) -> bool:
        """Cache fingerprint de dispositivo"""
        return self.set("device", "device_fingerprint", device_id, value=fingerprint)

    def get_device_fingerprint(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Obtém fingerprint de dispositivo"""
        return self.get("device", "device_fingerprint", device_id)

    def is_blacklisted(self, list_type: str, identifier: str) -> bool:
        """Verifica se item está na blacklist"""
        result = self.get("rule", "blacklist_check", list_type, identifier)
        return result is not None

    def add_to_blacklist(self, list_type: str, identifier: str, reason: str = None) -> bool:
        """Adiciona item à blacklist"""
        data = {"added_at": datetime.now().isoformat(), "reason": reason, "list_type": list_type}
        return self.set("rule", "blacklist_check", list_type, identifier, value=data)

    def cache_ensemble_result(self, input_hash: str, ensemble_result: Dict[str, Any]) -> bool:
        """Cache resultado de ensemble"""
        return self.set("model", "ensemble_result", input_hash, value=ensemble_result)

    def get_ensemble_result(self, input_hash: str) -> Optional[Dict[str, Any]]:
        """Obtém resultado de ensemble"""
        return self.get("model", "ensemble_result", input_hash)

    def batch_get(self, requests: List[Tuple[str, str, tuple]]) -> Dict[str, Any]:
        """Obtém múltiplos valores em lote"""
        results = {}

        # Processa em paralelo
        futures = []
        for category, subcategory, identifiers in requests:
            future = self.executor.submit(self.get, category, subcategory, *identifiers)
            futures.append((future, (category, subcategory, identifiers)))

        for future, (category, subcategory, identifiers) in futures:
            try:
                key = self._generate_cache_key(category, subcategory, *identifiers)
                results[key] = future.result(timeout=1.0)
            except Exception as e:
                logger.error(f"Erro no batch_get: {e}")
                results[key] = None

        return results

    def batch_set(self, requests: List[Tuple[str, str, tuple, Any, int]]) -> Dict[str, bool]:
        """Define múltiplos valores em lote"""
        results = {}

        # Processa em paralelo
        futures = []
        for category, subcategory, identifiers, value, ttl in requests:
            future = self.executor.submit(
                self.set, category, subcategory, *identifiers, value=value, ttl=ttl
            )
            futures.append((future, (category, subcategory, identifiers)))

        for future, (category, subcategory, identifiers) in futures:
            try:
                key = self._generate_cache_key(category, subcategory, *identifiers)
                results[key] = future.result(timeout=1.0)
            except Exception as e:
                logger.error(f"Erro no batch_set: {e}")
                results[key] = False

        return results

    def warm_up_cache(self, warm_up_data: Dict[str, Any]):
        """Aquece o cache com dados frequentemente acessados"""
        logger.info("Iniciando warm-up do cache...")

        warm_up_requests = []

        # Prepara requests de warm-up
        for category, subcategories in warm_up_data.items():
            for subcategory, items in subcategories.items():
                for identifiers, value in items.items():
                    if isinstance(identifiers, str):
                        identifiers = (identifiers,)

                    ttl = self.fraud_ttls.get(subcategory, 3600)
                    warm_up_requests.append((category, subcategory, identifiers, value, ttl))

        # Executa warm-up em lote
        results = self.batch_set(warm_up_requests)

        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Warm-up concluído: {success_count}/{len(results)} itens carregados")

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas completas do sistema de cache"""
        local_stats = self.local_cache.get_stats()
        redis_stats = self.redis_cache.get_stats()

        return {
            "local_cache": local_stats,
            "redis_cache": redis_stats,
            "cache_layers": self.cache_layers,
            "fraud_ttls_configured": len(self.fraud_ttls),
            "prefixes_configured": len(self.prefixes),
            "system_health": {
                "local_healthy": True,
                "redis_healthy": self.redis_cache.connection_manager.is_healthy(),
            },
        }

    def clear_fraud_cache(self, category: str = None) -> Dict[str, int]:
        """Limpa cache de fraude por categoria"""
        if category:
            # Limpa categoria específica
            pattern = f"{self.prefixes.get(category, category)}:*"
            redis_deleted = self.redis_cache.clear_pattern(pattern)

            # Limpa cache local (não há padrão, então limpa tudo)
            self.local_cache.clear()

            return {f"{category}_redis": redis_deleted, "local_cleared": True}
        else:
            # Limpa tudo
            results = {}
            for cat, prefix in self.prefixes.items():
                pattern = f"{prefix}:*"
                deleted = self.redis_cache.clear_pattern(pattern)
                results[f"{cat}_redis"] = deleted

            self.local_cache.clear()
            results["local_cleared"] = True

            return results


# Teste do sistema
if __name__ == "__main__":
    logger.info("🚀 Testando Sistema de Cache Distribuído para Fraude...")

    # Simula sistema Redis (seria importado em produção)
    class MockRedisSystem:
        def __init__(self):
            self.data = {}
            self.counters = {}

            # Mock connection manager
            class MockConnectionManager:
                def is_healthy(self):
                    return True

                def get_client(self):
                    return MockRedisClient(self.parent)

                def __init__(self, parent):
                    self.parent = parent

            # Mock Redis client
            # CORRECAO 10/10: Adicionado suporte a Lua scripts e pipelines para testes
            class MockRedisClient:
                def __init__(self, parent):
                    self.parent = parent
                    self._scripts = {}

                def incr(self, key):
                    self.parent.counters[key] = self.parent.counters.get(key, 0) + 1
                    return self.parent.counters[key]

                def expire(self, key, ttl):
                    return True

                def script_load(self, script):
                    """Mock para carregar Lua script"""
                    import hashlib
                    sha = hashlib.sha1(script.encode()).hexdigest()
                    self._scripts[sha] = script
                    return sha

                def evalsha(self, sha, numkeys, *args):
                    """Mock para executar Lua script - simula INCR+EXPIRE atômico"""
                    key = args[0]
                    ttl = args[1] if len(args) > 1 else 3600
                    count = self.incr(key)
                    self.expire(key, ttl)
                    return count

                def pipeline(self, transaction=True):
                    """Mock para pipeline Redis"""
                    return MockPipeline(self)

            class MockPipeline:
                def __init__(self, client):
                    self.client = client
                    self._commands = []

                def incr(self, key):
                    self._commands.append(('incr', key))
                    return self

                def expire(self, key, ttl):
                    self._commands.append(('expire', key, ttl))
                    return self

                def execute(self):
                    results = []
                    for cmd in self._commands:
                        if cmd[0] == 'incr':
                            results.append(self.client.incr(cmd[1]))
                        elif cmd[0] == 'expire':
                            results.append(self.client.expire(cmd[1], cmd[2]))
                    return results

            self.connection_manager = MockConnectionManager(self)

        def get(self, key, default=None):
            return self.data.get(key, default)

        def set(self, key, value, ttl):
            self.data[key] = value
            return True

        def delete(self, key):
            return self.data.pop(key, None) is not None

        def clear_pattern(self, pattern):
            keys_to_delete = [k for k in self.data.keys() if pattern.replace("*", "") in k]
            for key in keys_to_delete:
                del self.data[key]
            return len(keys_to_delete)

        def get_stats(self):
            return {"operations": {"hits": 10, "misses": 2}}

    # Inicializa sistema
    mock_redis = MockRedisSystem()
    fraud_cache = DistributedFraudCache(mock_redis)

    # Testa operações básicas
    transaction_id = "txn_123456"
    features = {
        "amount": 1500.00,
        "merchant_category": "grocery",
        "time_of_day": "evening",
        "day_of_week": "friday",
    }

    # Cache features
    success = fraud_cache.cache_transaction_features(transaction_id, features)
    logger.info(f"✅ Features cacheadas: {success}")

    # Recupera features
    cached_features = fraud_cache.get_transaction_features(transaction_id)
    logger.info(f"✅ Features recuperadas: {cached_features['amount']}")

    # Testa contador de velocidade
    count = fraud_cache.increment_velocity_counter("card_usage", "1234567890", "1h")
    logger.info(f"✅ Contador de velocidade: {count}")

    # Testa cache de comportamento
    user_id = "user_789"
    behavior = {
        "avg_transaction_amount": 250.00,
        "preferred_merchants": ["grocery", "gas"],
        "typical_hours": [9, 12, 18, 20],
    }

    fraud_cache.cache_user_behavior(user_id, behavior)
    cached_behavior = fraud_cache.get_user_behavior(user_id)
    logger.info(f"✅ Comportamento do usuário: {cached_behavior['avg_transaction_amount']}")

    # Testa blacklist
    fraud_cache.add_to_blacklist("ip", "192.168.1.100", "Suspicious activity")
    is_blacklisted = fraud_cache.is_blacklisted("ip", "192.168.1.100")
    logger.info(f"✅ IP na blacklist: {is_blacklisted}")

    # Estatísticas
    stats = fraud_cache.get_comprehensive_stats()
    logger.info(f"✅ Cache local hit rate: {stats['local_cache']['hit_rate']:.2%}")
    logger.info(f"✅ Sistema saudável: {stats['system_health']['redis_healthy']}")

    logger.info("🚀 Teste do Sistema de Cache Distribuído para Fraude concluído!")
