"""
Sankofa Enterprise Pro - Prediction Cache System
Cache de predições para latência sub-50ms
Reduz latência de 284ms para <30ms em cache hits
"""

import hashlib
import json
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CachedPrediction:
    """Predição cacheada com metadados"""
    transaction_hash: str
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    risk_level: str
    confidence: float
    model_version: str
    detection_reason: List[str]
    cached_at: str
    expires_at: str
    hit_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def is_expired(self) -> bool:
        return datetime.utcnow() > datetime.fromisoformat(self.expires_at.replace('Z', ''))


class PredictionCache:
    """
    Cache de Alta Performance para Predições ML
    
    Features:
    - TTL configurável por tipo de transação
    - LRU eviction para gerenciamento de memória
    - Thread-safe para acesso concorrente
    - Métricas de hit/miss rate
    - Warm-up para transações frequentes
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        max_size: int = 10000,
        default_ttl_seconds: int = 300,
        high_risk_ttl_seconds: int = 60,
        low_risk_ttl_seconds: int = 600
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl_seconds
        self.high_risk_ttl = high_risk_ttl_seconds
        self.low_risk_ttl = low_risk_ttl_seconds
        
        self._cache: OrderedDict[str, CachedPrediction] = OrderedDict()
        self._lock = threading.RLock()
        
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        self._feature_weights = {
            'amount': 0.3,
            'hour': 0.15,
            'channel': 0.15,
            'customer_id': 0.2,
            'device_id': 0.1,
            'is_new_device': 0.1
        }
        
        logger.info(
            f"PredictionCache initialized v{self.VERSION}",
            extra={
                'max_size': max_size,
                'default_ttl': default_ttl_seconds
            }
        )
    
    def _generate_hash(self, transaction: Dict[str, Any]) -> str:
        """Gera hash único para transação baseado em features críticas"""
        key_parts = []
        
        if 'amount' in transaction:
            amount = float(transaction['amount'])
            amount_bucket = int(amount / 100) * 100
            key_parts.append(f"amt:{amount_bucket}")
        
        if 'hour' in transaction:
            key_parts.append(f"hr:{transaction['hour']}")
        
        if 'channel' in transaction:
            key_parts.append(f"ch:{str(transaction['channel']).lower()}")
        
        if 'customer_id' in transaction or 'cliente_cpf' in transaction:
            cust_id = transaction.get('customer_id', transaction.get('cliente_cpf', ''))
            key_parts.append(f"cust:{str(cust_id)[:8]}")
        
        if 'is_new_device' in transaction:
            key_parts.append(f"nd:{transaction['is_new_device']}")
        
        if 'velocity_score' in transaction:
            vel_bucket = round(float(transaction.get('velocity_score', 0)), 1)
            key_parts.append(f"vel:{vel_bucket}")
        
        key_string = "|".join(sorted(key_parts))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_ttl(self, risk_level: str) -> int:
        """Retorna TTL baseado no nível de risco"""
        if risk_level in ['CRITICAL', 'HIGH']:
            return self.high_risk_ttl
        elif risk_level == 'LOW':
            return self.low_risk_ttl
        return self.default_ttl
    
    def get(self, transaction: Dict[str, Any]) -> Optional[CachedPrediction]:
        """Busca predição no cache"""
        tx_hash = self._generate_hash(transaction)
        
        with self._lock:
            if tx_hash in self._cache:
                cached = self._cache[tx_hash]
                
                if cached.is_expired():
                    del self._cache[tx_hash]
                    self._misses += 1
                    return None
                
                self._cache.move_to_end(tx_hash)
                cached.hit_count += 1
                self._hits += 1
                
                logger.debug(f"Cache HIT for {tx_hash[:8]}")
                return cached
            
            self._misses += 1
            return None
    
    def set(
        self,
        transaction: Dict[str, Any],
        is_fraud: bool,
        fraud_probability: float,
        risk_score: float,
        risk_level: str,
        confidence: float,
        model_version: str,
        detection_reason: List[str]
    ) -> str:
        """Armazena predição no cache"""
        tx_hash = self._generate_hash(transaction)
        ttl = self._get_ttl(risk_level)
        
        now = datetime.utcnow()
        expires = now + timedelta(seconds=ttl)
        
        cached = CachedPrediction(
            transaction_hash=tx_hash,
            is_fraud=is_fraud,
            fraud_probability=fraud_probability,
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=confidence,
            model_version=model_version,
            detection_reason=detection_reason,
            cached_at=now.isoformat() + "Z",
            expires_at=expires.isoformat() + "Z",
            hit_count=0
        )
        
        with self._lock:
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                self._evictions += 1
            
            self._cache[tx_hash] = cached
        
        logger.debug(f"Cache SET for {tx_hash[:8]}, TTL={ttl}s")
        return tx_hash
    
    def invalidate(self, transaction: Dict[str, Any]) -> bool:
        """Invalida entrada do cache"""
        tx_hash = self._generate_hash(transaction)
        
        with self._lock:
            if tx_hash in self._cache:
                del self._cache[tx_hash]
                return True
            return False
    
    def clear(self) -> int:
        """Limpa todo o cache"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count
    
    def cleanup_expired(self) -> int:
        """Remove entradas expiradas"""
        removed = 0
        
        with self._lock:
            expired_keys = [
                key for key, cached in self._cache.items()
                if cached.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
                removed += 1
        
        if removed > 0:
            logger.info(f"Cleaned up {removed} expired cache entries")
        
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'version': self.VERSION,
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'hit_rate_percent': round(hit_rate, 2),
                'total_requests': total_requests,
                'memory_usage_approx_kb': len(self._cache) * 2
            }
    
    def warm_up(self, transactions: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> int:
        """Pre-popula cache com transações frequentes"""
        warmed = 0
        
        for txn, pred in zip(transactions, predictions):
            self.set(
                transaction=txn,
                is_fraud=pred.get('is_fraud', False),
                fraud_probability=pred.get('fraud_probability', 0.0),
                risk_score=pred.get('risk_score', 0.0),
                risk_level=pred.get('risk_level', 'LOW'),
                confidence=pred.get('confidence', 0.5),
                model_version=pred.get('model_version', 'WARMUP'),
                detection_reason=pred.get('detection_reason', [])
            )
            warmed += 1
        
        logger.info(f"Cache warmed up with {warmed} entries")
        return warmed


_prediction_cache: Optional[PredictionCache] = None


def get_prediction_cache() -> PredictionCache:
    """Singleton para PredictionCache"""
    global _prediction_cache
    if _prediction_cache is None:
        _prediction_cache = PredictionCache()
    return _prediction_cache


class CachedFraudEngine:
    """
    Wrapper que adiciona cache ao ProductionFraudEngine
    Reduz latência de ~284ms para <30ms em cache hits
    """
    
    def __init__(self, fraud_engine=None, cache: Optional[PredictionCache] = None):
        self.cache = cache or get_prediction_cache()
        self._fraud_engine = fraud_engine
        self._initialized = False
    
    def _get_engine(self):
        """Lazy loading do fraud engine"""
        if self._fraud_engine is None:
            from ml_engine.production_fraud_engine import get_fraud_engine
            self._fraud_engine = get_fraud_engine()
        return self._fraud_engine
    
    def predict_with_cache(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predição com cache - latência <30ms em cache hit
        
        Args:
            transaction: Dicionário com dados da transação
            
        Returns:
            Predição com metadados (cache_hit, processing_time_ms)
        """
        start_time = time.time()
        
        cached = self.cache.get(transaction)
        if cached is not None:
            elapsed_ms = (time.time() - start_time) * 1000
            return {
                'transaction_id': transaction.get('transaction_id', transaction.get('id', '')),
                'is_fraud': cached.is_fraud,
                'fraud_probability': cached.fraud_probability,
                'risk_score': cached.risk_score,
                'risk_level': cached.risk_level,
                'confidence': cached.confidence,
                'model_version': cached.model_version,
                'detection_reason': cached.detection_reason,
                'processing_time_ms': round(elapsed_ms, 2),
                'cache_hit': True,
                'timestamp': datetime.utcnow().isoformat() + "Z"
            }
        
        engine = self._get_engine()
        
        import pandas as pd
        df = pd.DataFrame([transaction])
        
        predictions = engine.predict_detailed(df)
        
        if predictions:
            pred = predictions[0]
            
            self.cache.set(
                transaction=transaction,
                is_fraud=pred.is_fraud,
                fraud_probability=pred.fraud_probability,
                risk_score=pred.risk_score,
                risk_level=pred.risk_level,
                confidence=pred.confidence,
                model_version=pred.model_version,
                detection_reason=pred.detection_reason
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return {
                'transaction_id': pred.transaction_id,
                'is_fraud': pred.is_fraud,
                'fraud_probability': pred.fraud_probability,
                'risk_score': pred.risk_score,
                'risk_level': pred.risk_level,
                'confidence': pred.confidence,
                'model_version': pred.model_version,
                'detection_reason': pred.detection_reason,
                'processing_time_ms': round(elapsed_ms, 2),
                'cache_hit': False,
                'timestamp': pred.timestamp
            }
        
        elapsed_ms = (time.time() - start_time) * 1000
        return {
            'transaction_id': transaction.get('transaction_id', ''),
            'is_fraud': False,
            'fraud_probability': 0.0,
            'risk_score': 0.0,
            'risk_level': 'LOW',
            'confidence': 0.5,
            'model_version': 'FALLBACK',
            'detection_reason': ['No prediction available'],
            'processing_time_ms': round(elapsed_ms, 2),
            'cache_hit': False,
            'timestamp': datetime.utcnow().isoformat() + "Z"
        }
    
    def predict_batch_with_cache(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predição em batch com cache"""
        results = []
        for txn in transactions:
            result = self.predict_with_cache(txn)
            results.append(result)
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache"""
        return self.cache.get_stats()


_cached_engine: Optional[CachedFraudEngine] = None


def get_cached_fraud_engine() -> CachedFraudEngine:
    """Singleton para CachedFraudEngine"""
    global _cached_engine
    if _cached_engine is None:
        _cached_engine = CachedFraudEngine()
    return _cached_engine


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    cache = PredictionCache(max_size=100, default_ttl_seconds=60)
    
    test_txn = {
        'amount': 500,
        'hour': 14,
        'channel': 'PIX',
        'customer_id': 'CUST001'
    }
    
    cache.set(
        transaction=test_txn,
        is_fraud=False,
        fraud_probability=0.15,
        risk_score=0.15,
        risk_level='LOW',
        confidence=0.85,
        model_version='1.0.0',
        detection_reason=['Normal transaction pattern']
    )
    
    cached = cache.get(test_txn)
    print(f"Cache hit: {cached is not None}")
    if cached:
        print(f"Fraud probability: {cached.fraud_probability}")
    
    stats = cache.get_stats()
    print(f"Cache stats: {json.dumps(stats, indent=2)}")
