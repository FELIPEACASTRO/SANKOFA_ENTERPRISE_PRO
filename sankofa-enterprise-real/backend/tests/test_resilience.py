"""
Sankofa Enterprise Pro - Resilience Tests
Tests for system behavior under failure conditions
"""

import pytest
import os
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '.')

from cache.redis_cache_system import (
    RedisConnectionManager,
    RedisCacheSystem,
    CacheConfig,
    InMemoryCache,
)
from infrastructure.redis_cluster import RedisCache, RedisClusterConfig, MemoryCache


class TestRedisFallback:
    """Tests for Redis fallback behavior"""
    
    def test_memory_cache_in_development(self):
        """Test that MemoryCache is used in development without Redis"""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            if "REDIS_URL" in os.environ:
                del os.environ["REDIS_URL"]
            
            config = CacheConfig()
            manager = RedisConnectionManager(config)
            
            assert manager._use_memory_only == True
            assert manager._is_healthy == True
    
    def test_memory_cache_fallback_production(self):
        """Test that production falls back to MemoryCache when Redis unavailable"""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False):
            if "REDIS_URL" in os.environ:
                del os.environ["REDIS_URL"]
            
            config = CacheConfig()
            manager = RedisConnectionManager(config)
            
            assert manager._is_healthy == True
    
    def test_memory_cache_operations(self):
        """Test InMemoryCache basic operations"""
        cache = InMemoryCache()
        
        cache.setex("test_key", 3600, b"test_value")
        result = cache.get("test_key")
        
        assert result == b"test_value"
        
        cache.delete("test_key")
        result = cache.get("test_key")
        
        assert result is None
    
    def test_memory_cache_lru_eviction(self):
        """Test LRU eviction in InMemoryCache"""
        cache = InMemoryCache(max_size=5)
        
        for i in range(10):
            cache.setex(f"key_{i}", 3600, f"value_{i}".encode())
        
        assert len(cache._cache) <= 5


class TestRedisClusterFallback:
    """Tests for RedisCluster fallback behavior"""
    
    def test_development_mode_skips_redis(self):
        """Test that development mode skips Redis connection"""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            if "REDIS_URL" in os.environ:
                del os.environ["REDIS_URL"]
            
            config = RedisClusterConfig()
            cache = RedisCache(config)
            
            assert cache._connected == False
            assert cache._use_fallback() == True
    
    def test_fallback_operations_work(self):
        """Test that fallback cache operations work correctly"""
        config = RedisClusterConfig()
        
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            cache = RedisCache(config)
            
            cache.set("test_key", "test_value", ttl=60)
            result = cache.get("test_key")
            
            assert result == "test_value"
    
    def test_memory_cache_stats(self):
        """Test MemoryCache statistics"""
        cache = MemoryCache()
        
        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("missing_key")
        
        stats = cache.get_stats()
        
        assert stats["type"] == "memory"
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1


class TestMLModelResilience:
    """Tests for ML model resilience"""
    
    def test_model_handles_missing_features(self):
        """Test model handles missing features gracefully"""
        from ml_engine.production_fraud_engine import ProductionFraudEngine
        
        engine = ProductionFraudEngine()
        
        X = pd.DataFrame([{
            'amount': 1000,
        }])
        
        missing = engine._validate_required_features(X)
        assert isinstance(missing, list)
    
    def test_model_handles_invalid_data(self):
        """Test model handles invalid data gracefully"""
        from ml_engine.production_fraud_engine import ProductionFraudEngine
        
        engine = ProductionFraudEngine()
        
        X = pd.DataFrame([{
            'amount': 0,
            'hour': 14,
            'channel': 'web',
        }])
        
        X_filled = X.fillna(0)
        
        predictions = engine.predict_detailed(X_filled)
        assert len(predictions) == 1
    
    def test_model_handles_empty_dataframe(self):
        """Test model handles empty dataframe"""
        from ml_engine.production_fraud_engine import ProductionFraudEngine
        
        engine = ProductionFraudEngine()
        
        X = pd.DataFrame(columns=['amount', 'hour', 'channel'])
        
        predictions = engine.predict_detailed(X)
        assert len(predictions) == 0


class TestDatabaseResilience:
    """Tests for database connection resilience"""
    
    def test_database_connection_error_handling(self):
        """Test that database errors are handled gracefully"""
        from utils.error_handling import handle_error, DatabaseError
        
        error = DatabaseError("Connection timeout")
        context = handle_error(error, raise_exception=False)
        
        assert context is not None
        assert "database" in context.category.value


class TestErrorHandlingResilience:
    """Tests for error handling under various conditions"""
    
    def test_validation_error_returns_400(self):
        """Test that ValidationError is handled with 400 status"""
        from utils.error_handling import ValidationError, ErrorCategory
        
        error = ValidationError("Invalid input")
        context = error.get_context()
        
        assert context.category == ErrorCategory.VALIDATION
    
    def test_error_context_serialization(self):
        """Test that error context can be serialized"""
        from utils.error_handling import ValidationError
        
        error = ValidationError("Test", context={"field": "amount"})
        context = error.get_context()
        
        context_dict = context.to_dict()
        
        assert isinstance(context_dict, dict)
        assert "error_id" in context_dict
        assert "category" in context_dict
    
    def test_unknown_error_handling(self):
        """Test handling of unexpected errors"""
        from utils.error_handling import handle_error
        
        error = RuntimeError("Unexpected error")
        context = handle_error(error, raise_exception=False)
        
        assert context is not None
        assert context.message is not None


class TestCacheResilience:
    """Tests for cache system resilience"""
    
    def test_cache_system_initialization(self):
        """Test RedisCacheSystem initializes correctly"""
        config = CacheConfig()
        system = RedisCacheSystem(config)
        
        assert system is not None
        assert system.connection_manager is not None
    
    def test_cache_serialization_resilience(self):
        """Test cache handles various data types"""
        from cache.redis_cache_system import CacheSerializer
        
        test_data = [
            "string",
            123,
            {"key": "value"},
            [1, 2, 3],
            True,
        ]
        
        for data in test_data:
            serialized = CacheSerializer.serialize(data)
            deserialized = CacheSerializer.deserialize(serialized)
            
            assert deserialized == data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
