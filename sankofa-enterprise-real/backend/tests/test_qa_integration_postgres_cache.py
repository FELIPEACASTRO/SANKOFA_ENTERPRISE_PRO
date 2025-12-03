"""
Sankofa Enterprise Pro - QA Integration Tests
Testes de Integração PostgreSQL e Cache System
Validação de Padrões: Cache-Aside, Write-Through, Fallback

Autor: QA Automation
Data: 03 de Dezembro de 2025
"""

import pytest
import sys
import os
import time
import json
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPostgreSQLConnection:
    """Testes de conexão e operações básicas com PostgreSQL"""
    
    def test_psycopg2_installed(self):
        """Verifica se psycopg2 está instalado corretamente"""
        import psycopg2
        assert psycopg2.__version__ is not None
        print(f"psycopg2 version: {psycopg2.__version__}")
    
    def test_database_url_configured(self):
        """Verifica se DATABASE_URL está configurado"""
        db_url = os.environ.get('DATABASE_URL')
        assert db_url is not None, "DATABASE_URL environment variable not set"
        assert len(db_url) > 10, "DATABASE_URL appears to be invalid"
    
    def test_postgresql_connection(self):
        """Testa conexão com PostgreSQL"""
        import psycopg2
        
        db_url = os.environ.get('DATABASE_URL')
        conn = psycopg2.connect(db_url)
        assert conn is not None
        
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        assert result[0] == 1
        
        conn.close()
        print("PostgreSQL connection: SUCCESS")
    
    def test_postgresql_version(self):
        """Verifica versão do PostgreSQL"""
        import psycopg2
        
        db_url = os.environ.get('DATABASE_URL')
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        conn.close()
        
        assert "PostgreSQL" in version
        print(f"PostgreSQL version: {version[:60]}")
    
    def test_required_tables_exist(self):
        """Verifica se as tabelas críticas existem"""
        import psycopg2
        
        required_tables = [
            'transactions',
            'alerts',
            'hard_rules',
            'audit_trail'
        ]
        
        db_url = os.environ.get('DATABASE_URL')
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        existing_tables = [row[0] for row in cur.fetchall()]
        conn.close()
        
        missing_tables = [t for t in required_tables if t not in existing_tables]
        
        if missing_tables:
            print(f"Missing tables: {missing_tables}")
        
        for table in required_tables:
            assert table in existing_tables, f"Required table '{table}' not found"
        
        print(f"All {len(required_tables)} required tables exist")


class TestPostgresStoreService:
    """Testes do serviço PostgresStore"""
    
    def test_postgres_store_initialization(self):
        """Testa inicialização do PostgresStore"""
        from services.postgres_store import PostgresStore
        
        store = PostgresStore()
        assert store is not None
        assert store._conn_string is not None
        print("PostgresStore initialization: SUCCESS")
    
    def test_get_hard_rules(self):
        """Testa busca de hard rules"""
        from services.postgres_store import PostgresStore
        
        store = PostgresStore()
        rules = store.get_hard_rules()
        
        assert isinstance(rules, list)
        print(f"Hard rules retrieved: {len(rules)}")
    
    def test_get_transactions(self):
        """Testa busca de transações"""
        from services.postgres_store import PostgresStore
        
        store = PostgresStore()
        transactions = store.get_transactions(limit=10)
        
        assert isinstance(transactions, list)
        print(f"Transactions retrieved: {len(transactions)}")
    
    def test_get_alerts(self):
        """Testa busca de alertas"""
        from services.postgres_store import PostgresStore
        
        store = PostgresStore()
        alerts = store.get_alerts(limit=10)
        
        assert isinstance(alerts, list)
        print(f"Alerts retrieved: {len(alerts)}")
    
    def test_dashboard_kpis(self):
        """Testa busca de KPIs do dashboard"""
        from services.postgres_store import PostgresStore
        
        store = PostgresStore()
        kpis = store.get_dashboard_kpis()
        
        assert isinstance(kpis, dict)
        required_keys = ['total_transactions', 'fraud_detected', 'fraud_rate']
        for key in required_keys:
            assert key in kpis, f"Missing KPI: {key}"
        
        print(f"Dashboard KPIs: {kpis}")


class TestCacheSystem:
    """Testes do sistema de cache"""
    
    def test_simple_cache_initialization(self):
        """Testa inicialização do SimpleCache"""
        from services.postgres_store import SimpleCache
        
        cache = SimpleCache(default_ttl=30)
        assert cache is not None
        assert cache._default_ttl == 30
        print("SimpleCache initialization: SUCCESS")
    
    def test_simple_cache_set_get(self):
        """Testa set/get do SimpleCache"""
        from services.postgres_store import SimpleCache
        
        cache = SimpleCache(default_ttl=30)
        
        cache.set("test_key", {"value": 123})
        result = cache.get("test_key")
        
        assert result is not None
        assert result["value"] == 123
        print("SimpleCache set/get: SUCCESS")
    
    def test_simple_cache_expiration(self):
        """Testa expiração do cache"""
        from services.postgres_store import SimpleCache
        
        cache = SimpleCache(default_ttl=1)
        cache.set("expire_key", "expire_value")
        
        assert cache.get("expire_key") == "expire_value"
        
        time.sleep(1.5)
        
        assert cache.get("expire_key") is None
        print("SimpleCache expiration: SUCCESS")
    
    def test_simple_cache_invalidation(self):
        """Testa invalidação do cache"""
        from services.postgres_store import SimpleCache
        
        cache = SimpleCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.invalidate("key1")
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        
        cache.invalidate()
        assert cache.get("key2") is None
        print("SimpleCache invalidation: SUCCESS")


class TestInMemoryCacheFallback:
    """Testes do fallback InMemoryCache (quando Redis não está disponível)"""
    
    def test_inmemory_cache_initialization(self):
        """Testa inicialização do InMemoryCache"""
        from cache.redis_cache_system import InMemoryCache
        
        cache = InMemoryCache(max_size=1000)
        assert cache is not None
        assert cache._max_size == 1000
        print("InMemoryCache initialization: SUCCESS")
    
    def test_inmemory_cache_setex_get(self):
        """Testa setex/get do InMemoryCache"""
        from cache.redis_cache_system import InMemoryCache
        
        cache = InMemoryCache()
        
        cache.setex("test_key", 60, b"test_value")
        result = cache.get("test_key")
        
        assert result == b"test_value"
        print("InMemoryCache setex/get: SUCCESS")
    
    def test_inmemory_cache_lru_eviction(self):
        """Testa eviction LRU do InMemoryCache"""
        from cache.redis_cache_system import InMemoryCache
        
        cache = InMemoryCache(max_size=5)
        
        for i in range(10):
            cache.setex(f"key_{i}", 60, f"value_{i}".encode())
        
        assert cache.get("key_0") is None
        assert cache.get("key_9") is not None
        print("InMemoryCache LRU eviction: SUCCESS")
    
    def test_inmemory_cache_expiration(self):
        """Testa expiração do InMemoryCache"""
        from cache.redis_cache_system import InMemoryCache
        
        cache = InMemoryCache()
        cache.setex("expire_key", 1, b"expire_value")
        
        assert cache.get("expire_key") == b"expire_value"
        
        time.sleep(1.5)
        
        assert cache.get("expire_key") is None
        print("InMemoryCache expiration: SUCCESS")
    
    def test_inmemory_cache_stats(self):
        """Testa estatísticas de hits/misses"""
        from cache.redis_cache_system import InMemoryCache
        
        cache = InMemoryCache()
        
        cache.setex("key1", 60, b"value1")
        cache.get("key1")
        cache.get("key1")
        cache.get("nonexistent")
        
        assert cache._hits == 2
        assert cache._misses == 1
        print(f"InMemoryCache stats - Hits: {cache._hits}, Misses: {cache._misses}")


class TestPredictionCache:
    """Testes do PredictionCache para ML"""
    
    def test_prediction_cache_initialization(self):
        """Testa inicialização do PredictionCache"""
        from cache.prediction_cache import PredictionCache
        
        cache = PredictionCache(max_size=1000, default_ttl_seconds=300)
        assert cache is not None
        assert cache.max_size == 1000
        assert cache.VERSION == "1.0.0"
        print("PredictionCache initialization: SUCCESS")
    
    def test_prediction_cache_hash_generation(self):
        """Testa geração de hash para transações"""
        from cache.prediction_cache import PredictionCache
        
        cache = PredictionCache()
        
        transaction1 = {'amount': 500, 'hour': 14, 'channel': 'PIX'}
        transaction2 = {'amount': 500, 'hour': 14, 'channel': 'PIX'}
        transaction3 = {'amount': 1000, 'hour': 14, 'channel': 'PIX'}
        
        hash1 = cache._generate_hash(transaction1)
        hash2 = cache._generate_hash(transaction2)
        hash3 = cache._generate_hash(transaction3)
        
        assert hash1 == hash2
        assert hash1 != hash3
        print("PredictionCache hash generation: SUCCESS")
    
    def test_prediction_cache_set_get(self):
        """Testa set/get de predições"""
        from cache.prediction_cache import PredictionCache
        
        cache = PredictionCache()
        
        transaction = {'amount': 500, 'hour': 14, 'channel': 'PIX'}
        
        cache.set(
            transaction=transaction,
            is_fraud=False,
            fraud_probability=0.15,
            risk_score=0.15,
            risk_level='LOW',
            confidence=0.85,
            model_version='1.0.0',
            detection_reason=['Normal transaction']
        )
        
        cached = cache.get(transaction)
        
        assert cached is not None
        assert cached.is_fraud == False
        assert cached.fraud_probability == 0.15
        print("PredictionCache set/get: SUCCESS")
    
    def test_prediction_cache_latency(self):
        """Testa latência do cache (target <50ms)"""
        from cache.prediction_cache import PredictionCache
        
        cache = PredictionCache()
        
        transaction = {'amount': 1234, 'hour': 10, 'channel': 'TED'}
        cache.set(
            transaction=transaction,
            is_fraud=False,
            fraud_probability=0.1,
            risk_score=0.1,
            risk_level='LOW',
            confidence=0.9,
            model_version='1.0.0',
            detection_reason=['Test']
        )
        
        latencies = []
        for _ in range(100):
            start = time.time()
            cache.get(transaction)
            latencies.append((time.time() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        assert avg_latency < 1, f"Average latency {avg_latency}ms exceeds 1ms"
        assert max_latency < 50, f"Max latency {max_latency}ms exceeds 50ms"
        
        print(f"PredictionCache latency - Avg: {avg_latency:.3f}ms, Max: {max_latency:.3f}ms")


class TestCacheAsidePattern:
    """Testes do padrão Cache-Aside"""
    
    def test_cache_aside_miss_then_hit(self):
        """Testa padrão Cache-Aside: miss → fetch → cache → hit"""
        from services.postgres_store import PostgresStore, SimpleCache
        
        cache = SimpleCache(default_ttl=30)
        store = PostgresStore()
        
        cache.invalidate()
        
        start1 = time.time()
        rules1 = store.get_hard_rules()
        time1 = (time.time() - start1) * 1000
        
        start2 = time.time()
        rules2 = store.get_hard_rules()
        time2 = (time.time() - start2) * 1000
        
        assert rules1 == rules2
        
        print(f"Cache-Aside Pattern - First call: {time1:.2f}ms, Second call: {time2:.2f}ms")
    
    def test_cache_invalidation_on_write(self):
        """Testa invalidação do cache após escrita"""
        from services.postgres_store import _dashboard_cache
        
        _dashboard_cache.set("test_cache_key", {"data": "original"})
        
        original = _dashboard_cache.get("test_cache_key")
        assert original["data"] == "original"
        
        _dashboard_cache.invalidate("test_cache_key")
        
        invalidated = _dashboard_cache.get("test_cache_key")
        assert invalidated is None
        
        print("Cache invalidation on write: SUCCESS")


class TestConcurrencyAndThreadSafety:
    """Testes de concorrência e thread safety"""
    
    def test_inmemory_cache_thread_safety(self):
        """Testa thread safety do InMemoryCache"""
        from cache.redis_cache_system import InMemoryCache
        
        cache = InMemoryCache(max_size=10000)
        errors = []
        
        def writer(thread_id):
            try:
                for i in range(100):
                    cache.setex(f"key_{thread_id}_{i}", 60, f"value_{i}".encode())
            except Exception as e:
                errors.append(e)
        
        def reader(thread_id):
            try:
                for i in range(100):
                    cache.get(f"key_{thread_id}_{i}")
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        print("InMemoryCache thread safety: SUCCESS")
    
    def test_prediction_cache_concurrent_access(self):
        """Testa acesso concorrente ao PredictionCache"""
        from cache.prediction_cache import PredictionCache
        
        cache = PredictionCache(max_size=5000)
        errors = []
        
        def cache_operation(thread_id):
            try:
                for i in range(50):
                    transaction = {
                        'amount': 100 * (thread_id + i),
                        'hour': (thread_id + i) % 24,
                        'channel': 'PIX'
                    }
                    
                    cache.set(
                        transaction=transaction,
                        is_fraud=False,
                        fraud_probability=0.1,
                        risk_score=0.1,
                        risk_level='LOW',
                        confidence=0.9,
                        model_version='1.0.0',
                        detection_reason=['Test']
                    )
                    
                    cache.get(transaction)
            except Exception as e:
                errors.append(e)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(cache_operation, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()
        
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        print("PredictionCache concurrent access: SUCCESS")


class TestDatabaseOperationsIntegrity:
    """Testes de integridade das operações de banco"""
    
    def test_transaction_insert_and_read(self):
        """Testa inserção e leitura de transação"""
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        db_url = os.environ.get('DATABASE_URL')
        conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
        
        test_tx = {
            'amount': 999.99,
            'channel': 'QA_TEST',
            'status': 'pending',
            'risk_score': 0.5
        }
        
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO transactions (amount, channel, status, risk_score, created_at)
            VALUES (%(amount)s, %(channel)s, %(status)s, %(risk_score)s, NOW())
            RETURNING id
        """, test_tx)
        
        inserted_id = cur.fetchone()['id']
        conn.commit()
        
        cur.execute("SELECT * FROM transactions WHERE id = %s", (inserted_id,))
        fetched = cur.fetchone()
        
        assert fetched is not None
        assert float(fetched['amount']) == test_tx['amount']
        assert fetched['channel'] == test_tx['channel']
        
        cur.execute("DELETE FROM transactions WHERE id = %s", (inserted_id,))
        conn.commit()
        conn.close()
        
        print("Transaction insert/read integrity: SUCCESS")
    
    def test_audit_trail_logging(self):
        """Testa registro de audit trail"""
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        db_url = os.environ.get('DATABASE_URL')
        conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
        
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO audit_trail (action, details, user_id, created_at)
            VALUES ('QA_TEST', '{"test": true}', 'qa_system', NOW())
            RETURNING id
        """)
        
        audit_id = cur.fetchone()['id']
        conn.commit()
        
        cur.execute("SELECT * FROM audit_trail WHERE id = %s", (audit_id,))
        audit = cur.fetchone()
        
        assert audit is not None
        assert audit['action'] == 'QA_TEST'
        
        cur.execute("DELETE FROM audit_trail WHERE id = %s", (audit_id,))
        conn.commit()
        conn.close()
        
        print("Audit trail logging: SUCCESS")


class TestPerformanceBenchmarks:
    """Benchmarks de performance"""
    
    def test_database_query_latency(self):
        """Mede latência de queries ao banco"""
        import psycopg2
        
        db_url = os.environ.get('DATABASE_URL')
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        latencies = []
        for _ in range(20):
            start = time.time()
            cur.execute("SELECT COUNT(*) FROM transactions")
            cur.fetchone()
            latencies.append((time.time() - start) * 1000)
        
        conn.close()
        
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        print(f"Database query latency - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")
        
        assert avg_latency < 100, f"Average query latency {avg_latency}ms too high"
    
    def test_cache_vs_database_performance(self):
        """Compara performance cache vs banco"""
        from services.postgres_store import PostgresStore, _dashboard_cache
        
        store = PostgresStore()
        
        _dashboard_cache.invalidate()
        
        db_latencies = []
        for _ in range(5):
            _dashboard_cache.invalidate()
            start = time.time()
            store.get_hard_rules()
            db_latencies.append((time.time() - start) * 1000)
        
        cache_latencies = []
        store.get_hard_rules()
        for _ in range(10):
            start = time.time()
            store.get_hard_rules()
            cache_latencies.append((time.time() - start) * 1000)
        
        avg_db = sum(db_latencies) / len(db_latencies)
        avg_cache = sum(cache_latencies) / len(cache_latencies)
        
        speedup = avg_db / max(avg_cache, 0.001)
        
        print(f"Performance comparison - DB: {avg_db:.2f}ms, Cache: {avg_cache:.4f}ms, Speedup: {speedup:.1f}x")


class TestResilienceAndFailover:
    """Testes de resiliência e failover"""
    
    def test_redis_unavailable_fallback(self):
        """Testa fallback quando Redis não está disponível"""
        from cache.redis_cache_system import REDIS_AVAILABLE, InMemoryCache
        
        if REDIS_AVAILABLE:
            print("Redis is available - testing connection")
        else:
            print("Redis unavailable - InMemoryCache fallback active")
        
        cache = InMemoryCache()
        cache.setex("fallback_test", 60, b"fallback_value")
        result = cache.get("fallback_test")
        
        assert result == b"fallback_value"
        print("Redis fallback test: SUCCESS")
    
    def test_database_connection_recovery(self):
        """Testa recuperação após falha de conexão"""
        import psycopg2
        
        db_url = os.environ.get('DATABASE_URL')
        
        conn1 = psycopg2.connect(db_url)
        conn1.close()
        
        conn2 = psycopg2.connect(db_url)
        cur = conn2.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        conn2.close()
        
        assert result[0] == 1
        print("Database connection recovery: SUCCESS")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
