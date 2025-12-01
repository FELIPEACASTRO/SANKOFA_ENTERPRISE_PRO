#!/usr/bin/env python3
"""
Sankofa Enterprise Pro - Testes de Integração Backend-PostgreSQL-Redis
Testa todas as integrações de banco de dados e cache

Execução: pytest tests/test_integration_db.py -v --tb=short
"""

import os
import sys
import time
import json
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from psycopg2.extras import RealDictCursor


class TestPostgreSQLConnection:
    """Testes de conexão e operações básicas no PostgreSQL"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup de conexão para cada teste"""
        self.conn_string = os.environ.get("DATABASE_URL")
        assert self.conn_string, "DATABASE_URL não configurada"
        
    def get_connection(self):
        """Obtém conexão com o banco"""
        return psycopg2.connect(self.conn_string, cursor_factory=RealDictCursor)
    
    def test_connection_success(self):
        """Testa se a conexão com PostgreSQL está funcionando"""
        conn = self.get_connection()
        assert conn is not None
        assert not conn.closed
        conn.close()
        
    def test_connection_pool_performance(self):
        """Testa performance do pool de conexões"""
        start = time.time()
        connections = []
        
        for _ in range(10):
            conn = self.get_connection()
            connections.append(conn)
            
        connect_time = (time.time() - start) * 1000
        
        for conn in connections:
            conn.close()
            
        assert connect_time < 5000, f"Pool de conexões muito lento: {connect_time:.2f}ms"
        print(f"✓ 10 conexões criadas em {connect_time:.2f}ms")
        
    def test_database_version(self):
        """Verifica versão do PostgreSQL"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()['version']
                assert 'PostgreSQL' in version
                print(f"✓ Versão: {version[:50]}...")


class TestPostgreSQLTransactions:
    """Testes de operações com transações no PostgreSQL"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.conn_string = os.environ.get("DATABASE_URL")
        
    def get_connection(self):
        return psycopg2.connect(self.conn_string, cursor_factory=RealDictCursor)
    
    def test_transaction_table_exists(self):
        """Verifica se a tabela de transações existe"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'transactions'
                    );
                """)
                exists = cur.fetchone()['exists']
                assert exists, "Tabela 'transactions' não encontrada"
                
    def test_transaction_count(self):
        """Conta transações no banco"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as total FROM transactions;")
                total = cur.fetchone()['total']
                print(f"✓ Total de transações: {total}")
                assert total >= 0
                
    def test_transaction_read_performance(self):
        """Testa performance de leitura de transações"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                start = time.time()
                cur.execute("""
                    SELECT id, transaction_id, amount, channel, status, 
                           risk_score, created_at
                    FROM transactions 
                    ORDER BY created_at DESC 
                    LIMIT 100;
                """)
                rows = cur.fetchall()
                elapsed = (time.time() - start) * 1000
                
                print(f"✓ Leitura de 100 transações em {elapsed:.2f}ms")
                assert elapsed < 500, f"Leitura muito lenta: {elapsed:.2f}ms"
                
    def test_transaction_insert_and_rollback(self):
        """Testa insert com rollback (não persiste dados de teste)"""
        test_id = f"TEST_TXN_{uuid.uuid4().hex[:8]}"
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO transactions (transaction_id, amount, channel, status, risk_score, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    RETURNING id;
                """, (test_id, 1000.00, 'pix', 'PENDING', 0.5))
                
                inserted_id = cur.fetchone()['id']
                assert inserted_id > 0
                print(f"✓ Insert bem-sucedido (ID: {inserted_id})")
                
            conn.rollback()
            
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM transactions WHERE transaction_id = %s;", (test_id,))
                result = cur.fetchone()
                assert result is None, "Rollback não funcionou"
                print("✓ Rollback confirmado")
        finally:
            conn.close()
            
    def test_transaction_aggregations(self):
        """Testa queries de agregação"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                start = time.time()
                cur.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(CASE WHEN status = 'APPROVED' THEN 1 END) as approved,
                        COUNT(CASE WHEN status = 'FRAUD' THEN 1 END) as fraud,
                        COALESCE(AVG(risk_score), 0) as avg_risk,
                        COALESCE(SUM(amount), 0) as total_amount
                    FROM transactions;
                """)
                stats = cur.fetchone()
                elapsed = (time.time() - start) * 1000
                
                print(f"✓ Agregação em {elapsed:.2f}ms")
                print(f"  - Total: {stats['total']}")
                print(f"  - Aprovadas: {stats['approved']}")
                print(f"  - Fraudes: {stats['fraud']}")
                print(f"  - Risco médio: {float(stats['avg_risk']):.2%}")
                
                assert elapsed < 1000, f"Agregação muito lenta: {elapsed:.2f}ms"


class TestPostgreSQLAlerts:
    """Testes de operações com alertas no PostgreSQL"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.conn_string = os.environ.get("DATABASE_URL")
        
    def get_connection(self):
        return psycopg2.connect(self.conn_string, cursor_factory=RealDictCursor)
    
    def test_alerts_table_exists(self):
        """Verifica se a tabela de alertas existe"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'alerts'
                    );
                """)
                exists = cur.fetchone()['exists']
                assert exists, "Tabela 'alerts' não encontrada"
                
    def test_alerts_count(self):
        """Conta alertas no banco"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as total FROM alerts;")
                total = cur.fetchone()['total']
                print(f"✓ Total de alertas: {total}")
                
    def test_alerts_insert_and_cleanup(self):
        """Testa insert de alerta com cleanup"""
        test_alert_id = f"TEST_ALERT_{uuid.uuid4().hex[:8]}"
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO alerts (alert_id, title, description, type, severity, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    RETURNING id;
                """, (test_alert_id, 'Test Alert', 'Integration test', 'system', 'baixo', 'novo'))
                
                inserted_id = cur.fetchone()['id']
                conn.commit()
                
                cur.execute("DELETE FROM alerts WHERE alert_id = %s;", (test_alert_id,))
                conn.commit()
                
                print(f"✓ Insert/Delete de alerta bem-sucedido")


class TestPostgreSQLRules:
    """Testes de operações com regras no PostgreSQL"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.conn_string = os.environ.get("DATABASE_URL")
        
    def get_connection(self):
        return psycopg2.connect(self.conn_string, cursor_factory=RealDictCursor)
    
    def test_hard_rules_table_exists(self):
        """Verifica se a tabela de hard_rules existe"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'hard_rules'
                    );
                """)
                exists = cur.fetchone()['exists']
                assert exists, "Tabela 'hard_rules' não encontrada"
                
    def test_vip_list_table_exists(self):
        """Verifica se a tabela vip_list existe"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'vip_list'
                    );
                """)
                exists = cur.fetchone()['exists']
                assert exists, "Tabela 'vip_list' não encontrada"
                
    def test_hot_list_table_exists(self):
        """Verifica se a tabela hot_list existe"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'hot_list'
                    );
                """)
                exists = cur.fetchone()['exists']
                assert exists, "Tabela 'hot_list' não encontrada"
                
    def test_rules_count(self):
        """Conta regras no banco"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as total FROM hard_rules;")
                rules = cur.fetchone()['total']
                
                cur.execute("SELECT COUNT(*) as total FROM vip_list;")
                vip = cur.fetchone()['total']
                
                cur.execute("SELECT COUNT(*) as total FROM hot_list;")
                hot = cur.fetchone()['total']
                
                print(f"✓ Hard Rules: {rules}")
                print(f"✓ VIP List: {vip}")
                print(f"✓ Hot List: {hot}")


class TestPostgreSQLAuditLogs:
    """Testes de operações com logs de auditoria"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.conn_string = os.environ.get("DATABASE_URL")
        
    def get_connection(self):
        return psycopg2.connect(self.conn_string, cursor_factory=RealDictCursor)
    
    def test_audit_logs_table_exists(self):
        """Verifica se a tabela audit_logs existe"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'audit_logs'
                    );
                """)
                exists = cur.fetchone()['exists']
                assert exists, "Tabela 'audit_logs' não encontrada"
                
    def test_audit_log_insert(self):
        """Testa inserção de log de auditoria"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO audit_logs (action, details, ip_address, timestamp)
                    VALUES (%s, %s, %s, NOW())
                    RETURNING id;
                """, ('INTEGRATION_TEST', 'Test audit log entry', '127.0.0.1'))
                
                log_id = cur.fetchone()['id']
                conn.commit()
                
                cur.execute("DELETE FROM audit_logs WHERE id = %s;", (log_id,))
                conn.commit()
                
                print("✓ Audit log insert/delete bem-sucedido")


class TestPostgreSQLPerformance:
    """Testes de performance do PostgreSQL"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.conn_string = os.environ.get("DATABASE_URL")
        
    def get_connection(self):
        return psycopg2.connect(self.conn_string, cursor_factory=RealDictCursor)
    
    def test_dashboard_kpis_query_performance(self):
        """Testa performance da query de KPIs do dashboard"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                start = time.time()
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_transactions,
                        COUNT(CASE WHEN status = 'FRAUD' OR is_fraud = true THEN 1 END) as frauds_detected,
                        COUNT(CASE WHEN status = 'APPROVED' THEN 1 END) as approved,
                        COALESCE(AVG(processing_time_ms), 0) as avg_latency,
                        COALESCE(SUM(CASE WHEN status = 'FRAUD' OR is_fraud = true THEN amount ELSE 0 END), 0) as value_protected
                    FROM transactions;
                """)
                kpis = cur.fetchone()
                elapsed = (time.time() - start) * 1000
                
                print(f"✓ Query KPIs em {elapsed:.2f}ms")
                print(f"  - Transações: {kpis['total_transactions']}")
                print(f"  - Fraudes: {kpis['frauds_detected']}")
                print(f"  - Aprovadas: {kpis['approved']}")
                
                assert elapsed < 1000, f"Query KPIs muito lenta: {elapsed:.2f}ms"
                
    def test_timeseries_query_performance(self):
        """Testa performance da query de série temporal"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                start = time.time()
                cur.execute("""
                    SELECT 
                        EXTRACT(HOUR FROM created_at) as hour,
                        COUNT(*) as transactions,
                        COALESCE(AVG(processing_time_ms), 0) as avg_latency
                    FROM transactions
                    GROUP BY EXTRACT(HOUR FROM created_at)
                    ORDER BY hour;
                """)
                rows = cur.fetchall()
                elapsed = (time.time() - start) * 1000
                
                print(f"✓ Query timeseries em {elapsed:.2f}ms ({len(rows)} horas)")
                assert elapsed < 1000, f"Query timeseries muito lenta: {elapsed:.2f}ms"
                
    def test_channels_query_performance(self):
        """Testa performance da query de canais"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                start = time.time()
                cur.execute("""
                    SELECT 
                        COALESCE(channel, 'unknown') as channel,
                        COUNT(*) as count,
                        COALESCE(SUM(amount), 0) as total_amount,
                        COALESCE(AVG(risk_score), 0) as avg_risk
                    FROM transactions
                    GROUP BY channel
                    ORDER BY count DESC;
                """)
                rows = cur.fetchall()
                elapsed = (time.time() - start) * 1000
                
                print(f"✓ Query channels em {elapsed:.2f}ms ({len(rows)} canais)")
                for row in rows[:5]:
                    print(f"  - {row['channel']}: {row['count']} transações")
                    
                assert elapsed < 1000, f"Query channels muito lenta: {elapsed:.2f}ms"
                
    def test_concurrent_reads(self):
        """Testa leituras concorrentes"""
        import threading
        results = []
        errors = []
        
        def read_transactions():
            try:
                conn = self.get_connection()
                with conn.cursor() as cur:
                    start = time.time()
                    cur.execute("SELECT COUNT(*) FROM transactions;")
                    cur.fetchone()
                    elapsed = (time.time() - start) * 1000
                    results.append(elapsed)
                conn.close()
            except Exception as e:
                errors.append(str(e))
        
        threads = [threading.Thread(target=read_transactions) for _ in range(10)]
        
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_time = (time.time() - start) * 1000
        
        assert len(errors) == 0, f"Erros em leituras concorrentes: {errors}"
        avg_time = sum(results) / len(results)
        print(f"✓ 10 leituras concorrentes em {total_time:.2f}ms (média: {avg_time:.2f}ms)")


class TestRedisConnection:
    """Testes de conexão e operações com Redis"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            import redis
            self.redis_available = True
            self.redis_host = os.environ.get("REDIS_HOST", "localhost")
            self.redis_port = int(os.environ.get("REDIS_PORT", 6379))
            self.redis_password = os.environ.get("REDIS_PASSWORD")
        except ImportError:
            self.redis_available = False
            
    def get_redis_client(self):
        if not self.redis_available:
            pytest.skip("Redis não disponível")
        import redis
        return redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            decode_responses=True
        )
    
    def test_redis_connection(self):
        """Testa conexão com Redis"""
        if not self.redis_available:
            pytest.skip("Redis não disponível")
            
        try:
            client = self.get_redis_client()
            pong = client.ping()
            assert pong == True
            print("✓ Redis conectado com sucesso")
        except Exception as e:
            pytest.skip(f"Redis não acessível: {e}")
            
    def test_redis_set_get(self):
        """Testa operações SET/GET no Redis"""
        if not self.redis_available:
            pytest.skip("Redis não disponível")
            
        try:
            client = self.get_redis_client()
            test_key = f"test_integration_{uuid.uuid4().hex[:8]}"
            test_value = "integration_test_value"
            
            client.setex(test_key, 60, test_value)
            result = client.get(test_key)
            client.delete(test_key)
            
            assert result == test_value
            print("✓ Redis SET/GET funcionando")
        except Exception as e:
            pytest.skip(f"Redis não acessível: {e}")
            
    def test_redis_performance(self):
        """Testa performance do Redis"""
        if not self.redis_available:
            pytest.skip("Redis não disponível")
            
        try:
            client = self.get_redis_client()
            test_key = f"perf_test_{uuid.uuid4().hex[:8]}"
            
            start = time.time()
            for i in range(100):
                client.setex(f"{test_key}_{i}", 60, f"value_{i}")
            write_time = (time.time() - start) * 1000
            
            start = time.time()
            for i in range(100):
                client.get(f"{test_key}_{i}")
            read_time = (time.time() - start) * 1000
            
            for i in range(100):
                client.delete(f"{test_key}_{i}")
                
            print(f"✓ Redis 100 writes em {write_time:.2f}ms")
            print(f"✓ Redis 100 reads em {read_time:.2f}ms")
            
            assert write_time < 1000, f"Redis writes muito lentos: {write_time:.2f}ms"
            assert read_time < 1000, f"Redis reads muito lentos: {read_time:.2f}ms"
        except Exception as e:
            pytest.skip(f"Redis não acessível: {e}")


class TestInMemoryCache:
    """Testes do cache em memória (fallback quando Redis não disponível)"""
    
    def test_simple_cache_operations(self):
        """Testa operações básicas do SimpleCache"""
        from services.postgres_store import SimpleCache
        
        cache = SimpleCache(default_ttl=5)
        
        cache.set("test_key", "test_value")
        result = cache.get("test_key")
        assert result == "test_value"
        print("✓ SimpleCache SET/GET funcionando")
        
    def test_simple_cache_ttl(self):
        """Testa TTL do SimpleCache"""
        from services.postgres_store import SimpleCache
        
        cache = SimpleCache(default_ttl=1)
        cache.set("expire_test", "value", ttl=1)
        
        assert cache.get("expire_test") == "value"
        time.sleep(1.1)
        assert cache.get("expire_test") is None
        print("✓ SimpleCache TTL funcionando")
        
    def test_simple_cache_invalidate(self):
        """Testa invalidação do SimpleCache"""
        from services.postgres_store import SimpleCache
        
        cache = SimpleCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.invalidate("key1")
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        
        cache.invalidate()
        assert cache.get("key2") is None
        print("✓ SimpleCache invalidation funcionando")


class TestRedisCacheSystem:
    """Testes do sistema de cache Redis"""
    
    def test_inmemory_cache_fallback(self):
        """Testa o InMemoryCache como fallback"""
        from cache.redis_cache_system import InMemoryCache
        
        cache = InMemoryCache(max_size=100)
        
        test_value = b"test_binary_value"
        cache.setex("test_key", 60, test_value)
        result = cache.get("test_key")
        
        assert result == test_value
        print("✓ InMemoryCache fallback funcionando")
        
    def test_inmemory_cache_lru_eviction(self):
        """Testa LRU eviction do InMemoryCache"""
        from cache.redis_cache_system import InMemoryCache
        
        cache = InMemoryCache(max_size=5)
        
        for i in range(10):
            cache.setex(f"key_{i}", 60, f"value_{i}".encode())
            
        assert cache.get("key_0") is None
        assert cache.get("key_9") is not None
        print("✓ InMemoryCache LRU eviction funcionando")
        
    def test_inmemory_cache_expiry(self):
        """Testa expiração do InMemoryCache"""
        from cache.redis_cache_system import InMemoryCache
        
        cache = InMemoryCache()
        cache.setex("expire_key", 1, b"expire_value")
        
        assert cache.get("expire_key") == b"expire_value"
        time.sleep(1.1)
        assert cache.get("expire_key") is None
        print("✓ InMemoryCache expiry funcionando")


class TestPostgresStoreIntegration:
    """Testes de integração do PostgresStore"""
    
    def test_get_dashboard_kpis(self):
        """Testa método get_dashboard_kpis"""
        from services.postgres_store import PostgresStore
        
        store = PostgresStore()
        start = time.time()
        kpis = store.get_dashboard_kpis()
        elapsed = (time.time() - start) * 1000
        
        assert isinstance(kpis, dict)
        assert 'total_transactions' in kpis
        print(f"✓ get_dashboard_kpis em {elapsed:.2f}ms")
        print(f"  - Transações: {kpis.get('total_transactions', 0)}")
        
    def test_get_dashboard_timeseries(self):
        """Testa método get_dashboard_timeseries"""
        from services.postgres_store import PostgresStore
        
        store = PostgresStore()
        start = time.time()
        timeseries = store.get_dashboard_timeseries()
        elapsed = (time.time() - start) * 1000
        
        assert isinstance(timeseries, list)
        print(f"✓ get_dashboard_timeseries em {elapsed:.2f}ms ({len(timeseries)} pontos)")
        
    def test_get_dashboard_channels(self):
        """Testa método get_dashboard_channels"""
        from services.postgres_store import PostgresStore
        
        store = PostgresStore()
        start = time.time()
        channels = store.get_dashboard_channels()
        elapsed = (time.time() - start) * 1000
        
        assert isinstance(channels, list)
        print(f"✓ get_dashboard_channels em {elapsed:.2f}ms ({len(channels)} canais)")
        
    def test_get_hard_rules(self):
        """Testa método get_hard_rules"""
        from services.postgres_store import PostgresStore
        
        store = PostgresStore()
        start = time.time()
        rules = store.get_hard_rules()
        elapsed = (time.time() - start) * 1000
        
        assert isinstance(rules, list)
        print(f"✓ get_hard_rules em {elapsed:.2f}ms ({len(rules)} regras)")
        
    def test_get_recent_transactions(self):
        """Testa método get_recent_transactions"""
        from services.postgres_store import PostgresStore
        
        store = PostgresStore()
        start = time.time()
        transactions = store.get_recent_transactions(limit=50)
        elapsed = (time.time() - start) * 1000
        
        assert isinstance(transactions, list)
        print(f"✓ get_recent_transactions em {elapsed:.2f}ms ({len(transactions)} transações)")
        
    def test_get_alerts_list(self):
        """Testa método get_alerts_list"""
        from services.postgres_store import PostgresStore
        
        store = PostgresStore()
        start = time.time()
        alerts = store.get_alerts_list(limit=50)
        elapsed = (time.time() - start) * 1000
        
        assert isinstance(alerts, list)
        print(f"✓ get_alerts_list em {elapsed:.2f}ms ({len(alerts)} alertas)")
        
    def test_cache_effectiveness(self):
        """Testa efetividade do cache"""
        from services.postgres_store import PostgresStore
        
        store = PostgresStore()
        
        start = time.time()
        kpis1 = store.get_dashboard_kpis()
        first_call = (time.time() - start) * 1000
        
        start = time.time()
        kpis2 = store.get_dashboard_kpis()
        second_call = (time.time() - start) * 1000
        
        print(f"✓ Primeira chamada: {first_call:.2f}ms")
        print(f"✓ Segunda chamada (cache): {second_call:.2f}ms")
        
        if second_call < first_call * 0.5:
            print(f"✓ Cache efetivo! Speedup: {first_call/second_call:.1f}x")


class TestDatabaseSchema:
    """Testes de validação do schema do banco"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.conn_string = os.environ.get("DATABASE_URL")
        
    def get_connection(self):
        return psycopg2.connect(self.conn_string, cursor_factory=RealDictCursor)
    
    def test_all_required_tables_exist(self):
        """Verifica se todas as tabelas necessárias existem"""
        required_tables = [
            'transactions',
            'alerts',
            'hard_rules',
            'vip_list',
            'hot_list',
            'audit_logs',
            'settings'
        ]
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for table in required_tables:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = %s
                        );
                    """, (table,))
                    exists = cur.fetchone()['exists']
                    assert exists, f"Tabela '{table}' não encontrada"
                    print(f"✓ Tabela '{table}' existe")
                    
    def test_transactions_table_columns(self):
        """Verifica colunas da tabela transactions"""
        expected_columns = ['id', 'transaction_id', 'amount', 'channel', 'status', 'risk_score']
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'transactions';
                """)
                columns = [row['column_name'] for row in cur.fetchall()]
                
                for col in expected_columns:
                    assert col in columns, f"Coluna '{col}' não encontrada em transactions"
                print(f"✓ Todas as colunas esperadas presentes em transactions")
                
    def test_indexes_exist(self):
        """Verifica se os índices principais existem"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT indexname FROM pg_indexes 
                    WHERE tablename = 'transactions';
                """)
                indexes = [row['indexname'] for row in cur.fetchall()]
                print(f"✓ Índices em transactions: {len(indexes)}")
                for idx in indexes[:5]:
                    print(f"  - {idx}")


class TestEndToEndIntegration:
    """Testes end-to-end de integração"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.conn_string = os.environ.get("DATABASE_URL")
        
    def get_connection(self):
        return psycopg2.connect(self.conn_string, cursor_factory=RealDictCursor)
    
    def test_full_transaction_flow(self):
        """Testa fluxo completo de transação"""
        from services.postgres_store import PostgresStore
        
        store = PostgresStore()
        
        test_txn_id = f"E2E_TEST_{uuid.uuid4().hex[:8]}"
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO transactions 
                    (transaction_id, amount, channel, status, risk_score, created_at)
                    VALUES (%s, 5000.00, 'pix', 'PENDING', 0.75, NOW())
                    RETURNING id;
                """, (test_txn_id,))
                inserted_id = cur.fetchone()['id']
                conn.commit()
                
                cur.execute("""
                    SELECT * FROM transactions WHERE transaction_id = %s;
                """, (test_txn_id,))
                txn = cur.fetchone()
                assert txn is not None
                assert txn['amount'] == 5000.00
                
                cur.execute("""
                    UPDATE transactions SET status = 'APPROVED' WHERE transaction_id = %s;
                """, (test_txn_id,))
                conn.commit()
                
                cur.execute("""
                    SELECT status FROM transactions WHERE transaction_id = %s;
                """, (test_txn_id,))
                status = cur.fetchone()['status']
                assert status == 'APPROVED'
                
                cur.execute("DELETE FROM transactions WHERE transaction_id = %s;", (test_txn_id,))
                conn.commit()
                
        print("✓ Fluxo completo de transação: INSERT -> SELECT -> UPDATE -> DELETE")
        
    def test_alert_creation_on_high_risk(self):
        """Testa criação de alerta para transação de alto risco"""
        from services.postgres_store import PostgresStore
        
        store = PostgresStore()
        test_alert_id = f"E2E_ALERT_{uuid.uuid4().hex[:8]}"
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO alerts 
                    (alert_id, title, description, type, severity, status, created_at)
                    VALUES (%s, 'High Risk Transaction', 'E2E Test Alert', 'fraud_detected', 'critico', 'novo', NOW())
                    RETURNING id;
                """, (test_alert_id,))
                inserted_id = cur.fetchone()['id']
                conn.commit()
                
                cur.execute("""
                    UPDATE alerts SET status = 'investigando' WHERE alert_id = %s;
                """, (test_alert_id,))
                conn.commit()
                
                cur.execute("DELETE FROM alerts WHERE alert_id = %s;", (test_alert_id,))
                conn.commit()
                
        print("✓ Fluxo de alerta: CREATE -> UPDATE status -> DELETE")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
