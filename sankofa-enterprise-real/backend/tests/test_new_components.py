"""
Sankofa Enterprise Pro - Testes dos Novos Componentes
Testes para: Prediction Cache, Experiment Tracker, Shadow Mode, Fairness Analyzer
"""

import pytest
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd


class TestPredictionCache:
    """Testes do sistema de cache de predições"""
    
    def test_cache_initialization(self):
        """Testa inicialização do cache"""
        from cache.prediction_cache import PredictionCache
        
        cache = PredictionCache(max_size=100, default_ttl_seconds=60)
        
        assert cache.max_size == 100
        assert cache.default_ttl == 60
        assert cache.VERSION == "1.0.0"
    
    def test_cache_set_and_get(self):
        """Testa armazenamento e recuperação do cache"""
        from cache.prediction_cache import PredictionCache
        
        cache = PredictionCache(max_size=100, default_ttl_seconds=60)
        
        transaction = {
            'amount': 500,
            'hour': 14,
            'channel': 'PIX',
            'customer_id': 'CUST001'
        }
        
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
        assert cached.risk_level == 'LOW'
    
    def test_cache_hit_rate(self):
        """Testa taxa de hit do cache"""
        from cache.prediction_cache import PredictionCache
        
        cache = PredictionCache(max_size=100, default_ttl_seconds=60)
        
        transactions = [
            {'amount': 100 * i, 'hour': 10, 'channel': 'PIX', 'customer_id': f'C{i}'}
            for i in range(10)
        ]
        
        for txn in transactions:
            cache.set(
                transaction=txn,
                is_fraud=False,
                fraud_probability=0.1,
                risk_score=0.1,
                risk_level='LOW',
                confidence=0.9,
                model_version='1.0.0',
                detection_reason=[]
            )
        
        for txn in transactions:
            cache.get(txn)
        
        stats = cache.get_stats()
        
        assert stats['hit_rate_percent'] == 100.0
        assert stats['hits'] == 10
        assert stats['misses'] == 0
    
    def test_cache_eviction(self):
        """Testa evicção LRU do cache"""
        from cache.prediction_cache import PredictionCache
        
        cache = PredictionCache(max_size=5, default_ttl_seconds=60)
        
        for i in range(10):
            txn = {'amount': 100 * i, 'customer_id': f'C{i}'}
            cache.set(
                transaction=txn,
                is_fraud=False,
                fraud_probability=0.1,
                risk_score=0.1,
                risk_level='LOW',
                confidence=0.9,
                model_version='1.0.0',
                detection_reason=[]
            )
        
        stats = cache.get_stats()
        
        assert stats['size'] <= 5
        assert stats['evictions'] >= 5
    
    def test_cache_latency_improvement(self):
        """Testa melhoria de latência com cache"""
        from cache.prediction_cache import PredictionCache
        
        cache = PredictionCache(max_size=100, default_ttl_seconds=60)
        
        transaction = {'amount': 1000, 'hour': 14, 'channel': 'PIX'}
        
        cache.set(
            transaction=transaction,
            is_fraud=False,
            fraud_probability=0.2,
            risk_score=0.2,
            risk_level='LOW',
            confidence=0.8,
            model_version='1.0.0',
            detection_reason=[]
        )
        
        start = time.time()
        for _ in range(100):
            cache.get(transaction)
        cache_time = (time.time() - start) * 1000
        
        avg_time_per_hit = cache_time / 100
        
        assert avg_time_per_hit < 1.0
    
    def test_cache_cleanup_expired(self):
        """Testa limpeza de entradas expiradas"""
        from cache.prediction_cache import PredictionCache
        
        cache = PredictionCache(
            max_size=100, 
            default_ttl_seconds=1,
            high_risk_ttl_seconds=1,
            low_risk_ttl_seconds=1
        )
        
        txn = {'amount': 100, 'customer_id': 'C1'}
        cache.set(
            transaction=txn,
            is_fraud=False,
            fraud_probability=0.1,
            risk_score=0.1,
            risk_level='LOW',
            confidence=0.9,
            model_version='1.0.0',
            detection_reason=[]
        )
        
        cached = cache.get(txn)
        assert cached is not None
        
        time.sleep(1.5)
        
        cached_after = cache.get(txn)
        
        assert cached_after is None


class TestExperimentTracker:
    """Testes do sistema de experiment tracking"""
    
    def test_tracker_initialization(self):
        """Testa inicialização do tracker"""
        from mlops.experiment_tracker import ExperimentTracker
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tracking_dir=tmpdir)
            
            assert tracker.VERSION == "1.0.0"
    
    def test_create_experiment(self):
        """Testa criação de experimento"""
        from mlops.experiment_tracker import ExperimentTracker
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tracking_dir=tmpdir)
            
            exp_id = tracker.create_experiment("test_fraud_detection", "Test experiment")
            
            assert exp_id is not None
            assert len(exp_id) == 8
            assert "test_fraud_detection" in tracker.experiments
    
    def test_run_lifecycle(self):
        """Testa ciclo de vida de uma run"""
        from mlops.experiment_tracker import ExperimentTracker
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tracking_dir=tmpdir)
            
            run_id = tracker.start_run(
                experiment_name="fraud_model_v1",
                run_name="baseline",
                parameters={'n_estimators': 100},
                tags={'env': 'test'}
            )
            
            tracker.log_param("learning_rate", 0.1)
            tracker.log_metric("accuracy", 0.95)
            tracker.log_metrics({
                'precision': 0.92,
                'recall': 0.88,
                'f1_score': 0.90
            })
            
            result = tracker.end_run()
            
            assert result['run_id'] == run_id
            assert result['status'] == "COMPLETED"
            assert len(result['metrics']) == 4
            assert result['parameters']['n_estimators'] == 100
    
    def test_get_best_run(self):
        """Testa obtenção da melhor run"""
        from mlops.experiment_tracker import ExperimentTracker
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tracking_dir=tmpdir)
            
            for i, f1 in enumerate([0.85, 0.90, 0.88]):
                tracker.start_run(
                    experiment_name="comparison",
                    run_name=f"run_{i}"
                )
                tracker.log_metric("f1_score", f1)
                tracker.end_run()
            
            best = tracker.get_best_run("comparison", "f1_score", maximize=True)
            
            assert best is not None
            best_f1 = next(m['value'] for m in best['metrics'] if m['name'] == 'f1_score')
            assert best_f1 == 0.90
    
    def test_compare_runs(self):
        """Testa comparação de runs"""
        from mlops.experiment_tracker import ExperimentTracker
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tracking_dir=tmpdir)
            
            run_ids = []
            for i in range(3):
                run_id = tracker.start_run(
                    experiment_name="comparison_test",
                    run_name=f"run_{i}"
                )
                tracker.log_metric("accuracy", 0.90 + i * 0.02)
                tracker.end_run()
                run_ids.append(run_id)
            
            comparison = tracker.compare_runs(run_ids)
            
            assert len(comparison['runs']) == 3
            assert 'accuracy' in comparison['metrics_comparison']
    
    def test_get_summary(self):
        """Testa resumo de experimentos"""
        from mlops.experiment_tracker import ExperimentTracker
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tracking_dir=tmpdir)
            
            tracker.create_experiment("exp1", "Test 1")
            tracker.create_experiment("exp2", "Test 2")
            
            summary = tracker.get_summary()
            
            assert summary['total_experiments'] == 2
            assert len(summary['experiments']) == 2


class TestShadowMode:
    """Testes do sistema de Shadow Mode"""
    
    def test_shadow_mode_initialization(self):
        """Testa inicialização do shadow mode"""
        from mlops.shadow_mode import ShadowModeManager
        
        manager = ShadowModeManager()
        
        assert manager.VERSION == "1.0.0"
        assert manager.is_active == False
    
    def test_shadow_mode_start_stop(self):
        """Testa início e parada do shadow mode"""
        from mlops.shadow_mode import ShadowModeManager
        
        def model_a(txn):
            return {'is_fraud': False, 'fraud_probability': 0.1, 'risk_level': 'LOW'}
        
        def model_b(txn):
            return {'is_fraud': False, 'fraud_probability': 0.1, 'risk_level': 'LOW'}
        
        manager = ShadowModeManager()
        manager.start(primary_model=model_a, shadow_model=model_b)
        
        assert manager.is_active == True
        
        report = manager.stop()
        
        assert manager.is_active == False
        assert 'statistics' in report
    
    def test_shadow_mode_comparison(self):
        """Testa comparação entre modelos"""
        from mlops.shadow_mode import ShadowModeManager
        
        def primary_model(txn):
            return {
                'is_fraud': txn.get('amount', 0) > 10000,
                'fraud_probability': min(txn.get('amount', 0) / 20000, 1.0),
                'risk_level': 'HIGH' if txn.get('amount', 0) > 10000 else 'LOW'
            }
        
        def shadow_model(txn):
            return {
                'is_fraud': txn.get('amount', 0) > 8000,
                'fraud_probability': min(txn.get('amount', 0) / 15000, 1.0),
                'risk_level': 'HIGH' if txn.get('amount', 0) > 8000 else 'LOW'
            }
        
        manager = ShadowModeManager()
        manager.start(
            primary_model=primary_model,
            shadow_model=shadow_model,
            shadow_traffic_percent=100
        )
        
        test_transactions = [
            {'transaction_id': 'T1', 'amount': 500},
            {'transaction_id': 'T2', 'amount': 9000},
            {'transaction_id': 'T3', 'amount': 15000},
        ]
        
        for txn in test_transactions:
            result, comparison = manager.predict_with_shadow(txn, force_shadow=True)
            assert result is not None
            assert comparison is not None
        
        stats = manager.get_stats()
        
        assert stats['total_comparisons'] == 3
    
    def test_shadow_mode_divergent_detection(self):
        """Testa detecção de divergências"""
        from mlops.shadow_mode import ShadowModeManager
        
        def model_agree(txn):
            return {'is_fraud': False, 'fraud_probability': 0.1, 'risk_level': 'LOW'}
        
        call_count = [0]
        def model_disagree(txn):
            call_count[0] += 1
            return {
                'is_fraud': call_count[0] % 2 == 0,
                'fraud_probability': 0.9 if call_count[0] % 2 == 0 else 0.1,
                'risk_level': 'HIGH' if call_count[0] % 2 == 0 else 'LOW'
            }
        
        manager = ShadowModeManager()
        manager.start(primary_model=model_agree, shadow_model=model_disagree)
        
        for i in range(10):
            manager.predict_with_shadow({'transaction_id': f'T{i}'}, force_shadow=True)
        
        divergent = manager.get_divergent_transactions()
        
        assert len(divergent) > 0


class TestFairnessAnalyzer:
    """Testes do analisador de fairness"""
    
    def test_analyzer_initialization(self):
        """Testa inicialização do analisador"""
        from mlops.fairness_analyzer import FairnessAnalyzer
        
        analyzer = FairnessAnalyzer()
        
        assert analyzer.VERSION == "1.0.0"
        assert analyzer.fairness_threshold == 0.8
    
    def test_subgroup_analysis(self):
        """Testa análise por subgrupo"""
        from mlops.fairness_analyzer import FairnessAnalyzer
        
        predictions = [
            {'is_fraud': True, 'fraud_probability': 0.85, 'regiao': 'Sudeste'},
            {'is_fraud': False, 'fraud_probability': 0.15, 'regiao': 'Sudeste'},
            {'is_fraud': True, 'fraud_probability': 0.92, 'regiao': 'Nordeste'},
            {'is_fraud': False, 'fraud_probability': 0.25, 'regiao': 'Nordeste'},
        ]
        
        ground_truth = [True, False, True, False]
        
        analyzer = FairnessAnalyzer()
        report = analyzer.analyze(predictions, ground_truth)
        
        assert report is not None
        assert len(report.subgroup_metrics) > 0
        assert report.risk_level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    
    def test_fairness_metrics(self):
        """Testa métricas de fairness"""
        from mlops.fairness_analyzer import FairnessAnalyzer
        
        predictions = []
        ground_truth = []
        
        for i in range(100):
            region = 'Sudeste' if i % 2 == 0 else 'Nordeste'
            is_fraud = i % 3 == 0
            predictions.append({
                'is_fraud': is_fraud,
                'fraud_probability': 0.8 if is_fraud else 0.2,
                'regiao': region
            })
            ground_truth.append(is_fraud)
        
        analyzer = FairnessAnalyzer()
        report = analyzer.analyze(predictions, ground_truth)
        
        assert report.fairness_metrics is not None
        assert 0 <= report.fairness_metrics.demographic_parity_ratio <= 2
        assert 0 <= report.fairness_metrics.overall_fairness_score <= 2
    
    def test_compliance_status(self):
        """Testa status de conformidade"""
        from mlops.fairness_analyzer import FairnessAnalyzer
        
        predictions = [
            {'is_fraud': True, 'fraud_probability': 0.9, 'regiao': 'Sudeste'},
            {'is_fraud': True, 'fraud_probability': 0.9, 'regiao': 'Nordeste'},
            {'is_fraud': False, 'fraud_probability': 0.1, 'regiao': 'Sudeste'},
            {'is_fraud': False, 'fraud_probability': 0.1, 'regiao': 'Nordeste'},
        ]
        ground_truth = [True, True, False, False]
        
        analyzer = FairnessAnalyzer()
        report = analyzer.analyze(predictions, ground_truth)
        
        assert report.compliance_status in ['COMPLIANT', 'PARTIALLY_COMPLIANT', 'NON_COMPLIANT']
    
    def test_recommendations_generation(self):
        """Testa geração de recomendações"""
        from mlops.fairness_analyzer import FairnessAnalyzer
        
        predictions = [
            {'is_fraud': True, 'fraud_probability': 0.9, 'regiao': 'Sudeste'},
            {'is_fraud': False, 'fraud_probability': 0.1, 'regiao': 'Nordeste'},
        ]
        
        analyzer = FairnessAnalyzer()
        report = analyzer.analyze(predictions)
        
        assert len(report.recommendations) > 0


class TestAuditTrail:
    """Testes da trilha de auditoria"""
    
    def test_audit_trail_table_exists(self):
        """Verifica se tabela audit_trail existe"""
        import psycopg2
        
        conn = psycopg2.connect(os.environ.get('DATABASE_URL'))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'audit_trail'
            );
        """)
        
        exists = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        assert exists == True
    
    def test_audit_trail_structure(self):
        """Verifica estrutura da tabela audit_trail"""
        import psycopg2
        
        conn = psycopg2.connect(os.environ.get('DATABASE_URL'))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'audit_trail'
            ORDER BY ordinal_position;
        """)
        
        columns = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        required_columns = [
            'event_id', 'event_type', 'entity_type', 'action',
            'metadata', 'risk_level', 'lgpd_consent', 'created_at'
        ]
        
        for col in required_columns:
            assert col in columns, f"Column {col} missing from audit_trail"
    
    def test_audit_trail_insert(self):
        """Testa inserção na audit_trail"""
        import psycopg2
        import uuid
        
        conn = psycopg2.connect(os.environ.get('DATABASE_URL'))
        cursor = conn.cursor()
        
        event_id = str(uuid.uuid4())
        
        cursor.execute("""
            INSERT INTO audit_trail (
                event_id, event_type, entity_type, entity_id,
                action, metadata, risk_level
            ) VALUES (
                %s, 'TEST_EVENT', 'TRANSACTION', 'TXN_TEST_001',
                'EVALUATE', '{"test": true}'::jsonb, 'LOW'
            )
            RETURNING id;
        """, (event_id,))
        
        inserted_id = cursor.fetchone()[0]
        
        conn.commit()
        
        cursor.execute("DELETE FROM audit_trail WHERE event_id = %s", (event_id,))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        assert inserted_id is not None


class TestIntegrationNewComponents:
    """Testes de integração dos novos componentes"""
    
    def test_cache_with_ml_engine(self):
        """Testa integração do cache com ML engine"""
        from cache.prediction_cache import CachedFraudEngine
        
        cached_engine = CachedFraudEngine()
        
        transaction = {
            'transaction_id': 'INT_TEST_001',
            'amount': 5000,
            'hour': 14,
            'channel': 'PIX'
        }
        
        result1 = cached_engine.predict_with_cache(transaction)
        
        assert 'is_fraud' in result1
        assert 'cache_hit' in result1
        assert result1['cache_hit'] == False
        
        result2 = cached_engine.predict_with_cache(transaction)
        
        assert result2['cache_hit'] == True
    
    def test_production_engine_cache_integration(self):
        """Testa integração REAL do cache no ProductionFraudEngine.predict_detailed()"""
        import pandas as pd
        import time
        from ml_engine.production_fraud_engine import ProductionFraudEngine
        
        engine = ProductionFraudEngine()
        
        transaction = pd.DataFrame([{
            'amount': 12345,
            'hour': 7,
            'channel': 'PIX'
        }])
        
        start1 = time.time()
        result1 = engine.predict_detailed(transaction, use_cache=True)
        elapsed1 = (time.time() - start1) * 1000
        
        assert len(result1) == 1
        
        start2 = time.time()
        result2 = engine.predict_detailed(transaction, use_cache=True)
        elapsed2 = (time.time() - start2) * 1000
        
        assert len(result2) == 1
        assert result1[0].fraud_probability == result2[0].fraud_probability
        assert result1[0].risk_level == result2[0].risk_level
        
        stats = engine.get_cache_stats()
        assert stats.get('hits', 0) >= 1, f"Expected at least 1 cache hit, got {stats.get('hits', 0)}"
        assert elapsed2 < elapsed1, f"Cache hit should be faster: {elapsed2}ms vs {elapsed1}ms"
        assert elapsed2 < 50, f"Cache hit should be <50ms, got {elapsed2}ms"
    
    def test_production_engine_cache_batch_mixed(self):
        """Testa cache com batch misto (hits + misses)"""
        import pandas as pd
        from ml_engine.production_fraud_engine import ProductionFraudEngine
        
        engine = ProductionFraudEngine()
        
        first_batch = pd.DataFrame([
            {'amount': 100, 'hour': 10, 'channel': 'PIX'},
            {'amount': 200, 'hour': 11, 'channel': 'TED'},
        ])
        engine.predict_detailed(first_batch, use_cache=True)
        
        mixed_batch = pd.DataFrame([
            {'amount': 999, 'hour': 23, 'channel': 'PIX'},
            {'amount': 100, 'hour': 10, 'channel': 'PIX'},
            {'amount': 888, 'hour': 1, 'channel': 'Mobile'},
            {'amount': 200, 'hour': 11, 'channel': 'TED'},
        ])
        
        result = engine.predict_detailed(mixed_batch, use_cache=True)
        
        assert len(result) == 4, "Should return 4 predictions"
        
        stats = engine.get_cache_stats()
        assert stats.get('hits', 0) >= 2, f"Expected at least 2 cache hits, got stats: {stats}"
    
    def test_tracker_with_training(self):
        """Testa integração do tracker com treinamento"""
        from mlops.experiment_tracker import log_training_run
        
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = log_training_run(
                experiment_name="integration_test",
                model_name="test_model",
                parameters={
                    'n_estimators': 100,
                    'max_depth': 5
                },
                metrics={
                    'accuracy': 0.95,
                    'f1_score': 0.92
                }
            )
            
            assert run_id is not None
            assert len(run_id) == 12
    
    def test_bahnsen_feature_training(self):
        """Testa treino com features Bahnsen e split temporal estrito"""
        import pandas as pd
        import numpy as np
        from ml_engine.production_fraud_engine import ProductionFraudEngine
        
        np.random.seed(42)
        n_samples = 500
        fraud_rate = 0.15
        
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='h')
        
        is_fraud = np.random.random(n_samples) < fraud_rate
        
        amounts = np.where(
            is_fraud,
            np.random.uniform(5000, 50000, n_samples),
            np.random.exponential(500, n_samples)
        )
        
        hours = np.where(
            is_fraud,
            np.random.choice([0, 1, 2, 3, 23], n_samples),
            np.random.choice(range(8, 22), n_samples)
        )
        
        data = {
            'user_id': [f'user_{i % 50}' for i in range(n_samples)],
            'amount': amounts,
            'hour': hours,
            'channel': np.random.choice(['PIX', 'TED', 'BOLETO'], n_samples),
            'created_at': dates
        }
        
        X = pd.DataFrame(data)
        y = is_fraud.astype(int)
        
        engine = ProductionFraudEngine()
        engine.train_with_bahnsen_features(
            X, y,
            timestamp_col='created_at',
            use_temporal_split=True
        )
        
        assert engine.is_trained == True
        assert engine.metrics is not None
        assert len(engine.feature_names) >= 20, f"Expected 20+ features, got {len(engine.feature_names)}"
        
        assert hasattr(engine.metrics, 'accuracy')
        assert hasattr(engine.metrics, 'precision')
        assert hasattr(engine.metrics, 'recall')
        assert hasattr(engine.metrics, 'f1_score')
        
        print(f"Bahnsen training completed with {len(engine.feature_names)} features")
        print(f"Metrics: Acc={engine.metrics.accuracy:.3f}, P={engine.metrics.precision:.3f}, R={engine.metrics.recall:.3f}, F1={engine.metrics.f1_score:.3f}")
    
    def test_bahnsen_no_timestamp_fallback(self):
        """Testa fallback para stratified split quando não há timestamps válidos"""
        import pandas as pd
        import numpy as np
        from ml_engine.production_fraud_engine import ProductionFraudEngine
        
        np.random.seed(42)
        n_samples = 300
        fraud_rate = 0.15
        
        is_fraud = np.random.random(n_samples) < fraud_rate
        
        data = {
            'user_id': [f'user_{i % 30}' for i in range(n_samples)],
            'amount': np.random.exponential(500, n_samples),
            'hour': np.random.choice(range(24), n_samples),
            'channel': np.random.choice(['PIX', 'TED'], n_samples),
        }
        
        X = pd.DataFrame(data)
        y = is_fraud.astype(int)
        
        engine = ProductionFraudEngine()
        engine.train_with_bahnsen_features(
            X, y,
            timestamp_col='created_at',
            use_temporal_split=True
        )
        
        assert engine.is_trained == True
        assert engine.metrics is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
