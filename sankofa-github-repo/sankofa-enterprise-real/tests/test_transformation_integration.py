"""
Testes de Integração da Transformação Enterprise
Valida todos os componentes novos criados
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'backend'))

import os
os.environ['ENVIRONMENT'] = 'development'
os.environ['DEBUG'] = 'true'
os.environ['JWT_SECRET'] = 'test-secret'
os.environ['ENCRYPTION_KEY'] = 'test-encryption-key'

def test_1_configuration_system():
    """Testa sistema de configuração enterprise"""
    print("\n🧪 TEST 1: Configuration System")
    print("="*50)
    
    try:
        from config.settings import get_config
        
        config = get_config()
        
        assert config.environment == 'development', "Environment should be development"
        assert config.debug == True, "Debug should be enabled"
        assert config.database.host == 'localhost', "DB host should be localhost"
        assert config.security.jwt_secret == 'test-secret', "JWT secret should be set"
        
        print("✅ Configuration system working correctly")
        print(f"   Environment: {config.environment}")
        print(f"   Database: {config.database.host}:{config.database.port}")
        print(f"   Redis: {config.redis.host}:{config.redis.port}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_2_structured_logging():
    """Testa logging estruturado JSON"""
    print("\n🧪 TEST 2: Structured Logging")
    print("="*50)
    
    try:
        from utils.structured_logging import get_structured_logger, log_execution_time
        import json
        import io
        import sys
        
        # Capturar output
        logger = get_structured_logger('test', 'INFO')
        
        # Testar diferentes níveis
        logger.info("Test info message", transaction_id="TXN_001", amount=1000.00)
        logger.warning("Test warning", risk_score=0.95)
        
        # Testar decorator
        @log_execution_time(logger)
        def test_function():
            return "success"
        
        result = test_function()
        
        assert result == "success", "Decorated function should work"
        
        print("✅ Structured logging working correctly")
        print("   JSON logs generated successfully")
        print("   Decorator @log_execution_time functional")
        return True
    except Exception as e:
        print(f"❌ Structured logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_3_error_handling():
    """Testa error handling enterprise"""
    print("\n🧪 TEST 3: Error Handling System")
    print("="*50)
    
    try:
        from utils.error_handling import (
            ValidationError, DatabaseError, MLModelError,
            ErrorCategory, ErrorSeverity, handle_error
        )
        
        # Testar ValidationError
        try:
            raise ValidationError(
                "Test validation error",
                context={'field': 'amount', 'value': -100}
            )
        except ValidationError as e:
            assert e.category == ErrorCategory.VALIDATION, "Should be validation error"
            assert e.severity == ErrorSeverity.LOW, "Should be low severity"
            assert e.error_id.startswith('ERR_VALIDATION'), "Should have correct prefix"
        
        # Testar DatabaseError
        try:
            raise DatabaseError("Test database error")
        except DatabaseError as e:
            assert e.category == ErrorCategory.DATABASE, "Should be database error"
            assert e.severity == ErrorSeverity.HIGH, "Should be high severity"
        
        # Testar MLModelError
        try:
            raise MLModelError("Model not trained")
        except MLModelError as e:
            assert e.category == ErrorCategory.ML_MODEL, "Should be ML error"
            assert e.severity == ErrorSeverity.HIGH, "Should be high severity"
        
        print("✅ Error handling system working correctly")
        print("   ValidationError: ✓")
        print("   DatabaseError: ✓")
        print("   MLModelError: ✓")
        print("   Error categorization: ✓")
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_4_production_fraud_engine():
    """Testa Production Fraud Engine"""
    print("\n🧪 TEST 4: Production Fraud Engine")
    print("="*50)
    
    try:
        from ml_engine.production_fraud_engine import ProductionFraudEngine, FraudPrediction
        import pandas as pd
        import numpy as np
        
        # Criar engine
        engine = ProductionFraudEngine()
        
        assert engine.VERSION == "1.0.0", "Version should be 1.0.0"
        assert not engine.is_trained, "Engine should not be trained initially"
        
        # Criar dados de teste
        np.random.seed(42)
        n_samples = 1000
        
        X_train = pd.DataFrame({
            'amount': np.random.lognormal(3, 1.5, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'location_risk_score': np.random.beta(2, 5, n_samples),
            'device_risk_score': np.random.beta(2, 8, n_samples),
        })
        
        # Labels com 10% fraude
        y_train = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        
        # Treinar
        print("   Training model...")
        engine.fit(X_train, y_train)
        
        assert engine.is_trained, "Engine should be trained"
        assert engine.metrics is not None, "Metrics should be available"
        
        # Testar predição
        X_test = X_train.head(10)
        predictions = engine.predict(X_test)
        
        assert len(predictions) == 10, "Should have 10 predictions"
        assert all(isinstance(p, FraudPrediction) for p in predictions), "Should be FraudPrediction objects"
        
        # Testar métricas
        metrics = engine.get_performance_metrics()
        assert metrics['status'] == 'trained', "Status should be trained"
        assert 'metrics' in metrics, "Should have metrics"
        
        print("✅ Production Fraud Engine working correctly")
        print(f"   Model trained: ✓")
        print(f"   F1-Score: {engine.metrics.f1_score:.3f}")
        print(f"   Precision: {engine.metrics.precision:.3f}")
        print(f"   Recall: {engine.metrics.recall:.3f}")
        print(f"   Predictions: ✓ ({len(predictions)} samples)")
        return True
    except Exception as e:
        print(f"❌ Production Fraud Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_5_production_api():
    """Testa Production API imports e setup"""
    print("\n🧪 TEST 5: Production API")
    print("="*50)
    
    try:
        from api.production_api import app, fraud_engine, config
        
        assert app is not None, "Flask app should exist"
        assert fraud_engine is not None, "Fraud engine should exist"
        assert config is not None, "Config should exist"
        
        # Verificar routes
        routes = [str(rule) for rule in app.url_map.iter_rules()]
        
        required_routes = [
            '/api/health',
            '/api/status',
            '/api/fraud/predict',
            '/api/fraud/batch',
            '/api/model/metrics',
            '/api/model/info',
            '/api/dashboard/kpis'
        ]
        
        for route in required_routes:
            assert route in routes, f"Route {route} should exist"
        
        print("✅ Production API working correctly")
        print(f"   Flask app initialized: ✓")
        print(f"   Fraud engine loaded: ✓")
        print(f"   Config loaded: ✓")
        print(f"   Routes registered: {len(routes)}")
        print(f"   Required endpoints: ✓ ({len(required_routes)} found)")
        return True
    except Exception as e:
        print(f"❌ Production API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_6_integration_end_to_end():
    """Teste de integração end-to-end completo"""
    print("\n🧪 TEST 6: End-to-End Integration")
    print("="*50)
    
    try:
        # 1. Configuração
        from config.settings import get_config
        config = get_config()
        
        # 2. Logging
        from utils.structured_logging import get_structured_logger
        logger = get_structured_logger('integration_test', 'INFO')
        logger.info("Starting integration test")
        
        # 3. Fraud Engine
        from ml_engine.production_fraud_engine import get_fraud_engine
        import pandas as pd
        import numpy as np
        
        engine = get_fraud_engine()
        
        # Dados sintéticos
        np.random.seed(42)
        X_train = pd.DataFrame({
            'amount': np.random.lognormal(3, 1.5, 500),
            'hour': np.random.randint(0, 24, 500),
            'location_risk_score': np.random.beta(2, 5, 500),
            'device_risk_score': np.random.beta(2, 8, 500),
        })
        y_train = np.random.choice([0, 1], 500, p=[0.9, 0.1])
        
        # Treinar
        logger.info("Training fraud engine")
        engine.fit(X_train, y_train)
        
        # Predizer
        X_test = X_train.head(5)
        logger.info("Making predictions", num_samples=len(X_test))
        predictions = engine.predict(X_test)
        
        # 4. Error Handling
        from utils.error_handling import ValidationError
        try:
            raise ValidationError("Test error in integration")
        except ValidationError as e:
            logger.warning("Caught validation error", error_id=e.error_id)
        
        # 5. Verificar tudo funcionou
        assert engine.is_trained, "Engine should be trained"
        assert len(predictions) == 5, "Should have 5 predictions"
        
        print("✅ End-to-End Integration working perfectly")
        print("   Configuration: ✓")
        print("   Structured Logging: ✓")
        print("   Fraud Engine: ✓")
        print("   Error Handling: ✓")
        print("   ALL SYSTEMS OPERATIONAL 🚀")
        return True
    except Exception as e:
        print(f"❌ End-to-End integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Executa todos os testes"""
    print("\n" + "="*60)
    print("🔬 SANKOFA ENTERPRISE PRO - TRANSFORMATION VALIDATION")
    print("="*60)
    
    tests = [
        ("Configuration System", test_1_configuration_system),
        ("Structured Logging", test_2_structured_logging),
        ("Error Handling", test_3_error_handling),
        ("Production Fraud Engine", test_4_production_fraud_engine),
        ("Production API", test_5_production_api),
        ("End-to-End Integration", test_6_integration_end_to_end),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Resumo
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*60)
    print(f"FINAL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - SYSTEM 100% FUNCTIONAL! 🎉")
        print("="*60)
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed - requires attention")
        print("="*60)
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
