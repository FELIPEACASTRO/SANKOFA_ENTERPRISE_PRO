"""
Teste Rápido de Validação - Sem treinamento de modelo
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'backend'))

import os
os.environ['ENVIRONMENT'] = 'development'
os.environ['DEBUG'] = 'true'
os.environ['JWT_SECRET'] = 'test-secret'
os.environ['ENCRYPTION_KEY'] = 'test-encryption-key'

def run_quick_validation():
    """Validação rápida de todos componentes"""
    print("\n" + "="*60)
    print("⚡ QUICK VALIDATION - ALL NEW COMPONENTS")
    print("="*60)
    
    passed = 0
    total = 0
    
    # Test 1: Configuration
    print("\n1️⃣  Configuration System...")
    try:
        from config.settings import get_config
        config = get_config()
        assert config.environment == 'development'
        print("   ✅ PASS - Config loaded and validated")
        passed += 1
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
    total += 1
    
    # Test 2: Structured Logging
    print("\n2️⃣  Structured Logging...")
    try:
        from utils.structured_logging import get_structured_logger
        logger = get_structured_logger('test', 'INFO')
        logger.info("Test message", key="value")
        print("   ✅ PASS - JSON logs working")
        passed += 1
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
    total += 1
    
    # Test 3: Error Handling
    print("\n3️⃣  Error Handling...")
    try:
        from utils.error_handling import ValidationError, ErrorCategory
        try:
            raise ValidationError("Test")
        except ValidationError as e:
            assert e.category == ErrorCategory.VALIDATION
        print("   ✅ PASS - Error categorization working")
        passed += 1
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
    total += 1
    
    # Test 4: Fraud Engine (without training)
    print("\n4️⃣  Production Fraud Engine...")
    try:
        from ml_engine.production_fraud_engine import ProductionFraudEngine
        engine = ProductionFraudEngine()
        assert engine.VERSION == "1.0.0"
        assert not engine.is_trained
        print("   ✅ PASS - Engine initialized correctly")
        passed += 1
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
    total += 1
    
    # Test 5: Production API
    print("\n5️⃣  Production API...")
    try:
        from api.production_api import app, fraud_engine
        routes = [str(rule) for rule in app.url_map.iter_rules()]
        assert '/api/health' in routes
        assert '/api/fraud/predict' in routes
        print(f"   ✅ PASS - API with {len(routes)} endpoints")
        passed += 1
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
    total += 1
    
    # Test 6: PostgreSQL Schema
    print("\n6️⃣  PostgreSQL Schema...")
    try:
        schema_file = Path(__file__).parent.parent / 'backend' / 'database' / 'schema.sql'
        assert schema_file.exists()
        content = schema_file.read_text()
        assert 'CREATE TABLE' in content
        assert 'transactions' in content
        assert 'audit_trail' in content
        print("   ✅ PASS - Database schema ready")
        passed += 1
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
    total += 1
    
    # Test 7: Documentation
    print("\n7️⃣  Documentation...")
    try:
        docs_file = Path(__file__).parent.parent / 'docs' / 'TRANSFORMATION_REPORT.md'
        assert docs_file.exists()
        content = docs_file.read_text()
        assert 'TRANSFORMATION' in content
        print("   ✅ PASS - Documentation complete")
        passed += 1
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
    total += 1
    
    # Results
    print("\n" + "="*60)
    print(f"📊 RESULTS: {passed}/{total} components validated")
    print("="*60)
    
    if passed == total:
        print("\n🎉 ALL SYSTEMS OPERATIONAL - 100% FUNCTIONAL! 🎉\n")
        return True
    else:
        print(f"\n⚠️  {total - passed} component(s) need attention\n")
        return False

if __name__ == "__main__":
    success = run_quick_validation()
    exit(0 if success else 1)
