"""
CHECKPOINT 3 - Contract Tests: FraudDetectionService
BankingProduction_QA + MLEngineer_QA

Contrato ABC FraudDetectionService:
- analyze_transaction(transaction: Transaction) -> FraudAnalysisResult
- get_model_info() -> Dict[str, Any]

Este teste verifica o contrato da interface sem testar implementações específicas
"""
import pytest
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


class TestFraudDetectionServiceContract:
    """Contract tests para FraudDetectionService"""

    def test_CONTRACT_001_interface_exists(self):
        """CONTRACT-001: FraudDetectionService deve existir"""
        from backend.core.interfaces import FraudDetectionService
        assert FraudDetectionService is not None

    def test_CONTRACT_002_is_abstract(self):
        """CONTRACT-002: Deve ser classe abstrata"""
        from backend.core.interfaces import FraudDetectionService
        from abc import ABC

        assert issubclass(FraudDetectionService, ABC), "FraudDetectionService deve herdar de ABC"

    def test_CONTRACT_003_has_analyze_transaction_method(self):
        """CONTRACT-003: Deve ter método analyze_transaction"""
        from backend.core.interfaces import FraudDetectionService

        assert hasattr(FraudDetectionService, 'analyze_transaction'), "Falta método analyze_transaction"

    def test_CONTRACT_004_analyze_transaction_is_abstract(self):
        """CONTRACT-004: analyze_transaction deve ser abstrato"""
        from backend.core.interfaces import FraudDetectionService

        method = getattr(FraudDetectionService, 'analyze_transaction')
        assert getattr(method, '__isabstractmethod__', False), "analyze_transaction deve ser abstrato"

    def test_CONTRACT_005_has_get_model_info_method(self):
        """CONTRACT-005: Deve ter método get_model_info"""
        from backend.core.interfaces import FraudDetectionService

        assert hasattr(FraudDetectionService, 'get_model_info'), "Falta método get_model_info"

    def test_CONTRACT_006_get_model_info_is_abstract(self):
        """CONTRACT-006: get_model_info deve ser abstrato"""
        from backend.core.interfaces import FraudDetectionService

        method = getattr(FraudDetectionService, 'get_model_info')
        assert getattr(method, '__isabstractmethod__', False), "get_model_info deve ser abstrato"

    def test_CONTRACT_007_cannot_instantiate(self):
        """CONTRACT-007: Não deve ser possível instanciar diretamente"""
        from backend.core.interfaces import FraudDetectionService

        with pytest.raises(TypeError):
            FraudDetectionService()


class TestTransactionRepositoryContract:
    """Contract tests para TransactionRepository"""

    def test_CONTRACT_001_interface_exists(self):
        """CONTRACT-001: TransactionRepository deve existir"""
        from backend.core.interfaces import TransactionRepository
        assert TransactionRepository is not None

    def test_CONTRACT_002_is_abstract(self):
        """CONTRACT-002: Deve ser classe abstrata"""
        from backend.core.interfaces import TransactionRepository
        from abc import ABC

        assert issubclass(TransactionRepository, ABC)

    def test_CONTRACT_003_has_save_method(self):
        """CONTRACT-003: Deve ter método save"""
        from backend.core.interfaces import TransactionRepository
        assert hasattr(TransactionRepository, 'save'), "Falta método save"

    def test_CONTRACT_004_has_find_by_id_method(self):
        """CONTRACT-004: Deve ter método find_by_id"""
        from backend.core.interfaces import TransactionRepository
        assert hasattr(TransactionRepository, 'find_by_id'), "Falta método find_by_id"

    def test_CONTRACT_005_has_find_by_customer_method(self):
        """CONTRACT-005: Deve ter método find_by_customer"""
        from backend.core.interfaces import TransactionRepository
        assert hasattr(TransactionRepository, 'find_by_customer'), "Falta método find_by_customer"

    def test_CONTRACT_006_has_find_by_date_range_method(self):
        """CONTRACT-006: Deve ter método find_by_date_range"""
        from backend.core.interfaces import TransactionRepository
        assert hasattr(TransactionRepository, 'find_by_date_range'), "Falta método find_by_date_range"

    def test_CONTRACT_007_cannot_instantiate(self):
        """CONTRACT-007: Não deve ser possível instanciar diretamente"""
        from backend.core.interfaces import TransactionRepository

        with pytest.raises(TypeError):
            TransactionRepository()


class TestCacheServiceContract:
    """Contract tests para CacheService"""

    def test_CONTRACT_001_interface_exists(self):
        """CONTRACT-001: CacheService deve existir"""
        from backend.core.interfaces import CacheService
        assert CacheService is not None

    def test_CONTRACT_002_has_get_method(self):
        """CONTRACT-002: Deve ter método get"""
        from backend.core.interfaces import CacheService
        assert hasattr(CacheService, 'get'), "Falta método get"

    def test_CONTRACT_003_has_set_method(self):
        """CONTRACT-003: Deve ter método set"""
        from backend.core.interfaces import CacheService
        assert hasattr(CacheService, 'set'), "Falta método set"

    def test_CONTRACT_004_has_delete_method(self):
        """CONTRACT-004: Deve ter método delete"""
        from backend.core.interfaces import CacheService
        assert hasattr(CacheService, 'delete'), "Falta método delete"

    def test_CONTRACT_005_cannot_instantiate(self):
        """CONTRACT-005: Não deve ser possível instanciar diretamente"""
        from backend.core.interfaces import CacheService

        with pytest.raises(TypeError):
            CacheService()


class TestMLModelStrategyContract:
    """Contract tests para MLModelStrategy"""

    def test_CONTRACT_001_interface_exists(self):
        """CONTRACT-001: MLModelStrategy deve existir"""
        from backend.infrastructure.ml_service import MLModelStrategy
        assert MLModelStrategy is not None

    def test_CONTRACT_002_is_abstract(self):
        """CONTRACT-002: Deve ser classe abstrata"""
        from backend.infrastructure.ml_service import MLModelStrategy
        from abc import ABC

        assert issubclass(MLModelStrategy, ABC)

    def test_CONTRACT_003_has_predict_method(self):
        """CONTRACT-003: Deve ter método predict"""
        from backend.infrastructure.ml_service import MLModelStrategy
        assert hasattr(MLModelStrategy, 'predict'), "Falta método predict"

    def test_CONTRACT_004_cannot_instantiate(self):
        """CONTRACT-004: Não deve ser possível instanciar diretamente"""
        from backend.infrastructure.ml_service import MLModelStrategy

        with pytest.raises(TypeError):
            MLModelStrategy()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
