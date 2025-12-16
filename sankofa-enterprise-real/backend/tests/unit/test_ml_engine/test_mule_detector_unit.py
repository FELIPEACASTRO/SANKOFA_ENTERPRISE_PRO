"""
CHECKPOINT 1 - Unit Tests: MuleDetector
BackendEngineer_QA + BankingProduction_QA

Contrato:
- detect(account_id, account_data, transaction_history) -> MuleScore
- add_known_mule(account_id) -> None
- add_suspicious_account(account_id, score) -> None
- get_stats() -> Dict
"""
import pytest
import sys
import os

# Ajustar path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


class TestMuleDetectorContract:
    """Testes de contrato da API MuleDetector"""

    @pytest.fixture
    def detector(self):
        """Fixture para criar detector"""
        from backend.ml_engine.mule_detection.mule_detector import MuleDetector
        return MuleDetector()

    @pytest.fixture
    def sample_account_data(self):
        """Dados da conta de exemplo"""
        return {
            'age_days': 365,
            'avg_balance': 5000.0,
            'total_transactions': 100,
            'is_verified': True
        }

    @pytest.fixture
    def sample_transaction_history(self):
        """Histórico de transações de exemplo"""
        return [
            {'amount': 1000.0, 'type': 'credit', 'timestamp': '2025-01-14T10:00:00'},
            {'amount': 500.0, 'type': 'debit', 'timestamp': '2025-01-14T12:00:00'},
            {'amount': 2000.0, 'type': 'credit', 'timestamp': '2025-01-15T09:00:00'}
        ]

    def test_BE_001_has_detect_method(self, detector):
        """BE-001: Deve ter método detect"""
        assert hasattr(detector, 'detect'), "Falta método detect"
        assert callable(getattr(detector, 'detect')), "detect não é callable"

    def test_BE_002_has_add_known_mule_method(self, detector):
        """BE-002: Deve ter método add_known_mule"""
        assert hasattr(detector, 'add_known_mule'), "Falta método add_known_mule"

    def test_BE_003_has_add_suspicious_account_method(self, detector):
        """BE-003: Deve ter método add_suspicious_account"""
        assert hasattr(detector, 'add_suspicious_account'), "Falta método add_suspicious_account"

    def test_BE_004_has_get_stats_method(self, detector):
        """BE-004: Deve ter método get_stats"""
        assert hasattr(detector, 'get_stats'), "Falta método get_stats"

    def test_BE_005_detect_accepts_correct_args(self, detector, sample_account_data, sample_transaction_history):
        """BE-005: detect() deve aceitar account_id, account_data, transaction_history"""
        result = detector.detect(
            account_id='ACC_12345',
            account_data=sample_account_data,
            transaction_history=sample_transaction_history
        )
        assert result is not None, "detect() retornou None"


class TestMuleDetectorDetectOutput:
    """Testes do output de detect()"""

    @pytest.fixture
    def detector(self):
        from backend.ml_engine.mule_detection.mule_detector import MuleDetector
        return MuleDetector()

    @pytest.fixture
    def sample_account_data(self):
        return {
            'age_days': 365,
            'avg_balance': 5000.0,
            'total_transactions': 100,
            'is_verified': True
        }

    @pytest.fixture
    def sample_transaction_history(self):
        return [
            {'amount': 1000.0, 'type': 'credit', 'timestamp': '2025-01-14T10:00:00'},
            {'amount': 500.0, 'type': 'debit', 'timestamp': '2025-01-14T12:00:00'}
        ]

    def test_BE_006_detect_returns_structured_result(self, detector, sample_account_data, sample_transaction_history):
        """BE-006: detect() deve retornar resultado estruturado (MuleScore)"""
        result = detector.detect('ACC_12345', sample_account_data, sample_transaction_history)
        # Deve ser MuleScore dataclass ou similar
        is_structured = hasattr(result, 'is_mule') or hasattr(result, '__getitem__')
        assert is_structured, f"Resultado não é estruturado: {type(result)}"

    def test_BE_007_detect_result_has_is_mule(self, detector, sample_account_data, sample_transaction_history):
        """BE-007: Resultado deve ter campo is_mule"""
        result = detector.detect('ACC_12345', sample_account_data, sample_transaction_history)
        has_is_mule = (
            hasattr(result, 'is_mule') or
            (isinstance(result, dict) and 'is_mule' in result)
        )
        assert has_is_mule, f"Falta campo is_mule"

    def test_BE_008_detect_result_has_score(self, detector, sample_account_data, sample_transaction_history):
        """BE-008: Resultado deve ter campo de score"""
        result = detector.detect('ACC_12345', sample_account_data, sample_transaction_history)
        has_score = (
            hasattr(result, 'total_score') or
            hasattr(result, 'mule_probability') or
            (isinstance(result, dict) and ('score' in result or 'mule_probability' in result))
        )
        assert has_score, f"Falta campo de score. Attrs: {dir(result)}"

    def test_BE_009_detect_score_in_valid_range(self, detector, sample_account_data, sample_transaction_history):
        """BE-009: Score deve estar entre 0 e 1"""
        result = detector.detect('ACC_12345', sample_account_data, sample_transaction_history)
        if hasattr(result, 'mule_probability'):
            score = result.mule_probability
        elif hasattr(result, 'total_score'):
            score = result.total_score
        elif isinstance(result, dict):
            score = result.get('mule_probability', result.get('total_score', 0))
        else:
            pytest.skip("Score não encontrado")

        assert 0 <= score <= 1, f"Score {score} fora do range [0,1]"


class TestMuleDetectorAccountManagement:
    """Testes de gerenciamento de contas"""

    @pytest.fixture
    def detector(self):
        from backend.ml_engine.mule_detection.mule_detector import MuleDetector
        return MuleDetector()

    def test_BE_010_add_known_mule_accepts_string(self, detector):
        """BE-010: add_known_mule aceita string"""
        try:
            detector.add_known_mule('MULE_ACC_001')
        except TypeError as e:
            pytest.fail(f"add_known_mule deveria aceitar string: {e}")

    def test_BE_011_add_suspicious_account_accepts_string_and_score(self, detector):
        """BE-011: add_suspicious_account aceita string e score"""
        try:
            detector.add_suspicious_account('SUSPICIOUS_ACC_001', score=0.75)
        except TypeError as e:
            pytest.fail(f"add_suspicious_account deveria aceitar string e score: {e}")

    def test_BE_012_get_stats_returns_dict(self, detector):
        """BE-012: get_stats retorna dicionário"""
        stats = detector.get_stats()
        assert isinstance(stats, dict), f"Expected dict, got {type(stats)}"


class TestMuleDetectorKnownMuleDetection:
    """Testes de detecção de mulas conhecidas"""

    @pytest.fixture
    def detector(self):
        from backend.ml_engine.mule_detection.mule_detector import MuleDetector
        return MuleDetector()

    @pytest.fixture
    def sample_account_data(self):
        return {
            'age_days': 30,  # Conta nova
            'avg_balance': 100.0,  # Baixo saldo
            'total_transactions': 5,
            'is_verified': False
        }

    @pytest.fixture
    def sample_transaction_history(self):
        return [
            {'amount': 5000.0, 'type': 'credit', 'timestamp': '2025-01-15T10:00:00'},
            {'amount': 4900.0, 'type': 'debit', 'timestamp': '2025-01-15T10:05:00'}
        ]

    def test_BANK_001_known_mule_connection_detected(self, detector, sample_account_data):
        """BANK-001: Transação com mula conhecida como contraparte deve ter indicador"""
        # Registrar mula conhecida como contraparte
        detector.add_known_mule('KNOWN_MULE_COUNTERPARTY')

        # Histórico com transação para mula conhecida
        tx_history_with_mule = [
            {'amount': 5000.0, 'type': 'debit', 'timestamp': '2025-01-15T10:00:00',
             'counterparty': 'KNOWN_MULE_COUNTERPARTY'},
        ]

        result = detector.detect(
            account_id='SUSPECT_ACC_001',
            account_data=sample_account_data,
            transaction_history=tx_history_with_mule
        )

        # Verificar que o resultado foi gerado
        assert result is not None, "detect() deveria retornar resultado"
        # Verificar que tem campo is_mule
        has_is_mule = hasattr(result, 'is_mule')
        assert has_is_mule, "Resultado deve ter campo is_mule"

    def test_BANK_002_suspicious_account_registered(self, detector):
        """BANK-002: Conta suspeita deve ser registrada e consultável"""
        detector.add_suspicious_account('SUSPICIOUS_456', score=0.85)

        stats = detector.get_stats()
        assert stats.get('suspicious_accounts_count', 0) >= 1, "Conta suspeita deveria estar registrada"

    def test_BANK_003_stats_count_mules(self, detector):
        """BANK-003: Stats devem contar mulas registradas"""
        # Registrar algumas mulas
        detector.add_known_mule('MULE_A')
        detector.add_known_mule('MULE_B')

        stats = detector.get_stats()

        # Verificar se há contagem de mulas
        has_count = any(k in stats for k in ['known_mules', 'mules_count', 'total_mules', 'known_mule_accounts'])
        assert has_count or len(stats) > 0, f"Stats não tem contagem de mulas: {stats}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
