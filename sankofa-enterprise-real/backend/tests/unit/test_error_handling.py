"""
Testes Unitários - Error Handling
Checklist 4.3: Tratamento e exposição de erros
"""

import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.error_handling import (
    SanEnterpriseError,
    ValidationError,
    DatabaseError,
    MLError,
    SecurityError,
    ComplianceError,
    handle_errors
)


class TestErrorHierarchy:
    """Testa hierarquia de erros personalizados"""
    
    def test_base_error(self):
        """Testa erro base"""
        error = SanEnterpriseError("Test error", error_code="TEST001")
        
        assert str(error) == "Test error"
        assert error.error_code == "TEST001"
        assert error.severity == "ERROR"
    
    def test_validation_error(self):
        """Testa erro de validação"""
        error = ValidationError("Invalid input", field="amount")
        
        assert isinstance(error, SanEnterpriseError)
        assert error.error_code == "VAL001"
        assert error.severity == "WARNING"
        assert error.field == "amount"
    
    def test_database_error(self):
        """Testa erro de banco de dados"""
        error = DatabaseError("Connection failed", operation="SELECT")
        
        assert error.error_code == "DB001"
        assert error.severity == "CRITICAL"
        assert error.operation == "SELECT"
    
    def test_ml_error(self):
        """Testa erro de ML"""
        error = MLError("Model not trained", model="FraudEngine")
        
        assert error.error_code == "ML001"
        assert error.severity == "ERROR"
        assert error.model == "FraudEngine"
    
    def test_security_error(self):
        """Testa erro de segurança"""
        error = SecurityError("Unauthorized access", resource="admin")
        
        assert error.error_code == "SEC001"
        assert error.severity == "CRITICAL"
        assert error.resource == "admin"
    
    def test_compliance_error(self):
        """Testa erro de compliance"""
        error = ComplianceError("LGPD violation", regulation="LGPD")
        
        assert error.error_code == "CMP001"
        assert error.severity == "CRITICAL"
        assert error.regulation == "LGPD"


class TestErrorHandler:
    """Testa decorator de erro"""
    
    def test_handle_errors_success(self):
        """Testa que função normal funciona"""
        @handle_errors
        def working_function():
            return "success"
        
        result = working_function()
        assert result == "success"
    
    def test_handle_errors_catches_exception(self):
        """Testa que decorator captura exceção"""
        @handle_errors
        def failing_function():
            raise ValueError("Test error")
        
        # Não deve propagar a exceção
        result = failing_function()
        assert result is None
    
    def test_handle_errors_with_custom_exception(self):
        """Testa com exceção customizada"""
        @handle_errors
        def custom_error_function():
            raise ValidationError("Invalid data", field="test")
        
        result = custom_error_function()
        assert result is None
