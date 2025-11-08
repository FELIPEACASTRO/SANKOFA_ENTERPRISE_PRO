"""
Testes Unitários - Error Handling
Checklist 4.3: Tratamento e exposição de erros
"""

import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.error_handling import (
    SankoException,
    ValidationError,
    DatabaseError,
    MLModelError,
    SecurityError,
    ComplianceError,
    ErrorCategory,
    ErrorSeverity
)


class TestErrorHierarchy:
    """Testa hierarquia de erros personalizados"""
    
    def test_base_error(self):
        """Testa erro base"""
        error = SankoException("Test error", category=ErrorCategory.UNKNOWN)
        
        assert str(error) == "Test error"
        assert error.category == ErrorCategory.UNKNOWN
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.error_id.startswith("ERR_UNKNOWN_")
    
    def test_validation_error(self):
        """Testa erro de validação"""
        error = ValidationError("Invalid input", context={'field': 'amount'})
        
        assert isinstance(error, SankoException)
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.LOW
        assert error.context['field'] == 'amount'
        assert error.recovery_action == "Validate and correct input data"
    
    def test_database_error(self):
        """Testa erro de banco de dados"""
        error = DatabaseError("Connection failed", context={'operation': 'SELECT'})
        
        assert error.category == ErrorCategory.DATABASE
        assert error.severity == ErrorSeverity.HIGH
        assert error.context['operation'] == 'SELECT'
        assert error.recovery_action == "Check database connection and retry"
    
    def test_ml_model_error(self):
        """Testa erro de ML"""
        error = MLModelError("Model not trained", context={'model': 'FraudEngine'})
        
        assert error.category == ErrorCategory.ML_MODEL
        assert error.severity == ErrorSeverity.HIGH
        assert error.context['model'] == 'FraudEngine'
    
    def test_security_error(self):
        """Testa erro de segurança"""
        error = SecurityError("Unauthorized access", context={'resource': 'admin'})
        
        assert error.category == ErrorCategory.SECURITY
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.context['resource'] == 'admin'
    
    def test_compliance_error(self):
        """Testa erro de compliance"""
        error = ComplianceError("LGPD violation", context={'regulation': 'LGPD'})
        
        assert error.category == ErrorCategory.COMPLIANCE
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.context['regulation'] == 'LGPD'
    
    def test_error_context_generation(self):
        """Testa geração de contexto do erro"""
        error = ValidationError("Test", context={'key': 'value'})
        context = error.get_context()
        
        assert context.error_id == error.error_id
        assert context.category == ErrorCategory.VALIDATION
        assert context.severity == ErrorSeverity.LOW
        assert context.message == "Test"
        assert context.context_data == {'key': 'value'}
    
    def test_error_context_to_dict(self):
        """Testa conversão de contexto para dict"""
        error = DatabaseError("Connection error")
        context = error.get_context()
        context_dict = context.to_dict()
        
        assert 'error_id' in context_dict
        assert 'category' in context_dict
        assert 'severity' in context_dict
        assert 'message' in context_dict
        assert context_dict['category'] == 'database'
        assert context_dict['severity'] == 'high'
