"""
Sankofa Enterprise Pro - Error Handling System
Sistema de tratamento de erros enterprise com categorização e recovery
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
import traceback
from datetime import datetime

from utils.structured_logging import get_structured_logger

logger = get_structured_logger("error_handling", "INFO")


class ErrorSeverity(Enum):
    """Severidade do erro"""

    LOW = "low"  # Erro recuperável, não impacta operação
    MEDIUM = "medium"  # Erro recuperável, pode impactar performance
    HIGH = "high"  # Erro crítico, impacta funcionalidade
    CRITICAL = "critical"  # Erro fatal, requer intervenção imediata


class ErrorCategory(Enum):
    """Categoria do erro"""

    VALIDATION = "validation"  # Erro de validação de dados
    DATABASE = "database"  # Erro de banco de dados
    ML_MODEL = "ml_model"  # Erro do modelo de ML
    NETWORK = "network"  # Erro de rede/API externa
    SECURITY = "security"  # Erro de segurança
    COMPLIANCE = "compliance"  # Erro de compliance
    CONFIGURATION = "configuration"  # Erro de configuração
    UNKNOWN = "unknown"  # Erro desconhecido


@dataclass
class ErrorContext:
    """Contexto do erro"""

    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception]
    stack_trace: Optional[str]
    context_data: Dict[str, Any]
    timestamp: str
    recovery_action: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "exception_type": type(self.exception).__name__ if self.exception else None,
            "exception_message": str(self.exception) if self.exception else None,
            "stack_trace": self.stack_trace,
            "context_data": self.context_data,
            "timestamp": self.timestamp,
            "recovery_action": self.recovery_action,
        }


class SankoException(Exception):
    """Base exception para todas as exceções do sistema"""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        recovery_action: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.recovery_action = recovery_action
        self.error_id = self._generate_error_id()
        self.timestamp = datetime.utcnow().isoformat() + "Z"

    def _generate_error_id(self) -> str:
        """Gera ID único para o erro"""
        import uuid

        return f"ERR_{self.category.value.upper()}_{uuid.uuid4().hex[:8]}"

    def get_context(self) -> ErrorContext:
        """Retorna contexto completo do erro"""
        return ErrorContext(
            error_id=self.error_id,
            category=self.category,
            severity=self.severity,
            message=self.message,
            exception=self,
            stack_trace=traceback.format_exc(),
            context_data=self.context,
            timestamp=self.timestamp,
            recovery_action=self.recovery_action,
        )


# Exceções específicas


class ValidationError(SankoException):
    """Erro de validação de dados"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            context=context,
            recovery_action="Validate and correct input data",
        )


class DatabaseError(SankoException):
    """Erro de banco de dados"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_action="Check database connection and retry",
        )


class MLModelError(SankoException):
    """Erro do modelo de ML"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.ML_MODEL,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_action="Check model files and retrain if necessary",
        )


class SecurityError(SankoException):
    """Erro de segurança"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            recovery_action="Review security logs and investigate immediately",
        )


class ComplianceError(SankoException):
    """Erro de compliance"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.COMPLIANCE,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            recovery_action="Review compliance requirements and correct violation",
        )


class ConfigurationError(SankoException):
    """Erro de configuração"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_action="Check environment variables and configuration files",
        )


def handle_error(
    exception: Exception, context: Optional[Dict[str, Any]] = None, raise_exception: bool = True
) -> ErrorContext:
    """
    Handler centralizado de erros

    Args:
        exception: Exceção capturada
        context: Contexto adicional
        raise_exception: Se True, re-raise a exceção

    Returns:
        Contexto do erro
    """
    # Se for SankoException, já tem contexto
    if isinstance(exception, SankoException):
        error_context = exception.get_context()
        if context:
            error_context.context_data.update(context)
    else:
        # Criar contexto para exceção genérica
        error_context = ErrorContext(
            error_id=f"ERR_GENERIC_{datetime.utcnow().timestamp()}",
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            message=str(exception),
            exception=exception,
            stack_trace=traceback.format_exc(),
            context_data=context or {},
            timestamp=datetime.utcnow().isoformat() + "Z",
            recovery_action="Review error logs and context",
        )

    # Log estruturado do erro
    log_data = error_context.to_dict()

    if error_context.severity == ErrorSeverity.CRITICAL:
        logger.critical("Critical error occurred", **log_data)
    elif error_context.severity == ErrorSeverity.HIGH:
        logger.error("High severity error occurred", **log_data)
    elif error_context.severity == ErrorSeverity.MEDIUM:
        logger.warning("Medium severity error occurred", **log_data)
    else:
        logger.info("Low severity error occurred", **log_data)

    # Re-raise se solicitado
    if raise_exception:
        raise exception

    return error_context


# Decorator para tratamento automático de erros
def with_error_handling(
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    raise_exception: bool = True,
):
    """Decorator para adicionar error handling a funções"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except SankoException:
                # Já é SankoException, apenas re-raise
                raise
            except Exception as e:
                # Converter para SankoException
                sanko_exception = SankoException(
                    message=f"Error in {func.__name__}: {str(e)}",
                    category=category,
                    severity=severity,
                    context={
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs),
                    },
                )
                handle_error(sanko_exception, raise_exception=raise_exception)

        return wrapper

    return decorator
