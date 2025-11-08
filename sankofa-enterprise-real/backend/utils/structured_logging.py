"""
Sankofa Enterprise Pro - Sistema de Logging Estruturado
Logging em JSON para observabilidade enterprise (DataDog, Splunk, ELK)
"""

import logging
import json
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
from functools import wraps
import sys

class StructuredLogger:
    """Logger estruturado com output JSON"""
    
    def __init__(self, name: str, level: str = 'INFO'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove handlers existentes
        self.logger.handlers = []
        
        # Handler com formato JSON
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)
    
    def _build_log_entry(
        self,
        level: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ) -> Dict[str, Any]:
        """Constrói entrada de log estruturada"""
        entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': level,
            'message': message,
            'logger': self.logger.name
        }
        
        if extra:
            entry.update(extra)
        
        if exception:
            entry['error'] = {
                'type': type(exception).__name__,
                'message': str(exception),
                'traceback': traceback.format_exc()
            }
        
        return entry
    
    def debug(self, message: str, **extra):
        """Log debug"""
        entry = self._build_log_entry('DEBUG', message, extra)
        self.logger.debug(json.dumps(entry))
    
    def info(self, message: str, **extra):
        """Log info"""
        entry = self._build_log_entry('INFO', message, extra)
        self.logger.info(json.dumps(entry))
    
    def warning(self, message: str, **extra):
        """Log warning"""
        entry = self._build_log_entry('WARNING', message, extra)
        self.logger.warning(json.dumps(entry))
    
    def error(self, message: str, exception: Optional[Exception] = None, **extra):
        """Log error"""
        entry = self._build_log_entry('ERROR', message, extra, exception)
        self.logger.error(json.dumps(entry))
    
    def critical(self, message: str, exception: Optional[Exception] = None, **extra):
        """Log critical"""
        entry = self._build_log_entry('CRITICAL', message, extra, exception)
        self.logger.critical(json.dumps(entry))

class JSONFormatter(logging.Formatter):
    """Formatter que converte logs para JSON"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Formata log record como JSON"""
        # Se já for JSON, retornar diretamente
        try:
            json.loads(record.msg)
            return record.msg
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Caso contrário, criar estrutura JSON
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if record.exc_info:
            log_entry['error'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        return json.dumps(log_entry)

def log_execution_time(logger: StructuredLogger):
    """Decorator para logar tempo de execução"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func.__name__
            
            logger.debug(
                f"Starting execution: {function_name}",
                function=function_name,
                args_count=len(args),
                kwargs_count=len(kwargs)
            )
            
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # ms
                
                logger.info(
                    f"Completed execution: {function_name}",
                    function=function_name,
                    execution_time_ms=round(execution_time, 2),
                    status='success'
                )
                
                return result
            
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                logger.error(
                    f"Failed execution: {function_name}",
                    exception=e,
                    function=function_name,
                    execution_time_ms=round(execution_time, 2),
                    status='failed'
                )
                
                raise
        
        return wrapper
    return decorator

def get_structured_logger(name: str, level: str = 'INFO') -> StructuredLogger:
    """Factory para criar loggers estruturados"""
    return StructuredLogger(name, level)

# Logger global para a aplicação
app_logger = get_structured_logger('sankofa.enterprise', 'INFO')

# Exemplo de uso
if __name__ == "__main__":
    logger = get_structured_logger('test', 'DEBUG')
    
    logger.debug("Debug message", user_id="123", action="test")
    logger.info("Info message", transaction_id="TXN_001", amount=1000.00)
    logger.warning("Warning message", threshold_exceeded=True, current_value=95)
    
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.error("Error occurred", exception=e, context="test_execution")
    
    # Com decorator
    @log_execution_time(logger)
    def slow_function():
        time.sleep(0.1)
        return "completed"
    
    slow_function()
