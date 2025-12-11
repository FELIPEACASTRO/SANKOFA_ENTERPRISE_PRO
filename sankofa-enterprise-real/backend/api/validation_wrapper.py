"""
Global Validation Wrapper
Applies Pydantic validation to all POST/PUT endpoints automatically
"""

from functools import wraps
from flask import request, jsonify
from pydantic import ValidationError as PydanticValidationError
from utils.log_sanitizer import sanitize_log_data
import logging

logger = logging.getLogger(__name__)

def validate_request(schema_class=None):
    """
    Decorator to validate request with Pydantic schema
    If no schema provided, validates basic structure
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.method in ['POST', 'PUT', 'PATCH']:
                try:
                    if not request.json:
                        return jsonify({
                            'success': False,
                            'error': 'Request body is required'
                        }), 400

                    # If schema provided, validate
                    if schema_class:
                        validated_data = schema_class(**request.json)
                        # Store validated data in request context
                        request.validated_data = validated_data

                except PydanticValidationError as e:
                    logger.warning(
                        "Validation failed",
                        extra=sanitize_log_data({
                            'endpoint': request.path,
                            'errors': e.errors()
                        })
                    )
                    return jsonify({
                        'success': False,
                        'error': 'Validation failed',
                        'details': e.errors()
                    }), 400
                except Exception as e:
                    logger.error(
                        "Validation error",
                        extra=sanitize_log_data({
                            'endpoint': request.path,
                            'error': str(e)
                        })
                    )
                    return jsonify({
                        'success': False,
                        'error': 'Validation error'
                    }), 400

            return f(*args, **kwargs)
        return decorated_function
    return decorator
