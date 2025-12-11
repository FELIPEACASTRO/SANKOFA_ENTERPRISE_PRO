"""
OpenAPI 3.0 Specification for Sankofa Enterprise Pro API
Provides automatic API documentation with Swagger UI
"""

from flask import Flask
from flasgger import Swagger, swag_from

# OpenAPI 3.0 Configuration
SWAGGER_CONFIG = {
    "openapi": "3.0.0",
    "title": "Sankofa Enterprise Pro API",
    "description": """
    ## Production-Grade Fraud Detection API

    Sistema de detecção de fraude em tempo real usando ML avançado.

    ### Features:
    - **Real-time Fraud Detection**: Análise em <50ms (P95)
    - **Explainability**: LGPD Art. 20 compliant explanations
    - **RBAC**: Role-based access control
    - **Rate Limiting**: Protection against abuse
    - **Compliance**: LGPD, BACEN, PCI-DSS

    ### Authentication:
    Use JWT tokens in `Authorization: Bearer <token>` header.

    ### Rate Limits:
    - Default: 100 requests/minute
    - Prediction: 500 requests/minute
    - Auth: 10 requests/minute
    """,
    "version": "1.0.0",
    "contact": {
        "name": "Sankofa Support",
        "email": "support@sankofa.com",
        "url": "https://sankofa.com/support"
    },
    "license": {
        "name": "Proprietary",
        "url": "https://sankofa.com/license"
    },
    "servers": [
        {
            "url": "http://localhost:5000",
            "description": "Development server"
        },
        {
            "url": "https://api-staging.sankofa.com",
            "description": "Staging server"
        },
        {
            "url": "https://api.sankofa.com",
            "description": "Production server"
        }
    ],
    "components": {
        "securitySchemes": {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT token obtained from /api/v1/auth/login"
            }
        },
        "schemas": {
            "Transaction": {
                "type": "object",
                "required": ["amount", "cpf", "channel"],
                "properties": {
                    "amount": {
                        "type": "number",
                        "format": "decimal",
                        "example": 1000.50,
                        "description": "Transaction amount in BRL"
                    },
                    "cpf": {
                        "type": "string",
                        "pattern": "^[0-9]{11}$",
                        "example": "12345678901",
                        "description": "Customer CPF (11 digits)"
                    },
                    "channel": {
                        "type": "string",
                        "enum": ["PIX", "TED", "DOC", "BOLETO", "CARTAO"],
                        "example": "PIX",
                        "description": "Payment channel"
                    },
                    "merchant_id": {
                        "type": "string",
                        "example": "MERCHANT_123",
                        "description": "Merchant identifier"
                    },
                    "customer_id": {
                        "type": "string",
                        "example": "CUSTOMER_456",
                        "description": "Customer identifier"
                    }
                }
            },
            "FraudPrediction": {
                "type": "object",
                "properties": {
                    "is_fraud": {
                        "type": "boolean",
                        "example": False,
                        "description": "Whether transaction is fraudulent"
                    },
                    "fraud_probability": {
                        "type": "number",
                        "format": "float",
                        "example": 0.15,
                        "description": "Probability of fraud (0-1)"
                    },
                    "risk_score": {
                        "type": "number",
                        "format": "float",
                        "example": 0.23,
                        "description": "Overall risk score (0-1)"
                    },
                    "decision": {
                        "type": "string",
                        "enum": ["APPROVE", "DENY", "MANUAL_REVIEW"],
                        "example": "APPROVE",
                        "description": "Recommended decision"
                    },
                    "explanation": {
                        "type": "object",
                        "description": "LGPD Art. 20 compliant explanation",
                        "properties": {
                            "top_features": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "feature": {"type": "string"},
                                        "importance": {"type": "number"},
                                        "value": {"type": "number"}
                                    }
                                }
                            },
                            "human_readable": {
                                "type": "string",
                                "example": "Transação aprovada: valor normal para este cliente"
                            }
                        }
                    },
                    "latency_ms": {
                        "type": "number",
                        "example": 37.5,
                        "description": "Processing latency in milliseconds"
                    }
                }
            },
            "Error": {
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "example": False
                    },
                    "error": {
                        "type": "string",
                        "example": "Validation failed"
                    },
                    "details": {
                        "type": "object",
                        "description": "Detailed error information"
                    },
                    "status_code": {
                        "type": "integer",
                        "example": 400
                    }
                }
            }
        },
        "responses": {
            "UnauthorizedError": {
                "description": "Authentication required",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "ForbiddenError": {
                "description": "Insufficient permissions",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "ValidationError": {
                "description": "Input validation failed",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "RateLimitError": {
                "description": "Rate limit exceeded",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            }
        }
    },
    "security": [
        {"BearerAuth": []}
    ],
    "tags": [
        {
            "name": "Fraud Detection",
            "description": "Real-time fraud prediction endpoints"
        },
        {
            "name": "Authentication",
            "description": "User authentication and authorization"
        },
        {
            "name": "Transactions",
            "description": "Transaction management"
        },
        {
            "name": "Dashboard",
            "description": "KPIs and analytics"
        },
        {
            "name": "Admin",
            "description": "Administrative endpoints (admin only)"
        },
        {
            "name": "Compliance",
            "description": "LGPD Data Subject Rights"
        },
        {
            "name": "Health",
            "description": "Health checks and observability"
        }
    ]
}


def init_swagger(app: Flask):
    """Initialize Swagger/OpenAPI documentation"""
    swagger = Swagger(app, config=SWAGGER_CONFIG, template={
        "swagger": "2.0",
        "info": {
            "title": SWAGGER_CONFIG["title"],
            "description": SWAGGER_CONFIG["description"],
            "version": SWAGGER_CONFIG["version"],
            "contact": SWAGGER_CONFIG["contact"],
            "license": SWAGGER_CONFIG["license"]
        },
        "basePath": "/api/v1",
        "schemes": ["http", "https"],
        "securityDefinitions": {
            "Bearer": {
                "type": "apiKey",
                "name": "Authorization",
                "in": "header",
                "description": "JWT Authorization header using the Bearer scheme. Example: \"Authorization: Bearer {token}\""
            }
        }
    })
    return swagger


# Endpoint documentation decorators
PREDICT_DOCS = {
    "tags": ["Fraud Detection"],
    "summary": "Predict fraud for a transaction",
    "description": "Analyzes a single transaction and returns fraud prediction with explanation",
    "parameters": [
        {
            "name": "body",
            "in": "body",
            "required": True,
            "schema": {
                "$ref": "#/components/schemas/Transaction"
            }
        }
    ],
    "responses": {
        "200": {
            "description": "Successful prediction",
            "schema": {
                "$ref": "#/components/schemas/FraudPrediction"
            }
        },
        "400": {
            "$ref": "#/components/responses/ValidationError"
        },
        "401": {
            "$ref": "#/components/responses/UnauthorizedError"
        },
        "429": {
            "$ref": "#/components/responses/RateLimitError"
        }
    },
    "security": [
        {"Bearer": []}
    ]
}

BATCH_PREDICT_DOCS = {
    "tags": ["Fraud Detection"],
    "summary": "Batch fraud prediction",
    "description": "Analyzes multiple transactions in a single request (up to 100)",
    "parameters": [
        {
            "name": "body",
            "in": "body",
            "required": True,
            "schema": {
                "type": "object",
                "properties": {
                    "transactions": {
                        "type": "array",
                        "items": {
                            "$ref": "#/components/schemas/Transaction"
                        },
                        "maxItems": 100
                    }
                }
            }
        }
    ],
    "responses": {
        "200": {
            "description": "Batch prediction results",
            "schema": {
                "type": "object",
                "properties": {
                    "predictions": {
                        "type": "array",
                        "items": {
                            "$ref": "#/components/schemas/FraudPrediction"
                        }
                    },
                    "total_processed": {"type": "integer"},
                    "batch_latency_ms": {"type": "number"}
                }
            }
        }
    }
}

HEALTH_DOCS = {
    "tags": ["Health"],
    "summary": "Health check",
    "description": "Returns system health status",
    "responses": {
        "200": {
            "description": "System is healthy",
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "example": "healthy"},
                    "timestamp": {"type": "string"},
                    "version": {"type": "string"},
                    "dependencies": {
                        "type": "object",
                        "properties": {
                            "database": {"type": "string"},
                            "redis": {"type": "string"},
                            "ml_model": {"type": "string"}
                        }
                    }
                }
            }
        },
        "503": {
            "description": "System is unhealthy"
        }
    }
}
