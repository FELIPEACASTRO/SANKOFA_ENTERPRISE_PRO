"""
Sankofa Enterprise Pro - CORS Configuration
Production-ready CORS settings with domain whitelisting
"""

import os
from typing import List, Dict, Any

# Environment detection
ENVIRONMENT = os.getenv("FLASK_ENV", os.getenv("ENVIRONMENT", "development"))
IS_PRODUCTION = ENVIRONMENT == "production"

# Allowed origins by environment
ALLOWED_ORIGINS_PRODUCTION: List[str] = [
    # Production domains - customize for your deployment
    "https://sankofa.yourdomain.com",
    "https://api.sankofa.yourdomain.com",
    "https://dashboard.sankofa.yourdomain.com",
    # Add your production domains here
]

ALLOWED_ORIGINS_STAGING: List[str] = [
    "https://staging.sankofa.yourdomain.com",
    "https://staging-api.sankofa.yourdomain.com",
]

ALLOWED_ORIGINS_DEVELOPMENT: List[str] = [
    "http://localhost:3000",
    "http://localhost:5000",
    "http://localhost:5173",  # Common frontend dev servers (Vite, etc.)
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5000",
    "http://127.0.0.1:5173",
    # Replit and other development environments
    "https://*.replit.dev",
    "https://*.repl.co",
]

# Custom origins from environment variable (comma-separated)
CUSTOM_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
CUSTOM_ORIGINS = [origin.strip() for origin in CUSTOM_ORIGINS if origin.strip()]


def get_allowed_origins() -> List[str]:
    """
    Get allowed origins based on environment
    
    Returns:
        List of allowed origin URLs
    """
    origins = []
    
    if ENVIRONMENT == "production":
        origins = ALLOWED_ORIGINS_PRODUCTION.copy()
    elif ENVIRONMENT == "staging":
        origins = ALLOWED_ORIGINS_STAGING.copy()
    else:
        origins = ALLOWED_ORIGINS_DEVELOPMENT.copy()
    
    # Add custom origins from environment
    origins.extend(CUSTOM_ORIGINS)
    
    return list(set(origins))  # Remove duplicates


def get_cors_config() -> Dict[str, Any]:
    """
    Get CORS configuration for Flask-CORS
    
    Production settings are more restrictive:
    - Specific origins only (no wildcards)
    - Limited methods
    - Specific headers
    - Credentials disabled by default
    
    Development settings are more permissive for easier testing.
    
    Returns:
        Dictionary with CORS configuration
    """
    if IS_PRODUCTION:
        return {
            # Specific origins only in production
            "origins": get_allowed_origins(),
            
            # Allowed HTTP methods
            "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            
            # Allowed request headers
            "allow_headers": [
                "Content-Type",
                "Authorization",
                "X-Request-ID",
                "X-Requested-With",
                "Accept",
                "Accept-Language",
                "Content-Language",
            ],
            
            # Headers exposed to the browser
            "expose_headers": [
                "X-Request-ID",
                "X-Response-Time-Ms",
                "X-API-Version",
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset",
            ],
            
            # Whether to support credentials (cookies, authorization headers)
            "supports_credentials": False,
            
            # Preflight request cache time (in seconds)
            "max_age": 600,  # 10 minutes
            
            # Whether to vary the response based on Origin header
            "vary_header": True,
        }
    else:
        # Development/staging - more permissive
        return {
            # Allow all origins in development
            "origins": "*",
            
            # All methods
            "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
            
            # All headers
            "allow_headers": "*",
            
            # Expose all custom headers
            "expose_headers": [
                "X-Request-ID",
                "X-Response-Time-Ms",
                "X-API-Version",
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset",
                "Content-Disposition",
            ],
            
            # Support credentials in dev for easier testing
            "supports_credentials": True,
            
            # Short cache for faster development iteration
            "max_age": 60,
            
            "vary_header": True,
        }


def apply_cors(app):
    """
    Apply CORS configuration to Flask app
    
    Args:
        app: Flask application instance
    
    Returns:
        Configured Flask app with CORS
    """
    from flask_cors import CORS
    
    cors_config = get_cors_config()
    
    # Apply CORS globally
    CORS(app, resources={
        r"/api/*": cors_config,
        r"/docs/*": cors_config,
    })
    
    # Log configuration in development
    if not IS_PRODUCTION:
        print(f"[CORS] Environment: {ENVIRONMENT}")
        print(f"[CORS] Allowed origins: {cors_config.get('origins', 'all')}")
    
    return app


# Example usage in production_api.py:
#
# from config.cors_config import apply_cors
#
# app = Flask(__name__)
# apply_cors(app)  # Instead of CORS(app)
