"""
Flask-Talisman Configuration
Implements security headers and HTTPS enforcement
"""

from flask import Flask
from flask_talisman import Talisman
import os


def init_talisman(app: Flask) -> Talisman:
    """
    Initialize Flask-Talisman with production-grade security headers

    Security headers configured:
    - HTTPS enforcement (production only)
    - HSTS (HTTP Strict Transport Security)
    - CSP (Content Security Policy)
    - X-Frame-Options (Clickjacking protection)
    - X-Content-Type-Options (MIME sniffing protection)
    - X-XSS-Protection

    Args:
        app: Flask application instance

    Returns:
        Configured Talisman instance
    """

    # Environment detection
    is_production = os.getenv('FLASK_ENV') == 'production'
    is_development = os.getenv('FLASK_ENV') == 'development'

    # Content Security Policy
    csp = {
        'default-src': [
            "'self'",
        ],
        'script-src': [
            "'self'",
            "'unsafe-inline'",  # Required for some inline scripts (minimize in production)
            'cdn.jsdelivr.net',
            'unpkg.com',
        ],
        'style-src': [
            "'self'",
            "'unsafe-inline'",  # Required for inline styles
            'fonts.googleapis.com',
            'cdn.jsdelivr.net',
        ],
        'font-src': [
            "'self'",
            'fonts.gstatic.com',
            'data:',
        ],
        'img-src': [
            "'self'",
            'data:',
            'https:',
        ],
        'connect-src': [
            "'self'",
            'api.sankofa.com',
            'api-staging.sankofa.com',
        ],
        'frame-ancestors': [
            "'none'",  # Prevent framing (clickjacking protection)
        ],
        'base-uri': [
            "'self'",
        ],
        'form-action': [
            "'self'",
        ],
    }

    # Feature Policy / Permissions Policy
    feature_policy = {
        'geolocation': "'none'",
        'microphone': "'none'",
        'camera': "'none'",
        'payment': "'self'",
        'usb': "'none'",
    }

    # HSTS configuration (HTTP Strict Transport Security)
    # Tells browsers to always use HTTPS for this domain
    force_https = is_production
    force_https_permanent = is_production

    # Session cookie security
    session_cookie_secure = is_production
    session_cookie_http_only = True

    # Initialize Talisman
    talisman = Talisman(
        app,
        # HTTPS enforcement
        force_https=force_https,
        force_https_permanent=force_https_permanent,

        # HSTS settings
        strict_transport_security=True,
        strict_transport_security_max_age=31536000,  # 1 year
        strict_transport_security_include_subdomains=True,
        strict_transport_security_preload=is_production,

        # Content Security Policy
        content_security_policy=csp,
        content_security_policy_report_only=is_development,
        content_security_policy_report_uri='/api/v1/csp-report',

        # Feature Policy (Permissions Policy)
        feature_policy=feature_policy,

        # Referrer Policy
        referrer_policy='strict-origin-when-cross-origin',

        # Session cookies
        session_cookie_secure=session_cookie_secure,
        session_cookie_http_only=session_cookie_http_only,
        session_cookie_samesite='Lax',

        # X-Frame-Options (prevent clickjacking)
        frame_options='DENY',
        frame_options_allow_from=None,

        # X-Content-Type-Options (prevent MIME sniffing)
        content_type_options=True,
        content_type_options_value='nosniff',

        # X-XSS-Protection (legacy, but still good practice)
        x_xss_protection=True,
    )

    # Exempt health check endpoints from HTTPS redirect (for load balancers)
    talisman.force_https_exempt = [
        '/health',
        '/readiness',
        '/liveness',
        '/metrics',  # Prometheus scraper
    ]

    app.logger.info(
        "Flask-Talisman initialized",
        extra={
            'force_https': force_https,
            'csp_enabled': True,
            'hsts_enabled': True,
            'environment': os.getenv('FLASK_ENV', 'development')
        }
    )

    return talisman


def add_custom_security_headers(response):
    """
    Add additional custom security headers not covered by Talisman

    Usage:
        app.after_request(add_custom_security_headers)
    """
    # Permissions Policy (modern Feature-Policy)
    response.headers['Permissions-Policy'] = (
        'geolocation=(), microphone=(), camera=(), payment=(self), usb=()'
    )

    # Cross-Origin Resource Policy
    response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'

    # Cross-Origin Embedder Policy
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'

    # Cross-Origin Opener Policy
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'

    # X-DNS-Prefetch-Control
    response.headers['X-DNS-Prefetch-Control'] = 'off'

    # X-Download-Options (IE specific)
    response.headers['X-Download-Options'] = 'noopen'

    # X-Permitted-Cross-Domain-Policies (Adobe)
    response.headers['X-Permitted-Cross-Domain-Policies'] = 'none'

    # Remove server header (information disclosure)
    response.headers.pop('Server', None)

    # Custom security header with version
    response.headers['X-Sankofa-Security-Version'] = '1.0'

    return response


def init_security_headers(app: Flask):
    """
    Initialize all security headers for Flask app

    This is the main function to call during app initialization.

    Usage:
        from api.middleware.talisman_config import init_security_headers
        init_security_headers(app)
    """
    # Initialize Talisman
    talisman = init_talisman(app)

    # Add custom headers
    app.after_request(add_custom_security_headers)

    return talisman
