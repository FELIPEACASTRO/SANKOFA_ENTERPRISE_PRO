"""
Sankofa Enterprise Pro - Security Headers Middleware
Implements OWASP security headers best practices

V004 FIX: CSP sem unsafe-inline/unsafe-eval em produção
"""

import os
import secrets
from flask import Flask, Response, request, g
from typing import Optional


class SecurityHeadersMiddleware:
    """
    Adds security headers to all HTTP responses

    Headers implemented:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Strict-Transport-Security (HSTS)
    - Content-Security-Policy (CSP) - V004 FIX: Sem unsafe-inline em produção
    - Referrer-Policy
    - Permissions-Policy
    """

    def __init__(self, app: Optional[Flask] = None):
        self.app = app
        self._is_production = os.environ.get("ENVIRONMENT", "development") == "production"
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask):
        """Initialize middleware with Flask app"""
        # V004 FIX: Gerar nonce antes de cada request para CSP
        app.before_request(self._generate_nonce)
        app.after_request(self.add_security_headers)

    @staticmethod
    def _generate_nonce():
        """
        V004 FIX: Gera nonce único para cada request

        Nonces permitem scripts inline específicos sem usar unsafe-inline
        """
        g.csp_nonce = secrets.token_urlsafe(16)

    def add_security_headers(self, response: Response) -> Response:
        """
        Add security headers to response

        V004 FIX: CSP com nonce em vez de unsafe-inline em produção

        Args:
            response: Flask response object

        Returns:
            Response with security headers added
        """
        # Prevent MIME type sniffing
        response.headers['X-Content-Type-Options'] = 'nosniff'

        # Prevent clickjacking
        response.headers['X-Frame-Options'] = 'DENY'

        # Enable XSS filter (legacy browsers)
        response.headers['X-XSS-Protection'] = '1; mode=block'

        # Force HTTPS (1 year)
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'

        # V004 FIX: Content Security Policy - Restritiva em produção
        nonce = getattr(g, 'csp_nonce', secrets.token_urlsafe(16))

        if self._is_production:
            # PRODUÇÃO: CSP restritiva sem unsafe-inline/unsafe-eval
            csp_directives = [
                "default-src 'self'",
                f"script-src 'self' 'nonce-{nonce}'",  # V004 FIX: Usa nonce em vez de unsafe-inline
                f"style-src 'self' 'nonce-{nonce}'",   # V004 FIX: Usa nonce em vez de unsafe-inline
                "img-src 'self' data: https:",
                "font-src 'self' data:",
                "connect-src 'self'",
                "frame-ancestors 'none'",
                "base-uri 'self'",
                "form-action 'self'",
                "object-src 'none'",
                "script-src-attr 'none'",
                "upgrade-insecure-requests",
            ]
        else:
            # DESENVOLVIMENTO: CSP permissiva para facilitar debug
            # NOTA: Isso é aceitável apenas em ambiente de desenvolvimento
            csp_directives = [
                "default-src 'self'",
                f"script-src 'self' 'unsafe-inline' 'unsafe-eval' 'nonce-{nonce}'",
                f"style-src 'self' 'unsafe-inline' 'nonce-{nonce}'",
                "img-src 'self' data: https: blob:",
                "font-src 'self' data:",
                "connect-src 'self' ws: wss:",  # WebSocket para HMR
                "frame-ancestors 'none'",
                "base-uri 'self'",
                "form-action 'self'",
            ]

        response.headers['Content-Security-Policy'] = "; ".join(csp_directives)

        # Expor nonce para o frontend poder usar scripts inline autorizados
        response.headers['X-CSP-Nonce'] = nonce

        # Referrer policy - protect privacy
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

        # Permissions policy (formerly Feature-Policy)
        permissions_directives = [
            "geolocation=()",
            "microphone=()",
            "camera=()",
            "payment=()",
            "usb=()",
            "magnetometer=()",
            "gyroscope=()",
            "accelerometer=()",
            "autoplay=()",
            "fullscreen=(self)",
        ]
        response.headers['Permissions-Policy'] = ", ".join(permissions_directives)

        # Remove headers que podem facilitar fingerprinting
        response.headers.pop('X-Powered-By', None)
        response.headers.pop('Server', None)

        # Cache control para API
        if request.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'

        return response


def configure_security_headers(app: Flask):
    """
    Convenience function to apply security headers to Flask app

    V004 FIX: CSP restritiva em produção (sem unsafe-inline)

    Usage:
        from api.middleware.security_headers import configure_security_headers

        app = Flask(__name__)
        configure_security_headers(app)

    Args:
        app: Flask application instance
    """
    SecurityHeadersMiddleware(app)
    return app


def get_csp_nonce() -> str:
    """
    V004 FIX: Retorna o nonce CSP atual para usar em scripts inline

    Usage no template Jinja2:
        <script nonce="{{ csp_nonce }}">
            // Este script será permitido pela CSP
        </script>
    """
    return getattr(g, 'csp_nonce', '')
