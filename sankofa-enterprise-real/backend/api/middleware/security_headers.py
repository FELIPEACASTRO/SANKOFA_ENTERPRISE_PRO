"""
Sankofa Enterprise Pro - Security Headers Middleware
Implements OWASP security headers best practices
"""

from flask import Flask, Response
from typing import Optional


class SecurityHeadersMiddleware:
    """
    Adds security headers to all HTTP responses

    Headers implemented:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Strict-Transport-Security (HSTS)
    - Content-Security-Policy (CSP)
    - Referrer-Policy
    - Permissions-Policy
    """

    def __init__(self, app: Optional[Flask] = None):
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask):
        """Initialize middleware with Flask app"""
        app.after_request(self.add_security_headers)

    @staticmethod
    def add_security_headers(response: Response) -> Response:
        """
        Add security headers to response

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

        # Content Security Policy
        # Restrictive policy - adjust as needed for your frontend
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # TODO: Remove unsafe-* in production
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self' data:",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        ]
        response.headers['Content-Security-Policy'] = "; ".join(csp_directives)

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
        ]
        response.headers['Permissions-Policy'] = ", ".join(permissions_directives)

        # X-Powered-By - remove to prevent fingerprinting
        response.headers.pop('X-Powered-By', None)
        response.headers.pop('Server', None)

        return response


def configure_security_headers(app: Flask):
    """
    Convenience function to apply security headers to Flask app

    Usage:
        from api.middleware.security_headers import configure_security_headers

        app = Flask(__name__)
        configure_security_headers(app)

    Args:
        app: Flask application instance
    """
    SecurityHeadersMiddleware(app)
    return app
