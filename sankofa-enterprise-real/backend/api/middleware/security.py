"""
Security Middleware
Implementa CSRF protection, security headers, e proteções contra ataques comuns
Compliance com OWASP Top 10
"""

import secrets
import hashlib
import time
from functools import wraps
from typing import Optional, Dict, Any, Callable
from flask import request, g, jsonify, session, Response
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# SECURITY HEADERS MIDDLEWARE
# ============================================================================

class SecurityHeadersMiddleware:
    """
    Adiciona security headers a todas as respostas
    Proteção contra: XSS, Clickjacking, MIME sniffing, etc.
    """

    # Headers de segurança recomendados pela OWASP
    SECURITY_HEADERS = {
        # HSTS - Force HTTPS
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',

        # Previne MIME sniffing
        'X-Content-Type-Options': 'nosniff',

        # Previne Clickjacking
        'X-Frame-Options': 'DENY',

        # XSS Protection (legacy, mas ainda útil para browsers antigos)
        'X-XSS-Protection': '1; mode=block',

        # Content Security Policy - Previne XSS e data injection
        'Content-Security-Policy': (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        ),

        # Referrer Policy - Controla informação de referrer
        'Referrer-Policy': 'strict-origin-when-cross-origin',

        # Permissions Policy - Controla features do browser
        'Permissions-Policy': (
            'geolocation=(), '
            'microphone=(), '
            'camera=(), '
            'payment=(), '
            'usb=(), '
            'magnetometer=(), '
            'gyroscope=(), '
            'accelerometer=()'
        ),

        # Remove header que expõe versão do servidor
        'Server': 'Sankofa-API',

        # Cache control para dados sensíveis
        'Cache-Control': 'no-store, no-cache, must-revalidate, private',
        'Pragma': 'no-cache',
        'Expires': '0',
    }

    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        """Registra middleware no Flask app"""
        app.after_request(self.add_security_headers)

    def add_security_headers(self, response: Response) -> Response:
        """
        Adiciona headers de segurança a todas as respostas

        Args:
            response: Flask Response object

        Returns:
            Response com headers de segurança
        """
        for header, value in self.SECURITY_HEADERS.items():
            response.headers[header] = value

        # CORS headers (se necessário)
        # Apenas para endpoints específicos, não usar '*'
        if request.path.startswith('/api/'):
            origin = request.headers.get('Origin')
            # Whitelist de origins permitidos
            allowed_origins = [
                'http://localhost:3000',
                'http://localhost:5000',
                'https://sankofa.example.com',  # Substituir por domínio real
            ]

            if origin in allowed_origins:
                response.headers['Access-Control-Allow-Origin'] = origin
                response.headers['Access-Control-Allow-Credentials'] = 'true'
                response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-CSRF-Token'

        return response


# ============================================================================
# CSRF PROTECTION
# ============================================================================

class CSRFProtection:
    """
    Implementa proteção CSRF (Cross-Site Request Forgery)
    Baseado em Double Submit Cookie pattern
    """

    def __init__(self, app=None, exempt_methods=None):
        self.app = app
        self.exempt_methods = exempt_methods or {'GET', 'HEAD', 'OPTIONS'}
        self.exempt_endpoints = set()

        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        """Registra CSRF protection no Flask app"""
        app.before_request(self.protect)

        # Adiciona função para gerar token
        @app.route('/api/csrf-token', methods=['GET'])
        def get_csrf_token():
            token = self.generate_csrf_token()
            return jsonify({'csrf_token': token})

    def generate_csrf_token(self) -> str:
        """
        Gera token CSRF criptograficamente seguro

        Returns:
            Token CSRF
        """
        if 'csrf_token' not in session:
            session['csrf_token'] = secrets.token_hex(32)

        return session['csrf_token']

    def validate_csrf_token(self, token: str) -> bool:
        """
        Valida token CSRF

        Args:
            token: Token fornecido no request

        Returns:
            True se válido, False caso contrário
        """
        if 'csrf_token' not in session:
            return False

        # Constant-time comparison para prevenir timing attacks
        expected = session['csrf_token']
        return secrets.compare_digest(expected, token)

    def exempt(self, view_func: Callable) -> Callable:
        """
        Decorator para isentar endpoint de CSRF protection

        Args:
            view_func: View function a isentar

        Returns:
            Decorated function

        Example:
            @app.route('/api/webhook', methods=['POST'])
            @csrf.exempt
            def webhook():
                return 'OK'
        """
        endpoint = view_func.__name__
        self.exempt_endpoints.add(endpoint)
        return view_func

    def protect(self):
        """
        Middleware que valida CSRF token em requests não-GET

        Raises:
            403 se token inválido
        """
        # Skip se método é safe (GET, HEAD, OPTIONS)
        if request.method in self.exempt_methods:
            return

        # Skip se endpoint está isento
        if request.endpoint in self.exempt_endpoints:
            return

        # Skip se é API com autenticação JWT (usa outro mecanismo)
        # Mas ainda requer token para requests de browsers
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer ') and not self._is_browser_request():
            return

        # Valida token
        token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')

        if not token or not self.validate_csrf_token(token):
            logger.warning(
                'CSRF token validation failed',
                extra={
                    'ip': request.remote_addr,
                    'endpoint': request.endpoint,
                    'method': request.method,
                }
            )

            return jsonify({
                'success': False,
                'error': 'CSRF token missing or invalid',
                'code': 'CSRF_ERROR'
            }), 403

    def _is_browser_request(self) -> bool:
        """
        Detecta se request vem de browser (vs API client)

        Returns:
            True se é browser
        """
        user_agent = request.headers.get('User-Agent', '').lower()
        browser_indicators = ['mozilla', 'chrome', 'safari', 'edge', 'opera']
        return any(indicator in user_agent for indicator in browser_indicators)


# ============================================================================
# RATE LIMITING AVANÇADO
# ============================================================================

class AdvancedRateLimiter:
    """
    Rate limiting com backoff progressivo e detecção de brute force
    """

    def __init__(self, app=None):
        self.app = app
        self.attempts = {}  # {ip: [(timestamp, endpoint), ...]}
        self.blocked_ips = {}  # {ip: blocked_until_timestamp}

        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        """Registra rate limiter no Flask app"""
        app.before_request(self.check_rate_limit)

        # Cleanup task periódico
        @app.before_first_request
        def start_cleanup():
            # TODO: Implementar cleanup com APScheduler ou Celery
            pass

    def check_rate_limit(self):
        """
        Verifica rate limit antes de processar request

        Returns:
            429 se limit excedido
        """
        ip = self._get_client_ip()
        endpoint = request.endpoint or 'unknown'

        # Verifica se IP está bloqueado
        if ip in self.blocked_ips:
            if time.time() < self.blocked_ips[ip]:
                return jsonify({
                    'success': False,
                    'error': 'Too many requests. IP temporarily blocked.',
                    'code': 'RATE_LIMIT_EXCEEDED',
                    'retry_after': int(self.blocked_ips[ip] - time.time())
                }), 429
            else:
                # Desbloqueio
                del self.blocked_ips[ip]

        # Rate limits específicos por endpoint
        limits = self._get_endpoint_limits(endpoint)

        # Registra tentativa
        now = time.time()
        if ip not in self.attempts:
            self.attempts[ip] = []

        self.attempts[ip].append((now, endpoint))

        # Remove tentativas antigas (> 1 hora)
        self.attempts[ip] = [
            (ts, ep) for ts, ep in self.attempts[ip]
            if now - ts < 3600
        ]

        # Verifica limits
        for window, max_requests in limits.items():
            recent_attempts = [
                ts for ts, ep in self.attempts[ip]
                if now - ts < window and ep == endpoint
            ]

            if len(recent_attempts) > max_requests:
                # Bloqueia IP
                block_duration = self._calculate_block_duration(ip, endpoint)
                self.blocked_ips[ip] = now + block_duration

                logger.warning(
                    'Rate limit exceeded - IP blocked',
                    extra={
                        'ip': ip,
                        'endpoint': endpoint,
                        'attempts': len(recent_attempts),
                        'block_duration': block_duration
                    }
                )

                return jsonify({
                    'success': False,
                    'error': f'Rate limit exceeded. Blocked for {block_duration} seconds.',
                    'code': 'RATE_LIMIT_EXCEEDED',
                    'retry_after': block_duration
                }), 429

        return None

    def _get_endpoint_limits(self, endpoint: str) -> Dict[int, int]:
        """
        Retorna limites para endpoint específico

        Args:
            endpoint: Nome do endpoint

        Returns:
            Dict {window_seconds: max_requests}
        """
        # Limits rigorosos para endpoints sensíveis
        sensitive_limits = {
            60: 5,      # 5 req/min
            3600: 50,   # 50 req/hora
        }

        # Limits normais para endpoints regulares
        normal_limits = {
            60: 100,     # 100 req/min
            3600: 1000,  # 1000 req/hora
        }

        # Endpoints sensíveis (autenticação, alteração de dados)
        if endpoint and any(x in endpoint for x in ['login', 'auth', 'password', 'update', 'delete', 'create']):
            return sensitive_limits

        return normal_limits

    def _calculate_block_duration(self, ip: str, endpoint: str) -> int:
        """
        Calcula duração do bloqueio com backoff progressivo

        Args:
            ip: IP do cliente
            endpoint: Endpoint acessado

        Returns:
            Duração em segundos
        """
        # Conta quantas vezes IP foi bloqueado
        # TODO: Persistir em Redis para distribuição
        block_count = 1  # Simplificado

        # Backoff exponencial: 5min, 15min, 1h, 6h, 24h
        durations = [300, 900, 3600, 21600, 86400]
        index = min(block_count - 1, len(durations) - 1)

        return durations[index]

    def _get_client_ip(self) -> str:
        """
        Obtém IP real do cliente (considerando proxies)

        Returns:
            IP do cliente
        """
        # Verifica headers de proxy
        if request.headers.get('X-Forwarded-For'):
            return request.headers.get('X-Forwarded-For').split(',')[0].strip()
        elif request.headers.get('X-Real-IP'):
            return request.headers.get('X-Real-IP')

        return request.remote_addr or '0.0.0.0'


# ============================================================================
# INPUT SANITIZATION
# ============================================================================

def sanitize_input(data: Any) -> Any:
    """
    Sanitiza input para prevenir XSS e injection attacks

    Args:
        data: Dados a sanitizar

    Returns:
        Dados sanitizados
    """
    if isinstance(data, str):
        # Remove scripts e HTML perigoso
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe',
            r'<object',
            r'<embed',
        ]

        sanitized = data
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)

        return sanitized

    elif isinstance(data, dict):
        return {k: sanitize_input(v) for k, v in data.items()}

    elif isinstance(data, list):
        return [sanitize_input(item) for item in data]

    return data


# ============================================================================
# REQUEST VALIDATION
# ============================================================================

def require_https():
    """
    Decorator que força HTTPS em produção

    Raises:
        403 se não HTTPS
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from config.settings import get_config
            config = get_config()

            if config.environment == 'production' and not request.is_secure:
                return jsonify({
                    'success': False,
                    'error': 'HTTPS required',
                    'code': 'HTTPS_REQUIRED'
                }), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def require_content_type(content_type: str = 'application/json'):
    """
    Decorator que valida Content-Type do request

    Args:
        content_type: Content-Type esperado

    Raises:
        415 se Content-Type inválido
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.method in ['POST', 'PUT', 'PATCH']:
                if content_type not in request.headers.get('Content-Type', ''):
                    return jsonify({
                        'success': False,
                        'error': f'Content-Type must be {content_type}',
                        'code': 'INVALID_CONTENT_TYPE'
                    }), 415

            return f(*args, **kwargs)
        return decorated_function
    return decorator


# ============================================================================
# IP WHITELIST/BLACKLIST
# ============================================================================

class IPFilter:
    """
    Filtra requests por IP (whitelist/blacklist)
    """

    def __init__(self, whitelist=None, blacklist=None):
        self.whitelist = set(whitelist or [])
        self.blacklist = set(blacklist or [])

    def is_allowed(self, ip: str) -> bool:
        """
        Verifica se IP é permitido

        Args:
            ip: IP a verificar

        Returns:
            True se permitido
        """
        # Se há whitelist, IP deve estar nela
        if self.whitelist:
            return ip in self.whitelist

        # Se não há whitelist, verifica blacklist
        return ip not in self.blacklist

    def middleware(self, f):
        """
        Decorator para filtrar por IP

        Example:
            ip_filter = IPFilter(blacklist=['1.2.3.4'])

            @app.route('/api/sensitive')
            @ip_filter.middleware
            def sensitive_endpoint():
                return 'OK'
        """
        @wraps(f)
        def decorated_function(*args, **kwargs):
            ip = request.remote_addr

            if not self.is_allowed(ip):
                logger.warning(f'Blocked request from blacklisted IP: {ip}')
                return jsonify({
                    'success': False,
                    'error': 'Access denied',
                    'code': 'IP_BLOCKED'
                }), 403

            return f(*args, **kwargs)
        return decorated_function
