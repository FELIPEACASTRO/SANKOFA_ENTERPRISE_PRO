"""
Production Security Infrastructure
Implements enterprise-grade security with encryption, authentication, and audit
"""

import hashlib
import hmac
import secrets
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import logging
import asyncio
from dataclasses import dataclass
import re
import ipaddress
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class SecurityContext:
    """Security context for authenticated requests"""

    user_id: str
    api_key_id: str
    permissions: List[str]
    rate_limit: int
    ip_address: str
    user_agent: str
    authenticated_at: datetime


class EncryptionService:
    """Production-grade encryption service"""

    def __init__(self, master_key: str):
        self.master_key = master_key.encode()
        self._fernet = self._create_fernet()

    def _create_fernet(self) -> Fernet:
        """Create Fernet cipher from master key"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"sankofa_salt_2024",  # In production, use random salt per encryption
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        return Fernet(key)

    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted = self._fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self._fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

    def hash_pii(self, pii_data: str) -> str:
        """Hash PII data for privacy compliance (LGPD)"""
        # Use SHA-256 with salt for irreversible hashing
        salt = b"sankofa_pii_salt_2024"
        return hashlib.sha256(salt + pii_data.encode()).hexdigest()


class AuthenticationService:
    """Production authentication service"""

    def __init__(self, jwt_secret: str, db_manager):
        self.jwt_secret = jwt_secret
        self.db_manager = db_manager
        self.algorithm = "HS256"
        self.token_expiry = timedelta(hours=8)

    def generate_api_key(self) -> tuple[str, str]:
        """Generate API key and its hash"""
        api_key = f"sk_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_api_key(api_key)
        return api_key, key_hash

    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()

    async def create_api_key(
        self,
        name: str,
        permissions: List[str],
        rate_limit: int = 1000,
        expires_at: Optional[datetime] = None,
    ) -> str:
        """Create new API key"""
        api_key, key_hash = self.generate_api_key()

        async with self.db_manager.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO api_keys (key_hash, name, permissions, rate_limit, expires_at)
                VALUES ($1, $2, $3, $4, $5)
            """,
                key_hash,
                name,
                permissions,
                rate_limit,
                expires_at,
            )

        logger.info(f"API key created: {name}")
        return api_key

    async def authenticate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Authenticate API key"""
        try:
            key_hash = self._hash_api_key(api_key)

            async with self.db_manager.get_connection() as conn:
                result = await conn.fetchrow(
                    """
                    SELECT id, name, permissions, rate_limit, expires_at, is_active
                    FROM api_keys 
                    WHERE key_hash = $1 AND is_active = true
                """,
                    key_hash,
                )

                if not result:
                    return None

                # Check expiration
                if result["expires_at"] and result["expires_at"] < datetime.utcnow():
                    return None

                # Update last used
                await conn.execute(
                    "UPDATE api_keys SET last_used_at = CURRENT_TIMESTAMP WHERE key_hash = $1",
                    key_hash,
                )

                return {
                    "id": result["id"],
                    "name": result["name"],
                    "permissions": result["permissions"],
                    "rate_limit": result["rate_limit"],
                }

        except Exception as e:
            logger.error(f"API key authentication failed: {e}")
            return None

    def generate_jwt_token(self, user_id: str, permissions: List[str]) -> str:
        """Generate JWT token"""
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + self.token_expiry,
        }

        return jwt.encode(payload, self.jwt_secret, algorithm=self.algorithm)

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None


class RateLimitService:
    """Production rate limiting service"""

    def __init__(self, redis_manager):
        self.redis_manager = redis_manager
        self.window_size = 60  # 1 minute windows

    async def check_rate_limit(
        self, identifier: str, limit: int, window_seconds: int = 60
    ) -> tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limit"""
        try:
            current_time = datetime.utcnow()
            window_start = current_time.replace(second=0, microsecond=0)

            # Redis key for this window
            key = f"rate_limit:{identifier}:{window_start.timestamp()}"

            # Get current count
            current_count = await self.redis_manager.incr(key, 1)

            if current_count == 1:
                # First request in this window, set expiry
                await self.redis_manager.expire(key, window_seconds)

            # Check if over limit
            is_allowed = current_count <= limit

            # Calculate reset time
            reset_time = window_start + timedelta(seconds=window_seconds)

            return is_allowed, {
                "limit": limit,
                "remaining": max(0, limit - current_count),
                "reset_time": reset_time.isoformat(),
                "current_count": current_count,
            }

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if rate limiting fails
            return True, {"limit": limit, "remaining": limit}

    async def get_rate_limit_info(self, identifier: str) -> Dict[str, Any]:
        """Get current rate limit status"""
        try:
            current_time = datetime.utcnow()
            window_start = current_time.replace(second=0, microsecond=0)
            key = f"rate_limit:{identifier}:{window_start.timestamp()}"

            current_count = await self.redis_manager.get(key)
            current_count = int(current_count) if current_count else 0

            return {"current_count": current_count, "window_start": window_start.isoformat()}
        except Exception as e:
            logger.error(f"Rate limit info failed: {e}")
            return {"current_count": 0}


class AuditService:
    """Production audit service for compliance"""

    def __init__(self, db_manager, encryption_service: EncryptionService):
        self.db_manager = db_manager
        self.encryption_service = encryption_service

    async def log_event(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        user_id: Optional[str] = None,
        transaction_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        severity: str = "info",
    ) -> None:
        """Log audit event for compliance"""
        try:
            # Encrypt sensitive details
            encrypted_details = {}
            if details:
                for key, value in details.items():
                    if self._is_sensitive_field(key):
                        encrypted_details[key] = self.encryption_service.encrypt(str(value))
                    else:
                        encrypted_details[key] = value

            async with self.db_manager.get_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO audit_logs (
                        transaction_id, user_id, action, resource_type, resource_id,
                        details, ip_address, user_agent, severity
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                    transaction_id,
                    user_id,
                    action,
                    resource_type,
                    resource_id,
                    encrypted_details,
                    ip_address,
                    user_agent,
                    severity,
                )

            logger.info(f"Audit event logged: {action} on {resource_type}:{resource_id}")

        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
            # Don't raise - audit failure shouldn't break business logic

    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if field contains sensitive data"""
        sensitive_fields = {
            "email",
            "phone",
            "cpf",
            "cnpj",
            "card_number",
            "account_number",
            "password",
            "token",
            "key",
        }
        return any(sensitive in field_name.lower() for sensitive in sensitive_fields)

    async def get_audit_trail(
        self, resource_type: str, resource_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit trail for a resource"""
        try:
            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(
                    """
                    SELECT action, user_id, details, ip_address, timestamp, severity
                    FROM audit_logs
                    WHERE resource_type = $1 AND resource_id = $2
                    ORDER BY timestamp DESC
                    LIMIT $3
                """,
                    resource_type,
                    resource_id,
                    limit,
                )

                # Decrypt sensitive details for authorized access
                audit_trail = []
                for row in results:
                    details = dict(row["details"]) if row["details"] else {}

                    # Note: In production, you'd check authorization before decrypting
                    for key, value in details.items():
                        if self._is_sensitive_field(key) and isinstance(value, str):
                            try:
                                details[key] = self.encryption_service.decrypt(value)
                            except:
                                details[key] = "[ENCRYPTED]"

                    audit_trail.append(
                        {
                            "action": row["action"],
                            "user_id": row["user_id"],
                            "details": details,
                            "ip_address": row["ip_address"],
                            "timestamp": row["timestamp"].isoformat(),
                            "severity": row["severity"],
                        }
                    )

                return audit_trail

        except Exception as e:
            logger.error(f"Audit trail retrieval failed: {e}")
            return []


class SecurityMiddleware:
    """Security middleware for API requests"""

    def __init__(
        self,
        auth_service: AuthenticationService,
        rate_limit_service: RateLimitService,
        audit_service: AuditService,
    ):
        self.auth_service = auth_service
        self.rate_limit_service = rate_limit_service
        self.audit_service = audit_service

    def require_auth(self, permissions: List[str] = None):
        """Decorator for requiring authentication"""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract request from args (Flask/FastAPI specific)
                request = self._extract_request(args, kwargs)

                # Authenticate
                security_context = await self._authenticate_request(request)
                if not security_context:
                    return {"error": "Authentication required"}, 401

                # Check permissions
                if permissions and not self._check_permissions(security_context, permissions):
                    await self.audit_service.log_event(
                        action="access_denied",
                        resource_type="api_endpoint",
                        resource_id=request.path,
                        user_id=security_context.user_id,
                        ip_address=security_context.ip_address,
                        severity="warning",
                    )
                    return {"error": "Insufficient permissions"}, 403

                # Check rate limit
                is_allowed, rate_info = await self.rate_limit_service.check_rate_limit(
                    security_context.api_key_id, security_context.rate_limit
                )

                if not is_allowed:
                    await self.audit_service.log_event(
                        action="rate_limit_exceeded",
                        resource_type="api_endpoint",
                        resource_id=request.path,
                        user_id=security_context.user_id,
                        ip_address=security_context.ip_address,
                        severity="warning",
                    )
                    return {"error": "Rate limit exceeded"}, 429

                # Add security context to kwargs
                kwargs["security_context"] = security_context

                # Execute function
                result = await func(*args, **kwargs)

                # Log successful access
                await self.audit_service.log_event(
                    action="api_access",
                    resource_type="api_endpoint",
                    resource_id=request.path,
                    user_id=security_context.user_id,
                    ip_address=security_context.ip_address,
                )

                return result

            return wrapper

        return decorator

    async def _authenticate_request(self, request) -> Optional[SecurityContext]:
        """Authenticate request and return security context"""
        try:
            # Extract API key from header
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return None

            api_key = auth_header[7:]  # Remove 'Bearer '

            # Authenticate API key
            key_info = await self.auth_service.authenticate_api_key(api_key)
            if not key_info:
                return None

            # Create security context
            return SecurityContext(
                user_id=key_info["name"],  # Using name as user_id for API keys
                api_key_id=str(key_info["id"]),
                permissions=key_info["permissions"],
                rate_limit=key_info["rate_limit"],
                ip_address=self._get_client_ip(request),
                user_agent=request.headers.get("User-Agent", ""),
                authenticated_at=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None

    def _check_permissions(self, context: SecurityContext, required: List[str]) -> bool:
        """Check if user has required permissions"""
        if "admin" in context.permissions:
            return True

        return all(perm in context.permissions for perm in required)

    def _extract_request(self, args, kwargs):
        """Extract request object from function arguments"""
        # This would be framework-specific
        # For now, assume first arg is request
        return args[0] if args else None

    def _get_client_ip(self, request) -> str:
        """Get client IP address"""
        # Check for forwarded headers (load balancer/proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.remote_addr or "unknown"


class InputValidator:
    """Input validation for security"""

    @staticmethod
    def validate_transaction_amount(amount: float) -> bool:
        """Validate transaction amount"""
        return 0 < amount <= 1000000  # Max 1M per transaction

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return re.match(pattern, email) is not None

    @staticmethod
    def validate_merchant_id(merchant_id: str) -> bool:
        """Validate merchant ID format"""
        return re.match(r"^[A-Z0-9_]{3,50}$", merchant_id) is not None

    @staticmethod
    def validate_customer_id(customer_id: str) -> bool:
        """Validate customer ID format"""
        return re.match(r"^[A-Z0-9_]{3,50}$", customer_id) is not None

    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 255) -> str:
        """Sanitize string input"""
        if not isinstance(input_str, str):
            return ""

        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\';]', "", input_str)

        # Truncate to max length
        return sanitized[:max_length]

    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """Validate IP address"""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False


# Security configuration factory
class SecurityFactory:
    """Factory for creating security services"""

    @staticmethod
    def create_security_services(config: Dict[str, Any], db_manager, redis_manager):
        """Create all security services"""

        # Encryption service
        encryption_service = EncryptionService(config["encryption_master_key"])

        # Authentication service
        auth_service = AuthenticationService(config["jwt_secret"], db_manager)

        # Rate limiting service
        rate_limit_service = RateLimitService(redis_manager)

        # Audit service
        audit_service = AuditService(db_manager, encryption_service)

        # Security middleware
        security_middleware = SecurityMiddleware(auth_service, rate_limit_service, audit_service)

        return {
            "encryption": encryption_service,
            "auth": auth_service,
            "rate_limit": rate_limit_service,
            "audit": audit_service,
            "middleware": security_middleware,
            "validator": InputValidator(),
        }
