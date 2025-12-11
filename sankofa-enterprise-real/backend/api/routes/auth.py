"""
Auth Blueprint - Sankofa Enterprise Pro
Handles authentication-related endpoints
Refactored from production_api.py for better code organization
"""

from flask import Blueprint, request, jsonify, g
from functools import wraps
from datetime import datetime, timedelta, timezone
import jwt as pyjwt
import bcrypt

from config.settings import get_config
from utils.structured_logging import get_structured_logger
from utils.error_handling import ValidationError
from utils.log_sanitizer import sanitize_log_data

try:
    from api.schemas import UserLogin
    from pydantic import ValidationError as PydanticValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    PydanticValidationError = Exception

config = get_config()
logger = get_structured_logger("auth_blueprint", config.monitoring.log_level)

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

# Role permissions configuration
ROLE_PERMISSIONS = {
    "admin": ["*"],
    "analyst": [
        "fraud:view", "fraud:predict", "fraud:explain", "fraud:feedback",
        "transactions:view", "transactions:search",
        "alerts:view", "alerts:acknowledge", "alerts:update",
        "reports:view", "reports:generate",
        "dashboard:view", "metrics:view", "model:view",
        "investigation:view", "audit:view", "observability:view",
    ],
    "operator": [
        "fraud:view", "fraud:predict", "transactions:view",
        "alerts:view", "dashboard:view", "metrics:view", "observability:view",
    ],
    "viewer": ["dashboard:view", "metrics:view", "transactions:view", "alerts:view"],
    "system": ["fraud:predict", "fraud:batch", "model:train", "model:view", "observability:view"],
}


def check_permission(roles: list, required_permission: str) -> bool:
    """Verifica se algum dos roles tem a permissão necessária"""
    for role in roles:
        perms = ROLE_PERMISSIONS.get(role, [])
        if "*" in perms:
            return True
        if required_permission in perms:
            return True
        category = required_permission.split(":")[0] + ":*"
        if category in perms:
            return True
    return False


def require_auth(f):
    """Decorator para exigir autenticação JWT em endpoints sensíveis"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"success": False, "error": "Missing or invalid Authorization header"}), 401

        token = auth_header[7:]
        try:
            payload = pyjwt.decode(
                token, config.security.jwt_secret, algorithms=[config.security.jwt_algorithm]
            )
            g.user = payload
        except pyjwt.ExpiredSignatureError:
            return jsonify({"success": False, "error": "Token expired"}), 401
        except pyjwt.InvalidTokenError as e:
            return jsonify({"success": False, "error": f"Invalid token: {str(e)}"}), 401

        return f(*args, **kwargs)
    return decorated


def require_permission(permission: str):
    """Decorator para exigir permissão RBAC específica"""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return jsonify({"success": False, "error": "Missing or invalid Authorization header"}), 401

            token = auth_header[7:]
            try:
                payload = pyjwt.decode(
                    token, config.security.jwt_secret, algorithms=[config.security.jwt_algorithm]
                )
                g.user = payload
            except pyjwt.ExpiredSignatureError:
                return jsonify({"success": False, "error": "Token expired"}), 401
            except pyjwt.InvalidTokenError as e:
                return jsonify({"success": False, "error": f"Invalid token: {str(e)}"}), 401

            user_roles = payload.get("roles", [])
            if not user_roles:
                user_roles = [payload.get("role", "viewer")]

            if not check_permission(user_roles, permission):
                return jsonify({
                    "success": False,
                    "error": f"Insufficient permissions. Required: {permission}",
                    "code": "FORBIDDEN",
                }), 403

            return f(*args, **kwargs)
        return decorated
    return decorator


def verify_password(password: str, password_hash: str) -> bool:
    """Verifica senha usando bcrypt"""
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except Exception:
        return False


def is_account_locked(locked_until) -> bool:
    """Verifica se conta está bloqueada (timezone-aware)"""
    if not locked_until:
        return False
    now = datetime.now(timezone.utc)
    if locked_until.tzinfo is None:
        locked_until = locked_until.replace(tzinfo=timezone.utc)
    return locked_until > now


# Database helper functions (to be injected)
_db_persistence = None
_get_user_from_db = None
_update_login_attempt = None


def init_auth_blueprint(db_persistence, get_user_func, update_login_func):
    """Initialize auth blueprint with database dependencies"""
    global _db_persistence, _get_user_from_db, _update_login_attempt
    _db_persistence = db_persistence
    _get_user_from_db = get_user_func
    _update_login_attempt = update_login_func


@auth_bp.route("/login", methods=["POST"])
def login():
    """Autenticação de usuário com bcrypt e PostgreSQL"""
    try:
        if not request.json:
            raise ValidationError("Request body is required", context={"endpoint": "/api/auth/login"})

        if PYDANTIC_AVAILABLE:
            validated_request = UserLogin(**request.json)
            username = validated_request.username.strip().lower()
            password = validated_request.password
        else:
            username = request.json.get("username", "").strip().lower()
            password = request.json.get("password", "")

    except PydanticValidationError as e:
        logger.warning("Pydantic validation failed on login", extra=sanitize_log_data({
            'endpoint': '/api/auth/login', 'errors': e.errors()
        }))
        return jsonify({'success': False, 'error': 'Validation failed', 'details': e.errors()}), 400

    if not username or not password:
        return jsonify({"success": False, "error": {"message": "Username and password are required"}}), 400

    if _get_user_from_db is None:
        return jsonify({"success": False, "error": {"message": "Auth service not initialized"}}), 500

    user = _get_user_from_db(username)
    if not user:
        logger.warning("Login attempt for unknown user", username=username)
        return jsonify({"success": False, "error": {"message": "Invalid credentials"}}), 401

    if not user.get("is_active", False):
        logger.warning("Login attempt for inactive user", username=username)
        return jsonify({"success": False, "error": {"message": "Account is disabled"}}), 401

    if is_account_locked(user.get("locked_until")):
        logger.warning("Login attempt for locked user", username=username)
        return jsonify({"success": False, "error": {"message": "Account is temporarily locked. Try again later."}}), 401

    if not verify_password(password, user["password_hash"]):
        if _update_login_attempt:
            _update_login_attempt(user["id"], username, success=False)
        logger.warning("Invalid password for user", username=username)
        return jsonify({"success": False, "error": {"message": "Invalid credentials"}}), 401

    if _update_login_attempt:
        _update_login_attempt(user["id"], username, success=True)

    roles = user.get("roles", [user["role"]])
    primary_role = roles[0] if roles else user["role"]

    token_payload = {
        "sub": username,
        "user_id": user["id"],
        "name": user["name"],
        "role": primary_role,
        "roles": roles,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=24),
    }

    token = pyjwt.encode(token_payload, config.security.jwt_secret, algorithm=config.security.jwt_algorithm)
    logger.info("User logged in successfully", username=username, role=primary_role)

    return jsonify({
        "success": True,
        "data": {
            "token": token,
            "user": {
                "id": user["id"],
                "username": username,
                "name": user["name"],
                "role": primary_role,
                "roles": roles,
                "email": user.get("email"),
            },
            "expires_in": 86400,
        },
    })


@auth_bp.route("/verify", methods=["GET"])
def verify_token():
    """Verifica se o token JWT é válido"""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({
            "success": False, "valid": False,
            "error": {"message": "Missing or invalid Authorization header"},
        }), 401

    token = auth_header[7:]
    try:
        payload = pyjwt.decode(token, config.security.jwt_secret, algorithms=[config.security.jwt_algorithm])
        return jsonify({
            "success": True, "valid": True,
            "data": {"user": {
                "username": payload.get("sub"),
                "name": payload.get("name"),
                "role": payload.get("role"),
            }},
        })
    except pyjwt.ExpiredSignatureError:
        return jsonify({"success": False, "valid": False, "error": {"message": "Token expired"}}), 401
    except pyjwt.InvalidTokenError as e:
        return jsonify({"success": False, "valid": False, "error": {"message": f"Invalid token: {str(e)}"}}), 401


@auth_bp.route("/refresh", methods=["POST"])
def refresh_token():
    """Renova um token JWT válido"""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"success": False, "error": {"message": "Missing or invalid Authorization header"}}), 401

    token = auth_header[7:]
    try:
        payload = pyjwt.decode(token, config.security.jwt_secret, algorithms=[config.security.jwt_algorithm])

        new_payload = {
            "sub": payload.get("sub"),
            "user_id": payload.get("user_id"),
            "name": payload.get("name"),
            "role": payload.get("role"),
            "roles": payload.get("roles", []),
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=24),
        }

        new_token = pyjwt.encode(new_payload, config.security.jwt_secret, algorithm=config.security.jwt_algorithm)
        return jsonify({"success": True, "data": {"token": new_token, "expires_in": 86400}})
    except pyjwt.ExpiredSignatureError:
        return jsonify({"success": False, "error": {"message": "Token expired, please login again"}}), 401
    except pyjwt.InvalidTokenError as e:
        return jsonify({"success": False, "error": {"message": f"Invalid token: {str(e)}"}}), 401
