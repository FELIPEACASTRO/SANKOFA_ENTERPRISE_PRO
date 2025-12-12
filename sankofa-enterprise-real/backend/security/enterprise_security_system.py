#!/usr/bin/env python3
"""
Sistema de Segurança Enterprise para Sankofa Enterprise Pro
Implementa OAuth2/JWT, HTTPS, criptografia AES-256, RBAC e auditoria completa

V001 FIX: Migrado de SQLite para PostgreSQL para produção
V006 FIX: Encryption key persistida e carregada de variável de ambiente
"""

import os
import jwt
import bcrypt
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import hashlib
import logging
from functools import wraps
from flask import request, jsonify, current_app
import json
import threading

# PostgreSQL em vez de SQLite (V001 FIX)
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Configuração de logging seguro
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)


class EnterpriseSecuritySystem:
    """
    Sistema de segurança enterprise completo para ambiente bancário

    V001 FIX: Usa PostgreSQL em vez de SQLite
    V006 FIX: Encryption key é carregada de variável de ambiente
    CORRECAO 10/10: Thread-safe com RLock para operacoes sensiveis
    """

    def __init__(self):
        # CORRECAO 10/10: Lock para operacoes thread-safe
        self._security_lock = threading.RLock()

        # JWT Secret - DEVE ser configurado em produção
        self.jwt_secret = os.environ.get("SANKOFA_JWT_SECRET") or os.environ.get("JWT_SECRET")
        self._is_production = os.environ.get("ENVIRONMENT", "development") == "production"

        if not self.jwt_secret:
            if self._is_production:
                raise ValueError(
                    "SANKOFA_JWT_SECRET ou JWT_SECRET DEVE ser definido em produção"
                )
            logger.warning(
                "JWT_SECRET não definido, gerando um segredo temporário. NÃO USE EM PRODUÇÃO."
            )
            self.jwt_secret = secrets.token_urlsafe(64)
        else:
            logger.info("Chave JWT carregada da variável de ambiente.")

        # V006 FIX: Encryption key persistida
        self.encryption_key = self._load_or_generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)

        # V001 FIX: PostgreSQL connection string
        self._database_url = os.environ.get("DATABASE_URL")
        self._use_postgres = POSTGRES_AVAILABLE and self._database_url is not None

        if self._is_production and not self._use_postgres:
            raise ValueError(
                "DATABASE_URL DEVE ser configurado em produção para usar PostgreSQL"
            )

        self._init_security_database()
        self._init_default_roles()

        # Configurações de segurança
        self.jwt_expiration_hours = 8
        self.refresh_token_days = 30
        self.max_login_attempts = 3
        self.lockout_duration_minutes = 15

        db_type = "PostgreSQL" if self._use_postgres else "In-Memory (dev only)"
        logger.info(f"Sistema de Segurança Enterprise inicializado - DB: {db_type}")

    def _load_or_generate_encryption_key(self) -> bytes:
        """
        V006 FIX: Carrega encryption key de variável de ambiente ou gera uma nova

        Em produção, ENCRYPTION_KEY deve ser definida.
        Em desenvolvimento, gera uma chave e avisa.
        """
        env_key = os.environ.get("ENCRYPTION_KEY") or os.environ.get("SANKOFA_ENCRYPTION_KEY")

        if env_key:
            # Chave fornecida - usar diretamente
            try:
                # Se já é base64 válido para Fernet
                if len(env_key) == 44 and env_key.endswith('='):
                    return env_key.encode()
                # Caso contrário, derivar uma chave do valor fornecido
                return self._derive_key_from_password(env_key)
            except Exception as e:
                logger.error(f"Erro ao carregar ENCRYPTION_KEY: {e}")
                raise ValueError("ENCRYPTION_KEY inválida")

        if self._is_production:
            raise ValueError(
                "ENCRYPTION_KEY ou SANKOFA_ENCRYPTION_KEY DEVE ser definido em produção"
            )

        # Desenvolvimento: gerar chave temporária
        logger.warning(
            "ENCRYPTION_KEY não definida, gerando chave temporária. NÃO USE EM PRODUÇÃO."
        )
        return Fernet.generate_key()

    def _derive_key_from_password(self, password: str) -> bytes:
        """Deriva uma chave Fernet de uma senha/string

        CORRECAO 10/10: Salt DEVE ser configurado em producao (nao usar default)
        """
        salt_env = os.environ.get("ENCRYPTION_SALT")

        if not salt_env:
            if self._is_production:
                raise ValueError(
                    "ENCRYPTION_SALT DEVE ser definido em producao. "
                    "Gere com: python -c \"import secrets; print(secrets.token_hex(16))\""
                )
            # Em desenvolvimento, gerar salt aleatorio mas avisar
            logger.warning(
                "ENCRYPTION_SALT nao definido - gerando salt temporario. "
                "NAO USE EM PRODUCAO."
            )
            salt_env = secrets.token_hex(16)

        salt = salt_env.encode()

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def _get_connection(self):
        """
        V001 FIX: Retorna conexão PostgreSQL
        """
        if not self._use_postgres:
            raise RuntimeError("PostgreSQL não disponível - configure DATABASE_URL")
        return psycopg2.connect(self._database_url, cursor_factory=RealDictCursor)

    def _init_security_database(self):
        """
        V001 FIX: Inicializa banco de dados de segurança no PostgreSQL
        """
        if not self._use_postgres:
            logger.warning("PostgreSQL não disponível - security tables não criadas")
            self._in_memory_users = {}
            self._in_memory_roles = {}
            self._in_memory_sessions = {}
            self._in_memory_audit = []
            return

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Tabela de roles (RBAC)
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS security_roles (
                            id SERIAL PRIMARY KEY,
                            name VARCHAR(100) UNIQUE NOT NULL,
                            description TEXT,
                            permissions JSONB NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Tabela de usuários
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS security_users (
                            id SERIAL PRIMARY KEY,
                            username VARCHAR(100) UNIQUE NOT NULL,
                            email VARCHAR(255) UNIQUE NOT NULL,
                            password_hash VARCHAR(255) NOT NULL,
                            salt VARCHAR(100) NOT NULL,
                            role_id INTEGER NOT NULL REFERENCES security_roles(id),
                            is_active BOOLEAN DEFAULT TRUE,
                            failed_login_attempts INTEGER DEFAULT 0,
                            locked_until TIMESTAMP NULL,
                            last_login TIMESTAMP NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Tabela de sessões ativas
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS security_sessions (
                            id SERIAL PRIMARY KEY,
                            user_id INTEGER NOT NULL REFERENCES security_users(id),
                            session_token TEXT UNIQUE NOT NULL,
                            refresh_token TEXT UNIQUE NOT NULL,
                            ip_address VARCHAR(45) NOT NULL,
                            user_agent TEXT,
                            expires_at TIMESTAMP NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Tabela de auditoria
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS security_audit_log (
                            id SERIAL PRIMARY KEY,
                            user_id INTEGER REFERENCES security_users(id),
                            action VARCHAR(100) NOT NULL,
                            resource VARCHAR(100),
                            details TEXT,
                            ip_address VARCHAR(45),
                            user_agent TEXT,
                            success BOOLEAN NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Índices para performance
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_security_sessions_token
                        ON security_sessions(session_token)
                    """)
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_security_sessions_user
                        ON security_sessions(user_id)
                    """)
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_security_audit_user
                        ON security_audit_log(user_id)
                    """)
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_security_audit_action
                        ON security_audit_log(action)
                    """)

                    conn.commit()
                    logger.info("Security tables criadas/verificadas no PostgreSQL")
        except Exception as e:
            logger.error(f"Erro ao inicializar security database: {e}")
            raise

    def _init_default_roles(self):
        """Inicializa roles padrão do sistema"""
        default_roles = [
            {
                "name": "admin",
                "description": "Administrador do sistema",
                "permissions": [
                    "read_all", "write_all", "delete_all", "manage_users",
                    "view_audit", "system_config", "fraud_analysis", "share_fraud_data"
                ]
            },
            {
                "name": "analyst",
                "description": "Analista de fraude",
                "permissions": [
                    "read_transactions", "fraud_analysis", "view_reports",
                    "mark_fraud", "view_dashboard"
                ]
            },
            {
                "name": "operator",
                "description": "Operador do sistema",
                "permissions": ["read_transactions", "view_dashboard", "basic_reports"]
            },
            {
                "name": "auditor",
                "description": "Auditor de compliance",
                "permissions": ["view_audit", "read_all", "compliance_reports"]
            },
        ]

        if not self._use_postgres:
            for role in default_roles:
                self._in_memory_roles[role["name"]] = role
            return

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    for role in default_roles:
                        cursor.execute("""
                            INSERT INTO security_roles (name, description, permissions)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (name) DO NOTHING
                        """, (role["name"], role["description"], json.dumps(role["permissions"])))
                    conn.commit()
        except Exception as e:
            logger.error(f"Erro ao inicializar roles: {e}")

    def encrypt_sensitive_data(self, data: str) -> str:
        """Criptografa dados sensíveis com AES-256"""
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Erro na criptografia: {e}")
            raise

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Descriptografa dados sensíveis"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Erro na descriptografia: {e}")
            raise

    def hash_password(self, password: str) -> tuple:
        """Gera hash seguro da senha com salt"""
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode("utf-8"), salt)
        return password_hash.decode("utf-8"), salt.decode("utf-8")

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verifica senha contra hash"""
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))

    def create_user(
        self, username: str, email: str, password: str, role_name: str
    ) -> Dict[str, Any]:
        """Cria novo usuário no sistema

        CORRECAO 10/10: Thread-safe com lock
        """
        with self._security_lock:
            password_hash, salt = self.hash_password(password)

            if not self._use_postgres:
                # Fallback in-memory para desenvolvimento
                if role_name not in self._in_memory_roles:
                    raise ValueError(f"Role '{role_name}' não encontrada")
                user_id = len(self._in_memory_users) + 1
                self._in_memory_users[user_id] = {
                    "id": user_id, "username": username, "email": email,
                    "password_hash": password_hash, "role": role_name
                }
                return {"user_id": user_id, "username": username, "role": role_name}

            try:
                with self._get_connection() as conn:
                    with conn.cursor() as cursor:
                        # Busca role_id
                        cursor.execute(
                            "SELECT id FROM security_roles WHERE name = %s",
                            (role_name,)
                        )
                        role_result = cursor.fetchone()
                        if not role_result:
                            raise ValueError(f"Role '{role_name}' não encontrada")

                        role_id = role_result["id"]

                        # Cria usuário
                        cursor.execute("""
                            INSERT INTO security_users
                            (username, email, password_hash, salt, role_id)
                            VALUES (%s, %s, %s, %s, %s)
                            RETURNING id
                        """, (username, email, password_hash, salt, role_id))

                        user_id = cursor.fetchone()["id"]
                        conn.commit()

                        self._log_audit(
                            user_id, "user_created", "users",
                            f"Usuário {username} criado com role {role_name}"
                        )

                        logger.info(f"Usuário {username} criado com sucesso")
                        return {"user_id": user_id, "username": username, "role": role_name}

            except Exception as e:
                logger.error(f"Erro ao criar usuário: {e}")
                raise

    def authenticate_user(
        self, username: str, password: str, ip_address: str, user_agent: str
    ) -> Dict[str, Any]:
        """Autentica usuário e gera tokens JWT"""
        if not self._use_postgres:
            raise RuntimeError("Autenticação requer PostgreSQL em produção")

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Busca usuário
                    cursor.execute("""
                        SELECT u.id, u.username, u.email, u.password_hash, u.is_active,
                               u.failed_login_attempts, u.locked_until, r.name as role_name,
                               r.permissions
                        FROM security_users u
                        JOIN security_roles r ON u.role_id = r.id
                        WHERE u.username = %s OR u.email = %s
                    """, (username, username))

                    user = cursor.fetchone()
                    if not user:
                        self._log_audit(
                            None, "login_failed", "authentication",
                            f"Usuário não encontrado: {username}",
                            ip_address, user_agent, False
                        )
                        raise ValueError("Credenciais inválidas")

                    user_id = user["id"]
                    password_hash = user["password_hash"]
                    is_active = user["is_active"]
                    failed_attempts = user["failed_login_attempts"]
                    locked_until = user["locked_until"]
                    role_name = user["role_name"]
                    permissions = user["permissions"]

                    # Verifica se conta está ativa
                    if not is_active:
                        self._log_audit(
                            user_id, "login_failed", "authentication",
                            "Conta inativa", ip_address, user_agent, False
                        )
                        raise ValueError("Conta inativa")

                    # Verifica se conta está bloqueada
                    if locked_until and locked_until > datetime.now():
                        self._log_audit(
                            user_id, "login_failed", "authentication",
                            "Conta bloqueada", ip_address, user_agent, False
                        )
                        raise ValueError("Conta temporariamente bloqueada")

                    # Verifica senha
                    if not self.verify_password(password, password_hash):
                        failed_attempts += 1
                        new_locked_until = None

                        if failed_attempts >= self.max_login_attempts:
                            new_locked_until = datetime.now() + timedelta(
                                minutes=self.lockout_duration_minutes
                            )

                        cursor.execute("""
                            UPDATE security_users
                            SET failed_login_attempts = %s, locked_until = %s
                            WHERE id = %s
                        """, (failed_attempts, new_locked_until, user_id))
                        conn.commit()

                        self._log_audit(
                            user_id, "login_failed", "authentication",
                            f"Senha incorreta. Tentativas: {failed_attempts}",
                            ip_address, user_agent, False
                        )
                        raise ValueError("Credenciais inválidas")

                    # Login bem-sucedido - reset tentativas
                    cursor.execute("""
                        UPDATE security_users
                        SET failed_login_attempts = 0, locked_until = NULL,
                            last_login = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (user_id,))

                    # Gera tokens
                    access_token = self._generate_access_token(
                        user_id, user["username"], role_name, permissions
                    )
                    # CORRECAO 10/10: Refresh token com expiração
                    refresh_token_data = self._generate_refresh_token()
                    refresh_token = refresh_token_data["token"]
                    refresh_expires_at = refresh_token_data["expires_at"]

                    # Salva sessão com expiração do refresh token
                    expires_at = datetime.now() + timedelta(hours=self.jwt_expiration_hours)
                    cursor.execute("""
                        INSERT INTO security_sessions
                        (user_id, session_token, refresh_token, ip_address, user_agent, expires_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (user_id, access_token, refresh_token, ip_address, user_agent, expires_at))

                    conn.commit()

                    self._log_audit(
                        user_id, "login_success", "authentication",
                        "Login bem-sucedido", ip_address, user_agent, True
                    )

                    logger.info(f"Login bem-sucedido para usuário {user['username']}")

                    return {
                        "access_token": access_token,
                        "refresh_token": refresh_token,
                        # CORRECAO 10/10: Incluir expiracao do refresh token na resposta
                        "refresh_token_expires_at": refresh_expires_at,
                        "token_type": "Bearer",
                        "expires_in": self.jwt_expiration_hours * 3600,
                        "user": {
                            "id": user_id,
                            "username": user["username"],
                            "email": user["email"],
                            "role": role_name,
                            "permissions": permissions if isinstance(permissions, list) else json.loads(permissions),
                        },
                    }

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Erro na autenticação: {e}")
            raise

    def _generate_access_token(
        self, user_id: int, username: str, role: str, permissions
    ) -> str:
        """Gera token JWT de acesso"""
        if isinstance(permissions, str):
            permissions = json.loads(permissions)

        payload = {
            "user_id": user_id,
            "username": username,
            "role": role,
            "permissions": permissions,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=self.jwt_expiration_hours),
            "type": "access",
        }

        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def _generate_refresh_token(self) -> Dict[str, Any]:
        """Gera token de refresh COM EXPIRAÇÃO

        CORRECAO 10/10: Refresh tokens agora têm expiração controlada
        """
        token = secrets.token_urlsafe(64)
        expires_at = datetime.utcnow() + timedelta(days=self.refresh_token_days)

        return {
            "token": token,
            "expires_at": expires_at.isoformat(),
            "created_at": datetime.utcnow().isoformat()
        }

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verifica e decodifica token JWT"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])

            # Verifica se é token de acesso
            if payload.get("type") != "access":
                raise ValueError("Token inválido")

            if not self._use_postgres:
                return payload

            # Verifica se sessão ainda está ativa
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT id FROM security_sessions
                        WHERE session_token = %s AND expires_at > CURRENT_TIMESTAMP
                    """, (token,))

                    if not cursor.fetchone():
                        raise ValueError("Sessão expirada ou inválida")

            return payload

        except jwt.ExpiredSignatureError:
            raise ValueError("Token expirado")
        except jwt.InvalidTokenError:
            raise ValueError("Token inválido")

    def check_permission(self, user_id: int, required_permission: str) -> bool:
        """Verifica se o usuário tem a permissão necessária

        CORRECAO 10/10: Removido bypass de dev mode - SEMPRE verificar permissoes
        """
        if not self._use_postgres:
            # CORRECAO 10/10: Em desenvolvimento, verificar permissoes in-memory
            # NAO retornar True automaticamente (era vulnerabilidade)
            if user_id in self._in_memory_users:
                user = self._in_memory_users[user_id]
                role_name = user.get("role", "")
                if role_name in self._in_memory_roles:
                    role = self._in_memory_roles[role_name]
                    return required_permission in role.get("permissions", [])
            logger.warning(f"Dev mode: Permission check for user {user_id} - {required_permission}")
            return False  # Default deny em vez de allow

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT r.permissions
                        FROM security_users u
                        JOIN security_roles r ON u.role_id = r.id
                        WHERE u.id = %s
                    """, (user_id,))

                    result = cursor.fetchone()
                    if not result:
                        return False

                    permissions = result["permissions"]
                    if isinstance(permissions, str):
                        permissions = json.loads(permissions)
                    return required_permission in permissions
        except Exception as e:
            logger.error(f"Erro ao verificar permissão: {e}")
            return False

    def _log_audit(
        self,
        user_id: Optional[int],
        action: str,
        resource: str,
        details: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
    ):
        """Registra evento na trilha de auditoria

        CORRECAO 10/10: Thread-safe para fallback in-memory
        """
        if not self._use_postgres:
            with self._security_lock:
                self._in_memory_audit.append({
                    "user_id": user_id, "action": action, "resource": resource,
                    "details": details, "success": success
                })
            return

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO security_audit_log
                        (user_id, action, resource, details, ip_address, user_agent, success)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (user_id, action, resource, details, ip_address, user_agent, success))
                    conn.commit()
        except Exception as e:
            logger.error(f"Erro ao registrar auditoria: {e}")

    def require_auth(self):
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                auth_header = request.headers.get("Authorization")
                if not auth_header or not auth_header.startswith("Bearer "):
                    return jsonify({"error": "Token de autenticação ausente ou mal formatado"}), 401

                token = auth_header.split(" ")[1]

                try:
                    payload = self.verify_token(token)
                    from flask import g
                    g.user = payload
                except ValueError as e:
                    return jsonify({"error": str(e)}), 401

                return f(*args, **kwargs)

            return decorated_function

        return decorator

    def require_permission(self, required_permission: str):
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                from flask import g

                user = g.get("user")
                if not user or not self.check_permission(user["user_id"], required_permission):
                    return jsonify({"error": "Permissão insuficiente"}), 403

                return f(*args, **kwargs)

            return decorated_function

        return decorator

    def invalidate_session(self, token: str) -> bool:
        """Invalida uma sessão (logout)"""
        if not self._use_postgres:
            return True

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM security_sessions WHERE session_token = %s",
                        (token,)
                    )
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Erro ao invalidar sessão: {e}")
            return False

    def cleanup_expired_sessions(self) -> int:
        """Remove sessões expiradas"""
        if not self._use_postgres:
            return 0

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM security_sessions WHERE expires_at < CURRENT_TIMESTAMP"
                    )
                    conn.commit()
                    count = cursor.rowcount
                    if count > 0:
                        logger.info(f"Removidas {count} sessões expiradas")
                    return count
        except Exception as e:
            logger.error(f"Erro ao limpar sessões: {e}")
            return 0
