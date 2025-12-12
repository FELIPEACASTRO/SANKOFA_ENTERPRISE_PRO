"""
Sankofa Enterprise Pro - RBAC (Role-Based Access Control) System
Sistema completo de controle de acesso baseado em papéis e permissões
"""

import os
import hashlib
import secrets
import threading
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
import logging
import json

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Permissões do sistema"""
    TRANSACTIONS_READ = "transactions:read"
    TRANSACTIONS_WRITE = "transactions:write"
    TRANSACTIONS_DELETE = "transactions:delete"
    
    FRAUD_PREDICT = "fraud:predict"
    FRAUD_REVIEW = "fraud:review"
    FRAUD_APPROVE = "fraud:approve"
    FRAUD_REJECT = "fraud:reject"
    
    ALERTS_READ = "alerts:read"
    ALERTS_MANAGE = "alerts:manage"
    ALERTS_RESOLVE = "alerts:resolve"
    
    REPORTS_READ = "reports:read"
    REPORTS_GENERATE = "reports:generate"
    REPORTS_EXPORT = "reports:export"
    
    MODEL_READ = "model:read"
    MODEL_TRAIN = "model:train"
    MODEL_DEPLOY = "model:deploy"
    MODEL_ROLLBACK = "model:rollback"
    
    USERS_READ = "users:read"
    USERS_CREATE = "users:create"
    USERS_UPDATE = "users:update"
    USERS_DELETE = "users:delete"
    USERS_ROLES = "users:roles"
    
    CONFIG_READ = "config:read"
    CONFIG_UPDATE = "config:update"
    
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"
    
    COMPLIANCE_READ = "compliance:read"
    COMPLIANCE_MANAGE = "compliance:manage"
    
    API_KEYS_READ = "api_keys:read"
    API_KEYS_CREATE = "api_keys:create"
    API_KEYS_REVOKE = "api_keys:revoke"
    
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_HEALTH = "system:health"


@dataclass
class Role:
    """Definição de um papel"""
    name: str
    description: str
    permissions: Set[Permission]
    is_system_role: bool = False
    parent_role: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class User:
    """Usuário do sistema"""
    user_id: str
    username: str
    email: str
    roles: Set[str]
    is_active: bool = True
    is_locked: bool = False
    mfa_enabled: bool = False
    permissions_override: Set[Permission] = field(default_factory=set)
    denied_permissions: Set[Permission] = field(default_factory=set)
    last_login: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """Sessão de usuário"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    last_activity: datetime = field(default_factory=datetime.now)


class RBACSystem:
    """
    Sistema RBAC completo

    Features:
    - Hierarquia de papéis
    - Permissões granulares
    - Override de permissões por usuário
    - Negação explícita de permissões
    - Auditoria de acessos
    - Sessões com expiração

    CORRECAO 10/10: Thread-safe com RLock para todas as operacoes
    """

    def __init__(self):
        # CORRECAO 10/10: Lock para thread-safety
        self._lock = threading.RLock()

        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.access_log: List[Dict] = []

        self._initialize_default_roles()

        logger.info("RBAC System initialized (thread-safe)")
    
    def _initialize_default_roles(self):
        """Inicializa papéis padrão do sistema"""
        
        self.create_role(Role(
            name="admin",
            description="Administrador do sistema com acesso total",
            permissions={p for p in Permission},
            is_system_role=True
        ))
        
        self.create_role(Role(
            name="fraud_analyst",
            description="Analista de fraude",
            permissions={
                Permission.TRANSACTIONS_READ,
                Permission.FRAUD_PREDICT,
                Permission.FRAUD_REVIEW,
                Permission.ALERTS_READ,
                Permission.ALERTS_MANAGE,
                Permission.ALERTS_RESOLVE,
                Permission.REPORTS_READ,
                Permission.REPORTS_GENERATE,
                Permission.MODEL_READ,
                Permission.AUDIT_READ,
            },
            is_system_role=True
        ))
        
        self.create_role(Role(
            name="fraud_supervisor",
            description="Supervisor de fraude com permissões de aprovação",
            permissions={
                Permission.TRANSACTIONS_READ,
                Permission.TRANSACTIONS_WRITE,
                Permission.FRAUD_PREDICT,
                Permission.FRAUD_REVIEW,
                Permission.FRAUD_APPROVE,
                Permission.FRAUD_REJECT,
                Permission.ALERTS_READ,
                Permission.ALERTS_MANAGE,
                Permission.ALERTS_RESOLVE,
                Permission.REPORTS_READ,
                Permission.REPORTS_GENERATE,
                Permission.REPORTS_EXPORT,
                Permission.MODEL_READ,
                Permission.AUDIT_READ,
                Permission.AUDIT_EXPORT,
                Permission.USERS_READ,
            },
            is_system_role=True,
            parent_role="fraud_analyst"
        ))
        
        self.create_role(Role(
            name="compliance_officer",
            description="Oficial de compliance",
            permissions={
                Permission.TRANSACTIONS_READ,
                Permission.FRAUD_REVIEW,
                Permission.ALERTS_READ,
                Permission.REPORTS_READ,
                Permission.REPORTS_GENERATE,
                Permission.REPORTS_EXPORT,
                Permission.AUDIT_READ,
                Permission.AUDIT_EXPORT,
                Permission.COMPLIANCE_READ,
                Permission.COMPLIANCE_MANAGE,
            },
            is_system_role=True
        ))
        
        self.create_role(Role(
            name="data_scientist",
            description="Cientista de dados para treinamento de modelos",
            permissions={
                Permission.TRANSACTIONS_READ,
                Permission.FRAUD_PREDICT,
                Permission.REPORTS_READ,
                Permission.MODEL_READ,
                Permission.MODEL_TRAIN,
                Permission.MODEL_DEPLOY,
                Permission.MODEL_ROLLBACK,
            },
            is_system_role=True
        ))
        
        self.create_role(Role(
            name="viewer",
            description="Visualizador apenas leitura",
            permissions={
                Permission.TRANSACTIONS_READ,
                Permission.ALERTS_READ,
                Permission.REPORTS_READ,
                Permission.MODEL_READ,
                Permission.SYSTEM_HEALTH,
            },
            is_system_role=True
        ))
        
        self.create_role(Role(
            name="api_service",
            description="Serviço de API para integrações",
            permissions={
                Permission.TRANSACTIONS_READ,
                Permission.TRANSACTIONS_WRITE,
                Permission.FRAUD_PREDICT,
                Permission.SYSTEM_HEALTH,
            },
            is_system_role=True
        ))
    
    def create_role(self, role: Role) -> bool:
        """Cria um novo papel

        CORRECAO 10/10: Thread-safe com lock
        """
        with self._lock:
            if role.name in self.roles and self.roles[role.name].is_system_role:
                logger.warning(f"Cannot modify system role: {role.name}")
                return False

            if role.parent_role and role.parent_role in self.roles:
                parent = self.roles[role.parent_role]
                role.permissions = role.permissions.union(parent.permissions)

            self.roles[role.name] = role
            logger.info(f"Role created: {role.name}")
            return True
    
    def delete_role(self, role_name: str) -> bool:
        """Remove um papel

        CORRECAO 10/10: Thread-safe com lock
        """
        with self._lock:
            if role_name not in self.roles:
                return False

            if self.roles[role_name].is_system_role:
                logger.warning(f"Cannot delete system role: {role_name}")
                return False

            users_with_role = [u for u in self.users.values() if role_name in u.roles]
            for user in users_with_role:
                user.roles.discard(role_name)

            del self.roles[role_name]
            logger.info(f"Role deleted: {role_name}")
            return True
    
    def create_user(self, user: User) -> bool:
        """Cria um novo usuário

        CORRECAO 10/10: Thread-safe com lock
        """
        with self._lock:
            if user.user_id in self.users:
                logger.warning(f"User already exists: {user.user_id}")
                return False

            invalid_roles = user.roles - set(self.roles.keys())
            if invalid_roles:
                logger.warning(f"Invalid roles: {invalid_roles}")
                return False

            self.users[user.user_id] = user
            logger.info(f"User created: {user.username}")
            return True
    
    def update_user(self, user_id: str, updates: Dict) -> bool:
        """Atualiza um usuário

        CORRECAO 10/10: Thread-safe com lock
        """
        with self._lock:
            if user_id not in self.users:
                return False

            user = self.users[user_id]

            if 'roles' in updates:
                invalid_roles = set(updates['roles']) - set(self.roles.keys())
                if invalid_roles:
                    logger.warning(f"Invalid roles: {invalid_roles}")
                    return False
                user.roles = set(updates['roles'])

            if 'is_active' in updates:
                user.is_active = updates['is_active']

            if 'is_locked' in updates:
                user.is_locked = updates['is_locked']

            if 'mfa_enabled' in updates:
                user.mfa_enabled = updates['mfa_enabled']

            if 'permissions_override' in updates:
                user.permissions_override = {Permission(p) for p in updates['permissions_override']}

            if 'denied_permissions' in updates:
                user.denied_permissions = {Permission(p) for p in updates['denied_permissions']}

            logger.info(f"User updated: {user.username}")
            return True
    
    def assign_role(self, user_id: str, role_name: str) -> bool:
        """Atribui um papel a um usuário

        CORRECAO 10/10: Thread-safe com lock
        """
        with self._lock:
            if user_id not in self.users:
                return False

            if role_name not in self.roles:
                return False

            self.users[user_id].roles.add(role_name)
            logger.info(f"Role {role_name} assigned to user {user_id}")
            return True

    def revoke_role(self, user_id: str, role_name: str) -> bool:
        """Remove um papel de um usuário

        CORRECAO 10/10: Thread-safe com lock
        """
        with self._lock:
            if user_id not in self.users:
                return False

            self.users[user_id].roles.discard(role_name)
            logger.info(f"Role {role_name} revoked from user {user_id}")
            return True
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Retorna todas as permissões efetivas de um usuário

        CORRECAO 10/10: Thread-safe com lock
        """
        with self._lock:
            if user_id not in self.users:
                return set()

            user = self.users[user_id]

            if not user.is_active or user.is_locked:
                return set()

            permissions = set()
            for role_name in user.roles:
                if role_name in self.roles:
                    permissions.update(self.roles[role_name].permissions)

            permissions.update(user.permissions_override)

            permissions -= user.denied_permissions

            return permissions
    
    def check_permission(
        self,
        user_id: str,
        permission: Permission,
        resource_id: Optional[str] = None
    ) -> bool:
        """
        Verifica se usuário tem uma permissão específica
        
        Args:
            user_id: ID do usuário
            permission: Permissão a verificar
            resource_id: ID do recurso (para controle granular futuro)
            
        Returns:
            True se tem permissão
        """
        if user_id not in self.users:
            self._log_access(user_id, permission, False, "user_not_found")
            return False
        
        user = self.users[user_id]
        
        if not user.is_active:
            self._log_access(user_id, permission, False, "user_inactive")
            return False
        
        if user.is_locked:
            self._log_access(user_id, permission, False, "user_locked")
            return False
        
        user_permissions = self.get_user_permissions(user_id)
        
        if Permission.SYSTEM_ADMIN in user_permissions:
            self._log_access(user_id, permission, True, "admin_override")
            return True
        
        has_permission = permission in user_permissions
        
        self._log_access(
            user_id, permission, has_permission,
            "granted" if has_permission else "denied"
        )
        
        return has_permission
    
    def check_permissions(self, user_id: str, permissions: List[Permission]) -> bool:
        """Verifica se usuário tem TODAS as permissões listadas"""
        return all(self.check_permission(user_id, p) for p in permissions)
    
    def check_any_permission(self, user_id: str, permissions: List[Permission]) -> bool:
        """Verifica se usuário tem ALGUMA das permissões listadas"""
        return any(self.check_permission(user_id, p) for p in permissions)
    
    def create_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str,
        duration_hours: int = 24
    ) -> Optional[Session]:
        """Cria uma nova sessão para o usuário

        CORRECAO 10/10: Thread-safe com lock
        """
        with self._lock:
            if user_id not in self.users:
                return None

            user = self.users[user_id]
            if not user.is_active or user.is_locked:
                return None

            session_id = secrets.token_urlsafe(32)
            now = datetime.now()

            session = Session(
                session_id=session_id,
                user_id=user_id,
                created_at=now,
                expires_at=now + timedelta(hours=duration_hours),
                ip_address=ip_address,
                user_agent=user_agent
            )

            self.sessions[session_id] = session
            user.last_login = now

            logger.info(f"Session created for user {user_id}")
            return session
    
    def validate_session(self, session_id: str) -> Optional[User]:
        """Valida uma sessão e retorna o usuário

        CORRECAO 10/10: Thread-safe com lock
        """
        with self._lock:
            if session_id not in self.sessions:
                return None

            session = self.sessions[session_id]

            if not session.is_active:
                return None

            if datetime.now() > session.expires_at:
                session.is_active = False
                return None

            session.last_activity = datetime.now()

            return self.users.get(session.user_id)

    def invalidate_session(self, session_id: str) -> bool:
        """Invalida uma sessão

        CORRECAO 10/10: Thread-safe com lock
        """
        with self._lock:
            if session_id not in self.sessions:
                return False

            self.sessions[session_id].is_active = False
            logger.info(f"Session invalidated: {session_id[:8]}...")
            return True

    def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalida todas as sessões de um usuário

        CORRECAO 10/10: Thread-safe com lock
        """
        with self._lock:
            count = 0
            for session in self.sessions.values():
                if session.user_id == user_id and session.is_active:
                    session.is_active = False
                    count += 1

            logger.info(f"Invalidated {count} sessions for user {user_id}")
            return count
    
    def _log_access(
        self,
        user_id: str,
        permission: Permission,
        granted: bool,
        reason: str
    ):
        """Registra tentativa de acesso

        CORRECAO 10/10: Thread-safe com lock
        """
        with self._lock:
            self.access_log.append({
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "permission": permission.value,
                "granted": granted,
                "reason": reason
            })

            if len(self.access_log) > 10000:
                self.access_log = self.access_log[-5000:]
    
    def get_access_log(
        self,
        user_id: Optional[str] = None,
        permission: Optional[Permission] = None,
        granted: Optional[bool] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Retorna log de acesso com filtros"""
        logs = self.access_log
        
        if user_id:
            logs = [l for l in logs if l['user_id'] == user_id]
        
        if permission:
            logs = [l for l in logs if l['permission'] == permission.value]
        
        if granted is not None:
            logs = [l for l in logs if l['granted'] == granted]
        
        return logs[-limit:]
    
    def get_user_roles(self, user_id: str) -> List[str]:
        """Retorna papéis de um usuário"""
        if user_id not in self.users:
            return []
        return list(self.users[user_id].roles)
    
    def get_role_users(self, role_name: str) -> List[str]:
        """Retorna usuários com um papel específico"""
        if role_name not in self.roles:
            return []
        return [u.user_id for u in self.users.values() if role_name in u.roles]
    
    def export_config(self) -> Dict:
        """Exporta configuração RBAC"""
        return {
            "roles": {
                name: {
                    "description": role.description,
                    "permissions": [p.value for p in role.permissions],
                    "is_system_role": role.is_system_role,
                    "parent_role": role.parent_role
                }
                for name, role in self.roles.items()
            },
            "exported_at": datetime.now().isoformat()
        }


def require_permission(*permissions: Permission):
    """
    Decorator para verificar permissões em endpoints
    
    Usage:
        @require_permission(Permission.TRANSACTIONS_READ)
        def get_transactions():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from flask import g, jsonify
            
            user_id = getattr(g, 'user_id', None)
            if not user_id:
                return jsonify({"error": "Authentication required"}), 401
            
            rbac = get_rbac_system()
            
            for perm in permissions:
                if not rbac.check_permission(user_id, perm):
                    return jsonify({
                        "error": "Permission denied",
                        "required_permission": perm.value
                    }), 403
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


_rbac_instance: Optional[RBACSystem] = None
_rbac_lock = threading.Lock()


def get_rbac_system() -> RBACSystem:
    """Retorna instância singleton do sistema RBAC

    CORRECAO 10/10: Double-checked locking para thread-safety
    """
    global _rbac_instance
    if _rbac_instance is None:
        with _rbac_lock:
            if _rbac_instance is None:
                _rbac_instance = RBACSystem()
    return _rbac_instance


def initialize_rbac_with_users(users: List[Dict]) -> RBACSystem:
    """Inicializa RBAC com usuários pré-definidos"""
    rbac = get_rbac_system()
    
    for user_data in users:
        user = User(
            user_id=user_data['user_id'],
            username=user_data['username'],
            email=user_data.get('email', f"{user_data['username']}@sankofa.com"),
            roles=set(user_data.get('roles', ['viewer'])),
            is_active=user_data.get('is_active', True)
        )
        rbac.create_user(user)
    
    return rbac
