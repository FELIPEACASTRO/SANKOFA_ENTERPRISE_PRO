"""
Sankofa Enterprise Pro - RBAC PostgreSQL Persistence
Persistência de RBAC em PostgreSQL para produção (thread-safe)
"""

import os
import json
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

psycopg2 = None
RealDictCursor = None
try:
    import psycopg2 as _psycopg2
    from psycopg2.extras import RealDictCursor as _RealDictCursor
    psycopg2 = _psycopg2
    RealDictCursor = _RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger.warning("psycopg2 not available - RBAC persistence disabled")


class RBACPersistence:
    """
    Camada de persistência PostgreSQL para RBAC (thread-safe)
    
    Cada operação cria nova conexão para garantir thread-safety
    """
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self._available = PSYCOPG2_AVAILABLE and self.database_url is not None
        
        if self._available:
            self._test_connection()
    
    def _test_connection(self):
        """Testa se conexão é possível"""
        if psycopg2 is None:
            self._available = False
            return
        try:
            conn = psycopg2.connect(self.database_url)
            conn.close()
            logger.info("RBAC persistence: PostgreSQL available")
        except Exception as e:
            logger.warning(f"RBAC persistence: PostgreSQL not available - {e}")
            self._available = False
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def _execute(self, query: str, params: tuple = (), fetch: str = "none"):
        """Executa query com conexão dedicada"""
        if not self.is_available or psycopg2 is None or RealDictCursor is None:
            return None if fetch != "none" else False
        
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            conn.autocommit = True
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)
            
            if fetch == "one":
                result = cursor.fetchone()
                return dict(result) if result else None
            elif fetch == "all":
                return [dict(row) for row in cursor.fetchall()]
            elif fetch == "rowcount":
                return cursor.rowcount
            else:
                return True
        except Exception as e:
            logger.error(f"RBAC persistence error: {e}")
            return None if fetch != "none" else False
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def load_role(self, role_name: str) -> Optional[Dict]:
        """Carrega papel do banco de dados"""
        return self._execute(
            "SELECT * FROM rbac_roles WHERE name = %s",
            (role_name,),
            fetch="one"
        )
    
    def load_all_roles(self) -> List[Dict]:
        """Carrega todos os papéis"""
        result = self._execute("SELECT * FROM rbac_roles ORDER BY name", fetch="all")
        return result or []
    
    def save_role(self, name: str, description: str, permissions: List[str], 
                  is_system_role: bool = False, parent_role: Optional[str] = None) -> bool:
        """Salva ou atualiza papel"""
        result = self._execute("""
            INSERT INTO rbac_roles (name, description, permissions, is_system_role, parent_role)
            VALUES (%s, %s, %s::jsonb, %s, %s)
            ON CONFLICT (name) DO UPDATE SET
                description = EXCLUDED.description,
                permissions = EXCLUDED.permissions,
                parent_role = EXCLUDED.parent_role,
                updated_at = NOW()
        """, (name, description, json.dumps(permissions), is_system_role, parent_role))
        return result is True
    
    def delete_role(self, role_name: str) -> bool:
        """Remove papel (apenas não-sistema)"""
        result = self._execute(
            "DELETE FROM rbac_roles WHERE name = %s AND is_system_role = FALSE",
            (role_name,),
            fetch="rowcount"
        )
        return result is not None and result > 0
    
    def get_user_roles(self, user_id: str) -> List[str]:
        """Retorna papéis de um usuário"""
        result = self._execute(
            """SELECT role_name FROM rbac_user_roles 
               WHERE user_id = %s AND (expires_at IS NULL OR expires_at > NOW())""",
            (user_id,),
            fetch="all"
        )
        return [row['role_name'] for row in (result or [])]
    
    def assign_role(self, user_id: str, role_name: str, granted_by: Optional[str] = None,
                    expires_at: Optional[datetime] = None) -> bool:
        """Atribui papel a usuário"""
        result = self._execute("""
            INSERT INTO rbac_user_roles (user_id, role_name, granted_by, expires_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id, role_name) DO UPDATE SET
                granted_by = EXCLUDED.granted_by,
                expires_at = EXCLUDED.expires_at,
                granted_at = NOW()
        """, (user_id, role_name, granted_by, expires_at))
        return result is True
    
    def revoke_role(self, user_id: str, role_name: str) -> bool:
        """Remove papel de usuário"""
        result = self._execute(
            "DELETE FROM rbac_user_roles WHERE user_id = %s AND role_name = %s",
            (user_id, role_name),
            fetch="rowcount"
        )
        return result is not None and result > 0
    
    def create_session(self, session_id: str, user_id: str, expires_at: datetime,
                       ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                       metadata: Optional[Dict] = None) -> bool:
        """Cria sessão de usuário"""
        result = self._execute("""
            INSERT INTO rbac_sessions (session_id, user_id, ip_address, user_agent, expires_at, metadata)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        """, (session_id, user_id, ip_address, user_agent, expires_at, json.dumps(metadata or {})))
        return result is True
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Recupera sessão e atualiza last_activity"""
        result = self._execute(
            """SELECT * FROM rbac_sessions 
               WHERE session_id = %s AND is_active = TRUE AND expires_at > NOW()""",
            (session_id,),
            fetch="one"
        )
        if result:
            self._execute(
                "UPDATE rbac_sessions SET last_activity = NOW() WHERE session_id = %s",
                (session_id,)
            )
        return result
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalida sessão"""
        result = self._execute(
            "UPDATE rbac_sessions SET is_active = FALSE WHERE session_id = %s",
            (session_id,),
            fetch="rowcount"
        )
        return result is not None and result > 0
    
    def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalida todas as sessões de um usuário"""
        result = self._execute(
            "UPDATE rbac_sessions SET is_active = FALSE WHERE user_id = %s",
            (user_id,),
            fetch="rowcount"
        )
        return result or 0
    
    def cleanup_expired_sessions(self) -> int:
        """Remove sessões expiradas"""
        result = self._execute(
            "DELETE FROM rbac_sessions WHERE expires_at < NOW() OR is_active = FALSE",
            fetch="rowcount"
        )
        deleted = result or 0
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} expired sessions")
        return deleted
    
    def get_permission_overrides(self, user_id: str) -> Dict[str, bool]:
        """Retorna overrides de permissão para um usuário"""
        result = self._execute(
            "SELECT permission, is_granted FROM rbac_permissions_override WHERE user_id = %s",
            (user_id,),
            fetch="all"
        )
        return {row['permission']: row['is_granted'] for row in (result or [])}
    
    def set_permission_override(self, user_id: str, permission: str, is_granted: bool,
                                 reason: Optional[str] = None, granted_by: Optional[str] = None) -> bool:
        """Define override de permissão"""
        result = self._execute("""
            INSERT INTO rbac_permissions_override (user_id, permission, is_granted, reason, granted_by)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id, permission) DO UPDATE SET
                is_granted = EXCLUDED.is_granted,
                reason = EXCLUDED.reason,
                granted_by = EXCLUDED.granted_by,
                created_at = NOW()
        """, (user_id, permission, is_granted, reason, granted_by))
        return result is True
    
    def remove_permission_override(self, user_id: str, permission: str) -> bool:
        """Remove override de permissão"""
        result = self._execute(
            "DELETE FROM rbac_permissions_override WHERE user_id = %s AND permission = %s",
            (user_id, permission),
            fetch="rowcount"
        )
        return result is not None and result > 0


_rbac_persistence: Optional[RBACPersistence] = None


def get_rbac_persistence() -> RBACPersistence:
    """Retorna instância singleton de persistência RBAC"""
    global _rbac_persistence
    if _rbac_persistence is None:
        _rbac_persistence = RBACPersistence()
    return _rbac_persistence
