"""
Sankofa Enterprise Pro - RBAC PostgreSQL Persistence
Persistência de RBAC em PostgreSQL para produção
"""

import os
import json
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
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


import threading


class RBACPersistence:
    """
    Camada de persistência PostgreSQL para RBAC
    
    Persiste:
    - Papéis e permissões (rbac_roles)
    - Relacionamento usuário-papel (rbac_user_roles)  
    - Sessões (rbac_sessions)
    - Overrides de permissão (rbac_permissions_override)
    
    Thread-safe: Cada operação cria nova conexão
    """
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self._available = PSYCOPG2_AVAILABLE and self.database_url is not None
        self._lock = threading.Lock()
        
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
    
    def _get_connection(self):
        """Cria nova conexão para esta operação (thread-safe)"""
        if not self.is_available or psycopg2 is None:
            return None
        try:
            conn = psycopg2.connect(self.database_url)
            conn.autocommit = True
            return conn
        except Exception as e:
            logger.error(f"RBAC persistence: Connection failed - {e}")
            return None
    
    def _execute_with_connection(self, operation):
        """Executa operação com conexão dedicada (thread-safe)"""
        conn = self._get_connection()
        if conn is None or RealDictCursor is None:
            return None
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            result = operation(cursor)
            cursor.close()
            return result
        except Exception as e:
            logger.error(f"RBAC persistence: Operation failed - {e}")
            return None
        finally:
            try:
                conn.close()
            except:
                pass
    
    def load_role(self, role_name: str) -> Optional[Dict]:
        """Carrega papel do banco de dados"""
        cursor = self._get_cursor()
        if not cursor:
            return None
        
        try:
            cursor.execute(
                "SELECT * FROM rbac_roles WHERE name = %s",
                (role_name,)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"Error loading role {role_name}: {e}")
            return None
        finally:
            cursor.close()
    
    def load_all_roles(self) -> List[Dict]:
        """Carrega todos os papéis"""
        cursor = self._get_cursor()
        if not cursor:
            return []
        
        try:
            cursor.execute("SELECT * FROM rbac_roles ORDER BY name")
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error loading roles: {e}")
            return []
        finally:
            cursor.close()
    
    def save_role(self, name: str, description: str, permissions: List[str], 
                  is_system_role: bool = False, parent_role: Optional[str] = None) -> bool:
        """Salva ou atualiza papel"""
        cursor = self._get_cursor()
        if not cursor:
            return False
        
        try:
            cursor.execute("""
                INSERT INTO rbac_roles (name, description, permissions, is_system_role, parent_role)
                VALUES (%s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    description = EXCLUDED.description,
                    permissions = EXCLUDED.permissions,
                    parent_role = EXCLUDED.parent_role,
                    updated_at = NOW()
            """, (name, description, json.dumps(permissions), is_system_role, parent_role))
            return True
        except Exception as e:
            logger.error(f"Error saving role {name}: {e}")
            return False
        finally:
            cursor.close()
    
    def delete_role(self, role_name: str) -> bool:
        """Remove papel (apenas não-sistema)"""
        cursor = self._get_cursor()
        if not cursor:
            return False
        
        try:
            cursor.execute(
                "DELETE FROM rbac_roles WHERE name = %s AND is_system_role = FALSE",
                (role_name,)
            )
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting role {role_name}: {e}")
            return False
        finally:
            cursor.close()
    
    def get_user_roles(self, user_id: str) -> List[str]:
        """Retorna papéis de um usuário"""
        cursor = self._get_cursor()
        if not cursor:
            return []
        
        try:
            cursor.execute(
                """SELECT role_name FROM rbac_user_roles 
                   WHERE user_id = %s AND (expires_at IS NULL OR expires_at > NOW())""",
                (user_id,)
            )
            return [row['role_name'] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting user roles for {user_id}: {e}")
            return []
        finally:
            cursor.close()
    
    def assign_role(self, user_id: str, role_name: str, granted_by: Optional[str] = None,
                    expires_at: Optional[datetime] = None) -> bool:
        """Atribui papel a usuário"""
        cursor = self._get_cursor()
        if not cursor:
            return False
        
        try:
            cursor.execute("""
                INSERT INTO rbac_user_roles (user_id, role_name, granted_by, expires_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, role_name) DO UPDATE SET
                    granted_by = EXCLUDED.granted_by,
                    expires_at = EXCLUDED.expires_at,
                    granted_at = NOW()
            """, (user_id, role_name, granted_by, expires_at))
            return True
        except Exception as e:
            logger.error(f"Error assigning role {role_name} to user {user_id}: {e}")
            return False
        finally:
            cursor.close()
    
    def revoke_role(self, user_id: str, role_name: str) -> bool:
        """Remove papel de usuário"""
        cursor = self._get_cursor()
        if not cursor:
            return False
        
        try:
            cursor.execute(
                "DELETE FROM rbac_user_roles WHERE user_id = %s AND role_name = %s",
                (user_id, role_name)
            )
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error revoking role {role_name} from user {user_id}: {e}")
            return False
        finally:
            cursor.close()
    
    def create_session(self, session_id: str, user_id: str, expires_at: datetime,
                       ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                       metadata: Optional[Dict] = None) -> bool:
        """Cria sessão de usuário"""
        cursor = self._get_cursor()
        if not cursor:
            return False
        
        try:
            cursor.execute("""
                INSERT INTO rbac_sessions (session_id, user_id, ip_address, user_agent, expires_at, metadata)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            """, (session_id, user_id, ip_address, user_agent, expires_at, 
                  json.dumps(metadata or {})))
            return True
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return False
        finally:
            cursor.close()
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Recupera sessão"""
        cursor = self._get_cursor()
        if not cursor:
            return None
        
        try:
            cursor.execute(
                """SELECT * FROM rbac_sessions 
                   WHERE session_id = %s AND is_active = TRUE AND expires_at > NOW()""",
                (session_id,)
            )
            row = cursor.fetchone()
            if row:
                cursor.execute(
                    "UPDATE rbac_sessions SET last_activity = NOW() WHERE session_id = %s",
                    (session_id,)
                )
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            return None
        finally:
            cursor.close()
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalida sessão"""
        cursor = self._get_cursor()
        if not cursor:
            return False
        
        try:
            cursor.execute(
                "UPDATE rbac_sessions SET is_active = FALSE WHERE session_id = %s",
                (session_id,)
            )
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error invalidating session {session_id}: {e}")
            return False
        finally:
            cursor.close()
    
    def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalida todas as sessões de um usuário"""
        cursor = self._get_cursor()
        if not cursor:
            return 0
        
        try:
            cursor.execute(
                "UPDATE rbac_sessions SET is_active = FALSE WHERE user_id = %s",
                (user_id,)
            )
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Error invalidating sessions for user {user_id}: {e}")
            return 0
        finally:
            cursor.close()
    
    def cleanup_expired_sessions(self) -> int:
        """Remove sessões expiradas"""
        cursor = self._get_cursor()
        if not cursor:
            return 0
        
        try:
            cursor.execute(
                "DELETE FROM rbac_sessions WHERE expires_at < NOW() OR is_active = FALSE"
            )
            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} expired sessions")
            return deleted
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
            return 0
        finally:
            cursor.close()
    
    def get_permission_overrides(self, user_id: str) -> Dict[str, bool]:
        """Retorna overrides de permissão para um usuário"""
        cursor = self._get_cursor()
        if not cursor:
            return {}
        
        try:
            cursor.execute(
                "SELECT permission, is_granted FROM rbac_permissions_override WHERE user_id = %s",
                (user_id,)
            )
            return {row['permission']: row['is_granted'] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Error getting permission overrides for {user_id}: {e}")
            return {}
        finally:
            cursor.close()
    
    def set_permission_override(self, user_id: str, permission: str, is_granted: bool,
                                 reason: Optional[str] = None, granted_by: Optional[str] = None) -> bool:
        """Define override de permissão"""
        cursor = self._get_cursor()
        if not cursor:
            return False
        
        try:
            cursor.execute("""
                INSERT INTO rbac_permissions_override (user_id, permission, is_granted, reason, granted_by)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id, permission) DO UPDATE SET
                    is_granted = EXCLUDED.is_granted,
                    reason = EXCLUDED.reason,
                    granted_by = EXCLUDED.granted_by,
                    created_at = NOW()
            """, (user_id, permission, is_granted, reason, granted_by))
            return True
        except Exception as e:
            logger.error(f"Error setting permission override: {e}")
            return False
        finally:
            cursor.close()
    
    def remove_permission_override(self, user_id: str, permission: str) -> bool:
        """Remove override de permissão"""
        cursor = self._get_cursor()
        if not cursor:
            return False
        
        try:
            cursor.execute(
                "DELETE FROM rbac_permissions_override WHERE user_id = %s AND permission = %s",
                (user_id, permission)
            )
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error removing permission override: {e}")
            return False
        finally:
            cursor.close()


_rbac_persistence: Optional[RBACPersistence] = None


def get_rbac_persistence() -> RBACPersistence:
    """Retorna instância singleton de persistência RBAC"""
    global _rbac_persistence
    if _rbac_persistence is None:
        _rbac_persistence = RBACPersistence()
    return _rbac_persistence
