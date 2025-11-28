"""
Sankofa Enterprise Pro - CPF Tokenization PostgreSQL Persistence
Persistência de tokens CPF em PostgreSQL para produção
"""

import os
import json
from typing import Dict, List, Optional
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
    logger.warning("psycopg2 not available - CPF persistence disabled")


class CPFPersistence:
    """
    Camada de persistência PostgreSQL para tokenização de CPF
    
    Persiste:
    - Tokens e CPFs criptografados (cpf_tokens)
    - Log de acesso (cpf_access_log)
    
    LGPD Compliance:
    - CPFs são armazenados criptografados (AES-256)
    - Apenas o hash SHA-256 do CPF é indexável
    - Todos os acessos são registrados
    """
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self._conn = None
        self._available = False
        
        if PSYCOPG2_AVAILABLE and self.database_url:
            self._try_connect()
    
    def _try_connect(self):
        """Tenta conectar ao PostgreSQL"""
        if psycopg2 is None:
            return
        try:
            self._conn = psycopg2.connect(self.database_url)
            self._conn.autocommit = True
            self._available = True
            logger.info("CPF persistence connected to PostgreSQL")
        except Exception as e:
            logger.warning(f"CPF persistence: PostgreSQL not available - {e}")
            self._available = False
    
    @property
    def is_available(self) -> bool:
        return self._available and self._conn is not None
    
    def _get_cursor(self):
        if not self.is_available or RealDictCursor is None:
            return None
        try:
            return self._conn.cursor(cursor_factory=RealDictCursor)
        except:
            self._try_connect()
            if self.is_available and RealDictCursor is not None:
                return self._conn.cursor(cursor_factory=RealDictCursor)
            return None
    
    def save_token(self, token: str, encrypted_cpf: bytes, cpf_hash: str,
                   expires_at: Optional[datetime] = None, 
                   metadata: Optional[Dict] = None) -> bool:
        """Salva token no banco de dados"""
        cursor = self._get_cursor()
        if not cursor:
            return False
        
        try:
            cursor.execute("""
                INSERT INTO cpf_tokens (token, encrypted_cpf, cpf_hash, expires_at, metadata)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (token) DO UPDATE SET
                    expires_at = EXCLUDED.expires_at,
                    metadata = EXCLUDED.metadata
            """, (token, encrypted_cpf, cpf_hash, expires_at, json.dumps(metadata or {})))
            return True
        except Exception as e:
            logger.error(f"Error saving token: {e}")
            return False
        finally:
            cursor.close()
    
    def get_token_by_hash(self, cpf_hash: str) -> Optional[Dict]:
        """Busca token pelo hash do CPF"""
        cursor = self._get_cursor()
        if not cursor:
            return None
        
        try:
            cursor.execute(
                """SELECT * FROM cpf_tokens 
                   WHERE cpf_hash = %s AND (expires_at IS NULL OR expires_at > NOW())""",
                (cpf_hash,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting token by hash: {e}")
            return None
        finally:
            cursor.close()
    
    def get_token_data(self, token: str) -> Optional[Dict]:
        """Busca dados do token"""
        cursor = self._get_cursor()
        if not cursor:
            return None
        
        try:
            cursor.execute(
                """SELECT * FROM cpf_tokens 
                   WHERE token = %s AND (expires_at IS NULL OR expires_at > NOW())""",
                (token,)
            )
            row = cursor.fetchone()
            if row:
                cursor.execute(
                    """UPDATE cpf_tokens 
                       SET access_count = access_count + 1, last_accessed = NOW() 
                       WHERE token = %s""",
                    (token,)
                )
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"Error getting token data: {e}")
            return None
        finally:
            cursor.close()
    
    def delete_token(self, token: str) -> bool:
        """Remove token do banco"""
        cursor = self._get_cursor()
        if not cursor:
            return False
        
        try:
            cursor.execute("DELETE FROM cpf_tokens WHERE token = %s", (token,))
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting token: {e}")
            return False
        finally:
            cursor.close()
    
    def log_access(self, token: str, action: str, purpose: Optional[str] = None,
                   user_id: Optional[str] = None, ip_address: Optional[str] = None,
                   metadata: Optional[Dict] = None) -> bool:
        """Registra acesso ao token (auditoria LGPD)"""
        cursor = self._get_cursor()
        if not cursor:
            return False
        
        try:
            cursor.execute("""
                INSERT INTO cpf_access_log (token, action, purpose, user_id, ip_address, metadata)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            """, (token, action, purpose, user_id, ip_address, json.dumps(metadata or {})))
            return True
        except Exception as e:
            logger.error(f"Error logging access: {e}")
            return False
        finally:
            cursor.close()
    
    def get_access_log(self, token: str, limit: int = 100) -> List[Dict]:
        """Retorna log de acesso de um token"""
        cursor = self._get_cursor()
        if not cursor:
            return []
        
        try:
            cursor.execute(
                """SELECT * FROM cpf_access_log 
                   WHERE token = %s 
                   ORDER BY accessed_at DESC 
                   LIMIT %s""",
                (token, limit)
            )
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting access log: {e}")
            return []
        finally:
            cursor.close()
    
    def get_token_count(self) -> int:
        """Retorna contagem de tokens ativos"""
        cursor = self._get_cursor()
        if not cursor:
            return 0
        
        try:
            cursor.execute(
                "SELECT COUNT(*) as count FROM cpf_tokens WHERE expires_at IS NULL OR expires_at > NOW()"
            )
            row = cursor.fetchone()
            return row['count'] if row else 0
        except Exception as e:
            logger.error(f"Error getting token count: {e}")
            return 0
        finally:
            cursor.close()
    
    def cleanup_expired(self) -> int:
        """Remove tokens expirados"""
        cursor = self._get_cursor()
        if not cursor:
            return 0
        
        try:
            cursor.execute("DELETE FROM cpf_tokens WHERE expires_at < NOW()")
            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} expired CPF tokens")
            return deleted
        except Exception as e:
            logger.error(f"Error cleaning up expired tokens: {e}")
            return 0
        finally:
            cursor.close()
    
    def load_all_tokens(self) -> List[Dict]:
        """Carrega todos os tokens (para inicialização do cache em memória)"""
        cursor = self._get_cursor()
        if not cursor:
            return []
        
        try:
            cursor.execute(
                """SELECT token, encrypted_cpf, cpf_hash, created_at, expires_at, access_count, last_accessed
                   FROM cpf_tokens 
                   WHERE expires_at IS NULL OR expires_at > NOW()"""
            )
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error loading all tokens: {e}")
            return []
        finally:
            cursor.close()


_cpf_persistence: Optional[CPFPersistence] = None


def get_cpf_persistence() -> CPFPersistence:
    """Retorna instância singleton de persistência CPF"""
    global _cpf_persistence
    if _cpf_persistence is None:
        _cpf_persistence = CPFPersistence()
    return _cpf_persistence
