"""
Sankofa Enterprise Pro - CPF Tokenization PostgreSQL Persistence
Persistência de tokens CPF em PostgreSQL para produção (thread-safe)
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
    Camada de persistência PostgreSQL para tokenização de CPF (thread-safe)
    
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
            logger.info("CPF persistence: PostgreSQL available")
        except Exception as e:
            logger.warning(f"CPF persistence: PostgreSQL not available - {e}")
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
            elif fetch == "scalar":
                result = cursor.fetchone()
                return list(result.values())[0] if result else None
            else:
                return True
        except Exception as e:
            logger.error(f"CPF persistence error: {e}")
            return None if fetch != "none" else False
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def save_token(self, token: str, encrypted_cpf: bytes, cpf_hash: str,
                   expires_at: Optional[datetime] = None, 
                   metadata: Optional[Dict] = None) -> bool:
        """Salva token no banco de dados"""
        result = self._execute("""
            INSERT INTO cpf_tokens (token, encrypted_cpf, cpf_hash, expires_at, metadata)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (token) DO UPDATE SET
                expires_at = EXCLUDED.expires_at,
                metadata = EXCLUDED.metadata
        """, (token, encrypted_cpf, cpf_hash, expires_at, json.dumps(metadata or {})))
        return result is True
    
    def get_token_by_hash(self, cpf_hash: str) -> Optional[Dict]:
        """Busca token pelo hash do CPF"""
        return self._execute(
            """SELECT * FROM cpf_tokens 
               WHERE cpf_hash = %s AND (expires_at IS NULL OR expires_at > NOW())""",
            (cpf_hash,),
            fetch="one"
        )
    
    def get_token_data(self, token: str) -> Optional[Dict]:
        """Busca dados do token e atualiza contadores"""
        result = self._execute(
            """SELECT * FROM cpf_tokens 
               WHERE token = %s AND (expires_at IS NULL OR expires_at > NOW())""",
            (token,),
            fetch="one"
        )
        if result:
            self._execute(
                """UPDATE cpf_tokens 
                   SET access_count = access_count + 1, last_accessed = NOW() 
                   WHERE token = %s""",
                (token,)
            )
        return result
    
    def delete_token(self, token: str) -> bool:
        """Remove token do banco"""
        result = self._execute(
            "DELETE FROM cpf_tokens WHERE token = %s",
            (token,),
            fetch="rowcount"
        )
        return result is not None and result > 0
    
    def log_access(self, token: str, action: str, purpose: Optional[str] = None,
                   user_id: Optional[str] = None, ip_address: Optional[str] = None,
                   metadata: Optional[Dict] = None) -> bool:
        """Registra acesso ao token (auditoria LGPD)"""
        result = self._execute("""
            INSERT INTO cpf_access_log (token, action, purpose, user_id, ip_address, metadata)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        """, (token, action, purpose, user_id, ip_address, json.dumps(metadata or {})))
        return result is True
    
    def get_access_log(self, token: str, limit: int = 100) -> List[Dict]:
        """Retorna log de acesso de um token"""
        result = self._execute(
            """SELECT * FROM cpf_access_log 
               WHERE token = %s 
               ORDER BY accessed_at DESC 
               LIMIT %s""",
            (token, limit),
            fetch="all"
        )
        return result or []
    
    def get_token_count(self) -> int:
        """Retorna contagem de tokens ativos"""
        result = self._execute(
            "SELECT COUNT(*) as count FROM cpf_tokens WHERE expires_at IS NULL OR expires_at > NOW()",
            fetch="scalar"
        )
        return result or 0
    
    def cleanup_expired(self) -> int:
        """Remove tokens expirados"""
        result = self._execute(
            "DELETE FROM cpf_tokens WHERE expires_at < NOW()",
            fetch="rowcount"
        )
        deleted = result or 0
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} expired CPF tokens")
        return deleted
    
    def load_all_tokens(self) -> List[Dict]:
        """Carrega todos os tokens (para inicialização do cache em memória)"""
        result = self._execute(
            """SELECT token, encrypted_cpf, cpf_hash, created_at, expires_at, access_count, last_accessed
               FROM cpf_tokens 
               WHERE expires_at IS NULL OR expires_at > NOW()""",
            fetch="all"
        )
        return result or []


_cpf_persistence: Optional[CPFPersistence] = None


def get_cpf_persistence() -> CPFPersistence:
    """Retorna instância singleton de persistência CPF"""
    global _cpf_persistence
    if _cpf_persistence is None:
        _cpf_persistence = CPFPersistence()
    return _cpf_persistence
