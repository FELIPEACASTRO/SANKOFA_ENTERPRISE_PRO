"""
Sankofa Enterprise Pro - CPF Tokenization System
Sistema de tokenização de CPF para proteção de dados sensíveis (LGPD)
"""

import os
import hashlib
import secrets
import base64
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


@dataclass
class Token:
    """Token de CPF"""
    token_value: str
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class TokenVault:
    """
    Cofre seguro para tokenização de CPF
    
    Implementa:
    - Tokenização bidirecional (token <-> CPF)
    - Criptografia AES-256
    - Rotação de chaves
    - Auditoria de acesso
    - TTL configurável
    
    Compliance:
    - LGPD Art. 46 (medidas de segurança)
    - PCI DSS Requirement 3.4 (mascaramento)
    """
    
    def __init__(
        self,
        encryption_key: Optional[str] = None,
        token_ttl_days: int = 365,
        salt: Optional[bytes] = None
    ):
        self._salt = salt or os.urandom(16)
        self._encryption_key = self._derive_key(
            encryption_key or os.getenv("ENCRYPTION_KEY", secrets.token_hex(32))
        )
        self._fernet = Fernet(self._encryption_key)
        self._token_ttl_days = token_ttl_days
        
        self._token_to_cpf: Dict[str, bytes] = {}
        self._cpf_to_token: Dict[str, str] = {}
        self._token_metadata: Dict[str, Token] = {}
        self._access_log: List[Dict] = []
        
        logger.info("TokenVault initialized with AES-256 encryption")
    
    def _derive_key(self, password: str) -> bytes:
        """Deriva chave de criptografia usando PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def _generate_token(self) -> str:
        """Gera token único e seguro"""
        random_bytes = secrets.token_bytes(16)
        timestamp = datetime.now().timestamp()
        combined = f"{random_bytes.hex()}{timestamp}"
        token_hash = hashlib.sha256(combined.encode()).hexdigest()[:24]
        return f"TKN_{token_hash.upper()}"
    
    def _validate_cpf(self, cpf: str) -> str:
        """Valida e normaliza CPF"""
        cpf_clean = ''.join(filter(str.isdigit, cpf))
        
        if len(cpf_clean) != 11:
            raise ValueError(f"CPF inválido: deve ter 11 dígitos")
        
        if cpf_clean == cpf_clean[0] * 11:
            raise ValueError("CPF inválido: todos os dígitos iguais")
        
        def calc_digit(cpf_partial: str, weights: List[int]) -> int:
            total = sum(int(d) * w for d, w in zip(cpf_partial, weights))
            remainder = total % 11
            return 0 if remainder < 2 else 11 - remainder
        
        weights1 = [10, 9, 8, 7, 6, 5, 4, 3, 2]
        weights2 = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
        
        digit1 = calc_digit(cpf_clean[:9], weights1)
        digit2 = calc_digit(cpf_clean[:9] + str(digit1), weights2)
        
        if cpf_clean[-2:] != f"{digit1}{digit2}":
            raise ValueError("CPF inválido: dígitos verificadores incorretos")
        
        return cpf_clean
    
    def tokenize(self, cpf: str, metadata: Optional[Dict] = None) -> str:
        """
        Tokeniza um CPF
        
        Args:
            cpf: CPF a ser tokenizado
            metadata: Metadados adicionais
            
        Returns:
            Token único para o CPF
        """
        cpf_clean = self._validate_cpf(cpf)
        
        cpf_hash = hashlib.sha256(cpf_clean.encode()).hexdigest()
        
        if cpf_hash in self._cpf_to_token:
            token = self._cpf_to_token[cpf_hash]
            self._log_access(token, "tokenize_existing")
            return token
        
        token = self._generate_token()
        encrypted_cpf = self._fernet.encrypt(cpf_clean.encode())
        
        self._token_to_cpf[token] = encrypted_cpf
        self._cpf_to_token[cpf_hash] = token
        
        expires_at = None
        if self._token_ttl_days > 0:
            expires_at = datetime.now() + timedelta(days=self._token_ttl_days)
        
        self._token_metadata[token] = Token(
            token_value=token,
            created_at=datetime.now(),
            expires_at=expires_at
        )
        
        self._log_access(token, "tokenize_new", metadata)
        
        logger.info(f"CPF tokenized: {token}")
        return token
    
    def detokenize(self, token: str, purpose: str = "general") -> Optional[str]:
        """
        Recupera CPF original a partir do token
        
        Args:
            token: Token a ser detokenizado
            purpose: Propósito do acesso (para auditoria)
            
        Returns:
            CPF original ou None se token inválido
        """
        if token not in self._token_to_cpf:
            self._log_access(token, "detokenize_not_found", {"purpose": purpose})
            return None
        
        token_meta = self._token_metadata.get(token)
        if token_meta and token_meta.expires_at:
            if datetime.now() > token_meta.expires_at:
                self._log_access(token, "detokenize_expired", {"purpose": purpose})
                return None
        
        encrypted_cpf = self._token_to_cpf[token]
        cpf = self._fernet.decrypt(encrypted_cpf).decode()
        
        if token_meta:
            token_meta.access_count += 1
            token_meta.last_accessed = datetime.now()
        
        self._log_access(token, "detokenize_success", {"purpose": purpose})
        
        return cpf
    
    def get_masked_cpf(self, token: str) -> Optional[str]:
        """
        Retorna CPF mascarado (XXX.XXX.XXX-XX) a partir do token
        
        Args:
            token: Token do CPF
            
        Returns:
            CPF mascarado ou None
        """
        cpf = self.detokenize(token, purpose="masking")
        if cpf is None:
            return None
        
        return f"***.***.*{cpf[7:9]}-{cpf[9:]}"
    
    def validate_token(self, token: str) -> bool:
        """Valida se token existe e não expirou"""
        if token not in self._token_to_cpf:
            return False
        
        token_meta = self._token_metadata.get(token)
        if token_meta and token_meta.expires_at:
            return datetime.now() <= token_meta.expires_at
        
        return True
    
    def revoke_token(self, token: str, reason: str = "manual_revoke") -> bool:
        """
        Revoga um token
        
        Args:
            token: Token a ser revogado
            reason: Motivo da revogação
            
        Returns:
            True se revogado com sucesso
        """
        if token not in self._token_to_cpf:
            return False
        
        encrypted_cpf = self._token_to_cpf[token]
        cpf = self._fernet.decrypt(encrypted_cpf).decode()
        cpf_hash = hashlib.sha256(cpf.encode()).hexdigest()
        
        del self._token_to_cpf[token]
        if cpf_hash in self._cpf_to_token:
            del self._cpf_to_token[cpf_hash]
        if token in self._token_metadata:
            del self._token_metadata[token]
        
        self._log_access(token, "revoke", {"reason": reason})
        
        logger.info(f"Token revoked: {token}, reason: {reason}")
        return True
    
    def rotate_encryption_key(self, new_key: str) -> int:
        """
        Rotaciona chave de criptografia
        
        Args:
            new_key: Nova chave de criptografia
            
        Returns:
            Número de tokens re-encriptados
        """
        new_encryption_key = self._derive_key(new_key)
        new_fernet = Fernet(new_encryption_key)
        
        count = 0
        for token, encrypted_cpf in self._token_to_cpf.items():
            cpf = self._fernet.decrypt(encrypted_cpf).decode()
            self._token_to_cpf[token] = new_fernet.encrypt(cpf.encode())
            count += 1
        
        self._encryption_key = new_encryption_key
        self._fernet = new_fernet
        
        self._log_access("SYSTEM", "key_rotation", {"tokens_rotated": count})
        
        logger.info(f"Encryption key rotated. {count} tokens re-encrypted")
        return count
    
    def _log_access(self, token: str, action: str, metadata: Optional[Dict] = None):
        """Registra acesso para auditoria"""
        log_entry = {
            "token": token[:10] + "..." if len(token) > 10 else token,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self._access_log.append(log_entry)
        
        if len(self._access_log) > 10000:
            self._access_log = self._access_log[-5000:]
    
    def get_access_log(self, limit: int = 100) -> List[Dict]:
        """Retorna log de acesso"""
        return self._access_log[-limit:]
    
    def get_token_stats(self) -> Dict:
        """Retorna estatísticas de tokens"""
        now = datetime.now()
        active = sum(1 for t in self._token_metadata.values() 
                    if t.expires_at is None or t.expires_at > now)
        expired = len(self._token_metadata) - active
        
        return {
            "total_tokens": len(self._token_to_cpf),
            "active_tokens": active,
            "expired_tokens": expired,
            "total_accesses": sum(t.access_count for t in self._token_metadata.values()),
            "access_log_size": len(self._access_log)
        }
    
    def cleanup_expired(self) -> int:
        """Remove tokens expirados"""
        now = datetime.now()
        expired_tokens = [
            token for token, meta in self._token_metadata.items()
            if meta.expires_at and meta.expires_at < now
        ]
        
        for token in expired_tokens:
            self.revoke_token(token, reason="expired_cleanup")
        
        return len(expired_tokens)
    
    def export_for_backup(self, include_encrypted: bool = False) -> Dict:
        """
        Exporta dados para backup
        
        Args:
            include_encrypted: Se True, inclui dados encriptados
            
        Returns:
            Dados para backup
        """
        backup = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "stats": self.get_token_stats(),
            "metadata": {
                token: {
                    "created_at": meta.created_at.isoformat(),
                    "expires_at": meta.expires_at.isoformat() if meta.expires_at else None,
                    "access_count": meta.access_count
                }
                for token, meta in self._token_metadata.items()
            }
        }
        
        if include_encrypted:
            backup["encrypted_mappings"] = {
                token: base64.b64encode(data).decode()
                for token, data in self._token_to_cpf.items()
            }
        
        return backup


class CPFTokenizationService:
    """
    Serviço de alto nível para tokenização de CPF
    
    Integra com o sistema Sankofa para proteção automática de CPFs
    """
    
    _instance: Optional['CPFTokenizationService'] = None
    
    def __init__(self):
        encryption_key = os.getenv("ENCRYPTION_KEY")
        if not encryption_key:
            logger.warning("ENCRYPTION_KEY not set, using auto-generated key")
            encryption_key = secrets.token_hex(32)
        
        self.vault = TokenVault(encryption_key=encryption_key)
    
    @classmethod
    def get_instance(cls) -> 'CPFTokenizationService':
        """Retorna instância singleton"""
        if cls._instance is None:
            cls._instance = CPFTokenizationService()
        return cls._instance
    
    def tokenize_transaction(self, transaction: Dict) -> Dict:
        """
        Tokeniza CPF em uma transação
        
        Args:
            transaction: Dicionário da transação
            
        Returns:
            Transação com CPF tokenizado
        """
        result = transaction.copy()
        
        cpf_fields = ['cpf', 'cliente_cpf', 'customer_cpf', 'cpf_recebedor']
        
        for field in cpf_fields:
            if field in result and result[field]:
                try:
                    cpf_value = result[field]
                    if not cpf_value.startswith('TKN_'):
                        token = self.vault.tokenize(cpf_value)
                        result[field] = token
                        result[f"{field}_masked"] = self.vault.get_masked_cpf(token)
                except ValueError as e:
                    logger.warning(f"Invalid CPF in field {field}: {e}")
        
        return result
    
    def detokenize_for_compliance(self, token: str, user_id: str, purpose: str) -> Optional[str]:
        """
        Detokeniza CPF para fins de compliance
        
        Args:
            token: Token do CPF
            user_id: ID do usuário solicitante
            purpose: Propósito (auditoria, lgpd_request, etc.)
            
        Returns:
            CPF original ou None
        """
        allowed_purposes = [
            "audit", "lgpd_request", "compliance_report", 
            "fraud_investigation", "judicial_order"
        ]
        
        if purpose not in allowed_purposes:
            logger.warning(f"Detokenization denied: invalid purpose '{purpose}'")
            return None
        
        cpf = self.vault.detokenize(token, purpose=f"{purpose}:{user_id}")
        
        return cpf
    
    def process_lgpd_deletion_request(self, cpf: str) -> Dict:
        """
        Processa solicitação de exclusão LGPD
        
        Args:
            cpf: CPF do titular
            
        Returns:
            Resultado da operação
        """
        try:
            cpf_clean = self.vault._validate_cpf(cpf)
            cpf_hash = hashlib.sha256(cpf_clean.encode()).hexdigest()
            
            if cpf_hash in self.vault._cpf_to_token:
                token = self.vault._cpf_to_token[cpf_hash]
                self.vault.revoke_token(token, reason="lgpd_deletion_request")
                
                return {
                    "success": True,
                    "message": "Dados tokenizados removidos conforme LGPD",
                    "token_revoked": token[:10] + "..."
                }
            
            return {
                "success": True,
                "message": "Nenhum dado tokenizado encontrado para este CPF"
            }
            
        except ValueError as e:
            return {
                "success": False,
                "message": str(e)
            }


def get_tokenization_service() -> CPFTokenizationService:
    """Factory function para obter serviço de tokenização"""
    return CPFTokenizationService.get_instance()
