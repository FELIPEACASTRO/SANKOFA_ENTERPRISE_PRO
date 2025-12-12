#!/usr/bin/env python3
"""
Sankofa Enterprise Pro - Compliance LGPD (Lei Geral de Proteção de Dados)

V002 FIX: Implementação REAL de DSR (Data Subject Request)
- Busca dados reais do PostgreSQL
- Suporte a ACCESS, DELETE, PORTABILITY, RECTIFICATION
- Audit trail de todas as operações DSR
- Anonimização reversível para portabilidade
"""

import logging
import hashlib
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

# PostgreSQL
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


class DSRType(Enum):
    """Tipos de Data Subject Request conforme LGPD Art. 18"""
    ACCESS = "access"           # Art. 18, II - Acesso aos dados
    DELETE = "delete"           # Art. 18, VI - Eliminação
    PORTABILITY = "portability" # Art. 18, V - Portabilidade
    RECTIFICATION = "rectification"  # Art. 18, III - Correção
    ANONYMIZATION = "anonymization"  # Art. 18, IV - Anonimização
    REVOKE_CONSENT = "revoke_consent"  # Art. 18, IX - Revogação


class DSRStatus(Enum):
    """Status de processamento do DSR"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"


class LgpdCompliance:
    """
    Implementa compliance LGPD com operações DSR reais

    V002 FIX: Todas as operações agora consultam o banco de dados real
    """

    def __init__(self):
        self._database_url = os.environ.get("DATABASE_URL")
        self._use_postgres = POSTGRES_AVAILABLE and self._database_url is not None
        self._init_dsr_tables()

    def _get_connection(self):
        """Retorna conexão PostgreSQL"""
        if not self._use_postgres:
            raise RuntimeError("PostgreSQL não disponível - configure DATABASE_URL")
        return psycopg2.connect(self._database_url, cursor_factory=RealDictCursor)

    def _init_dsr_tables(self):
        """Inicializa tabelas para rastreamento de DSR"""
        if not self._use_postgres:
            logger.warning("PostgreSQL não disponível - DSR tables não criadas")
            return

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Tabela de requisições DSR
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS lgpd_dsr_requests (
                            id SERIAL PRIMARY KEY,
                            request_id VARCHAR(100) UNIQUE NOT NULL,
                            subject_identifier VARCHAR(255) NOT NULL,
                            subject_type VARCHAR(50) NOT NULL,
                            request_type VARCHAR(50) NOT NULL,
                            status VARCHAR(50) NOT NULL DEFAULT 'pending',
                            requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            processed_at TIMESTAMP,
                            completed_at TIMESTAMP,
                            response_data JSONB,
                            error_message TEXT,
                            processed_by VARCHAR(100),
                            ip_address VARCHAR(45),
                            audit_notes TEXT
                        )
                    """)

                    # Tabela de dados exportados (para portabilidade)
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS lgpd_data_exports (
                            id SERIAL PRIMARY KEY,
                            dsr_request_id INTEGER REFERENCES lgpd_dsr_requests(id),
                            export_format VARCHAR(20) NOT NULL,
                            export_data JSONB NOT NULL,
                            file_hash VARCHAR(64),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            expires_at TIMESTAMP,
                            downloaded BOOLEAN DEFAULT FALSE,
                            download_count INTEGER DEFAULT 0
                        )
                    """)

                    # Índices
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_dsr_subject
                        ON lgpd_dsr_requests(subject_identifier)
                    """)
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_dsr_status
                        ON lgpd_dsr_requests(status)
                    """)

                    conn.commit()
                    logger.info("LGPD DSR tables criadas/verificadas")
        except Exception as e:
            logger.error(f"Erro ao inicializar DSR tables: {e}")

    def anonymize_data_for_sharing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonimiza dados pessoais antes do compartilhamento

        Args:
            data: Dicionário com dados a anonimizar

        Returns:
            Dicionário com dados anonimizados
        """
        anonymized_data = data.copy()

        # Campos PII a serem hasheados
        pii_fields = [
            "cpf", "user_document", "destination_owner_document",
            "customer_cpf", "sender_cpf", "receiver_cpf"
        ]

        # Campos a serem mascarados (parcialmente visíveis)
        mask_fields = ["email", "phone", "telefone", "celular"]

        for field in pii_fields:
            if field in anonymized_data and anonymized_data[field]:
                anonymized_data[field] = self.hash_data(str(anonymized_data[field]))

        for field in mask_fields:
            if field in anonymized_data and anonymized_data[field]:
                anonymized_data[field] = self._mask_field(str(anonymized_data[field]), field)

        logger.info("Dados anonimizados para compartilhamento LGPD")
        return anonymized_data

    def _mask_field(self, value: str, field_type: str) -> str:
        """Mascara parcialmente um campo"""
        if field_type == "email" and "@" in value:
            local, domain = value.split("@", 1)
            return f"{local[:2]}***@{domain}"
        elif field_type in ["phone", "telefone", "celular"]:
            # Mostra apenas últimos 4 dígitos
            digits = "".join(c for c in value if c.isdigit())
            return f"***-{digits[-4:]}" if len(digits) >= 4 else "***"
        return f"{value[:2]}***"

    def hash_data(self, data_string: str) -> str:
        """Gera hash SHA-256 irreversível"""
        return hashlib.sha256(data_string.encode()).hexdigest()

    def process_data_subject_request(
        self,
        request_type: str,
        subject_id: str,
        subject_type: str = "cpf",
        requester_ip: Optional[str] = None,
        additional_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        V002 FIX: Processa DSR com dados REAIS do banco de dados

        Args:
            request_type: Tipo de requisição (ACCESS, DELETE, PORTABILITY, etc.)
            subject_id: Identificador do titular (CPF, email, user_id)
            subject_type: Tipo do identificador
            requester_ip: IP do requisitante
            additional_data: Dados adicionais para a requisição

        Returns:
            Resultado da operação DSR
        """
        import uuid
        request_id = f"DSR-{uuid.uuid4().hex[:12].upper()}"

        logger.info(f"Processando DSR {request_id}: {request_type} para {subject_type}:{subject_id[:4]}***")

        if not self._use_postgres:
            return {
                "success": False,
                "request_id": request_id,
                "error": "PostgreSQL não disponível - DSR requer banco de dados"
            }

        try:
            # Registra a requisição
            self._register_dsr_request(
                request_id, subject_id, subject_type,
                request_type, requester_ip
            )

            # Processa conforme o tipo
            request_type_upper = request_type.upper()

            if request_type_upper == "ACCESS":
                result = self._process_access_request(subject_id, subject_type)
            elif request_type_upper == "DELETE":
                result = self._process_delete_request(subject_id, subject_type)
            elif request_type_upper == "PORTABILITY":
                result = self._process_portability_request(
                    request_id, subject_id, subject_type
                )
            elif request_type_upper == "RECTIFICATION":
                result = self._process_rectification_request(
                    subject_id, subject_type, additional_data
                )
            elif request_type_upper == "ANONYMIZATION":
                result = self._process_anonymization_request(subject_id, subject_type)
            elif request_type_upper == "REVOKE_CONSENT":
                result = self._process_revoke_consent(subject_id, subject_type)
            else:
                raise ValueError(f"Tipo de DSR desconhecido: {request_type}")

            # Atualiza status da requisição
            self._update_dsr_status(
                request_id,
                DSRStatus.COMPLETED.value,
                result
            )

            result["request_id"] = request_id
            result["success"] = True

            logger.info(f"DSR {request_id} concluído com sucesso")
            return result

        except Exception as e:
            logger.error(f"Erro no DSR {request_id}: {e}")
            self._update_dsr_status(request_id, DSRStatus.FAILED.value, error=str(e))
            return {
                "success": False,
                "request_id": request_id,
                "error": str(e)
            }

    def _register_dsr_request(
        self,
        request_id: str,
        subject_id: str,
        subject_type: str,
        request_type: str,
        ip_address: Optional[str]
    ):
        """Registra requisição DSR no banco"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO lgpd_dsr_requests
                    (request_id, subject_identifier, subject_type, request_type, status, ip_address)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (request_id, self.hash_data(subject_id), subject_type,
                      request_type, DSRStatus.PROCESSING.value, ip_address))
                conn.commit()

    def _update_dsr_status(
        self,
        request_id: str,
        status: str,
        response_data: Optional[Dict] = None,
        error: Optional[str] = None
    ):
        """Atualiza status do DSR"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE lgpd_dsr_requests
                    SET status = %s,
                        response_data = %s,
                        error_message = %s,
                        completed_at = CASE WHEN %s IN ('completed', 'failed') THEN CURRENT_TIMESTAMP ELSE completed_at END
                    WHERE request_id = %s
                """, (status, json.dumps(response_data) if response_data else None,
                      error, status, request_id))
                conn.commit()

    def _process_access_request(
        self,
        subject_id: str,
        subject_type: str
    ) -> Dict[str, Any]:
        """
        V002 FIX: Busca dados REAIS do titular no banco de dados

        Consulta todas as tabelas relevantes para coletar dados do titular
        """
        collected_data = {
            "subject_type": subject_type,
            "collected_at": datetime.now().isoformat(),
            "data_categories": {}
        }

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                # 1. Buscar transações do titular
                cursor.execute("""
                    SELECT transaction_id, amount, channel, created_at, is_fraud, risk_score
                    FROM transactions
                    WHERE customer_cpf = %s OR sender_cpf = %s
                    ORDER BY created_at DESC
                    LIMIT 100
                """, (subject_id, subject_id))
                transactions = cursor.fetchall()

                if transactions:
                    collected_data["data_categories"]["transactions"] = {
                        "count": len(transactions),
                        "data": [dict(t) for t in transactions]
                    }

                # 2. Buscar análises de fraude
                cursor.execute("""
                    SELECT fa.transaction_id, fa.fraud_probability, fa.risk_level,
                           fa.detection_reason, fa.analyzed_at
                    FROM fraud_analysis fa
                    JOIN transactions t ON fa.transaction_id = t.transaction_id
                    WHERE t.customer_cpf = %s
                    ORDER BY fa.analyzed_at DESC
                    LIMIT 50
                """, (subject_id,))
                analyses = cursor.fetchall()

                if analyses:
                    collected_data["data_categories"]["fraud_analyses"] = {
                        "count": len(analyses),
                        "data": [dict(a) for a in analyses]
                    }

                # 3. Buscar dados de usuário no sistema de segurança
                cursor.execute("""
                    SELECT id, username, email, created_at, last_login, is_active
                    FROM security_users
                    WHERE email LIKE %s OR username = %s
                """, (f"%{subject_id}%", subject_id))
                users = cursor.fetchall()

                if users:
                    collected_data["data_categories"]["user_accounts"] = {
                        "count": len(users),
                        "data": [dict(u) for u in users]
                    }

                # 4. Buscar logs de auditoria
                cursor.execute("""
                    SELECT action, resource, details, created_at, ip_address
                    FROM security_audit_log
                    WHERE user_id IN (
                        SELECT id FROM security_users
                        WHERE email LIKE %s OR username = %s
                    )
                    ORDER BY created_at DESC
                    LIMIT 100
                """, (f"%{subject_id}%", subject_id))
                audit_logs = cursor.fetchall()

                if audit_logs:
                    collected_data["data_categories"]["audit_logs"] = {
                        "count": len(audit_logs),
                        "data": [dict(a) for a in audit_logs]
                    }

        # Calcular totais
        total_records = sum(
            cat.get("count", 0)
            for cat in collected_data["data_categories"].values()
        )

        return {
            "message": "Dados do titular recuperados com sucesso",
            "total_records": total_records,
            "categories_found": list(collected_data["data_categories"].keys()),
            "data": collected_data
        }

    def _process_delete_request(
        self,
        subject_id: str,
        subject_type: str
    ) -> Dict[str, Any]:
        """
        V002 FIX: Executa exclusão/anonimização REAL dos dados

        Conforme LGPD, alguns dados devem ser retidos (ex: obrigações legais)
        """
        deleted_counts = {}
        retained_counts = {}

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                # 1. Anonimizar transações (não deletar por obrigação BACEN - 5 anos)
                cursor.execute("""
                    UPDATE transactions
                    SET customer_cpf = %s,
                        sender_cpf = CASE WHEN sender_cpf = %s THEN %s ELSE sender_cpf END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE customer_cpf = %s OR sender_cpf = %s
                """, (
                    self.hash_data(subject_id),
                    subject_id, self.hash_data(subject_id),
                    subject_id, subject_id
                ))
                retained_counts["transactions_anonymized"] = cursor.rowcount

                # 2. Deletar dados de usuário (após período de retenção)
                cursor.execute("""
                    UPDATE security_users
                    SET email = %s,
                        username = %s,
                        is_active = FALSE,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE email LIKE %s OR username = %s
                """, (
                    f"deleted_{self.hash_data(subject_id)[:8]}@deleted.local",
                    f"deleted_{self.hash_data(subject_id)[:8]}",
                    f"%{subject_id}%", subject_id
                ))
                deleted_counts["user_accounts_anonymized"] = cursor.rowcount

                # 3. Anonimizar logs de auditoria (reter por compliance)
                cursor.execute("""
                    UPDATE security_audit_log
                    SET details = REGEXP_REPLACE(details, %s, '[REDACTED]', 'gi'),
                        ip_address = '0.0.0.0'
                    WHERE user_id IN (
                        SELECT id FROM security_users
                        WHERE email LIKE %s OR username LIKE %s
                    )
                """, (subject_id, f"%deleted_{self.hash_data(subject_id)[:8]}%",
                      f"%deleted_{self.hash_data(subject_id)[:8]}%"))
                retained_counts["audit_logs_redacted"] = cursor.rowcount

                conn.commit()

        return {
            "message": f"Dados do titular processados para exclusão/anonimização",
            "deleted": deleted_counts,
            "retained_anonymized": retained_counts,
            "retention_note": "Alguns dados foram anonimizados em vez de deletados devido a obrigações legais (BACEN - 5 anos)"
        }

    def _process_portability_request(
        self,
        request_id: str,
        subject_id: str,
        subject_type: str
    ) -> Dict[str, Any]:
        """
        V002 FIX: Gera pacote de portabilidade em formato estruturado
        """
        # Primeiro, buscar todos os dados
        access_result = self._process_access_request(subject_id, subject_type)

        if access_result.get("total_records", 0) == 0:
            return {
                "message": "Nenhum dado encontrado para portabilidade",
                "export_available": False
            }

        # Preparar dados para exportação
        export_data = {
            "export_metadata": {
                "request_id": request_id,
                "generated_at": datetime.now().isoformat(),
                "format": "json",
                "lgpd_article": "Art. 18, V"
            },
            "subject_data": access_result["data"]
        }

        # Calcular hash para integridade
        data_hash = hashlib.sha256(
            json.dumps(export_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        # Salvar exportação no banco
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO lgpd_data_exports
                    (dsr_request_id, export_format, export_data, file_hash, expires_at)
                    SELECT id, 'json', %s, %s, CURRENT_TIMESTAMP + INTERVAL '7 days'
                    FROM lgpd_dsr_requests WHERE request_id = %s
                """, (json.dumps(export_data, default=str), data_hash, request_id))
                conn.commit()

        return {
            "message": "Dados preparados para portabilidade",
            "export_available": True,
            "file_hash": data_hash,
            "expires_in_days": 7,
            "total_records": access_result["total_records"],
            "download_endpoint": f"/api/lgpd/export/{request_id}"
        }

    def _process_rectification_request(
        self,
        subject_id: str,
        subject_type: str,
        rectification_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        V002 FIX: Processa correção de dados
        """
        if not rectification_data:
            return {
                "message": "Nenhum dado fornecido para retificação",
                "rectified": False
            }

        rectified_fields = []

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                # Permitir correção apenas de campos específicos
                allowed_fields = ["email", "phone", "name"]

                for field, new_value in rectification_data.items():
                    if field in allowed_fields:
                        # Atualizar no sistema de segurança
                        if field == "email":
                            cursor.execute("""
                                UPDATE security_users
                                SET email = %s, updated_at = CURRENT_TIMESTAMP
                                WHERE email LIKE %s OR username = %s
                            """, (new_value, f"%{subject_id}%", subject_id))

                            if cursor.rowcount > 0:
                                rectified_fields.append(field)

                conn.commit()

        return {
            "message": "Solicitação de retificação processada",
            "rectified_fields": rectified_fields,
            "rectified": len(rectified_fields) > 0
        }

    def _process_anonymization_request(
        self,
        subject_id: str,
        subject_type: str
    ) -> Dict[str, Any]:
        """
        V002 FIX: Anonimiza dados mantendo para análise estatística
        """
        anonymized_hash = self.hash_data(subject_id)

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                # Anonimizar em transações
                cursor.execute("""
                    UPDATE transactions
                    SET customer_cpf = %s,
                        sender_cpf = CASE WHEN sender_cpf = %s THEN %s ELSE sender_cpf END
                    WHERE customer_cpf = %s OR sender_cpf = %s
                """, (anonymized_hash, subject_id, anonymized_hash, subject_id, subject_id))
                txn_count = cursor.rowcount

                conn.commit()

        return {
            "message": "Dados anonimizados com sucesso",
            "records_anonymized": txn_count,
            "anonymization_method": "SHA-256 hash irreversível"
        }

    def _process_revoke_consent(
        self,
        subject_id: str,
        subject_type: str
    ) -> Dict[str, Any]:
        """
        V002 FIX: Revoga consentimento e desativa processamento
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                # Desativar conta de usuário
                cursor.execute("""
                    UPDATE security_users
                    SET is_active = FALSE,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE email LIKE %s OR username = %s
                """, (f"%{subject_id}%", subject_id))
                deactivated = cursor.rowcount

                # Invalidar todas as sessões
                cursor.execute("""
                    DELETE FROM security_sessions
                    WHERE user_id IN (
                        SELECT id FROM security_users
                        WHERE email LIKE %s OR username = %s
                    )
                """, (f"%{subject_id}%", subject_id))
                sessions_revoked = cursor.rowcount

                conn.commit()

        return {
            "message": "Consentimento revogado com sucesso",
            "accounts_deactivated": deactivated,
            "sessions_revoked": sessions_revoked,
            "note": "O titular não poderá mais acessar o sistema até fornecer novo consentimento"
        }

    def get_dsr_status(self, request_id: str) -> Dict[str, Any]:
        """Consulta status de uma requisição DSR"""
        if not self._use_postgres:
            return {"error": "PostgreSQL não disponível"}

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT request_id, request_type, status, requested_at,
                           processed_at, completed_at, error_message
                    FROM lgpd_dsr_requests
                    WHERE request_id = %s
                """, (request_id,))
                result = cursor.fetchone()

                if not result:
                    return {"error": "Requisição não encontrada"}

                return dict(result)
