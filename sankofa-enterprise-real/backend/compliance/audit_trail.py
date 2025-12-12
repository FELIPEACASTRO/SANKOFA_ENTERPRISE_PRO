#!/usr/bin/env python3
"""
Módulo de Trilha de Auditoria para Compliance
Registra todas as ações sensíveis relacionadas a compliance para garantir rastreabilidade.

CORRECAO 10/10: Logs imutaveis com hash chain para detectar alteracoes
"""

import logging
import json
import os
import hashlib
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Configura um logger específico para a trilha de auditoria
audit_logger = logging.getLogger("compliance_audit")

# Usa caminho relativo ao projeto
project_root = Path(__file__).resolve().parent.parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)


class ImmutableAuditHandler(logging.Handler):
    """
    CORRECAO 10/10: Handler de logging imutável com hash chain

    Cada entrada de log inclui o hash da entrada anterior,
    criando uma cadeia que detecta adulteração.
    """

    def __init__(self, filename: Path, mode: str = "a"):
        super().__init__()
        self._lock = threading.RLock()
        self._filename = filename
        self._file = open(filename, mode, encoding="utf-8")
        self._previous_hash = self._calculate_initial_hash()
        self._entry_count = self._count_existing_entries()

    def _calculate_initial_hash(self) -> str:
        """Calcula hash inicial baseado no conteudo existente do arquivo"""
        if self._filename.exists() and self._filename.stat().st_size > 0:
            with open(self._filename, "rb") as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        return hashlib.sha256(b"GENESIS_BLOCK").hexdigest()

    def _count_existing_entries(self) -> int:
        """Conta entradas existentes no arquivo"""
        if self._filename.exists():
            with open(self._filename, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        return 0

    def emit(self, record):
        """Emite registro de log com hash chain"""
        with self._lock:
            try:
                msg = self.format(record)

                # Criar entrada com hash chain
                self._entry_count += 1
                entry = {
                    "sequence": self._entry_count,
                    "timestamp": datetime.utcnow().isoformat(),
                    "previous_hash": self._previous_hash,
                    "data": msg
                }

                # Calcular hash desta entrada
                entry_json = json.dumps(entry, sort_keys=True)
                current_hash = hashlib.sha256(entry_json.encode()).hexdigest()
                entry["hash"] = current_hash

                # Escrever no arquivo
                self._file.write(json.dumps(entry) + "\n")
                self._file.flush()
                os.fsync(self._file.fileno())  # Garantir escrita em disco

                # Atualizar hash anterior
                self._previous_hash = current_hash

            except Exception:
                self.handleError(record)

    def close(self):
        """Fecha o handler de forma segura"""
        with self._lock:
            self._file.close()
        super().close()

    @classmethod
    def verify_integrity(cls, filename: Path) -> Dict[str, Any]:
        """
        Verifica integridade da cadeia de logs

        Returns:
            Dict com resultado da verificacao
        """
        if not filename.exists():
            return {"valid": False, "error": "File not found"}

        valid_entries = 0
        invalid_entries = []
        previous_hash = hashlib.sha256(b"GENESIS_BLOCK").hexdigest()

        with open(filename, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line)

                    # Verificar hash anterior
                    if entry.get("previous_hash") != previous_hash:
                        invalid_entries.append({
                            "line": line_num,
                            "error": "Invalid previous_hash - chain broken"
                        })
                        continue

                    # Verificar hash da entrada
                    stored_hash = entry.pop("hash", None)
                    entry_json = json.dumps(entry, sort_keys=True)
                    calculated_hash = hashlib.sha256(entry_json.encode()).hexdigest()

                    if stored_hash != calculated_hash:
                        invalid_entries.append({
                            "line": line_num,
                            "error": "Hash mismatch - entry tampered"
                        })
                        continue

                    previous_hash = stored_hash
                    valid_entries += 1

                except json.JSONDecodeError:
                    invalid_entries.append({
                        "line": line_num,
                        "error": "Invalid JSON"
                    })

        return {
            "valid": len(invalid_entries) == 0,
            "valid_entries": valid_entries,
            "invalid_entries": invalid_entries,
            "integrity": "VERIFIED" if len(invalid_entries) == 0 else "COMPROMISED"
        }


# Usar handler imutavel em vez de FileHandler simples
immutable_handler = ImmutableAuditHandler(logs_dir / "compliance_audit.log")
formatter = logging.Formatter("%(message)s")  # Formatacao simplificada, dados em JSON
immutable_handler.setFormatter(formatter)
audit_logger.addHandler(immutable_handler)
audit_logger.setLevel(logging.INFO)


class AuditTrail:
    """Classe responsável por registrar trilhas de auditoria de compliance."""

    def log_compliance_action(self, action: str, details: Dict[str, Any], user: str):
        """
        Registra uma ação de compliance na trilha de auditoria.

        Args:
            action: A ação que foi realizada (e.g., "SHARE_FRAUD_DATA", "DSR_ACCESS").
            details: Um dicionário com detalhes relevantes sobre a ação.
            user: O usuário ou sistema que realizou a ação.
        """
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": action,
                "user": user,
                "details": details,
                "status": "SUCCESS",
            }
            # Serializa o dicionário para uma string JSON
            audit_logger.info(json.dumps(log_entry))
        except Exception as e:
            # Log de erro em caso de falha ao registrar a auditoria
            logging.error(f"Falha crítica ao registrar na trilha de auditoria: {e}")


# Exemplo de uso
if __name__ == "__main__":
    # Cria o diretório de logs se não existir
    import os

    if not os.path.exists("/home/ubuntu/sankofa-enterprise-real/logs"):
        os.makedirs("/home/ubuntu/sankofa-enterprise-real/logs")

    audit = AuditTrail()

    logger.info("--- Registrando ações de auditoria ---")

    # Log 1: Compartilhamento de dados
    audit.log_compliance_action(
        action="SHARE_FRAUD_DATA",
        details={"destination": "BACEN", "fraud_id": "FRD123"},
        user="compliance_officer",
    )
    logger.info("Log de compartilhamento de dados registrado.")

    # Log 2: Requisição de titular de dados
    audit.log_compliance_action(
        action="DSR_DELETE", details={"subject_id": "USR456"}, user="data_privacy_team"
    )
    logger.info("Log de requisição de titular de dados registrado.")

    # Log 3: Ação do sistema
    audit.log_compliance_action(
        action="APPLY_DATA_RETENTION", details={"policy": "PCI-DSS-3.1"}, user="system_cron_job"
    )
    logger.info("Log de ação do sistema registrado.")

    logger.info(
        "\nVerifique o arquivo '/home/ubuntu/sankofa-enterprise-real/logs/compliance_audit.log' para ver os registros."
    )
