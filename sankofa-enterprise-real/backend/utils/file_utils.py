#!/usr/bin/env python3
"""
Sankofa Enterprise Pro - File Utilities
Utilitários seguros para manipulação de arquivos

OWASP A01: Broken Access Control - Path Traversal Prevention
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


# Diretórios permitidos para operações de arquivo
ALLOWED_BASE_DIRS: List[Path] = [
    Path(__file__).resolve().parent.parent.parent / "uploads",
    Path(__file__).resolve().parent.parent.parent / "exports",
    Path(__file__).resolve().parent.parent.parent / "reports",
    Path(__file__).resolve().parent.parent.parent / "logs",
]


class SecurityError(ValueError):
    """Exceção de segurança para operações de arquivo."""
    pass


def validate_safe_path(
    user_path: str,
    base_dir: Optional[Path] = None,
    allowed_base: Optional[str] = None,  # Alias para compatibilidade com testes
    allowed_extensions: Optional[List[str]] = None
) -> Path:
    """
    Valida e sanitiza um caminho de arquivo para prevenir Path Traversal.

    OWASP A01: Broken Access Control
    Previne ataques de path traversal como:
    - ../../../etc/passwd
    - ..\\..\\..\\windows\\system32
    - /etc/passwd (absolute paths)

    Args:
        user_path: Caminho fornecido pelo usuário
        base_dir: Diretório base permitido (default: primeiro de ALLOWED_BASE_DIRS)
        allowed_extensions: Lista de extensões permitidas (ex: ['.pdf', '.csv'])

    Returns:
        Path: Caminho seguro e validado

    Raises:
        ValueError: Se o caminho for inválido ou tentar escapar do diretório base
        PermissionError: Se o caminho tentar acessar diretório não permitido
    """
    if not user_path:
        raise SecurityError("Caminho não pode ser vazio")

    # Usar allowed_base se fornecido (alias para base_dir)
    if allowed_base is not None:
        base_dir = Path(allowed_base)

    # Usar primeiro diretório permitido como default
    if base_dir is None:
        base_dir = ALLOWED_BASE_DIRS[0]
        # Criar se não existir
        base_dir.mkdir(parents=True, exist_ok=True)

    # Converter para Path e resolver
    base_dir = Path(base_dir).resolve()

    # Verificar se base_dir está na lista de permitidos
    if not any(base_dir == allowed or base_dir in allowed.parents or allowed in base_dir.parents
               for allowed in ALLOWED_BASE_DIRS):
        # Permitir se base_dir for subdiretório de um permitido
        is_allowed = False
        for allowed in ALLOWED_BASE_DIRS:
            try:
                base_dir.relative_to(allowed)
                is_allowed = True
                break
            except ValueError:
                continue

        if not is_allowed:
            logger.warning(
                f"PATH_TRAVERSAL_ATTEMPT: Tentativa de acesso a diretório não permitido: {base_dir}"
            )
            raise PermissionError(
                f"Diretório base não permitido. Diretórios válidos: {[str(d) for d in ALLOWED_BASE_DIRS]}"
            )

    # Sanitizar o caminho do usuário
    # Remover caracteres perigosos
    sanitized = user_path

    # Bloquear sequências de path traversal
    dangerous_patterns = [
        r'\.\.',           # ..
        r'\.\./',          # ../
        r'\.\\.\\',        # ..\
        r'%2e%2e',         # URL encoded ..
        r'%252e%252e',     # Double URL encoded ..
        r'\.\.%2f',        # Mixed encoding
        r'%2e%2e%2f',      # URL encoded ../
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            logger.warning(
                f"PATH_TRAVERSAL_BLOCKED: Padrão perigoso detectado: {pattern} em {user_path}"
            )
            raise SecurityError(f"Caminho contém sequência não permitida: {pattern}")

    # Remover barras iniciais (previne caminhos absolutos)
    sanitized = sanitized.lstrip('/\\')

    # Construir caminho completo
    try:
        full_path = (base_dir / sanitized).resolve()
    except Exception as e:
        raise SecurityError(f"Caminho inválido: {e}")

    # Verificar se o caminho resultante está dentro do base_dir
    try:
        full_path.relative_to(base_dir)
    except ValueError:
        logger.warning(
            f"PATH_TRAVERSAL_BLOCKED: Tentativa de escapar do diretório base. "
            f"Base: {base_dir}, Tentativa: {full_path}"
        )
        raise SecurityError(
            "Caminho tenta acessar fora do diretório permitido"
        )

    # Verificar extensão se especificada
    if allowed_extensions:
        ext = full_path.suffix.lower()
        if ext not in [e.lower() for e in allowed_extensions]:
            raise SecurityError(
                f"Extensão '{ext}' não permitida. Extensões válidas: {allowed_extensions}"
            )

    logger.debug(f"PATH_VALIDATED: {user_path} -> {full_path}")
    return full_path


def safe_read_file(
    user_path: str,
    base_dir: Optional[Path] = None,
    max_size_mb: float = 10.0
) -> bytes:
    """
    Lê um arquivo de forma segura após validar o caminho.

    Args:
        user_path: Caminho fornecido pelo usuário
        base_dir: Diretório base permitido
        max_size_mb: Tamanho máximo do arquivo em MB

    Returns:
        bytes: Conteúdo do arquivo

    Raises:
        ValueError: Se o caminho for inválido
        FileNotFoundError: Se o arquivo não existir
        PermissionError: Se o arquivo for muito grande
    """
    safe_path = validate_safe_path(user_path, base_dir)

    if not safe_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {user_path}")

    if not safe_path.is_file():
        raise ValueError(f"Caminho não é um arquivo: {user_path}")

    # Verificar tamanho
    file_size = safe_path.stat().st_size
    max_size_bytes = max_size_mb * 1024 * 1024

    if file_size > max_size_bytes:
        raise PermissionError(
            f"Arquivo muito grande: {file_size / 1024 / 1024:.2f}MB "
            f"(máximo: {max_size_mb}MB)"
        )

    return safe_path.read_bytes()


def safe_write_file(
    user_path: str,
    content: bytes,
    base_dir: Optional[Path] = None,
    allowed_extensions: Optional[List[str]] = None
) -> Path:
    """
    Escreve um arquivo de forma segura após validar o caminho.

    Args:
        user_path: Caminho fornecido pelo usuário
        content: Conteúdo a ser escrito
        base_dir: Diretório base permitido
        allowed_extensions: Extensões permitidas

    Returns:
        Path: Caminho do arquivo escrito

    Raises:
        ValueError: Se o caminho for inválido
    """
    safe_path = validate_safe_path(user_path, base_dir, allowed_extensions)

    # Criar diretório pai se não existir
    safe_path.parent.mkdir(parents=True, exist_ok=True)

    safe_path.write_bytes(content)
    logger.info(f"FILE_WRITTEN: {safe_path}")

    return safe_path


def list_safe_directory(
    user_path: str = "",
    base_dir: Optional[Path] = None,
    pattern: str = "*"
) -> List[Path]:
    """
    Lista arquivos em um diretório de forma segura.

    Args:
        user_path: Subcaminho dentro do diretório base
        base_dir: Diretório base permitido
        pattern: Padrão glob para filtrar arquivos

    Returns:
        List[Path]: Lista de arquivos encontrados
    """
    if user_path:
        safe_path = validate_safe_path(user_path, base_dir)
    else:
        safe_path = base_dir or ALLOWED_BASE_DIRS[0]

    if not safe_path.is_dir():
        raise ValueError(f"Caminho não é um diretório: {user_path}")

    return list(safe_path.glob(pattern))


# Exemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Testes
    print("=== Testes de Path Traversal Prevention ===")

    # Teste 1: Caminho válido
    try:
        path = validate_safe_path("report.pdf", allowed_extensions=[".pdf"])
        print(f"[OK] Caminho válido: {path}")
    except Exception as e:
        print(f"[FAIL] {e}")

    # Teste 2: Tentativa de path traversal
    dangerous_paths = [
        "../../../etc/passwd",
        "..\\..\\windows\\system32\\config",
        "reports/../../../etc/shadow",
        "%2e%2e%2f%2e%2e%2fetc/passwd",
    ]

    for dangerous in dangerous_paths:
        try:
            path = validate_safe_path(dangerous)
            print(f"[FAIL] Deveria ter bloqueado: {dangerous}")
        except (ValueError, PermissionError) as e:
            print(f"[OK] Bloqueado: {dangerous} -> {e}")
