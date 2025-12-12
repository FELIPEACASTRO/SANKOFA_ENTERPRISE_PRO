#!/usr/bin/env python3
"""
Sankofa Enterprise Pro - Authentication Services
Serviços de autenticação com hash seguro de senhas

OWASP A02: Cryptographic Failures - Password Hashing
OWASP A07: Authentication Failures - Password Validation
"""

import re
import secrets
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Tentar importar bcrypt, fallback para hashlib se não disponível
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    import hashlib
    logger.warning(
        "bcrypt não disponível - usando fallback com PBKDF2. "
        "Instale bcrypt para segurança máxima: pip install bcrypt"
    )


# Configurações de política de senha
PASSWORD_POLICY = {
    "min_length": 12,
    "max_length": 128,
    "require_uppercase": True,
    "require_lowercase": True,
    "require_digit": True,
    "require_special": True,
    "special_chars": "!@#$%^&*()_+-=[]{}|;:,.<>?",
    "max_consecutive_chars": 3,
    "check_common_passwords": True,
}

# Lista de senhas comuns a serem bloqueadas
COMMON_PASSWORDS = frozenset({
    "password", "123456", "12345678", "qwerty", "abc123",
    "monkey", "1234567", "letmein", "trustno1", "dragon",
    "baseball", "iloveyou", "master", "sunshine", "ashley",
    "bailey", "shadow", "123123", "654321", "superman",
    "qazwsx", "michael", "football", "password1", "password123",
    "welcome", "welcome1", "admin", "admin123", "root",
    "toor", "pass", "test", "guest", "master",
    "changeme", "123qwe", "1q2w3e", "1q2w3e4r", "zaq12wsx",
    "qwerty123", "1234567890", "0987654321", "senha", "senha123",
})


def hash_password(password: str, rounds: int = 12) -> str:
    """
    Gera hash seguro de senha usando bcrypt.

    OWASP A02: Cryptographic Failures
    Usa bcrypt com salt automático e custo configurável.

    Args:
        password: Senha em texto plano
        rounds: Número de rounds para bcrypt (default: 12, recomendado: 10-14)

    Returns:
        str: Hash da senha no formato bcrypt

    Raises:
        ValueError: Se a senha for inválida
    """
    if not password:
        raise ValueError("Senha não pode ser vazia")

    if len(password) > PASSWORD_POLICY["max_length"]:
        raise ValueError(f"Senha muito longa (máximo: {PASSWORD_POLICY['max_length']} caracteres)")

    password_bytes = password.encode('utf-8')

    if BCRYPT_AVAILABLE:
        # Usar bcrypt (recomendado)
        salt = bcrypt.gensalt(rounds=rounds)
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')
    else:
        # Fallback para PBKDF2 com SHA-256
        salt = secrets.token_hex(16)
        iterations = 2 ** rounds * 1000  # Aproximar custo do bcrypt
        import hashlib
        dk = hashlib.pbkdf2_hmac(
            'sha256',
            password_bytes,
            salt.encode('utf-8'),
            iterations
        )
        return f"$pbkdf2-sha256${iterations}${salt}${dk.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """
    Verifica se a senha corresponde ao hash.

    Args:
        password: Senha em texto plano
        hashed: Hash armazenado

    Returns:
        bool: True se a senha corresponder
    """
    if not password or not hashed:
        return False

    password_bytes = password.encode('utf-8')

    if BCRYPT_AVAILABLE and hashed.startswith('$2'):
        # Hash bcrypt
        try:
            return bcrypt.checkpw(password_bytes, hashed.encode('utf-8'))
        except Exception:
            return False
    elif hashed.startswith('$pbkdf2-sha256$'):
        # Hash PBKDF2 (fallback)
        try:
            parts = hashed.split('$')
            iterations = int(parts[2])
            salt = parts[3]
            stored_hash = parts[4]

            import hashlib
            dk = hashlib.pbkdf2_hmac(
                'sha256',
                password_bytes,
                salt.encode('utf-8'),
                iterations
            )
            return secrets.compare_digest(dk.hex(), stored_hash)
        except Exception:
            return False

    return False


def validate_password_strength(password: str) -> Dict[str, Any]:
    """
    Valida a força da senha conforme política de segurança.

    OWASP A07: Authentication Failures
    Implementa validação robusta de senha.

    Args:
        password: Senha a ser validada

    Returns:
        Dict com:
            - valid: bool indicando se a senha é válida
            - score: pontuação de 0-100
            - errors: lista de erros encontrados
            - suggestions: sugestões de melhoria
    """
    errors = []
    suggestions = []
    score = 0

    if not password:
        return {
            "valid": False,
            "score": 0,
            "errors": ["Senha não pode ser vazia"],
            "suggestions": ["Forneça uma senha"]
        }

    # Verificar comprimento mínimo
    if len(password) < PASSWORD_POLICY["min_length"]:
        errors.append(
            f"Senha muito curta (mínimo: {PASSWORD_POLICY['min_length']} caracteres)"
        )
        suggestions.append("Aumente o comprimento da senha")
    else:
        score += 20

    # Verificar comprimento máximo
    if len(password) > PASSWORD_POLICY["max_length"]:
        errors.append(
            f"Senha muito longa (máximo: {PASSWORD_POLICY['max_length']} caracteres)"
        )

    # Verificar letra maiúscula
    if PASSWORD_POLICY["require_uppercase"]:
        if not re.search(r'[A-Z]', password):
            errors.append("Senha deve conter pelo menos uma letra maiúscula")
            suggestions.append("Adicione letras maiúsculas (A-Z)")
        else:
            score += 15

    # Verificar letra minúscula
    if PASSWORD_POLICY["require_lowercase"]:
        if not re.search(r'[a-z]', password):
            errors.append("Senha deve conter pelo menos uma letra minúscula")
            suggestions.append("Adicione letras minúsculas (a-z)")
        else:
            score += 15

    # Verificar dígito
    if PASSWORD_POLICY["require_digit"]:
        if not re.search(r'\d', password):
            errors.append("Senha deve conter pelo menos um número")
            suggestions.append("Adicione números (0-9)")
        else:
            score += 15

    # Verificar caractere especial
    if PASSWORD_POLICY["require_special"]:
        special_pattern = f'[{re.escape(PASSWORD_POLICY["special_chars"])}]'
        if not re.search(special_pattern, password):
            errors.append(
                f"Senha deve conter pelo menos um caractere especial: "
                f"{PASSWORD_POLICY['special_chars']}"
            )
            suggestions.append("Adicione caracteres especiais (!@#$%...)")
        else:
            score += 15

    # Verificar caracteres consecutivos repetidos
    max_consecutive = PASSWORD_POLICY["max_consecutive_chars"]
    if re.search(r'(.)\1{' + str(max_consecutive) + r',}', password):
        errors.append(
            f"Senha não pode ter mais de {max_consecutive} caracteres iguais consecutivos"
        )
        suggestions.append("Evite repetir o mesmo caractere muitas vezes")
        score -= 10

    # Verificar senhas comuns
    if PASSWORD_POLICY["check_common_passwords"]:
        password_lower = password.lower()
        if password_lower in COMMON_PASSWORDS:
            errors.append("Senha muito comum - escolha uma senha mais única")
            suggestions.append("Use uma senha menos previsível")
            score = min(score, 20)

        # Verificar se contém padrões comuns
        common_patterns = ["123", "abc", "qwerty", "password", "admin"]
        for pattern in common_patterns:
            if pattern in password_lower:
                suggestions.append(f"Evite padrões comuns como '{pattern}'")
                score -= 5

    # Bônus por comprimento extra
    if len(password) >= 16:
        score += 10
    if len(password) >= 20:
        score += 10

    # Normalizar score
    score = max(0, min(100, score))

    # Determinar se é válida
    is_valid = len(errors) == 0 and score >= 50

    # Log para auditoria (sem expor a senha)
    if not is_valid:
        logger.warning(
            f"PASSWORD_VALIDATION_FAILED: {len(errors)} erros, score={score}"
        )

    return {
        "valid": is_valid,
        "score": score,
        "strength": _get_strength_label(score),
        "errors": errors,
        "suggestions": suggestions if not is_valid else []
    }


def _get_strength_label(score: int) -> str:
    """Retorna label de força baseado no score."""
    if score >= 80:
        return "STRONG"
    elif score >= 60:
        return "GOOD"
    elif score >= 40:
        return "FAIR"
    elif score >= 20:
        return "WEAK"
    else:
        return "VERY_WEAK"


def generate_secure_password(length: int = 16) -> str:
    """
    Gera uma senha segura aleatória.

    Args:
        length: Comprimento da senha (mínimo: 12)

    Returns:
        str: Senha gerada
    """
    if length < 12:
        length = 12

    # Garantir que tenha todos os tipos de caracteres
    import string

    password_chars = []

    # Pelo menos um de cada tipo
    password_chars.append(secrets.choice(string.ascii_uppercase))
    password_chars.append(secrets.choice(string.ascii_lowercase))
    password_chars.append(secrets.choice(string.digits))
    password_chars.append(secrets.choice(PASSWORD_POLICY["special_chars"]))

    # Preencher o resto
    all_chars = (
        string.ascii_letters +
        string.digits +
        PASSWORD_POLICY["special_chars"]
    )

    for _ in range(length - 4):
        password_chars.append(secrets.choice(all_chars))

    # Embaralhar
    secrets.SystemRandom().shuffle(password_chars)

    return ''.join(password_chars)


def needs_rehash(hashed: str, target_rounds: int = 12) -> bool:
    """
    Verifica se o hash precisa ser atualizado (e.g., rounds aumentados).

    Args:
        hashed: Hash atual
        target_rounds: Rounds desejados

    Returns:
        bool: True se precisa rehash
    """
    if not BCRYPT_AVAILABLE:
        return False

    if not hashed.startswith('$2'):
        return True  # Não é bcrypt, precisa migrar

    try:
        # Extrair rounds do hash bcrypt
        # Formato: $2b$12$...
        parts = hashed.split('$')
        if len(parts) >= 4:
            current_rounds = int(parts[2])
            return current_rounds < target_rounds
    except Exception:
        return True

    return False


# Exemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Testes de Autenticação ===")

    # Teste 1: Hash de senha
    password = "MySecureP@ssw0rd!"
    hashed = hash_password(password)
    print(f"\n1. Hash de senha:")
    print(f"   Senha: {password}")
    print(f"   Hash: {hashed[:50]}...")

    # Teste 2: Verificação
    print(f"\n2. Verificação:")
    print(f"   Correta: {verify_password(password, hashed)}")
    print(f"   Incorreta: {verify_password('wrong', hashed)}")

    # Teste 3: Validação de força
    print(f"\n3. Validação de força:")
    test_passwords = [
        "123",
        "password",
        "Password1",
        "Password1!",
        "MySecureP@ssw0rd!",
        "X#k9$mP2vL&nQ4wR",
    ]

    for pwd in test_passwords:
        result = validate_password_strength(pwd)
        print(f"   '{pwd}': score={result['score']}, strength={result['strength']}, valid={result['valid']}")

    # Teste 4: Geração de senha
    print(f"\n4. Senha gerada: {generate_secure_password(20)}")
