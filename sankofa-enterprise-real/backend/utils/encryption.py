#!/usr/bin/env python3
"""
Sankofa Enterprise Pro - PII Encryption
Criptografia de dados pessoais identificáveis (PII)

OWASP A02: Cryptographic Failures - PII Encryption at Rest
PCI-DSS Requirement 3.4: Render PAN unreadable
LGPD Art. 46: Medidas de segurança para proteção de dados pessoais
"""

import os
import base64
import secrets
import logging
import hashlib
from typing import Optional, Dict, Any, Union
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Tentar importar cryptography
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logger.warning(
        "cryptography não disponível. Instale: pip install cryptography"
    )


# Configurações de criptografia
ENCRYPTION_CONFIG = {
    "algorithm": "AES-256-GCM",
    "key_length": 32,  # 256 bits
    "nonce_length": 12,  # 96 bits para GCM
    "tag_length": 16,  # 128 bits
    "pbkdf2_iterations": 100000,
}


def _get_encryption_key() -> bytes:
    """
    Obtém a chave de criptografia do ambiente.

    Em produção, a chave deve ser:
    - Armazenada em HSM ou serviço de secrets (AWS KMS, HashiCorp Vault)
    - Rotacionada periodicamente
    - Nunca commitada no código

    Returns:
        bytes: Chave de 256 bits

    Raises:
        RuntimeError: Se a chave não estiver configurada
    """
    key_env = os.environ.get("PII_ENCRYPTION_KEY")

    if key_env:
        # Chave fornecida como hex ou base64
        try:
            if len(key_env) == 64:  # Hex string de 32 bytes
                return bytes.fromhex(key_env)
            else:
                return base64.b64decode(key_env)
        except Exception as e:
            logger.error(f"Erro ao decodificar PII_ENCRYPTION_KEY: {e}")
            raise RuntimeError("PII_ENCRYPTION_KEY inválida")

    # Em desenvolvimento, derivar chave de uma senha
    dev_password = os.environ.get("PII_DEV_PASSWORD", "sankofa-dev-only-not-for-production")

    if os.environ.get("FLASK_ENV") == "production":
        raise RuntimeError(
            "PII_ENCRYPTION_KEY deve ser configurada em produção. "
            "Gere uma chave com: python -c \"import secrets; print(secrets.token_hex(32))\""
        )

    logger.warning(
        "PII_ENCRYPTION_KEY não configurada - usando chave derivada de desenvolvimento. "
        "NÃO USE EM PRODUÇÃO!"
    )

    # Derivar chave usando PBKDF2
    salt = b"sankofa-enterprise-dev-salt"  # Salt fixo apenas para dev
    if CRYPTOGRAPHY_AVAILABLE:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=ENCRYPTION_CONFIG["pbkdf2_iterations"],
            backend=default_backend()
        )
        return kdf.derive(dev_password.encode())
    else:
        # Fallback sem cryptography
        return hashlib.pbkdf2_hmac(
            'sha256',
            dev_password.encode(),
            salt,
            ENCRYPTION_CONFIG["pbkdf2_iterations"]
        )


def encrypt_pii(plaintext: Union[str, bytes], additional_data: Optional[bytes] = None) -> str:
    """
    Criptografa dados PII usando AES-256-GCM.

    OWASP A02: Cryptographic Failures
    - Usa algoritmo autenticado (GCM) para integridade
    - Nonce único para cada operação
    - Chave de 256 bits

    Args:
        plaintext: Dados a serem criptografados (string ou bytes)
        additional_data: Dados adicionais autenticados (AAD) - não criptografados mas verificados

    Returns:
        str: Dados criptografados em base64 no formato:
             version:nonce:ciphertext:tag

    Raises:
        RuntimeError: Se cryptography não estiver disponível
        ValueError: Se os dados forem inválidos
    """
    if not CRYPTOGRAPHY_AVAILABLE:
        raise RuntimeError(
            "Módulo cryptography não disponível. "
            "Instale: pip install cryptography"
        )

    if not plaintext:
        raise ValueError("Dados para criptografar não podem ser vazios")

    # Converter para bytes se necessário
    if isinstance(plaintext, str):
        plaintext_bytes = plaintext.encode('utf-8')
    else:
        plaintext_bytes = plaintext

    # Obter chave
    key = _get_encryption_key()

    # Gerar nonce único (NUNCA reutilizar com a mesma chave!)
    nonce = secrets.token_bytes(ENCRYPTION_CONFIG["nonce_length"])

    # Criar cipher e criptografar
    aesgcm = AESGCM(key)

    try:
        ciphertext = aesgcm.encrypt(nonce, plaintext_bytes, additional_data)
    except Exception as e:
        logger.error(f"Erro na criptografia: {e}")
        raise ValueError(f"Falha na criptografia: {e}")

    # Formato: v1:nonce_b64:ciphertext_b64
    result = f"v1:{base64.b64encode(nonce).decode()}:{base64.b64encode(ciphertext).decode()}"

    logger.debug(f"PII_ENCRYPTED: {len(plaintext_bytes)} bytes -> {len(result)} chars")
    return result


def decrypt_pii(encrypted: str, additional_data: Optional[bytes] = None) -> str:
    """
    Descriptografa dados PII.

    Args:
        encrypted: Dados criptografados no formato v1:nonce:ciphertext
        additional_data: Dados adicionais autenticados (AAD) - deve ser o mesmo usado na criptografia

    Returns:
        str: Dados descriptografados

    Raises:
        RuntimeError: Se cryptography não estiver disponível
        ValueError: Se os dados forem inválidos ou a autenticação falhar
    """
    if not CRYPTOGRAPHY_AVAILABLE:
        raise RuntimeError(
            "Módulo cryptography não disponível. "
            "Instale: pip install cryptography"
        )

    if not encrypted:
        raise ValueError("Dados para descriptografar não podem ser vazios")

    # Parse do formato
    try:
        parts = encrypted.split(':')
        if len(parts) != 3:
            raise ValueError("Formato inválido")

        version = parts[0]
        if version != "v1":
            raise ValueError(f"Versão não suportada: {version}")

        nonce = base64.b64decode(parts[1])
        ciphertext = base64.b64decode(parts[2])

    except Exception as e:
        logger.error(f"Erro ao parsear dados criptografados: {e}")
        raise ValueError(f"Formato de dados criptografados inválido: {e}")

    # Obter chave
    key = _get_encryption_key()

    # Descriptografar
    aesgcm = AESGCM(key)

    try:
        plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, additional_data)
    except Exception as e:
        logger.error(f"Erro na descriptografia (possível adulteração): {e}")
        raise ValueError(
            "Falha na descriptografia - dados podem ter sido adulterados ou chave incorreta"
        )

    result = plaintext_bytes.decode('utf-8')
    logger.debug(f"PII_DECRYPTED: {len(encrypted)} chars -> {len(result)} chars")
    return result


def encrypt_field(data: Dict[str, Any], field_name: str) -> Dict[str, Any]:
    """
    Criptografa um campo específico em um dicionário.

    Args:
        data: Dicionário com os dados
        field_name: Nome do campo a criptografar

    Returns:
        Dict com o campo criptografado e metadados
    """
    if field_name not in data:
        return data

    value = data[field_name]
    if value is None:
        return data

    encrypted = encrypt_pii(str(value))

    result = data.copy()
    result[field_name] = encrypted
    result[f"{field_name}_encrypted"] = True
    result[f"{field_name}_encrypted_at"] = datetime.now(timezone.utc).isoformat()

    return result


def decrypt_field(data: Dict[str, Any], field_name: str) -> Dict[str, Any]:
    """
    Descriptografa um campo específico em um dicionário.

    Args:
        data: Dicionário com os dados
        field_name: Nome do campo a descriptografar

    Returns:
        Dict com o campo descriptografado
    """
    if field_name not in data:
        return data

    # Verificar se está criptografado
    if not data.get(f"{field_name}_encrypted"):
        return data

    encrypted = data[field_name]
    if encrypted is None:
        return data

    decrypted = decrypt_pii(encrypted)

    result = data.copy()
    result[field_name] = decrypted
    del result[f"{field_name}_encrypted"]
    if f"{field_name}_encrypted_at" in result:
        del result[f"{field_name}_encrypted_at"]

    return result


def hash_pii_for_lookup(value: str, purpose: str = "lookup") -> str:
    """
    Gera hash de PII para busca (não reversível).

    Útil para buscar registros sem expor o valor original.

    Args:
        value: Valor PII
        purpose: Propósito do hash (para separar diferentes usos)

    Returns:
        str: Hash em hex
    """
    if not value:
        raise ValueError("Valor não pode ser vazio")

    # Usar HMAC para evitar rainbow tables
    key = _get_encryption_key()
    data = f"{purpose}:{value}".encode('utf-8')

    import hmac
    h = hmac.new(key, data, hashlib.sha256)
    return h.hexdigest()


def mask_pii(value: str, pii_type: str = "generic") -> str:
    """
    Mascara PII para exibição (não reversível).

    Args:
        value: Valor PII
        pii_type: Tipo de PII (cpf, email, phone, card, generic)

    Returns:
        str: Valor mascarado
    """
    if not value:
        return "***"

    if pii_type == "cpf":
        # CPF: 123.456.789-00 -> ***.***.789-00
        clean = value.replace(".", "").replace("-", "")
        if len(clean) == 11:
            return f"***.***{clean[6:9]}-{clean[9:]}"
        return "***.***.***-**"

    elif pii_type == "email":
        # email@domain.com -> e***@domain.com
        if "@" in value:
            local, domain = value.split("@", 1)
            masked_local = local[0] + "***" if local else "***"
            return f"{masked_local}@{domain}"
        return "***@***"

    elif pii_type == "phone":
        # (11) 99999-1234 -> (11) *****-1234
        clean = ''.join(c for c in value if c.isdigit())
        if len(clean) >= 4:
            return f"{'*' * (len(clean) - 4)}{clean[-4:]}"
        return "****"

    elif pii_type == "card":
        # 4111111111111111 -> 411111******1111
        clean = ''.join(c for c in value if c.isdigit())
        if len(clean) >= 10:
            return f"{clean[:6]}{'*' * (len(clean) - 10)}{clean[-4:]}"
        return "****"

    else:
        # Generic: mostrar apenas primeiros e últimos caracteres
        if len(value) <= 4:
            return "*" * len(value)
        return f"{value[0]}{'*' * (len(value) - 2)}{value[-1]}"


def get_encryption_status() -> Dict[str, Any]:
    """
    Retorna status do sistema de criptografia.

    Returns:
        Dict com status detalhado
    """
    return {
        "cryptography_available": CRYPTOGRAPHY_AVAILABLE,
        "algorithm": ENCRYPTION_CONFIG["algorithm"],
        "key_length_bits": ENCRYPTION_CONFIG["key_length"] * 8,
        "key_configured": bool(os.environ.get("PII_ENCRYPTION_KEY")),
        "mode": "production" if os.environ.get("PII_ENCRYPTION_KEY") else "development",
        "compliance": {
            "pci_dss_3.4": "compliant" if CRYPTOGRAPHY_AVAILABLE else "not_compliant",
            "lgpd_art_46": "compliant" if CRYPTOGRAPHY_AVAILABLE else "not_compliant",
        }
    }


# Exemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Testes de Criptografia PII ===")

    if not CRYPTOGRAPHY_AVAILABLE:
        print("ERRO: cryptography não disponível")
        exit(1)

    # Teste 1: Criptografar e descriptografar
    print("\n1. Criptografia/Descriptografia:")
    original = "123.456.789-00"
    encrypted = encrypt_pii(original)
    decrypted = decrypt_pii(encrypted)
    print(f"   Original: {original}")
    print(f"   Criptografado: {encrypted[:50]}...")
    print(f"   Descriptografado: {decrypted}")
    print(f"   Match: {original == decrypted}")

    # Teste 2: Criptografar campo
    print("\n2. Criptografia de campo:")
    data = {"cpf": "123.456.789-00", "name": "João"}
    encrypted_data = encrypt_field(data, "cpf")
    print(f"   Original: {data}")
    print(f"   Criptografado: cpf={encrypted_data['cpf'][:30]}...")
    decrypted_data = decrypt_field(encrypted_data, "cpf")
    print(f"   Descriptografado: {decrypted_data}")

    # Teste 3: Mascaramento
    print("\n3. Mascaramento:")
    print(f"   CPF: {mask_pii('12345678900', 'cpf')}")
    print(f"   Email: {mask_pii('joao@empresa.com', 'email')}")
    print(f"   Telefone: {mask_pii('11999991234', 'phone')}")
    print(f"   Cartão: {mask_pii('4111111111111111', 'card')}")

    # Teste 4: Hash para lookup
    print("\n4. Hash para lookup:")
    cpf = "123.456.789-00"
    hash1 = hash_pii_for_lookup(cpf)
    hash2 = hash_pii_for_lookup(cpf)
    print(f"   Hash: {hash1}")
    print(f"   Consistente: {hash1 == hash2}")

    # Teste 5: Status
    print("\n5. Status:")
    import json
    print(json.dumps(get_encryption_status(), indent=2))
