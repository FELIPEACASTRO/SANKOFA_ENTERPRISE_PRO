"""
Log Sanitization Module
Remove PII (Personally Identifiable Information) dos logs
Compliance com LGPD Art. 46 - Segurança de Dados Pessoais
"""

import re
import hashlib
from typing import Any, Dict, List, Union
from copy import deepcopy


# ============================================================================
# PII PATTERNS - Regex para detectar dados sensíveis
# ============================================================================

PII_PATTERNS = {
    'cpf': r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b',
    'cnpj': r'\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}-?\d{4}\b',
    'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
    'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
    'jwt_token': r'eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*',
    'bearer_token': r'Bearer\s+[a-zA-Z0-9\-._~+/]+=*',
}

# Campos que SEMPRE devem ser mascarados
SENSITIVE_FIELDS = {
    'cpf',
    'customer_cpf',
    'cliente_cpf',
    'cpf_hash',
    'cnpj',
    'email',
    'customer_email',
    'phone',
    'telefone',
    'password',
    'senha',
    'token',
    'jwt',
    'jwt_token',
    'access_token',
    'refresh_token',
    'api_key',
    'secret',
    'secret_key',
    'private_key',
    'credit_card',
    'card_number',
    'cvv',
    'security_code',
    'ssn',
    'passport',
    'driver_license',
    'birth_date',
    'data_nascimento',
    'ip_address',
    'user_agent',
}


# ============================================================================
# MASKING FUNCTIONS
# ============================================================================

def mask_cpf(cpf: str) -> str:
    """
    Mascara CPF mostrando apenas últimos 2 dígitos
    Exemplo: 123.456.789-01 → ***.***.*79-**

    Args:
        cpf: CPF a mascarar

    Returns:
        CPF mascarado
    """
    if not cpf:
        return ""

    # Remove formatação
    cpf_clean = re.sub(r'\D', '', str(cpf))

    if len(cpf_clean) != 11:
        return "***.***.***-**"

    # Mostra apenas 2 dígitos do meio para evitar identificação
    return f"***.***.*{cpf_clean[-4:-2]}-**"


def mask_email(email: str) -> str:
    """
    Mascara email mostrando apenas domínio
    Exemplo: user@example.com → ***@example.com

    Args:
        email: Email a mascarar

    Returns:
        Email mascarado
    """
    if not email or '@' not in str(email):
        return "***@***.***"

    parts = str(email).split('@')
    return f"***@{parts[1]}"


def mask_phone(phone: str) -> str:
    """
    Mascara telefone mostrando apenas últimos 2 dígitos
    Exemplo: (11) 98765-4321 → (••) •••••-••21

    Args:
        phone: Telefone a mascarar

    Returns:
        Telefone mascarado
    """
    if not phone:
        return "••"

    phone_clean = re.sub(r'\D', '', str(phone))

    if len(phone_clean) < 8:
        return "••••••••"

    return f"(••) •••••-••{phone_clean[-2:]}"


def mask_credit_card(card: str) -> str:
    """
    Mascara cartão de crédito mostrando apenas últimos 4 dígitos
    Exemplo: 1234 5678 9012 3456 → **** **** **** 3456

    Args:
        card: Número do cartão

    Returns:
        Cartão mascarado
    """
    if not card:
        return "****"

    card_clean = re.sub(r'\D', '', str(card))

    if len(card_clean) < 12:
        return "**** **** **** ****"

    return f"**** **** **** {card_clean[-4:]}"


def mask_token(token: str) -> str:
    """
    Mascara tokens JWT e API keys mostrando apenas primeiros e últimos 4 caracteres
    Exemplo: eyJhbGc...xyz123 → eyJh...123

    Args:
        token: Token a mascarar

    Returns:
        Token mascarado
    """
    if not token or len(str(token)) < 10:
        return "****"

    token_str = str(token)
    return f"{token_str[:4]}...{token_str[-4:]}"


def hash_value(value: str) -> str:
    """
    Cria hash SHA-256 de um valor para logs
    Permite correlação sem expor dado original

    Args:
        value: Valor a fazer hash

    Returns:
        Hash do valor (primeiros 16 caracteres)
    """
    if not value:
        return "null"

    hash_obj = hashlib.sha256(str(value).encode())
    return f"hash_{hash_obj.hexdigest()[:16]}"


# ============================================================================
# SANITIZATION FUNCTIONS
# ============================================================================

def sanitize_string(text: str) -> str:
    """
    Sanitiza string aplicando máscaras a padrões de PII detectados

    Args:
        text: Texto a sanitizar

    Returns:
        Texto sanitizado
    """
    if not isinstance(text, str):
        return text

    sanitized = text

    # Aplica máscaras para cada padrão
    sanitized = re.sub(PII_PATTERNS['cpf'], '[CPF_MASKED]', sanitized)
    sanitized = re.sub(PII_PATTERNS['cnpj'], '[CNPJ_MASKED]', sanitized)
    sanitized = re.sub(PII_PATTERNS['email'], '[EMAIL_MASKED]', sanitized)
    sanitized = re.sub(PII_PATTERNS['phone'], '[PHONE_MASKED]', sanitized)
    sanitized = re.sub(PII_PATTERNS['credit_card'], '[CARD_MASKED]', sanitized)
    sanitized = re.sub(PII_PATTERNS['ip_address'], '[IP_MASKED]', sanitized)
    sanitized = re.sub(PII_PATTERNS['jwt_token'], '[JWT_MASKED]', sanitized)
    sanitized = re.sub(PII_PATTERNS['bearer_token'], 'Bearer [TOKEN_MASKED]', sanitized)

    return sanitized


def sanitize_dict(data: Dict[str, Any], mask_mode: str = 'mask') -> Dict[str, Any]:
    """
    Sanitiza dicionário mascarando campos sensíveis

    Args:
        data: Dicionário a sanitizar
        mask_mode: 'mask' (mascara) ou 'hash' (faz hash) ou 'remove' (remove campo)

    Returns:
        Dicionário sanitizado
    """
    if not isinstance(data, dict):
        return data

    sanitized = deepcopy(data)

    for key, value in sanitized.items():
        key_lower = key.lower()

        # Verifica se é campo sensível
        if key_lower in SENSITIVE_FIELDS:
            if mask_mode == 'remove':
                sanitized[key] = '[REMOVED]'
            elif mask_mode == 'hash':
                sanitized[key] = hash_value(value)
            else:  # mask
                # Aplica máscara apropriada baseado no tipo de campo
                if 'cpf' in key_lower:
                    sanitized[key] = mask_cpf(value)
                elif 'email' in key_lower:
                    sanitized[key] = mask_email(value)
                elif 'phone' in key_lower or 'telefone' in key_lower:
                    sanitized[key] = mask_phone(value)
                elif 'card' in key_lower or 'cartao' in key_lower:
                    sanitized[key] = mask_credit_card(value)
                elif 'token' in key_lower or 'jwt' in key_lower or 'key' in key_lower:
                    sanitized[key] = mask_token(value)
                elif 'password' in key_lower or 'senha' in key_lower:
                    sanitized[key] = '********'
                else:
                    sanitized[key] = '[MASKED]'

        # Recursão para dicionários aninhados
        elif isinstance(value, dict):
            sanitized[key] = sanitize_dict(value, mask_mode)

        # Recursão para listas
        elif isinstance(value, list):
            sanitized[key] = sanitize_list(value, mask_mode)

        # Sanitiza strings que podem conter PII
        elif isinstance(value, str):
            sanitized[key] = sanitize_string(value)

    return sanitized


def sanitize_list(data: List[Any], mask_mode: str = 'mask') -> List[Any]:
    """
    Sanitiza lista aplicando sanitização recursiva

    Args:
        data: Lista a sanitizar
        mask_mode: Modo de mascaramento

    Returns:
        Lista sanitizada
    """
    if not isinstance(data, list):
        return data

    sanitized = []

    for item in data:
        if isinstance(item, dict):
            sanitized.append(sanitize_dict(item, mask_mode))
        elif isinstance(item, list):
            sanitized.append(sanitize_list(item, mask_mode))
        elif isinstance(item, str):
            sanitized.append(sanitize_string(item))
        else:
            sanitized.append(item)

    return sanitized


def sanitize_log_data(data: Any, mask_mode: str = 'mask') -> Any:
    """
    Função principal para sanitizar dados antes de logar

    Args:
        data: Dados a sanitizar (dict, list, str, ou outro)
        mask_mode: 'mask', 'hash', ou 'remove'

    Returns:
        Dados sanitizados

    Example:
        >>> logger.info('Transaction created', extra=sanitize_log_data({
        ...     'cpf': '12345678901',
        ...     'amount': 1000,
        ...     'email': 'user@example.com'
        ... }))
        # Log: {'cpf': '***.***.*89-**', 'amount': 1000, 'email': '***@example.com'}
    """
    if isinstance(data, dict):
        return sanitize_dict(data, mask_mode)
    elif isinstance(data, list):
        return sanitize_list(data, mask_mode)
    elif isinstance(data, str):
        return sanitize_string(data)
    else:
        return data


# ============================================================================
# STRUCTURED LOGGING INTEGRATION
# ============================================================================

def create_safe_log_context(**kwargs) -> Dict[str, Any]:
    """
    Cria contexto de log seguro com sanitização automática

    Args:
        **kwargs: Campos do log

    Returns:
        Contexto sanitizado

    Example:
        >>> context = create_safe_log_context(
        ...     user_id='123',
        ...     cpf='12345678901',
        ...     action='login'
        ... )
        >>> logger.info('User login', **context)
    """
    return sanitize_log_data(kwargs, mask_mode='mask')


# ============================================================================
# TESTING HELPERS
# ============================================================================

def is_pii_exposed(text: str) -> bool:
    """
    Verifica se há PII exposto em texto

    Args:
        text: Texto a verificar

    Returns:
        True se PII detectado, False caso contrário

    Example:
        >>> is_pii_exposed("User CPF: 123.456.789-01")
        True
        >>> is_pii_exposed("User ID: 12345")
        False
    """
    if not isinstance(text, str):
        return False

    for pattern_name, pattern in PII_PATTERNS.items():
        if re.search(pattern, text):
            return True

    return False


def validate_log_safety(log_entry: Union[str, Dict]) -> List[str]:
    """
    Valida se entrada de log está segura (sem PII)

    Args:
        log_entry: Entrada de log a validar

    Returns:
        Lista de violações encontradas (vazio se seguro)

    Example:
        >>> violations = validate_log_safety({
        ...     'user': 'john',
        ...     'cpf': '123.456.789-01'
        ... })
        >>> print(violations)
        ['Field "cpf" contains PII', 'CPF pattern detected in values']
    """
    violations = []

    if isinstance(log_entry, dict):
        # Verifica campos sensíveis
        for key in log_entry.keys():
            if key.lower() in SENSITIVE_FIELDS:
                violations.append(f'Field "{key}" is sensitive and should be masked')

        # Verifica valores
        for value in log_entry.values():
            if isinstance(value, str) and is_pii_exposed(value):
                violations.append('PII pattern detected in values')

    elif isinstance(log_entry, str):
        if is_pii_exposed(log_entry):
            violations.append('PII pattern detected in log message')

    return violations


# ============================================================================
# PRE-COMMIT HOOK HELPER
# ============================================================================

def scan_file_for_pii(file_path: str) -> List[Dict[str, Any]]:
    """
    Escaneia arquivo Python em busca de logs com PII
    Útil para pre-commit hooks

    Args:
        file_path: Caminho do arquivo a escanear

    Returns:
        Lista de violações com linha e tipo

    Example:
        >>> violations = scan_file_for_pii('backend/api/app.py')
        >>> for v in violations:
        ...     print(f"Line {v['line']}: {v['violation']}")
    """
    violations = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Detecta chamadas de logger
                if 'logger.' in line or 'logging.' in line or 'print(' in line:
                    # Verifica se há padrões de PII
                    for pattern_name, pattern in PII_PATTERNS.items():
                        if re.search(pattern, line):
                            violations.append({
                                'file': file_path,
                                'line': line_num,
                                'type': pattern_name,
                                'violation': f'{pattern_name.upper()} pattern detected in log statement',
                                'line_content': line.strip()
                            })
    except Exception as e:
        violations.append({
            'file': file_path,
            'line': 0,
            'type': 'error',
            'violation': f'Error scanning file: {str(e)}',
            'line_content': ''
        })

    return violations
