"""
Script to batch-apply Pydantic validation to critical endpoints
Applies validation systematically to production_api.py
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

# Top 20 Critical Endpoints - Priority Order
CRITICAL_ENDPOINTS = {
    # Authentication (CRITICAL - P0)
    '/api/auth/login': {
        'line': 1286,
        'schema': 'UserLogin',
        'method': 'POST',
        'fields': ['username', 'password']
    },
    '/api/auth/refresh': {
        'line': 1416,
        'schema': 'RefreshTokenRequest',
        'method': 'POST',
        'fields': ['refresh_token']
    },

    # Fraud Detection (CRITICAL - P0)
    '/api/fraud/batch': {
        'line': 1684,
        'schema': 'FraudPredictionBatchRequest',
        'method': 'POST',
        'fields': ['transactions']
    },
    '/api/explainability/explain': {
        'line': 1786,
        'schema': 'TransactionRequest',
        'method': 'POST',
        'fields': ['amount', 'cpf', 'channel']
    },

    # Transaction Operations (HIGH - P1)
    '/api/transactions/<transaction_id>/review': {
        'line': 2188,
        'schema': 'ManualReviewCreate',
        'method': 'POST',
        'fields': ['decision', 'comments']
    },
    '/api/investigations': {
        'line': 2246,
        'schema': 'InvestigationCreate',
        'method': 'POST',
        'fields': ['transaction_id', 'investigation_type', 'description', 'priority']
    },

    # Hard Rules (HIGH - P1)
    '/api/hard-rules': {
        'line': 2444,
        'schema': 'HardRuleCreate',
        'method': 'POST',
        'fields': ['name', 'description', 'condition', 'action', 'priority', 'enabled']
    },
    '/api/hard-rules/<int:rule_id>': {
        'line': 2506,
        'schema': 'HardRuleUpdate',
        'method': 'PUT',
        'fields': ['name', 'description', 'condition', 'action', 'priority', 'enabled']
    },

    # VIP/Hot Lists (HIGH - P1)
    '/api/vip-list': {
        'line': 2915,
        'schema': 'VipListCreate',
        'method': 'POST',
        'fields': ['cpf', 'reason', 'expires_at']
    },
    '/api/hot-list': {
        'line': 2972,
        'schema': 'HotListCreate',
        'method': 'POST',
        'fields': ['cpf', 'reason', 'severity', 'expires_at']
    },

    # Settings (MEDIUM - P2)
    '/api/settings': {
        'line': 3022,
        'schema': 'SettingsUpdate',
        'method': 'PUT',
        'fields': ['fraud_threshold', 'auto_block_enabled', 'email_notifications']
    },

    # Alerts (MEDIUM - P2)
    '/api/alerts/<int:alert_id>/status': {
        'line': 3159,
        'schema': 'AlertUpdate',
        'method': 'PUT',
        'fields': ['status', 'notes']
    },

    # Calibration (MEDIUM - P2)
    '/api/calibration': {
        'line': 3227,
        'schema': 'CalibrationUpdate',
        'method': 'PUT',
        'fields': ['model_id', 'calibration_method', 'parameters']
    },

    # Feedback (MEDIUM - P2)
    '/api/feedback': {
        'line': 3719,
        'schema': 'FeedbackCreate',
        'method': 'POST',
        'fields': ['transaction_id', 'feedback_type', 'correct_label', 'comments']
    },

    # Manual Review (MEDIUM - P2)
    '/api/manual-review': {
        'line': 2349,
        'schema': 'ManualReviewCreate',
        'method': 'POST',
        'fields': ['transaction_id', 'decision', 'comments']
    },

    # Batch Processing (MEDIUM - P2)
    '/api/infrastructure/batch/process': {
        'line': 4159,
        'schema': 'BatchProcessRequest',
        'method': 'POST',
        'fields': ['transaction_ids', 'process_type', 'options']
    },

    # Reports (LOW - P3)
    '/api/reports/generate': {
        'line': 3645,
        'schema': 'ExportRequest',
        'method': 'POST',
        'fields': ['export_format', 'start_date', 'end_date', 'filters']
    },
    '/api/audit/export': {
        'line': 3203,
        'schema': 'ExportRequest',
        'method': 'POST',
        'fields': ['export_format', 'start_date', 'end_date']
    },

    # Advanced Features (LOW - P3)
    '/api/advanced/predict/enriched': {
        'line': 4560,
        'schema': 'TransactionRequest',
        'method': 'POST',
        'fields': ['amount', 'cpf', 'channel']
    },
    '/api/research/bahnsen/features': {
        'line': 4304,
        'schema': 'TransactionRequest',
        'method': 'POST',
        'fields': ['amount', 'cpf', 'channel']
    },
}


def generate_pydantic_validation_code(schema_name: str, fields: List[str]) -> str:
    """
    Gera código de validação Pydantic

    Args:
        schema_name: Nome do schema Pydantic
        fields: Campos esperados

    Returns:
        String com código de validação
    """
    return f'''    # Pydantic Validation
    try:
        if not request.json:
            raise ValidationError(
                "Request body is required",
                context={{"endpoint": request.path}}
            )

        # Validar request com Pydantic
        if PYDANTIC_AVAILABLE:
            validated_request = {schema_name}(**request.json)
            # Extrair campos validados
            {generate_field_extraction(fields)}
        else:
            # Fallback para validação manual
            {generate_manual_validation(fields)}

    except PydanticValidationError as e:
        logger.warning(
            "Pydantic validation failed",
            extra=sanitize_log_data({{
                'endpoint': request.path,
                'errors': e.errors()
            }})
        )
        return jsonify({{
            'success': False,
            'error': 'Validation failed',
            'details': e.errors()
        }}), 400
    except ValidationError as e:
        logger.error(
            "Validation error",
            extra=sanitize_log_data({{
                'endpoint': request.path,
                'error': str(e)
            }})
        )
        return jsonify({{
            'success': False,
            'error': str(e)
        }}), 400
'''


def generate_field_extraction(fields: List[str]) -> str:
    """Gera código para extrair campos do schema validado"""
    extractions = []
    for field in fields:
        extractions.append(f"            {field} = validated_request.{field}")
    return '\n'.join(extractions)


def generate_manual_validation(fields: List[str]) -> str:
    """Gera código de validação manual como fallback"""
    validations = []
    for field in fields:
        validations.append(f"            {field} = request.json.get('{field}')")
        validations.append(f"            if {field} is None:")
        validations.append(f"                raise ValidationError('Field {field} is required')")
    return '\n'.join(validations)


def read_file(file_path: Path) -> List[str]:
    """Lê arquivo e retorna linhas"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()


def write_file(file_path: Path, lines: List[str]):
    """Escreve linhas em arquivo"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def find_endpoint_function(lines: List[str], start_line: int) -> Tuple[int, int]:
    """
    Encontra início e fim da função do endpoint

    Args:
        lines: Linhas do arquivo
        start_line: Linha onde está o decorator @app.route

    Returns:
        Tuple (start_index, end_index) da função
    """
    # Encontrar início da função (linha com def)
    func_start = start_line
    while func_start < len(lines) and not lines[func_start].strip().startswith('def '):
        func_start += 1

    if func_start >= len(lines):
        raise ValueError(f"Function not found after line {start_line}")

    # Encontrar fim da função (próxima linha com @app.route ou def em nível 0)
    func_end = func_start + 1
    indent_level = len(lines[func_start]) - len(lines[func_start].lstrip())

    while func_end < len(lines):
        line = lines[func_end]
        if line.strip() and not line.startswith(' '):
            # Nova definição de nível global
            break
        if line.strip().startswith('@app.route') or line.strip().startswith('@limiter.limit'):
            # Próximo endpoint
            break
        func_end += 1

    return func_start, func_end


def inject_pydantic_validation(lines: List[str], endpoint: str, config: Dict) -> List[str]:
    """
    Injeta validação Pydantic em um endpoint

    Args:
        lines: Linhas do arquivo
        endpoint: Nome do endpoint
        config: Configuração do endpoint (schema, fields, etc.)

    Returns:
        Linhas modificadas
    """
    line_num = config['line'] - 1  # Convert to 0-indexed

    # Encontrar a função
    func_start, func_end = find_endpoint_function(lines, line_num)

    print(f"📍 Processing {endpoint}")
    print(f"   Function: lines {func_start+1} to {func_end+1}")

    # Encontrar a primeira linha após o docstring (se houver)
    insert_line = func_start + 1
    while insert_line < func_end and (
        lines[insert_line].strip().startswith('"""') or
        lines[insert_line].strip().startswith("'''") or
        (insert_line > func_start + 1 and not lines[insert_line-1].strip().endswith('"""') and not lines[insert_line-1].strip().endswith("'''"))
    ):
        insert_line += 1

    # Verificar se já tem validação Pydantic
    function_body = ''.join(lines[func_start:func_end])
    if 'PydanticValidationError' in function_body or f"{config['schema']}(**request.json)" in function_body:
        print(f"   ⏭️  Already has Pydantic validation - skipping")
        return lines

    # Gerar código de validação
    validation_code = generate_pydantic_validation_code(config['schema'], config['fields'])
    validation_lines = [line + '\n' for line in validation_code.split('\n')]

    # Inserir validação
    new_lines = lines[:insert_line] + validation_lines + lines[insert_line:]

    print(f"   ✅ Injected Pydantic validation ({len(validation_lines)} lines)")

    return new_lines


def main():
    """Main execution"""
    api_file = Path(__file__).parent.parent / 'api' / 'production_api.py'

    if not api_file.exists():
        print(f"❌ File not found: {api_file}")
        return

    print(f"📄 Reading {api_file}")
    lines = read_file(api_file)
    print(f"   Total lines: {len(lines)}")

    # Aplicar validação em cada endpoint
    modified = False
    for endpoint, config in CRITICAL_ENDPOINTS.items():
        try:
            new_lines = inject_pydantic_validation(lines, endpoint, config)
            if new_lines != lines:
                lines = new_lines
                modified = True
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

    if modified:
        # Backup original
        backup_file = api_file.with_suffix('.py.backup')
        print(f"\n💾 Creating backup: {backup_file}")
        write_file(backup_file, read_file(api_file))

        # Escrever modificações
        print(f"💾 Writing modifications to {api_file}")
        write_file(api_file, lines)
        print(f"   New total lines: {len(lines)}")

        print(f"\n✅ Successfully applied Pydantic validation to {len(CRITICAL_ENDPOINTS)} endpoints")
    else:
        print(f"\n⏭️  No changes needed - all endpoints already validated")


if __name__ == '__main__':
    main()
