#!/usr/bin/env python3
"""
Script de Automação: Aplicar Validação Pydantic em Massa
Aplica schemas Pydantic em todos os endpoints POST/PUT do production_api.py
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Mapeamento de endpoints para schemas
ENDPOINT_SCHEMA_MAPPING = {
    '/api/auth/login': ('UserLogin', 'POST'),
    '/api/auth/refresh': ('RefreshTokenRequest', 'POST'),
    '/api/hard-rules': ('HardRuleCreate', 'POST'),
    '/api/hard-rules/<int:rule_id>': ('HardRuleUpdate', 'PUT'),
    '/api/vip-list': ('VipListCreate', 'POST'),
    '/api/hot-list': ('HotListCreate', 'POST'),
    '/api/feedback': ('FeedbackCreate', 'POST'),
    '/api/manual-review': ('ManualReviewCreate', 'POST'),
    '/api/alerts/<int:alert_id>/status': ('AlertUpdate', 'PUT'),
    '/api/settings': ('SettingsUpdate', 'PUT'),
    '/api/calibration': ('CalibrationUpdate', 'PUT'),
    '/api/investigations': ('InvestigationCreate', 'POST'),
    '/api/datasets': ('DatasetCreate', 'POST'),
}

def find_endpoint_function(content: str, route_pattern: str, method: str) -> Tuple[int, int, str]:
    """
    Encontra a função de um endpoint específico

    Returns:
        (start_line, end_line, function_content)
    """
    # Procura pelo decorator @app.route
    pattern = rf'@app\.route\(["\']' + re.escape(route_pattern) + rf'["\'].*?methods=\["{method}"\]\)'

    matches = list(re.finditer(pattern, content, re.MULTILINE))
    if not matches:
        return None, None, None

    match = matches[0]
    start_pos = match.start()

    # Encontra o início da próxima função
    lines = content[:start_pos].split('\n')
    start_line = len(lines) - 1

    # Encontra o fim da função (próximo @app.route ou EOF)
    next_route = re.search(r'@app\.route\(', content[match.end():])
    if next_route:
        end_pos = match.end() + next_route.start()
    else:
        end_pos = len(content)

    function_content = content[start_pos:end_pos]
    lines_in_function = function_content.split('\n')
    end_line = start_line + len(lines_in_function)

    return start_line, end_line, function_content


def add_pydantic_validation(function_content: str, schema_name: str) -> str:
    """
    Adiciona validação Pydantic a uma função
    """
    lines = function_content.split('\n')

    # Encontra a linha da definição da função
    func_def_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            func_def_idx = i
            break

    if func_def_idx is None:
        return function_content

    # Encontra o primeiro if not request.json ou data = request.json
    validation_insert_idx = func_def_idx + 2  # Após def e docstring

    # Template de validação
    validation_code = f"""    # Pydantic Validation
    try:
        if not request.json:
            return jsonify({{'success': False, 'error': 'Request body required'}}), 400

        if PYDANTIC_AVAILABLE:
            validated_data = {schema_name}(**request.json)
            data = validated_data.model_dump()
        else:
            data = request.json
    except PydanticValidationError as e:
        logger.warning(
            "Validation failed",
            extra=sanitize_log_data({{'endpoint': request.path, 'errors': e.errors()}})
        )
        return jsonify({{'success': False, 'error': 'Validation failed', 'details': e.errors()}}), 400

"""

    # Insere o código de validação
    lines.insert(validation_insert_idx, validation_code)

    return '\n'.join(lines)


def main():
    api_file = Path('backend/api/production_api.py')

    if not api_file.exists():
        print(f"Erro: {api_file} não encontrado")
        return 1

    with open(api_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print("Aplicando validação Pydantic em endpoints...")

    modifications = []
    for endpoint, (schema, method) in ENDPOINT_SCHEMA_MAPPING.items():
        print(f"  Processando {method} {endpoint} -> {schema}")
        start, end, func_content = find_endpoint_function(content, endpoint, method)

        if func_content:
            # Verifica se já tem validação
            if 'PydanticValidationError' in func_content:
                print(f"    ✓ Já tem validação")
                continue

            # Adiciona validação
            new_content = add_pydantic_validation(func_content, schema)
            modifications.append((start, end, new_content))
            print(f"    ✓ Validação adicionada")
        else:
            print(f"    ✗ Endpoint não encontrado")

    # Aplica modificações
    if modifications:
        print(f"\nAplicando {len(modifications)} modificações...")
        # Ordena por posição reversa para não bagunçar índices
        modifications.sort(key=lambda x: x[0], reverse=True)

        lines = content.split('\n')
        for start, end, new_content in modifications:
            lines[start:end] = new_content.split('\n')

        new_full_content = '\n'.join(lines)

        # Salva backup
        backup_file = api_file.with_suffix('.py.bak')
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Backup salvo em: {backup_file}")

        # Salva arquivo modificado
        with open(api_file, 'w', encoding='utf-8') as f:
            f.write(new_full_content)
        print(f"✓ Arquivo atualizado: {api_file}")
        print(f"✓ {len(modifications)} endpoints agora com validação Pydantic")
    else:
        print("\nNenhuma modificação necessária")

    return 0


if __name__ == '__main__':
    sys.exit(main())
