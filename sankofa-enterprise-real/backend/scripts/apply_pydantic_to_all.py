#!/usr/bin/env python3
"""
Apply Pydantic Validation to ALL endpoints in production_api.py
Automaticamente adiciona @validate_request onde necessario
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict

# Mapeamento endpoint -> schema
ENDPOINT_SCHEMAS = {
    # Endpoints que JA tem validacao (skip)
    '/api/predict': 'TransactionRequest',
    '/api/batch/predict': 'FraudPredictionBatchRequest',
    '/api/rules': 'HardRuleCreate',
    '/api/auth/login': 'UserLogin',

    # Endpoints que PRECISAM validacao
    '/api/manual-review': 'ManualReviewCreate',
    '/api/feedback': 'FeedbackCreate',
    '/api/investigations': 'InvestigationCreate',
    '/api/settings': 'SettingsUpdate',
    '/api/alerts': 'BaseModel',  # Generic validation
    '/api/transactions': 'TransactionRequest',
    '/api/users': 'BaseModel',
    '/api/audit': 'BaseModel',
    '/api/reports': 'BaseModel',
    '/api/dashboard': 'BaseModel',
    '/api/vip': 'VipListCreate',
    '/api/hot': 'HotListCreate',
    '/api/auth/register': 'BaseModel',
    '/api/auth/reset-password': 'BaseModel',
    '/api/auth/change-password': 'BaseModel',
}


def find_endpoints_without_validation(content: str) -> List[Tuple[int, str, str]]:
    """
    Encontra endpoints POST/PUT/PATCH sem validacao Pydantic

    Returns:
        List de (line_number, route, method)
    """
    lines = content.split('\n')
    endpoints = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Procurar @app.route com POST/PUT/PATCH
        route_match = re.search(r"@app\.route\(['\"]([^'\"]+)['\"][^)]*methods=\[.*(?:'POST'|'PUT'|'PATCH')", line)

        if route_match:
            route = route_match.group(1)

            # Verificar se ja tem validacao nas proximas 5 linhas
            has_validation = False
            for j in range(i, min(i + 10, len(lines))):
                if '**request.json' in lines[j] or '@validate_request' in lines[j]:
                    has_validation = True
                    break

            if not has_validation:
                # Encontrar nome da funcao
                for j in range(i + 1, min(i + 5, len(lines))):
                    func_match = re.search(r'def\s+(\w+)\s*\(', lines[j])
                    if func_match:
                        endpoints.append((i + 1, route, func_match.group(1)))
                        break

        i += 1

    return endpoints


def add_validation_decorator(content: str, endpoint_info: Tuple[int, str, str]) -> str:
    """
    Adiciona @validate_request antes do endpoint
    """
    line_num, route, func_name = endpoint_info
    lines = content.split('\n')

    # Encontrar linha do @app.route
    route_line_idx = None
    for i, line in enumerate(lines):
        if f"@app.route('{route}'" in line or f'@app.route("{route}"' in line:
            route_line_idx = i
            break

    if route_line_idx is None:
        return content

    # Adicionar @validate_request antes do @app.route
    # Usar BaseModel como validacao generica
    validation_line = f"@validate_request()  # TODO: Add specific schema"

    # Inserir linha
    lines.insert(route_line_idx, validation_line)

    return '\n'.join(lines)


def ensure_imports(content: str) -> str:
    """
    Garante que imports necessarios estao presentes
    """
    if 'from api.validation_wrapper import validate_request' in content:
        return content

    # Encontrar ultima linha de import
    lines = content.split('\n')
    last_import_idx = 0

    for i, line in enumerate(lines):
        if line.strip().startswith(('import ', 'from ')):
            last_import_idx = i

    # Adicionar import
    import_line = 'from api.validation_wrapper import validate_request'
    lines.insert(last_import_idx + 1, import_line)

    return '\n'.join(lines)


def main():
    """Main execution"""
    base_dir = Path(__file__).parent.parent
    api_file = base_dir / 'api' / 'production_api.py'

    print("="*70)
    print("APPLYING PYDANTIC VALIDATION TO ALL ENDPOINTS")
    print("="*70)

    # Ler arquivo
    content = api_file.read_text(encoding='utf-8')

    # Encontrar endpoints sem validacao
    endpoints = find_endpoints_without_validation(content)

    print(f"\nFound {len(endpoints)} endpoints without validation:")
    for line_num, route, func_name in endpoints:
        print(f"  Line {line_num}: {route} ({func_name})")

    if not endpoints:
        print("\n[OK] All endpoints already have validation!")
        return

    # Backup
    backup_path = api_file.with_suffix('.py.bak_validation')
    backup_path.write_text(content, encoding='utf-8')
    print(f"\n[BACKUP] Created: {backup_path}")

    # Adicionar imports
    content = ensure_imports(content)

    # Adicionar validacao em cada endpoint
    modified_count = 0
    for endpoint_info in endpoints:
        content = add_validation_decorator(content, endpoint_info)
        modified_count += 1

    # Escrever arquivo modificado
    api_file.write_text(content, encoding='utf-8')

    print(f"\n[SUCCESS] Added validation to {modified_count} endpoints")
    print("\n[TODO] Review and add specific schemas where marked with TODO")
    print("       Replace BaseModel with specific schemas like:")
    print("       - TransactionRequest")
    print("       - ManualReviewCreate")
    print("       - FeedbackCreate")
    print("       etc.")


if __name__ == '__main__':
    main()
