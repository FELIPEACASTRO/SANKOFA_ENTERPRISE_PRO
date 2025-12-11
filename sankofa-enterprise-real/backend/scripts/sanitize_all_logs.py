"""
Automated Log Sanitization Script
Applies log_sanitizer.sanitize_log_data() to ALL logger calls in production_api.py
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import ast


def read_file(file_path: Path) -> str:
    """Lê conteúdo do arquivo"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(file_path: Path, content: str):
    """Escreve conteúdo no arquivo"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def find_logger_calls(content: str) -> List[Tuple[int, str, str]]:
    """
    Encontra todas as chamadas de logger no código

    Returns:
        List de tuplas (line_number, log_level, full_line)
    """
    logger_pattern = r'^\s*logger\.(debug|info|warning|error|critical)\('

    lines = content.split('\n')
    logger_calls = []

    for i, line in enumerate(lines, 1):
        if re.search(logger_pattern, line):
            # Extrair nível de log
            match = re.search(r'logger\.(\w+)\(', line)
            if match:
                log_level = match.group(1)
                logger_calls.append((i, log_level, line))

    return logger_calls


def needs_sanitization(line: str) -> bool:
    """
    Verifica se a linha de log precisa de sanitização

    Returns:
        True se a linha não tem sanitize_log_data() e tem variáveis/f-strings
    """
    # Já sanitizado
    if 'sanitize_log_data' in line:
        return False

    # Tem f-string ou variáveis que podem conter PII
    has_fstring = re.search(r'f["\']', line)
    has_variables = re.search(r'\{[^}]+\}', line) or re.search(r'extra\s*=', line)
    has_formatting = re.search(r'%[sd]', line)

    return bool(has_fstring or has_variables or has_formatting)


def sanitize_logger_call(line: str, log_level: str) -> str:
    """
    Transforma uma linha de log para usar sanitize_log_data()

    Exemplos:
        logger.info(f"User {username} logged in")
        -> logger.info("User logged in", extra=sanitize_log_data({'username': username}))

        logger.warning(f"Failed login attempt for {user}", extra={'ip': ip_address})
        -> logger.warning("Failed login attempt", extra=sanitize_log_data({'user': user, 'ip': ip_address}))
    """
    indent = len(line) - len(line.lstrip())
    indent_str = ' ' * indent

    # Caso 1: f-string simples: logger.info(f"message {var}")
    fstring_pattern = r'logger\.(\w+)\(f["\']([^"\']*)["\'](?:,\s*(.*))?'
    match = re.match(fstring_pattern, line.strip())

    if match:
        level = match.group(1)
        message = match.group(2)
        extra_args = match.group(3) or ""

        # Extrair variáveis do f-string
        variables = re.findall(r'\{([^}]+)\}', message)

        if not variables:
            return line  # Sem variáveis, não precisa sanitizar

        # Limpar mensagem (remover {variables})
        clean_message = re.sub(r'\{[^}]+\}', '', message).strip()
        clean_message = re.sub(r'\s+', ' ', clean_message)  # Normalizar espaços

        # Criar dict de extra
        extra_dict = '{' + ', '.join([f"'{v}': {v}" for v in variables]) + '}'

        # Se já tem extra, merge
        if 'extra=' in extra_args:
            # Extrair dict existente
            existing_extra_match = re.search(r'extra\s*=\s*(\{[^}]+\})', extra_args)
            if existing_extra_match:
                existing_dict = existing_extra_match.group(1)
                # Merge dicts
                extra_dict = existing_dict.replace('}', f", {', '.join([f'{v!r}: {v}' for v in variables])}}}")

        # Construir nova linha
        if extra_args and not 'extra=' in extra_args:
            # Tem outros args, adicionar extra
            new_line = f'{indent_str}logger.{level}("{clean_message}", extra=sanitize_log_data({extra_dict}), {extra_args})\n'
        else:
            new_line = f'{indent_str}logger.{level}("{clean_message}", extra=sanitize_log_data({extra_dict}))\n'

        return new_line

    # Caso 2: logger com extra já existente, mas sem sanitize
    extra_pattern = r'logger\.(\w+)\((.+?),\s*extra\s*=\s*(\{.+\})'
    match = re.search(extra_pattern, line)

    if match:
        level = match.group(1)
        message = match.group(2).strip()
        extra_dict = match.group(3)

        # Já tem extra, apenas envolver com sanitize_log_data
        new_line = line.replace(f'extra={extra_dict}', f'extra=sanitize_log_data({extra_dict})')
        return new_line

    # Caso 3: logger simples sem extra (pode ter % formatting ou .format())
    if '%' in line or '.format(' in line:
        # Converter para extra dict
        simple_pattern = r'logger\.(\w+)\((.+)\)'
        match = re.search(simple_pattern, line.strip())

        if match:
            level = match.group(1)
            full_message = match.group(2)

            # Por segurança, adicionar extra vazio sanitizado
            new_line = f'{indent_str}logger.{level}({full_message}, extra=sanitize_log_data({{}}))\n'
            return new_line

    return line  # Não conseguiu processar, manter original


def sanitize_file(file_path: Path, dry_run: bool = False) -> Dict[str, int]:
    """
    Sanitiza todos os logs em um arquivo

    Args:
        file_path: Caminho do arquivo
        dry_run: Se True, apenas mostra o que seria feito sem modificar

    Returns:
        Dict com estatísticas
    """
    print(f"\n[FILE] Processing: {file_path}")

    content = read_file(file_path)
    lines = content.split('\n')

    # Encontrar todos os logger calls
    logger_calls = find_logger_calls(content)
    print(f"   Found {len(logger_calls)} logger calls")

    # Filtrar os que precisam de sanitização
    to_sanitize = []
    for line_num, log_level, line in logger_calls:
        if needs_sanitization(line):
            to_sanitize.append((line_num, log_level, line))

    print(f"   {len(to_sanitize)} calls need sanitization")

    if not to_sanitize:
        print("   [OK] Already fully sanitized!")
        return {'total': len(logger_calls), 'sanitized': 0, 'skipped': len(logger_calls)}

    # Aplicar sanitização
    modified_count = 0
    new_lines = lines.copy()

    for line_num, log_level, original_line in to_sanitize:
        line_idx = line_num - 1

        sanitized_line = sanitize_logger_call(original_line, log_level)

        if sanitized_line != original_line:
            new_lines[line_idx] = sanitized_line.rstrip('\n')
            modified_count += 1

            if dry_run:
                print(f"\n   Line {line_num}:")
                print(f"     BEFORE: {original_line.strip()}")
                print(f"     AFTER:  {sanitized_line.strip()}")

    if not dry_run and modified_count > 0:
        # Verificar se sanitize_log_data está importado
        if 'from utils.log_sanitizer import sanitize_log_data' not in content:
            print("   [WARN] Adding missing import: from utils.log_sanitizer import sanitize_log_data")

            # Encontrar última linha de import
            import_lines = []
            for i, line in enumerate(new_lines):
                if line.strip().startswith(('import ', 'from ')):
                    import_lines.append(i)

            if import_lines:
                last_import_idx = max(import_lines)
                new_lines.insert(last_import_idx + 1, 'from utils.log_sanitizer import sanitize_log_data')

        # Escrever arquivo modificado
        new_content = '\n'.join(new_lines)

        # Backup
        backup_path = file_path.with_suffix('.py.bak_logs')
        write_file(backup_path, content)
        print(f"   [BACKUP] Backup created: {backup_path}")

        # Escrever modificado
        write_file(file_path, new_content)
        print(f"   [SUCCESS] Sanitized {modified_count} logger calls")

    return {
        'total': len(logger_calls),
        'sanitized': modified_count,
        'skipped': len(logger_calls) - modified_count
    }


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Sanitize logger calls in Python files')
    parser.add_argument('files', nargs='*', help='Files to sanitize (default: production_api.py)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without modifying files')
    parser.add_argument('--all', action='store_true', help='Sanitize all Python files in api/ and services/')

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # Determinar arquivos a processar
    if args.all:
        files_to_process = []
        files_to_process.extend((base_dir / 'api').glob('*.py'))
        files_to_process.extend((base_dir / 'api' / 'services').glob('*.py'))
    elif args.files:
        files_to_process = [Path(f) for f in args.files]
    else:
        # Default: apenas production_api.py
        files_to_process = [base_dir / 'api' / 'production_api.py']

    print(f"{'='*60}")
    print(f"LOG SANITIZATION SCRIPT")
    print(f"{'='*60}")
    print(f"Mode: {'DRY RUN (no changes)' if args.dry_run else 'LIVE (will modify files)'}")
    print(f"Files to process: {len(files_to_process)}")

    total_stats = {
        'files_processed': 0,
        'total_loggers': 0,
        'sanitized': 0,
        'skipped': 0
    }

    for file_path in files_to_process:
        if not file_path.exists():
            print(f"\n[ERROR] File not found: {file_path}")
            continue

        try:
            stats = sanitize_file(file_path, dry_run=args.dry_run)

            total_stats['files_processed'] += 1
            total_stats['total_loggers'] += stats['total']
            total_stats['sanitized'] += stats['sanitized']
            total_stats['skipped'] += stats['skipped']

        except Exception as e:
            print(f"\n[ERROR] Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Total logger calls: {total_stats['total_loggers']}")
    print(f"Sanitized: {total_stats['sanitized']}")
    print(f"Already sanitized: {total_stats['skipped']}")
    print(f"Sanitization coverage: {(total_stats['skipped'] / total_stats['total_loggers'] * 100) if total_stats['total_loggers'] > 0 else 0:.1f}%")

    if args.dry_run:
        print(f"\n[WARN] This was a DRY RUN - no files were modified")
        print(f"       Run without --dry-run to apply changes")
    else:
        print(f"\n[SUCCESS] Log sanitization complete!")


if __name__ == '__main__':
    main()
