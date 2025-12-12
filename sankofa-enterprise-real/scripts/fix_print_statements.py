"""
Script to replace print() statements with proper logging
"""

import re
import sys
from pathlib import Path

def fix_print_statements(file_path: Path) -> int:
    """
    Replace print() with logger.error() in exception handlers

    Returns:
        Number of replacements made
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Pattern: print(f"Error something: {e}")
    # Replace with: logger.error(f"Error something: {e}", exc_info=True)
    pattern = r'(\s+)print\(f"(Error [^"]+: \{e\})"\)'
    replacement = r'\1logger.error(f"\2", exc_info=True)'

    content = re.sub(pattern, replacement, content)

    # Count replacements
    replacements = content.count('logger.error') - original_content.count('logger.error')

    if replacements > 0:
        # Add logger import if not present
        if 'import logging' not in content:
            # Find first import
            first_import_match = re.search(r'^import |^from ', content, re.MULTILINE)
            if first_import_match:
                insert_pos = first_import_match.start()
                content = content[:insert_pos] + 'import logging\n' + content[insert_pos:]

        # Add logger declaration if not present
        if 'logger = logging.getLogger(__name__)' not in content:
            # Find after imports (after first blank line after imports)
            imports_end = 0
            for match in re.finditer(r'\n\n', content):
                if match.start() > 100:  # After imports section
                    imports_end = match.end()
                    break

            if imports_end > 0:
                content = content[:imports_end] + '\nlogger = logging.getLogger(__name__)\n' + content[imports_end:]

        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    return replacements


def main():
    # Files to fix
    files_to_fix = [
        'backend/api/services/postgres_store.py',
        'backend/api/production_api.py',
        'backend/ml_engine/production_fraud_engine.py',
    ]

    root = Path(__file__).parent.parent
    total_replacements = 0

    for file_path_str in files_to_fix:
        file_path = root / file_path_str

        if not file_path.exists():
            print(f"[WARN] File not found: {file_path}")
            continue

        replacements = fix_print_statements(file_path)
        total_replacements += replacements

        if replacements > 0:
            print(f"[OK] Fixed {replacements} print statements in {file_path.name}")
        else:
            print(f"[INFO] No print statements found in {file_path.name}")

    print(f"\n[SUCCESS] Total: {total_replacements} print statements replaced with logger.error()")
    return 0


if __name__ == '__main__':
    sys.exit(main())
