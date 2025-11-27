#!/usr/bin/env python3
"""
Sankofa Enterprise Pro - Database Migration Script
Executa migrações pendentes no banco de dados.
"""

import os
import sys
import glob
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

# Diretório de migrações
MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'migrations')


def get_connection():
    """Obtém conexão com o banco de dados."""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("ERRO: DATABASE_URL não definida")
        sys.exit(1)
    
    return psycopg2.connect(database_url)


def ensure_migrations_table(conn):
    """Garante que a tabela de migrações existe."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(50) PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                applied_by VARCHAR(100),
                checksum VARCHAR(64)
            )
        """)
        conn.commit()


def get_applied_migrations(conn):
    """Retorna lista de migrações já aplicadas."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT version, checksum FROM schema_migrations ORDER BY version")
        return {row['version']: row['checksum'] for row in cur.fetchall()}


def get_pending_migrations(applied):
    """Retorna lista de migrações pendentes."""
    migration_files = sorted(glob.glob(os.path.join(MIGRATIONS_DIR, '*.sql')))
    pending = []
    
    for filepath in migration_files:
        filename = os.path.basename(filepath)
        version = filename.replace('.sql', '')
        
        if version not in applied:
            with open(filepath, 'r') as f:
                content = f.read()
                checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
            
            pending.append({
                'version': version,
                'filepath': filepath,
                'content': content,
                'checksum': checksum
            })
    
    return pending


def apply_migration(conn, migration):
    """Aplica uma migração."""
    version = migration['version']
    content = migration['content']
    checksum = migration['checksum']
    
    print(f"Aplicando migração: {version}...")
    
    try:
        with conn.cursor() as cur:
            # Executar SQL da migração
            cur.execute(content)
            
            # Registrar migração (se não foi registrada pelo próprio SQL)
            cur.execute("""
                INSERT INTO schema_migrations (version, applied_by, checksum)
                VALUES (%s, %s, %s)
                ON CONFLICT (version) DO UPDATE SET checksum = EXCLUDED.checksum
            """, (version, 'migrate.py', checksum))
            
            conn.commit()
            print(f"  ✓ Migração {version} aplicada com sucesso")
            return True
            
    except Exception as e:
        conn.rollback()
        print(f"  ✗ ERRO na migração {version}: {e}")
        return False


def run_migrations():
    """Executa todas as migrações pendentes."""
    print("=" * 60)
    print("Sankofa Enterprise Pro - Database Migration")
    print("=" * 60)
    print()
    
    conn = get_connection()
    
    try:
        # Garantir tabela de migrações
        ensure_migrations_table(conn)
        
        # Obter migrações aplicadas
        applied = get_applied_migrations(conn)
        print(f"Migrações já aplicadas: {len(applied)}")
        
        # Obter migrações pendentes
        pending = get_pending_migrations(applied)
        
        if not pending:
            print("\n✓ Banco de dados está atualizado!")
            return 0
        
        print(f"Migrações pendentes: {len(pending)}")
        print()
        
        # Aplicar migrações
        success_count = 0
        for migration in pending:
            if apply_migration(conn, migration):
                success_count += 1
            else:
                print("\n✗ Migração falhou. Abortando...")
                return 1
        
        print()
        print("=" * 60)
        print(f"✓ {success_count} migração(ões) aplicada(s) com sucesso!")
        print("=" * 60)
        
        return 0
        
    finally:
        conn.close()


def rollback_migration(version):
    """Reverte uma migração específica."""
    print(f"Revertendo migração: {version}...")
    
    # Procurar seção de rollback no arquivo
    filepath = os.path.join(MIGRATIONS_DIR, f"{version}.sql")
    
    if not os.path.exists(filepath):
        print(f"ERRO: Arquivo de migração não encontrado: {filepath}")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Procurar seção "Down Migration"
    if '-- Down Migration' in content:
        # Extrair SQL de rollback
        parts = content.split('-- Down Migration')
        if len(parts) > 1:
            rollback_sql = parts[1].strip()
            # Remover comentários de linha única
            rollback_sql = '\n'.join(
                line for line in rollback_sql.split('\n')
                if not line.strip().startswith('--')
            )
            
            if rollback_sql:
                conn = get_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute(rollback_sql)
                        cur.execute("DELETE FROM schema_migrations WHERE version = %s", (version,))
                        conn.commit()
                        print(f"  ✓ Migração {version} revertida com sucesso")
                        return True
                except Exception as e:
                    conn.rollback()
                    print(f"  ✗ ERRO ao reverter: {e}")
                    return False
                finally:
                    conn.close()
    
    print("  ✗ Seção de rollback não encontrada")
    return False


def show_status():
    """Mostra status das migrações."""
    print("=" * 60)
    print("Status das Migrações")
    print("=" * 60)
    print()
    
    conn = get_connection()
    
    try:
        ensure_migrations_table(conn)
        applied = get_applied_migrations(conn)
        pending = get_pending_migrations(applied)
        
        print("Migrações Aplicadas:")
        if applied:
            for version in sorted(applied.keys()):
                print(f"  ✓ {version}")
        else:
            print("  (nenhuma)")
        
        print()
        print("Migrações Pendentes:")
        if pending:
            for m in pending:
                print(f"  ○ {m['version']}")
        else:
            print("  (nenhuma)")
        
    finally:
        conn.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'status':
            show_status()
        elif command == 'rollback' and len(sys.argv) > 2:
            rollback_migration(sys.argv[2])
        else:
            print("Uso: python migrate.py [status|rollback <version>]")
    else:
        sys.exit(run_migrations())
