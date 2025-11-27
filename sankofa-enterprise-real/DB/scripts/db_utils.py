#!/usr/bin/env python3
"""
Sankofa Enterprise Pro - Database Utilities
Utilitários para manutenção e operações do banco de dados.
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import json


def get_connection():
    """Obtém conexão com o banco de dados."""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("ERRO: DATABASE_URL não definida")
        sys.exit(1)
    return psycopg2.connect(database_url)


def show_tables():
    """Lista todas as tabelas do banco."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    table_name,
                    pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as size,
                    (SELECT COUNT(*) FROM information_schema.columns 
                     WHERE table_name = t.table_name) as columns
                FROM information_schema.tables t
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            
            print("\n" + "=" * 60)
            print("TABELAS DO BANCO DE DADOS")
            print("=" * 60)
            print(f"{'Tabela':<30} {'Tamanho':<15} {'Colunas':<10}")
            print("-" * 60)
            
            for row in cur.fetchall():
                print(f"{row['table_name']:<30} {row['size']:<15} {row['columns']:<10}")
    finally:
        conn.close()


def show_indexes():
    """Lista todos os índices."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    tablename,
                    indexname,
                    pg_size_pretty(pg_relation_size(quote_ident(indexname)::regclass)) as size
                FROM pg_indexes
                WHERE schemaname = 'public'
                ORDER BY tablename, indexname
            """)
            
            print("\n" + "=" * 80)
            print("ÍNDICES DO BANCO DE DADOS")
            print("=" * 80)
            print(f"{'Tabela':<25} {'Índice':<40} {'Tamanho':<15}")
            print("-" * 80)
            
            for row in cur.fetchall():
                print(f"{row['tablename']:<25} {row['indexname']:<40} {row['size']:<15}")
    finally:
        conn.close()


def show_stats():
    """Mostra estatísticas do banco."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Tamanho do banco
            cur.execute("SELECT pg_size_pretty(pg_database_size(current_database())) as size")
            db_size = cur.fetchone()['size']
            
            # Contagem de tabelas
            cur.execute("""
                SELECT COUNT(*) as count FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)
            table_count = cur.fetchone()['count']
            
            # Contagem de índices
            cur.execute("SELECT COUNT(*) as count FROM pg_indexes WHERE schemaname = 'public'")
            index_count = cur.fetchone()['count']
            
            # Transações
            cur.execute("SELECT COUNT(*) as count FROM transactions")
            txn_count = cur.fetchone()['count']
            
            # Fraudes
            cur.execute("SELECT COUNT(*) as count FROM transactions WHERE is_fraud = TRUE")
            fraud_count = cur.fetchone()['count']
            
            print("\n" + "=" * 50)
            print("ESTATÍSTICAS DO BANCO DE DADOS")
            print("=" * 50)
            print(f"Tamanho total:     {db_size}")
            print(f"Tabelas:           {table_count}")
            print(f"Índices:           {index_count}")
            print(f"Transações:        {txn_count:,}")
            print(f"Fraudes:           {fraud_count:,}")
            if txn_count > 0:
                fraud_rate = (fraud_count / txn_count) * 100
                print(f"Taxa de fraude:    {fraud_rate:.2f}%")
            print("=" * 50)
    finally:
        conn.close()


def vacuum_analyze():
    """Executa VACUUM ANALYZE em todas as tabelas."""
    conn = get_connection()
    conn.autocommit = True  # VACUUM não pode rodar em transação
    try:
        with conn.cursor() as cur:
            print("\nExecutando VACUUM ANALYZE...")
            
            # Listar tabelas
            cur.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)
            tables = [row[0] for row in cur.fetchall()]
            
            for table in tables:
                print(f"  Processando {table}...")
                cur.execute(f"VACUUM ANALYZE {table}")
            
            print("\n✓ VACUUM ANALYZE concluído!")
    finally:
        conn.close()


def cleanup_old_data(days=90):
    """Remove dados antigos (exceto audit_trail)."""
    conn = get_connection()
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        print(f"\nRemovendo dados anteriores a {cutoff_date.strftime('%Y-%m-%d')}...")
        
        with conn.cursor() as cur:
            # Rate limits (manter apenas últimas 24h)
            cur.execute("DELETE FROM rate_limits WHERE window_start < NOW() - INTERVAL '24 hours'")
            rate_deleted = cur.rowcount
            
            conn.commit()
            
            print(f"  Rate limits removidos: {rate_deleted}")
            print("\n✓ Limpeza concluída!")
            print("\nNOTA: audit_trail não é afetado (retenção de 7 anos)")
    finally:
        conn.close()


def reset_sequences():
    """Reseta sequências para valores corretos."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Listar sequências
            cur.execute("""
                SELECT sequence_name FROM information_schema.sequences 
                WHERE sequence_schema = 'public'
            """)
            sequences = [row[0] for row in cur.fetchall()]
            
            print("\nResetando sequências...")
            for seq in sequences:
                # Obter nome da tabela e coluna
                table = seq.replace('_id_seq', '')
                if table:
                    try:
                        cur.execute(f"""
                            SELECT setval('{seq}', COALESCE((SELECT MAX(id) FROM {table}), 1))
                        """)
                        print(f"  ✓ {seq}")
                    except Exception as e:
                        print(f"  ✗ {seq}: {e}")
            
            conn.commit()
            print("\n✓ Sequências resetadas!")
    finally:
        conn.close()


def export_schema():
    """Exporta schema atual como DDL."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Este é um exemplo simplificado
            # Em produção, use pg_dump
            print("\nPara exportar o schema completo, use:")
            print(f"  pg_dump --schema-only $DATABASE_URL > schema_export.sql")
    finally:
        conn.close()


def main():
    """Função principal."""
    if len(sys.argv) < 2:
        print("""
Sankofa Database Utilities

Uso: python db_utils.py <comando>

Comandos:
  tables       Lista todas as tabelas
  indexes      Lista todos os índices
  stats        Mostra estatísticas do banco
  vacuum       Executa VACUUM ANALYZE
  cleanup      Remove dados antigos
  sequences    Reseta sequências
  export       Exporta schema
        """)
        return
    
    command = sys.argv[1]
    
    if command == 'tables':
        show_tables()
    elif command == 'indexes':
        show_indexes()
    elif command == 'stats':
        show_stats()
    elif command == 'vacuum':
        vacuum_analyze()
    elif command == 'cleanup':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 90
        cleanup_old_data(days)
    elif command == 'sequences':
        reset_sequences()
    elif command == 'export':
        export_schema()
    else:
        print(f"Comando desconhecido: {command}")


if __name__ == '__main__':
    main()
