#!/bin/bash
# =====================================================
# Sankofa Enterprise Pro - Database Restore Script
# =====================================================
#
# Uso: ./restore.sh <backup_file>
#
# ATENÇÃO: Este script irá SOBRESCREVER todos os dados!
# =====================================================

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================"
echo " Sankofa Enterprise Pro - Database Restore"
echo "======================================================"
echo ""

# Verificar argumento
if [ -z "$1" ]; then
    echo -e "${RED}ERRO: Arquivo de backup não especificado${NC}"
    echo ""
    echo "Uso: ./restore.sh <backup_file>"
    echo ""
    echo "Backups disponíveis:"
    ls -lh ./backups/sankofa_backup_*.dump 2>/dev/null || echo "  (nenhum encontrado)"
    exit 1
fi

BACKUP_FILE="$1"

# Verificar se arquivo existe
if [ ! -f "$BACKUP_FILE" ]; then
    echo -e "${RED}ERRO: Arquivo não encontrado: ${BACKUP_FILE}${NC}"
    exit 1
fi

# Verificar variáveis de ambiente
if [ -z "$DATABASE_URL" ] && [ -z "$PGHOST" ]; then
    echo -e "${RED}ERRO: DATABASE_URL ou PGHOST não definida${NC}"
    exit 1
fi

# Confirmação
echo -e "${RED}ATENÇÃO: Este processo irá SOBRESCREVER todos os dados!${NC}"
echo ""
echo "Arquivo: ${BACKUP_FILE}"
echo "Tamanho: $(du -h "$BACKUP_FILE" | cut -f1)"
echo ""
read -p "Tem certeza que deseja continuar? (digite 'SIM' para confirmar): " confirm

if [ "$confirm" != "SIM" ]; then
    echo ""
    echo -e "${YELLOW}Operação cancelada.${NC}"
    exit 0
fi

echo ""
echo -e "${YELLOW}Iniciando restore...${NC}"

# Executar restore
if [ -n "$DATABASE_URL" ]; then
    pg_restore "$DATABASE_URL" \
        --clean \
        --if-exists \
        --no-owner \
        --no-privileges \
        --verbose \
        "$BACKUP_FILE"
else
    PGPASSWORD=$PGPASSWORD pg_restore \
        -h "$PGHOST" \
        -p "${PGPORT:-5432}" \
        -U "$PGUSER" \
        -d "$PGDATABASE" \
        --clean \
        --if-exists \
        --no-owner \
        --no-privileges \
        --verbose \
        "$BACKUP_FILE"
fi

# Verificar sucesso
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================================"
    echo " Restore concluído com sucesso!"
    echo "======================================================"
    echo " Arquivo restaurado: ${BACKUP_FILE}"
    echo " Data: $(date)"
    echo "======================================================${NC}"
else
    echo -e "${RED}ERRO: Restore falhou!${NC}"
    exit 1
fi
