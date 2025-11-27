#!/bin/bash
# =====================================================
# Sankofa Enterprise Pro - Database Backup Script
# =====================================================
#
# Uso: ./backup.sh [output_dir]
#
# Variáveis de ambiente necessárias:
#   - DATABASE_URL ou PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE
# =====================================================

set -e

# Configurações
BACKUP_DIR="${1:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="sankofa_backup_${TIMESTAMP}.sql"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_FILE}"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================"
echo " Sankofa Enterprise Pro - Database Backup"
echo "======================================================"
echo ""

# Verificar variáveis de ambiente
if [ -z "$DATABASE_URL" ] && [ -z "$PGHOST" ]; then
    echo -e "${RED}ERRO: DATABASE_URL ou PGHOST não definida${NC}"
    exit 1
fi

# Criar diretório de backup
mkdir -p "$BACKUP_DIR"
echo -e "${YELLOW}Diretório de backup: ${BACKUP_DIR}${NC}"

# Executar backup
echo -e "${YELLOW}Iniciando backup...${NC}"

if [ -n "$DATABASE_URL" ]; then
    pg_dump "$DATABASE_URL" \
        --format=custom \
        --file="${BACKUP_PATH}.dump" \
        --verbose \
        --no-owner \
        --no-privileges
else
    PGPASSWORD=$PGPASSWORD pg_dump \
        -h "$PGHOST" \
        -p "${PGPORT:-5432}" \
        -U "$PGUSER" \
        -d "$PGDATABASE" \
        --format=custom \
        --file="${BACKUP_PATH}.dump" \
        --verbose \
        --no-owner \
        --no-privileges
fi

# Verificar sucesso
if [ $? -eq 0 ]; then
    # Tamanho do arquivo
    SIZE=$(du -h "${BACKUP_PATH}.dump" | cut -f1)
    
    echo ""
    echo -e "${GREEN}======================================================"
    echo " Backup concluído com sucesso!"
    echo "======================================================"
    echo " Arquivo: ${BACKUP_PATH}.dump"
    echo " Tamanho: ${SIZE}"
    echo " Data: $(date)"
    echo "======================================================${NC}"
    
    # Limpar backups antigos (manter últimos 7)
    echo ""
    echo -e "${YELLOW}Limpando backups antigos (mantendo últimos 7)...${NC}"
    ls -t ${BACKUP_DIR}/sankofa_backup_*.dump 2>/dev/null | tail -n +8 | xargs -r rm -v
    
    echo ""
    echo -e "${GREEN}Backup completo!${NC}"
else
    echo -e "${RED}ERRO: Backup falhou!${NC}"
    exit 1
fi
