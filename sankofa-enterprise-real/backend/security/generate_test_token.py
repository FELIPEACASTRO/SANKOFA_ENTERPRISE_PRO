import logging

logger = logging.getLogger(__name__)
"""
Script para gerar um token de autenticação para testes.
"""

import os
import sys

# Adiciona o diretório pai ao path para importações
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from security.enterprise_security_system import EnterpriseSecuritySystem


def generate_token():
    security_system = EnterpriseSecuritySystem()

    # 1. Criar um usuário de teste se não existir
    try:
        security_system.create_user(
            "test_admin", "test_admin@sankofa.com", "StrongPassword123!", "admin"
        )
        logger.info("Usuário de teste 'test_admin' criado com sucesso.")
    except Exception as e:
        if "UNIQUE constraint failed" in str(e):
            logger.info("Usuário de teste 'test_admin' já existe.")
        else:
            logger.info(f"Erro ao criar usuário: {e}")

    # 2. Autenticar o usuário e obter o token
    auth_data = security_system.authenticate_user(
        "test_admin", "StrongPassword123!", "127.0.0.1", "Test-Client/1.0"
    )

    access_token = auth_data.get("access_token")
    logger.info(f"\n--- TOKEN DE ACESSO GERADO ---\n")
    logger.info(access_token)
    logger.info(f"\n--- FIM DO TOKEN ---\n")


if __name__ == "__main__":
    generate_token()
