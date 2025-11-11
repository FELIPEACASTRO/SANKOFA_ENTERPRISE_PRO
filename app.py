#!/usr/bin/env python3
"""
🏦 Sankofa Enterprise Pro - Ponto de Entrada Principal
Sistema de Detecção de Fraude Bancária de Classe Mundial

Este é o arquivo principal que inicia toda a aplicação.
"""

import os
import sys
import logging
from pathlib import Path

# Adiciona o diretório do projeto ao Python path
PROJECT_ROOT = Path(__file__).parent
BACKEND_PATH = PROJECT_ROOT / "sankofa-enterprise-real" / "backend"
sys.path.insert(0, str(BACKEND_PATH))

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sankofa_enterprise.log')
    ]
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Configura o ambiente da aplicação"""
    logger.info("🔧 Configurando ambiente...")
    
    # Define variáveis de ambiente padrão se não estiverem definidas
    env_defaults = {
        'ENVIRONMENT': 'development',
        'FLASK_DEBUG': 'false',
        'VERIFY_SSL_CERTS': 'true',
        'JWT_SECRET': 'sankofa-enterprise-secret-key-2024-change-in-production',
        'DB_HOST': 'localhost',
        'DB_PORT': '5432',
        'DB_NAME': 'sankofa_fraud_db',
        'REDIS_HOST': 'localhost',
        'REDIS_PORT': '6379',
        'API_PORT': '8445',
        'FRONTEND_PORT': '5000'
    }
    
    for key, default_value in env_defaults.items():
        if key not in os.environ:
            os.environ[key] = default_value
    
    # Validações de segurança
    environment = os.getenv('ENVIRONMENT')
    if environment == 'production':
        # Em produção, certas configurações devem ser obrigatórias
        required_prod_vars = ['JWT_SECRET', 'DB_PASSWORD', 'REDIS_PASSWORD']
        missing_vars = [var for var in required_prod_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"❌ Variáveis obrigatórias em produção não definidas: {missing_vars}")
            sys.exit(1)
        
        if os.getenv('FLASK_DEBUG', '').lower() == 'true':
            logger.error("❌ DEBUG MODE não pode estar ativo em produção!")
            sys.exit(1)
    
    logger.info(f"✅ Ambiente configurado: {environment}")

def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    logger.info("📦 Verificando dependências...")
    
    required_packages = [
        'flask', 'redis', 'pandas', 'numpy', 'scikit-learn',
        'psycopg2', 'cryptography', 'structlog'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Pacotes não instalados: {missing_packages}")
        logger.info("💡 Execute: pip install -r sankofa-enterprise-real/backend/requirements.txt")
        sys.exit(1)
    
    logger.info("✅ Todas as dependências estão instaladas")

def start_backend():
    """Inicia o backend da aplicação"""
    logger.info("🚀 Iniciando backend...")
    
    try:
        # Importa e inicia a API principal
        from api.main_integrated_api import app, logger as api_logger
        
        # Configurações do servidor
        environment = os.getenv('ENVIRONMENT')
        debug_mode = environment == 'development' and os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        host = "127.0.0.1" if environment == 'development' else "0.0.0.0"
        port = int(os.getenv('API_PORT', 8445))
        
        if debug_mode:
            api_logger.warning("⚠️  DEBUG MODE ATIVO - Use apenas em desenvolvimento!")
        
        api_logger.info(f"🌐 Servidor iniciando em http://{host}:{port}")
        api_logger.info("📊 Dashboard disponível em: http://localhost:5000")
        api_logger.info("🔍 Health check: http://localhost:8445/api/health")
        
        # Inicia o servidor
        app.run(
            host=host,
            port=port,
            debug=debug_mode,
            threaded=True,
            use_reloader=False  # Evita reinicialização dupla
        )
        
    except ImportError as e:
        logger.error(f"❌ Erro ao importar módulos do backend: {e}")
        logger.info("💡 Verifique se está no diretório correto e se as dependências estão instaladas")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar backend: {e}")
        sys.exit(1)

def show_startup_info():
    """Mostra informações de inicialização"""
    print("=" * 70)
    print("🏦 SANKOFA ENTERPRISE PRO - Sistema de Detecção de Fraude")
    print("=" * 70)
    print(f"📁 Diretório do projeto: {PROJECT_ROOT}")
    print(f"🌍 Ambiente: {os.getenv('ENVIRONMENT')}")
    print(f"🔧 Debug: {os.getenv('FLASK_DEBUG')}")
    print(f"🔒 SSL Verification: {os.getenv('VERIFY_SSL_CERTS')}")
    print("=" * 70)
    print()

def main():
    """Função principal"""
    try:
        show_startup_info()
        setup_environment()
        check_dependencies()
        start_backend()
        
    except KeyboardInterrupt:
        logger.info("👋 Aplicação interrompida pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro crítico: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()