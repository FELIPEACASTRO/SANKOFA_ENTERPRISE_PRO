"""
Sankofa Enterprise Pro - Configuração Centralizada
Sistema de configuração enterprise com variáveis de ambiente
"""

import os
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Configuração do banco de dados"""
    host: str
    port: int
    name: str
    user: str
    password: str
    pool_size: int
    pool_timeout: int
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Carrega configuração de variáveis de ambiente"""
        return cls(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            name=os.getenv('DB_NAME', 'sankofa_fraud_db'),
            user=os.getenv('DB_USER', 'sankofa'),
            password=os.getenv('DB_PASSWORD', ''),
            pool_size=int(os.getenv('DB_POOL_SIZE', '20')),
            pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', '30'))
        )
    
    @property
    def connection_string(self) -> str:
        """String de conexão PostgreSQL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

@dataclass
class RedisConfig:
    """Configuração do Redis"""
    host: str
    port: int
    password: Optional[str]
    db: int
    pool_size: int
    ttl_default: int
    
    @classmethod
    def from_env(cls) -> 'RedisConfig':
        """Carrega configuração de variáveis de ambiente"""
        return cls(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            password=os.getenv('REDIS_PASSWORD'),
            db=int(os.getenv('REDIS_DB', '0')),
            pool_size=int(os.getenv('REDIS_POOL_SIZE', '50')),
            ttl_default=int(os.getenv('REDIS_TTL_DEFAULT', '3600'))
        )

@dataclass
class SecurityConfig:
    """Configuração de segurança"""
    jwt_secret: str
    jwt_algorithm: str
    jwt_expiration: int
    encryption_key: str
    rate_limit_per_minute: int
    max_login_attempts: int
    
    @classmethod
    def from_env(cls) -> 'SecurityConfig':
        """Carrega configuração de variáveis de ambiente"""
        jwt_secret = os.getenv('JWT_SECRET')
        if not jwt_secret:
            raise ValueError("JWT_SECRET environment variable is required for production")
        
        encryption_key = os.getenv('ENCRYPTION_KEY')
        if not encryption_key:
            raise ValueError("ENCRYPTION_KEY environment variable is required for production")
        
        return cls(
            jwt_secret=jwt_secret,
            jwt_algorithm=os.getenv('JWT_ALGORITHM', 'HS256'),
            jwt_expiration=int(os.getenv('JWT_EXPIRATION', '3600')),
            encryption_key=encryption_key,
            rate_limit_per_minute=int(os.getenv('RATE_LIMIT_PER_MINUTE', '1000')),
            max_login_attempts=int(os.getenv('MAX_LOGIN_ATTEMPTS', '5'))
        )

@dataclass
class MLConfig:
    """Configuração do sistema de ML"""
    model_path: str
    confidence_threshold: float
    batch_size: int
    max_latency_ms: float
    enable_continuous_learning: bool
    drift_detection_window: int
    
    @classmethod
    def from_env(cls) -> 'MLConfig':
        """Carrega configuração de variáveis de ambiente"""
        return cls(
            model_path=os.getenv('ML_MODEL_PATH', './models'),
            confidence_threshold=float(os.getenv('ML_CONFIDENCE_THRESHOLD', '0.5')),
            batch_size=int(os.getenv('ML_BATCH_SIZE', '100')),
            max_latency_ms=float(os.getenv('ML_MAX_LATENCY_MS', '50.0')),
            enable_continuous_learning=os.getenv('ML_ENABLE_CONTINUOUS_LEARNING', 'true').lower() == 'true',
            drift_detection_window=int(os.getenv('ML_DRIFT_DETECTION_WINDOW', '10000'))
        )

@dataclass
class MonitoringConfig:
    """Configuração de monitoring"""
    datadog_api_key: Optional[str]
    datadog_app_key: Optional[str]
    prometheus_port: int
    log_level: str
    enable_metrics: bool
    
    @classmethod
    def from_env(cls) -> 'MonitoringConfig':
        """Carrega configuração de variáveis de ambiente"""
        return cls(
            datadog_api_key=os.getenv('DATADOG_API_KEY'),
            datadog_app_key=os.getenv('DATADOG_APP_KEY'),
            prometheus_port=int(os.getenv('PROMETHEUS_PORT', '9090')),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            enable_metrics=os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
        )

@dataclass
class ComplianceConfig:
    """Configuração de compliance"""
    bacen_api_url: Optional[str]
    bacen_institution_code: Optional[str]
    enable_lgpd: bool
    enable_pci_dss: bool
    audit_retention_days: int
    
    @classmethod
    def from_env(cls) -> 'ComplianceConfig':
        """Carrega configuração de variáveis de ambiente"""
        return cls(
            bacen_api_url=os.getenv('BACEN_API_URL'),
            bacen_institution_code=os.getenv('BACEN_INSTITUTION_CODE'),
            enable_lgpd=os.getenv('ENABLE_LGPD', 'true').lower() == 'true',
            enable_pci_dss=os.getenv('ENABLE_PCI_DSS', 'true').lower() == 'true',
            audit_retention_days=int(os.getenv('AUDIT_RETENTION_DAYS', '2555'))  # 7 anos BACEN
        )

@dataclass
class AppConfig:
    """Configuração completa da aplicação"""
    environment: str
    debug: bool
    database: DatabaseConfig
    redis: RedisConfig
    security: SecurityConfig
    ml: MLConfig
    monitoring: MonitoringConfig
    compliance: ComplianceConfig
    
    @classmethod
    def load(cls) -> 'AppConfig':
        """Carrega toda a configuração"""
        environment = os.getenv('ENVIRONMENT', 'development')
        debug = os.getenv('DEBUG', 'false').lower() == 'true'
        
        # Em desenvolvimento, permitir valores padrão
        # Em produção, exigir configuração completa
        if environment == 'production':
            security = SecurityConfig.from_env()
        else:
            # Development mode - usar valores padrão seguros
            try:
                security = SecurityConfig.from_env()
            except ValueError:
                logger.warning("⚠️  Using development security defaults - NOT FOR PRODUCTION")
                security = SecurityConfig(
                    jwt_secret='dev-secret-change-in-production',
                    jwt_algorithm='HS256',
                    jwt_expiration=3600,
                    encryption_key='dev-encryption-key-change-in-production',
                    rate_limit_per_minute=1000,
                    max_login_attempts=5
                )
        
        return cls(
            environment=environment,
            debug=debug,
            database=DatabaseConfig.from_env(),
            redis=RedisConfig.from_env(),
            security=security,
            ml=MLConfig.from_env(),
            monitoring=MonitoringConfig.from_env(),
            compliance=ComplianceConfig.from_env()
        )
    
    def validate(self):
        """Valida a configuração"""
        errors = []
        
        # Validações de produção
        if self.environment == 'production':
            if self.security.jwt_secret == 'dev-secret-change-in-production':
                errors.append("JWT_SECRET must be set in production")
            
            if self.security.encryption_key == 'dev-encryption-key-change-in-production':
                errors.append("ENCRYPTION_KEY must be set in production")
            
            if not self.database.password:
                errors.append("DB_PASSWORD must be set in production")
            
            if self.debug:
                errors.append("DEBUG must be false in production")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors))
        
        logger.info(f"✅ Configuration validated for environment: {self.environment}")

# Singleton global
_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Obtém a configuração global"""
    global _config
    if _config is None:
        _config = AppConfig.load()
        _config.validate()
    return _config

def reload_config():
    """Recarrega a configuração"""
    global _config
    _config = None
    return get_config()
