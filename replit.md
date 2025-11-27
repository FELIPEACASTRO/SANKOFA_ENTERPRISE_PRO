# Sankofa Enterprise Pro - Fraud Detection System

## Overview

Sankofa Enterprise Pro is a production-ready fraud detection system designed for banking environments processing 300M+ requests/day. The system combines machine learning ensemble models, real-time transaction analysis, MLOps infrastructure, and regulatory compliance (BACEN, LGPD, PCI DSS) with a React-based dashboard interface.

## Documentation

Complete documentation is available in the `sankofa-enterprise-real/docs/` folder:

| Document | Description |
|----------|-------------|
| [ARQUITETURA_TECNICA.md](sankofa-enterprise-real/docs/ARQUITETURA_TECNICA.md) | Technical architecture, stack, components, APIs |
| [DOCUMENTACAO_FUNCIONAL.md](sankofa-enterprise-real/docs/DOCUMENTACAO_FUNCIONAL.md) | Use cases, business rules, compliance |
| [DIAGRAMAS.md](sankofa-enterprise-real/docs/DIAGRAMAS.md) | Flowcharts, architecture diagrams (Mermaid + ASCII) |
| [MANUAL_USUARIO.md](sankofa-enterprise-real/docs/MANUAL_USUARIO.md) | User guide for dashboard operation |

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Architecture Pattern

The system follows a **Clean Architecture** approach with clear separation between:

- **Domain Layer**: Core business entities (`Transaction`, `FraudPrediction`) and value objects (`Money`, `TransactionId`)
- **Application Layer**: Use cases and business logic orchestration
- **Infrastructure Layer**: External services, APIs, database, cache
- **Presentation Layer**: React frontend dashboard

### Technology Stack

**Backend**:
- Python 3.11+ with Flask/FastAPI for REST API
- Machine Learning: scikit-learn, XGBoost, LightGBM (ensemble models)
- Caching: Redis for high-performance data access
- Database: PostgreSQL (optional, with SQLAlchemy ORM)

**Frontend**:
- React with Vite for development
- Modern JavaScript (ES6+)
- Component-based UI architecture

**Infrastructure**:
- Docker Compose for orchestration
- Nginx for load balancing
- DataDog integration for monitoring

### ML Engine Architecture

The fraud detection engine uses a **Stacking Ensemble** approach:

1. **Base Models (Layer 0)**:
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - Extra Trees Classifier
   - Logistic Regression
   - Support Vector Classifier

2. **Calibration Layer**: Probability calibration using `CalibratedClassifierCV`

3. **Meta-Model (Layer 1)**: Logistic Regression combining calibrated predictions

4. **Feature Engineering**: 47+ automated feature extraction techniques including:
   - Temporal features (hour, day, weekend patterns)
   - Value-based features (log, squared, statistical)
   - Geographic features (distance calculations, location risk)
   - Behavioral features (transaction velocity, patterns)

### MLOps Components

**Drift Detection** (`drift_detector.py`):
- Jensen-Shannon divergence for distribution monitoring
- Chi-square tests for categorical features
- Automatic severity level classification
- Alerts when model performance degrades

**A/B Testing** (`ab_testing_manager.py`):
- Traffic splitting between model variants
- Hash-based consistent routing
- Statistical significance testing
- Automated variant comparison

**Canary Deployment** (`canary_deployment_manager.py`):
- Gradual traffic rollout (steps: 5%, 10%, 25%, 50%, 100%)
- Health checks and automatic rollback
- Performance monitoring during deployment

**Model Lifecycle** (`model_lifecycle_manager.py`):
- Version management with joblib serialization
- Model registry with metadata
- Champion-challenger pattern support

### API Architecture

**Main API** (`production_api.py`):
- 30+ REST endpoints for fraud detection, model management, and configuration
- JWT authentication with 30-day key rotation
- HTTPS/TLS 1.3 encryption
- Role-based access control (RBAC)
- Rate limiting for DDoS protection

**Key Endpoints**:
- `/api/fraud/predict` - Real-time fraud detection
- `/api/fraud/batch` - Batch processing
- `/api/model/metrics` - Model performance metrics
- `/api/model/train` - Trigger model retraining
- `/api/feedback` - Human analyst feedback loop
- `/api/dashboard/*` - Dashboard data endpoints

### Caching Strategy

**Redis Cache System** (`redis_cache_system.py`):
- Multi-layer caching (Redis + in-memory fallback)
- Connection pooling (max 100 connections)
- Automatic serialization (JSON + Pickle)
- TTL-based cache invalidation
- Cache hit rate monitoring

### Security Architecture

**Authentication & Authorization**:
- JWT tokens with automatic rotation
- Secret key management via environment variables
- TLS 1.3 for all communications
- AES-256 encryption for data at rest

**Compliance**:
- BACEN Resolution 6/2023 compliance (fraud data sharing)
- LGPD compliance (Brazilian data protection law)
- PCI DSS adherence for payment data
- Audit trail for all operations

### Performance Characteristics

**Validated Metrics**:
- Throughput: 118,720+ TPS (transactions per second)
- Latency P95: 11ms
- Latency P99: 11.35ms
- Availability: 99.9%

**ML Performance**:
- Recall: 90.9% (fraud detection rate)
- Precision: 100% (no false positives in tests)
- F1-Score: 95.2%

## External Dependencies

### Required Services

**Redis** (v7.0+):
- Purpose: High-performance caching and session storage
- Configuration: Environment variable `REDIS_HOST`, `REDIS_PORT`
- Fallback: In-memory cache if Redis unavailable

**PostgreSQL** (v13+, optional):
- Purpose: Persistent data storage
- Configuration: `DATABASE_URL` environment variable
- Note: System can run without database using file-based storage

**Kaggle API** (for dataset access):
- Purpose: Download real fraud detection datasets for model training
- Required: `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables
- Datasets: 4+ integrated Kaggle datasets for training

### Third-Party Libraries

**Core ML Libraries**:
- scikit-learn 1.5.2+ (ML algorithms)
- XGBoost 2.1.2+ (gradient boosting)
- LightGBM 4.5.0+ (fast gradient boosting)
- NumPy 1.26.4+ (numerical computing)
- Pandas 2.2.3+ (data manipulation)

**Web Framework**:
- Flask 3.0.0 (REST API)
- Flask-CORS 4.0.0 (cross-origin requests)
- Flask-JWT-Extended 4.6.0 (authentication)

**Monitoring**:
- DataDog integration (optional, configured via environment)
- Custom metrics collection system

### Configuration Management

**Environment Variables** (see `.env.example`):
- `ENVIRONMENT`: development/staging/production
- `FLASK_DEBUG`: false (must be false in production)
- `VERIFY_SSL_CERTS`: true (SSL verification)
- `JWT_SECRET`: Secret key for JWT tokens
- `DB_HOST`, `DB_PORT`, `DB_NAME`: Database connection
- `REDIS_HOST`, `REDIS_PORT`: Redis connection
- `API_PORT`: Backend API port (default 8445)
- `FRONTEND_PORT`: Frontend port (default 5000)

**Security Notes**:
- Never commit secrets to version control
- Use secret management systems in production (AWS Secrets Manager, Azure Key Vault)
- Generate secure random keys (32+ characters)
- Rotate JWT secrets every 30 days

### Data Storage

**Model Persistence**:
- Location: `models/` directory
- Format: Joblib serialization
- Versioning: Filename includes version number
- Backup: Automatic backup before overwrite

**Configuration Storage**:
- Format: JSON files in `config/` directory
- Files: `configuration_rules.json`, `configuration_changes.json`
- Validation: Schema validation before load

**Logs**:
- Location: `logs/` directory
- Format: Structured JSON logging
- Rotation: Automatic log rotation by size/date