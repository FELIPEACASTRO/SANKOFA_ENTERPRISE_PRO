# Sankofa Enterprise Pro - Fraud Detection System v2.0

## Overview
Sankofa Enterprise Pro is a production-ready fraud detection system designed for banking environments. Its primary purpose is to process financial transactions with low latency, identify and prevent fraud using advanced machine learning models, and ensure compliance with relevant financial regulations (LGPD/BACEN/PCI DSS). The system features a comprehensive React-based dashboard for monitoring and management, aiming for high performance and reliability in a critical banking context.

## User Preferences
- Communication style: Simple, everyday language
- Avoid technical jargon when possible
- Documentation in Portuguese for main documents

## System Architecture

### Core Architecture Pattern
The system adheres to a Clean Architecture pattern, separating concerns into Domain, Application, Infrastructure, and Presentation layers.

### UI/UX Decisions
The frontend is built with React 18, utilizing Vite for fast development, `shadcn/ui` for UI components, and TailwindCSS for styling. This combination provides a modern, responsive, and accessible user interface across 16 fully operational pages, including a dashboard, transaction management, alert handling, and various configuration and monitoring screens.

### Technical Implementations
- **Backend**: Developed with Python 3.12+ using the Flask framework. It integrates machine learning libraries such as scikit-learn and XGBoost for fraud detection.
- **ML Models**: Employs a stacking ensemble model combining Random Forest, Gradient Boosting, and CatBoost for robust fraud detection. Transfer Learning techniques are used across four phases.
- **Data Storage**: PostgreSQL serves as the primary database, storing transactional data, audit logs, and configuration information.
- **Caching**: A `SimpleCache` implementation provides in-memory caching with a 30-second TTL (Time-To-Live) to significantly reduce latency for frequently accessed data, falling back to an in-memory solution if Redis is not configured.
- **Security**: Implements JWT Authentication, Role-Based Access Control (RBAC) with 5 defined roles, and compliance features like data masking, audit trails, and explainability to meet LGPD, BACEN, and PCI DSS standards.
- **API Endpoints**: The system exposes 27 fully functional API endpoints covering health checks, dashboard data, transaction processing, alerts, rules management, observability, configuration, and 6 new research ML endpoints.

### Feature Specifications
- **Fraud Detection**: Processes transactions with <50ms latency, utilizing ML models and hard rules for real-time fraud scoring.
- **Dashboard**: Provides KPIs, time-series data, and channel-specific insights for comprehensive system overview.
- **Transaction Management**: Allows for filtering, sorting, pagination, and actions on transactions (approve, reject, investigate).
- **Alerts & Rules**: Manages fraud alerts, hard business rules, VIP (whitelist) and Hot (blacklist) lists.
- **Observability**: Offers metrics, performance monitoring, and health checks for system and ML model status.
- **Calibration**: Enables adjustment of model thresholds and parameters.
- **Compliance**: Integrates features for LGPD (data privacy), BACEN (banking regulations), and PCI DSS (payment card security).

### Advanced Hard Rules Engine (NEW - December 2025)
The hard rules system has been upgraded to a full-featured rules engine:

**Multiple Conditions Support:**
- Support for AND/OR logic operators
- Up to 10+ conditions per rule
- Dynamic condition builder in frontend

**20 Available Fields in 7 Categories:**
- **Transaction (5)**: amount, channel, type, status, is_scheduled
- **Customer (4)**: cpf, location, is_first_transaction, account_age_days
- **Device (3)**: device_id, ip_address, is_new_device
- **Temporal (2)**: hour, day_of_week
- **Velocity (3)**: velocity_1h, velocity_24h, amount_24h
- **ML (2)**: risk_score, ml_confidence
- **PIX (1)**: pix_key_type

**16 Operators:**
- Comparison: ==, !=, >, <, >=, <=
- Text: contains, not_contains, starts_with, ends_with
- Lists: in, not_in
- Range: between
- Advanced: regex, is_null, is_not_null

**6 Action Types:**
- block, review, alert, approve, step_up, score_adjust

**4 Rule Types:**
- blocking, scoring, routing, alerting

**API Endpoints:**
- `GET /api/hard-rules/metadata` - Returns all fields, operators, actions, and rule types
- `POST /api/hard-rules` - Create rule with conditions_json and logic_operator
- `PUT /api/hard-rules/:id` - Update rule
- `DELETE /api/hard-rules/:id` - Delete rule

### Research-Based ML Modules (New)
Four new modules based on academic research have been implemented:

1. **Bahnsen Feature Engineering (v2.0.0)** - Based on Bahnsen et al. 2016:
   - Temporal aggregations (1h, 6h, 24h, 72h, 168h windows)
   - Von Mises periodic features (sin/cos for hour/day/month)
   - Behavioral deviation detection (Z-scores)
   - Velocity features and channel risk scoring
   - Generates 62+ features per transaction

2. **PIX Fraud Taxonomy (v1.0.0)** - Based on arXiv:2511.20902:
   - 10+ Brazilian PIX fraud types (Mão Fantasma, Clone WhatsApp, QR adulterado, etc.)
   - Remote access detection with high confidence
   - BACEN/LGPD compliance flags
   - Explainable recommendations for compliance

3. **NLP Social Engineering Detector (v1.0.0)** - Based on DIFrauD Dataset:
   - SMS phishing (smishing) detection
   - WhatsApp clone patterns
   - Bank impersonation detection
   - Urgency and emotional manipulation scoring
   - Batch analysis support

4. **Transfer Learning Pipeline (v1.0.0)**:
   - Support for Nigerian Financial (5M tx), PaySim (6.3M tx), Feedzai BAF (6M tx), IEEE-CIS datasets
   - Feature mapping and alignment
   - Model fine-tuning pipeline

### Research Module API Endpoints (NEW v2.0)
All 6 research endpoints are fully functional and tested:
- `GET /api/research/modules/status` - Status of all research modules (4 modules available)
- `POST /api/research/bahnsen/features` - Generate 62+ Bahnsen features for a transaction
- `POST /api/research/pix/analyze` - Analyze PIX transaction for fraud (10+ types)
- `POST /api/research/nlp/analyze` - Analyze text for social engineering (70%+ detection)
- `POST /api/research/nlp/batch` - Batch analyze multiple texts
- `GET /api/research/transfer/datasets` - List supported datasets (4 datasets, 17M+ transactions)

### Documentation (Updated December 2025)
Ultra-didactic ML documentation available at:
- `/docs/GUIA_COMPLETO_ML.md` - Complete ML guide explaining how models work together
  - Covers 7 integrated modules: Bahnsen, Random Forest, Gradient Boosting, Logistic Regression, CatBoost, GNN, PIX Taxonomy, NLP, Transfer Learning
  - Includes ensemble formula: P = 0.50×Base + 0.25×CatBoost + 0.25×GNN
  - ASCII diagrams and step-by-step calculation examples

### Mock Data Elimination (December 2025)
All endpoints now return real PostgreSQL data. Changes made:
- **Frontend Monitoring.jsx**: Fallback values changed from hardcoded (45, 60, 127) to 0/null
- **Frontend Metrics.jsx**: Mock data no catch block removido, agora retorna zeros + indicador de erro visual
- **Frontend Metrics.jsx**: Taxa de bloqueio hardcoded (78%) removida, agora usa `metrics.block_rate`
- **Backend /api/observability/ml**: Returns error with zeros instead of fake metrics on exception
- **Backend /api/observability/health**: Returns error status instead of fake "healthy"
- **Backend /api/alerts**: Uses postgres_store.get_alerts_list() instead of np.random
- **Backend /api/transactions**: Returns real transactions only, no mock generation

### Data Sources Analysis (December 2025)
Análise completa das 17 páginas (excluindo Manual/Documentação):
- **13 páginas com 100% PostgreSQL Real**: Dashboard, Transactions, Alerts, HardRules, VipList, HotList, Investigation, ManualReview, FeedbackAnalyst, Audit, Settings, Datasets, Reports
- **2 páginas mistas (Real + Fallback zeros)**: Monitoring, Metrics
- **1 página com dados sintéticos (intencional para ML)**: /api/model/train usa np.random para gerar dados de treinamento

## External Dependencies

- **PostgreSQL**: Used as the primary relational database for persistent data storage.
- **Redis**: Optional caching layer; if not configured, the system defaults to an in-memory cache.
- **Hugging Face**: Utilized for accessing pre-trained machine learning models.
- **Stanford SNAP Datasets**: Provides datasets used for machine learning model training and evaluation.