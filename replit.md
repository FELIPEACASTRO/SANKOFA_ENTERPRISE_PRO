# Sankofa Enterprise Pro - Fraud Detection System v2.0

## Overview
Sankofa Enterprise Pro is a production-ready fraud detection system for banking environments. Its core purpose is to process financial transactions with low latency, identify and prevent fraud using advanced machine learning models, and ensure compliance with financial regulations (LGPD/BACEN/PCI DSS). The system includes a comprehensive React-based dashboard for monitoring and management, aiming for high performance and reliability in a critical banking context. The project aims to provide a robust solution for fraud prevention, supporting the banking sector with cutting-edge technology and regulatory adherence.

## User Preferences
- Communication style: Simple, everyday language
- Avoid technical jargon when possible
- Documentation in Portuguese for main documents

## System Architecture

### Core Architecture Pattern
The system follows a Clean Architecture pattern, segmenting concerns into Domain, Application, Infrastructure, and Presentation layers.

### UI/UX Decisions
The frontend is built with React 18, Vite, `shadcn/ui` components, and TailwindCSS, providing a modern, responsive user interface across 16 pages for dashboard, transaction management, alerts, and configuration.

### Technical Implementations
- **Backend**: Developed with Python 3.12+ and Flask, integrating scikit-learn and XGBoost for ML.
- **ML Models**: Employs a stacking ensemble model (Random Forest, Gradient Boosting, CatBoost) with Transfer Learning across four phases for robust fraud detection.
- **Data Storage**: PostgreSQL is the primary database for transactional data, audit logs, and configuration.
- **Prediction Cache**: LRU-based cache with TTL (300s default, 60s high-risk, 600s low-risk) integrated into ProductionFraudEngine. Achieves 0.6ms cache hits (99.9% improvement vs cold calls).
- **Caching**: A `SimpleCache` provides in-memory caching with a 30-second TTL, with Redis as an optional external caching layer.
- **Security**: Features JWT Authentication, Role-Based Access Control (RBAC) with 5 roles, data masking, audit trails, and explainability for LGPD, BACEN, and PCI DSS compliance.
- **MLOps Components**:
    1. **Experiment Tracker (v1.0.0)**: MLflow-like experiment tracking with run lifecycle, metrics/artifacts logging, and comparison.
    2. **Shadow Mode (v1.0.0)**: Gradual deployment with model comparison, traffic splitting, and divergence detection.
    3. **Fairness Analyzer (v1.0.0)**: Demographic parity, equalized odds, and disparate impact analysis for bias detection.
- **API Endpoints**: 27 functional API endpoints for health checks, dashboard, transaction processing, alerts, rules, observability, configuration, and 6 research ML endpoints.
- **Hard Rules Engine**: An advanced rules engine supports multiple conditions (AND/OR logic, up to 10+ conditions), 20 available fields across 7 categories, 16 operators, 6 action types (block, review, alert, approve, step_up, score_adjust), and 4 rule types. It provides a unified response format identical to ML model output, with 216 active rules derived from real-world fraud scenarios and academic research.
- **Research-Based ML Modules**: Four new modules based on academic research are integrated:
    1.  **Bahnsen Feature Engineering (v2.0.0)**: Generates 62+ features per transaction (temporal aggregations, Von Mises features, behavioral deviation, velocity, channel risk).
    2.  **PIX Fraud Taxonomy (v1.0.0)**: Detects 10+ Brazilian PIX fraud types, including remote access, with compliance flags.
    3.  **NLP Social Engineering Detector (v1.0.0)**: Detects SMS phishing, WhatsApp cloning, and bank impersonation patterns.
    4.  **Transfer Learning Pipeline (v1.0.0)**: Supports fine-tuning models using various financial datasets (e.g., Nigerian Financial, PaySim, Feedzai BAF, IEEE-CIS).

### Feature Specifications
- **Fraud Detection**: Processes transactions with <50ms latency using ML models and hard rules.
- **Dashboard**: Displays KPIs, time-series data, and channel-specific insights.
- **Transaction Management**: Allows filtering, sorting, pagination, and actions (approve, reject, investigate).
- **Alerts & Rules**: Manages fraud alerts, business rules, VIP (whitelist), and Hot (blacklist) lists.
- **Observability**: Provides metrics, performance monitoring, and health checks.
- **Calibration**: Adjusts model thresholds and parameters.
- **Compliance**: Integrates features for LGPD, BACEN, and PCI DSS.
- **Documentation**: Comprehensive internal documentation including an ML guide, database setup, and frontend interactive manuals.

## External Dependencies

- **PostgreSQL**: Primary relational database for persistent data storage.
- **Redis**: Optional caching layer; if not configured, an in-memory cache is used.
- **Hugging Face**: Used for accessing pre-trained machine learning models.
- **Stanford SNAP Datasets**: Provides datasets for ML model training and evaluation.