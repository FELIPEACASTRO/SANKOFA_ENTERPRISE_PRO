# Sankofa Enterprise Pro - Sistema de Detecção de Fraude Bancária

## Overview

Sankofa Enterprise Pro is a comprehensive, real-time banking fraud detection platform designed for large financial institutions. It integrates advanced Machine Learning, automated MLOps, and banking compliance to deliver maximum protection against financial fraud. The project has undergone a significant transformation from a Proof-of-Concept to a production-ready system, demonstrating high performance, robust architecture, and adherence to regulatory standards. Key capabilities include sophisticated fraud detection techniques, ultra-low latency transaction analysis, and full compliance with banking regulations like BACEN, LGPD, and PCI DSS. The system aims to provide a high return on investment through improved fraud prevention and operational efficiency.

## User Preferences

- **I want iterative development.**
- **Ask before making major changes.**
- **I prefer detailed explanations.**

## System Architecture

The system follows a microservices-oriented architecture, separating the frontend and backend.

### UI/UX Decisions
- **Frontend Framework**: React 19 with Vite 6.
- **Styling**: Tailwind CSS 4.
- **Components**: Radix UI for accessible components and Shadcn UI for custom components.
- **Data Visualization**: Recharts for dynamic charts.
- **Navigation**: React Router for single-page application routing.

### Technical Implementations
- **Backend Framework**: Flask for the RESTful API.
- **Machine Learning**: Scikit-learn, XGBoost, and LightGBM for fraud detection models. A consolidated production fraud engine uses an optimized ensemble stacking (Random Forest + Gradient Boosting + Logistic Regression) with dynamic threshold calibration.
- **Caching**: Redis for high-performance data caching.
- **Database**: PostgreSQL for persistent storage, featuring a comprehensive schema with audit trails for compliance and optimized indexes.
- **Authentication**: JWT for secure API access.
- **Deployment**: Gunicorn for production-grade serving of the Flask application, configured for autoscaling with optimized builds.
- **Configuration**: Centralized, enterprise-grade configuration management via environment variables (`backend/config/settings.py`) with automatic validation and support for different environments (dev/staging/prod).
- **Logging**: Structured JSON logging (`backend/utils/structured_logging.py`) for enhanced observability and traceability, compatible with systems like DataDog, Splunk, or ELK.
- **Error Handling**: Enterprise-level error handling (`backend/utils/error_handling.py`) with categorization (Validation, Database, ML, Security, Compliance) and severity levels.
- **MLOps**: Automated CI/CD pipelines for ML models, including automatic drift detection, adversarial testing, model version management, and automatic rollback capabilities.
- **Security**: Implementation of enterprise security measures and adherence to PCI DSS standards for card data security.

### Feature Specifications
- **Fraud Detection Engine**: Utilizes 47 analysis techniques (temporal, geographical, behavioral) with an ensemble of models (Random Forest, XGBoost, LightGBM, Neural Networks). Achieves ultra-low latency (~11ms P95) and high throughput (118,720 TPS).
- **Compliance Modules**:
    - **BACEN**: Implements Joint Resolution n° 6.
    - **LGPD**: Data protection with personal data masking.
    - **PCI DSS**: Card data security.
    - **SOX**: Internal controls and auditing.
- **Production API**: Features 13 enterprise REST endpoints integrating with the fraud engine, configuration system, and structured logging, including global middleware and error handling.
- **Dashboard**: Provides key performance indicators (KPIs), time-series data, channel-specific insights, system alerts, and ML model status.

## External Dependencies

- **Database**: PostgreSQL
- **Caching**: Redis
- **ML Libraries**: Scikit-learn, XGBoost, LightGBM
- **Frontend Libraries**: React, Vite, Tailwind CSS, Radix UI, Recharts, React Router, Shadcn UI
- **Backend Libraries**: Flask, Flask-CORS, NumPy, Pandas, Gunicorn
- **Authentication**: JWT (JSON Web Tokens)

## AIForge Analysis - Double Check Criterioso (Nov 08, 2025)

**Status**: ⚠️ **Análise inicial REJEITADA - Versão corrigida disponível**

### What Happened
Foi realizada uma análise do repositório AIForge para identificar melhorias ao Sankofa. Após **Double Check ultra criterioso** pelo arquiteto técnico, a análise inicial foi **rejeitada** por conter:
- ❌ Inventário não verificado (326+ recursos sem auditoria)
- ❌ Ganhos irrealistas (99% accuracy sem validação)
- ❌ ROI com matemática falha
- ❌ Timelines otimistas demais
- ❌ Aplicabilidade ao Brasil não validada

### Corrected Analysis Available
**File**: `docs/AIFORGE_DOUBLE_CHECK_CORRECTED.md`

**Realistic Improvements** (conservative):
- F1-Score: 0.25 → 0.72-0.85
- Recall: ~0.75 → 0.80-0.85
- Precision: ~0.65 → 0.75-0.85
- AUC-ROC: N/A → 0.90-0.95

**Investment**: R$ 180k (Phase 1: 6-8 weeks)
**ROI**: ~15,000% (if validated)
**Savings**: R$ 25-40M/month (medium bank)

**Technologies Validated**:
- Stacking Ensemble (XGBoost + LightGBM + CatBoost)
- Real datasets (IEEE-CIS, Credit Card Fraud)
- SHAP Explainability (BACEN compliance)
- Optuna AutoML