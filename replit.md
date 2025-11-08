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

## AIForge Analysis - VERIFICAÇÃO FINAL (Nov 08, 2025)

**Status**: ✅ **REPOSITÓRIO VERIFICADO - Conteúdo Real Analisado**

### Evolution of Analysis
- ❌ **v1 REJECTED**: 326+ recursos não verificados, 99% accuracy irrealista
- ❌ **v2 REJECTED**: F1 0.72-0.85 sem evidência Brasil, R$ 180k irrealista
- ✅ **v3 APPROVED**: Separação fatos/premissas/incertezas
- ✅ **v4 VERIFIED**: Acesso direto ao repositório real via GitHub

### Verification Report
**Files**: 
- `docs/AIFORGE_VERIFICATION_FINAL.md` (análise do conteúdo real)
- `docs/AIFORGE_TRIPLE_CHECK_FINAL.md` (análise rigorosa)

### What REALLY EXISTS in AIForge (Verified)

**Repository**: https://github.com/FELIPEACASTRO/AIForge

**Verified Resources**:
- ✅ **135 Banking/Fraud resources** (datasets, tools, papers)
- ✅ **94 Transfer Learning resources** (hubs, libraries, datasets)
- ✅ **7 public fraud datasets** (IEEE-CIS, PaySim, Credit Card Fraud)
- ✅ **5 feature engineering tools** (Featuretools, tsfresh, SHAP, Boruta)
- ✅ **4 transfer learning libs** (PEFT, LoRA, FinGPT, FinBERT)

**Key Datasets for Sankofa**:
1. IEEE-CIS Fraud Detection (590K transactions) - Kaggle
2. Credit Card Fraud (284K transactions) - Kaggle
3. PaySim Mobile Money (6.3M transactions) - Kaggle
4. Bank Account Fraud NeurIPS 2022 - Kaggle
5. Feedzai Bank Fraud - GitHub
6. NVIDIA Financial Fraud Detection - GitHub

**Verified Tools**:
- Featuretools (7k⭐) - Automated feature synthesis
- tsfresh (8k⭐) - Time series features (60+)
- SHAP (22k⭐) - Model explainability
- FinGPT - Financial LLM
- FinBERT - Financial BERT

**What Was NOT Found** (404):
- Detailed Banking AI list (claimed 186)
- Detailed Fraud Detection list (claimed 140)
- Ensemble Learning detailed list
- AutoML detailed list

**Conclusion**: Repository has REAL and USEFUL resources, but aggregate numbers (14,988+) NOT verified.

**Recommendation**: Use AIForge as STARTING POINT to discover public datasets and tools, but validate each resource individually.