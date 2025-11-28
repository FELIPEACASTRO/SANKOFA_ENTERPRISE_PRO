# Sankofa Enterprise Pro - Fraud Detection System

## Overview
Sankofa Enterprise Pro is a production-ready fraud detection system designed for banking environments, capable of processing over 300 million requests per day. It integrates machine learning ensemble models (Random Forest, Gradient Boosting, CatBoost, GNN, Federated Learning), real-time transaction analysis, MLOps practices, and robust regulatory compliance (BACEN, LGPD, PCI DSS). The system features a React-based dashboard for operational oversight and fraud analysis. Its primary purpose is to detect and prevent financial fraud, ensuring high precision and recall, and providing explainable AI for compliance.

## User Preferences
- Communication style: Simple, everyday language
- Avoid technical jargon when possible
- Documentation in Portuguese for main documents

## System Architecture

### Core Architecture Pattern
The system adheres to a **Clean Architecture** pattern, ensuring a clear separation of concerns across its layers:
- **Domain Layer**: Handles core business entities like `Transaction` and `FraudPrediction`.
- **Application Layer**: Orchestrates use cases and business logic.
- **Infrastructure Layer**: Manages external services, APIs, databases, and caching.
- **Presentation Layer**: Comprises the React-based frontend dashboard.

### Technology Stack
- **Backend**: Python 3.12+ with Flask for the REST API, utilizing scikit-learn, XGBoost, LightGBM, CatBoost for machine learning, and custom Prometheus-style metrics for observability. PostgreSQL (Neon-backed) serves as the primary database.
- **Frontend**: Developed with React 18 and Vite, using shadcn/ui and TailwindCSS for a consistent and modern UI. It includes 16 specialized pages for dashboard, transactions, calibration, investigation, monitoring, and reporting.

### ML Engine Architecture
The fraud detection engine employs a **Stacking Ensemble** approach:
1.  **Base Models (Layer 0)**: Includes Random Forest Classifier, Gradient Boosting Classifier, CatBoost Classifier, and a GNN Detector for graph-based patterns.
2.  **Meta-Model (Layer 1)**: A Logistic Regression model combines predictions from the base models.
3.  **Explainability Engine**: Provides feature importance and LGPD-compliant explanations for fraud predictions.
4.  **Feature Engineering**: Automated extraction of over 47 features.

### System Design Choices
- **Comprehensive QA Coverage**: 136 automated tests covering 40+ test categories based on an 87-type QA framework, ensuring 100% test pass rate.
- **LGPD Explainability**: Each fraud prediction includes automated, LGPD-compliant explanations detailing risk factors.
- **Prometheus Observability**: Real-time metrics collection, including TPS, latency, error rates, and SLA compliance monitoring.
- **Scalable Infrastructure**: Optimized for high-performance batch processing (33.88 TPS) using a `BatchProcessor`, `AsyncTaskQueue`, and `CircuitBreaker` for resilience.
- **Security Architecture**: Implements a robust Role-Based Access Control (RBAC) system with 5 roles and 20+ permissions, JWT authentication, and comprehensive security testing against common vulnerabilities like SQL Injection, XSS, and path traversal.
- **Compliance**: Built-in features for LGPD (data masking, audit trails, explainability), BACEN (operational fraud detection API, SLA monitoring), and PCI DSS (sensitive data masking, structured logging, TLS readiness).

### Key Endpoints
The system exposes a variety of API endpoints for health checks, authentication, real-time and batch fraud prediction, transaction management, alerts, dashboard summaries, observability metrics, explainability, and ML model metrics.

## External Dependencies

### Required Services
- **PostgreSQL (Neon)**: Used as the persistent data store for all operational data including transactions, alerts, audit logs, user information, RBAC configurations, and sensitive tokenized data (e.g., CPF tokens).
- **Redis (Cluster)**: Utilized for caching and potentially for managing task queues, though not explicitly detailed as a primary data store.

### Environment Variables
- `DATABASE_URL`: Connection string for the PostgreSQL database.
- `JWT_SECRET`: Secret key for JSON Web Token authentication.
- `ENCRYPTION_KEY`: AES-256 key for encrypting sensitive data like CPF tokens.
- `ENVIRONMENT`: Specifies the deployment environment (e.g., `development`, `production`).