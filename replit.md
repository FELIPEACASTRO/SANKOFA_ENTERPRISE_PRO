# Sankofa Enterprise Pro - Fraud Detection System

## Overview
Sankofa Enterprise Pro is a production-ready fraud detection system for banking environments, designed to detect and prevent financial fraud. It processes over 300 million requests daily, utilizing machine learning ensemble models (Random Forest, Gradient Boosting, CatBoost, GNN, Federated Learning), real-time transaction analysis, and MLOps practices. The system provides explainable AI for regulatory compliance (BACEN, LGPD, PCI DSS) and features a React-based dashboard for operational oversight and fraud analysis. The primary goal is to achieve high precision and recall in fraud detection.

## User Preferences
- Communication style: Simple, everyday language
- Avoid technical jargon when possible
- Documentation in Portuguese for main documents

## System Architecture

### Core Architecture Pattern
The system employs a **Clean Architecture** pattern, separating concerns into Domain, Application, Infrastructure, and Presentation layers.

### Technology Stack
- **Backend**: Python 3.12+ with Flask, utilizing scikit-learn, XGBoost, LightGBM, and CatBoost for ML. PostgreSQL (Neon-backed) serves as the primary database, and custom Prometheus-style metrics are used for observability.
- **Frontend**: React 18 with Vite, using shadcn/ui and TailwindCSS for a modern UI. It includes 16 specialized pages for various functionalities.

### ML Engine Architecture
The fraud detection engine uses a **Stacking Ensemble** approach:
1.  **Base Models**: Random Forest, Gradient Boosting, CatBoost, and a GNN Detector.
2.  **Meta-Model**: A Logistic Regression model combines base model predictions.
3.  **Explainability Engine**: Provides LGPD-compliant explanations and feature importance.
4.  **Feature Engineering**: Automated extraction of over 47 features, including PIX-specific, velocity, and behavioral features.

### System Design Choices
- **Comprehensive QA Coverage**: 136 automated tests across 40+ categories, ensuring 100% test pass rate.
- **LGPD Explainability**: Automated, compliant explanations for each fraud prediction.
- **Prometheus Observability**: Real-time metrics for TPS, latency, error rates, and SLA compliance.
- **Scalable Infrastructure**: Optimized for high-performance batch processing (33.88 TPS) using `BatchProcessor`, `AsyncTaskQueue`, and `CircuitBreaker`.
- **Security Architecture**: Role-Based Access Control (RBAC) with 5 roles and 20+ permissions, JWT authentication, and security testing against common vulnerabilities.
- **Compliance**: Built-in features for LGPD (data masking, audit trails, explainability), BACEN (operational fraud detection API, SLA monitoring), and PCI DSS (sensitive data masking, structured logging, TLS readiness).
- **UI/UX Decisions**: Consistent and modern UI using shadcn/ui and TailwindCSS.

### Key Endpoints
The system provides API endpoints for health checks, authentication, real-time and batch fraud prediction, transaction management, alerts, dashboard summaries, observability metrics, explainability, and ML model metrics.

## External Dependencies

### Required Services
-   **PostgreSQL (Neon)**: Persistent data store for operational data, including transactions, alerts, audit logs, user information, RBAC, and tokenized sensitive data.
-   **Redis (Cluster)**: Used for caching.
-   **Hugging Face**: Integration of 4 pre-trained models and 4 datasets for enhanced fraud detection capabilities.
-   **Stanford SNAP Datasets**: Elliptic and Elliptic++ datasets for graph-based fraud analysis.

### Environment Variables
-   `DATABASE_URL`: PostgreSQL connection string.
-   `JWT_SECRET`: JSON Web Token authentication key.
-   `ENCRYPTION_KEY`: AES-256 key for sensitive data encryption.
-   `ENVIRONMENT`: Specifies the deployment environment.

## Recent Changes (November 30, 2025)

### Database Integration Fixes
The following improvements were made to ensure full PostgreSQL integration:

1. **Transaction IDs**: `/api/transactions` endpoint now returns real PostgreSQL transaction IDs (format: `TXN{timestamp}{sequence}`) instead of synthetic frontend-generated IDs.

2. **PostgresStore Enhancements**:
   - Added `get_recent_transactions(limit)` method for fetching transactions from PostgreSQL
   - Added `get_transaction_by_id(transaction_id)` method for single transaction lookup
   - Fixed `get_feedback_list()` to use correct `risk_score` column instead of `fraud_probability`

3. **Manual Review Pipeline**: Complete workflow now functional:
   - `add_to_manual_review()` - Updates transaction status to `pending_review`
   - `complete_review()` - Updates status and creates feedback record
   - Fixed SQL queries to match actual database schema (removed `updated_at` column references)

4. **Status Mapping**: Proper conversion between database statuses (APPROVED, FRAUD, pending_review) and frontend display statuses (APROVADA, REJEITADA, EM_REVISAO).

### Current Integration Status
All 16 pages now have working backend integration:
- Transactions, Dashboard, Alerts, Investigations - PostgreSQL-backed
- Hard Rules, VIP List, Hot List, Settings - CRUD operations with PostgreSQL
- Manual Review, Feedback - Complete workflow with audit logging
- Audit Logs - All operations logged to PostgreSQL