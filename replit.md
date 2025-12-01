# Sankofa Enterprise Pro - Fraud Detection System

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
- **API Endpoints**: The system exposes 21 fully functional API endpoints covering health checks, dashboard data, transaction processing, alerts, rules management, observability, and configuration.

### Feature Specifications
- **Fraud Detection**: Processes transactions with <50ms latency, utilizing ML models and hard rules for real-time fraud scoring.
- **Dashboard**: Provides KPIs, time-series data, and channel-specific insights for comprehensive system overview.
- **Transaction Management**: Allows for filtering, sorting, pagination, and actions on transactions (approve, reject, investigate).
- **Alerts & Rules**: Manages fraud alerts, hard business rules, VIP (whitelist) and Hot (blacklist) lists.
- **Observability**: Offers metrics, performance monitoring, and health checks for system and ML model status.
- **Calibration**: Enables adjustment of model thresholds and parameters.
- **Compliance**: Integrates features for LGPD (data privacy), BACEN (banking regulations), and PCI DSS (payment card security).

## External Dependencies

- **PostgreSQL**: Used as the primary relational database for persistent data storage.
- **Redis**: Optional caching layer; if not configured, the system defaults to an in-memory cache.
- **Hugging Face**: Utilized for accessing pre-trained machine learning models.
- **Stanford SNAP Datasets**: Provides datasets used for machine learning model training and evaluation.