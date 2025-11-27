# Sankofa Enterprise Pro - Fraud Detection System

## Overview
Sistema de detecção de fraude em tempo real com capacidade de processar 300M+ requests/dia, incluindo modelos de ML, infraestrutura MLOps, conformidade (LGPD/Bacen) e dashboard React.

## Recent Changes (2025-11-27)

### Bug Fixes
- Fixed 44 LSP errors across 4 files (drift_detector.py, redis_cache_system.py, ab_testing_manager.py, canary_deployment_manager.py)
- Fixed deadlock in MetricsCollector by changing threading.Lock() to threading.RLock()
- Removed fabricated/mock data from KPIs - now uses real historical data from persistence

### New Features
- Added 15+ missing API endpoints for frontend integration
- Implemented real-time metrics collection with historical persistence
- Added train() method alias in ProductionFraudEngine
- Created ConfigStore and TransactionStore classes for data persistence

### API Endpoints Added
- `/api/manual-review` (GET, POST, PUT, DELETE)
- `/api/hard-rules` (GET, POST, PUT, DELETE)
- `/api/vip-list` (GET, POST, DELETE)
- `/api/hot-list` (GET, POST, DELETE)
- `/api/settings` (GET, PUT)
- `/api/alerts` (GET, POST)
- `/api/audit` (GET, POST)
- `/api/calibration` (GET, PUT)
- `/api/datasets` (GET)
- `/api/reports` (GET, POST)
- `/api/investigation/<id>` (GET)
- `/api/feedback` (POST)
- `/api/dashboard/recent-alerts` (GET)
- `/api/dashboard/model-status` (GET)
- `/api/model/train` (POST)

## Project Architecture

### Backend (Python/Flask)
```
sankofa-enterprise-real/backend/
├── api/
│   └── production_api.py          # Main API with 30+ endpoints
├── ml_engine/
│   └── production_fraud_engine.py # ML model with ensemble (RF+GB+LR)
├── mlops/
│   ├── drift_detector.py          # Data/concept drift detection
│   ├── ab_testing_manager.py      # A/B testing for models
│   └── canary_deployment_manager.py # Canary deployments
├── cache/
│   └── redis_cache_system.py      # Redis cache with in-memory fallback
├── config/
│   └── settings.py                # Configuration management
└── utils/
    ├── structured_logging.py      # Structured logging
    └── error_handling.py          # Error handling utilities
```

### Frontend (React/Vite)
```
sankofa-enterprise-real/frontend/
├── src/
│   ├── pages/
│   │   ├── Dashboard.jsx
│   │   ├── Transactions.jsx
│   │   ├── ManualReview.jsx
│   │   ├── Calibration.jsx
│   │   ├── Monitoring.jsx
│   │   └── ...
│   └── components/
└── vite.config.js
```

## Key Features

### ML Model
- Ensemble stacking (Random Forest + Gradient Boosting + Logistic Regression)
- Probability calibration with isotonic regression
- Dynamic threshold optimization
- Precision-boosting rules

### MLOps
- Data/concept drift detection (Jensen-Shannon divergence)
- A/B testing framework for model comparison
- Canary deployment with automatic rollback
- Model versioning and metrics tracking

### Compliance
- LGPD-compliant data handling
- Bacen regulatory compliance
- Audit logging
- Data retention policies

## How to Use

### Train the Model
```bash
curl -X POST http://localhost:8000/api/model/train \
  -H "Content-Type: application/json" \
  -d '{"n_samples": 10000}'
```

### Make Predictions
```bash
curl -X POST http://localhost:8000/api/fraud/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [{
      "amount": 1000,
      "hour": 14,
      "location_risk_score": 0.3,
      "device_risk_score": 0.2,
      "velocity_score": 0.1,
      "is_new_device": 0
    }]
  }'
```

## Configuration

### Environment Variables
- `ENVIRONMENT`: development/production
- `DEBUG`: true/false
- `ML_CONFIDENCE_THRESHOLD`: 0.5 (default)

### Settings (via API)
- `fraud_threshold`: Threshold for fraud classification
- `step_up_threshold`: Threshold for step-up authentication
- `review_threshold`: Threshold for manual review
- `max_transaction_value`: Maximum allowed transaction value

## Monitoring

### Health Check
```bash
curl http://localhost:8000/api/health
```

### Metrics
```bash
curl http://localhost:8000/api/metrics/dashboard
```

## Notes
- Redis cache is optional (system uses in-memory fallback)
- Model must be trained before making predictions
- Historical KPIs require at least one day of data collection
