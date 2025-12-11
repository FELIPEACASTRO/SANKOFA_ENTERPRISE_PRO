# REFACTORING GUIDE - Blueprint Migration

## Overview

This guide documents the refactoring of `production_api.py` (4000+ lines) into modular blueprints.

## Current Status

### Completed
- [OK] DSR blueprint created (`api/routes/dsr.py`)
- [OK] App factory created (`api/app_factory.py`)
- [OK] Validation wrapper created (`api/validation_wrapper.py`)
- [OK] Security middleware integrated

### TODO
- [ ] Migrate auth endpoints to `api/routes/auth.py`
- [ ] Migrate fraud endpoints to `api/routes/fraud.py`
- [ ] Migrate admin endpoints to `api/routes/admin.py`
- [ ] Migrate observability endpoints to `api/routes/observability.py`

## Blueprint Structure

```
backend/api/routes/
├── __init__.py
├── auth.py          # Authentication endpoints (/api/auth/*)
├── fraud.py         # Fraud detection endpoints (/api/predict, /api/batch)
├── admin.py         # Admin/management endpoints (/api/rules/*, /api/vip/*)
├── dsr.py           # LGPD DSR endpoints (/api/dsr/*) [COMPLETE]
└── observability.py # Health checks, metrics (/health, /metrics)
```

## Migration Steps

### 1. Identify Endpoints by Domain

Group endpoints in `production_api.py` by functional area:

**Auth Endpoints:**
- POST /api/auth/login
- POST /api/auth/logout
- POST /api/auth/refresh
- GET /api/auth/validate

**Fraud Endpoints:**
- POST /api/predict
- POST /api/batch/predict
- POST /api/fraud/explain

**Admin Endpoints:**
- GET /api/rules
- POST /api/rules
- PUT /api/rules/<id>
- DELETE /api/rules/<id>
- GET /api/vip
- POST /api/vip
- GET /api/hot
- POST /api/hot

**Observability Endpoints:**
- GET /health
- GET /metrics
- GET /api/status

### 2. Extract to Blueprint

For each endpoint group:

1. Create blueprint file (e.g., `api/routes/fraud.py`)
2. Import blueprint template:

```python
from flask import Blueprint, request, jsonify, g
from api.schemas import TransactionRequest
from api.validation_wrapper import validate_request
from api.middleware.auth import require_permission
from utils.log_sanitizer import sanitize_log_data
import logging

fraud_bp = Blueprint('fraud', __name__)
logger = logging.getLogger(__name__)

@fraud_bp.route('/predict', methods=['POST'])
@require_permission('fraud:predict')
@validate_request(TransactionRequest)
def predict_fraud():
    """Predict fraud in single transaction"""
    try:
        validated = request.validated_data
        # Implementation here
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
```

3. Register blueprint in `app_factory.py`:

```python
from api.routes.fraud import fraud_bp
app.register_blueprint(fraud_bp, url_prefix='/api')
```

### 3. Test Migration

After extracting each blueprint:

```bash
# Test specific blueprint
pytest tests/integration/test_fraud_endpoints.py -v

# Test all endpoints
pytest tests/integration/ -v

# Check for broken imports
python -m api.app_factory
```

### 4. Remove from production_api.py

Once blueprint is tested:
- Delete endpoint code from `production_api.py`
- Add comment: `# Migrated to api/routes/fraud.py`
- Verify no duplicate route errors

## DSR Blueprint Example (COMPLETE)

The DSR blueprint is complete and can serve as a template:

**File:** `backend/api/routes/dsr.py`

```python
from flask import Blueprint, request, jsonify
from api.schemas import DSRAccessRequest, DSRDeletionRequest
from compliance.dsr_service import dsr_service
from pydantic import ValidationError

dsr_bp = Blueprint('dsr', __name__)

@dsr_bp.route('/access', methods=['POST'])
async def dsr_access():
    """LGPD Art. 18, I - Access to personal data"""
    try:
        validated = DSRAccessRequest(**request.json)
        request_id = f"DSR-ACCESS-{int(time.time())}"
        report = await dsr_service.access_request(validated.cpf, request_id)
        return jsonify({'success': True, 'data': report})
    except ValidationError as e:
        return jsonify({'success': False, 'error': 'Validation failed', 'details': e.errors()}), 400
```

**Registered in:** `backend/api/app_factory.py`

```python
from api.routes.dsr import dsr_bp
app.register_blueprint(dsr_bp, url_prefix='/api/dsr')
```

## Testing Blueprint Migration

```bash
# Test DSR endpoints
curl -X POST http://localhost:5000/api/dsr/access \
  -H "Content-Type: application/json" \
  -d '{"cpf": "12345678909", "request_reason": "I want to access my data", "requester_email": "user@example.com"}'

# Expected: 200 OK with data report
```

## Validation Wrapper Usage

All POST/PUT/PATCH endpoints should use the validation wrapper:

```python
from api.validation_wrapper import validate_request
from api.schemas import TransactionRequest

@app.route('/api/predict', methods=['POST'])
@validate_request(TransactionRequest)
def predict():
    # request.validated_data contains validated Pydantic object
    validated = request.validated_data
    result = fraud_engine.predict(validated.dict())
    return jsonify(result)
```

This automatically:
- Validates input against Pydantic schema
- Returns 400 error on validation failure
- Sanitizes error messages in logs
- Stores validated data in `request.validated_data`

## Next Steps

1. **Migrate Auth Endpoints** (Estimated: 2 hours)
   - Extract login, logout, refresh, validate
   - Update tests in `tests/integration/test_auth_endpoints.py`

2. **Migrate Fraud Endpoints** (Estimated: 3 hours)
   - Extract predict, batch predict, explain
   - Update tests in `tests/integration/test_fraud_endpoints.py`

3. **Migrate Admin Endpoints** (Estimated: 4 hours)
   - Extract rules, vip, hot endpoints
   - Update tests in `tests/integration/test_admin_endpoints.py`

4. **Migrate Observability Endpoints** (Estimated: 1 hour)
   - Extract health, metrics, status
   - Update tests in `tests/integration/test_observability.py`

5. **Remove production_api.py** (Final step)
   - Verify all endpoints migrated
   - Update import statements across codebase
   - Update documentation

## Benefits

- **Modularity:** Each blueprint is self-contained
- **Testability:** Test blueprints independently
- **Maintainability:** Easier to navigate 200-line files vs 4000-line file
- **Collaboration:** Multiple devs can work on different blueprints
- **Reusability:** Blueprints can be reused across projects

## Common Issues

### Issue: Circular imports
**Solution:** Use `from api.routes import blueprint` at registration time, not globally

### Issue: Duplicate route errors
**Solution:** Ensure endpoint is removed from `production_api.py` after migration

### Issue: Missing dependencies
**Solution:** Import all required services, schemas, middleware in blueprint file

### Issue: Tests fail after migration
**Solution:** Update test imports to point to new blueprint location

## Resources

- Flask Blueprints Documentation: https://flask.palletsprojects.com/en/2.3.x/blueprints/
- Pydantic Validation: https://docs.pydantic.dev/latest/
- LGPD Art. 18: https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm

## Completion Criteria

Blueprint migration is complete when:
- [ ] All endpoints extracted to blueprints
- [ ] `production_api.py` < 500 lines (only initialization)
- [ ] All integration tests pass
- [ ] Documentation updated
- [ ] Code review approved
