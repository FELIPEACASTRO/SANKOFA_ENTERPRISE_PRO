"""
Sankofa Enterprise Pro - Production API
API production-grade integrando TODOS os novos componentes enterprise
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from typing import Dict, Any, List

# Novos componentes enterprise
from config.settings import get_config
from utils.structured_logging import get_structured_logger
from utils.error_handling import (
    ValidationError, DatabaseError, MLModelError,
    handle_error, with_error_handling, ErrorCategory, ErrorSeverity
)
from ml_engine.production_fraud_engine import get_fraud_engine, FraudPrediction

# Configuração
config = get_config()
logger = get_structured_logger('production_api', config.monitoring.log_level)

# Flask app
app = Flask(__name__)
CORS(app)

# Fraud engine (singleton)
fraud_engine = get_fraud_engine()

logger.info(
    "Production API initialized",
    environment=config.environment,
    debug=config.debug
)

# ==========================================
# MIDDLEWARE
# ==========================================

@app.before_request
def before_request():
    """Middleware executado antes de cada request"""
    g.start_time = time.time()
    g.request_id = f"REQ_{int(time.time()*1000)}"
    
    logger.debug(
        "Request started",
        request_id=g.request_id,
        method=request.method,
        path=request.path,
        ip=request.remote_addr
    )

@app.after_request
def after_request(response):
    """Middleware executado após cada request"""
    duration_ms = (time.time() - g.start_time) * 1000
    
    response.headers['X-Request-ID'] = g.request_id
    response.headers['X-Response-Time-Ms'] = f"{duration_ms:.2f}"
    response.headers['X-API-Version'] = fraud_engine.VERSION
    
    logger.info(
        "Request completed",
        request_id=g.request_id,
        method=request.method,
        path=request.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2)
    )
    
    return response

@app.errorhandler(Exception)
def handle_exception(error):
    """Handler global de exceções"""
    error_context = handle_error(error, raise_exception=False)
    
    return jsonify({
        'success': False,
        'error': {
            'id': error_context.error_id,
            'category': error_context.category.value,
            'severity': error_context.severity.value,
            'message': error_context.message,
            'recovery_action': error_context.recovery_action
        }
    }), 500

# ==========================================
# HEALTH & STATUS
# ==========================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'version': fraud_engine.VERSION,
        'environment': config.environment
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    """Status detalhado do sistema"""
    metrics = fraud_engine.get_performance_metrics()
    
    return jsonify({
        'success': True,
        'data': {
            'fraud_engine': metrics,
            'environment': config.environment,
            'debug_mode': config.debug,
            'api_version': fraud_engine.VERSION,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    })

# ==========================================
# FRAUD DETECTION
# ==========================================

@app.route('/api/fraud/predict', methods=['POST'])
def predict_fraud():
    """
    Prediz fraude para uma ou mais transações
    
    Body:
    {
        "transactions": [
            {
                "amount": 1000.00,
                "hour": 14,
                "location_risk_score": 0.3,
                "device_risk_score": 0.2,
                ...
            }
        ]
    }
    """
    # Validação
    if not request.json:
        raise ValidationError(
            "Request body is required",
            context={'endpoint': '/api/fraud/predict'}
        )
    
    transactions_data = request.json.get('transactions')
    if not transactions_data:
        raise ValidationError(
            "transactions field is required",
            context={'body': request.json}
        )
    
    if not isinstance(transactions_data, list):
        raise ValidationError(
            "transactions must be a list",
            context={'type': type(transactions_data).__name__}
        )
    
    # Converter para DataFrame
    try:
        df = pd.DataFrame(transactions_data)
    except Exception as e:
        raise ValidationError(
            f"Invalid transaction data: {str(e)}",
            context={'error': str(e)}
        )
    
    logger.info(
        "Starting fraud predictions",
        request_id=g.request_id,
        num_transactions=len(df)
    )
    
    # Predição
    if not fraud_engine.is_trained:
        logger.warning("Fraud engine not trained, using demo mode")
        raise MLModelError(
            "Fraud detection model is not trained. Please train the model first.",
            context={'endpoint': '/api/fraud/predict'}
        )
    
    predictions = fraud_engine.predict_detailed(df)
    
    # Converter para JSON
    results = [pred.to_dict() for pred in predictions]
    
    logger.info(
        "Fraud predictions completed",
        request_id=g.request_id,
        num_predictions=len(results),
        num_frauds=sum(1 for p in predictions if p.is_fraud)
    )
    
    return jsonify({
        'success': True,
        'data': {
            'predictions': results,
            'summary': {
                'total': len(results),
                'frauds_detected': sum(1 for p in predictions if p.is_fraud),
                'avg_risk_score': sum(p.risk_score for p in predictions) / len(predictions),
                'model_version': fraud_engine.VERSION
            }
        }
    })

@app.route('/api/fraud/batch', methods=['POST'])
def predict_fraud_batch():
    """
    Processa lote grande de transações com otimização
    Similar ao /predict mas com batching interno
    """
    if not request.json or 'transactions' not in request.json:
        raise ValidationError("transactions field is required")
    
    transactions_data = request.json['transactions']
    batch_size = request.json.get('batch_size', config.ml.batch_size)
    
    df = pd.DataFrame(transactions_data)
    
    logger.info(
        "Starting batch fraud predictions",
        request_id=g.request_id,
        num_transactions=len(df),
        batch_size=batch_size
    )
    
    # Processar em lotes
    all_predictions = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        predictions = fraud_engine.predict_detailed(batch)
        all_predictions.extend(predictions)
    
    results = [pred.to_dict() for pred in all_predictions]
    
    logger.info(
        "Batch predictions completed",
        request_id=g.request_id,
        num_predictions=len(results)
    )
    
    return jsonify({
        'success': True,
        'data': {
            'predictions': results,
            'summary': {
                'total': len(results),
                'frauds_detected': sum(1 for p in all_predictions if p.is_fraud),
                'batches_processed': (len(df) + batch_size - 1) // batch_size
            }
        }
    })

# ==========================================
# MODEL MANAGEMENT
# ==========================================

@app.route('/api/model/metrics', methods=['GET'])
def get_model_metrics():
    """Retorna métricas do modelo"""
    metrics = fraud_engine.get_performance_metrics()
    
    return jsonify({
        'success': True,
        'data': metrics
    })

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Retorna informações do modelo"""
    return jsonify({
        'success': True,
        'data': {
            'version': fraud_engine.VERSION,
            'is_trained': fraud_engine.is_trained,
            'threshold': fraud_engine.threshold,
            'feature_count': len(fraud_engine.feature_names),
            'features': fraud_engine.feature_names if fraud_engine.is_trained else []
        }
    })

# ==========================================
# DASHBOARD (MOCK DATA para demo)
# ==========================================

@app.route('/api/dashboard/kpis', methods=['GET'])
def get_dashboard_kpis():
    """KPIs do dashboard (mock data)"""
    import random
    
    kpis = {
        "transacoes_hoje": random.randint(10000, 15000),
        "transacoes_ontem": random.randint(9000, 14000),
        "fraudes_detectadas": random.randint(15, 30),
        "fraudes_ontem": random.randint(12, 28),
        "taxa_aprovacao": round(random.uniform(96.0, 99.0), 1),
        "taxa_aprovacao_ontem": round(random.uniform(95.0, 98.0), 1),
        "latencia_media": round(random.uniform(8.0, 15.0), 1),
        "latencia_ontem": round(random.uniform(9.0, 16.0), 1),
        "valor_protegido_hoje": round(random.uniform(2000000, 3000000), 2),
        "valor_protegido_ano": round(random.uniform(1000000000, 1500000000), 2),
        "familias_protegidas": random.randint(7000, 9000)
    }
    
    return jsonify({'success': True, 'data': kpis})

@app.route('/api/dashboard/timeseries', methods=['GET'])
def get_dashboard_timeseries():
    """Série temporal (mock data)"""
    import random
    
    timeseries = []
    for hour in range(24):
        timeseries.append({
            "time": f"{hour:02d}:00",
            "transactions": random.randint(150, 850),
            "latency": round(random.uniform(7.0, 12.0), 1)
        })
    
    return jsonify({'success': True, 'data': timeseries})

@app.route('/api/dashboard/channels', methods=['GET'])
def get_dashboard_channels():
    """Dados por canal (mock data)"""
    import random
    
    channels = [
        {"name": "PIX", "frauds": random.randint(5, 15), "value": random.randint(3000, 5000)},
        {"name": "Cartão", "frauds": random.randint(8, 20), "value": random.randint(2500, 4500)},
        {"name": "TED", "frauds": random.randint(1, 5), "value": random.randint(800, 1500)},
        {"name": "DOC", "frauds": random.randint(0, 3), "value": random.randint(500, 1200)}
    ]
    
    return jsonify({'success': True, 'data': channels})

@app.route('/api/dashboard/alerts', methods=['GET'])
def get_dashboard_alerts():
    """Alertas do sistema (mock data)"""
    alerts = [
        {
            "id": 1,
            "message": "Taxa de fraude acima do limite em PIX",
            "severity": "alto",
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        },
        {
            "id": 2,
            "message": "Latência elevada detectada no modelo",
            "severity": "medio",
            "timestamp": (datetime.utcnow() - timedelta(minutes=15)).isoformat() + 'Z'
        }
    ]
    
    return jsonify({'success': True, 'data': alerts})

@app.route('/api/dashboard/models', methods=['GET'])
def get_dashboard_models():
    """Status dos modelos"""
    metrics = fraud_engine.get_performance_metrics()
    
    if metrics['status'] == 'trained':
        models = [
            {
                "name": "Production Ensemble (RF+GB+LR)",
                "status": "healthy",
                "accuracy": round(metrics['metrics']['accuracy'] * 100, 1),
                "f1_score": round(metrics['metrics']['f1_score'] * 100, 1),
                "version": fraud_engine.VERSION
            }
        ]
    else:
        models = [
            {
                "name": "Production Ensemble",
                "status": "not_trained",
                "message": "Model needs to be trained"
            }
        ]
    
    return jsonify({'success': True, 'data': models})

@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    """Lista de transações (mock data)"""
    import random
    
    transactions = []
    for i in range(20):
        transactions.append({
            "id": f"TXN{i:06d}",
            "timestamp": (datetime.utcnow() - timedelta(minutes=i*5)).isoformat() + 'Z',
            "amount": round(random.uniform(10, 5000), 2),
            "channel": random.choice(["PIX", "Cartão", "TED", "DOC"]),
            "status": random.choice(["approved"] * 8 + ["fraud"]),
            "risk_score": round(random.uniform(0, 100), 1)
        })
    
    return jsonify({'success': True, 'data': transactions, 'total': len(transactions)})

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    logger.info(
        "🚀 Starting Sankofa Enterprise Pro - Production API",
        version=fraud_engine.VERSION,
        environment=config.environment,
        port=8000
    )
    
    # Verificar se modelo está treinado
    if not fraud_engine.is_trained:
        logger.warning(
            "⚠️  Fraud engine not trained - API will return errors for predictions",
            action_required="Train the model using /api/model/train endpoint or load pre-trained model"
        )
    else:
        logger.info(
            "✅ Fraud engine ready",
            metrics=fraud_engine.get_performance_metrics()
        )
    
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=config.debug,
        threaded=True
    )
