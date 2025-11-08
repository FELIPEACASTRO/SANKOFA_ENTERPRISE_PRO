#!/usr/bin/env python3
"""
API Simplificada do Sankofa Enterprise Pro para demonstração
"""

import os
from datetime import datetime, timedelta
from flask import Flask, jsonify
from flask_cors import CORS
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Mock data for dashboard
MOCK_KPI_DATA = {
    "transacoes_hoje": 12547,
    "transacoes_ontem": 11832,
    "fraudes_detectadas": 23,
    "fraudes_ontem": 18,
    "taxa_aprovacao": 97.8,
    "taxa_aprovacao_ontem": 97.2,
    "latencia_media": 10.3,
    "latencia_ontem": 12.1,
    "valor_protegido_hoje": 2847392.50,
    "valor_protegido_ano": 1247382940.00,
    "familias_protegidas": 8432
}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/dashboard/kpis', methods=['GET'])
def get_dashboard_kpis():
    """Returns dashboard KPIs"""
    return jsonify({"success": True, "data": MOCK_KPI_DATA})

@app.route('/api/dashboard/timeseries', methods=['GET'])
def get_dashboard_timeseries():
    """Returns timeseries data"""
    timeseries = []
    for hour in range(24):
        timeseries.append({
            "time": f"{hour:02d}:00",
            "transactions": random.randint(150, 850),
            "latency": round(random.uniform(7.0, 12.0), 1)
        })
    return jsonify({"success": True, "data": timeseries})

@app.route('/api/dashboard/channels', methods=['GET'])
def get_dashboard_channels():
    """Returns channel fraud data"""
    channels = [
        {"name": "PIX", "frauds": 8, "value": 4523},
        {"name": "Cartão", "frauds": 12, "value": 3892},
        {"name": "TED", "frauds": 3, "value": 1247},
        {"name": "DOC", "frauds": 0, "value": 892}
    ]
    return jsonify({"success": True, "data": channels})

@app.route('/api/dashboard/alerts', methods=['GET'])
def get_dashboard_alerts():
    """Returns system alerts"""
    alerts = [
        {
            "id": 1,
            "message": "Taxa de fraude acima do limite em PIX",
            "severity": "alto",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": 2,
            "message": "Latência elevada detectada no modelo XGBoost",
            "severity": "medio",
            "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat()
        }
    ]
    return jsonify({"success": True, "data": alerts})

@app.route('/api/dashboard/models', methods=['GET'])
def get_dashboard_models():
    """Returns ML model status"""
    models = [
        {"name": "XGBoost Ensemble", "status": "healthy", "accuracy": 94.2},
        {"name": "Random Forest", "status": "healthy", "accuracy": 92.8},
        {"name": "Neural Network", "status": "healthy", "accuracy": 93.5},
        {"name": "Isolation Forest", "status": "healthy", "accuracy": 89.1},
        {"name": "LSTM Temporal", "status": "healthy", "accuracy": 91.7}
    ]
    return jsonify({"success": True, "data": models})

@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    """Returns mock transactions"""
    transactions = []
    for i in range(20):
        transactions.append({
            "id": f"TXN{i:06d}",
            "timestamp": (datetime.now() - timedelta(minutes=i*5)).isoformat(),
            "amount": round(random.uniform(10, 5000), 2),
            "channel": random.choice(["PIX", "Cartão", "TED", "DOC"]),
            "status": random.choice(["approved", "approved", "approved", "approved", "fraud"]),
            "risk_score": round(random.uniform(0, 100), 1)
        })
    return jsonify({"success": True, "data": transactions, "total": len(transactions)})

if __name__ == "__main__":
    logger.info("🚀 Iniciando Sankofa Enterprise Pro API Simplificada...")
    logger.info("✅ API pronta para receber requisições")
    app.run(host="localhost", port=8445, debug=True, threaded=True)
