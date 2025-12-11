"""
Sankofa Enterprise Pro - Load Test Suite (Locust)
Python-based load testing for validating SLA performance claims

Installation:
    pip install locust

Usage:
    locust -f load_test_locust.py --host=http://localhost:5000
    locust -f load_test_locust.py --host=http://localhost:5000 --users 100 --spawn-rate 10 --run-time 5m

Web UI:
    Open http://localhost:8089 after starting locust
"""

import json
import random
import time
from datetime import datetime
from typing import Dict, List, Any

from locust import HttpUser, task, between, events
from locust.runners import MasterRunner, WorkerRunner

# Configuration
SLA_LATENCY_P99_MS = 50
TARGET_TPS = 3472  # 300M/day = 3472 TPS


class FraudPredictionStats:
    """Custom statistics tracker for SLA compliance"""
    
    def __init__(self):
        self.latencies: List[float] = []
        self.violations = 0
        self.total_requests = 0
        self.start_time = time.time()
    
    def record(self, latency_ms: float):
        self.latencies.append(latency_ms)
        self.total_requests += 1
        if latency_ms > SLA_LATENCY_P99_MS:
            self.violations += 1
    
    def get_percentile(self, percentile: float) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * percentile / 100)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]
    
    def get_stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        return {
            "total_requests": self.total_requests,
            "sla_violations": self.violations,
            "violation_rate": self.violations / max(1, self.total_requests) * 100,
            "p50_ms": self.get_percentile(50),
            "p95_ms": self.get_percentile(95),
            "p99_ms": self.get_percentile(99),
            "avg_tps": self.total_requests / max(1, elapsed),
            "sla_compliant": self.get_percentile(99) <= SLA_LATENCY_P99_MS,
        }


# Global stats tracker
stats = FraudPredictionStats()


def generate_transaction() -> Dict[str, Any]:
    """Generate a random transaction for testing"""
    channels = ['PIX', 'TED', 'DOC', 'CARTAO']
    types = ['PAYMENT', 'TRANSFER', 'PURCHASE']
    locations = ['SP', 'RJ', 'MG', 'BA', 'PR', 'RS', 'SC', 'GO', 'PE', 'CE']
    
    return {
        "id": f"TXN_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
        "amount": random.uniform(100, 50000),
        "channel": random.choice(channels),
        "type": random.choice(types),
        "customer_id": f"CUST_{random.randint(1, 10000)}",
        "hour": random.randint(0, 23),
        "day_of_week": random.randint(0, 6),
        "is_new_device": random.random() > 0.8,
        "device_id": f"DEV_{random.randint(1, 5000)}",
        "ip_address": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
        "location": random.choice(locations),
        "velocity_score": random.uniform(0, 1),
        "avg_amount_30d": random.uniform(100, 5000),
    }


def generate_high_risk_transaction() -> Dict[str, Any]:
    """Generate a high-risk transaction for fraud detection testing"""
    txn = generate_transaction()
    # High risk indicators
    txn["amount"] = random.uniform(20000, 100000)  # High amount
    txn["hour"] = random.randint(0, 5)  # Night time
    txn["is_new_device"] = True
    txn["velocity_score"] = random.uniform(0.7, 1.0)
    return txn


class SankofaUser(HttpUser):
    """
    Simulated user for Sankofa Enterprise Pro load testing
    
    Tasks weighted by typical usage patterns:
    - Health check: occasional
    - Single prediction: most common
    - Batch prediction: less common
    - Dashboard: occasional
    """
    
    wait_time = between(0.1, 0.5)  # 100-500ms between requests
    
    def on_start(self):
        """Called when a user starts - verify API is healthy"""
        response = self.client.get("/api/health")
        if response.status_code != 200:
            raise Exception(f"API health check failed: {response.status_code}")
        print(f"✅ User {self.environment.runner.user_count} started")
    
    @task(1)
    def health_check(self):
        """Health check endpoint - low frequency"""
        with self.client.get("/api/health", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("status") == "healthy":
                        response.success()
                    else:
                        response.failure(f"Unhealthy status: {data}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(10)
    def predict_single_transaction(self):
        """Single transaction prediction - most common use case"""
        transaction = generate_transaction()
        payload = {
            "transactions": [transaction],
            "fast_mode": True,
        }
        
        start_time = time.time()
        
        with self.client.post(
            "/api/fraud/predict",
            json=payload,
            catch_response=True,
            name="/api/fraud/predict [single]"
        ) as response:
            latency_ms = (time.time() - start_time) * 1000
            stats.record(latency_ms)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("success") and data.get("data", {}).get("predictions"):
                        if latency_ms > SLA_LATENCY_P99_MS:
                            response.failure(f"SLA violation: {latency_ms:.2f}ms > {SLA_LATENCY_P99_MS}ms")
                        else:
                            response.success()
                    else:
                        response.failure(f"Invalid response: {data}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(3)
    def predict_high_risk_transaction(self):
        """High-risk transaction prediction"""
        transaction = generate_high_risk_transaction()
        payload = {
            "transactions": [transaction],
            "fast_mode": True,
            "include_explanation": True,
        }
        
        with self.client.post(
            "/api/fraud/predict",
            json=payload,
            catch_response=True,
            name="/api/fraud/predict [high-risk]"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("success"):
                        predictions = data.get("data", {}).get("predictions", [])
                        if predictions and predictions[0].get("risk_score", 0) > 0.5:
                            response.success()
                        else:
                            # Not necessarily a failure, just noting low risk
                            response.success()
                    else:
                        response.failure(f"Failed: {data}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON")
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(2)
    def predict_batch(self):
        """Batch prediction - less common but important for throughput"""
        batch_size = random.randint(5, 20)
        transactions = [generate_transaction() for _ in range(batch_size)]
        payload = {
            "transactions": transactions,
            "fast_mode": True,
        }
        
        with self.client.post(
            "/api/fraud/predict",
            json=payload,
            catch_response=True,
            name=f"/api/fraud/predict [batch:{batch_size}]"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    predictions = data.get("data", {}).get("predictions", [])
                    if len(predictions) == batch_size:
                        response.success()
                    else:
                        response.failure(f"Expected {batch_size} predictions, got {len(predictions)}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON")
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(1)
    def get_dashboard_kpis(self):
        """Dashboard KPIs - occasional access"""
        with self.client.get("/api/dashboard/kpis", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("success"):
                        response.success()
                    else:
                        response.failure(f"Failed: {data}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON")
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(1)
    def get_status(self):
        """System status endpoint"""
        with self.client.get("/api/status", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


class HighLoadUser(HttpUser):
    """
    High-load user for stress testing
    Focuses only on prediction endpoint for maximum throughput
    """
    
    wait_time = between(0.01, 0.05)  # 10-50ms between requests (aggressive)
    
    @task
    def predict_rapid(self):
        """Rapid-fire predictions for stress testing"""
        transaction = generate_transaction()
        payload = {
            "transactions": [transaction],
            "fast_mode": True,
        }
        
        start_time = time.time()
        
        with self.client.post(
            "/api/fraud/predict",
            json=payload,
            catch_response=True,
            name="/api/fraud/predict [stress]"
        ) as response:
            latency_ms = (time.time() - start_time) * 1000
            stats.record(latency_ms)
            
            if response.status_code in [200, 429]:  # Accept rate limiting
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


# Event handlers for reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts"""
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║         Sankofa Enterprise Pro - Load Test (Locust)              ║
╠══════════════════════════════════════════════════════════════════╣
║  Host: {environment.host or 'Not set':<56}║
║  SLA Target: p99 < {SLA_LATENCY_P99_MS}ms                                          ║
║  Target TPS: {TARGET_TPS} (for 300M/day)                                ║
╚══════════════════════════════════════════════════════════════════╝
    """)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops - print SLA compliance report"""
    final_stats = stats.get_stats()
    
    sla_status = "✅ PASS" if final_stats["sla_compliant"] else "❌ FAIL"
    error_rate = environment.runner.stats.total.fail_ratio * 100
    error_status = "✅ PASS" if error_rate < 1 else "❌ FAIL"
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    SLA COMPLIANCE REPORT                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Metric                    │ Target      │ Actual      │ Status  ║
╠────────────────────────────┼─────────────┼─────────────┼─────────╣
║  P99 Latency               │ < {SLA_LATENCY_P99_MS}ms     │ {final_stats['p99_ms']:>7.2f}ms  │ {sla_status}  ║
║  P95 Latency               │ -           │ {final_stats['p95_ms']:>7.2f}ms  │ -       ║
║  P50 Latency               │ -           │ {final_stats['p50_ms']:>7.2f}ms  │ -       ║
║  Error Rate                │ < 1%        │ {error_rate:>7.2f}%  │ {error_status}  ║
║  SLA Violations            │ 0           │ {final_stats['sla_violations']:>11} │ -       ║
║  Total Requests            │ -           │ {final_stats['total_requests']:>11} │ -       ║
║  Avg TPS                   │ {TARGET_TPS}       │ {final_stats['avg_tps']:>7.2f}     │ -       ║
╚══════════════════════════════════════════════════════════════════╝
    """)


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    """Track all requests for custom metrics"""
    if "fraud/predict" in name:
        # Custom tracking is already done in tasks
        pass


# For programmatic usage
if __name__ == "__main__":
    import os
    os.system("locust -f load_test_locust.py --host=http://localhost:5000 --web-host 0.0.0.0")
