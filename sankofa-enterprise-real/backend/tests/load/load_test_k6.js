/**
 * Sankofa Enterprise Pro - Load Test Suite
 * k6 script for validating SLA performance claims
 * 
 * Installation: https://k6.io/docs/getting-started/installation/
 * 
 * Usage:
 *   k6 run load_test_k6.js
 *   k6 run --vus 100 --duration 60s load_test_k6.js
 *   K6_PROMETHEUS_RW_SERVER_URL=http://localhost:9090 k6 run load_test_k6.js
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Counter, Rate, Trend, Gauge } from 'k6/metrics';
import { randomIntBetween, randomItem } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics
const fraudPredictionLatency = new Trend('fraud_prediction_latency_ms');
const healthCheckLatency = new Trend('health_check_latency_ms');
const slaViolations = new Counter('sla_violations_total');
const slaCompliance = new Rate('sla_compliance_rate');
const concurrentUsers = new Gauge('concurrent_users');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:5000';
const SLA_LATENCY_P99_MS = parseInt(__ENV.SLA_LATENCY_MS || '50');

// Test scenarios
export const options = {
    scenarios: {
        // Scenario 1: Smoke test - Basic functionality
        smoke: {
            executor: 'constant-vus',
            vus: 1,
            duration: '30s',
            tags: { scenario: 'smoke' },
        },
        
        // Scenario 2: Load test - Normal load
        load: {
            executor: 'ramping-vus',
            startVUs: 0,
            stages: [
                { duration: '1m', target: 50 },   // Ramp up
                { duration: '3m', target: 50 },   // Steady state
                { duration: '1m', target: 100 },  // Peak load
                { duration: '2m', target: 100 },  // Sustained peak
                { duration: '1m', target: 0 },    // Ramp down
            ],
            startTime: '35s',
            tags: { scenario: 'load' },
        },
        
        // Scenario 3: Stress test - Find breaking point
        stress: {
            executor: 'ramping-arrival-rate',
            startRate: 10,
            timeUnit: '1s',
            preAllocatedVUs: 200,
            maxVUs: 500,
            stages: [
                { duration: '1m', target: 100 },   // 100 req/s
                { duration: '2m', target: 500 },   // 500 req/s
                { duration: '2m', target: 1000 },  // 1000 req/s (target: 10k/s for 300M/day)
                { duration: '1m', target: 0 },
            ],
            startTime: '10m',
            tags: { scenario: 'stress' },
        },
        
        // Scenario 4: Spike test - Sudden traffic spike
        spike: {
            executor: 'ramping-vus',
            startVUs: 10,
            stages: [
                { duration: '10s', target: 10 },
                { duration: '10s', target: 200 },  // Spike!
                { duration: '30s', target: 200 },  // Hold
                { duration: '10s', target: 10 },   // Back to normal
            ],
            startTime: '20m',
            tags: { scenario: 'spike' },
        },
    },
    
    thresholds: {
        // SLA Thresholds
        'http_req_duration': ['p(95)<100', 'p(99)<200'],
        'fraud_prediction_latency_ms': [`p(99)<${SLA_LATENCY_P99_MS}`],
        'http_req_failed': ['rate<0.01'],  // <1% errors
        'sla_compliance_rate': ['rate>0.99'],  // >99% compliance
        
        // Per-scenario thresholds
        'http_req_duration{scenario:smoke}': ['p(99)<100'],
        'http_req_duration{scenario:load}': ['p(99)<200'],
    },
};

// Test data generators
function generateTransaction() {
    const channels = ['PIX', 'TED', 'DOC', 'CARTAO'];
    const types = ['PAYMENT', 'TRANSFER', 'PURCHASE'];
    
    return {
        id: `TXN_${Date.now()}_${randomIntBetween(1000, 9999)}`,
        amount: randomIntBetween(100, 50000) + Math.random(),
        channel: randomItem(channels),
        type: randomItem(types),
        customer_id: `CUST_${randomIntBetween(1, 10000)}`,
        hour: randomIntBetween(0, 23),
        day_of_week: randomIntBetween(0, 6),
        is_new_device: Math.random() > 0.8,
        device_id: `DEV_${randomIntBetween(1, 5000)}`,
        ip_address: `192.168.${randomIntBetween(1, 255)}.${randomIntBetween(1, 255)}`,
        location: randomItem(['SP', 'RJ', 'MG', 'BA', 'PR']),
    };
}

function generateBatch(size) {
    return Array.from({ length: size }, generateTransaction);
}

// Main test function
export default function() {
    concurrentUsers.add(__VU);
    
    group('Health Check', function() {
        const healthRes = http.get(`${BASE_URL}/api/health`);
        
        healthCheckLatency.add(healthRes.timings.duration);
        
        check(healthRes, {
            'health status is 200': (r) => r.status === 200,
            'health response is healthy': (r) => {
                try {
                    return JSON.parse(r.body).status === 'healthy';
                } catch (e) {
                    return false;
                }
            },
        });
    });
    
    group('Fraud Prediction - Single', function() {
        const transaction = generateTransaction();
        
        const payload = JSON.stringify({
            transactions: [transaction],
            fast_mode: true,
        });
        
        const params = {
            headers: {
                'Content-Type': 'application/json',
            },
        };
        
        const startTime = Date.now();
        const res = http.post(`${BASE_URL}/api/fraud/predict`, payload, params);
        const latency = Date.now() - startTime;
        
        fraudPredictionLatency.add(latency);
        
        // Check SLA compliance
        const slaCompliant = latency <= SLA_LATENCY_P99_MS;
        slaCompliance.add(slaCompliant);
        
        if (!slaCompliant) {
            slaViolations.add(1);
        }
        
        check(res, {
            'predict status is 200': (r) => r.status === 200,
            'predict response has success': (r) => {
                try {
                    return JSON.parse(r.body).success === true;
                } catch (e) {
                    return false;
                }
            },
            'predict response has predictions': (r) => {
                try {
                    const body = JSON.parse(r.body);
                    return body.data && body.data.predictions && body.data.predictions.length > 0;
                } catch (e) {
                    return false;
                }
            },
            [`predict latency < ${SLA_LATENCY_P99_MS}ms`]: () => slaCompliant,
        });
    });
    
    group('Fraud Prediction - Batch', function() {
        const batchSize = randomIntBetween(5, 20);
        const transactions = generateBatch(batchSize);
        
        const payload = JSON.stringify({
            transactions: transactions,
            fast_mode: true,
        });
        
        const params = {
            headers: {
                'Content-Type': 'application/json',
            },
        };
        
        const res = http.post(`${BASE_URL}/api/fraud/predict`, payload, params);
        
        check(res, {
            'batch predict status is 200': (r) => r.status === 200,
            'batch predict processes all transactions': (r) => {
                try {
                    const body = JSON.parse(r.body);
                    return body.data && body.data.predictions && 
                           body.data.predictions.length === batchSize;
                } catch (e) {
                    return false;
                }
            },
        });
    });
    
    group('Dashboard KPIs', function() {
        const res = http.get(`${BASE_URL}/api/dashboard/kpis`);
        
        check(res, {
            'kpis status is 200': (r) => r.status === 200,
            'kpis response has data': (r) => {
                try {
                    const body = JSON.parse(r.body);
                    return body.success === true && body.data !== undefined;
                } catch (e) {
                    return false;
                }
            },
        });
    });
    
    // Random sleep between 100ms and 500ms to simulate realistic traffic
    sleep(randomIntBetween(100, 500) / 1000);
}

// Lifecycle hooks
export function setup() {
    console.log(`
╔══════════════════════════════════════════════════════════════════╗
║         Sankofa Enterprise Pro - Load Test Suite                 ║
╠══════════════════════════════════════════════════════════════════╣
║  Target URL: ${BASE_URL.padEnd(49)}║
║  SLA Target: p99 < ${SLA_LATENCY_P99_MS}ms                                          ║
║  Target TPS: 3,472 (for 300M/day)                                ║
╚══════════════════════════════════════════════════════════════════╝
    `);
    
    // Verify API is up
    const healthRes = http.get(`${BASE_URL}/api/health`);
    if (healthRes.status !== 200) {
        throw new Error(`API health check failed: ${healthRes.status}`);
    }
    
    return { startTime: Date.now() };
}

export function teardown(data) {
    const duration = (Date.now() - data.startTime) / 1000;
    console.log(`
╔══════════════════════════════════════════════════════════════════╗
║                    Load Test Complete                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Duration: ${duration.toFixed(2)}s                                               ║
║  Check the k6 output above for detailed metrics                  ║
╚══════════════════════════════════════════════════════════════════╝
    `);
}

// Custom summary handler
export function handleSummary(data) {
    const p99Latency = data.metrics.fraud_prediction_latency_ms?.values?.['p(99)'] || 0;
    const slaCompliant = p99Latency <= SLA_LATENCY_P99_MS;
    const totalRequests = data.metrics.http_reqs?.values?.count || 0;
    const errorRate = data.metrics.http_req_failed?.values?.rate || 0;
    
    console.log(`
╔══════════════════════════════════════════════════════════════════╗
║                    SLA COMPLIANCE REPORT                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Metric                    │ Target      │ Actual      │ Status  ║
╠────────────────────────────┼─────────────┼─────────────┼─────────╣
║  P99 Latency               │ < ${SLA_LATENCY_P99_MS}ms     │ ${p99Latency.toFixed(2).padStart(7)}ms  │ ${slaCompliant ? '✅ PASS' : '❌ FAIL'}  ║
║  Error Rate                │ < 1%        │ ${(errorRate * 100).toFixed(2).padStart(7)}%  │ ${errorRate < 0.01 ? '✅ PASS' : '❌ FAIL'}  ║
║  Total Requests            │ -           │ ${String(totalRequests).padStart(11)} │ -       ║
╚══════════════════════════════════════════════════════════════════╝
    `);
    
    return {
        'stdout': JSON.stringify(data, null, 2),
        'summary.json': JSON.stringify(data),
        'summary.html': generateHTMLReport(data),
    };
}

function generateHTMLReport(data) {
    return `<!DOCTYPE html>
<html>
<head>
    <title>Sankofa Load Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        h1 { color: #333; }
        .metric { display: inline-block; margin: 10px; padding: 15px; background: #f0f0f0; border-radius: 4px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2196F3; }
        .metric-name { font-size: 12px; color: #666; }
        .pass { color: green; }
        .fail { color: red; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏦 Sankofa Enterprise Pro - Load Test Report</h1>
        <p>Generated: ${new Date().toISOString()}</p>
        <div class="metric">
            <div class="metric-value">${(data.metrics.fraud_prediction_latency_ms?.values?.['p(99)'] || 0).toFixed(2)}ms</div>
            <div class="metric-name">P99 Latency</div>
        </div>
        <div class="metric">
            <div class="metric-value">${data.metrics.http_reqs?.values?.count || 0}</div>
            <div class="metric-name">Total Requests</div>
        </div>
        <div class="metric">
            <div class="metric-value">${((data.metrics.http_req_failed?.values?.rate || 0) * 100).toFixed(2)}%</div>
            <div class="metric-name">Error Rate</div>
        </div>
    </div>
</body>
</html>`;
}
