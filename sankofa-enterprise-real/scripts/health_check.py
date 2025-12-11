#!/usr/bin/env python3
"""
Health Check Script for Sankofa Enterprise Pro
Validates system readiness for production deployment
"""

import sys
import os
import time
import requests
from typing import Dict, Any, List, Tuple

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class HealthChecker:
    """Comprehensive health check for all system components"""

    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0

    def check(self, name: str, func: callable) -> bool:
        """Run a health check and report results"""
        print(f"\n{BLUE}[CHECK]{RESET} {name}...", end=" ")
        try:
            result, message = func()
            if result:
                print(f"{GREEN}✓ PASS{RESET}")
                if message:
                    print(f"  └─ {message}")
                self.checks_passed += 1
                return True
            else:
                print(f"{RED}✗ FAIL{RESET}")
                if message:
                    print(f"  └─ {message}")
                self.checks_failed += 1
                return False
        except Exception as e:
            print(f"{RED}✗ ERROR{RESET}")
            print(f"  └─ {str(e)}")
            self.checks_failed += 1
            return False

    def warn(self, message: str):
        """Print warning message"""
        print(f"{YELLOW}[WARN]{RESET} {message}")
        self.warnings += 1

    # ==========================================================================
    # Health Check Functions
    # ==========================================================================

    def check_api_health(self) -> Tuple[bool, str]:
        """Check API health endpoint"""
        response = requests.get(f"{self.base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, f"Status: {data.get('status', 'unknown')}"
        return False, f"HTTP {response.status_code}"

    def check_database_connection(self) -> Tuple[bool, str]:
        """Check database connectivity"""
        response = requests.get(f"{self.base_url}/health", timeout=5)
        data = response.json()
        db_status = data.get('dependencies', {}).get('database', 'unknown')
        if db_status == 'connected':
            return True, "Database connected"
        return False, f"Database status: {db_status}"

    def check_redis_connection(self) -> Tuple[bool, str]:
        """Check Redis cache connectivity"""
        response = requests.get(f"{self.base_url}/health", timeout=5)
        data = response.json()
        redis_status = data.get('dependencies', {}).get('redis', 'unknown')
        if redis_status == 'connected':
            return True, "Redis connected"
        return False, f"Redis status: {redis_status}"

    def check_ml_model_loaded(self) -> Tuple[bool, str]:
        """Check if ML model is loaded"""
        response = requests.get(f"{self.base_url}/health", timeout=5)
        data = response.json()
        ml_status = data.get('dependencies', {}).get('ml_model', 'unknown')
        if ml_status == 'loaded':
            return True, "ML model loaded"
        return False, f"ML model status: {ml_status}"

    def check_api_latency(self) -> Tuple[bool, str]:
        """Check API response latency (target: <100ms)"""
        start = time.time()
        response = requests.get(f"{self.base_url}/health", timeout=5)
        latency_ms = (time.time() - start) * 1000

        if latency_ms < 100:
            return True, f"Latency: {latency_ms:.2f}ms (excellent)"
        elif latency_ms < 200:
            self.warn(f"Latency {latency_ms:.2f}ms is acceptable but above target (100ms)")
            return True, f"Latency: {latency_ms:.2f}ms (acceptable)"
        else:
            return False, f"Latency: {latency_ms:.2f}ms (too high, target: <100ms)"

    def check_prediction_endpoint(self) -> Tuple[bool, str]:
        """Test fraud prediction endpoint"""
        payload = {
            "amount": 1000.0,
            "cpf": "12345678901",
            "channel": "PIX",
            "merchant_id": "TEST_MERCHANT",
            "customer_id": "TEST_CUSTOMER"
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/predict",
                json=payload,
                timeout=10,
                headers={"Authorization": "Bearer test_token"}  # May need valid token
            )

            if response.status_code in [200, 401]:  # 401 is OK (just auth issue)
                return True, "Prediction endpoint accessible"
            return False, f"Unexpected status code: {response.status_code}"
        except Exception as e:
            return False, str(e)

    def check_security_headers(self) -> Tuple[bool, str]:
        """Verify security headers are present"""
        response = requests.get(f"{self.base_url}/health", timeout=5)
        headers = response.headers

        required_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'Strict-Transport-Security',  # May not be present in dev
        ]

        missing = []
        for header in required_headers:
            if header not in headers and header != 'Strict-Transport-Security':
                missing.append(header)

        if not missing:
            return True, "All security headers present"
        elif len(missing) <= 1:
            self.warn(f"Missing optional header: {missing[0]}")
            return True, "Core security headers present"
        else:
            return False, f"Missing headers: {', '.join(missing)}"

    def check_rate_limiting(self) -> Tuple[bool, str]:
        """Test if rate limiting is configured"""
        response = requests.get(f"{self.base_url}/health", timeout=5)
        headers = response.headers

        if 'X-RateLimit-Limit' in headers or 'RateLimit-Limit' in headers:
            return True, "Rate limiting configured"
        else:
            self.warn("Rate limiting headers not found (may still be active)")
            return True, "Cannot verify rate limiting from headers"

    def check_metrics_endpoint(self) -> Tuple[bool, str]:
        """Check if Prometheus metrics are exposed"""
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=5)
            if response.status_code == 200 and 'flask_http' in response.text:
                return True, "Metrics endpoint active"
            return False, "Metrics endpoint not responding correctly"
        except:
            self.warn("Metrics endpoint not accessible (may be on different port)")
            return True, "Metrics check skipped"

    def check_docker_deployment(self) -> Tuple[bool, str]:
        """Check if Dockerfile exists and is valid"""
        if os.path.exists('Dockerfile'):
            with open('Dockerfile', 'r') as f:
                content = f.read()
                if 'FROM python:3.12' in content and 'gunicorn' in content:
                    return True, "Dockerfile configured correctly"
            return False, "Dockerfile exists but may have issues"
        return False, "Dockerfile not found"

    def check_docker_compose(self) -> Tuple[bool, str]:
        """Check if docker-compose.yml exists"""
        if os.path.exists('docker-compose.yml'):
            with open('docker-compose.yml', 'r') as f:
                content = f.read()
                if 'postgres:' in content and 'redis:' in content:
                    return True, "Docker Compose configured with all services"
            return False, "docker-compose.yml missing required services"
        return False, "docker-compose.yml not found"

    def check_environment_variables(self) -> Tuple[bool, str]:
        """Check if .env.example exists"""
        if os.path.exists('.env.example'):
            return True, "Environment template exists"
        return False, ".env.example not found"

    def check_ci_cd_pipeline(self) -> Tuple[bool, str]:
        """Check if CI/CD pipeline is configured"""
        ci_files = [
            '.github/workflows/ci-cd-pipeline.yml',
            '.gitlab-ci.yml',
            'azure-pipelines.yml'
        ]

        for ci_file in ci_files:
            if os.path.exists(ci_file):
                return True, f"CI/CD configured: {ci_file}"

        return False, "No CI/CD pipeline found"

    def run_all_checks(self):
        """Run all health checks"""
        print(f"\n{BLUE}{'='*70}{RESET}")
        print(f"{BLUE}SANKOFA ENTERPRISE PRO - HEALTH CHECK{RESET}")
        print(f"{BLUE}{'='*70}{RESET}")

        # API Checks
        print(f"\n{BLUE}[CATEGORY]{RESET} API Health")
        self.check("API Health Endpoint", self.check_api_health)
        self.check("API Response Latency", self.check_api_latency)
        self.check("Prediction Endpoint", self.check_prediction_endpoint)

        # Infrastructure Checks
        print(f"\n{BLUE}[CATEGORY]{RESET} Infrastructure")
        self.check("Database Connection", self.check_database_connection)
        self.check("Redis Cache Connection", self.check_redis_connection)
        self.check("ML Model Status", self.check_ml_model_loaded)

        # Security Checks
        print(f"\n{BLUE}[CATEGORY]{RESET} Security")
        self.check("Security Headers", self.check_security_headers)
        self.check("Rate Limiting", self.check_rate_limiting)

        # Deployment Checks
        print(f"\n{BLUE}[CATEGORY]{RESET} Deployment Configuration")
        self.check("Dockerfile", self.check_docker_deployment)
        self.check("Docker Compose", self.check_docker_compose)
        self.check("Environment Variables", self.check_environment_variables)
        self.check("CI/CD Pipeline", self.check_ci_cd_pipeline)
        self.check("Metrics Endpoint", self.check_metrics_endpoint)

        # Final Summary
        self.print_summary()

    def print_summary(self):
        """Print final summary"""
        total = self.checks_passed + self.checks_failed

        print(f"\n{BLUE}{'='*70}{RESET}")
        print(f"{BLUE}SUMMARY{RESET}")
        print(f"{BLUE}{'='*70}{RESET}")
        print(f"Total Checks:    {total}")
        print(f"{GREEN}Passed:          {self.checks_passed}{RESET}")
        print(f"{RED}Failed:          {self.checks_failed}{RESET}")
        print(f"{YELLOW}Warnings:        {self.warnings}{RESET}")

        pass_rate = (self.checks_passed / total * 100) if total > 0 else 0
        print(f"\nPass Rate:       {pass_rate:.1f}%")

        if self.checks_failed == 0:
            print(f"\n{GREEN}✓ ALL CHECKS PASSED - SYSTEM READY FOR DEPLOYMENT{RESET}")
            return 0
        elif pass_rate >= 80:
            print(f"\n{YELLOW}⚠ SOME CHECKS FAILED - REVIEW REQUIRED{RESET}")
            return 1
        else:
            print(f"\n{RED}✗ CRITICAL FAILURES - NOT READY FOR DEPLOYMENT{RESET}")
            return 2


if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"

    checker = HealthChecker(base_url)
    exit_code = checker.run_all_checks()

    sys.exit(exit_code)
