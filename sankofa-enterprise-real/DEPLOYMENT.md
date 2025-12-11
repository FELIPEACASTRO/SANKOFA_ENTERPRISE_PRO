# SANKOFA ENTERPRISE PRO - DEPLOYMENT GUIDE

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Monitoring & Observability](#monitoring--observability)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

- **Docker**: >= 24.0
- **Docker Compose**: >= 2.20
- **Python**: 3.12+
- **Node.js**: 20+ (for frontend development)
- **PostgreSQL**: 16+ (if running locally)
- **Redis**: 7+ (if running locally)

### Cloud Services (Production)

- **AWS Account** (recommended)
  - ECS/Fargate for container orchestration
  - RDS PostgreSQL for database
  - ElastiCache Redis for caching
  - S3 for ML model storage
  - Secrets Manager for credential management
  - CloudWatch for logging

---

## Local Development

### 1. Clone Repository

```bash
git clone https://github.com/your-org/sankofa-enterprise-real.git
cd sankofa-enterprise-real
```

### 2. Setup Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and configure:

```env
# Database
POSTGRES_DB=sankofa_fraud_db
POSTGRES_USER=sankofa_admin
POSTGRES_PASSWORD=your_secure_password

# Redis
REDIS_PASSWORD=your_redis_password

# Application
FLASK_ENV=development
SECRET_KEY=your_secret_key_min_32_chars
JWT_SECRET_KEY=your_jwt_secret_key
```

### 3. Start Services with Docker Compose

```bash
# Start all services (API, PostgreSQL, Redis, Prometheus, Grafana)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### 4. Verify Installation

```bash
# Health check
curl http://localhost:5000/health

# API documentation
open http://localhost:5000/apidocs

# Grafana dashboards
open http://localhost:3001  # admin/admin123
```

---

## Docker Deployment

### Build Docker Image

```bash
# Build image
docker build -t sankofa-api:latest .

# Build with version tag
docker build \
  --build-arg VERSION=1.0.0 \
  --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  -t sankofa-api:1.0.0 \
  .

# Verify image size (target: <500MB)
docker images sankofa-api
```

### Run Container

```bash
# Run API container
docker run -d \
  --name sankofa-api \
  -p 5000:5000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  -e REDIS_HOST=redis-host \
  -e REDIS_PASSWORD=redis-pass \
  -e SECRET_KEY=your-secret-key \
  -e JWT_SECRET_KEY=your-jwt-key \
  sankofa-api:latest

# Check logs
docker logs -f sankofa-api

# Execute commands inside container
docker exec -it sankofa-api bash
```

### Push to Registry

```bash
# GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin

docker tag sankofa-api:latest ghcr.io/your-org/sankofa-api:latest
docker push ghcr.io/your-org/sankofa-api:latest

# AWS ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

docker tag sankofa-api:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/sankofa-api:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/sankofa-api:latest
```

---

## Production Deployment

### AWS ECS/Fargate Deployment

#### 1. Prepare Infrastructure

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name sankofa-production

# Create task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Create service
aws ecs create-service \
  --cluster sankofa-production \
  --service-name sankofa-api \
  --task-definition sankofa-api:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=sankofa-api,containerPort=5000"
```

#### 2. Database Setup (RDS)

```bash
# Create PostgreSQL RDS instance
aws rds create-db-instance \
  --db-instance-identifier sankofa-postgres \
  --db-instance-class db.r6g.xlarge \
  --engine postgres \
  --engine-version 16.1 \
  --master-username sankofa_admin \
  --master-user-password $DB_PASSWORD \
  --allocated-storage 100 \
  --storage-type gp3 \
  --multi-az \
  --vpc-security-group-ids sg-xxx \
  --backup-retention-period 7 \
  --preferred-backup-window "03:00-04:00"

# Run migrations
docker run --rm \
  -e DATABASE_URL=$PRODUCTION_DB_URL \
  sankofa-api:latest \
  python -m infrastructure.database migrate
```

#### 3. Cache Setup (ElastiCache)

```bash
# Create Redis cluster
aws elasticache create-replication-group \
  --replication-group-id sankofa-redis \
  --replication-group-description "Sankofa Redis Cluster" \
  --engine redis \
  --cache-node-type cache.r6g.large \
  --num-cache-clusters 3 \
  --automatic-failover-enabled \
  --multi-az-enabled \
  --at-rest-encryption-enabled \
  --transit-encryption-enabled
```

#### 4. Secrets Management

```bash
# Store secrets in AWS Secrets Manager
aws secretsmanager create-secret \
  --name sankofa/production/db \
  --secret-string '{"username":"sankofa_admin","password":"xxx"}'

aws secretsmanager create-secret \
  --name sankofa/production/jwt \
  --secret-string '{"secret_key":"xxx","jwt_secret":"xxx"}'

# Update ECS task definition to reference secrets
# (See ecs-task-definition.json for secret references)
```

### Environment Variables (Production)

```env
# Application
FLASK_ENV=production
FLASK_DEBUG=0

# Database (from Secrets Manager)
DB_HOST=sankofa-postgres.xxx.us-east-1.rds.amazonaws.com
DB_PORT=5432
DB_NAME=sankofa_fraud_db
DB_POOL_MIN=10
DB_POOL_MAX=100

# Redis (from ElastiCache)
REDIS_HOST=sankofa-redis.xxx.cache.amazonaws.com
REDIS_PORT=6379
REDIS_PASSWORD=${REDIS_PASSWORD}  # From Secrets Manager

# Security
SECRET_KEY=${SECRET_KEY}  # From Secrets Manager
JWT_SECRET_KEY=${JWT_SECRET_KEY}  # From Secrets Manager
ENABLE_CORS=true
CORS_ORIGINS=https://app.sankofa.com

# Observability
PROMETHEUS_ENABLED=true
LOG_LEVEL=INFO
SENTRY_DSN=${SENTRY_DSN}
DATADOG_API_KEY=${DATADOG_API_KEY}

# ML
ML_MODEL_PATH=s3://sankofa-models/production/
FEATURE_STORE_ENABLED=true
```

### Load Balancer Configuration

```bash
# Create Application Load Balancer
aws elbv2 create-load-balancer \
  --name sankofa-alb \
  --subnets subnet-xxx subnet-yyy \
  --security-groups sg-xxx \
  --scheme internet-facing

# Create target group
aws elbv2 create-target-group \
  --name sankofa-api-tg \
  --protocol HTTP \
  --port 5000 \
  --vpc-id vpc-xxx \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 3

# Create listener
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:... \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=arn:aws:acm:... \
  --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...
```

---

## Monitoring & Observability

### Prometheus + Grafana Setup

1. **Access Grafana**: `http://localhost:3001` (local) or `https://grafana.sankofa.com` (production)
2. **Default credentials**: admin / admin123 (change immediately)
3. **Import dashboards**:
   - Go to Dashboards > Import
   - Upload `monitoring/grafana/dashboards/*.json`

### Key Metrics to Monitor

| Metric | Target | Alert Threshold |
|--------|--------|----------------|
| API Latency (P95) | <50ms | >100ms |
| API Latency (P99) | <100ms | >200ms |
| Throughput | >2000 req/s | <1000 req/s |
| Error Rate | <0.1% | >1% |
| CPU Usage | <70% | >85% |
| Memory Usage | <80% | >90% |
| Database Connections | <80 | >90 |
| Cache Hit Rate | >80% | <70% |

### CloudWatch Alarms (Production)

```bash
# High error rate alarm
aws cloudwatch put-metric-alarm \
  --alarm-name sankofa-high-error-rate \
  --alarm-description "Alert when API error rate exceeds 1%" \
  --metric-name ErrorRate \
  --namespace Sankofa/API \
  --statistic Average \
  --period 300 \
  --evaluation-periods 2 \
  --threshold 1.0 \
  --comparison-operator GreaterThanThreshold \
  --alarm-actions arn:aws:sns:us-east-1:xxx:sankofa-alerts

# High latency alarm
aws cloudwatch put-metric-alarm \
  --alarm-name sankofa-high-latency \
  --metric-name Latency_P95 \
  --namespace Sankofa/API \
  --statistic Average \
  --period 300 \
  --threshold 100 \
  --comparison-operator GreaterThanThreshold
```

---

## Troubleshooting

### Common Issues

#### 1. Database Connection Failures

```bash
# Check database connectivity
docker exec -it sankofa-api python -c "
from infrastructure.database import test_connection
test_connection()
"

# Check pool status
docker exec -it sankofa-api python -c "
from api.production_api import db_persistence
print(f'Pool available: {db_persistence.is_available}')
"
```

#### 2. Redis Cache Issues

```bash
# Test Redis connection
docker exec -it sankofa-redis redis-cli ping

# Check cache stats
curl http://localhost:5000/api/v1/cache/stats
```

#### 3. High Latency

```bash
# Check Prometheus metrics
curl http://localhost:9090/api/v1/query?query=flask_http_request_duration_seconds_bucket

# Profile API endpoint
docker exec -it sankofa-api python -m cProfile -o profile.stats api/production_api.py
```

#### 4. Container Crashes

```bash
# Check logs
docker logs --tail 100 sankofa-api

# Check resource usage
docker stats sankofa-api

# Inspect container
docker inspect sankofa-api
```

### Performance Tuning

#### Database

```sql
-- Check slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
WHERE mean_exec_time > 100
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Analyze table
ANALYZE transactions;

-- Reindex
REINDEX TABLE transactions;
```

#### Application

```bash
# Increase Gunicorn workers
docker run -e GUNICORN_WORKERS=8 sankofa-api:latest

# Tune DB pool
docker run -e DB_POOL_MAX=200 sankofa-api:latest

# Enable query cache
docker run -e FEATURE_STORE_ENABLED=true sankofa-api:latest
```

---

## Deployment Checklist

### Pre-Production

- [ ] All tests passing (269/269)
- [ ] Security scan completed (Bandit, Trivy)
- [ ] Load testing completed (>1000 concurrent users)
- [ ] Database migrations tested
- [ ] Secrets configured in Secrets Manager
- [ ] SSL certificates provisioned
- [ ] DNS configured
- [ ] Monitoring dashboards created
- [ ] Alerting configured
- [ ] Runbook documented

### Production Deployment

- [ ] Database backup created
- [ ] Blue-green deployment configured
- [ ] Canary deployment tested
- [ ] Rollback plan documented
- [ ] On-call rotation configured
- [ ] Stakeholders notified
- [ ] Feature flags disabled (if any)
- [ ] Rate limits configured
- [ ] CORS origins whitelisted
- [ ] Security headers verified

### Post-Deployment

- [ ] Smoke tests passed
- [ ] Metrics dashboard green
- [ ] Error rate <0.1%
- [ ] Latency P95 <50ms
- [ ] Database performance normal
- [ ] Cache hit rate >80%
- [ ] No alerts firing
- [ ] Documentation updated

---

## Support

- **Documentation**: https://docs.sankofa.com
- **Issues**: https://github.com/your-org/sankofa-enterprise-real/issues
- **Slack**: #sankofa-support
- **Email**: support@sankofa.com
- **On-call**: PagerDuty escalation

---

**Last Updated**: 2025-12-11
**Version**: 1.0.0
