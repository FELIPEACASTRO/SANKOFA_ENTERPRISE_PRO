# ============================================================================
# SANKOFA ENTERPRISE PRO - PRODUCTION DOCKERFILE
# ============================================================================
# Multi-stage build for optimized production image
# Target size: <500MB
# Python 3.12 + Production-ready security hardening
# ============================================================================

# ============================================================================
# STAGE 1: Builder - Install dependencies and compile
# ============================================================================
FROM python:3.12-slim AS builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG VCS_REF

# Labels for image metadata
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.title="Sankofa Enterprise Pro" \
      org.opencontainers.image.description="Production-grade fraud detection system" \
      org.opencontainers.image.vendor="Sankofa Enterprise" \
      maintainer="Sankofa Team"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
WORKDIR /app
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# STAGE 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.12-slim AS runtime

# Security: Create non-root user
RUN groupadd -r sankofa && \
    useradd -r -g sankofa -s /bin/false -c "Sankofa App User" sankofa

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app directory
WORKDIR /app

# Copy application code
COPY --chown=sankofa:sankofa backend/ /app/backend/
COPY --chown=sankofa:sankofa frontend/dist/ /app/frontend/dist/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/experiments && \
    chown -R sankofa:sankofa /app

# Security: Remove setuid/setgid permissions
RUN find / -xdev -perm /6000 -type f -exec chmod a-s {} \; || true

# Switch to non-root user
USER sankofa

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Set working directory
WORKDIR /app/backend

# Run application
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "4", \
     "--threads", "2", \
     "--worker-class", "gthread", \
     "--worker-tmp-dir", "/dev/shm", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--keep-alive", "5", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "api.production_api:app"]
