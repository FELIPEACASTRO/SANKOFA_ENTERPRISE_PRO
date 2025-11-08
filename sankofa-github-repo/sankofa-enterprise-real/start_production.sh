#!/bin/bash

echo "================================================================"
echo "🚀 SANKOFA ENTERPRISE PRO - PRODUCTION API STARTER"
echo "================================================================"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found! Please install Python 3.11+"
    exit 1
fi

echo "✅ Python found: $(python --version)"
echo ""

# Set environment
export ENVIRONMENT=development
export DEBUG=true
export JWT_SECRET=dev-secret-change-in-production
export ENCRYPTION_KEY=dev-encryption-key-change-in-production

echo "📋 Environment configured:"
echo "   - ENVIRONMENT: $ENVIRONMENT"
echo "   - DEBUG: $DEBUG"
echo ""

# Navigate to backend
cd backend

echo "🔥 Starting Production API..."
echo "   API will be available at: http://localhost:8445"
echo "   Health check: http://localhost:8445/api/health"
echo "   Status: http://localhost:8445/api/status"
echo ""
echo "================================================================"
echo ""

# Start API
python api/production_api.py
