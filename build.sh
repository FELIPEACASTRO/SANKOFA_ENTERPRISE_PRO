#!/bin/bash
set -e

echo "=== Sankofa Enterprise Pro - Build Script ==="
echo "Current directory: $(pwd)"

echo "Installing frontend dependencies..."
cd sankofa-enterprise-real/frontend
npm install

echo "Building frontend..."
npm run build

echo "Build completed successfully!"
ls -la dist/
