#!/bin/bash
set -e

echo "=== Sankofa Enterprise Pro - Build Script ==="
echo "Current directory: $(pwd)"

WORKSPACE_ROOT="/home/runner/workspace"
FRONTEND_DIR="$WORKSPACE_ROOT/sankofa-enterprise-real/frontend"
BACKEND_STATIC="$WORKSPACE_ROOT/sankofa-enterprise-real/backend/static"

echo "Installing frontend dependencies..."
cd "$FRONTEND_DIR"
npm install

echo "Building frontend..."
npm run build

echo "Copying build to backend static..."
rm -rf "$BACKEND_STATIC"/*
cp -r dist/* "$BACKEND_STATIC/"

echo "Build completed successfully!"
ls -la "$BACKEND_STATIC/"
