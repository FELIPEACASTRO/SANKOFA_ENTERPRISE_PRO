#!/bin/bash

# Start backend API in background
cd sankofa-enterprise-real/backend && python api/main_integrated_api.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start frontend
cd ../frontend && npm run dev

# Cleanup on exit
trap "kill $BACKEND_PID" EXIT
