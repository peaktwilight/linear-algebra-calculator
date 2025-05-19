#!/bin/bash

# Wrapper script to ensure Streamlit binds properly

echo "Starting Streamlit application..."
echo "Environment: PORT=$PORT, STREAMLIT_SERVER_PORT=$STREAMLIT_SERVER_PORT"

# Set default port if not set
export PORT=${PORT:-8501}

# Make sure Streamlit uses the correct binding
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false

# Print network config for debugging
echo "Network interfaces:"
ip addr

echo "Starting Streamlit on port 8501, binding to 0.0.0.0..."
exec streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0