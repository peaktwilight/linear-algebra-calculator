#!/bin/bash

# Script to run the Linear Algebra Calculator with Docker Compose

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in the PATH."
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed or not in the PATH."
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

echo "Starting Linear Algebra Calculator with Docker Compose..."
echo "The application will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the application."

# Run Docker Compose
docker-compose up

# This will be executed when the user presses Ctrl+C
echo "Stopping Linear Algebra Calculator..."
docker-compose down

echo "Linear Algebra Calculator has been stopped."