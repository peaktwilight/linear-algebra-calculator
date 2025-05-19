#!/bin/bash

# Script to redeploy the Linear Algebra Calculator to Fly.io with fixes

# Check if Fly CLI is installed
if ! command -v fly &> /dev/null; then
    echo "Error: Fly CLI is not installed or not in the PATH."
    echo "Please install Fly CLI first:"
    echo "  For macOS: brew install flyctl"
    echo "  For Linux: curl -L https://fly.io/install.sh | sh"
    echo "  For Windows: iwr https://fly.io/install.ps1 -useb | iex"
    exit 1
fi

# Check if the user is logged in
fly auth whoami &> /dev/null
if [ $? -ne 0 ]; then
    echo "You need to log in to Fly.io first."
    echo "Running: fly auth login"
    fly auth login
    
    if [ $? -ne 0 ]; then
        echo "Failed to log in to Fly.io. Please try again manually."
        exit 1
    fi
fi

echo "Redeploying Linear Algebra Calculator to Fly.io with fixed configuration..."

# Ask for confirmation to destroy and recreate
read -p "Would you like to destroy and recreate the app (recommended)? (y/n): " DESTROY_CONFIRM

if [[ $DESTROY_CONFIRM == "y" || $DESTROY_CONFIRM == "Y" ]]; then
    echo "Destroying the existing app..."
    fly apps destroy linear-algebra-calculator --yes
    
    echo "Creating app again..."
    fly apps create linear-algebra-calculator
    
    echo "Allocating volume..."
    fly volumes create linalgdata --size 1 --region fra
    
    echo "Deploying the app with new configuration..."
    fly deploy
else
    # Just redeploy with the existing configuration
    echo "Deploying with updated configuration..."
    fly deploy --strategy immediate
fi

if [ $? -eq 0 ]; then
    echo "Deployment successful!"
    echo "Checking the deployment status..."
    fly status
    
    echo "Opening the app in your browser..."
    fly open
else
    echo "Deployment failed. Please check the error messages above."
    exit 1
fi