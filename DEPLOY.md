# Deploying to Fly.io

This document explains how to deploy the Linear Algebra Calculator app to Fly.io.

## Prerequisites

1. Install the Fly CLI:
   ```
   # For macOS
   brew install flyctl
   
   # For Linux
   curl -L https://fly.io/install.sh | sh
   
   # For Windows
   powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
   ```

2. Sign up and log in to Fly.io:
   ```
   fly auth signup
   # OR if you already have an account
   fly auth login
   ```

## Deployment Steps

1. Navigate to the project directory:
   ```
   cd /path/to/linear-algebra-calculator
   ```

2. Launch the app on Fly.io:
   ```
   fly launch
   ```
   - When prompted, choose an app name or use the default
   - Select a region closest to you or your users
   - Create a volume for persistent data (optional)
   - Skip adding a PostgreSQL database
   - Skip adding a Redis database

3. Deploy the app:
   ```
   fly deploy
   ```

4. Open the deployed app:
   ```
   fly open
   ```

## Updating the App

After making changes to the code, you can update the deployed app with:
```
fly deploy
```

## Monitoring and Logs

To see logs from your app:
```
fly logs
```

To monitor your app's status:
```
fly status
```

## Scaling

To add more VMs or change the size of your VMs:
```
fly scale count 2  # Scale to 2 VMs
fly scale vm shared-cpu-1x  # Change VM size
```