# Deploying the Linear Algebra Calculator to Fly.io

This document provides the exact terminal commands to deploy the application to Fly.io.

## Prerequisites

1. Create a Fly.io account at https://fly.io/
2. Install the Fly CLI:
   ```bash
   # For macOS
   brew install flyctl
   
   # For Linux
   curl -L https://fly.io/install.sh | sh
   
   # For Windows
   powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
   ```

## Step 1: Authenticate with Fly.io

Login to your Fly.io account:
```bash
fly auth login
```

## Step 2: Initialize the Fly.io Application

Navigate to your project directory and launch the app:
```bash
cd /Users/peak/Downloads/python_25fs
fly launch
```

During the launch process, Fly.io will detect the Dockerfile in your project. You'll be prompted to:
- Create an app name (or use the default "linear-algebra-calculator")
- Choose an organization
- Select a region (choose one closest to your users)
- Set up PostgreSQL/Redis (select "no")

## Step 3: Deploy the Application

After initializing, deploy the application:
```bash
fly deploy
```

This command will:
1. Build your Docker image
2. Push the image to Fly.io
3. Deploy the application
4. Provide a URL where your app is deployed (e.g., https://linear-algebra-calculator.fly.dev)

## Step 4: Open Your Deployed Application

Once deployed, you can open your application using:
```bash
fly open
```

This will open your default browser to the application URL.

## Additional Commands

### View Deployment Logs

To view logs from your running application:
```bash
fly logs
```

### Check Application Status

To see your application status:
```bash
fly status
```

### Scale Your Application

To add more instances or change VM size:
```bash
# Add more VMs (default is 1)
fly scale count 2

# Change VM size
fly scale vm shared-cpu-1x
```

### Update Your Application

After making changes to your code, redeploy with:
```bash
fly deploy
```

## Troubleshooting

If you encounter issues during deployment:

1. Check your application logs:
   ```bash
   fly logs
   ```

2. Verify your Dockerfile is correct and your application runs locally:
   ```bash
   docker build -t linear-algebra-calculator .
   docker run -p 8501:8501 linear-algebra-calculator
   ```

3. Visit the Fly.io documentation for more help:
   https://fly.io/docs/

4. If you need to destroy the application and start over:
   ```bash
   fly apps destroy linear-algebra-calculator
   ```