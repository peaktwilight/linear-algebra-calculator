#!/bin/bash

# Script to rename GitHub repository and update local settings

# Define variables
OLD_REPO_NAME="python_25fs"
NEW_REPO_NAME="linear-algebra-calculator"

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI is not installed. Please install it first."
    echo "Visit: https://cli.github.com/"
    exit 1
fi

# Check if the user is authenticated with GitHub
if ! gh auth status &> /dev/null; then
    echo "You need to authenticate with GitHub first."
    echo "Running: gh auth login"
    gh auth login
    
    if [ $? -ne 0 ]; then
        echo "Failed to log in to GitHub. Please try again manually."
        exit 1
    fi
fi

# Confirm the renaming
echo "This script will rename your GitHub repository from '$OLD_REPO_NAME' to '$NEW_REPO_NAME'."
echo "This action is irreversible and will update all references."
read -p "Do you want to proceed? (y/n): " CONFIRM

if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Repository renaming cancelled."
    exit 0
fi

# Get the GitHub username
GITHUB_USERNAME=$(gh api user | jq -r '.login')
if [ -z "$GITHUB_USERNAME" ]; then
    echo "Failed to get GitHub username."
    read -p "Please enter your GitHub username: " GITHUB_USERNAME
fi

echo "GitHub username: $GITHUB_USERNAME"

# Rename the repository on GitHub
echo "Renaming GitHub repository..."
gh repo rename $NEW_REPO_NAME --repo $GITHUB_USERNAME/$OLD_REPO_NAME

if [ $? -ne 0 ]; then
    echo "Failed to rename the repository on GitHub."
    exit 1
fi

echo "Repository renamed on GitHub successfully to: $GITHUB_USERNAME/$NEW_REPO_NAME"

# Update local git remote URLs
echo "Updating local Git remote URL..."
git remote set-url origin https://github.com/$GITHUB_USERNAME/$NEW_REPO_NAME.git

# Verify the change
echo "New Git remote URL:"
git remote -v

echo "Success! Repository renamed from '$OLD_REPO_NAME' to '$NEW_REPO_NAME'."
echo "The local Git configuration has been updated."
echo ""
echo "Things to note:"
echo "1. All GitHub URLs have been updated automatically."
echo "2. Any hard-coded references to the old repository name in your code should be updated manually."
echo "3. Clone URLs have changed - update them in any documentation."
echo ""
echo "New repository URL: https://github.com/$GITHUB_USERNAME/$NEW_REPO_NAME"