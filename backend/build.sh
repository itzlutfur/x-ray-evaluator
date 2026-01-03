#!/usr/bin/env bash
# Exit on error
set -o errexit

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Note: Model files (.keras) are NOT included in the repository
# If you need to download models from external storage, add commands here:
# Example:
# mkdir -p models
# curl -o models/ResNet50.keras "YOUR_MODEL_URL_HERE"
# OR use AWS CLI, gsutil, etc. to download from cloud storage

echo "Build completed successfully!"
