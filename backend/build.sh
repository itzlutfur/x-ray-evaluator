#!/usr/bin/env bash
# Exit on error
set -o errexit

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# =============================================================================
# MODEL FILES SETUP
# =============================================================================
# Your .keras model files are NOT in Git (too large).
# You MUST provide at least ONE model file for the app to work.
#
# Option 1: Download from public URL (Google Drive, Dropbox, GitHub Release)
# Example:
mkdir -p models
# curl -L -o models/ResNet50.keras "YOUR_DIRECT_DOWNLOAD_URL_HERE"
#
# Option 2: Use environment variable with model URL
# if [ -n "$MODEL_DOWNLOAD_URL" ]; then
#   curl -L -o models/ResNet50.keras "$MODEL_DOWNLOAD_URL"
# fi
#
# Option 3: Upload to Render Disk (paid feature) via SSH after deployment
#
# WARNING: Without at least one model file, the API will return errors!
# =============================================================================

echo "Build completed successfully!"
echo "NOTE: Ensure at least one .keras model file is available in models/ directory"
