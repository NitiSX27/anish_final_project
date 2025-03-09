#!/bin/bash
# Build script for Render deployment

# Install system dependencies if needed
# apt-get update && apt-get install -y some-package

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Make scripts executable
chmod +x build.sh

echo "Build completed successfully!"
