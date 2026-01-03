#!/bin/bash
# Frontend static site build configuration for Render

# Install dependencies
npm ci

# Build the application
npm run build

echo "Frontend build completed successfully!"
