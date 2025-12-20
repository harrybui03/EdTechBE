#!/bin/bash

# Build script for Agentic RAG Docker image

set -e

IMAGE_NAME="agentic-rag"
IMAGE_TAG="${1:-latest}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building Docker image: ${FULL_IMAGE_NAME}"

# Build the image
docker build -t "${FULL_IMAGE_NAME}" .

echo "✅ Docker image built successfully: ${FULL_IMAGE_NAME}"

# Optionally tag as latest if not already
if [ "${IMAGE_TAG}" != "latest" ]; then
    docker tag "${FULL_IMAGE_NAME}" "${IMAGE_NAME}:latest"
    echo "✅ Tagged as ${IMAGE_NAME}:latest"
fi

echo ""
echo "To run the container:"
echo "  docker run -p 8002:8002 --env-file .env ${FULL_IMAGE_NAME}"
echo ""
echo "Or use docker-compose:"
echo "  docker-compose up -d"

