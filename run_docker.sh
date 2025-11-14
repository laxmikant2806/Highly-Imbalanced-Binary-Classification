#!/bin/bash

echo "Building the Docker image..."
docker build -t fraud-detection-api .

echo "Running the Docker container..."
docker run -p 8000:8000 fraud-detection-api
