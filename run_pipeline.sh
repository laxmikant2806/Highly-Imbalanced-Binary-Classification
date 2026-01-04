#!/bin/bash

echo "Running the fraud detection pipeline..."

# Run the main script as a module to ensure proper imports
python -m src.main

echo "Pipeline finished."