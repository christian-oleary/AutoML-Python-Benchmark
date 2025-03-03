#!/bin/bash

echo "Starting at $(date)"

# Input should be path to output directory
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <output_dir>"
    exit 1
fi

# Run clone_or_pull.sh
echo "Cloning/pulling repositories..."
./scripts/clone_or_pull.sh

# Run sonar_scanner.sh
echo "Running sonar scanner..."
./scripts/sonar_scanner.sh "$1" || echo "continuing"

# Run source code analysis
echo "Running source code analysis..."
./scripts/source_code_analysis.sh "$1"

echo "Done at $(date)"
