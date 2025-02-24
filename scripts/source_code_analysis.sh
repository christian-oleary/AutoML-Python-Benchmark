#!/bin/bash

# Input should be path to output directory
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <output_dir>"
    exit 1
fi

# Clone/pull the repositories
./scripts/clone_or_pull.sh

# Activate the automl conda environment
# shellcheck source=/dev/null
source activate automl

# Run the Python module ml.sca
python -m ml.sca repositories "$1"
