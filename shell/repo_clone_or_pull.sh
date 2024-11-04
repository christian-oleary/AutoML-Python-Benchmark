#!/bin/bash

# Usage: ./shell/repo_clone_or_pull.sh

# Exit on error
# set -e

GREEN='\033[0;32m'
LIGHT_BLUE='\033[1;36m'
RESET_COLOR='\033[0m'

# Functions to print messages with fancy formatting
print_heading() {
    local message="  $1  "
    local border
    border=$(printf '%*s' "${#message}" '' | tr ' ' '=')
    echo -e "\n${GREEN}${border}\n${message}\n${border}${RESET_COLOR}\n"
}

print_subheading() {
    local message="  $1  "
    echo -e "\n${LIGHT_BLUE}-> ${message}${RESET_COLOR}"
}

# Create directory for repositories
print_heading "Creating repositories directory..."
repo_dir=./repositories/
mkdir -p ${repo_dir}

# List of repository URLs
print_heading Cloning repositories...
repos=(
    "https://github.com/tensorflow/adanet.git"
    "https://github.com/automl/Auto-PyTorch.git"
    "https://github.com/automl/auto-sklearn"
    "https://github.com/awslabs/autogluon"
    "https://github.com/keras-team/autokeras.git"
    "https://github.com/winedarksea/AutoTS"
    "https://github.com/tinkoff-ai/etna"
    "https://github.com/alteryx/evalml"
    "https://github.com/nccr-itmo/FEDOT"
    "https://github.com/microsoft/FLAML"
    "https://github.com/h2oai/h2o-3"
    "https://github.com/hyperopt/hyperopt-sklearn"
    "https://github.com/facebookresearch/Kats"
    "https://github.com/AILab-MLTools/LightAutoML"
    "https://github.com/ludwig-ai/ludwig"
    "https://github.com/daochenzha/Meta-AAD"
    "https://github.com/yzhao062/MetaOD"
    "https://github.com/AxeldeRomblay/MLBox"
    "https://github.com/mljar/mljar-supervised"
    "https://github.com/pycaret/pycaret"
    "https://github.com/datamllab/pyodds"
    "https://github.com/epistasislab/tpot"
)
echo "Repositories to clone: ${repos[*]}"

# Loop over repositories and clone or pull if already existing
for repo_url in "${repos[@]}"; do
    print_subheading "Cloning or pulling ${repo_url}..."
    repo_name=$(basename "${repo_url}" .git)
    repo_path="${repo_dir}${repo_name}"

    if [ -d "${repo_path}" ]; then
        echo "Repository ${repo_name} already exists. Pulling latest changes..."
        git -C "${repo_path}" pull
    else
        echo "Cloning ${repo_name}..."
        git clone "${repo_url}" "${repo_path}"
    fi
done

print_heading "Repositories ready"
