#!/bin/bash

echo Cloning repositories...

# ANSI color codes
LIGHT_BLUE='\033[1;36m'
GREEN='\033[1;32m'
RESET='\033[0m'

print_heading() {
    local message="  $1  "
    local border
    border=$(printf '%*s' "${#message}" '' | tr ' ' '=')
    echo -e "\n${LIGHT_BLUE}${border}\n${message}\n${border}${RESET}\n"
}

print_line() {
    local message=$1
    echo -e "${GREEN}$0:${BASH_LINENO[0]}: -> ${message}${RESET}"
}

# Create directory for repositories
print_heading "Creating repositories directory..."
repo_dir=./repositories/
mkdir -p ${repo_dir}

# List of repository URLs
print_heading Cloning repositories...
repos=(
    # "https://github.com/tensorflow/adanet.git"
    "https://github.com/automl/Auto-PyTorch.git"
    "https://github.com/automl/auto-sklearn"
    "https://github.com/awslabs/autogluon"
    "https://github.com/keras-team/autokeras.git"
    "https://github.com/winedarksea/AutoTS"
    # "https://github.com/tinkoff-ai/etna"  # ETNA has been archived by owner
    "https://github.com/alteryx/evalml"
    "https://github.com/nccr-itmo/FEDOT"
    "https://github.com/microsoft/FLAML"
    "https://github.com/openml-labs/gama"
    "https://github.com/h2oai/h2o-3"
    "https://github.com/hyperopt/hyperopt-sklearn"
    "https://github.com/facebookresearch/Kats"
    "https://github.com/AILab-MLTools/LightAutoML"
    "https://github.com/ludwig-ai/ludwig"
    # "https://github.com/daochenzha/Meta-AAD"
    # "https://github.com/yzhao062/MetaOD"
    "https://github.com/AxeldeRomblay/MLBox"
    "https://github.com/mljar/mljar-supervised"
    "https://github.com/pycaret/pycaret"
    # "https://github.com/datamllab/pyodds"  # No updates since 2019
    "https://github.com/epistasislab/tpot"
)
echo "Repositories to clone: ${repos[*]}"

# Loop over repositories and clone or pull if already existing
for repo_url in "${repos[@]}"; do
    # Extract repository name
    repo_name=$(basename "${repo_url}" .git)
    # Convert to lowercase and replace hyphens with underscores
    repo_name=$(echo "${repo_name}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    print_heading "Cloning ${repo_name}"
    # Define repository path
    repo_path="${repo_dir}${repo_name}"

    # Clone or pull repository
    if [ -d "${repo_path}" ]; then
        print_line "Repository ${repo_name} already exists. Pulling latest changes..."
        git -C "${repo_path}" pull &
    else
        print_line "Cloning ${repo_name}..."
        git clone "${repo_url}" "${repo_path}"
    fi
done

print_heading "Repositories ready"
