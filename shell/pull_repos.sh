#!/bin/bash

# Usage: ./shell/pull_repos.sh

echo Cloning repositories...

# Create directory for repositories
repo_dir=./repositories/
mkdir -p ${repo_dir}

# List of repository URLs
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

# Loop over repositories and clone or pull if already existing
for repo_url in "${repos[@]}"; do
    repo_name=$(basename ${repo_url} .git)
    repo_path="${repo_dir}${repo_name}"

    if [ -d "${repo_path}" ]; then
        echo "Repository ${repo_name} already exists. Pulling latest changes..."
        git -C "${repo_path}" pull
    else
        echo "Cloning ${repo_name}..."
        git clone "${repo_url}" "${repo_path}"
    fi
done

echo "Repositories ready"
