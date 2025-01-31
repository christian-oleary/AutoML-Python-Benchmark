# Description: Dockerfile for AutoGluon

# FROM autogluon/autogluon:1.0.0-cuda11.8-framework-ubuntu20.04-py3.10
# # FROM autogluon/autogluon:0.8.2-cuda11.7-framework-ubuntu20.04-py3.9
# # FROM autogluon/autogluon:0.5.2-cuda11.2-jupyter-ubuntu20.04-py3.8

FROM pytorch/pytorch:latest
# FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

# Allow torch to access the GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN echo "\n-> OS: $(uname -r)\n-> CUDA_HOME: {$CUDA_HOME}\n-> conda: $(which conda)"

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install git and AutoGluon dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y git graphviz libgraphviz-dev pkg-config

# Clone the AutoGluon repository which contains source code and tests
RUN rm -rf /src/autogluon && git clone https://github.com/awslabs/autogluon.git /src/autogluon

# Set the working directory to the AutoGluon source code
WORKDIR /src/autogluon/

# Initialize any submodules
RUN git submodule update --init --recursive

# Configure the packages to be tested
RUN echo '[tool.setuptools]' >> pyproject.toml && \
    echo 'packages = ["autogluon","autogluon.common","autogluon.core","autogluon.eda","autogluon.features","autogluon.multimodal","autogluon.tabular","autogluon.timeseries"]' >> pyproject.toml

# Attempts to allow torch to access GPU.
# RUN conda install -c conda-forge mamba
# RUN mamba install -c conda-forge -c pytorch -c nvidia autogluon "pytorch=*=*cuda*"
# RUN python -m pip install -q -e autogluon torch
# RUN python -m pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"

# Install the required dependencies
RUN ./full_install.sh && \
    python -m pip install -e autogluon[tabular,extra,timeseries] && \
    python -m pip install -e autogluon[test] coverage pytest pytest-cov

# Directory to the AutoGluon CI scripts
ENV CI_DIR=.github/workflow_scripts
RUN chmod +x $CI_DIR/*.sh

# Remove any existing coverage.xml files
RUN rm -rf results.xml, coverage.xml, .coverage, ./**/results.xml ./**/coverage.xml ./**/.coverage

# Run the test scripts
# Common
RUN sed -i 's/--junitxml=results.xml/--cov-report xml:coverage.xml --cov=. --cov-fail-under=1/g' $CI_DIR/test_common.sh && \
    $CI_DIR/test_common.sh && (ls common/.coverage || exit)

# Core
RUN sed -i 's/--junitxml=results.xml/--cov-report xml:coverage.xml --cov=. --cov-fail-under=1/g' $CI_DIR/test_core.sh && \
    $CI_DIR/test_core.sh && (ls core/.coverage || exit)

# EDA
# test_eda.sh fails. It is commented out in their CI.
# RUN sed -i 's/--junitxml=results.xml/--cov-report xml:coverage.xml --cov=. --cov-fail-under=1/g' $CI_DIR/test_eda.sh && \
#     $CI_DIR/test_eda.sh && (ls eda/.coverage || exit)

# Features
RUN sed -i 's/--junitxml=results.xml/--cov-report xml:coverage.xml --cov=. --cov-fail-under=1/g' $CI_DIR/test_features.sh && \
    $CI_DIR/test_features.sh && (ls features/.coverage || exit)

# Time series
RUN sed -i 's/--junitxml=results.xml/--cov-report xml:coverage.xml --cov=. --cov-fail-under=1/g' $CI_DIR/test_timeseries.sh && \
    $CI_DIR/test_timeseries.sh && (ls timeseries/.coverage || exit)

# Multimodal
RUN sed -i 's/--junitxml=results.xml/--cov-report xml:coverage.xml --cov=. --cov-fail-under=1/g' $CI_DIR/test_multimodal.sh && \
    $CI_DIR/test_multimodal.sh && (ls multimodal/.coverage || exit)

# Tabular
RUN sed -i 's/--junitxml=results.xml/--cov-report xml:coverage.xml --cov=. --cov-fail-under=1/g' $CI_DIR/test_tabular.sh && \
    $CI_DIR/test_tabular.sh && (ls tabular/.coverage || exit)

# Combine .coverage files
RUN ls && echo "--------" && python -m coverage combine ./**/.coverage

# Generate the coverage report
RUN ls && echo "--------" && python -m coverage xml -o coverage.xml && ls

# Print the number of lines with non-zero coverage
RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l

# Print the GPU status
RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"
