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
RUN echo "" >> pyproject.toml && \
    echo '[tool.setuptools]' >> pyproject.toml && \
    echo 'packages = ["autogluon","autogluon.common","autogluon.core","autogluon.eda","autogluon.features","autogluon.multimodal","autogluon.tabular","autogluon.timeseries"]' >> pyproject.toml && \
    echo "" >> pyproject.toml

# Install the required dependencies
RUN ./full_install.sh && \
    python -m pip install -e autogluon[tabular,extra,timeseries] && \
    python -m pip install -e autogluon[test] coverage pytest pytest-cov

# Directory to the AutoGluon CI scripts
ENV CI_DIR=.github/workflow_scripts
RUN chmod +x $CI_DIR/*.sh

# Remove any existing coverage.xml files
RUN rm -rf results.xml, coverage.xml, .coverage, ./**/results.xml ./**/coverage.xml ./**/.coverage

# Install dependencies for the test scripts
RUN apt-get update && apt-get install -y g++ && g++ --version
RUN sed -i 's/pip install/pip install -q/g' $CI_DIR/env_setup.sh
RUN sed -i 's/setup_hf_model_mirror "$SUB_FOLDER"/# setup_hf_model_mirror "$SUB_FOLDER"/g' $CI_DIR/test_multimodal.sh
RUN "${CI_DIR}/test_install.sh"

##### Run the test scripts #####
# Format commands
RUN cd $CI_DIR && \
    (find . -type f -exec sed -i.bak "s/--junitxml=results.xml/--cov-branch --cov-append --cov-report xml:coverage.xml --cov=. --cov-fail-under=1/g" {} \;) && \
    cd ..

# Run the test scripts
RUN $CI_DIR/test_common.sh || ls common/.coverage
RUN ls common/coverage.xml && wc -l common/coverage.xml

# Attempts to allow torch to access GPU.
RUN conda install -y -c conda-forge mamba
RUN mamba install -y -c conda-forge -c pytorch -c nvidia autogluon "pytorch=*=*cuda*"
# RUN python -m pip install -q -e autogluon torch
RUN python -m pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"

RUN $CI_DIR/test_core.sh || ls core/.coverage
RUN ls core/coverage.xml && wc -l core/coverage.xml

# RUN $CI_DIR/test_eda.sh && ls eda/.coverage
RUN $CI_DIR/test_features.sh || ls features/.coverage
RUN ls features/coverage.xml && wc -l features/coverage.xml

RUN $CI_DIR/test_tabular.sh || ls tabular/.coverage
RUN ls tabular/coverage.xml && wc -l tabular/coverage.xml

RUN $CI_DIR/test_timeseries.sh || ls timeseries/.coverage
RUN ls timeseries/coverage.xml && wc -l timeseries/coverage.xml

RUN $CI_DIR/test_multimodal_predictor.sh --run_single_gpu || ls multimodal/.coverage
RUN ls multimodal/coverage.xml && wc -l multimodal/coverage.xml

RUN $CI_DIR/test_multimodal_others.sh --run_single_gpu || ls multimodal/.coverage
RUN ls multimodal/coverage.xml && wc -l multimodal/coverage.xml

# Combine .coverage files, generate report, and show coverage
# RUN python -m coverage combine --keep ./**/.coverage
# RUN python -m coverage xml -i -o coverage.xml
# RUN coverage report -mi && \
RUN python -m coverage combine --keep ./**/.coverage && \
    python -m coverage xml -i -o coverage.xml && \
    coverage report -mi && \
    wc -l common/coverage.xml && \
    wc -l core/coverage.xml && \
    wc -l features/coverage.xml && \
    wc -l tabular/coverage.xml && \
    wc -l timeseries/coverage.xml && \
    wc -l multimodal/coverage.xml && \
    cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l

# Print the GPU status
RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"

RUN head -n 3 coverage.xml && \
    echo "COMMON:" && \
    head -n 3 common/coverage.xml && \
    echo "CORE:" && \
    head -n 3 core/coverage.xml && \
    echo "TABULAR:" && \
    head -n 3 tabular/coverage.xml && \
    echo "TIMESERIES:" && \
    head -n 3 timeseries/coverage.xml && \
    echo "MULTIMODAL:" && \
    head -n 3 multimodal/coverage.xml
