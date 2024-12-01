# Description: Dockerfile for AutoGluon
FROM autogluon/autogluon:1.0.0-cuda11.8-framework-ubuntu20.04-py3.10
# FROM autogluon/autogluon:0.8.2-cuda11.7-framework-ubuntu20.04-py3.9
# FROM autogluon/autogluon:0.5.2-cuda11.2-jupyter-ubuntu20.04-py3.8

# Allow torch to access the GPU
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

RUN echo "\n-> OS: $(uname -r)\n-> CUDA_HOME: {$CUDA_HOME}\n-> conda: $(which conda)"

RUN apt-get update --fix-missing && apt-get install -y graphviz libgraphviz-dev pkg-config

# Clone the AutoGluon repository which contains source code and tests
RUN git clone https://github.com/awslabs/autogluon.git /src/autogluon

# Set the working directory to the AutoGluon source code
WORKDIR /src/autogluon/

# AutoGluon requires the submodules to be initialized for tests
RUN git submodule update --init --recursive

# Configure the packages to be tested
RUN echo '[tool.setuptools]' >> pyproject.toml
RUN echo 'packages = ["autogluon","autogluon.common","autogluon.core","autogluon.eda","autogluon.features","autogluon.multimodal","autogluon.tabular","autogluon.timeseries"]' >> pyproject.toml

# Attempts to allow torch to access GPU.
# RUN conda install -c conda-forge mamba
# RUN mamba install -c conda-forge -c pytorch -c nvidia autogluon "pytorch=*=*cuda*"
# RUN python -m pip install -q -e autogluon torch
# RUN python -m pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"

# Install the required dependencies
RUN ./full_install.sh
# RUN pip freeze | grep torch && python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"
RUN python -m pip install -e autogluon[tabular,extra,timeseries]
# RUN pip freeze | grep torch && python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"
RUN python -m pip install -e autogluon[test] coverage pytest pytest-cov
# RUN pip freeze | grep torch && python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"
RUN python -m pip install coverage pytest pytest-cov
# RUN pip freeze | grep torch && python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"

ENV CI_DIR=.github/workflow_scripts
RUN chmod +x $CI_DIR/*.sh
RUN $CI_DIR/test_common.sh
RUN $CI_DIR/test_core.sh
# test_eda.sh fails. It is commented out in their CI.
RUN $CI_DIR/test_eda.sh || echo "FAILED"
RUN $CI_DIR/test_features.sh

RUN $CI_DIR/test_timeseries.sh
#  || echo "FAILED"
# RUN /usr/bin/bash $CI_DIR/test_tabular.sh || echo "FAILED"
# RUN $CI_DIR/test_multimodal.sh || echo "FAILED"

# RUN $CI_DIR/test_timeseries.sh || ls -la && ls timeseries/coverage.xml
# RUN $CI_DIR/test_tabular.sh    || ls -la && ls tabular/coverage.xml
# RUN $CI_DIR/test_multimodal.sh || ls -la && ls multimodal/coverage.xml

# # # Run tests if the run_tests is set to "true"
# # ARG run_tests
# # RUN if [ "$run_tests" = "true" ]; then \
# #         echo "Installing test dependencies..." && \
# #         python -m pip install -e autogluon[test] coverage pytest pytest-cov && \
# #         echo "Running unit tests..." && \
# #         python -m pytest --cov-report xml:coverage.xml --cov=. --disable-warnings && \
# #         ls coverage.xml && \
# #         echo "Tests complete"; \
# #     else \
# #         echo "Skipping tests"; \
# #     fi

RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"
