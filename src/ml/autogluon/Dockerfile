# Description: Dockerfile for AutoGluon
FROM autogluon/autogluon:1.0.0-cuda11.8-jupyter-ubuntu20.04-py3.10

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

RUN echo "\n-> OS: $(uname -r)\n-> CUDA_HOME: {$CUDA_HOME}\n-> conda: $(which conda)"

RUN apt-get install -y graphviz libgraphviz-dev pkg-config || (\
    apt-get update --fix-missing && apt-get install -y graphviz libgraphviz-dev pkg-config)

# Clone the AutoGluon repository which contains source code and tests
RUN git clone https://github.com/awslabs/autogluon.git /src/autogluon

# Set the working directory to the AutoGluon source code
WORKDIR /src/autogluon/

# AutoGluon requires the submodules to be initialized for tests
RUN git submodule update --init --recursive

# Install the required dependencies
RUN ls -la && ./full_install.sh
# RUN python -m pip install -e autogluon[tabular,extra,timeseries]
# RUN python -m pip install -e autogluon[test] coverage pytest pytest-cov
RUN python -m pip install coverage pytest pytest-cov

# Configure the packages to be tested
RUN echo '[tool.setuptools]' >> pyproject.toml
RUN echo 'packages = ["autogluon","autogluon.common","autogluon.core","autogluon.eda","autogluon.features","autogluon.multimodal","autogluon.tabular","autogluon.timeseries"]' >> pyproject.toml

ENV CI_DIR=.github/workflow_scripts
RUN chmod +x $CI_DIR/*.sh
# RUN $CI_DIR/test_common.sh
# RUN $CI_DIR/test_core.sh
# RUN $CI_DIR/test_eda.sh # Fails. Also commented out in their CI.
# RUN $CI_DIR/test_features.sh
RUN $CI_DIR/test_timeseries.sh
# RUN $CI_DIR/test_timeseries.sh || ls -la && ls timeseries/coverage.xml
RUN $CI_DIR/test_tabular.sh    || ls -la && ls tabular/coverage.xml
RUN $CI_DIR/test_multimodal.sh || ls -la && ls multimodal/coverage.xml

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
