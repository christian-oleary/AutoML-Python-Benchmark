# Description: Dockerfile for PyOD

FROM pytorch/pytorch:latest
# FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install git and dependencies
RUN apt-get install -y git || (apt-get update --fix-missing && apt-get install -y git)

# Clone the PyOD repository which contains source code and tests
RUN rm -rf /src/pyod && git clone https://github.com/yzhao062/pyod.git /src/pyod

# Set the working directory to the PyOD source code
WORKDIR /src/pyod/

# Initialize any submodules
RUN git submodule update --init --recursive

# Install the required dependencies
RUN python -m pip install --upgrade pip && \
    python -m pip install -r docs/requirements.txt && \
    python -m pip install . pytest-cov xgboost

# Remove old test reports if they exist
RUN rm -f report.xml coverage.xml .coverage ./**/report.xml ./**/coverage.xml ./**/.coverage

# Run tests if the run_tests is set to "true"
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        (python3 -m pytest --cov-branch --cov-report xml:coverage.xml --cov=. --cov-fail-under=1 || ls coverage.xml) && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "FAILED" && exit 1)) \
    ; fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)

RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"
