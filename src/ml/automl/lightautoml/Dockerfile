# Description: Dockerfile for LightAutoML

FROM pytorch/pytorch:latest
# FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

RUN uname -r && echo $CUDA_HOME && which conda

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install git and LightAutoML dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y git build-essential libcairo2 libpango-1.0-0 \
        libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info

# Clone the LightAutoML repository
RUN rm -rf /src/lightautoml && git clone https://github.com/sb-ai-lab/LightAutoML.git /src/lightautoml

# Set the working directory
WORKDIR /src/lightautoml

# Install the required packages
RUN python -m pip install --upgrade pip
RUN python -m pip install -e .[all] pytest-cov
RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"

# Remove old test reports if they exist
RUN rm -f report.xml coverage.xml .coverage ./**/report.xml ./**/coverage.xml ./**/.coverage

# Run tests
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        echo "Running unit tests..." && \
        python -m pytest --cov-report xml:coverage.xml --cov=. && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)) && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)
RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"
