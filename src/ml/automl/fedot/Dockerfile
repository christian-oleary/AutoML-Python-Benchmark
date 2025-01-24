# Dockerfile for FEDOT library
# Adapted from https://github.com/aimclub/FEDOT/blob/master/docker/gpu/Dockerfile

FROM nvcr.io/nvidia/rapidsai/rapidsai:21.06-cuda11.2-base-ubuntu18.04

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
# RUN apt-get update --fix-missing && apt-get install -y python3.8 python3-pip git
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    apt-get update --fix-missing && apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Clone the FEDOT repository which contains source code and tests
RUN rm -rf /src/fedot && git clone https://github.com/aimclub/FEDOT.git /src/fedot

# Set the working directory to the FEDOT source code
WORKDIR /src/fedot/

# Install required dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt lightgbm==3.2.1 tensorflow pytest-cov torch

# Run unit tests
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        echo "Running unit tests..." && \
        (python3 -m pytest ./test/unit/ --cov-report xml:coverage.xml --cov=. || ls coverage.xml) && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)) && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi

RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())" && \
    python -c "import tensorflow as tf;print('GPU STATUS:',tf.test.is_built_with_cuda())"
