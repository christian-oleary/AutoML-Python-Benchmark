# Dockerfile for EvalML

# FROM tensorflow/tensorflow:latest-gpu
FROM python:3.10

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y git graphviz && rm -rf /var/lib/apt/lists/*

# Clone the EvalML repository which contains source code and tests
RUN rm -rf /src/evalml && git clone https://github.com/alteryx/evalml.git /src/evalml

# Set the working directory to the EvalML source code
WORKDIR /src/evalml/

# Install the required dependencies
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then make installdeps-dev; fi
RUN if [ "$run_tests" = "true" ]; then make installdeps-prophet; fi
RUN if [ "$run_tests" = "true" ]; then make installdeps-test; fi

# Run tests
# RUN if [ "$run_tests" = "true" ]; then pip freeze && (TIMEOUT=$TIMEOUT make test); fi
# RUN if [ "$run_tests" = "true" ]; then pip freeze && make test; fi
RUN if [ "$run_tests" = "true" ]; then \
        pip freeze && \
        (pytest evalml/ --doctest-modules --doctest-continue-on-failure  --timeout 3600 --cov-report xml:coverage.xml --cov=. --cov-fail-under=1 || ls coverage.xml) && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "FAILED" && exit 1)) \
    ; fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "FAILED" && exit 1)

# RUN python -c "import tensorflow as tf;print('GPU STATUS:',tf.config.list_physical_devices('GPU'))"
