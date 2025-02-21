# Dockerfile for GAMA

FROM python:3.9-slim

# Install git
RUN apt-get update && apt-get install -y git

# Clone the GAMA repository
RUN rm -rf /src/gama && git clone https://github.com/openml-labs/gama /src/gama

# Change working directory to GAMA
WORKDIR /src/gama

# Install the required dependencies
RUN pip install .[test] coverage numpy==1.20.*

RUN pip freeze | grep numpy && rm -rf coverage.xml

# Run tests
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        echo "Running unit tests..." && \
        python -m pytest ./tests/unit --cov-report xml:coverage.xml --cov=. && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)) && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)
