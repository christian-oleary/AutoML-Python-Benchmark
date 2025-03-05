# Dockerfile for mljar

FROM python:3.9-slim

# Install git
RUN apt-get update && \
    apt-get install -y git libgomp1 && \
    rm -rf /src/mljar_supervised

# Clone the mljar repository
RUN git clone https://github.com/mljar/mljar-supervised.git /src/mljar_supervised

# Change working directory to mljar
WORKDIR /src/mljar_supervised

# Install the required dependencies
RUN pip install -r requirements.txt && \
    pip install -r requirements_dev.txt importlib-metadata>=1.7.0 ipython setuptools && \
    python setup.py install

# Run tests
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        rm -rf coverage.xml && \
        echo "Running unit tests..." && \
        python -m pytest --cov-report xml:coverage.xml --cov=. && \
        cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l
