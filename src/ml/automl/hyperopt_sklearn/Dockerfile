# Dockerfile for hyperopt-sklearn

FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Clone the hyperopt-sklearn repository
RUN rm -rf /src/hyperopt_sklearn && git clone https://github.com/hyperopt/hyperopt-sklearn.git /src/hyperopt_sklearn

# Change working directory to the cloned repository
WORKDIR /src/hyperopt_sklearn

# Install Python dependencies
RUN python -m pip install -r requirements.txt
RUN python -m pip install -r requirements_dev.txt
RUN python -m pip install .[test] pytest-cov

# Run tests
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        echo "Running unit tests..." && \
        python3 -m pytest --cov-branch --cov-report xml:coverage.xml --cov=. && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)) && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l
