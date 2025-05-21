# FROM python:3.9-slim
FROM python:3.10-slim

# Install git
RUN apt-get update && apt-get install -y git

# Clone the TPOT repository
RUN rm -rf /src/tpot && git clone https://github.com/EpistasisLab/tpot /src/tpot

# Change working directory to TPOT
WORKDIR /src/tpot

# [OLD] Install the required dependencies
# RUN pip install -r requirements.txt
# RUN pip install -r optional-requirements.txt
# RUN pip install . pytest-cov

# [UPDATED April 2025] Install the required dependencies
RUN pip install .
RUN pip install .[extra]
RUN pip install .[extra] pytest-cov
RUN pip install -r requirements_dev.txt

# Run tests
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        rm -rf coverage.xml && \
        echo "Running unit tests..." && \
        (pytest --cov-branch --cov-report xml:coverage.xml --cov=. || ls coverage.xml) && \
        cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l
