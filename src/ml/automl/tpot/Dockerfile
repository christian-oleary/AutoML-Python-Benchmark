# FROM python:3.9-slim
FROM python:3.10-slim

# Install git
# RUN apt-get update && apt-get install -y git

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*


# Clone the TPOT repository
RUN rm -rf /src/tpot && git clone https://github.com/EpistasisLab/tpot /src/tpot

# Change working directory to TPOT
WORKDIR /src/tpot

# Install the required dependencies
# RUN pip install -r requirements.txt && pip install -r optional-requirements.txt && pip install . pytest-cov
# [UPDATED April 2025]
# RUN pip install .[extra] pytest-cov && pip install -r requirements_dev.txt
# [UPDATED January 2026]
RUN pip install .[skrebate,mdr,sklearnex,amltk,testing]

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

# RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l
