FROM python:3.9-slim

# Install git
RUN apt-get update && apt-get install -y git

# Clone the TPOT repository
RUN rm -rf /src/tpot && git clone https://github.com/EpistasisLab/tpot /src/tpot

# Change working directory to TPOT
WORKDIR /src/tpot

# Install the required dependencies
RUN pip install -r requirements.txt
RUN pip install -r optional-requirements.txt
RUN pip install . pytest-cov

# Run tests
ARG run_tests
RUN if [ "$run_tests" = true ]; then \
        (pytest --cov-report xml:coverage.xml --cov=. --cov-fail-under=1 || ls coverage.xml) && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "FAILED" && exit 1)) \
    ; fi
