FROM python:3.9-slim

# Install git
RUN apt-get update && apt-get install -y git libgomp1

# Clone the MLBox repository
RUN rm -rf /src/mlbox && git clone https://github.com/mljar/mljar-supervised.git /src/mlbox

# Change working directory to MLBox
WORKDIR /src/mlbox

# Install the required dependencies
RUN pip install -r requirements.txt
RUN pip install . pytest-cov

# Run tests
ARG run_tests
RUN if [ "$run_tests" = true ]; then \
        python3 -m pytest --cov-report xml:coverage.xml --cov=. --cov-fail-under=1 \
    ; fi

# Ensure the coverage.xml file is created and some lines are covered
RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l
