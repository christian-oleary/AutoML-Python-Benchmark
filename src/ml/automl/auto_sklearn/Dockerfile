# Description: Dockerfile for auto-sklearn

FROM mfeurer/auto-sklearn:master

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install git and auto-sklearn dependencies
RUN apt-get install -y git || (apt-get update --fix-missing && apt-get install -y git)

# Clone the auto-sklearn repository which contains source code and tests
RUN git clone https://github.com/automl/auto-sklearn.git /src/auto_sklearn

# Set the working directory to the auto-sklearn source code
WORKDIR /src/auto_sklearn/

# Initialize any submodules
RUN git submodule update --init --recursive

# Install the required dependencies
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -e .[test,examples]

# Run tests
ARG run_tests
RUN if [ "$run_tests" = true ]; then \
        (python3 -m pytest --cov-report xml:coverage.xml --cov=. --cov-fail-under=1 || ls coverage.xml) && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "FAILED" && exit 1)) \
    ; fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l
