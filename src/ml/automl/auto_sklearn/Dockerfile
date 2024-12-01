# Description: Dockerfile for auto-sklearn

FROM mfeurer/auto-sklearn:master

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# # Install the git and Auto-Sklearn dependencies
# RUN apt-get install -y git || (apt-get update --fix-missing && apt-get install -y git)

# Clone the Auto-Sklearn repository which contains source code and tests
RUN git clone https://github.com/automl/auto-sklearn.git /src/auto_sklearn

# Set the working directory to the Auto-Sklearn source code
WORKDIR /src/auto_sklearn/

# AutoGluon requires the submodules to be initialized for tests
RUN git submodule update --init --recursive

# Install the required dependencies
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -e .[test,examples]

# Run tests
ARG run_tests
RUN if [ "$run_tests" = true ]; then \
    python3 -m pytest --cov-report xml:coverage.xml --cov=. --cov-fail-under=1; fi

# Ensure the coverage.xml file is created and some lines are covered
RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l
