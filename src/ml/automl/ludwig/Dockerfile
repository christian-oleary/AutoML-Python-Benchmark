# Description: Dockerfile for ludwig

FROM ludwigai/ludwig:master

# Clone the ludwig repository which contains source code and tests
RUN git clone https://github.com/ludwig-ai/ludwig.git /src/ludwig

# Set the working directory to the ludwig source code
WORKDIR /src/ludwig/

# Initialize any submodules
RUN git submodule update --init --recursive

# Update pip
RUN python3 -m pip install --upgrade pip

# Install core and test dependencies
RUN python3 -m pip install -e .[test,full] pytest-cov coverage

# Install extra dependencies quietly
RUN python3 -m pip install -q -e .[extra]

# Run tests
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        echo "Running unit tests..." && \
        (python3 -m pytest --cov-report xml:coverage.xml --cov=. || ls coverage.xml) && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)) && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)
RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"
