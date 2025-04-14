# Description: Dockerfile for ludwig

FROM ludwigai/ludwig:master

# Clone the ludwig repository which contains source code and tests
RUN git clone https://github.com/ludwig-ai/ludwig.git /src/ludwig

# Set the working directory to the ludwig source code
WORKDIR /src/ludwig/

# Initialize any submodules
RUN git submodule update --init --recursive && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -e .[test,full] pytest-cov coverage && \
    python3 -m pip install -q -e .[extra]

# Run tests
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        echo "Running unit tests..." && \
        (python3 -m pytest --cov-branch --cov-report xml:coverage.xml --cov=. || ls coverage.xml) && \
        cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l
RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"

# Override the entrypoint which is set in the base image to "ludwig"
ENTRYPOINT []
