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
RUN if [ "$run_tests" = "true" ]; then \
        rm -rf coverage.xml && \
        echo "Running unit tests..." && \
        (pytest --cov-report xml:coverage.xml --cov=. || ls coverage.xml) && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)) && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)
RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"
