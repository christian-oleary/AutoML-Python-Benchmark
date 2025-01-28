# Dockerfile for mljar

FROM python:3.9-slim

# Install git
RUN apt-get update && apt-get install -y git libgomp1

# Clone the mljar repository
RUN rm -rf /src/mljar && git clone https://github.com/mljar/mljar-supervised.git /src/mlbox

# Change working directory to mljar
WORKDIR /src/mljar

# Install the required dependencies
RUN pip install -r requirements.txt
RUN pip install -r requirements_dev.txt importlib-metadata>=1.7.0 ipython setuptools
RUN python setup.py install

# Run tests
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        rm -rf coverage.xml && \
        echo "Running unit tests..." && \
        python -m pytest ./test/ --cov-report xml:coverage.xml --cov=. && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)) && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)
RUN python -c "import tensorflow as tf;print('GPU STATUS:',tf.test.is_built_with_cuda())"
RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"
