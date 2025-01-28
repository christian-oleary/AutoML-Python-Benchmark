# Dockerfile for MLBox

FROM python:3.7-slim

# Disable interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install git
RUN apt-get update --fix-missing && apt-get install -y git libgomp1

# Clone the MLBox repository
RUN rm -rf /src/mlbox && git clone https://github.com/AxeldeRomblay/MLBox.git /src/mlbox

# Change working directory to MLBox
WORKDIR /src/mlbox

# Install the required dependencies
RUN python -m pip install --upgrade pip && \
    python -m pip install setuptools wheel

RUN python -m pip install Cython==0.29.37 && \
    python -m pip install numpy==1.18.2 --no-build-isolation

RUN python -m pip install -r requirements.txt

RUN python -m pip install . pytest-cov protobuf==3.20.*

# Run tests
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        echo "Running unit tests..." && \
        rm -rf coverage.xml && \
        cd tests && \
        rm -rf coverage.xml && \
        python -m pytest --cov-report xml:coverage.xml --cov=. && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)) && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)
RUN python -c "import tensorflow as tf;print('GPU STATUS:',tf.test.is_built_with_cuda())"
