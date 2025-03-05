# Description: Dockerfile for PyODDS

FROM tensorflow/tensorflow:latest-gpu

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install git and PyODDS dependencies
RUN apt-get update --fix-missing && apt-get install -y git openjdk-8-jdk

# Clone the PyODDS repository which contains source code and tests
RUN rm -rf /src/pyodds && git clone https://github.com/datamllab/pyodds.git /src/pyodds

# Set the working directory to the PyODDS source code
WORKDIR /src/pyodds/

# Initialize any submodules
RUN git submodule update --init --recursive

# Replace "tensorflow==2.0.0b1" with "tensorflow" in requirements.txt
RUN sed -i 's/tensorflow==2.0.0b1/tensorflow/g' requirements.txt

# Install the required dependencies
RUN python3 -m pip install .

RUN python3 -m pip install pytest-cov taos

# Run tests
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        rm -rf coverage.xml && \
        echo "Running unit tests..." && \
        python3 -m pytest --cov-report xml:coverage.xml --cov=. && \
        cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l
RUN python -c "import tensorflow as tf;print('GPU STATUS:',tf.test.is_built_with_cuda())"
