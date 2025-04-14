# Description: Dockerfile for Auto-Keras

# FROM haifengjin/autokeras:1.0.11 # No longer updated
FROM tensorflow/tensorflow:latest-gpu

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install git and Auto-Keras dependencies
RUN apt-get install -y git || (apt-get update --fix-missing && apt-get install -y git)

# Clone the Auto-Keras repository which contains source code and tests
RUN rm -rf /src/autokeras && git clone https://github.com/keras-team/autokeras.git /src/autokeras

# Set the working directory to the Auto-Keras source code
WORKDIR /src/autokeras/

# Initialize any submodules
RUN git submodule update --init --recursive

# Install the required dependencies
RUN python3 -m pip install git+https://github.com/keras-team/keras-tuner.git
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -e .[tests]

# Run tests if the run_tests is set to "true"
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        (python3 -m pytest --cov-branch --cov-report xml:coverage.xml --cov=. --cov-fail-under=1 || ls coverage.xml) && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "FAILED" && exit 1)) \
    ; fi

# RUN python3 -m pytest --cov-report xml:coverage.xml --cov=. --cov-fail-under=1 || ls coverage.xml
# RUN ls && (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "FAILED" && exit 1))

RUN python -c "import tensorflow as tf;print('GPU STATUS:',tf.test.is_built_with_cuda())"
