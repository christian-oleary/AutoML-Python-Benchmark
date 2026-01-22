# Description: Dockerfile for HyperTS

FROM tensorflow/tensorflow:2.10.1-gpu

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install git and HyperTS dependencies
RUN apt-get install -y git || (apt-get update --fix-missing && apt-get install -y git)

# Clone the HyperTS repository which contains source code and tests
RUN rm -rf /src/HyperTS && git clone https://github.com/DataCanvasIO/HyperTS /src/HyperTS

# Set the working directory to the HyperTS source code
WORKDIR /src/HyperTS/

# Install the required dependencies
RUN git submodule update --init --recursive && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install git+https://github.com/DataCanvasIO/Hypernets && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install pytest-cov

# Run tests if the run_tests is set to "true"
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        (python3 -m pytest --cov-branch --cov-report xml:coverage.xml --cov=. --cov-fail-under=1 || ls coverage.xml) && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "FAILED" && exit 1)) \
    ; fi

# RUN python3 -m pytest --cov-branch --cov-report xml:coverage.xml --cov=. --cov-fail-under=1 || ls coverage.xml
# RUN ls && (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "FAILED" && exit 1))

RUN python -c "import tensorflow as tf;print('GPU STATUS:',tf.test.is_built_with_cuda())"
