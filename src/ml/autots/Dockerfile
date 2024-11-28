# Description: Dockerfile for AutoTS

FROM tensorflow/tensorflow:latest-gpu

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install the git and AutoTS dependencies
RUN apt-get install -y git || (apt-get update --fix-missing && apt-get install -y git)

# Clone the AutoTS repository which contains source code and tests
RUN rm -rf /src/autots && git clone https://github.com/winedarksea/AutoTS.git /src/autots

# Set the working directory to the AutoTS source code
WORKDIR /src/autots/

# Install the required dependencies
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -e .['additional']

# Run unit tests if the run_tests is set to "true"
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        echo "Installing test dependencies..." && \
        python3 -m pip install -e .['additional'] coverage pytest pytest-cov && \
        echo "Running unit tests..." && \
        python3 -m pytest ./test/ --cov-report xml:coverage.xml --cov=. && \
        ls coverage.xml && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi
    # || ls coverage.xml
# Check GPU access
RUN python -c "import tensorflow as tf;print('GPU STATUS:',tf.test.is_built_with_cuda())"
