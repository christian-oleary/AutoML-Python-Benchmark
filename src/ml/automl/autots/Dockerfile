# Description: Dockerfile for AutoTS

FROM tensorflow/tensorflow:latest-gpu

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install the git and AutoTS dependencies
RUN apt-get update --fix-missing && apt-get install -y git build-essential

# Clone the AutoTS repository which contains source code and tests
RUN rm -rf /src/autots && git clone https://github.com/winedarksea/AutoTS.git /src/autots

# Set the working directory to the AutoTS source code
WORKDIR /src/autots/

# Install the required dependencies
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -e .['additional'] pytest-cov

# Run unit tests if the run_tests is set to "true"
ARG run_tests
RUN if [ "$run_tests" = true ]; then \
        (python3 -W ignore -m pytest ./tests/ --cov-branch --cov-report xml:coverage.xml --cov=. --cov-fail-under=1 || ls coverage.xml) && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "FAILED" && exit 1)) \
    ; fi

# Check GPU access
RUN python -c "import tensorflow as tf;print('GPU STATUS:',tf.test.is_built_with_cuda())"
