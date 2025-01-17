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

# Install the required dependencies
RUN python3 -m pip install .

RUN python3 -m pytest --cov-report xml:coverage.xml --cov=. --cov-fail-under=1

# Ensure the coverage.xml file is created and some lines are covered
RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l

RUN python -c "import tensorflow as tf;print('GPU STATUS:',tf.test.is_built_with_cuda())"
