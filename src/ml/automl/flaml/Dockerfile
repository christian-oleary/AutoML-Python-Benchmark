# Dockerfile for FLAML development environment
# Adapted from https://github.com/microsoft/FLAML/blob/main/Dockerfile

# basic setup
FROM mcr.microsoft.com/devcontainers/python:3.8

RUN apt-get update && apt-get -y update
RUN apt-get install -y sudo git npm

# Install Spark
RUN sudo apt-get update && sudo apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        ca-certificates-java ca-certificates openjdk-17-jdk-headless \
        wget \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*
RUN wget --progress=dot:giga "https://www.apache.org/dyn/closer.lua/spark/spark-3.3.0/spark-3.3.0-bin-hadoop2.tgz?action=download" -O - | tar -xzC /tmp; archive=$(basename "spark-3.3.0/spark-3.3.0-bin-hadoop2.tgz") bash -c "sudo mv -v /tmp/\${archive/%.tgz/} /spark"
ENV SPARK_HOME=/spark \
    PYTHONPATH=/spark/python/lib/py4j-0.10.9.5-src.zip:/spark/python
ENV PATH="${PATH}:${SPARK_HOME}/bin"

# # Setup user to not run as root
# RUN adduser --disabled-password --gecos '' flaml-dev
# RUN adduser flaml-dev sudo
# RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# USER flaml-dev

# Pull repository
RUN rm -rf /src/flaml/ && git clone https://github.com/microsoft/FLAML.git /src/flaml/
WORKDIR /src/flaml/

# Install FLAML
# not "benchmark"
RUN pip install -e .[test,automl,spark,catboost,blendsearch,ray,azureml,nni,vw,hf,ts_forecast,forecast,openai,autogen,mathchat,retrievechat,synapse,autozero] pytest-cov coverage

# Run tests
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        rm -rf coverage.xml && \
        echo "Running unit tests..." && \
        (python -m pytest ./test/ --cov-branch --cov-report xml:coverage.xml --cov=. || ls coverage.xml) && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)) && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)
RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"
