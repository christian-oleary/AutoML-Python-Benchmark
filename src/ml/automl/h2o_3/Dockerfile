# Description: Dockerfile for H2O-3

FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# Install git and AutoGluon dependencies
RUN apt-get update --fix-missing && apt-get install -y git libmysqlclient-dev wget unzip openjdk-8-jdk

# Clone the H2O-3 repository which contains source code and tests
RUN git clone https://github.com/h2oai/h2o-3.git /src/h2o_3

# Set the working directory to the H2O-3 source code
WORKDIR /src/h2o_3/h2o-py/

# Initialize any submodules
RUN git submodule update --init --recursive

RUN python3 -m pip install --upgrade pip && pip install requests tabulate

# Install required dependencies
RUN python3 -m pip install --upgrade pip && pip install requests tabulate && \
    python3 -m pip uninstall -y h2o && \
    python3 -m pip install -r requirements.txt
    # python3 -m pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o && \

# Fix and install test dependencies
RUN sed -i '/pyarrow==10.0.1; python_version > '\''3.6'\''/d' test-requirements.txt && \
    sed -i '/causalml==0.14.1; python_version > '\''3.7'\'' and python_version < '\''3.11'\''/d' test-requirements.txt && \
    python3 -m pip install -r test-requirements.txt pytest-cov

RUN echo "3.46.0.6" > "/src/h2o_3/h2o-py/h2o/version.txt" && \
    echo ""        >> "/src/h2o_3/h2o-py/h2o/version.txt" && \
    python3 -m pip install -e .

RUN python3 -m pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o

RUN python3 ./scripts/verify_requirements.py --metayaml conda/h2o-main/meta.yaml

ADD h2o.jar /opt/h2o.jar
ADD h2o.jar /opt/conda/h2o_jar/h2o.jar

# RUN python3 -c "import h2o;h2o.init(strict_version_check=False)"
# RUN curl localhost:54321
# RUN python3 -c "import h2o;h2o.init(strict_version_check=False)" && \
    # python3 -m pytest tests --cov-report xml:coverage.xml --cov=.
# RUN cd tests && python3 -m pytest --cov-report xml:coverage.xml --cov=. && cd -

# Run tests
ARG run_tests
# RUN python3 -m pytest tests --cov-report xml:coverage.xml --cov=.
RUN python3 -m pytest --cov-report xml:coverage.xml --cov=. || ls coverage.xml

RUN if [ "$run_tests" = "true" ]; then \
        echo "Running unit tests..." && \
        (python3 -m pytest --cov-report xml:coverage.xml --cov=.) && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)) && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)
RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"
