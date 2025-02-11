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

# # Download H2O server jar file
# RUN wget --progress=dot:giga https://h2o-release.s3.amazonaws.com/h2o/rel-3.46.0/6/h2o-3.46.0.6.zip && \
#     unzip h2o-3.46.0.6.zip

# install from wheel
# RUN python3 -m pip install h2o-3.46.0.6/**/h2o_client-3.46.0.6-py2.py3-none-any.whl

# Install required dependencies
RUN python3 -m pip install --upgrade pip && pip install requests tabulate && \
    python3 -m pip uninstall -y h2o && \
    python3 -m pip install -r requirements.txt
    # python3 -m pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o && \

# Fix and install test dependencies
RUN sed -i '/pyarrow==10.0.1; python_version > '\''3.6'\''/d' test-requirements.txt && \
    sed -i '/causalml==0.14.1; python_version > '\''3.7'\'' and python_version < '\''3.11'\''/d' test-requirements.txt && \
    python3 -m pip install -r test-requirements.txt pytest-cov

# # # Download H2O server jar file
# # RUN wget --progress=dot:giga https://h2o-release.s3.amazonaws.com/h2o/rel-3.46.0/6/h2o-3.46.0.6.zip && \
# #     unzip h2o-3.46.0.6.zip

# # RUN ls -la && cp ./h2o-3.46.0.6/h2o.jar ./h2o.jar
# # RUN cp ./h2o-3.46.0.6/h2o.jar ../h2o.jar
# # RUN cp ./h2o-3.46.0.6/h2o.jar ../../h2o.jar
# RUN mkdir -p /src/h2o_3/h2o-py/h2o/backend/bin/
# RUN cp ./h2o-3.46.0.6/h2o.jar /src/h2o_3/h2o-py/h2o/backend/bin/h2o.jar
# # # cd h2o-3.46.0.6 && (java -jar h2o.jar &) && cd ..

# Start the H2O-3 server
# RUN pwd && python3 -c "import h2o; h2o.init()"

# Run tests
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        echo "Running unit tests..." && \
        (python3 -m pytest --cov-report xml:coverage.xml --cov=.) && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)) && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi
        # (python3 -m pytest --cov-report xml:coverage.xml --cov=. || ls coverage.xml) && \

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)
RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"
