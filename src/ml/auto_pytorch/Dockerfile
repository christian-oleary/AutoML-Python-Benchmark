# Description: Dockerfile for Auto-PyTorch
FROM automlorg/autopytorch:v0.2

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Check the version of g++ installed
RUN g++ -v

# Install the git and Auto-PyTorch dependencies (ffmpeg, libsm6, libxext6)
RUN apt-get install -y swig3.0 git ffmpeg libsm6 libxext6 || (\
apt-get update --fix-missing && apt-get install -y swig3.0 git ffmpeg libsm6 libxext6)

RUN ln -s /usr/bin/swig3.0 /usr/bin/swig || echo "swig already exists"

# Clone the Auto-PyTorch repository which contains source code and tests
RUN git clone https://github.com/automl/Auto-PyTorch.git /src/auto_pytorch

# Set the working directory to the Auto-PyTorch source code
WORKDIR /src/auto_pytorch/

# Auto-PyTorch requires(?) the submodules to be initialized for tests
RUN git submodule update --init --recursive

# RUN python3 -m pip install swig
# RUN conda --version

# Install the required dependencies and remove any existing test reports
RUN which python3 && pwd && ls && \
    python3 -m pip install -e .[forecasting] swig gluonts==0.10.* numpy==1.20.* torch==1.10.* && \
    rm -f report.xml coverage.xml

# Folders in test dir at 2024/11/25:
# test_data    test_ensemble  test_pipeline
# test_api     test_datasets  test_evaluation  test_utils

# Run tests if the run_tests is set to "true"
ARG run_tests
# Install test dependencies
RUN if [ "$run_tests" = true ]; then python3 -m pip install -e .[test] swig gluonts==0.10.* numpy==1.20.* torch==1.10.8 coverage pytest pytest-cov; fi
# Run unit tests
RUN if [ "$run_tests" = true ]; then python3 -m pytest ./test/test_api/test_base_api.py -sx --cov-report xml:coverage.xml --cov=. --cov-fail-under=1; fi
# Ensure the test reports are generated
RUN if [ "$run_tests" = true ]; then ls coverage.xml && ls report.xml && echo "TESTS COMPLETE"; fi


# OSError: /opt/conda/lib/python3.10/site-packages/torchaudio/lib/libtorchaudio.so: undefined symbol: _ZN2at4_ops10zeros_like4callERKNS_6TensorEN3c108optionalINS5_10ScalarTypeEEENS6_INS5_6LayoutEEENS6_INS5_6DeviceEEENS6_IbEENS6_INS5_12MemoryFormatEEE
