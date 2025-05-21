# Description: Dockerfile for PyCaret

# Adapted from: https://github.com/pycaret/pycaret/blob/master/Docker_files/pycaret_full/Dockerfile

# FROM pycaret/full:latest  # Python version too old
# FROM jupyter/base-notebook:latest  # No need for Jupyter

# Using the PyTorch image instead
FROM pytorch/pytorch:latest
# FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

RUN uname -r && echo $CUDA_HOME && which conda

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# # Set user as root for installing dependencies
# USER root

# Install the git
RUN apt-get update --fix-missing && apt-get install -y git

# name your environment and choose python 3.x version
ARG conda_env=pycaret_full
ARG py_ver=3.11

# # Add additional libraries with mamba
# RUN mamba create --quiet --yes -p "${CONDA_DIR}/envs/${conda_env}" python=${py_ver} ipython ipykernel && \
#     mamba clean --all -f -y

# # Create Python 3.x environment
# RUN "${CONDA_DIR}/envs/${conda_env}/bin/python" install --user --name="${conda_env}" && \
#     fix-permissions "${CONDA_DIR}"

# # create Python 3.x environment and link it to jupyter
# RUN "${CONDA_DIR}/envs/${conda_env}/bin/python" -m ipykernel install --user --name="${conda_env}" && \
#     fix-permissions "${CONDA_DIR}" && \
#     fix-permissions "/home/${NB_USER}"

RUN conda create --quiet --yes -p "${CONDA_DIR}/envs/${conda_env}" python=${py_ver} -y && \
    conda clean --all -f -y

# Install pycaret full version
RUN "${CONDA_DIR}/envs/${conda_env}/bin/pip" install pycaret[full]>=3.3.2

# Prepend conda environment to path
ENV PATH "${CONDA_DIR}/envs/${conda_env}/bin:${PATH}"

# make the env default
ENV CONDA_DEFAULT_ENV ${conda_env}

# Clone the PyCaret repository which contains source code and tests
RUN rm -rf /src/pycaret
RUN git clone https://github.com/pycaret/pycaret.git /src/pycaret

# Set the working directory to the PyCaret source code
WORKDIR /src/pycaret/

# Initialize any submodules
RUN git submodule update --init --recursive

# # Change the owner of the PyCaret source code to the non-root user
# RUN chown 1000 /src/pycaret
# USER 1000

# Install the required dependencies
# RUN conda init /bin/bash
# RUN conda activate pycaret_full
# RUN python3 -m pip install --upgrade pip
# Need to upgrade Python to prevent: "No matching distribution found for pandas>=2.2.0"
RUN python3 -m pip --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org install pandas>=2.2.0
RUN python3 --version && python3 -m pip install pandas==2.2.0
RUN python3 -m pip install -e .[full] pytest-cov

# Run tests
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        rm -rf coverage.xml && \
        echo "Running unit tests..." && \
        (python3 -m pytest --cov-branch --cov-report xml:coverage.xml --cov=. || ls coverage.xml)&& \
        cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l
# RUN python -c "import tensorflow as tf;print('GPU STATUS:',tf.test.is_built_with_cuda())"
# RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"
