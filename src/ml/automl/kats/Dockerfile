# Use an official Python runtime as a parent image
# FROM python:3.8-slim
# FROM python:3.9-slim
FROM python:3.10-slim

# Install git and gcc. Clone repository
RUN apt-get update && \
    apt-get install -y build-essential git gcc g++ && \
    rm -rf /src/kats && \
    git clone https://github.com/facebookresearch/Kats.git /src/kats

# Change working directory to Kats
WORKDIR /src/kats

# Upgrade pip
RUN python -m pip install --upgrade pip
# RUN python -m pip install -r requirements.txt  # Fails
# RUN python -m pip install -r test_requirements.txt  # Fails

# RUN python -m pip install Cython==0.21
# RUN python -m pip install numpy>=1.21,<1.22
RUN python -m pip install Cython numpy==1.*
RUN python -m pip install attrs>=21.2.0
RUN python -m pip install deprecated>=1.2.12
RUN python -m pip install importlib_metadata

# RUN python -m pip install holidays<=0.57
# RUN python -m pip install matplotlib>=2.0.0
# RUN python -m pip install pandas>=1.0.4,<=1.3.5
RUN python -m pip install packaging==21
RUN python -m pip install python-dateutil>=2.8.0
# RUN python -m pip install scikit-learn==0.24.2
RUN python -m pip install scipy>=1.10.0
RUN python -m pip install seaborn>=0.11.1
RUN python -m pip install setuptools-git>=1.2
# RUN python -m pip install statsmodels==0.12.2
RUN python -m pip install lightgbm>=2.3.0
RUN python -m pip install numba>=0.55.0
RUN python -m pip install typing-extensions
RUN python -m pip install pandas
RUN python -m pip install matplotlib
RUN python -m pip install statsmodels

RUN python -m pip install LunarCalendar>=0.0.9

# RUN python -m pip install ax-platform==0.2.9
RUN python -m pip install prophet
# RUN python -m pip install gpytorch==1.9.0
RUN python -m pip install holidays>=0.10.2
RUN python -m pip install parameterized>=0.8.1
RUN python -m pip install plotly>=2.2.1
RUN python -m pip install pymannkendall>=1.4.1
RUN python -m pip install pystan==2.19.1.1
RUN python -m pip install pytest-mpl>=0.12
RUN python -m pip install torch
RUN python -m pip install tqdm>=4.36.1
RUN python -m pip install pytest-cov
RUN python -m pip install scikit-learn==0.24.2
RUN python -m pip install setuptools

# RUN rm -rf test_requirements.txt requirements.txt && \
#     echo "" > test_requirements.txt && \
#     echo "" > requirements.txt && \
#     python -m pip install -e .

RUN python -m pip install --no-deps kats
RUN python -c "import kats"

# Run tests
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        echo "Running unit tests..." && \
        python -m pytest --cov-branch --cov-report xml:coverage.xml --cov=. && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)) && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)
RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"
