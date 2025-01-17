# Use an official Python runtime as a parent image
# FROM python:3.8-slim
# FROM python:3.9-slim
FROM python:3.10-slim

# Install git and gcc
RUN apt-get update && apt-get install -y build-essential git

# Check the version of g++ installed
RUN g++ -v

# Clone the Kats repository
RUN rm -rf /src/kats && git clone https://github.com/facebookresearch/Kats.git /src/kats

# Change working directory to Kats
WORKDIR /src/kats

# Upgrade pip
RUN python -m pip install --upgrade pip

# Remove test_requirements.txt and requirements.txt. Major dependency issues with current versions
RUN rm -rf test_requirements.txt requirements.txt

# Create requirements.txt with the necessary dependencies
# importlib_metadata; python_version < '3.8' \n\
RUN echo $'\n\
attrs>=21.2.0           \n\
deprecated>=1.2.12      \n\
holidays>=0.10.2,<=0.57 \n\
matplotlib>=2.0.0       \n\
numpy>=1.21,<1.22       \n\
# numpy>=1.21,<1.22     \n\
pandas>=1.0.4,<=1.3.5   \n\
packaging<22            \n\
scikit-learn>=0.24.2    \n\
python-dateutil>=2.8.0  \n\
scipy>=1.10.0           \n\
seaborn>=0.11.1         \n\
setuptools-git>=1.2     \n\
# statsmodels==0.12.2   \n\
statsmodels>=0.12.2     \n\
lightgbm>=2.3.0         \n\
# numba>=0.55.0         \n\
numba>=0.52.0           \n\
typing-extensions       \n\
LunarCalendar>=0.0.9    \n\
ax-platform==0.2.9      \n\
gpytorch==1.9.0         \n\
parameterized>=0.8.1    \n\
plotly>=2.2.1           \n\
pymannkendall>=1.4.1    \n\
pytest-mpl>=0.12,<0.16  \n\
torch                   \n\
tqdm>=4.36.1            \n\
###############         \n\
# Added:                \n\
###############         \n\
Cython>=0.22,<3         \n\
convertdate             \n\
httpstan                \n\
lunarcalendar           \n\
wheel                   \n\
pytest-cov              \n\
coverage                \n\
nevergrad               \n\
####################### \n\
# pystan>=2.19.1.1      \n\
# pystan==2.18.0.0      \n\
# fbprophet             \n\
# fbprophet==0.7.1      \n\
# fbprophet==0.6        \n\
# prophet               \n\
' > requirements.txt

# Remove dollar sign, print requirements and install dependencies
RUN sed -i 's/\$//g'  requirements.txt && \
    cat requirements.txt && \
    python -m pip install -r requirements.txt

# RUN python -m pip install pystan
RUN python -m pip install pystan==2.18.0.0
# RUN python -m pip install pystan==2.19.1.1
RUN python -m pip install fbprophet

RUN echo "" > test_requirements.txt
RUN echo "" > requirements.txt

RUN ls -la
RUN ls -la && python -m pip install -e .

# Run tests
ARG run_tests
RUN if [ "$run_tests" = true ]; then \
        (python -W ignore -m pytest --cov-report xml:coverage.xml --cov=. --cov-fail-under=1 || ls coverage.xml) && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "FAILED" && exit 1)) \
    ; fi

RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"
