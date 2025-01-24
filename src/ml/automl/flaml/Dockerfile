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

# Setup user to not run as root
RUN adduser --disabled-password --gecos '' flaml-dev
RUN adduser flaml-dev sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER flaml-dev

# Pull repository
RUN rm -rf /src/flaml/ && git clone https://github.com/microsoft/FLAML.git /src/flaml/
WORKDIR /src/flaml/

# Install FLAML
RUN echo "01" && sudo pip install -e .[test]
RUN echo "02" && sudo pip install -e .[test,automl]
RUN echo "03" && sudo pip install -e .[test,automl,spark]
RUN echo "04" && sudo pip install -e .[test,automl,spark,catboost]
RUN echo "05" && sudo pip install -e .[test,automl,spark,catboost,blendsearch]
RUN echo "06" && sudo pip install -e .[test,automl,spark,catboost,blendsearch,ray]
RUN echo "07" && sudo pip install -e .[test,automl,spark,catboost,blendsearch,ray,azureml]
RUN echo "08" && sudo pip install -e .[test,automl,spark,catboost,blendsearch,ray,azureml,nni]
RUN echo "09" && sudo pip install -e .[test,automl,spark,catboost,blendsearch,ray,azureml,nni,vw]
RUN echo "10" && sudo pip install -e .[test,automl,spark,catboost,blendsearch,ray,azureml,nni,vw,hf]
RUN echo "11" && sudo pip install -e .[test,automl,spark,catboost,blendsearch,ray,azureml,nni,vw,hf,ts_forecast]
RUN echo "12" && sudo pip install -e .[test,automl,spark,catboost,blendsearch,ray,azureml,nni,vw,hf,ts_forecast,forecast]
RUN echo "13" && sudo pip install -e .[test,automl,spark,catboost,blendsearch,ray,azureml,nni,vw,hf,ts_forecast,forecast,benchmark]
RUN echo "14" && sudo pip install -e .[test,automl,spark,catboost,blendsearch,ray,azureml,nni,vw,hf,ts_forecast,forecast,benchmark,openai]
RUN echo "15" && sudo pip install -e .[test,automl,spark,catboost,blendsearch,ray,azureml,nni,vw,hf,ts_forecast,forecast,benchmark,openai,autogen]
RUN echo "16" && sudo pip install -e .[test,automl,spark,catboost,blendsearch,ray,azureml,nni,vw,hf,ts_forecast,forecast,benchmark,openai,autogen,mathchat]
RUN echo "17" && sudo pip install -e .[test,automl,spark,catboost,blendsearch,ray,azureml,nni,vw,hf,ts_forecast,forecast,benchmark,openai,autogen,mathchat,retrievechat]
RUN echo "18" && sudo pip install -e .[test,automl,spark,catboost,blendsearch,ray,azureml,nni,vw,hf,ts_forecast,forecast,benchmark,openai,autogen,mathchat,retrievechat,synapse]
RUN echo "19" && sudo pip install -e .[test,automl,spark,catboost,blendsearch,ray,azureml,nni,vw,hf,ts_forecast,forecast,benchmark,openai,autogen,mathchat,retrievechat,synapse,autozero]

# "automl": [ "lightgbm>=2.3.1", "xgboost>=0.90,<3.0.0", "scipy>=1.4.1", "pandas>=1.1.4", "scikit-learn>=1.0.0", ],
# "spark": [ "pyspark>=3.2.0", "joblibspark>=0.5.0", "joblib<=1.3.2", ],
# "catboost": [ "catboost>=0.26,<1.2; python_version<'3.11'", "catboost>=0.26,<=1.2.5; python_version>='3.11'" ],
# "blendsearch": [ "optuna>=2.8.0,<=3.6.1", "packaging", ],
# "ray": [ "ray[tune]~=1.13", ],
# "azureml": [ "azureml-mlflow", ],
# "nni": [ "nni",  ],
# "vw": [ "vowpalwabbit>=8.10.0, <9.0.0", "scikit-learn", ],
# "hf": [ "transformers[torch]==4.26", "datasets", "nltk<=3.8.1", "rouge_score", "seqeval", ],
# "ts_forecast": [ "holidays<0.14", "prophet>=1.0.1", "statsmodels>=0.12.2", "hcrystalball==0.1.10", ],
# "forecast": [
#     "holidays<0.14", "prophet>=1.0.1", "statsmodels>=0.12.2", "hcrystalball==0.1.10",
#     "pytorch-forecasting>=0.9.0; python_version<'3.11'",
#     # "pytorch-forecasting==0.10.1; python_version=='3.11'",
#     "pytorch-lightning==1.9.0", "tensorboardX==2.6",
# ],
# "benchmark": ["catboost>=0.26", "psutil==5.8.0", "xgboost==1.3.3", "pandas==1.1.4"],
# "openai": ["openai==0.27.8", "diskcache"],
# "autogen": ["openai==0.27.8", "diskcache", "termcolor"],
# "mathchat": ["openai==0.27.8", "diskcache", "termcolor", "sympy", "pydantic==1.10.9", "wolframalpha"],
# "retrievechat": [ "openai==0.27.8", "diskcache", "termcolor", "chromadb", "tiktoken", "sentence_transformers", ],
# "synapse": [ "joblibspark>=0.5.0", "optuna>=2.8.0,<=3.6.1", "pyspark>=3.2.0", ],
# "autozero": ["scikit-learn", "pandas", "packaging"],

# Run tests
ARG run_tests
RUN if [ "$run_tests" = "true" ]; then \
        echo "Running unit tests..." && \
        python -m pytest ./tests/ --cov-report xml:coverage.xml --cov=. --cov-fail-under=1 && \
        ls coverage.xml && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)) && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi

RUN cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)
RUN python -c "import tensorflow as tf;print('GPU STATUS:',tf.test.is_built_with_cuda())"
RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"
