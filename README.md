# AutoML-Python-Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellowgreen.svg)](https://opensource.org/licenses/MIT)
[![testing: bandit](https://github.com/christian-oleary/AutoML-Python-Benchmark/actions/workflows/bandit.yml/badge.svg)](https://github.com/christian-oleary/AutoML-Python-Benchmark/actions/workflows/bandit.yml)
[![linting: pylint](https://github.com/christian-oleary/AutoML-Python-Benchmark/actions/workflows/pylint.yml/badge.svg)](https://github.com/christian-oleary/AutoML-Python-Benchmark/actions/workflows/pylint.yml)
[![testing: pytest](https://github.com/christian-oleary/AutoML-Python-Benchmark/actions/workflows/pytest.yml/badge.svg)](https://github.com/christian-oleary/AutoML-Python-Benchmark/actions/workflows/pytest.yml)

Benchmarks of AutoML Frameworks for time series forecasting and anomaly detection.

Python version 3.9.

## Table of Contents

1. [Publications](#publications)
2. [Datasets](#datasets)
3. [CUDA](#cuda)
4. [Installation](#installation)
5. [Experiments](#experiments)
6. [Tests](#tests)
7. [Contact](#contact)
8. [Citation](#citation)

## Publications

A Comparative Analysis of Automated Machine Learning Libraries for Electricity Price Forecasting (2024)

- [Tag](https://github.com/christian-oleary/AutoML-Python-Benchmark/releases/tag/electricity_price_forecasting)
- [Code](https://github.com/christian-oleary/AutoML-Python-Benchmark/tree/c436f3f83e6872ab8a4bb430923fc5aaf64f5ade)
- These experiments are run with Python 3.9 and CUDA versions 11.2 and 11.7.

## Datasets

```bash
./shell/download_univariate_forecasting_dataset.sh
./shell/download_global_forecasting_dataset.sh
./shell/download_anomaly_detection_dataset.sh
```

## CUDA

To run this code, you will need to install CUDA for TensorFlow and PyTorch.

CUDA compatibilities for TensorFlow are listed [here](https://www.tensorflow.org/install/source_windows).

CUDA compatibilities for PyTorch are listed [here](https://pytorch.org/blog/deprecation-cuda-python-support/)

**NOTE: TensorFlow (GPU) on windows only supports CUDA <=11.2 while PyTorch (GPU) requires >=11.3. You will need multiple versions installed.**

The experiment results are based on CUDA 11.2 for TensorFlow and CUDA 11.7 for PyTorch which are officially recommended versions as of the 10th of March 2023.

## Installation

The following installation commands support *each* of the tested libraries. If you know which libraries you want to run and are aware of the requirements, then feel free to customize.

Replace pip with pip3 if not using Windows.

Note: cuDF is supported only on Linux, and with Python versions 3.9 and later.

Auto-PyTorch requires Linux. PyCaret and EvalML may conflict. Other conflicts are to be expected if you try to install every library.

```bash
# AutoML library-specific envs recommended
conda create -y -n env python=3.9
```

Installation for AutoTS specifically:

```bash
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 && \
python -m pip install numpy==1.21 scipy scikit-learn statsmodels lightgbm xgboost numexpr bottleneck yfinance pytrends fredapi plotly sktime==0.18.0 --exists-action i && \
python -m pip install Cython --exists-action i && \
python -m pip install pystan prophet --exists-action i && \
python -m pip install mxnet --no-deps && \
python -m pip install gluonts arch && \
python -m pip install holidays==0.24  holidays-ext pmdarima dill greykite --exists-action i --no-deps && \
python -m pip install holidays==0.24 prophet==1.1.3 cvxpy neuralprophet pytorch-forecasting && \
python -m pip install pandas --exists-action i && \
python -m pip install numpy==1.21 i && \
python -m pip install tensorflow==2.10.0 i && \
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 && \
python -m pip install autots --exists-action i
```

Installation for Auto-PyTorch specifically. Auto-PyTorch is Linux only:

```bash
# For CUDA setup in WSL:
# wget https://developer.download.nvidia.com/compute/cuda/12.2.1/local_installers/cuda_12.2.1_535.86.10_linux.run
# sudo sh cuda_12.2.1_535.86.10_linux.run

sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y
sudo add-apt-repository multiverse
sudo apt update
sudo apt install nvidia-cuda-toolkit
conda config --append channels conda-forge
conda install -c conda-forge pytorch # can also try: conda install -c conda-forge torch
pip install --force-reinstall charset-normalizer==3.1.0
pip install -r requirements.txt
pip install -r ./src/autopytorch/requirements.txt
```

Initial steps for AutoGluon:

<!-- # Conda does not support PyTorch installation for AutoGluon with GPU support
conda install -y -c conda-forge mamba
mamba install -y -c conda-forge autogluon
-->
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r ./src/autogluon/requirements.txt
pip install tensorflow # Only used in tests
```

Note: AutoGluon does not work with PyTorch 2.* yet: <https://github.com/autogluon/autogluon/issues/3250>

For all libraries except AutoGluon, AutoTS and AutoPyTorch:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install git+https://github.com/keras-team/keras-tuner.git
pip install -r requirements.txt
```

Install the libraries you want to run (they may conflict). One per environment is recommended.

```bash
pip install -r ./src/autokeras/requirements.txt
pip install -r ./src/etna/requirements.txt && pip install etna[prophet]
pip install -r ./src/evalml/requirements.txt
pip install -r ./src/fedot/requirements.txt
pip install -r ./src/flaml/requirements.txt && pip uninstall prophet # Prophet gives errors with FLAML
pip install -r ./src/ludwig/requirements.txt
pip install -r ./src/pycaret/requirements.txt
```

If you have trouble installing PyTorch, you can try building from source: <https://github.com/pytorch/pytorch#from-source>

## Experiments

Before running the code, datasets and repositories must be downloaded

```bash
sh ./shell/download_anomaly_datasets.sh # download anomaly detection datasets
sh ./shell/download_forecasting_datasets.sh # download forecasting datasets
sh ./shell/line_counts.sh # download repositories and count lines of code
```

After downloading repositories and datasets, you can run experiments with the following:

```bash
python run.py
```

## Tests

After installation and the download of repositories and datasets, you can run functional tests with:

```bash
pip install -r ./tests/requirements.txt
python -m pytest tests/functional_tests.py
```

Linting:

```bash
pip install -r ./tests/requirements.txt
python -m pylint src
python -m pylint src --disable=all --enable=W0102
```

Check if TensorFlow/PyTorch can access GPUs:

```bash
python ./tests/gpu_test.py
```

Profiling:

```bash
python -m cProfile -s time run.py > profile_verbose.txt
cat profile_verbose.txt | grep -e dataset_formatting.py -e forecasting.py -e util.py -e cumtime | grep -v "(<" > profile_summary.txt
```

Coverage report:

```bash
pip install -r ./tests/requirements.txt
coverage run -m pytest tests/functional_tests.py
coverage report --omit="env/*,venv/*,.env/*,.venv/*,*AppData*,*python37*,tests/*"
rm .coverage
```

## Contact

Please feel free to get in touch at <christian.oleary@mtu.ie>

## Citation

Christian O'Leary (2024) AutoML Python Benchmark.

```latex
@Misc{AutoML-Python-Benchmark,
author = {Christian O'Leary},
title = {AutoML Python Benchmark},
howpublished = {\url{https://github.com/christian-oleary/AutoML-Python-Benchmark}},
year = {2024}
}
```
