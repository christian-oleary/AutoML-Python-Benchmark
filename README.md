# AutoML-Python-Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellowgreen.svg)](https://opensource.org/licenses/MIT)
[![testing: bandit](https://github.com/christian-oleary/AutoML-Python-Benchmark/actions/workflows/bandit.yml/badge.svg)](https://github.com/christian-oleary/AutoML-Python-Benchmark/actions/workflows/bandit.yml)
[![linting: pylint](https://github.com/christian-oleary/AutoML-Python-Benchmark/actions/workflows/pylint.yml/badge.svg)](https://github.com/christian-oleary/AutoML-Python-Benchmark/actions/workflows/pylint.yml)
[![testing: pytest](https://github.com/christian-oleary/AutoML-Python-Benchmark/actions/workflows/pytest.yml/badge.svg)](https://github.com/christian-oleary/AutoML-Python-Benchmark/actions/workflows/pytest.yml)

Benchmarks of AutoML Frameworks for time series forecasting, anomaly detection and classification.

Primary Python version: 3.10.14

## Table of Contents

1. [Publications](#publications)
2. [Datasets](#datasets)
3. [CUDA](#cuda)
4. [Installation](#installation)
5. [Experiments](#experiments)
6. [Development](#development)
7. [Contact](#contact)
8. [Citation](#citation)

## Publications

A Comparative Analysis of Automated Machine Learning Libraries for Electricity Price Forecasting (2024)

- [Tag](https://github.com/christian-oleary/AutoML-Python-Benchmark/releases/tag/electricity_price_forecasting)
- [Code](https://github.com/christian-oleary/AutoML-Python-Benchmark/tree/c436f3f83e6872ab8a4bb430923fc5aaf64f5ade)
- These experiments are run with Python 3.9 and CUDA versions 11.2 and 11.7.



## Installation

Removed. To be redrafted.

### CUDA

To run this code, you will need to install CUDA for TensorFlow and PyTorch.

- CUDA compatibilities for TensorFlow are listed [here](https://www.tensorflow.org/install/source_windows).
- CUDA compatibilities for PyTorch are listed [here](https://pytorch.org/blog/deprecation-cuda-python-support/)

To be redrafted.

## Datasets

Removed. To be redrafted.

Before running the code, datasets and repositories must be downloaded

## Experiments

After downloading repositories and datasets, you can run experiments with the following:

```bash
python run.py
```

## Development

Removed. To be redrafted.

After installation and the download of repositories and datasets, you can run functional tests with:

<!--
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

SonarQube:

```bash
# 1. Run server
docker run -d --name sonarqube -e SONAR_ES_BOOTSTRAP_CHECKS_DISABLE=true -p 9000:9000 sonarqube:latest
# 2. (Create a project and token on the server at http://localhost:9000)
# 3. Run scanner
sonar-scanner -D"sonar.projectKey=PROJECT_NAME" -D"sonar.sources=." -D"sonar.host.url=http://localhost:9000" -D"sonar.token=GENERATED_TOKEN"
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
``` -->

## Contact

Please feel free to get in touch at <christian.oleary@mtu.ie>

## Citation

Christian O'Leary (2024) AutoML Python Benchmark.

```latex
@Misc{AutoML-Python-Benchmark,
author = {Christian O'Leary},
title = {AutoML Python Benchmark},
doi = {10.5281/zenodo.13133203},
howpublished = {\url{https://github.com/christian-oleary/AutoML-Python-Benchmark}},
year = {2024}
}
```
