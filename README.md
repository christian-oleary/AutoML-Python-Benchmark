# AutoML-Python-Benchmark

Benchmarks of AutoML Frameworks for time series forecasting and anomaly detection.

These experiments are run with Python 3.8 and CUDA versions 11.2 and 11.7.

## CUDA Setup

To run this code, you will need to install CUDA for TensorFlow and PyTorch.

CUDA compatibilities for TensorFlow are listed [here](https://www.tensorflow.org/install/source_windows).

CUDA compatibilities for PyTorch are listed [here](https://pytorch.org/blog/deprecation-cuda-python-support/)

**NOTE: TensorFlow (GPU) on windows only supports CUDA <=11.2 while PyTorch (GPU) requires >=11.3. You will need multiple versions installed.**

The experiment results are based on CUDA 11.2 for TensorFlow and CUDA 11.7 for PyTorch which are officially recommended versions as of the 10th of March 2023.

## Full Installation

The following installation commands support *each* of the tested libraries. If you know which libraries you want to run and are aware of the requirements, then feel free to customize.

Replace pip with pip3 if not using Windows.

Auto-PyTorch requires Linux. PyCaret and EvalML may conflict. Other conflicts are to be expected if you try to install every library.

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install git+https://github.com/keras-team/keras-tuner.git
pip install -r requirements.txt
```

Install the libraries you want to run (they may conflict), e.g.:

```bash
pip install -r ./src/autogluon/requirements.txt
pip install -r ./src/autokeras/requirements.txt
pip install -r ./src/autots/requirements.txt
pip install -r ./src/etna/requirements.txt
pip install -r ./src/evaml/requirements.txt
pip install -r ./src/fedot/requirements.txt
pip install -r ./src/flaml/requirements.txt
pip install -r ./src/ludwig/requirements.txt
pip install -r ./src/pycaret/requirements.txt
```

Auto-PyTorch is Linux only:

```bash
pip3 install -r ./src/autopytorch/requirements.txt
```

## Run experiments

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

## Tests and Linting

After installation and the download of repositories and datasets, you can run functional tests with:

```bash
pip install -r ./tests/requirements.txt
python -m pytest tests/functional_tests.py
```

Linting:

```bash
pip install -r ./tests/requirements.txt
python -m pylint src
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

Please feel free to get in touch at christian.oleary@mtu.ie

## Citation

Christian O'Leary (2023) AutoML Python Benchmark.

```latex
@Misc{AutoML-Python-Benchmark,
author = {Christian O'Leary},
title = {AutoML Python Benchmark},
howpublished = {\url{https://github.com/christian-oleary/AutoML-Python-Benchmark}},
year = {2023}
}
```
