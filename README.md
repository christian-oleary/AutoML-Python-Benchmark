# AutoML-Python-Benchmark

Benchmarks of AutoML Frameworks for time series forecasting and anomaly detection.

Perform CUDA setup, then install using requirements.txt

## CUDA Setup

To run this code, you will need to install CUDA for TensorFlow and PyTorch.

CUDA compatibilities for TensorFlow are listed [here](https://www.tensorflow.org/install/source_windows).

CUDA compatibilities for PyTorch are listed [here](https://pytorch.org/blog/deprecation-cuda-python-support/)

**NOTE: TensorFlow (GPU) on windows only supports CUDA <=11.2 while PyTorch (GPU) requires >=11.3. You will need multiple versions installed.**

## Downloading Repositories and Datasets

```bash
sh ./shell/download_anomaly_datasets.sh # download anomaly detection datasets
sh ./shell/download_forecasting_datasets.sh # download forecasting datasets
sh ./shell/line_counts.sh # download repositories and count lines of code
```

## Run experiments

After downloading repositories and datasets, you can run experiments with the following:

```bash
pip install -r requirements.txt
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
