# AutoML-Python-Benchmark

Benchmarks of AutoML Frameworks.

## Downloading Repositories and Datasets

```bash
sh ./shell/download_anomaly_datasets.sh # download anomaly detection datasets
sh ./shell/download_forecasting_datasets.sh # download forecasting datasets
sh ./shell/line_counts.sh # download repositories and count lines of code
```

## Tests and Linting

Download repositories and datasets first. Then run tests with:

```bash
pip install -r ./tests/requirements.txt
python -m pytest tests
```

Linting:

```bash
pip install -r ./tests/requirements.txt
python -m pylint src
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
