# AutoML Library Benchmark Code

This folder contains code for benchmarking AutoML libraries for time series forecasting and anomaly detection.

The `src/ml/ad` folder contains code for anomaly detection models located in `anomaly_detection.py`. The `src/ml/sca` folder contains code for source code analysis of AutoML repositories via Sonar Scanner. The `src/ml/archived` folder contains code for the electricity price forecasting benchmark and the anomaly detection benchmark, which are now archived. Some of the Python files in the `src/ml` folder are no longer used, but are kept for reference.

```txt
src/ml/
│
├── ad/
│   └── anomaly_detection.py (AD models)
│
├── archived/
│   ├── automl                        (can be ignored)
│   ├── electricity_price_forecasting (forecasting benchmark)
│   └── TSB-AD                        (anomaly detection benchmark)
│
├── automl/ (contains Dockerfiles)
│
├── sca/             (source code analysis)
│   └── __main__.py  (entry point)
|   └── analysis.py  (analyse repositories and their sonar scanner results)
|   └── repo.py      (code to clone/pull repositories)
|   └── reporting.py (generate reports based on analysis)
│
├── __init__.py (contains repositories list)
├── logs.py     (logging configuration)
│
├── (other *.py files are no longer used; can be ignored)
```
