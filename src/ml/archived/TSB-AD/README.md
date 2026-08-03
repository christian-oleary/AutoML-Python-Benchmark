# Adapted TSB-AD Benchmark

This repository is a stripped-down, adapted version of the original [TSB-AD benchmark](https://github.com/TheDatumOrg/TSB-AD).

The experiments the original models included in TSB-AD except for the following:

- CBLOF (unstable)
- Lag-Llama (incompatible with recent PyTorch versions)
- NORMA and Series2Graph (omitted from the original repository due to patent applications)

The original [README](https://github.com/thedatumorg/TSB-AD/blob/main/README.md) file includes information about the models and datasets used. A local copy of this file is included here as [README.original.md](README.original.md).

## Installation

This was run in WSL on Windows 11.

To install TSB-AD from source, you will need the following tools:

- `git`
- `conda` (anaconda or miniconda)

**Step 1:** Clone this repository using `git` and change into its root directory.

```bash
git clone https://github.com/TheDatumOrg/TSB-AD.git
```

**Step 2:** Create and activate a `conda` environment named `TSB-AD`.

```bash
conda create -n TSB-AD python=3.11 -y # They claim to support python>=3.8, up to 3.12
conda activate TSB-AD
```

**Step 3:** Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install pycaret[models,tuners] || python -m pip install --no-deps pycaret[models,tuners]
python -m pip install --no-deps momentfm autogluon.common autogluon.core autogluon.features autogluon.tabular[catboost,lightgbm,xgboost] autogluon.timeseries[TimeSeriesDataFrame,TimeSeriesPredictor]
python -m pip install --no-deps git+https://github.com/christian-oleary/AutoML-Python-Benchmark.git
python -m pip install --ignore-requires-python git+https://github.com/ibm-granite/granite-tsfm.git
```

<!-- # Download Lag-Llama model:
wget https://huggingface.co/time-series-foundation-models/Lag-Llama/blob/main/lag-llama.ckpt -->
<!-- conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia -->

Check PyTorch's access to GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Run Experiments

Results will be located in `benchmark_exp/eval/`.

```bash
conda activate TSB-AD
cd benchmark_exp/

# HPO for univariate models (results in 'benchmark_exp/eval/HP_tuning/uni/')
python HP_Tuning_U.py --AD_Name LunarADModel
python HP_Tuning_U.py --AD_Name PyCaretADModel
python HP_Tuning_U.py --AD_Name TimeSeriesODModel

# HPO for multivariate models (results in 'benchmark_exp/eval/HP_tuning/multi/')
python HP_Tuning_M.py --AD_Name LunarADModel
python HP_Tuning_M.py --AD_Name PyCaretADModel
python HP_Tuning_M.py --AD_Name TimeSeriesODModel

# Model evaluation for univariate models (results in 'benchmark_exp/eval/metrics/uni/')
python Run_Detector_U.py --AD_Name LunarADModel
python Run_Detector_U.py --AD_Name PyCaretADModel
python Run_Detector_U.py --AD_Name TimeSeriesODModel

# Model evaluation for multivariate models (results in 'benchmark_exp/eval/metrics/multi/')
python Run_Detector_M.py --AD_Name LunarADModel
python Run_Detector_M.py --AD_Name PyCaretADModel
python Run_Detector_M.py --AD_Name TimeSeriesODModel
```

Statistics and plots can be generated using the `benchmark_exp/analysis.ipynb` Jupyter Notebook. Resulting plots are located in `benchmark_exp/eval/plots`.
