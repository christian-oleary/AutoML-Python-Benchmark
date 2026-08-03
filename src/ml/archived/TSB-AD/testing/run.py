"""Adapted from README.md of TSB-AD repository."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import torch

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.model_wrapper import (
    run_Semisupervise_AD,
    run_Unsupervise_AD,
    Semisupervise_AD_Pool,
    Unsupervise_AD_Pool,
)
from TSB_AD.utils.slidingWindows import find_length_rank

from testing.analysis import Analysis

DATA_DIR = Path('Datasets')
DATA_SUBDIRS = [
    'TSB-AD-U',
    'TSB-AD-M',
]

RESULTS_DIR = Path('results')

EXCLUDED_MODELS = [
    'COF',  #
    'CBLOF',  # Unstable
    'Lag_Llama',  # Excluded due to outdated dependencies.
    'NORMA',
    'Series2Graph',  # Not included in the repository due to patents.
]
# MODELS = [model for model in Unsupervise_AD_Pool + Semisupervise_AD_Pool if model not in EXCLUDED_MODELS]
MODELS = [model for model in Unsupervise_AD_Pool if model not in EXCLUDED_MODELS]
# MODELS = [model for model in Semisupervise_AD_Pool if model not in missing_models]


def run_experiments(models: list = MODELS):
    """Run the AD experiments and save results.

    :param list models: List of model names to include in the experiments.
    """
    # Log GPU access
    logger.info(f'Torch CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        logger.info(f'GPU Device Name: {torch.cuda.get_device_name(0)}')
    else:
        logger.error('No GPU available.')

    logger.debug(f'Using models: {models}')

    all_results = []
    # Iterate through the two subdirectories (TSB-AD-U and TSB-AD-M)
    for subdir in DATA_SUBDIRS:
        data_dir = DATA_DIR / subdir

        # Iterate through all CSV files in the data directory
        for file_name in data_dir.glob('*.csv'):

            # if 'MITDB' in file_name.stem: continue

            # Create a results directory for the current file
            results_dir = RESULTS_DIR / subdir / file_name.stem
            results_dir.mkdir(parents=True, exist_ok=True)

            # Results file for the current dataset
            local_results_file = results_dir / f'results_{file_name.stem}.csv'
            logger.info(f'Processing file: {file_name}')

            # Check if results already exist for this file
            df_existing, existing_models = read_dataset_results(local_results_file, models)

            # Add all existing local results to the local results list
            local_results = []
            if len(df_existing) > 0:
                local_results.extend(df_existing.to_dict('records'))

            # Loop over models and run experiments
            data, label = None, None
            for model_name in models:

                # Read results for this model if they already exist instead of re-training
                if len(df_existing) > 0 and model_name in existing_models:
                    #     results_ = df_existing[df_existing['model_name'] == model_name].iloc[0].to_dict()
                    #     local_results.append(results_)
                    continue

                # Load data if not already loaded (only load once per dataset)
                if data is None or label is None:
                    data, label, data_train, slidingWindow = load_data(file_name, results_dir)

                # Train and evaluate model
                try:
                    model_results = train_model(
                        model_name=model_name,
                        data=data,
                        label=label,
                        data_train=data_train,
                        slidingWindow=slidingWindow,
                        file_name=file_name,
                        df_existing=df_existing,
                    )

                    # Save local results after each model
                    local_results.append(model_results[0])
                    save_local_results(df_existing, local_results, local_results_file)
                except Exception as e:
                    logger.error(f'Error processing file {file_name}: {e}')

            # Add local results to the overall results list
            all_results.extend(local_results)

    # Save all results to a CSV file
    results_df = pd.DataFrame(all_results)
    results_path = RESULTS_DIR / 'all_results.csv'
    results_df.to_csv(results_path, index=False)
    logger.success(f'Experiments done. Results saved to {results_path}')

    # Save list of all model names to a tex file
    save_list_as_text(results_df['model_name'].unique().tolist(), RESULTS_DIR / 'model_names.tex')
    # save_list_as_text(results_df['dataset'].unique().tolist(), RESULTS_DIR / 'dataset_names.tex')


def read_dataset_results(results_file: Path, models: list) -> tuple[pd.DataFrame, list, list]:
    """Read existing results for a dataset if they exist.

    :param Path results_file: Path to the results CSV file for the dataset.
    :param list models: List of model names to check for existing results.
    """
    df_existing, existing_models = pd.DataFrame(), []

    if results_file.exists():
        df_existing = pd.read_csv(results_file)

        # Check if results for all models already exist
        existing_models = df_existing['model_name'].unique()
        if set(existing_models.tolist()) == set(models):
            logger.debug(f'All results already exist for {results_file.name}, skipping...')
        else:
            models_to_run = [model for model in models if model not in existing_models]
            if len(models_to_run) > 0:
                logger.debug(f'Results found for: {existing_models}. Training: {models_to_run}')
    return df_existing, existing_models


def load_data(file_name: Path, results_dir: Path) -> tuple:
    """Load data from a CSV file and save data information.

    :param Path file_name: Path to the dataset CSV file.
    :param Path results_dir: Path to the directory where results will be saved.
    :return tuple: A tuple containing the data, labels, training data, and sliding window.
    """
    df = pd.read_csv(file_name).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)  # Assuming the last column is the label
    label = df['Label'].astype(int).to_numpy()  # Convert the 'Label' column to integers

    slidingWindow = find_length_rank(data, rank=1)
    train_index = str(file_name).split('.')[0].split('_')[-3]
    data_train = data[: int(train_index), :]

    # Save data information to a text file in the results directory
    with open(results_dir / 'data_info.txt', 'w') as f:
        f.write(f'File: {file_name}\nShapes: data={data.shape}, label={label.shape}\n')
    return data, label, data_train, slidingWindow


def train_model(
    model_name: str,
    data: np.ndarray,
    label: np.ndarray,
    data_train: np.ndarray,
    slidingWindow: int,
    file_name: Path,
    df_existing: pd.DataFrame,
):
    """Train and evaluate anomaly detection model.

    :param str model_name: Name of the model to train.
    :param np.ndarray data: Input data for training and evaluation.
    :param np.ndarray label: Ground truth labels for evaluation.
    :param np.ndarray data_train: Training data.
    :param int slidingWindow: Length of the sliding window.
    :param Path file_name: Name of the dataset file (for logging purposes).
    :param pd.DataFrame df_existing: DataFrame containing existing results for the dataset (if any).
    :return tuple: A tuple containing the evaluation results and a standardized result dictionary.
    """
    logger.debug(f'Training: {model_name}')
    start_time = datetime.now()

    # Apply Anomaly Detector
    if model_name in Semisupervise_AD_Pool:
        output = run_Semisupervise_AD(model_name, data_train, data)
    elif model_name in Unsupervise_AD_Pool:
        output = run_Unsupervise_AD(model_name, data)
    else:
        raise ValueError()

    # Evaluation
    total_time = datetime.now() - start_time
    pred = output > (np.mean(output) + 3 * np.std(output))  # type: ignore
    result = {
        'model_name': model_name,
        'dataset': file_name.stem,
        'Time': total_time.total_seconds(),
        # **get_metrics(output, label),
        **get_metrics(output, label, slidingWindow=slidingWindow, pred=pred),
    }

    # Newly added metric:
    new_metrics = [
        # 'Accuracy', 'Balanced Accuracy', 'Detection Rate',
        # 'FPR', 'TPR', 'Precision', 'Recall', 'MCC', 'Time',
    ]
    # Omit new metrics if they are not in old results to avoid concatenation issues
    for new_metric in new_metrics:
        if new_metric not in result:
            continue
        if len(df_existing) > 0 and new_metric not in df_existing.columns:
            del result[new_metric]

    result_standardised = {k: v for k, v in result.items() if k not in new_metrics}
    return result, result_standardised


def save_local_results(df_existing: pd.DataFrame, local_results: list, local_results_file: Path):
    """Save local results to a CSV file.

    :param pd.DataFrame df_existing: DataFrame containing existing results.
    :param list local_results: List of dictionaries containing local results.
    :param Path local_results_file: Path to the file where results will be saved.
    """
    # Create a DataFrame for local results and existing results
    df_subresults = pd.DataFrame(local_results)
    if len(df_existing) > 0:
        df_subresults = pd.concat([df_existing, df_subresults], ignore_index=True)

    # Move the 'model_name' and 'dataset' columns to the front
    cols = df_subresults.columns.tolist()
    cols.insert(0, cols.pop(cols.index('model_name')))
    cols.insert(1, cols.pop(cols.index('dataset')))
    df_subresults = df_subresults[cols]

    # Save local results to a CSV file in the results directory
    logger.trace(f'Saving results to {local_results_file}')
    df_subresults.to_csv(local_results_file, index=False)


def save_list_as_text(items: list, path: Path) -> str:
    """Format a list of strings for human reading and save to file.

    :param list items: List of strings to format.
    :param Path path: Path to the file where the formatted string will be saved.
    :return str: Formatted string.
    """
    text = ', '.join(items)
    if len(items) > 1:
        text = text.rsplit(', ', 1)
        text = ' and '.join(text)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

    # logger.debug(f'List saved to {path}: "{text}"')
    return text


if __name__ == "__main__":
    custom_models = ['PyCaretADModel', 'TimeSeriesODModel', 'LunarADModel']
    run_experiments(custom_models)
    Analysis().run_analysis()
    run_experiments(MODELS)
    Analysis().run_analysis()
