"""Anomaly Detection module."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.ad.anomaly_detection import BaseADModel, PyCaretADModel
from ml.metrics import Metrics


def load_skab(root_dir: str) -> dict[str, pd.DataFrame]:
    """Load the SKAB dataset.

    :param str root_dir: Path to the SKAB 'data' directory.
    :return: A dictionary mapping 'subfolder/filename.csv' -> DataFrame.
    """
    dataframes = {}
    # Walk through all directories and files under the root directory
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.csv'):
                file_path = os.path.join(subdir, file)
                # Load CSV
                df = pd.read_csv(file_path, delimiter=';')
                # Normalise column names
                df.columns = [c.strip().lower() for c in df.columns]
                # Parse timestamp column if present
                for col in df.columns:
                    if 'datetime' in col or 'time' in col or 'timestamp' in col:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                # Dictionary key: relative path
                dataframes[os.path.relpath(file_path, root_dir)] = df
    return dataframes


def prepare_data(
    df: pd.DataFrame, target_col: str = 'anomaly', window_size: int | None = None
) -> dict[str, Any]:
    """Prepare features, labels, and metadata for anomaly detection.

    :param pd.DataFrame df: The input DataFrame containing the time series data and target column.
    :param str target_col: The name of the target column to drop before creating features.
    :param int | None window_size: Sliding window  size. If None, no windows are created.
    :return dict: Dictionary of split data (X_train, y_train, X_test, y_test) and other metadata.
    """
    logger.debug(f'Input DataFrame shape: {df.shape}')
    logger.debug(f'Input DataFrame columns: {df.columns.tolist()}')
    logger.debug(f'Label counts: {df[target_col].value_counts().to_dict()}')

    # Calculate proportions of anomalies
    contamination = df[target_col].mean()
    logger.debug(f'Contamination (proportion of anomalies): {contamination:.4f}')

    # Generate features using sliding windows
    if window_size is not None:
        features, labels = make_windows(df, target_col=target_col, window_size=window_size)
    else:
        labels = df[target_col].values
        X = df[[c for c in df.columns if c != target_col]].values
        features = pd.DataFrame(X, columns=[c for c in df.columns if c != target_col])

    # Deal with any date and time columns
    # Commented out as PyCaret seems to be able to handle datetime columns:
    dropped_cols = [  # ]  # type: ignore
        c for c in features.columns if any(s in c for s in ['datetime', 'time', 'timestamp'])
    ]
    features = features[[c for c in features.columns if c not in dropped_cols]]

    # Split into train/test sets (75/25 split)
    split_idx = int(0.75 * len(features))
    X_train, y_train = features.iloc[:split_idx], labels[:split_idx]
    X_test, y_test = features.iloc[split_idx:], labels[split_idx:]
    logger.debug(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
    logger.debug(f'X_test: {X_test.shape}, y_test: {y_test.shape}')
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'contamination': contamination,
        'dropped_cols': dropped_cols,
    }


def make_windows(df: pd.DataFrame, window_size: int, target_col: str) -> tuple:
    """Convert SKAB time series into supervised learning samples.

    :param pd.DataFrame df: The input DataFrame containing the time series data and target column.
    :param int window_size: The size of the sliding window to create features from.
    :param str target_col: The name of the target column to drop before creating features.
    :return: A tuple (features_df, labels) where features_df is a DataFrame
    """
    features = [c for c in df.columns if c != target_col]
    X, y = [], []
    values = df[features].values
    labels = df[target_col].values

    # Create sliding windows of features and corresponding labels
    for i in range(window_size, len(df)):
        X.append(values[i - window_size : i].flatten())
        y.append(labels[i])

    X = np.array(X)  # type: ignore
    y = np.array(y)  # type: ignore
    columns = [f'f{j}' for j in range(X.shape[1])]  # type: ignore
    features = pd.DataFrame(X, columns=columns)
    return features, y


def invoke_ad_class(
    ad_class: type[BaseADModel],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    contamination: float,
    **kwargs,
) -> tuple:
    """Run the specified anomaly detection class.

    :param type[BaseADModel] ad_class: The anomaly detection class to invoke.
    :param pd.DataFrame X_train: The training features.
    :param pd.DataFrame X_test: The test features.
    :param float contamination: The proportion of anomalies in the dataset.
    :param dict kwargs: Additional keyword arguments to pass to the anomaly detection class.
    :return: A tuple (estimator, train predictions, test predictions).
    """
    logger.info(f'Running {ad_class.__name__}: contamination={contamination}')
    if 'model_name' in kwargs:
        logger.info(f'Model name: {kwargs["model_name"]}')
    estimator = ad_class(**kwargs)

    # Fit models
    start_time = perf_counter()
    estimator.fit(X_train, contamination=contamination)
    fit_time = perf_counter() - start_time
    logger.debug(f'Fit time: {fit_time:.2f} seconds')

    # Make predictions
    start_time = perf_counter()
    predictions_train = estimator.predict(X_train)
    predictions_test = estimator.predict(X_test)
    predict_time = perf_counter() - start_time
    logger.debug(f'Prediction time: {predict_time:.2f} seconds')
    return estimator, predictions_train, predictions_test, fit_time, predict_time


def iterate_ad_options(class_name: str, results_subdir: Path, **kwargs):
    """Iterate through different anomaly detection options.

    :param str class_name: The name of the anomaly detection class to invoke.
    :param Path results_subdir: The directory to save results.
    :return: A generator yielding tuples of (estimator, scores) for each parameter combination.
    """
    # Determine which class to use based on the class_name
    if class_name in ['pycaret', PyCaretADModel.__name__]:
        ad_class = PyCaretADModel
    else:
        raise ValueError(f'Unknown class name: {class_name}')

    # Check if this class has already been completed for this dataset/window size
    completion_file = results_subdir / f'{class_name}_completed.txt'
    if completion_file.exists():
        logger.info(f'{class_name} already completed. Skipping...')
        return

    # Loop through all parameter combinations
    for param_name, options in ad_class.parameter_options.items():
        for param_value in options:
            kwargs[param_name] = param_value
            # Fit models and calculate scores
            results = invoke_ad_class(ad_class, **kwargs)
            scores = calculate_scores(*results, **kwargs)

            # Save scores
            results_subdir.mkdir(parents=True, exist_ok=True)
            Metrics.write_to_csv(results_subdir / f'{class_name}.csv', scores)
            yield results[0], scores

    # Record completion
    completion_file.touch()


def calculate_scores(
    estimator: BaseADModel,
    predictions_train: np.ndarray,
    predictions_test: np.ndarray,
    fit_time: float,
    predict_time: float,
    y_train: np.ndarray,
    y_test: np.ndarray,
    **_,
) -> dict:
    """Calculate evaluation scores for the anomaly detection model.

    :param BaseADModel estimator: The fitted anomaly detection model.
    :param np.ndarray predictions_train: The training predictions.
    :param np.ndarray predictions_test: The test predictions.
    :param float fit_time: The time taken to fit the model.
    :param float predict_time: The time taken to make predictions.
    :param np.ndarray y_train: The true labels for the training set.
    :param np.ndarray y_test: The true labels for the test set.
    :return: A dictionary of evaluation scores (e.g., precision, recall, F1-score).
    """
    scores = {
        'model': estimator.__class__.__name__,
        'fit_time': fit_time,
        'predict_time': predict_time,
    }
    # Calculate scores for both train and test sets
    for name, labels, predictions in [
        ('train', y_train, predictions_train),
        ('test', y_test, predictions_test),
    ]:
        # Supervised metrics (using true labels)
        scores = {
            **scores,
            f'{name}__accuracy': accuracy_score(labels, predictions),
            f'{name}__average_precision': average_precision_score(labels, predictions),
            f'{name}__balanced_accuracy': balanced_accuracy_score(labels, predictions),
            f'{name}__f1_score': f1_score(labels, predictions),
            f'{name}__matthews_corrcoef': matthews_corrcoef(labels, predictions),
            f'{name}__precision': precision_score(labels, predictions),
            f'{name}__recall': recall_score(labels, predictions),
            f'{name}__roc_auc': roc_auc_score(labels, predictions),
        }

    # Record model parameters
    for param, value in estimator.__dict__.items():
        if param == 'model':
            continue
        scores[f'param__{param}'] = value
    return scores


def save_metadata(
    results_subdir: Path, dataset_name: str, df: pd.DataFrame, split_data: dict, **kwargs
) -> dict:
    """Save metadata to a JSON file, e.g. dataset name, shapes, columns, etc.

    :param Path results_subdir: The directory to save the metadata file in.
    :param str dataset_name: The name of the dataset.
    :param pd.DataFrame df: The original DataFrame before splitting.
    :param dict split_data: Split data (X_train, y_train, X_test, etc.).
    :return: The metadata dictionary that was saved.
    """
    dropped_cols = split_data.get('dropped_cols', [])
    columns = [c for c in df.columns if c not in dropped_cols]
    metadata = {
        'dataset_name': dataset_name,
        'df_shape': df.shape,
        'contamination': split_data['contamination'],
        'X_train_shape': split_data['X_train'].shape,
        'y_train_shape': split_data['y_train'].shape,
        'X_test_shape': split_data['X_test'].shape,
        'y_test_shape': split_data['y_test'].shape,
        'dropped_cols': dropped_cols,
        'columns': columns,
        **kwargs,
    }
    results_subdir.mkdir(parents=True, exist_ok=True)
    with open(results_subdir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    return metadata


def main():
    """Main function to run anomaly detection experiments."""
    skab_data = load_skab('data/SKAB/')
    logger.info(f'Loaded {len(skab_data)} datasets.')
    all_metadata = {}

    # Size of the sliding window to create features from
    window_sizes = [None]
    # window_sizes = [2, 20]
    for window_size in window_sizes:
        # Iterate through datasets and run anomaly detection
        for name, df in skab_data.items():
            if 'valve' not in name:  # Only run on valve datasets
                continue
            # df = df.head(100)
            logger.info(f'DATASET: {name}, shape: {df.shape}')

            # Process dataset
            dataset_name = str(Path(name))
            df_ = skab_data[dataset_name].drop(columns=['changepoint'], errors='ignore')
            data_ = prepare_data(df_, target_col='anomaly', window_size=window_size)

            # Create results subdirectory based on dataset name and window size
            results_subdir_ = Path(
                'results/ad', dataset_name.replace(os.sep, '__').replace('.csv', '')
            )
            if window_size is not None:
                results_subdir_ = results_subdir_ / f'window_size_{window_size}'
            else:
                results_subdir_ = results_subdir_ / 'original_columns'

            # Save metadata
            logger.info(f'results_subdir: {results_subdir_}')
            metadata = save_metadata(
                results_subdir_, dataset_name, df_, data_, window_size=window_size
            )

            # Save metadata in a dictionary keyed by dataset name and window size
            if window_size is not None:
                all_metadata[f'{dataset_name}_window-{window_size}'] = metadata
            else:
                all_metadata[f'{dataset_name}_original_columns'] = metadata

            # Run anomaly detection
            for tool_ in ['pycaret']:
                for _, scores_ in iterate_ad_options(tool_, results_subdir_, **data_):
                    logger.success(f'{tool_}:\n{json.dumps(scores_, indent=2)}')

    # Save all metadata
    with open('results/ad/all_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=4)
    logger.success('All experiments completed.')


if __name__ == "__main__":
    main()
