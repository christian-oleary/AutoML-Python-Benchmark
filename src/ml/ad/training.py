"""Anomaly detection training and evaluation."""

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

from ml.ad.anomaly_detection import BaseADModel, LunarADModel, PyCaretADModel, TimeSeriesODModel
from ml.metrics import Metrics


class SKABTrainer:
    """Class to handle training and evaluation of anomaly detection models.

    :param str tool: The anomaly detection tool to use (e.g. PyCaretADModel, TimeSeriesODModel).
    :param float | None contamination: The expected proportion of anomalies in the dataset.
    :param Path results_dir: The base directory to save results in.
    :param bool skip_done_experiments: Whether to skip experiments that have already been completed.
    :param float test_set_size: The proportion of data to use as the test set (between 0 and 1).
    :param int verbosity: The verbosity level for logging and library outputs (0-3).
    :param int | None window_size: The size of the sliding window to create features from (if applicable).
    """

    def __init__(
        self,
        tool: str,
        contamination: float | str | None,
        results_dir: Path,
        skip_done_experiments: bool,
        test_set_size: float,
        verbosity: int,
        window_size: int | str | None,
    ):
        if contamination == 'None':
            contamination = None
        elif isinstance(contamination, str):
            contamination = float(contamination)

        if window_size == 'None':
            window_size = None
        elif isinstance(window_size, str):
            window_size = int(window_size)

        self.tool = tool
        self.contamination = contamination
        self.results_dir = Path(results_dir)
        self.skip_done_experiments = skip_done_experiments
        self.test_set_size = test_set_size
        self.verbosity = verbosity
        self.window_size = window_size

    def train_models(
        self,
        name: str,
        dataframes: dict[str, pd.DataFrame],
        all_metadata: dict,
        all_results: list[dict],
    ) -> None:
        """Train anomaly detection models on the specified dataset and save results.

        :param str name: The name of the dataset to train on.
        :param dict dataframes: A dictionary mapping dataset names to DataFrames.
        :param dict all_metadata: A dictionary to store metadata for all experiments.
        :param list all_results: A list to store results for all experiments.
        """
        # Process dataset and prepare features/labels
        self.dataset_name = str(Path(name))
        df_ = dataframes[self.dataset_name].drop(columns=['changepoint'], errors='ignore')
        true_labels = df_['anomaly']
        data_ = self.prepare_data(df_, target_col='anomaly')

        # Create results subdirectory based on dataset name and window size
        self.results_subdir = Path(
            self.results_dir,
            self.dataset_name.replace(os.sep, '__').replace('.csv', ''),
            'original_columns' if self.window_size is None else f'window_size_{self.window_size}',
        )

        # Save metadata
        logger.debug(f'results_subdir: {self.results_subdir}')
        metadata = self._save_metadata(df_, data_)

        # Save metadata in a dictionary keyed by dataset name and window size
        if self.window_size is not None:
            key = f'{self.dataset_name}_window-{self.window_size}'
        else:
            key = f'{self.dataset_name}_no-window'
        all_metadata[key] = metadata

        # Run anomaly detection
        for scores_ in self._iterate_ad_options(true_labels, **data_):
            all_results.append(scores_)

    def prepare_data(self, df: pd.DataFrame, target_col: str = 'anomaly') -> dict[str, Any]:
        """Prepare features, labels, and metadata for anomaly detection.

        :param pd.DataFrame df: The input DataFrame containing the time series data and target column.
        :param str target_col: The name of the target column to drop before creating features.
        :return dict: Dictionary of split data (X_train, y_train, X_test, y_test) and other metadata.
        """
        self.contamination_calculated = df[target_col].mean()  # Proportion of anomalies
        if self.contamination is None:
            self.contamination = self.contamination_calculated

        logger.debug(
            f'df.shape: {df.shape}; df.columns: {df.columns.tolist()}; '
            f'label counts: {df[target_col].value_counts().to_dict()}; '
            f'contamination: {self.contamination:.4f}'
        )

        # Deal with any date and time columns
        if self.tool in ['pycaret', PyCaretADModel.__name__]:
            dropped_cols = []
        else:
            dropped_cols = [
                c for c in df.columns if any(s in c for s in ['datetime', 'time', 'timestamp'])
            ]
            df = df[[c for c in df.columns if c not in dropped_cols]]

        # Generate features using sliding windows
        if self.window_size is not None and self.tool in ['pycaret', PyCaretADModel.__name__]:
            features, labels = self._make_windows(df, target_col=target_col)
        else:
            labels = df[target_col].values
            X = df[[c for c in df.columns if c != target_col]].values
            features = pd.DataFrame(X, columns=[c for c in df.columns if c != target_col])

        # Split into train/test sets
        split_idx = int((1 - self.test_set_size) * len(features))
        X_train, y_train = features.iloc[:split_idx], labels[:split_idx]
        X_test, y_test = features.iloc[split_idx:], labels[split_idx:]
        logger.trace(
            f'X_train: {X_train.shape}, y_train: {y_train.shape}, '
            f'X_test: {X_test.shape}, y_test: {y_test.shape}'
        )
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'dropped_cols': dropped_cols,
        }

    def _make_windows(self, df: pd.DataFrame, target_col: str) -> tuple:
        """Convert SKAB time series into supervised learning samples.

        :param pd.DataFrame df: The input DataFrame containing the time series data and target column.
        :param str target_col: The name of the target column to drop before creating features.
        :return: A tuple (features_df, labels) where features_df is a DataFrame
        """
        if self.window_size is None:
            raise ValueError('window_size must be specified to create sliding window features.')

        features = [c for c in df.columns if c != target_col]
        X, y = [], []
        values = df[features].values
        labels = df[target_col].values

        # Create sliding windows of features and corresponding labels
        for i in range(self.window_size, len(df)):
            X.append(values[i - self.window_size : i].flatten())
            y.append(labels[i])

        X = np.array(X)  # type: ignore
        y = np.array(y)  # type: ignore
        columns = [f'f{j}' for j in range(X.shape[1])]  # type: ignore
        features = pd.DataFrame(X, columns=columns)
        return features, y

    def _save_metadata(self, df: pd.DataFrame, split_data: dict, **kwargs) -> dict:
        """Save metadata to a JSON file, e.g. dataset name, shapes, columns, etc.

        :param pd.DataFrame df: The original DataFrame before splitting.
        :param dict split_data: Split data (X_train, y_train, X_test, etc.).
        :return: The metadata dictionary that was saved.
        """
        dropped_cols = split_data.get('dropped_cols', [])
        columns = [c for c in df.columns if c not in dropped_cols]
        metadata = {
            'dataset_name': self.dataset_name,
            'df_shape': df.shape,
            'contamination': self.contamination,
            'contamination_calculated': self.contamination_calculated,
            'window_size': self.window_size,
            'X_train_shape': split_data['X_train'].shape,
            'y_train_shape': split_data['y_train'].shape,
            'X_test_shape': split_data['X_test'].shape,
            'y_test_shape': split_data['y_test'].shape,
            'dropped_cols': dropped_cols,
            'columns': columns,
            **kwargs,
        }
        self.results_subdir.mkdir(parents=True, exist_ok=True)
        with open(self.results_subdir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        return metadata

    def _iterate_ad_options(self, true_labels: pd.Series, **kwargs):
        """Iterate through different anomaly detection options.

        :param pd.Series true_labels: The true labels for the dataset.
        :param dict kwargs: The data and metadata to pass to the anomaly detection class.
        :return: A generator yielding scores for each model and parameter combination.
        """
        # Determine which class to use based on the class_name
        if self.tool in ['pycaret', PyCaretADModel.__name__]:
            ad_class = PyCaretADModel
        elif self.tool in ['ts_od', TimeSeriesODModel.__name__]:
            ad_class = TimeSeriesODModel
        elif self.tool in ['lunar', 'LunarADModel']:
            ad_class = LunarADModel
        else:
            raise ValueError(f'Unknown tool: {self.tool}')
        class_name = ad_class.__name__

        # File to save results for this class and dataset
        results_file = self.results_subdir / f'{class_name}.csv'

        # Results of completed runs
        df_existing = pd.DataFrame()
        if results_file.exists():
            df_existing = pd.read_csv(results_file)

        # Loop through all parameter combinations
        for param_name, options in ad_class.parameter_options.items():
            for param_value in options:
                kwargs[param_name] = param_value

                # Path to save predictions for this model-parameter combination
                param_str = '__'.join(
                    f'{k}-{v}' for k, v in kwargs.items() if k in ad_class.parameter_options
                )
                predictions_file = Path(
                    self.results_subdir, 'predictions', f'{class_name}__{param_str}.csv'
                )

                # Check if predictions and scores exist already
                predictions_found = self.skip_done_experiments and predictions_file.exists()
                if predictions_found and self._combination_found(df_existing, ad_class, **kwargs):
                    # Return existing scores for this combination
                    df_scores = df_existing[(df_existing[f'param__{param_name}'] == param_value)]
                    scores = df_scores.to_dict(orient='records')[0]
                    yield scores
                else:
                    # Fit models and calculate scores
                    kwargs['contamination'] = self.contamination
                    kwargs['window_size'] = self.window_size
                    kwargs['verbose'] = self.verbosity
                    results = self._invoke_ad_class(ad_class, **kwargs)
                    scores = self._calculate_scores(self.dataset_name, **results, **kwargs)

                    # Save scores
                    self.results_subdir.mkdir(parents=True, exist_ok=True)
                    Metrics.write_to_csv(results_file, scores)
                    # Save predictions
                    self._save_predictions(predictions_file, true_labels, results)

                    # logger.success(f'{class_name}:\n{json.dumps(scores_, indent=2)}')
                    logger.success(
                        f'{class_name}: F1-score={scores["test__f1_score"]:.3f}, Precision='
                        f'{scores["test__precision"]:.3f}, Recall={scores["test__recall"]:.3f}, '
                        f'Train time={scores["fit_time"]:.2f}s, '
                        f'Predict time={scores["predict_time"]:.2f}s'
                    )
                    yield scores

    def _invoke_ad_class(
        self,
        ad_class: type[BaseADModel | TimeSeriesODModel | LunarADModel],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        **kwargs,
    ) -> dict:
        """Run the specified anomaly detection class.

        :param type[BaseADModel | TimeSeriesODModel | LunarADModel] ad_class: The anomaly detection class to invoke.
        :param pd.DataFrame X_train: The training features.
        :param pd.DataFrame X_test: The test features.
        :param dict kwargs: Additional keyword arguments to pass to the anomaly detection class.
        :return: A dictionary containing the estimator and predictions.
        """
        text = (
            f'Running {ad_class.__name__}: contamination={self.contamination}, '
            f'window_size={self.window_size}'
        )
        for param_name, _ in ad_class.parameter_options.items():
            param_value = kwargs.get(param_name, None)
            text += f', {param_name}={param_value}'
        logger.info(text)

        estimator = ad_class(**kwargs)

        # Fit models
        start_time = perf_counter()
        estimator.fit(X_train)
        fit_time = perf_counter() - start_time
        logger.trace(f'Fit time: {fit_time:.2f} seconds')

        # Make predictions
        start_time = perf_counter()
        predictions_train = estimator.predict(X_train)
        predictions_test = estimator.predict(X_test)
        predict_time = perf_counter() - start_time
        logger.trace(f'Prediction time: {predict_time:.2f} seconds')

        return {
            'estimator': estimator,
            'predictions_train': predictions_train,
            'predictions_test': predictions_test,
            'fit_time': fit_time,
            'predict_time': predict_time,
        }

    def _combination_found(
        self,
        df_existing: pd.DataFrame,
        ad_class: type[BaseADModel | TimeSeriesODModel | LunarADModel],
        **kwargs,
    ) -> bool:
        """Check if a given parameter combination has been completed for this class and dataset.

        :param pd.DataFrame | None df_existing: The existing results DataFrame.
        :param type[BaseADModel | TimeSeriesODModel | LunarADModel] ad_class: The anomaly detection class.
        :param dict kwargs: The parameter combination to check.
        :return: True if the combination has already been completed, False otherwise.
        """
        # Extract only the relevant parameters for this class from kwargs
        params = {k: v for k, v in kwargs.items() if k in ad_class.parameter_options}
        # If there are no existing results, return False
        if len(df_existing) > 0:
            # Check if all parameters match any existing row in the results DataFrame
            if all((df_existing[f'param__{k}'] == v).any() for k, v in params.items()):
                return True
        return False

    def _calculate_scores(
        self,
        dataset_name: str,
        estimator: BaseADModel | TimeSeriesODModel | LunarADModel,
        predictions_train: np.ndarray,
        predictions_test: np.ndarray,
        fit_time: float,
        predict_time: float,
        y_train: np.ndarray,
        y_test: np.ndarray,
        **_,
    ) -> dict:
        """Calculate evaluation scores for the anomaly detection model.

        :param str dataset_name: The name of the dataset.
        :param int | None window_size: The size of the sliding window used for features.
        :param BaseADModel | TimeSeriesODModel | LunarADModel estimator: The fitted anomaly detection model.
        :param np.ndarray predictions_train: The training predictions.
        :param np.ndarray predictions_test: The test predictions.
        :param float fit_time: The time taken to fit the model.
        :param float predict_time: The time taken to make predictions.
        :param np.ndarray y_train: The true labels for the training set.
        :param np.ndarray y_test: The true labels for the test set.
        :return: A dictionary of evaluation scores (e.g., precision, recall, F1-score).
        """
        scores = {
            'dataset': dataset_name,
            'window_size': self.window_size,
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
            if param in estimator.parameter_options:
                scores[f'param__{param}'] = value
        return scores

    def _save_predictions(self, predictions_file: Path, true_labels: pd.Series, results: dict):
        """Save predictions and true labels to a CSV file.

        :param Path predictions_file: The path to save the predictions CSV file.
        :param pd.Series true_labels: The true labels for the dataset.
        :param dict results: Dictionary containing predictions and other results.
        """
        # Combine train and test predictions into a single array/series
        predictions_ = [results['predictions_train'], results['predictions_test']]
        if isinstance(predictions_[0], np.ndarray):
            predictions = np.concatenate(predictions_)
        else:
            predictions = pd.concat(predictions_)

        # Save predictions and true labels to CSV
        df_predictions = pd.DataFrame(
            {'true_labels': true_labels, 'predictions': predictions},
            columns=['true_labels', 'predictions'],
            index=true_labels.index,
        )
        logger.trace(f'Saving predictions to: {predictions_file}')
        predictions_file.parent.mkdir(parents=True, exist_ok=True)
        df_predictions.to_csv(predictions_file, index=True)
