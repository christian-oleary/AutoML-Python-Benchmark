"""Anomaly detection training and evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from loguru import logger
import matplotlib.pyplot as plt
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
from ml.plots import Plotter


class SKABTrainer:
    """Class to handle training and evaluation of anomaly detection models."""

    def train_models(
        self,
        name: str,
        dataframes: dict[str, pd.DataFrame],
        results_dir: Path,
        all_metadata: dict,
        all_results: list[dict],
        contamination: float | None = None,
        window_size: int | None = None,
    ) -> None:
        """Train anomaly detection models on the specified dataset and save results.

        :param str name: The name of the dataset to train on.
        :param dict dataframes: A dictionary mapping dataset names to DataFrames.
        :param Path results_dir: The directory to save results in.
        :param dict all_metadata: A dictionary to store metadata for all experiments.
        :param list all_results: A list to store results for all experiments.
        :param float | None contamination: The expected proportion of anomalies in the dataset.
        :param int | None window_size: The size of the sliding window to create features from
        """
        if contamination == 'None':
            contamination = None

        # Process dataset and prepare features/labels
        dataset_name = str(Path(name))
        df_ = dataframes[dataset_name].drop(columns=['changepoint'], errors='ignore')
        data_ = self.prepare_data(
            df_, target_col='anomaly', window_size=window_size, contamination=contamination
        )

        # Create results subdirectory based on dataset name and window size
        results_subdir_ = Path(
            results_dir,
            dataset_name.replace(os.sep, '__').replace('.csv', ''),
            'original_columns' if window_size is None else f'window_size_{window_size}',
        )

        # Save metadata
        logger.info(f'results_subdir: {results_subdir_}')
        metadata = self._save_metadata(
            results_subdir_, dataset_name, df_, data_, window_size=window_size
        )

        # Save metadata in a dictionary keyed by dataset name and window size
        if window_size is not None:
            key = f'{dataset_name}_window-{window_size}'
        else:
            key = f'{dataset_name}_no-window'
        all_metadata[key] = metadata

        # Run anomaly detection
        for tool_ in ['pycaret']:
            for _, scores_ in self._iterate_ad_options(
                tool_, results_subdir_, dataset_name, window_size, **data_
            ):
                all_results.append(scores_)

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'anomaly',
        window_size: int | None = None,
        contamination: float | None = None,
    ) -> dict[str, Any]:
        """Prepare features, labels, and metadata for anomaly detection.

        :param pd.DataFrame df: The input DataFrame containing the time series data and target column.
        :param str target_col: The name of the target column to drop before creating features.
        :param int | None window_size: Sliding window  size. If None, no windows are created.
        :param float | None contamination: The expected proportion of anomalies in the dataset.
        :return dict: Dictionary of split data (X_train, y_train, X_test, y_test) and other metadata.
        """
        contamination_calculated = df[target_col].mean()  # Proportion of anomalies in the dataset
        if contamination is None:
            contamination = contamination_calculated

        logger.debug(
            f'df.shape: {df.shape}; df.columns: {df.columns.tolist()}; '
            f'label counts: {df[target_col].value_counts().to_dict()}; '
            f'contamination: {contamination:.4f}'
        )

        # Generate features using sliding windows
        if window_size is not None:
            features, labels = self._make_windows(
                df, window_size=window_size, target_col=target_col
            )
        else:
            labels = df[target_col].values
            X = df[[c for c in df.columns if c != target_col]].values
            features = pd.DataFrame(X, columns=[c for c in df.columns if c != target_col])

        # Deal with any date and time columns
        dropped_cols = [
            c for c in features.columns if any(s in c for s in ['datetime', 'time', 'timestamp'])
        ]
        features = features[[c for c in features.columns if c not in dropped_cols]]

        # Split into train/test sets (75/25 split)
        split_idx = int(0.75 * len(features))
        X_train, y_train = features.iloc[:split_idx], labels[:split_idx]
        X_test, y_test = features.iloc[split_idx:], labels[split_idx:]
        logger.debug(
            f'X_train: {X_train.shape}, y_train: {y_train.shape}, '
            f'X_test: {X_test.shape}, y_test: {y_test.shape}'
        )
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'contamination': contamination,
            'contamination_calculated': contamination_calculated,
            'dropped_cols': dropped_cols,
        }

    def _make_windows(self, df: pd.DataFrame, window_size: int, target_col: str) -> tuple:
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

    def _save_metadata(
        self, results_subdir: Path, dataset_name: str, df: pd.DataFrame, split_data: dict, **kwargs
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
            'contamination_calculated': split_data['contamination_calculated'],
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

    def _iterate_ad_options(
        self,
        class_name: str,
        results_subdir: Path,
        dataset_name: str,
        window_size: int | None,
        **kwargs,
    ):
        """Iterate through different anomaly detection options.

        :param str class_name: The name of the anomaly detection class to invoke.
        :param Path results_subdir: The directory to save results.
        :param str dataset_name: The name of the dataset.
        :param int | None window_size: The size of the sliding window used for features.
        :param dict kwargs: The data and metadata to pass to the anomaly detection class.
        :return: A generator yielding tuples of (estimator, scores) for each parameter combination.
        """
        # Determine which class to use based on the class_name
        if class_name in ['pycaret', PyCaretADModel.__name__]:
            ad_class = PyCaretADModel
        else:
            raise ValueError(f'Unknown class name: {class_name}')

        # File to save results for this class and dataset
        results_file = results_subdir / f'{class_name}.csv'

        # Check if this class has already been completed for this dataset/window size
        completion_file = results_subdir / f'{class_name}_completed.txt'
        if completion_file.exists():
            logger.info(f'{class_name} already completed. Skipping...')
            # Read existing scores from CSV and yield them
            if results_file.exists():
                existing_scores = pd.read_csv(results_file)
                for _, row in existing_scores.iterrows():
                    scores = row.to_dict()
                    scores['dataset'] = dataset_name  # Ensure dataset name is included
                    scores['window_size'] = window_size
                    yield None, scores
            else:
                raise ValueError(
                    f'Completion file exists but results file does not: {results_file}'
                )

        # Loop through all parameter combinations
        for param_name, options in ad_class.parameter_options.items():
            for param_value in options:
                kwargs[param_name] = param_value

                # Check if this parameter combination has already been run
                if self._combination_found(results_file, class_name, ad_class, **kwargs):
                    continue

                # Fit models and calculate scores
                results = self._invoke_ad_class(ad_class, **kwargs)
                scores = self._calculate_scores(dataset_name, window_size, *results, **kwargs)

                # Save scores
                results_subdir.mkdir(parents=True, exist_ok=True)
                Metrics.write_to_csv(results_file, scores)

                # logger.success(f'{class_name}:\n{json.dumps(scores_, indent=2)}')
                logger.success(
                    f'{class_name}: F1-score={scores["test__f1_score"]:.4f}, Precision='
                    f'{scores["test__precision"]:.4f}, Recall={scores["test__recall"]:.4f}'
                )

                # Yield results for analysis
                yield results[0], scores
        # Record completion
        completion_file.touch()

    def _invoke_ad_class(
        self,
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
        text = f'Running {ad_class.__name__}: contamination={contamination}'
        for param_name, _ in ad_class.parameter_options.items():
            param_value = kwargs.get(param_name, None)
            text += f', {param_name}={param_value}'
        logger.info(text)

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

    def _combination_found(
        self, results_file: Path, class_name: str, ad_class: type[BaseADModel], **kwargs
    ) -> bool:
        """Check if a given parameter combination has been completed for this class and dataset.

        :param Path results_file: The file where results are saved.
        :param str class_name: The name of the anomaly detection class.
        :param type[BaseADModel] ad_class: The anomaly detection class to check parameters for.
        :param dict kwargs: The parameter combination to check.
        :return: True if the combination has already been completed, False otherwise.
        """
        # Extract only the relevant parameters for this class from kwargs
        params = {k: v for k, v in kwargs.items() if k in ad_class.parameter_options}

        # Check if the results file exists and contains a row with the same parameter values
        if results_file.exists():
            existing_scores = pd.read_csv(results_file)
            if all((existing_scores[f'param__{k}'] == v).any() for k, v in params.items()):
                return True

        logger.debug(f'Running {class_name} with parameters: {params}')
        return False

    def _calculate_scores(
        self,
        dataset_name: str,
        window_size: int | None,
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

        :param str dataset_name: The name of the dataset.
        :param int | None window_size: The size of the sliding window used for features.
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
            'dataset': dataset_name,
            'window_size': window_size,
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

    def analyse_results(
        self,
        all_results: list[dict],
        all_metadata: dict,
        results_dir: Path,
        plots_dir: str = 'plots',
    ) -> None:
        """Analyze the results of all experiments and save summary statistics and plots.

        :param list[dict] all_results: A list of dictionaries of results for each experiment.
        :param dict all_metadata: A dictionary containing the metadata for all experiments.
        :param Path results_dir: The directory to save the analysis results in.
        :param str plots_dir: The directory name to save the plots in.
        """
        logger.debug(f'Saving metadata and results to: {results_dir}')

        # Save all metadata to a single JSON file and to CSV
        with open(results_dir / 'all_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, indent=4)
        df_metadata = pd.DataFrame(all_metadata)
        self._save_df_csv_and_tex(df_metadata, results_dir / 'all_metadata.csv')
        self._save_df_csv_and_tex(df_metadata.T, results_dir / 'all_metadata_T.csv')

        # Save all results to a single CSV file
        df = pd.DataFrame(all_results)
        self._save_df_csv_and_tex(df.copy(), results_dir / 'all_results.csv')

        # Save summary statistics of results to a separate CSV file
        df_info = df.describe(include='all').transpose()
        self._save_df_csv_and_tex(df_info, results_dir / 'all_results_info.csv')

        # Save latex table of results
        self._save_tables(df.copy(), results_dir)
        # Save box plots of test scores by the key column (e.g., dataset or model)
        self._save_plots(df.copy(), results_dir, plots_dir)

    def _save_tables(self, df: pd.DataFrame, results_dir: Path, min_f1_score: float = 0.5) -> None:
        """Save a LaTeX table of results to a .tex file in the results directory.

        :param pd.DataFrame df: The DataFrame containing the results to save as a LaTeX table.
        :param Path results_dir: The directory to save the LaTeX table in.
        :param float min_f1_score: The minimum F1-score for rows to include, defaults to 0.5.
        """
        cols_tex = ['dataset'] + [
            col for col in df.columns if col.startswith('test__') or col.startswith('param__')
        ]

        # 1. TABLE: Mean, median, min, max, etc. of F1-score by dataset and model
        f1_summary = df.groupby(['dataset', 'param__model_name'])['test__f1_score'].agg(
            ['mean', 'median', 'min', 'max', 'std', 'count']
        )
        self._save_df_csv_and_tex(
            f1_summary.reset_index(), results_dir / 'f1_score_summary_by_dataset_and_model.csv'
        )

        # 2. TABLE: MEAN SCORES BY MODEL AND DATASET (regardless of F1-score)
        df_mean = df[cols_tex].rename(columns=self.readable)
        df_mean = df_mean.sort_values(by=['DATASET', 'MODEL NAME'])
        self._save_df_csv_and_tex(df_mean, results_dir / 'table_mean_unfiltered.csv')

        # 3. TEX FILE: LIST OF DROPPED MODELS (F1-SCORE BELOW min_f1_score)
        # Filter columns and rows where test__f1_score is greater than min_f1_score
        df_filtered = df[df['test__f1_score'] > min_f1_score]
        # Get names of models that were dropped entirely and save to tex file
        dropped_models = set(df['param__model_name']) - set(df_filtered['param__model_name'])
        with open(results_dir / 'dropped_models.tex', 'w', encoding='utf-8') as f:
            if dropped_models:
                f.write(
                    f'Excluded models with F1-score below {min_f1_score:.2f}: '
                    f'{", ".join(dropped_models)}.'
                )
            else:
                f.write(f'No models were dropped (all had F1-score > {min_f1_score:.2f}).')

        # 4. TABLE: ALL SCORES BY MODEL AND DATASET (FILTERED TO F1-SCORE > min_f1_score)
        # Make column names more readable
        df_filtered = df_filtered[cols_tex].rename(columns=self.readable)
        # Sort by dataset and model name
        df_filtered = df_filtered.sort_values(by=['DATASET', 'MODEL NAME'])
        # Save LaTeX table to file
        self._save_df_csv_and_tex(df_filtered, results_dir / 'table_all_scores_filtered.csv')

    def readable(self, col_name: str) -> str:
        """Convert column names to a more readable format for plot titles and labels."""
        return col_name.replace('param__', '').replace('test__', '').replace('_', ' ').upper()

    def _save_df_csv_and_tex(self, df: pd.DataFrame, csv_path: Path) -> None:
        """Save a DataFrame to both CSV and LaTeX files in the results directory.

        :param pd.DataFrame df: The DataFrame to save.
        :param Path csv_path: Path to save the CSV file. The LaTeX file will use the same name.
        """
        # Readable dataset and model names for LaTeX table
        if 'dataset' in df.columns:
            df['dataset'] = df['dataset'].apply(self.readable)
        if 'model_name' in df.columns:
            df['model_name'] = df['model_name'].apply(self.readable)

        # Readable column names for LaTeX table
        df.columns = [self.readable(c) for c in df.columns]

        # Save to CSV
        df.to_csv(csv_path, index=False)
        # Save to LaTeX
        latex_table = df.to_latex(index=False, float_format='%.4f')
        with open(csv_path.with_suffix('.tex'), 'w', encoding='utf-8') as f:
            f.write(latex_table)

    def _save_plots(self, df, results_dir: Path, plots_dir: str = 'plots') -> None:
        """Save box plots of test scores by the key column (e.g., dataset or model).

        :param pd.DataFrame df: The DataFrame containing the results to plot.
        :param Path results_dir: The base directory to save the plots in.
        :param str plots_dir: The subdirectory to save the plots in.
        """
        key_cols = ['dataset', 'param__model_name']
        test_score_cols = [col for col in df.columns if col.startswith('test__')]

        # Loop through each key column and create box plots for each test score column
        for key_col in key_cols:
            subdir = results_dir / plots_dir / f'box_plots_by_{key_col}'
            for score_col in test_score_cols:
                self._generate_plot(subdir, 'box', df, score_col, key_col)

        # Calculate mean F1-score by model
        mean_f1_by_model = df.groupby('param__model_name')['test__f1_score'].mean()
        logger.debug(f'Mean F1-score by model:\n{mean_f1_by_model}')

        # Plot by dataset again but only for the best model
        best_model = mean_f1_by_model.idxmax()
        df_best_model = df[df['param__model_name'] == best_model]
        subdir = results_dir / plots_dir / 'box_plots_by_dataset-best_model'
        for score_col in test_score_cols:
            self._generate_plot(
                results_subdir=subdir,
                plot_type='bar',
                df=df_best_model,
                y_col=score_col,
                x_col='dataset',
                title=f'{score_col} by dataset for best model: {best_model}',
            )

    def _generate_plot(
        self,
        results_subdir: Path,
        plot_type: str,
        df: pd.DataFrame,
        y_col: str,
        x_col: str,
        title: str | None = None,
        figsize=(8, 4),
    ) -> None:
        """Create and save a box plot of the specified score column by the specified key column.

        :param Path results_subdir: The directory to save the plot in.
        :param str plot_type: The type of plot to create (e.g., 'box').
        :param pd.DataFrame df: The DataFrame containing the results to plot.
        :param str y_col: The name of the column containing the scores to plot on the Y-axis.
        :param str x_col: The name of the column containing the categories to plot on the X-axis.
        :param str | None title: The title of the plot. If None, a default title will be generated.
        :param tuple figsize: The size of the figure to create, defaults to (8, 4).
        """
        plt.figure(figsize=figsize)  # Set figure size

        # Create plot based on the specified type
        if plot_type == 'box':
            df.boxplot(column=y_col, by=x_col)
        elif plot_type == 'bar':
            df.groupby(x_col)[y_col].mean().plot(kind='bar')
        else:
            raise ValueError(f'Unknown plot type: {plot_type}')

        # Generate a readable title if not provided
        title = title or f'{plot_type} plot: {y_col} by {x_col}'

        # Save plot to file
        kwargs = {
            'title': self.readable(title),
            'xlabel': self.readable(x_col),
            'ylabel': self.readable(y_col),
            'xlabel_rotation': 90,
            'ylim': (0.0, 1.0),
        }
        for ext in ['svg']:  # , 'png']:
            save_path = results_subdir / f'box-plot_{y_col}_by_{x_col}.{ext}'
            Plotter.save_plot(save_path=save_path, **kwargs)
