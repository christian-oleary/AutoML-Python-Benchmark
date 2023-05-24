"""
Utility functions
"""

import csv
import logging
import math
import os
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, t
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, median_absolute_error,
                             mean_squared_error, r2_score)
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError


logger_name = 'Benchmark'
logger = logging.getLogger(logger_name)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(level=logging.DEBUG)


class Utils:
    """Utility functions"""

    logger = logging.getLogger(logger_name)


    @staticmethod
    def regression_scores(actual, predicted,
                          scores_dir=None,
                          forecaster_name=None,
                          multioutput='uniform_average',
                          **kwargs):
        """Calculate forecasting metrics and optionally save results.

        :param actual: Original time series values
        :param predicted: Predicted time series values
        :param scores_dir: Path to file to record scores (str or None), defaults to None
        :param forecaster_name: Name of model (str)
        :param multioutput: 'raw_values' (raw errors), 'uniform_average' (averaged errors), defaults to 'uniform_average'
        :raises TypeError: If forecaster_name is not provided when saving results to file
        :return results: Dictionary of results
        """

        # Convert pd.Series to NumPy Array
        if predicted.shape == (actual.shape[0], 1) and not pd.core.frame.DataFrame:
            predicted = predicted.flatten()

        if predicted.shape != actual.shape:
            raise ValueError(f'Predicted ({predicted.shape}) and actual ({actual.shape}) shapes do not match!')

        mase = MeanAbsoluteScaledError(multioutput='uniform_average')
        pearson = Utils.correlation(actual, predicted, method='pearson')
        spearman = Utils.correlation(actual, predicted, method='spearman')

        results = {
            'MAE': mean_absolute_error(actual, predicted, multioutput=multioutput),
            'MAE2': median_absolute_error(actual, predicted),
            'MAPE': mean_absolute_percentage_error(actual, predicted, multioutput=multioutput),
            # 'MASE': mase(actual, predicted),
            'ME': np.mean(actual - predicted),
            'MSE': mean_squared_error(actual, predicted, multioutput=multioutput),
            'Pearson Correlation': pearson[0],
            'Pearson P-value': pearson[1],
            'R2': r2_score(actual, predicted, multioutput=multioutput),
            'RMSE': math.sqrt(mean_squared_error(actual, predicted, multioutput=multioutput)),
            'sMAPE': Utils.smape(actual, predicted),
            'Spearman Correlation': spearman[0],
            'Spearman P-value': spearman[1],
        }

        if 'duration' in kwargs.keys():
            results['duration'] = kwargs['duration']

        if scores_dir != None:
            if forecaster_name == None:
                raise TypeError('Forecaster name required to save scores')
            os.makedirs(scores_dir, exist_ok=True)
            Utils.write_to_csv(os.path.join(scores_dir, f'{forecaster_name}.csv'), results)

        return results


    def correlation(actual, predicted, method='pearson'):
        """Wrapper to extract correlations and p-values from scipy

        :param actual: Actual values
        :param predicted: Predicted values
        :param method: Correlation type, defaults to 'pearson'
        :raises ValueError: If unknown correlation method is passed
        :return: Correlation (float) and pvalue (float)
        """
        if method == 'pearson':
            result = pearsonr(actual, predicted)
        elif method == 'spearman':
            result = spearmanr(actual, predicted)
        else:
            raise ValueError(f'Unknown correlation method: {method}')

        try:
            correlation = result.correlation
            pvalue = result.pvalue
        except: # older scipy versions returned a tuple instead of an object
            correlation = result[0]
            pvalue = result[1]

        return correlation, pvalue


    @staticmethod
    def smape(actual, predicted):
        return 100/len(actual) * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))


    @staticmethod
    def write_to_csv(path, results):
        """Record modelling results in a CSV file.

        :param str path: the result file path
        :param dict results: a dict containing results from running a model
        """

        np.set_printoptions(precision=4)

        # Remove unneeded values
        # unused_cols = []
        # for col in unused_cols:
        #     if col in results.keys():
        #         del results[col]

        if len(results) > 0:
            HEADERS = sorted(list(results.keys()), key=lambda v: str(v).upper())
            if 'model' in HEADERS:
                HEADERS.insert(0, HEADERS.pop(HEADERS.index('model')))

            for key, value in results.items():
                if value == None or value == '':
                    results[key] = 'None'

            try:
                Utils._write_to_csv(path, results, HEADERS)
            except OSError as _:
                # try a second time: permission error can be due to Python not
                # having closed the file fast enough after the previous write
                time.sleep(1) # in seconds
                Utils._write_to_csv(path, results, HEADERS)


    @staticmethod
    def _write_to_csv(path, results, headers):
        """Open and write results to CSV file.

        :param str path: Path to file
        :param dict results: Values to write
        :param list headers: A list of strings to order values by
        """

        is_new_file = not os.path.exists(path)
        with open(path, 'a+', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            if is_new_file:
                writer.writerow(headers)
            writer.writerow([results[header] for header in headers])


    @staticmethod
    def plot_forecast(actual, predicted, results_subdir, forecaster_name):
        """Plot forecasted vs actual values

        :param actual: Original time series values
        :param predicted: Forecasted values
        :param results_subdir: Path to output directory (str)
        :param forecaster_name: Model name (str)
        """
        pd.plotting.register_matplotlib_converters()

        # Create plot
        plt.figure(figsize=(20, 3))
        plt.plot(actual, label='actual')
        plt.plot(predicted, label='predicted')

        # Add title and legend
        plt.title(forecaster_name)
        plt.legend(loc='upper left')

        output_dir = os.path.join(results_subdir, 'plots')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{forecaster_name}.png'), bbox_inches='tight')


    @staticmethod
    def get_csv_datasets(datasets_directory):
        """Fetch list of file names of CSV datasets

        :param datasets_directory: Path to datasets directory (str)
        :raises NotADirectoryError: If datasets_directory does not exist
        :raises IOError: If datasets_directory does not have CSV files
        :return: list of dataset file names
        """

        if not os.path.exists(datasets_directory):
            raise NotADirectoryError('Datasets direcotry path does not exist')

        csv_files = [
            f for f in os.listdir(datasets_directory)
            if f.endswith('csv') and '0_metadata' not in f
            ]

        if len(csv_files) == 0:
            raise IOError('No CSV files found')

        return csv_files


    @staticmethod
    def split_test_set(test_df, horizon):
        """Split test dataset into list of smaller sets for rolling origin forecasting

        :param test_df: Test dataset (pandas DataFrame)
        :param horizon: Forecasting horizon (int)
        :return: List of DataFrame objects
        """

        test_splits = []
        total = 0 # total length of test splits
        for _ in range(int(len(test_df) / horizon)):
            test_splits.append(test_df.iloc[total:total+horizon, :])
            total += horizon

        # Leftover rows
        if total < len(test_df):
            test_splits.append(test_df.tail(len(test_df) - total))
        return test_splits
