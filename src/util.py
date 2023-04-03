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

        mase = MeanAbsoluteScaledError(multioutput='uniform_average')
        results = {
            'MAE': mean_absolute_error(actual, predicted, multioutput=multioutput),
            'MAE2': median_absolute_error(actual, predicted),
            'MAPE': mean_absolute_percentage_error(actual, predicted, multioutput=multioutput),
            # 'MASE': mase(actual.values, predicted.values),
            'ME': np.mean(actual.values - predicted.values),
            'MSE': mean_squared_error(actual, predicted, multioutput=multioutput),
            'Pearson Correlation': pearsonr(actual, predicted).correlation,
            'Pearson P-value': pearsonr(actual, predicted).pvalue,
            'R2': r2_score(actual, predicted, multioutput=multioutput),
            'RMSE': math.sqrt(mean_squared_error(actual, predicted, multioutput=multioutput)),
            'sMAPE': Utils.smape(actual.values, predicted.values),
            'Spearman Correlation': spearmanr(actual, predicted).correlation,
            'Spearman P-value': spearmanr(actual, predicted).pvalue,
        }

        if 'duration' in kwargs.keys():
            results['duration'] = kwargs['duration']

        if scores_dir != None:
            if forecaster_name == None:
                raise TypeError('Forecaster name required to save scores')
            os.makedirs(scores_dir, exist_ok=True)
            Utils.write_to_csv(os.path.join(scores_dir, f'{forecaster_name}.csv'), results)

        return results


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
        # Create plot
        plt.figure(figsize=(20, 3))
        plt.plot(actual.values, label='actual')
        plt.plot(predicted.values, label='predicted')

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
